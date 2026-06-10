// ============================================================================
// Pilotless Poses — Frontend Application
// WebSocket client, UI logic, canvas skeleton rendering
// ============================================================================

// ---- State ------------------------------------------------------------------
const STATE = {
  ws: null,
  connected: false,
  reconnectTimer: null,
  reconnectAttempts: 0,
  maxReconnectAttempts: 10,

  // Scan state
  scanning: false,
  totalFiles: 0,
  processedFiles: 0,
  errors: [],

  // Preview state
  results: [],           // { source, result, status, error_msg }
  currentResultIndex: -1,
  currentPoseData: null,
  currentImage: null,    // HTMLImageElement
  showOverlay: true,
  showLabels: true,
  showInfo: false,
  overlayOpacity: 0.85,

  // Processor state
  processors: [],          // { id, name, code, active, createdAt, updatedAt }
  editingProcessorId: null, // id being edited, or "__new__" for an unsaved one
  filteredOut: {},         // result path -> reason string (image hidden in preview)
  showFiltered: false,     // when true, filtered images stay visible (dimmed)
  runningProcessors: false,
  monacoEditor: null,
  monacoLoading: null,     // Promise while Monaco is loading
};

// ---- DOM refs (cached after DOMContentLoaded) --------------------------------
let DOM = {};

// ---- Constants --------------------------------------------------------------
const COCO_SKELETON = [
  // Face
  ["nose", "left_eye"], ["nose", "right_eye"],
  ["left_eye", "left_ear"], ["right_eye", "right_ear"],
  // Upper body
  ["left_shoulder", "right_shoulder"],
  ["left_shoulder", "left_elbow"], ["right_shoulder", "right_elbow"],
  ["left_elbow", "left_wrist"], ["right_elbow", "right_wrist"],
  // Torso
  ["left_shoulder", "left_hip"], ["right_shoulder", "right_hip"],
  ["left_hip", "right_hip"],
  // Lower body
  ["left_hip", "left_knee"], ["right_hip", "right_knee"],
  ["left_knee", "left_ankle"], ["right_knee", "right_ankle"],
];

const PERSON_COLORS = [
  "#5b6eea", // blue
  "#f0c145", // yellow
  "#43d9a3", // green
  "#f2556b", // red
  "#c084fc", // purple
  "#fb923c", // orange
  "#38bdf8", // sky
  "#f472b6", // pink
];

const RECONNECT_BASE_MS = 1000;
const RECONNECT_MAX_MS = 30000;

// ============================================================================
// WebSocket
// ============================================================================

function connectWebSocket() {
  if (STATE.ws && STATE.ws.readyState !== WebSocket.CLOSED) return;

  const protocol = location.protocol === "https:" ? "wss:" : "ws:";
  const wsUrl = `${protocol}//${location.host}/ws`;

  try {
    STATE.ws = new WebSocket(wsUrl);
  } catch (err) {
    console.error("WebSocket creation failed:", err);
    scheduleReconnect();
    return;
  }

  STATE.ws.addEventListener("open", () => {
    STATE.connected = true;
    STATE.reconnectAttempts = 0;
    updateConnectionIndicator(true);
    console.log("WebSocket connected");

    // Load initial file browser listing — restore saved path if any
    navigateBrowserTo("scan", FILE_BROWSER.scanPath);
  });

  STATE.ws.addEventListener("message", (event) => {
    try {
      const msg = JSON.parse(event.data);
      handleMessage(msg);
    } catch (e) {
      console.error("Failed to parse WebSocket message:", e);
    }
  });

  STATE.ws.addEventListener("close", (event) => {
    STATE.connected = false;
    updateConnectionIndicator(false);
    console.log("WebSocket closed:", event.code, event.reason);
    scheduleReconnect();
  });

  STATE.ws.addEventListener("error", (err) => {
    console.error("WebSocket error:", err);
    STATE.connected = false;
    updateConnectionIndicator(false);
  });
}

function scheduleReconnect() {
  if (STATE.reconnectTimer) return;
  if (STATE.reconnectAttempts >= STATE.maxReconnectAttempts) {
    console.warn("Max reconnect attempts reached");
    return;
  }

  const delay = Math.min(
    RECONNECT_BASE_MS * Math.pow(2, STATE.reconnectAttempts),
    RECONNECT_MAX_MS
  );
  STATE.reconnectAttempts++;

  STATE.reconnectTimer = setTimeout(() => {
    STATE.reconnectTimer = null;
    connectWebSocket();
  }, delay);
}

function sendMessage(msg) {
  if (!STATE.ws || STATE.ws.readyState !== WebSocket.OPEN) {
    console.warn("WebSocket not connected, cannot send:", msg);
    return false;
  }
  STATE.ws.send(JSON.stringify(msg));
  return true;
}

// Base64-encode bytes in chunks (avoids call-stack limits on large buffers).
function bytesToBase64(bytes) {
  let binary = "";
  const CHUNK = 0x8000;
  for (let i = 0; i < bytes.length; i += CHUNK) {
    binary += String.fromCharCode.apply(null, bytes.subarray(i, i + CHUNK));
  }
  return btoa(binary);
}

// Server-side Deno.writeFile, exposed to processor functions as
// `f_deno_write_file`. Accepts a string, Uint8Array, ArrayBuffer, or number[];
// the write is proxied to the server over the WebSocket. Returns a Promise that
// resolves true once the write is acknowledged (false if it failed or could not
// be sent).
function f_deno_write_file(path, data) {
  let bytes;
  if (typeof data === "string") {
    bytes = new TextEncoder().encode(data);
  } else if (data instanceof ArrayBuffer) {
    bytes = new Uint8Array(data);
  } else if (data instanceof Uint8Array) {
    bytes = data;
  } else if (Array.isArray(data)) {
    bytes = Uint8Array.from(data);
  } else {
    return Promise.reject(
      new TypeError(
        "f_deno_write_file: data must be a string, Uint8Array, ArrayBuffer, or number[]",
      ),
    );
  }

  const sent = sendMessage({
    type: "write_file",
    path,
    dataBase64: bytesToBase64(bytes),
  });
  if (!sent) return Promise.resolve(false);

  // Resolve when the server reports this path's write result.
  return new Promise((resolve) => {
    STATE.pendingWrites = STATE.pendingWrites || new Map();
    const queue = STATE.pendingWrites.get(path) || [];
    queue.push(resolve);
    STATE.pendingWrites.set(path, queue);
  });
}

function handleWriteFileResult(msg) {
  if (!msg.ok) {
    console.error(`f_deno_write_file failed for "${msg.path}":`, msg.error);
  }
  const queue = STATE.pendingWrites && STATE.pendingWrites.get(msg.path);
  if (queue && queue.length) {
    queue.shift()(!!msg.ok);
    if (queue.length === 0) STATE.pendingWrites.delete(msg.path);
  }
}

// ---- Processor helper functions ---------------------------------------------
//
// In addition to o_img and f_deno_write_file, each processor receives a set of
// convenience helpers bound to the current image, to simplify exporting it:
//   f_save_image(dir)    — copy the image file into dir
//   f_save_json(dir)     — write the image's pose JSON into dir (<name>_pose.json)
//   f_save_filtered(dir) — do both (image + JSON)
// All return a Promise resolving to whether the write(s) succeeded.

const join_path = (dir, name) => String(dir).replace(/\/+$/, "") + "/" + name;

// Build the image-bound helpers. `pose` is the raw pose JSON for the image
// (may be null when inference failed).
function makeImageHelpers(o_img, pose) {
  async function f_save_image(destDir) {
    const res = await fetch(
      "/api/image?path=" + encodeURIComponent(o_img.s_path_abs),
    );
    if (!res.ok) {
      throw new Error(
        `f_save_image: HTTP ${res.status} fetching "${o_img.s_name_file}"`,
      );
    }
    const buf = await res.arrayBuffer();
    return f_deno_write_file(
      join_path(destDir, o_img.s_name_file),
      new Uint8Array(buf),
    );
  }

  function f_save_json(destDir) {
    // Mirror the server's naming (<filename>_pose.json) and contents.
    const info = pose ??
      { image: o_img.s_path_abs, error: o_img.s_msg_error || "no pose data" };
    return f_deno_write_file(
      join_path(destDir, o_img.s_name_file + "_pose.json"),
      JSON.stringify(info, null, 2) + "\n",
    );
  }

  async function f_save_filtered(destDir) {
    const [okImg, okJson] = await Promise.all([
      f_save_image(destDir),
      f_save_json(destDir),
    ]);
    return okImg && okJson;
  }

  return { f_save_image, f_save_json, f_save_filtered };
}

// Argument names (and order) injected into every processor function. Keep the
// compile sites and the call site in sync via this single list.
const PROCESSOR_ARG_NAMES = [
  "o_img",
  "f_deno_write_file",
  "f_save_image",
  "f_save_json",
  "f_save_filtered",
];

// Produce the concrete argument values for a processor call on one image.
function processorArgs(o_img, pose) {
  const h = makeImageHelpers(o_img, pose);
  return [o_img, f_deno_write_file, h.f_save_image, h.f_save_json, h.f_save_filtered];
}

function updateConnectionIndicator(connected) {
  const el = document.getElementById("connection-indicator");
  if (!el) return;
  if (connected) {
    el.classList.add("connected");
    el.querySelector(".label").textContent = "Connected";
  } else {
    el.classList.remove("connected");
    el.querySelector(".label").textContent = "Disconnected";
  }
}

// ============================================================================
// Message Handlers
// ============================================================================

function handleMessage(msg) {
  switch (msg.type) {
    case "progress":
      handleProgress(msg);
      break;
    case "batch_complete":
      handleBatchComplete(msg);
      break;
    case "results_list":
      handleResultsList(msg);
      break;
    case "pose_data":
      handlePoseData(msg);
      break;
    case "browse_result":
      handleBrowseResult(msg);
      break;
    case "processors_list":
      handleProcessorsList(msg);
      break;
    case "all_pose_data":
      handleAllPoseData(msg);
      break;
    case "write_file_result":
      handleWriteFileResult(msg);
      break;
    case "error":
      handleError(msg);
      break;
    default:
      console.warn("Unknown message type:", msg.type);
  }
}

function handleProgress(msg) {
  STATE.processedFiles = msg.current;
  STATE.totalFiles = msg.total;

  // Update progress bar
  const pct = msg.total > 0 ? Math.round((msg.current / msg.total) * 100) : 0;
  const bar = DOM.progressBar;
  const text = DOM.progressText;
  if (bar) bar.style.width = `${pct}%`;
  if (text) text.textContent = `${msg.current} / ${msg.total}`;

  // Update summary
  const ok = msg.current - (STATE.errors.length);
  if (DOM.progressSummary) {
    DOM.progressSummary.textContent =
      `${ok} of ${msg.total} processed. ${STATE.errors.length} errors.`;
  }

  // Update file log entry
  updateFileLogEntry(msg);
}

function handleBatchComplete(msg) {
  STATE.scanning = false;
  STATE.errors = msg.errors || [];
  updateScanButtons(false);

  if (DOM.progressSummary) {
    const ok = msg.total - STATE.errors.length;
    DOM.progressSummary.textContent =
      `Batch complete: ${ok} of ${msg.total} images processed successfully. ${STATE.errors.length} errors.`;
  }
}

function handleResultsList(msg) {
  STATE.results = msg.results || [];
  renderImageList();
}

function handlePoseData(msg) {
  if (msg.error) {
    STATE.currentPoseData = null;
    showPoseError(msg.error);
    return;
  }
  STATE.currentPoseData = msg.data;
  renderPoseCanvas();
}

function handleError(msg) {
  console.error("Server error:", msg.message);
  if (STATE.scanning) {
    addLogEntry({ file: msg.message, status: "error", error: msg.message });
  }
}

// ============================================================================
// Page Navigation
// ============================================================================

function navigateTo(page) {
  document.querySelectorAll(".page").forEach((p) => p.classList.remove("active"));
  document.querySelectorAll(".nav-tab").forEach((t) => t.classList.remove("active"));

  const pageEl = document.getElementById(`page-${page}`);
  if (pageEl) pageEl.classList.add("active");
  const tabEl = document.getElementById(`nav-${page}`);
  if (tabEl) tabEl.classList.add("active");

  // Auto-load results when switching to the preview page. Also refresh the
  // processor list so the pipeline column reflects current active processors.
  if (page === "preview") {
    loadResults();
    sendMessage({ type: "list_processors" });
  }

  // Load processors and bootstrap Monaco when entering the processors page
  if (page === "processors") {
    sendMessage({ type: "list_processors" });
    ensureMonaco();
  }
}

// ============================================================================
// Page 1: Folder Scan
// ============================================================================

function initScanPage() {
  DOM.folderPath = document.getElementById("folder-path");
  DOM.extFilter = document.getElementById("ext-filter");
  DOM.outputDir = document.getElementById("output-dir");
  DOM.btnStartScan = document.getElementById("btn-start-scan");
  DOM.btnCancelScan = document.getElementById("btn-cancel-scan");
  DOM.progressBar = document.getElementById("progress-bar");
  DOM.progressText = document.getElementById("progress-text");
  DOM.progressSummary = document.getElementById("progress-summary");
  DOM.fileLog = document.getElementById("file-log");

  // Start scan
  DOM.btnStartScan.addEventListener("click", startScan);
  DOM.btnCancelScan.addEventListener("click", cancelScan);
}

function updateScanButtons(scanning) {
  DOM.btnStartScan.disabled = scanning;
  DOM.btnCancelScan.classList.toggle("hidden", !scanning);
  DOM.btnStartScan.textContent = scanning ? "Scanning..." : "Start Scan";
}

function startScan() {
  const folderPath = DOM.folderPath.value.trim();
  if (!folderPath) {
    alert("Please enter a folder path.");
    return;
  }

  // Sync the file browser state so the preview page can find results
  FILE_BROWSER.scanPath = folderPath;
  saveState();

  const extRaw = DOM.extFilter.value.trim() || ".jpg,.jpeg,.png";
  const extensions = extRaw.split(",").map((s) => s.trim()).filter(Boolean);
  const outputDir = DOM.outputDir.value.trim() || "./pose_results/";

  STATE.scanning = true;
  STATE.errors = [];
  STATE.totalFiles = 0;
  STATE.processedFiles = 0;

  updateScanButtons(true);

  // Reset UI
  if (DOM.progressBar) DOM.progressBar.style.width = "0%";
  if (DOM.progressText) DOM.progressText.textContent = "0 / 0";
  if (DOM.progressSummary) DOM.progressSummary.textContent = "";
  if (DOM.fileLog) DOM.fileLog.innerHTML = "";

  const sent = sendMessage({
    type: "scan_folder",
    folderPath,
    extensions,
    outputDir,
  });

  if (!sent) {
    alert("WebSocket not connected. Please wait for reconnection and try again.");
    STATE.scanning = false;
    updateScanButtons(false);
  }
}

function cancelScan() {
  sendMessage({ type: "cancel_scan" });
  STATE.scanning = false;
  updateScanButtons(false);
}

function addLogEntry(entry) {
  if (!DOM.fileLog) return;

  // Remove placeholder if present
  const placeholder = DOM.fileLog.querySelector(".log-placeholder");
  if (placeholder) placeholder.remove();

  const icons = {
    queued: "○",      // ○
    processing: "◐",  // ◐
    done: "✓",        // ✓
    error: "✗",       // ✗
  };

  const div = document.createElement("div");
  div.className = "log-entry";
  div.dataset.file = entry.file;

  const icon = document.createElement("span");
  icon.className = `status-icon ${entry.status}`;
  icon.textContent = icons[entry.status] || "";

  const nameEl = document.createElement("span");
  nameEl.className = "filename";
  nameEl.textContent = entry.file;
  nameEl.title = entry.file;

  div.appendChild(icon);
  div.appendChild(nameEl);

  if (entry.error) {
    const errEl = document.createElement("span");
    errEl.className = "error-msg";
    errEl.textContent = entry.error;
    errEl.title = entry.error;
    div.appendChild(errEl);
  }

  DOM.fileLog.appendChild(div);
  DOM.fileLog.scrollTop = DOM.fileLog.scrollHeight;
}

function updateFileLogEntry(msg) {
  if (!DOM.fileLog) return;

  // Try to update existing entry
  let entry = DOM.fileLog.querySelector(`[data-file="${CSS.escape(msg.file)}"]`);
  if (!entry) {
    addLogEntry(msg);
    return;
  }

  const icon = entry.querySelector(".status-icon");
  if (icon) {
    const icons = {
      queued: "○",
      processing: "◐",
      done: "✓",
      error: "✗",
    };
    icon.textContent = icons[msg.status] || "";
    icon.className = `status-icon ${msg.status}`;
  }

  // Add error message if present
  if (msg.error) {
    let errEl = entry.querySelector(".error-msg");
    if (!errEl) {
      errEl = document.createElement("span");
      errEl.className = "error-msg";
      entry.appendChild(errEl);
    }
    errEl.textContent = msg.error;
    errEl.title = msg.error;
  }
}

// ============================================================================
// Page 2: Pose Preview
// ============================================================================

function initPreviewPage() {
  DOM.imageList = document.getElementById("image-list");
  DOM.imageCount = document.getElementById("image-count");
  DOM.btnRunProcessors = document.getElementById("btn-run-processors");
  DOM.chkShowFiltered = document.getElementById("chk-show-filtered");
  DOM.processorFilterStatus = document.getElementById("processor-filter-status");
  DOM.processorOrderList = document.getElementById("processor-order-list");
  DOM.processorOrderCount = document.getElementById("processor-order-count");
  DOM.viewerPlaceholder = document.getElementById("viewer-placeholder");
  DOM.canvasContainer = document.getElementById("canvas-container");
  DOM.poseCanvas = document.getElementById("pose-canvas");
  DOM.canvasWrap = document.getElementById("canvas-wrap");
  DOM.canvasInfo = document.getElementById("canvas-info");

  DOM.btnToggleOverlay = document.getElementById("btn-toggle-overlay");
  DOM.btnToggleLabels = document.getElementById("btn-toggle-labels");
  DOM.btnToggleInfo = document.getElementById("btn-toggle-info");
  DOM.infoOverlay = document.getElementById("info-overlay");
  DOM.infoOverlayBody = document.getElementById("info-overlay-body");
  DOM.opacitySlider = document.getElementById("opacity-slider");
  DOM.btnPrevImage = document.getElementById("btn-prev-image");
  DOM.btnNextImage = document.getElementById("btn-next-image");
  DOM.navIndex = document.getElementById("nav-index");

  // Overlay toggles
  DOM.btnToggleOverlay.addEventListener("click", toggleOverlay);
  DOM.btnToggleLabels.addEventListener("click", toggleLabels);
  DOM.btnToggleInfo.addEventListener("click", toggleInfo);
  DOM.opacitySlider.addEventListener("input", () => {
    STATE.overlayOpacity = parseFloat(DOM.opacitySlider.value);
    renderPoseCanvas();
  });

  // Navigation
  DOM.btnPrevImage.addEventListener("click", () => navigateImages(-1));
  DOM.btnNextImage.addEventListener("click", () => navigateImages(1));

  // Processor run + filter visibility
  DOM.btnRunProcessors.addEventListener("click", runProcessors);
  DOM.chkShowFiltered.addEventListener("change", () => {
    STATE.showFiltered = DOM.chkShowFiltered.checked;
    renderImageList();
  });

  // Keyboard shortcuts
  document.addEventListener("keydown", handleKeyboard);

  // Re-fit the canvas to the viewer when the window resizes
  let resizeTimer = null;
  window.addEventListener("resize", () => {
    if (resizeTimer) clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {
      if (STATE.currentImage) renderPoseCanvas();
    }, 120);
  });
}

function loadResults() {
  // Use the folder that was last scanned (from the file browser)
  const folderPath = FILE_BROWSER.scanPath;
  if (!folderPath || folderPath === "/") {
    // Nothing scanned yet — show empty state
    STATE.results = [];
    renderImageList();
    return;
  }

  sendMessage({ type: "get_results", folderPath });
}

function isFilteredOut(result) {
  return Object.prototype.hasOwnProperty.call(STATE.filteredOut, result.result);
}

function renderImageList() {
  if (!DOM.imageList) return;

  DOM.imageList.innerHTML = "";

  if (STATE.results.length === 0) {
    DOM.imageList.innerHTML =
      '<div class="placeholder">No results found for this folder.</div>';
    DOM.imageCount.textContent = "0";
    updateFilterStatus();
    return;
  }

  // Apply processor filtering. When showFiltered is off, hidden images are
  // dropped from the list entirely; when on, they remain but are dimmed.
  const visible = STATE.results.filter(
    (r) => STATE.showFiltered || !isFilteredOut(r),
  );

  DOM.imageCount.textContent = String(visible.length);
  updateFilterStatus();

  if (visible.length === 0) {
    DOM.imageList.innerHTML =
      '<div class="placeholder">All images were filtered out by active processors.</div>';
    return;
  }

  visible.forEach((r) => {
    const idx = STATE.results.indexOf(r);
    const div = document.createElement("div");
    div.className = "image-item";
    if (isFilteredOut(r)) div.classList.add("filtered-out");
    div.dataset.index = idx;
    div.title = r.source;

    // Thumbnail — use the source image path via a data URL approach
    // We'll generate a thumbnail URL using the server
    const thumb = document.createElement("img");
    thumb.className = "thumb";
    thumb.src = `/api/thumbnail?path=${encodeURIComponent(r.source)}`;
    thumb.alt = r.source;
    thumb.loading = "lazy";
    thumb.onerror = () => { thumb.src = ""; };

    const info = document.createElement("div");
    info.className = "info";
    const displayName = r.source.split("/").pop() || r.source;
    const nameEl = document.createElement("span");
    nameEl.className = "name";
    nameEl.textContent = displayName;
    nameEl.title = r.source;
    const statusEl = document.createElement("span");
    statusEl.className = "status";
    statusEl.textContent = r.status === "ok" ? "Pose data available" : (r.error_msg || "Error");

    info.appendChild(nameEl);
    info.appendChild(statusEl);

    if (isFilteredOut(r)) {
      const tag = document.createElement("span");
      tag.className = "filter-tag";
      tag.textContent = "filtered";
      tag.title = STATE.filteredOut[r.result] || "Filtered out by a processor";
      info.title = tag.title;
      div.appendChild(thumb);
      div.appendChild(info);
      div.appendChild(tag);
    } else {
      const dot = document.createElement("span");
      dot.className = `status-dot ${r.status}`;
      div.appendChild(thumb);
      div.appendChild(info);
      div.appendChild(dot);
    }

    div.addEventListener("click", () => selectImage(idx));

    DOM.imageList.appendChild(div);
  });
}

function updateFilterStatus() {
  const el = DOM.processorFilterStatus;
  if (!el) return;
  const total = STATE.results.length;
  const filtered = STATE.results.filter(isFilteredOut).length;
  if (filtered === 0) {
    el.textContent = total > 0 ? `${total} image${total !== 1 ? "s" : ""}` : "";
    el.classList.remove("has-filter");
  } else {
    el.textContent =
      `${filtered} of ${total} filtered out by processors` +
      (STATE.showFiltered ? " (shown, dimmed)" : "");
    el.classList.add("has-filter");
  }
}

function selectImage(index) {
  STATE.currentResultIndex = index;
  STATE.currentPoseData = null;
  STATE.currentImage = null;

  // Update selection highlight
  document.querySelectorAll(".image-item").forEach((el) => el.classList.remove("active"));
  const item = document.querySelector(`.image-item[data-index="${index}"]`);
  if (item) item.classList.add("active");

  // Update nav display
  if (DOM.navIndex) {
    DOM.navIndex.textContent = `${index + 1} / ${STATE.results.length}`;
  }

  // Show canvas area
  if (DOM.viewerPlaceholder) DOM.viewerPlaceholder.classList.add("hidden");
  if (DOM.canvasContainer) DOM.canvasContainer.classList.remove("hidden");

  // Clear canvas
  const canvas = DOM.poseCanvas;
  if (canvas) {
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    canvas.width = 0;
    canvas.height = 0;
  }
  if (DOM.canvasInfo) DOM.canvasInfo.textContent = "Loading...";

  // Load the source image
  const result = STATE.results[index];
  const img = new Image();
  img.onload = () => {
    STATE.currentImage = img;
    // Draw the image right away; the skeleton is layered on once pose data
    // arrives. This keeps the viewer stable instead of flashing empty.
    renderPoseCanvas();
    sendMessage({
      type: "get_pose_data",
      resultPath: result.result,
    });
  };
  img.onerror = () => {
    STATE.currentImage = null;
    STATE.currentPoseData = null;
    clearCanvasWithMessage("Failed to load image.");
  };
  img.src = `/api/image?path=${encodeURIComponent(result.source)}`;
}

function showPoseError(errorMsg) {
  // Keep the image visible — the overlay simply has no skeleton to draw.
  STATE.currentPoseData = null;
  renderPoseCanvas();
  if (DOM.canvasInfo) DOM.canvasInfo.textContent = `Pose data error: ${errorMsg}`;
}

function clearCanvasWithMessage(message) {
  const canvas = DOM.poseCanvas;
  if (canvas) {
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    canvas.width = 0;
    canvas.height = 0;
  }
  if (DOM.canvasInfo) DOM.canvasInfo.textContent = message;
  updateInfoOverlay();
}

function renderPoseCanvas() {
  const canvas = DOM.poseCanvas;
  const img = STATE.currentImage;
  const poseData = STATE.currentPoseData;

  if (!canvas || !img) return;

  // Fit the image inside the viewer area on BOTH axes so the canvas always
  // occupies the same stable region — its layout never shifts based on whether
  // pose data is present or how tall the image is.
  const wrap = DOM.canvasWrap;
  const availW = (wrap ? wrap.clientWidth : img.naturalWidth) - 2;
  const availH = (wrap ? wrap.clientHeight : img.naturalHeight) - 2;
  const scale =
    Math.min(availW / img.naturalWidth, availH / img.naturalHeight) || 1;
  canvas.width = Math.max(1, Math.floor(img.naturalWidth * scale));
  canvas.height = Math.max(1, Math.floor(img.naturalHeight * scale));

  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Draw image (always — even when there is no pose data)
  ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

  const people = poseData && poseData.people ? poseData.people : [];

  // Draw skeletons (if any)
  people.forEach((person, pidx) => {
    const color = PERSON_COLORS[pidx % PERSON_COLORS.length];
    drawSkeleton(ctx, person, scale, scale, color);
  });

  // Update info line
  if (DOM.canvasInfo) {
    const ppl = people.length;
    DOM.canvasInfo.textContent = poseData
      ? `${img.naturalWidth}×${img.naturalHeight} — ${ppl} person${ppl !== 1 ? "s" : ""} detected`
      : `${img.naturalWidth}×${img.naturalHeight}`;
  }

  updateInfoOverlay();
}

// ---- Image data overlay ----

function toggleInfo() {
  STATE.showInfo = !STATE.showInfo;
  DOM.btnToggleInfo.classList.toggle("btn-active", STATE.showInfo);
  DOM.btnToggleInfo.textContent = STATE.showInfo ? "Hide Info" : "Show Info";
  DOM.infoOverlay.classList.toggle("hidden", !STATE.showInfo);
  if (STATE.showInfo) updateInfoOverlay();
}

function updateInfoOverlay() {
  const body = DOM.infoOverlayBody;
  if (!body || STATE.showInfo === false) return;

  const result = STATE.results[STATE.currentResultIndex];
  const img = STATE.currentImage;
  const pose = STATE.currentPoseData;
  const people = pose && pose.people ? pose.people : [];

  const rows = [];
  if (result) {
    rows.push(["File", result.source.split("/").pop() || result.source]);
    rows.push(["Path", result.source]);
    rows.push([
      "Status",
      result.status === "ok" ? "OK" : (result.error_msg || "Error"),
    ]);
  }
  if (img) rows.push(["Dimensions", `${img.naturalWidth}×${img.naturalHeight}`]);
  rows.push(["People", String(people.length)]);

  let html = "";
  for (const [k, v] of rows) {
    html += `<dt>${escapeHtml(k)}</dt><dd>${escapeHtml(String(v))}</dd>`;
  }

  people.forEach((p, i) => {
    const kpts = p.keypoints || [];
    const visible = kpts.filter((kp) => kp.confidence >= 0.3).length;
    const avg = kpts.length
      ? kpts.reduce((s, kp) => s + kp.confidence, 0) / kpts.length
      : 0;
    html += `<div class="group-label">Person ${i + 1}</div>`;
    html += `<dt>Keypoints</dt><dd>${kpts.length}</dd>`;
    html += `<dt>Visible ≥0.3</dt><dd>${visible}</dd>`;
    html += `<dt>Avg conf</dt><dd>${avg.toFixed(3)}</dd>`;
  });

  body.innerHTML = html || "<dd>No data</dd>";
}

function drawSkeleton(ctx, person, sx, sy, color) {
  const kpts = person.keypoints;
  if (!kpts || kpts.length === 0) return;

  // Build a name -> keypoint map
  const kptMap = {};
  kpts.forEach((kp) => {
    kptMap[kp.name] = kp;
  });

  const alpha = STATE.overlayOpacity;

  // ---- Draw skeleton lines ----
  if (STATE.showOverlay) {
    COCO_SKELETON.forEach(([aName, bName]) => {
      const a = kptMap[aName];
      const b = kptMap[bName];
      if (!a || !b) return;

      // Average confidence determines visibility
      const avgConf = (a.confidence + b.confidence) / 2;
      if (avgConf < 0.1) return;

      const lineColor = confidenceColor(color, avgConf);
      ctx.strokeStyle = applyAlpha(lineColor, alpha);
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(a.x * sx, a.y * sy);
      ctx.lineTo(b.x * sx, b.y * sy);
      ctx.stroke();
    });
  }

  // ---- Draw keypoints ----
  if (STATE.showOverlay) {
    kpts.forEach((kp) => {
      const fillColor = confidenceColor(color, kp.confidence);
      ctx.fillStyle = applyAlpha(fillColor, alpha);
      ctx.strokeStyle = applyAlpha("#ffffff", alpha * 0.7);
      ctx.lineWidth = 1;
      ctx.beginPath();
      const r = 4;
      ctx.arc(kp.x * sx, kp.y * sy, r, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    });
  }

  // ---- Draw labels ----
  if (STATE.showLabels) {
    ctx.font = '10px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif';
    ctx.textAlign = "left";
    ctx.textBaseline = "bottom";

    kpts.forEach((kp) => {
      if (kp.confidence < 0.15) return;
      const label = kp.name
        .replace(/_/g, " ")
        .replace(/\b\w/g, (c) => c.toUpperCase());
      const lx = kp.x * sx + 7;
      const ly = kp.y * sy - 5;
      ctx.fillStyle = applyAlpha(color, alpha);
      ctx.fillText(label, lx, ly);
    });
  }
}

// ---- Color helpers ----

function confidenceColor(baseColor, confidence) {
  if (confidence >= 0.7) return baseColor;
  if (confidence >= 0.3) return "#f0c145"; // yellow
  return "#f2556b"; // red
}

function applyAlpha(hex, alpha) {
  // Convert hex to rgba
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

// ---- Overlay toggles ----

function toggleOverlay() {
  STATE.showOverlay = !STATE.showOverlay;
  DOM.btnToggleOverlay.classList.toggle("off", !STATE.showOverlay);
  DOM.btnToggleOverlay.textContent = STATE.showOverlay ? "Hide Skeleton" : "Show Skeleton";
  renderPoseCanvas();
}

function toggleLabels() {
  STATE.showLabels = !STATE.showLabels;
  DOM.btnToggleLabels.classList.toggle("off", !STATE.showLabels);
  DOM.btnToggleLabels.textContent = STATE.showLabels ? "Hide Labels" : "Show Labels";
  renderPoseCanvas();
}

function navigateImages(delta) {
  if (STATE.results.length === 0) return;
  let idx = STATE.currentResultIndex + delta;
  if (idx < 0) idx = STATE.results.length - 1;
  if (idx >= STATE.results.length) idx = 0;
  selectImage(idx);
}

function handleKeyboard(e) {
  // Only handle when preview page is active
  const previewPage = document.getElementById("page-preview");
  if (!previewPage || !previewPage.classList.contains("active")) return;

  // Don't capture when typing in input fields
  if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;

  switch (e.key.toLowerCase()) {
    case "arrowleft":
      e.preventDefault();
      navigateImages(-1);
      break;
    case "arrowright":
      e.preventDefault();
      navigateImages(1);
      break;
    case "h":
      e.preventDefault();
      toggleOverlay();
      break;
    case "l":
      e.preventDefault();
      toggleLabels();
      break;
    case "i":
      e.preventDefault();
      toggleInfo();
      break;
  }
}

// ============================================================================
// Page 3: Processors
// ============================================================================

const MONACO_VS = "https://cdn.jsdelivr.net/npm/monaco-editor@0.45.0/min/vs";

const PROCESSOR_TEMPLATE =
  `// Return true to keep the image, false to filter it out of the preview.\n` +
  `// The callback receives o_img — e.g. keep only images with a person:\n` +
  `return o_img.n_persons > 0;`;

function initProcessorsPage() {
  DOM.processorList = document.getElementById("processor-list");
  DOM.processorCount = document.getElementById("processor-count");
  DOM.btnNewProcessor = document.getElementById("btn-new-processor");
  DOM.editorPlaceholder = document.getElementById("editor-placeholder");
  DOM.editorPanel = document.getElementById("editor-panel");
  DOM.processorName = document.getElementById("processor-name");
  DOM.processorActive = document.getElementById("processor-active");
  DOM.monacoHost = document.getElementById("monaco-editor");
  DOM.editorMessage = document.getElementById("editor-message");
  DOM.btnSaveProcessor = document.getElementById("btn-save-processor");
  DOM.btnDeleteProcessor = document.getElementById("btn-delete-processor");
  DOM.btnCancelProcessor = document.getElementById("btn-cancel-processor");

  DOM.btnNewProcessor.addEventListener("click", newProcessor);
  DOM.btnSaveProcessor.addEventListener("click", saveProcessor);
  DOM.btnDeleteProcessor.addEventListener("click", deleteProcessor);
  DOM.btnCancelProcessor.addEventListener("click", closeEditor);
}

// ---- Monaco (lazy) ----

function ensureMonaco() {
  if (STATE.monacoEditor || STATE.fallbackTextarea) {
    return Promise.resolve();
  }
  if (STATE.monacoLoading) return STATE.monacoLoading;

  STATE.monacoLoading = new Promise((resolve) => {
    const fallback = () => {
      // Monaco unavailable (e.g. offline) — degrade to a plain textarea so the
      // feature still works.
      if (!DOM.monacoHost) return resolve();
      const ta = document.createElement("textarea");
      ta.className = "fallback-editor";
      ta.spellcheck = false;
      ta.style.cssText =
        "width:100%;height:100%;min-height:200px;background:#1e1e1e;color:#e1e3ec;" +
        "border:none;font-family:var(--font-mono);font-size:13px;padding:8px;resize:none;outline:none;";
      DOM.monacoHost.appendChild(ta);
      STATE.fallbackTextarea = ta;
      resolve();
    };

    if (typeof require === "undefined") {
      fallback();
      return;
    }

    try {
      require.config({ paths: { vs: MONACO_VS } });
      require(["vs/editor/editor.main"], () => {
        try {
          STATE.monacoEditor = monaco.editor.create(DOM.monacoHost, {
            value: "",
            language: "javascript",
            theme: "vs-dark",
            minimap: { enabled: false },
            automaticLayout: true,
            fontSize: 13,
            scrollBeyondLastLine: false,
            tabSize: 2,
            lineNumbers: "on",
          });
          resolve();
        } catch {
          fallback();
        }
      }, fallback);
    } catch {
      fallback();
    }
  });

  return STATE.monacoLoading;
}

function getEditorCode() {
  if (STATE.monacoEditor) return STATE.monacoEditor.getValue();
  if (STATE.fallbackTextarea) return STATE.fallbackTextarea.value;
  return "";
}

function setEditorCode(code) {
  if (STATE.monacoEditor) STATE.monacoEditor.setValue(code);
  else if (STATE.fallbackTextarea) STATE.fallbackTextarea.value = code;
}

// ---- Processor list / CRUD ----

function handleProcessorsList(msg) {
  STATE.processors = msg.processors || [];
  if (msg.error) console.error("Processors error:", msg.error);

  // After creating a new processor, the server assigns it an id. Resolve our
  // "__new__" placeholder to the concrete record so editing continues smoothly.
  if (STATE.editingProcessorId === "__new__" && STATE.pendingSelectName) {
    const created = [...STATE.processors]
      .reverse()
      .find((p) => p.name === STATE.pendingSelectName);
    if (created) {
      STATE.editingProcessorId = created.id;
      STATE.pendingSelectName = null;
      if (DOM.btnDeleteProcessor) {
        DOM.btnDeleteProcessor.classList.remove("hidden");
      }
    }
  }

  renderProcessorList();
  renderProcessorOrder();

  // If we were editing a processor that no longer exists, close the editor.
  if (
    STATE.editingProcessorId &&
    STATE.editingProcessorId !== "__new__" &&
    !STATE.processors.some((p) => p.id === STATE.editingProcessorId)
  ) {
    closeEditor();
  }
}

function renderProcessorList() {
  const list = DOM.processorList;
  if (!list) return;
  list.innerHTML = "";

  if (DOM.processorCount) {
    DOM.processorCount.textContent = String(STATE.processors.length);
  }

  if (STATE.processors.length === 0) {
    list.innerHTML =
      '<div class="placeholder">No processors yet. Create one to get started.</div>';
    return;
  }

  STATE.processors.forEach((p) => {
    const div = document.createElement("div");
    div.className = "processor-entry";
    if (p.id === STATE.editingProcessorId) div.classList.add("active");

    const name = document.createElement("span");
    name.className = "p-name";
    name.textContent = p.name || "(unnamed)";
    name.title = p.name || "";

    const toggle = document.createElement("button");
    toggle.className = `p-toggle ${p.active ? "on" : ""}`;
    toggle.textContent = p.active ? "Active" : "Off";
    toggle.title = "Toggle active";
    toggle.addEventListener("click", (e) => {
      e.stopPropagation();
      toggleProcessorActive(p);
    });

    div.appendChild(name);
    div.appendChild(toggle);
    div.addEventListener("click", () => editProcessor(p.id));
    list.appendChild(div);
  });
}

// ---- Pipeline column (active processors, reorderable) on the preview page ----

let dragProcId = null;

function renderProcessorOrder() {
  const list = DOM.processorOrderList;
  if (!list) return;

  const active = STATE.processors.filter((p) => p.active);
  if (DOM.processorOrderCount) {
    DOM.processorOrderCount.textContent = String(active.length);
  }

  list.innerHTML = "";
  if (active.length === 0) {
    list.innerHTML = '<div class="placeholder">No active processors.</div>';
    return;
  }

  active.forEach((p, i) => {
    const div = document.createElement("div");
    div.className = "processor-order-entry";
    div.draggable = true;
    div.dataset.id = p.id;

    const handle = document.createElement("span");
    handle.className = "po-handle";
    handle.textContent = "⠿";
    handle.title = "Drag to reorder";

    const num = document.createElement("span");
    num.className = "po-num";
    num.textContent = String(i + 1);

    const name = document.createElement("span");
    name.className = "po-name";
    name.textContent = p.name || "(unnamed)";
    name.title = p.name || "";

    div.append(handle, num, name);

    div.addEventListener("dragstart", (e) => {
      dragProcId = p.id;
      div.classList.add("dragging");
      e.dataTransfer.effectAllowed = "move";
      // Firefox requires data to be set for dragging to start.
      e.dataTransfer.setData("text/plain", p.id);
    });
    div.addEventListener("dragend", () => {
      dragProcId = null;
      div.classList.remove("dragging");
      list.querySelectorAll(".drop-before, .drop-after")
        .forEach((el) => el.classList.remove("drop-before", "drop-after"));
    });
    div.addEventListener("dragover", (e) => {
      if (!dragProcId || dragProcId === p.id) return;
      e.preventDefault();
      e.dataTransfer.dropEffect = "move";
      const after = isPointerInLowerHalf(e, div);
      div.classList.toggle("drop-after", after);
      div.classList.toggle("drop-before", !after);
    });
    div.addEventListener("dragleave", () => {
      div.classList.remove("drop-before", "drop-after");
    });
    div.addEventListener("drop", (e) => {
      e.preventDefault();
      div.classList.remove("drop-before", "drop-after");
      if (!dragProcId || dragProcId === p.id) return;
      reorderActiveProcessors(dragProcId, p.id, isPointerInLowerHalf(e, div));
    });

    list.appendChild(div);
  });
}

function isPointerInLowerHalf(event, el) {
  const rect = el.getBoundingClientRect();
  return event.clientY > rect.top + rect.height / 2;
}

// Move `draggedId` next to `targetId` among the active processors, then rebuild
// STATE.processors so inactive entries keep their slots. Persists the new order
// to the server and re-applies the pipeline if a run already produced results.
function reorderActiveProcessors(draggedId, targetId, placeAfter) {
  const activeIds = STATE.processors.filter((p) => p.active).map((p) => p.id);
  const from = activeIds.indexOf(draggedId);
  if (from === -1) return;
  activeIds.splice(from, 1);

  let to = activeIds.indexOf(targetId);
  if (to === -1) to = activeIds.length;
  else if (placeAfter) to += 1;
  activeIds.splice(to, 0, draggedId);

  // Fill the active slots in the new order; leave inactive entries in place.
  const byId = Object.fromEntries(STATE.processors.map((p) => [p.id, p]));
  let ai = 0;
  STATE.processors = STATE.processors.map((p) =>
    p.active ? byId[activeIds[ai++]] : p
  );

  renderProcessorOrder();
  renderProcessorList();
  sendMessage({
    type: "reorder_processors",
    order: STATE.processors.map((p) => p.id),
  });

  // Reflect the new order immediately if processors were already run.
  if (STATE.results.length > 0) runProcessors();
}

async function newProcessor() {
  STATE.editingProcessorId = "__new__";
  await openEditor();
  DOM.processorName.value = "";
  DOM.processorActive.checked = true;
  setEditorCode(PROCESSOR_TEMPLATE);
  DOM.btnDeleteProcessor.classList.add("hidden");
  setEditorMessage("");
  renderProcessorList();
  DOM.processorName.focus();
}

async function editProcessor(id) {
  const p = STATE.processors.find((x) => x.id === id);
  if (!p) return;
  STATE.editingProcessorId = id;
  await openEditor();
  DOM.processorName.value = p.name || "";
  DOM.processorActive.checked = !!p.active;
  setEditorCode(p.code || "");
  DOM.btnDeleteProcessor.classList.remove("hidden");
  setEditorMessage("");
  renderProcessorList();
}

async function openEditor() {
  DOM.editorPlaceholder.classList.add("hidden");
  DOM.editorPanel.classList.remove("hidden");
  await ensureMonaco();
}

function closeEditor() {
  STATE.editingProcessorId = null;
  DOM.editorPanel.classList.add("hidden");
  DOM.editorPlaceholder.classList.remove("hidden");
  setEditorMessage("");
  renderProcessorList();
}

function setEditorMessage(text, kind) {
  if (!DOM.editorMessage) return;
  DOM.editorMessage.textContent = text;
  DOM.editorMessage.className = "editor-message" + (kind ? " " + kind : "");
}

function saveProcessor() {
  const name = DOM.processorName.value.trim();
  const code = getEditorCode();
  const active = DOM.processorActive.checked;

  if (!name) {
    setEditorMessage("Please give the processor a name.", "error");
    return;
  }

  // Validate the code compiles before saving so users get immediate feedback.
  try {
    new Function(...PROCESSOR_ARG_NAMES, code);
  } catch (e) {
    setEditorMessage(`Syntax error: ${e.message}`, "error");
    return;
  }

  const id = STATE.editingProcessorId === "__new__"
    ? undefined
    : STATE.editingProcessorId;

  const sent = sendMessage({
    type: "save_processor",
    processor: { id, name, code, active },
  });

  if (!sent) {
    setEditorMessage("Not connected — could not save.", "error");
    return;
  }

  // The server responds with the full updated list. Keep the editor open on
  // the saved item; resolve "__new__" to a concrete id once the list returns.
  if (!id) {
    STATE.pendingSelectName = name;
  }
  setEditorMessage("Saved.", "success");
}

function deleteProcessor() {
  if (!STATE.editingProcessorId || STATE.editingProcessorId === "__new__") {
    closeEditor();
    return;
  }
  const p = STATE.processors.find((x) => x.id === STATE.editingProcessorId);
  if (p && !confirm(`Delete processor "${p.name}"?`)) return;
  sendMessage({ type: "delete_processor", id: STATE.editingProcessorId });
  closeEditor();
}

function toggleProcessorActive(p) {
  sendMessage({
    type: "save_processor",
    processor: { id: p.id, name: p.name, code: p.code, active: !p.active },
  });
}

// ---- Running processors against all images ----

// Convert a raw person ({ keypoints: [{ name, x, y, confidence }] }) into the
// nested o_person shape processors see, e.g. o_person.o_wrist.o_left.n_x.
// Paired keypoints (left_*/right_*) become { o_left, o_right }; the lone "nose"
// keypoint sits directly on o_person. A missing keypoint is null.
function buildPersonObject(person) {
  const byName = {};
  for (const k of (person && person.keypoints) || []) byName[k.name] = k;

  const toPoint = (kp) =>
    kp ? { n_x: kp.x, n_y: kp.y, n_conf: kp.confidence } : null;
  const pair = (base) => ({
    o_left: toPoint(byName["left_" + base]),
    o_right: toPoint(byName["right_" + base]),
  });

  return {
    o_nose: toPoint(byName["nose"]),
    o_eye: pair("eye"),
    o_ear: pair("ear"),
    o_shoulder: pair("shoulder"),
    o_elbow: pair("elbow"),
    o_wrist: pair("wrist"),
    o_hip: pair("hip"),
    o_knee: pair("knee"),
    o_ankle: pair("ankle"),
  };
}

async function runProcessors() {
  if (STATE.runningProcessors) return;

  const active = STATE.processors.filter((p) => p.active);
  if (active.length === 0) {
    updateFilterStatusText("No active processors to run.", true);
    return;
  }

  // Compile all active processors up front so a syntax error stops the run.
  const compiled = [];
  for (const p of active) {
    try {
      compiled.push({
        name: p.name,
        fn: new Function(...PROCESSOR_ARG_NAMES, p.code),
      });
    } catch (e) {
      updateFilterStatusText(
        `Processor "${p.name}" has a syntax error: ${e.message}`,
        true,
      );
      return;
    }
  }

  STATE._compiledRun = compiled;
  STATE.runningProcessors = true;
  setRunButton(true);
  updateFilterStatusText("Loading pose data…");

  const folderPath = FILE_BROWSER.scanPath;
  const sent = sendMessage({ type: "get_all_pose_data", folderPath });
  if (!sent) {
    STATE.runningProcessors = false;
    setRunButton(false);
    updateFilterStatusText("Not connected — could not run.", true);
  }
}

function setRunButton(running) {
  if (!DOM.btnRunProcessors) return;
  DOM.btnRunProcessors.disabled = running;
  DOM.btnRunProcessors.textContent = running ? "Running…" : "Run Processors";
}

async function handleAllPoseData(msg) {
  if (!STATE.runningProcessors) return;
  const compiled = STATE._compiledRun || [];
  const items = msg.items || [];

  if (msg.error) {
    STATE.runningProcessors = false;
    setRunButton(false);
    updateFilterStatusText(`Run failed: ${msg.error}`, true);
    return;
  }

  const filtered = {};
  let passed = 0;
  let errors = 0;

  // Resolve image dimensions with limited concurrency so we don't open
  // hundreds of image requests at once.
  const dimsByResult = await mapWithConcurrency(items, 6, async (item) => {
    if (item.status !== "ok") return [item.result, { width: 0, height: 0 }];
    const dims = await loadImageDims(item.source).catch(() => ({
      width: 0,
      height: 0,
    }));
    return [item.result, dims];
  });
  const dimMap = Object.fromEntries(dimsByResult);

  for (const item of items) {
    const pose = item.data || null;
    const people = (pose && pose.people) ? pose.people : [];
    const dims = dimMap[item.result] || { width: 0, height: 0 };

    const a_o_person = people.map(buildPersonObject);
    const o_img = {
      s_path_abs: item.source,
      s_name_file: (item.source || "").split("/").pop() || item.source,
      s_msg_error: item.error_msg || "",
      n_scl_x: dims.width || (pose && pose.image_width) || 0,
      n_scl_y: dims.height || (pose && pose.image_height) || 0,
      n_persons: a_o_person.length,
      a_o_person,
    };

    const procArgs = processorArgs(o_img, pose);

    let keep = true;
    let reason = "";
    for (const proc of compiled) {
      let result;
      try {
        result = proc.fn(...procArgs);
      } catch (e) {
        errors++;
        // A throwing processor neither keeps nor filters; record and continue.
        console.error(
          `Processor "${proc.name}" threw on ${o_img.s_name_file}:`,
          e,
        );
        continue;
      }
      if (!result) {
        keep = false;
        reason = `Filtered by "${proc.name}"`;
        break;
      }
    }

    if (keep) passed++;
    else filtered[item.result] = reason;
  }

  STATE.filteredOut = filtered;
  STATE.runningProcessors = false;
  STATE._compiledRun = null;
  setRunButton(false);
  renderImageList();

  const filteredCount = Object.keys(filtered).length;
  let summary =
    `Done: ${passed} kept, ${filteredCount} filtered out of ${items.length}.`;
  if (errors > 0) summary += ` ${errors} processor error(s) — see console.`;
  updateFilterStatusText(summary, filteredCount > 0);
}

function updateFilterStatusText(text, highlight) {
  const el = DOM.processorFilterStatus;
  if (!el) return;
  el.textContent = text;
  el.classList.toggle("has-filter", !!highlight);
}

// Resolve an image's natural dimensions, with a small cache so repeated runs
// don't reload the same images.
const _dimCache = {};
function loadImageDims(path) {
  if (_dimCache[path]) return Promise.resolve(_dimCache[path]);
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      const dims = { width: img.naturalWidth, height: img.naturalHeight };
      _dimCache[path] = dims;
      resolve(dims);
    };
    img.onerror = () => reject(new Error("Failed to load image"));
    img.src = `/api/image?path=${encodeURIComponent(path)}`;
  });
}

// Run `task` over `items` with at most `limit` in flight; returns results in
// the original order.
async function mapWithConcurrency(items, limit, task) {
  const results = new Array(items.length);
  let next = 0;
  async function worker() {
    while (next < items.length) {
      const i = next++;
      results[i] = await task(items[i], i);
    }
  }
  const workers = [];
  for (let i = 0; i < Math.min(limit, items.length); i++) {
    workers.push(worker());
  }
  await Promise.all(workers);
  return results;
}

// ============================================================================
// File Browser
// ============================================================================

const STATE_KEY = "pilotless_poses_state";

const FILE_BROWSER = {
  scanPath: "/",
  scanEntries: [],
};

function saveState() {
  const state = {
    scanPath: FILE_BROWSER.scanPath,
    extFilter: document.getElementById("ext-filter")?.value || ".jpg,.jpeg,.png",
    outputDir: document.getElementById("output-dir")?.value || "./pose_results/",
  };
  try { localStorage.setItem(STATE_KEY, JSON.stringify(state)); } catch {}
}

function loadState() {
  try {
    const raw = localStorage.getItem(STATE_KEY);
    if (!raw) return;
    const state = JSON.parse(raw);
    if (state.scanPath) FILE_BROWSER.scanPath = state.scanPath;
    const extInput = document.getElementById("ext-filter");
    if (extInput && state.extFilter) extInput.value = state.extFilter;
    const outInput = document.getElementById("output-dir");
    if (outInput && state.outputDir) outInput.value = state.outputDir;
    const scanInput = document.getElementById("folder-path");
    if (scanInput && state.scanPath) scanInput.value = state.scanPath;
  } catch {}
}

function initFileBrowser() {
  // Back button — scan page only
  document.getElementById("btn-back-scan").addEventListener("click", () => {
    const parent = parentPath(FILE_BROWSER.scanPath);
    if (parent !== null) navigateBrowserTo("scan", parent);
  });

  // Filter as you type
  const inputScan = document.getElementById("folder-path");
  inputScan.addEventListener("input", () => filterBrowserOnInput("scan"));

  // Initial listing loaded once WebSocket connects
}

function parentPath(path) {
  if (path === "/") return null;
  return path.split("/").slice(0, -1).join("/") || "/";
}

function navigateBrowserTo(page, path) {
  if (page === "scan") FILE_BROWSER.scanPath = path;
  saveState();

  const input = document.getElementById("folder-path");
  if (input) input.value = path;

  const list = document.getElementById("browser-list-scan");
  if (list) list.innerHTML = '<div class="browser-placeholder">Loading...</div>';

  if (!sendMessage({ type: "browse_folder", path, page })) {
    if (list) {
      list.innerHTML = '<div class="browser-placeholder">WebSocket not connected. Please wait...</div>';
    }
  }
}

// Called from handleMessage when a browse_result arrives
function handleBrowseResult(data) {
  const list = document.getElementById("browser-list-scan");
  if (!list) return;

  if (data.error) {
    list.innerHTML = `<div class="browser-placeholder">Error: ${escapeHtml(data.error)}</div>`;
    return;
  }

  FILE_BROWSER.scanPath = data.path;
  FILE_BROWSER.scanEntries = data.entries;

  const input = document.getElementById("folder-path");
  if (input) input.value = data.path;

  renderBrowserList(list, data, "");
}

function filterBrowserOnInput(page) {
  const input = document.getElementById("folder-path");
  const list = document.getElementById("browser-list-scan");
  if (!input || !list) return;

  const inputValue = input.value;
  const currentPath = FILE_BROWSER.scanPath;
  const entries = FILE_BROWSER.scanEntries;

  // Determine the filter text: everything after the current browser path
  let filter = "";
  if (inputValue.startsWith(currentPath)) {
    filter = inputValue.slice(currentPath.length);
    // Strip leading slash
    if (filter.startsWith("/")) filter = filter.slice(1);
  }

  // Filter entries client-side
  const filtered = filter
    ? entries.filter(e => e.name.toLowerCase().includes(filter.toLowerCase()))
    : entries;

  // Build a fake data object for rendering
  const data = {
    path: currentPath,
    page,
    entries: filtered,
  };

  renderBrowserList(list, data, filter);
}

function renderBrowserList(list, data, filter) {
  list.innerHTML = "";

  data.entries.forEach((entry) => {
    const div = document.createElement("div");
    div.className = entry.isDirectory ? "browser-entry" : "browser-entry is-file";

    const icon = document.createElement("span");
    icon.className = "entry-icon";
    icon.textContent = entry.isDirectory ? "📁" : "📄";

    const name = document.createElement("span");
    name.className = "entry-name";
    name.title = entry.name;

    // Highlight matching text if a filter is active
    if (filter && filter.length > 0) {
      name.innerHTML = highlightMatch(entry.name, filter);
    } else {
      name.textContent = entry.name;
    }

    div.appendChild(icon);
    div.appendChild(name);

    if (entry.isDirectory) {
      const targetPath = data.path === "/"
        ? `/${entry.name}`
        : `${data.path}/${entry.name}`;
      div.addEventListener("click", () => navigateBrowserTo(data.page || "scan", targetPath));
    }

    list.appendChild(div);
  });

  if (data.entries.length === 0) {
    const msg = filter
      ? `No entries matching "${escapeHtml(filter)}"`
      : "This directory is empty.";
    list.innerHTML = `<div class="browser-placeholder">${msg}</div>`;
  }
}

function highlightMatch(text, filter) {
  // Case-insensitive highlight of filter within text
  const escaped = filter.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const re = new RegExp(`(${escaped})`, "gi");
  return text.replace(re, '<mark class="browser-highlight">$1</mark>');
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

// ============================================================================
// Init
// ============================================================================

document.addEventListener("DOMContentLoaded", () => {
  loadState();
  initScanPage();
  initPreviewPage();
  initProcessorsPage();
  initFileBrowser();

  // Navigation
  document.querySelectorAll(".nav-tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      navigateTo(tab.dataset.page);
    });
  });

  // Connect WebSocket
  connectWebSocket();
});
