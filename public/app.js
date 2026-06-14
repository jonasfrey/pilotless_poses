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
  tags: {},              // result path -> [tag, …] (persisted) — now only the
                         // `${base}_positive` / `${base}_negative` basetag labels
  filterPositive: false, // quick-filter: show only positives for the basetag
  filterNegative: false, // quick-filter: show only negatives for the basetag
  exportFolder: "filtered_images", // destination for the Export action (persisted)
  exporting: false,      // true while an export_images request is in flight
  currentResultIndex: -1,
  currentPoseData: null,
  currentImage: null,    // HTMLImageElement
  showOverlay: true,
  showLabels: true,
  showInfo: false,
  showKeypoints: false,  // per-keypoint pixel coordinates in the info overlay
  overlayOpacity: 0.85,

  // Canvas zoom / pan (applied as a CSS transform on the canvas element)
  view: { scale: 1, tx: 0, ty: 0 },

  // Processor state
  processors: [],          // { id, name, code, active, createdAt, updatedAt }
  editingProcessorId: null, // id being edited, or "__new__" for an unsaved one
  filteredOut: {},         // result path -> reason string (image hidden in preview)
  showFiltered: false,     // when true, filtered images stay visible (dimmed)
  showFailed: false,       // when true, images whose inference failed stay visible
  invertFilter: false,     // when true, keep images that DON'T match the processors
  limitCount: 0,           // cap the kept set to this many images (0 = no limit)
  limitFilteredOut: {},    // result path -> reason for images hidden by the limit
  procResult: null,        // { matched:Set, reasons:{}, evaluated:[] } from the last run
  procCache: {},           // procId -> { version, sig, matched:[rp…] } (persisted)
  modelCache: {},          // modelId -> { version, sig, prob:{rp:number} } (persisted)
  runningProcessors: false,
  monacoEditor: null,
  monacoLoading: null,     // Promise while Monaco is loading

  // Basetag state
  basetags: [],                // { id, name, createdAt, updatedAt }
  editingBasetagId: null,      // id being edited, or "__new__" for an unsaved one
  pendingSelectBasetagName: null, // resolve a freshly-saved "__new__" to its id
  selectedBasetagId: null,     // basetag chosen in the preview (persisted)

  // Inference model state
  inferenceModels: [],         // { id, name, positiveDir, negativeDir, status, metrics, ... }
  editingInferenceId: null,    // id being edited, or "__new__" for an unsaved one
  inferenceFilteredOut: {},    // result path -> reason string (hidden by an applied model)
  appliedInferenceId: null,    // id of the model currently applied in the preview
  inferenceThreshold: 0.5,     // probability threshold for the preview filter
  trainingInferenceId: null,   // id currently training (Train button disabled)
  applyingInference: false,    // true while an apply_inference_model request is in flight
  dirBrowseTarget: null,       // input id the inline folder browser writes back to
  dirBrowsePath: "/",          // current path of the inline folder browser

  // Processor GUI (lil-gui). Controls are rebuilt each run, but their VALUES
  // persist here so a processor can read user-entered settings across runs.
  gui: null,                   // active lil-gui GUI instance (mounted in preview)
  guiControllers: {},          // control label -> lil-gui controller (this run)
  guiValues: {},               // control label -> current value (persistent)
  lilGuiLoading: null,         // Promise while lil-gui is loading from the CDN
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

    // Basetags are needed by the preview + inference selectors regardless of
    // which page is shown first, so load them up front.
    sendMessage({ type: "list_basetags" });
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

// ---- Processor GUI (lil-gui) ------------------------------------------------
//
// Processors can render persistent controls on the preview page with
// f_gui(label, default), which returns the control's current value. The raw
// lil-gui namespace is also injected as `lil_gui` for advanced use
// (new lil_gui.GUI(...)). Controls are rebuilt at the start of each run so only
// the ones used by the current pipeline show, but their VALUES persist in
// STATE.guiValues across images and runs — letting a processor read a setting
// (e.g. a destination folder) the user typed earlier.

const LIL_GUI_URL =
  "https://cdn.jsdelivr.net/npm/lil-gui@0.20/dist/lil-gui.umd.min.js";

// Load lil-gui from the CDN once. Resolves with the lil namespace, or null if
// it can't load (offline) — processors then just fall back to default values.
function ensureLilGui() {
  if (window.lil && window.lil.GUI) return Promise.resolve(window.lil);
  if (STATE.lilGuiLoading) return STATE.lilGuiLoading;
  STATE.lilGuiLoading = new Promise((resolve) => {
    const s = document.createElement("script");
    s.src = LIL_GUI_URL;
    s.onload = () => resolve(window.lil || null);
    s.onerror = () => {
      console.warn("lil-gui failed to load — processors will use default values.");
      resolve(null);
    };
    document.head.appendChild(s);
  });
  return STATE.lilGuiLoading;
}

// Tear down and recreate the shared GUI panel. Called once at the start of a
// run / refresh; f_gui() re-adds each control as it's first referenced, reusing
// the persisted value. The host stays hidden until a control is actually added.
// Prefers a lil-gui panel; falls back to native inputs if lil-gui isn't loaded
// (offline / CDN blocked) so the fields ALWAYS appear.
function rebuildProcessorGui() {
  const host = DOM.processorGuiHost;
  if (!host) return;
  host.innerHTML = "";
  if (STATE.gui) {
    try { STATE.gui.destroy(); } catch { /* already gone */ }
  }
  STATE.gui = null;
  STATE.guiControllers = {};
  host.classList.add("hidden");

  if (window.lil && window.lil.GUI) {
    try {
      STATE.gui = new window.lil.GUI({ container: host, title: "Processor controls" });
    } catch {
      STATE.gui = null;
    }
  }
  if (!STATE.gui) {
    // Native fallback: a titled container; f_gui() appends plain input rows.
    const title = document.createElement("div");
    title.className = "pg-title";
    title.textContent = "Processor controls";
    host.appendChild(title);
  }
}

// Append a native (non-lil-gui) control row to the host and bind it to
// STATE.guiValues[key]. Type follows the value's type.
function addNativeControl(key, value) {
  const host = DOM.processorGuiHost;
  const row = document.createElement("label");
  row.className = "pg-row";
  const span = document.createElement("span");
  span.className = "pg-label";
  span.textContent = key;
  span.title = key;
  const input = document.createElement("input");
  input.className = "pg-input";
  if (typeof value === "boolean") {
    input.type = "checkbox";
    input.checked = !!value;
    input.addEventListener("change", () => { STATE.guiValues[key] = input.checked; });
  } else if (typeof value === "number") {
    input.type = "number";
    input.value = String(value);
    input.addEventListener("input", () => {
      const n = parseFloat(input.value);
      STATE.guiValues[key] = Number.isNaN(n) ? 0 : n;
    });
  } else {
    input.type = "text";
    input.value = String(value);
    input.addEventListener("input", () => { STATE.guiValues[key] = input.value; });
  }
  row.appendChild(span);
  row.appendChild(input);
  host.appendChild(row);
  return row;
}

// Injected into processors as `f_gui`. Returns the current value of a
// persistent control labelled `label`, creating it (bound to STATE.guiValues)
// on first reference this run. The control type follows the default's type
// (string -> text, number -> number, boolean -> checkbox).
function f_gui(label, def) {
  const key = String(label);
  if (!(key in STATE.guiValues)) STATE.guiValues[key] = def;
  if (!STATE.guiControllers[key]) {
    if (STATE.gui) {
      try {
        STATE.guiControllers[key] = STATE.gui.add(STATE.guiValues, key);
      } catch { /* unsupported type — value still returned, just no control */ }
    } else if (DOM.processorGuiHost) {
      STATE.guiControllers[key] = addNativeControl(key, STATE.guiValues[key]);
    }
    if (STATE.guiControllers[key] && DOM.processorGuiHost) {
      DOM.processorGuiHost.classList.remove("hidden");
    }
  }
  return STATE.guiValues[key];
}

// Populate the Pipeline's "Processor controls" panel from the currently ACTIVE
// processors WITHOUT a full run, so f_gui() inputs (e.g. a filter's threshold or
// min-people value) appear as soon as a processor is active — letting the user
// set values before running. Each active processor is executed once against a placeholder
// image with inert side-effect helpers, purely to register its controls; any
// error (or a processor that only calls f_gui inside an image-dependent branch)
// is ignored. A real run re-registers with live values afterwards.
async function refreshProcessorGui() {
  if (STATE.runningProcessors) return; // don't disturb the GUI mid-run
  await ensureLilGui();
  rebuildProcessorGui();

  const active = (STATE.processors || []).filter((p) => p.active);
  if (active.length === 0) return;

  const o_img = {
    s_path_abs: "", s_name_file: "", s_msg_error: "",
    n_scl_x: 0, n_scl_y: 0, n_persons: 0, a_s_tag: [],
    a_o_person: [],
  };
  // Inert helpers: a probe must never touch the filesystem. They return a
  // pending promise so processors that chain .then(...) produce no side effects.
  const inert = () => new Promise(() => {});
  const probeArgs = [o_img, inert, inert, inert, inert, window.lil || null, f_gui];

  // Silence console during the dry pass so processors' own log lines don't spam
  // devtools on every navigate/toggle. Restored immediately after (the probe is
  // synchronous and the inert helpers never resolve, so nothing logs later).
  const saved = { log: console.log, warn: console.warn, error: console.error };
  console.log = console.warn = console.error = () => {};
  try {
    for (const p of active) {
      try {
        const fn = new Function(...PROCESSOR_ARG_NAMES, p.code);
        fn(...probeArgs);
      } catch {
        // Best-effort control registration — ignore processor errors here.
      }
    }
  } finally {
    console.log = saved.log;
    console.warn = saved.warn;
    console.error = saved.error;
  }
}

// Argument names (and order) injected into every processor function. Keep the
// compile sites and the call site in sync via this single list.
const PROCESSOR_ARG_NAMES = [
  "o_img",
  "f_deno_write_file",
  "f_save_image",
  "f_save_json",
  "f_save_filtered",
  "lil_gui",
  "f_gui",
];

// Produce the concrete argument values for a processor call on one image.
function processorArgs(o_img, pose) {
  const h = makeImageHelpers(o_img, pose);
  return [
    o_img,
    f_deno_write_file,
    h.f_save_image,
    h.f_save_json,
    h.f_save_filtered,
    window.lil || null,
    f_gui,
  ];
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
    case "inference_models_list":
      handleInferenceModelsList(msg);
      break;
    case "inference_model_trained":
      handleInferenceModelTrained(msg);
      break;
    case "inference_model_applied":
      handleInferenceModelApplied(msg);
      break;
    case "all_pose_data":
      handleAllPoseData(msg);
      break;
    case "write_file_result":
      handleWriteFileResult(msg);
      break;
    case "export_result":
      handleExportResult(msg);
      break;
    case "unexport_result":
      handleUnexportResult(msg);
      break;
    case "basetags_list":
      handleBasetagsList(msg);
      break;
    case "scan_log":
      handleScanLog(msg);
      break;
    case "scan_status":
      handleScanStatus(msg);
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
  setLogStatus("idle");

  if (DOM.progressSummary) {
    const ok = msg.total - STATE.errors.length;
    DOM.progressSummary.textContent =
      `Batch complete: ${ok} of ${msg.total} images processed successfully. ${STATE.errors.length} errors.`;
  }

  // A scan changes what's on disk. If the preview is currently showing, re-fetch
  // results (and the processor / model lists) so it reflects the new state
  // instead of waiting for the user to re-navigate to the tab.
  const previewActive =
    document.getElementById("page-preview")?.classList.contains("active");
  if (previewActive) {
    loadResults();
    sendMessage({ type: "list_processors" });
    sendMessage({ type: "list_inference_models" });
  }
}

function handleResultsList(msg) {
  STATE.results = msg.results || [];

  // A fresh result set invalidates any prior processor / inference filtering:
  // those maps are keyed by result path and were computed against the OLD set,
  // so they'd otherwise keep hiding images that were never re-evaluated. Clear
  // them and the inference-filter UI so the preview reflects the server's actual
  // state; the user re-runs processors / re-applies a model explicitly.
  STATE.filteredOut = {};
  STATE.inferenceFilteredOut = {};
  STATE.limitFilteredOut = {};
  STATE.procResult = null; // last run no longer applies to this result set
  STATE.appliedInferenceId = null;
  if (DOM.btnClearInference) DOM.btnClearInference.classList.add("hidden");
  setInferenceFilterStatus("");

  console.log(
    "[handleResultsList] received",
    STATE.results.length,
    "result(s) from server.",
  );
  if (STATE.results.length === 0) {
    console.warn(
      "[handleResultsList] Server returned 0 results. Either the manifest is empty/missing " +
        "or its source_folder didn't match the requested folder. Check pose_results/manifest.json on this machine.",
    );
  } else {
    const byStatus = STATE.results.reduce((acc, r) => {
      acc[r.status] = (acc[r.status] || 0) + 1;
      return acc;
    }, {});
    console.log("[handleResultsList] status breakdown:", byStatus);
    console.log(
      "[handleResultsList] first result:",
      JSON.stringify(STATE.results[0]),
    );
  }
  applyAllFilters();
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
  // Surface server errors on the live log too, even outside an active scan.
  appendLogLine({ level: "error", line: msg.message });
}

// ============================================================================
// Live Log Page
// ============================================================================

function initLogPage() {
  DOM.liveLog = document.getElementById("live-log");
  DOM.logStatus = document.getElementById("log-status");
  DOM.logAutoscroll = document.getElementById("chk-log-autoscroll");
  DOM.btnClearLog = document.getElementById("btn-clear-log");
  DOM.logOverlay = document.getElementById("log-overlay");
  DOM.btnToggleLog = document.getElementById("btn-toggle-log");
  DOM.btnCloseLog = document.getElementById("btn-close-log");

  if (DOM.btnClearLog) DOM.btnClearLog.addEventListener("click", clearLog);
  if (DOM.btnToggleLog) DOM.btnToggleLog.addEventListener("click", toggleLogOverlay);
  if (DOM.btnCloseLog) DOM.btnCloseLog.addEventListener("click", hideLogOverlay);
  // Esc closes the overlay.
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && isLogOverlayOpen()) hideLogOverlay();
  });
}

// ---- Live Log overlay (toggled from the header, visible on any page) --------

function isLogOverlayOpen() {
  return !!(DOM.logOverlay && !DOM.logOverlay.classList.contains("hidden"));
}

function showLogOverlay() {
  if (!DOM.logOverlay) return;
  DOM.logOverlay.classList.remove("hidden");
  if (DOM.btnToggleLog) DOM.btnToggleLog.classList.add("active");
  if (DOM.liveLog && (!DOM.logAutoscroll || DOM.logAutoscroll.checked)) {
    DOM.liveLog.scrollTop = DOM.liveLog.scrollHeight;
  }
}

function hideLogOverlay() {
  if (!DOM.logOverlay) return;
  DOM.logOverlay.classList.add("hidden");
  if (DOM.btnToggleLog) DOM.btnToggleLog.classList.remove("active");
}

function toggleLogOverlay() {
  if (isLogOverlayOpen()) hideLogOverlay();
  else showLogOverlay();
}

function setLogStatus(state) {
  // state: "running" | "idle"
  if (DOM.logStatus) {
    const running = state === "running";
    DOM.logStatus.textContent = running ? "● Scan running" : "Idle";
    DOM.logStatus.classList.toggle("running", running);
    DOM.logStatus.classList.toggle("idle", !running);
  }
  // Pulse the header toggle while a scan is live so activity shows from any page.
  if (DOM.btnToggleLog) {
    DOM.btnToggleLog.classList.toggle("live", state === "running");
  }
}

function clearLog() {
  if (!DOM.liveLog) return;
  DOM.liveLog.innerHTML =
    '<div class="live-log-placeholder">Log cleared.</div>';
}

function handleScanLog(msg) {
  appendLogLine(msg);
}

// Sent by the server right after a (re)connection. Restores the live log and
// progress for a scan that may have started before this page was loaded, so a
// reload mid-scan doesn't show an empty log.
function handleScanStatus(msg) {
  // Replace the live log with the server's buffered lines.
  if (DOM.liveLog && Array.isArray(msg.logs)) {
    DOM.liveLog.innerHTML = "";
    if (msg.logs.length === 0) {
      DOM.liveLog.innerHTML =
        '<div class="live-log-placeholder">No activity yet. Start a scan to see live progress.</div>';
    } else {
      for (const line of msg.logs) appendLogLine(line);
    }
  }

  STATE.scanning = !!msg.scanning;
  STATE.totalFiles = msg.total || 0;
  STATE.processedFiles = msg.current || 0;

  updateScanButtons(STATE.scanning);
  setLogStatus(STATE.scanning ? "running" : "idle");

  if (msg.total > 0) {
    const pct = Math.round((msg.current / msg.total) * 100);
    if (DOM.progressBar) DOM.progressBar.style.width = `${pct}%`;
    if (DOM.progressText) DOM.progressText.textContent = `${msg.current} / ${msg.total}`;
    if (DOM.progressSummary && STATE.scanning) {
      DOM.progressSummary.textContent =
        `Scan in progress: ${msg.current} of ${msg.total} processed…`;
    }
  }
}

function appendLogLine(msg) {
  if (!DOM.liveLog) return;

  const placeholder = DOM.liveLog.querySelector(".live-log-placeholder");
  if (placeholder) placeholder.remove();

  const level = msg.level || "info";
  const row = document.createElement("div");
  row.className = `live-log-line level-${level}`;

  const timeEl = document.createElement("span");
  timeEl.className = "log-time";
  const d = msg.ts ? new Date(msg.ts) : new Date();
  timeEl.textContent = isNaN(d.getTime())
    ? ""
    : d.toLocaleTimeString([], { hour12: false });

  const textEl = document.createElement("span");
  textEl.className = "log-text";
  textEl.textContent = msg.line ?? "";

  row.appendChild(timeEl);
  row.appendChild(textEl);
  DOM.liveLog.appendChild(row);

  // Cap the DOM size so a huge download log doesn't grow unbounded.
  const MAX_LINES = 2000;
  while (DOM.liveLog.childElementCount > MAX_LINES) {
    DOM.liveLog.firstElementChild.remove();
  }

  if (!DOM.logAutoscroll || DOM.logAutoscroll.checked) {
    DOM.liveLog.scrollTop = DOM.liveLog.scrollHeight;
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
  // processor list so the pipeline column reflects current active processors,
  // and the inference models so the filter column lists trained models.
  if (page === "preview") {
    loadResults();
    sendMessage({ type: "list_processors" });
    sendMessage({ type: "list_inference_models" });
    // Show processor input fields right away from the cached processor list;
    // the list_processors response refreshes them again when it arrives.
    refreshProcessorGui();
  }

  // Load processors and bootstrap Monaco when entering the processors page
  if (page === "processors") {
    sendMessage({ type: "list_processors" });
    ensureMonaco();
  }

  // Load inference models when entering the inference models page
  if (page === "inference") {
    sendMessage({ type: "list_inference_models" });
    sendMessage({ type: "list_basetags" });
  }

  // Load basetags when entering the basetags page
  if (page === "basetags") {
    sendMessage({ type: "list_basetags" });
  }
}

// ============================================================================
// Page 1: Folder Scan
// ============================================================================

function initScanPage() {
  DOM.folderPath = document.getElementById("folder-path");
  DOM.extFilter = document.getElementById("ext-filter");
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
  if (DOM.liveLog) DOM.liveLog.innerHTML = "";

  const sent = sendMessage({
    type: "scan_folder",
    folderPath,
    extensions,
  });

  if (!sent) {
    alert("WebSocket not connected. Please wait for reconnection and try again.");
    STATE.scanning = false;
    updateScanButtons(false);
    return;
  }

  // Surface live server activity: mark the log running and open the overlay so
  // the user sees what's happening during the (potentially long) scan.
  setLogStatus("running");
  showLogOverlay();
}

function cancelScan() {
  sendMessage({ type: "cancel_scan" });
  STATE.scanning = false;
  updateScanButtons(false);
  setLogStatus("idle");
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
  DOM.posCount = document.getElementById("pos-count");
  DOM.negCount = document.getElementById("neg-count");
  DOM.btnFilterPos = document.getElementById("btn-filter-pos");
  DOM.btnFilterNeg = document.getElementById("btn-filter-neg");
  DOM.currentLabel = document.getElementById("current-label");
  DOM.btnRunProcessors = document.getElementById("btn-run-processors");
  DOM.chkShowFiltered = document.getElementById("chk-show-filtered");
  DOM.chkShowFailed = document.getElementById("chk-show-failed");
  DOM.processorFilterStatus = document.getElementById("processor-filter-status");
  DOM.processorOrderList = document.getElementById("processor-order-list");
  DOM.processorOrderCount = document.getElementById("processor-order-count");
  DOM.processorGuiHost = document.getElementById("processor-gui-host");
  DOM.viewerPlaceholder = document.getElementById("viewer-placeholder");
  DOM.canvasContainer = document.getElementById("canvas-container");
  DOM.poseCanvas = document.getElementById("pose-canvas");
  DOM.canvasWrap = document.getElementById("canvas-wrap");
  DOM.canvasInfo = document.getElementById("canvas-info");

  DOM.btnToggleOverlay = document.getElementById("btn-toggle-overlay");
  DOM.btnToggleLabels = document.getElementById("btn-toggle-labels");
  DOM.btnToggleInfo = document.getElementById("btn-toggle-info");
  DOM.btnToggleKeypoints = document.getElementById("btn-toggle-keypoints");
  DOM.btnResetView = document.getElementById("btn-reset-view");
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
  DOM.btnToggleKeypoints.addEventListener("click", toggleKeypoints);

  // Quick-filter toggles: restrict the list to positives and/or negatives for
  // the selected basetag (OR across the two when both are on).
  if (DOM.btnFilterPos) {
    DOM.btnFilterPos.addEventListener("click", () => toggleLabelFilter("pos"));
  }
  if (DOM.btnFilterNeg) {
    DOM.btnFilterNeg.addEventListener("click", () => toggleLabelFilter("neg"));
  }
  DOM.btnResetView.addEventListener("click", resetView);
  initCanvasZoomPan();
  DOM.opacitySlider.addEventListener("input", () => {
    STATE.overlayOpacity = parseFloat(DOM.opacitySlider.value);
    renderPoseCanvas();
  });

  // Navigation
  DOM.btnPrevImage.addEventListener("click", () => navigateImages(-1));
  DOM.btnNextImage.addEventListener("click", () => navigateImages(1));

  // Export action (copies the kept set to a folder, server-side)
  DOM.exportFolder = document.getElementById("export-folder");
  DOM.btnExportImages = document.getElementById("btn-export-images");
  DOM.exportStatus = document.getElementById("export-status");
  if (DOM.exportFolder && STATE.exportFolder) DOM.exportFolder.value = STATE.exportFolder;
  if (DOM.btnExportImages) DOM.btnExportImages.addEventListener("click", exportImages);
  if (DOM.exportFolder) {
    DOM.exportFolder.addEventListener("input", () => {
      STATE.exportFolder = DOM.exportFolder.value;
      saveState();
    });
  }

  // Processor run + filter visibility
  DOM.btnRunProcessors.addEventListener("click", runProcessors);
  DOM.chkShowFiltered.addEventListener("change", () => {
    STATE.showFiltered = DOM.chkShowFiltered.checked;
    renderImageList();
  });
  DOM.chkShowFailed.addEventListener("change", () => {
    STATE.showFailed = DOM.chkShowFailed.checked;
    renderImageList();
  });

  // Invert + limit — re-derive the filter instantly, no re-run needed.
  DOM.chkInvertFilter = document.getElementById("chk-invert-filter");
  DOM.limitCount = document.getElementById("limit-count");
  if (DOM.chkInvertFilter) {
    DOM.chkInvertFilter.checked = !!STATE.invertFilter;
    DOM.chkInvertFilter.addEventListener("change", () => {
      STATE.invertFilter = DOM.chkInvertFilter.checked;
      saveState();
      applyAllFilters();
    });
  }
  if (DOM.limitCount) {
    DOM.limitCount.value = String(STATE.limitCount || 0);
    DOM.limitCount.addEventListener("input", () => {
      const n = parseInt(DOM.limitCount.value, 10);
      STATE.limitCount = Number.isNaN(n) || n < 0 ? 0 : n;
      saveState();
      applyAllFilters();
    });
  }

  // Inference-model filter column (preview sidebar)
  DOM.inferenceModelChoices = document.getElementById("inference-model-choices");
  DOM.inferenceFilterCount = document.getElementById("inference-filter-count");
  DOM.inferenceThreshold = document.getElementById("inference-threshold");
  DOM.inferenceThresholdValue = document.getElementById("inference-threshold-value");
  DOM.btnApplyInference = document.getElementById("btn-apply-inference");
  DOM.btnClearInference = document.getElementById("btn-clear-inference");
  DOM.inferenceFilterStatus = document.getElementById("inference-filter-status");

  DOM.inferenceThreshold.addEventListener("input", () => {
    STATE.inferenceThreshold = parseFloat(DOM.inferenceThreshold.value);
    DOM.inferenceThresholdValue.textContent = STATE.inferenceThreshold.toFixed(2);
  });
  DOM.btnApplyInference.addEventListener("click", applyInferenceModel);
  DOM.btnClearInference.addEventListener("click", clearInferenceFilter);

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
  // The pose preview always shows ALL scanned results from the server's
  // ./pose_results manifest — it does not depend on the folder currently
  // selected on the Folder Scan page.
  console.log("[loadResults] requesting all scanned pose results");
  const sent = sendMessage({ type: "get_results" });
  if (!sent) {
    console.warn(
      "[loadResults] WebSocket not connected — get_results was NOT sent. ws.readyState =",
      STATE.ws ? STATE.ws.readyState : "no ws",
    );
  }
}

function isFilteredOut(result) {
  return Object.prototype.hasOwnProperty.call(STATE.filteredOut, result.result) ||
    Object.prototype.hasOwnProperty.call(STATE.inferenceFilteredOut, result.result) ||
    Object.prototype.hasOwnProperty.call(STATE.limitFilteredOut, result.result);
}

// Reason an image is hidden by a processor, applied model, or the limit (or "").
function filterReason(result) {
  return STATE.filteredOut[result.result] ||
    STATE.inferenceFilteredOut[result.result] ||
    STATE.limitFilteredOut[result.result] || "";
}

function isFailed(result) {
  return result.status !== "ok";
}

// ---- Derived processor / invert / limit filtering ---------------------------
//
// The processor run records which images "matched" (passed every active
// processor); the visible filter is DERIVED from that so the Invert and Limit
// controls re-apply instantly without re-running the pipeline.

// Rebuild STATE.filteredOut (the processor filter) from the last run + invert.
function recomputeProcessorFilter() {
  const res = STATE.procResult;
  if (!res) { STATE.filteredOut = {}; return; }
  const filtered = {};
  for (const rp of res.evaluated) {
    const matched = res.matched.has(rp);
    const hide = STATE.invertFilter ? matched : !matched;
    if (hide) {
      filtered[rp] = STATE.invertFilter
        ? "Matched (inverted filter)"
        : (res.reasons[rp] || "Filtered by processors");
    }
  }
  STATE.filteredOut = filtered;
}

// Cap the kept set (passed processor + inference filters, not failed) to
// STATE.limitCount, hiding the overflow. 0 = no limit.
function recomputeLimit() {
  const over = {};
  const limit = STATE.limitCount | 0;
  if (limit > 0) {
    let kept = 0;
    for (const r of STATE.results) {
      if (isFailed(r)) continue;
      if (
        Object.prototype.hasOwnProperty.call(STATE.filteredOut, r.result) ||
        Object.prototype.hasOwnProperty.call(STATE.inferenceFilteredOut, r.result)
      ) continue;
      kept++;
      if (kept > limit) over[r.result] = `Over limit (${limit})`;
    }
  }
  STATE.limitFilteredOut = over;
}

// Recompute every derived filter and repaint the list.
function applyAllFilters() {
  recomputeProcessorFilter();
  recomputeLimit();
  renderImageList();
}

// ---- Tagging ----------------------------------------------------------------
//
// Each image can carry multiple string tags (e.g. "marked_hands_in_the_air").
// The "Active tag" box names the tag that "m" / the row ★ toggles on/off, so the
// user can curate several independent sets. Tags are keyed by result path and
// persisted in localStorage (result paths are stable across reloads/re-scans),
// and surfaced to processors as o_img.a_s_tag (see the has_tag processor).

const TAGS_KEY = "pilotless_poses_tags";
const MARKS_KEY = "pilotless_poses_marks"; // legacy boolean marks (migrated once)

function loadTags() {
  try {
    const raw = localStorage.getItem(TAGS_KEY);
    STATE.tags = raw ? (JSON.parse(raw) || {}) : {};
  } catch {
    STATE.tags = {};
  }
  // One-time migration from the old boolean marks → a "marked" tag.
  if (Object.keys(STATE.tags).length === 0) {
    try {
      const old = JSON.parse(localStorage.getItem(MARKS_KEY) || "null");
      if (old && typeof old === "object") {
        for (const key of Object.keys(old)) {
          if (old[key]) STATE.tags[key] = ["marked"];
        }
        if (Object.keys(STATE.tags).length) saveTags();
      }
    } catch { /* no legacy marks */ }
  }
}

function saveTags() {
  try {
    localStorage.setItem(TAGS_KEY, JSON.stringify(STATE.tags));
  } catch { /* storage full / unavailable — tags stay in-memory only */ }
}

function tagsFor(result) {
  return (result && STATE.tags[result.result]) || [];
}

function hasTag(result, tag) {
  return !!tag && tagsFor(result).includes(tag);
}

// Add/remove `tag` on an image; cleans up empty arrays so untagged images leave
// no entry behind.
function toggleTag(result, tag) {
  if (!result || !tag) return;
  const list = STATE.tags[result.result] ? [...STATE.tags[result.result]] : [];
  const i = list.indexOf(tag);
  if (i === -1) list.push(tag);
  else list.splice(i, 1);
  if (list.length) STATE.tags[result.result] = list;
  else delete STATE.tags[result.result];
  saveTags();
}

// ---- Basetag positive/negative labels ---------------------------------------
//
// Images carry only `${base}_positive` / `${base}_negative` tags now (set with
// ↑ / ↓ in labelCurrent). These helpers surface that state in the list rows, the
// header counts, the toolbar badge, and the quick-filter buttons.

// "pos" | "neg" | null for the currently-selected basetag.
function labelStateFor(result) {
  const base = selectedBasetagName();
  if (!base || !result) return null;
  if (hasTag(result, `${base}_positive`)) return "pos";
  if (hasTag(result, `${base}_negative`)) return "neg";
  return null;
}

// Header badges: how many images are positive / negative for the basetag.
function updateLabelCounts() {
  const base = selectedBasetagName();
  let pos = 0, neg = 0;
  if (base) {
    for (const r of STATE.results) {
      if (hasTag(r, `${base}_positive`)) pos++;
      else if (hasTag(r, `${base}_negative`)) neg++;
    }
  }
  if (DOM.posCount) {
    DOM.posCount.textContent = `↑ ${pos}`;
    DOM.posCount.title = `${pos} image(s) marked “${base}_positive”`;
    DOM.posCount.classList.toggle("hidden", !base);
  }
  if (DOM.negCount) {
    DOM.negCount.textContent = `↓ ${neg}`;
    DOM.negCount.title = `${neg} image(s) marked “${base}_negative”`;
    DOM.negCount.classList.toggle("hidden", !base);
  }
}

// The toolbar badge showing the CURRENT image's label for the basetag.
function updateCurrentLabel() {
  const el = DOM.currentLabel;
  if (!el) return;
  const base = selectedBasetagName();
  const r = STATE.results[STATE.currentResultIndex];
  const state = labelStateFor(r);
  el.classList.remove("pos", "neg", "none");
  if (!base) {
    el.classList.add("none");
    el.textContent = "No basetag";
  } else if (state === "pos") {
    el.classList.add("pos");
    el.textContent = "↑ Positive";
  } else if (state === "neg") {
    el.classList.add("neg");
    el.textContent = "↓ Negative";
  } else {
    el.classList.add("none");
    el.textContent = "Unlabelled";
  }
}

// Paint one row's positive/negative indicator + tint.
function renderRowLabel(div, result) {
  const state = labelStateFor(result);
  div.classList.toggle("row-pos", state === "pos");
  div.classList.toggle("row-neg", state === "neg");
  const ind = div.querySelector(".label-indicator");
  if (ind) {
    ind.classList.remove("pos", "neg", "none");
    ind.classList.add(state || "none");
    ind.textContent = state === "pos" ? "↑" : state === "neg" ? "↓" : "";
    ind.title = state === "pos"
      ? "Positive"
      : state === "neg"
      ? "Negative"
      : "Unlabelled";
  }
}

// Update one row in place after labelling (no full re-render, so the list
// doesn't jump-scroll). Also refreshes the toolbar badge if it's the current row.
function refreshRowTagUI(index) {
  const r = STATE.results[index];
  const div = document.querySelector(`.image-item[data-index="${index}"]`);
  if (!r || !div) return;
  renderRowLabel(div, r);
  if (index === STATE.currentResultIndex) updateCurrentLabel();
}

// Toggle a quick-filter button and re-render the list under the new filter.
function toggleLabelFilter(which) {
  if (which === "pos") STATE.filterPositive = !STATE.filterPositive;
  else STATE.filterNegative = !STATE.filterNegative;
  saveState();
  updateLabelFilterButtons();
  renderImageList();
}

// Reflect the quick-filter buttons' active/disabled state (disabled when no
// basetag is selected — there is nothing to filter on).
function updateLabelFilterButtons() {
  const base = selectedBasetagName();
  if (DOM.btnFilterPos) {
    DOM.btnFilterPos.disabled = !base;
    DOM.btnFilterPos.classList.toggle("active", !!base && STATE.filterPositive);
  }
  if (DOM.btnFilterNeg) {
    DOM.btnFilterNeg.disabled = !base;
    DOM.btnFilterNeg.classList.toggle("active", !!base && STATE.filterNegative);
  }
}

// Re-sync everything that depends on the selected basetag: the quick-filter
// buttons, the row indicators + header counts (via renderImageList), and the
// toolbar badge. Called when the basetag selection changes or basetags load.
function refreshBasetagLabelUI() {
  updateLabelFilterButtons();
  renderImageList();
  updateCurrentLabel();
}

// ---- Export -----------------------------------------------------------------
//
// A first-class action (not a processor): copy the images that passed the
// active filters/marks — and their pose JSON — into a folder. The server does
// the copy with Deno.copyFile and reports a summary. The "what" is whatever the
// filters kept; Export just decides "where" and "when".

// Images that survived the active filters: not failed, not filtered out. This
// ignores the "show filtered/failed" view toggles — those only affect what's
// visible, not what counts as kept.
function keptResults() {
  return STATE.results.filter((r) => !isFailed(r) && !isFilteredOut(r));
}

function updateExportButton() {
  const btn = DOM.btnExportImages;
  if (!btn) return;
  if (STATE.exporting) return; // leave the "Exporting…" label alone
  const n = keptResults().length;
  btn.textContent = n > 0 ? `Export ${n}` : "Export";
  btn.disabled = n === 0;
}

function exportImages() {
  if (STATE.exporting) return;
  const folder = (DOM.exportFolder?.value || "").trim();
  if (!folder) {
    setExportStatus("Enter a folder name first.", true);
    return;
  }
  const kept = keptResults();
  if (kept.length === 0) {
    setExportStatus("Nothing to export — no images passed the filters.", true);
    return;
  }

  const items = kept.map((r) => ({ source: r.source, result: r.result }));
  const sent = sendMessage({ type: "export_images", folder, items });
  if (!sent) {
    setExportStatus("Not connected — could not export.", true);
    return;
  }

  STATE.exporting = true;
  STATE.exportFolder = folder;
  saveState();
  if (DOM.btnExportImages) {
    DOM.btnExportImages.disabled = true;
    DOM.btnExportImages.textContent = "Exporting…";
  }
  setExportStatus(`Exporting ${items.length} image(s) to ./${folder}…`);
}

function handleExportResult(msg) {
  STATE.exporting = false;
  updateExportButton();
  if (msg.failed > 0) {
    const first = (msg.errors && msg.errors[0]) || null;
    const detail = first ? ` First error (${first.file}): ${first.error}` : "";
    setExportStatus(
      `Exported ${msg.copied} to ./${msg.folder}, ${msg.failed} failed.${detail}`,
      true,
    );
  } else {
    setExportStatus(`Exported ${msg.copied} image(s) to ./${msg.folder}.`);
  }
}

function setExportStatus(text, highlight) {
  const el = DOM.exportStatus;
  if (!el) return;
  el.textContent = text || "";
  el.classList.toggle("has-filter", !!highlight);
}

// The images currently shown in the sidebar, in order. Navigation (arrow keys
// and the prev/next buttons) walks THIS list, not STATE.results — so once
// processors filter images out, the arrows only step through what's visible.
//
// Visibility rules:
//  - Failed images (inference errored) are hidden unless "Show failed" is on.
//  - Processor- / inference-filtered images are hidden unless "Show filtered"
//    is on (then they remain but are dimmed).
function visibleResults() {
  const base = selectedBasetagName();
  const labelFilter = base && (STATE.filterPositive || STATE.filterNegative);
  return STATE.results.filter((r) => {
    if (isFailed(r) && !STATE.showFailed) return false;
    if (isFilteredOut(r) && !STATE.showFiltered) return false;
    // Quick basetag-label filter: keep positives and/or negatives (OR when both
    // toggles are on). Only applies when a basetag is selected.
    if (labelFilter) {
      const state = labelStateFor(r);
      const keep = (STATE.filterPositive && state === "pos") ||
        (STATE.filterNegative && state === "neg");
      if (!keep) return false;
    }
    return true;
  });
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

  const visible = visibleResults();

  console.log(
    `[renderImageList] ${STATE.results.length} total, ${
      Object.keys(STATE.filteredOut).length
    } filtered by processors, ${
      Object.keys(STATE.inferenceFilteredOut).length
    } filtered by inference, ${visible.length} visible ` +
      `(showFiltered=${STATE.showFiltered}, showFailed=${STATE.showFailed}).`,
  );

  DOM.imageCount.textContent = String(visible.length);
  updateFilterStatus();
  updateLabelCounts();
  updateExportButton();

  if (visible.length === 0) {
    DOM.imageList.innerHTML =
      '<div class="placeholder">No images to show with the current filters.</div>';
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
    thumb.onerror = () => {
      console.warn(
        "[thumbnail] failed to load source image:", r.source,
        "\n  → The server could not serve this path. On a different machine the manifest's",
        "absolute source paths often don't exist (different home dir / checkout location).",
      );
      thumb.src = "";
    };

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
      tag.title = filterReason(r) || "Filtered out";
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

    // Positive/negative indicator (↑ / ↓ / none) for the selected basetag.
    const ind = document.createElement("span");
    ind.className = "label-indicator";
    div.appendChild(ind);
    renderRowLabel(div, r);

    div.addEventListener("click", () => selectImage(idx));

    DOM.imageList.appendChild(div);
  });
}

function updateFilterStatus() {
  const el = DOM.processorFilterStatus;
  if (!el) return;
  const total = STATE.results.length;
  const filtered = STATE.results.filter(isFilteredOut).length;
  const failed = STATE.results.filter(isFailed).length;

  const parts = [];
  if (filtered > 0) {
    parts.push(
      `${filtered} of ${total} filtered out` +
        (STATE.showFiltered ? " (shown, dimmed)" : ""),
    );
  }
  if (failed > 0) {
    parts.push(
      `${failed} failed` + (STATE.showFailed ? " (shown)" : " (hidden)"),
    );
  }

  if (parts.length === 0) {
    el.textContent = total > 0 ? `${total} image${total !== 1 ? "s" : ""}` : "";
    el.classList.remove("has-filter");
  } else {
    el.textContent = parts.join(" · ");
    el.classList.add("has-filter");
  }
}

function selectImage(index) {
  STATE.currentResultIndex = index;
  STATE.currentPoseData = null;
  STATE.currentImage = null;
  // Start each image fitted, not at the previous image's zoom/pan.
  STATE.view = { scale: 1, tx: 0, ty: 0 };

  // Update selection highlight
  document.querySelectorAll(".image-item").forEach((el) => el.classList.remove("active"));
  const item = document.querySelector(`.image-item[data-index="${index}"]`);
  if (item) item.classList.add("active");

  // Reflect this image's positive/negative label on the toolbar badge.
  updateCurrentLabel();

  // Update nav display — show position within the visible (shown) images, not
  // the full result set, so it matches what the arrows actually step through.
  if (DOM.navIndex) {
    const visible = visibleResults();
    const vpos = visible.indexOf(STATE.results[index]);
    DOM.navIndex.textContent = vpos >= 0
      ? `${vpos + 1} / ${visible.length}`
      : `– / ${visible.length}`;
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
    console.warn(
      "[selectImage] failed to load source image:", result.source,
      "— the server returned an error for /api/image. The path likely does not exist on this machine.",
    );
    STATE.currentImage = null;
    STATE.currentPoseData = null;
    clearCanvasWithMessage("Failed to load image.");
  };
  console.log("[selectImage] loading image", result.source, "pose result:", result.result);
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

  applyView();
  updateInfoOverlay();
}

// ---- Canvas zoom & pan ----
//
// Zoom/pan is a pure CSS transform on the <canvas> element (the pose pixels are
// still drawn 1:1, the browser just scales the result). transform-origin stays
// at the canvas centre; tx/ty are screen-pixel pan offsets.

const VIEW_MIN_SCALE = 0.25;
const VIEW_MAX_SCALE = 20;

function applyView() {
  const canvas = DOM.poseCanvas;
  if (!canvas) return;
  const { scale, tx, ty } = STATE.view;
  canvas.style.transformOrigin = "center center";
  canvas.style.transform = `translate(${tx}px, ${ty}px) scale(${scale})`;
  if (DOM.canvasWrap) {
    DOM.canvasWrap.classList.toggle("zoomed", scale !== 1 || tx !== 0 || ty !== 0);
  }
}

function resetView() {
  STATE.view = { scale: 1, tx: 0, ty: 0 };
  applyView();
}

function zoomAt(clientX, clientY, factor) {
  const canvas = DOM.poseCanvas;
  if (!canvas || !STATE.currentImage) return;
  const v = STATE.view;
  const next = Math.min(VIEW_MAX_SCALE, Math.max(VIEW_MIN_SCALE, v.scale * factor));
  if (next === v.scale) return;

  // Keep the point under the cursor fixed. Derive the canvas centre in screen
  // coordinates from its live bounding rect (which already includes the current
  // transform), then solve for the new translation.
  const rect = canvas.getBoundingClientRect();
  const centerX = rect.left + rect.width / 2;
  const centerY = rect.top + rect.height / 2;
  const ux = clientX - centerX + v.tx;
  const uy = clientY - centerY + v.ty;
  const ratio = next / v.scale;
  STATE.view = {
    scale: next,
    tx: ux - (ux - v.tx) * ratio,
    ty: uy - (uy - v.ty) * ratio,
  };
  applyView();
}

function initCanvasZoomPan() {
  const wrap = DOM.canvasWrap;
  if (!wrap) return;

  wrap.addEventListener("wheel", (e) => {
    if (!STATE.currentImage) return;
    e.preventDefault();
    // Trackpads send many small deltas; exponentiate for smooth, symmetric zoom.
    const factor = Math.exp(-e.deltaY * 0.0015);
    zoomAt(e.clientX, e.clientY, factor);
  }, { passive: false });

  let panning = false;
  let lastX = 0;
  let lastY = 0;

  wrap.addEventListener("pointerdown", (e) => {
    if (!STATE.currentImage || e.button !== 0) return;
    panning = true;
    lastX = e.clientX;
    lastY = e.clientY;
    wrap.classList.add("panning");
    wrap.setPointerCapture(e.pointerId);
  });
  wrap.addEventListener("pointermove", (e) => {
    if (!panning) return;
    STATE.view.tx += e.clientX - lastX;
    STATE.view.ty += e.clientY - lastY;
    lastX = e.clientX;
    lastY = e.clientY;
    applyView();
  });
  const endPan = (e) => {
    if (!panning) return;
    panning = false;
    wrap.classList.remove("panning");
    try { wrap.releasePointerCapture(e.pointerId); } catch { /* ignore */ }
  };
  wrap.addEventListener("pointerup", endPan);
  wrap.addEventListener("pointercancel", endPan);

  // Double-click toggles between fit (reset) and 2× at the cursor.
  wrap.addEventListener("dblclick", (e) => {
    if (!STATE.currentImage) return;
    e.preventDefault();
    if (STATE.view.scale !== 1 || STATE.view.tx !== 0 || STATE.view.ty !== 0) {
      resetView();
    } else {
      zoomAt(e.clientX, e.clientY, 2);
    }
  });
}

// ---- Image data overlay ----

function toggleInfo() {
  STATE.showInfo = !STATE.showInfo;
  DOM.btnToggleInfo.classList.toggle("btn-active", STATE.showInfo);
  DOM.btnToggleInfo.textContent = STATE.showInfo ? "Hide Info" : "Show Info";
  updateOverlayVisibility();
}

function toggleKeypoints() {
  STATE.showKeypoints = !STATE.showKeypoints;
  DOM.btnToggleKeypoints.classList.toggle("btn-active", STATE.showKeypoints);
  DOM.btnToggleKeypoints.textContent = STATE.showKeypoints
    ? "Hide Keypoints"
    : "Show Keypoints";
  updateOverlayVisibility();
}

// The shared overlay is visible when either the summary ("Info") or the
// per-keypoint coordinate list ("Keypoints") is requested.
function updateOverlayVisibility() {
  const show = STATE.showInfo || STATE.showKeypoints;
  DOM.infoOverlay.classList.toggle("hidden", !show);
  if (show) updateInfoOverlay();
}

function updateInfoOverlay() {
  const body = DOM.infoOverlayBody;
  if (!body || (!STATE.showInfo && !STATE.showKeypoints)) return;

  const result = STATE.results[STATE.currentResultIndex];
  const img = STATE.currentImage;
  const pose = STATE.currentPoseData;
  const people = pose && pose.people ? pose.people : [];

  let html = "";

  if (STATE.showInfo) {
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
    for (const [k, v] of rows) {
      html += `<dt>${escapeHtml(k)}</dt><dd>${escapeHtml(String(v))}</dd>`;
    }
  }

  people.forEach((p, i) => {
    const kpts = p.keypoints || [];
    const visible = kpts.filter((kp) => kp.confidence >= 0.3).length;
    const avg = kpts.length
      ? kpts.reduce((s, kp) => s + kp.confidence, 0) / kpts.length
      : 0;
    html += `<div class="group-label">Person ${i + 1}</div>`;
    if (STATE.showInfo) {
      html += `<dt>Keypoints</dt><dd>${kpts.length}</dd>`;
      html += `<dt>Visible ≥0.3</dt><dd>${visible}</dd>`;
      html += `<dt>Avg conf</dt><dd>${avg.toFixed(3)}</dd>`;
    }
    if (STATE.showKeypoints) {
      // Per-keypoint pixel coordinates (rounded to whole pixels) and confidence.
      // Confidence is dotted in its skeleton color so the overlay matches the
      // canvas: green ≥0.7, yellow 0.3–0.7, red <0.3.
      for (const kp of kpts) {
        const label = kp.name.replace(/_/g, " ");
        const cls = kp.confidence >= 0.7
          ? "kp-high"
          : (kp.confidence >= 0.3 ? "kp-mid" : "kp-low");
        html += `<dt class="kp-name">${escapeHtml(label)}</dt>` +
          `<dd class="kp-coord ${cls}">` +
          `${Math.round(kp.x)}, ${Math.round(kp.y)} px ` +
          `<span class="kp-conf">(${kp.confidence.toFixed(2)})</span></dd>`;
      }
    }
  });

  if (STATE.showKeypoints) {
    html += `<div class="kp-legend">` +
      `<span class="kp-high">● ≥0.70</span>` +
      `<span class="kp-mid">● 0.30–0.70</span>` +
      `<span class="kp-low">● &lt;0.30</span> confidence` +
      `</div>`;
  }

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
  const visible = visibleResults();
  if (visible.length === 0) return;

  // Step relative to the current image's position WITHIN the visible list. If
  // the current image isn't visible (or none is selected), start from an end so
  // the first press lands on a shown image.
  const current = STATE.results[STATE.currentResultIndex];
  let pos = current ? visible.indexOf(current) : -1;
  if (pos === -1) pos = delta >= 0 ? -1 : 0;

  let nextPos = pos + delta;
  if (nextPos < 0) nextPos = visible.length - 1;
  if (nextPos >= visible.length) nextPos = 0;

  selectImage(STATE.results.indexOf(visible[nextPos]));
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
    case "arrowup":
      e.preventDefault();
      labelCurrent(1);
      break;
    case "arrowdown":
      e.preventDefault();
      labelCurrent(-1);
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
    case "k":
      e.preventDefault();
      toggleKeypoints();
      break;
    case "0":
      e.preventDefault();
      resetView();
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
          // Processor bodies reference injected globals (o_img, console,
          // f_save_filtered, …). Keep syntax validation so real parse errors
          // squiggle, but turn off semantic validation so those globals don't
          // get flagged as "cannot find name".
          try {
            monaco.languages.typescript.javascriptDefaults.setDiagnosticsOptions({
              noSemanticValidation: true,
              noSyntaxValidation: false,
            });
          } catch { /* older Monaco — ignore */ }

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

          // Drop our custom error markers as soon as the user edits, so stale
          // highlights don't linger after the offending code is changed.
          STATE.monacoEditor.onDidChangeModelContent(() => clearEditorErrorMarkers());

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

  // Keep the Pipeline's "Processor controls" panel in sync with the active set
  // so f_gui() inputs appear/disappear immediately as processors are toggled,
  // reordered, or edited — without needing a run first.
  if (document.getElementById("page-preview")?.classList.contains("active")) {
    refreshProcessorGui();
  }

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

// ---- Pipeline column (ALL processors, toggle + reorderable) on the preview ----

let dragProcId = null;

function renderProcessorOrder() {
  const list = DOM.processorOrderList;
  if (!list) return;

  const all = STATE.processors;
  const activeCount = all.filter((p) => p.active).length;
  if (DOM.processorOrderCount) {
    DOM.processorOrderCount.textContent = String(activeCount);
  }

  list.innerHTML = "";
  if (all.length === 0) {
    list.innerHTML = '<div class="placeholder">No processors yet.</div>';
    return;
  }

  let activeIdx = 0;
  all.forEach((p) => {
    const div = document.createElement("div");
    div.className = "processor-order-entry" + (p.active ? "" : " inactive");
    div.draggable = true;
    div.dataset.id = p.id;

    const handle = document.createElement("span");
    handle.className = "po-handle";
    handle.textContent = "⠿";
    handle.title = "Drag to reorder";

    // Order number among ACTIVE processors (inactive show a dash).
    const num = document.createElement("span");
    num.className = "po-num";
    num.textContent = p.active ? String(++activeIdx) : "–";

    // Quick enable/disable toggle right in the pipeline.
    const toggle = document.createElement("button");
    toggle.className = `p-toggle ${p.active ? "on" : ""}`;
    toggle.textContent = p.active ? "On" : "Off";
    toggle.title = p.active ? "Disable this processor" : "Enable this processor";
    toggle.addEventListener("click", (e) => {
      e.stopPropagation();
      toggleProcessorActive(p);
    });

    const name = document.createElement("span");
    name.className = "po-name";
    name.textContent = p.name || "(unnamed)";
    name.title = p.name || "";

    div.append(handle, num, toggle, name);

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
      reorderProcessors(dragProcId, p.id, isPointerInLowerHalf(e, div));
    });

    list.appendChild(div);
  });
}

function isPointerInLowerHalf(event, el) {
  const rect = el.getBoundingClientRect();
  return event.clientY > rect.top + rect.height / 2;
}

// Move `draggedId` next to `targetId` in the full processor list (active and
// inactive alike). The active application order is just this list filtered to
// active. Persists the order and re-applies the pipeline if a run already ran.
function reorderProcessors(draggedId, targetId, placeAfter) {
  const ids = STATE.processors.map((p) => p.id);
  const from = ids.indexOf(draggedId);
  if (from === -1) return;
  ids.splice(from, 1);

  let to = ids.indexOf(targetId);
  if (to === -1) to = ids.length;
  else if (placeAfter) to += 1;
  ids.splice(to, 0, draggedId);

  const byId = Object.fromEntries(STATE.processors.map((p) => [p.id, p]));
  STATE.processors = ids.map((id) => byId[id]);

  renderProcessorOrder();
  renderProcessorList();
  sendMessage({ type: "reorder_processors", order: ids });

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

// ---- Processor error highlighting (Monaco) ----

const PROCESSOR_MARKER_OWNER = "processor-error";

function tsMessageText(messageText) {
  // TS diagnostics are either a string or a chained {messageText, next} object.
  return typeof messageText === "string"
    ? messageText
    : (messageText && messageText.messageText) || "Syntax error";
}

// Locate syntax errors in arbitrary processor code using Monaco's JS language
// service. Returns an array of marker-shaped objects (with 1-based line/column),
// or null when Monaco isn't available (offline / fallback textarea).
async function locateSyntaxErrors(code) {
  if (typeof monaco === "undefined" || !monaco.languages || !monaco.languages.typescript) {
    return null;
  }
  let model = null;
  try {
    const uri = monaco.Uri.parse(`inmemory://proc-check/${STATE._procCheckSeq = (STATE._procCheckSeq || 0) + 1}.js`);
    model = monaco.editor.createModel(code, "javascript", uri);
    const getWorker = await monaco.languages.typescript.getJavaScriptWorker();
    const worker = await getWorker(model.uri);
    const diags = await worker.getSyntacticDiagnostics(model.uri.toString());
    if (!diags || diags.length === 0) return [];
    return diags.map((d) => {
      const start = model.getPositionAt(d.start);
      const end = model.getPositionAt(d.start + (d.length || 1));
      return {
        startLineNumber: start.lineNumber,
        startColumn: start.column,
        endLineNumber: end.lineNumber,
        endColumn: end.column,
        message: tsMessageText(d.messageText),
      };
    });
  } catch {
    return null;
  } finally {
    if (model) model.dispose();
  }
}

// Best-effort {line, column} for an error thrown while EXECUTING a processor.
// Processors are compiled with new Function(...args, body); in V8 the generated
// wrapper is "function anonymous(<args>\n) {\n<body…>", so body line 1 lands on
// generated line 3 — subtract that offset to map back to editor lines.
function locateRuntimeError(error) {
  const stack = error && typeof error.stack === "string" ? error.stack : "";
  const BODY_LINE_OFFSET = 2;
  // V8 names the compiled wrapper "anonymous"/"<anonymous>"; the body frame is
  // the first one referencing it. Fall back to the first line:col pair if the
  // wrapper isn't named (other engines).
  const patterns = [
    /<anonymous>:(\d+):(\d+)/,
    /\banonymous:(\d+):(\d+)/,
    /:(\d+):(\d+)/,
  ];
  for (const re of patterns) {
    const m = re.exec(stack);
    if (m) {
      const line = Number(m[1]) - BODY_LINE_OFFSET;
      if (line >= 1) return { line, column: Number(m[2]) };
    }
  }
  return null;
}

function clearEditorErrorMarkers() {
  if (typeof monaco === "undefined" || !STATE.monacoEditor) return;
  const model = STATE.monacoEditor.getModel();
  if (model) monaco.editor.setModelMarkers(model, PROCESSOR_MARKER_OWNER, []);
}

// Mark the given positions as errors in the editor and jump to the first one.
function highlightEditorErrors(markers) {
  if (typeof monaco === "undefined" || !STATE.monacoEditor || !markers || !markers.length) {
    return;
  }
  const model = STATE.monacoEditor.getModel();
  if (!model) return;
  monaco.editor.setModelMarkers(
    model,
    PROCESSOR_MARKER_OWNER,
    markers.map((m) => ({
      severity: monaco.MarkerSeverity.Error,
      message: m.message,
      startLineNumber: m.startLineNumber,
      startColumn: m.startColumn,
      endLineNumber: m.endLineNumber,
      endColumn: m.endColumn,
    })),
  );
  const first = markers[0];
  STATE.monacoEditor.revealLineInCenter(first.startLineNumber);
  STATE.monacoEditor.setPosition({
    lineNumber: first.startLineNumber,
    column: first.startColumn,
  });
  STATE.monacoEditor.focus();
}

// True when the editor currently shows the processor with this name.
function isProcessorOpen(name) {
  return !!(DOM.editorPanel &&
    !DOM.editorPanel.classList.contains("hidden") &&
    DOM.processorName &&
    DOM.processorName.value.trim() === name);
}

async function saveProcessor() {
  const name = DOM.processorName.value.trim();
  const code = getEditorCode();
  const active = DOM.processorActive.checked;

  if (!name) {
    setEditorMessage("Please give the processor a name.", "error");
    return;
  }

  // Validate the code compiles before saving so users get immediate feedback.
  clearEditorErrorMarkers();
  try {
    new Function(...PROCESSOR_ARG_NAMES, code);
  } catch (e) {
    // new Function's SyntaxError has no usable position; ask Monaco where it is
    // and highlight it directly in the editor.
    const markers = await locateSyntaxErrors(code);
    if (markers && markers.length) {
      highlightEditorErrors(markers);
      const f = markers[0];
      setEditorMessage(
        `Syntax error (line ${f.startLineNumber}, col ${f.startColumn}): ${f.message}`,
        "error",
      );
    } else {
      setEditorMessage(`Syntax error: ${e.message}`, "error");
    }
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

// ---- Result cache (versioned) ----------------------------------------------
//
// Running a processor or applying a model over every image is expensive, and we
// re-do it constantly during curation. So we cache each processor's matched set
// and each model's per-image scores, keyed by the processor/model VERSION (from
// the server) and a signature of the current image set. A cache entry is valid
// only while both match — editing a processor (version bump), retraining a model
// (version bump), or scanning a different folder (signature change) invalidates
// it automatically and forces a fresh run.

const PROC_CACHE_KEY = "pilotless_poses_proc_cache";
const MODEL_CACHE_KEY = "pilotless_poses_model_cache";

function loadCaches() {
  try { STATE.procCache = JSON.parse(localStorage.getItem(PROC_CACHE_KEY)) || {}; }
  catch { STATE.procCache = {}; }
  try { STATE.modelCache = JSON.parse(localStorage.getItem(MODEL_CACHE_KEY)) || {}; }
  catch { STATE.modelCache = {}; }
}
function saveProcCache() {
  try { localStorage.setItem(PROC_CACHE_KEY, JSON.stringify(STATE.procCache)); } catch {}
}
function saveModelCache() {
  try { localStorage.setItem(MODEL_CACHE_KEY, JSON.stringify(STATE.modelCache)); } catch {}
}

// Cheap stable signature of the current result set; changes when the set of
// images changes, so caches built for a different scan don't apply.
function resultsSignature() {
  const s = STATE.results.map((r) => r.result).join("|");
  let h = 0;
  for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) | 0;
  return `${STATE.results.length}:${h}`;
}

const procVersion = (p) => (typeof p.version === "number" ? p.version : 1);
const modelVersion = (m) => (typeof m.version === "number" ? m.version : 0);

function processorCacheValid(p, sig) {
  const c = STATE.procCache[p.id];
  return !!c && c.version === procVersion(p) && c.sig === sig;
}
function modelCacheValid(m, sig) {
  const c = STATE.modelCache[m.id];
  return !!c && c.version === modelVersion(m) && c.sig === sig;
}

// Derive STATE.procResult (the pipeline match: passed EVERY active processor)
// from the per-processor caches.
function buildProcResultFromCache(active) {
  const sets = active.map((p) => new Set((STATE.procCache[p.id] || {}).matched || []));
  const matched = new Set();
  const reasons = {};
  const evaluated = [];
  for (const r of STATE.results) {
    const rp = r.result;
    evaluated.push(rp);
    let keep = true;
    for (let i = 0; i < active.length; i++) {
      if (!sets[i].has(rp)) {
        keep = false;
        reasons[rp] = `Filtered by "${active[i].name}"`;
        break;
      }
    }
    if (keep) matched.add(rp);
  }
  STATE.procResult = { matched, reasons, evaluated };
}

async function runProcessors() {
  if (STATE.runningProcessors) return;

  const active = STATE.processors.filter((p) => p.active);
  if (active.length === 0) {
    STATE.procResult = null;
    updateFilterStatusText("No active processors to run.", true);
    applyAllFilters();
    return;
  }

  // Compile all active processors up front so a syntax error stops the run.
  const compiled = [];
  for (const p of active) {
    try {
      compiled.push({
        p,
        name: p.name,
        fn: new Function(...PROCESSOR_ARG_NAMES, p.code),
      });
    } catch (e) {
      const markers = await locateSyntaxErrors(p.code);
      const f = markers && markers[0];
      // If this processor is the one open in the editor, highlight it there.
      if (f && isProcessorOpen(p.name)) {
        highlightEditorErrors(markers);
        setEditorMessage(
          `Syntax error (line ${f.startLineNumber}, col ${f.startColumn}): ${f.message}`,
          "error",
        );
      }
      const where = f ? ` (line ${f.startLineNumber}, col ${f.startColumn})` : "";
      updateFilterStatusText(
        `Processor "${p.name}" has a syntax error${where}: ${f ? f.message : e.message}`,
        true,
      );
      return;
    }
  }

  // Cache fast-path: if every active processor already has a valid cached result
  // for these exact images + versions, derive the filter instantly — no pose
  // data fetch, no image loading, no JS evaluation.
  const sig = resultsSignature();
  if (compiled.every((c) => processorCacheValid(c.p, sig))) {
    buildProcResultFromCache(active);
    applyAllFilters();
    const kept = keptResults().length;
    updateFilterStatusText(
      `Done (cached): ${kept} kept of ${STATE.results.length}.` +
        (STATE.invertFilter ? " Inverted." : "") +
        (STATE.limitCount > 0 ? ` Limited to ${kept}.` : ""),
      false,
    );
    return;
  }

  STATE._run = { compiled, active, sig };
  STATE.runningProcessors = true;
  setRunButton(true);
  updateFilterStatusText("Loading pose data…");

  // Make sure lil-gui is available before the processors run so any f_gui()
  // controls can be created synchronously during the run.
  await ensureLilGui();

  const sent = sendMessage({ type: "get_all_pose_data" });
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
  const run = STATE._run || {};
  const compiled = run.compiled || [];
  const active = run.active || [];
  const sig = run.sig || resultsSignature();
  const items = msg.items || [];

  if (msg.error) {
    STATE.runningProcessors = false;
    STATE._run = null;
    setRunButton(false);
    updateFilterStatusText(`Run failed: ${msg.error}`, true);
    return;
  }

  // Rebuild the processor GUI panel for this run; f_gui() re-adds controls as
  // they're referenced, reusing values the user entered on a previous run.
  rebuildProcessorGui();

  let errors = 0;
  // First runtime error seen per processor: name -> { error, loc, image }.
  const procErrors = new Map();
  // Per-processor matched sets — cached independently so each processor's result
  // can be reused without re-running the others.
  const perProc = new Map();
  for (const c of compiled) perProc.set(c.p.id, new Set());

  // Resolve image dimensions with limited concurrency so we don't open
  // hundreds of image requests at once. This is the slow part of a run, so
  // report progress (e.g. "22 / 200 images") as each image resolves.
  let processed = 0;
  updateFilterStatusText(`Running processors… 0 / ${items.length} images`);
  const dimsByResult = await mapWithConcurrency(items, 6, async (item) => {
    let dims = { width: 0, height: 0 };
    if (item.status === "ok") {
      dims = await loadImageDims(item.source).catch(() => ({
        width: 0,
        height: 0,
      }));
    }
    processed++;
    updateFilterStatusText(
      `Running processors… ${processed} / ${items.length} images`,
    );
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
      a_s_tag: (STATE.tags[item.result] || []).slice(),
      a_o_person,
    };

    const procArgs = processorArgs(o_img, pose);

    // Evaluate EVERY processor (no short-circuit) so each one's result can be
    // cached. A throwing processor is treated as a pass (doesn't filter).
    for (const proc of compiled) {
      let result;
      try {
        result = proc.fn(...procArgs);
      } catch (e) {
        errors++;
        if (!procErrors.has(proc.name)) {
          procErrors.set(proc.name, {
            error: e,
            loc: locateRuntimeError(e),
            image: o_img.s_name_file,
          });
        }
        console.error(`Processor "${proc.name}" threw on ${o_img.s_name_file}:`, e);
        result = true;
      }
      if (result) perProc.get(proc.p.id).add(item.result);
    }
  }

  // Persist each processor's cache, then derive the pipeline result from it.
  for (const c of compiled) {
    STATE.procCache[c.p.id] = {
      version: procVersion(c.p),
      sig,
      matched: [...perProc.get(c.p.id)],
    };
  }
  saveProcCache();

  STATE.runningProcessors = false;
  STATE._run = null;
  setRunButton(false);
  buildProcResultFromCache(active);
  applyAllFilters();

  const matchedCount = STATE.procResult.matched.size;
  const total = STATE.results.length;
  const keptCount = keptResults().length; // after invert + limit
  let summary = STATE.invertFilter
    ? `Done: ${total - matchedCount} not matched (inverted), ${matchedCount} matched, of ${total}.`
    : `Done: ${matchedCount} matched, ${total - matchedCount} filtered out of ${total}.`;
  if (STATE.limitCount > 0) summary += ` Limited to ${keptCount}.`;
  const filteredCount = total - keptCount;
  if (errors > 0) {
    const [name, info] = [...procErrors.entries()][0];
    const emsg = info.error && info.error.message
      ? info.error.message
      : String(info.error);
    const where = info.loc ? ` (line ${info.loc.line}, col ${info.loc.column})` : "";
    summary += ` ${errors} processor error(s). "${name}"${where}: ${emsg}` +
      ` — e.g. on ${info.image}.`;
    // Highlight in the editor if that processor is the one being edited.
    if (info.loc && isProcessorOpen(name)) {
      highlightEditorErrors([{
        startLineNumber: info.loc.line,
        startColumn: info.loc.column,
        endLineNumber: info.loc.line,
        endColumn: info.loc.column + 1,
        message: emsg,
      }]);
      setEditorMessage(
        `Runtime error (line ${info.loc.line}, col ${info.loc.column}): ${emsg}`,
        "error",
      );
    }
  }
  updateFilterStatusText(summary, filteredCount > 0 || errors > 0);
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
    exportFolder: STATE.exportFolder,
    selectedBasetagId: STATE.selectedBasetagId,
    filterPositive: STATE.filterPositive,
    filterNegative: STATE.filterNegative,
    invertFilter: STATE.invertFilter,
    limitCount: STATE.limitCount,
  };
  try { localStorage.setItem(STATE_KEY, JSON.stringify(state)); } catch {}
}

function loadState() {
  try {
    const raw = localStorage.getItem(STATE_KEY);
    if (!raw) return;
    const state = JSON.parse(raw);
    if (state.scanPath) FILE_BROWSER.scanPath = state.scanPath;
    if (state.exportFolder) STATE.exportFolder = state.exportFolder;
    if (typeof state.selectedBasetagId === "string") STATE.selectedBasetagId = state.selectedBasetagId;
    if (typeof state.filterPositive === "boolean") STATE.filterPositive = state.filterPositive;
    if (typeof state.filterNegative === "boolean") STATE.filterNegative = state.filterNegative;
    if (typeof state.invertFilter === "boolean") STATE.invertFilter = state.invertFilter;
    if (typeof state.limitCount === "number") STATE.limitCount = state.limitCount;
    const extInput = document.getElementById("ext-filter");
    if (extInput && state.extFilter) extInput.value = state.extFilter;
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
  // The inference page reuses the folder browser with page="inference".
  if (data.page === "inference") {
    handleInferenceBrowseResult(data);
    return;
  }

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
// Page 5: Inference Models (CRUD + training)
// ============================================================================

function initInferencePage() {
  DOM.inferenceList = document.getElementById("inference-list");
  DOM.inferenceCount = document.getElementById("inference-count");
  DOM.inferenceEditorPlaceholder = document.getElementById("inference-editor-placeholder");
  DOM.inferenceEditorPanel = document.getElementById("inference-editor-panel");
  DOM.inferenceName = document.getElementById("inference-name");
  DOM.inferencePosDir = document.getElementById("inference-pos-dir");
  DOM.inferenceNegDir = document.getElementById("inference-neg-dir");
  DOM.inferenceStatusBadge = document.getElementById("inference-status-badge");
  DOM.inferenceMetrics = document.getElementById("inference-metrics");
  DOM.inferenceEditorMessage = document.getElementById("inference-editor-message");
  DOM.btnTrainInference = document.getElementById("btn-train-inference");
  DOM.btnSaveInference = document.getElementById("btn-save-inference");
  DOM.btnDeleteInference = document.getElementById("btn-delete-inference");
  DOM.btnCancelInference = document.getElementById("btn-cancel-inference");

  // Shared inline folder browser
  DOM.inferenceDirBrowser = document.getElementById("inference-dir-browser");
  DOM.inferenceBrowsePath = document.getElementById("inference-browse-path");
  DOM.inferenceBrowserList = document.getElementById("inference-browser-list");

  document.getElementById("btn-new-inference")
    .addEventListener("click", newInferenceModel);
  DOM.btnSaveInference.addEventListener("click", () => saveInferenceModel(false));
  DOM.btnDeleteInference.addEventListener("click", deleteInferenceModel);
  DOM.btnCancelInference.addEventListener("click", closeInferenceEditor);
  DOM.btnTrainInference.addEventListener("click", trainInferenceModel);

  // 📁 buttons open the shared folder browser, writing back to their target.
  document.querySelectorAll(".btn-browse-dir").forEach((btn) => {
    btn.addEventListener("click", () => openDirBrowser(btn.dataset.target));
  });
  document.getElementById("btn-inference-browse-up").addEventListener("click", () => {
    const parent = parentPath(STATE.dirBrowsePath);
    if (parent !== null) navigateInferenceBrowserTo(parent);
  });
  document.getElementById("btn-inference-browse-use")
    .addEventListener("click", useDirBrowser);
}

function handleInferenceModelsList(msg) {
  STATE.inferenceModels = msg.models || [];
  if (msg.error) console.error("inference_models error:", msg.error);

  // Resolve a freshly-saved "__new__" model to its concrete id.
  if (STATE.pendingSelectInferenceName) {
    const m = STATE.inferenceModels.find(
      (x) => x.name === STATE.pendingSelectInferenceName,
    );
    if (m) {
      STATE.editingInferenceId = m.id;
      STATE.pendingSelectInferenceName = null;
    }
  }

  // If a save was triggered by Train, kick off training now that we have an id.
  if (STATE.pendingTrainInferenceName) {
    const m = STATE.inferenceModels.find(
      (x) => x.name === STATE.pendingTrainInferenceName,
    );
    STATE.pendingTrainInferenceName = null;
    if (m) startTraining(m.id);
  }

  renderInferenceList();
  renderInferenceModelChoices();

  // Refresh the open editor (e.g. status/metrics updated after training).
  if (STATE.editingInferenceId && STATE.editingInferenceId !== "__new__") {
    const m = STATE.inferenceModels.find((x) => x.id === STATE.editingInferenceId);
    if (m) refreshInferenceEditor(m);
  }
}

function renderInferenceList() {
  const list = DOM.inferenceList;
  if (!list) return;
  list.innerHTML = "";
  if (DOM.inferenceCount) {
    DOM.inferenceCount.textContent = String(STATE.inferenceModels.length);
  }

  if (STATE.inferenceModels.length === 0) {
    list.innerHTML =
      '<div class="placeholder">No models yet. Create one to get started.</div>';
    return;
  }

  STATE.inferenceModels.forEach((m) => {
    const div = document.createElement("div");
    div.className = "processor-entry";
    if (m.id === STATE.editingInferenceId) div.classList.add("active");

    const name = document.createElement("span");
    name.className = "p-name";
    name.textContent = m.name || "(unnamed)";
    name.title = m.name || "";

    const badge = document.createElement("span");
    badge.className = `status-badge ${m.status || "untrained"}`;
    badge.textContent = m.status || "untrained";

    div.appendChild(name);
    div.appendChild(badge);
    div.addEventListener("click", () => editInferenceModel(m.id));
    list.appendChild(div);
  });
}

function openInferenceEditor() {
  DOM.inferenceEditorPlaceholder.classList.add("hidden");
  DOM.inferenceEditorPanel.classList.remove("hidden");
  DOM.inferenceDirBrowser.classList.add("hidden");
}

function closeInferenceEditor() {
  STATE.editingInferenceId = null;
  DOM.inferenceEditorPanel.classList.add("hidden");
  DOM.inferenceEditorPlaceholder.classList.remove("hidden");
  setInferenceMessage("");
  renderInferenceList();
}

function newInferenceModel() {
  STATE.editingInferenceId = "__new__";
  openInferenceEditor();
  DOM.inferenceName.value = "";
  DOM.inferencePosDir.value = "";
  DOM.inferenceNegDir.value = "";
  DOM.btnDeleteInference.classList.add("hidden");
  setInferenceStatusBadge("untrained");
  renderInferenceMetrics(null);
  setInferenceMessage("Pick positive and negative folders, then Save.");
  renderInferenceList();
  DOM.inferenceName.focus();
}

function editInferenceModel(id) {
  const m = STATE.inferenceModels.find((x) => x.id === id);
  if (!m) return;
  STATE.editingInferenceId = id;
  openInferenceEditor();
  DOM.inferenceName.value = m.name || "";
  DOM.inferencePosDir.value = m.positiveDir || "";
  DOM.inferenceNegDir.value = m.negativeDir || "";
  DOM.btnDeleteInference.classList.remove("hidden");
  setInferenceMessage("");
  refreshInferenceEditor(m);
  renderInferenceList();
}

// Update only the live status/metrics of the currently-open editor without
// clobbering text fields the user may be editing.
function refreshInferenceEditor(m) {
  setInferenceStatusBadge(m.status || "untrained", m.error);
  renderInferenceMetrics(m.metrics || null);
  const training = STATE.trainingInferenceId === m.id || m.status === "training";
  DOM.btnTrainInference.disabled = training;
  DOM.btnTrainInference.textContent = training ? "Training…" : "Train";
}

function setInferenceStatusBadge(status, error) {
  if (!DOM.inferenceStatusBadge) return;
  DOM.inferenceStatusBadge.className = `status-badge ${status}`;
  DOM.inferenceStatusBadge.textContent = status;
  DOM.inferenceStatusBadge.title = error || "";
}

function renderInferenceMetrics(metrics) {
  const el = DOM.inferenceMetrics;
  if (!el) return;
  if (!metrics) {
    el.innerHTML = "";
    return;
  }
  const top = (metrics.topFeatures || [])
    .map((f) => `<li><code>${escapeHtml(f.name)}</code> — ${f.importance.toFixed(3)}</li>`)
    .join("");
  el.innerHTML =
    `<div class="metrics-grid">` +
    `<div><span class="metrics-label">Positive samples</span><span>${metrics.nPos}</span></div>` +
    `<div><span class="metrics-label">Negative samples</span><span>${metrics.nNeg}</span></div>` +
    `<div><span class="metrics-label">Train accuracy</span><span>${(metrics.trainAcc * 100).toFixed(1)}%</span></div>` +
    `<div><span class="metrics-label">Test accuracy</span><span>${(metrics.testAcc * 100).toFixed(1)}%</span></div>` +
    `<div><span class="metrics-label">Features</span><span>${metrics.features}</span></div>` +
    `</div>` +
    (top ? `<div class="metrics-top"><span class="metrics-label">Top features</span><ul>${top}</ul></div>` : "");
}

function setInferenceMessage(text, kind) {
  if (!DOM.inferenceEditorMessage) return;
  DOM.inferenceEditorMessage.textContent = text || "";
  DOM.inferenceEditorMessage.className =
    "editor-message" + (kind ? " " + kind : "");
}

// Read the editor fields and send a save. Returns false on validation failure.
// When `silent` is true, success messages are suppressed (used before Train).
function saveInferenceModel(silent) {
  const name = DOM.inferenceName.value.trim();
  const positiveDir = DOM.inferencePosDir.value.trim();
  const negativeDir = DOM.inferenceNegDir.value.trim();

  if (!name) {
    setInferenceMessage("Please give the model a name.", "error");
    return false;
  }
  if (!positiveDir || !negativeDir) {
    setInferenceMessage("Please set both the positive and negative folders.", "error");
    return false;
  }

  const id = STATE.editingInferenceId === "__new__"
    ? undefined
    : STATE.editingInferenceId;

  const sent = sendMessage({
    type: "save_inference_model",
    model: { id, name, positiveDir, negativeDir },
  });
  if (!sent) {
    setInferenceMessage("Not connected — could not save.", "error");
    return false;
  }

  if (!id) STATE.pendingSelectInferenceName = name;
  if (!silent) setInferenceMessage("Saved.", "success");
  return true;
}

function deleteInferenceModel() {
  if (!STATE.editingInferenceId || STATE.editingInferenceId === "__new__") {
    closeInferenceEditor();
    return;
  }
  const m = STATE.inferenceModels.find((x) => x.id === STATE.editingInferenceId);
  if (m && !confirm(`Delete model "${m.name}"?`)) return;
  sendMessage({ type: "delete_inference_model", id: STATE.editingInferenceId });
  closeInferenceEditor();
}

// Persist the current edits, then train. Training starts once the saved model
// (with a concrete id) comes back in the models list.
function trainInferenceModel() {
  const name = DOM.inferenceName.value.trim();
  if (!saveInferenceModel(true)) return;

  if (STATE.editingInferenceId && STATE.editingInferenceId !== "__new__") {
    startTraining(STATE.editingInferenceId);
  } else {
    // New model: defer until the save round-trip assigns an id.
    STATE.pendingTrainInferenceName = name;
  }
  setInferenceMessage("Training… progress streams to the Live Log tab.");
}

function startTraining(id) {
  STATE.trainingInferenceId = id;
  if (DOM.btnTrainInference) {
    DOM.btnTrainInference.disabled = true;
    DOM.btnTrainInference.textContent = "Training…";
  }
  setInferenceStatusBadge("training");
  sendMessage({ type: "train_inference_model", id });
}

function handleInferenceModelTrained(msg) {
  if (STATE.trainingInferenceId === msg.id) STATE.trainingInferenceId = null;
  if (DOM.btnTrainInference) {
    DOM.btnTrainInference.disabled = false;
    DOM.btnTrainInference.textContent = "Train";
  }
  if (msg.error) {
    if (STATE.editingInferenceId === msg.id) {
      setInferenceStatusBadge("error", msg.error);
      setInferenceMessage(`Training failed: ${msg.error}`, "error");
    }
  } else if (STATE.editingInferenceId === msg.id) {
    setInferenceMessage("Training complete.", "success");
  }
  // The updated models list (status/metrics) arrives separately via
  // inference_models_list and refreshes the editor + preview choices.
}

// ---- Shared inline folder browser (positive/negative pickers) --------------

function openDirBrowser(targetInputId) {
  STATE.dirBrowseTarget = targetInputId;
  DOM.inferenceDirBrowser.classList.remove("hidden");
  const current = (document.getElementById(targetInputId).value || "").trim();
  navigateInferenceBrowserTo(current || "/");
}

function navigateInferenceBrowserTo(path) {
  STATE.dirBrowsePath = path;
  if (DOM.inferenceBrowsePath) DOM.inferenceBrowsePath.value = path;
  if (DOM.inferenceBrowserList) {
    DOM.inferenceBrowserList.innerHTML =
      '<div class="browser-placeholder">Loading...</div>';
  }
  if (!sendMessage({ type: "browse_folder", path, page: "inference" })) {
    if (DOM.inferenceBrowserList) {
      DOM.inferenceBrowserList.innerHTML =
        '<div class="browser-placeholder">WebSocket not connected. Please wait...</div>';
    }
  }
}

function handleInferenceBrowseResult(data) {
  const list = DOM.inferenceBrowserList;
  if (!list) return;
  if (data.error) {
    list.innerHTML =
      `<div class="browser-placeholder">Error: ${escapeHtml(data.error)}</div>`;
    return;
  }
  STATE.dirBrowsePath = data.path;
  if (DOM.inferenceBrowsePath) DOM.inferenceBrowsePath.value = data.path;

  list.innerHTML = "";
  const dirs = (data.entries || []).filter((e) => e.isDirectory);
  dirs.forEach((entry) => {
    const div = document.createElement("div");
    div.className = "browser-entry";
    const icon = document.createElement("span");
    icon.className = "entry-icon";
    icon.textContent = "📁";
    const name = document.createElement("span");
    name.className = "entry-name";
    name.textContent = entry.name;
    div.appendChild(icon);
    div.appendChild(name);
    const targetPath = data.path === "/"
      ? `/${entry.name}`
      : `${data.path}/${entry.name}`;
    div.addEventListener("click", () => navigateInferenceBrowserTo(targetPath));
    list.appendChild(div);
  });
  if (dirs.length === 0) {
    list.innerHTML =
      '<div class="browser-placeholder">No sub-folders here. Use this folder, or go up.</div>';
  }
}

function useDirBrowser() {
  if (STATE.dirBrowseTarget) {
    const input = document.getElementById(STATE.dirBrowseTarget);
    if (input) input.value = STATE.dirBrowsePath;
  }
  DOM.inferenceDirBrowser.classList.add("hidden");
  STATE.dirBrowseTarget = null;
}

// ============================================================================
// Preview: inference-model filter column
// ============================================================================

function renderInferenceModelChoices() {
  const el = DOM.inferenceModelChoices;
  if (!el) return;
  const trained = STATE.inferenceModels.filter((m) => m.status === "trained");

  if (DOM.inferenceFilterCount) {
    DOM.inferenceFilterCount.textContent = String(trained.length);
  }

  el.innerHTML = "";
  if (trained.length === 0) {
    el.innerHTML =
      '<div class="placeholder">No trained models yet. Train one on the Inference Models tab.</div>';
    if (DOM.btnApplyInference) DOM.btnApplyInference.disabled = true;
    return;
  }

  trained.forEach((m) => {
    const label = document.createElement("label");
    label.className = "inference-choice";
    const radio = document.createElement("input");
    radio.type = "radio";
    radio.name = "inference-model";
    radio.value = m.id;
    if (STATE.appliedInferenceId === m.id) radio.checked = true;
    radio.addEventListener("change", () => {
      if (DOM.btnApplyInference) DOM.btnApplyInference.disabled = false;
    });
    const span = document.createElement("span");
    span.textContent = m.name;
    span.title = m.name;
    label.appendChild(radio);
    label.appendChild(span);
    el.appendChild(label);
  });

  // Enable Apply if something is selected (or pre-selected from a prior apply).
  const anyChecked = !!el.querySelector('input[name="inference-model"]:checked');
  if (DOM.btnApplyInference) DOM.btnApplyInference.disabled = !anyChecked;
}

function selectedInferenceModelId() {
  const el = DOM.inferenceModelChoices;
  const checked = el && el.querySelector('input[name="inference-model"]:checked');
  return checked ? checked.value : null;
}

function applyInferenceModel() {
  if (STATE.applyingInference) return;
  const id = selectedInferenceModelId();
  if (!id) {
    setInferenceFilterStatus("Select a trained model first.", true);
    return;
  }
  if (STATE.results.length === 0) {
    setInferenceFilterStatus("No images loaded — run a scan first.", true);
    return;
  }

  // Cache fast-path: the model's per-image scores are cached by version + image
  // set, so re-applying — even at a different threshold — needs no Python run.
  const model = STATE.inferenceModels.find((m) => m.id === id);
  const sig = resultsSignature();
  if (model && modelCacheValid(model, sig)) {
    applyModelScores(id, STATE.inferenceThreshold, STATE.modelCache[id].prob, true);
    return;
  }

  STATE.applyingInference = true;
  if (DOM.btnApplyInference) {
    DOM.btnApplyInference.disabled = true;
    DOM.btnApplyInference.textContent = "Applying…";
  }
  setInferenceFilterStatus("Running inference…");

  const sent = sendMessage({
    type: "apply_inference_model",
    id,
    threshold: STATE.inferenceThreshold,
  });
  if (!sent) {
    STATE.applyingInference = false;
    if (DOM.btnApplyInference) {
      DOM.btnApplyInference.disabled = false;
      DOM.btnApplyInference.textContent = "Apply";
    }
    setInferenceFilterStatus("Not connected — could not run.", true);
  }
}

function handleInferenceModelApplied(msg) {
  STATE.applyingInference = false;
  if (DOM.btnApplyInference) {
    DOM.btnApplyInference.disabled = false;
    DOM.btnApplyInference.textContent = "Apply";
  }
  if (msg.error) {
    setInferenceFilterStatus(`Inference failed: ${msg.error}`, true);
    return;
  }

  // Cache the per-image probabilities (keyed by the model's version + image set)
  // so future applies / threshold tweaks skip the Python run.
  const model = STATE.inferenceModels.find((m) => m.id === msg.id);
  const prob = {};
  for (const [rp, res] of Object.entries(msg.results || {})) {
    if (res && typeof res.prob === "number") prob[rp] = res.prob;
  }
  if (model) {
    STATE.modelCache[msg.id] = { version: modelVersion(model), sig: resultsSignature(), prob };
    saveModelCache();
  }

  applyModelScores(msg.id, msg.threshold, prob, false);
}

// Compute the inference filter for `modelId` at `threshold` from a per-image
// probability map (server-fresh or cached). Shared by the live and cached paths.
function applyModelScores(modelId, threshold, prob, fromCache) {
  const model = STATE.inferenceModels.find((m) => m.id === modelId);
  const modelName = model ? model.name : "model";

  const filtered = {};
  let passed = 0;
  let scored = 0;
  for (const r of STATE.results) {
    const p = prob[r.result];
    if (p === undefined) continue; // failed/unscored images aren't touched
    scored++;
    if (p >= threshold) {
      passed++;
    } else {
      filtered[r.result] = `Below threshold ${threshold.toFixed(2)} ` +
        `(model "${modelName}", score ${p.toFixed(2)})`;
    }
  }

  STATE.inferenceFilteredOut = filtered;
  STATE.appliedInferenceId = modelId;
  if (DOM.btnClearInference) DOM.btnClearInference.classList.remove("hidden");

  recomputeLimit(); // the limit applies after the inference filter
  renderImageList();
  const hidden = Object.keys(filtered).length;
  setInferenceFilterStatus(
    `"${modelName}": ${passed} passed, ${hidden} filtered of ${scored} scored.` +
      (fromCache ? " (cached)" : ""),
    hidden > 0,
  );
}

function clearInferenceFilter() {
  STATE.inferenceFilteredOut = {};
  STATE.appliedInferenceId = null;
  if (DOM.btnClearInference) DOM.btnClearInference.classList.add("hidden");
  setInferenceFilterStatus("");
  recomputeLimit(); // the limit set may grow now that the inference filter is gone
  renderImageList();
}

function setInferenceFilterStatus(text, highlight) {
  const el = DOM.inferenceFilterStatus;
  if (!el) return;
  el.textContent = text || "";
  el.classList.toggle("has-filter", !!highlight);
}

// ============================================================================
// Page 5: Basetags
// ============================================================================
// A basetag is a single base name (e.g. "hands_in_the_air"). It drives a
// keyboard labelling workflow on the preview page (↑ → `${name}_positive`,
// ↓ → `${name}_negative`, with the image auto-copied into the matching folder)
// and auto-fills the positive/negative dirs on the inference page.

// Slug-safe names so they map cleanly onto tag + on-disk folder names. Mirrors
// BASETAG_NAME_RE in server.ts.
const BASETAG_NAME_RE = /^[A-Za-z0-9_-]+$/;

function initBasetagsPage() {
  DOM.basetagList = document.getElementById("basetag-list");
  DOM.basetagCount = document.getElementById("basetag-count");
  DOM.basetagEditorPlaceholder = document.getElementById("basetag-editor-placeholder");
  DOM.basetagEditorPanel = document.getElementById("basetag-editor-panel");
  DOM.basetagName = document.getElementById("basetag-name");
  DOM.basetagEditorMessage = document.getElementById("basetag-editor-message");
  DOM.btnSaveBasetag = document.getElementById("btn-save-basetag");
  DOM.btnDeleteBasetag = document.getElementById("btn-delete-basetag");
  DOM.btnCancelBasetag = document.getElementById("btn-cancel-basetag");

  document.getElementById("btn-new-basetag").addEventListener("click", newBasetag);
  DOM.btnSaveBasetag.addEventListener("click", saveBasetag);
  DOM.btnDeleteBasetag.addEventListener("click", deleteBasetag);
  DOM.btnCancelBasetag.addEventListener("click", closeBasetagEditor);
  DOM.basetagName.addEventListener("keydown", (e) => {
    if (e.key === "Enter") { e.preventDefault(); saveBasetag(); }
  });

  // Preview-page basetag selector (drives the ↑/↓ labelling).
  DOM.basetagSelect = document.getElementById("basetag-select");
  DOM.basetagStatus = document.getElementById("basetag-status");
  if (DOM.basetagSelect) {
    DOM.basetagSelect.addEventListener("change", () => {
      STATE.selectedBasetagId = DOM.basetagSelect.value || null;
      saveState();
      setBasetagStatus(
        STATE.selectedBasetagId
          ? `Basetag “${selectedBasetagName()}” — ↑ positive · ↓ negative`
          : "",
      );
      // The label visuals/counts/filters all key off the selected basetag.
      refreshBasetagLabelUI();
    });
  }

  // Inference-page basetag selector — fills the positive/negative dir fields.
  DOM.inferenceBasetag = document.getElementById("inference-basetag");
  if (DOM.inferenceBasetag) {
    DOM.inferenceBasetag.addEventListener("change", () => {
      const b = STATE.basetags.find((x) => x.id === DOM.inferenceBasetag.value);
      if (!b) return;
      if (DOM.inferencePosDir) DOM.inferencePosDir.value = `${b.name}_positive`;
      if (DOM.inferenceNegDir) DOM.inferenceNegDir.value = `${b.name}_negative`;
      setInferenceMessage(`Folders set from basetag “${b.name}”. Edit if needed, then Save.`);
    });
  }

  populateBasetagSelectors();
}

function handleBasetagsList(msg) {
  STATE.basetags = msg.basetags || [];
  if (msg.error) {
    console.error("basetags error:", msg.error);
    setBasetagMessage(msg.error, "error");
  }

  // Resolve a freshly-saved "__new__" basetag to its concrete id so the editor
  // stays on it and Delete becomes available.
  if (STATE.pendingSelectBasetagName) {
    const b = STATE.basetags.find((x) => x.name === STATE.pendingSelectBasetagName);
    if (b) {
      STATE.editingBasetagId = b.id;
      STATE.pendingSelectBasetagName = null;
      DOM.btnDeleteBasetag?.classList.remove("hidden");
    }
  }

  renderBasetagList();
  populateBasetagSelectors();
  // Basetags arriving (or changing) can resolve/clear the preview selection, so
  // re-sync the positive/negative label visuals, counts and filters.
  refreshBasetagLabelUI();
}

function renderBasetagList() {
  const list = DOM.basetagList;
  if (!list) return;
  list.innerHTML = "";
  if (DOM.basetagCount) DOM.basetagCount.textContent = String(STATE.basetags.length);

  if (STATE.basetags.length === 0) {
    list.innerHTML =
      '<div class="placeholder">No basetags yet. Create one to get started.</div>';
    return;
  }

  STATE.basetags.forEach((b) => {
    const div = document.createElement("div");
    div.className = "processor-entry";
    if (b.id === STATE.editingBasetagId) div.classList.add("active");
    const name = document.createElement("span");
    name.className = "p-name";
    name.textContent = b.name;
    name.title = b.name;
    div.appendChild(name);
    div.addEventListener("click", () => editBasetag(b.id));
    list.appendChild(div);
  });
}

function openBasetagEditor() {
  DOM.basetagEditorPlaceholder.classList.add("hidden");
  DOM.basetagEditorPanel.classList.remove("hidden");
}

function closeBasetagEditor() {
  STATE.editingBasetagId = null;
  DOM.basetagEditorPanel.classList.add("hidden");
  DOM.basetagEditorPlaceholder.classList.remove("hidden");
  setBasetagMessage("");
  renderBasetagList();
}

function newBasetag() {
  STATE.editingBasetagId = "__new__";
  openBasetagEditor();
  DOM.basetagName.value = "";
  DOM.btnDeleteBasetag.classList.add("hidden");
  setBasetagMessage("Enter a name, then Save.");
  renderBasetagList();
  DOM.basetagName.focus();
}

function editBasetag(id) {
  const b = STATE.basetags.find((x) => x.id === id);
  if (!b) return;
  STATE.editingBasetagId = id;
  openBasetagEditor();
  DOM.basetagName.value = b.name;
  DOM.btnDeleteBasetag.classList.remove("hidden");
  setBasetagMessage("");
  renderBasetagList();
}

function saveBasetag() {
  const name = DOM.basetagName.value.trim();
  if (!name) {
    setBasetagMessage("Please give the basetag a name.", "error");
    return;
  }
  if (!BASETAG_NAME_RE.test(name)) {
    setBasetagMessage("Use letters, digits, _ or - only (no spaces).", "error");
    return;
  }
  const id = STATE.editingBasetagId === "__new__" ? undefined : STATE.editingBasetagId;
  const sent = sendMessage({ type: "save_basetag", basetag: { id, name } });
  if (!sent) {
    setBasetagMessage("Not connected — could not save.", "error");
    return;
  }
  if (!id) STATE.pendingSelectBasetagName = name;
  setBasetagMessage("Saved.", "success");
}

function deleteBasetag() {
  if (!STATE.editingBasetagId || STATE.editingBasetagId === "__new__") {
    closeBasetagEditor();
    return;
  }
  const b = STATE.basetags.find((x) => x.id === STATE.editingBasetagId);
  if (b && !confirm(`Delete basetag "${b.name}"? Its folders on disk are kept.`)) return;
  sendMessage({ type: "delete_basetag", id: STATE.editingBasetagId });
  closeBasetagEditor();
}

function setBasetagMessage(text, kind) {
  if (!DOM.basetagEditorMessage) return;
  DOM.basetagEditorMessage.textContent = text || "";
  DOM.basetagEditorMessage.className = "editor-message" + (kind ? " " + kind : "");
}

function setBasetagStatus(text, highlight) {
  const el = DOM.basetagStatus;
  if (!el) return;
  el.textContent = text || "";
  el.classList.toggle("has-filter", !!highlight);
}

// Fill both the preview and inference basetag <select>s from STATE.basetags,
// preserving the current selection where it still exists.
function populateBasetagSelectors() {
  const fill = (sel, keepValue) => {
    if (!sel) return;
    sel.innerHTML = '<option value="">— none —</option>';
    for (const b of STATE.basetags) {
      const opt = document.createElement("option");
      opt.value = b.id;
      opt.textContent = b.name;
      sel.appendChild(opt);
    }
    sel.value = STATE.basetags.some((b) => b.id === keepValue) ? keepValue : "";
  };

  // Preview selector mirrors STATE.selectedBasetagId; drop a stale selection.
  const sel = DOM.basetagSelect || document.getElementById("basetag-select");
  if (sel) {
    fill(sel, STATE.selectedBasetagId || "");
    if (STATE.selectedBasetagId && sel.value === "") {
      STATE.selectedBasetagId = null;
      saveState();
    }
  }

  // Inference selector is a transient convenience — keep whatever's chosen.
  const isel = DOM.inferenceBasetag || document.getElementById("inference-basetag");
  if (isel) fill(isel, isel.value);
}

function selectedBasetagName() {
  const b = STATE.basetags.find((x) => x.id === STATE.selectedBasetagId);
  return b ? b.name : "";
}

// ↑ / ↓ on the preview page: mark the current image positive / negative for the
// selected basetag. Mutually exclusive (sets one, clears the other); pressing
// the same direction again clears it. Each change keeps the on-disk
// `${base}_positive` / `${base}_negative` folder in sync via export/unexport.
function labelCurrent(sign) {
  const r = STATE.results[STATE.currentResultIndex];
  if (!r) return;
  const base = selectedBasetagName();
  if (!base) {
    setBasetagStatus("Pick a basetag first (create one on the Basetags page).", true);
    return;
  }

  const posTag = `${base}_positive`;
  const negTag = `${base}_negative`;
  const wantTag = sign > 0 ? posTag : negTag;
  const otherTag = sign > 0 ? negTag : posTag;
  const item = { source: r.source, result: r.result };

  if (hasTag(r, wantTag)) {
    // Same direction again → clear this label and remove the copied files.
    toggleTag(r, wantTag);
    sendMessage({ type: "unexport_images", folder: wantTag, items: [item] });
    setBasetagStatus(`Cleared “${wantTag}”.`);
  } else {
    // Switch sides: drop the opposite label + its copy first (mutual exclusion).
    if (hasTag(r, otherTag)) {
      toggleTag(r, otherTag);
      sendMessage({ type: "unexport_images", folder: otherTag, items: [item] });
    }
    toggleTag(r, wantTag);
    sendMessage({ type: "export_images", folder: wantTag, items: [item] });
    setBasetagStatus(`Tagged “${wantTag}” → ./${wantTag}/`);
  }

  // Reflect the change; selection stays put (no auto-advance). When a quick
  // filter is active the image's membership may change, so re-render the whole
  // list (and re-apply the active highlight); otherwise update just this row so
  // the list doesn't jump-scroll.
  if (STATE.filterPositive || STATE.filterNegative) {
    renderImageList();
    const cur = document.querySelector(
      `.image-item[data-index="${STATE.currentResultIndex}"]`,
    );
    if (cur) cur.classList.add("active");
  } else {
    refreshRowTagUI(STATE.currentResultIndex);
  }
  updateLabelCounts();
  updateCurrentLabel();
}

function handleUnexportResult(msg) {
  if (msg.errors && msg.errors.length) {
    const first = msg.errors[0];
    setBasetagStatus(
      `Removed from ./${msg.folder} with errors: ${first.error}`,
      true,
    );
  }
  // Success is silent — labelCurrent already showed the optimistic status.
}

// ============================================================================
// Init
// ============================================================================

document.addEventListener("DOMContentLoaded", () => {
  loadState();
  loadTags();
  loadCaches();
  initScanPage();
  initLogPage();
  initPreviewPage();
  initProcessorsPage();
  initInferencePage();
  initBasetagsPage();
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
