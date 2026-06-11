// ============================================================================
// Pilotless Poses — Deno HTTP + WebSocket Server
//
// Serves static frontend files, exposes a WebSocket endpoint for real-time
// communication, and spawns Python child processes for VitPose inference.
// ============================================================================

import { ensureDependencies } from "./setup.ts";

// All scan results and the manifest always live here, relative to the server's
// working directory. This is intentionally not configurable — the pose preview
// reads from this single, well-known location.
const OUTPUT_DIR = "./pose_results/";

// ---- Types ------------------------------------------------------------------

interface AppConfig {
  pythonPath: string;
  port: number;
  maxConcurrency: number;
  modelDir: string;
  venvDir: string;
}

interface ScanFolderMessage {
  type: "scan_folder";
  folderPath: string;
  extensions: string[];
}

interface GetResultsMessage {
  type: "get_results";
}

interface GetPoseDataMessage {
  type: "get_pose_data";
  resultPath: string;
}

interface CancelScanMessage {
  type: "cancel_scan";
}

interface BrowseFolderMessage {
  type: "browse_folder";
  path: string;
  page?: string;
}

interface Processor {
  id: string;
  name: string;
  code: string;
  active: boolean;
  createdAt: string;
  updatedAt: string;
}

interface ListProcessorsMessage {
  type: "list_processors";
}

interface SaveProcessorMessage {
  type: "save_processor";
  processor: {
    id?: string;
    name: string;
    code: string;
    active: boolean;
  };
}

interface DeleteProcessorMessage {
  type: "delete_processor";
  id: string;
}

interface ReorderProcessorsMessage {
  type: "reorder_processors";
  // Processor ids in the desired application order. Ids not present are kept
  // in their existing relative order after the listed ones.
  order: string[];
}

interface GetAllPoseDataMessage {
  type: "get_all_pose_data";
}

interface WriteFileMessage {
  type: "write_file";
  // Destination path, relative to the server's working directory.
  path: string;
  // File contents, base64-encoded (binary-safe).
  dataBase64: string;
}

interface InferenceMetrics {
  nPos: number;
  nNeg: number;
  nPosImages?: number;
  nNegImages?: number;
  trainAcc: number;
  testAcc: number;
  features: number;
  topFeatures: Array<{ name: string; importance: number }>;
}

interface InferenceModel {
  id: string;
  name: string;
  positiveDir: string;
  negativeDir: string;
  status: "untrained" | "training" | "trained" | "error";
  metrics?: InferenceMetrics;
  error?: string;
  trainedAt?: string;
  createdAt: string;
  updatedAt: string;
}

interface ListInferenceModelsMessage {
  type: "list_inference_models";
}

interface SaveInferenceModelMessage {
  type: "save_inference_model";
  model: {
    id?: string;
    name: string;
    positiveDir: string;
    negativeDir: string;
  };
}

interface DeleteInferenceModelMessage {
  type: "delete_inference_model";
  id: string;
}

interface TrainInferenceModelMessage {
  type: "train_inference_model";
  id: string;
}

interface ApplyInferenceModelMessage {
  type: "apply_inference_model";
  id: string;
  threshold: number;
}

type ClientMessage =
  | ScanFolderMessage
  | GetResultsMessage
  | GetPoseDataMessage
  | CancelScanMessage
  | BrowseFolderMessage
  | ListProcessorsMessage
  | SaveProcessorMessage
  | DeleteProcessorMessage
  | ReorderProcessorsMessage
  | GetAllPoseDataMessage
  | WriteFileMessage
  | ListInferenceModelsMessage
  | SaveInferenceModelMessage
  | DeleteInferenceModelMessage
  | TrainInferenceModelMessage
  | ApplyInferenceModelMessage;

interface ProgressMessage {
  type: "progress";
  current: number;
  total: number;
  file: string;
  status: "queued" | "processing" | "done" | "error";
  error?: string;
}

interface BatchCompleteMessage {
  type: "batch_complete";
  total: number;
  errors: Array<{ file: string; error: string }>;
}

interface ResultsListMessage {
  type: "results_list";
  results: Array<{
    source: string;
    result: string;
    status: string;
    error_msg?: string;
  }>;
}

interface PoseDataMessage {
  type: "pose_data";
  data: Record<string, unknown> | null;
  error?: string;
}

interface ErrorMessage {
  type: "error";
  message: string;
}

interface ScanLogMessage {
  type: "scan_log";
  // "info"   — server lifecycle events (scan started, N images found, …)
  // "python" — a line streamed live from the Python inference process's stderr
  // "error"  — something went wrong
  level: "info" | "python" | "error";
  line: string;
  ts: string;
}

interface ScanStatusMessage {
  type: "scan_status";
  // Snapshot sent to a (re)connecting client so it can restore the live log and
  // progress for a scan that started before this socket existed.
  scanning: boolean;
  current: number;
  total: number;
  logs: ScanLogMessage[];
}

interface BrowseResultMessage {
  type: "browse_result";
  path: string;
  parent: string | null;
  page?: string;
  entries: Array<{ name: string; isDirectory: boolean }>;
  error?: string;
}

interface ProcessorsListMessage {
  type: "processors_list";
  processors: Processor[];
  error?: string;
}

interface WriteFileResultMessage {
  type: "write_file_result";
  path: string;
  ok: boolean;
  error?: string;
}

interface AllPoseDataMessage {
  type: "all_pose_data";
  // Keyed by the manifest result path so the client can match against its
  // own results list. Entries include the source path and parsed pose data.
  items: Array<{
    source: string;
    result: string;
    status: string;
    error_msg?: string;
    data: Record<string, unknown> | null;
  }>;
  error?: string;
}

interface InferenceModelsListMessage {
  type: "inference_models_list";
  models: InferenceModel[];
  error?: string;
}

interface InferenceModelTrainedMessage {
  type: "inference_model_trained";
  id: string;
  metrics?: InferenceMetrics;
  error?: string;
}

interface InferenceModelAppliedMessage {
  type: "inference_model_applied";
  id: string;
  threshold: number;
  // Keyed by the manifest result path; pass=true means at least one person
  // scored at or above the threshold.
  results: Record<string, { prob: number; pass: boolean; scored?: boolean }>;
  error?: string;
}

type ServerMessage =
  | ProgressMessage
  | BatchCompleteMessage
  | ResultsListMessage
  | PoseDataMessage
  | ErrorMessage
  | ScanStatusMessage
  | BrowseResultMessage
  | ProcessorsListMessage
  | AllPoseDataMessage
  | WriteFileResultMessage
  | ScanLogMessage
  | InferenceModelsListMessage
  | InferenceModelTrainedMessage
  | InferenceModelAppliedMessage;

interface ManifestEntry {
  source: string;
  result: string;
  status: "ok" | "error";
  error_msg?: string;
}

interface Manifest {
  scan_date: string;
  source_folder: string;
  output_dir: string;
  results: ManifestEntry[];
}

// ---- Configuration ----------------------------------------------------------

const CONFIG_PATH = "./pose_app_config.json";

async function loadConfig(): Promise<AppConfig> {
  try {
    const raw = await Deno.readTextFile(CONFIG_PATH);
    const saved = JSON.parse(raw);
    return {
      pythonPath: saved.pythonPath || "python3",
      port: saved.port || 8000,
      maxConcurrency: saved.maxConcurrency || 3,
      modelDir: saved.modelDir || "./models",
      venvDir: saved.venvDir || "./.venv",
    };
  } catch {
    // First run — create default config
    const defaults: AppConfig = {
      pythonPath: "python3",
      port: 8000,
      maxConcurrency: 3,
      modelDir: "./models",
      venvDir: "./.venv",
    };
    try {
      await Deno.writeTextFile(
        CONFIG_PATH,
        JSON.stringify(defaults, null, 2) + "\n",
      );
      console.log(`Config created at ${CONFIG_PATH} — using defaults.`);
    } catch {
      console.warn("Could not write config file; using defaults.");
    }
    return defaults;
  }
}

// ---- MIME Types -------------------------------------------------------------

const MIME_TYPES: Record<string, string> = {
  ".html": "text/html; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".json": "application/json",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".gif": "image/gif",
  ".webp": "image/webp",
  ".bmp": "image/bmp",
  ".tiff": "image/tiff",
  ".ico": "image/x-icon",
  ".woff": "font/woff",
  ".woff2": "font/woff2",
};

function mimeType(path: string): string {
  const ext = path.slice(path.lastIndexOf(".")).toLowerCase();
  return MIME_TYPES[ext] || "application/octet-stream";
}

// ---- Image MIME -------------------------------------------------------------

const IMAGE_EXTENSIONS = new Set([
  ".jpg",
  ".jpeg",
  ".png",
  ".gif",
  ".webp",
  ".bmp",
  ".tiff",
  ".tif",
]);

function isImageExt(ext: string): boolean {
  return IMAGE_EXTENSIONS.has(ext.toLowerCase());
}

// ---- Static File Serving ----------------------------------------------------

async function serveStatic(url: URL): Promise<Response> {
  let pathname = url.pathname;
  // Prevent directory traversal
  if (pathname.includes("..")) {
    return new Response("Forbidden", { status: 403 });
  }
  // Default to index.html for root
  if (pathname === "/" || pathname === "") {
    pathname = "/index.html";
  }
  const filePath = `./public${pathname}`;

  try {
    const file = await Deno.open(filePath, { read: true });
    const stat = await file.stat();
    if (!stat.isFile) {
      file.close();
      return new Response("Not Found", { status: 404 });
    }

    // Read the file
    const contents = await Deno.readFile(filePath);
    file.close();

    return new Response(contents, {
      status: 200,
      headers: {
        "content-type": mimeType(filePath),
        "content-length": String(stat.size),
        // Cache static assets for 1 hour
        "cache-control": "no-cache",
      },
    });
  } catch (e) {
    if (e instanceof Deno.errors.NotFound) {
      return new Response("Not Found", { status: 404 });
    }
    console.error("Error serving static file:", e);
    return new Response("Internal Server Error", { status: 500 });
  }
}

// ---- File System Helpers ----------------------------------------------------

async function pathExists(p: string): Promise<boolean> {
  try {
    await Deno.stat(p);
    return true;
  } catch {
    return false;
  }
}

async function isDirectory(p: string): Promise<boolean> {
  try {
    const stat = await Deno.stat(p);
    return stat.isDirectory;
  } catch {
    return false;
  }
}

async function isFile(p: string): Promise<boolean> {
  try {
    const stat = await Deno.stat(p);
    return stat.isFile;
  } catch {
    return false;
  }
}

async function ensureDir(p: string): Promise<void> {
  try {
    await Deno.mkdir(p, { recursive: true });
  } catch (e) {
    if (!(e instanceof Deno.errors.AlreadyExists)) {
      throw e;
    }
  }
}

// ---- Image Scanning ---------------------------------------------------------

async function scanImages(
  folderPath: string,
  extensions: string[],
): Promise<string[]> {
  const images: string[] = [];
  const extSet = new Set(extensions.map((e) => e.toLowerCase()));

  async function walk(dir: string) {
    const entries = Deno.readDir(dir);
    for await (const entry of entries) {
      const fullPath = `${dir}/${entry.name}`;
      if (entry.isDirectory) {
        await walk(fullPath);
      } else if (entry.isFile) {
        const ext = entry.name.slice(entry.name.lastIndexOf(".")).toLowerCase();
        if (extSet.has(ext)) {
          images.push(fullPath);
        }
      }
    }
  }

  await walk(folderPath);
  images.sort();
  return images;
}

// ---- Python Process Management ----------------------------------------------

interface VitPoseResult {
  image_path: string;
  success: boolean;
  people_count: number;
  people: Array<{
    person_id: number;
    keypoints: Array<{
      name: string;
      x: number;
      y: number;
      confidence: number;
    }>;
  }>;
  error?: string;
}

interface VitPoseBatchSummary {
  total_images: number;
  successful: number;
  failed: number;
}

async function runPythonBatchInference(
  pythonPath: string,
  imagePaths: string[],
  modelDir: string,
  signal?: AbortSignal,
  onLog?: (line: string) => void,
  onResult?: (result: VitPoseResult) => Promise<void> | void,
): Promise<VitPoseBatchSummary> {
  const args = [
    "python/f_o_info_vitpose.py",
    ...imagePaths,
    "--model-dir", modelDir,
  ];

  const command = new Deno.Command(pythonPath, {
    args,
    stdout: "piped",
    stderr: "piped",
    signal,
  });

  const child = command.spawn();

  // stdout is now JSONL: one {"type":"result",...} object per image as it
  // finishes, then a final {"type":"summary",...}. Parse line-by-line and hand
  // each result to onResult so it can be persisted immediately.
  let summary: VitPoseBatchSummary = {
    total_images: imagePaths.length,
    successful: 0,
    failed: 0,
  };
  const handleStdoutLine = async (line: string) => {
    let msg: { type?: string; data?: VitPoseResult } & Partial<VitPoseBatchSummary>;
    try {
      msg = JSON.parse(line);
    } catch {
      // Non-JSON stray line on stdout — ignore rather than fail the whole scan.
      return;
    }
    if (msg.type === "result" && msg.data) {
      await onResult?.(msg.data);
    } else if (msg.type === "summary") {
      summary = {
        total_images: msg.total_images ?? imagePaths.length,
        successful: msg.successful ?? 0,
        failed: msg.failed ?? 0,
      };
    }
  };
  const stdoutDone = (async () => {
    const decoder = new TextDecoder();
    let buf = "";
    for await (const chunk of child.stdout) {
      buf += decoder.decode(chunk, { stream: true });
      let nl: number;
      while ((nl = buf.indexOf("\n")) !== -1) {
        const line = buf.slice(0, nl).trim();
        buf = buf.slice(nl + 1);
        if (line) await handleStdoutLine(line);
      }
    }
    buf += decoder.decode();
    const last = buf.trim();
    if (last) await handleStdoutLine(last);
  })();

  // stderr carries human-readable progress (model loading, per-image lines,
  // download progress). Stream it line-by-line so the client sees activity in
  // real time instead of one buffered dump when the process exits. We also keep
  // the lines around for error reporting on a non-zero exit.
  const stderrLines: string[] = [];
  const stderrDone = (async () => {
    const decoder = new TextDecoder();
    let buf = "";
    for await (const chunk of child.stderr) {
      buf += decoder.decode(chunk, { stream: true });
      // Split on newlines AND carriage returns so tqdm-style progress bars,
      // which rewrite the current line with \r, surface as they update.
      const parts = buf.split(/\r\n|\r|\n/);
      buf = parts.pop() ?? "";
      for (const line of parts) {
        if (!line.length) continue;
        stderrLines.push(line);
        if (line.trim()) onLog?.(line);
      }
    }
    buf += decoder.decode();
    if (buf.length) {
      stderrLines.push(buf);
      if (buf.trim()) onLog?.(buf);
    }
  })();

  await stdoutDone;
  await stderrDone;
  const { code } = await child.status;
  const stderrText = stderrLines.join("\n");

  if (stderrText) {
    // Show last 2000 chars — that's where the actual error usually is
    console.error("Python stderr (last 2000 chars):", stderrText.slice(-2000));
  }

  if (code !== 0) {
    // Grab the last meaningful line, skipping CUDA/XLA registration noise
    const lines = stderrText.split("\n").filter(l => l.trim() && !l.includes("Unable to register"));
    const tail = lines.slice(-5).join("\n");
    throw new Error(`Python exited with code ${code}: ${tail.slice(0, 800)}`);
  }

  return summary;
}

// ---- Result Storage ---------------------------------------------------------

async function saveResult(
  outputDir: string,
  imageBasename: string,
  result: Record<string, unknown>,
): Promise<string> {
  await ensureDir(outputDir);
  const filename = `${imageBasename}_pose.json`;
  const filePath = `${outputDir}/${filename}`;
  await Deno.writeTextFile(filePath, JSON.stringify(result, null, 2) + "\n");
  return filename;
}

async function loadManifest(outputDir: string): Promise<Manifest> {
  const manifestPath = `${outputDir}/manifest.json`;
  try {
    const raw = await Deno.readTextFile(manifestPath);
    const m = JSON.parse(raw) as Manifest;
    // Ensure output_dir is populated for backward compat
    if (!m.output_dir) m.output_dir = outputDir;
    return m;
  } catch {
    return {
      scan_date: new Date().toISOString(),
      source_folder: "",
      output_dir: outputDir,
      results: [],
    };
  }
}

async function saveManifest(outputDir: string, manifest: Manifest): Promise<void> {
  await ensureDir(outputDir);
  const manifestPath = `${outputDir}/manifest.json`;
  await Deno.writeTextFile(
    manifestPath,
    JSON.stringify(manifest, null, 2) + "\n",
  );
}

// ---- WebSocket Connection Manager -------------------------------------------

interface ScanState {
  abortController: AbortController;
  total: number;
  processed: number;
  errors: Array<{ file: string; error: string }>;
  scanning: boolean;
}

class ConnectionManager {
  private sockets: Set<WebSocket> = new Set();

  add(ws: WebSocket): void {
    this.sockets.add(ws);
  }

  remove(ws: WebSocket): void {
    // A running scan is global (see `activeScan`), not tied to this socket, so
    // closing a page/tab does NOT abort it — the scan keeps running and the live
    // log is replayed to the next client that connects.
    this.sockets.delete(ws);
  }

  send(ws: WebSocket, msg: ServerMessage): void {
    if (ws.readyState === WebSocket.OPEN) {
      try {
        ws.send(JSON.stringify(msg));
      } catch (e) {
        console.error("Failed to send WebSocket message:", e);
      }
    }
  }

  // Send to every connected client so scan progress/logs reach all open tabs
  // (and survive a reload that opens a fresh socket).
  broadcast(msg: ServerMessage): void {
    for (const ws of this.sockets) this.send(ws, msg);
  }
}

// ---- Global scan state ------------------------------------------------------
//
// A scan is server-global, not per-connection: it must outlive the socket that
// started it so reloading the page doesn't kill an in-progress scan. Live-log
// lines are buffered here too so a reconnecting client can replay them.

let activeScan: ScanState | null = null;

const RECENT_LOG_CAP = 1000;
const recentLogs: ScanLogMessage[] = [];

function bufferLog(msg: ScanLogMessage): void {
  recentLogs.push(msg);
  if (recentLogs.length > RECENT_LOG_CAP) {
    recentLogs.splice(0, recentLogs.length - RECENT_LOG_CAP);
  }
}

function broadcastScanLog(
  connManager: ConnectionManager,
  line: string,
  level: "info" | "python" | "error" = "info",
): void {
  const msg: ScanLogMessage = {
    type: "scan_log",
    level,
    line,
    ts: new Date().toISOString(),
  };
  bufferLog(msg);
  connManager.broadcast(msg);
}

// Snapshot for a (re)connecting client: current progress + buffered log lines.
function scanStatusMessage(): ScanStatusMessage {
  return {
    type: "scan_status",
    scanning: !!(activeScan && activeScan.scanning),
    current: activeScan ? activeScan.processed : 0,
    total: activeScan ? activeScan.total : 0,
    logs: recentLogs,
  };
}

// ---- Scanning Orchestration -------------------------------------------------

async function handleScanFolder(
  ws: WebSocket,
  msg: ScanFolderMessage,
  config: AppConfig,
  connManager: ConnectionManager,
): Promise<void> {
  const { folderPath, extensions } = msg;

  // Stream a line to every client's Live Log page (and buffer it so a client
  // that reloads mid-scan can replay it).
  const log = (line: string, level: "info" | "python" | "error" = "info") =>
    broadcastScanLog(connManager, line, level);

  // A fresh scan starts a fresh live-log buffer.
  recentLogs.length = 0;

  log(`▶ Scan requested: ${folderPath} (extensions: ${extensions.join(", ")})`);

  // Validate folder
  if (!(await isDirectory(folderPath))) {
    log(`Folder not found or not a directory: ${folderPath}`, "error");
    connManager.send(ws, {
      type: "error",
      message: `Folder not found or not a directory: ${folderPath}`,
    });
    return;
  }

  const outputDir = OUTPUT_DIR;

  // Cancel any existing (global) scan before starting a new one.
  if (activeScan) activeScan.abortController.abort();

  const abortController = new AbortController();

  try {
    // Scan for images
    log(`Scanning ${folderPath} for images…`);
    const images = await scanImages(folderPath, extensions);

    if (images.length === 0) {
      log(`No images found matching ${extensions.join(", ")}`, "error");
      connManager.send(ws, {
        type: "error",
        message: `No images found in ${folderPath} matching ${extensions.join(", ")}`,
      });
      return;
    }

    log(`Found ${images.length} image(s). Output dir: ${outputDir}`);

    // Ensure output directory exists
    await ensureDir(outputDir);

    // Initialize or load manifest
    const manifest = await loadManifest(outputDir);
    manifest.scan_date = new Date().toISOString();
    manifest.source_folder = folderPath;
    manifest.output_dir = outputDir;

    // Remove stale entries for images being reprocessed. entry.source stores
    // the full image path, so compare against the full paths (not basenames),
    // otherwise old entries are never dropped and duplicates accumulate.
    const imageSet = new Set(images);
    manifest.results = manifest.results.filter((entry) =>
      !imageSet.has(entry.source)
    );

    // Create the global scan state (survives the originating socket closing).
    const scanState: ScanState = {
      abortController,
      total: images.length,
      processed: 0,
      errors: [],
      scanning: true,
    };
    activeScan = scanState;

    // Send queued status for all files
    for (const img of images) {
      connManager.broadcast({
        type: "progress",
        current: 0,
        total: images.length,
        file: img.split("/").pop()!,
        status: "queued",
      });
    }

    // Send processing status for all files
    for (const img of images) {
      connManager.broadcast({
        type: "progress",
        current: 0,
        total: images.length,
        file: img.split("/").pop()!,
        status: "processing",
      });
    }

    // Persist the manifest incrementally, but throttled: rewriting the whole
    // manifest on every single image would be O(n²) on a large scan. Per-image
    // JSON files are always written immediately (durable); the manifest catches
    // up at most every MANIFEST_WRITE_INTERVAL_MS and is force-written at the end.
    const MANIFEST_WRITE_INTERVAL_MS = 1500;
    let lastManifestWrite = 0;
    const persistManifest = async (force = false): Promise<void> => {
      const now = Date.now();
      if (!force && now - lastManifestWrite < MANIFEST_WRITE_INTERVAL_MS) return;
      lastManifestWrite = now;
      try {
        await saveManifest(outputDir, manifest);
      } catch (e) {
        log(`⚠ Could not write manifest: ${e instanceof Error ? e.message : e}`, "error");
      }
    };

    // Called for each image as the Python process finishes it: write the result
    // to disk right away so nothing is lost if the scan is interrupted.
    const onResult = async (result: VitPoseResult): Promise<void> => {
      if (abortController.signal.aborted) return;
      const imagePath = result.image_path;
      const basename = imagePath.split("/").pop()!;

      if (result.success) {
        const poseData = {
          image: imagePath,
          people: result.people.map((p) => ({
            id: p.person_id,
            keypoints: p.keypoints,
          })),
          image_width: 0, // not provided by the batch script
          image_height: 0,
        };
        const resultFilename = await saveResult(
          outputDir,
          basename.replace(/\.[^.]+$/, ""),
          poseData,
        );
        manifest.results.push({
          source: imagePath,
          result: `${outputDir}/${resultFilename}`,
          status: "ok",
        });
        scanState.processed++;
        connManager.broadcast({
          type: "progress",
          current: scanState.processed,
          total: images.length,
          file: basename,
          status: "done",
        });
      } else {
        const errorMsg = result.error || "Unknown error";
        scanState.errors.push({ file: basename, error: errorMsg });
        manifest.results.push({
          source: imagePath,
          result: `${outputDir}/${basename.replace(/\.[^.]+$/, "")}_pose.json`,
          status: "error",
          error_msg: errorMsg,
        });
        scanState.processed++;
        connManager.broadcast({
          type: "progress",
          current: scanState.processed,
          total: images.length,
          file: basename,
          status: "error",
          error: errorMsg,
        });
      }
      await persistManifest();
    };

    try {
      // Call the VitPose batch script — processes all images in one invocation
      // (models load once, then images stream back one result at a time).
      const modelDir = config.modelDir || "./models";
      log("Launching VitPose inference (this can take a while on first run)…");
      const summary = await runPythonBatchInference(
        config.pythonPath,
        images,
        modelDir,
        abortController.signal,
        (line) => log(line, "python"),
        onResult,
      );

      if (abortController.signal.aborted) return;

      log(`Inference finished: ${summary.successful} ok, ${summary.failed} failed`);
    } catch (e) {
      if (abortController.signal.aborted) return;
      const errorMsg = e instanceof Error ? e.message : String(e);
      log(`Inference failed: ${errorMsg}`, "error");

      // Mark every image not yet recorded as errored.
      for (const imagePath of images) {
        const basename = imagePath.split("/").pop()!;
        if (manifest.results.some((r) => r.source === imagePath)) continue;
        scanState.errors.push({ file: basename, error: errorMsg });
        manifest.results.push({
          source: imagePath,
          result: `${outputDir}/${basename.replace(/\.[^.]+$/, "")}_pose.json`,
          status: "error",
          error_msg: errorMsg,
        });
        scanState.processed++;
        connManager.broadcast({
          type: "progress",
          current: scanState.processed,
          total: images.length,
          file: basename,
          status: "error",
          error: errorMsg,
        });
      }
    }

    // Final manifest write to capture everything.
    await persistManifest(true);
    log(`Saved results + manifest to ${outputDir}`);

    // Send batch complete
    if (!abortController.signal.aborted) {
      log(
        `✓ Scan complete: ${images.length - scanState.errors.length} of ` +
          `${images.length} succeeded, ${scanState.errors.length} error(s)`,
      );
      connManager.broadcast({
        type: "batch_complete",
        total: images.length,
        errors: scanState.errors,
      });
    }

    scanState.scanning = false;
    if (activeScan === scanState) activeScan = null;
  } catch (e) {
    if (!abortController.signal.aborted) {
      const errorMsg = e instanceof Error ? e.message : String(e);
      log(`Scan failed: ${errorMsg}`, "error");
      connManager.send(ws, {
        type: "error",
        message: `Scan failed: ${errorMsg}`,
      });
    }
    if (activeScan && activeScan.abortController === abortController) {
      activeScan = null;
    }
  }
}

async function handleGetResults(
  ws: WebSocket,
  _msg: GetResultsMessage,
  connManager: ConnectionManager,
): Promise<void> {
  // The pose preview always shows ALL scanned results from the single
  // manifest in OUTPUT_DIR — it is not scoped to any selected folder.
  const manifestPath = `${OUTPUT_DIR}/manifest.json`;
  console.log(`[get_results] reading ${manifestPath} (cwd: ${Deno.cwd()})`);

  if (!(await pathExists(manifestPath))) {
    console.warn(`[get_results] no manifest at ${manifestPath} — returning 0 results`);
    connManager.send(ws, { type: "results_list", results: [] });
    return;
  }

  const manifest = await loadManifest(OUTPUT_DIR);

  // Dedupe by source, keeping the most recent entry. Older manifests written
  // before the dedup fix can contain repeated entries for the same image.
  const bySource = new Map<string, typeof manifest.results[number]>();
  for (const entry of manifest.results) bySource.set(entry.source, entry);
  const results = [...bySource.values()];

  console.log(
    `[get_results] returning ${results.length} result(s) to client` +
      (results.length !== manifest.results.length
        ? ` (deduped from ${manifest.results.length})`
        : ""),
  );
  connManager.send(ws, {
    type: "results_list",
    results,
  });
}

async function handleGetPoseData(
  ws: WebSocket,
  msg: GetPoseDataMessage,
  connManager: ConnectionManager,
): Promise<void> {
  const { resultPath } = msg;

  // Security: prevent path traversal
  if (resultPath.includes("..")) {
    connManager.send(ws, {
      type: "pose_data",
      data: null,
      error: "Invalid result path",
    });
    return;
  }

  // The resultPath is typically a full path like "./pose_results/photo_pose.json"
  // or "/absolute/path/to/pose_results/photo_pose.json"
  // Try the path directly, then fall back to relative lookups
  const candidates: string[] = [];

  if (resultPath.startsWith("/") || resultPath.startsWith("./")) {
    candidates.push(resultPath);
  } else {
    // It might be just a filename — look in common locations
    candidates.push(`./pose_results/${resultPath}`);
    candidates.push(resultPath);
  }

  // Also try resolving relative to CWD
  if (!resultPath.startsWith("/")) {
    candidates.push(`${Deno.cwd()}/${resultPath}`);
  }

  try {
    for (const fullPath of candidates) {
      if (await pathExists(fullPath)) {
        const raw = await Deno.readTextFile(fullPath);
        const data = JSON.parse(raw);
        connManager.send(ws, { type: "pose_data", data });
        return;
      }
    }

    // Not found in any candidate location
    connManager.send(ws, {
      type: "pose_data",
      data: null,
      error: `Result file not found: ${resultPath}`,
    });
  } catch (e) {
    connManager.send(ws, {
      type: "pose_data",
      data: null,
      error: e instanceof Error ? e.message : String(e),
    });
  }
}

// ---- Filesystem Browsing -----------------------------------------------------

async function handleBrowseFolder(
  ws: WebSocket,
  msg: BrowseFolderMessage,
  connManager: ConnectionManager,
): Promise<void> {
  let targetPath = msg.path || "/";
  const page = msg.page; // echo back so the client knows which browser to update

  // Resolve to absolute, normalizing ".." / "."
  try {
    targetPath = await Deno.realPath(targetPath);
  } catch {
    connManager.send(ws, {
      type: "browse_result",
      path: targetPath,
      parent: null,
      page,
      entries: [],
      error: `Path not found: ${targetPath}`,
    });
    return;
  }

  const stat = await Deno.stat(targetPath).catch(() => null);
  if (!stat || !stat.isDirectory) {
    connManager.send(ws, {
      type: "browse_result",
      path: targetPath,
      parent: null,
      page,
      entries: [],
      error: !stat ? `Path not found: ${targetPath}` : "Not a directory",
    });
    return;
  }

  const entries: Array<{ name: string; isDirectory: boolean }> = [];
  try {
    for await (const entry of Deno.readDir(targetPath)) {
      entries.push({ name: entry.name, isDirectory: entry.isDirectory });
    }
  } catch {
    connManager.send(ws, {
      type: "browse_result",
      path: targetPath,
      parent: null,
      page,
      entries: [],
      error: "Permission denied",
    });
    return;
  }

  // Sort: directories first, then alphabetical
  entries.sort((a, b) => {
    if (a.isDirectory !== b.isDirectory) return a.isDirectory ? -1 : 1;
    return a.name.localeCompare(b.name);
  });

  const parent = targetPath === "/"
    ? null
    : targetPath.split("/").slice(0, -1).join("/") || "/";

  connManager.send(ws, {
    type: "browse_result",
    path: targetPath,
    parent,
    page,
    entries,
  });
}

// ---- Processors -------------------------------------------------------------
//
// A "processor" is a user-authored JavaScript predicate that runs (in the
// browser) against each image's pose data. Definitions are persisted here so
// they survive restarts; the actual execution happens client-side.

const PROCESSORS_PATH = "./processors.json";
// Committed, version-controlled list of built-in example processors. It is
// regenerated on startup and always merged into the user's processors so the
// examples are available in the app even on an existing processors.json.
const EXAMPLE_PROCESSORS_PATH = "./processors.example.json";
// Fixed timestamp for the built-in examples so the generated file is stable
// (no git churn between runs).
const EXAMPLE_TIMESTAMP = "2026-01-01T00:00:00.000Z";

// Built-in example processor (always available). Filters out images where any
// detected person has a hand raised — a wrist sitting above the matching
// shoulder. Note the image y-axis grows downward, so "above" means smaller y.
const HANDS_IN_AIR_CODE =
  `// Keep images where a person has a hand in the air (a wrist positioned above
// the corresponding shoulder); filter out images where nobody raised a hand.
// Image y grows downward, so "above" means a smaller y value.
const MIN_CONF = 0.3;
for (let i = 0; i < o_img.a_o_person.length; i++) {
  const o_person = o_img.a_o_person[i];
  const sides = [
    ["left", o_person.o_wrist.o_left, o_person.o_shoulder.o_left],
    ["right", o_person.o_wrist.o_right, o_person.o_shoulder.o_right],
  ];
  for (const [side, o_wrist, o_shoulder] of sides) {
    if (!o_wrist || !o_shoulder) continue;
    if (o_wrist.n_conf < MIN_CONF || o_shoulder.n_conf < MIN_CONF) continue;
    if (o_wrist.n_y < o_shoulder.n_y) {
      console.log(
        \`[hands_in_air] KEPT "\${o_img.s_name_file}": person \${i} \${side} \` +
          \`wrist (y=\${o_wrist.n_y.toFixed(1)}) is above shoulder \` +
          \`(y=\${o_shoulder.n_y.toFixed(1)}) — hand raised by \` +
          \`\${(o_shoulder.n_y - o_wrist.n_y).toFixed(1)}px\`
      );
      return true; // hand in the air — keep this image
    }
  }
}
console.log(
  \`[hands_in_air] FILTERED OUT "\${o_img.s_name_file}": no raised hands detected\`,
);
return false; // nobody has a hand raised — filter this image out
`;

// Built-in processor that copies every image reaching it into ./filtered_images.
// It keeps all images (always returns true), so place it LAST in the pipeline:
// then it only sees images that survived the earlier filters. Writing happens
// through f_deno_write_file, the server-side Deno.writeFile proxy.
const SAVE_IMAGES_CODE =
  `// Save each surviving image and its pose JSON into ./filtered_images, then
// keep it. Put this processor LAST in the pipeline so only the images that
// passed the earlier filters get saved.
//
// f_save_filtered() copies both the image file and its <name>_pose.json. It's
// fire-and-forget here: the predicate returns immediately while the copy runs.
const DEST_DIR = "./filtered_images";
f_save_filtered(DEST_DIR)
  .then((ok) =>
    console.log(
      ok
        ? \`[save_images] saved "\${o_img.s_name_file}" + pose JSON -> \${DEST_DIR}\`
        : \`[save_images] save incomplete for "\${o_img.s_name_file}"\`,
    )
  )
  .catch((err) =>
    console.error(\`[save_images] failed "\${o_img.s_name_file}":\`, err)
  );
return true; // never filters — only saves
`;

// Built-in example processors that ship with the app. Stable ids + fixed
// timestamps keep the generated processors.example.json deterministic.
function exampleProcessors(): Processor[] {
  return [
    {
      id: "example-hands_in_air",
      name: "hands_in_air",
      code: HANDS_IN_AIR_CODE,
      active: true,
      createdAt: EXAMPLE_TIMESTAMP,
      updatedAt: EXAMPLE_TIMESTAMP,
    },
    {
      id: "example-save_images",
      name: "save_images",
      code: SAVE_IMAGES_CODE,
      // Off by default so images aren't written to disk unexpectedly; the user
      // enables it and orders it last in the pipeline when they want to export.
      active: false,
      createdAt: EXAMPLE_TIMESTAMP,
      updatedAt: EXAMPLE_TIMESTAMP,
    },
  ];
}

// Write the committed example file from the built-in examples so it always
// exists and stays in sync with the code. Non-fatal if it can't be written.
async function ensureExampleProcessorsFile(): Promise<void> {
  try {
    await Deno.writeTextFile(
      EXAMPLE_PROCESSORS_PATH,
      JSON.stringify(exampleProcessors(), null, 2) + "\n",
    );
  } catch (e) {
    console.warn(`⚠ Could not write ${EXAMPLE_PROCESSORS_PATH}:`, e);
  }
}

async function loadProcessors(): Promise<Processor[]> {
  let userList: Processor[] = [];
  let existed = true;
  try {
    const raw = await Deno.readTextFile(PROCESSORS_PATH);
    const parsed = JSON.parse(raw);
    userList = Array.isArray(parsed) ? parsed as Processor[] : [];
  } catch {
    existed = false; // first run, or unreadable — start from the examples
  }

  // Refresh the code of built-in examples the user hasn't modified, so
  // improvements to the built-ins reach existing installs. An example counts as
  // "unmodified" while its timestamp still equals EXAMPLE_TIMESTAMP (editing it
  // in the UI bumps updatedAt), so user customizations are never clobbered.
  const examplesById = new Map(exampleProcessors().map((e) => [e.id, e]));
  let changed = false;
  for (const p of userList) {
    const ex = examplesById.get(p.id);
    if (ex && p.updatedAt === EXAMPLE_TIMESTAMP && p.code !== ex.code) {
      p.code = ex.code;
      p.name = ex.name;
      changed = true;
    }
  }

  // Ensure every built-in example is always available: append any example
  // whose name isn't already present in the user's list.
  const names = new Set(userList.map((p) => p.name));
  let added = false;
  for (const ex of exampleProcessors()) {
    if (!names.has(ex.name)) {
      userList.push(ex);
      added = true;
    }
  }

  if (!existed || added || changed) {
    try {
      await saveProcessors(userList);
    } catch {
      // Non-fatal: still return the merged list even if we couldn't persist it.
    }
  }
  return userList;
}

async function saveProcessors(processors: Processor[]): Promise<void> {
  await Deno.writeTextFile(
    PROCESSORS_PATH,
    JSON.stringify(processors, null, 2) + "\n",
  );
}

async function handleListProcessors(
  ws: WebSocket,
  connManager: ConnectionManager,
): Promise<void> {
  try {
    const processors = await loadProcessors();
    connManager.send(ws, { type: "processors_list", processors });
  } catch (e) {
    connManager.send(ws, {
      type: "processors_list",
      processors: [],
      error: e instanceof Error ? e.message : String(e),
    });
  }
}

async function handleSaveProcessor(
  ws: WebSocket,
  msg: SaveProcessorMessage,
  connManager: ConnectionManager,
): Promise<void> {
  try {
    const processors = await loadProcessors();
    const now = new Date().toISOString();
    const incoming = msg.processor;

    if (incoming.id) {
      const existing = processors.find((p) => p.id === incoming.id);
      if (existing) {
        existing.name = incoming.name;
        existing.code = incoming.code;
        existing.active = incoming.active;
        existing.updatedAt = now;
      } else {
        processors.push({
          id: incoming.id,
          name: incoming.name,
          code: incoming.code,
          active: incoming.active,
          createdAt: now,
          updatedAt: now,
        });
      }
    } else {
      processors.push({
        id: crypto.randomUUID(),
        name: incoming.name,
        code: incoming.code,
        active: incoming.active,
        createdAt: now,
        updatedAt: now,
      });
    }

    await saveProcessors(processors);
    connManager.send(ws, { type: "processors_list", processors });
  } catch (e) {
    connManager.send(ws, {
      type: "processors_list",
      processors: await loadProcessors().catch(() => []),
      error: e instanceof Error ? e.message : String(e),
    });
  }
}

async function handleDeleteProcessor(
  ws: WebSocket,
  msg: DeleteProcessorMessage,
  connManager: ConnectionManager,
): Promise<void> {
  try {
    const processors = (await loadProcessors()).filter((p) => p.id !== msg.id);
    await saveProcessors(processors);
    connManager.send(ws, { type: "processors_list", processors });
  } catch (e) {
    connManager.send(ws, {
      type: "processors_list",
      processors: await loadProcessors().catch(() => []),
      error: e instanceof Error ? e.message : String(e),
    });
  }
}

async function handleReorderProcessors(
  ws: WebSocket,
  msg: ReorderProcessorsMessage,
  connManager: ConnectionManager,
): Promise<void> {
  try {
    const processors = await loadProcessors();
    const order = Array.isArray(msg.order) ? msg.order : [];

    // Rebuild the list in the requested order; any processor whose id wasn't
    // listed is appended afterwards, preserving its original relative order.
    const byId = new Map(processors.map((p) => [p.id, p]));
    const reordered: Processor[] = [];
    for (const id of order) {
      const p = byId.get(id);
      if (p) {
        reordered.push(p);
        byId.delete(id);
      }
    }
    for (const p of byId.values()) reordered.push(p);

    await saveProcessors(reordered);
    connManager.send(ws, { type: "processors_list", processors: reordered });
  } catch (e) {
    connManager.send(ws, {
      type: "processors_list",
      processors: await loadProcessors().catch(() => []),
      error: e instanceof Error ? e.message : String(e),
    });
  }
}

// Proxy for Deno.writeFile exposed to client-side processors as
// `f_deno_write_file`. Writes base64-decoded contents to a path relative to the
// server's working directory, creating parent directories as needed.
async function handleWriteFile(
  ws: WebSocket,
  msg: WriteFileMessage,
  connManager: ConnectionManager,
): Promise<void> {
  const path = msg.path ?? "";
  try {
    // Keep writes inside the working directory: no parent traversal, no
    // absolute paths.
    if (!path || path.includes("..") || path.startsWith("/")) {
      throw new Error(`Refusing to write to unsafe path: "${path}"`);
    }

    const bin = atob(msg.dataBase64 ?? "");
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);

    const dir = path.split("/").slice(0, -1).join("/");
    if (dir) await Deno.mkdir(dir, { recursive: true });
    await Deno.writeFile(path, bytes);

    connManager.send(ws, { type: "write_file_result", path, ok: true });
  } catch (e) {
    connManager.send(ws, {
      type: "write_file_result",
      path,
      ok: false,
      error: e instanceof Error ? e.message : String(e),
    });
  }
}

// ---- Inference models -------------------------------------------------------

const INFERENCE_MODELS_PATH = "./inference_models.json";
const INFERENCE_MODELS_DIR = "./inference_models";
const INFERENCE_SCRIPT = "python/f_o_train_inference.py";

async function loadInferenceModels(): Promise<InferenceModel[]> {
  try {
    const raw = await Deno.readTextFile(INFERENCE_MODELS_PATH);
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed as InferenceModel[] : [];
  } catch {
    return []; // first run or unreadable — start empty
  }
}

async function saveInferenceModels(models: InferenceModel[]): Promise<void> {
  await Deno.writeTextFile(
    INFERENCE_MODELS_PATH,
    JSON.stringify(models, null, 2) + "\n",
  );
}

function modelFilePath(id: string): string {
  return `${INFERENCE_MODELS_DIR}/${id}.joblib`;
}

// Run the train/predict Python script, streaming stderr to `onLog` and
// returning the trimmed stdout (the script's machine-readable JSON result).
async function runInferenceScript(
  pythonPath: string,
  args: string[],
  onLog?: (line: string) => void,
): Promise<string> {
  const command = new Deno.Command(pythonPath, {
    args: [INFERENCE_SCRIPT, ...args],
    stdout: "piped",
    stderr: "piped",
  });
  const child = command.spawn();

  const stdoutPromise = new Response(child.stdout).text();

  const stderrLines: string[] = [];
  const stderrDone = (async () => {
    const decoder = new TextDecoder();
    let buf = "";
    for await (const chunk of child.stderr) {
      buf += decoder.decode(chunk, { stream: true });
      const parts = buf.split(/\r\n|\r|\n/);
      buf = parts.pop() ?? "";
      for (const line of parts) {
        if (!line.trim()) continue;
        stderrLines.push(line);
        onLog?.(line);
      }
    }
    buf += decoder.decode();
    if (buf.trim()) {
      stderrLines.push(buf);
      onLog?.(buf);
    }
  })();

  const stdoutText = await stdoutPromise;
  await stderrDone;
  const { code } = await child.status;

  if (code !== 0) {
    const tail = stderrLines.slice(-6).join("\n");
    throw new Error(`Python exited with code ${code}: ${tail.slice(0, 800)}`);
  }
  return stdoutText.trim();
}

async function handleListInferenceModels(
  ws: WebSocket,
  connManager: ConnectionManager,
): Promise<void> {
  try {
    const models = await loadInferenceModels();
    connManager.send(ws, { type: "inference_models_list", models });
  } catch (e) {
    connManager.send(ws, {
      type: "inference_models_list",
      models: [],
      error: e instanceof Error ? e.message : String(e),
    });
  }
}

async function handleSaveInferenceModel(
  ws: WebSocket,
  msg: SaveInferenceModelMessage,
  connManager: ConnectionManager,
): Promise<void> {
  try {
    const models = await loadInferenceModels();
    const now = new Date().toISOString();
    const incoming = msg.model;

    const existing = incoming.id
      ? models.find((m) => m.id === incoming.id)
      : undefined;

    if (existing) {
      // Changing either folder invalidates a previously trained model.
      const dirsChanged = existing.positiveDir !== incoming.positiveDir ||
        existing.negativeDir !== incoming.negativeDir;
      existing.name = incoming.name;
      existing.positiveDir = incoming.positiveDir;
      existing.negativeDir = incoming.negativeDir;
      existing.updatedAt = now;
      if (dirsChanged && existing.status === "trained") {
        existing.status = "untrained";
        existing.metrics = undefined;
        existing.error = undefined;
      }
    } else {
      models.push({
        id: incoming.id || crypto.randomUUID(),
        name: incoming.name,
        positiveDir: incoming.positiveDir,
        negativeDir: incoming.negativeDir,
        status: "untrained",
        createdAt: now,
        updatedAt: now,
      });
    }

    await saveInferenceModels(models);
    connManager.send(ws, { type: "inference_models_list", models });
  } catch (e) {
    connManager.send(ws, {
      type: "inference_models_list",
      models: await loadInferenceModels().catch(() => []),
      error: e instanceof Error ? e.message : String(e),
    });
  }
}

async function handleDeleteInferenceModel(
  ws: WebSocket,
  msg: DeleteInferenceModelMessage,
  connManager: ConnectionManager,
): Promise<void> {
  try {
    const models = (await loadInferenceModels()).filter((m) => m.id !== msg.id);
    await saveInferenceModels(models);
    // Best-effort removal of the trained model file.
    try {
      await Deno.remove(modelFilePath(msg.id));
    } catch { /* file may not exist */ }
    connManager.send(ws, { type: "inference_models_list", models });
  } catch (e) {
    connManager.send(ws, {
      type: "inference_models_list",
      models: await loadInferenceModels().catch(() => []),
      error: e instanceof Error ? e.message : String(e),
    });
  }
}

function logToClient(
  ws: WebSocket,
  connManager: ConnectionManager,
  line: string,
  level: "info" | "python" | "error" = "python",
): void {
  connManager.send(ws, {
    type: "scan_log",
    level,
    line,
    ts: new Date().toISOString(),
  });
}

async function handleTrainInferenceModel(
  ws: WebSocket,
  msg: TrainInferenceModelMessage,
  config: AppConfig,
  connManager: ConnectionManager,
): Promise<void> {
  const models = await loadInferenceModels();
  const model = models.find((m) => m.id === msg.id);
  if (!model) {
    connManager.send(ws, {
      type: "inference_model_trained",
      id: msg.id,
      error: "Model not found",
    });
    return;
  }

  // Mark training and broadcast the updated list so the UI reflects it.
  model.status = "training";
  model.error = undefined;
  await saveInferenceModels(models);
  connManager.send(ws, { type: "inference_models_list", models });
  logToClient(ws, connManager, `■ Training model "${model.name}"…`, "info");

  try {
    await ensureDir(INFERENCE_MODELS_DIR);
    const out = modelFilePath(model.id);
    const stdout = await runInferenceScript(
      config.pythonPath,
      ["train", "--pos", model.positiveDir, "--neg", model.negativeDir, "--out", out],
      (line) => logToClient(ws, connManager, line),
    );

    let metrics: InferenceMetrics;
    try {
      metrics = JSON.parse(stdout) as InferenceMetrics;
    } catch {
      throw new Error(`Could not parse training metrics: ${stdout.slice(0, 300)}`);
    }

    // Reload in case the list changed during the (long) training run.
    const fresh = await loadInferenceModels();
    const m = fresh.find((x) => x.id === model.id);
    if (m) {
      m.status = "trained";
      m.metrics = metrics;
      m.error = undefined;
      m.trainedAt = new Date().toISOString();
      m.updatedAt = m.trainedAt;
      await saveInferenceModels(fresh);
    }

    logToClient(
      ws,
      connManager,
      `✔ Trained "${model.name}": test accuracy ${(metrics.testAcc * 100).toFixed(1)}%`,
      "info",
    );
    connManager.send(ws, { type: "inference_model_trained", id: model.id, metrics });
    connManager.send(ws, { type: "inference_models_list", models: fresh });
  } catch (e) {
    const errMsg = e instanceof Error ? e.message : String(e);
    const fresh = await loadInferenceModels();
    const m = fresh.find((x) => x.id === model.id);
    if (m) {
      m.status = "error";
      m.error = errMsg;
      m.updatedAt = new Date().toISOString();
      await saveInferenceModels(fresh);
    }
    logToClient(ws, connManager, `✗ Training failed: ${errMsg}`, "error");
    connManager.send(ws, {
      type: "inference_model_trained",
      id: model.id,
      error: errMsg,
    });
    connManager.send(ws, { type: "inference_models_list", models: fresh });
  }
}

// Resolve a manifest result path to a file that actually exists, mirroring the
// candidate lookup in handleGetPoseData. Returns null when nothing matches.
async function resolveResultPath(resultPath: string): Promise<string | null> {
  if (resultPath.includes("..")) return null;
  const candidates: string[] = [];
  if (resultPath.startsWith("/") || resultPath.startsWith("./")) {
    candidates.push(resultPath);
  } else {
    candidates.push(`${OUTPUT_DIR}/${resultPath}`);
    candidates.push(resultPath);
  }
  if (!resultPath.startsWith("/")) candidates.push(`${Deno.cwd()}/${resultPath}`);
  for (const c of candidates) {
    if (await pathExists(c)) return c;
  }
  return null;
}

async function handleApplyInferenceModel(
  ws: WebSocket,
  msg: ApplyInferenceModelMessage,
  config: AppConfig,
  connManager: ConnectionManager,
): Promise<void> {
  try {
    const models = await loadInferenceModels();
    const model = models.find((m) => m.id === msg.id);
    if (!model) throw new Error("Model not found");
    const out = modelFilePath(model.id);
    if (!(await isFile(out))) {
      throw new Error("Model is not trained yet — train it first.");
    }

    // Gather successful results from the manifest and resolve to real paths.
    const manifest = await loadManifest(OUTPUT_DIR);
    const bySource = new Map<string, ManifestEntry>();
    for (const entry of manifest.results) bySource.set(entry.source, entry);

    const resolvedToOriginal = new Map<string, string>();
    for (const entry of bySource.values()) {
      if (entry.status !== "ok") continue;
      const resolved = await resolveResultPath(entry.result);
      if (resolved) resolvedToOriginal.set(resolved, entry.result);
    }

    if (resolvedToOriginal.size === 0) {
      connManager.send(ws, {
        type: "inference_model_applied",
        id: msg.id,
        threshold: msg.threshold,
        results: {},
      });
      return;
    }

    const stdout = await runInferenceScript(
      config.pythonPath,
      [
        "predict",
        "--model", out,
        "--threshold", String(msg.threshold),
        ...resolvedToOriginal.keys(),
      ],
    );

    const byResolved = JSON.parse(stdout) as Record<
      string,
      { prob: number; pass: boolean; scored?: boolean }
    >;

    // Remap the script's keys (resolved paths) back to manifest result paths.
    const results: Record<string, { prob: number; pass: boolean; scored?: boolean }> = {};
    for (const [resolved, original] of resolvedToOriginal) {
      if (byResolved[resolved]) results[original] = byResolved[resolved];
    }

    connManager.send(ws, {
      type: "inference_model_applied",
      id: msg.id,
      threshold: msg.threshold,
      results,
    });
  } catch (e) {
    connManager.send(ws, {
      type: "inference_model_applied",
      id: msg.id,
      threshold: msg.threshold,
      results: {},
      error: e instanceof Error ? e.message : String(e),
    });
  }
}

async function handleGetAllPoseData(
  ws: WebSocket,
  _msg: GetAllPoseDataMessage,
  connManager: ConnectionManager,
): Promise<void> {
  // Operate on the same single manifest the preview page loads.
  if (!(await pathExists(`${OUTPUT_DIR}/manifest.json`))) {
    connManager.send(ws, { type: "all_pose_data", items: [] });
    return;
  }

  const manifest = await loadManifest(OUTPUT_DIR);

  if (manifest.results.length === 0) {
    connManager.send(ws, { type: "all_pose_data", items: [] });
    return;
  }

  const items: AllPoseDataMessage["items"] = [];
  for (const entry of manifest.results) {
    let data: Record<string, unknown> | null = null;
    if (entry.status === "ok" && !entry.result.includes("..")) {
      try {
        if (await pathExists(entry.result)) {
          data = JSON.parse(await Deno.readTextFile(entry.result));
        }
      } catch {
        data = null;
      }
    }
    items.push({
      source: entry.source,
      result: entry.result,
      status: entry.status,
      error_msg: entry.error_msg,
      data,
    });
  }

  connManager.send(ws, { type: "all_pose_data", items });
}

// ---- API Handlers -----------------------------------------------------------

async function serveImage(url: URL): Promise<Response> {
  const path = url.searchParams.get("path");
  if (!path) {
    return new Response("Missing path parameter", { status: 400 });
  }

  // Prevent path traversal
  if (path.includes("..")) {
    return new Response("Forbidden", { status: 403 });
  }

  try {
    if (!(await isFile(path))) {
      console.warn(
        `[serveImage] 404 — file not found: ${path} ` +
          `(if this is an absolute path from a manifest created on another machine, it won't exist here)`,
      );
      return new Response("File not found", { status: 404 });
    }

    const file = await Deno.open(path, { read: true });
    const stat = await file.stat();
    const ext = path.slice(path.lastIndexOf(".")).toLowerCase();

    if (!isImageExt(ext)) {
      file.close();
      return new Response("Not an image file", { status: 400 });
    }

    const contents = await Deno.readFile(path);
    file.close();

    return new Response(contents, {
      status: 200,
      headers: {
        "content-type": mimeType(path),
        "content-length": String(stat.size),
        "cache-control": "public, max-age=3600",
      },
    });
  } catch (e) {
    if (e instanceof Deno.errors.NotFound) {
      return new Response("File not found", { status: 404 });
    }
    console.error("Error serving image:", e);
    return new Response("Internal Server Error", { status: 500 });
  }
}

// ---- Request Handler --------------------------------------------------------

async function handleRequest(
  req: Request,
  config: AppConfig,
  connManager: ConnectionManager,
): Promise<Response> {
  const url = new URL(req.url);

  // WebSocket upgrade
  if (url.pathname === "/ws") {
    const { socket, response } = Deno.upgradeWebSocket(req);

    connManager.add(socket);

    // Replay the current scan's progress + buffered live-log to this client, so
    // reloading the page mid-scan restores the Live Log and progress instead of
    // showing nothing. Send on open (and immediately if already open).
    if (socket.readyState === WebSocket.OPEN) {
      connManager.send(socket, scanStatusMessage());
    } else {
      socket.addEventListener("open", () => {
        connManager.send(socket, scanStatusMessage());
      });
    }

    socket.addEventListener("message", async (event) => {
      let msg: ClientMessage;
      try {
        msg = JSON.parse(
          typeof event.data === "string" ? event.data : new TextDecoder().decode(event.data),
        ) as ClientMessage;
      } catch {
        connManager.send(socket, {
          type: "error",
          message: "Invalid JSON message",
        });
        return;
      }

      switch (msg.type) {
        case "scan_folder":
          await handleScanFolder(socket, msg, config, connManager);
          break;
        case "get_results":
          await handleGetResults(socket, msg, connManager);
          break;
        case "get_pose_data":
          await handleGetPoseData(socket, msg, connManager);
          break;
        case "cancel_scan": {
          const state = activeScan;
          if (state) {
            state.abortController.abort();
            state.scanning = false;
            activeScan = null;
            broadcastScanLog(connManager, "■ Scan cancelled by user.", "error");
            connManager.broadcast({
              type: "batch_complete",
              total: state.total,
              errors: state.errors,
            });
          }
          break;
        }
        case "browse_folder":
          await handleBrowseFolder(socket, msg, connManager);
          break;
        case "list_processors":
          await handleListProcessors(socket, connManager);
          break;
        case "save_processor":
          await handleSaveProcessor(socket, msg, connManager);
          break;
        case "delete_processor":
          await handleDeleteProcessor(socket, msg, connManager);
          break;
        case "reorder_processors":
          await handleReorderProcessors(socket, msg, connManager);
          break;
        case "get_all_pose_data":
          await handleGetAllPoseData(socket, msg, connManager);
          break;
        case "write_file":
          await handleWriteFile(socket, msg, connManager);
          break;
        case "list_inference_models":
          await handleListInferenceModels(socket, connManager);
          break;
        case "save_inference_model":
          await handleSaveInferenceModel(socket, msg, connManager);
          break;
        case "delete_inference_model":
          await handleDeleteInferenceModel(socket, msg, connManager);
          break;
        case "train_inference_model":
          await handleTrainInferenceModel(socket, msg, config, connManager);
          break;
        case "apply_inference_model":
          await handleApplyInferenceModel(socket, msg, config, connManager);
          break;
        default:
          connManager.send(socket, {
            type: "error",
            message: `Unknown message type: ${(msg as ClientMessage).type}`,
          });
      }
    });

    socket.addEventListener("close", () => {
      connManager.remove(socket);
    });

    socket.addEventListener("error", (event) => {
      console.error("WebSocket error:", event);
      connManager.remove(socket);
    });

    return response;
  }

  // API endpoints
  if (url.pathname === "/api/image" || url.pathname === "/api/thumbnail") {
    return await serveImage(url);
  }

  // Static file serving (fallback)
  return await serveStatic(url);
}

// ---- Test data (COCO val2017) -----------------------------------------------

// Downloaded once on startup, in the background. The download streams to a
// ".part" file and is only renamed to the final name when it finishes in full,
// so an interrupted download (server closed mid-transfer) leaves no final zip
// and is restarted from scratch next launch. Once the final zip exists it is
// extracted exactly once (guarded by a marker file) and never re-downloaded.
const TESTDATA_DIR = "./testdata";
const TESTDATA_URL = "http://images.cocodataset.org/zips/val2017.zip";
const TESTDATA_ZIP_PART = `${TESTDATA_DIR}/val2017.zip.part`;
const TESTDATA_ZIP = `${TESTDATA_DIR}/val2017.zip`;
const TESTDATA_EXTRACT_MARKER = `${TESTDATA_DIR}/.val2017_extracted`;

async function downloadTestDataZip(): Promise<void> {
  console.log(`[testdata] downloading ${TESTDATA_URL} …`);

  // Discard any partial file from a previous interrupted run, then re-download.
  try {
    await Deno.remove(TESTDATA_ZIP_PART);
  } catch { /* nothing to remove */ }

  const resp = await fetch(TESTDATA_URL);
  if (!resp.ok || !resp.body) {
    throw new Error(`download failed: HTTP ${resp.status}`);
  }
  const total = Number(resp.headers.get("content-length")) || 0;

  const file = await Deno.open(TESTDATA_ZIP_PART, {
    write: true,
    create: true,
    truncate: true,
  });
  let received = 0;
  let nextLogAt = 50 * 1024 * 1024; // log progress every ~50 MB
  try {
    for await (const chunk of resp.body) {
      // FsFile.write may write fewer bytes than supplied — loop until drained.
      let pos = 0;
      while (pos < chunk.length) pos += await file.write(chunk.subarray(pos));
      received += chunk.length;
      if (received >= nextLogAt) {
        const mb = (received / 1024 / 1024).toFixed(0);
        const pct = total ? ` (${((received / total) * 100).toFixed(0)}%)` : "";
        console.log(`[testdata] downloaded ${mb} MB${pct}`);
        nextLogAt += 50 * 1024 * 1024;
      }
    }
  } finally {
    file.close();
  }

  // A truncated transfer (dropped connection) won't match Content-Length — treat
  // it as a failure so the partial file is never promoted to the final name.
  if (total && received !== total) {
    throw new Error(`incomplete download: got ${received} of ${total} bytes`);
  }

  await Deno.rename(TESTDATA_ZIP_PART, TESTDATA_ZIP);
  console.log(`[testdata] download complete -> ${TESTDATA_ZIP}`);
}

async function extractTestDataZip(pythonPath: string): Promise<void> {
  console.log(`[testdata] extracting ${TESTDATA_ZIP} …`);
  // The zip nests everything under a top-level "val2017/" folder, so extract
  // straight into TESTDATA_DIR. Python's zipfile is always available here.
  const py =
    "import sys, zipfile\n" +
    "with zipfile.ZipFile(sys.argv[1]) as z:\n" +
    "    z.extractall(sys.argv[2])\n";
  const cmd = new Deno.Command(pythonPath, {
    args: ["-c", py, TESTDATA_ZIP, TESTDATA_DIR],
    stdout: "null",
    stderr: "piped",
  });
  const { code, stderr } = await cmd.output();
  if (code !== 0) {
    throw new Error(
      `unzip failed (code ${code}): ${new TextDecoder().decode(stderr).slice(-400)}`,
    );
  }
  await Deno.writeTextFile(TESTDATA_EXTRACT_MARKER, new Date().toISOString());
  console.log(`[testdata] ready -> ${TESTDATA_DIR}/val2017/`);
}

// Ensure the COCO val2017 test images are present. Safe to call unawaited; all
// errors are caught and logged so a download failure never crashes the server.
async function ensureTestData(pythonPath: string): Promise<void> {
  try {
    if (await pathExists(TESTDATA_EXTRACT_MARKER)) {
      console.log("[testdata] already downloaded and extracted — skipping.");
      return;
    }
    await ensureDir(TESTDATA_DIR);

    // Only a fully-downloaded zip (final name present) skips re-downloading.
    if (!(await isFile(TESTDATA_ZIP))) {
      await downloadTestDataZip();
    } else {
      console.log("[testdata] zip already downloaded — extracting.");
    }

    await extractTestDataZip(pythonPath);

    // Free the ~780 MB archive now that the images are extracted; the marker
    // keeps us from ever downloading it again.
    try {
      await Deno.remove(TESTDATA_ZIP);
    } catch { /* keep going even if removal fails */ }
  } catch (e) {
    console.error(
      `[testdata] background download/extract failed: ${
        e instanceof Error ? e.message : e
      }`,
    );
  }
}

// ---- Main -------------------------------------------------------------------

async function main() {
  const config = await loadConfig();
  const connManager = new ConnectionManager();

  // Keep the committed example-processors file in sync with the built-ins.
  await ensureExampleProcessorsFile();

  console.log("=".repeat(60));
  console.log("  Pilotless Poses Server");
  console.log("=".repeat(60));
  console.log(`  Python:     ${config.pythonPath}`);
  console.log(`  Concurrency: ${config.maxConcurrency}`);
  console.log(`  Listening:  http://localhost:${config.port}`);
  console.log(`  WebSocket:  ws://localhost:${config.port}/ws`);
  console.log("=".repeat(60));
  console.log("");

  // Check and install every dependency (Deno → Python → venv → pip packages)
  // before serving. If the environment can't be made to work, abort instead of
  // starting in a broken state.
  try {
    config.pythonPath = await ensureDependencies(config.pythonPath, config.venvDir);
    console.log(`✓ Using Python: ${config.pythonPath}`);
  } catch (e) {
    console.error(`\n✗ ${e instanceof Error ? e.message : e}`);
    console.error(
      "  Fix the dependency above (or run `deno task setup`) and start again.",
    );
    Deno.exit(1);
  }

  // Start server
  Deno.serve(
    {
      port: config.port,
      hostname: "0.0.0.0",
    },
    (req) => handleRequest(req, config, connManager),
  );

  console.log("Server started. Open http://localhost:8000 in your browser.");

  // Fetch the COCO val2017 test images in the background — never block serving.
  ensureTestData(config.pythonPath);
}

// Run
if (import.meta.main) {
  main();
}
