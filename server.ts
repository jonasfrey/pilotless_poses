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
  // Bumped whenever `code` changes. Clients key their result cache on it, so a
  // code edit invalidates the cache; toggling active does NOT bump it.
  version: number;
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

interface ExportImagesMessage {
  type: "export_images";
  // Destination folder, relative to the server's working directory.
  folder: string;
  // The images to export (the client-side kept set). result is the manifest
  // pose-JSON path; both the image and its pose JSON are copied.
  items: Array<{ source: string; result: string }>;
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
  // Bumped on every successful (re)train. Clients key their score cache on it,
  // so retraining invalidates the cache.
  version: number;
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
  | ExportImagesMessage
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

interface ExportResultMessage {
  type: "export_result";
  folder: string;
  copied: number;
  failed: number;
  errors: Array<{ file: string; error: string }>;
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
  | ExportResultMessage
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

// Reconcile the manifest against the pose JSON files actually on disk — files
// are the source of truth. Two directions of drift are repaired:
//   - an "ok" entry whose result file has vanished is dropped (the data is
//     genuinely gone); "error" entries are kept (their file was never written)
//   - a <name>_pose.json present on disk but absent from the manifest is
//     surfaced as a fresh "ok" entry (e.g. the manifest was deleted while the
//     results remained). The source image path is read back from the file's
//     own "image" field.
// The repaired manifest is persisted so subsequent reads are consistent.
async function reconcileManifestWithDisk(outputDir: string): Promise<Manifest> {
  const manifest = await loadManifest(outputDir);
  let changed = false;

  // Drop "ok" entries whose pose JSON has disappeared. pathExists (not a raw
  // string match) so absolute paths from manifests created elsewhere still
  // resolve correctly.
  const kept: ManifestEntry[] = [];
  for (const e of manifest.results) {
    if (e.status === "ok" && !(await pathExists(e.result))) {
      changed = true;
      continue;
    }
    kept.push(e);
  }

  // Surface pose JSON files on disk the manifest doesn't know about.
  const knownSources = new Set(kept.map((e) => e.source));
  try {
    for await (const entry of Deno.readDir(outputDir)) {
      if (!entry.isFile || !entry.name.endsWith("_pose.json")) continue;
      const resultPath = `${outputDir}/${entry.name}`;
      try {
        const parsed = JSON.parse(await Deno.readTextFile(resultPath));
        const source = typeof parsed.image === "string" ? parsed.image : null;
        if (!source || knownSources.has(source)) continue;
        kept.push({ source, result: resultPath, status: "ok" });
        knownSources.add(source);
        changed = true;
      } catch {
        // Unreadable / not JSON — skip it.
      }
    }
  } catch {
    // Output dir doesn't exist yet — nothing on disk to discover.
  }

  manifest.results = kept;
  if (changed) {
    try {
      await saveManifest(outputDir, manifest);
    } catch {
      // Non-fatal: still return the reconciled view even if we can't persist it.
    }
  }
  return manifest;
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

    // Skip images that already have a pose JSON on disk. The per-image
    // <name>_pose.json file is the source of truth (files always win): if it
    // exists we trust it and don't re-run inference, so re-scanning a folder is
    // idempotent and only processes images that are genuinely new. This path
    // matches what onResult writes, so reconciliation lines up exactly.
    const resultPathFor = (img: string) =>
      `${outputDir}/${img.split("/").pop()!.replace(/\.[^.]+$/, "")}_pose.json`;
    const toProcess: string[] = [];
    const alreadyDone: string[] = [];
    for (const img of images) {
      if (await isFile(resultPathFor(img))) alreadyDone.push(img);
      else toProcess.push(img);
    }

    // Re-derive this scan's manifest entries from authoritative state. Drop any
    // existing entries for these images (full-path match — entry.source stores
    // the full path), then re-add an `ok` entry for every already-done image
    // straight from its file on disk. New images get their entries from
    // onResult as inference produces them.
    const imageSet = new Set(images);
    manifest.results = manifest.results.filter((entry) =>
      !imageSet.has(entry.source)
    );
    for (const img of alreadyDone) {
      manifest.results.push({ source: img, result: resultPathFor(img), status: "ok" });
    }

    if (alreadyDone.length > 0) {
      log(
        `${alreadyDone.length} of ${images.length} image(s) already processed ` +
          `— skipping. ${toProcess.length} to do.`,
      );
    }

    // Create the global scan state (survives the originating socket closing).
    // Already-done images count as processed up front so progress is accurate.
    const scanState: ScanState = {
      abortController,
      total: images.length,
      processed: alreadyDone.length,
      errors: [],
      scanning: true,
    };
    activeScan = scanState;

    // Reflect already-done images as completed immediately.
    for (const img of alreadyDone) {
      connManager.broadcast({
        type: "progress",
        current: scanState.processed,
        total: images.length,
        file: img.split("/").pop()!,
        status: "done",
      });
    }

    // Nothing new to process: persist the reconciled manifest and finish. This
    // is the common case when re-scanning a folder that's already been done —
    // the scan is now a fast no-op instead of a full recompute.
    if (toProcess.length === 0) {
      await saveManifest(outputDir, manifest);
      log(`All ${images.length} image(s) already processed — nothing to do.`);
      connManager.broadcast({
        type: "batch_complete",
        total: images.length,
        errors: [],
      });
      scanState.scanning = false;
      if (activeScan === scanState) activeScan = null;
      return;
    }

    // Send queued, then processing, status for the images we will actually run.
    for (const img of toProcess) {
      connManager.broadcast({
        type: "progress",
        current: scanState.processed,
        total: images.length,
        file: img.split("/").pop()!,
        status: "queued",
      });
    }
    for (const img of toProcess) {
      connManager.broadcast({
        type: "progress",
        current: scanState.processed,
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
        toProcess,
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

      // Mark every image we tried to process but didn't record as errored.
      for (const imagePath of toProcess) {
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
  // The pose preview always shows ALL scanned results from the single manifest
  // in OUTPUT_DIR — it is not scoped to any selected folder. Reconcile against
  // the files on disk first (files always win) so a deleted manifest, a vanished
  // result file, or pose JSONs produced out-of-band are all reflected correctly.
  console.log(`[get_results] reconciling manifest in ${OUTPUT_DIR} (cwd: ${Deno.cwd()})`);
  const manifest = await reconcileManifestWithDisk(OUTPUT_DIR);

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

// Built-in processor that keeps only images carrying a given tag. The tag name
// is a parameter (f_gui), so the same processor works for any tag. Pairs with a
// coarse filter: run e.g. hands_in_air first, tag the keepers in the preview,
// then use this to keep exactly the tagged set before Exporting.
const HAS_TAG_CODE =
  `// Keep only images carrying a given tag. Set the tag in the "Processor
// controls" panel (defaults to "marked_hands_in_the_air").
//
// Tag images in the preview: type a tag in the "Active tag" box, then press "m"
// (or click the ★ on a row) to toggle that tag on an image. An image can carry
// several tags; o_img.a_s_tag is the array of its tags.
//
// Workflow: run a coarse filter like hands_in_air, tag the keepers, then enable
// this processor to narrow to the tagged set before using Export.
const tag = f_gui("Tag", "marked_hands_in_the_air");
return (o_img.a_s_tag || []).includes(tag);
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
      version: 1,
      createdAt: EXAMPLE_TIMESTAMP,
      updatedAt: EXAMPLE_TIMESTAMP,
    },
    {
      id: "example-has_tag",
      name: "has_tag",
      code: HAS_TAG_CODE,
      // Off by default: with no images tagged it would filter everything out.
      // The user tags images, then enables and orders it in the pipeline.
      active: false,
      version: 1,
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

  // Migration: drop retired built-ins from existing installs. save_images was
  // replaced by the dedicated Export action; all_marked_manual by the
  // tag-based has_tag processor. Custom processors (random ids) are untouched.
  const RETIRED_EXAMPLE_IDS = new Set([
    "example-save_images",
    "example-all_marked_manual",
  ]);
  const beforeLen = userList.length;
  userList = userList.filter((p) => !RETIRED_EXAMPLE_IDS.has(p.id));
  const removedRetired = userList.length !== beforeLen;

  // Refresh the code of built-in examples the user hasn't modified, so
  // improvements to the built-ins reach existing installs. An example counts as
  // "unmodified" while its timestamp still equals EXAMPLE_TIMESTAMP (editing it
  // in the UI bumps updatedAt), so user customizations are never clobbered.
  const examplesById = new Map(exampleProcessors().map((e) => [e.id, e]));
  let changed = false;
  for (const p of userList) {
    // Backfill version for processors saved before versioning existed.
    if (typeof p.version !== "number") { p.version = 1; changed = true; }
    const ex = examplesById.get(p.id);
    if (ex && p.updatedAt === EXAMPLE_TIMESTAMP && p.code !== ex.code) {
      p.code = ex.code;
      p.name = ex.name;
      p.version = (p.version || 1) + 1; // code changed -> invalidate caches
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

  if (!existed || added || changed || removedRetired) {
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
        // Bump version only when the code actually changes (so toggling active
        // or renaming doesn't invalidate a client's result cache).
        if (existing.code !== incoming.code) {
          existing.version = (existing.version || 1) + 1;
        }
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
          version: 1,
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
        version: 1,
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

// ---- Image export -----------------------------------------------------------
//
// A first-class action (not a processor): copy the images the client decided to
// keep — plus each one's pose JSON — into a folder, server-side via
// Deno.copyFile (no browser round-trip). Idempotent (overwrites in place) and
// reports a real per-file summary.
async function handleExportImages(
  ws: WebSocket,
  msg: ExportImagesMessage,
  connManager: ConnectionManager,
): Promise<void> {
  const folderRaw = (msg.folder ?? "").trim();
  // Keep the destination inside the working directory: strip leading slashes,
  // reject parent traversal — same rules as handleWriteFile.
  const folder = folderRaw.replace(/^\/+/, "");
  const items = Array.isArray(msg.items) ? msg.items : [];
  const log = (line: string, level: "info" | "python" | "error" = "info") =>
    broadcastScanLog(connManager, line, level);

  if (!folder || folder.includes("..")) {
    connManager.send(ws, {
      type: "export_result",
      folder: folderRaw,
      copied: 0,
      failed: 0,
      errors: [{ file: "", error: `Invalid destination folder: "${folderRaw}"` }],
    });
    return;
  }

  log(`▶ Exporting ${items.length} image(s) to ./${folder}…`);
  try {
    await ensureDir(folder);
  } catch (e) {
    const error = e instanceof Error ? e.message : String(e);
    log(`Export failed: could not create ./${folder}: ${error}`, "error");
    connManager.send(ws, {
      type: "export_result",
      folder,
      copied: 0,
      failed: items.length,
      errors: [{ file: "", error: `Could not create folder: ${error}` }],
    });
    return;
  }

  let copied = 0;
  let failed = 0;
  const errors: Array<{ file: string; error: string }> = [];

  for (const it of items) {
    const source = it?.source ?? "";
    const basename = source.split("/").pop() || source;
    try {
      // Copy the source image (its name is kept as-is).
      await Deno.copyFile(source, `${folder}/${basename}`);
      // Copy its pose JSON alongside, preserving the canonical <base>_pose.json
      // name. Missing pose JSON isn't fatal — the image still exports.
      const resolved = it?.result ? await resolveResultPath(it.result) : null;
      if (resolved) {
        const jsonName = resolved.split("/").pop() || `${basename}_pose.json`;
        await Deno.copyFile(resolved, `${folder}/${jsonName}`);
      }
      copied++;
    } catch (e) {
      failed++;
      if (errors.length < 50) {
        errors.push({ file: basename, error: e instanceof Error ? e.message : String(e) });
      }
    }
    if ((copied + failed) % 25 === 0) {
      log(`  …exported ${copied + failed}/${items.length}`);
    }
  }

  log(
    `✓ Export complete: ${copied} copied, ${failed} failed -> ./${folder}`,
    failed > 0 ? "error" : "info",
  );
  connManager.send(ws, { type: "export_result", folder, copied, failed, errors });
}

// ---- Inference models -------------------------------------------------------

const INFERENCE_MODELS_PATH = "./inference_models.json";
const INFERENCE_MODELS_DIR = "./inference_models";
const INFERENCE_SCRIPT = "python/f_o_train_inference.py";

// Ids of models training in THIS server process. Lets us tell a genuinely
// in-flight "training" status apart from one orphaned by a server restart.
const activeTrainings = new Set<string>();

async function loadInferenceModels(): Promise<InferenceModel[]> {
  let models: InferenceModel[];
  try {
    const raw = await Deno.readTextFile(INFERENCE_MODELS_PATH);
    const parsed = JSON.parse(raw);
    models = Array.isArray(parsed) ? parsed as InferenceModel[] : [];
  } catch {
    return []; // first run or unreadable — start empty
  }

  // Reconcile persisted status against reality — live tasks and files win:
  //   - "training" with no live training task in this process was orphaned by a
  //     restart/crash; reset to "error" so the user can retrain (otherwise the
  //     UI disables Train forever).
  //   - "trained" with no .joblib on disk means the artifact is gone (deleted,
  //     or a models.json copied without its model files); reset to "untrained"
  //     so the UI doesn't offer Apply for a model that can't run.
  let changed = false;
  for (const m of models) {
    // Backfill version for models saved before versioning existed.
    if (typeof m.version !== "number") {
      m.version = m.status === "trained" ? 1 : 0;
      changed = true;
    }
    if (m.status === "training" && !activeTrainings.has(m.id)) {
      m.status = "error";
      m.error = "Training was interrupted (server restarted). Train again.";
      m.updatedAt = new Date().toISOString();
      changed = true;
    } else if (m.status === "trained" && !(await isFile(modelFilePath(m.id)))) {
      m.status = "untrained";
      m.metrics = undefined;
      m.error = "Trained model file is missing — retrain.";
      m.updatedAt = new Date().toISOString();
      changed = true;
    }
  }
  if (changed) {
    try {
      await saveInferenceModels(models);
    } catch {
      // Non-fatal: still return the reconciled view even if we can't persist it.
    }
  }
  return models;
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
        version: 0, // bumped on first successful train
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
  // Register as a live training so reconciliation in loadInferenceModels() does
  // not mistake this model's persisted "training" status for an orphaned one
  // (it gets saved to disk below, then reloaded mid-run).
  activeTrainings.add(msg.id);
  try {
    await trainInferenceModelInner(ws, msg, config, connManager);
  } finally {
    activeTrainings.delete(msg.id);
  }
}

async function trainInferenceModelInner(
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
      m.version = (m.version || 0) + 1; // new artifact -> invalidate score cache
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
  // Operate on the same single manifest the preview page loads, reconciled
  // against disk (files win) so processors run on exactly what's shown — even
  // if the manifest was deleted or an entry went stale.
  const manifest = await reconcileManifestWithDisk(OUTPUT_DIR);

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
        case "export_images":
          await handleExportImages(socket, msg, connManager);
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
