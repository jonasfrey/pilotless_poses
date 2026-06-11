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
  | WriteFileMessage;

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

type ServerMessage =
  | ProgressMessage
  | BatchCompleteMessage
  | ResultsListMessage
  | PoseDataMessage
  | ErrorMessage
  | BrowseResultMessage
  | ProcessorsListMessage
  | AllPoseDataMessage
  | WriteFileResultMessage
  | ScanLogMessage;

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

interface VitPoseBatchOutput {
  total_images: number;
  successful: number;
  failed: number;
  results: VitPoseResult[];
}

async function runPythonBatchInference(
  pythonPath: string,
  imagePaths: string[],
  modelDir: string,
  signal?: AbortSignal,
  onLog?: (line: string) => void,
): Promise<VitPoseBatchOutput> {
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

  // stdout carries the final JSON result — collect it in full.
  const stdoutPromise = new Response(child.stdout).text();

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

  const stdoutText = await stdoutPromise;
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

  let output: VitPoseBatchOutput;
  try {
    output = JSON.parse(stdoutText);
  } catch {
    throw new Error(`Failed to parse Python output as JSON. stdout: ${stdoutText.slice(0, 500)}`);
  }

  return output;
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
}

class ConnectionManager {
  private sockets: Map<WebSocket, ScanState | null> = new Map();

  add(ws: WebSocket): void {
    this.sockets.set(ws, null);
  }

  remove(ws: WebSocket): void {
    const state = this.sockets.get(ws);
    if (state) {
      state.abortController.abort();
    }
    this.sockets.delete(ws);
  }

  setScanState(ws: WebSocket, state: ScanState): void {
    this.sockets.set(ws, state);
  }

  clearScanState(ws: WebSocket): void {
    this.sockets.set(ws, null);
  }

  getScanState(ws: WebSocket): ScanState | null {
    return this.sockets.get(ws) ?? null;
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
}

// ---- Scanning Orchestration -------------------------------------------------

async function handleScanFolder(
  ws: WebSocket,
  msg: ScanFolderMessage,
  config: AppConfig,
  connManager: ConnectionManager,
): Promise<void> {
  const { folderPath, extensions } = msg;

  // Stream a line to the client's Live Log page.
  const log = (line: string, level: "info" | "python" | "error" = "info") => {
    connManager.send(ws, {
      type: "scan_log",
      level,
      line,
      ts: new Date().toISOString(),
    });
  };

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

  // Cancel any existing scan for this connection
  const existingState = connManager.getScanState(ws);
  if (existingState) {
    existingState.abortController.abort();
  }

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

    // Create scan state
    const scanState: ScanState = {
      abortController,
      total: images.length,
      processed: 0,
      errors: [],
    };
    connManager.setScanState(ws, scanState);

    // Send queued status for all files
    for (const img of images) {
      const basename = img.split("/").pop()!;
      connManager.send(ws, {
        type: "progress",
        current: 0,
        total: images.length,
        file: basename,
        status: "queued",
      });
    }

    // Send processing status for all files
    for (const img of images) {
      connManager.send(ws, {
        type: "progress",
        current: 0,
        total: images.length,
        file: img.split("/").pop()!,
        status: "processing",
      });
    }

    try {
      // Call the VitPose batch script — processes all images in one invocation
      // (models are loaded once, then all images processed sequentially)
      const modelDir = config.modelDir || "./models";
      log("Launching VitPose inference (this can take a while on first run)…");
      const batchOutput = await runPythonBatchInference(
        config.pythonPath,
        images,
        modelDir,
        abortController.signal,
        (line) => log(line, "python"),
      );

      if (abortController.signal.aborted) return;

      log(
        `Inference finished: ${batchOutput.successful} ok, ${batchOutput.failed} failed`,
      );

      // Distribute results to per-image files and send progress
      for (const result of batchOutput.results) {
        const imagePath = result.image_path;
        const basename = imagePath.split("/").pop()!;

        if (result.success) {
          // Transform to the expected output schema
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
          const resultFullPath = `${outputDir}/${resultFilename}`;
          manifest.results.push({
            source: imagePath,
            result: resultFullPath,
            status: "ok",
          });
          connManager.send(ws, {
            type: "progress",
            current: scanState.processed + 1,
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
          connManager.send(ws, {
            type: "progress",
            current: scanState.processed + 1,
            total: images.length,
            file: basename,
            status: "error",
            error: errorMsg,
          });
        }

        scanState.processed++;
      }
    } catch (e) {
      if (abortController.signal.aborted) return;
      const errorMsg = e instanceof Error ? e.message : String(e);
      log(`Inference failed: ${errorMsg}`, "error");

      // Mark all remaining images as errored
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
        connManager.send(ws, {
          type: "progress",
          current: scanState.processed + 1,
          total: images.length,
          file: basename,
          status: "error",
          error: errorMsg,
        });
        scanState.processed++;
      }
    }

    // Save manifest
    await saveManifest(outputDir, manifest);
    log(`Saved results + manifest to ${outputDir}`);

    // Send batch complete
    if (!abortController.signal.aborted) {
      log(
        `✓ Scan complete: ${images.length - scanState.errors.length} of ` +
          `${images.length} succeeded, ${scanState.errors.length} error(s)`,
      );
      connManager.send(ws, {
        type: "batch_complete",
        total: images.length,
        errors: scanState.errors,
      });
    }

    connManager.clearScanState(ws);
  } catch (e) {
    if (!abortController.signal.aborted) {
      const errorMsg = e instanceof Error ? e.message : String(e);
      log(`Scan failed: ${errorMsg}`, "error");
      connManager.send(ws, {
        type: "error",
        message: `Scan failed: ${errorMsg}`,
      });
    }
    connManager.clearScanState(ws);
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
          const state = connManager.getScanState(socket);
          if (state) {
            state.abortController.abort();
            connManager.clearScanState(socket);
            connManager.send(socket, {
              type: "scan_log",
              level: "error",
              line: "■ Scan cancelled by user.",
              ts: new Date().toISOString(),
            });
            connManager.send(socket, {
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
}

// Run
if (import.meta.main) {
  main();
}
