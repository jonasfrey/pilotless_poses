// setup.ts — Deno-driven dependency bootstrap for Pilotless Poses.
//
// Checks EVERY runtime dependency and installs whatever is missing, using only
// Deno to orchestrate it:
//   Deno → Python 3 → a working venv (with pip) → the pip requirements.
//
// Missing OS packages (python3, python3-venv, pip, …) are installed through the
// system package manager. When that needs root and we aren't root, the command
// is run via `sudo`, which prompts for the password on the terminal.
//
// Used two ways:
//   • `deno task setup`              — run standalone to provision a machine
//   • imported by server.ts (start)  — verified/installed on every server start
//
// If a mandatory dependency cannot be satisfied, ensureDependencies() throws so
// the caller can abort instead of starting in a broken state.

const REQUIREMENTS_FILE = "./py-requirements.txt";
const CONFIG_PATH = "./pose_app_config.json";

// Python packages required by python/f_o_info_vitpose.py (name = pip name,
// import = module name used to verify it's importable).
const PIP_PACKAGES = [
  { name: "torch", import: "torch" },
  { name: "transformers", import: "transformers" },
  { name: "numpy", import: "numpy" },
  { name: "Pillow", import: "PIL" },
  { name: "supervision", import: "supervision" },
];

// ---- process helpers --------------------------------------------------------

/** Run a command capturing its output (used for quick probes). */
async function capture(
  cmd: string,
  args: string[],
): Promise<{ code: number; out: string; err: string }> {
  try {
    const { code, stdout, stderr } = await new Deno.Command(cmd, {
      args,
      stdout: "piped",
      stderr: "piped",
    }).output();
    return {
      code,
      out: new TextDecoder().decode(stdout).trim(),
      err: new TextDecoder().decode(stderr).trim(),
    };
  } catch {
    return { code: 127, out: "", err: "" };
  }
}

/** Run a command with inherited stdio (so output and sudo prompts show live). */
async function runInherit(cmd: string, args: string[]): Promise<number> {
  try {
    const child = new Deno.Command(cmd, {
      args,
      stdin: "inherit",
      stdout: "inherit",
      stderr: "inherit",
    }).spawn();
    const { code } = await child.status;
    return code;
  } catch (e) {
    console.error(`  ✗ Could not run "${cmd}": ${e instanceof Error ? e.message : e}`);
    return 127;
  }
}

async function commandExists(name: string): Promise<boolean> {
  const { code } = await capture("sh", ["-c", `command -v ${name}`]);
  return code === 0;
}

async function isFile(path: string): Promise<boolean> {
  try {
    return (await Deno.stat(path)).isFile;
  } catch {
    return false;
  }
}

async function currentUid(): Promise<number> {
  const { out } = await capture("id", ["-u"]);
  const n = parseInt(out, 10);
  return Number.isNaN(n) ? -1 : n;
}

// ---- privileged execution (sudo) -------------------------------------------

/**
 * Run a command with root privileges. If we're already root, run it directly;
 * otherwise wrap it in `sudo` (which prompts for the password on the terminal).
 * Returns the exit code, or 1 if elevation isn't possible.
 */
async function runPrivileged(cmd: string, args: string[]): Promise<number> {
  const uid = await currentUid();
  if (uid === 0) return await runInherit(cmd, args);

  if (!(await commandExists("sudo"))) {
    console.error(
      `  ✗ Root privileges are needed to run: ${cmd} ${args.join(" ")}\n` +
        `    but 'sudo' is not available. Re-run as root, or install the ` +
        `packages manually.`,
    );
    return 1;
  }

  // Heads-up if sudo will actually prompt (no cached/passwordless credentials).
  const { code: cached } = await capture("sudo", ["-n", "true"]);
  if (cached !== 0) {
    console.log("  → Your password is required to install system packages (sudo):");
  }
  return await runInherit("sudo", [cmd, ...args]);
}

// ---- package manager --------------------------------------------------------

interface PackageManager {
  name: string;
  needsRoot: boolean;
  refresh?: () => Promise<number>;
  install: (pkgs: string[]) => Promise<number>;
}

async function detectPackageManager(): Promise<PackageManager | null> {
  if (await commandExists("apt-get")) {
    return {
      name: "apt-get",
      needsRoot: true,
      refresh: () => runPrivileged("apt-get", ["update"]),
      install: (p) => runPrivileged("apt-get", ["install", "-y", ...p]),
    };
  }
  if (await commandExists("dnf")) {
    return {
      name: "dnf",
      needsRoot: true,
      install: (p) => runPrivileged("dnf", ["install", "-y", ...p]),
    };
  }
  if (await commandExists("pacman")) {
    return {
      name: "pacman",
      needsRoot: true,
      install: (p) => runPrivileged("pacman", ["-S", "--noconfirm", ...p]),
    };
  }
  if (await commandExists("zypper")) {
    return {
      name: "zypper",
      needsRoot: true,
      install: (p) => runPrivileged("zypper", ["install", "-y", ...p]),
    };
  }
  if (await commandExists("apk")) {
    return {
      name: "apk",
      needsRoot: true,
      install: (p) => runPrivileged("apk", ["add", ...p]),
    };
  }
  if (await commandExists("brew")) {
    // Homebrew must never run as root.
    return {
      name: "brew",
      needsRoot: false,
      install: (p) => runInherit("brew", ["install", ...p]),
    };
  }
  return null;
}

/** Per-package-manager names for Python, venv support, and pip. */
function pkgNames(pm: string): { python: string[]; venv: string[]; pip: string[] } {
  switch (pm) {
    case "apt-get":
      return { python: ["python3"], venv: ["python3-venv"], pip: ["python3-pip"] };
    case "dnf":
    case "zypper":
      // venv + ensurepip ship with the base python3 package here.
      return { python: ["python3"], venv: ["python3"], pip: ["python3-pip"] };
    case "pacman":
      return { python: ["python"], venv: ["python"], pip: ["python-pip"] };
    case "apk":
      return { python: ["python3"], venv: ["python3"], pip: ["py3-pip"] };
    case "brew":
      return { python: ["python"], venv: ["python"], pip: ["python"] };
    default:
      return { python: ["python3"], venv: ["python3-venv"], pip: ["python3-pip"] };
  }
}

async function installSystem(pm: PackageManager, pkgs: string[]): Promise<boolean> {
  if (pkgs.length === 0) return true;
  console.log(`  Installing system packages via ${pm.name}: ${pkgs.join(", ")} ...`);
  if (pm.refresh) await pm.refresh();
  const code = await pm.install(pkgs);
  if (code !== 0) {
    console.error(`  ✗ ${pm.name} failed to install: ${pkgs.join(", ")}`);
    return false;
  }
  return true;
}

// ---- python probes ----------------------------------------------------------

async function pythonWorks(python: string): Promise<boolean> {
  const { code } = await capture(python, ["--version"]);
  return code === 0;
}

async function pyImportOk(python: string, moduleName: string): Promise<boolean> {
  const { code } = await capture(python, ["-c", `import ${moduleName.replace(/-/g, "_")}`]);
  return code === 0;
}

async function hasPip(python: string): Promise<boolean> {
  return await pyImportOk(python, "pip");
}

// ---- ensure steps -----------------------------------------------------------

async function ensurePython3(basePython: string): Promise<boolean> {
  if (await pythonWorks(basePython)) return true;

  console.log(`  Python ("${basePython}") not found — installing...`);
  const pm = await detectPackageManager();
  if (!pm) {
    console.error("  ✗ No supported package manager found to install Python.");
    return false;
  }
  const names = pkgNames(pm.name);
  await installSystem(pm, [...names.python, ...names.pip]);
  return await pythonWorks(basePython);
}

async function createVenv(basePython: string, venvDir: string): Promise<boolean> {
  console.log(`  Creating virtual environment at ${venvDir} ...`);
  const code = await runInherit(basePython, ["-m", "venv", venvDir]);
  return code === 0 && (await isFile(`${venvDir}/bin/python`));
}

/**
 * Ensure pip exists for the given interpreter. On Debian/Ubuntu the `ensurepip`
 * module lives in python3-venv, so a venv can be created without pip — leaving
 * `python -m pip` to fail with "No module named pip". Recover via ensurepip.
 */
async function ensurePipFor(python: string): Promise<boolean> {
  if (await hasPip(python)) return true;
  console.log("  pip missing — bootstrapping with ensurepip ...");
  await runInherit(python, ["-m", "ensurepip", "--upgrade"]);
  return await hasPip(python);
}

/**
 * Ensure a working venv at `venvDir` (interpreter + pip), creating it and
 * installing the OS venv package if necessary. Returns the venv interpreter
 * path, or null if it could not be made to work.
 */
async function ensureVenv(basePython: string, venvDir: string): Promise<string | null> {
  const venvPython = `${venvDir}/bin/python`;

  // Reuse an existing venv only if it actually has a working pip.
  if (await isFile(venvPython)) {
    if (await ensurePipFor(venvPython)) return venvPython;
    console.warn(`  ⚠ Existing venv at ${venvDir} lacks pip — recreating.`);
    await Deno.remove(venvDir, { recursive: true }).catch(() => {});
  }

  // First attempt to create the venv.
  if (!(await createVenv(basePython, venvDir))) {
    // Likely missing OS venv support (e.g. python3-venv). Install and retry once.
    const pm = await detectPackageManager();
    if (pm) {
      const names = pkgNames(pm.name);
      console.log("  venv creation failed — installing venv/pip support...");
      await installSystem(pm, [...names.venv, ...names.pip]);
      await Deno.remove(venvDir, { recursive: true }).catch(() => {});
      if (!(await createVenv(basePython, venvDir))) return null;
    } else {
      return null;
    }
  }

  if (!(await ensurePipFor(venvPython))) return null;

  // Keep pip itself current inside the fresh venv (best-effort).
  await runInherit(venvPython, ["-m", "pip", "install", "--upgrade", "pip"]);
  return venvPython;
}

async function ensurePyPackages(venvPython: string): Promise<boolean> {
  const missing: string[] = [];
  for (const pkg of PIP_PACKAGES) {
    if (!(await pyImportOk(venvPython, pkg.import))) missing.push(pkg.name);
  }
  if (missing.length === 0) return true;

  console.log(`  Installing Python packages (${missing.join(", ")}) ...`);
  // Install the full pinned set so versions stay consistent.
  const code = await runInherit(venvPython, [
    "-m",
    "pip",
    "install",
    "-r",
    REQUIREMENTS_FILE,
  ]);
  if (code !== 0) return false;

  for (const pkg of PIP_PACKAGES) {
    if (!(await pyImportOk(venvPython, pkg.import))) {
      console.error(`  ✗ Package still not importable after install: ${pkg.name}`);
      return false;
    }
  }
  return true;
}

// ---- public API -------------------------------------------------------------

async function configuredPaths(): Promise<{ basePython: string; venvDir: string }> {
  try {
    const c = JSON.parse(await Deno.readTextFile(CONFIG_PATH));
    // pythonPath in config is the BASE interpreter used to build the venv. If a
    // previous run left a venv path there, fall back to a plain "python3".
    let basePython = typeof c.pythonPath === "string" ? c.pythonPath : "python3";
    if (basePython.includes("/")) basePython = "python3";
    const venvDir = typeof c.venvDir === "string" ? c.venvDir : "./.venv";
    return { basePython, venvDir };
  } catch {
    return { basePython: "python3", venvDir: "./.venv" };
  }
}

/**
 * Check and, if necessary, install every dependency. Returns the path to the
 * venv Python interpreter to use for inference. Throws if a mandatory
 * dependency cannot be satisfied.
 */
export async function ensureDependencies(
  basePython?: string,
  venvDir?: string,
): Promise<string> {
  if (!basePython || !venvDir) {
    const cfg = await configuredPaths();
    basePython ??= cfg.basePython;
    venvDir ??= cfg.venvDir;
  }

  console.log("▶ Checking dependencies (Deno-managed)...");

  // 1. Deno — we're running inside it.
  console.log(`  ✓ Deno ${Deno.version.deno}`);

  // 2. Python 3
  if (!(await ensurePython3(basePython))) {
    throw new Error(
      `Python 3 is required but could not be found or installed (tried "${basePython}").`,
    );
  }
  console.log("  ✓ Python 3");

  // 3. venv (+ OS venv/pip package if needed)
  const venvPython = await ensureVenv(basePython, venvDir);
  if (!venvPython) {
    throw new Error(
      "Could not create a working Python virtual environment (venv/pip unavailable).",
    );
  }
  console.log(`  ✓ venv: ${venvPython}`);

  // 4. Python packages
  if (!(await ensurePyPackages(venvPython))) {
    throw new Error("Failed to install the required Python packages.");
  }
  console.log("  ✓ Python packages");

  console.log("✔ All dependencies satisfied.\n");
  return venvPython;
}

// ---- standalone CLI ---------------------------------------------------------

if (import.meta.main) {
  try {
    await ensureDependencies();
    Deno.exit(0);
  } catch (e) {
    console.error(`\n✗ Setup failed: ${e instanceof Error ? e.message : e}`);
    Deno.exit(1);
  }
}
