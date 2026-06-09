# Copyright (C) 2026 Jonas Immanuel Frey - Licensed under GPL-2.0-only. See LICENSE file for details.

import sys
import os
import time

n_sec__start = time.monotonic()
a_o_timing = []

def f_n_sec__elapsed():
    return time.monotonic() - n_sec__start

def f_s_ts():
    n_elapsed = f_n_sec__elapsed()
    n_min = int(n_elapsed // 60)
    n_sec = n_elapsed % 60
    return f"[{n_min:02d}:{n_sec:06.3f}]"

def f_log(s_msg):
    print(f"{f_s_ts()} {s_msg}")

def f_time_start(s_name):
    return { 's_name': s_name, 'n_sec__start': time.monotonic() }

def f_time_end(o_timer):
    n_sec__elapsed = time.monotonic() - o_timer['n_sec__start']
    a_o_timing.append({ 's_name': o_timer['s_name'], 'n_sec': n_sec__elapsed })
    f_log(f"{o_timer['s_name']} ({n_sec__elapsed:.3f}s)")
    return n_sec__elapsed

# --- 1. dependency guard ---

try:
    import json
except ImportError:
    print("Missing required package: json")
    sys.exit(1)

try:
    import argparse
except ImportError:
    print("Missing required package: argparse")
    sys.exit(1)

try:
    from dotenv import load_dotenv
except ImportError:
    print("Missing required package: python-dotenv")
    print("\nUse a virtual environment:\n")
    print("  python3 -m venv venv")
    print("  source venv/bin/activate")
    print("  pip install python-dotenv")
    sys.exit(1)

try:
    import pyttsx3
except ImportError:
    print("Missing required package: pyttsx3")
    print("\nUse a virtual environment:\n")
    print("  python3 -m venv venv")
    print("  source venv/bin/activate")
    print("  pip install pyttsx3")
    sys.exit(1)

s_path__model_constructor = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '.gitignored', 'model_constructors'
)
sys.path.insert(0, s_path__model_constructor)
try:
    from model_constructors import f_o_utterance, f_o_fsnode
except ImportError:
    print("Missing model_constructors.py in .gitignored/model_constructors/")
    print("Run the model constructor generator first.")
    sys.exit(1)

# --- 2. argument parsing & summary ---

s_path__script_dir = os.path.dirname(os.path.abspath(__file__))
s_path__root_dir = os.path.dirname(s_path__script_dir)
s_path__env = os.path.join(s_path__root_dir, '.env')
if os.path.exists(s_path__env):
    load_dotenv(s_path__env)

s_uuid__default = os.environ.get('S_UUID', '')
s_path_dir__output_default = os.path.join(s_path__root_dir, '.gitignored', 'audio')

o_parser = argparse.ArgumentParser(
    description="Generate a TTS audio file from text using pyttsx3 (espeak). Outputs file metadata via IPC."
)
o_parser.add_argument("s_text", nargs="*", help="Text to synthesize into speech")
o_parser.add_argument("--s-uuid", default=s_uuid__default, help="S_UUID for IPC output (default from .env)")
o_parser.add_argument("--s-path-dir-output", default=s_path_dir__output_default, help=f"Output directory for audio files (default: {s_path_dir__output_default})")
o_parser.add_argument("--n-rate", type=int, default=150, help="Speech rate in words per minute (default: 150)")

o_arg = o_parser.parse_args()
s_text = " ".join(o_arg.s_text).strip()
s_uuid = o_arg.s_uuid
s_path_dir__output = o_arg.s_path_dir_output
n_rate = o_arg.n_rate

# detect which args were explicitly provided via sys.argv
a_s_arg__provided = set()
for s_a in sys.argv[1:]:
    if s_a.startswith('--'):
        a_s_arg__provided.add(s_a.split('=')[0])

def f_s_source(s_flag):
    return "(provided)" if s_flag in a_s_arg__provided else "(default)"

print("  +-Arguments -------------------------------------------+")
print(f"  | s_text              {repr(s_text[:25]):30s} {f_s_source('s_text'):12s}|")
print(f"  | --s-uuid            {s_uuid[:25]:30s} {f_s_source('--s-uuid'):12s}|")
print(f"  | --s-path-dir-output {s_path_dir__output[:25]:30s} {f_s_source('--s-path-dir-output'):12s}|")
print(f"  | --n-rate            {str(n_rate):30s} {f_s_source('--n-rate'):12s}|")
print("  +------------------------------------------------------+")

if not s_text:
    print("Reading from stdin (Ctrl+D to finish)...")
    s_text = sys.stdin.read().strip()

if not s_text:
    print("Error: No text provided.", file=sys.stderr)
    sys.exit(1)

# --- 3. processing with logging & timing ---

o_timer__mkdir = f_time_start("create_output_dir")
f_log(f"Ensuring output directory: {s_path_dir__output}")
os.makedirs(s_path_dir__output, exist_ok=True)
f_time_end(o_timer__mkdir)

o_timer__init_engine = f_time_start("init_engine")
f_log("Initializing pyttsx3 engine...")
o_engine = pyttsx3.init()
o_engine.setProperty('rate', n_rate)
f_time_end(o_timer__init_engine)

n_ts_ms = int(time.time() * 1000)
s_name__file = f"utterance_{n_ts_ms}.wav"
s_path_absolute__file = os.path.join(s_path_dir__output, s_name__file)

n_its__retry = 3
b_synthesized = False
o_timer__synthesize = f_time_start("synthesize_audio")
f_log(f"Synthesizing: \"{s_text[:80]}{'...' if len(s_text) > 80 else ''}\"")
for n_it__retry in range(n_its__retry):
    o_engine.save_to_file(s_text, s_path_absolute__file)
    o_engine.runAndWait()
    if os.path.exists(s_path_absolute__file) and os.path.getsize(s_path_absolute__file) > 0:
        b_synthesized = True
        break
    f_log(f"Attempt {n_it__retry + 1}/{n_its__retry}: file not created, reinitializing engine...")
    o_engine.stop()
    o_engine = pyttsx3.init()
    o_engine.setProperty('rate', n_rate)
f_time_end(o_timer__synthesize)

if not b_synthesized:
    print(f"Error: pyttsx3 failed to create audio file after {n_its__retry} attempts.", file=sys.stderr)
    sys.exit(2)

n_bytes = os.path.getsize(s_path_absolute__file)
f_log(f"Audio saved: {s_path_absolute__file} ({n_bytes} bytes)")

# --- 4. machine-readable output (IPC protocol) ---

o_fsnode = f_o_fsnode({
    'n_bytes': n_bytes,
    's_name': s_name__file,
    's_path_absolute': s_path_absolute__file,
    'b_folder': False,
    'b_image': False,
    'b_video': False,
    'n_ts_ms_created': n_ts_ms,
})

o_utterance = f_o_utterance({
    's_text': s_text,
    'n_ts_ms_created': n_ts_ms,
})

o_uttdatainfo = {
    'o_utterance': o_utterance,
    'o_fsnode': o_fsnode,
}

if s_uuid:
    s_json = json.dumps(o_uttdatainfo)
    print(f"{s_uuid}_start_json")
    print(s_json)
    print(f"{s_uuid}_end_json")

# --- 5. performance summary ---

n_sec__total = f_n_sec__elapsed()
print("  +-Performance -----------------------------------------+")
for o_t in a_o_timing:
    print(f"  | {o_t['s_name']:25s} {o_t['n_sec']:8.3f}s               |")
print(f"  | {'---':25s} {'--------':8s}                |")
print(f"  | {'Total':25s} {n_sec__total:8.3f}s               |")
print("  +------------------------------------------------------+")

sys.exit(0)
