# Copyright (C) 2026 Jonas Immanuel Frey - Licensed under GPL-2.0-only. See LICENSE file for details.

import sys
import os
import time
import json
import argparse

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
    from ultralytics import YOLO
except ImportError:
    print("Missing required package: ultralytics")
    print("\nUse a virtual environment:\n")
    print("  python3 -m venv venv")
    print("  source venv/bin/activate")
    print("  pip install ultralytics")
    sys.exit(1)

# --- 2. argument parsing ---

s_path__script_dir = os.path.dirname(os.path.abspath(__file__))
s_path__root_dir = os.path.dirname(s_path__script_dir)

s_uuid__default = os.environ.get('S_UUID', '')

o_parser = argparse.ArgumentParser(
    description="Detect objects in images using a pretrained YOLO detection model. Outputs COCO class detections with bounding boxes via IPC."
)
o_parser.add_argument("a_s_path", nargs="+", help="Paths to image files to detect objects in")
o_parser.add_argument("--s-uuid", default=s_uuid__default, help="S_UUID for IPC output (default from env)")

o_arg = o_parser.parse_args()
a_s_path = o_arg.a_s_path
s_uuid = o_arg.s_uuid

print("  +-Arguments -------------------------------------------+")
print(f"  | images              {str(len(a_s_path)) + ' file(s)':30s}             |")
print(f"  | --s-uuid            {s_uuid[:25]:30s}             |")
print("  +------------------------------------------------------+")

# --- 3. load model ---

o_timer__model = f_time_start("load_yolo_model")
f_log("Loading YOLO detection model...")
o_model = YOLO(".gitignored/yolov8m.pt")
f_time_end(o_timer__model)

# --- 4. detect objects in each image ---

a_o_result = []

for s_path in a_s_path:
    if not os.path.isfile(s_path):
        f_log(f"skipping (not a file): {s_path}")
        continue

    o_timer__detect = f_time_start(f"detect {os.path.basename(s_path)}")
    a_o_detection_result = o_model(s_path, verbose=False)
    f_time_end(o_timer__detect)

    a_o_detection = []
    n_inference_ms = 0
    n_width = 0
    n_height = 0
    for o_r in a_o_detection_result:
        # sum up preprocessing + inference + postprocessing
        n_inference_ms = round(sum(o_r.speed.values()))
        # orig_shape is (height, width)
        n_height, n_width = o_r.orig_shape

        for box in o_r.boxes:
            n_cls_idx = int(box.cls[0])
            n_conf = float(box.conf[0])
            n_x1, n_y1, n_x2, n_y2 = box.xyxy[0].tolist()
            s_name = o_r.names[n_cls_idx]
            a_o_detection.append({
                'n_index': n_cls_idx,
                's_name': s_name,
                'n_confidence': round(n_conf, 6),
                'n_x1': round(n_x1, 1),
                'n_y1': round(n_y1, 1),
                'n_x2': round(n_x2, 1),
                'n_y2': round(n_y2, 1),
            })
            f_log(f"  {s_name:20s} {n_conf:.2%}  [{n_x1:.0f}, {n_y1:.0f}, {n_x2:.0f}, {n_y2:.0f}]")

    f_log(f"  {len(a_o_detection)} object(s) detected, inference {n_inference_ms}ms")

    a_o_result.append({
        's_path_absolute': os.path.abspath(s_path),
        'n_width': n_width,
        'n_height': n_height,
        'n_inference_ms': n_inference_ms,
        'a_o_detection': a_o_detection,
    })

# --- 5. IPC output ---

if s_uuid:
    s_json = json.dumps(a_o_result)
    print(f"{s_uuid}_start_json")
    print(s_json)
    print(f"{s_uuid}_end_json")

# --- 6. performance summary ---

n_sec__total = f_n_sec__elapsed()
print("  +-Performance -----------------------------------------+")
for o_t in a_o_timing:
    print(f"  | {o_t['s_name'][:25]:25s} {o_t['n_sec']:8.3f}s               |")
print(f"  | {'---':25s} {'--------':8s}                |")
print(f"  | {'Total':25s} {n_sec__total:8.3f}s               |")
print("  +------------------------------------------------------+")

sys.exit(0)
