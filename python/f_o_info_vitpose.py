import torch
import numpy as np
import json
import sys
import os
import argparse
from pathlib import Path
from PIL import Image
import supervision as sv
from transformers import AutoProcessor, RTDetrForObjectDetection, VitPoseForPoseEstimation

COCO_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

# Selectable VitPose pose-estimation models -> their HuggingFace repo ids. The
# short key is what we record in each pose JSON ("pose_model") and accept via
# --pose-model. "vitpose-base" is the default (more accurate); "vitpose-base-
# simple" is the older, lighter variant.
POSE_MODELS = {
    "vitpose-base": "usyd-community/vitpose-base",
    "vitpose-base-simple": "usyd-community/vitpose-base-simple",
}
DEFAULT_POSE_MODEL = "vitpose-base"


def pose_estimator_dir(s_dir_model, model_name):
    """Per-model cache dir, so switching models never reuses cached weights."""
    return os.path.join(s_dir_model, "pose_estimator_" + model_name)

def select_device():
    """Pick the inference device, preferring CUDA but degrading gracefully.

    torch.cuda.is_available() can return True on machines where CUDA then fails
    at runtime with errors like "CUDA driver error: invalid argument" — e.g. a
    driver/runtime version mismatch or broken GPU passthrough in a container.
    So we don't just trust is_available(): we run a tiny op on the GPU and only
    keep CUDA if it actually works, otherwise fall back to CPU.

    Override with the POSE_DEVICE env var ("cpu" or "cuda") to skip detection.
    """
    forced = os.environ.get("POSE_DEVICE", "").strip().lower()
    if forced in ("cpu", "cuda"):
        if forced == "cuda" and not torch.cuda.is_available():
            print("POSE_DEVICE=cuda but CUDA is unavailable; using CPU.", file=sys.stderr)
            return torch.device("cpu")
        return torch.device(forced)

    if not torch.cuda.is_available():
        return torch.device("cpu")

    # CUDA claims to be available — verify real operations succeed before
    # committing the whole pipeline to the GPU. Some torch builds (e.g.
    # 2.12.0+cu130) ship broken reduction/scan kernels: elementwise ops work
    # but prod()/cumsum() fail with "CUDA driver error: invalid argument",
    # which then crashes RT-DETR mid-inference. Exercise those ops here so we
    # detect the broken build up front and fall back to CPU.
    try:
        probe = torch.tensor([[2, 3], [4, 5]], device="cuda")
        _ = probe + 1                 # elementwise
        _ = probe.prod(1).cumsum(0)   # reduction + scan (the ops RT-DETR needs)
        torch.cuda.synchronize()
        return torch.device("cuda")
    except Exception as exc:
        print(
            f"CUDA reported available but failed a test op ({exc}); "
            "falling back to CPU. This usually means a broken torch CUDA build "
            "rather than a driver problem. Set POSE_DEVICE=cpu to silence this.",
            file=sys.stderr,
        )
        return torch.device("cpu")


def process_image(image_path, person_processor, person_model, pose_processor, pose_model, device):
    """Process a single image and return pose data"""
    
    # Get absolute path
    abs_path = str(Path(image_path).resolve())
    
    try:
        # Load image
        image = Image.open(image_path)
        
        # Detect people
        inputs = person_processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = person_model(**inputs)

        results = person_processor.post_process_object_detection(
            outputs, target_sizes=torch.tensor([(image.height, image.width)]).to(device), threshold=0.3
        )
        result = results[0]

        person_boxes = result["boxes"][result["labels"] == 0]
        person_boxes = person_boxes.cpu().numpy()

        if len(person_boxes) == 0:
            return {
                "image_path": abs_path,
                "success": True,
                "people_count": 0,
                "people": []
            }

        # Convert boxes
        person_boxes[:, 2] = person_boxes[:, 2] - person_boxes[:, 0]
        person_boxes[:, 3] = person_boxes[:, 3] - person_boxes[:, 1]

        # Detect poses
        inputs = pose_processor(image, boxes=[person_boxes], return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = pose_model(**inputs)

        pose_results = pose_processor.post_process_pose_estimation(outputs, boxes=[person_boxes])
        image_pose_result = pose_results[0]

        # Extract keypoint data
        xy = torch.stack([pose_result['keypoints'] for pose_result in image_pose_result]).cpu().numpy()
        scores = torch.stack([pose_result['scores'] for pose_result in image_pose_result]).cpu().numpy()

        # Format data
        people = []
        for person_idx in range(len(xy)):
            keypoints = []
            for kp_idx in range(len(xy[person_idx])):
                keypoints.append({
                    "name": COCO_KEYPOINT_NAMES[kp_idx],
                    "x": float(xy[person_idx][kp_idx][0]),
                    "y": float(xy[person_idx][kp_idx][1]),
                    "confidence": float(scores[person_idx][kp_idx])
                })
            
            people.append({
                "person_id": person_idx,
                "keypoints": keypoints
            })
        
        return {
            "image_path": abs_path,
            "success": True,
            "people_count": len(people),
            "people": people
        }
        
    except Exception as e:
        return {
            "image_path": abs_path,
            "success": False,
            "error": str(e),
            "people_count": 0,
            "people": []
        }

def download_models(s_path_person_detector, s_path_pose_estimator, model_name,
                    need_person=True, need_pose=True):
    """Download the person detector and/or the chosen pose estimator locally."""
    repo_id = POSE_MODELS[model_name]

    if need_person:
        os.makedirs(s_path_person_detector, exist_ok=True)
        print(f"Downloading person detector to {s_path_person_detector}...", file=sys.stderr)
        processor = AutoProcessor.from_pretrained("PekingU/rtdetr_r50vd_coco_o365")
        processor.save_pretrained(s_path_person_detector)
        model = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r50vd_coco_o365")
        model.save_pretrained(s_path_person_detector)

    if need_pose:
        os.makedirs(s_path_pose_estimator, exist_ok=True)
        print(f"Downloading pose estimator '{model_name}' ({repo_id}) to "
              f"{s_path_pose_estimator}...", file=sys.stderr)
        processor = AutoProcessor.from_pretrained(repo_id)
        processor.save_pretrained(s_path_pose_estimator)
        model = VitPoseForPoseEstimation.from_pretrained(repo_id)
        model.save_pretrained(s_path_pose_estimator)

    print("Models downloaded!", file=sys.stderr)

def resolve_model_paths(s_dir_model, model_name):
    """Return (person_detector_dir, pose_estimator_dir) for `model_name`.

    The pose estimator lives in a per-model dir. For backward compatibility the
    old flat "pose_estimator" dir (which only ever held vitpose-base-simple) is
    honored when that's the requested model and the new dir doesn't exist yet.
    """
    s_path_person_detector = os.path.join(s_dir_model, "person_detector")
    s_path_pose_estimator = pose_estimator_dir(s_dir_model, model_name)
    legacy_pose = os.path.join(s_dir_model, "pose_estimator")
    if (model_name == "vitpose-base-simple"
            and not os.path.isdir(s_path_pose_estimator)
            and os.path.isdir(legacy_pose)):
        s_path_pose_estimator = legacy_pose
    return s_path_person_detector, s_path_pose_estimator


def main():
    parser = argparse.ArgumentParser(description="Batch pose estimation with VitPose")
    parser.add_argument("image_path", nargs="*", help="Image file paths to process")
    parser.add_argument("--model-dir", default="./models", help="Directory containing model subdirectories (person_detector, pose_estimator_<model>)")
    parser.add_argument("--pose-model", default=DEFAULT_POSE_MODEL, choices=sorted(POSE_MODELS),
                        help=f"VitPose model to use (default: {DEFAULT_POSE_MODEL})")
    parser.add_argument("--download-models", action="store_true", help="Download models to --model-dir and exit")
    o_arg = parser.parse_args()

    model_name = o_arg.pose_model

    if o_arg.download_models:
        s_dir_model = str(Path(o_arg.model_dir).resolve())
        person_dir, pose_dir = resolve_model_paths(s_dir_model, model_name)
        download_models(person_dir, pose_dir, model_name)
        return

    if len(o_arg.image_path) == 0:
        print(json.dumps({
            "error": "No image paths provided",
            "usage": "python f_o_info_vitpose.py [--model-dir DIR] [--pose-model NAME] <image1> <image2> ..."
        }), file=sys.stderr)
        sys.exit(1)

    image_path = o_arg.image_path
    s_dir_model = str(Path(o_arg.model_dir).resolve())
    s_path_person_detector, s_path_pose_estimator = resolve_model_paths(s_dir_model, model_name)

    need_person = not os.path.isdir(s_path_person_detector)
    need_pose = not os.path.isdir(s_path_pose_estimator)
    if need_person or need_pose:
        print(f"Models not found in {s_dir_model}, downloading automatically...", file=sys.stderr)
        download_models(s_path_person_detector, s_path_pose_estimator, model_name,
                        need_person=need_person, need_pose=need_pose)

    # Print to stderr so it doesn't interfere with JSON output
    print(f"Processing {len(image_path)} images...", file=sys.stderr)

    device = select_device()
    print(f"Using device: {device}", file=sys.stderr)

    # Load models ONCE
    # do not use local_files_only=True with absolute paths, it causes huggingface_hub repo_id validation errors
    print(f"Loading models from {s_dir_model} (pose model: {model_name})...", file=sys.stderr)

    person_processor = AutoProcessor.from_pretrained(s_path_person_detector)
    person_model = RTDetrForObjectDetection.from_pretrained(s_path_person_detector).to(device)

    pose_processor = AutoProcessor.from_pretrained(s_path_pose_estimator)
    pose_model = VitPoseForPoseEstimation.from_pretrained(s_path_pose_estimator).to(device)

    print("Models loaded!", file=sys.stderr)

    # Process all images, streaming each result to stdout as a single JSON line
    # (JSONL) the moment it's ready. This lets the server persist results
    # incrementally, so a crash or page close mid-scan keeps everything done so
    # far instead of losing the whole batch. A final "summary" line closes it.
    successful = 0
    failed = 0
    for idx, s_path_image in enumerate(image_path, 1):
        print(f"Processing {idx}/{len(image_path)}: {s_path_image}", file=sys.stderr)
        result = process_image(s_path_image, person_processor, person_model, pose_processor, pose_model, device)
        # Record which pose model produced this result (stored in the pose JSON).
        result["pose_model"] = model_name
        if result["success"]:
            successful += 1
        else:
            failed += 1
        print(json.dumps({"type": "result", "data": result}), flush=True)

    print(json.dumps({
        "type": "summary",
        "total_images": len(image_path),
        "successful": successful,
        "failed": failed,
    }), flush=True)

if __name__ == "__main__":
    main()