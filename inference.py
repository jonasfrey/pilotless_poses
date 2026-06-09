#!/usr/bin/env python3
"""
VitPose inference script for single-image human pose detection.

Usage:
    python inference.py <image_path> [--device cpu|cuda] [--det-model <model>] [--pose-model <model>]

Outputs a JSON object to stdout with COCO 17-keypoint pose data.
On error, outputs {"error": "message"} to stdout and exits with code 1.

Model weights are downloaded automatically on first run via mim/mmpose.
"""

import argparse
import json
import sys
import traceback
from pathlib import Path

# ---------------------------------------------------------------------------
# COCO 17 keypoint names (in standard order)
# ---------------------------------------------------------------------------
COCO_KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

# Default model configurations
DEFAULT_DET_MODEL = "rtmdet_m_8xb32-300e_coco"
DEFAULT_POSE_MODEL = "vitpose-b-coco"


# Cache directory for downloaded model files
MODEL_CACHE = Path(__file__).parent / ".model_cache"
MODEL_CACHE.mkdir(exist_ok=True)


def download_model(package: str, config_name: str):
    """Download model config + checkpoint via mim. Returns (config_path, checkpoint_path)."""
    import torch
    from mim.commands.download import download

    dest = str(MODEL_CACHE / package)
    config_path = str(Path(dest) / f"{config_name}.py")

    # Check if we already have a valid checkpoint
    pth_files = sorted(Path(dest).glob("*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
    checkpoint_path = str(pth_files[0]) if pth_files else None

    valid = False
    if Path(config_path).exists() and checkpoint_path:
        # Validate the checkpoint isn't corrupted
        try:
            torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            valid = True
        except Exception:
            # Corrupted — delete and re-download
            Path(checkpoint_path).unlink(missing_ok=True)
            checkpoint_path = None

    if not valid:
        download(package=package, configs=[config_name], dest_root=dest)

        # Find config
        if not Path(config_path).exists():
            raise RuntimeError(
                f"Failed to download config '{config_name}' from {package}. "
                f"No .py config found in {dest}."
            )

        # Find checkpoint — the one that was just downloaded
        pth_files = sorted(Path(dest).glob("*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not pth_files:
            raise RuntimeError(
                f"Failed to find checkpoint for '{config_name}' from {package}. "
                f"No .pth file found in {dest}."
            )
        checkpoint_path = str(pth_files[0])

    return config_path, checkpoint_path


def load_models(det_model_name: str, pose_model_name: str, device: str):
    """Load person detector and pose estimator models.

    Uses mmdet for person detection and mmpose (ViTPose) for keypoints.
    Model configs and weights are downloaded automatically on first use.
    """
    from mmdet.apis import init_detector
    from mmpose.apis import init_model as init_pose_model

    # --- Person detector ---
    det_config, det_checkpoint = download_model("mmdet", det_model_name)
    detector = init_detector(det_config, det_checkpoint, device=device)

    # --- Pose estimator (ViTPose) ---
    pose_config, pose_checkpoint = download_model("mmpose", pose_model_name)
    pose_model = init_pose_model(pose_config, pose_checkpoint, device=device)

    return detector, pose_model


def run_inference(image_path: str, detector, pose_model) -> dict:
    """Run person detection + pose estimation on a single image.

    Returns a dictionary conforming to the output schema.
    """
    import mmcv
    import numpy as np
    from mmdet.apis import inference_detector
    from mmpose.apis import inference_topdown
    from mmpose.structures import merge_data_samples

    # Load image
    img = mmcv.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    h, w = img.shape[:2]

    # --- Person detection ---
    det_result = inference_detector(detector, img)

    # mmdet 3.x returns a DetDataSample; extract bboxes
    if hasattr(det_result, "pred_instances"):
        # DetDataSample (mmdet 3.x)
        instances = det_result.pred_instances
        bboxes = instances.bboxes.cpu().numpy()
        scores = instances.scores.cpu().numpy()
        labels = instances.labels.cpu().numpy() if hasattr(instances, "labels") else None

        # Filter to person class only (class 0 in COCO)
        if labels is not None:
            person_mask = labels == 0
            bboxes = bboxes[person_mask]
            scores = scores[person_mask]
    elif isinstance(det_result, (list, tuple)):
        # Legacy list-of-arrays format
        bboxes = np.array(det_result[0]) if len(det_result) > 0 else np.array([])
        if bboxes.size > 0:
            scores = bboxes[:, 4] if bboxes.shape[1] >= 5 else np.ones(len(bboxes))
            bboxes = bboxes[:, :4]
        else:
            scores = np.array([])
    else:
        bboxes = np.array([])
        scores = np.array([])

    people = []

    if len(bboxes) == 0:
        return {
            "image": str(image_path),
            "people": [],
            "image_width": w,
            "image_height": h,
        }

    # Filter low-confidence detections
    conf_threshold = 0.3
    valid = scores >= conf_threshold
    bboxes = bboxes[valid]
    scores = scores[valid]

    if len(bboxes) == 0:
        return {
            "image": str(image_path),
            "people": [],
            "image_width": w,
            "image_height": h,
        }

    # --- Pose estimation for each person ---
    pose_results = inference_topdown(pose_model, img, bboxes)

    for idx, (pose_result, bbox, score) in enumerate(
        zip(pose_results, bboxes, scores)
    ):
        # Extract keypoints
        if hasattr(pose_result, "pred_instances"):
            # PoseDataSample (mmpose 1.x)
            instances = pose_result.pred_instances
            kpts = instances.keypoints.cpu().numpy()  # shape (N_kpts, 2)
            kpt_scores = instances.keypoint_scores.cpu().numpy()  # shape (N_kpts,)
        elif isinstance(pose_result, dict):
            kpts = np.array(pose_result.get("keypoints", []))
            kpt_scores = (
                np.array(pose_result.get("keypoint_scores", []))
                if "keypoint_scores" in pose_result
                else np.ones(kpts.shape[0])
            )
        else:
            continue

        keypoints = []
        for kpt_idx, name in enumerate(COCO_KEYPOINT_NAMES):
            if kpt_idx < len(kpts):
                kp = kpts[kpt_idx]
                conf = float(kpt_scores[kpt_idx]) if kpt_idx < len(kpt_scores) else 0.0
                keypoints.append(
                    {
                        "name": name,
                        "x": round(float(kp[0]), 2),
                        "y": round(float(kp[1]), 2),
                        "confidence": round(float(conf), 4),
                    }
                )

        # Round bbox values
        bbox_list = [round(float(v), 1) for v in bbox[:4]]

        people.append(
            {
                "id": idx,
                "bbox": bbox_list,
                "keypoints": keypoints,
            }
        )

    return {
        "image": str(image_path),
        "people": people,
        "image_width": w,
        "image_height": h,
    }


def main():
    parser = argparse.ArgumentParser(
        description="VitPose single-image pose inference"
    )
    parser.add_argument("image", help="Path to the input image")
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda", "cuda:0", "cuda:1"],
        help="Device for inference (default: cpu)",
    )
    parser.add_argument(
        "--det-model",
        default=DEFAULT_DET_MODEL,
        help=f"Person detection model (default: {DEFAULT_DET_MODEL})",
    )
    parser.add_argument(
        "--pose-model",
        default=DEFAULT_POSE_MODEL,
        help=f"Pose estimation model (default: {DEFAULT_POSE_MODEL})",
    )
    args = parser.parse_args()

    image_path = args.image

    # Validate image exists
    if not Path(image_path).is_file():
        result = {"error": f"Image not found: {image_path}"}
        json.dump(result, sys.stdout)
        sys.exit(1)

    # Pre-flight dependency check
    missing = []
    for mod in ["mmcv", "mmdet", "mmpose", "mim"]:
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        result = {
            "error": (
                f"Missing Python packages: {', '.join(missing)}. "
                "Install them with:\n"
                "  pip install openmim\n"
                "  mim install mmcv mmpose mmdet mmengine"
            ),
        }
        json.dump(result, sys.stdout)
        sys.exit(1)

    try:
        detector, pose_model = load_models(
            args.det_model, args.pose_model, args.device
        )
        result = run_inference(image_path, detector, pose_model)
        json.dump(result, sys.stdout)
        sys.stdout.flush()
    except Exception as exc:
        result = {
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        json.dump(result, sys.stdout)
        sys.exit(1)


if __name__ == "__main__":
    main()
