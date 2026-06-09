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

def download_models(s_dir_model):
    """Download models from HuggingFace and save locally"""
    s_dir_model = str(Path(s_dir_model).resolve())
    s_path_person_detector = os.path.join(s_dir_model, "person_detector")
    s_path_pose_estimator = os.path.join(s_dir_model, "pose_estimator")

    os.makedirs(s_path_person_detector, exist_ok=True)
    os.makedirs(s_path_pose_estimator, exist_ok=True)

    print(f"Downloading person detector to {s_path_person_detector}...", file=sys.stderr)
    processor = AutoProcessor.from_pretrained("PekingU/rtdetr_r50vd_coco_o365")
    processor.save_pretrained(s_path_person_detector)
    model = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r50vd_coco_o365")
    model.save_pretrained(s_path_person_detector)

    print(f"Downloading pose estimator to {s_path_pose_estimator}...", file=sys.stderr)
    processor = AutoProcessor.from_pretrained("usyd-community/vitpose-base-simple")
    processor.save_pretrained(s_path_pose_estimator)
    model = VitPoseForPoseEstimation.from_pretrained("usyd-community/vitpose-base-simple")
    model.save_pretrained(s_path_pose_estimator)

    print("Models downloaded!", file=sys.stderr)

def main():
    parser = argparse.ArgumentParser(description="Batch pose estimation with VitPose")
    parser.add_argument("image_path", nargs="*", help="Image file paths to process")
    parser.add_argument("--model-dir", default="./models", help="Directory containing model subdirectories (person_detector, pose_estimator)")
    parser.add_argument("--download-models", action="store_true", help="Download models to --model-dir and exit")
    o_arg = parser.parse_args()

    if o_arg.download_models:
        download_models(o_arg.model_dir)
        return

    if len(o_arg.image_path) == 0:
        print(json.dumps({
            "error": "No image paths provided",
            "usage": "python vitpose_batch_processing.py [--model-dir DIR] <image1> <image2> ..."
        }), file=sys.stderr)
        sys.exit(1)

    image_path = o_arg.image_path
    s_dir_model = str(Path(o_arg.model_dir).resolve())
    s_path_person_detector = os.path.join(s_dir_model, "person_detector")
    s_path_pose_estimator = os.path.join(s_dir_model, "pose_estimator")

    if not os.path.isdir(s_path_person_detector) or not os.path.isdir(s_path_pose_estimator):
        print(f"Models not found in {s_dir_model}, downloading automatically...", file=sys.stderr)
        download_models(s_dir_model)

    # Print to stderr so it doesn't interfere with JSON output
    print(f"Processing {len(image_path)} images...", file=sys.stderr)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", file=sys.stderr)

    # Load models ONCE
    # do not use local_files_only=True with absolute paths, it causes huggingface_hub repo_id validation errors
    print(f"Loading models from {s_dir_model}...", file=sys.stderr)

    person_processor = AutoProcessor.from_pretrained(s_path_person_detector)
    person_model = RTDetrForObjectDetection.from_pretrained(s_path_person_detector).to(device)

    pose_processor = AutoProcessor.from_pretrained(s_path_pose_estimator)
    pose_model = VitPoseForPoseEstimation.from_pretrained(s_path_pose_estimator).to(device)

    print("Models loaded!", file=sys.stderr)

    # Process all images
    results = []
    for idx, s_path_image in enumerate(image_path, 1):
        print(f"Processing {idx}/{len(image_path)}: {s_path_image}", file=sys.stderr)
        result = process_image(s_path_image, person_processor, person_model, pose_processor, pose_model, device)
        results.append(result)

    # Output JSON to stdout
    output = {
        "total_images": len(image_path),
        "successful": sum(1 for r in results if r["success"]),
        "failed": sum(1 for r in results if not r["success"]),
        "results": results
    }

    print(json.dumps(output, indent=2))

if __name__ == "__main__":
    main()