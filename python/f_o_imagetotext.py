# Copyright (C) 2026 Jonas Immanuel Frey - Licensed under GPL-2.0-only. See LICENSE file for details.

import sys
import os
import json
import argparse
from pathlib import Path
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch


def process_image(image_path, processor, model, device):
    """Process a single image and return a description."""
    abs_path = str(Path(image_path).resolve())

    try:
        image = Image.open(image_path).convert("RGB")

        inputs = processor(image, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=150)
        s_description = processor.decode(output[0], skip_special_tokens=True)

        return {
            "image_path": abs_path,
            "success": True,
            "s_description": s_description,
        }

    except Exception as e:
        return {
            "image_path": abs_path,
            "success": False,
            "error": str(e),
            "s_description": "",
        }


def download_models(s_dir_model):
    """Download models from HuggingFace and save locally."""
    s_dir_model = str(Path(s_dir_model).resolve())
    s_path_imagetotext = os.path.join(s_dir_model, "imagetotext")

    os.makedirs(s_path_imagetotext, exist_ok=True)

    print(f"Downloading BLIP image captioning model to {s_path_imagetotext}...", file=sys.stderr)
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    processor.save_pretrained(s_path_imagetotext)
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
    model.save_pretrained(s_path_imagetotext)

    print("Models downloaded!", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Batch image captioning with BLIP")
    parser.add_argument("image_path", nargs="*", help="Image file paths to process")
    parser.add_argument("--model-dir", default="./models", help="Directory containing model subdirectory (imagetotext)")
    parser.add_argument("--download-models", action="store_true", help="Download models to --model-dir and exit")
    o_arg = parser.parse_args()

    if o_arg.download_models:
        download_models(o_arg.model_dir)
        return

    if len(o_arg.image_path) == 0:
        print(json.dumps({
            "error": "No image paths provided",
            "usage": "python f_o_imagetotext.py [--model-dir DIR] <image1> <image2> ..."
        }), file=sys.stderr)
        sys.exit(1)

    image_paths = o_arg.image_path
    s_dir_model = str(Path(o_arg.model_dir).resolve())
    s_path_imagetotext = os.path.join(s_dir_model, "imagetotext")

    a_s_weights = ["model.safetensors", "pytorch_model.bin"]
    b_has_weights = any(os.path.isfile(os.path.join(s_path_imagetotext, s)) for s in a_s_weights)
    if not b_has_weights:
        print(f"Model weights not found in {s_path_imagetotext}, downloading...", file=sys.stderr)
        download_models(s_dir_model)

    print(f"Processing {len(image_paths)} images...", file=sys.stderr)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", file=sys.stderr)

    print(f"Loading models from {s_dir_model}...", file=sys.stderr)
    processor = BlipProcessor.from_pretrained(s_path_imagetotext)
    model = BlipForConditionalGeneration.from_pretrained(s_path_imagetotext).to(device)
    print("Models loaded!", file=sys.stderr)

    results = []
    for idx, s_path_image in enumerate(image_paths, 1):
        print(f"Processing {idx}/{len(image_paths)}: {s_path_image}", file=sys.stderr)
        result = process_image(s_path_image, processor, model, device)
        results.append(result)

    output = {
        "total_images": len(image_paths),
        "successful": sum(1 for r in results if r["success"]),
        "failed": sum(1 for r in results if not r["success"]),
        "results": results
    }

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
