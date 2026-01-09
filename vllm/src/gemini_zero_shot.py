import os
import json
import time
from pathlib import Path
from tqdm import tqdm
import numpy as np
import google.generativeai as genai
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from zero_shot_detection import calculate_iou, load_yolo_annotation, evaluate_predictions, parse_model_output


def process_single_image(model, img_path, prompt, test_labels_dir):
    try:
        image = Image.open(img_path).convert("RGB")
        img_width, img_height = image.size
        label_path = test_labels_dir / (img_path.stem + ".txt")
        gt_boxes = load_yolo_annotation(label_path)
        response = model.generate_content([prompt, image])
        output_text = response.text
        # Gemini uses 0-1000 scale by default
        pred_boxes = parse_model_output(output_text, img_width, img_height, target_scale=1000)
        return {
            "image": img_path.name,
            "ground_truth": gt_boxes,
            "predictions": pred_boxes,
            "output_text": output_text,
            "success": True
        }
    except Exception as e:
        print(f"Error processing {img_path.name}: {e}")
        return {
            "image": img_path.name,
            "success": False,
            "error": str(e)
        }


def main():
    api_key = os.environ.get("GEMINI_API_KEY2")
    if not api_key:
        print("Error: GEMINI_API_KEY2 environment variable not set.")
        return
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-3-flash-preview')
    data_root = Path("/Users/andrei.ogurtsov/NUP/ImProc/ImageProcessing-HW-01/vllm/data")
    test_images_dir = data_root / "test" / "images"
    test_labels_dir = data_root / "test" / "labels"
    test_images = sorted(list(test_images_dir.glob("*.jpg")))
    print(f"Using {len(test_images)} images as test set")
    prompt = """Detect all MacBook laptops (Apple laptops) in this image. 
MacBooks are Apple-branded laptops with distinctive aluminum design and Apple logo.
Do NOT detect other laptop brands like Acer, Asus, Dell, HP, Lenovo, etc.
Only detect MacBooks.

Please provide the bounding box coordinates and confidence score (0 to 1) in JSON format:
{
  "objects": [
    {"name": "macbook", "bbox": [x_min, y_min, x_max, y_max], "confidence": score}
  ]
}
Where coordinates are normalized to [0, 1000] scale (i.e. top-left is [0, 0] and bottom-right is [1000, 1000]).
Return ONLY the JSON object."""
    results = []
    max_workers = 32
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_single_image, model, img_path, prompt, test_labels_dir) for img_path in
                   test_images]
        for future in tqdm(futures, desc="Processing images with Gemini (parallel)"):
            results.append(future.result())
    all_gt_boxes = []
    all_pred_boxes = []
    processed_results = []
    for res in results:
        if res["success"]:
            all_gt_boxes.append(res["ground_truth"])
            all_pred_boxes.append(res["predictions"])
            processed_results.append(res)
        else:
            img_name = res["image"]
            label_path = test_labels_dir / (Path(img_name).stem + ".txt")
            gt_boxes = load_yolo_annotation(label_path)
            all_gt_boxes.append(gt_boxes)
            all_pred_boxes.append([])
    if all_gt_boxes:
        metrics = evaluate_predictions(all_gt_boxes, all_pred_boxes)
        print(f"\n{'=' * 50}")
        print(f"Gemini Zero-shot Results:")
        print(f"{'=' * 50}")
        print(f"Total test images: {len(test_images)}")
        print(f"Mean IoU: {metrics['mean_iou']:.4f}")
        print(f"mAP@0.5: {metrics['map50']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"{'=' * 50}")
        with open("/Users/andrei.ogurtsov/NUP/ImProc/ImageProcessing-HW-01/vllm/results/gemini_zero_shot_results.txt",
                  "w") as f:
            f.write(f"Gemini Zero-shot Results:\n")
            f.write(f"Mean IoU: {metrics['mean_iou']:.4f}\n")
            f.write(f"mAP@0.5: {metrics['map50']:.4f}\n")
            f.write(f"Precision: {metrics['precision']:.4f}\n")
            f.write(f"Recall: {metrics['recall']:.4f}\n")
        with open("/Users/andrei.ogurtsov/NUP/ImProc/ImageProcessing-HW-01/vllm/results/gemini_zero_shot_results.json",
                  "w") as f:
            json.dump({
                "mean_iou": float(metrics['mean_iou']),
                "map50": float(metrics['map50']),
                "precision": float(metrics['precision']),
                "recall": float(metrics['recall']),
                "num_images": len(test_images),
                "detailed_results": processed_results
            }, f, indent=2)
    else:
        print("No results to report")


if __name__ == "__main__":
    main()
