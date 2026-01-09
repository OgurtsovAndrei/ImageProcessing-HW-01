import os
import json
import torch
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import numpy as np
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
def calculate_iou(box1, box2):
    """
    Calculate IoU between two bounding boxes.
    Boxes are in format [x_center, y_center, width, height] (normalized 0-1)
    """
    box1_x1 = box1[0] - box1[2] / 2
    box1_y1 = box1[1] - box1[3] / 2
    box1_x2 = box1[0] + box1[2] / 2
    box1_y2 = box1[1] + box1[3] / 2
    box2_x1 = box2[0] - box2[2] / 2
    box2_y1 = box2[1] - box2[3] / 2
    box2_x2 = box2[0] + box2[2] / 2
    box2_y2 = box2[1] + box2[3] / 2
    inter_x1 = max(box1_x1, box2_x1)
    inter_y1 = max(box1_y1, box2_y1)
    inter_x2 = min(box1_x2, box2_x2)
    inter_y2 = min(box1_y2, box2_y2)
    inter_width = max(0, inter_x2 - inter_x1)
    inter_height = max(0, inter_y2 - inter_y1)
    inter_area = inter_width * inter_height
    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]
    union_area = box1_area + box2_area - inter_area
    iou = inter_area / union_area if union_area > 0 else 0
    return iou
def load_yolo_annotation(label_path):
    """Load YOLO format annotation: class_id x_center y_center width height"""
    boxes = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id, x_center, y_center, width, height = parts
                    boxes.append([float(x_center), float(y_center), float(width), float(height)])
    return boxes
def parse_model_output(output_text, img_width, img_height):
    """
    Parse model output to extract bounding boxes.
    Expected format: JSON with bounding boxes
    """
    boxes = []
    try:
        start_idx = output_text.find('{')
        end_idx = output_text.rfind('}')
        if start_idx != -1 and end_idx != -1:
            json_str = output_text[start_idx:end_idx+1]
            data = json.loads(json_str)
            if 'objects' in data:
                for obj in data['objects']:
                    if 'bbox' in obj or 'bounding_box' in obj:
                        bbox = obj.get('bbox', obj.get('bounding_box'))
                        if isinstance(bbox, list) and len(bbox) == 4:
                            x1, y1, x2, y2 = bbox
                            if x2 > 1 or y2 > 1:
                                x1, y1, x2, y2 = x1/img_width, y1/img_height, x2/img_width, y2/img_height
                            x_center = (x1 + x2) / 2
                            y_center = (y1 + y2) / 2
                            width = abs(x2 - x1)
                            height = abs(y2 - y1)
                            boxes.append([x_center, y_center, width, height])
            elif 'bounding_boxes' in data:
                for bbox in data['bounding_boxes']:
                    if isinstance(bbox, list) and len(bbox) == 4:
                        x1, y1, x2, y2 = bbox
                        if x2 > 1 or y2 > 1:
                            x1, y1, x2, y2 = x1/img_width, y1/img_height, x2/img_width, y2/img_height
                        x_center = (x1 + x2) / 2
                        y_center = (y1 + y2) / 2
                        width = abs(x2 - x1)
                        height = abs(y2 - y1)
                        boxes.append([x_center, y_center, width, height])
        if not boxes:
            import re
            coord_pattern = r'\[(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\]'
            matches = re.findall(coord_pattern, output_text)
            for match in matches:
                x1, y1, x2, y2 = [float(x) for x in match]
                if x2 > 1 or y2 > 1:
                    x1, y1, x2, y2 = x1/img_width, y1/img_height, x2/img_width, y2/img_height
                x_center = (x1 + x2) / 2
                y_center = (y1 + y2) / 2
                width = abs(x2 - x1)
                height = abs(y2 - y1)
                boxes.append([x_center, y_center, width, height])
    except Exception as e:
        print(f"Error parsing output: {e}")
    return boxes
def evaluate_predictions(all_gt_boxes, all_pred_boxes, iou_threshold=0.5):
    """
    Calculate Mean IoU and mAP@0.5.
    all_gt_boxes: list of lists of boxes [x_center, y_center, width, height] (normalized)
    all_pred_boxes: list of lists of boxes [x_center, y_center, width, height] (normalized)
    """
    ious = []
    tp = 0
    fp = 0
    fn = 0
    for gt_boxes, pred_boxes in zip(all_gt_boxes, all_pred_boxes):
        if not gt_boxes:
            fp += len(pred_boxes)
            continue
        if not pred_boxes:
            fn += len(gt_boxes)
            for _ in gt_boxes:
                ious.append(0.0)
            continue
        for gt_box in gt_boxes:
            max_iou = 0
            for pred_box in pred_boxes:
                iou = calculate_iou(gt_box, pred_box)
                max_iou = max(max_iou, iou)
            ious.append(max_iou)
        matched_gt = [False] * len(gt_boxes)
        for p_box in pred_boxes:
            best_iou = 0
            best_gt_idx = -1
            for g_idx, g_box in enumerate(gt_boxes):
                if matched_gt[g_idx]: continue
                iou = calculate_iou(g_box, p_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = g_idx
            if best_iou >= iou_threshold:
                tp += 1
                matched_gt[best_gt_idx] = True
            else:
                fp += 1
        fn += len(gt_boxes) - sum(matched_gt)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    map50 = precision * recall
    mean_iou = np.mean(ious) if ious else 0
    return {
        "mean_iou": mean_iou,
        "map50": map50,
        "precision": precision,
        "recall": recall,
        "tp": tp,
        "fp": fp,
        "fn": fn
    }
def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    data_root = Path("/Users/andrei.ogurtsov/NUP/ImProc/ImageProcessing-HW-01/vllm/data")
    test_images_dir = data_root / "test" / "images"
    test_labels_dir = data_root / "test" / "labels"
    test_images = sorted(list(test_images_dir.glob("*.jpg")))
    test_images = test_images[:50]
    print(f"Using {len(test_images)} images as test set")
    print("Loading Qwen2-VL model...")
    model_name = "Qwen/Qwen2-VL-2B-Instruct"
    try:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "mps" else torch.float32,
            device_map=None
        )
        model = model.to(device)
        model.eval()
        processor = AutoProcessor.from_pretrained(
            model_name,
            min_pixels=256*28*28,
            max_pixels=512*28*28
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Model might not be available. Please check the model name or network connection.")
        return
    prompt = """Detect all MacBook laptops (Apple laptops) in this image. 
MacBooks are Apple-branded laptops with distinctive aluminum design and Apple logo.
Do NOT detect other laptop brands like Acer, Asus, Dell, HP, Lenovo, etc.
Only detect MacBooks.

Please provide the bounding box coordinates in JSON format:
{
  "objects": [
    {"name": "macbook", "bbox": [x_min, y_min, x_max, y_max]}
  ]
}
Where coordinates are in pixels relative to the image dimensions."""
    results = []
    all_gt_boxes = []
    all_pred_boxes = []
    for img_path in tqdm(test_images, desc="Processing images"):
        image = Image.open(img_path).convert("RGB")
        img_width, img_height = image.size
        label_path = test_labels_dir / (img_path.stem + ".txt")
        gt_boxes = load_yolo_annotation(label_path)
        all_gt_boxes.append(gt_boxes)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img_path.as_posix()},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        try:
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=512)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            pred_boxes = parse_model_output(output_text, img_width, img_height)
            all_pred_boxes.append(pred_boxes)
            results.append({
                "image": img_path.name,
                "ground_truth": gt_boxes,
                "predictions": pred_boxes,
                "output_text": output_text
            })
        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")
            all_pred_boxes.append([])
    if all_gt_boxes:
        metrics = evaluate_predictions(all_gt_boxes, all_pred_boxes)
        print(f"\n{'='*50}")
        print(f"Results:")
        print(f"{'='*50}")
        print(f"Total test images: {len(test_images)}")
        print(f"Successfully processed: {len(results)}")
        print(f"Mean IoU: {metrics['mean_iou']:.4f}")
        print(f"mAP@0.5: {metrics['map50']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"{'='*50}")
        output_file = Path("/Users/andrei.ogurtsov/NUP/ImProc/ImageProcessing-HW-01/vllm/src/results.json")
        with open(output_file, 'w') as f:
            json.dump({
                "mean_iou": float(metrics['mean_iou']),
                "map50": float(metrics['map50']),
                "precision": float(metrics['precision']),
                "recall": float(metrics['recall']),
                "num_images": len(test_images),
                "num_processed": len(results),
                "detailed_results": results[:10]
            }, f, indent=2)
        print(f"Detailed results saved to: {output_file}")
    else:
        print("No results to report")
if __name__ == "__main__":
    main()
