import os
import torch
from ultralytics import YOLO
from pathlib import Path
from tqdm import tqdm
import numpy as np
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from zero_shot_detection import calculate_iou, load_yolo_annotation, evaluate_predictions
def get_yolo_predictions(model, img_path, conf=0.01):
    results = model(img_path, conf=conf, verbose=False)
    pred_boxes = []
    for r in results:
        if r.boxes is not None:
            for box, cls in zip(r.boxes.xywhn, r.boxes.cls):
                if int(cls) == 0:
                    pred_boxes.append(box.tolist())
    return pred_boxes
def evaluate_model(model, test_images, test_labels_dir):
    all_gt_boxes = []
    all_pred_boxes = []
    for img_path in tqdm(test_images, desc="Evaluating", leave=False):
        gt_boxes = load_yolo_annotation(test_labels_dir / (img_path.stem + ".txt"))
        all_gt_boxes.append(gt_boxes)
        pred_boxes = get_yolo_predictions(model, img_path)
        all_pred_boxes.append(pred_boxes)
    return evaluate_predictions(all_gt_boxes, all_pred_boxes)
def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    data_root = Path("/Users/andrei.ogurtsov/NUP/ImProc/ImageProcessing-HW-01/vllm/data")
    test_images_dir = data_root / "test" / "images"
    test_labels_dir = data_root / "test" / "labels"
    test_images = sorted(list(test_images_dir.glob("*.jpg")))
    print(f"Evaluating on {len(test_images)} images")
    print("Evaluating Zero-shot YOLO...")
    model_zs = YOLO("yolov8n.pt")
    metrics_zs = evaluate_model(model_zs, test_images, test_labels_dir)
    print(f"Zero-shot YOLO Mean IoU: {metrics_zs['mean_iou']:.4f}, mAP@0.5: {metrics_zs['map50']:.4f}")
    output_file = "yolo_iou_results.txt"
    with open(output_file, "w") as f:
        f.write("Size\tmAP@0.5\tMean_IoU\n")
        f.write(f"0\t{metrics_zs['map50']:.4f}\t{metrics_zs['mean_iou']:.4f}\n")
        subset_sizes = [2, 4, 8, 16, 32, 64, 128, 256]
        for size in subset_sizes:
            model_path = Path(f"vllm_training/size_{size}/weights/best.pt")
            if model_path.exists():
                print(f"Evaluating YOLO size {size}...")
                model = YOLO(model_path)
                metrics = evaluate_model(model, test_images, test_labels_dir)
                print(f"Size {size} Mean IoU: {metrics['mean_iou']:.4f}, mAP@0.5: {metrics['map50']:.4f}")
                f.write(f"{size}\t{metrics['map50']:.4f}\t{metrics['mean_iou']:.4f}\n")
            else:
                print(f"Warning: Model for size {size} not found at {model_path}")
if __name__ == "__main__":
    main()
