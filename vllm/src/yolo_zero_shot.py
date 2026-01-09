from ultralytics import YOLO
import os


def main():
    model = YOLO("yolov8n.pt")
    data_path = "/Users/andrei.ogurtsov/NUP/ImProc/ImageProcessing-HW-01/vllm/data/data.yaml"
    print("Running zero-shot YOLO on test set...")
    results = model.val(data=data_path, split='test', device='mps')
    map50 = results.results_dict['metrics/mAP50(B)']
    print(f"\n{'=' * 50}")
    print(f"Zero-shot YOLO mAP@0.5: {map50:.4f}")
    print(f"{'=' * 50}")
    with open("/Users/andrei.ogurtsov/NUP/ImProc/ImageProcessing-HW-01/vllm/results/yolo_zero_shot_results.txt",
              "w") as f:
        f.write(f"Zero-shot YOLO mAP@0.5: {map50:.4f}\n")


if __name__ == "__main__":
    main()
