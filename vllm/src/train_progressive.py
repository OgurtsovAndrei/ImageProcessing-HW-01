from ultralytics import YOLO
import os
import shutil
from pathlib import Path
import yaml


def create_subset_data(data_root, subset_size, train_files, background_files):
    data_root = Path(data_root)
    subset_dir = data_root / f"subset_{subset_size}"
    train_images_dir = subset_dir / "images"
    train_labels_dir = subset_dir / "labels"
    train_images_dir.mkdir(parents=True, exist_ok=True)
    train_labels_dir.mkdir(parents=True, exist_ok=True)
    orig_train_images = data_root / "train" / "images"
    orig_train_labels = data_root / "train" / "labels"
    
    num_annotated = subset_size // 2
    num_background = subset_size - num_annotated
    
    # Copy annotated files
    for base in train_files[:num_annotated]:
        shutil.copy(str(orig_train_images / f"{base}.jpg"), str(train_images_dir / f"{base}.jpg"))
        shutil.copy(str(orig_train_labels / f"{base}.txt"), str(train_labels_dir / f"{base}.txt"))
    
    # Copy background files
    for base in background_files[:num_background]:
        shutil.copy(str(orig_train_images / f"{base}.jpg"), str(train_images_dir / f"{base}.jpg"))
        shutil.copy(str(orig_train_labels / f"{base}.txt"), str(train_labels_dir / f"{base}.txt"))

    temp_yaml = {
        'train': str(train_images_dir.absolute()),
        'val': "/Users/andrei.ogurtsov/NUP/ImProc/ImageProcessing-HW-01/vllm/data/valid/images",
        'test': "/Users/andrei.ogurtsov/NUP/ImProc/ImageProcessing-HW-01/vllm/data/test/images",
        'nc': 80,
        'names': {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}
    }
    yaml_path = subset_dir / "data.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(temp_yaml, f)
    return str(yaml_path.absolute())


def main():
    data_root = "/Users/andrei.ogurtsov/NUP/ImProc/ImageProcessing-HW-01/vllm/data"
    train_labels_dir = Path(data_root) / "train" / "labels"
    train_files = []
    background_files = []
    for label_path in train_labels_dir.glob("*.txt"):
        if os.path.getsize(label_path) > 0:
            train_files.append(label_path.stem)
        else:
            background_files.append(label_path.stem)
    
    print(f"Total available annotated train samples: {len(train_files)}")
    print(f"Total available background train samples: {len(background_files)}")
    
    subset_sizes = [2, 4, 8, 16, 32, 64, 128, 256, 512]
    subset_sizes = [s for s in subset_sizes if (s // 2 <= len(train_files)) and (s - s // 2 <= len(background_files))]
    results = []
    for size in subset_sizes:
        num_annotated = size // 2
        num_background = size - num_annotated
        print(f"\n--- Training with subset size: {size} ({num_annotated} annotated + {num_background} backgrounds) ---")
        yaml_path = create_subset_data(data_root, size, train_files, background_files)
        model_path = Path(f"vllm_training/size_{size}/weights/best.pt")
        if not model_path.exists():
            model = YOLO("yolov8n.pt")
            model.train(data=yaml_path, epochs=10, imgsz=640, device='mps', verbose=False, exist_ok=True,
                        project="vllm_training", name=f"size_{size}")
        
        model = YOLO(model_path)
        val_results = model.val(data=yaml_path, split='test', device='mps', verbose=False)
        map50 = val_results.results_dict['metrics/mAP50(B)']
        print(f"Result for size {size}: mAP@0.5 = {map50:.4f}")
        results.append((size, map50))
    print("\nSummary of Progressive Training:")
    print("Size\tmAP@0.5")
    with open("/Users/andrei.ogurtsov/NUP/ImProc/ImageProcessing-HW-01/vllm/results/yolo_progressive_results.txt",
              "w") as f:
        f.write("Size\tmAP@0.5\n")
        for size, mAP in results:
            line = f"{size}\t{mAP:.4f}"
            print(line)
            f.write(line + "\n")


if __name__ == "__main__":
    main()
