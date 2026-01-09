from ultralytics import YOLO
import os
import shutil
from pathlib import Path
import yaml
def create_subset_data(data_root, subset_size, train_files):
    data_root = Path(data_root)
    subset_dir = data_root / f"subset_{subset_size}"
    train_images_dir = subset_dir / "images"
    train_labels_dir = subset_dir / "labels"
    train_images_dir.mkdir(parents=True, exist_ok=True)
    train_labels_dir.mkdir(parents=True, exist_ok=True)
    orig_train_images = data_root / "train" / "images"
    orig_train_labels = data_root / "train" / "labels"
    for base in train_files[:subset_size]:
        shutil.copy(str(orig_train_images / f"{base}.jpg"), str(train_images_dir / f"{base}.jpg"))
        shutil.copy(str(orig_train_labels / f"{base}.txt"), str(train_labels_dir / f"{base}.txt"))
    temp_yaml = {
        'train': str(train_images_dir.absolute()),
        'val': "/Users/andrei.ogurtsov/NUP/ImProc/ImageProcessing-HW-01/vllm/data/valid/images",
        'test': "/Users/andrei.ogurtsov/NUP/ImProc/ImageProcessing-HW-01/vllm/data/test/images",
        'nc': 1,
        'names': ['macbook']
    }
    yaml_path = subset_dir / "data.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(temp_yaml, f)
    return str(yaml_path.absolute())
def main():
    data_root = "/Users/andrei.ogurtsov/NUP/ImProc/ImageProcessing-HW-01/vllm/data"
    train_labels_dir = Path(data_root) / "train" / "labels"
    train_files = []
    for label_path in train_labels_dir.glob("*.txt"):
        if os.path.getsize(label_path) > 0:
            train_files.append(label_path.stem)
    print(f"Total available annotated train samples: {len(train_files)}")
    subset_sizes = [2, 4, 8, 16, 32, 64, 128, 256]
    subset_sizes = [s for s in subset_sizes if s <= len(train_files)]
    results = []
    for size in subset_sizes:
        print(f"\n--- Training with subset size: {size} ---")
        yaml_path = create_subset_data(data_root, size, train_files)
        model = YOLO("yolov8n.pt")
        model.train(data=yaml_path, epochs=10, imgsz=640, device='mps', verbose=False, exist_ok=True, project="vllm_training", name=f"size_{size}")
        val_results = model.val(data=yaml_path, split='test', device='mps', verbose=False)
        map50 = val_results.results_dict['metrics/mAP50(B)']
        print(f"Result for size {size}: mAP@0.5 = {map50:.4f}")
        results.append((size, map50))
    print("\nSummary of Progressive Training:")
    print("Size\tmAP@0.5")
    with open("/Users/andrei.ogurtsov/NUP/ImProc/ImageProcessing-HW-01/vllm/src/progressive_results.txt", "w") as f:
        f.write("Size\tmAP@0.5\n")
        for size, mAP in results:
            line = f"{size}\t{mAP:.4f}"
            print(line)
            f.write(line + "\n")
if __name__ == "__main__":
    main()
