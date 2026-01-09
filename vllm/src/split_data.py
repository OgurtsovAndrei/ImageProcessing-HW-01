import os
import shutil
from pathlib import Path
import random


def split_annotated_data(data_root, val_size=50, test_size=50):
    data_root = Path(data_root)
    train_images = data_root / "train" / "images"
    train_labels = data_root / "train" / "labels"
    valid_images = data_root / "valid" / "images"
    valid_labels = data_root / "valid" / "labels"
    test_images = data_root / "test" / "images"
    test_labels = data_root / "test" / "labels"
    for d in [valid_images, valid_labels, test_images, test_labels]:
        d.mkdir(parents=True, exist_ok=True)
    annotated_files = []
    for label_path in train_labels.glob("*.txt"):
        if os.path.getsize(label_path) > 0:
            base = label_path.stem
            img_path = train_images / f"{base}.jpg"
            if img_path.exists():
                annotated_files.append(base)
    print(f"Found {len(annotated_files)} annotated images")
    random.seed(42)
    random.shuffle(annotated_files)
    test_files = annotated_files[:test_size]
    val_files = annotated_files[test_size:test_size + val_size]
    for base in test_files:
        shutil.move(str(train_images / f"{base}.jpg"), str(test_images / f"{base}.jpg"))
        shutil.move(str(train_labels / f"{base}.txt"), str(test_labels / f"{base}.txt"))
    for base in val_files:
        shutil.move(str(train_images / f"{base}.jpg"), str(valid_images / f"{base}.jpg"))
        shutil.move(str(train_labels / f"{base}.txt"), str(valid_labels / f"{base}.txt"))
    print(f"Moved {len(test_files)} files to test set")
    print(f"Moved {len(val_files)} files to valid set")


if __name__ == "__main__":
    split_annotated_data("/Users/andrei.ogurtsov/NUP/ImProc/ImageProcessing-HW-01/vllm/data")
