import os
import shutil
from pathlib import Path


def move_labels(images_dir, labels_src_dir, labels_dst_dir):
    images_dir = Path(images_dir)
    labels_src_dir = Path(labels_src_dir)
    labels_dst_dir = Path(labels_dst_dir)
    if not labels_dst_dir.exists():
        labels_dst_dir.mkdir(parents=True)
    for img_path in images_dir.glob("*.jpg"):
        base = img_path.stem
        src_label = labels_src_dir / f"{base}.txt"
        if src_label.exists():
            shutil.move(str(src_label), str(labels_dst_dir / f"{base}.txt"))
            print(f"Moved {src_label.name}")


if __name__ == "__main__":
    data_root = "/Users/andrei.ogurtsov/NUP/ImProc/ImageProcessing-HW-01/vllm/data"
    print("Moving valid labels...")
    move_labels(f"{data_root}/valid/images", f"{data_root}/train/labels", f"{data_root}/valid/labels")
    print("Moving test labels...")
    move_labels(f"{data_root}/test/images", f"{data_root}/train/labels", f"{data_root}/test/labels")
