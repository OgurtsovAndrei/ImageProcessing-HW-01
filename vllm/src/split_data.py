import os
import shutil
from pathlib import Path
import random

def split_dataset(data_root, val_annotated=50, test_annotated=50, val_bg=50, test_bg=50):
    data_root = Path(data_root)
    train_images = data_root / "train" / "images"
    train_labels = data_root / "train" / "labels"
    valid_images = data_root / "valid" / "images"
    valid_labels = data_root / "valid" / "labels"
    test_images = data_root / "test" / "images"
    test_labels = data_root / "test" / "labels"

    print("Resetting split: moving files back to train...")
    for d_img, d_lbl in [(valid_images, valid_labels), (test_images, test_labels)]:
        if d_img.exists():
            for img in d_img.glob("*.jpg"):
                dest = train_images / img.name
                if dest.exists():
                    os.remove(dest)
                shutil.move(str(img), str(dest))
        if d_lbl.exists():
            for lbl in d_lbl.glob("*.txt"):
                dest = train_labels / lbl.name
                if dest.exists():
                    os.remove(dest)
                shutil.move(str(lbl), str(dest))

    for d in [valid_images, valid_labels, test_images, test_labels]:
        d.mkdir(parents=True, exist_ok=True)

    annotated_files = []
    bg_files = []
    
    for label_path in train_labels.glob("*.txt"):
        base = label_path.stem
        img_path = train_images / f"{base}.jpg"
        if not img_path.exists():
            continue
            
        if os.path.getsize(label_path) > 0:
            annotated_files.append(base)
        else:
            bg_files.append(base)

    print(f"Found {len(annotated_files)} annotated and {len(bg_files)} background images")

    random.seed(42)
    random.shuffle(annotated_files)
    random.shuffle(bg_files)

    selected_test_ann = annotated_files[:test_annotated]
    selected_val_ann = annotated_files[test_annotated : test_annotated + val_annotated]
    
    selected_test_bg = bg_files[:test_bg]
    selected_val_bg = bg_files[test_bg : test_bg + val_bg]

    def move_files(files, target_img_dir, target_lbl_dir):
        for base in files:
            shutil.move(str(train_images / f"{base}.jpg"), str(target_img_dir / f"{base}.jpg"))
            shutil.move(str(train_labels / f"{base}.txt"), str(target_lbl_dir / f"{base}.txt"))

    move_files(selected_test_ann, test_images, test_labels)
    move_files(selected_test_bg, test_images, test_labels)
    move_files(selected_val_ann, valid_images, valid_labels)
    move_files(selected_val_bg, valid_images, valid_labels)

    print(f"Test set: {len(selected_test_ann)} annotated, {len(selected_test_bg)} background")
    print(f"Valid set: {len(selected_val_ann)} annotated, {len(selected_val_bg)} background")

if __name__ == "__main__":
    split_dataset("/Users/andrei.ogurtsov/NUP/ImProc/ImageProcessing-HW-01/vllm/data")
