import os
import shutil
import random
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from torchvision.utils import make_grid
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def _augment_single_image(args):
    """
    Worker function to augment a single image. It's designed to be called
    by a ProcessPoolExecutor.
    """
    source_image_path, class_dir, augmentation_transform, i = args
    try:
        image = Image.open(source_image_path).convert('RGB')
        augmented_image = augmentation_transform(image)

        # Use a completely new, unique name to avoid any conflicts
        output_filename = f"aug_set_{random.randint(10000, 99999)}_{i}.jpg"
        output_path = os.path.join(class_dir, output_filename)

        augmented_image.save(output_path, "JPEG")  # Save as JPEG for consistency
        return output_path
    except Exception as e:
        # Return error message instead of path
        return f"Error processing {source_image_path}: {e}"


def _augment_and_balance_class(class_dir, target_count):
    """
    Replaces the original images in a class directory with a new, fully augmented
    set of images of a specific target count, processed in parallel.
    """
    print(f"\nProcessing class: '{os.path.basename(class_dir)}'")
    original_image_paths = [os.path.join(class_dir, f) for f in os.listdir(class_dir) if
                            f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    current_count = len(original_image_paths)
    print(f"  - Initial image count: {current_count}")

    if current_count == 0:
        print("    - Warning: No images found in directory to augment. Skipping.")
        return

    if target_count == 0:
        print("   - Warning: target_count is 0. All original images will be deleted.")
        for f_path in tqdm(original_image_paths, desc="Deleting all images"):
            os.remove(f_path)
        return

    augmentation_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    ])

    # --- STAGE 1: Generate a new, fully augmented dataset in parallel ---
    print(f"  - Stage 1: Generating {target_count} new augmented images...")
    repeats = target_count // current_count
    remainder = target_count % current_count
    source_list = original_image_paths * repeats + original_image_paths[:remainder]
    random.shuffle(source_list)

    new_image_paths = []
    tasks = [(path, class_dir, augmentation_transform, i) for i, path in enumerate(source_list)]

    num_workers = multiprocessing.cpu_count()
    print(f"    - Using {num_workers} workers for parallel augmentation...")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(_augment_single_image, task) for task in tasks]
        for future in tqdm(as_completed(futures), total=len(tasks), desc="Generating new set"):
            result = future.result()
            if isinstance(result, str) and not result.startswith("Error"):
                new_image_paths.append(result)
            elif isinstance(result, str):
                print(f"\n{result}") # Print augmentation errors

    # --- STAGE 2: Delete all original images ---
    print(f"  - Stage 2: Deleting {len(original_image_paths)} original images...")
    for f_path in tqdm(original_image_paths, desc="Deleting originals"):
        try:
            if f_path not in new_image_paths:
                os.remove(f_path)
        except OSError as e:
            print(f"\nWarning: Could not delete file {f_path}. Error: {e}")

    # Final verification
    final_image_paths = [os.path.join(class_dir, f) for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"  - Final image count: {len(final_image_paths)}")


def _copy_files_to_split_dirs(splits, source_class_names, processed_base_dir):
    for split_name, (paths, labels) in splits.items():
        print(f"\nProcessing '{split_name}' split...")
        for i, path in tqdm(enumerate(paths), total=len(paths)):
            label = labels[i]
            class_name = source_class_names[label]

            target_dir = os.path.join(processed_base_dir, split_name, class_name)
            os.makedirs(target_dir, exist_ok=True)

            shutil.copy(path, target_dir)
        print(f"Copied {len(paths)} original images.")


def _augment_train_set(processed_base_dir, source_class_names, target_aug_count):
    print("\n--- Augmenting and Balancing Training Set ---")
    train_dir = os.path.join(processed_base_dir, 'train')
    train_class_dirs = [os.path.join(train_dir, name) for name in source_class_names]

    if target_aug_count is None:
        counts = [len(os.listdir(d)) for d in train_class_dirs if os.path.isdir(d)]
        if counts:
            target_aug_count = max(counts)
            print(f"Target count not provided. Balancing to the largest class size: {target_aug_count} images.")
        else:
            print("Warning: No class directories found to determine target count.")
            return

    for class_dir in train_class_dirs:
        if os.path.isdir(class_dir):
            _augment_and_balance_class(class_dir, target_aug_count)
        else:
            print(f"Warning: Class directory not found, skipping: {class_dir}")


def setup_dataset(
        source_base_dir='data',
        processed_base_dir='data/processed',
        source_class_names=None,
        test_size=0.15,
        val_size=0.15,
        target_aug_count=None,
        random_state=42
):
    if source_class_names is None:
        source_class_names = ['mac-merged', 'laptops-merged']

    if os.path.isdir(processed_base_dir):
        print(f"Directory '{processed_base_dir}' already exists. Skipping dataset creation and augmentation.")
        return

    print(f"Creating new dataset structure in '{processed_base_dir}'...")

    all_image_paths = []
    all_labels = []
    for i, class_name in enumerate(source_class_names):
        class_dir = os.path.join(source_base_dir, class_name)
        if not os.path.isdir(class_dir):
            raise FileNotFoundError(f"Source directory not found: {class_dir}")

        for filename in tqdm(os.listdir(class_dir)):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                all_image_paths.append(os.path.join(class_dir, filename))
                all_labels.append(i)

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        all_image_paths, all_labels, test_size=test_size, random_state=random_state, stratify=all_labels
    )
    relative_val_size = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=relative_val_size, random_state=random_state, stratify=y_train_val
    )

    splits = {'train': (X_train, y_train), 'val': (X_val, y_val), 'test': (X_test, y_test)}

    _copy_files_to_split_dirs(splits, source_class_names, processed_base_dir)

    _augment_train_set(processed_base_dir, source_class_names, target_aug_count)

    print("\nDataset preparation complete!")


def get_dataloaders(processed_dir='data/processed', batch_size=32):
    if not os.path.isdir(processed_dir):
        raise FileNotFoundError(f"Processed data directory '{processed_dir}' not found. "
                                f"Run setup_dataset() first.")

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        # Random augmentations are still useful during training
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    eval_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(root=os.path.join(processed_dir, 'train'), transform=train_transform)
    val_dataset = datasets.ImageFolder(root=os.path.join(processed_dir, 'val'), transform=eval_test_transform)
    test_dataset = datasets.ImageFolder(root=os.path.join(processed_dir, 'test'), transform=eval_test_transform)

    print("\nDataset Information:")
    print(f"  Training set: {len(train_dataset)} images in {len(train_dataset.classes)} classes.")
    print(f"  Validation set: {len(val_dataset)} images.")
    print(f"  Test set: {len(test_dataset)} images.")
    print(f"  Classes: {train_dataset.class_to_idx}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

    print(f"\nDataLoaders created with batch size {batch_size}.")

    return train_loader, val_loader, test_loader


# ========================= New real-time dataset setup ========================= #

class ImagePathsDataset(Dataset):
    """Simple dataset over (path, label) pairs with a transform applied at fetch time."""

    def __init__(self, samples: Sequence[Tuple[str, int]], transform: Optional[Callable] = None):
        self.samples: List[Tuple[str, int]] = list(samples)
        self.transform: Optional[Callable] = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        with Image.open(path).convert('RGB') as img:
            image = img.copy()
        if self.transform is not None:
            image = self.transform(image)
        return image, label


@dataclass
class DatasetBundle:
    """Container returned by setup_dataset_realtime with all relevant objects and metadata."""
    # Datasets
    train_dataset: Dataset
    val_dataset: Dataset
    test_dataset: Dataset

    # Loaders
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader

    # Metadata
    classes: List[str]
    class_to_idx: dict
    idx_to_class: List[str]
    num_classes: int
    class_counts: List[int]
    class_weights: torch.Tensor


def _default_augmentation_transform() -> transforms.Compose:
    """Augmentation-only transform applied to TRAIN split before model_transform."""
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    ])


def _default_model_transform() -> transforms.Compose:
    """Model/preprocessing transform applied to ALL splits (after augmentation on train)."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def _compute_class_weights(y: Sequence[int], num_classes: int) -> Tuple[List[int], torch.Tensor]:
    """Return (counts_per_class, weights_tensor) based on inverse frequency."""
    if len(y) == 0:
        counts = [0] * num_classes
        return counts, torch.ones(num_classes, dtype=torch.float32)
    counts_tensor = torch.bincount(torch.tensor(y, dtype=torch.long), minlength=num_classes)
    counts = counts_tensor.tolist()
    # Avoid division by zero: assign zero weight to empty classes
    weights = []
    n = counts_tensor.sum().item()
    for c in counts:
        if c == 0:
            weights.append(0.0)
        else:
            weights.append(n / (num_classes * float(c)))
    return counts, torch.tensor(weights, dtype=torch.float32)


def setup_dataset_realtime(
    source_base_dir: str = 'data',
    batch_size: int = 32,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
    augmentation_transform: Optional[Callable] = None,
    model_transform: Optional[Callable] = None,
    num_workers: int = 4,
    pin_memory: bool = True,
    persistent_workers: bool = True,
) -> DatasetBundle:
    """
    Create train/val/test datasets and loaders with on-the-fly augmentation, without
    generating or copying files. Class weights are computed from the training split.

    Different models can pass their own transforms via augmentation_transform and model_transform.
    - model_transform is ALWAYS applied to all splits.
    - augmentation_transform is applied ONLY on the training split, BEFORE model_transform.
    """

    if not os.path.isdir(source_base_dir):
        raise FileNotFoundError(f"Source data directory not found: {source_base_dir}")

    # Use ImageFolder to enumerate (path, class) pairs according to directory names
    base: datasets.ImageFolder = datasets.ImageFolder(root=source_base_dir)
    classes: List[str] = list(base.classes)
    class_to_idx: Dict[str, int] = dict(base.class_to_idx)
    num_classes: int = len(classes)

    all_paths_tuple, all_labels_tuple = zip(*base.samples) if len(base.samples) > 0 else ([], [])
    all_paths: List[str] = list(all_paths_tuple)
    all_labels: List[int] = list(all_labels_tuple)

    if len(all_paths) == 0:
        raise RuntimeError(f"No images found in '{source_base_dir}'. Expected class subfolders with images.")

    # Stratified splits
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        all_paths, all_labels, test_size=test_size, random_state=random_state, stratify=all_labels
    )
    relative_val_size: float = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=relative_val_size, random_state=random_state, stratify=y_train_val
    )

    # Transforms
    aug_tf = augmentation_transform if augmentation_transform is not None else _default_augmentation_transform()
    model_tf = model_transform if model_transform is not None else _default_model_transform()

    # Compose train transform as [augmentation -> model]
    if aug_tf is None:  # mypy safety; logically never None here
        train_tf = model_tf
    else:
        train_tf = transforms.Compose([aug_tf, model_tf])

    # Validation/Test use only the model transform
    eval_tf = model_tf

    # Build datasets
    train_ds = ImagePathsDataset(list(zip(X_train, y_train)), transform=train_tf)
    val_ds = ImagePathsDataset(list(zip(X_val, y_val)), transform=eval_tf)
    test_ds = ImagePathsDataset(list(zip(X_test, y_test)), transform=eval_tf)

    # Class weights from training labels
    class_counts, class_weights = _compute_class_weights(y_train, num_classes)

    # Loaders
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=pin_memory, persistent_workers=persistent_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=pin_memory, persistent_workers=persistent_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=pin_memory, persistent_workers=persistent_workers
    )

    idx_to_class: List[str] = [None] * num_classes  # type: ignore[assignment]
    for cls, idx in class_to_idx.items():
        idx_to_class[idx] = cls

    print("\nReal-time dataset setup complete:")
    print(f"  Classes: {classes}")
    print(f"  Train/Val/Test sizes: {len(train_ds)}/{len(val_ds)}/{len(test_ds)}")
    print(f"  Train class counts: {class_counts}")

    return DatasetBundle(
        train_dataset=train_ds,
        val_dataset=val_ds,
        test_dataset=test_ds,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        classes=classes,
        class_to_idx=class_to_idx,
        idx_to_class=idx_to_class,
        num_classes=num_classes,
        class_counts=class_counts,
        class_weights=class_weights,
    )


# ============== Visualization helper (used only in __main__) ==============
def _imshow(inp: torch.Tensor, title: Optional[str] = None) -> None:
    """Quickly display a tensor image or grid with ImageNet un-normalization.

    Note: Intended for debugging/visualization in the __main__ block only.
    """
    # Make sure tensor is on CPU and detached
    inp_np = inp.detach().cpu().numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp_np = std * inp_np + mean
    inp_np = np.clip(inp_np, 0, 1)
    plt.imshow(inp_np)
    if title is not None:
        plt.title(title)
    plt.axis('off')
    plt.pause(0.001)


if __name__ == '__main__':
    # Demo entrypoint: use the new real-time pipeline (no offline augmentation or folder copies)
    BATCH_SIZE: int = 32

    try:
        bundle: DatasetBundle = setup_dataset_realtime(
            source_base_dir='data',
            batch_size=BATCH_SIZE,
        )

        print("\n--- Verification (real-time pipeline) ---")
        print(f"Classes: {bundle.classes}")
        print(f"Num classes: {bundle.num_classes}")
        print(f"Class-to-idx: {bundle.class_to_idx}")
        print(f"Train/Val/Test sizes: {len(bundle.train_dataset)}/{len(bundle.val_dataset)}/{len(bundle.test_dataset)}")
        print(f"Train class counts: {bundle.class_counts}")
        print(f"Class weights: {bundle.class_weights.tolist()}")

        # Fetch one batch from each loader to ensure transforms/dataloader work and show images
        train_images, train_labels = next(iter(bundle.train_loader))
        print("Successfully retrieved one training batch.")
        print(f"  Train image batch shape: {train_images.shape}")
        print(f"  Train label batch shape: {train_labels.shape}")
        grid_train: torch.Tensor = make_grid(train_images[:8], nrow=4)
        _imshow(grid_train, title="Train batch (first 8)")

        val_images, val_labels = next(iter(bundle.val_loader))
        print(f"  Val image batch shape: {val_images.shape}")
        print(f"  Val label batch shape: {val_labels.shape}")
        grid_val: torch.Tensor = make_grid(val_images[:8], nrow=4)
        _imshow(grid_val, title="Val batch (first 8)")

        test_images, test_labels = next(iter(bundle.test_loader))
        print(f"  Test image batch shape: {test_images.shape}")
        print(f"  Test label batch shape: {test_labels.shape}")
        grid_test: torch.Tensor = make_grid(test_images[:8], nrow=4)
        _imshow(grid_test, title="Test batch (first 8)")

    except Exception as e:
        print(f"\nDataset real-time setup failed: {e}")

