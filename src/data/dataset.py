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
from torchvision.datasets.vision import VisionDataset
from torchvision.utils import make_grid
from sklearn.model_selection import train_test_split
from tqdm import tqdm

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
    dataset: VisionDataset,
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

    Caller must pass a prebuilt torchvision VisionDataset via `dataset`. It must expose
    ImageFolder-like attributes (classes, class_to_idx, and either samples or imgs containing
    (path, label) pairs).

    Different models can pass their own transforms via augmentation_transform and model_transform.
    - model_transform is ALWAYS applied to all splits.
    - augmentation_transform is applied ONLY on the training split, BEFORE model_transform.
    """

    # Resolve base dataset (now required)
    base: VisionDataset = dataset

    # Extract metadata
    classes_attr = getattr(base, 'classes', None)
    class_to_idx_attr = getattr(base, 'class_to_idx', None)

    classes: List[str] = list(classes_attr) if classes_attr is not None else []
    class_to_idx: Dict[str, int] = dict(class_to_idx_attr) if class_to_idx_attr is not None else {}

    # Extract samples: prefer `samples`, fallback to legacy `imgs`
    samples = getattr(base, 'samples', None)
    if samples is None:
        samples = getattr(base, 'imgs', None)

    if samples is None:
        # We require file-based datasets. Provide a clear error.
        raise AttributeError(
            "The provided dataset must expose a 'samples' (or legacy 'imgs') attribute with (path, label) pairs."
        )

    all_paths_tuple, all_labels_tuple = zip(*samples) if len(samples) > 0 else ([], [])
    all_paths: List[str] = list(all_paths_tuple)
    all_labels: List[int] = list(all_labels_tuple)

    if len(all_paths) == 0:
        where = getattr(base, 'root', '<unknown>')
        raise RuntimeError(f"No images found in '{where}'. Expected class subfolders with images.")

    # If classes are missing but we have class_to_idx, reconstruct the list in index order
    if not classes and class_to_idx:
        max_idx = max(class_to_idx.values()) if class_to_idx else -1
        tmp_classes: List[Optional[str]] = [None] * (max_idx + 1)
        for cls, idx in class_to_idx.items():
            if idx < len(tmp_classes):
                tmp_classes[idx] = cls
        classes = [c if c is not None else str(i) for i, c in enumerate(tmp_classes)]

    # If still missing, derive minimal metadata from labels
    if not classes:
        num_classes_derived = max(all_labels) + 1 if all_labels else 0
        classes = [str(i) for i in range(num_classes_derived)]
        class_to_idx = {c: i for i, c in enumerate(classes)}

    num_classes: int = len(classes)

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
        ds_demo = datasets.ImageFolder(root='data')
        bundle: DatasetBundle = setup_dataset_realtime(
            dataset=ds_demo,
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

