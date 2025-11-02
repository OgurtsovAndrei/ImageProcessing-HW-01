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
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms, datasets
from torchvision.datasets import STL10
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


class TransformedSubset(Dataset):
    """Wrap a torch.utils.data.Subset and apply transforms to image/target.

    This lets us keep a single base dataset instance while using different
    transforms for train/val/test without mutating the base dataset's transform.
    """

    def __init__(self, subset: Subset, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None):
        self.subset = subset
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx: int):
        image, label = self.subset[idx]
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return image, label


def _extract_labels(dataset: Dataset) -> List[int]:
    """Best-effort extraction of numeric labels from an arbitrary torchvision dataset.

    Tries (in order):
      - dataset.targets
      - dataset.labels
      - dataset.samples / dataset.imgs (list of (path, label))
      - iterate over dataset[i][1]
    """
    # Common attrs
    y = getattr(dataset, 'targets', None)
    if y is None:
        y = getattr(dataset, 'labels', None)
    if y is not None:
        return list(map(int, list(y)))

    # ImageFolder-like
    samples = getattr(dataset, 'samples', None)
    if samples is None:
        samples = getattr(dataset, 'imgs', None)
    if samples is not None:
        return [int(lbl) for _, lbl in samples]

    # Fallback: iterate (may be slower but robust)
    return [int(dataset[i][1]) for i in range(len(dataset))]


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

    Caller must pass a prebuilt torchvision VisionDataset via `dataset`. Works with both
    ImageFolder-like datasets and array-backed datasets (e.g., STL10, CIFAR10). Splits are
    created via torch.utils.data.Subset with stratification by labels.

    Different models can pass their own transforms via augmentation_transform and model_transform.
    - model_transform is ALWAYS applied to all splits.
    - augmentation_transform is applied ONLY on the training split, BEFORE model_transform.
    """

    # Base dataset instance
    base: VisionDataset = dataset
    # Defensive check to catch accidental lists/tuples of indices
    if not isinstance(base, Dataset):
        raise TypeError(
            f"setup_dataset_realtime expects a torch.utils.data.Dataset (e.g., a VisionDataset or a Subset), "
            f"but got {type(base).__name__}. Did you pass a list of indices instead of a Dataset?"
        )

    # Extract labels robustly for stratified split
    all_labels: List[int] = _extract_labels(base)
    if len(all_labels) != len(base):
        raise RuntimeError(
            "Failed to extract labels for all items: "
            f"got {len(all_labels)} labels for dataset of length {len(base)}."
        )

    # If classes are missing but we have class_to_idx, reconstruct the list in index order
    # Unwrap Subset for metadata so we can read underlying dataset's classes/class_to_idx
    meta_src = base
    while isinstance(meta_src, Subset):
        meta_src = meta_src.dataset  # type: ignore[attr-defined]
    classes_attr = getattr(meta_src, 'classes', None)
    class_to_idx_attr = getattr(meta_src, 'class_to_idx', None)
    orig_classes: List[str] = list(classes_attr) if classes_attr is not None else []
    orig_class_to_idx: Dict[str, int] = dict(class_to_idx_attr) if class_to_idx_attr is not None else {}

    if not orig_classes and orig_class_to_idx:
        max_idx = max(orig_class_to_idx.values()) if orig_class_to_idx else -1
        tmp_classes: List[Optional[str]] = [None] * (max_idx + 1)
        for cls, idx in orig_class_to_idx.items():
            if idx < len(tmp_classes):
                tmp_classes[idx] = cls
        orig_classes = [c if c is not None else str(i) for i, c in enumerate(tmp_classes)]

    # Determine which label ids are actually present in the provided dataset (handles Subset cases)
    present_label_ids: List[int] = sorted(set(int(y) for y in all_labels))
    # Map original label ids -> compact [0..K-1]
    label_map: Dict[int, int] = {orig: new for new, orig in enumerate(present_label_ids)}

    # Build classes for the present labels only
    if orig_classes:
        classes: List[str] = [orig_classes[i] if 0 <= i < len(orig_classes) else str(i) for i in present_label_ids]
    else:
        # Fallback to stringified original ids
        classes = [str(i) for i in present_label_ids]
    class_to_idx: Dict[str, int] = {name: i for i, name in enumerate(classes)}

    num_classes: int = len(classes)

    # Stratified splits over indices
    all_indices = list(range(len(base)))
    idx_train_val, idx_test, y_train_val, y_test = train_test_split(
        all_indices, all_labels, test_size=test_size, random_state=random_state, stratify=all_labels
    )
    relative_val_size: float = val_size / (1 - test_size)
    idx_train, idx_val, y_train, y_val = train_test_split(
        idx_train_val, y_train_val, test_size=relative_val_size, random_state=random_state, stratify=y_train_val
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

    # Build Subsets with per-split transforms (without mutating base); remap targets to [0..K-1]
    train_subset = Subset(base, idx_train)
    val_subset = Subset(base, idx_val)
    test_subset = Subset(base, idx_test)

    # Small, fast callable for mapping labels
    def _map_label(y: int) -> int:
        return label_map.get(int(y), int(y))

    train_ds: Dataset = TransformedSubset(train_subset, transform=train_tf, target_transform=_map_label)
    val_ds: Dataset = TransformedSubset(val_subset, transform=eval_tf, target_transform=_map_label)
    test_ds: Dataset = TransformedSubset(test_subset, transform=eval_tf, target_transform=_map_label)

    # Class weights from training labels
    y_train_mapped: List[int] = [label_map[int(y)] for y in y_train]
    class_counts, class_weights = _compute_class_weights(y_train_mapped, num_classes)

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

    # Build idx_to_class directly from the compact class order
    idx_to_class: List[str] = list(classes)

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


def get_STL_dataset(folder = 'data/STL10'):
    ds = STL10(root=folder, split="train", download=True)
    classes = ds.classes
    cat_id, dog_id = classes.index("cat"), classes.index("dog")

    def keep_idx(ds, keep):
        return [i for i in range(len(ds)) if ds[i][1] in keep]

    # Keep only cat and dog samples by wrapping the dataset with a Subset of indices
    idx_keep = keep_idx(ds, (cat_id, dog_id))
    return Subset(ds, idx_keep)


if __name__ == '__main__':
    # Demo entrypoint: use the new real-time pipeline (no offline augmentation or folder copies)
    BATCH_SIZE: int = 32

    try:
        ds_demo = datasets.ImageFolder(root='data/MacVsNonMac')

        ds_2 = get_STL_dataset()

        bundle: DatasetBundle = setup_dataset_realtime(
            dataset=ds_2,
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
