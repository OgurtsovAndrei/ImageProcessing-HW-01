import random
from pathlib import Path
from typing import Tuple, Optional, List
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from monai import transforms as MT

# from monai.data import Read
from dataset import SurfaceDataset3D
from config import BATCH_SIZE, NUM_WORKERS

def custom_collate(batch):
    """Custom collate to handle variable size 3D volumes.
    Returns a list of items instead of stacking them, allowing GPU resizing later.
    """
    return batch

class SurfaceDataModule(pl.LightningDataModule):
    """Lightning DataModule for Surface Detection.

    Handles all data loading, splitting, and dataloader creation.
    Updated to use MONAI 3D augmentations on GPU with dynamic resizing.
    """

    def __init__(
        self,
        train_images_dir: Path,
        train_labels_dir: Path,
        volume_shape: Tuple[int, int, int] = (64, 64, 64),
        val_split: float = 0.2,
        batch_size: int = BATCH_SIZE,
        num_workers: int = NUM_WORKERS
    ):
        super().__init__()
        self.train_images_dir = train_images_dir
        self.train_labels_dir = train_labels_dir
        self.volume_shape = volume_shape
        self.val_split = val_split
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Will be set in setup()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        # Define GPU-based augmentations using MONAI
        # 1. Resize to target shape (trilinear for image, nearest for label)
        # 2. Apply Augmentations
        self.gpu_augments = MT.Compose([
            MT.Resized(keys=["image", "label"], spatial_size=self.volume_shape, mode=["trilinear", "nearest"]),
            MT.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            MT.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            MT.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            MT.RandRotated(keys=["image", "label"], range_x=0.1, range_y=0.1, range_z=0.1, prob=0.3, keep_size=True, mode=["bilinear", "nearest"]),
            MT.RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
            MT.RandGaussianNoised(keys=["image"], prob=0.3, mean=0.0, std=0.01),
        ])
        # Validation transforms: Just Resize (for image AND label)
        self.val_augments = MT.Compose([
            MT.Resized(keys=["image", "label"], spatial_size=self.volume_shape, mode=["trilinear", "nearest"])
        ])
        # Validation transforms for Image ONLY (for test set where labels are None)
        self.val_image_augments = MT.Compose([
            MT.Resized(keys=["image"], spatial_size=self.volume_shape, mode=["trilinear"])
        ])

    def setup(self, stage: Optional[str] = None):
        """Setup datasets for different stages."""
        print(f"\nSetting up training data...")

        # Get all available training files (scanning for npy, npz, tif)
        extensions = [".npy", ".npz", ".tif"]
        all_files = []
        for ext in extensions:
             files = sorted([f.name for f in self.train_images_dir.glob(f"*{ext}")])
             all_files.extend(files)

        if not all_files:
            raise RuntimeError(f"No volume files found in {self.train_images_dir}. Check your data path.")

        # Shuffle for random split (deterministic with seed)
        random.seed(42)
        random.shuffle(all_files)
        # Split into train/val
        split_idx = int(len(all_files) * (1 - self.val_split))
        train_files = all_files[:split_idx]
        val_files = all_files[split_idx:]

        print(f"Total files: {len(all_files)}")
        print(f"Train files: {len(train_files)}")
        print(f"Val files: {len(val_files)}")

        # Create train dataset
        self.train_dataset = SurfaceDataset3D(
            images_dir=self.train_images_dir,
            labels_dir=self.train_labels_dir,
            volume_files=train_files,
            volume_shape=self.volume_shape
        )
        # Create validation dataset
        self.val_dataset = SurfaceDataset3D(
            images_dir=self.train_images_dir,
            labels_dir=self.train_labels_dir,
            volume_files=val_files,
            volume_shape=self.volume_shape
        )

    def train_dataloader(self) -> DataLoader:
        """Create train dataloader with custom collate for variable sizes."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=bool(self.num_workers > 0),
            collate_fn=custom_collate
        )

    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader with custom collate for variable sizes."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=bool(self.num_workers > 0),
            collate_fn=custom_collate
        )

    def on_after_batch_transfer(self, batch, dataloader_idx):
        """Apply MONAI GPU-accelerated 3D augmentations to the batch."""
        # If custom_collate is used, batch is a list of tuples [(x, y, id), ...]
        if not isinstance(batch, list):
            return super().on_after_batch_transfer(batch, dataloader_idx)

        x_list, y_list, frag_ids = [], [], []
        # Determine device to ensure we process on GPU
        # self.trainer.strategy.root_device is reliable in Lightning
        device = self.trainer.strategy.root_device if self.trainer else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Select transform
        transforms = self.gpu_augments if self.trainer.training else self.val_augments

        for item in batch:
            x, y, frag_id = item
            
            # Workaround for MPS not supporting float64 which MONAI uses internally for some transforms
            if device.type == "mps":
                 # Ensure data is on CPU for transforms
                 x = x.cpu()
                 y = y.cpu()
                 data = {"image": x, "label": y}
                 # Apply transforms on CPU
                 data = transforms(data)
                 # Move to MPS device
                 x_list.append(data["image"].to(device, non_blocking=True))
                 y_list.append(data["label"].to(device, non_blocking=True))
            else:
                # IMPORTANT: Explicitly move to GPU now.
                # This ensures the Resized and other transforms run on VRAM, avoiding CPU RAM spikes.
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
    
                data = {"image": x, "label": y}
                # Apply transforms (Resize + Augments)
                data = transforms(data)
    
                x_list.append(data["image"])
                y_list.append(data["label"])
            
            frag_ids.append(frag_id)

        # Stack into tensors -> (B, C, D, H, W)
        return torch.stack(x_list), torch.stack(y_list), frag_ids
