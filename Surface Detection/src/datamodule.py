import random
from pathlib import Path
from typing import Tuple, Optional, List
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from monai import transforms as MT

import config
from dataset import SurfaceDataset3D
from config import BATCH_SIZE, NUM_WORKERS, DEVICE, SIDE_SIZE


def custom_collate(batch):
    return batch


class SurfaceDataModule(pl.LightningDataModule):
    def __init__(
            self,
            train_images_dir: Path,
            train_labels_dir: Path,
            volume_shape: Tuple[int, int, int] = (64, 64, 64),
            val_split: float = 0.1,
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

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.crop_transforms = MT.Compose([
            MT.RandSpatialCropd(
                keys=["image", "label"],
                roi_size=self.volume_shape,
                random_size=False,
                # use num_samples = ...
            ),
        ])

        self.gpu_augments = MT.Compose([
            MT.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            MT.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            MT.RandRotate90d(keys=["image", "label"], prob=0.75, spatial_axes=(1, 2)),

            # small random rotation / shift / scale (ShiftScaleRotate)
            MT.RandAffined(
                keys=["image", "label"],
                prob=0.1,
            ),

            # brightness / contrast
            MT.RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
            MT.RandAdjustContrastd(keys=["image"], prob=0.5, gamma=(0.9, 1.1)),

            # OneOf noise / blur / motion blur
            MT.OneOf(
                [
                    MT.RandGaussianNoised(keys=["image"], mean=0.0, std=0.05, prob=0.1),
                    MT.RandGaussianSmoothd(keys=["image"], sigma_x=(0.3, 1.2), prob=0.1),
                ],
                weights=[1, 1]
            ),

            # grid distortion
            MT.RandGridDistortiond(
                keys=["image", "label"],
                prob=0.1,
                distort_limit=0.3,
                num_cells=5
            ),

            # coarse dropout
            MT.RandCoarseDropoutd(
                keys=["image"],
                holes=1,
                spatial_size=(SIDE_SIZE // 16, SIDE_SIZE // 16, SIDE_SIZE // 16),
                prob=0.1
            ),
        ])

        self.val_augments = MT.Compose([])
        self.val_image_augments = MT.Compose([])

    def setup(self, stage: Optional[str] = None):
        print(f"\nSetting up training data...")

        extensions = [".npy", ".npz", ".tif"]
        all_files = []
        for ext in extensions:
            files = sorted([f.name for f in self.train_images_dir.glob(f"*{ext}")])
            all_files.extend(files)

        if not all_files:
            raise RuntimeError(f"No volume files found in {self.train_images_dir}. Check your data path.")

        random.seed(42)
        random.shuffle(all_files)
        split_idx = int(len(all_files) * (1 - self.val_split))
        train_files = all_files[:split_idx]
        val_files = all_files[split_idx:]

        print(f"Total files: {len(all_files)}")
        print(f"Train files: {len(train_files)}")
        print(f"Val files: {len(val_files)}")

        self.train_dataset = SurfaceDataset3D(
            images_dir=self.train_images_dir,
            labels_dir=self.train_labels_dir,
            volume_files=train_files,
            volume_shape=self.volume_shape
        )
        self.val_dataset = SurfaceDataset3D(
            images_dir=self.train_images_dir,
            labels_dir=self.train_labels_dir,
            volume_files=val_files,
            volume_shape=self.volume_shape
        )

    def train_dataloader(self) -> DataLoader:
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
        return DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=bool(self.num_workers > 0),
            collate_fn=custom_collate
        )

    def on_after_batch_transfer(self, batch, dataloader_idx):
        if not isinstance(batch, list):
            return super().on_after_batch_transfer(batch, dataloader_idx)

        x_list, y_list, frag_ids = [], [], []

        device = self.trainer.strategy.root_device if self.trainer else torch.device(DEVICE)

        # Выбираем точность
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            target_dtype = torch.bfloat16
        else:
            target_dtype = torch.float32

        aug_transforms = self.gpu_augments if self.trainer.training else self.val_augments

        for item in batch:
            x, y, frag_id = item

            if device.type != "mps":
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
            else:
                x = x.to(device)
                y = y.to(device)

            if self.trainer.training:
                for _ in range(config.SAMPLES_PER_VOLUME):
                    data = {"image": x, "label": y}

                    data = self.crop_transforms(data)

                    if device.type != "mps":
                        data["image"] = data["image"].to(dtype=target_dtype).div_(255.0)
                    else:
                        data["image"] = data["image"].float().div_(255.0)

                    if device.type == "mps":
                        data["label"] = data["label"].float()
                        data["image"] = data["image"].contiguous()
                        data["label"] = data["label"].contiguous()

                        # Fix: Move to CPU for stable augmentation on MPS
                        data["image"] = data["image"].cpu()
                        data["label"] = data["label"].cpu()

                        with torch.no_grad():
                            data = aug_transforms(data)

                        # Move back to MPS
                        data["image"] = data["image"].to(device)
                        data["label"] = data["label"].to(device)
                    else:
                        data["label"] = data["label"].long()
                        data = aug_transforms(data)

                    x_list.append(data["image"])
                    y_list.append(data["label"].long())
                    frag_ids.append(frag_id)
            else:
                if device.type != "mps":
                    x_final = x.to(dtype=target_dtype).div_(255.0)
                else:
                    x_final = x.float().div_(255.0)

                y_final = y.long()

                x_list.append(x_final)
                y_list.append(y_final)
                frag_ids.append(frag_id)

        return torch.stack(x_list), torch.stack(y_list), frag_ids
