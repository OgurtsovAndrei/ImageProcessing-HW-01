import pytorch_lightning as pl
from torch.utils.data import DataLoader
import albumentations
import cv2
import os

from data import prepare_data
from dataset import Dataset


class CovidDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=24, source_size=512, target_size=256):
        super().__init__()
        self.train_batch_size = batch_size
        self.source_size = source_size
        self.target_size = target_size
        self.val_batch_size = batch_size
        try:
            self.num_workers = os.cpu_count() // 2
        except Exception:
            self.num_workers = 4

    def setup(self, stage=None):
        (
            self.train_images,
            self.train_masks,
            self.val_images,
            self.val_masks,
            self.test_images,
        ) = prepare_data()

        self.train_augs = albumentations.Compose([
            albumentations.Rotate(limit=360, p=0.9, border_mode=cv2.BORDER_REPLICATE),
            albumentations.RandomSizedCrop(
                (int(self.source_size * 0.75), self.source_size),
                (self.target_size, self.target_size),
                interpolation=cv2.INTER_NEAREST
            ),
            albumentations.HorizontalFlip(p=0.5),
        ])

        self.val_augs = albumentations.Compose([
            albumentations.Resize(self.target_size, self.target_size, interpolation=cv2.INTER_NEAREST)
        ])

        if stage == 'fit' or stage is None:
            self.train_dataset = Dataset(self.train_images, self.train_masks, self.train_augs)
            self.val_dataset = Dataset(self.val_images, self.val_masks, self.val_augs)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True
        )