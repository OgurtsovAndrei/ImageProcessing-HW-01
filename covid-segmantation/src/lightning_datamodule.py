import pytorch_lightning as pl
from torch.utils.data import DataLoader
import albumentations
import cv2
import os
import albumentations as A
import matplotlib.pyplot as plt
import numpy as np

from data import prepare_data
from dataset import Dataset


class CovidDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=24, source_size=512, target_size=256, use_radiopedia: bool = False):
        super().__init__()
        self.train_batch_size = batch_size
        self.source_size = source_size
        self.target_size = target_size
        self.val_batch_size = batch_size
        self.use_radiopedia = use_radiopedia
        # Will be populated in setup() based on training masks
        self.class_weights = None
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
        ) = prepare_data(self.use_radiopedia)
        self.train_augs = albumentations.Compose([
            A.CropNonEmptyMaskIfExists(
                height=self.target_size,
                width=self.target_size,
                p=1.0
            ),
            albumentations.Rotate(limit=360, p=0.9, border_mode=cv2.BORDER_REPLICATE),
            # albumentations.RandomSizedCrop(
            #     (int(self.source_size * 0.75), self.source_size),
            #     (self.target_size, self.target_size),
            #     interpolation=cv2.INTER_NEAREST
            # ),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.RandomBrightnessContrast(p=0.5),
            A.GaussNoise(p=0.5),

            # A.OneOf([
            #     A.GaussNoise(p=0.5),
            #     A.CoarseDropout(
            #         num_holes_range=(1, 8),
            #         hole_height_range=(int(self.target_size * 0.05), int(self.target_size * 0.1)),
            #         hole_width_range=(int(self.target_size * 0.05), int(self.target_size * 0.1)),
            #         fill=0,
            #         fill_mask=0,
            #         p=0.5
            #     )
            # ], 0.9),
        ])

        self.val_augs = albumentations.Compose([
            albumentations.Resize(self.target_size, self.target_size, interpolation=cv2.INTER_NEAREST)
        ])

        if stage == 'fit' or stage is None:
            self.train_dataset = Dataset(self.train_images, self.train_masks, self.train_augs)
            self.val_dataset = Dataset(self.val_images, self.val_masks, self.val_augs)

        # Compute class weights from train masks for imbalance handling
        try:
            flat = self.train_masks.reshape(-1).astype(np.int64)
            # Ensure labels start from 0 and are integers
            max_label = int(flat.max()) if flat.size > 0 else 0
            counts = np.bincount(flat, minlength=max_label + 1).astype(np.float64)
            counts[counts == 0] = 1.0  # avoid division by zero for missing classes
            inv_freq = counts.sum() / counts
            # Normalize weights to sum to number of classes (stable scale)
            inv_freq *= (len(inv_freq) / inv_freq.sum())
            self.class_weights = inv_freq.tolist()
            print(f"[DataModule] Computed class weights: {[round(x, 4) for x in self.class_weights]}")
        except Exception as e:
            print(f"[DataModule] Could not compute class weights automatically: {e}")

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

    def plot_sample_batch(self):
        batch = next(iter(self.train_dataloader()))
        images, masks = batch

        plt.figure(figsize=(16, 8))
        for i in range(min(4, len(images))):
            plt.subplot(2, 4, i + 1)
            plt.imshow(images[i][0].numpy(), cmap='gray')
            plt.title(f'Image {i + 1}')
            plt.axis('off')

            plt.subplot(2, 4, i + 5)
            # FIX: Removed the [0] index from masks[i]
            plt.imshow(masks[i].numpy(), cmap='gray')
            plt.title(f'Mask {i + 1}')
            plt.axis('off')
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    data = CovidDataModule(
        batch_size=32,
        source_size=512,
        target_size=512,
        use_radiopedia=False
    )
    data.setup("fit")
    data.plot_sample_batch()
