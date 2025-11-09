import os

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import scipy
import scipy.ndimage
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from torch import nn
from torchvision import transforms as T

from data import visualize
from freeze_utils import FreezeStrategy
from lightning_datamodule import CovidDataModule
from lightning_module import CovidSegmenter
from plot import plot_loss, plot_score, plot_acc, generate_plots_from_logs
from test import run_test_predictions, run_val_tta_evaluation

SOURCE_SIZE: int = 512
TARGET_SIZE: int = SOURCE_SIZE
MAX_LR: float = 5e-5
WEIGHT_DECAY: float = 1e-5
L1_REG: float = 1e-6
BATCH_SIZE: int = 16


EPOCHS: int = 70
PATIENCE: int = 20

FINETUNE_EPOCHS: int = 100
FINETUNE_PATIENCE: int = 30

if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    datamodule = CovidDataModule(
        batch_size=BATCH_SIZE,
        source_size=SOURCE_SIZE,
        target_size=TARGET_SIZE,
        use_radiopedia=True
    )

    model = CovidSegmenter(
        num_classes=4,
        max_lr=MAX_LR,
        weight_decay=WEIGHT_DECAY,
        freeze_strategy=FreezeStrategy.NO,
        l1_lambda=L1_REG,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor='val_miou',
        dirpath='checkpoints',
        filename='best_model-{epoch:02d}-{val_miou:.3f}',
        save_top_k=1,
        mode='max',
    )

    early_stopping_callback = EarlyStopping(
        monitor='val_miou',
        patience=PATIENCE,
        mode='max'
    )

    csv_logger = CSVLogger(save_dir="logs/")

    trainer = pl.Trainer(
        accelerator='mps',
        devices=1,
        max_epochs=EPOCHS,
        callbacks=[checkpoint_callback],  # early_stopping_callback,
        logger=csv_logger,
        enable_progress_bar=True,
    )

    print("Starting PyTorch Lightning training...")
    print(f"Using full dataset with radiopedia")
    trainer.fit(model, datamodule=datamodule)

    print("Starting finetune on medseg...")
    datamodule = CovidDataModule(
        batch_size=BATCH_SIZE,
        source_size=SOURCE_SIZE,
        target_size=TARGET_SIZE,
        use_radiopedia=False
    )

    finetune_early_stopping_callback = EarlyStopping(
        monitor='val_miou',
        patience=FINETUNE_PATIENCE,
        mode='max',
        verbose=True
    )

    finetune_trainer = pl.Trainer(
        accelerator='mps',
        devices=1,
        max_epochs=FINETUNE_EPOCHS,
        callbacks=[checkpoint_callback],
        logger=csv_logger,
        enable_progress_bar=True,
    )

    finetune_trainer.fit(model, datamodule=datamodule)

    print("Training complete.")
    print(f"Best model saved at: {checkpoint_callback.best_model_path}")

    log_dir = csv_logger.experiment.log_dir
    if log_dir:
        generate_plots_from_logs(log_dir)
    else:
        print("Could not find log directory, skipping plot generation.")

    # Run TTA on validation set and show its result for comparison with last-epoch val metrics
    try:
        tta_val_miou = run_val_tta_evaluation(checkpoint_callback, datamodule, device, num_classes=4, visualize_samples=8)
        if tta_val_miou is not None:
            print(f"Validation TTA mIoU (macro): {tta_val_miou:.4f}")
    except Exception as e:
        print(f"TTA validation evaluation failed: {e}")

    run_test_predictions(checkpoint_callback, datamodule, device, TARGET_SIZE, checkpoint_callback.best_model_score.item())
