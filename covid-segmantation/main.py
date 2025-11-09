import os

import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger

from lightning_module import CovidSegmenter
from lightning_datamodule import CovidDataModule
from freeze_utils import FreezeStrategy
from plot import plot_loss, plot_score, plot_acc

SOURCE_SIZE: int = 512
TARGET_SIZE: int = 256
MAX_LR: float = 1e-3
EPOCHS: int = 20
WEIGHT_DECAY: float = 1e-4
BATCH_SIZE: int = 32


def generate_plots_from_logs(log_dir: str):
    print(f"Attempting to generate plots from logs in: {log_dir}")
    try:
        metrics_path = os.path.join(log_dir, "metrics.csv")
        if not os.path.exists(metrics_path):
            print(f"Error: Could not find metrics.csv at {metrics_path}")
            return

        metrics_df = pd.read_csv(metrics_path)

        epoch_metrics = metrics_df.drop_duplicates(subset='epoch', keep='last')

        history = {
            'val_loss': epoch_metrics['val_loss'].dropna().tolist(),
            'train_loss': epoch_metrics['train_loss'].dropna().tolist(),
            'val_miou': epoch_metrics['val_miou'].dropna().tolist(),
            'train_miou': epoch_metrics['train_miou'].dropna().tolist(),
            'val_acc': epoch_metrics['val_acc'].dropna().tolist(),
            'train_acc': epoch_metrics['train_acc'].dropna().tolist()
        }

        print("Generating loss plot...")
        plot_loss(history)
        print("Generating mIoU score plot...")
        plot_score(history)
        print("Generating accuracy plot...")
        plot_acc(history)
        print("Plots generated successfully.")

    except Exception as e:
        print(f"Error generating plots: {e}")
        print("Please check if 'val_loss', 'train_loss', 'val_miou', etc. exist in your metrics.csv")


if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')

    datamodule = CovidDataModule(
        batch_size=BATCH_SIZE,
        source_size=SOURCE_SIZE,
        target_size=TARGET_SIZE
    )

    model = CovidSegmenter(
        num_classes=4,
        max_lr=MAX_LR,
        weight_decay=WEIGHT_DECAY,
        freeze_strategy=FreezeStrategy.PCT70
    )

    checkpoint_callback = ModelCheckpoint(
        monitor='val_miou',
        dirpath='checkpoints',
        filename='best_model-{epoch:02d}-{val_miou:.3f}',
        save_top_k=1,
        mode='max',
    )

    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=7,
        mode='min'
    )

    csv_logger = CSVLogger(save_dir="logs/")

    trainer = pl.Trainer(
        accelerator='mps',
        devices=1,
        max_epochs=EPOCHS,
        callbacks=[checkpoint_callback, early_stopping_callback],
        logger=csv_logger
    )

    print("Starting PyTorch Lightning training...")
    trainer.fit(model, datamodule=datamodule)

    print("Training complete.")
    print(f"Best model saved at: {checkpoint_callback.best_model_path}")

    log_dir = csv_logger.experiment.log_dir
    generate_plots_from_logs(log_dir)