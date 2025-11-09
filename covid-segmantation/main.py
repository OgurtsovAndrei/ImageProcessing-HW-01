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
from plot import plot_loss, plot_score, plot_acc

SOURCE_SIZE: int = 512
TARGET_SIZE: int = 256
MAX_LR: float = 1e-3
EPOCHS: int = 50
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

        epoch_metrics = metrics_df.groupby('epoch').mean()

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


def predict_single_image(model, image_np, device, val_augs):
    model.eval()

    image_aug = val_augs(image=image_np)['image']

    mean = [0.485]
    std = [0.229]
    transform = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    image_t = transform(image_aug)

    image_t = image_t.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_t)
        output = nn.Softmax(dim=1)(output)
        output = output.permute(0, 2, 3, 1)

    return output.squeeze(0).cpu().numpy()


def run_test_predictions(checkpoint_callback, datamodule, device, target_size):
    print("\nStarting test predictions...")

    best_model_path = checkpoint_callback.best_model_path
    if not best_model_path or not os.path.exists(best_model_path):
        print("No best model checkpoint found. Skipping test predictions.")
        return

    print(f"Loading best model from: {best_model_path}")
    best_model = CovidSegmenter.load_from_checkpoint(best_model_path)
    best_model.to(device)
    best_model.eval()

    datamodule.setup('fit')
    test_images = datamodule.test_images
    val_augs = datamodule.val_augs

    image_batch = np.stack([val_augs(image=img)['image'] for img in test_images], axis=0)

    print(f"Test image batch shape (after augs): {image_batch.shape}")

    output = np.zeros((len(test_images), target_size, target_size, 4))
    for i in range(len(test_images)):
        output[i] = predict_single_image(best_model, image_batch[i], device, val_augs)

    print(f"Output prediction shape: {output.shape}")
    test_masks_prediction = output > 0.5
    visualize(image_batch, test_masks_prediction, num_samples=len(test_images))

    print("Resizing test predictions to original size...")
    test_masks_prediction_original_size = scipy.ndimage.zoom(test_masks_prediction[..., :-2], (1, 2, 2, 1), order=0)
    print(f"Resized predictions shape: {test_masks_prediction_original_size.shape}")

    print("Creating submission file (sub.csv)...")
    frame = pd.DataFrame(
        data=np.stack(
            (np.arange(len(test_masks_prediction_original_size.ravel())),
             test_masks_prediction_original_size.ravel().astype(int)),
            axis=-1
        ),
        columns=['Id', 'Predicted']
    ) .set_index('Id')
    frame.to_csv('sub.csv')
    print("Submission file created successfully.")


if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

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
        monitor='val_miou',
        patience=10,
        mode='max'
    )

    csv_logger = CSVLogger(save_dir="logs/")

    trainer = pl.Trainer(
        accelerator='mps',
        devices=1,
        max_epochs=EPOCHS,
        callbacks=[checkpoint_callback, early_stopping_callback],
        logger=csv_logger,
        enable_progress_bar=True,
    )

    print("Starting PyTorch Lightning training...")
    trainer.fit(model, datamodule=datamodule)

    print("Training complete.")
    print(f"Best model saved at: {checkpoint_callback.best_model_path}")

    log_dir = csv_logger.experiment.log_dir
    if log_dir:
        generate_plots_from_logs(log_dir)
    else:
        print("Could not find log directory, skipping plot generation.")

    run_test_predictions(checkpoint_callback, datamodule, device, TARGET_SIZE)