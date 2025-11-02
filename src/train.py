import os
import warnings
from datetime import datetime
from typing import Tuple, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# PyTorch Lightning + TorchMetrics
import pytorch_lightning as pl
from torchmetrics import Accuracy, F1Score

from src.data.dataset import setup_dataset, get_dataloaders
from src.model.dino_vit import create_dino_vit_classifier
from src.model.dino_convnext import create_dino_conv_classifier
from src.model.resnet import create_resnet_classifier

warnings.filterwarnings("ignore", message=".*Redirects are currently not supported in Windows or MacOs.*")


def get_device() -> torch.device:
    if torch.cuda.is_available():
        print("Using CUDA.")
        return torch.device("cuda:0")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("Using Apple MPS.")
        return torch.device("mps")
    print("Using CPU.")
    return torch.device("cpu")

def save_and_plot_history(history: List[dict], model_name: str, run_timestamp: str, weights_dir: str):
    history_df = pd.DataFrame(history)
    csv_filename = f"{model_name}-{run_timestamp}-metrics.csv"
    csv_path = os.path.join(weights_dir, csv_filename)
    history_df.to_csv(csv_path, index=False)
    print(f"Metrics history saved to {csv_path}")

    plt.figure(figsize=(15, 6))

    plt.subplot(1, 2, 1)
    plt.plot(history_df['epoch'], history_df['train_loss'], label='Train Loss')
    plt.plot(history_df['epoch'], history_df['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'{model_name} - Loss vs. Epoch')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history_df['epoch'], history_df['train_acc'], label='Train Accuracy')
    plt.plot(history_df['epoch'], history_df['val_acc'], label='Validation Accuracy')
    plt.plot(history_df['epoch'], history_df['val_f1'], label='Validation F1-score')
    plt.xlabel('Epoch')
    plt.ylabel('Metric')
    plt.legend()
    plt.title(f'{model_name} - Metrics vs. Epoch')
    plt.grid(True)

    plt.suptitle(f'Training Metrics for {model_name} ({run_timestamp})')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plot_filename = f"{model_name}-{run_timestamp}-metrics.png"
    plot_path = os.path.join(weights_dir, plot_filename)
    plt.savefig(plot_path)
    plt.close()
    print(f"Metrics plot saved to {plot_path}")


class ClassificationLightningModule(pl.LightningModule):
    """LightningModule wrapper around a standard nn.Module classifier."""

    def __init__(self, base_model: nn.Module, num_classes: int, lr: float, weight_decay: float, l1_lambda: float):
        super().__init__()
        self.model = base_model
        self.save_hyperparameters(ignore=["base_model"])  # logs lr/decays/etc.

        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr
        self.weight_decay = weight_decay
        self.l1_lambda = l1_lambda

        # Metrics
        self.train_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.val_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.val_f1 = F1Score(task='multiclass', num_classes=num_classes, average='macro')

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)
        if self.l1_lambda > 0:
            l1_loss = sum(param.abs().sum() for param in self.model.parameters())
            loss = loss + self.l1_lambda * l1_loss
        preds = torch.argmax(outputs, dim=1)
        self.train_acc.update(preds, labels)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def on_train_epoch_end(self):
        acc = self.train_acc.compute()
        self.log('train_acc', acc, prog_bar=True, logger=True)
        self.train_acc.reset()

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        self.val_acc.update(preds, labels)
        self.val_f1.update(preds, labels)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def on_validation_epoch_end(self):
        acc = self.val_acc.compute()
        f1 = self.val_f1.compute()
        self.log('val_acc', acc, prog_bar=True, logger=True)
        self.log('val_f1', f1, prog_bar=True, logger=True)
        self.val_acc.reset()
        self.val_f1.reset()

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer


class HistoryLogger(pl.Callback):
    """Collects per-epoch metrics for CSV/plotting to keep previous behavior."""

    def __init__(self):
        super().__init__()
        self.history: List[dict] = []

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        m = trainer.callback_metrics
        # Handle possible naming differences across Lightning versions
        train_loss = m.get('train_loss_epoch') or m.get('train_loss')
        val_loss = m.get('val_loss')
        val_acc = m.get('val_acc')
        val_f1 = m.get('val_f1')
        train_acc = m.get('train_acc')

        def _to_float(x: Optional[torch.Tensor]):
            if x is None:
                return None
            return x.item() if isinstance(x, torch.Tensor) else float(x)

        row = {
            'epoch': trainer.current_epoch + 1,
            'train_loss': _to_float(train_loss) if train_loss is not None else None,
            'train_acc': _to_float(train_acc) if train_acc is not None else None,
            'val_loss': _to_float(val_loss) if val_loss is not None else None,
            'val_acc': _to_float(val_acc) if val_acc is not None else None,
            'val_f1': _to_float(val_f1) if val_f1 is not None else None,
        }
        self.history.append(row)


class BestF1Saver(pl.Callback):
    """Saves best model .pth (state_dict) in the legacy naming format when val_f1 improves."""

    def __init__(self, model_name: str, weights_dir: str):
        super().__init__()
        self.best_val_f1: float = 0.0
        self.model_name = model_name
        self.weights_dir = weights_dir
        os.makedirs(self.weights_dir, exist_ok=True)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        m = trainer.callback_metrics
        f1 = m.get('val_f1')
        if f1 is None:
            return
        if isinstance(f1, torch.Tensor):
            f1 = f1.item()
        if f1 > self.best_val_f1:
            self.best_val_f1 = f1
            model_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_path = os.path.join(self.weights_dir, f"{self.model_name}-{model_timestamp}-f1_{self.best_val_f1:.4f}.pth")
            torch.save(pl_module.model.state_dict(), save_path)
            print(f"New best model saved to {save_path} with F1-score: {self.best_val_f1:.4f}")


def train_model(
        model: nn.Module,
        model_name: str,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device,
        num_epochs: int,
        weights_dir: str
) -> None:
    # Note: criterion and optimizer params kept for backward compatibility but handled by Lightning.
    os.makedirs(weights_dir, exist_ok=True)
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Build Lightning module
    lit_module = ClassificationLightningModule(
        base_model=model,
        num_classes=NUM_CLASSES,
        lr=LR,
        weight_decay=L2_LAMBDA,
        l1_lambda=L1_LAMBDA,
    )

    # Callbacks to preserve legacy behavior
    history_cb = HistoryLogger()
    best_saver_cb = BestF1Saver(model_name=model_name, weights_dir=weights_dir)

    # Map device to Lightning accelerator
    if device.type == 'cuda':
        accelerator = 'gpu'
    elif device.type == 'mps':
        accelerator = 'mps'
    else:
        accelerator = 'cpu'

    trainer = pl.Trainer(
        max_epochs=num_epochs,
        accelerator=accelerator,
        devices=1,
        log_every_n_steps=10,
        enable_progress_bar=True,
        callbacks=[history_cb, best_saver_cb],
    )

    print("\nStarting training with PyTorch Lightning...")
    trainer.fit(lit_module, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print("\nTraining complete!")

    # Filter out epochs that might have None values if early epoch logs were missing
    history = [row for row in history_cb.history if all(k in row for k in ['train_loss','train_acc','val_loss','val_acc','val_f1'])]

    save_and_plot_history(
        history=history,
        model_name=model_name,
        run_timestamp=run_timestamp,
        weights_dir=weights_dir
    )


SOURCE_DIRS = ['mac-merged', 'laptops-merged']
PROCESSED_DIR = 'data/processed'
WEIGHTS_DIR = 'models/weights'
MODEL_NAME = 'convnext'

NUM_CLASSES = 2
BATCH_SIZE = 32
NUM_EPOCHS = 50
TARGET_AUG_COUNT = 1024
L1_LAMBDA = 1e-6
L2_LAMBDA = 1e-5
LR = 1e-4

if __name__ == '__main__':
    setup_dataset(
        source_base_dir='data',
        processed_base_dir=PROCESSED_DIR,
        source_class_names=SOURCE_DIRS,
        target_aug_count=TARGET_AUG_COUNT
    )

    train_loader, val_loader, test_loader = get_dataloaders(
        processed_dir=PROCESSED_DIR,
        batch_size=BATCH_SIZE
    )

    device = get_device()
    print(f"Selected device: {device}")

    model = create_resnet_classifier(num_classes=NUM_CLASSES)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    # Optimizer kept for backward compatibility with train_model signature; Lightning will create its own.
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=L2_LAMBDA)

    train_model(
        model=model,
        model_name=MODEL_NAME,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=NUM_EPOCHS,
        weights_dir=WEIGHTS_DIR
    )
