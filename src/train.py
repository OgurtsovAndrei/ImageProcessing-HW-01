import os
import warnings
from datetime import datetime
from typing import List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib.cbook import STEP_LOOKUP_MAP
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

# PyTorch Lightning + TorchMetrics
import pytorch_lightning as pl
from torchmetrics import Accuracy, F1Score
from torchvision.datasets import STL10, VisionDataset

from src.data.dataset import setup_dataset_realtime, DatasetBundle, get_STL_dataset
from torchvision import datasets

from src.model.freeze_utils import FreezeStrategy
from src.model.resnet import create_resnet_classifier
from src.model.swin import create_swin_classifier
from src.save_res import save_and_plot_history

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

class ClassificationLightningModule(pl.LightningModule):
    """LightningModule wrapper around a standard nn.Module classifier."""

    def __init__(self, base_model: nn.Module, num_classes: int, lr: float, weight_decay: float, l1_lambda: float,
                 class_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.model = base_model
        self.save_hyperparameters(ignore=["base_model"])  # logs lr/decays/etc.

        # Register criterion with optional class weights (moved with module to device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
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
        weights_dir: str,
        num_classes: int,
        class_weights: Optional[torch.Tensor] = None,
) -> None:
    # Note: criterion and optimizer params kept for backward compatibility but handled by Lightning.
    os.makedirs(weights_dir, exist_ok=True)
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Build Lightning module
    lit_module = ClassificationLightningModule(
        base_model=model,
        num_classes=num_classes,
        lr=LR,
        weight_decay=L2_LAMBDA,
        l1_lambda=L1_LAMBDA,
        class_weights=class_weights,
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

MAC_DATA_ROOT = 'data/MacVsNonMac'
STL_DATA_ROOT = 'data/STL10'
WEIGHTS_DIR = 'models/weights'
MODEL_NAME = 'resnet'

NUM_CLASSES = 2
BATCH_SIZE = 64
NUM_EPOCHS = 5
L1_LAMBDA = 1e-6
L2_LAMBDA = 1e-5
LR = 1e-4

if __name__ == '__main__':
    # ds = datasets.ImageFolder(root=MAC_DATA_ROOT)
    # ds: VisionDataset = STL10(root=STP_DATA_ROOT, split="train", download=True)
    ds = get_STL_dataset(STL_DATA_ROOT)

    bundle: DatasetBundle = setup_dataset_realtime(
        dataset=ds,
        batch_size=BATCH_SIZE,
    )

    device = get_device()
    print(f"Selected device: {device}")

    # Derive actual number of classes from data
    effective_num_classes = bundle.num_classes
    model = create_swin_classifier(num_classes=effective_num_classes, freeze=FreezeStrategy.PCT70)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    # Optimizer kept for backward compatibility with train_model signature; Lightning will create its own.
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=L2_LAMBDA)

    train_model(
        model=model,
        model_name=MODEL_NAME,
        train_loader=bundle.train_loader,
        val_loader=bundle.val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=NUM_EPOCHS,
        weights_dir=WEIGHTS_DIR,
        num_classes=effective_num_classes,
        class_weights=bundle.class_weights,
    )
