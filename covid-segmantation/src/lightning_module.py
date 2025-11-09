import torch
import torch.nn as nn
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from huggingface_hub import hf_hub_download
import torchmetrics
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from collections import OrderedDict

from torchinfo import summary

from freeze_utils import apply_freeze, FreezeStrategy


def adapt_radimagenet_weights(device):
    print("Downloading RadImageNet DenseNet121 weights...")
    weights_path = hf_hub_download(
        repo_id="Lab-Rasool/RadImageNet",
        filename="DenseNet121.pt"
    )

    model_or_weights = torch.load(weights_path, map_location=torch.device('cpu'))

    if isinstance(model_or_weights, nn.Module):
        state_dict = model_or_weights.state_dict()
    else:
        state_dict = model_or_weights

    conv1_weights_3ch = state_dict['backbone.0.conv0.weight']
    conv1_weights_1ch = conv1_weights_3ch.sum(dim=1, keepdim=True)
    state_dict['backbone.0.conv0.weight'] = conv1_weights_1ch
    print("Weights adapted for 1-channel input.")

    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith('backbone.0.'):
            new_key = key.replace('backbone.0.', 'features.', 1)
            new_state_dict[new_key] = value

    print("State_dict keys renamed from 'backbone.0.*' to 'features.*'")
    return new_state_dict


class CovidSegmenter(pl.LightningModule):
    def __init__(self, num_classes=4, max_lr=1e-3, weight_decay=1e-4,
                 freeze_strategy: FreezeStrategy = FreezeStrategy.PCT70):
        super().__init__()
        self.save_hyperparameters()

        print("Creating 2D Unet model with densenet121 backbone...")
        self.model = smp.Unet(
            encoder_name="densenet121",
            encoder_weights=None,
            in_channels=1,
            classes=num_classes,
        )

        try:
            adapted_weights = adapt_radimagenet_weights(self.device)
            self.model.encoder.load_state_dict(adapted_weights)
            print("Successfully loaded RadImageNet weights into 1-channel encoder.")

        except Exception as e:
            print(f"Error loading RadImageNet weights: {e}")
            print("Proceeding with randomly initialized weights.")

        if self.hparams.freeze_strategy != FreezeStrategy.NO:
            print(f"Applying freeze strategy: {self.hparams.freeze_strategy.name}")
            classifier_prefixes = ('decoder.', 'segmentation_head.')
            apply_freeze(self.model, classifier_prefixes, strategy=self.hparams.freeze_strategy)

        self.criterion = nn.CrossEntropyLoss()

        self.train_miou = torchmetrics.JaccardIndex(task="multiclass", num_classes=num_classes, average='macro')
        self.val_miou = torchmetrics.JaccardIndex(task="multiclass", num_classes=num_classes, average='macro')

        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, average='macro')
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, average='macro')

    def forward(self, x):
        return self.model(x)

    def _common_step(self, batch, batch_idx, stage):
        image, mask = batch
        output = self(image)
        loss = self.criterion(output, mask)

        self.log(f'{stage}_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        preds = torch.argmax(output, dim=1)
        
        self.val_miou.update(preds, mask)
        self.val_acc.update(preds, mask)
        self.log(f'{stage}_miou', self.val_miou, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f'{stage}_acc', self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, 'val')

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.max_lr, weight_decay=self.hparams.weight_decay)

        if self.trainer and self.trainer.datamodule:
            try:
                total_steps = len(self.trainer.datamodule.train_dataloader()) * self.trainer.max_epochs
            except Exception:
                total_steps = 1000 * self.trainer.max_epochs
        else:
            total_steps = 1000 * self.trainer.max_epochs if self.trainer else 1000

        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.hparams.max_lr,
            total_steps=total_steps
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }


if __name__ == '__main__':

    device_str = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device_str}")

    print("\n--- Attempting to initialize CovidSegmenter ---")

    try:
        model = CovidSegmenter(
            num_classes=4,
            max_lr=1e-3,
            weight_decay=1e-4,
            freeze_strategy=FreezeStrategy.PCT70
        )

        print("\n--- Model initialization successful ---")

        print("\n--- Model Summary (Checking freeze status) ---")
        summary(
            model,
            input_size=(1, 1, 256, 256),
            device=device_str,
            col_names=["output_size", "num_params", "trainable"]
        )

    except Exception as e:
        print(f"\n--- ERROR during model initialization ---")
        print(e)
        print("Model check failed.")