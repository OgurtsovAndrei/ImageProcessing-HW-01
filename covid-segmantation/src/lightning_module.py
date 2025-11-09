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


def create_unet_with_densenet121(num_classes: int,
                                 freeze_strategy: FreezeStrategy = FreezeStrategy.PCT70,
                                 device: torch.device = torch.device("cpu")):
    print("Creating 2D Unet model with densenet121 backbone...")
    model = smp.Unet(
        encoder_name="densenet121",
        encoder_weights=None,
        in_channels=1,
        classes=num_classes,
    )

    adapted_weights = adapt_radimagenet_weights(device)
    model.encoder.load_state_dict(adapted_weights)
    print("Successfully loaded RadImageNet weights into 1-channel encoder.")

    if freeze_strategy != FreezeStrategy.NO:
        print(f"Applying freeze strategy: {freeze_strategy.name}")
        classifier_prefixes = ('decoder.', 'segmentation_head.')
        apply_freeze(model, classifier_prefixes, strategy=freeze_strategy)

    return model


def adapt_radimagenet_weights_resnet(device):
    print("Downloading RadImageNet ResNet-50 weights...")
    weights_path = hf_hub_download(
        repo_id="Lab-Rasool/RadImageNet",
        filename="ResNet50.pt"
    )

    model_or_weights = torch.load(weights_path, map_location=torch.device('cpu'))

    if isinstance(model_or_weights, nn.Module):
        state_dict = model_or_weights.state_dict()
    else:
        state_dict = model_or_weights

    first_conv_key = 'backbone.0.weight'

    if first_conv_key not in state_dict:
        print(f"ERROR: Could not find key '{first_conv_key}' in state_dict.")
        print("Available keys:", state_dict.keys())
        raise KeyError(f"Weight adaptation failed: Key '{first_conv_key}' not found.")

    conv1_weights_3ch = state_dict[first_conv_key]
    conv1_weights_1ch = conv1_weights_3ch.sum(dim=1, keepdim=True)
    state_dict[first_conv_key] = conv1_weights_1ch
    print("Weights adapted for 1-channel input (backbone.0.weight).")

    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        new_key = key
        if key.startswith('backbone.0.'):
            new_key = key.replace('backbone.0.', 'conv1.', 1)
        elif key.startswith('backbone.1.'):
            new_key = key.replace('backbone.1.', 'bn1.', 1)
        elif key.startswith('backbone.4.'):
            new_key = key.replace('backbone.4.', 'layer1.', 1)
        elif key.startswith('backbone.5.'):
            new_key = key.replace('backbone.5.', 'layer2.', 1)
        elif key.startswith('backbone.6.'):
            new_key = key.replace('backbone.6.', 'layer3.', 1)
        elif key.startswith('backbone.7.'):
            new_key = key.replace('backbone.7.', 'layer4.', 1)

        new_state_dict[new_key] = value

    print("State_dict keys remapped for smp encoder (e.g., 'backbone.0' -> 'conv1', 'backbone.4' -> 'layer1').")
    return new_state_dict


def create_unet_with_resnet(num_classes: int,
                            freeze_strategy: FreezeStrategy = FreezeStrategy.PCT70,
                            device: torch.device = torch.device("cpu")):
    print("Creating 2D Unet model with resnet50 backbone...")
    model = smp.Unet(
        encoder_name="resnet50",
        encoder_weights=None,
        in_channels=1,
        classes=num_classes,
    )

    adapted_weights = adapt_radimagenet_weights_resnet(device)
    try:
        model.encoder.load_state_dict(adapted_weights)
        print("Successfully loaded RadImageNet weights into 1-channel ResNet-50 encoder.")
    except Exception as e:
        print("\n--- ERROR loading state_dict into ResNet-50 encoder ---")
        print("This often happens if the keys don't match.")
        print(e)
        raise e

    if freeze_strategy != FreezeStrategy.NO:
        print(f"Applying freeze strategy: {freeze_strategy.name}")
        classifier_prefixes = ('decoder.', 'segmentation_head.')
        apply_freeze(model, classifier_prefixes, strategy=freeze_strategy)

    return model

def create_fpn_with_resnet(num_classes: int,
                            freeze_strategy: FreezeStrategy = FreezeStrategy.PCT70,
                            device: torch.device = torch.device("cpu")):
    print("Creating 2D Unet model with resnet50 backbone...")
    model = smp.FPN(
        encoder_name="resnet50",
        encoder_weights=None,
        in_channels=1,
        classes=num_classes,
    )

    adapted_weights = adapt_radimagenet_weights_resnet(device)
    try:
        model.encoder.load_state_dict(adapted_weights)
        print("Successfully loaded RadImageNet weights into 1-channel ResNet-50 encoder.")
    except Exception as e:
        print("\n--- ERROR loading state_dict into ResNet-50 encoder ---")
        print("This often happens if the keys don't match.")
        print(e)
        raise e

    if freeze_strategy != FreezeStrategy.NO:
        print(f"Applying freeze strategy: {freeze_strategy.name}")
        classifier_prefixes = ('decoder.', 'segmentation_head.')
        apply_freeze(model, classifier_prefixes, strategy=freeze_strategy)

    return model


def create_swin_model(num_classes: int,
                      freeze_strategy: FreezeStrategy = FreezeStrategy.PCT70,
                      _device: torch.device = torch.device("cpu")):
    print("Creating 2D Unet model with Swin Transformer backbone...")

    model = smp.Unet(
        encoder_name="tu-swin_tiny_patch4_window7_224",
        encoder_weights="imagenet",
        in_channels=1,
        classes=num_classes,
    )

    print(f"Applying freeze strategy: {freeze_strategy.name}")
    classifier_prefixes = ('decoder.', 'segmentation_head.')
    apply_freeze(model, classifier_prefixes, strategy=freeze_strategy)

    return model

def create_efficientnet_model(num_classes: int,
                      freeze_strategy: FreezeStrategy = FreezeStrategy.NO,
                      _device: torch.device = torch.device("cpu")):
    print("Creating 2D Unet model with efficientnet backbone...")

    model = smp.UnetPlusPlus(
        encoder_name="efficientnet-b7",
        encoder_weights="imagenet",
        in_channels=1,
        classes=num_classes,
    )

    print(f"Applying freeze strategy: {freeze_strategy.name}")
    classifier_prefixes = ('decoder.', 'segmentation_head.')
    apply_freeze(model, classifier_prefixes, strategy=freeze_strategy)

    return model


class CovidSegmenter(pl.LightningModule):
    def __init__(self,
                 num_classes=4,
                 max_lr=1e-3,
                 weight_decay=1e-4,
                 freeze_strategy: FreezeStrategy = FreezeStrategy.PCT70,
                 l1_lambda: float = 1e-5,
                 ):
        super().__init__()
        self.save_hyperparameters()
        self.hparams.l1_lambda = l1_lambda

        self.model = create_efficientnet_model(num_classes, freeze_strategy, self.device)
        self.model.to(self.device)

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

        l1_penalty = 0.0
        for param in self.model.parameters():
            if param.requires_grad:
                l1_penalty += torch.abs(param).sum()

        total_loss = loss + self.hparams.l1_lambda * l1_penalty

        self.log(f'{stage}_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f'{stage}_l1_penalty', l1_penalty, on_step=False, on_epoch=True, logger=True)
        self.log(f'{stage}_total_loss', total_loss, on_step=False, on_epoch=True, logger=True)

        preds = torch.argmax(output, dim=1)

        if stage == 'train':
            self.train_miou.update(preds, mask)
            self.train_acc.update(preds, mask)
            self.log(f'{stage}_miou', self.train_miou, on_step=False, on_epoch=True, prog_bar=True)
            self.log(f'{stage}_acc', self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        else:
            self.val_miou.update(preds, mask)
            self.val_acc.update(preds, mask)
            self.log(f'{stage}_miou', self.val_miou, on_step=False, on_epoch=True, prog_bar=True)
            self.log(f'{stage}_acc', self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

        return total_loss

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

    model = CovidSegmenter(
        num_classes=4,
        max_lr=1e-3,
        weight_decay=1e-4,
        freeze_strategy=FreezeStrategy.NO
    )

    print("\n--- Model initialization successful ---")

    print("\n--- Model Summary (Checking freeze status) ---")
    summary(
        model,
        input_size=(1, 1, 512, 512),
        device=device_str,
        col_names=["output_size", "num_params", "trainable"]
    )
