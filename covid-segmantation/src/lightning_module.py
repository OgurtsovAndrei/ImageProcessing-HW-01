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
        encoder_name="efficientnet-b2",
        encoder_weights="imagenet",
        in_channels=1,
        classes=num_classes,
    )

    print(f"Applying freeze strategy: {freeze_strategy.name}")
    classifier_prefixes = ('decoder.', 'segmentation_head.')
    apply_freeze(model, classifier_prefixes, strategy=freeze_strategy)

    return model

def create_mask2former_model(num_classes: int,
                      freeze_strategy: FreezeStrategy = FreezeStrategy.NO,
                      _device: torch.device = torch.device("cpu")):

    print("Creating Mask2Former (transformers) model for semantic segmentation...")

    try:
        from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
    except Exception as e:  # pragma: no cover - provide clear guidance
        raise ImportError(
            "Mask2Former requires the 'transformers' package. Please install it, e.g. `pip install transformers`"
        ) from e

    class _Mask2FormerSemanticWrapper(nn.Module):

        def __init__(self, checkpoint: str, num_classes: int):
            super().__init__()
            self.image_processor = AutoImageProcessor.from_pretrained(checkpoint)
            self.model = Mask2FormerForUniversalSegmentation.from_pretrained(checkpoint)
            self._reinit_num_classes(num_classes)
            self.num_classes = num_classes

            mean = torch.tensor(self.image_processor.image_mean, dtype=torch.float32).view(1, -1, 1, 1)
            std = torch.tensor(self.image_processor.image_std, dtype=torch.float32).view(1, -1, 1, 1)
            self.register_buffer("_img_mean", mean, persistent=False)
            self.register_buffer("_img_std", std, persistent=False)

        def _reinit_num_classes(self, num_classes: int) -> None:
            try:
                self.model.config.num_labels = num_classes
            except Exception:
                pass

            try:
                in_features = self.model.class_predictor.in_features
            except Exception:
                in_features = 256  # sensible default for most Mask2Former variants
            new_cls = nn.Linear(in_features, num_classes)
            nn.init.normal_(new_cls.weight, std=0.01)
            nn.init.zeros_(new_cls.bias)
            self.model.class_predictor = new_cls

            try:
                dec_class_embed = getattr(self.model, "decoder", None)
                if dec_class_embed is not None and hasattr(dec_class_embed, "class_embed"):
                    ce = dec_class_embed.class_embed
                    if isinstance(ce, nn.ModuleList):
                        for i in range(len(ce)):
                            in_f = ce[i].in_features
                            ce[i] = nn.Linear(in_f, num_classes)
                            nn.init.normal_(ce[i].weight, std=0.01)
                            nn.init.zeros_(ce[i].bias)
                    elif isinstance(ce, nn.Linear):
                        in_f = ce.in_features
                        new_ce = nn.Linear(in_f, num_classes)
                        nn.init.normal_(new_ce.weight, std=0.01)
                        nn.init.zeros_(new_ce.bias)
                        dec_class_embed.class_embed = new_ce
            except Exception:
                pass

            try:
                print(f"[Mask2Former] Reinitialized classification head and decoder for {num_classes} classes (no 'no-object'; background included).")
            except Exception:
                pass

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            b, c, h, w = x.shape
            if c == 1:
                x = x.repeat(1, 3, 1, 1)
            elif c != 3:
                if c > 3:
                    x = x[:, :3]
                else:
                    x = x.repeat(1, (3 + c - 1) // c, 1, 1)[:, :3]

            x = (x - self._img_mean) / (self._img_std + 1e-6)

            pixel_mask = torch.ones((b, h, w), dtype=torch.long, device=x.device)

            outputs = self.model(pixel_values=x, pixel_mask=pixel_mask)

            class_logits = outputs.class_queries_logits  # raw logits per query
            mask_logits = outputs.masks_queries_logits   # raw logits per query mask

            class_probs = class_logits.softmax(dim=-1)  # [B, Q, C]
            mask_probs = mask_logits.sigmoid()  # [B, Q, Hm, Wm]

            score_maps = torch.einsum("bqc,bqhw->bchw", class_probs, mask_probs)

            score_maps = torch.nn.functional.interpolate(
                score_maps, size=(h, w), mode="bilinear", align_corners=False
            )

            return score_maps

    checkpoint = "facebook/mask2former-swin-small-ade-semantic"
    model = _Mask2FormerSemanticWrapper(checkpoint=checkpoint, num_classes=num_classes)

    if freeze_strategy != FreezeStrategy.NO:
        print(f"Applying freeze strategy: {freeze_strategy.name}")
        classifier_prefixes = (
            'model.class_predictor',
            'model.decoder.class_embed',
            'model.mask_embed',
        )
        apply_freeze(model, classifier_prefixes, strategy=freeze_strategy)

    model.to(_device)
    return model

class CovidSegmenter(pl.LightningModule):
    def __init__(self,
                 num_classes=4,
                 max_lr=1e-3,
                 weight_decay=1e-4,
                 freeze_strategy: FreezeStrategy = FreezeStrategy.PCT70,
                 l1_lambda: float = 1e-5,
                 class_weights=None,
                 ):
        super().__init__()
        self.save_hyperparameters()
        self.hparams.l1_lambda = l1_lambda
        # Keep a python-list for hparams serialization friendliness; convert to tensor later
        self._class_weights_list = (
            class_weights.tolist() if hasattr(class_weights, 'tolist') else class_weights
        )

        # self.model = create_unet_with_resnet(num_classes, freeze_strategy, self.device)
        self.model = create_mask2former_model(num_classes, freeze_strategy, self.device)
        self.model.to(self.device)

        # Initialize criterion without weights; will (optionally) be updated in on_fit_start
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
        # Safety check: ensure the model outputs exactly num_classes channels (background included)
        if output.shape[1] != self.hparams.num_classes:
            raise RuntimeError(
                f"Model returned {output.shape[1]} channels but num_classes={self.hparams.num_classes}. "
                f"Ensure the segmentation head is configured without a 'no-object' channel."
            )
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

    def on_fit_start(self) -> None:
        """Attach class weights to the loss if provided or available from the DataModule.

        Note: AdamW optimizer itself does not support class weights; class imbalance should be
        handled in the loss function (e.g., CrossEntropyLoss(weight=...)). This hook retrieves
        weights either from the constructor argument or from the DataModule (if it exposes
        `class_weights`).
        """
        weights_tensor = None

        # 1) From constructor
        if self._class_weights_list is not None:
            try:
                weights_tensor = torch.as_tensor(self._class_weights_list, dtype=torch.float32, device=self.device)
            except Exception:
                # Fallback to CPU then move
                weights_tensor = torch.as_tensor(self._class_weights_list, dtype=torch.float32)
        # 2) From DataModule if not passed explicitly
        elif self.trainer is not None and getattr(self.trainer, 'datamodule', None) is not None:
            dm = self.trainer.datamodule
            if hasattr(dm, 'class_weights') and dm.class_weights is not None:
                cw = dm.class_weights
                # Accept list, numpy array, or tensor
                try:
                    weights_tensor = torch.as_tensor(cw, dtype=torch.float32, device=self.device)
                except Exception:
                    weights_tensor = torch.as_tensor(cw, dtype=torch.float32)

        if weights_tensor is not None:
            # Rebuild criterion with weights on the correct device
            self.criterion = nn.CrossEntropyLoss(weight=weights_tensor)
            try:
                # Helpful console output for verification
                print(f"Using class weights for loss: {weights_tensor.detach().cpu().numpy().round(4).tolist()}")
            except Exception:
                pass
        else:
            print("No class weights provided; using unweighted CrossEntropyLoss.")

    # --- Robust checkpoint loading for resume training ---
    def load_state_dict(self, state_dict, strict: bool = True):
        """
        Lightning restores checkpoints with strict=True by default when resuming
        (Trainer.fit(..., ckpt_path=...)). Some earlier checkpoints may contain
        a key like 'criterion.weight' (when the loss had class weights), while
        the current model's loss may not have this buffer registered. In such
        cases PyTorch would raise an "Unexpected key(s)" error.

        To make resume robust without relaxing strictness for all weights, we
        precisely drop 'criterion.weight' from the incoming state_dict if the
        current module does not expect it.
        """
        try:
            current_keys = set(super().state_dict().keys())
            if 'criterion.weight' in state_dict and 'criterion.weight' not in current_keys:
                # Drop only the offending key; keep strict for everything else
                state_dict = {k: v for k, v in state_dict.items() if k != 'criterion.weight'}
                print("[CovidSegmenter] Removed 'criterion.weight' from checkpoint while loading (not present in current model).")
        except Exception:
            # If any issue arises while checking, fall back to default behavior
            pass

        return super().load_state_dict(state_dict, strict=strict)


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
