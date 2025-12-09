import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from typing import Tuple, Dict
from monai.losses import DiceCELoss, TverskyLoss
from monai.inferers import sliding_window_inference
from config import MAX_EPOCHS, SW_ROI_SIZE, SW_OVERLAP, VALIDATION_TTA, VAE_LAMBDA


class SurfaceSegmentation3D(pl.LightningModule):
    """3D Surface Segmentation using a custom network.

    Key Design Choices:
    - **Loss**: Combined DiceCELoss + TverskyLoss to handle structural imbalance.
    - **Metrics**: Manual computation of Dice and IoU ignoring class 2.
    """

    def __init__(
            self,
            net: nn.Module,
            out_channels: int = 2,
            spatial_dims: int = 3,
            learning_rate: float = 1e-3,
            weight_decay: float = 1e-4,
            ignore_index_val: int = 2
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["net"])
        self.net_module = net
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.ignore_index_val = ignore_index_val

        # Loss function configuration
        # TverskyLoss with alpha=0.7 emphasizes minimizing False Negatives (Recall)
        self.criterion_tversky = TverskyLoss(
            softmax=True,
            to_onehot_y=False,
            include_background=True,
            alpha=0.7,
            beta=0.3
        )
        # DiceCELoss combines Dice Loss and Cross Entropy Loss
        self.criterion_dice_ce = DiceCELoss(
            softmax=True,
            to_onehot_y=False,
            include_background=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net_module(x)
        if isinstance(out, tuple):  # FIX: SegResNetVAE возвращает (logits, loss), нам нужны только logits
            return out[0]
        return out

    def _compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute loss excluding class 2 (unlabeled). Optimized for GPU."""
        # targets shape: (B, 1, D, H, W)
        mask = (targets != self.ignore_index_val)
        # Prepare targets for One-Hot Encoding (replace ignore index with 0 temporary)
        targets_sq = targets.squeeze(1)
        targets_clean = torch.where(mask.squeeze(1), targets_sq, torch.tensor(0, device=targets.device))
        # One-Hot Encode
        targets_onehot = torch.nn.functional.one_hot(
            targets_clean.long(),
            num_classes=self.hparams.out_channels
        ).float()
        if self.hparams.spatial_dims == 3:
            targets_onehot = targets_onehot.permute(0, 4, 1, 2, 3)
        else:
            targets_onehot = targets_onehot.permute(0, 3, 1, 2)
        # Mask One-Hot Targets
        targets_masked_ohe = targets_onehot * mask.half()

        # Compute both losses and sum them
        loss_tversky = self.criterion_tversky(logits, targets_masked_ohe)
        loss_dice_ce = self.criterion_dice_ce(logits, targets_masked_ohe)

        return loss_tversky + loss_dice_ce

    def _compute_metrics(self, preds_logits: torch.Tensor, targets_class_indices: torch.Tensor) -> dict:
        preds_proba = torch.softmax(preds_logits, dim=1)
        preds_hard = torch.argmax(preds_proba, dim=1, keepdim=True)
        valid_mask = (targets_class_indices != self.ignore_index_val).float()  # (B, 1, D, H, W)
        num_classes = preds_logits.shape[1]  # This will be 2 (background, foreground)
        dice_scores_per_class = []
        iou_scores_per_class = []

        for i in range(num_classes):
            pred_class_i = (preds_hard == i).float()  # (B, 1, D, H, W)
            target_class_i = (targets_class_indices == i).float()  # (B, 1, D, H, W)

            pred_class_i_valid = pred_class_i * valid_mask
            target_class_i_valid = target_class_i * valid_mask

            intersection = (pred_class_i_valid * target_class_i_valid).sum()
            union_sum_dice = pred_class_i_valid.sum() + target_class_i_valid.sum()
            union_sum_iou = pred_class_i_valid.sum() + target_class_i_valid.sum() - intersection
            dice = (2 * intersection + 1e-8) / (union_sum_dice + 1e-8)
            iou = (intersection + 1e-8) / (union_sum_iou + 1e-8)
            dice_scores_per_class.append(dice)
            iou_scores_per_class.append(iou)

        mean_dice = torch.mean(torch.stack(dice_scores_per_class))
        mean_iou = torch.mean(torch.stack(iou_scores_per_class))
        return {"dice": mean_dice, "iou": mean_iou}

    def training_step(self, batch, batch_idx):
        inputs, targets, _ = batch

        out = self.net_module(inputs)
        if isinstance(out, tuple):
            logits, vae_loss = out
        else:
            logits, vae_loss = out, None

        seg_loss = self._compute_loss(logits, targets)

        lambda_vae = VAE_LAMBDA
        if vae_loss is not None:
            loss = seg_loss + lambda_vae * vae_loss
            self.log("train_vae_loss", vae_loss, prog_bar=False, on_step=False, on_epoch=True)
            ratio = (lambda_vae * vae_loss) / seg_loss
            self.log("vae_ratio", ratio, on_step=False, on_epoch=True)
        else:
            loss = seg_loss

        metrics = self._compute_metrics(logits, targets)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_dice", metrics["dice"], on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_iou", metrics["iou"], on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Tuple, batch_idx: int) -> torch.Tensor:
        inputs, targets, _ = batch
        # Step A (Standard): Run the existing sliding_window_inference logic
        logits = sliding_window_inference(
            inputs=inputs,
            roi_size=SW_ROI_SIZE,
            sw_batch_size=4,
            predictor=self.forward,
            overlap=SW_OVERLAP
        )
        loss = self._compute_loss(logits, targets)

        metrics = self._compute_metrics(logits, targets)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_dice", metrics["dice"], on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_iou", metrics["iou"], on_step=False, on_epoch=True, prog_bar=True)

        # Step B (TTA - Conditional)
        if VALIDATION_TTA:
            avg_probs = self.tta_inference(inputs, overlap=SW_OVERLAP)

            # Calculate metrics (Dice/IoU) on this averaged result.
            # Passing avg_probs as "logits" works for Hard Dice because argmax(softmax(probs)) == argmax(probs).
            tta_metrics = self._compute_metrics(avg_probs, targets)

            self.log("val_dice_tta", tta_metrics["dice"], on_step=False, on_epoch=True, prog_bar=True)
            self.log("val_iou_tta", tta_metrics["iou"], on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def tta_inference(self, inputs: torch.Tensor, overlap: float) -> torch.Tensor:
        """
        Performs Test-Time Augmentation (TTA) inference by averaging predictions from 4 views:
        Original, Flip Horizontal (dim 3), Flip Vertical (dim 4), Flip Both (dim 3 and 4).
        """
        tta_probs_sum = None
        flip_configs = [[], [3], [4], [3, 4]]

        for dims in flip_configs:
            # Flip input
            x_aug = torch.flip(inputs, dims=dims) if dims else inputs

            # Inference
            logits_aug = sliding_window_inference(
                inputs=x_aug,
                roi_size=SW_ROI_SIZE,
                sw_batch_size=4,
                predictor=self.forward,
                overlap=overlap
            )

            # Flip output logits back
            if dims:
                logits_aug = torch.flip(logits_aug, dims=dims)

            # Get probabilities
            probs_aug = torch.softmax(logits_aug, dim=1)

            if tta_probs_sum is None:
                tta_probs_sum = probs_aug
            else:
                tta_probs_sum += probs_aug

        # Average probabilities
        return tta_probs_sum / len(flip_configs)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        # Cosine Annealing Scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs if self.trainer else MAX_EPOCHS,
            eta_min=1e-6
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch"
            }
        }

    def predict_step(self, batch: Tuple, batch_idx: int) -> Dict:
        inputs, _, frag_id = batch
        logits = self(inputs)
        probs = torch.softmax(logits, dim=1)
        pred_class = torch.argmax(probs, dim=1)
        return {"prediction": pred_class, "fragment_id": frag_id}
