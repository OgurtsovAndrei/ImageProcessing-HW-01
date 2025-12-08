import os
import warnings
import datetime
from pathlib import Path
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from monai.inferers import sliding_window_inference
import pandas as pd
import matplotlib.pyplot as plt
import tifffile
from tqdm.auto import tqdm
import torch.nn.functional as F
import zipfile
import numpy as np
import json

from config import *
from datamodule import SurfaceDataModule
from model import SurfaceSegmentation3D
from dataset import SurfaceDataset3D
from utils import get_best_checkpoint, post_process_3d, plot_three_axis_cuts
from model_factory import get_model

warnings.filterwarnings("ignore")


def main():
    # 1. Data
    datamodule = SurfaceDataModule(
        train_images_dir=TRAIN_IMAGES_DIR,
        train_labels_dir=TRAIN_LABELS_DIR,
        volume_shape=MODEL_INPUT_SIZE,
    )
    datamodule.setup()
    
    # Dynamic Output Directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_DIR / f"{MODEL_NAME}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}")

    net = get_model(MODEL_NAME)
    net_name = net.__class__.__name__
    model = SurfaceSegmentation3D(net=net, learning_rate=LEARNING_RATE)

    # 3. Checkpoint Search (for resuming or initial weights)
    # We only look in global CHECKPOINT_DIR to avoid confusing with other runs
    ckpt_path, ckpt_score = get_best_checkpoint(
        [CHECKPOINT_DIR], name=net_name
    )

    # 4. Trainer
    checkpoint_callback = ModelCheckpoint(
        dirpath=run_dir,
        filename=net_name + "-{epoch:02d}-{val_dice:.4f}",
        monitor="val_dice",
        mode="max",
        save_top_k=3,
        verbose=True
    )

    early_stop_callback = EarlyStopping(
        monitor="val_dice",
        patience=10,
        mode="max",
        verbose=True
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    csv_logger = CSVLogger(save_dir=run_dir)

    # Precision logic
    if DEVICE == "mps":
        precision_val = 32
    elif torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        precision_val = "bf16-mixed"
    else:
        precision_val = "16-mixed"

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="auto",
        devices="auto",
        logger=csv_logger,
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        precision=precision_val,
        limit_val_batches=VAL_RATIO,
        log_every_n_steps=10,
        enable_progress_bar=True,
        accumulate_grad_batches=18,
        gradient_clip_val=1.0,
    )

    # 5. Train
    print("Starting training...")
    try:
        trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)
    except MisconfigurationException as ex:
        print(ex)

    # 6. Metrics Plotting
    log_base_dir = Path(trainer.logger.save_dir) / 'lightning_logs'
    if trainer.logger.version is not None:
        version = trainer.logger.version
        if isinstance(version, int):
            metrics_path = log_base_dir / f"version_{version}" / 'metrics.csv'
        else:
            metrics_path = log_base_dir / f"{version}" / 'metrics.csv'
    else:
        # Fallback
        metrics_path = log_base_dir / 'version_0' / 'metrics.csv'  # Placeholder

    print(f"Loading metrics from: {metrics_path}")
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        metrics.ffill(inplace=True)
        metrics_melted = metrics.reset_index().melt(
            id_vars='epoch', var_name='metric', value_name='value')
        metric_groups = {
            'Loss': [c for c in metrics.columns if '_loss' in c],
            'Dice Score': [c for c in metrics.columns if '_dice' in c],
            'IoU Score': [c for c in metrics.columns if '_iou' in c],
        }
        for title, metric_list in metric_groups.items():
            group_metrics = metrics_melted[metrics_melted['metric'].isin(metric_list)]
            plt.figure(figsize=(10, 5))

            # Plot each metric
            for metric in metric_list:
                subset = group_metrics[group_metrics['metric'] == metric]
                # Sort by epoch just in case
                subset = subset.sort_values('epoch')
                plt.plot(subset['epoch'], subset['value'], label=metric)

            plt.title(f'{title} over Epochs')
            plt.xlabel('Epoch')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True, alpha=0.3)

            if title == 'Loss':
                plt.yscale('log')
            plt.tight_layout()
            plt.savefig(run_dir / f"{title.replace(' ', '_')}.png")  # Save to run_dir
            plt.close()

    # 7. Inference
    print("Starting inference...")
    test_files = sorted([f.name for f in TEST_IMAGES_DIR.glob("*.tif")])
    print(f"Test files: {test_files}")
    test_dataset = SurfaceDataset3D(
        images_dir=TEST_IMAGES_DIR,
        labels_dir=None,
        volume_files=test_files,
        volume_shape=datamodule.volume_shape

    )

    # Load best model from THIS run
    best_checkpoint_path, _ = get_best_checkpoint(
        [run_dir], name=net_name)

    # If no checkpoint found, we proceed with the current model state (which might be untrained if fit failed or wasn't needed)
    if best_checkpoint_path:
        print(f"Loading best checkpoint: {best_checkpoint_path}")
        model = SurfaceSegmentation3D.load_from_checkpoint(best_checkpoint_path, net=net)
    else:
        print("No checkpoint found in run_dir. Using current model state.")

    model.eval()
    model.to(DEVICE)

    # === Threshold Optimization ===
    print("Starting Threshold Optimization...")
    thresholds = np.arange(0.05, 1.0, 0.05)

    global_tp = torch.zeros(len(thresholds), device=DEVICE)
    global_pred_sum = torch.zeros(len(thresholds), device=DEVICE)
    global_target_sum = torch.zeros(1, device=DEVICE)

    val_loader = datamodule.val_dataloader()

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Optimizing Threshold"):
            for sample in batch:
                # Batch is list of tuples from custom_collate
                x, y, _ = sample

                inputs = x.unsqueeze(0).to(DEVICE).float().div_(255.0)
                targets = y.unsqueeze(0).to(DEVICE).long()

                if TEST_TTA:
                    probs = model.tta_inference(inputs, overlap=SW_OVERLAP)
                else:
                    logits = sliding_window_inference(
                        inputs=inputs,
                        roi_size=SW_ROI_SIZE,
                        sw_batch_size=4,
                        predictor=model,
                        overlap=SW_OVERLAP,
                        progress=False
                    )
                    probs = torch.softmax(logits, dim=1)

                targets_sq = targets.squeeze(1)
                valid_mask = (targets_sq != 2)

                # Flatten relevant pixels
                t_flat = targets_sq[valid_mask]
                p_flat = probs[:, 1, ...][valid_mask]

                # Vectorized stats
                threshold_values = torch.from_numpy(thresholds)
                if DEVICE == "mps":
                    threshold_values = threshold_values.to(torch.float32)

                t_tensor = threshold_values.to(DEVICE).unsqueeze(0)
                preds = p_flat.unsqueeze(1) > t_tensor

                t_expanded = t_flat.unsqueeze(1)

                tp_batch = (preds & (t_expanded == 1)).sum(dim=0)
                pred_sum_batch = preds.sum(dim=0)
                target_sum_batch = (t_flat == 1).sum()

                global_tp += tp_batch
                global_pred_sum += pred_sum_batch
                global_target_sum += target_sum_batch

    epsilon = 1e-6
    dices = (2 * global_tp + epsilon) / (global_pred_sum + global_target_sum + epsilon)
    ious = (global_tp + epsilon) / (global_pred_sum + global_target_sum - global_tp + epsilon)

    # Print Table
    print(f"\n{'Threshold':<10} | {'Dice':<8} | {'IoU':<8}")
    print("-" * 32)
    for i, thr in enumerate(thresholds):
        print(f"{thr:<10.2f} | {dices[i].item():<8.4f} | {ious[i].item():<8.4f}")
    print("-" * 32)

    best_idx = torch.argmax(dices).item()
    best_threshold = thresholds[best_idx]
    best_dice = dices[best_idx].item()
    best_iou = ious[best_idx].item()

    print(f"Best Threshold: {best_threshold:.2f}, Dice: {best_dice:.4f}, IoU: {best_iou:.4f}")

    with open(run_dir / "threshold.json", "w") as f:
        json.dump({"best_threshold": float(best_threshold), "val_dice": best_dice, "val_iou": best_iou}, f, indent=4)
    # ==============================

    predictions_tif = []

    # Iterate directly over the dataset
    for image, _, frag_id in tqdm(test_dataset, desc="Processing and saving 3D predictions"):
        # image is (C, D, H, W) tensor
        # sliding_window_inference expects (B, C, D, H, W)
        input_tensor = image.unsqueeze(0).to(model.device).float().div_(255.0)

        with torch.no_grad():
            if TEST_TTA:
                # TTA Ensemble Strategy via Model Method
                probs = model.tta_inference(input_tensor, overlap=SW_OVERLAP_TEST)
            else:
                # Sliding window inference (no global resizing)
                logits = sliding_window_inference(
                    inputs=input_tensor,
                    roi_size=SW_ROI_SIZE,
                    sw_batch_size=4,
                    predictor=model,
                    overlap=SW_OVERLAP_TEST,
                    progress=True,
                )
                # logits: (B, Out_Channels, D, H, W)
                probs = torch.softmax(logits, dim=1)

            # Apply Best Threshold
            pred_class = (probs[:, 1] > best_threshold).long()[0]  # (D, H, W)

        # No resizing needed, output is already full size
        pred_saved = pred_class.byte().cpu().numpy()
        pred_saved = post_process_3d(pred_saved, min_size=1000, device=DEVICE)

        prediction_tif_name = f"{frag_id}.tif"
        save_path = run_dir / prediction_tif_name
        tifffile.imwrite(str(save_path), pred_saved)
        predictions_tif.append(prediction_tif_name)

    # 8. Zip
    if predictions_tif:
        print("Visualizing first prediction...")
        mask_path = run_dir / predictions_tif[0]
        image_path = TEST_IMAGES_DIR / predictions_tif[0]

        # Check if image exists
        if image_path.exists() and mask_path.exists():
            plot_three_axis_cuts(str(image_path), str(mask_path))

        print(f"Zipping {len(predictions_tif)} files...")
        with zipfile.ZipFile(run_dir / 'submission.zip', 'w', zipfile.ZIP_DEFLATED) as zipf:
            for filename in tqdm(predictions_tif, desc="Zipping files"):
                filepath = run_dir / filename

                if not filepath.exists():
                    print(f"Missing file: {filepath}")
                    continue

                zipf.write(filepath, arcname=filename)
                # os.remove(filepath) 
        print("Submission.zip created successfully.")
    else:
        print("No predictions generated.")