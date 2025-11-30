import os
import warnings
from pathlib import Path
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from monai.networks.nets import SegResNet
import pandas as pd
import matplotlib.pyplot as plt
import tifffile
from tqdm.auto import tqdm
import torch.nn.functional as F
import zipfile

from src.config import (
    DATA_DIR, TRAIN_IMAGES_DIR, TRAIN_LABELS_DIR, TEST_IMAGES_DIR, OUTPUT_DIR, CHECKPOINT_DIR,
    MODEL_INPUT_SIZE, IN_CHANNELS, OUT_CHANNELS, MAX_EPOCHS, DEVICE
)
from src.datamodule import SurfaceDataModule
from src.model import SurfaceSegmentation3D
from src.dataset import SurfaceDataset3D
from src.utils import get_best_checkpoint, post_process_3d, plot_three_axis_cuts

warnings.filterwarnings("ignore")

def main():
    # 1. Data
    datamodule = SurfaceDataModule(
        train_images_dir=TRAIN_IMAGES_DIR,
        train_labels_dir=TRAIN_LABELS_DIR,
        volume_shape=MODEL_INPUT_SIZE,
    )
    datamodule.setup()

    # 2. Model
    net = SegResNet(
        spatial_dims=3,
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        init_filters=16,
        dropout_prob=0.2
    )
    net_name = net.__class__.__name__
    model = SurfaceSegmentation3D(net=net)

    # 3. Checkpoint Search
    ckpt_path, ckpt_score = get_best_checkpoint(
        [OUTPUT_DIR, CHECKPOINT_DIR], name=net_name
    )

    # 4. Trainer
    checkpoint_callback = ModelCheckpoint(
        dirpath=OUTPUT_DIR,
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
    csv_logger = CSVLogger(save_dir=OUTPUT_DIR)

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="auto",
        devices="auto",
        logger=csv_logger,
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        precision="16-mixed",
        log_every_n_steps=1,
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
    # Get the latest version directory (assuming just trained)
    if trainer.logger.version is not None:
         version = trainer.logger.version
         # version can be int or str 'version_0'
         if isinstance(version, int):
             metrics_path = log_base_dir / f"version_{version}" / 'metrics.csv'
         else:
             metrics_path = log_base_dir / f"{version}" / 'metrics.csv'
    else:
         # Fallback
         metrics_path = log_base_dir / 'version_0' / 'metrics.csv' # Placeholder

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
            plt.savefig(f"{title.replace(' ', '_')}.png") # Save instead of show
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

    # Load best model
    best_checkpoint_path, _ = get_best_checkpoint(
        [OUTPUT_DIR, CHECKPOINT_DIR], name=net_name)
    
    # If no checkpoint found, we proceed with the current model state (which might be untrained if fit failed or wasn't needed)
    if best_checkpoint_path:
        print(f"Loading best checkpoint: {best_checkpoint_path}")
        model = SurfaceSegmentation3D.load_from_checkpoint(best_checkpoint_path, net=net)
    else:
        print("No checkpoint found. Using current model state.")

    model.eval()
    model.to(DEVICE)

    predictions_tif = []
    
    # Iterate directly over the dataset
    for image, _, frag_id in tqdm(test_dataset, desc="Processing and saving 3D predictions"):
        if model.device.type == 'mps':
             # Workaround for MPS/MONAI compatibility: run transforms on CPU
             # image is already on CPU from dataset
             processed_data = datamodule.val_image_augments({"image": image})
             # Move to device
             inputs_resized = processed_data["image"].to(model.device).unsqueeze(0)
        else:
             image_on_device = image.to(model.device)
             processed_data = datamodule.val_image_augments({"image": image_on_device})
             inputs_resized = processed_data["image"].unsqueeze(0)

        with torch.no_grad():
            frag_id_list = [frag_id]
            pred_dict = model.predict_step((inputs_resized, None, frag_id_list), 0)
            pred_class = pred_dict["prediction"][0]

        # Postprocess: Resize back to Original Shape
        image_path = test_dataset.images_dir / f"{frag_id}.tif"
        with tifffile.TiffFile(str(image_path)) as tif:
            original_shape = tif.series[0].shape # (D, H, W)
        
        pred_binary = pred_class.float()
        pred_input_tensor = pred_binary.unsqueeze(0).unsqueeze(0)
        pred_restored = F.interpolate(
            pred_input_tensor,
            size=original_shape,
            mode='nearest'
        ).squeeze(0).squeeze(0)
        
        pred_saved = pred_restored.byte().cpu().numpy()
        pred_saved = post_process_3d(pred_saved, min_size=20*20*35, device=DEVICE)

        prediction_tif_name = f"{frag_id}.tif"
        save_path = OUTPUT_DIR / prediction_tif_name
        tifffile.imwrite(str(save_path), pred_saved)
        predictions_tif.append(prediction_tif_name)

    # 8. Zip
    if predictions_tif:
        # Visualize the first processed volume
        print("Visualizing first prediction...")
        mask_path = OUTPUT_DIR / predictions_tif[0]
        image_path = TEST_IMAGES_DIR / predictions_tif[0]
        # Check if image exists (it should)
        if image_path.exists() and mask_path.exists():
             plot_three_axis_cuts(str(image_path), str(mask_path))
        
        print(f"Zipping {len(predictions_tif)} files...")
        with zipfile.ZipFile('submission.zip', 'w', zipfile.ZIP_DEFLATED) as zipf:
            for filename in tqdm(predictions_tif, desc="Zipping files"):
                if not os.path.exists(filename):
                    print(f"Missing <> {filename}")
                    continue
                zipf.write(filename)
                # Remove original file to save space
                os.remove(filename)
        print("Submission.zip created successfully.")
    else:
        print("No predictions generated.")

if __name__ == "__main__":
    main()
