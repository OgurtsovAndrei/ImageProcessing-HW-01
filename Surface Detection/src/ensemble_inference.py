import argparse
import os
import zipfile
from pathlib import Path
from typing import List

import torch
import tifffile
import numpy as np
from tqdm.auto import tqdm
from monai.inferers import sliding_window_inference

from config import *
from dataset import SurfaceDataset3D
from model import SurfaceSegmentation3D
from utils import post_process_3d, get_best_checkpoint
from model_factory import get_model

def ensemble_inference(model_paths: List[str], output_dir: Path):
    print(f"Ensembling models from: {model_paths}")
    
    # 1. Setup Data
    test_files = sorted([f.name for f in TEST_IMAGES_DIR.glob("*.tif")])
    test_dataset = SurfaceDataset3D(
        images_dir=TEST_IMAGES_DIR,
        labels_dir=None,
        volume_files=test_files,
        volume_shape=MODEL_INPUT_SIZE
    )
    
    # 2. Load Models
    models = []
    for path_str in model_paths:
        path = Path(path_str)
        ckpt_path = None
        if path.is_dir():
            ckpt_path, _ = get_best_checkpoint([path]) # name="" finds any .ckpt
        elif path.is_file() and path.suffix == ".ckpt":
            ckpt_path = path
        
        if not ckpt_path:
            print(f"Warning: No checkpoint found for {path_str}. Skipping.")
            continue
            
        print(f"Loading checkpoint: {ckpt_path}")
        
        # Infer model type from filename or default to BasicUNet
        ckpt_filename = Path(ckpt_path).name
        if "SegResNet" in ckpt_filename:
            net = get_model("SegResNet")
        else:
            net = get_model("BasicUNet") 
            
        model = SurfaceSegmentation3D.load_from_checkpoint(str(ckpt_path), net=net)
        model.eval()
        model.to(DEVICE)
        models.append(model)
        
    if not models:
        print("No models loaded. Exiting.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_tif = []

    # 3. Inference Loop
    for image, _, frag_id in tqdm(test_dataset, desc="Ensemble Inference"):
        # Image: (C, D, H, W)
        input_tensor = image.unsqueeze(0).to(DEVICE).float().div_(255.0) # (1, C, D, H, W)
        
        accumulated_probs = None
        
        with torch.no_grad():
            for model in models:
                if TEST_TTA:
                    # model.tta_inference returns (B, C, D, H, W)
                    probs = model.tta_inference(input_tensor, overlap=SW_OVERLAP_TEST)
                else:
                    logits = sliding_window_inference(
                        inputs=input_tensor,
                        roi_size=SW_ROI_SIZE,
                        sw_batch_size=4,
                        predictor=model,
                        overlap=SW_OVERLAP_TEST,
                        progress=False
                    )
                    probs = torch.softmax(logits, dim=1) # (B, C, D, H, W)
                
                if accumulated_probs is None:
                    accumulated_probs = probs
                else:
                    accumulated_probs += probs
        
        # Average
        avg_probs = accumulated_probs / len(models)
        # avg_probs is (B, C, D, H, W) -> (1, 2, D, H, W)
        pred_class = torch.argmax(avg_probs, dim=1)[0] # (D, H, W)
        
        # Post-process
        pred_saved = pred_class.byte().cpu().numpy()
        pred_saved = post_process_3d(pred_saved, min_size=20 * 20 * 35, device=DEVICE)
        
        prediction_tif_name = f"{frag_id}.tif"
        save_path = output_dir / prediction_tif_name
        tifffile.imwrite(str(save_path), pred_saved)
        predictions_tif.append(prediction_tif_name)

    # 4. Zip
    if predictions_tif:
        print(f"Zipping {len(predictions_tif)} files...")
        zip_path = output_dir / 'submission.zip'
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for filename in tqdm(predictions_tif, desc="Zipping files"):
                filepath = output_dir / filename
                zipf.write(filepath, arcname=filename)
                # os.remove(filepath) 
        print(f"Ensemble submission saved to {zip_path}")

if __name__ == "__main__":
    models = [] # List of model paths (dirs or .ckpt files)
    out = "data/ensemble_out"
    
    ensemble_inference(models, Path(out))
