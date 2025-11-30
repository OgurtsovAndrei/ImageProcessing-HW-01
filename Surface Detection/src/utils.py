import re
import os
from pathlib import Path
from typing import Tuple, Union, List
import numpy as np
import torch
import torch.nn.functional as F
import tifffile
import matplotlib.pyplot as plt
from skimage.morphology import remove_small_objects, ball
from .config import DEVICE

def get_best_checkpoint(
    checkpoint_dirs: Union[str, Path, List[Union[str, Path]]],
    name: str = "",
) -> Tuple[str, float]:
    """Finds the checkpoint with the highest val_dice score across multiple directories."""
    # Normalize input to a list of Paths
    if not isinstance(checkpoint_dirs, list):
        checkpoint_dirs = [checkpoint_dirs]
    checkpoint_dirs = [d for d in checkpoint_dirs if Path(d).exists()]
    if not checkpoint_dirs:
        print("No valid folder provided.")
        return None, None
    
    checkpoints = []
    # Regex for val_dice
    pattern = re.compile(r"val_dice=?([0-9]+\.[0-9]+)")
    # Iterate over all files in all valid directories
    for path in [f for d in checkpoint_dirs for f in Path(d).glob(f"{name}*.ckpt")]:
        match = pattern.search(path.name)
        if not match:
            continue
        checkpoints.append((float(match.group(1)), str(path)))

    if not checkpoints:
        print("No valid checkpoints found.")
        return None, None

    # Sort by score descending so the best is first
    checkpoints.sort(key=lambda x: x[0], reverse=True)
    best_score, best_path = checkpoints[0]
    print(f"Found {len(checkpoints)} checkpoints.")
    print(f"Best  (Score={best_score}): {Path(best_path)}")
    return best_path, best_score

def get_spherical_kernel(radius):
    """Generates a spherical kernel (structuring element) for 3D morphological operations."""
    # Generate boolean ball on CPU
    kernel_np = ball(radius)
    # Convert to float tensor: (1, 1, D, H, W)
    kernel = torch.from_numpy(kernel_np.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    return kernel

def post_process_3d(
    volume: np.ndarray,
    min_size: int = 1000,
    closing_radius: int = 5,
    device: str = DEVICE,
) -> np.ndarray:
    """Applies 3D morphological operations to clean up a segmentation volume using a spherical element.

    Steps:
    1. Performs morphological closing (Dilation -> Erosion) with a spherical kernel.
    2. Removes small connected components.
    """
    # Ensure input is boolean
    binary_vol = volume > 0
    clean_vol_np = binary_vol

    # 1. Close gaps with Spherical Element (GPU accelerated)
    if closing_radius > 0:
        # print(f"Closing gaps with spherical radius {closing_radius} (GPU accelerated)...")
        # Prepare Input: (1, 1, D, H, W)
        input_tensor = torch.from_numpy(clean_vol_np.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
        # Prepare Kernel
        kernel = get_spherical_kernel(closing_radius).to(device)
        # Dilation: (Input * Kernel) > 0
        # We use padding=closing_radius to maintain the same spatial dimensions (same as 'same' padding)
        dilated = (F.conv3d(input_tensor, kernel, padding=closing_radius) > 0).float()
        # Erosion: 1 - ((1 - Dilated) * Kernel > 0)
        # This relies on the duality: Erosion(A) = ~Dilation(~A)
        eroded = 1.0 - (F.conv3d(1.0 - dilated, kernel, padding=closing_radius) > 0).float()
        # Retrieve result
        clean_vol_np = eroded.squeeze().cpu().numpy().astype(bool)

    # 2. Remove small objects (CPU-bound)
    # print(f"Removing small objects < {min_size} voxels (CPU-bound)...")
    clean_vol_np = remove_small_objects(clean_vol_np, min_size=min_size)

    return clean_vol_np.astype(np.uint8)

def plot_three_axis_cuts(image_vol_path, mask_vol_path):
    """Plots the middle slice of the XY, XZ, and YZ planes for both the image volume and the predicted mask."""
    print(f"Visualizing cuts for: {os.path.basename(image_vol_path)}")
    # Load volumes
    image_vol = tifffile.imread(image_vol_path)
    mask_vol = tifffile.imread(mask_vol_path)
    
    # Get dimensions
    d, h, w = image_vol.shape
    z_mid, y_mid, x_mid = d // 2, h // 2, w // 2
    
    # Extract slices
    slices = {
        'XY Plane (Z-axis)': (image_vol[z_mid, :, :], mask_vol[z_mid, :, :]),
        'XZ Plane (Y-axis)': (image_vol[:, y_mid, :], mask_vol[:, y_mid, :]),
        'YZ Plane (X-axis)': (image_vol[:, :, x_mid], mask_vol[:, :, x_mid])
    }
    
    fig, axes = plt.subplots(3, 2, figsize=(12, 15))
    for i, (plane_name, (img_slice, mask_slice)) in enumerate(slices.items()):
        # Image Volume
        axes[i, 0].imshow(img_slice, cmap='gray')
        axes[i, 0].set_title(f"{plane_name} - Image Volume")
        axes[i, 0].axis('off')
        
        # Mask
        axes[i, 1].imshow(mask_slice, cmap='gray')
        axes[i, 1].set_title(f"{plane_name} - Predicted Mask")
        axes[i, 1].axis('off')
        
    plt.tight_layout()
    plt.show()
