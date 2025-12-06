import tifffile
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Tuple, Optional, List

class SurfaceDataset3D(Dataset):
    """3D Surface Detection Dataset.

    Updated to support volume-based loading.
    Optimized for faster Torch conversion.
    Supports .tif, .npy, .npz formats.
    """

    def __init__(
        self,
        images_dir: Path,
        labels_dir: Optional[Path],
        volume_files: Optional[List[str]] = None,
        volume_shape: Tuple[int, int, int] = (64, 64, 64), # Kept for compatibility, unused
    ):
        super().__init__()
        self.images_dir = images_dir
        # Ensure labels_dir is a Path to prevent errors in _load_from_raw
        self.labels_dir = labels_dir if labels_dir is not None else Path("__no_label__")
        self.volume_files = volume_files
        self.volume_shape = volume_shape

        # Validate and populate volume_files
        self._prepare_volume_files()

    def _prepare_volume_files(self):
        """Validate and index provided volumes, updating self.volume_files."""
        # Determine source of files
        if self.volume_files is None:
             print(f"No volume files specified. Scanning {self.images_dir}...")
             # Priority: npy > npz > tif
             extensions = [".npy", ".npz", ".tif"]
             self.volume_files = []
             # TODO: Add check for duplicates if multiple formats exist for the same volume.
             # Currently assuming each volume appears only once across these formats.
             for ext in extensions:
                 # glob returns full paths, we just want filenames
                 files = sorted([p.name for p in self.images_dir.glob(f"*{ext}")])
                 self.volume_files.extend(files)

        print(f"Indexing volumes...")
        valid_files = []
        for filename in self.volume_files:
            image_path = self.images_dir / filename
            if not image_path.exists():
                print(f"Warning: {image_path} not found, skipping.")
                continue
            valid_files.append(filename)

        self.volume_files = valid_files
        print(f"Found {len(self.volume_files)} volumes.")

    def __len__(self) -> int:
        return len(self.volume_files)

    def __getitem__(self, idx: int):
        filename = self.volume_files[idx]
        # Load raw data -> (D, H, W)
        image, mask = self._load_from_raw(filename)
        # Optimization: Convert directly to Tensor to avoid intermediate numpy float64 copies
        # 1. Convert raw uint8/uint16 -> Tensor
        # 2. Cast to float32 (mps/half fix)
        # 3. Scale
        # 4. Add channel dim
        image_t = torch.from_numpy(image).float().div_(255.0).unsqueeze(0)

        # Handle Mask
        if mask is not None:
             mask_t = torch.from_numpy(mask).long().unsqueeze(0)
        else:
             # Return dummy mask for test set (class 2 is ignored in loss)
             mask_t = torch.full_like(image_t, 2, dtype=torch.long)

        # Return fragment ID (filename without extension) for prediction grouping
        frag_id = Path(filename).stem
        return image_t, mask_t, frag_id

    def _load_file(self, path: Path) -> np.ndarray:
        """Helper to load generic file formats."""
        if path.suffix == ".npy":
            return np.load(str(path))
        if path.suffix == ".npz":
            data = np.load(str(path))
            # Return the first array found in the archive
            return data[list(data.files)[0]]
        # Original dataset format
        return tifffile.imread(str(path))

    def _load_from_raw(
        self,
        volume_file: str,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Helper to load image and mask from raw TIFF files."""
        # Load from disk
        image_path = self.images_dir / volume_file
        image_volume = self._load_file(image_path)

        label_volume = None
        label_path = self.labels_dir / volume_file
        if label_path.exists():
            label_volume = self._load_file(label_path)

        return image_volume, label_volume
