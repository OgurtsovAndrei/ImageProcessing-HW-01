from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from torchvision import transforms as T
import torch


class Dataset:
    def __init__(
        self,
        images: np.ndarray,
        masks: np.ndarray,
        augmentations: Optional[Callable[..., Dict[str, Any]]] = None,
    ) -> None:
        self.images: np.ndarray = images
        self.masks: np.ndarray = masks
        self.augmentations: Optional[Callable[..., Dict[str, Any]]] = augmentations
        # single-channel normalization stats
        self.mean: List[float] = [0.485]
        self.std: List[float] = [0.229]

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image: np.ndarray = self.images[i]
        mask: np.ndarray = self.masks[i]

        if self.augmentations is not None:
            sample: Dict[str, Any] = self.augmentations(image=image, mask=mask)
            image = np.squeeze(sample["image"], axis=2)
            mask = sample["mask"]
            pil_image = Image.fromarray(image)
        else:
            pil_image = Image.fromarray(image)

        t = T.Compose([T.ToTensor(), T.Normalize(self.mean, self.std)])
        image_t: torch.Tensor = t(pil_image)
        mask_t: torch.Tensor = torch.from_numpy(mask).long()

        return image_t, mask_t

    def __len__(self) -> int:  # pragma: no cover - trivial
        return int(len(self.images))

    def tiles(self, image: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        img_patches = image.unfold(1, 512, 512).unfold(2, 768, 768)
        img_patches = img_patches.contiguous().view(3, -1, 512, 768)
        img_patches = img_patches.permute(1, 0, 2, 3)

        mask_patches = mask.unfold(0, 512, 512).unfold(1, 768, 768)
        mask_patches = mask_patches.contiguous().view(-1, 512, 768)

        return img_patches, mask_patches
