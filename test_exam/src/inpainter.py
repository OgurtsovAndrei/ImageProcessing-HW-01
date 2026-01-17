import torch
from PIL import Image, ImageDraw
from typing import List, Dict
from diffusers import AutoPipelineForInpainting
import test_exam.config as config
import numpy as np
import cv2


class BowlInpainter:
    def __init__(self) -> None:
        self.device: str = config.DEVICE
        self.dtype: torch.dtype = (
            torch.float16 if self.device == "mps" else torch.float32
        )

        self.pipe: AutoPipelineForInpainting = (
            AutoPipelineForInpainting.from_pretrained(
                config.INPAINT_MODEL_ID,
                torch_dtype=self.dtype,
                variant="fp16" if self.device == "mps" else None
            ).to(self.device)
        )

    def inpaint_bowls(
            self,
            image_path: str,
            bowl_boxes: List[Dict[str, float]]
    ) -> Image.Image:
        full_image: Image.Image = Image.open(image_path).convert("RGB")

        for bowl_box in bowl_boxes:
            full_image = self._inpaint_single_bowl(full_image, bowl_box)

        return full_image

    def _inpaint_single_bowl(
            self,
            full_image: Image.Image,
            bowl_box: Dict[str, float]
    ) -> Image.Image:
        img_w: int = full_image.size[0]
        img_h: int = full_image.size[1]

        bx: int = int(bowl_box["x"])
        by: int = int(bowl_box["y"])
        bw: int = int(bowl_box["w"])
        bh: int = int(bowl_box["h"])

        padding: int = config.INPAINT_CROP_PADDING
        x1: int = max(0, bx - padding)
        y1: int = max(0, by - padding)
        x2: int = min(img_w, bx + bw + padding)
        y2: int = min(img_h, by + bh + padding)

        crop_w: int = x2 - x1
        crop_h: int = y2 - y1

        crop_img: Image.Image = full_image.crop((x1, y1, x2, y2))

        input_img: Image.Image = crop_img.resize(
            (1024, 1024), Image.Resampling.LANCZOS
        )

        scale_x: float = 1024.0 / crop_w
        scale_y: float = 1024.0 / crop_h

        rel_bx1: float = (bx - x1) * scale_x
        rel_by1: float = (by - y1) * scale_y
        rel_bx2: float = (bx + bw - x1) * scale_x
        rel_by2: float = (by + bh - y1) * scale_y

        mask: Image.Image = Image.new("L", (1024, 1024), 0)
        draw: ImageDraw.ImageDraw = ImageDraw.Draw(mask)
        draw.rectangle([rel_bx1, rel_by1, rel_bx2, rel_by2], fill=255)

        mask_np: np.ndarray = np.array(mask)
        mask_np = cv2.GaussianBlur(mask_np, (21, 21), 0)
        mask = Image.fromarray(mask_np)

        inpainted_crop: Image.Image = self.pipe(
            prompt=config.INPAINT_PROMPT,
            negative_prompt=config.INPAINT_NEGATIVE_PROMPT,
            image=input_img,
            mask_image=mask,
            num_inference_steps=config.INPAINT_NUM_STEPS,
            guidance_scale=config.INPAINT_GUIDANCE_SCALE,
        ).images[0]  # type: ignore

        final_crop: Image.Image = inpainted_crop.resize(
            (crop_w, crop_h), Image.Resampling.LANCZOS
        )

        final_mask_np: np.ndarray = cv2.resize(mask_np, (crop_w, crop_h))
        final_mask: Image.Image = Image.fromarray(final_mask_np)

        full_image.paste(final_crop, (x1, y1), final_mask)

        return full_image
