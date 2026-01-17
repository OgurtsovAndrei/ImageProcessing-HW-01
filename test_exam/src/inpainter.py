import torch
import os
from PIL import Image, ImageDraw
from typing import List, Dict, Any
from diffusers import AutoPipelineForInpainting

import test_exam.config as config
import numpy as np
import cv2

from test_exam.src.template_inpainter import TemplateInpainter


class DiffusionInpainter:
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
        print(type(self.pipe))

    def inpaint_bowls(
            self,
            image_path: str,
            bowl_boxes: List[Dict[str, float]]
    ) -> Image.Image:
        full_image: Image.Image = Image.open(image_path).convert("RGB")
        image_name: str = os.path.basename(image_path).split('.')[0]

        print(f"Inpainting bowls for {image_name}: {bowl_boxes}")
        for i, bowl_box in enumerate(bowl_boxes):
            full_image = self._inpaint_single_bowl(
                full_image, bowl_box, image_name, i
            )
            if config.DEBUG_ENABLED:
                os.makedirs(config.DEBUG_DIR, exist_ok=True)
                debug_path = os.path.join(
                    config.DEBUG_DIR, f"{image_name}_step_{i}_full.png"
                )
                full_image.save(debug_path)

        return full_image

    def _inpaint_single_bowl(
            self,
            full_image: Image.Image,
            bowl_box: Dict[str, float],
            image_name: str = "debug",
            bowl_idx: int = 0
    ) -> Image.Image:
        img_w: int = full_image.size[0]
        img_h: int = full_image.size[1]

        bx: int = int(bowl_box["x"])
        by: int = int(bowl_box["y"])
        bw: int = int(bowl_box["w"])
        bh: int = int(bowl_box["h"])

        # Ensure the box is at least INPAINT_MIN_W and INPAINT_MIN_H
        if bw < config.INPAINT_MIN_W:
            diff = config.INPAINT_MIN_W - bw
            bx = int(bx - diff / 2)
            bw = int(config.INPAINT_MIN_W)

        if bh < config.INPAINT_MIN_H:
            diff = config.INPAINT_MIN_H - bh
            by = int(by - diff / 2)
            bh = int(config.INPAINT_MIN_H)

        padding: int = config.INPAINT_CROP_PADDING
        x1: int = max(0, bx - padding)
        y1: int = max(0, by - padding)
        x2: int = min(img_w, bx + bw + padding)
        y2: int = min(img_h, by + bh + padding)

        crop_w: int = x2 - x1
        crop_h: int = y2 - y1

        crop_img: Image.Image = full_image.crop((x1, y1, x2, y2))

        inp_size: int = config.INPAINT_SIZE
        input_img: Image.Image = crop_img.resize(
            (inp_size, inp_size), Image.Resampling.LANCZOS
        )

        scale_x: float = float(inp_size) / crop_w
        scale_y: float = float(inp_size) / crop_h

        rel_bx1: float = (bx - x1) * scale_x
        rel_by1: float = (by - y1) * scale_y
        rel_bx2: float = (bx + bw - x1) * scale_x
        rel_by2: float = (by + bh - y1) * scale_y

        margin_x: float = (rel_bx2 - rel_bx1) * config.INPAINT_MASK_MARGIN_RATIO
        margin_y: float = (rel_by2 - rel_by1) * config.INPAINT_MASK_MARGIN_RATIO

        rel_bx1_m: float = max(0.0, rel_bx1 - margin_x)
        rel_by1_m: float = max(0.0, rel_by1 - margin_y)
        rel_bx2_m: float = min(float(inp_size), rel_bx2 + margin_x)
        rel_by2_m: float = min(float(inp_size), rel_by2 + margin_y)

        mask: Image.Image = Image.new("L", (inp_size, inp_size), 0)
        draw: ImageDraw.ImageDraw = ImageDraw.Draw(mask)
        draw.rectangle(
            [rel_bx1_m, rel_by1_m, rel_bx2_m, rel_by2_m],
            fill=255
        )

        mask_np: np.ndarray = np.array(mask)
        blur_k: int = config.INPAINT_MASK_BLUR_KERNEL
        mask_np = cv2.GaussianBlur(mask_np, (blur_k, blur_k), 0)
        mask = Image.fromarray(mask_np)

        pipe_kwargs: Dict[str, Any] = {"prompt": config.INPAINT_PROMPT,
                                       "negative_prompt": config.INPAINT_NEGATIVE_PROMPT, "image": input_img,
                                       "mask_image": mask, "num_inference_steps": config.INPAINT_NUM_STEPS,
                                       "guidance_scale": config.INPAINT_GUIDANCE_SCALE,
                                       "strength": config.INPAINT_STRENGTH}

        # sig: inspect.Signature = inspect.signature(self.pipe.__call__)  # type: ignore
        # if "strength" in sig.parameters:

        inpainted_crop: Image.Image = self.pipe(**pipe_kwargs).images[0]  # type: ignore

        if config.DEBUG_ENABLED and bowl_idx < config.DEBUG_MAX_ARTIFACTS_PER_IMAGE:
            os.makedirs(config.DEBUG_DIR, exist_ok=True)
            prefix: str = f"{image_name}_bowl_{bowl_idx}"
            crop_img.save(os.path.join(config.DEBUG_DIR, f"{prefix}_0_crop.png"))
            input_img.save(os.path.join(config.DEBUG_DIR, f"{prefix}_1_input.png"))
            mask.save(os.path.join(config.DEBUG_DIR, f"{prefix}_2_mask.png"))
            inpainted_crop.save(
                os.path.join(config.DEBUG_DIR, f"{prefix}_3_inpainted.png")
            )

        final_crop: Image.Image = inpainted_crop.resize(
            (crop_w, crop_h), Image.Resampling.LANCZOS
        )

        final_mask_np: np.ndarray = cv2.resize(mask_np, (crop_w, crop_h))
        final_mask: Image.Image = Image.fromarray(final_mask_np)

        full_image.paste(final_crop, (x1, y1), final_mask)

        return full_image


def create_inpainter() -> Any:
    """Factory function to create the appropriate inpainter based on config"""
    if config.INPAINTER_TYPE == config.InpainterType.TEMPLATE:
        return TemplateInpainter()
    else:
        return DiffusionInpainter()


# Backwards compatibility
BowlInpainter = DiffusionInpainter
