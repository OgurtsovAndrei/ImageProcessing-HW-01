import os
import random
from typing import List, Dict

import cv2
import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline

from test_exam import config as config


class TemplateInpainter:
    def __init__(self) -> None:
        self.device: str = config.DEVICE
        self.dtype: torch.dtype = (
            torch.float16 if self.device == "mps" else torch.float32
        )

        # Load img2img pipeline for denoising
        self.pipe: StableDiffusionImg2ImgPipeline = (
            StableDiffusionImg2ImgPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=self.dtype,
                variant="fp16" if self.device == "mps" else None
            ).to(self.device)
        )

        # Load bowl templates
        self.bowl_templates: List[Image.Image] = self._load_bowl_templates()
        print(f"Loaded {len(self.bowl_templates)} bowl templates")

    def _load_bowl_templates(self) -> List[Image.Image]:
        templates: List[Image.Image] = []
        bowl_dir: str = config.TEMPLATE_BOWL_DIR

        if not os.path.exists(bowl_dir):
            print(f"Warning: Bowl directory {bowl_dir} not found")
            return templates

        for filename in os.listdir(bowl_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                path: str = os.path.join(bowl_dir, filename)
                img: Image.Image = Image.open(path).convert("RGBA")
                templates.append(img)

        return templates

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

        # Ensure minimum bowl size
        if bw < config.INPAINT_MIN_W:
            diff = config.INPAINT_MIN_W - bw
            bx = int(bx - diff / 2)
            bw = int(config.INPAINT_MIN_W)

        if bh < config.INPAINT_MIN_H:
            diff = config.INPAINT_MIN_H - bh
            by = int(by - diff / 2)
            bh = int(config.INPAINT_MIN_H)

        # Crop region with padding
        padding: int = config.INPAINT_CROP_PADDING
        x1: int = max(0, bx - padding)
        y1: int = max(0, by - padding)
        x2: int = min(img_w, bx + bw + padding)
        y2: int = min(img_h, by + bh + padding)

        crop_w: int = x2 - x1
        crop_h: int = y2 - y1

        crop_img: Image.Image = full_image.crop((x1, y1, x2, y2))

        # Select random bowl template
        if not self.bowl_templates:
            print("Warning: No bowl templates available, skipping")
            return full_image

        bowl_template: Image.Image = random.choice(self.bowl_templates)

        # Resize bowl template to fit the bowl box
        resized_bowl: Image.Image = bowl_template.resize(
            (bw, bh), Image.Resampling.LANCZOS
        )

        # Create composite image: paste bowl onto crop
        composite: Image.Image = crop_img.copy()

        # Calculate position relative to crop
        rel_bx: int = bx - x1
        rel_by: int = by - y1

        # Paste bowl using alpha channel as mask
        if resized_bowl.mode == 'RGBA':
            composite.paste(resized_bowl, (rel_bx, rel_by), resized_bowl)
        else:
            composite.paste(resized_bowl, (rel_bx, rel_by))

        # Add noise to the composite
        noisy_composite: Image.Image = self._add_noise(composite)

        # Use img2img to denoise and make it realistic
        inp_size: int = config.INPAINT_SIZE
        resized_composite: Image.Image = noisy_composite.resize(
            (inp_size, inp_size), Image.Resampling.LANCZOS
        )

        denoised: Image.Image = self.pipe(
            prompt=config.INPAINT_PROMPT,
            negative_prompt=config.INPAINT_NEGATIVE_PROMPT,
            image=resized_composite,
            strength=config.TEMPLATE_DENOISE_STRENGTH,
            num_inference_steps=config.TEMPLATE_DENOISE_STEPS,
            guidance_scale=config.TEMPLATE_DENOISE_GUIDANCE
        ).images[0]

        # Resize back to crop size
        final_crop: Image.Image = denoised.resize(
            (crop_w, crop_h), Image.Resampling.LANCZOS
        )

        # Create blend mask (feather edges)
        mask_np: np.ndarray = np.zeros((crop_h, crop_w), dtype=np.uint8)
        mask_np[rel_by:rel_by+bh, rel_bx:rel_bx+bw] = 255

        # Blur mask for smooth blending
        blur_k: int = config.INPAINT_MASK_BLUR_KERNEL
        mask_np = cv2.GaussianBlur(mask_np, (blur_k, blur_k), 0)
        mask: Image.Image = Image.fromarray(mask_np)

        # Save debug artifacts
        if config.DEBUG_ENABLED and bowl_idx < config.DEBUG_MAX_ARTIFACTS_PER_IMAGE:
            os.makedirs(config.DEBUG_DIR, exist_ok=True)
            prefix: str = f"{image_name}_bowl_{bowl_idx}"
            crop_img.save(os.path.join(config.DEBUG_DIR, f"{prefix}_0_crop.png"))
            composite.save(os.path.join(config.DEBUG_DIR, f"{prefix}_1_composite.png"))
            noisy_composite.save(os.path.join(config.DEBUG_DIR, f"{prefix}_2_noisy.png"))
            resized_composite.save(os.path.join(config.DEBUG_DIR, f"{prefix}_3_input.png"))
            mask.save(os.path.join(config.DEBUG_DIR, f"{prefix}_4_mask.png"))
            denoised.save(os.path.join(config.DEBUG_DIR, f"{prefix}_5_denoised.png"))
            final_crop.save(os.path.join(config.DEBUG_DIR, f"{prefix}_6_final_crop.png"))

        # Paste back into full image
        full_image.paste(final_crop, (x1, y1), mask)

        return full_image

    def _add_noise(self, image: Image.Image) -> Image.Image:
        """Add Gaussian noise to make the bowl look less perfect"""
        img_array: np.ndarray = np.array(image).astype(np.float32)

        noise_strength: float = config.TEMPLATE_NOISE_STRENGTH * 255
        noise: np.ndarray = np.random.normal(0, noise_strength, img_array.shape)

        noisy: np.ndarray = img_array + noise
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)

        return Image.fromarray(noisy)
