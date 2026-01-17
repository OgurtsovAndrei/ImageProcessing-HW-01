import os
import random
from typing import List, Dict, Tuple, Any

import cv2
import numpy as np
import torch
from PIL import Image, ImageFilter
from diffusers import AutoPipelineForInpainting

from test_exam import config as config


class TemplateInpainter:
    def __init__(self) -> None:
        self.device: str = config.DEVICE
        self.dtype: torch.dtype = (
            torch.float16 if self.device == "mps" else torch.float32
        )

        self.pipe: AutoPipelineForInpainting = (
            AutoPipelineForInpainting.from_pretrained(
                config.INPAINT_MODEL_ID,
                torch_dtype=self.dtype,
                variant="fp16" if self.device == "mps" else None,
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

        mask_crop: Image.Image = self._create_bowl_mask(
            crop_w=crop_w,
            crop_h=crop_h,
            rel_bx=rel_bx,
            rel_by=rel_by,
            bw=bw,
            bh=bh,
        )

        composite_noisy: Image.Image = self._apply_local_noise(
            image=composite,
            mask=mask_crop,
        )

        inp_size: int = config.INPAINT_SIZE
        sq_input, scale, pad_left, pad_top, new_w, new_h = letterbox_to_square(
            composite_noisy, inp_size
        )
        sq_mask: Image.Image = letterbox_mask_to_square(
            mask_crop, inp_size, scale, pad_left, pad_top, new_w, new_h
        )

        pipe_kwargs: dict[str, Any] = {
            "prompt": config.TEMPLATE_INPAINT_PROMPT,
            "negative_prompt": config.TEMPLATE_INPAINT_NEGATIVE_PROMPT,
            "image": sq_input,
            "mask_image": sq_mask,
            "num_inference_steps": config.TEMPLATE_INPAINT_NUM_STEPS,
            "guidance_scale": config.TEMPLATE_INPAINT_GUIDANCE_SCALE,
            "strength": config.TEMPLATE_INPAINT_STRENGTH,
        }

        try:
            pipe_result = self.pipe(**pipe_kwargs)  # type: ignore
            sq_inpainted: Image.Image = pipe_result.images[0]
        except TypeError:
            pipe_kwargs.pop("strength", None)
            pipe_result = self.pipe(**pipe_kwargs)  # type: ignore
            sq_inpainted = pipe_result.images[0]

        if sq_inpainted.size != sq_input.size:
            sq_inpainted = sq_inpainted.resize(
                sq_input.size,
                Image.Resampling.LANCZOS,
            )
        if sq_inpainted.mode != sq_input.mode:
            sq_inpainted = sq_inpainted.convert(sq_input.mode)

        if sq_mask.size != sq_input.size:
            sq_mask = sq_mask.resize(
                sq_input.size,
                Image.Resampling.NEAREST,
            )
        if sq_mask.mode != "L":
            sq_mask = sq_mask.convert("L")

        sq_final: Image.Image = Image.composite(
            sq_inpainted,
            sq_input,
            sq_mask,
        )

        final_crop: Image.Image = unletterbox_from_square(
            sq_final,
            crop_w,
            crop_h,
            scale,
            pad_left,
            pad_top,
            new_w,
            new_h,
        )

        # Save debug artifacts
        if (
                config.DEBUG_ENABLED
                and bowl_idx < config.DEBUG_MAX_ARTIFACTS_PER_IMAGE
        ):
            os.makedirs(config.DEBUG_DIR, exist_ok=True)
            prefix: str = f"{image_name}_bowl_{bowl_idx}"
            crop_img.save(
                os.path.join(config.DEBUG_DIR, f"{prefix}_0_crop.png")
            )
            composite.save(
                os.path.join(config.DEBUG_DIR, f"{prefix}_1_composite.png")
            )
            composite_noisy.save(
                os.path.join(config.DEBUG_DIR, f"{prefix}_2_noisy.png")
            )
            sq_input.save(
                os.path.join(
                    config.DEBUG_DIR,
                    f"{prefix}_3_input_letterbox.png",
                )
            )
            sq_mask.save(
                os.path.join(
                    config.DEBUG_DIR,
                    f"{prefix}_4_mask_letterbox.png",
                )
            )
            sq_inpainted.save(
                os.path.join(
                    config.DEBUG_DIR,
                    f"{prefix}_5_inpainted_square.png",
                )
            )
            sq_final.save(
                os.path.join(
                    config.DEBUG_DIR,
                    f"{prefix}_6_composited_square.png",
                )
            )
            final_crop.save(
                os.path.join(config.DEBUG_DIR, f"{prefix}_7_final_crop.png")
            )
            mask_crop.save(
                os.path.join(config.DEBUG_DIR, f"{prefix}_8_mask_crop.png")
            )

        # Paste back into full image
        full_image.paste(final_crop, (x1, y1), mask_crop)

        return full_image

    def _apply_local_noise(
            self,
            image: Image.Image,
            mask: Image.Image,
    ) -> Image.Image:
        sigma: float = float(config.TEMPLATE_NOISE_SIGMA)
        if sigma <= 0.0:
            return image

        if not config.TEMPLATE_NOISE_ON_BOWL_ONLY:
            img_array: np.ndarray = np.asarray(image).astype(np.float32)
            noise_float: np.ndarray = np.random.normal(
                0.0,
                sigma,
                img_array.shape,
            )
            noise: np.ndarray = noise_float.astype(np.float32)
            noisy: np.ndarray = np.clip(img_array + noise, 0.0, 255.0).astype(
                np.uint8
            )
            return Image.fromarray(noisy)

        img_array = np.asarray(image).astype(np.float32)
        mask_np: np.ndarray = np.asarray(mask).astype(np.float32) / 255.0
        if mask_np.ndim != 2:
            mask_np = mask_np[:, :, 0]
        mask_np = np.clip(mask_np, 0.0, 1.0)
        mask_np_3: np.ndarray = np.repeat(mask_np[:, :, None], 3, axis=2)

        noise = np.random.normal(0.0, sigma, img_array.shape).astype(
            np.float32
        )
        noisy = img_array + noise * mask_np_3
        noisy = np.clip(noisy, 0.0, 255.0).astype(np.uint8)
        return Image.fromarray(noisy)

    def _create_bowl_mask(
            self,
            crop_w: int,
            crop_h: int,
            rel_bx: int,
            rel_by: int,
            bw: int,
            bh: int,
    ) -> Image.Image:
        margin_ratio: float = float(config.TEMPLATE_MASK_MARGIN_RATIO)
        margin_x: int = int(round(float(bw) * margin_ratio))
        margin_y: int = int(round(float(bh) * margin_ratio))

        x1: int = max(0, rel_bx - margin_x)
        y1: int = max(0, rel_by - margin_y)
        x2: int = min(crop_w, rel_bx + bw + margin_x)
        y2: int = min(crop_h, rel_by + bh + margin_y)

        mask_np: np.ndarray = np.zeros((crop_h, crop_w), dtype=np.uint8)
        mask_np[y1:y2, x1:x2] = 255

        blur_k: int = int(config.TEMPLATE_MASK_BLUR_KERNEL)
        if blur_k < 1:
            blur_k = 1
        if blur_k % 2 == 0:
            blur_k += 1
        if blur_k > 1:
            mask_np = cv2.GaussianBlur(mask_np, (blur_k, blur_k), 0)

        return Image.fromarray(mask_np)


class FullTemplateInpainter(TemplateInpainter):
    def inpaint_bowls(
            self,
            image_path: str,
            bowl_boxes: List[Dict[str, float]]
    ) -> Image.Image:
        full_image: Image.Image = Image.open(image_path).convert("RGB")
        image_name: str = os.path.basename(image_path).split('.')[0]

        for bowl_box in bowl_boxes:
            full_image = self._paste_bowl(full_image, bowl_box)

        temp_dir: str = "test_exam/res-mult-cat-at-once"
        os.makedirs(temp_dir, exist_ok=True)
        temp_path: str = os.path.join(temp_dir, f"temp-{image_name}.jpg")
        full_image.save(temp_path)
        print(f"Saved intermediate image to {temp_path}")

        if config.DEBUG_ENABLED:
            os.makedirs(config.DEBUG_DIR, exist_ok=True)
            full_image.save(
                os.path.join(
                    config.DEBUG_DIR, f"{image_name}_1_added_bowls.jpg"
                )
            )

        img_w: int = full_image.size[0]
        img_h: int = full_image.size[1]
        global_mask_np: np.ndarray = np.full(
            (img_h, img_w), 255, dtype=np.uint8
        )
        global_mask: Image.Image = Image.fromarray(global_mask_np)

        blurred_image: Image.Image = full_image.filter(
            ImageFilter.GaussianBlur(radius=1.0)
        )

        if config.DEBUG_ENABLED:
            blurred_image.save(
                os.path.join(
                    config.DEBUG_DIR, f"{image_name}_2_noised.jpg"
                )
            )

        inp_size: int = config.INPAINT_SIZE
        sq_input: Image.Image
        scale: float
        pad_left: int
        pad_top: int
        new_w: int
        new_h: int
        sq_input, scale, pad_left, pad_top, new_w, new_h = letterbox_to_square(
            blurred_image, inp_size
        )
        sq_mask: Image.Image = letterbox_mask_to_square(
            global_mask, inp_size, scale, pad_left, pad_top, new_w, new_h
        )

        pipe_kwargs: Dict[str, Any] = {
            "prompt": config.TEMPLATE_INPAINT_PROMPT,
            "negative_prompt": config.TEMPLATE_INPAINT_NEGATIVE_PROMPT,
            "image": sq_input,
            "mask_image": sq_mask,
            "num_inference_steps": config.TEMPLATE_INPAINT_NUM_STEPS,
            "guidance_scale": config.TEMPLATE_INPAINT_GUIDANCE_SCALE,
            "strength": config.TEMPLATE_INPAINT_STRENGTH,
        }

        sq_inpainted: Image.Image
        try:
            pipe_result: Any = self.pipe(**pipe_kwargs)  # type: ignore
            sq_inpainted = pipe_result.images[0]
        except TypeError:
            pipe_kwargs.pop("strength", None)
            pipe_result = self.pipe(**pipe_kwargs)  # type: ignore
            sq_inpainted = pipe_result.images[0]

        final_image: Image.Image = unletterbox_from_square(
            sq_inpainted, img_w, img_h, scale, pad_left, pad_top, new_w, new_h
        )

        final_special_path: str = os.path.join(
            temp_dir, f"final-{image_name}-dumb-inpaint.jpg"
        )
        final_image.save(final_special_path)
        print(f"Saved final special result to {final_special_path}")

        if config.DEBUG_ENABLED:
            final_image.save(
                os.path.join(
                    config.DEBUG_DIR, f"{image_name}_3_denoised.jpg"
                )
            )

        return final_image

    def _paste_bowl(
            self,
            full_image: Image.Image,
            bowl_box: Dict[str, float]
    ) -> Image.Image:
        bx: int = int(bowl_box["x"])
        by: int = int(bowl_box["y"])
        bw: int = int(bowl_box["w"])
        bh: int = int(bowl_box["h"])

        if not self.bowl_templates:
            return full_image

        bowl_template: Image.Image = random.choice(self.bowl_templates)
        resized_bowl: Image.Image = bowl_template.resize(
            (bw, bh), Image.Resampling.LANCZOS
        )

        res: Image.Image = full_image.copy()
        if resized_bowl.mode == 'RGBA':
            res.paste(resized_bowl, (bx, by), resized_bowl)
        else:
            res.paste(resized_bowl, (bx, by))
        return res


def letterbox_to_square(
        image: Image.Image,
        target_size: int
) -> Tuple[Image.Image, float, int, int, int, int]:
    w: int
    h: int
    w, h = image.size
    if w <= 0 or h <= 0:
        raise ValueError("Invalid image size")

    scale: float = float(target_size) / float(max(w, h))
    new_w: int = max(1, int(round(float(w) * scale)))
    new_h: int = max(1, int(round(float(h) * scale)))
    resized: Image.Image = image.resize(
        (new_w, new_h),
        Image.Resampling.LANCZOS,
    )

    pad_left: int = (target_size - new_w) // 2
    pad_right: int = target_size - new_w - pad_left
    pad_top: int = (target_size - new_h) // 2
    pad_bottom: int = target_size - new_h - pad_top

    arr: np.ndarray = np.asarray(resized)
    if arr.ndim == 2:
        arr = arr[:, :, None]

    pad_width: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]] = (
        (pad_top, pad_bottom),
        (pad_left, pad_right),
        (0, 0),
    )
    padded: np.ndarray = np.pad(arr, pad_width, mode="edge")
    if padded.shape[0] != target_size or padded.shape[1] != target_size:
        padded = cv2.resize(
            padded,
            (target_size, target_size),
            interpolation=cv2.INTER_AREA,
        )

    sq_image: Image.Image = Image.fromarray(padded)
    return sq_image, scale, pad_left, pad_top, new_w, new_h


def letterbox_mask_to_square(
        mask: Image.Image,
        target_size: int,
        scale: float,
        pad_left: int,
        pad_top: int,
        new_w: int,
        new_h: int,
) -> Image.Image:
    _ = scale
    resized: Image.Image = mask.resize(
        (new_w, new_h),
        Image.Resampling.NEAREST,
    )
    arr: np.ndarray = np.asarray(resized)
    if arr.ndim != 2:
        arr = arr[:, :, 0]

    pad_right: int = target_size - new_w - pad_left
    pad_bottom: int = target_size - new_h - pad_top
    pad_width: Tuple[Tuple[int, int], Tuple[int, int]] = (
        (pad_top, pad_bottom),
        (pad_left, pad_right),
    )
    padded: np.ndarray = np.pad(
        arr,
        pad_width,
        mode="constant",
        constant_values=0,
    )
    if padded.shape[0] != target_size or padded.shape[1] != target_size:
        padded = cv2.resize(
            padded,
            (target_size, target_size),
            interpolation=cv2.INTER_NEAREST,
        )
    return Image.fromarray(padded.astype(np.uint8))


def unletterbox_from_square(
        square_output: Image.Image,
        original_crop_w: int,
        original_crop_h: int,
        scale: float,
        pad_left: int,
        pad_top: int,
        new_w: int,
        new_h: int,
) -> Image.Image:
    _ = scale
    crop_box: Tuple[int, int, int, int] = (
        pad_left,
        pad_top,
        pad_left + new_w,
        pad_top + new_h,
    )
    unpadded: Image.Image = square_output.crop(crop_box)
    return unpadded.resize(
        (original_crop_w, original_crop_h),
        Image.Resampling.LANCZOS,
    )
