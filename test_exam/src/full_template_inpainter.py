import os
import random
from typing import List, Dict, Any

import numpy as np
from PIL import Image, ImageFilter

from test_exam import config as config
from test_exam.src.template_inpainter import (
    TemplateInpainter, letterbox_to_square, letterbox_mask_to_square,
    unletterbox_from_square_full
)


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

        res_dir: str = "test_exam/result_final"
        os.makedirs(res_dir, exist_ok=True)
        temp_path: str = os.path.join(
            res_dir, f"intermediate-{image_name}-full-inpaint.jpg"
        )
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

        print(f"Debug: sq_inpainted size: {sq_inpainted.size}")
        print(
            f"Debug: expected size: ({new_w}, {new_h}) "
            f"in square of {inp_size}"
        )
        print(f"Debug: target restoration: {img_w}x{img_h}")

        final_image: Image.Image = unletterbox_from_square_full(
            sq_inpainted, img_w, img_h, scale, pad_left, pad_top, new_w, new_h
        )

        final_special_path: str = os.path.join(
            res_dir, f"final-{image_name}-full-inpaint.jpg"
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

        # Ensure minimum bowl size
        if bw < config.INPAINT_MIN_W:
            diff = config.INPAINT_MIN_W - bw
            bx = int(bx - diff / 2)
            bw = int(config.INPAINT_MIN_W)

        if bh < config.INPAINT_MIN_H:
            diff = config.INPAINT_MIN_H - bh
            by = int(by - diff / 2)
            bh = int(config.INPAINT_MIN_H)

        # Apply scaling
        bw_new: int = int(bw * config.BOWL_SCALE_FACTOR)
        bh_new: int = int(bh * config.BOWL_SCALE_FACTOR)
        bx = int(bx + (bw - bw_new) / 2)
        by = int(by + (bh - bh_new) / 2)
        bw = bw_new
        bh = bh_new

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
