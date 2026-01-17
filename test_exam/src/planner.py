import re
import torch
import numpy as np
from PIL import Image, ImageDraw
from typing import List, Dict, Any, Tuple, Optional
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import test_exam.config as config


class BowlPlanner:
    def __init__(
            self,
            model_id: str = config.VLM_MODEL_ID,
            device: str = config.DEVICE
    ) -> None:
        self.device: str = device
        self.model: Any = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=(
                torch.float16 if self.device == "mps" else torch.float32
            ),
            low_cpu_mem_usage=True
        ).to(self.device)
        self.processor: Any = AutoProcessor.from_pretrained(model_id)

    def plan_bowls(
            self,
            image_path: str,
            cat_boxes: List[Dict[str, float]]
    ) -> List[Dict[str, float]]:
        image: Image.Image = Image.open(image_path).convert("RGB")
        bowl_candidates: List[Tuple[Dict[str, float], Dict[str, float]]] = []

        for cat_box in cat_boxes:
            bowl: Optional[Dict[str, float]] = self._infer_bowl_for_cat(
                image=image,
                cat_box=cat_box
            )
            if bowl is not None:
                bowl_candidates.append((bowl, cat_box))

        bowl_boxes: List[Dict[str, float]] = self._remove_overlapping_bowls(
            bowl_candidates
        )

        return bowl_boxes

    def _extract_crop(
            self,
            image: Image.Image,
            cat_box: Dict[str, float]
    ) -> Tuple[Image.Image, int, int, int, int]:
        img_width: int = image.width
        img_height: int = image.height

        cat_cx: float = cat_box["x"] + cat_box["w"] / 2
        cat_cy: float = cat_box["y"] + cat_box["h"] / 2

        crop_w: float = cat_box["w"] * config.CROP_CONTEXT_MULTIPLIER
        crop_h: float = cat_box["h"] * config.CROP_CONTEXT_MULTIPLIER

        crop_x1: int = int(max(0, cat_cx - crop_w / 2))
        crop_y1: int = int(max(0, cat_cy - crop_h / 2))
        crop_x2: int = int(min(img_width, cat_cx + crop_w / 2))
        crop_y2: int = int(min(img_height, cat_cy + crop_h / 2))

        cropped_image: Image.Image = image.crop(
            (crop_x1, crop_y1, crop_x2, crop_y2)
        )

        return cropped_image, crop_x1, crop_y1, crop_x2, crop_y2

    def _render_single_cat_box(
            self,
            crop: Image.Image,
            cat_box: Dict[str, float],
            crop_x1: int,
            crop_y1: int
    ) -> Image.Image:
        crop_copy: Image.Image = crop.copy()
        drawer: ImageDraw.ImageDraw = ImageDraw.Draw(crop_copy)
        color: Tuple[int, int, int] = config.CAT_BOX_COLOR
        thickness: int = config.BOX_THICKNESS

        local_x1: int = int(cat_box["x"] - crop_x1)
        local_y1: int = int(cat_box["y"] - crop_y1)
        local_x2: int = int(cat_box["x"] + cat_box["w"] - crop_x1)
        local_y2: int = int(cat_box["y"] + cat_box["h"] - crop_y1)

        drawer.rectangle(
            [(local_x1, local_y1), (local_x2, local_y2)],
            outline=color,
            width=thickness
        )

        return crop_copy

    def _infer_bowl_for_cat(
            self,
            image: Image.Image,
            cat_box: Dict[str, float]
    ) -> Optional[Dict[str, float]]:
        crop, crop_x1, crop_y1, crop_x2, crop_y2 = self._extract_crop(
            image=image,
            cat_box=cat_box
        )

        crop_with_box: Image.Image = self._render_single_cat_box(
            crop=crop,
            cat_box=cat_box,
            crop_x1=crop_x1,
            crop_y1=crop_y1
        )

        crop_width: int = crop.width
        crop_height: int = crop.height

        local_cat_x1: int = int(
            (cat_box["x"] - crop_x1) * 1000 / crop_width
        )
        local_cat_y1: int = int(
            (cat_box["y"] - crop_y1) * 1000 / crop_height
        )
        local_cat_x2: int = int(
            (cat_box["x"] + cat_box["w"] - crop_x1) * 1000 / crop_width
        )
        local_cat_y2: int = int(
            (cat_box["y"] + cat_box["h"] - crop_y1) * 1000 / crop_height
        )

        prompt: str = (
            f"This image shows a single cat marked with a green bounding "
            f"box at [{local_cat_y1},{local_cat_x1},{local_cat_y2},"
            f"{local_cat_x2}].\n\n"

            "Your task is to determine if a food bowl can be placed near "
            "this cat.\n\n"

            "Rules:\n"
            "- If placement is possible, output ONE bowl bounding box.\n"
            "- If placement is not possible, output nothing.\n"
            "- The bowl MUST NOT overlap with the cat bounding box.\n"
            "- The bowl should be on the ground, typically below or "
            "slightly in front of the cat.\n"
            "- The bowl must be smaller than the cat.\n\n"

            "Output format (no extra text):\n"
            "bowl: [ymin, xmin, ymax, xmax]\n"
        )

        messages: List[Dict[str, Any]] = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": crop_with_box},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text: str = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs: Any = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.inference_mode():
            generated_ids: Any = self.model.generate(
                **inputs, max_new_tokens=512
            )

        generated_ids_trimmed: Any = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text: str = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        pattern: str = r"\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]"
        matches: List[Tuple[str, ...]] = re.findall(pattern, output_text)

        if len(matches) == 0:
            return None

        ymin, xmin, ymax, xmax = map(int, matches[0])

        x_crop: float = xmin * crop_width / 1000
        y_crop: float = ymin * crop_height / 1000
        w_crop: float = (xmax - xmin) * crop_width / 1000
        h_crop: float = (ymax - ymin) * crop_height / 1000

        x_abs: float = x_crop + crop_x1
        y_abs: float = y_crop + crop_y1

        return {
            "x": x_abs,
            "y": y_abs,
            "w": w_crop,
            "h": h_crop
        }

    def _remove_overlapping_bowls(
            self,
            bowl_candidates: List[Tuple[Dict[str, float], Dict[str, float]]]
    ) -> List[Dict[str, float]]:
        if len(bowl_candidates) == 0:
            return []

        kept_indices: List[int] = list(range(len(bowl_candidates)))

        for i in range(len(bowl_candidates)):
            if i not in kept_indices:
                continue

            bowl_i: Dict[str, float] = bowl_candidates[i][0]
            cat_i: Dict[str, float] = bowl_candidates[i][1]
            cat_i_cx: float = cat_i["x"] + cat_i["w"] / 2
            cat_i_cy: float = cat_i["y"] + cat_i["h"] / 2

            for j in range(i + 1, len(bowl_candidates)):
                if j not in kept_indices:
                    continue

                bowl_j: Dict[str, float] = bowl_candidates[j][0]

                if self._boxes_overlap(bowl_i, bowl_j):
                    cat_j: Dict[str, float] = bowl_candidates[j][1]
                    cat_j_cx: float = cat_j["x"] + cat_j["w"] / 2
                    cat_j_cy: float = cat_j["y"] + cat_j["h"] / 2

                    bowl_i_cx: float = bowl_i["x"] + bowl_i["w"] / 2
                    bowl_i_cy: float = bowl_i["y"] + bowl_i["h"] / 2
                    bowl_j_cx: float = bowl_j["x"] + bowl_j["w"] / 2
                    bowl_j_cy: float = bowl_j["y"] + bowl_j["h"] / 2

                    dist_i: float = np.sqrt(
                        (bowl_i_cx - cat_i_cx)**2 + (bowl_i_cy - cat_i_cy)**2
                    )
                    dist_j: float = np.sqrt(
                        (bowl_j_cx - cat_j_cx)**2 + (bowl_j_cy - cat_j_cy)**2
                    )

                    if dist_i <= dist_j:
                        kept_indices.remove(j)
                    else:
                        kept_indices.remove(i)
                        break

        return [bowl_candidates[i][0] for i in kept_indices]

    @staticmethod
    def _boxes_overlap(
            box1: Dict[str, float],
            box2: Dict[str, float]
    ) -> bool:
        x1_min: float = box1["x"]
        y1_min: float = box1["y"]
        x1_max: float = box1["x"] + box1["w"]
        y1_max: float = box1["y"] + box1["h"]

        x2_min: float = box2["x"]
        y2_min: float = box2["y"]
        x2_max: float = box2["x"] + box2["w"]
        y2_max: float = box2["y"] + box2["h"]

        if x1_max <= x2_min or x2_max <= x1_min:
            return False
        if y1_max <= y2_min or y2_max <= y1_min:
            return False

        return True
