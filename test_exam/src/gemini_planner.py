import os
import re
from PIL import Image
from typing import List, Dict, Tuple, Optional, Any
import google.generativeai as genai
import test_exam.config as config
from test_exam.src.planner import BowlPlanner


class GeminiBowlPlanner(BowlPlanner):
    def __init__(
            self,
            model_id: str = config.GEMINI_MODEL_ID,
            api_key: Optional[str] = None
    ) -> None:
        self.device: str = config.DEVICE

        if api_key is None:
            api_key = os.environ.get("GEMINI_API_KEY2")

        if not api_key:
            raise ValueError("GEMINI_API_KEY2 environment variable not set.")

        genai.configure(api_key=api_key)
        self.model: genai.GenerativeModel = genai.GenerativeModel(model_id)

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

        try:
            content: List[Any] = [prompt, crop_with_box]
            response: Any = self.model.generate_content(content)
            output_text: str = response.text
        except Exception as e:
            print(f"Gemini inference error: {e}")
            return None

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

        bowl: Dict[str, float] = {
            "x": x_abs,
            "y": y_abs,
            "w": w_crop,
            "h": h_crop
        }

        return self._validate_bowl(
            bowl=bowl,
            cat_box=cat_box,
            img_w=image.width,
            img_h=image.height
        )
