import os
import re
from PIL import Image, ImageDraw
from typing import List, Dict, Tuple, Optional, Any
import google.generativeai as genai
import test_exam.config as config
from test_exam.src.planner import BowlPlanner


class GeminiInstantPlanner(BowlPlanner):
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

    def plan_bowls(
            self,
            image_path: str,
            cat_boxes: List[Dict[str, float]]
    ) -> List[Dict[str, float]]:
        if not cat_boxes:
            return []

        image: Image.Image = Image.open(image_path).convert("RGB")
        img_width: int = image.width
        img_height: int = image.height

        annotated_image: Image.Image = image.copy()
        draw: ImageDraw.ImageDraw = ImageDraw.Draw(annotated_image)

        cat_texts: List[str] = []
        for i, box in enumerate(cat_boxes):
            x1: float = box["x"]
            y1: float = box["y"]
            x2: float = box["x"] + box["w"]
            y2: float = box["y"] + box["h"]

            draw.rectangle(
                [x1, y1, x2, y2],
                outline=config.CAT_BOX_COLOR,
                width=config.BOX_THICKNESS
            )

            norm_y1: int = int(y1 * 1000 / img_height)
            norm_x1: int = int(x1 * 1000 / img_width)
            norm_y2: int = int(y2 * 1000 / img_height)
            norm_x2: int = int(x2 * 1000 / img_width)
            cat_texts.append(
                f"cat_{i}: [{norm_y1}, {norm_x1}, {norm_y2}, {norm_x2}]"
            )

        cats_str: str = "\n".join(cat_texts)
        prompt: str = (
            f"I have an image with {len(cat_boxes)} cats. "
            f"The cats are located at the following coordinates "
            f"(normalized [0, 1000]):\n"
            f"{cats_str}\n\n"
            "I am providing two images: the original one and one where these "
            "cats are marked with green boxes.\n"
            "Your task is to place exactly one food bowl for each cat "
            "where possible.\n"
            "Rules:\n"
            "- A bowl should be placed on the ground near the cat "
            "(usually below or in front).\n"
            "- A bowl MUST NOT overlap with any cat bounding box.\n"
            "- A bowl must be smaller than the cat it belongs to.\n"
            "- If it is impossible to place a bowl for a specific cat due "
            "to space constraints, do not place it.\n"
            "- It will be nice to have bowl at the side cat is looking to.\n\n"
            "Output the detected bowls in the following format "
            "(no extra text):\n"
            "bowl: [ymin, xmin, ymax, xmax]\n"
            "bowl: [ymin, xmin, ymax, xmax]\n"
            "... (one for each valid placement)"
        )

        try:
            content: List[Any] = [prompt, image, annotated_image]
            response: Any = self.model.generate_content(content)
            output_text: str = response.text
        except Exception as e:
            print(f"Gemini Instant inference error: {e}")
            return []

        pattern: str = r"\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]"
        matches: List[Tuple[str, ...]] = re.findall(pattern, output_text)

        bowl_boxes: List[Dict[str, float]] = []
        for match in matches:
            ymin, xmin, ymax, xmax = map(int, match)

            x_abs: float = xmin * img_width / 1000
            y_abs: float = ymin * img_height / 1000
            w_abs: float = (xmax - xmin) * img_width / 1000
            h_abs: float = (ymax - ymin) * img_height / 1000

            bowl_boxes.append({
                "x": x_abs,
                "y": y_abs,
                "w": w_abs,
                "h": h_abs
            })

        bowl_candidates: List[Tuple[Dict[str, float], Dict[str, float]]] = []
        for bowl in bowl_boxes:
            bowl_cx: float = bowl["x"] + bowl["w"] / 2
            bowl_cy: float = bowl["y"] + bowl["h"] / 2

            min_dist: float = float('inf')
            best_cat: Optional[Dict[str, float]] = None

            for cat in cat_boxes:
                cat_cx: float = cat["x"] + cat["w"] / 2
                cat_cy: float = cat["y"] + cat["h"] / 2
                dist: float = (
                                      (bowl_cx - cat_cx) ** 2 + (bowl_cy - cat_cy) ** 2
                              ) ** 0.5
                if dist < min_dist:
                    min_dist = dist
                    best_cat = cat

            if best_cat:
                bowl_candidates.append((bowl, best_cat))

        final_bowls: List[Dict[str, float]] = self._remove_overlapping_bowls(
            bowl_candidates
        )
        return final_bowls
