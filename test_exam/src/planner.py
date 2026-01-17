import re
import torch
from PIL import Image
from typing import List, Dict, Any, Tuple
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
        width: int = image.width
        height: int = image.height

        cat_info: str = ""
        for i, box in enumerate(cat_boxes):
            y1: int = int(box["y"] * 1000 / height)
            x1: int = int(box["x"] * 1000 / width)
            y2: int = int((box["y"] + box["h"]) * 1000 / height)
            x2: int = int((box["x"] + box["w"]) * 1000 / width)
            cat_info += f"cat{i + 1}: [{y1},{x1},{y2},{x2}] "

        prompt: str = (
            f"Cats are located at the following bounding boxes:\n{cat_info}\n\n"

            "Your task is to propose bounding boxes for food bowls for the cats.\n\n"

            "Rules:\n"
            "- Each bowl must be placed NEAR its corresponding cat, but MUST NOT overlap with the cat bounding box.\n"
            "- The bowl bounding box must be strictly outside the cat bounding box.\n"
            "- Bowls should be placed on the ground, typically below or slightly in front of the cat.\n"
            "- Bowl bounding boxes must be smaller than the corresponding cat bounding box.\n"
            "- Bowl bounding boxes must not overlap with each other.\n"
            "- If a valid non-overlapping placement is not possible for a cat, do NOT output a bowl for that cat.\n\n"

            "Important:\n"
            "- Do NOT place bowls on top of cats.\n"
            "- Do NOT place bowls inside cat bounding boxes.\n"

            "Output format (no extra text):\n"
            "bowl1: [ymin, xmin, ymax, xmax]\n"
            "bowl2: [ymin, xmin, ymax, xmax]\n"
        )

        messages: List[Dict[str, Any]] = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
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

        bowl_boxes: List[Dict[str, float]] = []
        pattern: str = r"\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]"
        matches: List[Tuple[str, ...]] = re.findall(pattern, output_text)

        for match in matches:
            ymin, xmin, ymax, xmax = map(int, match)

            x_abs: float = xmin * width / 1000
            y_abs: float = ymin * height / 1000
            w_abs: float = (xmax - xmin) * width / 1000
            h_abs: float = (ymax - ymin) * height / 1000

            bowl_boxes.append({
                "x": x_abs,
                "y": y_abs,
                "w": w_abs,
                "h": h_abs
            })

        return bowl_boxes
