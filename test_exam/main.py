import os
from typing import List, Dict, Type
from test_exam.src.detector import CatDetector
from test_exam.src.planner import BowlPlanner
from test_exam.src.gemini_planner import GeminiBowlPlanner
from test_exam.src.gemini_instant_planner import GeminiInstantPlanner
from test_exam.src.inpainter import create_inpainter
import test_exam.config as config
from test_exam.config import PlannerType
from test_exam.src.utils import save_visualized_detections
from PIL import Image


def main() -> None:
    data_dir: str = config.DATA_DIR
    res1_dir: str = config.RESULT_DIR_STEP1
    res2_dir: str = config.RESULT_DIR_STEP2
    res_final_dir: str = config.RESULT_DIR_FINAL

    for d in [res1_dir, res2_dir, res_final_dir]:
        if not os.path.exists(d):
            os.makedirs(d)

    detector: CatDetector = CatDetector()

    if config.PLANNER_TYPE == PlannerType.GEMINI_LOCAL:
        planner_class: Type[BowlPlanner] = GeminiBowlPlanner
    elif config.PLANNER_TYPE == PlannerType.GEMINI_INSTANT:
        planner_class = GeminiInstantPlanner
    else:
        planner_class = BowlPlanner

    planner: BowlPlanner = planner_class()
    inpainter = create_inpainter()

    valid_exts: tuple[str, ...] = config.VALID_EXTENSIONS
    image_files: List[str] = [
        f for f in os.listdir(data_dir) if f.lower().endswith(valid_exts)
    ]

    for image_name in image_files:
        image_path: str = os.path.join(data_dir, image_name)
        print(f"Processing {image_path}...")

        cat_boxes: List[Dict[str, float]] = detector.detect(image_path)
        res1_path: str = os.path.join(res1_dir, image_name)
        save_visualized_detections(image_path, cat_boxes, res1_path)

        bowl_boxes: List[Dict[str, float]] = planner.plan_bowls(
            image_path, cat_boxes
        )
        res2_path: str = os.path.join(res2_dir, image_name)
        save_visualized_detections(
            image_path, cat_boxes, res2_path, bowl_boxes
        )

        final_image: Image.Image = inpainter.inpaint_bowls(
            image_path, bowl_boxes
        )
        res_final_path: str = os.path.join(res_final_dir, image_name)
        final_image.save(res_final_path)
        print(f"Saved final result to {res_final_path}")


if __name__ == "__main__":
    main()
