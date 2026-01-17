import os
from typing import List, Dict
from test_exam.src.detector import CatDetector
from test_exam.src.planner import BowlPlanner
import test_exam.config as config
from test_exam.src.utils import save_visualized_detections


def main() -> None:
    data_dir: str = config.DATA_DIR
    res1_dir: str = config.RESULT_DIR_STEP1
    res2_dir: str = config.RESULT_DIR_STEP2

    for d in [res1_dir, res2_dir]:
        if not os.path.exists(d):
            os.makedirs(d)

    detector: CatDetector = CatDetector()
    planner: BowlPlanner = BowlPlanner()

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


if __name__ == "__main__":
    main()
