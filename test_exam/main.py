import os
import cv2
from typing import List, Dict, Any, Optional
from test_exam.src.detector import CatDetector
import test_exam.config as config


def save_visualized_detections(
    image_path: str,
    boxes: List[Dict[str, float]],
    result_path: str
) -> None:
    image: Optional[Any] = cv2.imread(image_path)

    if image is None:
        print(f"Failed to load {image_path}")
        return

    for box in boxes:
        x: int = int(box["x"])
        y: int = int(box["y"])
        w: int = int(box["w"])
        h: int = int(box["h"])

        cv2.rectangle(
            image, (x, y), (x + w, y + h),
            config.BOX_COLOR, config.BOX_THICKNESS
        )

    cv2.imwrite(result_path, image)
    print(f"Saved result to {result_path}")


def main() -> None:
    data_dir: str = config.DATA_DIR
    result_dir: str = config.RESULT_DIR

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    detector: CatDetector = CatDetector()

    # Supported image extensions
    valid_exts: tuple[str, ...] = config.VALID_EXTENSIONS

    image_files: List[str] = [
        f for f in os.listdir(data_dir) if f.lower().endswith(valid_exts)
    ]

    for image_name in image_files:
        image_path: str = os.path.join(data_dir, image_name)
        print(f"Processing {image_path}...")

        boxes: List[Dict[str, float]] = detector.detect(image_path)

        result_path: str = os.path.join(result_dir, image_name)
        save_visualized_detections(image_path, boxes, result_path)


if __name__ == "__main__":
    main()
