from typing import List, Dict, Optional, Any, Tuple

import cv2

from test_exam import config as config


def save_visualized_detections(
        image_path: str,
        cat_boxes: List[Dict[str, float]],
        result_path: str,
        bowl_boxes: Optional[List[Dict[str, float]]] = None
) -> None:
    image: Optional[Any] = cv2.imread(image_path)

    if image is None:
        print(f"Failed to load {image_path}")
        return

    add_boxes_to_img(cat_boxes, image, config.CAT_BOX_COLOR)

    if bowl_boxes:
        add_boxes_to_img(bowl_boxes, image, config.BOWL_BOX_COLOR)

    cv2.imwrite(result_path, image)
    print(f"Saved result to {result_path}")


def add_boxes_to_img(
        boxes: list[dict[str, float]],
        image: Any,
        color: Tuple[int, int, int]
) -> None:
    for box in boxes:
        x: int = int(box["x"])
        y: int = int(box["y"])
        w: int = int(box["w"])
        h: int = int(box["h"])

        cv2.rectangle(
            image, (x, y), (x + w, y + h),
            color, config.BOX_THICKNESS
        )
