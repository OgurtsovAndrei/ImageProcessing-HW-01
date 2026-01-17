from typing import List, Dict, Any
from ultralytics import YOLO
import numpy as np
import test_exam.config as config


class CatDetector:
    def __init__(self, model_path: str = config.MODEL_PATH):
        self.model: YOLO = YOLO(model_path)
        self.cat_class_id: int = config.CAT_CLASS_ID

    def detect(
        self,
        image_path: str,
        conf: float = config.DETECTION_CONF,
        imgsz: int = config.DETECTION_IMGSZ,
        device: str = config.DEVICE
    ) -> List[Dict[str, float]]:
        """
        Detects cats in the image and returns a list of bounding boxes.
        Each box is a dictionary with keys 'x', 'y', 'w', 'h' in absolute
        pixel coordinates. 'x', 'y' are the top-left corner.
        """
        results: Any = self.model.predict(
            image_path,
            classes=[self.cat_class_id],
            conf=conf,
            imgsz=imgsz,
            device=device,
            verbose=False
        )
        boxes: List[Dict[str, float]] = []

        if len(results) > 0:
            result = results[0]
            # result.boxes.xywh contains [x_center, y_center, width, height]
            # but we want top-left x, y
            for box in result.boxes:
                # xyxy is [x1, y1, x2, y2]
                xyxy: np.ndarray = box.xyxy[0].cpu().numpy()
                x1: float = float(xyxy[0])
                y1: float = float(xyxy[1])
                x2: float = float(xyxy[2])
                y2: float = float(xyxy[3])

                boxes.append({
                    "x": x1,
                    "y": y1,
                    "w": x2 - x1,
                    "h": y2 - y1
                })

        return boxes
