from typing import Tuple

DATA_DIR: str = "test_exam/data"
RESULT_DIR: str = "test_exam/result1"
VALID_EXTENSIONS: Tuple[str, ...] = (".jpg", ".jpeg", ".png")

# Detection Hyperparameters
MODEL_PATH: str = "yolov8x.pt"
CAT_CLASS_ID: int = 15
DETECTION_CONF: float = 0.009
DETECTION_IMGSZ: int = 1280

# Visualization Hyperparameters
BOX_COLOR: Tuple[int, int, int] = (0, 255, 0)
BOX_THICKNESS: int = 2
TEXT_FONT_SCALE: float = 0.9
TEXT_THICKNESS: int = 2
LABEL_TEXT: str = "cat"
