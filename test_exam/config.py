from typing import Tuple
import torch

DATA_DIR: str = "test_exam/data"
RESULT_DIR_STEP1: str = "test_exam/result1"
RESULT_DIR_STEP2: str = "test_exam/result2"
VALID_EXTENSIONS: Tuple[str, ...] = (".jpg", ".jpeg", ".png")
DEVICE: str = "mps" if torch.backends.mps.is_available() else "cpu"

# Detection Hyperparameters
MODEL_PATH: str = "yolov8x.pt"
CAT_CLASS_ID: int = 15
DETECTION_CONF: float = 0.1
DETECTION_IMGSZ: int = 1280

# VLM Hyperparameters
VLM_MODEL_ID: str = "Qwen/Qwen2-VL-2B-Instruct"

# Visualization Hyperparameters
CAT_BOX_COLOR: Tuple[int, int, int] = (0, 255, 0)
BOWL_BOX_COLOR: Tuple[int, int, int] = (0, 0, 255)
BOX_THICKNESS: int = 1
TEXT_FONT_SCALE: float = 0.9
TEXT_THICKNESS: int = 1
CAT_LABEL: str = "cat"
BOWL_LABEL: str = "bowl"
