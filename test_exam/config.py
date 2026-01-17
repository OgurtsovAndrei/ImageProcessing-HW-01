from typing import Tuple
import torch
from enum import Enum


class PlannerType(Enum):
    QWEN = "qwen"
    GEMINI_LOCAL = "gemini_local"
    GEMINI_INSTANT = "gemini_instant"


DATA_DIR: str = "test_exam/data"
RESULT_DIR_STEP1: str = "test_exam/result1"
RESULT_DIR_STEP2: str = "test_exam/result2"
RESULT_DIR_FINAL: str = "test_exam/result_final"
VALID_EXTENSIONS: Tuple[str, ...] = (".jpg", ".jpeg", ".png")
DEVICE: str = "mps" if torch.backends.mps.is_available() else "cpu"

# Detection Hyperparameters
MODEL_PATH: str = "yolov8x.pt"
CAT_CLASS_ID: int = 15
DETECTION_CONF: float = 0.9
DETECTION_IMGSZ: int = 1280

# VLM Hyperparameters
VLM_MODEL_ID: str = "Qwen/Qwen2-VL-7B-Instruct"
GEMINI_MODEL_ID: str = "gemini-3-flash-preview"
PLANNER_TYPE: PlannerType = PlannerType.GEMINI_INSTANT
CROP_CONTEXT_MULTIPLIER: float = 2.5

# Inpainting Hyperparameters
INPAINT_MODEL_ID: str = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
INPAINT_PROMPT: str = (
    "A realistic bowl of cat food placed on the ground, "
    "matching the lighting, perspective, and style of the scene"
)
INPAINT_NEGATIVE_PROMPT: str = (
    "bad quality, blurry, distorted, low resolution, deformed"
)
INPAINT_NUM_STEPS: int = 30
INPAINT_GUIDANCE_SCALE: float = 7.5
INPAINT_CROP_PADDING: int = 64

# Visualization Hyperparameters
CAT_BOX_COLOR: Tuple[int, int, int] = (0, 255, 0)
BOWL_BOX_COLOR: Tuple[int, int, int] = (0, 0, 255)
BOX_THICKNESS: int = 2
TEXT_FONT_SCALE: float = 0.9
TEXT_THICKNESS: int = 2
CAT_LABEL: str = "cat"
BOWL_LABEL: str = "bowl"
