from typing import Tuple
import torch
from enum import Enum


class PlannerType(Enum):
    QWEN = "qwen"
    GEMINI_LOCAL = "gemini_local"
    GEMINI_INSTANT = "gemini_instant"


class InpainterType(Enum):
    DIFFUSION = "diffusion"
    TEMPLATE = "template"
    FULL_TEMPLATE = "full_template"


DATA_DIR: str = "test_exam/data"
RESULT_DIR_STEP1: str = "test_exam/result1"
RESULT_DIR_STEP2: str = "test_exam/result2"
RESULT_DIR_FINAL: str = "test_exam/result_final"
VALID_EXTENSIONS: Tuple[str, ...] = (".jpg", ".jpeg", ".png")
DEVICE: str = "mps" if torch.backends.mps.is_available() else "cpu"

# Detection Hyperparameters
MODEL_PATH: str = "yolov8x.pt"
CAT_CLASS_ID: int = 15
DETECTION_CONF: float = 0.1
DETECTION_IMGSZ: int = 1280

# VLM Hyperparameters
VLM_MODEL_ID: str = "Qwen/Qwen2-VL-7B-Instruct"
GEMINI_MODEL_ID: str = "gemini-3-flash-preview"
PLANNER_TYPE: PlannerType = PlannerType.GEMINI_INSTANT
INPAINTER_TYPE: InpainterType = InpainterType.TEMPLATE
CROP_CONTEXT_MULTIPLIER: float = 2.5
CROP_Y_SHIFT_RATIO: float = 0.8
MIN_BOWL_W: float = 20.0
MIN_BOWL_H: float = 10.0

# Inpainting Hyperparameters
INPAINT_MODEL_ID: str = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
INPAINT_SIZE: int = 512
INPAINT_MASK_MARGIN_RATIO: float = 0.2
INPAINT_MASK_BLUR_KERNEL: int = 21
INPAINT_STRENGTH: float = 1.0
INPAINT_MIN_W: float = 35.0
INPAINT_MIN_H: float = 20.0

DEBUG_ENABLED: bool = True
DEBUG_DIR: str = "test_exam/debug"
DEBUG_MAX_ARTIFACTS_PER_IMAGE: int = 5

INPAINT_PROMPT: str = (
    "A realistic bowl of cat food placed on the ground, "
    "matching the lighting, perspective, and style of the scene"
)
INPAINT_NEGATIVE_PROMPT: str = (
    "bad quality, blurry, distorted, low resolution, deformed"
)
INPAINT_NUM_STEPS: int = 50
INPAINT_GUIDANCE_SCALE: float = 10.0
INPAINT_CROP_PADDING: int = 128

# Template Inpainter Hyperparameters
TEMPLATE_BOWL_DIR: str = "test_exam/bowls"
BOWL_SCALE_FACTOR: float = 1.2
TEMPLATE_NOISE_STRENGTH: float = 0.3
TEMPLATE_DENOISE_STEPS: int = 20
TEMPLATE_DENOISE_STRENGTH: float = 0.5
TEMPLATE_DENOISE_GUIDANCE: float = 7.5

TEMPLATE_INPAINT_STRENGTH: float = 0.45
TEMPLATE_INPAINT_NUM_STEPS: int = 25
TEMPLATE_INPAINT_GUIDANCE_SCALE: float = 6.0
TEMPLATE_INPAINT_PROMPT: str = (
    "small cat food bowl on the stone ground, realistic shadow, "
    "match lighting and perspective"
)
TEMPLATE_INPAINT_NEGATIVE_PROMPT: str = (
    "no stylization, no cartoon, no texture-only fill, no artifacts"
)

TEMPLATE_NOISE_SIGMA: float = 5.0
TEMPLATE_NOISE_ON_BOWL_ONLY: bool = True

TEMPLATE_MASK_MARGIN_RATIO: float = 0.45
TEMPLATE_MASK_BLUR_KERNEL: int = 31

# Visualization Hyperparameters
CAT_BOX_COLOR: Tuple[int, int, int] = (0, 255, 0)
BOWL_BOX_COLOR: Tuple[int, int, int] = (0, 0, 255)
BOX_THICKNESS: int = 2
TEXT_FONT_SCALE: float = 0.9
TEXT_THICKNESS: int = 2
CAT_LABEL: str = "cat"
BOWL_LABEL: str = "bowl"
