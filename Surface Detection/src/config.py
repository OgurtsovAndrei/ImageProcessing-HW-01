from pathlib import Path
import torch

# Paths
DATA_DIR = Path("data/vesuvius-challenge-surface-detection")
CHECKPOINT_DIR = Path("data/checkpoints")
TRAIN_IMAGES_DIR = Path.joinpath(DATA_DIR, Path("train_images"))
TRAIN_LABELS_DIR = Path.joinpath(DATA_DIR, Path("train_labels"))
TEST_IMAGES_DIR = Path.joinpath(DATA_DIR, Path("test_images"))
OUTPUT_DIR = Path("data/out")

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# Model architecture
MODEL_NAME = "SegResNet"  # Or "SegResNet" - could be made configurable
PATCH_SIZE = (96, 96, 96)
MODEL_INPUT_SIZE = PATCH_SIZE
SW_ROI_SIZE = PATCH_SIZE
SW_OVERLAP = 0.1
SW_OVERLAP_TEST = 0.5
IN_CHANNELS = 1  # grayscale
OUT_CHANNELS = 2  # background + papyrus (ignore class 2)

# Training
BATCH_SIZE = 4
SAMPLES_PER_VOLUME = 4
NUM_WORKERS = 2
MAX_EPOCHS = 10
LEARNING_RATE = 2e-3

# dataset
USE_RATIO = 1.0

# Validation
VALIDATION_TTA = False
TEST_TTA = True