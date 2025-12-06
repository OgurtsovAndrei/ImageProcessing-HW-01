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
MODEL_INPUT_SIZE = (160, 160, 160)  # (depth, height, width) - resize volumes to this
IN_CHANNELS = 1  # grayscale
OUT_CHANNELS = 2  # background + papyrus (ignore class 2)

# Training
BATCH_SIZE = 2
NUM_WORKERS = 2
MAX_EPOCHS = 8
LEARNING_RATE = 2e-3
