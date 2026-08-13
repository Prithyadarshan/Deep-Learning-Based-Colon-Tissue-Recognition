import torch
import os

# -----------------------------
# Dataset
# -----------------------------

DATASET_ROOT = "dataset"

TRAIN_DIR = os.path.join(DATASET_ROOT, "train")
VAL_DIR = os.path.join(DATASET_ROOT, "val")
TEST_DIR = os.path.join(DATASET_ROOT, "test")

# -----------------------------
# Classes
# -----------------------------

CLASS_NAMES = [
    "ADI",
    "BACK",
    "DEB",
    "LYM",
    "MUC",
    "MUS",
    "NORM",
    "STR",
    "TUM"
]

NUM_CLASSES = len(CLASS_NAMES)

# -----------------------------
# Training
# -----------------------------

IMAGE_SIZE = 224

BATCH_SIZE = 32

NUM_EPOCHS = 30

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

PATIENCE = 7

NUM_WORKERS = 4

# -----------------------------
# Device
# -----------------------------

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# -----------------------------
# Paths
# -----------------------------

MODEL_DIR = "saved_models"

RESULT_DIR = "results"

LOG_DIR = "logs"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

BEST_MODEL = os.path.join(
    MODEL_DIR,
    "best_model.pth"
)

FINAL_MODEL = os.path.join(
    MODEL_DIR,
    "final_model.pth"
)
