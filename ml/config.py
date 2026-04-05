"""
config.py — Single source of truth for all hyperparameters and paths.

── Local (M4 MacBook) ──────────────────────────────────────────────────────
    python train.py                          (all defaults below)

── Cloud (Kaggle / Colab T4/A100) ──────────────────────────────────────────
    export MODEL_NAME=tf_efficientnetv2_m    # or convnext_base
    export IMG_SIZE=384                      # or 480 for V2-M
    export BATCH_SIZE=16                     # reduce if OOM
    export NUM_WORKERS=8
    export DATA_ROOT=/kaggle/input/forearm-fracture-xrays/alle Bilder
    python train.py

All other knobs (LR, dropout, augmentation scale, …) are also overridable
via the same environment-variable pattern.
"""

import os
import torch
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_ROOT = Path(
    os.getenv(
        "DATA_ROOT",
        "/Users/venthanvigneswaran/Library/Mobile Documents/"
        "com~apple~CloudDocs/neueste bilder/Promotion/alle Bilder",
    )
)

PROJECT_ROOT = Path(__file__).parent.parent  # ml/../ = project root

# On Kaggle the script may live inside a read-only input dataset.
# Always write to /kaggle/working/ so checkpoints survive the session
# and appear in the notebook Output tab for download.
_KAGGLE_WORKING = Path("/kaggle/working")
_ON_KAGGLE      = _KAGGLE_WORKING.exists()
_WRITE_ROOT     = _KAGGLE_WORKING if _ON_KAGGLE else PROJECT_ROOT

SPLITS_CSV     = Path(os.getenv("SPLITS_CSV",      str(_WRITE_ROOT / "data"        / "splits.csv")))
CHECKPOINT_DIR = Path(os.getenv("CHECKPOINT_DIR",  str(_WRITE_ROOT / "checkpoints")))
LOG_DIR        = Path(os.getenv("LOG_DIR",          str(_WRITE_ROOT / "logs")))

# ── Device ─────────────────────────────────────────────────────────────────
# Auto-detect: CUDA (cloud GPU) → MPS (Apple Silicon) → CPU (fallback)
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

# ── Model ──────────────────────────────────────────────────────────────────
# Local:  "efficientnet_b0"      → IMG_SIZE=224, BATCH_SIZE=32  (~50–80s/epoch on M4)
# Cloud:  "tf_efficientnetv2_m"  → IMG_SIZE=480, BATCH_SIZE=16  (~3–4 min/epoch on T4)
MODEL_NAME  = os.getenv("MODEL_NAME",  "efficientnet_b0")
IMG_SIZE    = int(os.getenv("IMG_SIZE",    "224"))
BATCH_SIZE  = int(os.getenv("BATCH_SIZE",  "32"))
NUM_CLASSES = 2  # fracture (1) vs normal (0)

# ── Data Split ─────────────────────────────────────────────────────────────
TRAIN_FRAC  = 0.80
VAL_FRAC    = 0.10
TEST_FRAC   = 0.10
RANDOM_SEED = 42

# ── DataLoader ─────────────────────────────────────────────────────────────
# PyTorch 2.x fixed MPS multiprocessing — num_workers=4 is safe on Apple Silicon
NUM_WORKERS = int(os.getenv("NUM_WORKERS", "4"))
PIN_MEMORY  = DEVICE == "cuda"

# ── Augmentation ───────────────────────────────────────────────────────────
AUG_HFLIP_PROB   = 0.5
AUG_ROTATION_DEG = 15
AUG_BRIGHTNESS   = 0.2
AUG_CONTRAST     = 0.2
# RandomResizedCrop scale range — lower bound controls how aggressively the
# image is zoomed in. 0.6 retains at least 60% of area; improves generalisation
# vs a fixed Resize which sees the same field-of-view every epoch.
CROP_SCALE_MIN   = float(os.getenv("CROP_SCALE_MIN", "0.6"))

# ── Normalisation (ImageNet statistics) ────────────────────────────────────
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD  = [0.229, 0.224, 0.225]

# ── Evaluation resize ratio (ImageNet standard: resize to 256, crop to 224) ─
EVAL_RESIZE_RATIO = 256 / 224  # e.g. IMG_SIZE=224 → resize 256 → crop 224

# ── Inference defaults ──────────────────────────────────────────────────────
DEFAULT_THRESHOLD  = 0.5   # fallback when optimal threshold is unavailable
UNCERTAINTY_MARGIN = 0.05  # ±5% around threshold → warn radiologist

# ── Grad-CAM ────────────────────────────────────────────────────────────────
GRADCAM_IMAGE_WEIGHT = 0.4  # fraction of original in overlay (lower = more CAM)

# ── Loss ───────────────────────────────────────────────────────────────────
LABEL_SMOOTHING = 0.1  # regularises predictions; improves calibration for medical use

# ── Phase 1: frozen backbone, train head only ──────────────────────────────
PHASE1_EPOCHS       = 20
PHASE1_LR           = 1e-3
PHASE1_WEIGHT_DECAY = 1e-4

# ── Phase 2: unfreeze last N backbone blocks, fine-tune ───────────────────
PHASE2_EPOCHS       = 40
PHASE2_LR           = 1e-5   # 100× lower than Phase1 — prevents catastrophic forgetting
PHASE2_WEIGHT_DECAY = 1e-4
UNFREEZE_LAST_N     = 2      # unfreeze last 2 block groups of the backbone

# ── Early Stopping ─────────────────────────────────────────────────────────
EARLY_STOP_PATIENCE  = 10    # epochs without val AUC improvement before stopping
EARLY_STOP_MIN_DELTA = 0.001 # minimum improvement threshold

# ── Head Architecture ──────────────────────────────────────────────────────
HEAD_HIDDEN_DIM = 256   # intermediate linear layer size in the classification head
# Applied to both dropout layers in the classification head.
# 0.4 adds stronger regularisation; tune via HPO or HEAD_DROPOUT env var.
HEAD_DROPOUT = float(os.getenv("HEAD_DROPOUT", "0.4"))

# ── Gradient Clipping ───────────────────────────────────────────────────────
GRAD_CLIP_MAX_NORM = 1.0  # prevents exploding gradients during Phase 2 fine-tuning

# ── LR Scheduler ───────────────────────────────────────────────────────────
# Phase 1 (head only): CosineAnnealingLR — smooth warm-up, fixed budget.
# Phase 2 (fine-tune): ReduceLROnPlateau — adapts to val AUC stagnation.
USE_SCHEDULER       = True
P2_SCHEDULER        = "plateau"  # "plateau" | "cosine" | "none"
P2_PLATEAU_FACTOR   = 0.2        # multiply LR by this when plateau is detected
P2_PLATEAU_PATIENCE = 3          # epochs of no val AUC improvement before reducing
