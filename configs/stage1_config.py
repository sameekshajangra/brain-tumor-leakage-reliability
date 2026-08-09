"""
configs/stage1_config.py
========================
Central configuration for Stage 1 of the brain-tumor patient-overlap study.

All paths are resolved relative to the project root or via DATA_DIR so that
the scripts can be run from any working directory.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# The raw dataset directory: the folder that contains the 4 ZIP files and
# cvind.mat.  Override via the environment variable BRAIN_TUMOR_DATA_DIR.
DATA_DIR = Path(
    os.environ.get(
        "BRAIN_TUMOR_DATA_DIR",
        "/Users/sameekshasharma/Desktop/brain-tumor",
    )
)

# Where the .mat files will be extracted
RAW_MAT_DIR = DATA_DIR / "data" / "raw"

# cvind.mat is already present at the root of the dataset directory
CVIND_PATH = DATA_DIR / "cvind.mat"

# Project root (parent of this configs/ directory)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_DIR   = PROJECT_ROOT / "results"  / "stage1"
FIGURES_DIR   = PROJECT_ROOT / "figures"  / "stage1"
TABLES_DIR    = PROJECT_ROOT / "tables"   / "stage1"

# ---------------------------------------------------------------------------
# Dataset ground-truth expectations (for automated verification)
# ---------------------------------------------------------------------------
EXPECTED_TOTAL_IMAGES   = 3064
EXPECTED_TOTAL_PATIENTS = 233
EXPECTED_CLASS_COUNTS   = {
    1: 708,   # Meningioma
    2: 1426,  # Glioma
    3: 930,   # Pituitary
}
EXPECTED_NUM_FOLDS = 5

CLASS_NAMES = {
    1: "Meningioma",
    2: "Glioma",
    3: "Pituitary",
}
VALID_LABELS = set(CLASS_NAMES.keys())

# ---------------------------------------------------------------------------
# Experiment design
# ---------------------------------------------------------------------------
# The fixed test fold for the later experiment
TEST_FOLD = 5

# ---------------------------------------------------------------------------
# ZIP files (resolved relative to DATA_DIR)
# ---------------------------------------------------------------------------
ZIP_FILENAMES = [
    "brainTumorDataPublic_1766.zip",
    "brainTumorDataPublic_7671532.zip",
    "brainTumorDataPublic_15332298.zip",
    "brainTumorDataPublic_22993064.zip",
]
