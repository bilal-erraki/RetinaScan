"""Central configuration for RetinaScan paths and model metadata."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = Path(os.getenv("RETINASCAN_RAW_DATA_DIR", PROJECT_ROOT / "data" / "raw"))
PROCESSED_DATA_DIR = Path(
    os.getenv("RETINASCAN_PROCESSED_DATA_DIR", PROJECT_ROOT / "data" / "processed")
)
REPORTS_DIR = Path(os.getenv("RETINASCAN_REPORTS_DIR", PROJECT_ROOT / "reports"))
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = Path(os.getenv("RETINASCAN_MODELS_DIR", PROJECT_ROOT / "models"))
MODEL_PATH = MODELS_DIR / "best_model.keras"

LEGACY_DATA_DIR = PROJECT_ROOT / "aptos2019-blindness-detection"
DATASET_DIR = RAW_DATA_DIR / "aptos2019-blindness-detection"

IMAGE_SIZE = 300
BATCH_SIZE = 16
RANDOM_STATE = 42
CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]

MODEL_NAME = "RetinaScan EfficientNetB3"
MODEL_VERSION = "1.0.0"
MEDICAL_DISCLAIMER = "Academic demo only. Not a medical diagnostic device."

VALIDATION_ACCURACY = 0.7858
VALIDATION_LOSS = 0.6081
QUADRATIC_WEIGHTED_KAPPA = 0.8603
MACRO_F1 = 0.6021
WEIGHTED_F1 = 0.7773


def ensure_directories() -> None:
    """Create the small, generated project directories when they do not exist."""

    for directory in (
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        REPORTS_DIR,
        FIGURES_DIR,
        MODELS_DIR,
        PROJECT_ROOT / "notebooks",
    ):
        directory.mkdir(parents=True, exist_ok=True)
