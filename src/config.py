from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "wisconsin_breast_cancer.csv"
SPLITS_PATH = BASE_DIR / "data" / "splits.npz"

ARTIFACTS_DIR = BASE_DIR / "artifacts"
BASELINE_MODELS_DIR = ARTIFACTS_DIR / "baseline_models"
BASELINE_METRICS_PATH = ARTIFACTS_DIR / "baseline_metrics.json"

TARGET_COL = "diagnosis"
ID_COLS = ["id", "ID number", "Unnamed: 32"]

RANDOM_STATE = 42
FITNESS_METRIC = "f1"

TEST_SIZE = 0.2
VAL_SIZE = 0.2

POSITIVE_LABEL = 1

USE_SCALER = True
