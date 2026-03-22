from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "wisconsin_breast_cancer.csv"
SPLITS_PATH = BASE_DIR / "data" / "splits.npz"

ARTIFACTS_DIR = BASE_DIR / "artifacts"
BASELINE_MODELS_DIR = ARTIFACTS_DIR / "baseline_models"
BASELINE_METRICS_PATH = ARTIFACTS_DIR / "baseline_metrics.json"
LOGS_DIR = ARTIFACTS_DIR / "logs"
TRAINING_LOG_PATH = LOGS_DIR / "training.jsonl"
EVALUATION_LOG_PATH = LOGS_DIR / "evaluation.jsonl"
LLM_STAGE_LOG_PATH = LOGS_DIR / "llm.jsonl"

TARGET_COL = "diagnosis"
ID_COLS = ["id", "ID number", "Unnamed: 32"]

RANDOM_STATE = 42
FITNESS_METRIC = "f1"

TEST_SIZE = 0.2
VAL_SIZE = 0.2

POSITIVE_LABEL = 1

USE_SCALER = True

FINETUNE_DIR           = BASE_DIR / "data" / "finetune"
FINETUNE_TRAIN_PATH    = FINETUNE_DIR / "train.jsonl"
FINETUNE_VAL_PATH      = FINETUNE_DIR / "val.jsonl"
FINETUNE_PREP_LOG_PATH = LOGS_DIR / "finetune_prep.jsonl"

FINETUNE_CHECKPOINTS_DIR   = ARTIFACTS_DIR / "finetune_checkpoints"
FINETUNE_FINAL_ADAPTER_DIR = FINETUNE_CHECKPOINTS_DIR / "final_adapter"
FINETUNE_LOG_PATH          = LOGS_DIR / "finetune.jsonl"

# ── Phase 3: paths ───────────────────────────────────────────────
DOCS_DIR              = BASE_DIR / "docs"
REPORTS_DIR           = DOCS_DIR / "reports"
KB_DIR                = BASE_DIR / "data" / "kb"
VECTORSTORE_DIR       = ARTIFACTS_DIR / "vectorstore"
ASSISTANT_LOG_PATH    = LOGS_DIR / "assistant.jsonl"
AUDIT_LOG_PATH        = LOGS_DIR / "audit.jsonl"
EXTERNAL_DATA_DIR     = BASE_DIR / "data" / "external"
PUBMEDQA_FILE         = EXTERNAL_DATA_DIR / "pubmedqa" / "data" / "ori_pqal.json"
MEDQUAD_DIR           = EXTERNAL_DATA_DIR / "medquad" / "1_CancerGov_QA"
BEST_MODEL_PATH       = ARTIFACTS_DIR / "models" / "best_model_with_threshold.joblib"
