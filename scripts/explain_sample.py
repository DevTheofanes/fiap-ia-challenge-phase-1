from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
from sklearn.inspection import permutation_importance

BASE_DIR = Path(__file__).resolve().parent.parent
import sys

sys.path.insert(0, str(BASE_DIR))

from src import config
from src.data.load import load_dataset
from src.data.split import apply_splits, make_or_load_splits
from src.llm.explain import build_case_payload, explain_case
from src.logging_utils import log_event, setup_json_logger


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explain a single sample with LLM.")
    parser.add_argument("--bundle-path", type=Path, default=None)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--sample-idx", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--importance", choices=["auto", "coef", "tree", "permutation"], default="auto")
    return parser.parse_args()


def _default_bundle_path() -> Path:
    candidates = [
        config.ARTIFACTS_DIR / "best_model_with_threshold.joblib",
        BASE_DIR / "best_model_with_threshold.joblib",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _unwrap_estimator(model):
    if hasattr(model, "named_steps"):
        return list(model.named_steps.values())[-1]
    return model


def _infer_model_name(model) -> str:
    estimator = _unwrap_estimator(model)
    name = estimator.__class__.__name__
    if name == "LogisticRegression":
        return "LR"
    if name == "RandomForestClassifier":
        return "RF"
    return name


def _compute_importance(model, X_ref, y_ref, feature_names, method: str) -> tuple[np.ndarray, str]:
    estimator = _unwrap_estimator(model)
    if method in ("auto", "coef") and hasattr(estimator, "coef_"):
        coef = np.array(estimator.coef_).ravel()
        return np.abs(coef), "coef"
    if method in ("auto", "tree") and hasattr(estimator, "feature_importances_"):
        return np.array(estimator.feature_importances_), "tree"

    result = permutation_importance(
        model,
        X_ref,
        y_ref,
        n_repeats=10,
        random_state=config.RANDOM_STATE,
        scoring="f1",
    )
    return np.array(result.importances_mean), "permutation"


def _predict_proba(model, X_row):
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(X_row)[0, 1])
    if hasattr(model, "decision_function"):
        score = float(model.decision_function(X_row)[0])
        return 1.0 / (1.0 + np.exp(-score))
    return None


def _predict_label(model, X_row, threshold: float | None, prob: float | None) -> int:
    if prob is not None and threshold is not None:
        return int(prob >= threshold)
    return int(model.predict(X_row)[0])


def _label_to_text(label: int) -> str:
    return "malignant" if label == 1 else "benign"


def main() -> None:
    args = _parse_args()
    bundle_path = args.bundle_path or _default_bundle_path()
    if not bundle_path.exists():
        raise FileNotFoundError(
            f"Model bundle not found: {bundle_path}. Generate best_model_with_threshold.joblib first."
        )
    llm_logger = setup_json_logger("llm.explain", config.LLM_STAGE_LOG_PATH)

    bundle = joblib.load(bundle_path)
    if isinstance(bundle, dict) and "model" in bundle:
        model = bundle["model"]
        threshold = float(bundle.get("threshold")) if bundle.get("threshold") is not None else None
        feature_names = bundle.get("features")
    else:
        model = bundle
        threshold = None
        feature_names = None

    X, y = load_dataset(config.DATA_PATH, config.TARGET_COL, config.ID_COLS)
    splits = make_or_load_splits(
        n_samples=len(X),
        y=y,
        test_size=config.TEST_SIZE,
        val_size=config.VAL_SIZE,
        random_state=config.RANDOM_STATE,
        splits_path=config.SPLITS_PATH,
    )
    X_train, X_val, X_test, y_train, y_val, y_test = apply_splits(X, y, splits)

    X_ref, y_ref = (X_val, y_val) if args.split == "val" else (X_test, y_test)
    if args.sample_idx < 0 or args.sample_idx >= len(X_ref):
        raise IndexError(f"sample-idx out of range for {args.split} split")

    X_row = X_ref.iloc[[args.sample_idx]].copy()
    prob = _predict_proba(model, X_row)
    label = _predict_label(model, X_row, threshold, prob)
    diagnosis = _label_to_text(label)

    feature_names = feature_names or list(X_ref.columns)
    importances, importance_type = _compute_importance(
        model, X_ref, y_ref, feature_names, method=args.importance
    )

    ranked_idx = np.argsort(importances)[::-1][: args.top_k]
    top_features = []
    for idx in ranked_idx:
        feature = feature_names[idx]
        value = float(X_row.iloc[0][feature])
        top_features.append(
            {
                "feature": feature,
                "importance": float(importances[idx]),
                "value": value,
                "importance_type": importance_type,
            }
        )

    payload = build_case_payload(
        diagnosis=diagnosis,
        probability=prob,
        threshold=threshold,
        threshold_reason="selected on validation metric" if threshold is not None else "model default",
        top_features=top_features,
        safety_notes=None,
    )

    log_path = config.ARTIFACTS_DIR / "llm" / "llm_logs.jsonl"
    start = time.time()
    explanation = explain_case(payload, log_path=log_path)
    duration_sec = time.time() - start

    output_path = config.ARTIFACTS_DIR / "llm" / "sample_explanations.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "bundle_path": str(bundle_path),
        "split": args.split,
        "sample_idx": args.sample_idx,
        "payload": asdict(payload),
        "explanation": asdict(explanation),
    }
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    log_event(
        llm_logger,
        "llm_explain",
        model=_infer_model_name(model),
        experiment="baseline",
        seed=config.RANDOM_STATE,
        split=args.split,
        sample_idx=args.sample_idx,
        duration_sec=duration_sec,
    )

    print(json.dumps(entry, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
