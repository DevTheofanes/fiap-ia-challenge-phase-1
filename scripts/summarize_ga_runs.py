from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
import sys

sys.path.insert(0, str(BASE_DIR))

from src import config
from src.data.load import load_dataset
from src.data.split import apply_splits, make_or_load_splits
from src.models.train import evaluate_on_test


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize GA experiment runs.")
    parser.add_argument("--model", choices=["LR", "RF", "all"], default="all")
    parser.add_argument("--runs-root", type=Path, default=config.ARTIFACTS_DIR / "ga_runs")
    parser.add_argument("--summary-root", type=Path, default=config.ARTIFACTS_DIR / "ga_summary")
    return parser.parse_args()


def _load_best_payload(best_path: Path) -> dict | None:
    try:
        return json.loads(best_path.read_text())
    except Exception:
        return None


def _load_data_splits():
    X, y = load_dataset(config.DATA_PATH, config.TARGET_COL, config.ID_COLS)
    splits = make_or_load_splits(
        n_samples=len(X),
        y=y,
        test_size=config.TEST_SIZE,
        val_size=config.VAL_SIZE,
        random_state=config.RANDOM_STATE,
        splits_path=config.SPLITS_PATH,
    )
    return apply_splits(X, y, splits)


def _recompute_holdout_metrics(model_path: Path, holdout: str, data_splits):
    if not model_path.exists():
        return {}
    X_train, X_val, X_test, y_train, y_val, y_test = data_splits
    model = joblib.load(model_path)
    if holdout == "val":
        return evaluate_on_test(model, X_val, y_val)
    return evaluate_on_test(model, X_test, y_test)


def _collect_runs(runs_root: Path, model: str, data_splits) -> list[dict]:
    runs = []
    model_dir = runs_root / model
    if not model_dir.exists():
        return runs

    for best_path in model_dir.glob("exp*_seed*/best.json"):
        payload = _load_best_payload(best_path)
        if not payload:
            continue
        config_payload = payload.get("config", {})
        holdout_split = payload.get("holdout") or "val"
        holdout_metrics = payload.get("holdout_metrics", {})
        recomputed = _recompute_holdout_metrics(best_path.parent / "best_model.joblib", holdout_split, data_splits)
        metrics = recomputed or holdout_metrics
        params = payload.get("decoded_params", {})
        runs.append(
            {
                "model": model,
                "exp_id": config_payload.get("exp_id"),
                "seed": config_payload.get("seed"),
                "best_cv_f1": payload.get("best_cv_f1"),
                "best_holdout_f1": metrics.get("f1"),
                "holdout_accuracy": metrics.get("accuracy"),
                "holdout_precision": metrics.get("precision"),
                "holdout_recall": metrics.get("recall"),
                "holdout_roc_auc": metrics.get("roc_auc"),
                "holdout_pr_auc": metrics.get("pr_auc"),
                "holdout_confusion_matrix": json.dumps(metrics.get("confusion_matrix", [])),
                "holdout_split": holdout_split,
                "training_time_sec": payload.get("training_time_sec"),
                "params": json.dumps(params, sort_keys=True),
                "run_dir": str(best_path.parent),
            }
        )
    return runs


def _write_summary(summary_root: Path, model: str, rows: list[dict]) -> Path:
    summary_root.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    out_path = summary_root / f"summary_{model}.csv"
    df.to_csv(out_path, index=False)
    return out_path


def _load_baseline_metrics() -> dict | None:
    if not config.BASELINE_METRICS_PATH.exists():
        return None
    try:
        return json.loads(config.BASELINE_METRICS_PATH.read_text())
    except Exception:
        return None


def _compute_baseline_metrics(data_splits):
    X_train, X_val, X_test, y_train, y_val, y_test = data_splits
    metrics = {}
    for model_key in ["LR", "RF"]:
        model_path = config.BASELINE_MODELS_DIR / f"{model_key}.joblib"
        if not model_path.exists():
            continue
        model = joblib.load(model_path)
        metrics[model_key] = {
            "val_metrics": evaluate_on_test(model, X_val, y_val),
            "test_metrics": evaluate_on_test(model, X_test, y_test),
        }
    return metrics


def _build_compare_table(summary_rows: list[dict], baseline: dict | None) -> pd.DataFrame:
    rows = []
    if not summary_rows:
        return pd.DataFrame(rows)

    by_model = {}
    for row in summary_rows:
        by_model.setdefault(row["model"], []).append(row)

    for model, items in by_model.items():
        best = max(
            items,
            key=lambda r: (
                r.get("best_holdout_f1") or 0.0,
                r.get("best_cv_f1") or 0.0,
            ),
        )
        holdout_split = best.get("holdout_split") or "val"
        baseline_metrics = {}
        baseline_params = {}
        if baseline and model in baseline.get("models", {}):
            metrics_key = "val_metrics" if holdout_split == "val" else "test_metrics"
            baseline_metrics = baseline["models"][model].get(metrics_key, {})
            baseline_params = baseline["models"][model].get("params", {})
        rows.append(
            {
                "model": model,
                "approach": "Baseline",
                "params": json.dumps(baseline_params, sort_keys=True),
                "f1_cv": "",
                "f1_holdout": baseline_metrics.get("f1"),
                "accuracy_holdout": baseline_metrics.get("accuracy"),
                "precision_holdout": baseline_metrics.get("precision"),
                "recall_holdout": baseline_metrics.get("recall"),
                "roc_auc_holdout": baseline_metrics.get("roc_auc"),
                "pr_auc_holdout": baseline_metrics.get("pr_auc"),
                "notes": f"baseline_{holdout_split}",
                "source_run": "",
            }
        )
        rows.append(
            {
                "model": model,
                "approach": "GA",
                "params": best.get("params"),
                "f1_cv": best.get("best_cv_f1"),
                "f1_holdout": best.get("best_holdout_f1"),
                "accuracy_holdout": best.get("holdout_accuracy"),
                "precision_holdout": best.get("holdout_precision"),
                "recall_holdout": best.get("holdout_recall"),
                "roc_auc_holdout": best.get("holdout_roc_auc"),
                "pr_auc_holdout": best.get("holdout_pr_auc"),
                "notes": f"exp{best.get('exp_id')}_seed{best.get('seed')}",
                "source_run": best.get("run_dir"),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    args = _parse_args()
    models = ["LR", "RF"] if args.model == "all" else [args.model]

    data_splits = _load_data_splits()
    all_rows = []
    for model in models:
        rows = _collect_runs(args.runs_root, model, data_splits)
        all_rows.extend(rows)
        _write_summary(args.summary_root, model, rows)

    baseline = _load_baseline_metrics() or {"models": {}}
    baseline_updates = _compute_baseline_metrics(data_splits)
    for model_key, payload in baseline_updates.items():
        baseline.setdefault("models", {}).setdefault(model_key, {})
        for split_key in ["val_metrics", "test_metrics"]:
            current = baseline["models"][model_key].get(split_key, {})
            recomputed = payload.get(split_key, {})
            merged = {**current, **{k: v for k, v in recomputed.items() if k not in current}}
            baseline["models"][model_key][split_key] = merged
    compare = _build_compare_table(all_rows, baseline)
    if not compare.empty:
        args.summary_root.mkdir(parents=True, exist_ok=True)
        compare.to_csv(args.summary_root / "compare_baseline_vs_ga.csv", index=False)

    print(f"GA summaries saved to {args.summary_root}")


if __name__ == "__main__":
    main()
