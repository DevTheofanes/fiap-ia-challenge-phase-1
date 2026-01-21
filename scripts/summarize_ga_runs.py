from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
import sys

sys.path.insert(0, str(BASE_DIR))

from src import config


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


def _collect_runs(runs_root: Path, model: str) -> list[dict]:
    runs = []
    model_dir = runs_root / model
    if not model_dir.exists():
        return runs

    for best_path in model_dir.glob("exp*_seed*/best.json"):
        payload = _load_best_payload(best_path)
        if not payload:
            continue
        config_payload = payload.get("config", {})
        holdout_metrics = payload.get("holdout_metrics", {})
        runs.append(
            {
                "model": model,
                "exp_id": config_payload.get("exp_id"),
                "seed": config_payload.get("seed"),
                "best_cv_f1": payload.get("best_cv_f1"),
                "best_holdout_f1": holdout_metrics.get("f1"),
                "holdout_recall": holdout_metrics.get("recall"),
                "holdout_split": payload.get("holdout"),
                "training_time_sec": payload.get("training_time_sec"),
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


def _build_compare_table(summary_rows: list[dict], baseline: dict | None) -> pd.DataFrame:
    rows = []
    if not summary_rows:
        return pd.DataFrame(rows)

    by_model = {}
    for row in summary_rows:
        by_model.setdefault(row["model"], []).append(row)

    for model, items in by_model.items():
        best = max(items, key=lambda r: (r.get("best_holdout_f1") or 0.0, r.get("best_cv_f1") or 0.0))
        holdout_split = best.get("holdout_split") or "val"
        baseline_metrics = {}
        if baseline and model in baseline.get("models", {}):
            metrics_key = "val_metrics" if holdout_split == "val" else "test_metrics"
            baseline_metrics = baseline["models"][model].get(metrics_key, {})
        rows.append(
            {
                "model": model,
                "approach": "Baseline",
                "f1_cv": "",
                "f1_holdout": baseline_metrics.get("f1"),
                "recall_holdout": baseline_metrics.get("recall"),
                "notes": f"baseline_{holdout_split}",
                "source_run": "",
            }
        )
        rows.append(
            {
                "model": model,
                "approach": "GA",
                "f1_cv": best.get("best_cv_f1"),
                "f1_holdout": best.get("best_holdout_f1"),
                "recall_holdout": best.get("holdout_recall"),
                "notes": f"exp{best.get('exp_id')}_seed{best.get('seed')}",
                "source_run": best.get("run_dir"),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    args = _parse_args()
    models = ["LR", "RF"] if args.model == "all" else [args.model]

    all_rows = []
    for model in models:
        rows = _collect_runs(args.runs_root, model)
        all_rows.extend(rows)
        _write_summary(args.summary_root, model, rows)

    baseline = _load_baseline_metrics()
    compare = _build_compare_table(all_rows, baseline)
    if not compare.empty:
        args.summary_root.mkdir(parents=True, exist_ok=True)
        compare.to_csv(args.summary_root / "compare_baseline_vs_ga.csv", index=False)

    print(f"GA summaries saved to {args.summary_root}")


if __name__ == "__main__":
    main()
