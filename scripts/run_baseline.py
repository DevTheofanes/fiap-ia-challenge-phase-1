from __future__ import annotations

import json
import random
import time
from pathlib import Path
import sys

import joblib
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src import config
from src.data.load import load_dataset
from src.data.split import apply_splits, make_or_load_splits
from src.evaluation.report import save_baseline_report
from src.logging_utils import log_event, setup_json_logger
from src.models.registry import default_model_registry
from src.models.train import evaluate_on_test, train_and_evaluate


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def main() -> None:
    _set_seeds(config.RANDOM_STATE)
    train_logger = setup_json_logger("baseline.train", config.TRAINING_LOG_PATH)
    eval_logger = setup_json_logger("baseline.eval", config.EVALUATION_LOG_PATH)

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

    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    config.BASELINE_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    report = {
        "random_state": config.RANDOM_STATE,
        "fitness_metric": config.FITNESS_METRIC,
        "splits": {
            "train_size": int(len(X_train)),
            "val_size": int(len(X_val)),
            "test_size": int(len(X_test)),
        },
        "models": {},
    }

    for model_name, params in default_model_registry().items():
        model_start = time.time()
        model, val_metrics = train_and_evaluate(
            model_name,
            params,
            X_train,
            y_train,
            X_val,
            y_val,
            config,
        )
        test_metrics = evaluate_on_test(model, X_test, y_test)
        duration_sec = time.time() - model_start

        model_path = config.BASELINE_MODELS_DIR / f"{model_name}.joblib"
        joblib.dump(model, model_path)

        report["models"][model_name] = {
            "params": model.named_steps["model"].get_params(),
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "artifact": str(model_path.relative_to(config.BASE_DIR)),
        }

        log_event(
            train_logger,
            "baseline",
            model=model_name,
            experiment="baseline",
            seed=config.RANDOM_STATE,
            params=json.dumps(model.named_steps["model"].get_params(), default=str),
            val_f1=val_metrics.get("f1"),
            val_recall=val_metrics.get("recall"),
            val_precision=val_metrics.get("precision"),
            duration_sec=duration_sec,
        )
        for split_name, metrics in [("val", val_metrics), ("test", test_metrics)]:
            log_event(
                eval_logger,
                "baseline_eval",
                model=model_name,
                experiment="baseline",
                seed=config.RANDOM_STATE,
                split=split_name,
                f1=metrics.get("f1"),
                recall=metrics.get("recall"),
                precision=metrics.get("precision"),
                accuracy=metrics.get("accuracy"),
                roc_auc=metrics.get("roc_auc"),
                pr_auc=metrics.get("pr_auc"),
            )

    save_baseline_report(report, config.BASELINE_METRICS_PATH)
    print(f"Baseline metrics saved to {config.BASELINE_METRICS_PATH}")


if __name__ == "__main__":
    main()
