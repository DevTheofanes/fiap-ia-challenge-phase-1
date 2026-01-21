from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
import sys

sys.path.insert(0, str(BASE_DIR))

from sklearn.pipeline import Pipeline

from src import config
from src.data.load import load_dataset
from src.data.split import apply_splits, make_or_load_splits
from src.genetic.encoding import decode
from src.genetic.fitness import evaluate_fitness
from src.genetic.ga import GAConfig, run_ga
from src.logging_utils import log_event, setup_json_logger


DEFAULT_CONFIGS = {
    "LR": GAConfig(
        pop_size=40,
        n_generations=50,
        mutation_rate=0.10,
        crossover_rate=0.9,
        tournament_k=3,
        elite_pct=0.05,
        seed=config.RANDOM_STATE,
    ),
    "RF": GAConfig(
        pop_size=30,
        n_generations=40,
        mutation_rate=0.15,
        crossover_rate=0.9,
        tournament_k=3,
        elite_pct=0.05,
        seed=config.RANDOM_STATE,
    ),
}


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GA hyperparameter search.")
    parser.add_argument("--model", choices=["LR", "RF"], required=True)
    parser.add_argument("--pop-size", type=int)
    parser.add_argument("--generations", type=int)
    parser.add_argument("--mutation-rate", type=float)
    parser.add_argument("--crossover-rate", type=float)
    parser.add_argument("--tournament-k", type=int)
    parser.add_argument("--elite-pct", type=float)
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


def _merge_config(model_key: str, args: argparse.Namespace) -> GAConfig:
    base = DEFAULT_CONFIGS[model_key]
    return GAConfig(
        pop_size=args.pop_size or base.pop_size,
        n_generations=args.generations or base.n_generations,
        mutation_rate=args.mutation_rate if args.mutation_rate is not None else base.mutation_rate,
        crossover_rate=args.crossover_rate if args.crossover_rate is not None else base.crossover_rate,
        tournament_k=args.tournament_k or base.tournament_k,
        elite_pct=args.elite_pct if args.elite_pct is not None else base.elite_pct,
        seed=args.seed if args.seed is not None else base.seed,
    )


def main() -> None:
    args = _parse_args()
    model_key = args.model
    ga_config = _merge_config(model_key, args)
    _set_seeds(ga_config.seed)
    train_logger = setup_json_logger("ga.train", config.TRAINING_LOG_PATH)

    X, y = load_dataset(config.DATA_PATH, config.TARGET_COL, config.ID_COLS)
    splits = make_or_load_splits(
        n_samples=len(X),
        y=y,
        test_size=config.TEST_SIZE,
        val_size=config.VAL_SIZE,
        random_state=config.RANDOM_STATE,
        splits_path=config.SPLITS_PATH,
    )
    X_train, X_val, _, y_train, y_val, _ = apply_splits(X, y, splits)

    fitness_cache = {}

    def fitness_fn(genes):
        return evaluate_fitness(
            genes,
            model_key,
            X_train,
            y_train,
            config,
            cache=fitness_cache,
            seed=ga_config.seed,
            n_splits=3,
            n_jobs=None,
        )

    run_start = time.time()
    best_individual, history = run_ga(model_key, fitness_fn, ga_config)
    run_duration = time.time() - run_start

    run_id = time.strftime("run_%Y%m%d_%H%M%S")
    run_dir = config.ARTIFACTS_DIR / "ga_runs" / model_key / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    config_path = run_dir / "config.json"
    config_payload = {
        "model": model_key,
        "seed": ga_config.seed,
        "fitness_metric": config.FITNESS_METRIC,
        "ga_config": ga_config.__dict__,
        "splits": {
            "train_size": int(len(X_train)),
            "val_size": int(len(X_val)),
        },
    }
    config_path.write_text(json.dumps(config_payload, indent=2))

    history_df = pd.DataFrame(history)
    history_df.to_csv(run_dir / "history.csv", index=False)

    estimator, preprocess, params_resolved = decode(best_individual.genes, model_key, config)

    best_payload = {
        "genes": best_individual.genes,
        "fitness": best_individual.fitness,
        "params_resolved": params_resolved,
    }
    (run_dir / "best_individual.json").write_text(json.dumps(best_payload, indent=2))

    X_full = pd.concat([X_train, X_val], axis=0)
    y_full = pd.concat([y_train, y_val], axis=0)
    pipeline = Pipeline([("preprocess", preprocess), ("model", estimator)])
    pipeline.fit(X_full, y_full)
    joblib.dump(pipeline, run_dir / "best_model.joblib")

    total_eval_count = sum(row.get("eval_count", 0) for row in history)
    total_eval_time = sum(row.get("eval_time_sec", 0.0) for row in history)
    mean_gen_time = float(np.mean([row.get("elapsed_sec", 0.0) for row in history])) if history else 0.0
    mean_eval_time = (total_eval_time / total_eval_count) if total_eval_count else 0.0
    log_event(
        train_logger,
        "ga_train",
        model=model_key,
        experiment="baseline",
        seed=ga_config.seed,
        ga_config=ga_config.__dict__,
        best_f1=best_individual.fitness,
        duration_sec=run_duration,
        mean_gen_time_sec=mean_gen_time,
        eval_count=total_eval_count,
        mean_eval_time_sec=mean_eval_time,
    )

    print(f"GA run saved to {run_dir}")


if __name__ == "__main__":
    main()
