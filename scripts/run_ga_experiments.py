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
from src.models.train import evaluate_on_test


EXPERIMENTS = {
    "A": {
        "pop_size": 30,
        "n_generations": 25,
        "mutation_rate": 0.10,
        "crossover_rate": 0.80,
        "elite_n": 1,
        "tournament_k": 3,
    },
    "B": {
        "pop_size": 60,
        "n_generations": 25,
        "mutation_rate": 0.20,
        "crossover_rate": 0.70,
        "elite_n": 2,
        "tournament_k": 2,
    },
    "C": {
        "pop_size": 30,
        "n_generations": 50,
        "mutation_rate": 0.05,
        "crossover_rate": 0.90,
        "elite_n": 2,
        "tournament_k": 4,
    },
}


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run controlled GA experiments.")
    parser.add_argument("--model", choices=["LR", "RF"], required=True)
    parser.add_argument("--exp", choices=sorted(EXPERIMENTS.keys()), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--holdout", choices=["val", "test"], default="val")
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument("--output-root", type=Path, default=config.ARTIFACTS_DIR / "ga_runs")
    return parser.parse_args()


def _make_ga_config(exp_id: str, seed: int) -> tuple[GAConfig, int]:
    exp = EXPERIMENTS[exp_id]
    elite_n = exp["elite_n"]
    elite_pct = elite_n / max(1, exp["pop_size"])
    ga_config = GAConfig(
        pop_size=exp["pop_size"],
        n_generations=exp["n_generations"],
        mutation_rate=exp["mutation_rate"],
        crossover_rate=exp["crossover_rate"],
        tournament_k=exp["tournament_k"],
        elite_pct=elite_pct,
        seed=seed,
    )
    return ga_config, elite_n


def _safe_run_dir(root: Path, model_key: str, exp_id: str, seed: int) -> Path:
    base = root / model_key / f"exp{exp_id}_seed{seed}"
    if not base.exists():
        return base
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return root / model_key / f"exp{exp_id}_seed{seed}_run_{stamp}"


def main() -> None:
    args = _parse_args()
    ga_config, elite_n = _make_ga_config(args.exp, args.seed)
    _set_seeds(ga_config.seed)

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

    fitness_cache = {}

    def fitness_fn(genes):
        return evaluate_fitness(
            genes,
            args.model,
            X_train,
            y_train,
            config,
            cache=fitness_cache,
            seed=ga_config.seed,
            n_splits=3,
            n_jobs=args.n_jobs,
        )

    run_start = time.time()
    best_individual, history = run_ga(args.model, fitness_fn, ga_config)

    estimator, preprocess, params_resolved = decode(best_individual.genes, args.model, config)
    pipeline = Pipeline([("preprocess", preprocess), ("model", estimator)])

    holdout_split = args.holdout
    if holdout_split == "val":
        pipeline.fit(X_train, y_train)
        holdout_metrics = evaluate_on_test(pipeline, X_val, y_val)
    else:
        X_full = pd.concat([X_train, X_val], axis=0)
        y_full = pd.concat([y_train, y_val], axis=0)
        pipeline.fit(X_full, y_full)
        holdout_metrics = evaluate_on_test(pipeline, X_test, y_test)

    training_time_sec = time.time() - run_start

    run_dir = _safe_run_dir(args.output_root, args.model, args.exp, args.seed)
    run_dir.mkdir(parents=True, exist_ok=True)

    history_rows = []
    for row in history:
        history_rows.append(
            {
                "exp_id": args.exp,
                "seed": args.seed,
                "model": args.model,
                "generation": row["generation"],
                "best_f1": row["best_fitness"],
                "mean_f1": row["mean_fitness"],
                "mutation_rate": ga_config.mutation_rate,
                "crossover_rate": ga_config.crossover_rate,
                "population_size": ga_config.pop_size,
                "tournament_k": ga_config.tournament_k,
                "elite_n": elite_n,
            }
        )
    pd.DataFrame(history_rows).to_csv(run_dir / "history.csv", index=False)

    best_payload = {
        "best_genome": best_individual.genes,
        "decoded_params": params_resolved,
        "best_cv_f1": best_individual.fitness,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "training_time_sec": training_time_sec,
        "holdout": holdout_split,
        "holdout_metrics": holdout_metrics,
        "config": {
            "exp_id": args.exp,
            "model": args.model,
            "seed": ga_config.seed,
            "ga_config": ga_config.__dict__,
            "elite_n": elite_n,
        },
    }
    (run_dir / "best.json").write_text(json.dumps(best_payload, indent=2))

    joblib.dump(pipeline, run_dir / "best_model.joblib")

    print(f"GA experiment saved to {run_dir}")


if __name__ == "__main__":
    main()
