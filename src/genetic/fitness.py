from __future__ import annotations

import json
import logging
import time
import warnings
from typing import Any, Dict, Tuple

from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import Pipeline

from src.evaluation.cv import stratified_f1_cv
from src.genetic.encoding import decode

logger = logging.getLogger(__name__)

FitnessCache = Dict[str, Dict[str, Any]]


def evaluate_fitness(
    individual: Dict[str, Any],
    model_name: str,
    X,
    y,
    config,
    cache: FitnessCache | None = None,
    seed: int | None = None,
    n_splits: int = 3,
    n_jobs: int | None = None,
) -> Tuple[float, Dict[str, Any]]:
    cache = cache or {}
    key = _cache_key(model_name, individual)
    if key in cache:
        return cache[key]["fitness"], cache[key]["params_resolved"]

    run_seed = seed if seed is not None else getattr(config, "RANDOM_STATE", 42)
    start = time.time()
    try:
        genes = dict(individual)
        mean_f1, std_f1, params_resolved = _run_cv_with_retry(
            genes,
            model_name,
            config,
            X,
            y,
            run_seed,
            n_splits,
            n_jobs,
        )
        payload = {
            "fitness": mean_f1,
            "params_resolved": params_resolved,
            "std_f1": std_f1,
            "elapsed_sec": time.time() - start,
        }
    except Exception as exc:
        logger.warning("Fitness failed for %s: %s", model_name, exc)
        payload = {
            "fitness": 0.0,
            "params_resolved": {},
            "std_f1": 0.0,
            "elapsed_sec": time.time() - start,
        }

    cache[key] = payload
    return payload["fitness"], payload["params_resolved"]


def _cache_key(model_name: str, individual: Dict[str, Any]) -> str:
    genes_json = json.dumps(individual, sort_keys=True, default=str)
    return f"{model_name}:{genes_json}"


def _run_cv_with_retry(
    genes: Dict[str, Any],
    model_name: str,
    config,
    X,
    y,
    seed: int,
    n_splits: int,
    n_jobs: int | None,
) -> Tuple[float, float, Dict[str, Any]]:
    attempts = 2 if model_name == "LR" else 1
    current = dict(genes)
    for attempt in range(attempts):
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always", ConvergenceWarning)
            estimator, preprocess, params_resolved = decode(current, model_name, config)
            pipeline = Pipeline([("preprocess", preprocess), ("model", estimator)])
            mean_f1, std_f1 = stratified_f1_cv(
                pipeline,
                X,
                y,
                seed=seed,
                n_splits=n_splits,
                n_jobs=n_jobs,
            )

        if _has_convergence_warning(captured) and model_name == "LR" and attempt == 0:
            current = _bump_lr_max_iter(current)
            continue
        return mean_f1, std_f1, params_resolved

    return mean_f1, std_f1, params_resolved


def _has_convergence_warning(warnings_list) -> bool:
    for warn in warnings_list:
        if issubclass(warn.category, ConvergenceWarning):
            return True
    return False


def _bump_lr_max_iter(genes: Dict[str, Any]) -> Dict[str, Any]:
    updated = dict(genes)
    base = updated.get("max_iter")
    try:
        base_val = int(base)
    except Exception:
        base_val = 1000
    updated["max_iter"] = min(max(base_val * 2, 1000), 10000)
    return updated
