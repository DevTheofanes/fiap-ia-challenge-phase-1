from __future__ import annotations

import math
import random
from typing import Any, Dict, Tuple

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.data.preprocess import build_preprocess_pipeline
from src.genetic.search_space import get_search_space, lr_valid_solvers

Individual = Dict[str, Any]


def random_individual(model_name: str, rng: random.Random | None = None) -> Individual:
    rng = rng or random
    space = get_search_space(model_name)
    individual: Individual = {}

    if model_name == "LR":
        penalty = rng.choice(space["penalty"]["values"])
        individual["penalty"] = penalty
        individual["solver"] = rng.choice(lr_valid_solvers(penalty))
    for gene, spec in space.items():
        if gene in individual:
            continue
        individual[gene] = _sample_gene(spec, rng)

    return repair(individual, model_name)


def mutate(
    individual: Individual,
    model_name: str,
    rng: random.Random | None = None,
    mutation_rate: float = 0.2,
) -> Individual:
    rng = rng or random
    space = get_search_space(model_name)
    mutated = dict(individual)

    for gene, spec in space.items():
        if rng.random() >= mutation_rate:
            continue
        mutated[gene] = _mutate_gene(mutated.get(gene), spec, rng)

    return repair(mutated, model_name)


def crossover(
    parent_a: Individual,
    parent_b: Individual,
    model_name: str,
    rng: random.Random | None = None,
) -> Individual:
    rng = rng or random
    space = get_search_space(model_name)
    child: Individual = {}

    for gene in space:
        if rng.random() < 0.5:
            child[gene] = parent_a.get(gene)
        else:
            child[gene] = parent_b.get(gene)

    return repair(child, model_name)


def decode(
    individual: Individual,
    model_name: str,
    config=None,
) -> Tuple[Any, Any, Dict[str, Any]]:
    repaired = repair(individual, model_name)
    params = dict(repaired)
    random_state = getattr(config, "RANDOM_STATE", 42)

    if model_name == "LR":
        estimator = _build_lr_estimator(params, random_state)
    elif model_name == "RF":
        estimator = RandomForestClassifier(
            class_weight="balanced",
            random_state=random_state,
            **params,
        )
    else:
        raise ValueError(f"Unknown model for decode: {model_name}")

    preprocess = build_preprocess_pipeline(config) if config is not None else "passthrough"
    params_resolved = estimator.get_params()
    return estimator, preprocess, params_resolved


def repair(individual: Individual, model_name: str) -> Individual:
    space = get_search_space(model_name)
    repaired = dict(individual)

    for gene, spec in space.items():
        if gene not in repaired:
            repaired[gene] = _sample_gene(spec, random)
            continue
        repaired[gene] = _coerce_and_clamp(repaired[gene], spec)

    if model_name == "LR":
        penalty = repaired.get("penalty", "l2")
        if penalty not in ("l1", "l2"):
            penalty = "l2"
        repaired["penalty"] = penalty

        valid_solvers = lr_valid_solvers(penalty)
        if repaired.get("solver") not in valid_solvers:
            repaired["solver"] = valid_solvers[0]

        repaired["max_iter"] = int(repaired["max_iter"])

        if repaired.get("class_weight") not in (None, "balanced"):
            repaired["class_weight"] = None

    if model_name == "RF":
        if not _is_valid_rf_criterion(repaired["criterion"]):
            repaired["criterion"] = "gini"

        min_split = int(repaired["min_samples_split"])
        min_leaf = int(repaired["min_samples_leaf"])
        if min_split <= min_leaf:
            min_split = min_leaf + 1
        repaired["min_samples_split"] = min_split
        repaired["min_samples_leaf"] = min_leaf

    return repaired


def _sample_gene(spec: Dict[str, Any], rng: random.Random) -> Any:
    if spec["type"] == "categorical":
        return rng.choice(spec["values"])
    if spec["type"] == "int":
        return rng.randint(int(spec["min"]), int(spec["max"]))
    if spec["type"] == "float":
        if spec.get("scale") == "log":
            return _sample_log_uniform(spec["min"], spec["max"], rng)
        return rng.uniform(float(spec["min"]), float(spec["max"]))
    raise ValueError(f"Unknown gene type: {spec['type']}")


def _mutate_gene(current: Any, spec: Dict[str, Any], rng: random.Random) -> Any:
    if spec["type"] == "categorical":
        values = spec["values"]
        if current in values and len(values) > 1:
            alternatives = [v for v in values if v != current]
            return rng.choice(alternatives)
        return rng.choice(values)

    if spec["type"] == "int":
        span = int(spec["max"]) - int(spec["min"])
        step = max(1, int(span * 0.1))
        delta = rng.randint(-step, step)
        return _clamp_int(int(current or 0) + delta, int(spec["min"]), int(spec["max"]))

    if spec["type"] == "float":
        if spec.get("scale") == "log":
            factor = 10 ** rng.uniform(-0.5, 0.5)
            return _clamp_float(float(current or spec["min"]) * factor, spec["min"], spec["max"])
        span = float(spec["max"]) - float(spec["min"])
        jitter = rng.uniform(-0.1 * span, 0.1 * span)
        return _clamp_float(float(current or spec["min"]) + jitter, spec["min"], spec["max"])

    return current


def _coerce_and_clamp(value: Any, spec: Dict[str, Any]) -> Any:
    if spec["type"] == "categorical":
        return value if value in spec["values"] else spec["values"][0]
    if spec["type"] == "int":
        if value is None:
            value = spec["min"]
        return _clamp_int(int(value), int(spec["min"]), int(spec["max"]))
    if spec["type"] == "float":
        if value is None:
            value = spec["min"]
        return _clamp_float(float(value), spec["min"], spec["max"])
    return value


def _sample_log_uniform(min_val: float, max_val: float, rng: random.Random) -> float:
    log_min = math.log10(min_val)
    log_max = math.log10(max_val)
    return 10 ** rng.uniform(log_min, log_max)


def _clamp_int(value: int, min_val: int, max_val: int) -> int:
    return max(min_val, min(value, max_val))


def _clamp_float(value: float, min_val: float, max_val: float) -> float:
    return max(min_val, min(value, max_val))


def _is_valid_rf_criterion(value: str) -> bool:
    try:
        RandomForestClassifier(criterion=value)
        return True
    except Exception:
        return False


def _build_lr_estimator(params: Dict[str, Any], random_state: int) -> LogisticRegression:
    params = dict(params)
    max_iter = int(params.pop("max_iter"))
    class_weight = params.pop("class_weight")
    penalty = params.pop("penalty", "l2")

    if _penalty_deprecated():
        l1_ratio = None
        if penalty == "l2":
            l1_ratio = 0.0
        elif penalty == "l1":
            l1_ratio = 1.0

        if l1_ratio is not None:
            params["l1_ratio"] = l1_ratio

        return LogisticRegression(
            max_iter=max_iter,
            class_weight=class_weight,
            random_state=random_state,
            **params,
        )

    params["penalty"] = penalty
    return LogisticRegression(
        max_iter=max_iter,
        class_weight=class_weight,
        random_state=random_state,
        **params,
    )


def _penalty_deprecated() -> bool:
    try:
        import sklearn
    except Exception:
        return False

    version = getattr(sklearn, "__version__", "0.0")
    parts = version.split(".")
    major = int(parts[0]) if parts and parts[0].isdigit() else 0
    minor = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
    return (major, minor) >= (1, 8)
