from __future__ import annotations

from typing import Dict, List

ModelSpace = Dict[str, dict]

SELECTED_MODELS = ("LR", "RF")


LR_SPACE: ModelSpace = {
    "C": {"type": "float", "min": 1e-4, "max": 1e2, "scale": "log"},
    "penalty": {"type": "categorical", "values": ["l1", "l2"]},
    "solver": {"type": "categorical", "values": ["lbfgs", "liblinear", "saga"]},
    "max_iter": {"type": "int", "min": 500, "max": 5000},
    "class_weight": {"type": "categorical", "values": [None, "balanced"]},
}

RF_SPACE: ModelSpace = {
    "n_estimators": {"type": "int", "min": 100, "max": 600},
    "max_depth": {"type": "int", "min": 1, "max": 20},
    "min_samples_split": {"type": "int", "min": 2, "max": 30},
    "min_samples_leaf": {"type": "int", "min": 1, "max": 20},
    "criterion": {"type": "categorical", "values": ["gini", "entropy", "log_loss"]},
    "ccp_alpha": {"type": "float", "min": 0.0, "max": 0.02, "scale": "linear"},
}

MODEL_SPACES: Dict[str, ModelSpace] = {
    "LR": LR_SPACE,
    "RF": RF_SPACE,
}


def lr_valid_solvers(penalty: str) -> List[str]:
    if penalty == "l1":
        return ["liblinear", "saga"]
    return ["lbfgs", "liblinear", "saga"]


def get_search_space(model_name: str) -> ModelSpace:
    if model_name not in MODEL_SPACES:
        raise ValueError(f"Unknown model for search space: {model_name}")
    return MODEL_SPACES[model_name]
