from __future__ import annotations

from typing import Dict

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier  # type: ignore
    HAS_XGB = True
except Exception:
    HAS_XGB = False


def build_model(model_name: str, params: dict | None, random_state: int):
    params = params or {}
    if model_name == "LR":
        return LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=random_state, **params
        )
    if model_name == "SVC":
        return SVC(
            probability=True,
            class_weight="balanced",
            random_state=random_state,
            **params,
        )
    if model_name == "RF":
        return RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=random_state,
            **params,
        )
    if model_name == "KNN":
        return KNeighborsClassifier(n_neighbors=9, **params)
    if model_name == "XGB":
        if not HAS_XGB:
            raise ValueError("XGBoost is not available.")
        return XGBClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            n_jobs=4,
            random_state=random_state,
            eval_metric="logloss",
            **params,
        )

    raise ValueError(f"Unknown model: {model_name}")


def default_model_registry() -> Dict[str, dict]:
    registry = {
        "LR": {},
        "SVC": {},
        "RF": {},
        "KNN": {},
    }
    if HAS_XGB:
        registry["XGB"] = {}
    return registry
