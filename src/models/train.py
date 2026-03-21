from __future__ import annotations

from typing import Dict, Tuple

from sklearn.pipeline import Pipeline

from src.data.preprocess import build_preprocess_pipeline
from src.evaluation.metrics import compute_metrics
from src.models.registry import build_model


def _predict_scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        smin, smax = scores.min(), scores.max()
        return (scores - smin) / (smax - smin + 1e-9)
    return None


def train_and_evaluate(
    model_name: str,
    params: dict,
    X_train,
    y_train,
    X_val,
    y_val,
    config,
) -> Tuple[Pipeline, Dict[str, float]]:
    preprocess = build_preprocess_pipeline(config)
    estimator = build_model(model_name, params, config.RANDOM_STATE)

    pipeline = Pipeline(steps=[("preprocess", preprocess), ("model", estimator)])
    pipeline.fit(X_train, y_train)

    y_val_pred = pipeline.predict(X_val)
    y_val_scores = _predict_scores(pipeline, X_val)

    metrics = compute_metrics(y_val, y_val_pred, y_val_scores)
    return pipeline, metrics


def evaluate_on_test(model, X_test, y_test) -> Dict[str, float]:
    y_test_pred = model.predict(X_test)
    y_test_scores = _predict_scores(model, X_test)
    return compute_metrics(y_test, y_test_pred, y_test_scores)
