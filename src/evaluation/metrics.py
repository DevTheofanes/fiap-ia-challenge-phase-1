from __future__ import annotations

from typing import Dict

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true, y_pred, y_scores=None) -> Dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    if y_scores is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_scores))
        except Exception:
            metrics["roc_auc"] = float("nan")
        try:
            metrics["pr_auc"] = float(average_precision_score(y_true, y_scores))
        except Exception:
            metrics["pr_auc"] = float("nan")
    else:
        metrics["roc_auc"] = float("nan")
        metrics["pr_auc"] = float("nan")

    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = cm.tolist()
    return metrics
