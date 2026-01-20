from __future__ import annotations

from typing import Tuple

from sklearn.model_selection import StratifiedKFold, cross_val_score


def stratified_f1_cv(
    estimator,
    X,
    y,
    seed: int,
    n_splits: int = 3,
    n_jobs: int | None = None,
) -> Tuple[float, float]:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = cross_val_score(
        estimator,
        X,
        y,
        scoring="f1",
        cv=cv,
        n_jobs=n_jobs,
    )
    return float(scores.mean()), float(scores.std())
