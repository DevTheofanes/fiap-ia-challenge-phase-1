from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class SplitIndices:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


def _validate_sizes(n_samples: int, splits: SplitIndices) -> None:
    total = len(splits.train_idx) + len(splits.val_idx) + len(splits.test_idx)
    if total != n_samples:
        raise ValueError(
            f"Split sizes do not match dataset size: {total} != {n_samples}"
        )


def load_splits(path: Path) -> SplitIndices:
    data = np.load(path)
    return SplitIndices(
        train_idx=data["train_idx"],
        val_idx=data["val_idx"],
        test_idx=data["test_idx"],
    )


def save_splits(path: Path, splits: SplitIndices) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        train_idx=splits.train_idx,
        val_idx=splits.val_idx,
        test_idx=splits.test_idx,
    )


def make_or_load_splits(
    n_samples: int,
    y,
    test_size: float,
    val_size: float,
    random_state: int,
    splits_path: Path,
) -> SplitIndices:
    if splits_path.exists():
        splits = load_splits(splits_path)
        _validate_sizes(n_samples, splits)
        return splits

    idx = np.arange(n_samples)
    train_idx, temp_idx, y_train, y_temp = train_test_split(
        idx,
        y,
        test_size=test_size + val_size,
        stratify=y,
        random_state=random_state,
    )

    val_ratio = val_size / (test_size + val_size)
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=1.0 - val_ratio,
        stratify=y_temp,
        random_state=random_state,
    )

    splits = SplitIndices(train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)
    save_splits(splits_path, splits)
    return splits


def apply_splits(X, y, splits: SplitIndices) -> Tuple:
    X_train = X.iloc[splits.train_idx].copy()
    X_val = X.iloc[splits.val_idx].copy()
    X_test = X.iloc[splits.test_idx].copy()

    y_train = y.iloc[splits.train_idx].copy()
    y_val = y.iloc[splits.val_idx].copy()
    y_test = y.iloc[splits.test_idx].copy()

    return X_train, X_val, X_test, y_train, y_val, y_test
