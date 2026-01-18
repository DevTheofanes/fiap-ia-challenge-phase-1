from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


def load_dataset(
    data_path: Path,
    target_col: str,
    id_cols: list[str],
) -> Tuple[pd.DataFrame, pd.Series]:
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path)

    cols_to_drop = [c for c in id_cols if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop, errors="ignore")

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in num_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    df = df.copy()
    df["target"] = df[target_col].map({"M": 1, "B": 0})
    if not set(df["target"].dropna().unique()).issubset({0, 1}):
        raise ValueError("Target mapping failed; expected values {0, 1}.")

    feature_cols = [c for c in df.columns if c not in [target_col, "target"]]
    X = df[feature_cols].select_dtypes(include=[np.number]).copy()
    y = df["target"].astype(int).copy()

    return X, y
