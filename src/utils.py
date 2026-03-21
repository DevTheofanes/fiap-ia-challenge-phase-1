"""Shared utilities used across pipeline scripts."""
from __future__ import annotations

import random

import numpy as np


def set_seeds(seed: int) -> None:
    """Set Python and NumPy random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
