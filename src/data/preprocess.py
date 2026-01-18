from __future__ import annotations

from sklearn.preprocessing import StandardScaler


def build_preprocess_pipeline(config):
    # Data is numeric-only; StandardScaler is enough for reproducible baseline.
    if not getattr(config, "USE_SCALER", True):
        return "passthrough"

    return StandardScaler()
