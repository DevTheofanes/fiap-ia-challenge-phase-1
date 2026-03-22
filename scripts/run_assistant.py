#!/usr/bin/env python
"""Interactive medical assistant CLI (Phase 3 — M3).

Usage:
    python scripts/run_assistant.py
    python scripts/run_assistant.py --features "17.99,10.38,122.8,1001,0.118,..."

The --features flag accepts 30 comma-separated float values (Wisconsin dataset
feature order) and injects an ML prediction into the assistant context.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from dotenv import load_dotenv

load_dotenv(BASE_DIR / ".env")

import joblib
import numpy as np

from src.assistant.graph import build_graph
from src.config import BEST_MODEL_PATH


def _parse_features(features_str: str) -> str | None:
    """Parse --features string, run ML model, return formatted prediction string."""
    try:
        values = [float(v.strip()) for v in features_str.split(",")]
    except ValueError as e:
        print(f"ERROR: Could not parse --features: {e}", file=sys.stderr)
        return None

    if len(values) != 30:
        print(f"ERROR: Expected 30 features, got {len(values)}", file=sys.stderr)
        return None

    if not BEST_MODEL_PATH.exists():
        print(f"WARNING: ML model not found at {BEST_MODEL_PATH}. Skipping ML context.")
        return None

    artifact = joblib.load(BEST_MODEL_PATH)
    if isinstance(artifact, dict):
        model = artifact["model"]
        threshold = artifact.get("threshold", 0.5)
    else:
        model = artifact
        threshold = 0.5
    X = np.array(values).reshape(1, -1)
    prob = model.predict_proba(X)[0][1]
    label = "Malignant" if prob >= threshold else "Benign"
    return f"ML Pipeline prediction: {prob:.0%} probability of malignancy ({label} — RF model)"


def main() -> None:
    parser = argparse.ArgumentParser(description="Medical assistant CLI (M3)")
    parser.add_argument(
        "--features",
        type=str,
        default=None,
        help="30 comma-separated Wisconsin feature values for ML prediction context",
    )
    args = parser.parse_args()

    ml_context: str | None = None
    if args.features:
        ml_context = _parse_features(args.features)
        if ml_context:
            print(f"ML context: {ml_context}\n")

    print("Building assistant graph …")
    graph = build_graph()
    print("Assistant ready. Type your question (Ctrl+C to exit).\n")

    while True:
        try:
            query = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            break

        if not query:
            continue

        state = {
            "query": query,
            "intent": "",
            "retrieved_docs": [],
            "ml_context": ml_context,
            "response": "",
            "sources": [],
            "refused": False,
            "error": None,
        }

        result = graph.invoke(state)

        print(f"\nAssistant: {result['response']}")

        if result.get("sources"):
            unique_sources = sorted(set(result["sources"]))
            print(f"Sources: {', '.join(unique_sources)}")

        print()


if __name__ == "__main__":
    main()
