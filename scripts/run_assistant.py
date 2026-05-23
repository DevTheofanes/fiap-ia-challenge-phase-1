#!/usr/bin/env python
"""Interactive medical assistant CLI (Phase 3 — M3).

Usage:
    python scripts/run_assistant.py
    python scripts/run_assistant.py --features "17.99,10.38,122.8,1001,0.118,..."
    python scripts/run_assistant.py --patient-id P-0001
    python scripts/run_assistant.py --audio consultation.wav --video procedure.mp4

The --features flag accepts 30 comma-separated float values (Wisconsin dataset
feature order) and injects an ML prediction into the assistant context when
no patient record is provided.
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
        "--patient-id",
        type=str,
        default=None,
        help="Synthetic patient record id under data/patients/",
    )
    parser.add_argument(
        "--features",
        type=str,
        default=None,
        help="30 comma-separated Wisconsin feature values for ML prediction context",
    )
    parser.add_argument(
        "--audio",
        type=str,
        default=None,
        help="Path to an audio file to transcribe and analyze before answering",
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path to a video file to analyze before answering",
    )
    args = parser.parse_args()

    ml_context: str | None = None
    if args.features and not args.patient_id:
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
            "patient_id": args.patient_id,
            "intent": "",
            "retrieved_docs": [],
            "kb_context": "",
            "patient_record": None,
            "patient_context": ml_context or "No patient-specific context provided.",
            "answer": "",
            "kb_sources": [],
            "patient_source": None,
            "used_patient_context": False,
            "refused": False,
            "error": None,
            "audio_path": args.audio,
            "audio_transcript": None,
            "audio_analysis": None,
            "video_path": args.video,
            "video_report": None,
        }

        result = graph.invoke(state)

        print(f"\nAssistant: {result['answer']}")
        print()


if __name__ == "__main__":
    main()
