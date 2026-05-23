#!/usr/bin/env python
"""Standalone CLI for the Phase 4 audio pipeline."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from dotenv import load_dotenv

load_dotenv(BASE_DIR / ".env")

from src.multimodal.audio import analyze_transcript, transcribe


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze a clinical audio recording.")
    parser.add_argument("audio_path", type=Path, help="Path to an audio file supported by OpenAI.")
    parser.add_argument(
        "--transcription-model",
        default="whisper-1",
        help="OpenAI transcription model to use. Defaults to whisper-1.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        transcript = transcribe(args.audio_path, model=args.transcription_model)
        analysis = analyze_transcript(transcript)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print("=== Transcript ===")
    print(transcript)
    print()
    print("=== Clinical Analysis ===")
    print(analysis)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
