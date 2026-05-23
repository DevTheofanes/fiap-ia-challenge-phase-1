#!/usr/bin/env python
"""Standalone CLI for the Phase 4 video pipeline."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.multimodal.video import analyze_video, generate_video_report


DEFAULT_MODEL_PATH = BASE_DIR / "artifacts" / "yolo" / "bleeding_yolov8n" / "weights" / "best.pt"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze a clinical video with YOLOv8.")
    parser.add_argument("video_path", type=Path, help="Path to a clinical video file.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to the trained YOLOv8 .pt model.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.25,
        help="Minimum confidence required to keep a detection.",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Analyze one frame every N frames.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        analysis = analyze_video(
            args.video_path,
            args.model_path,
            confidence_threshold=args.confidence_threshold,
            frame_stride=args.frame_stride,
        )
        report = generate_video_report(analysis.detections)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print("=== Video Analysis ===")
    print(f"Video: {analysis.video_path}")
    print(f"Model: {analysis.model_path}")
    print(f"Frames processed: {analysis.frames_processed}")
    print()
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
