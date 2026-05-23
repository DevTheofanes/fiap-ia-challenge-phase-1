#!/usr/bin/env python
"""Fine-tune YOLOv8 for synthetic anomalous bleeding detection."""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.logging_utils import log_event, setup_json_logger


DEFAULT_DATA = BASE_DIR / "data" / "synthetic_bleeding" / "dataset.yaml"
DEFAULT_PROJECT = BASE_DIR / "artifacts" / "yolo"
DEFAULT_MODEL = "yolov8n.pt"
DEFAULT_EPOCHS = 20
DEFAULT_IMGSZ = 640
DEFAULT_BATCH = 8
DEFAULT_DEVICE = "mps"
DEFAULT_RUN_NAME = "bleeding_yolov8n"
LOG_PATH = BASE_DIR / "artifacts" / "logs" / "yolo_training.jsonl"


@dataclass(frozen=True)
class TrainingSummary:
    run_dir: Path | None
    metrics: dict[str, float | None]


def _coerce_path(path: Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return BASE_DIR / path


def _get_nested_value(source: Any, path: str) -> Any:
    value = source
    for part in path.split("."):
        if isinstance(value, dict):
            value = value.get(part)
        else:
            value = getattr(value, part, None)
        if value is None:
            return None
    return value


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_available(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def extract_metrics(results: Any) -> dict[str, float | None]:
    results_dict = getattr(results, "results_dict", None)
    return {
        "mAP50": _coerce_float(
            _first_available(
                _get_nested_value(results, "box.map50"),
                (results_dict or {}).get("metrics/mAP50(B)"),
                (results_dict or {}).get("metrics/mAP50"),
            )
        ),
        "precision": _coerce_float(
            _first_available(
                _get_nested_value(results, "box.mp"),
                (results_dict or {}).get("metrics/precision(B)"),
                (results_dict or {}).get("metrics/precision"),
            )
        ),
        "recall": _coerce_float(
            _first_available(
                _get_nested_value(results, "box.mr"),
                (results_dict or {}).get("metrics/recall(B)"),
                (results_dict or {}).get("metrics/recall"),
            )
        ),
    }


def train_yolo(
    *,
    data: Path = DEFAULT_DATA,
    model: str = DEFAULT_MODEL,
    epochs: int = DEFAULT_EPOCHS,
    imgsz: int = DEFAULT_IMGSZ,
    batch: int = DEFAULT_BATCH,
    project: Path = DEFAULT_PROJECT,
    name: str = DEFAULT_RUN_NAME,
    device: str = DEFAULT_DEVICE,
    log_path: Path | None = LOG_PATH,
) -> TrainingSummary:
    data_path = _coerce_path(data)
    project_path = _coerce_path(project)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset YAML not found: {data_path}. "
            "Run scripts/generate_synthetic_data.py before training YOLOv8."
        )

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "ultralytics is required for YOLOv8 training. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc

    logger = setup_json_logger("phase4.yolo_training", _coerce_path(log_path)) if log_path else None
    if logger:
        log_event(
            logger,
            "yolo_training_started",
            data=str(data_path),
            model=model,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            project=str(project_path),
            name=name,
            device=device,
        )

    yolo_model = YOLO(model)
    results = yolo_model.train(
        data=str(data_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=str(project_path),
        name=name,
        device=device,
    )
    metrics = extract_metrics(results)
    run_dir = getattr(results, "save_dir", None)
    summary = TrainingSummary(run_dir=Path(run_dir) if run_dir else None, metrics=metrics)

    if logger:
        log_event(
            logger,
            "yolo_training_completed",
            run_dir=str(summary.run_dir) if summary.run_dir else None,
            **metrics,
        )
    return summary


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8 on synthetic bleeding data.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Path to dataset.yaml.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Base YOLO model checkpoint.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ, help="Training image size.")
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH, help="Training batch size.")
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT, help="Output project directory.")
    parser.add_argument("--name", default=DEFAULT_RUN_NAME, help="Ultralytics run name.")
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="Training device, e.g. mps, cpu, cuda.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        summary = train_yolo(
            data=args.data,
            model=args.model,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            project=args.project,
            name=args.name,
            device=args.device,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print("=== YOLOv8 Training Summary ===")
    if summary.run_dir:
        print(f"Run directory: {summary.run_dir}")
    for metric, value in summary.metrics.items():
        rendered = "unavailable" if value is None else f"{value:.4f}"
        print(f"{metric}: {rendered}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
