from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class VideoDetection:
    frame_index: int
    timestamp_sec: float
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]


@dataclass(frozen=True)
class VideoAnalysis:
    video_path: Path
    model_path: Path
    frames_processed: int
    detections: list[VideoDetection]


def _video_fps(video_path: Path) -> float | None:
    try:
        import cv2  # type: ignore
    except Exception:  # pragma: no cover - optional dependency
        return None

    capture = cv2.VideoCapture(str(video_path))
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
    finally:
        capture.release()
    return fps if fps > 0 else None


def _as_float_tuple(values: Any) -> tuple[float, float, float, float]:
    if hasattr(values, "tolist"):
        values = values.tolist()
    if values and isinstance(values[0], list):
        values = values[0]
    if len(values) != 4:
        raise ValueError(f"Expected four bounding box coordinates, got: {values}")
    return tuple(float(value) for value in values)  # type: ignore[return-value]


def _as_float(value: Any) -> float:
    if hasattr(value, "item"):
        return float(value.item())
    if hasattr(value, "tolist"):
        value = value.tolist()
        if isinstance(value, list):
            value = value[0]
    return float(value)


def _class_name(model: Any, class_id: int) -> str:
    names = getattr(model, "names", None) or {}
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if isinstance(names, list) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def _iter_boxes(result: Any) -> list[Any]:
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []
    try:
        return list(boxes)
    except TypeError:
        return [boxes]


def analyze_video(
    video_path: str | Path,
    model_path: str | Path,
    *,
    confidence_threshold: float = 0.25,
    frame_stride: int = 1,
) -> VideoAnalysis:
    """Run YOLOv8 inference over a video and return frame-level detections."""
    path = Path(video_path)
    model = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")
    if not path.is_file():
        raise ValueError(f"Video path is not a file: {path}")
    if not model.exists():
        raise FileNotFoundError(f"YOLO model file not found: {model}")
    if not model.is_file():
        raise ValueError(f"YOLO model path is not a file: {model}")
    if not 0 <= confidence_threshold <= 1:
        raise ValueError("confidence_threshold must be between 0 and 1.")
    if frame_stride < 1:
        raise ValueError("frame_stride must be at least 1.")

    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "ultralytics is required for YOLOv8 video analysis. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc

    yolo_model = YOLO(str(model))
    fps = _video_fps(path)
    detections: list[VideoDetection] = []
    frames_processed = 0

    results = yolo_model.predict(
        source=str(path),
        stream=True,
        conf=confidence_threshold,
        vid_stride=frame_stride,
        verbose=False,
    )
    for result_index, result in enumerate(results):
        frame_index = result_index * frame_stride
        timestamp_sec = frame_index / fps if fps else 0.0
        frames_processed += 1
        for box in _iter_boxes(result):
            confidence = _as_float(getattr(box, "conf", 0.0))
            if confidence < confidence_threshold:
                continue
            class_id = int(_as_float(getattr(box, "cls", 0)))
            detections.append(
                VideoDetection(
                    frame_index=frame_index,
                    timestamp_sec=timestamp_sec,
                    class_name=_class_name(yolo_model, class_id),
                    confidence=confidence,
                    bbox=_as_float_tuple(getattr(box, "xyxy")),
                )
            )

    return VideoAnalysis(
        video_path=path,
        model_path=model,
        frames_processed=frames_processed,
        detections=detections,
    )


def classify_video_risk(detections: list[VideoDetection]) -> str:
    if not detections:
        return "baixo"
    average_confidence = sum(detection.confidence for detection in detections) / len(detections)
    if len(detections) >= 5 or average_confidence >= 0.60:
        return "alto"
    return "moderado"


def generate_video_report(detections: list[VideoDetection]) -> str:
    """Generate a structured clinical screening report from YOLO detections."""
    total = len(detections)
    frames_with_detections = len({detection.frame_index for detection in detections})
    average_confidence = (
        sum(detection.confidence for detection in detections) / total if total else 0.0
    )
    risk = classify_video_risk(detections)

    if detections:
        detection_lines = [
            (
                f"- frame {detection.frame_index} "
                f"({detection.timestamp_sec:.2f}s): {detection.class_name}, "
                f"confidence {detection.confidence:.2f}, bbox {detection.bbox}"
            )
            for detection in detections[:10]
        ]
    else:
        detection_lines = ["- Nenhuma deteccao visual de sangramento anomalo foi encontrada."]

    if total > 10:
        detection_lines.append(f"- {total - 10} deteccoes adicionais omitidas do resumo.")

    return "\n".join(
        [
            "Laudo de Video - Deteccao de Sangramento Anomalo",
            "",
            "1. Resumo tecnico",
            f"- Total de deteccoes: {total}",
            f"- Frames com deteccao: {frames_with_detections}",
            f"- Confidence medio: {average_confidence:.2f}",
            f"- Classificacao de risco: {risk}",
            "",
            "2. Deteccoes observadas",
            *detection_lines,
            "",
            "3. Interpretacao clinica",
            (
                "- Achados visuais sugerem necessidade de revisao clinica prioritaria."
                if risk == "alto"
                else "- Achados visuais devem ser revisados no contexto do procedimento."
            ),
            "",
            "4. Limitacoes",
            "- Este laudo e apoio a triagem visual e nao substitui avaliacao profissional.",
            "- Falsos positivos e falsos negativos podem ocorrer conforme qualidade do video e modelo.",
        ]
    )
