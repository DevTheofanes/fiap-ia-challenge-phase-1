from __future__ import annotations

import sys
import builtins
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.multimodal import video
from src.multimodal.video import VideoDetection

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))

import analyze_video as analyze_video_cli


class _FakeBox:
    def __init__(self, *, conf: float, cls: int = 0, xyxy: tuple[float, float, float, float]):
        self.conf = conf
        self.cls = cls
        self.xyxy = xyxy


def test_analyze_video_rejects_missing_video(tmp_path):
    model_path = tmp_path / "best.pt"
    model_path.write_bytes(b"fake-model")

    with pytest.raises(FileNotFoundError, match="Video file not found"):
        video.analyze_video(tmp_path / "missing.mp4", model_path)


def test_analyze_video_rejects_missing_model(tmp_path):
    video_path = tmp_path / "sample.mp4"
    video_path.write_bytes(b"fake-video")

    with pytest.raises(FileNotFoundError, match="YOLO model file not found"):
        video.analyze_video(video_path, tmp_path / "missing.pt")


def test_analyze_video_requires_ultralytics(monkeypatch, tmp_path):
    video_path = tmp_path / "sample.mp4"
    model_path = tmp_path / "best.pt"
    video_path.write_bytes(b"fake-video")
    model_path.write_bytes(b"fake-model")
    monkeypatch.delitem(sys.modules, "ultralytics", raising=False)
    monkeypatch.setattr(video, "_video_fps", lambda path: 10.0)
    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "ultralytics":
            raise ImportError("blocked")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    with pytest.raises(RuntimeError, match="ultralytics is required"):
        video.analyze_video(video_path, model_path)


def test_analyze_video_passes_options_and_filters_detections(monkeypatch, tmp_path):
    video_path = tmp_path / "sample.mp4"
    model_path = tmp_path / "best.pt"
    video_path.write_bytes(b"fake-video")
    model_path.write_bytes(b"fake-model")
    calls = {}

    class _FakeYOLO:
        names = {0: "anomalous_bleeding"}

        def __init__(self, model):
            calls["model"] = model

        def predict(self, **kwargs):
            calls["predict"] = kwargs
            return iter(
                [
                    SimpleNamespace(
                        boxes=[
                            _FakeBox(conf=0.9, xyxy=(1, 2, 30, 40)),
                            _FakeBox(conf=0.2, xyxy=(5, 6, 7, 8)),
                        ]
                    ),
                    SimpleNamespace(boxes=[_FakeBox(conf=0.7, xyxy=(10, 20, 50, 80))]),
                ]
            )

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=_FakeYOLO))
    monkeypatch.setattr(video, "_video_fps", lambda path: 20.0)

    analysis = video.analyze_video(
        video_path,
        model_path,
        confidence_threshold=0.5,
        frame_stride=2,
    )

    assert calls["model"] == str(model_path)
    assert calls["predict"] == {
        "source": str(video_path),
        "stream": True,
        "conf": 0.5,
        "vid_stride": 2,
        "verbose": False,
    }
    assert analysis.frames_processed == 2
    assert analysis.detections == [
        VideoDetection(
            frame_index=0,
            timestamp_sec=0.0,
            class_name="anomalous_bleeding",
            confidence=0.9,
            bbox=(1.0, 2.0, 30.0, 40.0),
        ),
        VideoDetection(
            frame_index=2,
            timestamp_sec=0.1,
            class_name="anomalous_bleeding",
            confidence=0.7,
            bbox=(10.0, 20.0, 50.0, 80.0),
        ),
    ]


def test_generate_video_report_summarizes_risk_and_metrics():
    detections = [
        VideoDetection(0, 0.0, "anomalous_bleeding", 0.5, (1, 2, 3, 4)),
        VideoDetection(1, 0.1, "anomalous_bleeding", 0.7, (2, 3, 4, 5)),
    ]

    report = video.generate_video_report(detections)

    assert "Total de deteccoes: 2" in report
    assert "Frames com deteccao: 2" in report
    assert "Confidence medio: 0.60" in report
    assert "Classificacao de risco: alto" in report
    assert "nao substitui avaliacao profissional" in report


def test_generate_video_report_handles_no_detections():
    report = video.generate_video_report([])

    assert "Total de deteccoes: 0" in report
    assert "Confidence medio: 0.00" in report
    assert "Classificacao de risco: baixo" in report
    assert "Nenhuma deteccao visual" in report


def test_analyze_video_cli_defaults_to_phase4_model_path():
    args = analyze_video_cli._parse_args(["sample.mp4"])

    assert args.video_path == Path("sample.mp4")
    assert args.model_path == analyze_video_cli.DEFAULT_MODEL_PATH
    assert args.confidence_threshold == 0.25
    assert args.frame_stride == 1


def test_analyze_video_cli_prints_report(monkeypatch, tmp_path, capsys):
    video_path = tmp_path / "sample.mp4"
    model_path = tmp_path / "best.pt"

    monkeypatch.setattr(
        analyze_video_cli,
        "analyze_video",
        lambda *args, **kwargs: video.VideoAnalysis(video_path, model_path, 3, []),
    )
    monkeypatch.setattr(analyze_video_cli, "generate_video_report", lambda detections: "report")

    result = analyze_video_cli.main([str(video_path), "--model-path", str(model_path)])

    output = capsys.readouterr()
    assert result == 0
    assert "=== Video Analysis ===" in output.out
    assert "Frames processed: 3" in output.out
    assert "report" in output.out


def test_analyze_video_cli_returns_error(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(
        analyze_video_cli,
        "analyze_video",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    result = analyze_video_cli.main([str(tmp_path / "sample.mp4")])

    output = capsys.readouterr()
    assert result == 1
    assert "ERROR: boom" in output.err
