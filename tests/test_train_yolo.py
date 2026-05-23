from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))

import train_yolo


def test_parse_args_defaults_to_phase4_paths_and_mps_device():
    args = train_yolo._parse_args([])

    assert args.data == train_yolo.DEFAULT_DATA
    assert args.project == train_yolo.DEFAULT_PROJECT
    assert args.model == "yolov8n.pt"
    assert args.epochs == 20
    assert args.device == "mps"


def test_train_yolo_rejects_missing_dataset(monkeypatch, tmp_path):
    calls = []

    class _FakeYOLO:
        def __init__(self, model):
            calls.append(model)

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=_FakeYOLO))

    summary = train_yolo.main(["--data", str(tmp_path / "missing.yaml")])

    assert summary == 1
    assert calls == []


def test_train_yolo_passes_options_to_ultralytics(monkeypatch, tmp_path):
    dataset = tmp_path / "dataset.yaml"
    dataset.write_text("path: synthetic\ntrain: images/train\nval: images/val\n", encoding="utf-8")
    calls = {}

    class _FakeYOLO:
        def __init__(self, model):
            calls["model"] = model

        def train(self, **kwargs):
            calls["train"] = kwargs
            return SimpleNamespace(
                save_dir=tmp_path / "artifacts" / "yolo" / "run",
                box=SimpleNamespace(map50=0.91, mp=0.82, mr=0.73),
            )

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=_FakeYOLO))

    summary = train_yolo.train_yolo(
        data=dataset,
        model="custom.pt",
        epochs=3,
        imgsz=320,
        batch=4,
        project=tmp_path / "artifacts" / "yolo",
        name="unit-test",
        device="cpu",
        log_path=None,
    )

    assert calls["model"] == "custom.pt"
    assert calls["train"] == {
        "data": str(dataset),
        "epochs": 3,
        "imgsz": 320,
        "batch": 4,
        "project": str(tmp_path / "artifacts" / "yolo"),
        "name": "unit-test",
        "device": "cpu",
    }
    assert summary.run_dir == tmp_path / "artifacts" / "yolo" / "run"
    assert summary.metrics == {"mAP50": 0.91, "precision": 0.82, "recall": 0.73}


def test_extract_metrics_reads_results_dict_fallbacks():
    results = SimpleNamespace(
        results_dict={
            "metrics/mAP50(B)": 0.88,
            "metrics/precision(B)": 0.79,
            "metrics/recall(B)": 0.69,
        }
    )

    assert train_yolo.extract_metrics(results) == {
        "mAP50": 0.88,
        "precision": 0.79,
        "recall": 0.69,
    }


def test_extract_metrics_preserves_zero_values():
    results = SimpleNamespace(box=SimpleNamespace(map50=0.0, mp=0.0, mr=0.0))

    assert train_yolo.extract_metrics(results) == {
        "mAP50": 0.0,
        "precision": 0.0,
        "recall": 0.0,
    }
