from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))

import generate_synthetic_data as synthetic


def _label_rows(label_path: Path) -> list[list[float]]:
    rows = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        rows.append([float(value) for value in line.split()])
    return rows


def test_generate_frame_returns_image_and_yolo_compatible_boxes():
    rng = np.random.default_rng(123)

    image, boxes = synthetic.generate_frame(128, rng, min_blobs=2, max_blobs=2)

    assert image.shape == (128, 128, 3)
    assert image.dtype == np.uint8
    assert len(boxes) == 2
    for box in boxes:
        values = box.to_yolo(128)
        assert all(0 < value <= 1 for value in values)


def test_generate_dataset_writes_yolov8_structure(tmp_path):
    output_dir = tmp_path / "synthetic_bleeding"

    summary = synthetic.generate_dataset(
        output_dir,
        count=10,
        image_size=96,
        val_ratio=0.2,
        seed=7,
        min_blobs=1,
        max_blobs=1,
    )

    assert summary == {"train": 8, "val": 2, "total": 10}
    assert len(list((output_dir / "images" / "train").glob("*.png"))) == 8
    assert len(list((output_dir / "images" / "val").glob("*.png"))) == 2
    assert len(list((output_dir / "labels" / "train").glob("*.txt"))) == 8
    assert len(list((output_dir / "labels" / "val").glob("*.txt"))) == 2
    for split in ("train", "val"):
        image_stems = {path.stem for path in (output_dir / "images" / split).glob("*.png")}
        label_stems = {path.stem for path in (output_dir / "labels" / split).glob("*.txt")}
        assert image_stems == label_stems

    dataset_yaml = (output_dir / "dataset.yaml").read_text(encoding="utf-8")
    assert f"path: {output_dir.as_posix()}" in dataset_yaml
    assert "train: images/train" in dataset_yaml
    assert "val: images/val" in dataset_yaml
    assert "names: ['anomalous_bleeding']" in dataset_yaml


def test_generate_dataset_writes_valid_label_values(tmp_path):
    output_dir = tmp_path / "synthetic_bleeding"

    synthetic.generate_dataset(
        output_dir,
        count=4,
        image_size=96,
        val_ratio=0.5,
        seed=11,
        min_blobs=2,
        max_blobs=2,
    )

    label_paths = sorted((output_dir / "labels").rglob("*.txt"))
    assert len(label_paths) == 4
    for label_path in label_paths:
        rows = _label_rows(label_path)
        assert len(rows) == 2
        for row in rows:
            assert row[0] == 0
            assert all(0 <= value <= 1 for value in row[1:])
            assert row[3] > 0
            assert row[4] > 0


def test_generate_dataset_is_deterministic_for_same_seed(tmp_path):
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"

    synthetic.generate_dataset(first_dir, count=3, image_size=96, seed=99)
    synthetic.generate_dataset(second_dir, count=3, image_size=96, seed=99)

    first_image = first_dir / "images" / "train" / "bleeding_train_00000.png"
    assert first_image.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    first_labels = sorted(
        path.read_text(encoding="utf-8")
        for path in (first_dir / "labels").rglob("*.txt")
    )
    second_labels = sorted(
        path.read_text(encoding="utf-8")
        for path in (second_dir / "labels").rglob("*.txt")
    )
    assert first_labels == second_labels


def test_generate_dataset_rejects_invalid_arguments(tmp_path):
    rng = np.random.default_rng(1)

    with pytest.raises(ValueError, match="image_size"):
        synthetic.generate_frame(32, rng)

    with pytest.raises(ValueError, match="count"):
        synthetic.generate_dataset(tmp_path, count=0)
