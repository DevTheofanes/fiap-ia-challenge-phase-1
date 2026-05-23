#!/usr/bin/env python
"""Generate synthetic YOLOv8 data for anomalous bleeding detection."""
from __future__ import annotations

import argparse
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np


CLASS_NAME = "anomalous_bleeding"
DEFAULT_OUTPUT_DIR = Path("data/synthetic_bleeding")


@dataclass(frozen=True)
class BoundingBox:
    x_min: int
    y_min: int
    x_max: int
    y_max: int

    def to_yolo(self, image_size: int) -> tuple[float, float, float, float]:
        width = self.x_max - self.x_min
        height = self.y_max - self.y_min
        x_center = self.x_min + width / 2
        y_center = self.y_min + height / 2
        return (
            x_center / image_size,
            y_center / image_size,
            width / image_size,
            height / image_size,
        )


def _make_background(image_size: int, rng: np.random.Generator) -> np.ndarray:
    y, x = np.ogrid[:image_size, :image_size]
    center = image_size / 2
    radius = np.sqrt((x - center) ** 2 + (y - center) ** 2) / center
    vignette = np.clip(1 - radius * 0.85, 0.08, 0.65)

    base = np.zeros((image_size, image_size, 3), dtype=np.float32)
    base[..., 0] = 30 * vignette
    base[..., 1] = 13 * vignette
    base[..., 2] = 10 * vignette
    noise = rng.normal(0, 5, size=base.shape)
    return np.clip(base + noise, 0, 255).astype(np.uint8)


def _draw_bleeding_blob(
    image: np.ndarray,
    rng: np.random.Generator,
) -> BoundingBox:
    image_size = image.shape[0]
    rx = int(rng.integers(max(12, image_size // 35), max(18, image_size // 9)))
    ry = int(rng.integers(max(10, image_size // 45), max(16, image_size // 11)))
    cx = int(rng.integers(rx, image_size - rx))
    cy = int(rng.integers(ry, image_size - ry))

    y, x = np.ogrid[:image_size, :image_size]
    ellipse = ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1
    edge = ((x - cx) / max(rx + 4, 1)) ** 2 + ((y - cy) / max(ry + 4, 1)) ** 2 <= 1
    halo = edge & ~ellipse

    red = int(rng.integers(145, 230))
    image[halo] = np.clip(image[halo].astype(np.int16) + np.array([35, 0, 0]), 0, 255)
    image[ellipse] = np.array([red, int(rng.integers(5, 35)), int(rng.integers(8, 35))], dtype=np.uint8)

    return BoundingBox(
        x_min=max(cx - rx, 0),
        y_min=max(cy - ry, 0),
        x_max=min(cx + rx, image_size),
        y_max=min(cy + ry, image_size),
    )


def generate_frame(
    image_size: int,
    rng: np.random.Generator,
    *,
    min_blobs: int = 1,
    max_blobs: int = 3,
) -> tuple[np.ndarray, list[BoundingBox]]:
    if image_size < 64:
        raise ValueError("image_size must be at least 64")
    if min_blobs < 1 or max_blobs < min_blobs:
        raise ValueError("Expected 1 <= min_blobs <= max_blobs")

    image = _make_background(image_size, rng)
    blob_count = int(rng.integers(min_blobs, max_blobs + 1))
    boxes = [_draw_bleeding_blob(image, rng) for _ in range(blob_count)]
    return image, boxes


def _write_label(path: Path, boxes: list[BoundingBox], image_size: int) -> None:
    lines = []
    for box in boxes:
        x_center, y_center, width, height = box.to_yolo(image_size)
        lines.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + chunk_type
        + data
        + struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
    )


def _write_png(path: Path, image: np.ndarray) -> None:
    height, width, channels = image.shape
    if channels != 3:
        raise ValueError("Expected an RGB image")

    raw_rows = b"".join(b"\x00" + image[row].tobytes() for row in range(height))
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    png = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", zlib.compress(raw_rows))
        + _png_chunk(b"IEND", b"")
    )
    path.write_bytes(png)


def _write_dataset_yaml(output_dir: Path) -> None:
    content = "\n".join(
        [
            f"path: {output_dir.resolve().as_posix()}",
            "train: images/train",
            "val: images/val",
            "nc: 1",
            f"names: ['{CLASS_NAME}']",
            "",
        ]
    )
    (output_dir / "dataset.yaml").write_text(content, encoding="utf-8")


def generate_dataset(
    output_dir: Path,
    *,
    count: int = 100,
    image_size: int = 640,
    val_ratio: float = 0.2,
    seed: int = 42,
    min_blobs: int = 1,
    max_blobs: int = 3,
) -> dict[str, int]:
    if count < 1:
        raise ValueError("count must be at least 1")
    if not 0 <= val_ratio < 1:
        raise ValueError("val_ratio must be in the range [0, 1)")

    output_dir = Path(output_dir)
    train_count = count - int(round(count * val_ratio))
    val_count = count - train_count
    rng = np.random.default_rng(seed)

    for split in ("train", "val"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    for index in range(count):
        split = "train" if index < train_count else "val"
        local_index = index if split == "train" else index - train_count
        stem = f"bleeding_{split}_{local_index:05d}"
        image, boxes = generate_frame(
            image_size,
            rng,
            min_blobs=min_blobs,
            max_blobs=max_blobs,
        )
        _write_png(output_dir / "images" / split / f"{stem}.png", image)
        _write_label(output_dir / "labels" / split / f"{stem}.txt", boxes, image_size)

    _write_dataset_yaml(output_dir)
    return {"train": train_count, "val": val_count, "total": count}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic YOLOv8 bleeding data.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-blobs", type=int, default=1)
    parser.add_argument("--max-blobs", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    summary = generate_dataset(
        args.output_dir,
        count=args.count,
        image_size=args.image_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
        min_blobs=args.min_blobs,
        max_blobs=args.max_blobs,
    )
    print(
        f"Generated {summary['total']} images "
        f"({summary['train']} train, {summary['val']} val) in {args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
