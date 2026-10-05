"""TUM RGB-D sequence discovery and frame loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class RgbFrame:
    timestamp: float
    path: Path


def find_sequences(root: Path) -> list[Path]:
    """Every directory under `root` that holds a TUM `rgb.txt`, sorted by path."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"dataset root {root} does not exist")
    sequences = sorted(p.parent for p in root.rglob("rgb.txt"))
    if not sequences:
        raise FileNotFoundError(f"no TUM sequence (directory containing rgb.txt) under {root}")
    return sequences


def read_rgb_list(sequence: Path) -> list[RgbFrame]:
    """Frames listed in `rgb.txt`, in file order."""
    listing = Path(sequence) / "rgb.txt"
    frames = []
    for line in listing.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        timestamp, relative_path = line.split()[:2]
        frames.append(RgbFrame(float(timestamp), Path(sequence) / relative_path))
    if not frames:
        raise ValueError(f"{listing} lists no frames")
    return frames


def load_rgb(path: Path) -> np.ndarray:
    """An HxWx3 uint8 RGB image."""
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)
