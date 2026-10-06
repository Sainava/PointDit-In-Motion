"""Hole counting and hole-minimising window selection over kept-frame timestamps."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

HOLE_FACTOR = 1.5


@dataclass(frozen=True)
class WindowChoice:
    start: int
    count: int
    holes: int


def hole_gaps(timestamps: np.ndarray | list[float]) -> np.ndarray:
    """(N-1,) bool, True where a gap exceeds HOLE_FACTOR times the median gap of the whole list."""
    times = np.asarray(timestamps, dtype=np.float64)
    if times.ndim != 1 or len(times) < 2:
        raise ValueError(f"need at least two timestamps, got shape {times.shape}")
    gaps = np.diff(times)
    if np.any(gaps <= 0):
        raise ValueError("timestamps must be strictly increasing")
    return gaps > HOLE_FACTOR * np.median(gaps)


def count_holes(timestamps: np.ndarray | list[float]) -> int:
    return int(hole_gaps(timestamps).sum())


def select_window(timestamps: np.ndarray | list[float], count: int) -> WindowChoice:
    """The `count`-frame window with the fewest holes; ties go to the earliest start."""
    holes = hole_gaps(timestamps).astype(np.int64)
    n = len(holes) + 1
    if not 1 <= count <= n:
        raise ValueError(f"window count must be in [1, {n}], got {count}")
    # A window starting at s spans gaps s .. s + count - 2.
    cumulative = np.concatenate([[0], np.cumsum(holes)])
    per_start = cumulative[count - 1:n] - cumulative[:n - count + 1]
    start = int(np.argmin(per_start))
    return WindowChoice(start, count, int(per_start[start]))
