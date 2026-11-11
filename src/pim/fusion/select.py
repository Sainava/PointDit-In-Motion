"""Choice of source frames for a target time, by timestamp rather than list position."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from pim.data.tum import TIMESTAMP_EPS
from pim.fusion.types import FusionConfig


def select_sources(timestamps: Sequence[float], target_time: float, config: FusionConfig) -> list[int]:
    """Sorted indices of the chosen frames; any frame at the target time itself is never chosen."""
    times = np.asarray(timestamps, dtype=np.float64)
    if times.ndim != 1:
        raise ValueError(f"timestamps must be 1-D, got shape {times.shape}")
    if not np.all(np.isfinite(times)) or not np.isfinite(target_time):
        raise ValueError("timestamps and target_time must be finite")
    if np.any(np.diff(times) <= 0):
        raise ValueError("timestamps must be strictly increasing")
    if config.window == 0 or times.size == 0:
        return []

    before = np.flatnonzero(times < target_time - TIMESTAMP_EPS)[::-1]
    after = np.flatnonzero(times > target_time + TIMESTAMP_EPS)
    sides = [(before, -1.0)] + ([(after, 1.0)] if config.mode == "centred" else [])
    chosen: set[int] = set()
    for side, sign in sides:
        if config.selection == "adjacent":
            chosen.update(int(i) for i in side[: config.window])
        else:
            chosen.update(_near_strides(times, side, target_time, sign, config))
    return sorted(chosen)


def _near_strides(
    times: np.ndarray, side: np.ndarray, target_time: float, sign: float, config: FusionConfig
) -> list[int]:
    if side.size == 0:
        return []
    wanted = target_time + sign * config.stride_ms / 1000.0 * np.arange(1, config.window + 1)
    gaps = np.abs(times[side][None, :] - wanted[:, None])
    closest = np.argmin(gaps, axis=1)
    within = gaps[np.arange(wanted.size), closest] <= config.stride_tolerance_ms / 1000.0
    return [int(side[c]) for c in closest[within]]
