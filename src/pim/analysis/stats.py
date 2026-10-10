"""Rank correlations, moving-block bootstrap and temporal spike detection for autocorrelated per-frame series."""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

MIN_PAIRS = 3


def average_ranks(x: np.ndarray) -> np.ndarray:
    """1-based ranks; tied values share the mean of their ranks."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError(f"expected a 1-D array, got shape {x.shape}")
    if np.isnan(x).any():
        raise ValueError("cannot rank NaN values; drop them first")
    order = np.argsort(x, kind="stable")
    sorted_x = x[order]
    starts = np.flatnonzero(np.r_[True, sorted_x[1:] != sorted_x[:-1]])
    ends = np.r_[starts[1:], len(x)]
    ranks = np.empty(len(x))
    ranks[order] = np.repeat((starts + ends + 1) / 2.0, ends - starts)
    return ranks


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x, y = x - x.mean(), y - y.mean()
    denominator = np.sqrt((x @ x) * (y @ y))
    return float(x @ y / denominator) if denominator > 0 else float("nan")


def _rank_residuals(values: np.ndarray, control_ranks: np.ndarray) -> np.ndarray:
    ranks = average_ranks(values) / (len(values) + 1)
    design = np.column_stack([np.ones(len(values)), control_ranks])
    return ranks - design @ np.linalg.lstsq(design, ranks, rcond=None)[0]


def pooled_partial_spearman(groups: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> float:
    """Pearson correlation of rank(y) and rank(x) residuals pooled over groups; ranking and the regression on the
    controls' ranks are done within each group, so differences between groups do not enter the correlation.

    Each group is (x, y, controls) with controls of shape (n, k), k >= 0. Rows with any NaN are dropped.
    """
    x_parts, y_parts = [], []
    for x, y, controls in groups:
        x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
        controls = np.asarray(controls, dtype=np.float64).reshape(len(x), -1)
        keep = ~(np.isnan(x) | np.isnan(y) | np.isnan(controls).any(axis=1))
        if keep.sum() < MIN_PAIRS + controls.shape[1]:
            continue
        control_ranks = np.column_stack([average_ranks(c) for c in controls[keep].T]) if controls.shape[1] else \
            np.empty((int(keep.sum()), 0))
        x_parts.append(_rank_residuals(x[keep], control_ranks))
        y_parts.append(_rank_residuals(y[keep], control_ranks))
    if not x_parts:
        return float("nan")
    return pearson(np.concatenate(x_parts), np.concatenate(y_parts))


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return pooled_partial_spearman([(x, y, np.empty((len(x), 0)))])


def partial_spearman(x: np.ndarray, y: np.ndarray, controls: np.ndarray) -> float:
    return pooled_partial_spearman([(x, y, controls)])


def moving_block_indices(n: int, block_length: int, resamples: int, rng: np.random.Generator) -> np.ndarray:
    """(resamples, n) indices built from contiguous blocks with uniformly drawn starts, truncated to length n."""
    if not 1 <= block_length <= n:
        raise ValueError(f"block length must be in [1, {n}], got {block_length}")
    blocks = -(-n // block_length)
    starts = rng.integers(0, n - block_length + 1, size=(resamples, blocks))
    indices = starts[:, :, None] + np.arange(block_length)
    return indices.reshape(resamples, -1)[:, :n]


def bootstrap(statistic: Callable[[list[np.ndarray]], float], indices: Sequence[np.ndarray]) -> np.ndarray:
    """statistic evaluated on each resample; indices holds one (resamples, n_group) array per group."""
    resamples = {len(group) for group in indices}
    if len(resamples) != 1:
        raise ValueError(f"every group needs the same number of resamples, got {sorted(resamples)}")
    return np.array([statistic([group[b] for group in indices]) for b in range(resamples.pop())])


def percentile_ci(values: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return float("nan"), float("nan")
    tail = 50.0 * (1.0 - level)
    low, high = np.percentile(values, [tail, 100.0 - tail])
    return float(low), float(high)


def fast_slow_median_difference(speed: np.ndarray, value: np.ndarray) -> float:
    """Median value of the fastest third minus that of the slowest third (by speed), NaN rows dropped."""
    keep = ~(np.isnan(speed) | np.isnan(value))
    speed, value = speed[keep], value[keep]
    third = len(speed) // 3
    if third == 0:
        return float("nan")
    order = np.argsort(speed, kind="stable")
    return float(np.median(value[order[-third:]]) - np.median(value[order[:third]]))


def neighbour_medians(values: np.ndarray, radius: int) -> np.ndarray:
    """Median of the non-NaN values within ±radius positions of each position, excluding the position itself."""
    values = np.asarray(values, dtype=np.float64)
    medians = np.full(len(values), np.nan)
    for i in range(len(values)):
        window = np.r_[values[max(0, i - radius):i], values[i + 1:i + radius + 1]]
        window = window[~np.isnan(window)]
        if len(window):
            medians[i] = np.median(window)
    return medians


def detect_spikes(values: np.ndarray, radius: int, min_excess: float, min_relative_excess: float) -> np.ndarray:
    """A position is a spike if it exceeds its neighbour median by more than max(min_excess, min_relative_excess x median)."""
    values = np.asarray(values, dtype=np.float64)
    medians = neighbour_medians(values, radius)
    threshold = np.maximum(min_excess, min_relative_excess * medians)
    with np.errstate(invalid="ignore"):
        return np.nan_to_num(values - medians, nan=-np.inf) > threshold
