"""Per-pixel aggregation of a stack of warped depth candidates."""

from __future__ import annotations

import numpy as np

from pim.fusion.types import FusionConfig


def aggregate(
    depths: np.ndarray, weights: np.ndarray, hits: np.ndarray, config: FusionConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(fused float32 depth with NaN holes, total weight, number of contributing candidates) per pixel."""
    if depths.ndim != 3:
        raise ValueError(f"depths must be (K, h, w), got {depths.shape}")
    for name, array in (("weights", weights), ("hits", hits)):
        if array.shape != depths.shape:
            raise ValueError(f"{name} {array.shape} must match depths {depths.shape}")
    shape = depths.shape[1:]
    if depths.shape[0] == 0:
        return np.full(shape, np.nan, dtype=np.float32), np.zeros(shape, np.float32), np.zeros(shape, np.int32)

    with np.errstate(invalid="ignore"):
        kept = hits.astype(bool) & np.isfinite(depths) & (depths > 0) & (weights > 0)
    depths = np.where(kept, depths, np.nan)
    if config.min_support > 1:
        kept = _consistent(depths, kept, config.min_support, config.front_tolerance)
    if config.front_layer:
        kept = _front(depths, kept, config.front_tolerance)
    depths = np.where(kept, depths, np.nan)
    weights = np.where(kept, weights, 0).astype(np.float64)

    if config.aggregator == "weighted_mean":
        fused, total, count = _weighted_mean(depths, weights, kept)
    elif config.aggregator == "weighted_median":
        fused, total, count = _weighted_median(depths, weights, kept)
    else:
        fused, total, count = _trimmed_mean(depths, weights, kept, config.trim_fraction)

    # A lone candidate is returned as is, so that a single-layer fusion reproduces its input bitwise.
    single = kept.sum(axis=0) == 1
    fused = np.where(single, np.where(kept, depths, np.inf).min(axis=0), fused)
    fused = np.where(count > 0, fused, np.nan)
    return fused.astype(np.float32), total.astype(np.float32), count.astype(np.int32)


def _consistent(depths: np.ndarray, kept: np.ndarray, min_support: int, tolerance: float) -> np.ndarray:
    # Candidate k is supported by m when |d_m - d_k| <= tolerance * d_k, the visibility rule with d_k as
    # reference; k counts itself, and pixels with fewer than min_support candidates are left unfiltered.
    support = np.empty(depths.shape, dtype=np.int32)
    for k in range(depths.shape[0]):
        support[k] = (np.abs(depths - depths[k]) <= tolerance * depths[k]).sum(axis=0)
    too_few = kept.sum(axis=0) < min_support
    return kept & ((support >= min_support) | too_few)


def _front(depths: np.ndarray, kept: np.ndarray, tolerance: float) -> np.ndarray:
    # Only candidates within tolerance of the nearest surviving one are kept, so a deeper surface that
    # leaked through a gap in a nearer one, or is occluded in the target, cannot pull the depth back.
    front = np.where(kept, depths, np.inf).min(axis=0)
    return kept & (depths <= front * (1 + tolerance))


def _weighted_mean(
    depths: np.ndarray, weights: np.ndarray, kept: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    total = weights.sum(axis=0)
    weighted = (weights * np.where(kept, depths, 0)).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return weighted / total, total, kept.sum(axis=0)


def _sorted(depths: np.ndarray, weights: np.ndarray, kept: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(np.where(kept, depths, np.inf), axis=0)
    return np.take_along_axis(depths, order, axis=0), np.take_along_axis(weights, order, axis=0)


def _weighted_median(
    depths: np.ndarray, weights: np.ndarray, kept: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Midpoint of the lowest depths whose cumulative weight reaches and passes half the total, which is
    # the ordinary median for equal weights.
    depths, weights = _sorted(depths, weights, kept)
    cumulative = np.cumsum(weights, axis=0)
    total = cumulative[-1]
    half, eps = total / 2, 1e-9 * total
    low = np.argmax(cumulative >= half - eps, axis=0)[None]
    high = np.argmax(cumulative > half + eps, axis=0)[None]
    fused = 0.5 * (np.take_along_axis(depths, low, axis=0)[0] + np.take_along_axis(depths, high, axis=0)[0])
    return fused, total, kept.sum(axis=0)


def _trimmed_mean(
    depths: np.ndarray, weights: np.ndarray, kept: np.ndarray, trim_fraction: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    depths, weights = _sorted(depths, weights, kept)
    n = kept.sum(axis=0)
    trim = np.floor(trim_fraction * n).astype(np.int64)
    rank = np.arange(depths.shape[0])[:, None, None]
    used = (rank >= trim) & (rank < n - trim)
    return _weighted_mean(depths, np.where(used, weights, 0), used)
