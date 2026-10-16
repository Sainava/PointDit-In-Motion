"""Sampler-initialisation comparisons: same-frame disagreement between predictions, determinism of zero
initialisation, and block-bootstrap summaries of per-frame and per-pair differences between configs."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from pim.analysis.stats import moving_block_indices, percentile_ci
from pim.consistency.evaluation import METRIC_COLUMNS, cell_value
from pim.consistency.metrics import MIN_VISIBLE_PIXELS, align_depth_to_anchor

DETERMINISM_TOLERANCE = 2e-3
REGRESSION_TOLERANCE = 1e-3
BLOCK_LENGTH = 15
RESAMPLES = 2000
BOOTSTRAP_SEED = 0


@dataclass(frozen=True)
class Disagreement:
    pixels: int
    median: float
    p90: float


@dataclass(frozen=True)
class PredictionDifference:
    max_abs_diff: float
    rgb_file_a: str
    rgb_file_b: str
    valid_mismatch: int

    @property
    def rgb_match(self) -> bool:
        return self.rgb_file_a == self.rgb_file_b

    def passed(self, tolerance: float = DETERMINISM_TOLERANCE) -> bool:
        return self.rgb_match and self.max_abs_diff <= tolerance


@dataclass(frozen=True)
class Estimate:
    value: float
    ci_low: float
    ci_high: float


def comparisons(noise: Sequence[str], zeros: str) -> list[tuple[str, str]]:
    """(reference, aligned) config names: every pair of noise configs, then zeros against each noise config."""
    return [*combinations(noise, 2), *((zeros, name) for name in noise)]


def disagreement(reference_depth: np.ndarray, other_raw_depth: np.ndarray, mask: np.ndarray) -> Disagreement:
    """Relative difference x100 between the reference's GT-aligned depth and the other prediction's raw depth fitted
    to it (scale + shift on the 64x64 subsample, as M2 fits frame j to frame i's anchor), over mask."""
    mask = mask & (reference_depth > 0)
    pixels = int(mask.sum())
    if pixels < MIN_VISIBLE_PIXELS:
        return Disagreement(pixels, math.nan, math.nan)
    aligned = align_depth_to_anchor(other_raw_depth, reference_depth.astype(np.float64), mask)
    reference = reference_depth[mask].astype(np.float64)
    error = np.abs(aligned[mask] - reference) / reference
    median, p90 = np.percentile(error, [50, 90])
    return Disagreement(pixels, 100.0 * float(median), 100.0 * float(p90))


def prediction_difference(path_a: Path, path_b: Path) -> PredictionDifference:
    arrays = []
    for path in (path_a, path_b):
        if not path.is_file():
            raise FileNotFoundError(f"prediction {path} not found")
        with np.load(path) as data:
            arrays.append((data["points"].astype(np.float32), data["valid"].astype(bool), str(data["rgb_file"])))
    (points_a, valid_a, rgb_a), (points_b, valid_b, rgb_b) = arrays
    if points_a.shape != points_b.shape:
        raise ValueError(f"{path_a} has points {points_a.shape} but {path_b} has {points_b.shape}")
    return PredictionDifference(float(np.abs(points_a - points_b).max()), rgb_a, rgb_b,
                                int((valid_a != valid_b).sum()))


def pair_key(row: dict[str, str]) -> tuple[float, int, int]:
    return float(row["gap_ms"]), int(row["frame_i"]), int(row["frame_j"])


def consistency_regression(
    rows: Sequence[dict[str, str]], reference_rows: Sequence[dict[str, str]], tolerance: float = REGRESSION_TOLERANCE
) -> tuple[float, list[str]]:
    """(largest difference, failures): every row must match the reference row of the same pair in every metric
    column within tolerance, with NaN only where the reference is NaN."""
    reference = {pair_key(r): r for r in reference_rows}
    failures, largest = [], 0.0
    if not rows:
        failures.append("no pairs to check")
    for row in rows:
        key = pair_key(row)
        expected = reference.get(key)
        if expected is None:
            failures.append(f"pair {key} is not in the reference")
            continue
        if row["too_few_pixels"] != expected["too_few_pixels"]:
            failures.append(f"pair {key}: too_few_pixels {row['too_few_pixels']} vs {expected['too_few_pixels']}")
        for column in METRIC_COLUMNS:
            actual, wanted = cell_value(row[column]), cell_value(expected[column])
            if math.isnan(actual) and math.isnan(wanted):
                continue
            difference = abs(actual - wanted)
            if not difference <= tolerance:
                failures.append(f"pair {key} {column}: {actual} vs {wanted}")
            elif difference > largest:
                largest = difference
    return largest, failures


def nanmedian(values: np.ndarray) -> float:
    kept = values[~np.isnan(values)]
    return float(np.median(kept)) if kept.size else math.nan


def block_bootstrap(
    statistic: Callable[[np.ndarray], float], n: int, block_length: int = BLOCK_LENGTH, resamples: int = RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> Estimate:
    """statistic(indices) on the series in order and on moving-block resamples of its positions."""
    value = statistic(np.arange(n))
    if n < block_length:
        return Estimate(value, math.nan, math.nan)
    indices = moving_block_indices(n, block_length, resamples, np.random.default_rng(seed))
    low, high = percentile_ci(np.array([statistic(idx) for idx in indices]))
    return Estimate(value, low, high)


def median_estimate(values: np.ndarray, **bootstrap_kwargs) -> Estimate:
    values = np.asarray(values, dtype=np.float64)
    return block_bootstrap(lambda idx: nanmedian(values[idx]), len(values), **bootstrap_kwargs)


def median_ratio_estimate(numerator: np.ndarray, denominator: np.ndarray, **bootstrap_kwargs) -> Estimate:
    """median(numerator) / median(denominator), resampling the paired series jointly."""
    numerator, denominator = np.asarray(numerator, np.float64), np.asarray(denominator, np.float64)
    if numerator.shape != denominator.shape:
        raise ValueError(f"paired series differ in length: {numerator.shape} vs {denominator.shape}")

    def ratio(idx: np.ndarray) -> float:
        bottom = nanmedian(denominator[idx])
        return nanmedian(numerator[idx]) / bottom if bottom > 0 else math.nan

    return block_bootstrap(ratio, len(numerator), **bootstrap_kwargs)
