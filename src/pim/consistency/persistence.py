"""Persistence decomposition of per-pixel depth error between two posed frames: frame i's GT-aligned prediction warped
into j and frame j's own prediction, both scored as relative error against GT_j on the M1 pixels."""

from __future__ import annotations

import csv
import math
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pim.analysis.stats import average_ranks, pearson
from pim.consistency.evaluation import (
    PROGRESS_EVERY, FrameStore, PairTask, SequenceJob, atomic_write, cell_value, evaluation_order, load_window,
    peak_memory_mib, plan_pairs,
)
from pim.consistency.metrics import MIN_VISIBLE_PIXELS, FrameInputs, _tertile_masks, frame_inputs, visibility
from pim.consistency.warp import depth_pose, relative_pose, warp_depth
from pim.data.tum import Frame, read_groundtruth
from pim.evaluation import load_prediction, prediction_path, set_torch_threads
from pim.sampling import pair_key

REGIONS = ("visible", "interior", "visible_near", "visible_mid", "visible_far")
STATS = ("n", "median_abs_e_i", "median_abs_e_j", "median_abs_diff", "pearson", "spearman", "persistent_fraction")
PAIR_COLUMNS = ("sequence", "gap_ms", "frame_i", "frame_j", "actual_gap_ms", "too_few_pixels",
                *(f"{region}_{stat}" for region in REGIONS for stat in STATS))
M1_COLUMNS = {"visible_median_abs_diff": "m1_visible_median", "interior_median_abs_diff": "m1_interior_median"}
REGRESSION_TOLERANCE = 1e-6


@dataclass(frozen=True)
class PairErrors:
    e_i: np.ndarray
    """(ẑ_i→j − z_gt_j) / z_gt_j, NaN where not scored."""
    e_j: np.ndarray
    """(ẑ_j − z_gt_j) / z_gt_j, NaN where not scored."""
    regions: dict[str, np.ndarray]
    too_few: bool


def pair_errors(
    frame_i: FrameInputs, frame_j: FrameInputs, K: np.ndarray, T_j_from_i: np.ndarray, T_gt_j_from_i: np.ndarray,
) -> PairErrors:
    """Pixels, poses and thirds as M1 in pair_consistency: visibility from T_gt_j_from_i, frame i's aligned depth
    warped with T_j_from_i, thirds of GT_j depth over the visible (or interior) pixels."""
    if frame_i.aligned_depth is None or frame_j.aligned_depth is None:
        raise ValueError("both frames need a GT-aligned prediction")
    vis = visibility(frame_i, frame_j, K, T_gt_j_from_i)
    anchor, anchor_hit = warp_depth(frame_i.aligned_depth, frame_i.gt_valid, K, T_j_from_i, vis.visible.shape)
    scored = vis.visible & anchor_hit
    gt_j = frame_j.gt_depth.astype(np.float64)
    e_i, e_j = np.full(gt_j.shape, np.nan), np.full(gt_j.shape, np.nan)
    e_i[scored] = (anchor[scored].astype(np.float64) - gt_j[scored]) / gt_j[scored]
    e_j[scored] = (frame_j.aligned_depth[scored].astype(np.float64) - gt_j[scored]) / gt_j[scored]

    # As in pair_consistency, a pixel set with fewer than MIN_VISIBLE_PIXELS pixels is not scored at all.
    empty = np.zeros_like(scored)
    too_few = int(vis.visible.sum()) < MIN_VISIBLE_PIXELS
    if too_few:
        return PairErrors(e_i, e_j, {name: empty for name in REGIONS}, too_few)
    interior = scored & vis.interior if vis.interior.sum() >= MIN_VISIBLE_PIXELS else empty
    thirds = (scored & third for third in _tertile_masks(gt_j, vis.visible))
    return PairErrors(e_i, e_j, dict(zip(REGIONS, (scored, interior, *thirds))), too_few)


def persistence_stats(e_i: np.ndarray, e_j: np.ndarray) -> dict[str, float]:
    """Medians x100; P = 1 − Var(E_i − E_j) / (Var(E_i) + Var(E_j)), 0 for independent errors, 1 for identical."""
    if e_i.shape != e_j.shape or e_i.ndim != 1:
        raise ValueError(f"expected two 1-D arrays of equal length, got {e_i.shape} and {e_j.shape}")
    n = len(e_i)
    if n < MIN_VISIBLE_PIXELS:
        return {"n": n, **{stat: math.nan for stat in STATS[1:]}}
    diff = e_i - e_j
    total = float(e_i.var() + e_j.var())
    return {
        "n": n,
        "median_abs_e_i": 100.0 * float(np.median(np.abs(e_i))),
        "median_abs_e_j": 100.0 * float(np.median(np.abs(e_j))),
        "median_abs_diff": 100.0 * float(np.median(np.abs(diff))),
        "pearson": pearson(e_i, e_j),
        "spearman": pearson(average_ranks(e_i), average_ranks(e_j)),
        "persistent_fraction": 1.0 - float(diff.var()) / total if total > 0 else math.nan,
    }


def region_stats(errors: PairErrors) -> dict[str, dict[str, float]]:
    return {name: persistence_stats(errors.e_i[mask], errors.e_j[mask]) for name, mask in errors.regions.items()}


@dataclass(frozen=True)
class SequenceOutcome:
    name: str
    pairs: dict[str, int]
    seconds: float
    frame_seconds: float
    peak_mem_mib: float


def persistence_csv(out: Path, name: str) -> Path:
    return out / f"persistence_{name}.csv"


def complete(out: Path, name: str, pairs: Sequence[PairTask]) -> bool:
    path = persistence_csv(out, name)
    if not path.is_file():
        return False
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if tuple(reader.fieldnames or ()) != PAIR_COLUMNS:
            return False
        return {pair_key(r) for r in reader} == {p.key for p in pairs}


def evaluate_sequence(job: SequenceJob) -> SequenceOutcome:
    started = time.perf_counter()
    set_torch_threads(job.threads)
    sequence, frames = load_window(job)
    pairs = plan_pairs(frames, job.gaps_ms)
    if not pairs:
        raise ValueError(f"{job.name}: no frame pairs at gaps {job.gaps_ms} ms")
    gt_trajectory = read_groundtruth(sequence.root)
    frame_seconds = 0.0

    def load(frame: Frame) -> FrameInputs:
        nonlocal frame_seconds
        t0 = time.perf_counter()
        depth, valid = frame.load_depth()
        points = load_prediction(prediction_path(job.run_dir, job.name, frame.index), frame, sequence.root).points
        inputs = frame_inputs(depth, valid, sequence.K, points)
        frame_seconds += time.perf_counter() - t0
        return inputs

    store = FrameStore(load, ((p.frame_i, p.frame_j) for p in pairs))
    rows: dict[tuple[float, int, int], dict[str, Any]] = {}
    print(f"{job.name}: {len(frames)} frames, {len(pairs)} pairs at {list(job.gaps_ms)} ms (pid {os.getpid()})",
          flush=True)
    for done, pair in enumerate(evaluation_order(pairs), start=1):
        T_j_from_i = relative_pose(pair.frame_i.T_world_cam, pair.frame_j.T_world_cam)
        T_gt_j_from_i = relative_pose(depth_pose(pair.frame_i, *gt_trajectory), depth_pose(pair.frame_j, *gt_trajectory))
        errors = pair_errors(store.get(pair.frame_i), store.get(pair.frame_j), sequence.K, T_j_from_i, T_gt_j_from_i)
        store.release(pair.frame_i, pair.frame_j)
        rows[pair.key] = pair_row(job.name, pair, errors)
        if done % PROGRESS_EVERY == 0 or done == len(pairs):
            elapsed = time.perf_counter() - started
            print(f"  {job.name}: {done}/{len(pairs)} pairs, {elapsed / done:.2f} s/pair, "
                  f"{len(store.cached)} frames cached", flush=True)
    write_rows(persistence_csv(job.out, job.name), [rows[p.key] for p in pairs])
    return SequenceOutcome(job.name, {f"{gap:g}": sum(p.gap_ms == gap for p in pairs) for gap in job.gaps_ms},
                           time.perf_counter() - started, frame_seconds, peak_memory_mib())


def pair_row(name: str, pair: PairTask, errors: PairErrors) -> dict[str, Any]:
    row: dict[str, Any] = {
        "sequence": name,
        "gap_ms": f"{pair.gap_ms:g}",
        "frame_i": pair.frame_i.index,
        "frame_j": pair.frame_j.index,
        "actual_gap_ms": f"{(pair.frame_j.timestamp - pair.frame_i.timestamp) * 1000.0:.3f}",
        "too_few_pixels": errors.too_few,
    }
    for region, stats in region_stats(errors).items():
        row.update({f"{region}_{stat}": value if stat == "n" else _cell(value) for stat, value in stats.items()})
    return row


def _cell(value: float) -> str:
    return "" if math.isnan(value) else repr(float(value))


def write_rows(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    def write(f) -> None:
        writer = csv.DictWriter(f, fieldnames=PAIR_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    atomic_write(path, write, newline="")


def read_rows(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if tuple(reader.fieldnames or ()) != PAIR_COLUMNS:
            raise ValueError(f"{path} does not have the persistence CSV columns")
        return list(reader)


def m1_regression(
    rows: Sequence[dict[str, str]], reference_rows: Sequence[dict[str, str]], tolerance: float = REGRESSION_TOLERANCE,
) -> tuple[float, list[str]]:
    """(largest difference, failures): the pair keys must match the reference exactly, and each pair's median
    |E_i − E_j| must equal the reference M1 median within tolerance, NaN only where the reference is NaN."""
    actual = {pair_key(r): r for r in rows}
    reference = {pair_key(r): r for r in reference_rows}
    failures = []
    if len(actual) != len(rows) or len(reference) != len(reference_rows):
        failures.append(f"duplicate pair keys: {len(rows) - len(actual)} in the persistence rows, "
                        f"{len(reference_rows) - len(reference)} in the reference")
    only_actual, only_reference = sorted(actual.keys() - reference.keys()), sorted(reference.keys() - actual.keys())
    if only_actual or only_reference:
        failures.append(f"pair keys differ: {len(only_actual)} only in the persistence rows (e.g. {only_actual[:3]}), "
                        f"{len(only_reference)} only in the reference (e.g. {only_reference[:3]})")
    if not actual:
        failures.append("no pairs to check")
    largest = 0.0
    for key in sorted(actual.keys() & reference.keys()):
        row, expected = actual[key], reference[key]
        if row["too_few_pixels"] != expected["too_few_pixels"]:
            failures.append(f"pair {key}: too_few_pixels {row['too_few_pixels']} vs {expected['too_few_pixels']}")
        for column, reference_column in M1_COLUMNS.items():
            value, wanted = cell_value(row[column]), cell_value(expected[reference_column])
            if math.isnan(value) and math.isnan(wanted):
                continue
            difference = abs(value - wanted)
            if not difference <= tolerance:
                failures.append(f"pair {key} {column}: {value!r} vs {reference_column} {wanted!r}")
            else:
                largest = max(largest, difference)
    return largest, failures
