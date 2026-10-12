"""Per-sequence consistency evaluation over all frame pairs at fixed time gaps, with pair CSV and map output."""

from __future__ import annotations

import csv
import math
import os
import resource
import sys
import time
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pim.consistency.metrics import ErrorSummary, FrameInputs, PairConsistency, frame_inputs, pair_consistency
from pim.consistency.warp import depth_pose, pairs_by_gap, relative_pose
from pim.data.tum import Frame, TUMSequence, apply_window, load_sequence, read_groundtruth
from pim.evaluation import load_prediction, prediction_path, set_torch_threads
from pim.subsets import FrameRange

PAIR_TOLERANCE_MS = 10.0
MAPS_GAP_MS = 33.0
METRICS = {
    "gt_floor_depth": "gt_floor",
    "gt_floor_rgb": "gt_floor_rgb",
    "m1": "m1",
    "m2_floor": "m2_floor",
    "m2": "m2",
}
PIXEL_SETS = ("visible", "interior")
STATS = ("mean", "median", "p90", "near_median", "mid_median", "far_median")
METRIC_COLUMNS = tuple(f"{m}_{p}_{s}" for m in METRICS for p in PIXEL_SETS for s in STATS)
PAIR_COLUMNS = (
    "sequence", "gap_ms", "frame_i", "frame_j", "t_i", "t_j", "actual_gap_ms",
    "frac_visible", "frac_interior", "too_few_pixels", *METRIC_COLUMNS,
)
PROGRESS_EVERY = 100


@dataclass(frozen=True)
class PairTask:
    gap_ms: float
    frame_i: Frame
    frame_j: Frame

    @property
    def key(self) -> tuple[float, int, int]:
        return self.gap_ms, self.frame_i.index, self.frame_j.index


@dataclass(frozen=True)
class SequenceJob:
    name: str
    data_root: Path
    sequences_config: Path
    run_dir: Path
    gaps_ms: tuple[float, ...]
    out: Path
    save_maps: bool
    threads: int
    frame_range: FrameRange | None = None
    """Keep only pairs with both frames inside this range; pairing itself still runs over the whole window."""


@dataclass(frozen=True)
class SequenceOutcome:
    name: str
    frames: int
    pairs: dict[str, int]
    too_few_pairs: int
    seconds: float
    frame_seconds: float
    """Loading frames and upstream's per-frame alignment."""
    pair_seconds: float
    peak_mem_mib: float
    pid: int


class FrameStore:
    """Loads each frame's inputs on first use and drops them after the last pair that needs them."""

    def __init__(self, load: Callable[[Frame], FrameInputs], pairs: Iterable[tuple[Frame, Frame]]) -> None:
        self._load = load
        self._remaining = Counter(frame.index for pair in pairs for frame in pair)
        self._inputs: dict[int, FrameInputs] = {}
        self.loads = 0

    @property
    def cached(self) -> set[int]:
        return set(self._inputs)

    def get(self, frame: Frame) -> FrameInputs:
        if self._remaining[frame.index] <= 0:
            raise RuntimeError(f"frame {frame.index} was already released or is not part of any pair")
        if frame.index not in self._inputs:
            self._inputs[frame.index] = self._load(frame)
            self.loads += 1
        return self._inputs[frame.index]

    def release(self, *frames: Frame) -> None:
        for frame in frames:
            if self._remaining[frame.index] <= 0:
                raise RuntimeError(f"frame {frame.index} released more often than it is used")
            self._remaining[frame.index] -= 1
            if self._remaining[frame.index] == 0:
                self._inputs.pop(frame.index, None)


def plan_pairs(
    frames: Sequence[Frame], gaps_ms: Sequence[float], frame_range: FrameRange | None = None
) -> list[PairTask]:
    """Pairs in output order: by gap, then by frame i. With frame_range, the pairs of the whole window whose two
    frames both lie inside it, so they are exactly the corresponding rows of a whole-window run."""
    pairs = [PairTask(gap, i, j) for gap in gaps_ms for i, j in pairs_by_gap(frames, gap, PAIR_TOLERANCE_MS)]
    if frame_range is None:
        return pairs
    return [p for p in pairs if frame_range.contains(p.frame_i.index) and frame_range.contains(p.frame_j.index)]


def evaluation_order(pairs: Sequence[PairTask]) -> list[PairTask]:
    # Sweeping frame i in time order keeps only a max-gap-wide band of frames in memory.
    return sorted(pairs, key=lambda p: (p.frame_i.index, p.gap_ms))


def pairs_csv(out: Path, name: str) -> Path:
    return out / f"pairs_{name}.csv"


def map_path(out: Path, name: str, pair: PairTask) -> Path:
    return out / "maps" / name / f"{pair.frame_i.index:05d}_{pair.frame_j.index:05d}.npz"


def map_pairs(pairs: Sequence[PairTask]) -> list[PairTask]:
    return [p for p in pairs if p.gap_ms == MAPS_GAP_MS]


def incomplete_reason(out: Path, name: str, pairs: Sequence[PairTask], save_maps: bool) -> str | None:
    """None if the sequence's CSV (and maps, if requested) are complete for these pairs."""
    path = pairs_csv(out, name)
    if not path.is_file():
        return "no pair CSV"
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if tuple(reader.fieldnames or ()) != PAIR_COLUMNS:
            return f"{path.name} has different columns"
        keys = {(float(r["gap_ms"]), int(r["frame_i"]), int(r["frame_j"])) for r in reader}
    if keys != {p.key for p in pairs}:
        return f"{path.name} holds {len(keys)} pairs, expected {len(pairs)} for these gaps"
    if save_maps:
        missing = [p for p in map_pairs(pairs) if not map_path(out, name, p).is_file()]
        if missing:
            return f"{len(missing)} maps missing"
    return None


def load_window(job: SequenceJob) -> tuple[TUMSequence, list[Frame]]:
    sequence = load_sequence(job.data_root / job.name, job.sequences_config)
    return sequence, apply_window(sequence)


def evaluate_sequence(job: SequenceJob, compute: Callable[..., PairConsistency] = pair_consistency) -> SequenceOutcome:
    started = time.perf_counter()
    set_torch_threads(job.threads)
    sequence, frames = load_window(job)
    pairs = plan_pairs(frames, job.gaps_ms, job.frame_range)
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

    maps_wanted = {p.key for p in map_pairs(pairs)} if job.save_maps else set()
    store = FrameStore(load, ((p.frame_i, p.frame_j) for p in pairs))
    rows: dict[tuple[float, int, int], dict[str, Any]] = {}
    print(f"{job.name}: {len(frames)} frames, {len(pairs)} pairs at {list(job.gaps_ms)} ms (pid {os.getpid()})",
          flush=True)
    for done, pair in enumerate(evaluation_order(pairs), start=1):
        T_j_from_i = relative_pose(pair.frame_i.T_world_cam, pair.frame_j.T_world_cam)
        T_gt_j_from_i = relative_pose(depth_pose(pair.frame_i, *gt_trajectory), depth_pose(pair.frame_j, *gt_trajectory))
        result = compute(store.get(pair.frame_i), store.get(pair.frame_j), sequence.K, T_j_from_i, T_gt_j_from_i,
                         keep_maps=pair.key in maps_wanted)
        store.release(pair.frame_i, pair.frame_j)
        rows[pair.key] = pair_row(job.name, pair, result)
        if pair.key in maps_wanted:
            write_maps(map_path(job.out, job.name, pair), pair, result)
        if done % PROGRESS_EVERY == 0 or done == len(pairs):
            elapsed = time.perf_counter() - started
            print(f"  {job.name}: {done}/{len(pairs)} pairs, {elapsed / done:.2f} s/pair, "
                  f"{len(store.cached)} frames cached", flush=True)
    write_pairs_csv(pairs_csv(job.out, job.name), [rows[p.key] for p in pairs])
    seconds = time.perf_counter() - started
    return SequenceOutcome(
        name=job.name,
        frames=store.loads,
        pairs={f"{gap:g}": sum(p.gap_ms == gap for p in pairs) for gap in job.gaps_ms},
        too_few_pairs=sum(r["too_few_pixels"] for r in rows.values()),
        seconds=seconds,
        frame_seconds=frame_seconds,
        pair_seconds=seconds - frame_seconds,
        peak_mem_mib=peak_memory_mib(),
        pid=os.getpid(),
    )


def pair_row(name: str, pair: PairTask, result: PairConsistency) -> dict[str, Any]:
    row: dict[str, Any] = {
        "sequence": name,
        "gap_ms": f"{pair.gap_ms:g}",
        "frame_i": pair.frame_i.index,
        "frame_j": pair.frame_j.index,
        "t_i": f"{pair.frame_i.timestamp:.6f}",
        "t_j": f"{pair.frame_j.timestamp:.6f}",
        "actual_gap_ms": f"{(pair.frame_j.timestamp - pair.frame_i.timestamp) * 1000.0:.3f}",
        "frac_visible": _cell(result.visible.fraction),
        "frac_interior": _cell(result.interior.fraction),
        "too_few_pixels": result.too_few,
    }
    for metric, attribute in METRICS.items():
        for pixels in PIXEL_SETS:
            summary: ErrorSummary | None = getattr(getattr(result, pixels), attribute)
            values = (math.nan,) * len(STATS) if summary is None else (
                summary.overall.mean, summary.overall.median, summary.overall.p90,
                summary.near.median, summary.mid.median, summary.far.median,
            )
            row.update({f"{metric}_{pixels}_{stat}": _cell(v) for stat, v in zip(STATS, values)})
    return row


def _cell(value: float) -> str:
    return "" if math.isnan(value) else repr(float(value))


def cell_value(cell: str) -> float:
    return float(cell) if cell else math.nan


def write_pairs_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    def write(f) -> None:
        writer = csv.DictWriter(f, fieldnames=PAIR_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    atomic_write(path, write, newline="")


def read_pairs(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if tuple(reader.fieldnames or ()) != PAIR_COLUMNS:
            raise ValueError(f"{path} does not have the pair CSV columns")
        return list(reader)


def write_maps(path: Path, pair: PairTask, result: PairConsistency) -> None:
    if result.maps is None or result.maps.m2 is None:
        raise ValueError(f"pair {pair.frame_i.index} -> {pair.frame_j.index}: no M2 maps to save")
    arrays = {
        "m2": result.maps.m2.astype(np.float16),
        "anchor_depth": result.maps.anchor_depth.astype(np.float16),
        "frame_i": np.int64(pair.frame_i.index),
        "frame_j": np.int64(pair.frame_j.index),
        "t_i": np.float64(pair.frame_i.timestamp),
        "t_j": np.float64(pair.frame_j.timestamp),
    }
    atomic_write(path, lambda f: np.savez_compressed(f, **arrays), mode="wb")


def atomic_write(path: Path, write: Callable[[Any], None], mode: str = "w", **open_kwargs: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    try:
        with open(partial, mode, **open_kwargs) as f:
            write(f)
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)


def init_worker(threads: int) -> None:
    # Must be set before torch is first imported in this process.
    os.environ["OMP_NUM_THREADS"] = str(threads)


def peak_memory_mib() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is bytes on macOS and KiB on Linux.
    return peak / 2**20 if sys.platform == "darwin" else peak / 2**10
