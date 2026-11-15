"""Fused depth of every window frame for rendering, with JOINT-SOLVED scale and ORACLE GT poses.

Each frame with four earlier source frames by time runs exactly the joint_causal_adjacent_w4 pipeline of
fusion_joint.py. The scale and shift of the target and its sources are solved jointly from parallax under the GT poses,
the solved depths are fused with the median aggregator, and holes take the target's own solved depth. The raw and the
fused depth are then each aligned to the target GT by the depth only scorer of fusion_joint.py, as baseline_self is.
Frames with fewer sources keep only the raw depth and have has_fused false.

Writes out/<run name>/<sequence>/<index>.npz per frame, render_prepare.json and regression_status.json. After every
output is written, rel_d_raw must reproduce per_frame.csv within 1e-4 and rel_d_fused the joint_causal_adjacent_w4
rel_d of the joint results within 1e-3 on the targets those cover, or the script exits with code 3. Any other failure
exits with code 1. --preflight runs three of those targets per sequence, checks both values and writes nothing."""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import os
import sys
import time
from collections.abc import Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import fusion_joint  # noqa: E402
from pim.consistency.evaluation import atomic_write, init_worker, peak_memory_mib  # noqa: E402
from pim.evaluation import set_torch_threads  # noqa: E402
from pim.fusion.joint_scale import JointScaleConfig  # noqa: E402
from pim.fusion.oracle import (  # noqa: E402
    BASELINE_SELF_TOLERANCE, SCORE_THREADS, TARGET_COUNT, TARGET_FIRST, Mismatch, RegressionError, default_targets,
    load_window, plan_sources, read_baseline,
)
from pim.fusion.types import FusionConfig  # noqa: E402
from pim.metrics.geometry import depth_affine_metrics  # noqa: E402
from pim.paths import FORK_DIR, REPO_ROOT  # noqa: E402
from pim.runs import available_cpus, git_commit, now  # noqa: E402

SCALE = fusion_joint.SCALE
POSES = fusion_joint.POSES
BASE = "causal_adjacent_w4"
CONFIG = f"joint_{BASE}"
FUSED_TOLERANCE = 1e-3
RECORD_JSON = "render_prepare.json"
STATUS_JSON = "regression_status.json"
PREFLIGHT_TARGETS = 3
CHUNK_TARGETS = 25
EXIT_REGRESSION = 3
PROGRESS_EVERY = 10
DEPTH_DTYPE = np.float16


class Store(Protocol):
    def retain(self, positions: set[int]) -> None: ...
    def raw_z(self, position: int) -> np.ndarray: ...
    def gt(self, position: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...


@dataclass(frozen=True)
class FrameOutput:
    sequence: str
    index: int
    timestamp: float
    raw_aligned: np.ndarray
    fused_aligned: np.ndarray | None
    rel_d_raw: float
    rel_d_fused: float
    low_parallax: bool
    n_sources: int

    @property
    def has_fused(self) -> bool:
        return self.fused_aligned is not None


def fusion_config() -> FusionConfig:
    return fusion_joint.base_config(BASE)


def prepare_frame(name: str, store: Store, frames: Sequence[Any], K: np.ndarray, position: int,
                  fusion: FusionConfig, solver: JointScaleConfig) -> FrameOutput:
    frame = frames[position]
    sources = plan_sources(frames, position, fusion)
    store.retain({position, *sources})
    gt_depth, gt_valid, gt_points = store.gt(position)
    rel_d_raw, _, raw_aligned = depth_affine_metrics(store.raw_z(position), gt_points, gt_valid)
    if len(sources) < fusion.window:
        return FrameOutput(name, frame.index, frame.timestamp, raw_aligned, None, rel_d_raw, math.nan, False,
                           len(sources))
    poses = [frame.T_world_cam, *(frames[p].T_world_cam for p in sources)]
    window = fusion_joint.solve_window(store, K, [position, *sources], poses, solver)
    result, own = fusion_joint.fuse_window(store, frames, K, window, fusion)
    fused = np.where(result.hole, own, result.depth).astype(np.float32)
    rel_d_fused, _, fused_aligned = depth_affine_metrics(fused, gt_points, gt_valid)
    return FrameOutput(name, frame.index, frame.timestamp, raw_aligned, fused_aligned, rel_d_raw, rel_d_fused,
                       bool(window.solved.low_parallax), len(sources))


# Per frame files.

def frame_path(run_out: Path, sequence: str, index: int) -> Path:
    return run_out / sequence / f"{index:05d}.npz"


def write_frame(path: Path, output: FrameOutput) -> None:
    arrays: dict[str, np.ndarray] = {"raw_aligned": output.raw_aligned.astype(DEPTH_DTYPE)}
    if output.fused_aligned is not None:
        arrays["fused_aligned"] = output.fused_aligned.astype(DEPTH_DTYPE)
    arrays.update({
        "rel_d_raw": np.float64(output.rel_d_raw),
        "rel_d_fused": np.float64(output.rel_d_fused),
        "low_parallax": np.bool_(output.low_parallax),
        "n_sources": np.int64(output.n_sources),
        "has_fused": np.bool_(output.has_fused),
        "sequence": np.str_(output.sequence),
        "index": np.int64(output.index),
        "timestamp": np.float64(output.timestamp),
        "scale": np.str_(SCALE),
        "poses": np.str_(POSES),
    })
    atomic_write(path, lambda f: np.savez(f, **arrays), mode="wb")


def read_scores(path: Path) -> tuple[float, float, bool]:
    with np.load(path) as data:
        return float(data["rel_d_raw"]), float(data["rel_d_fused"]), bool(data["has_fused"])


def write_outputs(name: str, store: Store, frames: Sequence[Any], K: np.ndarray, positions: Iterable[int],
                  run_out: Path | None, fusion: FusionConfig, solver: JointScaleConfig,
                  progress: str = "") -> tuple[list[tuple[int, float, float]], int, int]:
    """(index, rel_d_raw, rel_d_fused) of every frame computed, and the counts written and already present. With
    run_out None nothing is written and nothing is skipped."""
    values, written, existing = [], 0, 0
    positions = list(positions)
    started = time.perf_counter()
    for done, position in enumerate(positions, start=1):
        path = None if run_out is None else frame_path(run_out, name, frames[position].index)
        if path is not None and path.is_file():
            existing += 1
            continue
        output = prepare_frame(name, store, frames, K, position, fusion, solver)
        if path is not None:
            write_frame(path, output)
            written += 1
        values.append((output.index, output.rel_d_raw, output.rel_d_fused))
        if progress and (done % PROGRESS_EVERY == 0 or done == len(positions)):
            print(f"  {progress} {done}/{len(positions)}, {(time.perf_counter() - started) / done:.1f} s per frame, "
                  f"peak {peak_memory_mib():.0f} MiB", flush=True)
    return values, written, existing


# Jobs.

@dataclass(frozen=True)
class RenderJob:
    name: str
    data_root: Path
    sequences_config: Path
    run_dir: Path
    cache_dir: Path
    run_out: Path | None
    """None in the preflight, which writes nothing."""
    targets: tuple[int, ...]
    threads: int = 1
    score_threads: int = SCORE_THREADS
    solver: JointScaleConfig = JointScaleConfig()


@dataclass(frozen=True)
class RenderOutcome:
    name: str
    targets: int
    written: int
    existing: int
    seconds: float
    peak_mem_mib: float
    pid: int
    values: tuple[tuple[int, float, float], ...]


def process_job(job: RenderJob) -> RenderOutcome:
    started = time.perf_counter()
    set_torch_threads(job.score_threads)
    sequence, frames = load_window(job.name, job.data_root, job.sequences_config)
    store = fusion_joint.JointStore(job.run_dir, job.cache_dir, job.name, sequence.root, sequence.K, frames)
    position = {f.index: k for k, f in enumerate(frames)}
    label = f"{job.name} frames {job.targets[0]} to {job.targets[-1]} (pid {os.getpid()})"
    values, written, existing = write_outputs(job.name, store, frames, sequence.K, [position[t] for t in job.targets],
                                              job.run_out, fusion_config(), job.solver, label)
    return RenderOutcome(job.name, len(job.targets), written, existing, time.perf_counter() - started,
                         peak_memory_mib(), os.getpid(), tuple(values))


def run_jobs(jobs: list[RenderJob], workers: int) -> list[RenderOutcome]:
    if workers == 1 or len(jobs) <= 1:
        return [process_job(job) for job in jobs]
    executor = ProcessPoolExecutor(
        max_workers=min(workers, len(jobs)), mp_context=multiprocessing.get_context("spawn"),
        initializer=init_worker, initargs=(jobs[0].threads,), max_tasks_per_child=1,
    )
    outcomes = []
    try:
        futures = [executor.submit(process_job, job) for job in jobs]
        for future in as_completed(futures):
            outcome = future.result()
            outcomes.append(outcome)
            print(f"{outcome.name} chunk of {outcome.targets} done in {outcome.seconds / 60:.1f} min, peak "
                  f"{outcome.peak_mem_mib:.0f} MiB, {len(outcomes)}/{len(jobs)} chunks", flush=True)
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    return outcomes


def chunks(indices: Sequence[int], size: int) -> list[tuple[int, ...]]:
    return [tuple(indices[k:k + size]) for k in range(0, len(indices), size)]


# Regression checks.

class RenderRegressionMismatch(RegressionError):
    """rel_d_raw or rel_d_fused differs from its reference after every output was written."""


def read_joint(path: Path, names: Sequence[str]) -> dict[tuple[str, int], float]:
    """rel_d of CONFIG by (sequence, target index), from one joint_fusion CSV or the folder holding them."""
    files = [path] if path.is_file() else [fusion_joint.fusion_csv(path, n) for n in names]
    values = {}
    for file in files:
        if not file.is_file():
            raise FileNotFoundError(f"joint results {file} not found")
        for row in fusion_joint.read_rows(file):
            if row["config"] != CONFIG:
                continue
            if row["scale"] != SCALE or row["poses"] != POSES:
                raise ValueError(f"{file} has scale {row['scale']} and poses {row['poses']}, expected {SCALE} and "
                                 f"{POSES}")
            values[(row["sequence"], int(row["target_index"]))] = float(row["rel_d"]) if row["rel_d"] else math.nan
    return values


def regression(scores: dict[tuple[str, int], tuple[float, float]], baseline: dict[tuple[str, int], float],
               joint: dict[tuple[str, int], float]) -> dict[str, Any]:
    """Every checked target's rel_d_raw against per_frame.csv and rel_d_fused against the joint results."""
    raw = [Mismatch(n, i, r, baseline.get((n, i), math.nan)) for (n, i), (r, _) in scores.items()]
    fused = [Mismatch(n, i, f, joint.get((n, i), math.nan)) for (n, i), (_, f) in scores.items()]
    report = {
        "rel_d_raw": fusion_joint.comparison_report(
            "rel_d_raw against per_frame.csv", sorted(raw, key=lambda m: m.diff, reverse=True),
            BASELINE_SELF_TOLERANCE),
        "rel_d_fused": fusion_joint.comparison_report(
            f"rel_d_fused against {CONFIG} of the joint results", sorted(fused, key=lambda m: m.diff, reverse=True),
            FUSED_TOLERANCE),
    }
    report["passed"] = report["rel_d_raw"]["passed"] and report["rel_d_fused"]["passed"]
    return report


def scores_from_files(run_out: Path, checked: dict[str, Sequence[int]]) -> dict[tuple[str, int], tuple[float, float]]:
    scores = {}
    for name, indices in checked.items():
        for index in indices:
            path = frame_path(run_out, name, index)
            scores[(name, index)] = read_scores(path)[:2] if path.is_file() else (math.nan, math.nan)
    return scores


def finish(run_out: Path, checked: dict[str, Sequence[int]], baseline: dict[tuple[str, int], float],
           joint: dict[tuple[str, int], float], record: dict[str, Any]) -> dict[str, Any]:
    """Runs the regression check on the written files, writes the record and the status, and raises on a mismatch."""
    report = regression(scores_from_files(run_out, checked), baseline, joint)
    record = {**record, "regression": report, "end_time": now()}
    atomic_write(run_out / RECORD_JSON, lambda f: f.write(json.dumps(record, indent=2) + "\n"))
    status = {"status": "pass" if report["passed"] else "fail", "scale": SCALE, "poses": POSES, "config": CONFIG,
              "rel_d_raw": report["rel_d_raw"], "rel_d_fused": report["rel_d_fused"]}
    atomic_write(run_out / STATUS_JSON, lambda f: f.write(json.dumps(status, indent=2) + "\n"))
    print(f"\nwrote {run_out}")
    if not report["passed"]:
        raise RenderRegressionMismatch(f"regression check FAILED (rel_d_raw passed {report['rel_d_raw']['passed']}, "
                                       f"rel_d_fused passed {report['rel_d_fused']['passed']}); every output is "
                                       f"already written to {run_out}")
    print("regression passed. rel_d_raw matches per_frame.csv and rel_d_fused matches the joint results")
    return record


def inventory(run_out: Path, windows: dict[str, list[int]]) -> dict[str, Any]:
    by_sequence = {}
    for name, indices in windows.items():
        paths = [frame_path(run_out, name, i) for i in indices]
        present = [p for p in paths if p.is_file()]
        fused = sum(read_scores(p)[2] for p in present)
        by_sequence[name] = {"frames": len(indices), "written": len(present), "fused": fused,
                             "raw_only": len(present) - fused,
                             "megabytes": round(sum(p.stat().st_size for p in present) / 1e6, 1)}
    return by_sequence


# Command line.

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True, help="run directory holding run.json")
    parser.add_argument("--cache", type=Path, required=True,
                        help="GT-aligned depth cache of fusion_prepare.py, read only for its provenance and windows")
    parser.add_argument("--data-root", type=Path, required=True, help="folder containing the sequence folders")
    parser.add_argument("--sequences-config", type=Path, default=Path("configs/sequences.yaml"))
    parser.add_argument("--baseline-csv", type=Path, required=True, help="per_frame.csv of the baseline evaluation")
    parser.add_argument("--joint-csv", type=Path, required=True,
                        help="joint_fusion_<sequence>.csv of fusion_joint.py, or the folder holding them")
    parser.add_argument("--sequences", nargs="+", help="default is every sequence in the cache")
    parser.add_argument("--out", type=Path, required=True, help="frames go to out/<run name>/<sequence>/")
    parser.add_argument("--workers", type=int, default=4, help="worker processes, each taking chunks of targets")
    parser.add_argument("--threads", type=int, default=1, help="OpenMP threads per worker")
    parser.add_argument("--score-threads", type=int, default=SCORE_THREADS,
                        help="torch threads for the float32 score reduction; must match the baseline evaluation run")
    parser.add_argument("--preflight", action="store_true",
                        help=f"run {PREFLIGHT_TARGETS} checked targets per sequence, check both values and write "
                             "nothing")
    return parser.parse_args(argv)


def setup(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], list[str], dict[str, list[int]],
                                             dict[str, list[int]], dict[tuple[str, int], float],
                                             dict[tuple[str, int], float]]:
    """(prepare record, run, sequences, window indices, checked targets, per_frame.csv, joint results)."""
    fusion_joint.validate(args)
    prepared, run, names = fusion_joint.read_inputs(args)
    windows, checked = {}, {}
    for name in names:
        _, frames = load_window(name, args.data_root, args.sequences_config)
        window = {"start": frames[0].index, "count": len(frames)}
        if prepared["sequences"][name]["window"] != window:
            raise ValueError(f"{name} cache covers window {prepared['sequences'][name]['window']}, but "
                             f"{args.sequences_config} gives {window}")
        windows[name] = [f.index for f in frames]
        checked[name] = default_targets(frames)
    return (prepared, run, names, windows, checked, read_baseline(args.baseline_csv),
            read_joint(args.joint_csv, names))


def job(args: argparse.Namespace, name: str, run_out: Path | None, targets: Sequence[int]) -> RenderJob:
    return RenderJob(name, args.data_root, args.sequences_config, args.predictions, args.cache, run_out,
                     tuple(targets), args.threads, args.score_threads)


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    _, _, names, _, checked, baseline, joint = setup(args)
    set_torch_threads(args.score_threads)
    outcomes = run_jobs([job(args, n, None, checked[n][:PREFLIGHT_TARGETS]) for n in names], args.workers)
    scores = {(o.name, i): (r, f) for o in outcomes for i, r, f in o.values}
    for o in outcomes:
        print(f"{o.name} {o.targets} targets in {o.seconds:.0f} s, peak {o.peak_mem_mib:.0f} MiB", flush=True)
    report = regression(scores, baseline, joint)
    if not report["passed"]:
        raise RenderRegressionMismatch("preflight FAILED. rel_d_raw passed "
                                       f"{report['rel_d_raw']['passed']}, rel_d_fused passed "
                                       f"{report['rel_d_fused']['passed']}")
    print(f"preflight passed on {len(scores)} targets")
    return report


def run_render(args: argparse.Namespace) -> dict[str, Any]:
    start_time, started = now(), time.perf_counter()
    prepared, run, names, windows, checked, baseline, joint = setup(args)
    set_torch_threads(args.score_threads)
    run_out = args.out / run["name"]
    jobs = []
    for name in names:
        todo = [i for i in windows[name] if not frame_path(run_out, name, i).is_file()]
        print(f"{name} {len(windows[name])} frames, {len(windows[name]) - len(todo)} already written", flush=True)
        jobs.extend(job(args, name, run_out, c) for c in chunks(todo, CHUNK_TARGETS))
    outcomes = run_jobs(jobs, args.workers)
    peak = {n: max((o.peak_mem_mib for o in outcomes if o.name == n), default=None) for n in names}
    seconds = {n: round(sum(o.seconds for o in outcomes if o.name == n), 1) for n in names}
    record = {
        "scale": SCALE,
        "poses": POSES,
        "label": f"scale = {SCALE}, poses = {POSES}",
        "config": CONFIG,
        "fusion": fusion_joint.SPECS[[s.name for s in fusion_joint.SPECS].index(CONFIG)].describe(),
        "solver": asdict(JointScaleConfig()),
        "protocol": {
            "pipeline": f"{CONFIG} of fusion_joint.py, run on every window frame with {fusion_config().window} earlier "
                        "sources by time; other frames keep the raw depth only and have has_fused false",
            "alignment": "raw_aligned and fused_aligned are the aligned depth of fusion_joint.score_depth, upstream "
                         "compute_metrics' depth affine branch against the target GT, as baseline_self is scored",
            "storage": f"depth arrays in {np.dtype(DEPTH_DTYPE).name}, scalars in float64, int64, bool and str",
            "regression": f"on window positions {TARGET_FIRST} to {TARGET_FIRST + TARGET_COUNT - 1}, rel_d_raw "
                          f"equals per_frame.csv within {BASELINE_SELF_TOLERANCE:g} and rel_d_fused equals {CONFIG} "
                          f"of the joint results within {FUSED_TOLERANCE:g}",
        },
        "predictions": {"dir": str(args.predictions), "run": run},
        "cache": {"dir": str(args.cache), "prepare": prepared, "use": "provenance and window check only"},
        "baseline_csv": str(args.baseline_csv),
        "joint_csv": str(args.joint_csv),
        "commits": {"project": git_commit(REPO_ROOT), "pointdit": git_commit(FORK_DIR)},
        "arguments": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "sequences": {n: {**inventory(run_out, {n: windows[n]})[n], "worker_seconds": seconds[n],
                          "peak_mem_mib": peak[n]} for n in names},
        "checked_targets": checked,
        "cpus": available_cpus(),
        "start_time": start_time,
        "seconds": round(time.perf_counter() - started, 1),
    }
    return finish(run_out, checked, baseline, joint, record)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    try:
        if args.preflight:
            run_preflight(args)
        else:
            run_render(args)
    except RenderRegressionMismatch as error:
        banner = "=" * 30 + " FAILED " + "=" * 30
        print(f"\n{banner}\n{error}\n{banner}", flush=True)
        sys.exit(EXIT_REGRESSION)


if __name__ == "__main__":
    main()
