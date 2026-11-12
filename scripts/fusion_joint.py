"""JOINT-SOLVED-SCALE fusion with ORACLE GT poses. For every target, the per frame depth scale and shift of the target
and its selected sources are solved jointly from parallax under the metric GT poses, with no GT depth. The solved
depths are fused with the median aggregator, and only the fused output is aligned to the target's GT, as baseline_self
is, before it is scored. The GT-aligned depth cache of fusion_prepare.py is read for a logging-only comparison of the
solved and the oracle affines and never reaches the solver or the fusion.

Writes joint_fusion_<sequence>.csv, joint_fusion_summary.csv, joint_fusion.json and regression_status.json. After
every output is written, baseline_self must reproduce per_frame.csv and the no-fusion control must reproduce
baseline_self within 1e-4, or the script exits with code 3; any other failure exits with code 1. --preflight scores
baseline_self on the first targets of each sequence, runs the solver on one target per sequence and writes nothing."""

from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pim.consistency.evaluation import atomic_write, init_worker, peak_memory_mib  # noqa: E402
from pim.data.tum import Frame  # noqa: E402
from pim.evaluation import load_prediction, prediction_path, set_torch_threads  # noqa: E402
from pim.fusion.engine import fuse_frame  # noqa: E402
from pim.fusion.joint_scale import (  # noqa: E402
    RIDGE_SENSITIVITY, JointScaleConfig, JointScaleResult, fit_affine, median_pixel_depth_error, ridge_sensitivity,
    solve_joint_scale,
)
from pim.fusion.oracle import (  # noqa: E402
    BASELINE_SELF_TOLERANCE, PREPARE_RECORD, REGRESSION_TOLERANCE, SCORE_THREADS, SEED, SPEC_GROUPS, TARGET_COUNT,
    TARGET_FIRST, WORST_SHOWN, Mismatch, RegressionError, baseline_self_comparison, cache_path, check_targets,
    default_targets, load_window, plan_sources, read_alignment, read_baseline, shuffled_positions, source_valid,
)
from pim.fusion.oracle import SCALE as ORACLE_SCALE  # noqa: E402
from pim.fusion.types import FusionConfig, FusionResult, SourceView, TargetSpec  # noqa: E402
from pim.metrics.error_maps import rel_d_by_depth_tertile, relative_depth_error  # noqa: E402
from pim.metrics.geometry import backproject, depth_affine_metrics, evaluate_frame  # noqa: E402
from pim.paths import FORK_DIR, REPO_ROOT  # noqa: E402
from pim.runs import available_cpus, git_commit, now, read_run  # noqa: E402

SCALE = "joint_solved"
POSES = "oracle_gt"
BASES = ("causal_adjacent_w4", "causal_diverse_w4", "centred_adjacent_w4")
CONTROL_BASE = "causal_adjacent_w4"
ORACLE_PREFIX = "rungb_"
CSV_PREFIX = "joint_fusion_"
SUMMARY_CSV = "joint_fusion_summary.csv"
RECORD_JSON = "joint_fusion.json"
STATUS_JSON = "regression_status.json"
PREFLIGHT_TARGETS = 3
EXIT_REGRESSION = 3
PROGRESS_EVERY = 10
DIAGNOSTIC_COLUMNS = (
    "fused_affine", "low_parallax", "baseline_ratio", "rotation_ratio", "solver_cost", "solver_iterations",
    "visible_fraction", "target_scale", "target_shift", "oracle_scale", "oracle_shift", "target_depth_error",
    "window_depth_error", "ridge_low_change", "ridge_high_change",
)
ROW_COLUMNS = (
    "sequence", "target_index", "config", "scale", "poses", "rel_d", "delta1_d", "rel_d_near", "rel_d_mid",
    "rel_d_far", "hole_fraction", "median_n_sources", *DIAGNOSTIC_COLUMNS, "n_selected", "sources",
)
SUMMARY_COLUMNS = (
    "sequence", "config", "scale", "poses", "targets", "rel_d_median", "delta_vs_baseline_median",
    "rel_change_pct_median", "share_improved", "delta1_d_median", "rel_d_near_median", "rel_d_mid_median",
    "rel_d_far_median", "hole_fraction_median", "low_parallax_targets", "target_depth_error_abs_median",
    "target_depth_error_abs_median_parallax", "target_depth_error_abs_median_low", "scale_correlation",
    "scale_correlation_parallax", "scale_correlation_low", "window_depth_error_median",
    "ridge_low_change_abs_median", "ridge_high_change_abs_median", "oracle_config", "oracle_rel_d_median",
    "oracle_rel_change_pct_median",
)
PROTOCOL = {
    "scale": f"{SCALE}. Each frame's raw z gets its own scale and shift, solved jointly over the target and its "
             "sources from parallax under the GT poses; GT depth never enters the solver or the fusion",
    "poses": f"{POSES}. RGB-timestamp metric GT poses, used as an ORACLE by the solver and the fusion",
    "solver": "pim.fusion.joint_scale. About 1500 pixels per frame on a regular grid over finite raw z inside a 10% "
              "margin; every ordered pair; residual (z_in_j - depth_j) / (s_j * IQR of raw_z_j), the relative "
              "residual rescaled by depth_j / (s_j * IQR); soft L1 truncated at the visibility limit; ridge on (log "
              "s_k, t_k / median depth) toward the window mean; 15 by 15 shared grid, then Levenberg Marquardt with "
              "a numerical Jacobian at visibility limits x4, x2, x1. Low parallax windows keep the shared grid "
              "affine for every frame",
    "fusion": "the engine with the median aggregator and no front layer or edge options (rungb_<base> of the oracle "
              "edge group), with the target's own solved depth as the self layer; holes take that depth",
    "scoring": "the fused depth alone is aligned to the target GT by upstream compute_metrics' depth affine branch "
               "(the code evaluate_frame runs for rel_d), then rel_d and delta1_d x100 over GT-valid pixels and "
               "rel_d near, mid and far over GT depth thirds. baseline_self is the raw prediction scored the same "
               "way and must reproduce per_frame.csv",
    "oracle_diagnostic": "logging only. The oracle affine is the least squares fit of the cached GT-aligned depth "
                         "on raw z. target_depth_error is the signed relative error of the solved depth at the "
                         "target's median raw z pixel, window_depth_error the median of its absolute value over the "
                         "window frames; scale_correlation is Pearson over targets of solved and oracle target scale",
    "oracle_comparison": "oracle_rel_d_median and oracle_rel_change_pct_median are read from the oracle edge "
                         "summary for rungb_<base>; those scores fused GT-aligned sources and were NOT realigned "
                         "after fusion, so they are an upper bound under a different protocol",
    "controls": f"nofusion_joint_{CONTROL_BASE} scores only the target's own depth in its solved affine and must "
                f"equal baseline_self; shuffled_poses_joint_{CONTROL_BASE} gives each source another random window "
                "frame's pose in both the solver and the fusion",
    "ridge_sensitivity": f"ridge_low_change and ridge_high_change are the relative change of the target depth at its "
                         f"median raw z pixel when the solution is refined with the ridge weight times "
                         f"{RIDGE_SENSITIVITY[0]:g} and {RIDGE_SENSITIVITY[1]:g}; the weight itself is fixed",
    "regression": f"baseline_self equals per_frame.csv and nofusion equals baseline_self within "
                  f"{BASELINE_SELF_TOLERANCE:g} (absolute, percent) for every target",
}


@dataclass(frozen=True)
class JointSpec:
    name: str
    kind: str
    """baseline, joint, nofusion or shuffled."""
    base: str | None = None

    @property
    def fusion(self) -> FusionConfig:
        return base_config(self.base) if self.base else FusionConfig(window=0)

    def describe(self) -> dict[str, Any]:
        fusion = {k: v for k, v in vars(self.fusion).items() if k != "keep_layers" and v is not None}
        return {"kind": self.kind, "base": self.base, "oracle_config": oracle_config(self), "fusion": fusion}


def base_config(base: str) -> FusionConfig:
    return SPEC_GROUPS["edge"][f"{ORACLE_PREFIX}{base}"].fusion


def build_specs() -> list[JointSpec]:
    return [JointSpec("baseline_self", "baseline"), *(JointSpec(f"joint_{b}", "joint", b) for b in BASES),
            JointSpec(f"nofusion_joint_{CONTROL_BASE}", "nofusion", CONTROL_BASE),
            JointSpec(f"shuffled_poses_joint_{CONTROL_BASE}", "shuffled", CONTROL_BASE)]


SPECS = build_specs()


def oracle_config(spec: JointSpec) -> str:
    if spec.kind == "baseline":
        return "baseline_self"
    return f"{ORACLE_PREFIX}{spec.base}" if spec.kind == "joint" else ""


# Scoring.

def score_depth(depth: np.ndarray, gt_depth: np.ndarray, gt_valid: np.ndarray, gt_points: np.ndarray
                ) -> dict[str, float]:
    """rel_d, delta1_d and GT depth thirds after upstream's depth affine alignment of depth alone to the GT."""
    rel_d, delta1_d, aligned = depth_affine_metrics(depth, gt_points, gt_valid)
    if not math.isfinite(rel_d):
        return {"rel_d": rel_d, "delta1_d": delta1_d, "rel_d_near": math.nan, "rel_d_mid": math.nan,
                "rel_d_far": math.nan}
    error = relative_depth_error(aligned, gt_depth, gt_valid)
    return {"rel_d": rel_d, "delta1_d": delta1_d, **rel_d_by_depth_tertile(error, gt_depth, gt_valid)}


# Frames of the current target.

class JointStore:
    """Raw z, oracle affine and GT of the frames the current target needs, dropped once a later target does not."""

    def __init__(self, run_dir: Path, cache_dir: Path, name: str, root: Path, K: np.ndarray,
                 frames: list[Frame]) -> None:
        self._run_dir, self._cache_dir, self._name, self._root, self._K = run_dir, cache_dir, name, root, K
        self._frames = frames
        self._raw: dict[int, np.ndarray] = {}
        self._oracle: dict[int, tuple[float, float]] = {}
        self._gt: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def retain(self, positions: set[int]) -> None:
        for store in (self._raw, self._oracle, self._gt):
            for position in [p for p in store if p not in positions]:
                del store[position]

    @property
    def held(self) -> int:
        return len(set(self._raw) | set(self._gt))

    def points(self, position: int) -> np.ndarray:
        frame = self._frames[position]
        return load_prediction(prediction_path(self._run_dir, self._name, frame.index), frame, self._root).points

    def raw_z(self, position: int) -> np.ndarray:
        if position not in self._raw:
            self._raw[position] = np.ascontiguousarray(self.points(position)[..., 2], dtype=np.float32)
        return self._raw[position]

    def oracle_affine(self, position: int) -> tuple[float, float]:
        """Logging only. Least squares fit of the cached GT-aligned depth on raw z."""
        if position not in self._oracle:
            frame = self._frames[position]
            aligned = read_alignment(cache_path(self._cache_dir, self._name, frame.index), frame.index).aligned_depth
            self._oracle[position] = fit_affine(self.raw_z(position), aligned)
        return self._oracle[position]

    def gt(self, position: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(GT depth, GT-valid mask, GT points) as evaluate.py builds them."""
        if position not in self._gt:
            depth, depth_valid = self._frames[position].load_depth()
            gt_points, gt_valid = backproject(depth, depth_valid, self._K)
            self._gt[position] = depth, gt_valid, gt_points
        return self._gt[position]


@dataclass(frozen=True)
class Window:
    positions: tuple[int, ...]
    """Target first, then its sources."""
    poses: tuple[np.ndarray, ...]
    solved: JointScaleResult


def solve_window(store: JointStore, K: np.ndarray, positions: list[int], poses: list[np.ndarray],
                 config: JointScaleConfig) -> Window:
    raw = [store.raw_z(p) for p in positions]
    return Window(tuple(positions), tuple(poses), solve_joint_scale(raw, poses, K, config))


def fuse_window(store: JointStore, frames: list[Frame], K: np.ndarray, window: Window, fusion: FusionConfig
                ) -> tuple[FusionResult, np.ndarray]:
    """Fused depth of the target from every window frame in its solved affine, and the target's own solved depth."""
    views = []
    for k, (position, pose) in enumerate(zip(window.positions, window.poses)):
        depth = window.solved.depth(k, store.raw_z(position))
        views.append(SourceView(depth, source_valid(depth), pose, K, frames[position].timestamp))
    own, sources = views[0], views[1:]
    target = frames[window.positions[0]]
    result = fuse_frame(sources, TargetSpec(own.T_world_cam, K, own.depth.shape), target.timestamp, fusion, own)
    if result.source_indices != tuple(range(len(sources))):
        raise RuntimeError(f"target {target.index} the engine chose sources {result.source_indices} of the planned "
                           f"{[frames[p].index for p in window.positions[1:]]}")
    return result, own.depth


def diagnostics(store: JointStore, window: Window, solver: JointScaleConfig, K: np.ndarray, sensitivity: bool
                ) -> dict[str, Any]:
    solved = window.solved
    raw = [store.raw_z(p) for p in window.positions]
    oracle = [store.oracle_affine(p) for p in window.positions]
    errors = [median_pixel_depth_error(solved.scale[k], solved.shift[k], *oracle[k], raw[k]) for k in range(len(raw))]
    changes = {"ridge_low_change": math.nan, "ridge_high_change": math.nan}
    if sensitivity and solved.per_frame:
        alternatives = ridge_sensitivity(raw, list(window.poses), K, solved, solver)
        for column, factor in zip(changes, RIDGE_SENSITIVITY):
            alt = alternatives[factor]
            changes[column] = median_pixel_depth_error(alt.scale[0], alt.shift[0], solved.scale[0], solved.shift[0],
                                                       raw[0])
    return {
        "fused_affine": "per_frame" if solved.per_frame else "shared",
        "low_parallax": solved.low_parallax,
        "baseline_ratio": solved.baseline_ratio,
        "rotation_ratio": solved.rotation_ratio,
        "solver_cost": solved.cost,
        "solver_iterations": solved.iterations,
        "visible_fraction": solved.valid_fraction,
        "target_scale": float(solved.scale[0]),
        "target_shift": float(solved.shift[0]),
        "oracle_scale": oracle[0][0],
        "oracle_shift": oracle[0][1],
        "target_depth_error": errors[0],
        "window_depth_error": float(np.median(np.abs(errors))),
        **changes,
    }


@dataclass
class SequenceContext:
    name: str
    K: np.ndarray
    frames: list[Frame]
    seed: int
    solver: JointScaleConfig
    sensitivity: bool = True


def evaluate_target(ctx: SequenceContext, store: JointStore, specs: list[JointSpec], position: int
                    ) -> list[dict[str, Any]]:
    frames, K = ctx.frames, ctx.K
    target = frames[position]
    plans = {s.name: plan_sources(frames, position, s.fusion) for s in specs}
    store.retain({position, *(p for plan in plans.values() for p in plan)})
    gt_depth, gt_valid, gt_points = store.gt(position)
    windows: dict[tuple[str, str], Window] = {}
    rows = []
    for spec in specs:
        sources = plans[spec.name]
        extra: dict[str, Any] = dict.fromkeys(DIAGNOSTIC_COLUMNS, "")
        fill = {"hole_fraction": 0.0, "median_n_sources": 1.0}
        if spec.kind == "baseline":
            scores = score_depth(store.raw_z(position), gt_depth, gt_valid, gt_points)
        else:
            pose_kind = "shuffled" if spec.kind == "shuffled" else "true"
            key = (spec.base, pose_kind)
            if key not in windows:
                picks = (shuffled_positions(ctx.seed, target.index, sources, len(frames)) if pose_kind == "shuffled"
                         else sources)
                poses = [target.T_world_cam, *(frames[q].T_world_cam for q in picks)]
                windows[key] = solve_window(store, K, [position, *sources], poses, ctx.solver)
            window = windows[key]
            extra = diagnostics(store, window, ctx.solver, K, ctx.sensitivity and spec.kind == "joint")
            if spec.kind == "nofusion":
                scores = score_depth(window.solved.depth(0, store.raw_z(position)), gt_depth, gt_valid, gt_points)
            else:
                result, own = fuse_window(store, frames, K, window, spec.fusion)
                fused = np.where(result.hole, own, result.depth).astype(np.float32)
                scores = score_depth(fused, gt_depth, gt_valid, gt_points)
                fill = {"hole_fraction": float(result.hole[gt_valid].mean()),
                        "median_n_sources": float(np.median(result.n_sources[gt_valid]))}
        shown = [] if spec.kind == "nofusion" else sources
        rows.append({
            "sequence": ctx.name, "target_index": target.index, "config": spec.name, "scale": SCALE, "poses": POSES,
            **{k: _cell(v) for k, v in scores.items()}, **{k: _cell(v) for k, v in fill.items()},
            **{k: _cell(v) for k, v in extra.items()},
            "n_selected": len(shown), "sources": " ".join(str(frames[p].index) for p in shown),
        })
    return rows


# Per-sequence jobs.

@dataclass(frozen=True)
class JointJob:
    name: str
    data_root: Path
    sequences_config: Path
    run_dir: Path
    cache_dir: Path
    out: Path
    targets: tuple[int, ...]
    baseline: dict[int, float]
    seed: int = SEED
    threads: int = 1
    score_threads: int = SCORE_THREADS
    solver: JointScaleConfig = JointScaleConfig()
    sensitivity: bool = True


@dataclass(frozen=True)
class JointOutcome:
    name: str
    targets: int
    seconds: float
    peak_mem_mib: float
    pid: int


def fusion_csv(out: Path, name: str) -> Path:
    return out / f"{CSV_PREFIX}{name}.csv"


def make_context(job: JointJob) -> tuple[SequenceContext, JointStore, dict[int, int]]:
    sequence, frames = load_window(job.name, job.data_root, job.sequences_config)
    check_targets(job.name, frames, job.targets)
    ctx = SequenceContext(job.name, sequence.K, frames, job.seed, job.solver, job.sensitivity)
    store = JointStore(job.run_dir, job.cache_dir, job.name, sequence.root, sequence.K, frames)
    return ctx, store, {f.index: k for k, f in enumerate(frames)}


def evaluate_sequence(job: JointJob) -> JointOutcome:
    started = time.perf_counter()
    set_torch_threads(job.score_threads)
    ctx, store, position = make_context(job)
    rows: list[dict[str, Any]] = []
    print(f"{job.name} {len(job.targets)} targets x {len(SPECS)} configs, scale {SCALE}, poses {POSES} "
          f"(pid {os.getpid()})", flush=True)
    for done, target_index in enumerate(job.targets, start=1):
        target_rows = evaluate_target(ctx, store, SPECS, position[target_index])
        for line in baseline_failures(target_rows, {(job.name, k): v for k, v in job.baseline.items()}):
            print(f"  WARNING {line}", flush=True)
        rows.extend(target_rows)
        if done % PROGRESS_EVERY == 0 or done == len(job.targets):
            elapsed = time.perf_counter() - started
            print(f"  {job.name} {done}/{len(job.targets)} targets, {elapsed / done:.1f} s per target, "
                  f"{store.held} frames held, peak {peak_memory_mib():.0f} MiB", flush=True)
    write_rows(fusion_csv(job.out, job.name), rows)
    return JointOutcome(job.name, len(job.targets), time.perf_counter() - started, peak_memory_mib(), os.getpid())


def run_jobs(jobs: list[JointJob], workers: int) -> dict[str, JointOutcome]:
    if workers == 1 or len(jobs) <= 1:
        return {job.name: evaluate_sequence(job) for job in jobs}
    executor = ProcessPoolExecutor(
        max_workers=min(workers, len(jobs)), mp_context=multiprocessing.get_context("spawn"),
        initializer=init_worker, initargs=(jobs[0].threads,), max_tasks_per_child=1,
    )
    outcomes = {}
    try:
        futures = {executor.submit(evaluate_sequence, job): job.name for job in jobs}
        for future in as_completed(futures):
            outcome = future.result()
            outcomes[outcome.name] = outcome
            print(f"{outcome.name} done in {outcome.seconds / 60:.1f} min, peak {outcome.peak_mem_mib:.0f} MiB",
                  flush=True)
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    return outcomes


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    def write(f) -> None:
        writer = csv.DictWriter(f, fieldnames=ROW_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    atomic_write(path, write, newline="")


def read_rows(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if tuple(reader.fieldnames or ()) != ROW_COLUMNS:
            raise ValueError(f"{path} does not have the joint fusion columns")
        return list(reader)


def complete(path: Path, name: str, targets: tuple[int, ...]) -> bool:
    if not path.is_file():
        return False
    try:
        rows = read_rows(path)
    except ValueError:
        return False
    return {(r["sequence"], int(r["target_index"]), r["config"]) for r in rows} == {
        (name, t, s.name) for t in targets for s in SPECS}


# Regression checks.

def baseline_failures(rows: list[dict[str, Any]], baseline: dict[tuple[str, int], float],
                      tolerance: float = BASELINE_SELF_TOLERANCE) -> list[str]:
    return [m.describe(tolerance) for m in baseline_self_comparison(rows, baseline) if m.diff > tolerance]


def nofusion_comparison(rows: list[dict[str, str]]) -> list[Mismatch]:
    """Every nofusion row against the baseline_self row of the same target, worst first."""
    spec = next(s for s in SPECS if s.kind == "nofusion")
    by_key = {(r["sequence"], int(r["target_index"]), r["config"]): _value(r["rel_d"]) for r in rows}
    compared = [Mismatch(seq, t, value, by_key.get((seq, t, "baseline_self"), math.nan))
                for (seq, t, config), value in by_key.items() if config == spec.name]
    return sorted(compared, key=lambda m: m.diff, reverse=True)


def comparison_report(title: str, comparison: list[Mismatch], tolerance: float = BASELINE_SELF_TOLERANCE
                      ) -> dict[str, Any]:
    over = [m for m in comparison if m.diff > tolerance]
    max_diff = max((m.diff for m in comparison), default=0.0)
    print(f"\n{title}. {len(over)} of {len(comparison)} targets over {tolerance:g}, max |diff| {max_diff:.3g}")
    for m in comparison[:WORST_SHOWN]:
        print(f"  {'FAIL' if m.diff > tolerance else 'ok  '} {m.sequence} frame {m.frame_index} rel_d {m.actual!r} "
              f"against {m.expected!r} (|diff| {m.diff:.3g})")
    return {"rows": len(comparison), "tolerance": tolerance, "over_tolerance": len(over),
            "max_abs_diff": _json_float(max_diff),
            "worst": [{"sequence": m.sequence, "frame_index": m.frame_index, "rel_d": _json_float(m.actual),
                       "expected": _json_float(m.expected), "abs_diff": _json_float(m.diff)}
                      for m in comparison[:WORST_SHOWN]],
            "passed": bool(comparison) and not over}


def control_report(summary: list[dict[str, Any]]) -> dict[str, Any]:
    """Shuffled poses must be clearly worse than true poses, in fused rel_d and in affine recovery."""
    joint, shuffled = f"joint_{CONTROL_BASE}", f"shuffled_poses_joint_{CONTROL_BASE}"
    by_key = {(e["sequence"], e["config"]): e for e in summary}
    sequences = {}
    for name in dict.fromkeys(e["sequence"] for e in summary):
        good, bad = by_key.get((name, joint)), by_key.get((name, shuffled))
        if good is None or bad is None:
            continue
        sequences[name] = {
            "rel_d_true_poses": _json_float(good["rel_d_median"]),
            "rel_d_shuffled_poses": _json_float(bad["rel_d_median"]),
            "depth_error_true_poses": _json_float(good["target_depth_error_abs_median"]),
            "depth_error_shuffled_poses": _json_float(bad["target_depth_error_abs_median"]),
            "worse": bool(bad["rel_d_median"] > good["rel_d_median"]
                          and bad["target_depth_error_abs_median"] > good["target_depth_error_abs_median"]),
        }
    passed = bool(sequences) and all(v["worse"] for v in sequences.values())
    print(f"\nshuffled pose control {'passed' if passed else 'FAILED'}")
    for name, v in sequences.items():
        print(f"  {name} rel_d {v['rel_d_true_poses']} true, {v['rel_d_shuffled_poses']} shuffled; depth error "
              f"{v['depth_error_true_poses']} true, {v['depth_error_shuffled_poses']} shuffled")
    return {"sequences": sequences, "passed": passed}


# Summary.

def read_oracle_summary(path: Path | None) -> dict[tuple[str, str], dict[str, str]]:
    if path is None or not path.is_file():
        print(f"WARNING oracle summary {path} not found; the oracle comparison columns stay empty", flush=True)
        return {}
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    wrong = {r.get("scale") for r in rows} - {ORACLE_SCALE}
    if wrong:
        raise ValueError(f"{path} holds scale {sorted(map(str, wrong))}, expected {ORACLE_SCALE}")
    return {(r["sequence"], r["config"]): r for r in rows}


def summarize(rows: list[dict[str, str]], oracle: dict[tuple[str, str], dict[str, str]]) -> list[dict[str, Any]]:
    by_sequence: dict[str, dict[str, dict[int, dict[str, str]]]] = {}
    for row in rows:
        by_sequence.setdefault(row["sequence"], {}).setdefault(row["config"], {})[int(row["target_index"])] = row
    specs = {s.name: s for s in SPECS}
    summary = []
    for name, configs in by_sequence.items():
        anchor = {t: _value(r["rel_d"]) for t, r in configs["baseline_self"].items()}
        for config, by_target in configs.items():
            rel = {t: _value(r["rel_d"]) for t, r in by_target.items()}
            diff = [rel[t] - anchor[t] for t in rel]
            change = [100.0 * (rel[t] / anchor[t] - 1.0) for t in rel]
            values = list(by_target.values())
            low = [r for r in values if r["low_parallax"] == "True"]
            high = [r for r in values if r["low_parallax"] == "False"]
            match = oracle_config(specs[config]) if config in specs else ""
            reference = oracle.get((name, match), {})
            summary.append({
                "sequence": name, "config": config, "scale": SCALE, "poses": POSES, "targets": len(rel),
                "rel_d_median": _median(rel.values()),
                "delta_vs_baseline_median": _median(diff),
                "rel_change_pct_median": _median(change),
                "share_improved": float(np.mean([d < 0 for d in diff])),
                **{f"{c}_median": _median(_value(r[c]) for r in values)
                   for c in ("delta1_d", "rel_d_near", "rel_d_mid", "rel_d_far", "hole_fraction")},
                "low_parallax_targets": len(low),
                "target_depth_error_abs_median": _median(abs(_value(r["target_depth_error"])) for r in values),
                "target_depth_error_abs_median_parallax": _median(abs(_value(r["target_depth_error"])) for r in high),
                "target_depth_error_abs_median_low": _median(abs(_value(r["target_depth_error"])) for r in low),
                "scale_correlation": _correlation(values),
                "scale_correlation_parallax": _correlation(high),
                "scale_correlation_low": _correlation(low),
                "window_depth_error_median": _median(_value(r["window_depth_error"]) for r in values),
                "ridge_low_change_abs_median": _median(abs(_value(r["ridge_low_change"])) for r in values),
                "ridge_high_change_abs_median": _median(abs(_value(r["ridge_high_change"])) for r in values),
                "oracle_config": match,
                "oracle_rel_d_median": _value(reference.get("rel_d_median", "")),
                "oracle_rel_change_pct_median": _value(reference.get("rel_change_pct_median", "")),
            })
    return summary


def write_summary(path: Path, summary: list[dict[str, Any]]) -> None:
    def write(f) -> None:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for entry in summary:
            writer.writerow({k: _cell(v) if isinstance(v, float) else v for k, v in entry.items()})

    atomic_write(path, write, newline="")


def print_table(summary: list[dict[str, Any]]) -> None:
    print(f"\nscale {SCALE}, poses {POSES} (ORACLE). median per-target rel_d x100 after per-frame alignment of the "
          "fused output; oracle columns are the GT-aligned upper bound, not realigned")
    for name in dict.fromkeys(e["sequence"] for e in summary):
        print(name)
        print(f"  {'config':<42} {'rel_d':>7} {'change%':>8} {'improved':>8} {'low':>4} {'depth_err':>9} "
              f"{'oracle':>7} {'oracle%':>8}")
        for e in (e for e in summary if e["sequence"] == name):
            print(f"  {e['config']:<42} {e['rel_d_median']:>7.3f} {e['rel_change_pct_median']:>8.2f} "
                  f"{e['share_improved']:>8.2f} {e['low_parallax_targets']:>4} "
                  f"{e['target_depth_error_abs_median']:>9.3f} {e['oracle_rel_d_median']:>7.3f} "
                  f"{e['oracle_rel_change_pct_median']:>8.2f}")


def _correlation(rows: list[dict[str, str]]) -> float:
    pairs = np.array([(_value(r["target_scale"]), _value(r["oracle_scale"])) for r in rows], dtype=np.float64)
    pairs = pairs[np.all(np.isfinite(pairs), axis=1)] if pairs.size else pairs
    if len(pairs) < 3 or np.std(pairs[:, 0]) == 0 or np.std(pairs[:, 1]) == 0:
        return math.nan
    return float(np.corrcoef(pairs[:, 0], pairs[:, 1])[0, 1])


def _median(values) -> float:
    kept = np.array([v for v in values if math.isfinite(v)], dtype=np.float64)
    return float(np.median(kept)) if kept.size else math.nan


def _value(cell: str) -> float:
    return float(cell) if cell not in ("", None) else math.nan


def _cell(value: Any) -> str:
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, str):
        return value
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    value = float(value)
    return "" if math.isnan(value) else repr(value)


def _json_float(value: float) -> float | None:
    return value if math.isfinite(value) else None


# Command line.

class RegressionMismatch(RegressionError):
    """baseline_self or the no-fusion control differs after every output was written."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True, help="run directory holding run.json")
    parser.add_argument("--cache", type=Path, required=True,
                        help="GT-aligned depth cache of fusion_prepare.py, read only for the oracle diagnostic")
    parser.add_argument("--data-root", type=Path, required=True, help="folder containing the sequence folders")
    parser.add_argument("--sequences-config", type=Path, default=Path("configs/sequences.yaml"))
    parser.add_argument("--baseline-csv", type=Path, required=True, help="per_frame.csv of the baseline evaluation")
    parser.add_argument("--oracle-summary", type=Path,
                        help="oracle_fusion_summary.csv of the oracle edge group; default is the bundled "
                             "results/fusion/oracle_eval_edge/<run name>/oracle_fusion_summary.csv")
    parser.add_argument("--sequences", nargs="+", help="default is every sequence in the cache")
    parser.add_argument("--targets", type=Path,
                        help="YAML mapping targets to {sequence: [kept-frame indices]}; default is window positions "
                             f"{TARGET_FIRST} to {TARGET_FIRST + TARGET_COUNT - 1}, the targets of fusion_eval.py")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1, help="one sequence per worker process")
    parser.add_argument("--threads", type=int, default=1, help="OpenMP threads per worker")
    parser.add_argument("--score-threads", type=int, default=SCORE_THREADS,
                        help="torch threads for the float32 score reduction; must match the baseline evaluation run")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--allow-low-parallax", action="store_true",
                        help="refine per frame affines in low parallax windows too")
    parser.add_argument("--no-ridge-sensitivity", action="store_true", help="skip the ridge sensitivity refits")
    parser.add_argument("--preflight", action="store_true",
                        help=f"score baseline_self on the first {PREFLIGHT_TARGETS} targets of each sequence, run "
                             "the solver on one target per sequence and write nothing")
    return parser.parse_args(argv)


def validate(args: argparse.Namespace) -> None:
    for name in ("workers", "threads", "score_threads"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be at least 1, got {getattr(args, name)}")
    for path in (args.data_root, args.cache, args.predictions):
        if not path.is_dir():
            raise FileNotFoundError(f"{path} does not exist")
    if not args.baseline_csv.is_file():
        raise FileNotFoundError(f"baseline per_frame.csv {args.baseline_csv} not found")


def read_inputs(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    path = args.cache / PREPARE_RECORD
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; run fusion_prepare.py into {args.cache} first")
    prepared = json.loads(path.read_text())
    if prepared.get("scale") != ORACLE_SCALE:
        raise ValueError(f"{path} has scale {prepared.get('scale')!r}, expected {ORACLE_SCALE}")
    run = read_run(args.predictions)
    if prepared["predictions"]["run"] != run:
        raise ValueError(f"{args.cache} was prepared from a different prediction run than {args.predictions}")
    available = list(prepared["sequences"])
    names = args.sequences or available
    missing = [n for n in names if n not in available]
    if missing:
        raise ValueError(f"sequences {missing} are not in the cache {args.cache}, which holds {available}")
    return prepared, run, names


def requested_targets(path: Path | None) -> dict[str, list[int]] | None:
    if path is None:
        return None
    raw = yaml.safe_load(path.read_text())
    entries = raw.get("targets") if isinstance(raw, dict) else None
    if not isinstance(entries, dict) or not entries:
        raise ValueError(f"{path} needs a non-empty targets mapping of sequence to kept-frame indices")
    return {name: [int(i) for i in indices] for name, indices in entries.items()}


def solver_config(args: argparse.Namespace) -> JointScaleConfig:
    return JointScaleConfig(allow_low_parallax=args.allow_low_parallax)


def jobs_for(args: argparse.Namespace, names: list[str], prepared: dict[str, Any],
             baseline: dict[tuple[str, int], float]) -> list[JointJob]:
    explicit = requested_targets(args.targets)
    jobs = []
    for name in names:
        _, frames = load_window(name, args.data_root, args.sequences_config)
        window = {"start": frames[0].index, "count": len(frames)}
        if prepared["sequences"][name]["window"] != window:
            raise ValueError(f"{name} cache covers window {prepared['sequences'][name]['window']}, but "
                             f"{args.sequences_config} gives {window}")
        targets = tuple(explicit[name] if explicit is not None else default_targets(frames))
        check_targets(name, frames, targets)
        jobs.append(JointJob(name, args.data_root, args.sequences_config, args.predictions, args.cache, args.out,
                             targets, {i: v for (n, i), v in baseline.items() if n == name}, args.seed,
                             args.threads, args.score_threads, solver_config(args), not args.no_ridge_sensitivity))
    return jobs


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    """baseline_self on the first targets of each sequence against per_frame.csv, the depth-only scorer against
    evaluate_frame on the first of them, and the solver on that target; writes nothing."""
    validate(args)
    prepared, _, names = read_inputs(args)
    baseline = read_baseline(args.baseline_csv)
    set_torch_threads(args.score_threads)
    rows, failures = [], []
    baseline_spec = [s for s in SPECS if s.kind == "baseline"]
    for job in jobs_for(args, names, prepared, baseline):
        ctx, store, position = make_context(job)
        chosen = job.targets[:PREFLIGHT_TARGETS]
        for t in chosen:
            rows.extend(evaluate_target(ctx, store, baseline_spec, position[t]))
        first = position[chosen[0]]
        frame = ctx.frames[first]
        points = store.points(first)
        gt_depth, gt_valid, gt_points = store.gt(first)
        full = evaluate_frame(points, gt_points, gt_valid).rel_d
        depth_only = score_depth(points[..., 2], gt_depth, gt_valid, gt_points)["rel_d"]
        if not abs(full - depth_only) <= REGRESSION_TOLERANCE:
            failures.append(f"{job.name} frame {frame.index} depth-only rel_d {depth_only!r} but evaluate_frame "
                            f"gives {full!r}")
        sources = plan_sources(ctx.frames, first, base_config(CONTROL_BASE))
        started = time.perf_counter()
        window = solve_window(store, ctx.K, [first, *sources],
                              [ctx.frames[p].T_world_cam for p in [first, *sources]], job.solver)
        solved = window.solved
        depths = [solved.scale[k] * float(np.median(store.raw_z(p)[np.isfinite(store.raw_z(p))])) + solved.shift[k]
                  for k, p in enumerate(window.positions)]
        ok = (np.all(np.isfinite(solved.scale)) and np.all(solved.scale > 0) and math.isfinite(solved.cost)
              and all(d > 0 for d in depths))
        print(f"{job.name} solver on target {frame.index} with {CONTROL_BASE}: {time.perf_counter() - started:.1f} s, "
              f"{solved.iterations} iterations, cost {solved.cost:.4g}, low parallax {solved.low_parallax}, "
              f"baseline over depth {solved.baseline_ratio:.4f}, target depth error against the oracle "
              f"{diagnostics(store, window, job.solver, ctx.K, False)['target_depth_error']:+.3f}", flush=True)
        if not ok:
            failures.append(f"{job.name} frame {frame.index} solver returned scales {solved.scale}, cost "
                            f"{solved.cost} and median depths {depths}")
    regression = comparison_report("baseline_self against per_frame.csv", baseline_self_comparison(rows, baseline))
    if not regression["passed"]:
        failures.append(f"baseline_self differs from per_frame.csv for {regression['over_tolerance']} of "
                        f"{regression['rows']} targets (max |diff| {regression['max_abs_diff']})")
    if failures:
        raise RegressionError("preflight FAILED. " + "; ".join(failures))
    print(f"preflight passed. {regression['rows']} baseline_self rows match per_frame.csv, the depth-only scorer "
          "matches evaluate_frame and the solver returned positive depth")
    return regression


def run_joint(args: argparse.Namespace) -> dict[str, Any]:
    validate(args)
    start_time, started = now(), time.perf_counter()
    prepared, run, names = read_inputs(args)
    baseline = read_baseline(args.baseline_csv)
    oracle_path = args.oracle_summary or (REPO_ROOT / "results" / "fusion" / "oracle_eval_edge" / run["name"]
                                          / "oracle_fusion_summary.csv")
    oracle = read_oracle_summary(oracle_path)
    set_torch_threads(args.score_threads)
    jobs = jobs_for(args, names, prepared, baseline)
    args.out.mkdir(parents=True, exist_ok=True)
    todo = [j for j in jobs if not complete(fusion_csv(args.out, j.name), j.name, j.targets)]
    reused = [j.name for j in jobs if j not in todo]
    for name in reused:
        print(f"{name} {fusion_csv(args.out, name).name} is complete, skipping")
    outcomes = run_jobs(todo, args.workers)

    rows = [r for name in names for r in read_rows(fusion_csv(args.out, name))]
    summary = summarize(rows, oracle)
    regression = {
        "baseline_self": comparison_report("baseline_self against per_frame.csv",
                                           baseline_self_comparison(rows, baseline)),
        "nofusion": comparison_report(f"nofusion_joint_{CONTROL_BASE} against baseline_self",
                                      nofusion_comparison(rows)),
    }
    regression["passed"] = regression["baseline_self"]["passed"] and regression["nofusion"]["passed"]
    controls = control_report(summary)
    record = {
        "scale": SCALE,
        "poses": POSES,
        "predictions": {"dir": str(args.predictions), "run": run},
        "cache": {"dir": str(args.cache), "prepare": prepared, "use": "oracle diagnostic only"},
        "baseline_csv": str(args.baseline_csv),
        "oracle_summary": str(oracle_path) if oracle else None,
        "commits": {"project": git_commit(REPO_ROOT), "pointdit": git_commit(FORK_DIR)},
        "arguments": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "protocol": PROTOCOL,
        "solver": asdict(solver_config(args)),
        "configs": {s.name: s.describe() for s in SPECS},
        "targets": {j.name: list(j.targets) for j in jobs},
        "target_positions": (f"window positions {TARGET_FIRST} to {TARGET_FIRST + TARGET_COUNT - 1}"
                             if args.targets is None else f"from {args.targets}"),
        "sequences": {**{n: {"reused": True} for n in reused},
                      **{n: {**asdict(o), "reused": False} for n, o in outcomes.items()}},
        "regression": regression,
        "controls": controls,
        "cpus": available_cpus(),
        "start_time": start_time,
        "end_time": now(),
        "seconds": round(time.perf_counter() - started, 1),
    }
    write_summary(args.out / SUMMARY_CSV, summary)
    atomic_write(args.out / RECORD_JSON, lambda f: f.write(json.dumps(record, indent=2) + "\n"))
    status = {"status": "pass" if regression["passed"] else "fail", "scale": SCALE, "poses": POSES,
              "baseline_self": regression["baseline_self"], "nofusion": regression["nofusion"],
              "shuffled_control_passed": controls["passed"]}
    atomic_write(args.out / STATUS_JSON, lambda f: f.write(json.dumps(status, indent=2) + "\n"))
    print_table(summary)
    print(f"\nwrote {args.out}")
    if not regression["passed"]:
        raise RegressionMismatch(f"regression check FAILED (baseline_self passed "
                                 f"{regression['baseline_self']['passed']}, nofusion passed "
                                 f"{regression['nofusion']['passed']}); every output is already written to {args.out}")
    print("regression passed. baseline_self matches per_frame.csv and the no-fusion control matches baseline_self")
    return record


def main() -> None:
    args = parse_args()
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    try:
        if args.preflight:
            run_preflight(args)
        else:
            run_joint(args)
    except RegressionMismatch as error:
        banner = "=" * 30 + " FAILED " + "=" * 30
        print(f"\n{banner}\n{error}\nsee {args.out / STATUS_JSON}\n{banner}", flush=True)
        sys.exit(EXIT_REGRESSION)
    except RegressionError as error:
        print(f"\n{error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
