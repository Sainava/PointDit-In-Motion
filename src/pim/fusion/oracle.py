"""ORACLE-SCALE fusion evaluation. Every frame's prediction is aligned to its own GT before it is fused, so these
numbers are an upper bound for fusion with estimated scale and must never be reported as deployable results."""

from __future__ import annotations

import csv
import math
import os
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from pim.consistency.evaluation import atomic_write, peak_memory_mib
from pim.consistency.warp import depth_pose
from pim.data.tum import Frame, TUMSequence, apply_window, load_sequence, read_groundtruth
from pim.evaluation import load_prediction, prediction_path, set_torch_threads
from pim.fusion.engine import edge_fallback, fuse_frame, view_edge_band, without_band
from pim.fusion.select import select_sources
from pim.fusion.types import FusionConfig, FusionResult, SourceView, TargetSpec
from pim.fusion.warp import WarpedLayer, self_layer, warp_to_target
from pim.metrics.error_maps import depth_edge_band, rel_d_by_depth_tertile, relative_depth_error
from pim.metrics.geometry import FrameMetrics, backproject, evaluate_frame
from pim.paths import configure_fork

SCALE = "oracle_gt_aligned"
PREPARE_RECORD = "oracle_prepare.json"
TARGET_FIRST = 100
TARGET_COUNT = 100
DIVERSE_STRIDE_MS = 100.0
REGRESSION_TOLERANCE = 1e-6
# rel_d is percent in float32, so reduction order alone moves the last digits by ~1e-6; a real error is orders of
# magnitude larger.
BASELINE_SELF_TOLERANCE = 1e-4
WORST_SHOWN = 5
# Stored rel_d comes from evaluate_frame in the prepare worker, whose torch thread count may differ from the
# baseline run, and a float32 mean over ~2e5 pixels then moves by a few ulps.
STORED_REL_D_TOLERANCE = 1e-4
# per_frame.csv was written by evaluate.py with one worker on 4 CPUs, so torch reduced with 4 threads, and the
# float32 mean only reproduces it bit for bit with the same thread count.
SCORE_THREADS = 4
SEED = 0
GROUND_TRUTH_BASE = "causal_diverse_w4"
CONTROL_BASE = "causal_diverse_w4_rung_d"
ABLATION_BASES = ("causal_diverse_w4", "centred_adjacent_w4")
RUNGS: dict[str, dict[str, Any]] = {
    "a": {},
    "b": {"aggregator": "weighted_median"},
    "c": {"aggregator": "trimmed_mean", "trim_fraction": 0.2},
    "d": {"aggregator": "weighted_median", "front_layer": True},
    "e": {"aggregator": "weighted_median", "front_layer": True, "min_support": 2},
    "f": {"aggregator": "weighted_median", "front_layer": True, "min_support": 2, "ray_angle_weight": True,
          "baseline_weight": True},
}
# One factor at a time, each sweep starting from the noise-free level.
POSE_NOISE_LEVELS = ((0.0, 0.0), (5.0, 0.0), (10.0, 0.0), (20.0, 0.0), (0.0, 0.25), (0.0, 0.5), (0.0, 1.0))
EDGE_BASES = ("causal_diverse_w4", "centred_adjacent_w4", "causal_adjacent_w4")
EDGE_VARIANTS: dict[str, dict[str, Any]] = {
    "rungb": {},
    "edge_src2": {"source_edge_mask_px": 2},
    "edge_src5": {"source_edge_mask_px": 5},
    "edge_tgt3": {"target_edge_fallback_px": 3},
    "edge_src2_tgt3": {"source_edge_mask_px": 2, "target_edge_fallback_px": 3},
}
CONFIG_GROUPS = ("ladder", "edge")
EDGE_COLUMNS = ("rel_d_interior", "rel_d_edge")
LEGACY_ROW_COLUMNS = (
    "sequence", "target_index", "config", "scale", "rel_d", "delta1_d", "rel_d_near", "rel_d_mid", "rel_d_far",
    "hole_fraction", "median_n_sources", "rel_d_covered", "rel_d_covered_interior", "rel_d_covered_edge",
    "n_selected", "sources",
)
"""Columns of CSVs written before rel_d_interior and rel_d_edge existed; read with those two left empty."""
ROW_COLUMNS = (*LEGACY_ROW_COLUMNS[:-2], *EDGE_COLUMNS, *LEGACY_ROW_COLUMNS[-2:])
SUMMARY_COLUMNS = (
    "sequence", "config", "scale", "targets", "rel_d_median", "delta_vs_baseline_median", "rel_change_pct_median",
    "share_improved", "hole_fraction_median", "rel_d_covered_median", "rel_d_covered_interior_median",
    "rel_d_covered_edge_median", "rel_d_interior_median", "rel_d_edge_median",
)
PROGRESS_EVERY = 10


class RegressionError(RuntimeError):
    pass


@dataclass(frozen=True)
class FusionSpec:
    name: str
    fusion: FusionConfig
    depth: str = "oracle"
    """oracle fuses each source's GT-aligned prediction, gt fuses GT depth with depth-timestamp poses."""
    poses: str = "true"
    """true, shuffled (each source gets another window frame's pose) or noisy."""
    noise_mm: float = 0.0
    noise_deg: float = 0.0
    gt_target_fallback_px: int | None = None
    """GT ceiling only. Pixels in this-width depth edge band of the target GT take the target GT after fusion, as
    target_edge_fallback_px does with the target's own layer, but without include_self."""

    def __post_init__(self) -> None:
        if self.gt_target_fallback_px is not None and self.depth != "gt":
            raise ValueError(f"{self.name} gt_target_fallback_px needs depth gt")

    @property
    def key(self) -> tuple[Any, ...]:
        poses = "true" if self.poses == "noisy" and self.noise_mm == 0 and self.noise_deg == 0 else self.poses
        noise = (self.noise_mm, self.noise_deg) if poses == "noisy" else (0.0, 0.0)
        return self.fusion, self.depth, poses, noise, self.gt_target_fallback_px

    def describe(self) -> dict[str, Any]:
        # Edge options are listed only when set, so ladder descriptions are as before.
        edge = ("source_edge_mask_px", "target_edge_fallback_px")
        fusion = {k: v for k, v in vars(self.fusion).items() if k != "keep_layers" and not (k in edge and v is None)}
        extra = {} if self.gt_target_fallback_px is None else {"gt_target_fallback_px": self.gt_target_fallback_px}
        return {"depth": self.depth, "poses": self.poses, "noise_mm": self.noise_mm, "noise_deg": self.noise_deg,
                **extra, "fusion": fusion}


def build_specs() -> dict[str, FusionSpec]:
    specs = [FusionSpec("baseline_self", FusionConfig(window=0))]
    grid = {"adjacent": {"causal": (1, 2, 4), "centred": (1, 2, 4)},
            "diverse": {"causal": (1, 2, 3, 4), "centred": (1, 2)}}
    for selection, modes in grid.items():
        for mode, windows in modes.items():
            for window in windows:
                config = FusionConfig(window=window, mode=mode, selection=selection, stride_ms=DIVERSE_STRIDE_MS)
                specs.append(FusionSpec(f"{mode}_{selection}_w{window}", config))
    by_name = {s.name: s for s in specs}
    for base in ABLATION_BASES:
        for rung, changes in RUNGS.items():
            specs.append(FusionSpec(f"{base}_rung_{rung}", replace(by_name[base].fusion, **changes)))
    ceiling = by_name[GROUND_TRUTH_BASE].fusion
    specs.append(FusionSpec(f"gt_ceiling_{GROUND_TRUTH_BASE}_self", ceiling, depth="gt"))
    specs.append(FusionSpec(f"gt_ceiling_{GROUND_TRUTH_BASE}_noself", replace(ceiling, include_self=False), depth="gt"))
    # Rung d settings on GT separate the engine's own edge growth from the method error of rungs d to f.
    specs.append(FusionSpec(f"gt_ceiling_{GROUND_TRUTH_BASE}_rung_d_noself",
                            replace(ceiling, **RUNGS["d"], include_self=False), depth="gt"))
    by_name = {s.name: s for s in specs}
    control = by_name[CONTROL_BASE].fusion
    specs.append(FusionSpec(f"shuffled_poses_{CONTROL_BASE}", control, poses="shuffled"))
    for mm, deg in POSE_NOISE_LEVELS:
        specs.append(FusionSpec(f"pose_noise_t{mm:g}mm_r{deg:g}deg_{CONTROL_BASE}", control, poses="noisy",
                                noise_mm=mm, noise_deg=deg))
    return {s.name: s for s in specs}


SPECS = build_specs()


def build_edge_specs(ladder: dict[str, FusionSpec] = SPECS) -> dict[str, FusionSpec]:
    """Edge-aware options on median-aggregated bases, with GT ceilings for the splat and for both options."""
    specs = [ladder["baseline_self"]]
    for base in EDGE_BASES:
        median = replace(ladder[base].fusion, **RUNGS["b"])
        for prefix, changes in EDGE_VARIANTS.items():
            specs.append(FusionSpec(f"{prefix}_{base}", replace(median, **changes)))
    specs.append(FusionSpec("gt_ceiling_causal_adjacent_w1_noself",
                            replace(ladder["causal_adjacent_w1"].fusion, include_self=False), depth="gt"))
    specs.append(FusionSpec(f"gt_ceiling_{GROUND_TRUTH_BASE}_noself_src2_tgt3",
                            replace(ladder[GROUND_TRUTH_BASE].fusion, **RUNGS["b"], include_self=False,
                                    source_edge_mask_px=2), depth="gt", gt_target_fallback_px=3))
    return {s.name: s for s in specs}


SPEC_GROUPS = {"ladder": SPECS, "edge": build_edge_specs()}


def group_specs(group: str) -> dict[str, FusionSpec]:
    if group not in SPEC_GROUPS:
        raise ValueError(f"unknown configuration group {group!r}; known are {list(SPEC_GROUPS)}")
    return SPEC_GROUPS[group]


def resolve_specs(names: Sequence[str] | None, group: str = "ladder") -> list[FusionSpec]:
    """The requested configurations of the group in registry order, always with baseline_self as the regression
    anchor."""
    registry = group_specs(group)
    wanted = set(registry) if names is None else set(names) | {"baseline_self"}
    unknown = sorted(wanted - set(registry))
    if unknown:
        raise ValueError(f"unknown fusion configurations {unknown} in group {group}; known are {list(registry)}")
    return [s for name, s in registry.items() if name in wanted]


# Cache of per-frame GT-aligned depth.

def cache_path(cache_dir: Path, sequence: str, frame_index: int) -> Path:
    return cache_dir / sequence / f"{frame_index:05d}.npz"


@dataclass(frozen=True)
class CachedAlignment:
    frame_index: int
    aligned_depth: np.ndarray
    """(H, W) float32 prediction after upstream's per-frame depth scale and shift fit to this frame's GT."""
    rel_d: float
    delta1_d: float
    num_valid: int
    skipped: bool


def write_alignment(path: Path, frame_index: int, rgb_file: str, metrics: FrameMetrics) -> None:
    arrays = {
        "aligned_depth": metrics.aligned_depth.astype(np.float32),
        "frame_index": np.int64(frame_index),
        "rel_d": np.float64(metrics.rel_d),
        "delta1_d": np.float64(metrics.delta1_d),
        "num_valid": np.int64(metrics.num_valid),
        "skipped": np.bool_(metrics.skipped),
        "rgb_file": np.str_(rgb_file),
        "scale": np.str_(SCALE),
    }
    atomic_write(path, lambda f: np.savez(f, **arrays), mode="wb")


def read_alignment(path: Path, frame_index: int) -> CachedAlignment:
    if not path.is_file():
        raise FileNotFoundError(f"aligned depth {path} not found; run fusion_prepare.py first")
    with np.load(path) as data:
        if int(data["frame_index"]) != frame_index or str(data["scale"]) != SCALE:
            raise ValueError(f"{path} holds frame {int(data['frame_index'])} at scale {str(data['scale'])}, "
                             f"expected frame {frame_index} at {SCALE}")
        return CachedAlignment(frame_index, data["aligned_depth"].astype(np.float32), float(data["rel_d"]),
                               float(data["delta1_d"]), int(data["num_valid"]), bool(data["skipped"]))


def source_valid(depth: np.ndarray) -> np.ndarray:
    # GT validity is never used here, since GT is not available in deployment.
    with np.errstate(invalid="ignore"):
        return np.isfinite(depth) & (depth > 0)


Aligner = Callable[[Frame, TUMSequence, Path], tuple[FrameMetrics, str]]


def align_frame(frame: Frame, sequence: TUMSequence, run_dir: Path) -> tuple[FrameMetrics, str]:
    """The same steps as evaluate.py's evaluate_task, so rel_d matches per_frame.csv."""
    prediction = load_prediction(prediction_path(run_dir, sequence.name, frame.index), frame, sequence.root)
    depth, depth_valid = frame.load_depth()
    gt_points, gt_valid = backproject(depth, depth_valid, sequence.K)
    return evaluate_frame(prediction.points, gt_points, gt_valid, pred_valid=None), prediction.rgb_file


@dataclass(frozen=True)
class PrepareJob:
    name: str
    data_root: Path
    sequences_config: Path
    run_dir: Path
    cache_dir: Path
    threads: int


@dataclass(frozen=True)
class PrepareOutcome:
    name: str
    frames: int
    written: int
    existing: int
    seconds: float
    peak_mem_mib: float
    pid: int


def load_window(name: str, data_root: Path, sequences_config: Path) -> tuple[TUMSequence, list[Frame]]:
    sequence = load_sequence(data_root / name, sequences_config)
    return sequence, apply_window(sequence)


def prepare_sequence(job: PrepareJob, align: Aligner = align_frame) -> PrepareOutcome:
    started = time.perf_counter()
    set_torch_threads(job.threads)
    sequence, frames = load_window(job.name, job.data_root, job.sequences_config)
    written = existing = 0
    for done, frame in enumerate(frames, start=1):
        path = cache_path(job.cache_dir, job.name, frame.index)
        if path.is_file():
            existing += 1
        else:
            metrics, rgb_file = align(frame, sequence, job.run_dir)
            write_alignment(path, frame.index, rgb_file, metrics)
            written += 1
        if done % 50 == 0 or done == len(frames):
            print(f"  {job.name} {done}/{len(frames)} frames ({written} aligned, {existing} already cached)",
                  flush=True)
    return PrepareOutcome(job.name, len(frames), written, existing, time.perf_counter() - started,
                          peak_memory_mib(), os.getpid())


# Targets, sources and poses.

def default_targets(frames: Sequence[Frame], all_targets: bool = False) -> list[int]:
    """Kept-frame indices of window positions 100 to 199, or of the whole window with all_targets."""
    if all_targets:
        return [f.index for f in frames]
    if len(frames) < TARGET_FIRST + TARGET_COUNT:
        raise ValueError(f"window of {len(frames)} frames is too short for targets at positions "
                         f"{TARGET_FIRST} to {TARGET_FIRST + TARGET_COUNT - 1}")
    return [f.index for f in frames[TARGET_FIRST:TARGET_FIRST + TARGET_COUNT]]


def check_targets(name: str, frames: Sequence[Frame], targets: Sequence[int]) -> None:
    inside = {f.index for f in frames}
    outside = [t for t in targets if t not in inside]
    if outside or not targets:
        raise ValueError(f"{name} targets {outside or 'none'} are not kept frames of the window "
                         f"{frames[0].index} to {frames[-1].index}")
    if len(set(targets)) != len(targets):
        raise ValueError(f"{name} lists a target twice")


def plan_sources(frames: Sequence[Frame], target_position: int, config: FusionConfig) -> list[int]:
    """Window positions of the sources, chosen by timestamp over the whole window by the engine's selection."""
    return select_sources([f.timestamp for f in frames], frames[target_position].timestamp, config)


def rotation_from_vector(w: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(w))
    if theta < 1e-15:
        return np.eye(3)
    k = w / theta
    cross = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(theta) * cross + (1 - math.cos(theta)) * cross @ cross


def noisy_pose(T_world_cam: np.ndarray, draw: np.ndarray, noise_mm: float, noise_deg: float) -> np.ndarray:
    """Camera-frame perturbation with per-axis sigma noise_mm on translation and noise_deg on the rotation vector."""
    delta = np.eye(4)
    delta[:3, :3] = rotation_from_vector(draw[3:] * math.radians(noise_deg))
    delta[:3, 3] = draw[:3] * noise_mm / 1000.0
    return T_world_cam @ delta


def pose_noise_draw(seed: int, target_index: int, source_index: int) -> np.ndarray:
    # The same standard normal draw scales with every noise level, so the sweep is nested.
    return np.random.default_rng([seed, 1, target_index, source_index]).standard_normal(6)


def shuffled_positions(seed: int, target_index: int, sources: Sequence[int], window: int) -> list[int]:
    """For each source position, a different random position of the window whose pose it is given."""
    rng = np.random.default_rng([seed, 0, target_index])
    picks = []
    for position in sources:
        other = int(rng.integers(window - 1))
        picks.append(other + (other >= position))
    return picks


# Scoring.

def upstream_depth_metrics(depth: np.ndarray, gt_depth: np.ndarray, gt_valid: np.ndarray) -> tuple[float, float]:
    """rel_d and delta1_d x100 with upstream's own float32 functions, as evaluate_frame computes them."""
    import torch

    configure_fork()
    from metrics.metrics import delta1_depth, rel_depth

    mask = torch.from_numpy(np.ascontiguousarray(gt_valid, dtype=bool))
    pred = torch.from_numpy(np.ascontiguousarray(depth, dtype=np.float32))[mask]
    gt = torch.from_numpy(np.ascontiguousarray(gt_depth, dtype=np.float32))[mask]
    with torch.no_grad():
        return rel_depth(pred, gt) * 100.0, delta1_depth(pred, gt) * 100.0


def score_fused(result: FusionResult, fallback: np.ndarray, gt_depth: np.ndarray, gt_valid: np.ndarray,
                edge_band: np.ndarray | None = None, target_band: np.ndarray | None = None) -> dict[str, float]:
    """Scored against GT on GT-valid pixels without realignment; holes take the target's own aligned depth. With
    edge_band, rel_d_covered is also split into interior pixels and pixels inside the band. rel_d is always split
    into rel_d_interior and rel_d_edge over target_band, by default depth_edge_band of the target GT."""
    valid = gt_valid.astype(bool)
    if target_band is None:
        target_band = depth_edge_band(gt_depth, valid)
    hole = result.hole
    depth = np.where(hole, fallback, result.depth).astype(np.float32)
    rel_d, delta1_d = upstream_depth_metrics(depth, gt_depth, valid)
    error = relative_depth_error(depth, gt_depth, valid)
    covered = valid & ~hole
    return {
        "rel_d": rel_d,
        "delta1_d": delta1_d,
        **rel_d_by_depth_tertile(error, gt_depth, valid),
        "hole_fraction": float(hole[valid].mean()),
        "median_n_sources": float(np.median(result.n_sources[valid])),
        "rel_d_covered": _mean_x100(error[covered]),
        "rel_d_covered_interior": math.nan if edge_band is None else _mean_x100(error[covered & ~edge_band]),
        "rel_d_covered_edge": math.nan if edge_band is None else _mean_x100(error[covered & edge_band]),
        "rel_d_interior": _mean_x100(error[valid & ~target_band]),
        "rel_d_edge": _mean_x100(error[valid & target_band]),
    }


def _mean_x100(values: np.ndarray) -> float:
    return 100.0 * float(values.mean(dtype=np.float64)) if values.size else math.nan


# Per-sequence evaluation.

@dataclass(frozen=True)
class EvalJob:
    name: str
    data_root: Path
    sequences_config: Path
    cache_dir: Path
    out: Path
    targets: tuple[int, ...]
    spec_names: tuple[str, ...]
    baseline: dict[int, float]
    """Baseline rel_d of per_frame.csv by kept-frame index."""
    seed: int = SEED
    threads: int = 1
    score_threads: int = SCORE_THREADS
    config_group: str = "ladder"


@dataclass(frozen=True)
class EvalOutcome:
    name: str
    targets: int
    configs: int
    fusions: int
    seconds: float
    peak_mem_mib: float
    pid: int


class ViewStore:
    """Aligned and GT depth of the frames the current target needs, dropped once a later target does not."""

    def __init__(self, cache_dir: Path, name: str, frames: Sequence[Frame]) -> None:
        self._cache_dir, self._name, self._frames = cache_dir, name, frames
        self._aligned: dict[int, np.ndarray] = {}
        self._gt: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._bands: dict[tuple[str, int, int], np.ndarray] = {}

    def retain(self, positions: Iterable[int]) -> None:
        keep = set(positions)
        for store in (self._aligned, self._gt):
            for position in [p for p in store if p not in keep]:
                del store[position]
        for key in [k for k in self._bands if k[1] not in keep]:
            del self._bands[key]

    def band(self, depth: str, position: int, width_px: int, compute: Callable[[], np.ndarray]) -> np.ndarray:
        """Edge band of a frame's oracle or GT depth at a width, computed once while the frame is held."""
        key = (depth, position, width_px)
        if key not in self._bands:
            self._bands[key] = compute()
        return self._bands[key]

    @property
    def held(self) -> int:
        return len(set(self._aligned) | set(self._gt))

    def aligned(self, position: int) -> np.ndarray:
        if position not in self._aligned:
            frame = self._frames[position]
            self._aligned[position] = read_alignment(cache_path(self._cache_dir, self._name, frame.index),
                                                     frame.index).aligned_depth
        return self._aligned[position]

    def gt(self, position: int) -> tuple[np.ndarray, np.ndarray]:
        if position not in self._gt:
            depth, valid = self._frames[position].load_depth()
            self._gt[position] = depth, valid & np.isfinite(depth)
        return self._gt[position]


@dataclass
class SequenceContext:
    name: str
    K: np.ndarray
    frames: Sequence[Frame]
    seed: int
    gt_trajectory: tuple[np.ndarray, np.ndarray] | None = None
    _depth_poses: dict[int, np.ndarray] | None = None

    def pose(self, position: int, depth_time: bool) -> np.ndarray:
        frame = self.frames[position]
        if not depth_time:
            return frame.T_world_cam
        if self.gt_trajectory is None:
            raise ValueError("depth-timestamp poses need the GT trajectory")
        if self._depth_poses is None:
            self._depth_poses = {}
        if position not in self._depth_poses:
            self._depth_poses[position] = depth_pose(frame, *self.gt_trajectory)
        return self._depth_poses[position]


class TargetFusion:
    """Fuses one target for several configurations, warping each distinct source view into it only once."""

    def __init__(self, ctx: SequenceContext, store: ViewStore, target_position: int) -> None:
        self.ctx, self.store, self.position = ctx, store, target_position
        self.target = ctx.frames[target_position]
        self._views: dict[tuple[Any, ...], SourceView] = {}
        self._keys: dict[int, tuple[Any, ...]] = {}
        self._layers: dict[tuple[Any, ...], WarpedLayer] = {}

    def _view(self, key: tuple[Any, ...], position: int, pose: Callable[[], np.ndarray]) -> SourceView:
        if key not in self._views:
            if key[0] == "gt":
                depth, valid = self.store.gt(position)
            else:
                depth = self.store.aligned(position)
                valid = source_valid(depth)
            view = SourceView(depth, valid, pose(), self.ctx.K, self.ctx.frames[position].timestamp)
            self._views[key] = view
            self._keys[id(view)] = key
        return self._views[key]

    def _warp(self, view: SourceView, target: TargetSpec) -> WarpedLayer:
        key = self._keys[id(view)]
        if key not in self._layers:
            self._layers[key] = warp_to_target(view, target)
        return self._layers[key]

    def _band(self, view: SourceView, width_px: int) -> np.ndarray:
        depth, position = self._keys[id(view)][:2]
        return self.store.band(depth, position, width_px, lambda: view_edge_band(view, width_px))

    def _masked(self, view: SourceView, width_px: int) -> SourceView:
        key = (*self._keys[id(view)], "edge_masked", width_px)
        if key not in self._views:
            masked = without_band(view, self._band(view, width_px))
            self._views[key] = masked
            self._keys[id(masked)] = key
        return self._views[key]

    def source(self, spec: FusionSpec, position: int, pose_position: int) -> SourceView:
        gt = spec.depth == "gt"
        if spec.poses == "shuffled":
            key: tuple[Any, ...] = (spec.depth, position, "shuffled", pose_position)
            return self._view(key, position, lambda: self.ctx.pose(pose_position, gt))
        if spec.poses == "noisy" and (spec.noise_mm or spec.noise_deg):
            def pose() -> np.ndarray:
                draw = pose_noise_draw(self.ctx.seed, self.target.index, self.ctx.frames[position].index)
                return noisy_pose(self.ctx.pose(position, gt), draw, spec.noise_mm, spec.noise_deg)
            return self._view((spec.depth, position, "noisy", spec.noise_mm, spec.noise_deg), position, pose)
        return self._view((spec.depth, position, "true"), position, lambda: self.ctx.pose(position, gt))

    def fuse(self, spec: FusionSpec, sources: list[int]) -> FusionResult:
        pose_positions = (shuffled_positions(self.ctx.seed, self.target.index, sources, len(self.ctx.frames))
                          if spec.poses == "shuffled" else sources)
        views = [self.source(spec, p, q) for p, q in zip(sources, pose_positions)]
        # The target's own layer and pose are never perturbed.
        own = self._view((spec.depth, self.position, "true"), self.position,
                         lambda: self.ctx.pose(self.position, spec.depth == "gt"))
        shape = self.store.aligned(self.position).shape
        result = fuse_frame(views, TargetSpec(own.T_world_cam, self.ctx.K, shape), self.target.timestamp,
                            spec.fusion, own if spec.fusion.include_self else None, warp=self._warp,
                            edge_band=self._band, mask_source=self._masked)
        if result.source_indices != tuple(range(len(views))):
            raise RuntimeError(f"{self.ctx.name} target {self.target.index} {spec.name} the engine chose sources "
                               f"{result.source_indices} from the planned {sources}")
        if spec.gt_target_fallback_px is not None:
            layer = self_layer(own, 1)
            result = edge_fallback(result, layer.depth, layer.hit, self._band(own, spec.gt_target_fallback_px))
        return result


def evaluate_target(ctx: SequenceContext, store: ViewStore, specs: Sequence[FusionSpec], target_position: int
                    ) -> list[dict[str, Any]]:
    target = ctx.frames[target_position]
    plans = {s.name: plan_sources(ctx.frames, target_position, s.fusion) for s in specs}
    store.retain({target_position, *(p for plan in plans.values() for p in plan)})
    fallback = store.aligned(target_position)
    gt_depth, gt_valid = store.gt(target_position)
    fusion = TargetFusion(ctx, store, target_position)
    edge_band = depth_edge_band(gt_depth, gt_valid)
    scored: dict[tuple[Any, ...], dict[str, float]] = {}
    rows = []
    for spec in specs:
        if spec.key not in scored:
            result = fusion.fuse(spec, plans[spec.name])
            scored[spec.key] = score_fused(result, fallback, gt_depth, gt_valid,
                                           edge_band if spec.depth == "gt" else None, edge_band)
        rows.append({
            "sequence": ctx.name, "target_index": target.index, "config": spec.name, "scale": SCALE,
            **{k: _cell(v) for k, v in scored[spec.key].items()},
            "n_selected": len(plans[spec.name]),
            "sources": " ".join(str(ctx.frames[p].index) for p in plans[spec.name]),
        })
    return rows


def fusion_csv(out: Path, name: str) -> Path:
    return out / f"oracle_fusion_{name}.csv"


def evaluate_sequence(job: EvalJob) -> EvalOutcome:
    started = time.perf_counter()
    # Fusion runs in numpy; torch only reduces the scores and must reduce as the baseline run did.
    set_torch_threads(job.score_threads)
    sequence, frames = load_window(job.name, job.data_root, job.sequences_config)
    check_targets(job.name, frames, job.targets)
    specs = resolve_specs(job.spec_names, job.config_group)
    needs_gt_poses = any(s.depth == "gt" for s in specs)
    ctx = SequenceContext(job.name, sequence.K, frames, job.seed,
                          read_groundtruth(sequence.root) if needs_gt_poses else None)
    store = ViewStore(job.cache_dir, job.name, frames)
    position = {f.index: k for k, f in enumerate(frames)}
    rows: list[dict[str, Any]] = []
    print(f"{job.name} {len(job.targets)} targets x {len(specs)} configs, ORACLE scale (pid {os.getpid()})",
          flush=True)
    for done, target_index in enumerate(job.targets, start=1):
        target_rows = evaluate_target(ctx, store, specs, position[target_index])
        # Warn only; the run checks every frame after all outputs are written.
        for line in baseline_self_failures(target_rows, {(job.name, k): v for k, v in job.baseline.items()}):
            print(f"  WARNING {line}", flush=True)
        rows.extend(target_rows)
        if done % PROGRESS_EVERY == 0 or done == len(job.targets):
            elapsed = time.perf_counter() - started
            print(f"  {job.name} {done}/{len(job.targets)} targets, {elapsed / done:.1f} s per target, "
                  f"{store.held} frames held, peak {peak_memory_mib():.0f} MiB", flush=True)
    write_rows(fusion_csv(job.out, job.name), rows)
    fusions = len({s.key for s in specs}) * len(job.targets)
    return EvalOutcome(job.name, len(job.targets), len(specs), fusions, time.perf_counter() - started,
                       peak_memory_mib(), os.getpid())


def expected_keys(name: str, targets: Iterable[int], specs: Iterable[FusionSpec]) -> set[tuple[str, int, str]]:
    return {(name, t, s.name) for t in targets for s in specs}


def row_key(row: dict[str, Any]) -> tuple[str, int, str]:
    return row["sequence"], int(row["target_index"]), row["config"]


def complete(path: Path, keys: set[tuple[str, int, str]]) -> bool:
    if not path.is_file():
        return False
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if tuple(reader.fieldnames or ()) != ROW_COLUMNS:
            return False
        return {row_key(r) for r in reader} == keys


def write_rows(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    def write(f) -> None:
        writer = csv.DictWriter(f, fieldnames=ROW_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    atomic_write(path, write, newline="")


def read_rows(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        columns = tuple(reader.fieldnames or ())
        if columns == LEGACY_ROW_COLUMNS:
            return [{c: r.get(c, "") for c in ROW_COLUMNS} for r in reader]
        if columns != ROW_COLUMNS:
            raise ValueError(f"{path} does not have the oracle fusion columns")
        return list(reader)


# Regression checks.

def read_baseline(path: Path) -> dict[tuple[str, int], float]:
    with open(path, newline="") as f:
        return {(r["sequence"], int(r["frame_index"])): float(r["rel_d"]) if r["rel_d"] else math.nan
                for r in csv.DictReader(f)}


@dataclass(frozen=True)
class Mismatch:
    sequence: str
    frame_index: int
    actual: float
    expected: float

    @property
    def diff(self) -> float:
        """|actual - expected|, inf when either side is missing or not finite."""
        if not (math.isfinite(self.actual) and math.isfinite(self.expected)):
            return math.inf
        return abs(self.actual - self.expected)

    def describe(self, tolerance: float) -> str:
        return (f"baseline_self {self.sequence} frame {self.frame_index} rel_d {self.actual!r} but per_frame.csv "
                f"has {self.expected!r} (|diff| {self.diff:.3g}, tolerance {tolerance:g})")


def baseline_self_comparison(rows: Iterable[dict[str, Any]],
                             baseline: dict[tuple[str, int], float]) -> list[Mismatch]:
    """Every baseline_self row against per_frame.csv, worst first."""
    compared = []
    for row in rows:
        if row["config"] != "baseline_self":
            continue
        key = (row["sequence"], int(row["target_index"]))
        actual = float(row["rel_d"]) if row["rel_d"] != "" else math.nan
        compared.append(Mismatch(key[0], key[1], actual, baseline.get(key, math.nan)))
    return sorted(compared, key=lambda m: m.diff, reverse=True)


def baseline_self_failures(rows: Iterable[dict[str, Any]], baseline: dict[tuple[str, int], float],
                           tolerance: float = BASELINE_SELF_TOLERANCE) -> list[str]:
    return [m.describe(tolerance) for m in baseline_self_comparison(rows, baseline) if m.diff > tolerance]


def regression_positions(count: int, n: int) -> list[int]:
    return sorted({int(round(p)) for p in np.linspace(0, count - 1, n)})


def cache_failures(cache_dir: Path, name: str, frames: Sequence[Frame], positions: Iterable[int],
                   baseline: dict[tuple[str, int], float], tolerance: float = REGRESSION_TOLERANCE) -> list[str]:
    """The cached aligned depth scored directly against GT must reproduce per_frame.csv rel_d."""
    failures = []
    for position in positions:
        frame = frames[position]
        cached = read_alignment(cache_path(cache_dir, name, frame.index), frame.index)
        depth, valid = frame.load_depth()
        rel_d, _ = upstream_depth_metrics(cached.aligned_depth, depth, valid)
        expected = baseline.get((name, frame.index), math.nan)
        for label, value, limit in (("rescored", rel_d, tolerance), ("stored", cached.rel_d, STORED_REL_D_TOLERANCE)):
            if not (math.isfinite(value) and abs(value - expected) <= limit):
                failures.append(f"cache {name} frame {frame.index} {label} rel_d {value!r} but per_frame.csv has "
                                f"{expected!r} (|diff| {abs(value - expected):.3g}, tolerance {limit:g})")
    return failures


# Summary.

def summarize(rows: Iterable[dict[str, str]]) -> list[dict[str, Any]]:
    by_sequence: dict[str, dict[str, dict[int, dict[str, str]]]] = {}
    for row in rows:
        by_sequence.setdefault(row["sequence"], {}).setdefault(row["config"], {})[int(row["target_index"])] = row
    summary = []
    for name, configs in by_sequence.items():
        anchor = {t: float(r["rel_d"]) for t, r in configs["baseline_self"].items()}
        for config, by_target in configs.items():
            rel = {t: float(r["rel_d"]) for t, r in by_target.items()}
            diff = [rel[t] - anchor[t] for t in rel]
            change = [100.0 * (rel[t] / anchor[t] - 1.0) for t in rel]
            summary.append({
                "sequence": name, "config": config, "scale": SCALE, "targets": len(rel),
                "rel_d_median": _median(rel.values()),
                "delta_vs_baseline_median": _median(diff),
                "rel_change_pct_median": _median(change),
                "share_improved": float(np.mean([d < 0 for d in diff])),
                "hole_fraction_median": _median(float(r["hole_fraction"]) for r in by_target.values()),
                **{f"{c}_median": _median(_value(r[c]) for r in by_target.values())
                   for c in ("rel_d_covered", "rel_d_covered_interior", "rel_d_covered_edge", *EDGE_COLUMNS)},
            })
    return summary


def write_summary(path: Path, summary: Sequence[dict[str, Any]]) -> None:
    def write(f) -> None:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for entry in summary:
            writer.writerow({k: _cell(v) if isinstance(v, float) else v for k, v in entry.items()})

    atomic_write(path, write, newline="")


def _median(values: Iterable[float]) -> float:
    kept = np.array([v for v in values if math.isfinite(v)], dtype=np.float64)
    return float(np.median(kept)) if kept.size else math.nan


def _value(cell: str) -> float:
    return float(cell) if cell else math.nan


def _cell(value: float) -> str:
    return "" if math.isnan(value) else repr(float(value))
