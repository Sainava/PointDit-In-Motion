from __future__ import annotations

import csv
import importlib.util
import json
import math
import shutil
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pim.data.tum import Frame, apply_window, load_sequence
from pim.evaluation import COLUMNS, build_tasks, evaluate_task
from pim.fusion import oracle
from pim.fusion.oracle import (
    ROW_COLUMNS, SCALE, SPECS, RegressionError, baseline_self_failures, cache_path, check_targets, default_targets,
    noisy_pose, plan_sources, read_alignment, resolve_specs, shuffled_positions,
)
from pim.metrics.geometry import FrameMetrics, backproject


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_script",
                                                  Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


fusion_prepare = load_script("fusion_prepare")
fusion_eval = load_script("fusion_eval")

H, W = 40, 52
FX, FY, CX, CY = 45.0, 45.0, 25.5, 19.5
K = np.array([[FX, 0.0, CX], [0.0, FY, CY], [0.0, 0.0, 1.0]])
NAME = "seq_a"
# Slot 5 has no frame, so kept-frame indices and 30 Hz slots differ after it.
SLOTS = tuple(s for s in range(25) if s != 5)
TARGETS = [15, 16, 17]


def camera_pose(t: float) -> np.ndarray:
    yaw = 0.15 * t
    T = np.eye(4)
    T[:3, :3] = [[math.cos(yaw), 0.0, math.sin(yaw)], [0.0, 1.0, 0.0], [-math.sin(yaw), 0.0, math.cos(yaw)]]
    T[:3, 3] = [0.3 * t, 0.02 * t, 0.0]
    return T


def render(T_world_cam: np.ndarray) -> np.ndarray:
    v, u = np.indices((H, W), dtype=np.float64)
    rays = np.stack([(u - CX) / FX, (v - CY) / FY, np.ones_like(u)], axis=-1)
    d = rays @ T_world_cam[:3, :3].T
    c = T_world_cam[:3, 3]
    wall = (3.0 - c[2]) / d[..., 2]
    box = (1.8 - c[2]) / d[..., 2]
    hit = c + box[..., None] * d
    on_box = (np.abs(hit[..., 0] - 0.1) < 0.25) & (np.abs(hit[..., 1]) < 0.2)
    return np.where(on_box, box, wall)


def quaternion(R: np.ndarray) -> tuple[float, float, float, float]:
    w = math.sqrt(max(1.0 + R[0, 0] + R[1, 1] + R[2, 2], 1e-12)) / 2
    return (R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w), (R[1, 0] - R[0, 1]) / (4 * w), w


def write_sequence(data_root: Path) -> None:
    seq = data_root / NAME
    (seq / "rgb").mkdir(parents=True)
    (seq / "depth").mkdir()
    rgb_lines, depth_lines = [], []
    for i, slot in enumerate(SLOTS):
        t = slot / 30
        depth_t = t + 0.002
        depth = render(camera_pose(depth_t))
        depth[:3, :5] = 0.0
        Image.fromarray(np.full((H, W, 3), 5 * i, np.uint8)).save(seq / "rgb" / f"{i}.png")
        Image.fromarray(np.round(depth * 5000).astype(np.uint16)).save(seq / "depth" / f"{i}.png")
        rgb_lines.append(f"{1000.0 + t:.6f} rgb/{i}.png")
        depth_lines.append(f"{1000.0 + depth_t:.6f} depth/{i}.png")
    (seq / "rgb.txt").write_text("\n".join(rgb_lines) + "\n")
    (seq / "depth.txt").write_text("\n".join(depth_lines) + "\n")
    gt = []
    for k in range(100):
        t = -0.05 + 0.01 * k
        T = camera_pose(t)
        gt.append(f"{1000.0 + t:.6f} " + " ".join(f"{x:.9f}" for x in (*T[:3, 3], *quaternion(T[:3, :3]))))
    (seq / "groundtruth.txt").write_text("\n".join(gt) + "\n")


def smooth_field(seed: int) -> np.ndarray:
    # Model error is spatially smooth; per-pixel noise would be biased low by the z-buffer over overlapping splats.
    rng = np.random.default_rng(seed)
    v, u = np.indices((H, W), dtype=np.float64)
    waves = [np.sin(rng.uniform(-0.3, 0.3) * u + rng.uniform(-0.3, 0.3) * v + rng.uniform(0, 2 * np.pi))
             for _ in range(4)]
    return np.sum(waves, axis=0) / 2


def write_predictions(run_dir: Path) -> None:
    (run_dir / NAME).mkdir(parents=True)
    for i in range(len(SLOTS)):
        with Image.open(run_dir.parents[1] / "data" / NAME / "depth" / f"{i}.png") as image:
            depth = np.array(image).astype(np.float32) / 5000
        points, _ = backproject(depth, depth > 0, K)
        noise = 1 + 0.03 * smooth_field(i)[..., None]
        points = 0.5 * points * noise + np.array([0.1, -0.2, 0.3])
        np.savez_compressed(run_dir / NAME / f"{i:05d}.npz", points=points.astype(np.float32),
                            rgb_file=np.str_(f"rgb/{i}.png"), latency_ms=np.float64(1.0), peak_mem_gb=np.float64(1.0))


def write_per_frame(path: Path, data_root: Path, config: Path, run_dir: Path) -> None:
    sequence = load_sequence(data_root / NAME, config)
    tasks = build_tasks(NAME, sequence.root, sequence.K, apply_window(sequence), run_dir)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(evaluate_task(t) for t in tasks)


@pytest.fixture(scope="module")
def setup(tmp_path_factory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("oracle")
    data_root, run_dir = root / "data", root / "predictions" / "test"
    config = root / "sequences.yaml"
    config.write_text(f"cameras:\n  cam: {{fx: {FX}, fy: {FY}, cx: {CX}, cy: {CY}}}\nsequences:\n"
                      f"  {NAME}:\n    camera: cam\n    window: {{start: 0, count: {len(SLOTS)}}}\n")
    write_sequence(data_root)
    write_predictions(run_dir)
    window = {"start": 0, "count": len(SLOTS)}
    (run_dir / "run.json").write_text(json.dumps({"name": "test", "sequences": {NAME: {"window": window}}}))
    targets = root / "targets.yaml"
    targets.write_text(f"targets:\n  {NAME}: {TARGETS}\n")
    per_frame = root / "per_frame.csv"
    write_per_frame(per_frame, data_root, config, run_dir)
    cache = root / "cache"
    fusion_prepare.run_prepare(fusion_prepare.parse_args(
        ["--predictions", str(run_dir), "--data-root", str(data_root), "--sequences-config", str(config),
         "--out", str(cache)]))
    return {"root": root, "data_root": data_root, "run_dir": run_dir, "config": config, "targets": targets,
            "per_frame": per_frame, "cache": cache}


def eval_args(setup: dict[str, Path], out: Path, per_frame: Path | None = None, *extra: str):
    return fusion_eval.parse_args(
        ["--cache", str(setup["cache"]), "--data-root", str(setup["data_root"]),
         "--sequences-config", str(setup["config"]), "--baseline-csv", str(per_frame or setup["per_frame"]),
         "--targets", str(setup["targets"]), "--out", str(out), *extra])


@pytest.fixture(scope="module")
def evaluated(setup, tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("eval")
    fusion_eval.run_fusion_eval(eval_args(setup, out))
    return out


def read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def mean_rel_d(rows: list[dict[str, str]], config: str, column: str = "rel_d") -> float:
    return float(np.mean([float(r[column]) for r in rows if r["config"] == config]))


def stub_aligner(calls: list[int], fail_at: int | None = None):
    def align(frame: Frame, sequence, run_dir) -> tuple[FrameMetrics, str]:
        calls.append(frame.index)
        if fail_at is not None and frame.index == fail_at:
            raise RuntimeError("alignment failed")
        depth, _ = frame.load_depth()
        metrics = FrameMetrics(1.0, 99.0, 2.0, 98.0, depth.size, False, np.zeros((H, W, 3), np.float32), depth)
        return metrics, f"rgb/{frame.index}.png"

    return align


def prepare_args(setup: dict[str, Path], out: Path):
    return fusion_prepare.parse_args(["--predictions", str(setup["run_dir"]), "--data-root", str(setup["data_root"]),
                                      "--sequences-config", str(setup["config"]), "--out", str(out)])


def test_prepare_skips_existing_files(setup, tmp_path: Path) -> None:
    existing = cache_path(tmp_path, NAME, 3)
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"left alone")
    calls: list[int] = []
    record = fusion_prepare.run_prepare(prepare_args(setup, tmp_path), stub_aligner(calls))
    assert 3 not in calls and len(calls) == len(SLOTS) - 1
    assert existing.read_bytes() == b"left alone"
    assert record["sequences"][NAME]["written"] == len(SLOTS) - 1 and record["scale"] == SCALE
    calls.clear()
    record = fusion_prepare.run_prepare(prepare_args(setup, tmp_path), stub_aligner(calls))
    assert calls == [] and record["sequences"][NAME]["existing"] == len(SLOTS)
    cached = read_alignment(cache_path(tmp_path, NAME, 0), 0)
    assert cached.aligned_depth.dtype == np.float32 and cached.rel_d == 2.0


def test_prepare_failure_leaves_no_partial_file(setup, tmp_path: Path, monkeypatch) -> None:
    with pytest.raises(RuntimeError, match="alignment failed"):
        fusion_prepare.run_prepare(prepare_args(setup, tmp_path), stub_aligner([], fail_at=4))
    assert cache_path(tmp_path, NAME, 3).is_file() and not cache_path(tmp_path, NAME, 4).exists()

    def broken_savez(f, **arrays) -> None:
        f.write(b"half a file")
        raise OSError("disk full")

    monkeypatch.setattr(oracle.np, "savez", broken_savez)
    with pytest.raises(OSError, match="disk full"):
        fusion_prepare.run_prepare(prepare_args(setup, tmp_path), stub_aligner([]))
    assert not cache_path(tmp_path, NAME, 4).exists()
    assert not list(tmp_path.rglob("*.partial"))


def test_prepare_cache_holds_the_oracle_aligned_depth(setup) -> None:
    rows = {int(r["frame_index"]): float(r["rel_d"]) for r in read_csv(setup["per_frame"])}
    for index in (0, 10, len(SLOTS) - 1):
        cached = read_alignment(cache_path(setup["cache"], NAME, index), index)
        assert cached.aligned_depth.shape == (H, W) and cached.aligned_depth.dtype == np.float32
        assert cached.rel_d == pytest.approx(rows[index], abs=1e-9)
    assert json.loads((setup["cache"] / oracle.PREPARE_RECORD).read_text())["scale"] == SCALE


def test_csv_has_all_columns_and_every_target_and_config(evaluated: Path) -> None:
    rows = read_csv(evaluated / f"oracle_fusion_{NAME}.csv")
    assert tuple(rows[0]) == ROW_COLUMNS
    assert {(int(r["target_index"]), r["config"]) for r in rows} == {(t, c) for t in TARGETS for c in SPECS}
    assert {r["scale"] for r in rows} == {SCALE}
    for r in rows:
        for column in ("rel_d", "delta1_d", "rel_d_near", "rel_d_mid", "rel_d_far", "hole_fraction",
                       "median_n_sources"):
            assert math.isfinite(float(r[column])), (r["config"], column)
    summary = read_csv(evaluated / "oracle_fusion_summary.csv")
    assert {r["config"] for r in summary} == set(SPECS)
    record = json.loads((evaluated / "oracle_fusion.json").read_text())
    assert record["scale"] == SCALE and record["regression"]["passed"] and record["targets"][NAME] == TARGETS


def test_baseline_self_matches_the_stored_baseline(setup, evaluated: Path) -> None:
    baseline = {int(r["frame_index"]): float(r["rel_d"]) for r in read_csv(setup["per_frame"])}
    rows = [r for r in read_csv(evaluated / f"oracle_fusion_{NAME}.csv") if r["config"] == "baseline_self"]
    assert len(rows) == len(TARGETS)
    for r in rows:
        assert abs(float(r["rel_d"]) - baseline[int(r["target_index"])]) <= 1e-6
        assert float(r["hole_fraction"]) == 0.0 and r["sources"] == ""


def test_fusion_helps_and_shuffled_poses_hurt(evaluated: Path) -> None:
    rows = read_csv(evaluated / f"oracle_fusion_{NAME}.csv")
    correct = mean_rel_d(rows, "causal_diverse_w4_rung_d")
    assert mean_rel_d(rows, "causal_diverse_w4") < mean_rel_d(rows, "baseline_self")
    assert mean_rel_d(rows, "shuffled_poses_causal_diverse_w4_rung_d") > correct
    assert mean_rel_d(rows, "pose_noise_t20mm_r0deg_causal_diverse_w4_rung_d") > correct
    assert mean_rel_d(rows, "pose_noise_t0mm_r0deg_causal_diverse_w4_rung_d") == correct


def test_gt_ceiling_error_is_small(evaluated: Path) -> None:
    rows = read_csv(evaluated / f"oracle_fusion_{NAME}.csv")
    noself = mean_rel_d(rows, "gt_ceiling_causal_diverse_w4_noself", "rel_d_covered")
    with_self = mean_rel_d(rows, "gt_ceiling_causal_diverse_w4_self", "rel_d_covered")
    assert noself < 1.0 and with_self <= noself
    assert noself < 0.5 * mean_rel_d(rows, "baseline_self")


def test_rung_d_gt_ceiling_and_edge_split_columns(evaluated: Path) -> None:
    rows = read_csv(evaluated / f"oracle_fusion_{NAME}.csv")
    assert {"rel_d_covered_interior", "rel_d_covered_edge"} <= set(rows[0])
    ceilings = [c for c in SPECS if c.startswith("gt_ceiling_")]
    assert "gt_ceiling_causal_diverse_w4_rung_d_noself" in ceilings and len(ceilings) == 3
    for r in rows:
        split = (r["rel_d_covered_interior"], r["rel_d_covered_edge"])
        if r["config"] in ceilings:
            assert all(math.isfinite(float(v)) for v in split), r["config"]
        else:
            assert split == ("", ""), r["config"]
    rung_a = mean_rel_d(rows, "gt_ceiling_causal_diverse_w4_noself", "rel_d_covered_edge")
    rung_d = mean_rel_d(rows, "gt_ceiling_causal_diverse_w4_rung_d_noself", "rel_d_covered_edge")
    assert rung_d >= rung_a
    summary = read_csv(evaluated / "oracle_fusion_summary.csv")
    assert {"rel_d_covered_interior_median", "rel_d_covered_edge_median"} <= set(summary[0])
    by_config = {r["config"]: r for r in summary}
    assert float(by_config["gt_ceiling_causal_diverse_w4_rung_d_noself"]["rel_d_covered_edge_median"]) >= 0


def test_rerun_reuses_a_complete_csv(setup, evaluated: Path, monkeypatch) -> None:
    def forbidden(job):
        raise AssertionError("complete sequence was evaluated again")

    monkeypatch.setattr(fusion_eval, "evaluate_sequence", forbidden)
    record = fusion_eval.run_fusion_eval(eval_args(setup, evaluated))
    assert record["sequences"][NAME]["reused"]


def perturbed_per_frame(setup: dict[str, Path], path: Path, frame_index: int) -> Path:
    rows = read_csv(setup["per_frame"])
    for r in rows:
        if int(r["frame_index"]) == frame_index:
            r["rel_d"] = repr(float(r["rel_d"]) + 1e-4)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_regression_fails_on_a_perturbed_target_baseline(setup, tmp_path: Path) -> None:
    per_frame = perturbed_per_frame(setup, tmp_path / "per_frame.csv", TARGETS[1])
    with pytest.raises(RegressionError, match=f"baseline_self {NAME} frame {TARGETS[1]}"):
        fusion_eval.run_fusion_eval(eval_args(setup, tmp_path / "out", per_frame, "--configs", "baseline_self"))
    assert not (tmp_path / "out" / f"oracle_fusion_{NAME}.csv").exists()


def test_regression_fails_on_a_perturbed_cached_frame(setup, tmp_path: Path) -> None:
    per_frame = perturbed_per_frame(setup, tmp_path / "per_frame.csv", 0)
    with pytest.raises(RegressionError, match=f"cache {NAME} frame 0 rescored"):
        fusion_eval.run_fusion_eval(eval_args(setup, tmp_path / "out", per_frame))


def test_regression_fails_on_a_reused_csv_with_a_perturbed_baseline(setup, evaluated: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    shutil.copytree(evaluated, out)
    per_frame = perturbed_per_frame(setup, tmp_path / "per_frame.csv", TARGETS[0])
    with pytest.raises(RegressionError, match="baseline_self"):
        fusion_eval.run_fusion_eval(eval_args(setup, out, per_frame))


def test_baseline_self_failures_checks_only_the_anchor_rows() -> None:
    rows = [{"sequence": "s", "target_index": "1", "config": "baseline_self", "rel_d": "5.0"},
            {"sequence": "s", "target_index": "1", "config": "causal_adjacent_w1", "rel_d": "4.0"}]
    assert baseline_self_failures(rows, {("s", 1): 5.0 + 5e-7}) == []
    assert len(baseline_self_failures(rows, {("s", 1): 5.0 + 2e-6})) == 1
    assert len(baseline_self_failures(rows, {})) == 1


def window_frames(start: int, slots: list[int]) -> list[Frame]:
    return [Frame(start + k, 1000.0 + s / 30, Path("rgb.png"), Path("depth.png"), 0.0, np.eye(4))
            for k, s in enumerate(slots)]


def test_default_targets_are_kept_frame_indices_in_the_middle_of_the_window() -> None:
    slots = [s for s in range(310) if s not in (50, 150)]
    frames = window_frames(37, slots)[:300]
    targets = default_targets(frames)
    assert targets == list(range(137, 237))
    assert all(frames[0].index <= t <= frames[-1].index for t in targets)
    assert default_targets(frames, all_targets=True) == [f.index for f in frames]
    with pytest.raises(ValueError, match="too short"):
        default_targets(frames[:150])
    with pytest.raises(ValueError, match="not kept frames"):
        check_targets("s", frames, [frames[-1].index + 1])


def test_sources_are_chosen_by_time_across_holes() -> None:
    # Slots 7 and 4 are missing, so the frame 100 ms before the target (slot 7) has no stand-in.
    slots = [s for s in range(20) if s not in (4, 7)]
    frames = window_frames(0, slots)
    target = slots.index(10)
    adjacent = plan_sources(frames, target, SPECS["causal_adjacent_w2"].fusion)
    assert [slots[p] for p in adjacent] == [8, 9]
    diverse = plan_sources(frames, target, SPECS["causal_diverse_w3"].fusion)
    assert [slots[p] for p in diverse] == [1]
    centred = plan_sources(frames, target, SPECS["centred_diverse_w2"].fusion)
    assert [slots[p] for p in centred] == [13, 16]


def test_config_registry_names_and_settings() -> None:
    assert list(SPECS)[0] == "baseline_self" and SPECS["baseline_self"].fusion.window == 0
    assert len(SPECS) == 36
    rung_f = SPECS["centred_adjacent_w4_rung_f"].fusion
    assert (rung_f.mode, rung_f.selection, rung_f.window, rung_f.aggregator) == ("centred", "adjacent", 4,
                                                                                 "weighted_median")
    assert rung_f.front_layer and rung_f.min_support == 2 and rung_f.ray_angle_weight and rung_f.baseline_weight
    assert SPECS["causal_diverse_w4_rung_c"].fusion.trim_fraction == 0.2
    assert SPECS["causal_diverse_w4_rung_a"].key == SPECS["causal_diverse_w4"].key
    assert SPECS["pose_noise_t0mm_r0deg_causal_diverse_w4_rung_d"].key == SPECS["causal_diverse_w4_rung_d"].key
    assert not SPECS["gt_ceiling_causal_diverse_w4_noself"].fusion.include_self
    rung_d_ceiling = SPECS["gt_ceiling_causal_diverse_w4_rung_d_noself"]
    assert rung_d_ceiling.depth == "gt"
    assert rung_d_ceiling.fusion == replace(SPECS["causal_diverse_w4_rung_d"].fusion, include_self=False)
    assert [s.name for s in resolve_specs(["causal_diverse_w1"])] == ["baseline_self", "causal_diverse_w1"]
    with pytest.raises(ValueError, match="unknown"):
        resolve_specs(["nope"])


def test_pose_controls_are_seeded_and_never_keep_the_own_pose() -> None:
    picks = shuffled_positions(0, 15, [3, 6, 9, 12], 24)
    assert picks == shuffled_positions(0, 15, [3, 6, 9, 12], 24)
    assert all(p != q and 0 <= q < 24 for p, q in zip([3, 6, 9, 12], picks))
    T = np.eye(4)
    T[:3, 3] = [1.0, 2.0, 3.0]
    draw = np.random.default_rng(0).standard_normal(6)
    assert np.array_equal(noisy_pose(T, draw, 0.0, 0.0), T)
    small, large = noisy_pose(T, draw, 5.0, 0.0), noisy_pose(T, draw, 10.0, 0.0)
    assert np.allclose(large[:3, 3] - T[:3, 3], 2 * (small[:3, 3] - T[:3, 3]))
    rotated = noisy_pose(T, draw, 0.0, 1.0)
    angle = math.degrees(math.acos((np.trace(rotated[:3, :3]) - 1) / 2))
    assert angle == pytest.approx(np.linalg.norm(draw[3:]), rel=1e-6)


def test_shared_warps_give_the_same_fusion_as_the_plain_engine(setup) -> None:
    from pim.data.tum import read_groundtruth
    from pim.fusion.engine import fuse_frame
    from pim.fusion.types import SourceView, TargetSpec

    sequence = load_sequence(setup["data_root"] / NAME, setup["config"])
    frames = apply_window(sequence)
    ctx = oracle.SequenceContext(NAME, sequence.K, frames, 0, read_groundtruth(sequence.root))
    store = oracle.ViewStore(setup["cache"], NAME, frames)
    target = TARGETS[0]
    fusion = oracle.TargetFusion(ctx, store, target)
    for name in ("causal_diverse_w4", "centred_adjacent_w4", "causal_diverse_w4_rung_f"):
        spec = SPECS[name]
        shared = fusion.fuse(spec, plan_sources(frames, target, spec.fusion))
        views = [SourceView(store.aligned(p), oracle.source_valid(store.aligned(p)), f.T_world_cam, K, f.timestamp)
                 for p, f in enumerate(frames) if p != target]
        own = SourceView(store.aligned(target), oracle.source_valid(store.aligned(target)), frames[target].T_world_cam,
                         K, frames[target].timestamp)
        plain = fuse_frame(views, TargetSpec(frames[target].T_world_cam, K, (H, W)), frames[target].timestamp,
                           spec.fusion, own)
        assert np.array_equal(shared.depth, plain.depth, equal_nan=True)
        assert np.array_equal(shared.n_sources, plain.n_sources)
