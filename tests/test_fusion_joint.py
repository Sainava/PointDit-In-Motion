from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from pim.fusion.engine import fuse_frame
from pim.fusion.joint_scale import JointScaleResult, fit_affine, solve_joint_scale, subsample
from pim.fusion.oracle import shuffled_positions, source_valid
from pim.fusion.types import FusionConfig, SourceView, TargetSpec
from pim.metrics.geometry import backproject
from pim.runs import find_input_files


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_script",
                                                  Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


fusion_joint = load_script("fusion_joint")

H, W = 120, 160
F = 132.0
K = np.array([[F, 0.0, (W - 1) / 2], [0.0, F, (H - 1) / 2], [0.0, 0.0, 1.0]])
N = 5
NOISE = 0.002


def yaw(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def camera_pose(k: int, translate: bool = True) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = yaw(0.01 * k)
    if translate:
        T[:3, 3] = [0.04 * k, 0.01 * k, 0.03 * k]
    return T


def render(T_world_cam: np.ndarray) -> np.ndarray:
    """Camera z of the nearest of a back wall, a floor, a side wall and a box face."""
    v, u = np.indices((H, W), dtype=np.float64)
    d = np.stack([(u - K[0, 2]) / F, (v - K[1, 2]) / F, np.ones_like(u)], axis=-1) @ T_world_cam[:3, :3].T
    c = T_world_cam[:3, 3]
    depth = np.full((H, W), np.inf)
    for axis, value, bounds in ((2, 4.0, ()), (1, 1.0, ()), (0, -1.5, ()),
                                (2, 2.5, ((0, -0.2, 0.6), (1, -0.2, 0.4)))):
        with np.errstate(divide="ignore", invalid="ignore"):
            s = (value - c[axis]) / d[..., axis]
        hit = c + s[..., None] * d
        ok = s > 0
        for b, lo, hi in bounds:
            ok &= (hit[..., b] > lo) & (hit[..., b] < hi)
        depth = np.where(ok & (s < depth), s, depth)
    return depth


def make_scene(translate: bool = True, seed: int = 0):
    rng = np.random.default_rng(seed)
    poses = [camera_pose(k, translate) for k in range(N)]
    scale = 0.8 * rng.uniform(0.9, 1.1, N)
    shift = 0.2 + rng.uniform(-0.05, 0.05, N)
    depths = [render(T) for T in poses]
    raw = [((d * (1 + NOISE * rng.standard_normal(d.shape)) - shift[k]) / scale[k]).astype(np.float32)
           for k, d in enumerate(depths)]
    return poses, raw, depths, scale, shift


def depth_errors(result: JointScaleResult, raw: list[np.ndarray], scale: np.ndarray, shift: np.ndarray) -> np.ndarray:
    """Relative depth error of the solved affine at the 10th, 50th and 90th percentile raw z of every frame."""
    errors = []
    for k, z in enumerate(raw):
        q = np.percentile(z, [10, 50, 90])
        errors.append((result.scale[k] * q + result.shift[k]) / (scale[k] * q + shift[k]) - 1)
    return np.abs(np.array(errors))


@pytest.fixture(scope="module")
def scene():
    return make_scene()


@pytest.fixture(scope="module")
def solved(scene) -> JointScaleResult:
    poses, raw, _, _, _ = scene
    return solve_joint_scale(raw, poses, K)


def test_solver_recovers_per_frame_affines_with_translation(scene, solved) -> None:
    _, raw, _, scale, shift = scene
    assert not solved.low_parallax and solved.per_frame and solved.iterations > 0
    # Scale and shift trade off over the scene's narrow depth range, so the depth map is held tighter than s alone.
    assert np.max(np.abs(solved.scale / scale - 1)) < 0.08
    assert np.max(depth_errors(solved, raw, scale, shift)) < 0.035
    assert solved.valid_fraction > 0.9


def test_rotation_only_window_is_low_parallax_and_keeps_the_shared_affine() -> None:
    poses, raw, _, _, _ = make_scene(translate=False)
    result = solve_joint_scale(raw, poses, K)
    assert result.low_parallax and not result.per_frame and result.iterations == 0
    assert result.baseline_ratio == 0.0
    np.testing.assert_allclose(result.scale, result.shared_scale, rtol=1e-12)
    np.testing.assert_allclose(result.shift, result.shared_shift, rtol=1e-12)


def test_solver_is_invariant_to_frame_order(scene, solved) -> None:
    poses, raw, _, _, _ = scene
    order = [3, 0, 4, 2, 1]
    permuted = solve_joint_scale([raw[i] for i in order], [poses[i] for i in order], K)
    np.testing.assert_allclose(permuted.scale, solved.scale[order], rtol=1e-4)
    np.testing.assert_allclose(permuted.shift, solved.shift[order], rtol=1e-4, atol=1e-6)
    assert permuted.low_parallax == solved.low_parallax


def test_solver_fails_clearly_on_fewer_than_two_frames(scene) -> None:
    poses, raw, _, _, _ = scene
    with pytest.raises(ValueError, match="at least two frames"):
        solve_joint_scale(raw[:1], poses[:1], K)
    with pytest.raises(ValueError, match="at least two frames"):
        solve_joint_scale([], [], K)


def test_subsample_is_a_regular_grid_over_finite_pixels() -> None:
    z = np.ones((480, 640), dtype=np.float32)
    z[:, :100] = np.nan
    v, u = subsample(z, 1500)
    assert 1200 < v.size < 1800
    assert np.all(np.isfinite(z[v, u]))
    assert len(np.unique(np.diff(np.unique(u)))) == 1


def gt_inputs(depth: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gt_depth = depth.astype(np.float32)
    gt_points, gt_valid = backproject(gt_depth, np.isfinite(gt_depth), K)
    return gt_depth, gt_valid, gt_points


def test_no_fusion_control_equals_the_baseline_after_alignment(scene, solved) -> None:
    _, raw, depths, _, _ = scene
    gt_depth, gt_valid, gt_points = gt_inputs(depths[0])
    baseline = fusion_joint.score_depth(raw[0], gt_depth, gt_valid, gt_points)
    control = fusion_joint.score_depth(solved.depth(0, raw[0]), gt_depth, gt_valid, gt_points)
    assert math.isfinite(baseline["rel_d"])
    for key in baseline:
        assert abs(control[key] - baseline[key]) <= fusion_joint.BASELINE_SELF_TOLERANCE, key


def fused_error(result: JointScaleResult, raw: list[np.ndarray], poses: list[np.ndarray], depths: list[np.ndarray]
                ) -> float:
    views = []
    for k in range(N):
        d = result.depth(k, raw[k])
        views.append(SourceView(d, source_valid(d), poses[k], K, float(k) / 30))
    target = N - 1
    own = views[target]
    config = FusionConfig(window=N - 1, mode="causal", aggregator="weighted_median")
    fused = fuse_frame(views[:target], TargetSpec(own.T_world_cam, K, (H, W)), own.timestamp, config, own)
    depth = np.where(fused.hole, own.depth, fused.depth)
    gt_depth, gt_valid, gt_points = gt_inputs(depths[target])
    return fusion_joint.score_depth(depth, gt_depth, gt_valid, gt_points)["rel_d"]


def test_shuffled_poses_recover_worse_and_fuse_worse(scene, solved) -> None:
    poses, raw, depths, scale, shift = scene
    target = N - 1
    sources = list(range(target))
    picks = shuffled_positions(0, target, sources, N)
    assert all(p != s for p, s in zip(picks, sources))
    wrong = [poses[p] for p in picks] + [poses[target]]
    shuffled = solve_joint_scale(raw, wrong, K)
    assert np.median(depth_errors(shuffled, raw, scale, shift)) > 2 * np.median(depth_errors(solved, raw, scale, shift))
    assert fused_error(shuffled, raw, wrong, depths) > fused_error(solved, raw, poses, depths) + 0.5


def test_fit_affine_recovers_an_exact_affine() -> None:
    z = np.linspace(0.5, 2.0, 200, dtype=np.float32).reshape(10, 20)
    z[0, 0] = np.nan
    assert fit_affine(z, 1.7 * z + 0.3) == pytest.approx((1.7, 0.3), rel=1e-5)


def test_discovery_ignores_paths_inside_a_project_checkout(tmp_path: Path) -> None:
    root = tmp_path / "input"
    wanted = root / "eval-output" / "eval" / "baseline" / "per_frame.csv"
    bundled = root / "eval-output" / "Pointdit-In-Motion" / "results" / "eval" / "per_frame.csv"
    run = root / "runner-output" / "predictions" / "baseline" / "run.json"
    bundled_run = root / "runner-output" / "Pointdit-In-Motion" / "predictions" / "run.json"
    for path in (wanted, bundled, run, bundled_run):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x")
    assert find_input_files(root, "per_frame.csv") == [wanted]
    assert find_input_files(root, "run.json") == [run]
    assert len(find_input_files(root, "per_frame.csv", ignored=())) == 2
