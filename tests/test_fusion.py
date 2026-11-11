from __future__ import annotations

import time
import warnings
from dataclasses import dataclass

import numpy as np
import pytest

from pim.consistency.warp import warp_depth
from pim.fusion.aggregate import aggregate
from pim.fusion.engine import fuse_frame
from pim.fusion.select import select_sources
from pim.fusion.types import FusionConfig, SourceView, TargetSpec, scale_intrinsics
from pim.fusion.warp import warp_to_target

H, W = 40, 52
K = np.array([[60.0, 0.0, 25.5], [0.0, 60.0, 19.5], [0.0, 0.0, 1.0]])
DT = 1 / 30


@dataclass(frozen=True)
class Scene:
    plane_normal: tuple[float, float, float] = (-0.15, 0.05, 1.0)
    plane_offset: float = 3.0
    box: tuple[tuple[float, float, float], tuple[float, float, float]] | None = (
        (-0.15, -0.2, 1.5), (0.15, 0.2, 1.7))

    def cast(self, T_world_cam: np.ndarray, rays_cam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(depth along camera-frame rays with z = 1, box-hit mask), inf where nothing is hit."""
        R, c = T_world_cam[:3, :3], T_world_cam[:3, 3]
        d = rays_cam @ R.T
        n = np.asarray(self.plane_normal)
        with np.errstate(divide="ignore", invalid="ignore"):
            t_plane = (self.plane_offset - n @ c) / (d @ n)
        depth = np.where(t_plane > 0, t_plane, np.inf)
        box_hit = np.zeros(depth.shape, dtype=bool)
        if self.box is not None:
            lo, hi = (np.asarray(b) for b in self.box)
            with np.errstate(divide="ignore", invalid="ignore"):
                t1, t2 = (lo - c) / d, (hi - c) / d
            t_near = np.minimum(t1, t2).max(axis=-1)
            t_far = np.maximum(t1, t2).min(axis=-1)
            box_hit = (t_far >= t_near) & (t_near > 0) & (t_near < depth)
            depth = np.where(box_hit, t_near, depth)
        return depth, box_hit

    def render(self, T_world_cam: np.ndarray, K_: np.ndarray = K, shape: tuple[int, int] = (H, W)):
        v, u = np.indices(shape, dtype=np.float64)
        rays = np.stack([(u - K_[0, 2]) / K_[0, 0], (v - K_[1, 2]) / K_[1, 1], np.ones(shape)], axis=-1)
        return self.cast(T_world_cam, rays)


def pose(t=(0.0, 0.0, 0.0), yaw_deg: float = 0.0, pitch_deg: float = 0.0) -> np.ndarray:
    a, b = np.radians(yaw_deg), np.radians(pitch_deg)
    yaw = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    pitch = np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]])
    T = np.eye(4)
    T[:3, :3] = yaw @ pitch
    T[:3, 3] = t
    return T


def view(scene: Scene, T: np.ndarray, timestamp: float, depth: np.ndarray | None = None) -> SourceView:
    if depth is None:
        depth = scene.render(T)[0]
    finite = np.isfinite(depth)
    return SourceView(np.where(finite, depth, 0).astype(np.float32), finite, T, K, timestamp)


def target_of(T: np.ndarray, grid_scale: int = 1) -> TargetSpec:
    return TargetSpec(T, K, (H, W), grid_scale)


def relative_error(fused: np.ndarray, truth: np.ndarray) -> np.ndarray:
    scored = np.isfinite(fused) & np.isfinite(truth)
    return np.abs(fused[scored] - truth[scored]) / truth[scored]


def dilate(mask: np.ndarray, steps: int) -> np.ndarray:
    for _ in range(steps):
        padded = np.pad(mask, 1, constant_values=False)
        mask = padded[1:-1, 1:-1] | padded[:-2, 1:-1] | padded[2:, 1:-1] | padded[1:-1, :-2] | padded[1:-1, 2:]
    return mask


def erode(mask: np.ndarray, steps: int) -> np.ndarray:
    for _ in range(steps):
        padded = np.pad(mask, 1, constant_values=False)
        mask = padded[1:-1, 1:-1] & padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:]
    return mask


SOURCE_POSES = [pose((-0.12, 0.02, 0.0), 2.0), pose((-0.06, -0.01, 0.02), 1.0),
                pose((0.06, 0.01, -0.02), -1.0), pose((0.12, -0.02, 0.0), -2.0, 1.0)]
TARGET_POSE = pose((0.0, 0.0, 0.0), 0.0, 0.5)


def held_out_frames(scene: Scene, poses=SOURCE_POSES) -> list[SourceView]:
    times = [0.0, DT, 3 * DT, 4 * DT]
    return [view(scene, T, t) for T, t in zip(poses, times)]


def test_fusing_sources_reproduces_held_out_pose():
    scene = Scene()
    result = fuse_frame(held_out_frames(scene), target_of(TARGET_POSE), 2 * DT,
                        FusionConfig(window=2, include_self=False))
    truth = scene.render(TARGET_POSE)[0]
    assert result.source_indices == (0, 1, 2, 3)
    assert (~result.hole).mean() > 0.9
    assert np.median(relative_error(result.depth, truth)) < 0.005


def test_window_zero_with_self_returns_own_depth_bitwise():
    scene = Scene()
    own = view(scene, TARGET_POSE, 2 * DT)
    own = SourceView(own.depth, own.valid & (np.arange(W)[None] % 7 != 0), own.T_world_cam, K, own.timestamp)
    for weight in (1.0, 0.37):
        result = fuse_frame(held_out_frames(scene), target_of(TARGET_POSE), 2 * DT,
                            FusionConfig(window=0, self_weight=weight), self_view=own)
        assert np.array_equal(result.depth[own.valid], own.depth[own.valid])
        assert np.all(result.hole[~own.valid])


def test_window_zero_without_self_is_all_holes():
    scene = Scene()
    result = fuse_frame(held_out_frames(scene), target_of(TARGET_POSE), 2 * DT,
                        FusionConfig(window=0, include_self=False))
    assert result.depth.shape == (H, W)
    assert np.all(result.hole) and np.all(np.isnan(result.depth))


def test_include_self_without_self_view_fails():
    with pytest.raises(ValueError, match="self_view"):
        fuse_frame([], target_of(TARGET_POSE), 0.0, FusionConfig())


def test_identity_copies_return_the_view():
    scene = Scene(box=None)
    copies = [view(scene, TARGET_POSE, t) for t in (0.0, DT, 3 * DT, 4 * DT)]
    truth = copies[0].depth
    for include_self in (False, True):
        result = fuse_frame(copies, target_of(TARGET_POSE), 2 * DT,
                            FusionConfig(window=2, include_self=include_self), self_view=copies[0])
        assert not result.hole.any()
        # The 2x2 splat with a nearest-depth z-buffer may pick a neighbour one pixel away.
        gradient = (np.abs(np.diff(truth, axis=0)).max() + np.abs(np.diff(truth, axis=1)).max()) / truth.min()
        assert relative_error(result.depth, truth).max() <= gradient + 1e-6
        assert np.median(relative_error(result.depth, truth)) < 0.005


def test_occluded_pixels_get_no_weight_from_the_occluded_source():
    scene = Scene()
    T_a, T_b = pose((0.6, 0.0, 0.0)), pose((-0.6, 0.0, 0.0))
    frames = [view(scene, T_a, 0.0), view(scene, T_b, 2 * DT)]
    target = target_of(TARGET_POSE)
    result = fuse_frame(frames, target, DT, FusionConfig(window=1, include_self=False, keep_layers=True))
    only_b = fuse_frame(frames[1:], target, DT, FusionConfig(window=1, include_self=False))

    truth, target_box = scene.render(TARGET_POSE)
    # The splat may spread a visible neighbour one pixel into the hidden region and the box one pixel
    # out of its silhouette, so the check skips those borders.
    hidden = erode(_hidden_from(scene, TARGET_POSE, truth, T_a), 1) & ~dilate(target_box, 2)
    assert hidden.sum() > 40
    assert np.all(result.layer_weights[0][hidden] == 0)
    assert np.array_equal(result.depth[hidden], only_b.depth[hidden])
    assert np.all(relative_error(result.depth[hidden], truth[hidden]) < 0.02)


def _into_source(T_target: np.ndarray, truth: np.ndarray, T_source: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(target pixels' surface points in the source camera frame, mask of those inside the source image)."""
    v, u = np.indices(truth.shape, dtype=np.float64)
    rays = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones(truth.shape)], axis=-1)
    world = (rays * np.where(np.isfinite(truth), truth, 1.0)[..., None]) @ T_target[:3, :3].T + T_target[:3, 3]
    points = (world - T_source[:3, 3]) @ T_source[:3, :3]
    uv = points[..., :2] / points[..., 2:] * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]
    inside = (uv[..., 0] >= 0) & (uv[..., 0] <= W - 1) & (uv[..., 1] >= 0) & (uv[..., 1] <= H - 1)
    return points, np.isfinite(truth) & (points[..., 2] > 0) & inside


def _hidden_from(scene: Scene, T_target: np.ndarray, truth: np.ndarray, T_source: np.ndarray) -> np.ndarray:
    points, inside = _into_source(T_target, truth, T_source)
    seen_depth = scene.cast(T_source, points / points[..., 2:])[0]
    return inside & (seen_depth < points[..., 2] * 0.98)


def _stack(*candidates: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    depths = np.array(candidates, dtype=np.float32).reshape(-1, 1, 1)
    hits = np.isfinite(depths)
    return depths, hits.astype(np.float32), hits


def test_front_layer_keeps_the_front_candidate():
    stack = _stack(2.0, 5.0)
    assert aggregate(*stack, FusionConfig())[0][0, 0] == pytest.approx(3.5)
    fused, total, count = aggregate(*stack, FusionConfig(front_layer=True))
    assert fused[0, 0] == pytest.approx(2.0) and count[0, 0] == 1 and total[0, 0] == 1


def test_median_resists_a_gross_outlier_where_the_mean_does_not():
    stack = _stack(1.99, 2.0, 2.01, 9.0)
    assert aggregate(*stack, FusionConfig(aggregator="weighted_median"))[0][0, 0] == pytest.approx(2.005, abs=1e-6)
    assert aggregate(*stack, FusionConfig())[0][0, 0] > 3.5
    assert aggregate(*stack, FusionConfig(aggregator="trimmed_mean", trim_fraction=0.25))[0][0, 0] == pytest.approx(
        2.005, abs=1e-6)


def test_weighted_median_follows_weights():
    depths, _, hits = _stack(1.0, 2.0, 3.0)
    weights = np.array([1.0, 1.0, 5.0], dtype=np.float32).reshape(-1, 1, 1)
    assert aggregate(depths, weights, hits, FusionConfig(aggregator="weighted_median"))[0][0, 0] == 3.0


def test_consistency_filter_drops_unsupported_candidates():
    stack = _stack(2.0, 2.02, 3.0)
    fused, _, count = aggregate(*stack, FusionConfig(min_support=2))
    assert fused[0, 0] == pytest.approx(2.01) and count[0, 0] == 2
    assert aggregate(*stack, FusionConfig(min_support=1))[2][0, 0] == 3
    lone = aggregate(*_stack(3.0), FusionConfig(min_support=2))
    assert lone[0][0, 0] == 3.0 and lone[2][0, 0] == 1


@pytest.mark.parametrize("aggregator", ["weighted_mean", "weighted_median", "trimmed_mean"])
def test_aggregate_handles_single_layer_and_empty_pixels_without_warnings(aggregator):
    depths = np.array([[[np.nan, 2.0], [0.0, 3.0]]], dtype=np.float32)
    hits = np.array([[[False, True], [True, True]]])
    config = FusionConfig(aggregator=aggregator, front_layer=True, min_support=2)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fused, total, count = aggregate(depths, hits.astype(np.float32), hits, config)
        empty = aggregate(np.full((3, 2, 2), np.nan, np.float32), np.zeros((3, 2, 2), np.float32),
                          np.zeros((3, 2, 2), bool), config)
        none = aggregate(np.zeros((0, 2, 2), np.float32), np.zeros((0, 2, 2), np.float32),
                         np.zeros((0, 2, 2), bool), config)
    assert np.array_equal(fused, np.array([[np.nan, 2.0], [np.nan, 3.0]], np.float32), equal_nan=True)
    assert np.array_equal(count, [[0, 1], [0, 1]])
    for result in (empty, none):
        assert np.all(np.isnan(result[0])) and np.all(result[2] == 0)


def _crack_fraction(hit: np.ndarray, scene: Scene, T_source: np.ndarray, T_target: np.ndarray) -> float:
    expected = _into_source(T_target, scene.render(T_target)[0], T_source)[1]
    return float((expected & ~hit).sum() / expected.sum())


def test_splat_closes_forward_warp_cracks():
    scene = Scene(plane_normal=(0.0, 0.0, 1.0), plane_offset=2.0, box=None)
    T_source = pose()
    T_target = pose((0.05, 0.02, 0.1), 2.0, 1.0)
    source = view(scene, T_source, 0.0)
    layer = warp_to_target(source, target_of(T_target))
    assert _crack_fraction(layer.hit, scene, T_source, T_target) < 0.01
    _, nearest_hit = warp_depth(source.depth, source.valid, K, np.linalg.inv(T_target) @ T_source, (H, W))
    assert _crack_fraction(nearest_hit, scene, T_source, T_target) > 0.02


def test_scale_intrinsics_uses_pixel_centres():
    scaled = scale_intrinsics(K, 2)
    assert scaled[0, 0] == 120 and scaled[1, 1] == 120
    assert scaled[0, 2] == (25.5 + 0.5) * 2 - 0.5 and scaled[1, 2] == (19.5 + 0.5) * 2 - 0.5
    assert np.array_equal(scale_intrinsics(K, 1), K)
    # The left edge of native pixel 0 (at -0.5) maps to the left edge of fine pixel 0.
    for s in (2, 3):
        Ks = scale_intrinsics(K, s)
        x = (-0.5 - K[0, 2]) / K[0, 0]
        assert Ks[0, 0] * x + Ks[0, 2] == pytest.approx(-0.5)


@pytest.mark.parametrize("include_self", [False, True])
def test_grid_scale_two_on_constant_plane(include_self):
    scene = Scene(plane_normal=(0.0, 0.0, 1.0), plane_offset=2.5, box=None)
    frames = [view(scene, pose(), 0.0), view(scene, pose(), 2 * DT)]
    result = fuse_frame(frames, target_of(pose(), grid_scale=2), DT,
                        FusionConfig(window=1, include_self=include_self), self_view=frames[0])
    assert result.depth.shape == (2 * H, 2 * W)
    assert np.all(result.depth == np.float32(2.5))


@pytest.mark.parametrize("n", [2, 4])
@pytest.mark.parametrize("rho", [0.0, 0.5, 0.9])
def test_weighted_mean_noise_reduction_matches_theory(n, rho):
    rng = np.random.default_rng(int(10 * rho) + n)
    shape, sigma, depth = (60, 80), 0.02, 2.0
    shared = rng.standard_normal(shape)
    noise = np.sqrt(rho) * shared + np.sqrt(1 - rho) * rng.standard_normal((n, *shape))
    depths = (depth + sigma * noise).astype(np.float32)
    ones = np.ones(depths.shape, dtype=np.float32)
    fused = aggregate(depths, ones, ones.astype(bool), FusionConfig())[0]
    ratio = np.std(fused - depth) / np.std(depths[0] - depth)
    assert ratio == pytest.approx(np.sqrt((1 + (n - 1) * rho) / n), rel=0.1)


def test_wrong_poses_hurt():
    scene = Scene()
    truth = scene.render(TARGET_POSE)[0]
    frames = held_out_frames(scene)
    config = FusionConfig(window=2, include_self=False)
    correct = relative_error(fuse_frame(frames, target_of(TARGET_POSE), 2 * DT, config).depth, truth)

    rng = np.random.default_rng(0)
    perturbed = [SourceView(f.depth, f.valid, f.T_world_cam @ pose(rng.normal(0, 0.03, 3), *rng.normal(0, 2, 2)),
                            K, f.timestamp) for f in frames]
    shuffled = [SourceView(f.depth, f.valid, frames[i].T_world_cam, K, f.timestamp)
                for f, i in zip(frames, (3, 2, 1, 0))]
    for wrong in (perturbed, shuffled):
        error = relative_error(fuse_frame(wrong, target_of(TARGET_POSE), 2 * DT, config).depth, truth)
        assert error.mean() > 2.5 * correct.mean()
        assert np.percentile(error, 90) > 5 * np.percentile(correct, 90)


TIMES = np.array([0, 1, 2, 4, 5, 6, 9, 10, 12, 13, 14, 15], dtype=np.float64) * DT


def test_causal_selection_never_returns_a_later_frame():
    target = TIMES[6]
    picks = select_sources(TIMES, target, FusionConfig(window=3, mode="causal"))
    assert picks == [3, 4, 5]
    assert select_sources(TIMES, 7 * DT, FusionConfig(window=3, mode="causal")) == [3, 4, 5]
    for selection in ("adjacent", "diverse"):
        for t in TIMES:
            chosen = select_sources(TIMES, t, FusionConfig(window=4, mode="causal", selection=selection))
            assert all(TIMES[i] < t for i in chosen)


def test_centred_selection_takes_both_sides_and_excludes_the_target():
    picks = select_sources(TIMES, TIMES[6], FusionConfig(window=2))
    assert picks == [4, 5, 7, 8]
    assert 6 not in select_sources(TIMES, TIMES[6], FusionConfig(window=20))
    assert select_sources(TIMES, TIMES[0], FusionConfig(window=2)) == [1, 2]


def test_diverse_selection_follows_strides_within_tolerance():
    config = FusionConfig(window=4, selection="diverse", stride_ms=100, stride_tolerance_ms=20)
    target = TIMES[6]
    picks = select_sources(TIMES, target, config)
    for i in picks:
        offset_ms = abs(TIMES[i] - target) * 1000
        assert abs(offset_ms - 100 * round(offset_ms / 100)) <= 20
    # 300 ms target. Strides 100 and 300 ms before hit frames 5 and 0 exactly, 200 ms before falls 33 ms
    # from frames 2 and 3, and 100 and 200 ms after hit frames 8 and 11.
    assert picks == [0, 5, 8, 11]
    # 333 ms target. Exact hits are frames 1, 3 and 9, and frames 5 and 11 lie 33 ms off a stride.
    assert select_sources(TIMES, TIMES[7], config) == [1, 3, 9]
    wider = FusionConfig(window=4, selection="diverse", stride_ms=100, stride_tolerance_ms=40)
    assert select_sources(TIMES, TIMES[7], wider) == [1, 3, 5, 9, 11]


def test_window_zero_selects_nothing():
    for mode in ("causal", "centred"):
        for selection in ("adjacent", "diverse"):
            assert select_sources(TIMES, TIMES[5], FusionConfig(window=0, mode=mode, selection=selection)) == []


def test_selection_rejects_unsorted_timestamps():
    with pytest.raises(ValueError, match="increasing"):
        select_sources([0.0, 0.2, 0.1], 0.1, FusionConfig(window=1))


def test_warp_and_aggregate_are_vectorised():
    shape = (200, 300)
    K_big = np.array([[250.0, 0, 149.5], [0, 250.0, 99.5], [0, 0, 1]])
    scene = Scene()
    depth = scene.render(pose(), K_big, shape)[0]
    source = SourceView(depth.astype(np.float32), np.isfinite(depth), pose(), K_big, 0.0)
    target = TargetSpec(pose((0.05, 0.0, 0.0), 1.0), K_big, shape)
    start = time.perf_counter()
    layers = [warp_to_target(source, target) for _ in range(3)]
    stack = np.stack([l.depth for l in layers]), np.stack([l.hit for l in layers]).astype(np.float32)
    for aggregator in ("weighted_mean", "weighted_median", "trimmed_mean"):
        aggregate(stack[0], stack[1], stack[1] > 0, FusionConfig(aggregator=aggregator, front_layer=True,
                                                                 min_support=2))
    assert time.perf_counter() - start < 3.0
