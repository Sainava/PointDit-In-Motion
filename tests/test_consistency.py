from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pim.consistency.metrics import FrameInputs, frame_inputs, pair_consistency, visibility
from pim.consistency.warp import depth_pose, pairs_by_gap, relative_pose, warp_depth, warp_image
from pim.data.tum import DEPTH_SCALE, Frame, TUMSequence, Window
from pim.metrics.geometry import backproject

H, W = 120, 160
K = np.array([[140.0, 0.0, 79.5], [0.0, 140.0, 59.5], [0.0, 0.0, 1.0]])
PLANE_Z = 4.0
BOX = (np.array([-0.5, -0.4, 2.2]), np.array([0.3, 0.4, 2.8]))
OFFSCREEN_BOX = (np.array([1.4, -0.3, 1.5]), np.array([1.8, 0.3, 1.8]))


def pose(yaw_deg: float = 0.0, translation: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> np.ndarray:
    a = math.radians(yaw_deg)
    T = np.eye(4)
    T[:3, :3] = [[math.cos(a), 0.0, math.sin(a)], [0.0, 1.0, 0.0], [-math.sin(a), 0.0, math.cos(a)]]
    T[:3, 3] = translation
    return T


def render(T_world_cam: np.ndarray, boxes: tuple[tuple[np.ndarray, np.ndarray], ...] = (BOX,)) -> np.ndarray:
    """Camera-z depth of a fronto-parallel plane at world z = PLANE_Z plus axis-aligned boxes, by ray casting."""
    v, u = np.indices((H, W), dtype=np.float64)
    rays_cam = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], axis=-1)
    rays = rays_cam @ T_world_cam[:3, :3].T
    origin = T_world_cam[:3, 3]
    # Rays have unit camera z, so the ray parameter t is the camera-frame depth.
    with np.errstate(divide="ignore", invalid="ignore"):
        depth = (PLANE_Z - origin[2]) / rays[..., 2]
        for lo, hi in boxes:
            t0, t1 = (lo - origin) / rays, (hi - origin) / rays
            t_near = np.minimum(t0, t1).max(axis=-1)
            t_far = np.maximum(t0, t1).min(axis=-1)
            hit = (t_near <= t_far) & (t_near > 0)
            depth = np.where(hit & (t_near < depth), t_near, depth)
    return depth.astype(np.float32)


def gt_frame(T_world_cam: np.ndarray, **kwargs) -> FrameInputs:
    depth = render(T_world_cam, **kwargs)
    return FrameInputs(depth, depth > 0)


def frame(index: int, timestamp: float) -> Frame:
    return Frame(index, timestamp, Path(f"rgb/{index}.png"), Path(f"depth/{index}.png"), 0.0, np.eye(4))


T_I = pose()
T_J = pose(yaw_deg=4.0, translation=(0.25, 0.05, 0.1))


def test_relative_pose_round_trip() -> None:
    T_ij = relative_pose(T_I, T_J)
    np.testing.assert_allclose(T_J @ T_ij, T_I, atol=1e-12)
    np.testing.assert_allclose(relative_pose(T_J, T_I) @ T_ij, np.eye(4), atol=1e-12)
    point_i = np.array([0.3, -0.2, 2.5, 1.0])
    np.testing.assert_allclose(np.linalg.inv(T_J) @ (T_I @ point_i), T_ij @ point_i, atol=1e-12)


def test_warped_depth_matches_rendered_depth_of_the_other_camera() -> None:
    depth_i, depth_j = render(T_I), render(T_J)

    warped, hit = warp_depth(depth_i, depth_i > 0, K, relative_pose(T_I, T_J), (H, W))

    vis = visibility(FrameInputs(depth_i, depth_i > 0), FrameInputs(depth_j, depth_j > 0), K, relative_pose(T_I, T_J))
    assert vis.visible.mean() > 0.7
    rel = np.abs(warped[vis.visible] - depth_j[vis.visible]) / depth_j[vis.visible]
    assert np.median(rel) < 0.005
    assert hit.mean() > 0.7


def test_self_pair_warp_is_identity() -> None:
    depth = render(T_I)
    valid = depth > 0
    valid[10:20, 30:50] = False
    rgb = np.random.default_rng(0).integers(0, 255, (H, W, 3), dtype=np.uint8)

    warped, hit = warp_depth(depth, valid, K, np.eye(4), (H, W))
    warped_rgb, rgb_hit = warp_image(rgb, depth, valid, K, np.eye(4))

    assert np.array_equal(hit, valid) and np.array_equal(rgb_hit, valid)
    assert np.array_equal(warped[valid], depth[valid])
    assert np.array_equal(warped_rgb[valid], rgb[valid])


@pytest.mark.parametrize("near_first", [True, False])
def test_z_buffer_keeps_the_nearest_point(near_first: bool) -> None:
    K_line = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    depth = np.zeros((1, 4), dtype=np.float32)
    # A pixel moves by fx * t / Z under an x-translation t, so the near point moves twice as far.
    if near_first:
        depth[0, 0], depth[0, 1], shift, target = 1.0, 2.0, 2.0, 2
    else:
        depth[0, 2], depth[0, 3], shift, target = 2.0, 1.0, -2.0, 1
    T = np.eye(4)
    T[0, 3] = shift

    warped, hit = warp_depth(depth, depth > 0, K_line, T, (1, 4))

    assert hit.sum() == 1 and hit[0, target]
    assert warped[0, target] == 1.0


def test_points_behind_camera_or_out_of_bounds_are_dropped() -> None:
    depth = render(T_I)
    behind = pose(yaw_deg=180.0)
    _, hit = warp_depth(depth, depth > 0, K, relative_pose(T_I, behind), (H, W))
    assert not hit.any()


def test_box_occluding_plane_in_j_is_not_visible() -> None:
    T_j = pose(translation=(1.0, 0.0, 0.0))
    frame_i = gt_frame(T_I, boxes=(OFFSCREEN_BOX,))
    frame_j = gt_frame(T_j, boxes=(OFFSCREEN_BOX,))
    plane_only_j = render(T_j, boxes=())
    assert np.allclose(frame_i.gt_depth, render(T_I, boxes=())), "box must be outside camera i's view"
    box_in_j = frame_j.gt_depth < plane_only_j - 1e-3

    vis = visibility(frame_i, frame_j, K, relative_pose(T_I, T_j))
    _, hit = warp_depth(frame_i.gt_depth, frame_i.gt_valid, K, relative_pose(T_I, T_j), (H, W))

    occluded = box_in_j & hit
    assert occluded.sum() > 100
    assert not vis.visible[occluded].any()
    assert vis.visible[hit & ~box_in_j].mean() > 0.99


def test_interior_excludes_edge_band() -> None:
    vis = visibility(gt_frame(T_I), gt_frame(T_J), K, relative_pose(T_I, T_J))
    assert vis.interior.sum() < vis.visible.sum()
    assert not (vis.interior & ~vis.visible).any()


def test_pairs_by_gap_uses_time_and_skips_holes() -> None:
    frames = [frame(0, 0.000), frame(1, 0.033), frame(2, 0.066), frame(3, 0.166), frame(4, 0.199)]

    pairs = pairs_by_gap(frames, gap_ms=33.0, tolerance_ms=10.0)

    assert [(i.index, j.index) for i, j in pairs] == [(0, 1), (1, 2), (3, 4)]
    assert [(i.index, j.index) for i, j in pairs_by_gap(frames, gap_ms=100.0)] == [(2, 3)]
    assert [(i.index, j.index) for i, j in pairs_by_gap(frames, gap_ms=0.0)] == [(f.index, f.index) for f in frames]


def test_pairs_by_gap_rejects_unsorted_frames() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        pairs_by_gap([frame(0, 0.1), frame(1, 0.0)], gap_ms=33.0)


def predicted_inputs(depth: np.ndarray, raw_depth: np.ndarray) -> FrameInputs:
    gt_points, valid = backproject(depth, depth > 0, K)
    pred_points = gt_points.copy()
    pred_points[..., 2] = raw_depth
    return frame_inputs(depth, valid, K, pred_points)


def test_affine_prediction_has_zero_m1_and_m2() -> None:
    depth_i, depth_j = render(T_I), render(T_J)
    frame_i = predicted_inputs(depth_i, 0.5 * depth_i + 0.2)
    frame_j = predicted_inputs(depth_j, 3.0 * depth_j - 1.0)

    result = pair_consistency(frame_i, frame_j, K, relative_pose(T_I, T_J))

    assert not result.too_few
    floor = result.visible.gt_floor.overall.median
    assert floor < 0.5
    assert abs(result.visible.m1.overall.median - floor) < 0.1
    assert result.visible.m2.overall.median < 0.5
    raw_error = np.median(np.abs(frame_j.raw_depth - depth_j) / depth_j)
    assert raw_error > 0.5, "the test needs frame j's raw prediction far from GT"


def test_m2_is_invariant_to_frame_j_affine_but_not_to_region_errors() -> None:
    depth_i, depth_j = render(T_I), render(T_J)
    frame_i = predicted_inputs(depth_i, depth_i)
    # Pixels at the same GT depth predicted differently: no scale and shift can undo this.
    region = np.zeros((H, W), dtype=bool)
    region[:, : W // 3] = True
    distorted = np.where(region, 1.25 * depth_j, depth_j)

    affine = pair_consistency(frame_i, predicted_inputs(depth_j, 0.1 * depth_j + 7.0), K, relative_pose(T_I, T_J))
    distorted_result = pair_consistency(frame_i, predicted_inputs(depth_j, distorted), K, relative_pose(T_I, T_J))

    assert affine.visible.m2.overall.mean < 1.0
    assert distorted_result.visible.m2.overall.mean > 3.0


def test_self_pair_metrics_are_zero() -> None:
    depth = render(T_I)
    inputs = predicted_inputs(depth, 2.0 * depth + 0.5)

    result = pair_consistency(inputs, inputs, K, np.eye(4))

    assert result.visible_fraction == 1.0
    for summary in (result.visible.gt_floor, result.visible.m1, result.visible.m2):
        assert summary.overall.p90 < 1e-3


def test_fewer_than_min_visible_pixels_gives_nan_and_flag() -> None:
    depth_i = render(T_I)
    frame_i = predicted_inputs(depth_i, depth_i)
    T_far = pose(yaw_deg=60.0)
    frame_j = predicted_inputs(render(T_far), render(T_far))

    result = pair_consistency(frame_i, frame_j, K, relative_pose(T_I, T_far))

    assert result.too_few and result.interior.too_few
    assert result.visible.num_pixels < 1000
    for summary in (result.visible.gt_floor, result.visible.m1, result.visible.m2):
        assert math.isnan(summary.overall.mean) and math.isnan(summary.far.median)


def test_prediction_on_only_one_frame_is_rejected() -> None:
    depth = render(T_I)
    with pytest.raises(ValueError, match="frame j does not"):
        pair_consistency(predicted_inputs(depth, depth), gt_frame(T_I), K, np.eye(4))


def test_m2_floor_is_zero_on_exact_gt() -> None:
    result = pair_consistency(gt_frame(T_I), gt_frame(T_J), K, relative_pose(T_I, T_J))

    assert result.visible.m1 is None and result.visible.m2 is None
    assert result.visible.m2_floor.overall.median < 0.5
    assert result.interior.m2_floor.overall.median < 0.5


def test_m2_floor_is_reported_with_predictions_and_zero_on_self_pair() -> None:
    depth = render(T_I)
    inputs = predicted_inputs(depth, 2.0 * depth + 0.5)
    assert pair_consistency(inputs, inputs, K, np.eye(4)).visible.m2_floor.overall.p90 < 1e-3


# Camera moving forward at 2 m/s with identity rotation; frame 1's depth is captured 20 ms after its RGB image.
SPEED = 2.0
RGB_TIMES = (0.1, 0.2)
DEPTH_OFFSETS = (0.0, 0.02)


def forward_pose(t: float) -> np.ndarray:
    return pose(translation=(0.0, 0.0, SPEED * t))


def write_offset_sequence(root: Path) -> TUMSequence:
    times = np.arange(0.0, 0.5001, 0.01)
    lines = [f"{t:.6f} 0 0 {SPEED * t:.6f} 0 0 0 1" for t in times]
    (root / "groundtruth.txt").write_text("\n".join(lines) + "\n")
    frames = []
    for index, (t, offset) in enumerate(zip(RGB_TIMES, DEPTH_OFFSETS)):
        depth_path = root / f"depth_{index}.png"
        depth = render(forward_pose(t + offset))
        Image.fromarray(np.round(depth * DEPTH_SCALE).astype(np.uint16)).save(depth_path)
        frames.append(Frame(index, t, root / f"rgb_{index}.png", depth_path, offset, forward_pose(t)))
    return TUMSequence("synthetic", root, K, Window(None, None), frames)


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_script", Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_depth_pose_is_interpolated_at_the_depth_timestamp(tmp_path: Path) -> None:
    sequence = write_offset_sequence(tmp_path)
    gt = (np.arange(0.0, 0.5001, 0.01), np.array([[0, 0, SPEED * t, 0, 0, 0, 1] for t in np.arange(0.0, 0.5001, 0.01)]))

    np.testing.assert_allclose(depth_pose(sequence.frames[0], *gt), sequence.frames[0].T_world_cam, atol=1e-9)
    np.testing.assert_allclose(depth_pose(sequence.frames[1], *gt), forward_pose(0.22), atol=1e-9)
    assert not np.allclose(depth_pose(sequence.frames[1], *gt), sequence.frames[1].T_world_cam)


def test_gt_floor_uses_depth_timestamp_poses_when_timestamps_differ(tmp_path: Path) -> None:
    check_reprojection = load_script("check_reprojection")
    sequence = write_offset_sequence(tmp_path)
    frame_0, frame_1 = sequence.frames

    depth_timed = check_reprojection.InputCache(sequence, None, "depth").pair(frame_0, frame_1)
    rgb_timed = check_reprojection.InputCache(sequence, None, "rgb").pair(frame_0, frame_1)

    assert depth_timed.visible.gt_floor.overall.median < 0.05
    # 4 cm of unmodelled forward motion against surfaces 2-3.6 m away.
    assert rgb_timed.visible.gt_floor.overall.median > 0.8
