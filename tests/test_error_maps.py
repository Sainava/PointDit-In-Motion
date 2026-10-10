from __future__ import annotations

import numpy as np
import pytest

from pim.metrics.error_maps import depth_edge_band, edge_breakdown, relative_depth_error
from pim.metrics.geometry import backproject, evaluate_frame

H, W = 96, 128
EDGE = W // 2
K = np.array([[110.0, 0.0, 63.5], [0.0, 105.0, 47.5], [0.0, 0.0, 1.0]])


def step_depth() -> tuple[np.ndarray, np.ndarray]:
    depth = np.full((H, W), 1.5, dtype=np.float32)
    depth[:, EDGE:] = 3.0
    return depth, np.ones((H, W), dtype=bool)


def test_error_confined_to_rectangle_is_non_zero_exactly_there() -> None:
    depth, valid = step_depth()
    aligned = depth.copy()
    aligned[20:40, 10:30] *= 1.2

    error = relative_depth_error(aligned, depth, valid)

    expected = np.zeros((H, W), dtype=bool)
    expected[20:40, 10:30] = True
    assert np.array_equal(error > 0, expected)
    np.testing.assert_allclose(error[expected], 0.2, rtol=1e-6)


def test_band_covers_width_px_either_side_of_a_depth_step() -> None:
    depth, valid = step_depth()

    band = depth_edge_band(depth, valid, rel_jump=0.04, width_px=5)

    expected = np.zeros((H, W), dtype=bool)
    expected[:, EDGE - 6:EDGE + 6] = True
    assert np.array_equal(band, expected)


def test_error_along_the_edge_concentrates_in_the_band() -> None:
    depth, valid = step_depth()
    aligned = depth.copy()
    aligned[:, EDGE - 2:EDGE + 2] *= 1.3

    error = relative_depth_error(aligned, depth, valid)
    stats = edge_breakdown(error, depth_edge_band(depth, valid), valid)

    assert stats["band_error_fraction"] == pytest.approx(1.0)
    assert stats["band_pixel_fraction"] == pytest.approx(12 / W)
    assert stats["band_error_fraction"] > 5 * stats["band_pixel_fraction"]
    assert stats["rel_d_outside"] == 0.0


def test_uniform_error_is_split_like_the_pixels() -> None:
    depth, valid = step_depth()

    error = relative_depth_error(depth * 1.1, depth, valid)
    stats = edge_breakdown(error, depth_edge_band(depth, valid), valid)

    assert stats["band_error_fraction"] == pytest.approx(stats["band_pixel_fraction"], rel=1e-5)
    assert stats["rel_d_band"] == pytest.approx(10.0, rel=1e-5)
    assert stats["rel_d_outside"] == pytest.approx(10.0, rel=1e-5)


def test_invalid_gt_pixels_are_nan_and_excluded_from_statistics() -> None:
    depth, valid = step_depth()
    aligned = depth * 1.1
    hole = np.zeros((H, W), dtype=bool)
    hole[60:80, 90:110] = True
    valid_with_hole = valid & ~hole
    depth_with_hole = np.where(hole, 0.0, depth).astype(np.float32)
    garbage = aligned.copy()
    garbage[hole] = 1e6

    error = relative_depth_error(garbage, depth_with_hole, valid_with_hole)
    band = depth_edge_band(depth_with_hole, valid_with_hole)
    stats = edge_breakdown(error, band, valid_with_hole)

    assert np.all(np.isnan(error[hole]))
    assert np.all(np.isfinite(error[valid_with_hole]))
    assert not band[hole].any()
    # The hole's rim is an invalid-region boundary, so it is in the band.
    assert band[55:60, 90:110].all() and band[60:80, 110:115].all()
    # Cropping the hole out of the statistics: same band, same uniform 10% error everywhere else.
    assert stats["band_pixel_fraction"] == pytest.approx(band.sum() / valid_with_hole.sum())
    assert stats["rel_d_band"] == pytest.approx(10.0, rel=1e-5)
    assert stats["rel_d_outside"] == pytest.approx(10.0, rel=1e-5)


def test_error_map_mean_matches_evaluate_frame_rel_d() -> None:
    v, u = np.indices((H, W))
    depth = (2.0 + 0.01 * v + 0.005 * u).astype(np.float32)
    depth[30:60, 40:80] = 1.2
    valid = np.ones((H, W), dtype=bool)
    valid[::7, ::5] = False
    gt_points, gt_valid = backproject(depth, valid, K)
    rng = np.random.default_rng(0)
    pred = (0.5 * gt_points + 0.05 * rng.standard_normal(gt_points.shape)).astype(np.float32)

    metrics = evaluate_frame(pred, gt_points, gt_valid)
    error = relative_depth_error(metrics.aligned_depth, depth, gt_valid)

    assert metrics.rel_d > 1.0
    assert 100.0 * np.nanmean(error.astype(np.float64)) == pytest.approx(metrics.rel_d, rel=1e-4)


def test_skipped_frame_is_rejected() -> None:
    depth, valid = step_depth()

    with pytest.raises(ValueError, match="skipped"):
        relative_depth_error(np.full((H, W), np.nan, dtype=np.float32), depth, valid)
