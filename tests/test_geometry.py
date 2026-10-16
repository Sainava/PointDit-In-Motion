from __future__ import annotations

import numpy as np
import pytest

from pim.metrics.geometry import backproject, evaluate_frame, project

H, W = 96, 128
K = np.array([[110.0, 0.0, 63.5], [0.0, 105.0, 47.5], [0.0, 0.0, 1.0]])


def synthetic_depth() -> tuple[np.ndarray, np.ndarray]:
    v, u = np.indices((H, W))
    depth = 2.0 + 0.01 * v + 0.005 * u
    depth[30:60, 40:80] = 1.2
    valid = np.ones((H, W), dtype=bool)
    valid[::7, ::5] = False
    valid[:10, -20:] = False
    return depth.astype(np.float32), valid


def gt_frame() -> tuple[np.ndarray, np.ndarray]:
    depth, valid = synthetic_depth()
    return backproject(depth, valid, K)


def noisy_prediction(gt_points: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(0)
    return (0.5 * gt_points + 0.05 * rng.standard_normal(gt_points.shape)).astype(np.float32)


def as_tuple(metrics) -> tuple[float, float, float, float, int]:
    return metrics.rel_p, metrics.delta1_p, metrics.rel_d, metrics.delta1_d, metrics.num_valid


def test_backproject_then_project_recovers_pixels_and_depth() -> None:
    depth, valid = synthetic_depth()

    points, points_valid = backproject(depth, valid, K)
    uv, z = project(points[points_valid], K)

    v, u = np.nonzero(points_valid)
    np.testing.assert_allclose(uv, np.stack([u, v], axis=-1), atol=1e-4)
    np.testing.assert_allclose(z, depth[points_valid], rtol=1e-6)
    assert np.array_equal(points_valid, valid)
    assert np.all(points[~points_valid] == 0.0)


def test_backproject_marks_non_finite_depth_invalid() -> None:
    depth, valid = synthetic_depth()
    depth[0, 0] = np.nan

    _, points_valid = backproject(depth, np.ones_like(valid), K)

    assert not points_valid[0, 0]


def test_prediction_equal_to_gt_up_to_scale_and_shift_is_perfect() -> None:
    gt_points, gt_valid = gt_frame()
    pred = (0.37 * gt_points + np.array([0.4, -0.2, 1.5], dtype=np.float32)).astype(np.float32)

    metrics = evaluate_frame(pred, gt_points, gt_valid)

    assert not metrics.skipped

    assert metrics.rel_p == pytest.approx(0.0, abs=1e-3)
    assert metrics.rel_d == pytest.approx(0.0, abs=1e-3)
    assert metrics.delta1_p == pytest.approx(100.0)
    assert metrics.delta1_d == pytest.approx(100.0)
    assert metrics.num_valid == gt_valid.sum()
    np.testing.assert_allclose(metrics.aligned_points[gt_valid], gt_points[gt_valid], atol=1e-4)
    np.testing.assert_allclose(metrics.aligned_depth[gt_valid], gt_points[gt_valid][:, 2], atol=1e-4)


def test_garbage_in_invalid_pixels_does_not_change_metrics() -> None:
    gt_points, gt_valid = gt_frame()
    pred = noisy_prediction(gt_points)
    pred_valid = np.ones_like(gt_valid)
    pred_valid[50:70, 10:30] = False
    clean = evaluate_frame(pred, gt_points, gt_valid, pred_valid=pred_valid)

    garbage_pred, garbage_gt = pred.copy(), gt_points.copy()
    garbage_pred[~pred_valid] = 1e6
    garbage_pred[~gt_valid] = np.nan
    garbage_gt[~gt_valid] = np.array([np.inf, -1e9, np.nan])
    garbage_gt[~pred_valid] = -5.0
    dirty = evaluate_frame(garbage_pred, garbage_gt, gt_valid, pred_valid=pred_valid)

    assert as_tuple(dirty) == as_tuple(clean)
    assert clean.num_valid == (gt_valid & pred_valid).sum()


def test_garbage_in_invalid_gt_pixels_does_not_change_default_metrics() -> None:
    gt_points, gt_valid = gt_frame()
    pred = noisy_prediction(gt_points)
    clean = evaluate_frame(pred, gt_points, gt_valid)

    garbage_pred, garbage_gt = pred.copy(), gt_points.copy()
    garbage_pred[~gt_valid] = np.nan
    garbage_gt[~gt_valid] = np.array([np.inf, -1e9, np.nan])
    dirty = evaluate_frame(garbage_pred, garbage_gt, gt_valid)

    assert as_tuple(dirty) == as_tuple(clean)


def test_default_masks_with_gt_validity_only() -> None:
    gt_points, gt_valid = gt_frame()
    pred = noisy_prediction(gt_points)
    pred_valid = np.ones_like(gt_valid)
    pred_valid[50:70, 10:30] = False

    default = evaluate_frame(pred, gt_points, gt_valid)
    all_valid = evaluate_frame(pred, gt_points, gt_valid, pred_valid=np.ones_like(gt_valid))
    with_pred_mask = evaluate_frame(pred, gt_points, gt_valid, pred_valid=pred_valid)

    assert default.num_valid == gt_valid.sum()
    assert as_tuple(default) == as_tuple(all_valid)
    assert with_pred_mask.num_valid < default.num_valid


def test_non_uniform_depth_scaling_is_clearly_worse() -> None:
    depth, valid = synthetic_depth()
    gt_points, gt_valid = backproject(depth, valid, K)
    ramp = 1.0 + 2.0 * np.arange(W, dtype=np.float32) / (W - 1)
    distorted_points, _ = backproject(depth * ramp[None, :], valid, K)

    good = evaluate_frame(0.5 * gt_points, gt_points, gt_valid)
    bad = evaluate_frame(distorted_points, gt_points, gt_valid)

    assert bad.rel_p > good.rel_p + 5.0
    assert bad.rel_d > good.rel_d + 5.0
    assert bad.delta1_p < good.delta1_p - 5.0
    assert bad.delta1_d < good.delta1_d - 5.0


def test_mismatched_sizes_fail_fast() -> None:
    gt_points, gt_valid = gt_frame()

    with pytest.raises(ValueError, match="share H and W"):
        evaluate_frame(gt_points[:-1], gt_points, gt_valid)
    with pytest.raises(ValueError, match="share H and W"):
        evaluate_frame(gt_points, gt_points, gt_valid, pred_valid=gt_valid[:-1])


@pytest.mark.parametrize("restrict", ["gt_valid", "pred_valid"])
def test_too_few_valid_pixels_is_skipped_not_raised(restrict: str) -> None:
    gt_points, gt_valid = gt_frame()
    few = np.zeros_like(gt_valid)
    few[1, :50] = True

    if restrict == "gt_valid":
        metrics = evaluate_frame(gt_points, gt_points, few)
    else:
        metrics = evaluate_frame(gt_points, gt_points, gt_valid, pred_valid=few)

    assert metrics.skipped
    assert metrics.num_valid == (few & gt_valid).sum()
    assert np.isnan([metrics.rel_p, metrics.delta1_p, metrics.rel_d, metrics.delta1_d]).all()
    assert metrics.aligned_points.shape == (H, W, 3) and np.isnan(metrics.aligned_points).all()
    assert metrics.aligned_depth.shape == (H, W) and np.isnan(metrics.aligned_depth).all()


def test_alignment_fit_larger_than_subsample_fails() -> None:
    from pim.metrics.geometry import MAX_ALIGNMENT_PIXELS, check_alignment_size

    check_alignment_size(MAX_ALIGNMENT_PIXELS)
    with pytest.raises(AssertionError, match="alignment fit"):
        check_alignment_size(MAX_ALIGNMENT_PIXELS + 1)


def test_upstream_aligners_are_size_checked() -> None:
    import torch

    from pim.metrics.geometry import MAX_ALIGNMENT_PIXELS, _upstream_compute_metrics

    _upstream_compute_metrics()
    import metrics.compute_metrics as module

    n = MAX_ALIGNMENT_PIXELS + 1
    with pytest.raises(AssertionError, match="alignment fit"):
        module.align_depth_affine(torch.ones(n), torch.ones(n), torch.ones(n))
    with pytest.raises(AssertionError, match="alignment fit"):
        module.align_points_scale_xyz_shift(torch.ones(n, 3), torch.ones(n, 3), torch.ones(n))
