"""Pinhole back-projection and per-frame affine-invariant metrics via upstream's evaluation code."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pim.paths import configure_fork

# engine.evaluate_img2point skips samples with fewer valid GT pixels than this.
MIN_VALID_PIXELS = 100
# Pixels near the principal column carry little focal information and divide by X ~ 0.
FOCAL_MIN_OFFSET_PX = 50.0


@dataclass(frozen=True)
class FrameMetrics:
    # Rel and δ1 are ×100, as upstream reports them; NaN when skipped.
    rel_p: float
    delta1_p: float
    rel_d: float
    delta1_d: float
    num_valid: int
    skipped: bool
    """True when num_valid < MIN_VALID_PIXELS, where upstream leaves the frame out."""
    aligned_points: np.ndarray
    """(H, W, 3) float32 prediction after upstream's scale + xyz-shift point alignment."""
    aligned_depth: np.ndarray
    """(H, W) float32 predicted z after upstream's separate scale + shift depth alignment."""


def backproject(depth: np.ndarray, valid: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    _check_hw(depth=depth, valid=valid)
    _check_intrinsics(K)
    valid = valid.astype(bool) & np.isfinite(depth)
    z = np.where(valid, depth, 0.0).astype(np.float64)
    v, u = np.indices(depth.shape, dtype=np.float64)
    x = (u - K[0, 2]) * z / K[0, 0]
    y = (v - K[1, 2]) * z / K[1, 1]
    return np.stack([x, y, z], axis=-1).astype(np.float32), valid


def project(points: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if points.shape[-1] != 3:
        raise ValueError(f"expected (..., 3) camera-frame points, got {points.shape}")
    _check_intrinsics(K)
    points = points.reshape(-1, 3).astype(np.float64)
    z = points[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = K[0, 0] * points[:, 0] / z + K[0, 2]
        v = K[1, 1] * points[:, 1] / z + K[1, 2]
    return np.stack([u, v], axis=-1), z


def evaluate_frame(
    pred_points: np.ndarray,
    gt_points: np.ndarray,
    gt_valid: np.ndarray,
    pred_valid: np.ndarray | None = None,
    device: str = "cpu",
) -> FrameMetrics:
    # Upstream masks with GT validity only; pred_valid is an explicit opt-in on top of that.
    masks = {"gt_valid": gt_valid} if pred_valid is None else {"gt_valid": gt_valid, "pred_valid": pred_valid}
    _check_hw(pred_points=pred_points, gt_points=gt_points, **masks)
    for name, points in (("pred_points", pred_points), ("gt_points", gt_points)):
        if points.ndim != 3 or points.shape[2] != 3:
            raise ValueError(f"{name} must be (H, W, 3), got {points.shape}")
    mask = gt_valid.astype(bool)
    if pred_valid is not None:
        mask = mask & pred_valid.astype(bool)
    num_valid = int(mask.sum())
    if num_valid < MIN_VALID_PIXELS:
        return _skipped(num_valid, mask.shape)

    import torch

    compute_metrics = _upstream_compute_metrics()
    pred = torch.from_numpy(np.ascontiguousarray(pred_points, dtype=np.float32)).to(device)[None]
    gt = torch.from_numpy(np.ascontiguousarray(gt_points, dtype=np.float32)).to(device)[None]
    with torch.no_grad():
        metrics, aligned = compute_metrics(
            {"points_affine_invariant": pred, "depth_affine_invariant": pred[..., 2]},
            {"points": gt, "depth": gt[..., 2], "depth_mask": torch.from_numpy(mask).to(device)[None]},
            vis=True,
        )
    points, depth = metrics["points_affine_invariant"], metrics["depth_affine_invariant"]
    return FrameMetrics(
        rel_p=points["rel"] * 100.0,
        delta1_p=points["delta1"] * 100.0,
        rel_d=depth["rel"] * 100.0,
        delta1_d=depth["delta1"] * 100.0,
        num_valid=num_valid,
        skipped=False,
        aligned_points=aligned["pred_points"][0].float().cpu().numpy(),
        aligned_depth=aligned["pred_depth"][0].float().cpu().numpy(),
    )


def implied_focal(points: np.ndarray, valid: np.ndarray, cx: float, min_offset_px: float = FOCAL_MIN_OFFSET_PX) -> float:
    """Horizontal focal length, in pixels, of a point map known only up to scale and xyz shift; NaN if unusable."""
    _check_hw(points=points, valid=valid)
    if points.ndim != 3 or points.shape[2] != 3:
        raise ValueError(f"points must be (H, W, 3), got {points.shape}")
    offset = np.broadcast_to(np.arange(points.shape[1], dtype=np.float64) - cx, points.shape[:2])
    x, z = points[..., 0].astype(np.float64), points[..., 2].astype(np.float64)
    usable = valid.astype(bool) & (np.abs(offset) > min_offset_px) & np.isfinite(x) & np.isfinite(z)
    if usable.sum() < 3:
        return float("nan")
    du, x, z = offset[usable], x[usable], z[usable]
    # X·f = (u - cx)·Z holds only before the unknown shift t, so fit X = a(u - cx)Z + b(u - cx) + c
    # with a = 1/f, b = -t_z/f, c = t_x. Fitting X·f = (u - cx)·Z directly lets t_z absorb most of a
    # focal error whenever the depth range is narrow.
    design = np.stack([du * z, du, np.ones_like(du)], axis=1)
    norms = np.linalg.norm(design, axis=0)
    if np.any(norms == 0):
        return float("nan")
    design = design / norms
    coef = _l1_regression(design, x)
    a = coef[0] / norms[0]
    return float(1.0 / a) if a > 0 else float("nan")


def _l1_regression(design: np.ndarray, target: np.ndarray, iterations: int = 30) -> np.ndarray:
    """Least-absolute-deviations fit by iteratively reweighted least squares."""
    floor = 1e-9 * max(float(np.median(np.abs(target))), 1e-12)
    weights = np.ones_like(target)
    coef = np.zeros(design.shape[1])
    for _ in range(iterations):
        root = np.sqrt(weights)
        coef = np.linalg.lstsq(design * root[:, None], target * root, rcond=None)[0]
        weights = 1.0 / np.maximum(np.abs(target - design @ coef), floor)
    return coef


def _skipped(num_valid: int, shape: tuple[int, int]) -> FrameMetrics:
    return FrameMetrics(
        rel_p=float("nan"),
        delta1_p=float("nan"),
        rel_d=float("nan"),
        delta1_d=float("nan"),
        num_valid=num_valid,
        skipped=True,
        aligned_points=np.full((*shape, 3), np.nan, dtype=np.float32),
        aligned_depth=np.full(shape, np.nan, dtype=np.float32),
    )


def _upstream_compute_metrics():
    configure_fork()
    from metrics.compute_metrics import compute_metrics

    return compute_metrics


def _check_hw(**arrays: np.ndarray) -> None:
    sizes = {name: array.shape[:2] for name, array in arrays.items()}
    if len(set(sizes.values())) != 1:
        raise ValueError(f"inputs must share H and W, got {sizes}")


def _check_intrinsics(K: np.ndarray) -> None:
    if K.shape != (3, 3):
        raise ValueError(f"K must be 3x3, got {K.shape}")
