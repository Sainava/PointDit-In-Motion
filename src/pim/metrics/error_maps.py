"""Per-pixel relative depth error and how much of it lies along GT depth edges."""

from __future__ import annotations

import numpy as np


def relative_depth_error(aligned_depth: np.ndarray, gt_depth: np.ndarray, gt_valid: np.ndarray) -> np.ndarray:
    _check_shapes(aligned_depth=aligned_depth, gt_depth=gt_depth, gt_valid=gt_valid)
    valid = gt_valid.astype(bool)
    if np.any(gt_depth[valid] <= 0) or not np.all(np.isfinite(gt_depth[valid])):
        raise ValueError("gt_depth must be finite and positive on every gt_valid pixel")
    if not np.all(np.isfinite(aligned_depth[valid])):
        raise ValueError("aligned_depth is not finite on every gt_valid pixel (was the frame skipped?)")
    error = np.full(gt_depth.shape, np.nan, dtype=np.float32)
    gt = gt_depth[valid].astype(np.float64)
    error[valid] = np.abs(aligned_depth[valid].astype(np.float64) - gt) / gt
    return error


def depth_edge_band(
    gt_depth: np.ndarray, gt_valid: np.ndarray, rel_jump: float = 0.04, width_px: int = 5
) -> np.ndarray:
    """Valid pixels within width_px (Chebyshev distance) of a GT depth jump or of the GT invalid region."""
    _check_shapes(gt_depth=gt_depth, gt_valid=gt_valid)
    if rel_jump <= 0:
        raise ValueError(f"rel_jump must be positive, got {rel_jump}")
    if width_px < 0:
        raise ValueError(f"width_px must be non-negative, got {width_px}")
    valid = gt_valid.astype(bool)
    depth = np.where(valid, gt_depth, np.nan).astype(np.float64)
    seeds = np.zeros(valid.shape, dtype=bool)
    for axis in (0, 1):
        a = _leading(depth, axis, 1)
        b = _trailing(depth, axis, 1)
        a_valid, b_valid = _leading(valid, axis, 1), _trailing(valid, axis, 1)
        with np.errstate(invalid="ignore"):
            jump = a_valid & b_valid & (np.abs(a - b) > rel_jump * np.minimum(a, b))
        boundary = a_valid != b_valid
        marked = jump | boundary
        _leading(seeds, axis, 1)[...] |= marked
        _trailing(seeds, axis, 1)[...] |= marked
    return _dilate(seeds, width_px) & valid


def edge_breakdown(error: np.ndarray, band: np.ndarray, gt_valid: np.ndarray) -> dict[str, float]:
    _check_shapes(error=error, band=band, gt_valid=gt_valid)
    valid = gt_valid.astype(bool)
    if not valid.any():
        raise ValueError("no valid GT pixels")
    if not np.all(np.isfinite(error[valid])):
        raise ValueError("error is not finite on every gt_valid pixel")
    inside = band.astype(bool) & valid
    outside = ~band.astype(bool) & valid
    total = float(error[valid].sum(dtype=np.float64))
    return {
        "band_pixel_fraction": float(inside.sum() / valid.sum()),
        "band_error_fraction": float(error[inside].sum(dtype=np.float64)) / total if total > 0 else float("nan"),
        "rel_d_band": _mean_x100(error[inside]),
        "rel_d_outside": _mean_x100(error[outside]),
    }


def _mean_x100(values: np.ndarray) -> float:
    return 100.0 * float(values.mean(dtype=np.float64)) if values.size else float("nan")


def _dilate(mask: np.ndarray, width_px: int) -> np.ndarray:
    out = mask.copy()
    for axis in (0, 1):
        source = out.copy()
        for shift in range(1, width_px + 1):
            _leading(out, axis, shift)[...] |= _trailing(source, axis, shift)
            _trailing(out, axis, shift)[...] |= _leading(source, axis, shift)
    return out


def _leading(array: np.ndarray, axis: int, shift: int) -> np.ndarray:
    return array[:-shift] if axis == 0 else array[:, :-shift]


def _trailing(array: np.ndarray, axis: int, shift: int) -> np.ndarray:
    return array[shift:] if axis == 0 else array[:, shift:]


def _check_shapes(**arrays: np.ndarray) -> None:
    shapes = {name: array.shape for name, array in arrays.items()}
    if len(set(shapes.values())) != 1 or any(len(s) != 2 for s in shapes.values()):
        raise ValueError(f"inputs must be HxW arrays of one shape, got {shapes}")
