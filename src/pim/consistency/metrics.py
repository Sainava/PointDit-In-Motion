"""Reprojection-based temporal consistency between two posed frames: GT visibility, GT noise floor, M1 and M2."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pim.consistency.warp import warp_depth
from pim.metrics.error_maps import depth_edge_band
from pim.metrics.geometry import backproject, check_alignment_size, evaluate_frame
from pim.paths import configure_fork

VISIBILITY_THRESHOLD = 0.15
MIN_VISIBLE_PIXELS = 1000
# Upstream's compute_metrics fits its alignment on a 64x64 masked-nearest subsample; the exact
# L1 solver is quadratic in the number of pixels, so M2 fits on the same subsample.
ALIGNMENT_GRID = (64, 64)


@dataclass(frozen=True)
class FrameInputs:
    gt_depth: np.ndarray
    gt_valid: np.ndarray
    raw_depth: np.ndarray | None = None
    """z channel of the cached prediction, in the prediction's own normalized coordinates."""
    aligned_depth: np.ndarray | None = None
    """Prediction depth after upstream's per-frame alignment to this frame's GT."""


@dataclass(frozen=True)
class Visibility:
    visible: np.ndarray
    interior: np.ndarray
    """visible minus depth_edge_band(GT_j)."""
    warped_gt: np.ndarray
    """GT depth of frame i warped into frame j (z_gt_i→j), 0 where not hit."""


@dataclass(frozen=True)
class Summary:
    mean: float
    median: float
    p90: float


@dataclass(frozen=True)
class ErrorSummary:
    overall: Summary
    near: Summary
    mid: Summary
    far: Summary


@dataclass(frozen=True)
class PixelSetMetrics:
    num_pixels: int
    fraction: float
    """num_pixels over GT-valid pixels of frame j."""
    too_few: bool
    gt_floor: ErrorSummary
    gt_floor_rgb: ErrorSummary
    """GT floor with GT_i warped by T_j_from_i (RGB-timestamp poses), scored on the same visible pixels."""
    m2_floor: ErrorSummary
    """M2 with GT_j as frame j's raw prediction and GT_i as the anchor."""
    m1: ErrorSummary | None
    m2: ErrorSummary | None


@dataclass(frozen=True)
class PairMaps:
    m2: np.ndarray | None
    """Per-pixel M2 relative error (not x100), NaN where not scored; None without predictions."""
    anchor_depth: np.ndarray | None
    """Frame i's GT-aligned prediction warped into j, NaN where not hit; None without predictions."""


@dataclass(frozen=True)
class PairConsistency:
    visible: PixelSetMetrics
    interior: PixelSetMetrics
    maps: PairMaps | None = None

    @property
    def visible_fraction(self) -> float:
        return self.visible.fraction

    @property
    def too_few(self) -> bool:
        return self.visible.too_few


def frame_inputs(
    gt_depth: np.ndarray, gt_valid: np.ndarray, K: np.ndarray, pred_points: np.ndarray | None = None
) -> FrameInputs:
    if pred_points is None:
        return FrameInputs(gt_depth, gt_valid)
    gt_points, valid = backproject(gt_depth, gt_valid, K)
    result = evaluate_frame(pred_points, gt_points, valid)
    if result.skipped:
        raise ValueError(f"prediction cannot be aligned to GT: only {result.num_valid} valid GT pixels")
    return FrameInputs(gt_depth, valid, pred_points[..., 2].astype(np.float32), result.aligned_depth)


def visibility(
    frame_i: FrameInputs, frame_j: FrameInputs, K: np.ndarray, T_j_from_i: np.ndarray,
    threshold: float = VISIBILITY_THRESHOLD,
) -> Visibility:
    warped, hit = warp_depth(frame_i.gt_depth, frame_i.gt_valid, K, T_j_from_i, frame_j.gt_depth.shape)
    valid_j = frame_j.gt_valid.astype(bool)
    with np.errstate(divide="ignore", invalid="ignore"):
        agree = np.abs(warped - frame_j.gt_depth) <= threshold * frame_j.gt_depth
    visible = hit & valid_j & agree
    interior = visible & ~depth_edge_band(frame_j.gt_depth, valid_j)
    return Visibility(visible, interior, warped)


def pair_consistency(
    frame_i: FrameInputs, frame_j: FrameInputs, K: np.ndarray, T_j_from_i: np.ndarray,
    T_gt_j_from_i: np.ndarray | None = None, keep_maps: bool = False,
) -> PairConsistency:
    """T_j_from_i drives M1, M2, the M2 floor and gt_floor_rgb; T_gt_j_from_i (default T_j_from_i) drives
    visibility and the GT floor."""
    vis = visibility(frame_i, frame_j, K, T_j_from_i if T_gt_j_from_i is None else T_gt_j_from_i)
    gt_j = frame_j.gt_depth.astype(np.float64)
    if T_gt_j_from_i is None:
        rgb_warped, rgb_scored = vis.warped_gt, vis.visible
    else:
        rgb_warped, rgb_hit = warp_depth(frame_i.gt_depth, frame_i.gt_valid, K, T_j_from_i, gt_j.shape)
        rgb_scored = vis.visible & rgb_hit
    m2_floor, m2_floor_scored, _ = _m2(frame_i.gt_depth, frame_i.gt_valid, frame_j.gt_depth, vis.visible, K,
                                       T_j_from_i)
    metrics = {
        "gt_floor": (_relative(vis.warped_gt, gt_j, vis.visible), vis.visible),
        "gt_floor_rgb": (_relative(rgb_warped, gt_j, rgb_scored), rgb_scored),
        "m2_floor": (m2_floor, m2_floor_scored),
    }
    maps = PairMaps(None, None) if keep_maps else None
    if frame_i.aligned_depth is not None:
        if frame_j.aligned_depth is None or frame_j.raw_depth is None:
            raise ValueError("frame i has a prediction but frame j does not")
        m2, scored, anchor = _m2(frame_i.aligned_depth, frame_i.gt_valid, frame_j.raw_depth, vis.visible, K,
                                 T_j_from_i)
        metrics["m1"] = (_relative(anchor, frame_j.aligned_depth, scored, denominator=gt_j), scored)
        metrics["m2"] = (m2, scored)
        if keep_maps:
            maps = PairMaps(m2, np.where(anchor > 0, anchor, np.nan))
    valid_count = int(frame_j.gt_valid.sum())
    return PairConsistency(
        visible=_pixel_set(vis.visible, valid_count, gt_j, metrics),
        interior=_pixel_set(vis.interior, valid_count, gt_j, metrics),
        maps=maps,
    )


def _m2(
    anchor_depth_i: np.ndarray, valid_i: np.ndarray, raw_depth_j: np.ndarray, visible: np.ndarray,
    K: np.ndarray, T_j_from_i: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(error map, scored mask, anchor depth warped into j with 0 where not hit)."""
    anchor, anchor_hit = warp_depth(anchor_depth_i, valid_i, K, T_j_from_i, visible.shape)
    scored = visible & anchor_hit
    anchor = anchor.astype(np.float64)
    return _relative(_align_to_anchor(raw_depth_j, anchor, scored), anchor, scored), scored, anchor


def _relative(
    estimate: np.ndarray, reference: np.ndarray, mask: np.ndarray, denominator: np.ndarray | None = None
) -> np.ndarray:
    denominator = reference if denominator is None else denominator
    error = np.full(mask.shape, np.nan)
    error[mask] = np.abs(estimate[mask].astype(np.float64) - reference[mask]) / denominator[mask]
    return error


def _align_to_anchor(raw_depth: np.ndarray, anchor: np.ndarray, mask: np.ndarray) -> np.ndarray:
    # M2 anchor: frame j's raw prediction is fitted to frame i's GT-aligned depth warped into j,
    # not to GT_j, so GT only fixes frame i's scale and shift.
    if mask.sum() < MIN_VISIBLE_PIXELS:
        return np.full(mask.shape, np.nan)
    configure_fork()
    import torch
    from metrics.alignment import align_depth_affine
    from metrics.masked_resize import masked_nearest_resize

    src = torch.from_numpy(np.ascontiguousarray(raw_depth, dtype=np.float32))[None]
    tgt = torch.from_numpy(np.ascontiguousarray(anchor, dtype=np.float32))[None]
    with torch.no_grad():
        lr_mask, lr_index = masked_nearest_resize(mask=torch.from_numpy(mask)[None], size=ALIGNMENT_GRID,
                                                  return_index=True)
        src_lr, tgt_lr = src[lr_index][lr_mask], tgt[lr_index][lr_mask]
        check_alignment_size(src_lr.shape[-1])
        scale, shift = align_depth_affine(src_lr, tgt_lr, 1 / tgt_lr)
    return raw_depth.astype(np.float64) * float(scale) + float(shift)


def _pixel_set(
    pixels: np.ndarray, valid_count: int, gt_depth: np.ndarray, metrics: dict[str, tuple[np.ndarray, np.ndarray]],
) -> PixelSetMetrics:
    num_pixels = int(pixels.sum())
    too_few = num_pixels < MIN_VISIBLE_PIXELS
    tertiles = _tertile_masks(gt_depth, pixels) if not too_few else None

    def summarize(error: np.ndarray, mask: np.ndarray) -> ErrorSummary:
        if too_few or mask.sum() < MIN_VISIBLE_PIXELS:
            return _nan_error_summary()
        near, mid, far = (mask & t for t in tertiles)
        return ErrorSummary(*(_summary(error[m]) for m in (mask, near, mid, far)))

    return PixelSetMetrics(
        num_pixels=num_pixels,
        fraction=num_pixels / valid_count if valid_count else float("nan"),
        too_few=too_few,
        **{name: summarize(error, scored & pixels) for name, (error, scored) in metrics.items()},
        **{name: None for name in ("m1", "m2") if name not in metrics},
    )


def _tertile_masks(gt_depth: np.ndarray, pixels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    low, high = np.percentile(gt_depth[pixels], [100 / 3, 200 / 3])
    return pixels & (gt_depth < low), pixels & (gt_depth >= low) & (gt_depth < high), pixels & (gt_depth >= high)


def _summary(values: np.ndarray) -> Summary:
    if values.size == 0:
        return Summary(float("nan"), float("nan"), float("nan"))
    median, p90 = np.percentile(values, [50, 90])
    return Summary(100.0 * float(values.mean()), 100.0 * float(median), 100.0 * float(p90))


def _nan_error_summary() -> ErrorSummary:
    nan = Summary(float("nan"), float("nan"), float("nan"))
    return ErrorSummary(nan, nan, nan, nan)
