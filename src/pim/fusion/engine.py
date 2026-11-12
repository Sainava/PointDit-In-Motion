"""Training-free temporal fusion of metric depth from several posed source frames into one target view."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import replace

import numpy as np

from pim.fusion.aggregate import aggregate
from pim.fusion.select import select_sources
from pim.fusion.types import FusionConfig, FusionResult, SourceView, TargetSpec
from pim.fusion.warp import WarpedLayer, self_layer, warp_to_target
from pim.metrics.error_maps import depth_edge_band

BASELINE_SOFTNESS = 0.05

EdgeBand = Callable[[SourceView, int], np.ndarray]
MaskSource = Callable[[SourceView, int], SourceView]


def view_edge_band(view: SourceView, width_px: int) -> np.ndarray:
    """depth_edge_band of the view's own depth over its finite positive pixels, at the default relative jump."""
    with np.errstate(invalid="ignore"):
        positive = np.isfinite(view.depth) & (view.depth > 0)
    return depth_edge_band(view.depth, positive, width_px=width_px)


def without_band(view: SourceView, band: np.ndarray) -> SourceView:
    return replace(view, valid=view.valid.astype(bool) & ~band)


def fuse_frame(
    frames: Sequence[SourceView], target: TargetSpec, target_time: float, config: FusionConfig,
    self_view: SourceView | None = None,
    warp: Callable[[SourceView, TargetSpec], WarpedLayer] = warp_to_target,
    edge_band: EdgeBand = view_edge_band,
    mask_source: MaskSource | None = None,
) -> FusionResult:
    """self_view is the target frame's own depth, required when config.include_self is set. warp lets a caller
    reuse layers it already warped into this target, edge_band and mask_source let it reuse edge bands and masked
    sources; mask_source defaults to removing edge_band from the source's valid mask."""
    if config.include_self:
        if self_view is None:
            raise ValueError("include_self is set but no self_view was given")
        if self_view.depth.shape != tuple(target.shape):
            raise ValueError(f"self_view depth {self_view.depth.shape} must match target shape {target.shape}")
    if config.target_edge_fallback_px is not None and not config.include_self:
        raise ValueError("target_edge_fallback_px needs include_self, since the fallback is the target's own depth")
    if config.source_edge_mask_px is not None:
        warp = _masked_warp(warp, mask_source or _band_mask(edge_band), config.source_edge_mask_px)
    indices = select_sources([f.timestamp for f in frames], target_time, config)
    n_layers = len(indices) + int(config.include_self)
    shape = target.grid_shape
    depths = np.full((n_layers, *shape), np.nan, dtype=np.float32)
    weights = np.zeros((n_layers, *shape), dtype=np.float32)
    hits = np.zeros((n_layers, *shape), dtype=bool)

    for k, layer in enumerate(_layers(frames, indices, target, self_view if config.include_self else None, warp)):
        is_self = config.include_self and k == 0
        depths[k], hits[k] = layer.depth, layer.hit
        weights[k] = layer.hit * np.float32(config.self_weight) if is_self else layer_weight(layer, config)

    fused, weight_sum, count = aggregate(depths, weights, hits, config)
    result = FusionResult(
        depth=fused, weight_sum=weight_sum, n_sources=count, source_indices=tuple(indices),
        layers=depths if config.keep_layers else None, layer_weights=weights if config.keep_layers else None,
    )
    if config.target_edge_fallback_px is not None:
        band = edge_band(self_view, config.target_edge_fallback_px)
        result = edge_fallback(result, depths[0], hits[0], band, config.self_weight)
    return result


def edge_fallback(result: FusionResult, own_depth: np.ndarray, own_hit: np.ndarray, band: np.ndarray,
                  own_weight: float = 1.0) -> FusionResult:
    """Pixels of band where the target's own grid-sized layer has depth take that depth as the only candidate. band
    may be at native resolution and is then repeated up to the grid."""
    scale = own_depth.shape[0] // band.shape[0]
    if scale > 1:
        band = band.repeat(scale, axis=0).repeat(scale, axis=1)
    if band.shape != own_depth.shape:
        raise ValueError(f"edge band {band.shape} does not fit the grid {own_depth.shape}")
    use = band & own_hit
    return replace(
        result,
        depth=np.where(use, own_depth, result.depth).astype(np.float32),
        weight_sum=np.where(use, np.float32(own_weight), result.weight_sum).astype(np.float32),
        n_sources=np.where(use, 1, result.n_sources).astype(np.int32),
    )


def _band_mask(edge_band: EdgeBand) -> MaskSource:
    def mask(view: SourceView, width_px: int) -> SourceView:
        return without_band(view, edge_band(view, width_px))
    return mask


def _masked_warp(warp: Callable[[SourceView, TargetSpec], WarpedLayer], mask: MaskSource, width_px: int
                 ) -> Callable[[SourceView, TargetSpec], WarpedLayer]:
    def masked(view: SourceView, target: TargetSpec) -> WarpedLayer:
        return warp(mask(view, width_px), target)
    return masked


def _layers(
    frames: Sequence[SourceView], indices: list[int], target: TargetSpec, self_view: SourceView | None,
    warp: Callable[[SourceView, TargetSpec], WarpedLayer] = warp_to_target,
) -> Iterator[WarpedLayer]:
    if self_view is not None:
        yield self_layer(self_view, target.grid_scale)
    for i in indices:
        yield warp(frames[i], target)


def layer_weight(layer: WarpedLayer, config: FusionConfig) -> np.ndarray:
    # Visibility weight is 1 on a hit and 0 elsewhere.
    weight = layer.hit.astype(np.float32)
    if config.ray_angle_weight:
        # Ray-angle weight is cos(angle between source and target rays to the point), clipped to [0, 1].
        weight *= np.clip(layer.cos_angle, 0, 1)
    if config.baseline_weight and layer.hit.any():
        # Baseline weight is r / (r + 0.05) with r = baseline / median valid source depth, so near-duplicate
        # views get weights near 0. The self layer never gets it.
        r = layer.baseline / layer.median_depth
        weight *= np.float32(r / (r + BASELINE_SOFTNESS))
    return weight
