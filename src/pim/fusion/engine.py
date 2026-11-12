"""Training-free temporal fusion of metric depth from several posed source frames into one target view."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence

import numpy as np

from pim.fusion.aggregate import aggregate
from pim.fusion.select import select_sources
from pim.fusion.types import FusionConfig, FusionResult, SourceView, TargetSpec
from pim.fusion.warp import WarpedLayer, self_layer, warp_to_target

BASELINE_SOFTNESS = 0.05


def fuse_frame(
    frames: Sequence[SourceView], target: TargetSpec, target_time: float, config: FusionConfig,
    self_view: SourceView | None = None,
    warp: Callable[[SourceView, TargetSpec], WarpedLayer] = warp_to_target,
) -> FusionResult:
    """self_view is the target frame's own depth, required when config.include_self is set. warp lets a caller
    reuse layers it already warped into this target."""
    if config.include_self:
        if self_view is None:
            raise ValueError("include_self is set but no self_view was given")
        if self_view.depth.shape != tuple(target.shape):
            raise ValueError(f"self_view depth {self_view.depth.shape} must match target shape {target.shape}")
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
    return FusionResult(
        depth=fused, weight_sum=weight_sum, n_sources=count, source_indices=tuple(indices),
        layers=depths if config.keep_layers else None, layer_weights=weights if config.keep_layers else None,
    )


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
