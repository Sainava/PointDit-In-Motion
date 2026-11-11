"""Forward warping of one source view's metric depth into a target camera's grid."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pim.consistency.warp import relative_pose
from pim.fusion.types import SourceView, TargetSpec
from pim.metrics.geometry import backproject, project

_SPLAT_DU = np.array([0, 1, 0, 1], dtype=np.int64)
_SPLAT_DV = np.array([0, 0, 1, 1], dtype=np.int64)


@dataclass(frozen=True)
class WarpedLayer:
    depth: np.ndarray
    """Grid-sized float32 depth in the target camera, NaN where not hit."""
    hit: np.ndarray
    cos_angle: np.ndarray
    """Cosine between the source ray and the target ray to the winning point, 0 where not hit."""
    baseline: float
    """Distance between the source and target camera centres, in metres."""
    median_depth: float
    """Median valid source depth, in metres; NaN when the source has none."""


def warp_to_target(source: SourceView, target: TargetSpec) -> WarpedLayer:
    height, width = target.grid_shape
    T = relative_pose(source.T_world_cam, target.T_world_cam)
    usable = source.usable
    baseline = float(np.linalg.norm(T[:3, 3]))
    if not usable.any():
        return _empty_layer((height, width), baseline)
    median_depth = float(np.median(source.depth[usable]))

    points, _ = backproject(source.depth, usable, source.K)
    p = points[usable].astype(np.float64) @ T[:3, :3].T + T[:3, 3]
    p = p[p[:, 2] > 0]
    uv, z = project(p, target.grid_K)
    on_grid = (uv[:, 0] >= -1) & (uv[:, 0] < width) & (uv[:, 1] >= -1) & (uv[:, 1] < height)
    uv, z, cos = uv[on_grid], z[on_grid], _ray_cosine(p[on_grid], T[:3, 3])

    # Each point writes its depth to the 2x2 block of pixel centres around its sub-pixel position, so
    # neighbouring points landing up to two pixels apart leave no crack between them.
    u = (np.floor(uv[:, 0]).astype(np.int64)[None] + _SPLAT_DU[:, None]).ravel()
    v = (np.floor(uv[:, 1]).astype(np.int64)[None] + _SPLAT_DV[:, None]).ravel()
    z, cos = np.tile(z, 4), np.tile(cos, 4)
    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    pixel, z, cos = v[inside] * width + u[inside], z[inside], cos[inside]

    order = np.lexsort((z, pixel))
    pixel, z, cos = pixel[order], z[order], cos[order]
    nearest = np.ones(pixel.size, dtype=bool)
    nearest[1:] = pixel[1:] != pixel[:-1]
    pixel = pixel[nearest]

    depth = np.full(height * width, np.nan, dtype=np.float32)
    hit = np.zeros(height * width, dtype=bool)
    cos_angle = np.zeros(height * width, dtype=np.float32)
    depth[pixel] = z[nearest]
    hit[pixel] = True
    cos_angle[pixel] = cos[nearest]
    shape = (height, width)
    return WarpedLayer(depth.reshape(shape), hit.reshape(shape), cos_angle.reshape(shape), baseline, median_depth)


def self_layer(view: SourceView, grid_scale: int) -> WarpedLayer:
    # Nearest-neighbour upsampling under the pixel-centre convention is a plain repeat, and at scale 1
    # this copies the view bitwise instead of splatting it through an identity warp.
    usable = view.usable
    depth = np.where(usable, view.depth, np.float32(np.nan))
    hit = usable
    if grid_scale > 1:
        depth = depth.repeat(grid_scale, axis=0).repeat(grid_scale, axis=1)
        hit = hit.repeat(grid_scale, axis=0).repeat(grid_scale, axis=1)
    median_depth = float(np.median(view.depth[usable])) if usable.any() else float("nan")
    return WarpedLayer(depth, hit, hit.astype(np.float32), 0.0, median_depth)


def _ray_cosine(points: np.ndarray, source_centre: np.ndarray) -> np.ndarray:
    target_ray = points / np.linalg.norm(points, axis=1, keepdims=True)
    source_ray = points - source_centre
    source_ray /= np.linalg.norm(source_ray, axis=1, keepdims=True)
    return np.clip(np.einsum("ij,ij->i", target_ray, source_ray), -1.0, 1.0)


def _empty_layer(shape: tuple[int, int], baseline: float) -> WarpedLayer:
    return WarpedLayer(np.full(shape, np.nan, dtype=np.float32), np.zeros(shape, dtype=bool),
                       np.zeros(shape, dtype=np.float32), baseline, float("nan"))
