"""Inputs, configuration and result of training-free temporal depth fusion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MODES = ("causal", "centred")
SELECTIONS = ("adjacent", "diverse")
AGGREGATORS = ("weighted_mean", "weighted_median", "trimmed_mean")


@dataclass(frozen=True)
class SourceView:
    depth: np.ndarray
    """(H, W) float32 metric depth, 0 or NaN where missing."""
    valid: np.ndarray
    T_world_cam: np.ndarray
    K: np.ndarray
    timestamp: float
    rgb: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.depth.ndim != 2:
            raise ValueError(f"depth must be (H, W), got {self.depth.shape}")
        if self.depth.dtype != np.float32:
            raise ValueError(f"depth must be float32, got {self.depth.dtype}")
        if self.valid.shape != self.depth.shape:
            raise ValueError(f"valid {self.valid.shape} must match depth {self.depth.shape}")
        if self.rgb is not None and (self.rgb.ndim != 3 or self.rgb.shape[:2] != self.depth.shape):
            raise ValueError(f"rgb must be (H, W, C) matching depth {self.depth.shape}, got {self.rgb.shape}")
        if not np.isfinite(self.timestamp):
            raise ValueError(f"timestamp must be finite, got {self.timestamp}")
        check_pose(self.T_world_cam, "T_world_cam")
        check_intrinsics(self.K)

    @property
    def usable(self) -> np.ndarray:
        return self.valid.astype(bool) & np.isfinite(self.depth) & (self.depth > 0)


@dataclass(frozen=True)
class TargetSpec:
    T_world_cam: np.ndarray
    K: np.ndarray
    shape: tuple[int, int]
    """Native (H, W); the fused grid is grid_scale times larger."""
    grid_scale: int = 1

    def __post_init__(self) -> None:
        check_pose(self.T_world_cam, "T_world_cam")
        check_intrinsics(self.K)
        if len(self.shape) != 2 or min(self.shape) < 1:
            raise ValueError(f"shape must be (H, W) with positive sizes, got {self.shape}")
        if not isinstance(self.grid_scale, (int, np.integer)) or self.grid_scale < 1:
            raise ValueError(f"grid_scale must be an integer >= 1, got {self.grid_scale!r}")

    @property
    def grid_shape(self) -> tuple[int, int]:
        return self.shape[0] * self.grid_scale, self.shape[1] * self.grid_scale

    @property
    def grid_K(self) -> np.ndarray:
        return scale_intrinsics(self.K, self.grid_scale)


@dataclass(frozen=True)
class FusionConfig:
    window: int = 0
    """Frames per side in centred mode, past frames in causal mode."""
    mode: str = "centred"
    selection: str = "adjacent"
    stride_ms: float = 100.0
    stride_tolerance_ms: float = 20.0
    aggregator: str = "weighted_mean"
    trim_fraction: float = 0.2
    front_layer: bool = False
    front_tolerance: float = 0.15
    min_support: int = 1
    """Agreeing candidates a candidate needs, counting itself; 1 disables the filter."""
    ray_angle_weight: bool = False
    baseline_weight: bool = False
    include_self: bool = True
    self_weight: float = 1.0
    keep_layers: bool = False
    source_edge_mask_px: int | None = None
    """Width of each source's own depth edge band, removed from its valid mask before warping; None disables."""
    target_edge_fallback_px: int | None = None
    """Width of the target's own depth edge band, where the target's own depth replaces the fused one; None
    disables. Needs include_self."""

    def __post_init__(self) -> None:
        if not isinstance(self.window, (int, np.integer)) or self.window < 0:
            raise ValueError(f"window must be a non-negative integer, got {self.window!r}")
        for name, value, allowed in (
            ("mode", self.mode, MODES), ("selection", self.selection, SELECTIONS),
            ("aggregator", self.aggregator, AGGREGATORS),
        ):
            if value not in allowed:
                raise ValueError(f"{name} must be one of {allowed}, got {value!r}")
        if not self.stride_ms > 0:
            raise ValueError(f"stride_ms must be positive, got {self.stride_ms}")
        if not 0 <= self.stride_tolerance_ms < self.stride_ms / 2:
            raise ValueError(f"stride_tolerance_ms must be in [0, stride_ms / 2), got {self.stride_tolerance_ms}")
        if not 0 <= self.trim_fraction < 0.5:
            raise ValueError(f"trim_fraction must be in [0, 0.5), got {self.trim_fraction}")
        if not self.front_tolerance >= 0:
            raise ValueError(f"front_tolerance must be non-negative, got {self.front_tolerance}")
        if not isinstance(self.min_support, (int, np.integer)) or self.min_support < 1:
            raise ValueError(f"min_support must be an integer >= 1, got {self.min_support!r}")
        if not self.self_weight > 0:
            raise ValueError(f"self_weight must be positive, got {self.self_weight}")
        for name in ("source_edge_mask_px", "target_edge_fallback_px"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, (int, np.integer)) or value < 0):
                raise ValueError(f"{name} must be None or a non-negative integer, got {value!r}")
        if self.target_edge_fallback_px is not None and not self.include_self:
            raise ValueError("target_edge_fallback_px needs include_self, since the fallback is the target's own depth")


@dataclass(frozen=True)
class FusionResult:
    depth: np.ndarray
    """Grid-sized float32 fused depth, NaN where no candidate survived."""
    weight_sum: np.ndarray
    n_sources: np.ndarray
    source_indices: tuple[int, ...]
    """Indices into the frame list of the warped sources, in layer order after the self layer."""
    layers: np.ndarray | None = None
    """(K, h, w) candidate depths with the self layer first when included; only with keep_layers."""
    layer_weights: np.ndarray | None = None

    @property
    def hole(self) -> np.ndarray:
        return self.n_sources == 0


def scale_intrinsics(K: np.ndarray, scale: int) -> np.ndarray:
    # Pixel centres sit at integer coordinates, so native pixel x spans [x - 0.5, x + 0.5] and its
    # left edge lands at (x - 0.5) * s on the fine grid, giving cx' = (cx + 0.5) * s - 0.5.
    check_intrinsics(K)
    scaled = K.astype(np.float64).copy()
    scaled[0, 0] *= scale
    scaled[1, 1] *= scale
    scaled[0, 2] = (K[0, 2] + 0.5) * scale - 0.5
    scaled[1, 2] = (K[1, 2] + 0.5) * scale - 0.5
    return scaled


def check_pose(T: np.ndarray, name: str) -> None:
    if T.shape != (4, 4):
        raise ValueError(f"{name} must be 4x4, got {T.shape}")
    if not np.all(np.isfinite(T)):
        raise ValueError(f"{name} must be finite")


def check_intrinsics(K: np.ndarray) -> None:
    if K.shape != (3, 3):
        raise ValueError(f"K must be 3x3, got {K.shape}")
    if not (K[0, 0] > 0 and K[1, 1] > 0):
        raise ValueError(f"K must have positive focal lengths, got fx={K[0, 0]}, fy={K[1, 1]}")
