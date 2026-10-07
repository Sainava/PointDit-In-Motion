"""Per-frame covariates for failure analysis: camera motion, image sharpness and GT scene geometry."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from pim.data.tum import Frame, interpolate_pose
from pim.metrics.error_maps import depth_edge_band

MOTION_HALF_WINDOW_S = 0.02
SHARPNESS_NEIGHBOURHOOD_S = 0.5
BORDER_FRACTION = 0.1
NEAR_DEPTH_FACTOR = 0.6
GRAY_WEIGHTS = np.array([0.299, 0.587, 0.114])

COLUMNS = (
    "sequence", "frame_index", "rgb_timestamp", "rgb_file", "depth_offset_ms",
    "angular_speed_dps", "linear_speed_mps", "image_motion_px_per_s", "expected_misalignment_px",
    "sharpness", "sharpness_rel",
    "depth_p5", "depth_p50", "depth_p95", "depth_range_ratio", "edge_density", "near_border_fraction",
)


def rotation_angle(R: np.ndarray) -> float:
    return float(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)))


def camera_motion(
    gt_times: np.ndarray, gt_poses: np.ndarray, t: float, h: float = MOTION_HALF_WINDOW_S
) -> tuple[float, float]:
    """(angular speed in deg/s, linear speed in m/s) from the GT poses at t - h and t + h; NaN if either is unavailable."""
    if h <= 0:
        raise ValueError(f"h must be positive, got {h}")
    before = interpolate_pose(gt_times, gt_poses, t - h)
    after = interpolate_pose(gt_times, gt_poses, t + h)
    if before is None or after is None:
        return math.nan, math.nan
    relative = np.linalg.inv(before) @ after
    angle = rotation_angle(relative[:3, :3])
    distance = float(np.linalg.norm(relative[:3, 3]))
    return math.degrees(angle) / (2 * h), distance / (2 * h)


def image_motion_px_per_s(angular_speed_dps: float, linear_speed_mps: float, fx: float, median_depth: float) -> float:
    # Order-of-magnitude estimate: rotation moves pixels by ~omega*fx and translation by ~v*fx/z,
    # summed as if both moved the image the same way and translation were parallel to the image plane.
    return math.radians(angular_speed_dps) * fx + linear_speed_mps * fx / median_depth


def grayscale(rgb: np.ndarray) -> np.ndarray:
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"expected an HxWx3 RGB image, got {rgb.shape}")
    if rgb.dtype != np.uint8:
        raise ValueError(f"expected uint8 RGB, got {rgb.dtype}")
    return rgb.astype(np.float64) @ GRAY_WEIGHTS / 255.0


def sharpness(rgb: np.ndarray) -> float:
    """Variance of the 3x3 Laplacian of the [0, 1] grayscale image, over interior pixels."""
    g = grayscale(rgb)
    if min(g.shape) < 3:
        raise ValueError(f"image {g.shape} is too small for a 3x3 Laplacian")
    laplacian = g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:] - 4.0 * g[1:-1, 1:-1]
    return float(laplacian.var())


def relative_sharpness(
    timestamps: np.ndarray, values: np.ndarray, radius_s: float = SHARPNESS_NEIGHBOURHOOD_S
) -> np.ndarray:
    """Each value divided by the median of the values (itself included) within radius_s seconds."""
    timestamps, values = np.asarray(timestamps, dtype=np.float64), np.asarray(values, dtype=np.float64)
    if timestamps.shape != values.shape or timestamps.ndim != 1:
        raise ValueError(f"timestamps {timestamps.shape} and values {values.shape} must be matching 1-D arrays")
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError("timestamps must be strictly increasing")
    lo = np.searchsorted(timestamps, timestamps - radius_s, side="left")
    hi = np.searchsorted(timestamps, timestamps + radius_s, side="right")
    medians = np.array([np.median(values[a:b]) for a, b in zip(lo, hi)])
    return values / medians


def depth_geometry(depth: np.ndarray, valid: np.ndarray) -> dict[str, float]:
    valid = valid.astype(bool)
    if depth.shape != valid.shape or depth.ndim != 2:
        raise ValueError(f"depth {depth.shape} and valid {valid.shape} must be matching HxW arrays")
    if not valid.any():
        return dict.fromkeys(
            ("depth_p5", "depth_p50", "depth_p95", "depth_range_ratio", "edge_density", "near_border_fraction"),
            math.nan)
    p5, p50, p95 = (float(v) for v in np.percentile(depth[valid], [5, 50, 95]))
    return {
        "depth_p5": p5,
        "depth_p50": p50,
        "depth_p95": p95,
        "depth_range_ratio": p95 / p5,
        "edge_density": float(depth_edge_band(depth, valid).sum() / valid.sum()),
        "near_border_fraction": near_border_fraction(depth, valid),
    }


def border_strip(shape: tuple[int, int], fraction: float = BORDER_FRACTION) -> np.ndarray:
    h, w = shape
    rows, cols = max(1, round(fraction * h)), max(1, round(fraction * w))
    strip = np.zeros(shape, dtype=bool)
    strip[:rows] = strip[-rows:] = True
    strip[:, :cols] = strip[:, -cols:] = True
    return strip


def near_border_fraction(
    depth: np.ndarray, valid: np.ndarray, fraction: float = BORDER_FRACTION, factor: float = NEAR_DEPTH_FACTOR
) -> float:
    # Heuristic for a near object cut off by the image border: valid pixels in the outer strip
    # much closer than the frame's median depth.
    valid = valid.astype(bool)
    strip = border_strip(depth.shape, fraction) & valid
    if not strip.any():
        return math.nan
    return float((depth[strip] < factor * np.median(depth[valid])).mean())


def frame_covariates(
    sequence: str, frame: Frame, rgb_file: str, fx: float, gt_times: np.ndarray, gt_poses: np.ndarray
) -> dict[str, Any]:
    rgb = frame.load_rgb()
    depth, valid = frame.load_depth()
    angular, linear = camera_motion(gt_times, gt_poses, frame.timestamp)
    geometry = depth_geometry(depth, valid)
    motion_px = image_motion_px_per_s(angular, linear, fx, geometry["depth_p50"])
    return {
        "sequence": sequence,
        "frame_index": frame.index,
        "rgb_timestamp": f"{frame.timestamp:.6f}",
        "rgb_file": rgb_file,
        "depth_offset_ms": abs(frame.depth_offset) * 1000.0,
        "angular_speed_dps": angular,
        "linear_speed_mps": linear,
        "image_motion_px_per_s": motion_px,
        "expected_misalignment_px": motion_px * abs(frame.depth_offset),
        "sharpness": sharpness(rgb),
        **geometry,
    }


def add_relative_sharpness(rows: list[dict[str, Any]]) -> None:
    times = np.array([float(r["rgb_timestamp"]) for r in rows])
    relative = relative_sharpness(times, np.array([r["sharpness"] for r in rows]))
    for row, value in zip(rows, relative):
        row["sharpness_rel"] = float(value)
