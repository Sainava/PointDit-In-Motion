"""Forward warping of depth and colour between posed frames, and pairing of frames by time gap."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from pim.data.tum import Frame, interpolate_pose
from pim.metrics.geometry import backproject, project


def relative_pose(T_world_cam_i: np.ndarray, T_world_cam_j: np.ndarray) -> np.ndarray:
    """T_j_from_i: maps points in camera i's frame into camera j's frame."""
    for name, T in (("T_world_cam_i", T_world_cam_i), ("T_world_cam_j", T_world_cam_j)):
        if T.shape != (4, 4):
            raise ValueError(f"{name} must be 4x4, got {T.shape}")
    return np.linalg.inv(T_world_cam_j) @ T_world_cam_i


def depth_pose(frame: Frame, gt_times: np.ndarray, gt_poses: np.ndarray) -> np.ndarray:
    # frame.T_world_cam is the pose at the RGB timestamp, which is what the model sees and what M1/M2 use.
    # The GT depth image was captured at the depth timestamp (10-18 ms later on fr1), so warping GT depth
    # with the RGB-time pose adds the camera motion in between to the GT floor and to visibility.
    t = frame.timestamp + frame.depth_offset
    pose = interpolate_pose(gt_times, gt_poses, t)
    if pose is None:
        raise ValueError(f"frame {frame.index}: no ground-truth pose at its depth timestamp {t:.6f}")
    return pose


def warp_depth(
    depth_i: np.ndarray, valid_i: np.ndarray, K: np.ndarray, T_j_from_i: np.ndarray, shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """(HxW depth of frame i's points seen from camera j, 0 where not hit; HxW hit mask)."""
    depth, hit, _ = _forward_warp(depth_i, valid_i, K, T_j_from_i, shape)
    return depth, hit


def warp_image(
    rgb_i: np.ndarray, depth_i: np.ndarray, valid_i: np.ndarray, K: np.ndarray, T_j_from_i: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """(HxWx3 colours of frame i's points seen from camera j, 0 where not hit; HxW hit mask)."""
    if rgb_i.shape[:2] != depth_i.shape or rgb_i.ndim != 3:
        raise ValueError(f"rgb_i must be HxWxC matching depth_i {depth_i.shape}, got {rgb_i.shape}")
    _, hit, source = _forward_warp(depth_i, valid_i, K, T_j_from_i, depth_i.shape)
    warped = np.zeros_like(rgb_i)
    warped[hit] = rgb_i.reshape(-1, rgb_i.shape[2])[source[hit]]
    return warped, hit


def _forward_warp(
    depth_i: np.ndarray, valid_i: np.ndarray, K: np.ndarray, T_j_from_i: np.ndarray, shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if T_j_from_i.shape != (4, 4):
        raise ValueError(f"T_j_from_i must be 4x4, got {T_j_from_i.shape}")
    if len(shape) != 2 or min(shape) < 1:
        raise ValueError(f"shape must be (H, W), got {shape}")
    points, valid = backproject(depth_i, valid_i, K)
    source = np.flatnonzero(valid)
    p = points.reshape(-1, 3)[source].astype(np.float64)
    p = p @ T_j_from_i[:3, :3].T + T_j_from_i[:3, 3]
    in_front = p[:, 2] > 0
    source, p = source[in_front], p[in_front]
    uv, z = project(p, K)
    u, v = np.rint(uv[:, 0]), np.rint(uv[:, 1])
    height, width = shape
    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    source, z = source[inside], z[inside]
    target = v[inside].astype(np.int64) * width + u[inside].astype(np.int64)

    # Z-buffer: sort by target pixel, then by depth, and keep the first (nearest) point of each pixel.
    order = np.lexsort((z, target))
    target, z, source = target[order], z[order], source[order]
    first = np.ones(target.size, dtype=bool)
    first[1:] = target[1:] != target[:-1]

    depth = np.zeros(height * width, dtype=np.float32)
    hit = np.zeros(height * width, dtype=bool)
    winner = np.full(height * width, -1, dtype=np.int64)
    depth[target[first]] = z[first]
    hit[target[first]] = True
    winner[target[first]] = source[first]
    return depth.reshape(shape), hit.reshape(shape), winner.reshape(shape)


def pairs_by_gap(frames: Sequence[Frame], gap_ms: float, tolerance_ms: float = 10.0) -> list[tuple[Frame, Frame]]:
    """(frame i, frame j) where j is the frame closest in time to t_i + gap_ms, kept only if within tolerance_ms."""
    if not np.isfinite(gap_ms):
        raise ValueError(f"gap_ms must be finite, got {gap_ms}")
    if not tolerance_ms >= 0:
        raise ValueError(f"tolerance_ms must be non-negative, got {tolerance_ms}")
    if not frames:
        return []
    times = np.array([f.timestamp for f in frames], dtype=np.float64)
    if np.any(np.diff(times) <= 0):
        raise ValueError("frame timestamps must be strictly increasing")
    targets = times + gap_ms / 1000.0
    right = np.clip(np.searchsorted(times, targets), 1, len(times) - 1) if len(times) > 1 else np.zeros(len(times), int)
    left = np.maximum(right - 1, 0)
    closest = np.where(np.abs(times[left] - targets) <= np.abs(times[right] - targets), left, right)
    within = np.abs(times[closest] - targets) <= tolerance_ms / 1000.0
    return [(frames[i], frames[int(closest[i])]) for i in np.flatnonzero(within)]
