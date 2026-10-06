"""TUM RGB-D sequence discovery, association, ground-truth poses and frame loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

DEPTH_SCALE = 5000.0
MAX_ASSOCIATION_DT = 0.02
MAX_GT_GAP = 0.05
# Differences of ~1.3e9 s epoch timestamps carry ~2e-7 s of float64 error, so a gap
# written as exactly 0.0500 s in the file can compute as 0.0500002 s.
TIMESTAMP_EPS = 1e-6


@dataclass(frozen=True)
class ListedImage:
    timestamp: float
    path: Path


@dataclass(frozen=True)
class Frame:
    index: int
    timestamp: float
    rgb_path: Path
    depth_path: Path
    depth_offset: float
    """Depth timestamp minus RGB timestamp, in seconds."""
    T_world_cam: np.ndarray

    def load_rgb(self) -> np.ndarray:
        return load_rgb(self.rgb_path)

    def load_depth(self) -> tuple[np.ndarray, np.ndarray]:
        return load_depth(self.depth_path)


@dataclass(frozen=True)
class Window:
    start: int | None
    count: int | None


@dataclass(frozen=True)
class TUMSequence:
    name: str
    root: Path
    K: np.ndarray
    window: Window
    frames: list[Frame]


def find_sequences(root: Path) -> list[Path]:
    """Every directory under `root` that holds a TUM `rgb.txt`, sorted by path."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"dataset root {root} does not exist")
    sequences = sorted(p.parent for p in root.rglob("rgb.txt"))
    if not sequences:
        raise FileNotFoundError(f"no TUM sequence (directory containing rgb.txt) under {root}")
    return sequences


def read_rgb_list(sequence: Path) -> list[ListedImage]:
    return _read_image_list(Path(sequence), "rgb.txt")


def read_depth_list(sequence: Path) -> list[ListedImage]:
    return _read_image_list(Path(sequence), "depth.txt")


def _read_image_list(sequence: Path, listing_name: str) -> list[ListedImage]:
    listing = sequence / listing_name
    images = [
        ListedImage(float(fields[0]), sequence / fields[1])
        for fields in _data_lines(listing, min_fields=2)
    ]
    if not images:
        raise ValueError(f"{listing} lists no images")
    return images


def read_groundtruth(sequence: Path) -> tuple[np.ndarray, np.ndarray]:
    """(N,) timestamps and (N, 7) poses as tx ty tz qx qy qz qw, in file order."""
    listing = Path(sequence) / "groundtruth.txt"
    rows = np.array([[float(v) for v in fields[:8]] for fields in _data_lines(listing, min_fields=8)])
    if len(rows) < 2:
        raise ValueError(f"{listing} needs at least two poses, found {len(rows)}")
    times = rows[:, 0]
    if np.any(np.diff(times) < 0):
        raise ValueError(f"{listing} timestamps are not sorted")
    return times, rows[:, 1:]


def _data_lines(listing: Path, min_fields: int) -> list[list[str]]:
    if not listing.is_file():
        raise FileNotFoundError(f"{listing} not found")
    lines = []
    for number, line in enumerate(listing.read_text().splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < min_fields:
            raise ValueError(f"{listing}:{number}: expected {min_fields} fields, got {line!r}")
        lines.append(fields)
    return lines


def associate(
    rgb: list[ListedImage], depth: list[ListedImage], max_dt: float = MAX_ASSOCIATION_DT
) -> list[tuple[ListedImage, ListedImage]]:
    # TUM's associate.py rule: take every candidate pair within max_dt, accept them
    # greedily from the smallest time difference up, and use each image at most once.
    depth_order = np.argsort([d.timestamp for d in depth])
    depth_times = np.array([depth[j].timestamp for j in depth_order])
    candidates = []
    for i, r in enumerate(rgb):
        lo = np.searchsorted(depth_times, r.timestamp - max_dt - TIMESTAMP_EPS, side="left")
        hi = np.searchsorted(depth_times, r.timestamp + max_dt + TIMESTAMP_EPS, side="right")
        for k in range(lo, hi):
            dt = abs(depth_times[k] - r.timestamp)
            if dt <= max_dt + TIMESTAMP_EPS:
                candidates.append((dt, i, int(depth_order[k])))
    candidates.sort()

    used_rgb: set[int] = set()
    used_depth: set[int] = set()
    pairs = []
    for _, i, j in candidates:
        if i in used_rgb or j in used_depth:
            continue
        used_rgb.add(i)
        used_depth.add(j)
        pairs.append((rgb[i], depth[j]))
    return sorted(pairs, key=lambda pair: pair[0].timestamp)


def quaternion_to_rotation(q: np.ndarray) -> np.ndarray:
    """3x3 rotation from a quaternion in (x, y, z, w) order."""
    x, y, z, w = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def slerp(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    q0 = np.asarray(q0, dtype=np.float64) / np.linalg.norm(q0)
    q1 = np.asarray(q1, dtype=np.float64) / np.linalg.norm(q1)
    dot = float(np.dot(q0, q1))
    # q and -q are the same rotation; flip one so we interpolate along the shorter arc.
    if dot < 0.0:
        q1, dot = -q1, -dot
    # Nearly parallel: sin(theta) -> 0, so fall back to normalised linear interpolation.
    if dot > 0.9995:
        q = q0 + alpha * (q1 - q0)
        return q / np.linalg.norm(q)
    theta = np.arccos(dot)
    return (np.sin((1 - alpha) * theta) * q0 + np.sin(alpha * theta) * q1) / np.sin(theta)


def interpolate_pose(
    gt_times: np.ndarray, gt_poses: np.ndarray, t: float, max_gap: float = MAX_GT_GAP
) -> np.ndarray | None:
    """4x4 T_world_cam at time t, or None if t is outside the GT range or across a gap."""
    if t < gt_times[0] or t > gt_times[-1]:
        return None
    i0 = min(int(np.searchsorted(gt_times, t, side="right")) - 1, len(gt_times) - 2)
    i1 = i0 + 1
    gap = gt_times[i1] - gt_times[i0]
    if gap > max_gap + TIMESTAMP_EPS:
        return None
    alpha = (t - gt_times[i0]) / gap if gap > 0 else 0.0
    pose = np.eye(4)
    pose[:3, 3] = (1 - alpha) * gt_poses[i0, :3] + alpha * gt_poses[i1, :3]
    pose[:3, :3] = quaternion_to_rotation(slerp(gt_poses[i0, 3:], gt_poses[i1, 3:], alpha))
    return pose


def load_rgb(path: Path) -> np.ndarray:
    """An HxWx3 uint8 RGB image."""
    with Image.open(path) as image:
        return np.array(image.convert("RGB"), dtype=np.uint8)


def load_depth(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """(HxW float32 depth in metres, 0.0 where missing; HxW bool valid mask)."""
    with Image.open(path) as image:
        if image.mode not in ("I;16", "I;16B", "I"):
            raise ValueError(f"{path}: expected a 16-bit depth PNG, got mode {image.mode}")
        raw = np.array(image)
    valid = raw > 0
    depth = np.where(valid, raw.astype(np.float32) / DEPTH_SCALE, np.float32(0.0))
    return depth.astype(np.float32), valid


def load_camera_config(config_path: Path, sequence_name: str) -> tuple[np.ndarray, Window]:
    config_path = Path(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"config {config_path} not found")
    config = yaml.safe_load(config_path.read_text())
    sequences = config.get("sequences") or {}
    if sequence_name not in sequences:
        raise KeyError(f"sequence {sequence_name!r} is not in {config_path}; known: {sorted(sequences)}")
    entry = sequences[sequence_name]
    camera_name = entry.get("camera")
    cameras = config.get("cameras") or {}
    if camera_name not in cameras:
        raise KeyError(f"camera {camera_name!r} of {sequence_name} is not in {config_path}")
    camera = cameras[camera_name]
    missing = [key for key in ("fx", "fy", "cx", "cy") if key not in camera]
    if missing:
        raise KeyError(f"camera {camera_name!r} in {config_path} is missing {missing}")
    K = np.array([
        [camera["fx"], 0.0, camera["cx"]],
        [0.0, camera["fy"], camera["cy"]],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    window = entry.get("window") or {}
    return K, Window(window.get("start"), window.get("count"))


def build_frames(
    pairs: list[tuple[ListedImage, ListedImage]], gt_times: np.ndarray, gt_poses: np.ndarray
) -> list[Frame]:
    frames = []
    for rgb, depth in pairs:
        pose = interpolate_pose(gt_times, gt_poses, rgb.timestamp)
        if pose is None:
            continue
        frames.append(Frame(len(frames), rgb.timestamp, rgb.path, depth.path,
                            depth.timestamp - rgb.timestamp, pose))
    return frames


def apply_window(sequence: TUMSequence) -> list[Frame]:
    """The frames of `sequence.window`, or all frames if start and count are both null."""
    start, count = sequence.window.start, sequence.window.count
    if start is None and count is None:
        return list(sequence.frames)
    if start is None or count is None:
        raise ValueError(f"{sequence.name}: window needs both start and count or neither, "
                         f"got start={start}, count={count}")
    for key, value in (("start", start), ("count", count)):
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"{sequence.name}: window {key} must be an integer, got {value!r}")
    if start < 0 or count < 1 or start + count > len(sequence.frames):
        raise ValueError(f"{sequence.name}: window start={start}, count={count} does not fit "
                         f"in {len(sequence.frames)} frames")
    return sequence.frames[start:start + count]


def load_sequence(root: Path, config_path: Path) -> TUMSequence:
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"sequence directory {root} does not exist")
    K, window = load_camera_config(config_path, root.name)
    pairs = associate(read_rgb_list(root), read_depth_list(root))
    gt_times, gt_poses = read_groundtruth(root)
    frames = build_frames(pairs, gt_times, gt_poses)
    if not frames:
        raise ValueError(f"{root}: no frame has both an associated depth image and a ground-truth pose")
    missing = [p for f in frames for p in (f.rgb_path, f.depth_path) if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"{root}: {len(missing)} listed images are missing, e.g. {missing[0]}")
    return TUMSequence(root.name, root, K, window, frames)
