"""World frame point maps of GT, raw and fused depth from fusion_render_prepare.py, with JOINT-SOLVED scale and ORACLE
GT poses, for orbit and flythrough renders.

Every --stride th frame that has fused depth is back projected for all three sources on the same pixels, GT valid,
finite in both predictions and outside the depth_edge_band of each of the three depths, so flying pixels at edges are
dropped. Predictions use the RGB time pose and GT the depth time pose, as the consistency evaluation does. Each source
keeps at most --max-points points with their RGB colour, written as binary PLY in TUM world coordinates, plus
map.json with the camera path, an orbit framing and a ghosting statistic per source."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from export_gt_ply import write_ply  # noqa: E402
from render_video import FOOTER, POSES, SCALE, SequenceInputs  # noqa: E402

from pim.metrics.error_maps import depth_edge_band  # noqa: E402
from pim.metrics.geometry import backproject  # noqa: E402

SOURCES = ("gt", "raw", "fused")
DEFAULT_STRIDE = 5
MAX_POINTS = 300_000
VOXEL_M = 0.05
MIN_VOXEL_POINTS = 10
SEED = 0
KEY_BITS = 21
MAP_JSON = "map.json"


def frame_mask(gt_depth: np.ndarray, gt_valid: np.ndarray, raw: np.ndarray, fused: np.ndarray) -> np.ndarray:
    gt_valid = gt_valid.astype(bool)
    base = gt_valid & np.isfinite(raw) & np.isfinite(fused) & (raw > 0) & (fused > 0)
    band = depth_edge_band(gt_depth, gt_valid)
    for depth in (raw, fused):
        band |= depth_edge_band(np.where(base, depth, 0.0), base)
    return base & ~band


def world_points(depth: np.ndarray, K: np.ndarray, T_world_cam: np.ndarray, pixels: np.ndarray) -> np.ndarray:
    points, _ = backproject(depth, np.ones(depth.shape, dtype=bool), K)
    p = points.reshape(-1, 3)[pixels].astype(np.float64)
    return (p @ T_world_cam[:3, :3].T + T_world_cam[:3, 3]).astype(np.float32)


def read_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """(N, 3) float32 points and (N, 3) uint8 colours of a PLY written by export_gt_ply.write_ply."""
    data = Path(path).read_bytes()
    end = data.index(b"end_header\n") + len(b"end_header\n")
    header = data[:end].decode("ascii").splitlines()
    count = next(int(line.split()[2]) for line in header if line.startswith("element vertex"))
    dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    vertices = np.frombuffer(data, dtype=dtype, count=count, offset=end)
    points = np.stack([vertices[a] for a in "xyz"], axis=1)
    colors = np.stack([vertices[c] for c in ("red", "green", "blue")], axis=1)
    return points, colors


# Ghosting. Each GT voxel with at least MIN_VOXEL_POINTS points gets a plane through its centroid along the smallest
# principal axis, the local GT surface normal, oriented toward the mean camera centre. A source point's ghosting
# distance is its signed distance to the plane of the voxel it falls in, and the statistic is the spread of those
# distances. Points in voxels without a GT plane count as uncovered rather than as a distance.

@dataclass(frozen=True)
class VoxelPlanes:
    origin: np.ndarray
    voxel: float
    keys: np.ndarray
    """Sorted packed keys of the voxels with a plane."""
    centroid: np.ndarray
    normal: np.ndarray


def voxel_keys(points: np.ndarray, origin: np.ndarray, voxel: float) -> tuple[np.ndarray, np.ndarray]:
    """(packed int64 key per point, mask of points inside the key range)."""
    cells = np.floor((points.astype(np.float64) - origin) / voxel).astype(np.int64)
    inside = np.all((cells >= 0) & (cells < 1 << KEY_BITS), axis=1)
    cells = np.clip(cells, 0, (1 << KEY_BITS) - 1)
    return (cells[:, 0] << 2 * KEY_BITS) | (cells[:, 1] << KEY_BITS) | cells[:, 2], inside


def voxel_planes(gt: np.ndarray, toward: np.ndarray, voxel: float = VOXEL_M,
                 min_points: int = MIN_VOXEL_POINTS) -> VoxelPlanes:
    # Half a voxel of slack so a flat map lying exactly on its minimum is not split across a cell boundary.
    origin = gt.min(axis=0).astype(np.float64) - 1.5 * voxel
    keys, _ = voxel_keys(gt, origin, voxel)
    unique, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
    p = gt.astype(np.float64)
    sums = np.stack([np.bincount(inverse, p[:, a], len(unique)) for a in range(3)], axis=1)
    centroid = sums / counts[:, None]
    d = p - centroid[inverse]
    cov = np.empty((len(unique), 3, 3))
    for a in range(3):
        for b in range(a, 3):
            cov[:, a, b] = cov[:, b, a] = np.bincount(inverse, d[:, a] * d[:, b], len(unique)) / counts
    keep = counts >= min_points
    normal = np.linalg.eigh(cov[keep])[1][:, :, 0]
    flip = np.einsum("ij,ij->i", toward - centroid[keep], normal) < 0
    normal[flip] *= -1
    return VoxelPlanes(origin, voxel, unique[keep], centroid[keep], normal)


def surface_distance(points: np.ndarray, planes: VoxelPlanes) -> np.ndarray:
    """Signed distance of each point to its voxel's GT plane, positive toward the cameras, NaN without a plane."""
    keys, inside = voxel_keys(points, planes.origin, planes.voxel)
    slot = np.clip(np.searchsorted(planes.keys, keys), 0, max(len(planes.keys) - 1, 0))
    found = inside & (planes.keys[slot] == keys) if len(planes.keys) else np.zeros(len(points), dtype=bool)
    distance = np.full(len(points), np.nan)
    offset = points[found].astype(np.float64) - planes.centroid[slot[found]]
    distance[found] = np.einsum("ij,ij->i", offset, planes.normal[slot[found]])
    return distance


def ghosting(points: np.ndarray, planes: VoxelPlanes) -> dict[str, float | None]:
    distance = surface_distance(points, planes)
    found = distance[np.isfinite(distance)]
    covered = round(found.size / len(points), 4) if len(points) else 0.0
    if not found.size:
        return {"std_m": None, "median_abs_m": None, "mean_signed_m": None, "covered_fraction": covered}
    return {
        "std_m": round(float(np.std(found)), 5),
        "median_abs_m": round(float(np.median(np.abs(found))), 5),
        "mean_signed_m": round(float(np.mean(found)), 5),
        "covered_fraction": covered,
    }


# Command line.

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("render_dir", type=Path, help="fusion/render/<run name>, the folder holding sequence folders")
    parser.add_argument("data_root", type=Path, help="folder containing the sequence folders")
    parser.add_argument("sequences_config", type=Path)
    parser.add_argument("sequence")
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE, help="every stride th frame with fused depth")
    parser.add_argument("--max-points", type=int, default=MAX_POINTS, help="per source")
    parser.add_argument("--voxel", type=float, default=VOXEL_M, help="ghosting voxel edge in metres")
    parser.add_argument("--max-frames", type=int)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, object]:
    args = parse_args(argv)
    inputs = SequenceInputs(args.render_dir, args.data_root, args.sequences_config, args.sequence, args.max_frames)
    fused_positions = []
    for position, path in enumerate(inputs.paths):
        with np.load(path) as data:
            if bool(data["has_fused"]):
                fused_positions.append(position)
    chosen = fused_positions[::args.stride]
    if not chosen:
        raise ValueError(f"{args.sequence} has no frame with fused depth")
    per_frame = math.ceil(args.max_points / len(chosen))
    rng = np.random.default_rng(SEED)
    parts: dict[str, list[np.ndarray]] = {s: [] for s in SOURCES}
    colors = []
    for position in chosen:
        loaded = inputs.load(position, with_rgb=True)
        raw, fused = loaded.raw, loaded.fused
        mask = frame_mask(raw.gt_depth, raw.gt_valid, raw.depth, fused.depth)
        pixels = np.flatnonzero(mask)
        pixels = np.sort(rng.choice(pixels, min(per_frame, pixels.size), replace=False))
        parts["gt"].append(world_points(raw.gt_depth, inputs.K, raw.T_depth, pixels))
        parts["raw"].append(world_points(raw.depth, inputs.K, raw.T_rgb, pixels))
        parts["fused"].append(world_points(fused.depth, inputs.K, fused.T_rgb, pixels))
        colors.append(loaded.rgb.reshape(-1, 3)[pixels])
    color = np.concatenate(colors)
    keep = np.sort(rng.choice(len(color), args.max_points, replace=False)) if len(color) > args.max_points else None
    points = {s: np.concatenate(parts[s]) for s in SOURCES}
    if keep is not None:
        points, color = {s: p[keep] for s, p in points.items()}, color[keep]

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    for source in SOURCES:
        files[source] = out / f"{source}.ply"
        write_ply(files[source], points[source], color)

    path = [inputs.frames[p].T_world_cam for p in range(len(inputs))]
    centres = np.array([T[:3, 3] for T in path])
    planes = voxel_planes(points["gt"], centres.mean(axis=0), args.voxel)
    up = -np.mean([T[:3, 1] for T in path], axis=0)
    up /= np.linalg.norm(up)
    centre = np.median(points["gt"], axis=0)
    radius = float(np.percentile(np.linalg.norm(points["gt"] - centre, axis=1), 90))
    record = {
        "scale": SCALE,
        "poses": POSES,
        "label": FOOTER,
        "sequence": args.sequence,
        "coordinates": "TUM world frame in metres, cameras in the OpenCV convention",
        "settings": {**{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
                     "min_voxel_points": MIN_VOXEL_POINTS, "seed": SEED,
                     "pixels": "GT valid, finite predictions, outside depth_edge_band of GT, raw and fused"},
        "frames_used": [inputs.frames[p].index for p in chosen],
        "points": {s: int(len(points[s])) for s in SOURCES},
        "files": {s: files[s].name for s in SOURCES},
        "ghosting": {s: ghosting(points[s], planes) for s in SOURCES},
        "ghosting_rule": f"signed distance to the GT plane of the {args.voxel:g} m voxel a point falls in, the plane "
                         f"fitted by PCA to at least {MIN_VOXEL_POINTS} GT map points; std over covered points",
        "camera": {"K": inputs.K.tolist(), "image_size": [640, 480],
                   "path_indices": [f.index for f in inputs.frames],
                   "path_timestamps": [f.timestamp for f in inputs.frames],
                   "path_T_world_cam": [np.round(T, 6).tolist() for T in path]},
        "orbit": {"centre": centre.round(4).tolist(), "up": up.round(6).tolist(), "radius_m": round(radius, 4),
                  "start_direction": (centres.mean(axis=0) - centre).round(4).tolist()},
    }
    (out / MAP_JSON).write_text(json.dumps(record, indent=2) + "\n")
    for source in SOURCES:
        g = record["ghosting"][source]
        print(f"{source} {record['points'][source]} points -> {files[source]}, ghosting std {g['std_m']} m, "
              f"covered {g['covered_fraction']}")
    print(f"wrote {out / MAP_JSON}")
    return record


if __name__ == "__main__":
    main()
