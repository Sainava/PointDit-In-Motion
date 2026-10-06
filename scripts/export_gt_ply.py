"""Write the ground-truth point cloud of one TUM frame as a binary PLY for the fork's ply_viewer.html."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pim.data.tum import load_sequence  # noqa: E402
from pim.metrics.geometry import backproject  # noqa: E402

# ply_viewer.html expects the fork's PLY convention: OpenGL camera axes (Y up, looking down -Z).
OPENCV_TO_OPENGL = np.array([1.0, -1.0, -1.0], dtype=np.float32)


def write_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    vertices = np.empty(len(points), dtype=[
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ])
    for i, axis in enumerate("xyz"):
        vertices[axis] = points[:, i]
    for i, channel in enumerate(("red", "green", "blue")):
        vertices[channel] = colors[:, i]
    header = "\n".join([
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {len(vertices)}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header",
    ]) + "\n"
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(vertices.tobytes())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=Path, required=True, help="TUM sequence directory")
    parser.add_argument("--config", type=Path, default=Path("configs/sequences.yaml"))
    parser.add_argument("--frame", type=int, required=True, help="frame index after association and pose filtering")
    parser.add_argument("--out", type=Path, default=Path("outputs/ply"))
    args = parser.parse_args()

    sequence = load_sequence(args.sequence, args.config)
    if not 0 <= args.frame < len(sequence.frames):
        raise IndexError(f"--frame {args.frame} out of range; {sequence.name} has {len(sequence.frames)} frames")
    frame = sequence.frames[args.frame]
    depth, valid = frame.load_depth()
    image = frame.load_rgb()
    if image.shape[:2] != depth.shape:
        raise ValueError(f"{frame.rgb_path} {image.shape[:2]} and {frame.depth_path} {depth.shape} differ")
    points, valid = backproject(depth, valid, sequence.K)

    out_dir = args.out / sequence.name
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"gt_{frame.index:04d}.ply"
    write_ply(path, points[valid] * OPENCV_TO_OPENGL, image[valid])
    print(f"{sequence.name} frame {frame.index} ({frame.rgb_path.name}): {int(valid.sum())} points -> {path}")


if __name__ == "__main__":
    main()
