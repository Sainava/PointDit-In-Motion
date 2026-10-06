"""Summarise one TUM sequence and save RGB-depth overlays and a trajectory plot."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pim.data.tum import (  # noqa: E402
    Frame,
    TUMSequence,
    associate,
    load_sequence,
    read_depth_list,
    read_rgb_list,
)

DEPTH_SAMPLES = 20
OVERLAY_SAMPLES = 3
EDGE_RELATIVE_JUMP = 0.04


def evenly_spaced(frames: list[Frame], count: int) -> list[Frame]:
    indices = np.linspace(0, len(frames) - 1, min(count, len(frames))).round().astype(int)
    return [frames[i] for i in indices]


def depth_stats(frames: list[Frame]) -> tuple[float, np.ndarray]:
    valid_fractions = []
    valid_depths = []
    for frame in evenly_spaced(frames, DEPTH_SAMPLES):
        depth, valid = frame.load_depth()
        valid_fractions.append(valid.mean())
        valid_depths.append(depth[valid])
    return float(np.mean(valid_fractions)), np.percentile(np.concatenate(valid_depths), [5, 50, 95])


def rotation_angle_deg(R: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))))


def trajectory_stats(frames: list[Frame]) -> tuple[float, float, float]:
    poses = [f.T_world_cam for f in frames]
    steps = [np.linalg.norm(b[:3, 3] - a[:3, 3]) for a, b in zip(poses, poses[1:])]
    turns = [rotation_angle_deg(a[:3, :3].T @ b[:3, :3]) for a, b in zip(poses, poses[1:])]
    return float(np.sum(steps)), float(np.max(steps, initial=0.0)), float(np.max(turns, initial=0.0))


def depth_edges(depth: np.ndarray, valid: np.ndarray) -> np.ndarray:
    edges = np.zeros_like(valid)
    # Relative jumps, so small objects on a desk at 0.6 m show up as well as walls at 4 m.
    jump_x = valid[:, 1:] & valid[:, :-1] & (
        np.abs(np.diff(depth, axis=1)) > EDGE_RELATIVE_JUMP * np.minimum(depth[:, 1:], depth[:, :-1]))
    jump_y = valid[1:, :] & valid[:-1, :] & (
        np.abs(np.diff(depth, axis=0)) > EDGE_RELATIVE_JUMP * np.minimum(depth[1:, :], depth[:-1, :]))
    edges[:, 1:] |= jump_x
    edges[:, :-1] |= jump_x
    edges[1:, :] |= jump_y
    edges[:-1, :] |= jump_y
    return edges


def save_overlays(frames: list[Frame], out_dir: Path) -> None:
    for frame in evenly_spaced(frames, OVERLAY_SAMPLES):
        image = frame.load_rgb()
        depth, valid = frame.load_depth()
        if image.shape[:2] != depth.shape:
            raise ValueError(f"{frame.rgb_path} {image.shape[:2]} and {frame.depth_path} {depth.shape} differ")
        image[depth_edges(depth, valid)] = (255, 0, 0)
        Image.fromarray(image).save(out_dir / f"overlay_{frame.index:04d}.png")


def save_trajectory(sequence: TUMSequence, path: Path) -> None:
    positions = np.array([f.T_world_cam[:3, 3] for f in sequence.frames])
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (a, b) in zip(axes, [(0, 1), (0, 2), (1, 2)]):
        ax.plot(positions[:, a], positions[:, b], linewidth=1)
        ax.scatter(*positions[0, [a, b]], color="green", label="start", zorder=3)
        ax.scatter(*positions[-1, [a, b]], color="red", label="end", zorder=3)
        ax.set_xlabel(f"{'xyz'[a]} (m)")
        ax.set_ylabel(f"{'xyz'[b]} (m)")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.3)
    axes[0].legend()
    fig.suptitle(f"{sequence.name}: camera positions (world frame)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=Path, required=True, help="TUM sequence directory")
    parser.add_argument("--config", type=Path, default=Path("configs/sequences.yaml"))
    parser.add_argument("--out", type=Path, default=Path("outputs/inspect"))
    args = parser.parse_args()

    rgb = read_rgb_list(args.sequence)
    associated = associate(rgb, read_depth_list(args.sequence))
    sequence = load_sequence(args.sequence, args.config)
    frames = sequence.frames
    offsets = np.abs([f.depth_offset for f in frames])
    valid_fraction, percentiles = depth_stats(frames)
    length, max_step, max_turn = trajectory_stats(frames)

    out_dir = args.out / sequence.name
    out_dir.mkdir(parents=True, exist_ok=True)
    save_overlays(frames, out_dir)
    save_trajectory(sequence, out_dir / "trajectory.png")

    duration = frames[-1].timestamp - frames[0].timestamp
    print(f"sequence: {sequence.name}  (fx={sequence.K[0, 0]}, fy={sequence.K[1, 1]}, "
          f"cx={sequence.K[0, 2]}, cy={sequence.K[1, 2]})")
    print(f"frames: rgb.txt {len(rgb)} -> associated {len(associated)} -> with pose {len(frames)}  "
          f"({duration:.1f} s)")
    print(f"|depth - rgb| offset: max {offsets.max() * 1e3:.1f} ms, mean {offsets.mean() * 1e3:.1f} ms")
    print(f"depth over {min(DEPTH_SAMPLES, len(frames))} frames: valid {valid_fraction:.1%}, "
          f"p5/p50/p95 {percentiles[0]:.2f}/{percentiles[1]:.2f}/{percentiles[2]:.2f} m")
    print(f"trajectory: length {length:.2f} m, max step {max_step * 100:.1f} cm, "
          f"max rotation step {max_turn:.2f} deg")
    print(f"saved to {out_dir}")


if __name__ == "__main__":
    main()
