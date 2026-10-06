"""Save RGB | predicted depth | GT depth panels for cached predictions, for visual spot checks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pim.data.tum import Frame, load_sequence  # noqa: E402

PERCENTILES = (2, 98)


def colorize(depth: np.ndarray, valid: np.ndarray, label: str) -> tuple[np.ndarray, float, float]:
    """RGB plasma image normalised to the PERCENTILES of valid pixels; invalid pixels black."""
    if not valid.any():
        raise ValueError(f"{label}: no valid pixels to normalise")
    lo, hi = np.percentile(depth[valid], PERCENTILES)
    scaled = np.clip((depth - lo) / max(hi - lo, 1e-12), 0.0, 1.0)
    image = matplotlib.colormaps["plasma"](scaled)[..., :3]
    image[~valid] = 0.0
    return image, float(lo), float(hi)


def load_prediction(path: Path, frame: Frame, sequence_root: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"prediction {path} not found")
    with np.load(path) as data:
        rgb_file = str(data["rgb_file"])
        points = data["points"].astype(np.float32)
        valid = data["valid"]
    expected = frame.rgb_path.relative_to(sequence_root).as_posix()
    if rgb_file != expected:
        raise ValueError(f"{path} was predicted from {rgb_file}, but frame {frame.index} is {expected}")
    return points, valid


def save_panel(frame: Frame, points: np.ndarray, pred_valid: np.ndarray, path: Path) -> None:
    rgb = frame.load_rgb()
    gt_depth, gt_valid = frame.load_depth()
    if points.shape[:2] != gt_depth.shape or rgb.shape[:2] != gt_depth.shape:
        raise ValueError(f"frame {frame.index}: rgb {rgb.shape[:2]}, prediction {points.shape[:2]} "
                         f"and GT {gt_depth.shape} sizes differ")
    pred_z = points[..., 2]
    pred_image, pred_lo, pred_hi = colorize(pred_z, pred_valid & np.isfinite(pred_z), "prediction")
    gt_image, gt_lo, gt_hi = colorize(gt_depth, gt_valid, "GT depth")

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))
    panels = [
        (rgb, f"RGB {frame.rgb_path.name}"),
        (pred_image, f"predicted z (p2-p98 {pred_lo:.3f}-{pred_hi:.3f}, normalised units)"),
        (gt_image, f"GT depth (p2-p98 {gt_lo:.2f}-{gt_hi:.2f} m, invalid black)"),
    ]
    for ax, (image, title) in zip(axes, panels):
        ax.imshow(image)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.suptitle(f"frame {frame.index}  t={frame.timestamp:.6f}")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True, help="run directory, e.g. .../predictions/baseline")
    parser.add_argument("--data-root", type=Path, required=True, help="folder containing the sequence folders")
    parser.add_argument("--sequences-config", type=Path, default=Path("configs/sequences.yaml"))
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--frames", type=int, nargs="+", required=True, help="frame indices (npz file names)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    sequence = load_sequence(args.data_root / args.sequence, args.sequences_config)
    out_of_range = [i for i in args.frames if not 0 <= i < len(sequence.frames)]
    if out_of_range:
        raise ValueError(f"frame indices {out_of_range} are outside 0..{len(sequence.frames) - 1}")
    args.out.mkdir(parents=True, exist_ok=True)
    for index in args.frames:
        frame = sequence.frames[index]
        points, valid = load_prediction(
            args.predictions / sequence.name / f"{index:05d}.npz", frame, sequence.root)
        path = args.out / f"{sequence.name}_{index:05d}.png"
        save_panel(frame, points, valid, path)
        print(f"saved {path}")


if __name__ == "__main__":
    main()
