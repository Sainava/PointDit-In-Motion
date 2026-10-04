"""Run PointDiT on the first frame of a TUM sequence and save the point map and a depth panel."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pim.data.tum import load_rgb, read_rgb_list  # noqa: E402
from pim.model.runner import PointDiTRunner, PredictionResult  # noqa: E402


def save_points(result: PredictionResult, path: Path) -> None:
    np.savez_compressed(
        path, points=result.points.astype(np.float16), valid_mask=result.valid_mask
    )


def save_depth_panel(image: np.ndarray, result: PredictionResult, path: Path) -> None:
    depth = np.ma.masked_where(~result.valid_mask, result.points[..., 2])
    fig, (ax_rgb, ax_depth) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax_rgb.imshow(image)
    ax_rgb.set_title("RGB")
    shown = ax_depth.imshow(depth, cmap="plasma")
    ax_depth.set_title("Predicted depth (z)")
    fig.colorbar(shown, ax=ax_depth, fraction=0.035)
    for ax in (ax_rgb, ax_depth):
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=Path, required=True, help="TUM sequence directory")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    frame = read_rgb_list(args.sequence)[0]
    image = load_rgb(frame.path)
    result = PointDiTRunner().predict(image)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = frame.path.stem
    save_points(result, args.output_dir / f"{stem}_points.npz")
    save_depth_panel(image, result, args.output_dir / f"{stem}_depth.png")

    print(f"frame: {frame.path}")
    print(f"points: {result.points.shape} (model ran at {result.inference_size})")
    print(f"latency: {result.latency_ms:.1f} ms (sampling {result.model_latency_ms:.1f} ms)")
    print(f"peak memory: {result.peak_memory_gb:.2f} GB")
    print(f"saved to {args.output_dir}")


if __name__ == "__main__":
    main()
