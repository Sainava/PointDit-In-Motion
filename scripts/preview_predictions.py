"""Save RGB | predicted depth | GT depth | relative error panels for cached predictions, plus error_breakdown.csv."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pim.data.tum import Frame, load_sequence  # noqa: E402
from pim.metrics.error_maps import depth_edge_band, edge_breakdown, relative_depth_error  # noqa: E402
from pim.metrics.geometry import backproject, evaluate_frame  # noqa: E402

PERCENTILES = (2, 98)
ERROR_SCALE = (0.0, 0.25)
BREAKDOWN = ("band_pixel_fraction", "band_error_fraction", "rel_d_band", "rel_d_outside")
BREAKDOWN_COLUMNS = ("sequence", "frame_index", "rel_d", *BREAKDOWN, "depth_offset_ms")
# x100 units; matches check_frame.py. Guards against a --csv written for different predictions.
REL_D_TOLERANCE = 1e-3


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


def error_analysis(
    points: np.ndarray, gt_depth: np.ndarray, gt_valid: np.ndarray, K: np.ndarray
) -> tuple[float, np.ndarray, dict[str, float]]:
    """Rel_d, error map and edge breakdown, with the evaluation protocol's GT-only mask."""
    gt_points, gt_valid = backproject(gt_depth, gt_valid, K)
    metrics = evaluate_frame(points, gt_points, gt_valid, pred_valid=None)
    if metrics.skipped:
        return metrics.rel_d, np.full(gt_depth.shape, np.nan, dtype=np.float32), dict.fromkeys(BREAKDOWN, math.nan)
    error = relative_depth_error(metrics.aligned_depth, gt_depth, gt_valid)
    return metrics.rel_d, error, edge_breakdown(error, depth_edge_band(gt_depth, gt_valid), gt_valid)


def save_panel(frame: Frame, K: np.ndarray, points: np.ndarray, pred_valid: np.ndarray, path: Path) -> dict[str, float]:
    rgb = frame.load_rgb()
    gt_depth, gt_valid = frame.load_depth()
    if points.shape[:2] != gt_depth.shape or rgb.shape[:2] != gt_depth.shape:
        raise ValueError(f"frame {frame.index}: rgb {rgb.shape[:2]}, prediction {points.shape[:2]} "
                         f"and GT {gt_depth.shape} sizes differ")
    pred_z = points[..., 2]
    pred_image, pred_lo, pred_hi = colorize(pred_z, pred_valid & np.isfinite(pred_z), "prediction")
    gt_image, gt_lo, gt_hi = colorize(gt_depth, gt_valid, "GT depth")
    rel_d, error, breakdown = error_analysis(points, gt_depth, gt_valid, K)

    fig, axes = plt.subplots(1, 4, figsize=(24, 4.8))
    panels = [
        (rgb, f"RGB {frame.rgb_path.name}"),
        (pred_image, f"predicted z (p2-p98 {pred_lo:.3f}-{pred_hi:.3f}, normalised units)"),
        (gt_image, f"GT depth (p2-p98 {gt_lo:.2f}-{gt_hi:.2f} m, invalid black)"),
    ]
    for ax, (image, title) in zip(axes, panels):
        ax.imshow(image)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    draw_error(fig, axes[3], error, rel_d, breakdown)
    fig.suptitle(f"frame {frame.index}  t={frame.timestamp:.6f}")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return {"rel_d": rel_d, **breakdown}


def draw_error(fig: plt.Figure, ax: plt.Axes, error: np.ndarray, rel_d: float, breakdown: dict[str, float]) -> None:
    cmap = matplotlib.colormaps["magma"].copy()
    cmap.set_bad("0.5")
    image = ax.imshow(np.ma.masked_invalid(error), cmap=cmap, vmin=ERROR_SCALE[0], vmax=ERROR_SCALE[1])
    fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    if math.isnan(rel_d):
        title = "|aligned - GT| / GT: frame skipped (too few valid GT pixels)"
    else:
        title = (f"|aligned - GT| / GT, invalid grey | Rel_d {rel_d:.2f}\n"
                 f"edge band: {breakdown['band_error_fraction']:.2f} of error "
                 f"on {breakdown['band_pixel_fraction']:.2f} of pixels")
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def read_per_frame(csv_path: Path, sequence: str) -> dict[int, dict[str, str]]:
    with open(csv_path, newline="") as f:
        rows = {int(r["frame_index"]): r for r in csv.DictReader(f) if r["sequence"] == sequence}
    if not rows:
        raise ValueError(f"{csv_path} has no rows for {sequence}")
    return rows


def check_against_csv(row: dict[str, str] | None, rel_d: float, sequence: str, index: int, csv_path: Path) -> None:
    if row is None:
        raise ValueError(f"{csv_path} has no row for {sequence} frame {index}")
    expected = float(row["rel_d"]) if row["rel_d"] else math.nan
    if math.isnan(expected) != math.isnan(rel_d) or abs(expected - rel_d) > REL_D_TOLERANCE:
        raise ValueError(f"{sequence} frame {index}: Rel_d {rel_d:.4f} here but {expected:.4f} in {csv_path}; "
                         "the CSV was written for different predictions or GT")


def write_breakdown(path: Path, rows: list[dict[str, object]]) -> None:
    """Merge rows into an existing file, replacing rows for the same frame, so sequences can be previewed one call at a time."""
    merged: dict[tuple[str, int], dict[str, object]] = {}
    if path.is_file():
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                merged[row["sequence"], int(row["frame_index"])] = row
    for row in rows:
        merged[str(row["sequence"]), int(row["frame_index"])] = row
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=BREAKDOWN_COLUMNS)
        writer.writeheader()
        writer.writerows(merged[key] for key in sorted(merged))


def cell(value: float) -> str:
    return "" if math.isnan(value) else f"{value:.6f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True, help="run directory, e.g. .../predictions/baseline")
    parser.add_argument("--data-root", type=Path, required=True, help="folder containing the sequence folders")
    parser.add_argument("--sequences-config", type=Path, default=Path("configs/sequences.yaml"))
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--frames", type=int, nargs="+", required=True, help="frame indices (npz file names)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--csv", type=Path, help="per_frame.csv from evaluate.py: adds depth_offset_ms and "
                                                 "checks that its rel_d matches the recomputed one")
    args = parser.parse_args()

    sequence = load_sequence(args.data_root / args.sequence, args.sequences_config)
    out_of_range = [i for i in args.frames if not 0 <= i < len(sequence.frames)]
    if out_of_range:
        raise ValueError(f"frame indices {out_of_range} are outside 0..{len(sequence.frames) - 1}")
    per_frame = read_per_frame(args.csv, sequence.name) if args.csv else None
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for index in args.frames:
        frame = sequence.frames[index]
        points, valid = load_prediction(
            args.predictions / sequence.name / f"{index:05d}.npz", frame, sequence.root)
        path = args.out / f"{sequence.name}_{index:05d}.png"
        stats = save_panel(frame, sequence.K, points, valid, path)
        if per_frame is not None:
            check_against_csv(per_frame.get(index), stats["rel_d"], sequence.name, index, args.csv)
        rows.append({"sequence": sequence.name, "frame_index": index,
                     **{name: cell(value) for name, value in stats.items()},
                     "depth_offset_ms": per_frame[index]["depth_offset_ms"] if per_frame else ""})
        print(f"saved {path}  Rel_d {stats['rel_d']:.2f}, edge band {stats['band_error_fraction']:.2f} of error "
              f"on {stats['band_pixel_fraction']:.2f} of pixels")
    breakdown = args.out / "error_breakdown.csv"
    write_breakdown(breakdown, rows)
    print(f"wrote {breakdown}")


if __name__ == "__main__":
    main()
