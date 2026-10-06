"""Recompute one frame's metrics in plain numpy and compare them with its per_frame.csv row."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pim.data.tum import load_sequence  # noqa: E402
from pim.evaluation import METRICS, load_prediction, prediction_path  # noqa: E402
from pim.metrics.geometry import evaluate_frame  # noqa: E402

EPS = 1e-6
# x100 units. The metrics run in float32 torch, the check in float64 numpy; a δ1 pixel on the
# threshold may flip, and one pixel in 300k is 3e-4.
TOLERANCE = 1e-3


def gt_points_numpy(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    v, u = np.indices(depth.shape, dtype=np.float64)
    z = depth.astype(np.float64)
    return np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], axis=-1)


def numpy_metrics(
    aligned_points: np.ndarray, aligned_depth: np.ndarray, gt_points: np.ndarray, mask: np.ndarray
) -> dict[str, float]:
    """upstream metrics/metrics.py rel/delta1 for points and depth over `mask`, x100."""
    p = aligned_points[mask].astype(np.float64)
    g = gt_points[mask].astype(np.float64)
    d = aligned_depth[mask].astype(np.float64)
    gz = g[:, 2]
    dist_err = np.linalg.norm(p - g, axis=-1)
    dist_gt = np.linalg.norm(g, axis=-1)
    dist_pred = np.linalg.norm(p, axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.maximum(gz / d, d / gz)
    return {
        "rel_p": 100.0 * float(np.mean(dist_err / (dist_gt + EPS))),
        "delta1_p": 100.0 * float(np.mean(dist_err < 0.25 * np.minimum(dist_gt, dist_pred))),
        "rel_d": 100.0 * float(np.mean(np.abs(d - gz) / (gz + EPS))),
        "delta1_d": 100.0 * float(np.mean(ratio < 1.25)),
    }


def read_csv_row(csv_path: Path, sequence: str, frame_index: int) -> dict[str, str]:
    with open(csv_path, newline="") as f:
        matches = [r for r in csv.DictReader(f) if r["sequence"] == sequence and int(r["frame_index"]) == frame_index]
    if len(matches) != 1:
        raise ValueError(f"{csv_path} has {len(matches)} rows for {sequence} frame {frame_index}, expected 1")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True, help="run directory holding run.json")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--sequences-config", type=Path, default=Path("configs/sequences.yaml"))
    parser.add_argument("--csv", type=Path, required=True, help="per_frame.csv written by evaluate.py")
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--frame", type=int, required=True, help="frame index (npz file name)")
    args = parser.parse_args()

    row = read_csv_row(args.csv, args.sequence, args.frame)
    sequence = load_sequence(args.data_root / args.sequence, args.sequences_config)
    if not 0 <= args.frame < len(sequence.frames):
        raise ValueError(f"frame {args.frame} is outside 0..{len(sequence.frames) - 1}")
    frame = sequence.frames[args.frame]
    prediction = load_prediction(prediction_path(args.predictions, sequence.name, frame.index), frame, sequence.root)
    if row["rgb_file"] != prediction.rgb_file or row["rgb_timestamp"] != f"{frame.timestamp:.6f}":
        raise ValueError(f"CSV row is {row['rgb_file']} at {row['rgb_timestamp']}, but frame {frame.index} "
                         f"is {prediction.rgb_file} at {frame.timestamp:.6f}")

    depth, _ = frame.load_depth()
    mask = depth > 0
    gt_points = gt_points_numpy(depth, sequence.K)
    result = evaluate_frame(prediction.points, gt_points.astype(np.float32), mask, pred_valid=None)
    print(f"{sequence.name} frame {frame.index} ({prediction.rgb_file}): "
          f"num_valid csv {row['num_valid']}, recomputed {int(mask.sum())}, evaluate_frame {result.num_valid}")
    if result.skipped:
        if row["skipped"] != "True":
            raise SystemExit(f"MISMATCH: frame is skipped on recompute but the CSV row says skipped={row['skipped']}")
        print("frame is skipped (fewer than 100 valid GT pixels) in both; nothing to compare")
        return

    recomputed = numpy_metrics(result.aligned_points, result.aligned_depth, gt_points, mask)
    print(f"{'metric':<9} {'csv':>12} {'evaluate_frame':>15} {'numpy':>12} {'|csv-numpy|':>12}")
    worst = abs(int(row["num_valid"]) - int(mask.sum()))
    for name in METRICS:
        csv_value = float(row[name]) if row[name] else float("nan")
        diff = abs(csv_value - recomputed[name])
        worst = max(worst, diff) if not np.isnan(diff) else float("inf")
        print(f"{name:<9} {csv_value:>12.6f} {getattr(result, name):>15.6f} {recomputed[name]:>12.6f} {diff:>12.2e}")
    if worst > TOLERANCE:
        raise SystemExit(f"MISMATCH: largest difference {worst:.2e} exceeds {TOLERANCE:.0e}")
    print(f"OK: all differences within {TOLERANCE:.0e}")


if __name__ == "__main__":
    main()
