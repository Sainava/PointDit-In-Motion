"""Split-half reliability of the implied focal length: fit it on the left and right image halves; write focal_split.csv."""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pim.data.tum import apply_window, load_sequence  # noqa: E402
from pim.evaluation import build_tasks, load_prediction  # noqa: E402
from pim.metrics.geometry import FOCAL_MIN_OFFSET_PX, MIN_VALID_PIXELS, backproject, implied_focal  # noqa: E402

COLUMNS = ("sequence", "frame_index", "focal_full", "focal_left", "focal_right", "focal_split_logdiff")
MIN_SIDE_PIXELS = 1000
TOLERANCE = 1e-6


def focal_split(points: np.ndarray, valid: np.ndarray, K: np.ndarray) -> dict[str, float]:
    """Implied fx / true fx on all pixels and on each side of cx; NaN where a fit has too few pixels or fails."""
    valid = valid.astype(bool)
    if valid.sum() < MIN_VALID_PIXELS:
        return dict.fromkeys(COLUMNS[2:], math.nan)
    fx, cx = float(K[0, 0]), float(K[0, 2])
    u = np.broadcast_to(np.arange(valid.shape[1], dtype=np.float64), valid.shape)
    left = valid & (u < cx - FOCAL_MIN_OFFSET_PX)
    right = valid & (u > cx + FOCAL_MIN_OFFSET_PX)

    def side(mask: np.ndarray) -> float:
        return implied_focal(points, mask, cx) / fx if mask.sum() >= MIN_SIDE_PIXELS else math.nan

    focal_left, focal_right = side(left), side(right)
    return {
        "focal_full": implied_focal(points, valid, cx) / fx,
        "focal_left": focal_left,
        "focal_right": focal_right,
        "focal_split_logdiff": abs(math.log(focal_left / focal_right)),
    }


def read_focal_ratios(csv_path: Path) -> dict[str, dict[int, float]]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"{csv_path} not found")
    ratios: dict[str, dict[int, float]] = defaultdict(dict)
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            ratios[row["sequence"]][int(row["frame_index"])] = float(row["focal_ratio"]) if row["focal_ratio"] else math.nan
    if not ratios:
        raise ValueError(f"{csv_path} has no rows")
    return dict(ratios)


def check_consistency(rows: list[dict[str, float]], expected: dict[str, dict[int, float]]) -> None:
    mismatches = []
    for row in rows:
        full, ratio = row["focal_full"], expected[row["sequence"]][row["frame_index"]]
        same = (math.isnan(full) and math.isnan(ratio)) or abs(full - ratio) <= TOLERANCE
        if not same:
            mismatches.append(f"{row['sequence']} frame {row['frame_index']}: focal_full {full!r}, focal_ratio {ratio!r}")
    if mismatches:
        raise ValueError(f"focal_full differs from per_frame.csv's focal_ratio by more than {TOLERANCE} on "
                         f"{len(mismatches)} frames:\n  " + "\n  ".join(mismatches[:20]))


def sequence_rows(name: str, run_dir: Path, data_root: Path, sequences_config: Path,
                  frame_indices: list[int]) -> list[dict[str, float]]:
    sequence = load_sequence(data_root / name, sequences_config)
    tasks = build_tasks(name, sequence.root, sequence.K, apply_window(sequence), run_dir)
    window = [task.frame.index for task in tasks]
    if window != frame_indices:
        raise ValueError(f"{name}: per_frame.csv has frames {frame_indices[0]}..{frame_indices[-1]} "
                         f"({len(frame_indices)}), but the configured window is {window[0]}..{window[-1]} ({len(window)})")
    rows = []
    for task in tasks:
        prediction = load_prediction(task.prediction, task.frame, task.sequence_root)
        depth, depth_valid = task.frame.load_depth()
        _, gt_valid = backproject(depth, depth_valid, task.K)
        rows.append({"sequence": name, "frame_index": task.frame.index,
                     **focal_split(prediction.points, gt_valid, task.K)})
    return rows


def write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".csv.partial")
    with open(partial, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if isinstance(value, float) and math.isnan(value) else value
                             for key, value in row.items()})
    os.replace(partial, path)


def print_summary(rows: list[dict[str, float]]) -> None:
    print(f"{'sequence':<45} {'frames':>6} {'split':>6} {'< ln 1.1':>9} {'median logdiff':>15}")
    for name in dict.fromkeys(r["sequence"] for r in rows):
        logdiff = np.array([r["focal_split_logdiff"] for r in rows if r["sequence"] == name])
        fitted = logdiff[~np.isnan(logdiff)]
        median = f"{np.median(fitted):.3f}" if len(fitted) else "n/a"
        print(f"{name:<45} {len(logdiff):>6} {len(fitted):>6} {int((fitted < math.log(1.1)).sum()):>9} {median:>15}")
    print(f"split: frames with both side fits (each side needs >= {MIN_SIDE_PIXELS} GT-valid pixels "
          f"more than {FOCAL_MIN_OFFSET_PX:.0f} px from cx)")


def run_focal_split(run_dir: Path, data_root: Path, sequences_config: Path, per_frame_csv: Path, out: Path) -> None:
    expected = read_focal_ratios(per_frame_csv)
    rows = []
    for name, ratios in expected.items():
        rows += sequence_rows(name, run_dir, data_root, sequences_config, sorted(ratios))
        print(f"  {name}: {len(ratios)} frames", flush=True)
    check_consistency(rows, expected)
    write_csv(out, rows)
    print_summary(rows)
    print(f"focal_full matches per_frame.csv's focal_ratio within {TOLERANCE} on all {len(rows)} frames; wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True, help="run directory written by predict.py")
    parser.add_argument("--data-root", type=Path, required=True, help="folder containing the sequence folders")
    parser.add_argument("--sequences-config", type=Path, default=Path("configs/sequences.yaml"))
    parser.add_argument("--csv", type=Path, required=True, help="per_frame.csv of the same run, for the consistency check")
    parser.add_argument("--out", type=Path, required=True, help="path of focal_split.csv")
    args = parser.parse_args()
    run_focal_split(args.predictions, args.data_root, args.sequences_config, args.csv, args.out)


if __name__ == "__main__":
    main()
