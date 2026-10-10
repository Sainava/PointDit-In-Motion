"""Compute per-frame motion, sharpness and GT-geometry covariates for each sequence window; write covariates.csv."""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pim.analysis.covariates import COLUMNS, add_relative_sharpness, frame_covariates  # noqa: E402
from pim.data.tum import apply_window, load_sequence, read_groundtruth  # noqa: E402


def sequence_rows(data_root: Path, sequences_config: Path, name: str) -> list[dict[str, Any]]:
    sequence = load_sequence(data_root / name, sequences_config)
    gt_times, gt_poses = read_groundtruth(sequence.root)
    fx = float(sequence.K[0, 0])
    rows = []
    for frame in apply_window(sequence):
        rgb_file = frame.rgb_path.relative_to(sequence.root).as_posix()
        rows.append(frame_covariates(name, frame, rgb_file, fx, gt_times, gt_poses))
    add_relative_sharpness(rows)
    return rows


def cell(value: Any) -> Any:
    if isinstance(value, float):
        return "" if math.isnan(value) else repr(value)
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".csv.partial")
    with open(partial, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows({key: cell(row[key]) for key in COLUMNS} for row in rows)
    os.replace(partial, path)


def print_summary(rows: dict[str, list[dict[str, Any]]]) -> None:
    print(f"{'sequence':<45} {'frames':>6} {'ang mean':>9} {'ang max':>8} {'lin mean':>9} "
          f"{'sharp med':>10} {'misalign':>9} {'motion NaN':>10} {'any NaN':>7}")
    print(f"{'':<45} {'':>6} {'(deg/s)':>9} {'(deg/s)':>8} {'(m/s)':>9} {'':>10} {'mean px':>9}")
    for name, sequence_rows in rows.items():
        def column(key: str) -> np.ndarray:
            return np.array([r[key] for r in sequence_rows], dtype=np.float64)

        angular = column("angular_speed_dps")
        nan_rows = sum(any(isinstance(v, float) and math.isnan(v) for v in r.values()) for r in sequence_rows)
        print(f"{name:<45} {len(sequence_rows):>6} {np.nanmean(angular):>9.2f} {np.nanmax(angular):>8.2f} "
              f"{np.nanmean(column('linear_speed_mps')):>9.3f} {np.median(column('sharpness')):>10.5f} "
              f"{np.nanmean(column('expected_misalignment_px')):>9.2f} {int(np.isnan(angular).sum()):>10} {nan_rows:>7}")
    print("motion NaN: t +/- 20 ms outside the GT range or across a GT gap > 50 ms; "
          "any NaN: also counts frames without valid GT depth.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True, help="folder containing the sequence folders")
    parser.add_argument("--sequences-config", type=Path, default=Path("configs/sequences.yaml"))
    parser.add_argument("--sequences", nargs="+", help="default: every sequence in the config")
    parser.add_argument("--out", type=Path, default=Path("results/covariates/covariates.csv"))
    args = parser.parse_args()

    known = list((yaml.safe_load(args.sequences_config.read_text()) or {}).get("sequences") or {})
    names = args.sequences or known
    unknown = [name for name in names if name not in known]
    if unknown:
        raise ValueError(f"sequences {unknown} are not in {args.sequences_config}; known: {known}")
    rows = {}
    for name in names:
        rows[name] = sequence_rows(args.data_root, args.sequences_config, name)
        print(f"  {name}: {len(rows[name])} frames", flush=True)
    write_csv(args.out, [row for sequence_rows in rows.values() for row in sequence_rows])
    print_summary(rows)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
