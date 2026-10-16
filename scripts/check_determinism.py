"""Check that a zero-init rerun reproduces cached zero-init predictions frame by frame: max |points difference|
within tolerance (float16 storage) and the same rgb_file. Writes determinism.csv; exits 1 on any failure."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pim.evaluation import prediction_path  # noqa: E402
from pim.runs import read_run  # noqa: E402
from pim.sampling import DETERMINISM_TOLERANCE, prediction_difference  # noqa: E402
from pim.subsets import Subsets, check_run_covers, load_subsets  # noqa: E402

COLUMNS = ("sequence", "frame_index", "max_abs_diff", "rgb_file", "reference_rgb_file", "valid_mismatch_pixels",
           "passed")


def check_zero_init(run: dict[str, Any], run_dir: Path) -> None:
    inference = run["config"]["inference"]
    if inference["init"] != "zeros":
        raise ValueError(f"{run_dir} was predicted with init {inference['init']!r}; both runs must use zeros")


def compare(run_dir: Path, reference_dir: Path, subsets: Subsets, tolerance: float) -> list[dict[str, Any]]:
    for directory in (run_dir, reference_dir):
        run = read_run(directory)
        check_zero_init(run, directory)
        check_run_covers(run, subsets)
    rows = []
    for name, frame_range in subsets.items():
        for index in range(frame_range.first, frame_range.last + 1):
            diff = prediction_difference(prediction_path(run_dir, name, index),
                                         prediction_path(reference_dir, name, index))
            rows.append({"sequence": name, "frame_index": index, "max_abs_diff": diff.max_abs_diff,
                         "rgb_file": diff.rgb_file_a, "reference_rgb_file": diff.rgb_file_b,
                         "valid_mismatch_pixels": diff.valid_mismatch, "passed": diff.passed(tolerance)})
    return rows


def report(rows: list[dict[str, Any]], tolerance: float) -> bool:
    print(f"{'sequence':<45} {'frames':>6} {'max |diff|':>11} {'median':>9} {'rgb ok':>6} {'valid diff px':>13} "
          f"{'failed':>6}")
    for name in dict.fromkeys(r["sequence"] for r in rows):
        sequence = [r for r in rows if r["sequence"] == name]
        diffs = sorted(r["max_abs_diff"] for r in sequence)
        print(f"{name:<45} {len(sequence):>6} {diffs[-1]:>11.3e} {diffs[len(diffs) // 2]:>9.3e} "
              f"{sum(r['rgb_file'] == r['reference_rgb_file'] for r in sequence):>6} "
              f"{sum(r['valid_mismatch_pixels'] for r in sequence):>13} "
              f"{sum(not r['passed'] for r in sequence):>6}")
    failed = [r for r in rows if not r["passed"]]
    for r in failed[:20]:
        print(f"  FAIL {r['sequence']} frame {r['frame_index']}: max |diff| {r['max_abs_diff']:.3e}, "
              f"rgb_file {r['rgb_file']} vs {r['reference_rgb_file']}")
    print(f"determinism: {len(rows) - len(failed)}/{len(rows)} frames within {tolerance:g} with matching rgb_file"
          + (f", {len(failed)} FAILED" if failed else ""))
    return not failed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True, help="zero-init rerun directory (run.json)")
    parser.add_argument("--reference", type=Path, required=True, help="cached zero-init run directory (run.json)")
    parser.add_argument("--subsets", type=Path, required=True, help="frames to compare")
    parser.add_argument("--tolerance", type=float, default=DETERMINISM_TOLERANCE)
    parser.add_argument("--out", type=Path, required=True, help="determinism.csv")
    args = parser.parse_args()

    rows = compare(args.predictions, args.reference, load_subsets(args.subsets), args.tolerance)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    passed = report(rows, args.tolerance)
    print(f"wrote {args.out}")
    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
