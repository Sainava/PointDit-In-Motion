"""Per sweep config, the median per-frame rel_d and its median per-frame difference against sweep_baseline_check
with a moving-block bootstrap interval; writes sweep_accuracy.csv.

Then checks that sweep_baseline_check reproduces the cached baseline evaluation on the same frames. A mismatch
prints a banner and exits 3, after the CSV is written.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pim.analysis.stats import bootstrap, moving_block_indices, percentile_ci  # noqa: E402
from pim.subsets import Subsets, load_subsets  # noqa: E402

REFERENCE = "sweep_baseline_check"
METRIC = "rel_d"
BLOCK_LENGTH = 15
RESAMPLES = 2000
SEED = 0
TOLERANCE = 1e-3
MISMATCH = 3
COLUMNS = ("config", "frames", "median_rel_d", "median_rel_d_diff", "diff_ci_low", "diff_ci_high")
BANNER = "!" * 78

Key = tuple[str, int]


def read_metric(path: Path, metric: str = METRIC) -> dict[Key, float]:
    with open(path, newline="") as f:
        return {(r["sequence"], int(r["frame_index"])): float(r[metric]) if r[metric] else math.nan
                for r in csv.DictReader(f)}


def subset_keys(subsets: Subsets) -> dict[str, list[Key]]:
    return {name: [(name, i) for i in range(r.first, r.last + 1)] for name, r in subsets.items()}


def nanmedian(values: np.ndarray) -> float:
    kept = values[~np.isnan(values)]
    return float(np.median(kept)) if kept.size else math.nan


def series(values: dict[Key, float], keys: Sequence[Key], where: str) -> np.ndarray:
    missing = [k for k in keys if k not in values]
    if missing:
        raise ValueError(f"{where} lacks {len(missing)} subset frames, first {missing[0]}")
    return np.array([values[k] for k in keys], dtype=np.float64)


def accuracy_rows(
    per_config: dict[str, dict[Key, float]], subsets: Subsets, reference: str = REFERENCE,
    block_length: int = BLOCK_LENGTH, resamples: int = RESAMPLES, seed: int = SEED,
) -> list[dict[str, Any]]:
    """Bootstrap resamples blocks within each sequence and pools the sequences, so the interval is per config."""
    if reference not in per_config:
        raise ValueError(f"{reference} has no evaluation; it is needed for the differences")
    keys = subset_keys(subsets)
    ref = {name: series(per_config[reference], k, reference) for name, k in keys.items()}
    rows = []
    for config, values in per_config.items():
        own = {name: series(values, k, config) for name, k in keys.items()}
        diffs = [own[name] - ref[name] for name in keys]
        pooled = np.concatenate(diffs)
        low = high = math.nan
        if all(len(d) >= block_length for d in diffs):
            rng = np.random.default_rng(seed)
            indices = [moving_block_indices(len(d), block_length, resamples, rng) for d in diffs]
            draws = bootstrap(lambda idx: nanmedian(np.concatenate([d[i] for d, i in zip(diffs, idx)])), indices)
            low, high = percentile_ci(draws)
        rows.append({"config": config, "frames": int((~np.isnan(pooled)).sum()),
                     "median_rel_d": nanmedian(np.concatenate(list(own.values()))),
                     "median_rel_d_diff": nanmedian(pooled), "diff_ci_low": low, "diff_ci_high": high})
    return rows


def regression(
    check: dict[Key, float], baseline: dict[Key, float], keys: Sequence[Key], tolerance: float = TOLERANCE,
) -> tuple[float, list[str]]:
    """(largest difference, failures) of check against baseline on keys; NaN must meet NaN."""
    largest, failures = 0.0, []
    for key in keys:
        if key not in check or key not in baseline:
            failures.append(f"{key}: missing from {'the sweep check' if key not in check else 'the baseline'}")
            continue
        a, b = check[key], baseline[key]
        if math.isnan(a) and math.isnan(b):
            continue
        difference = abs(a - b)
        if not difference <= tolerance:
            failures.append(f"{key}: {METRIC} {a} vs baseline {b}")
        elif difference > largest:
            largest = difference
    return largest, failures


def check_evaluates(eval_dir: Path, config: str) -> None:
    record = json.loads((eval_dir / "eval.json").read_text())
    evaluated = record["predictions"]["run"].get("name")
    if evaluated != config:
        raise ValueError(f"{eval_dir} evaluates run {evaluated!r}, expected {config!r}")


def write_rows(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=Path, required=True, help="holds <run name>/per_frame.csv")
    parser.add_argument("--configs", nargs="+", required=True, help="run names to tabulate")
    parser.add_argument("--subsets", type=Path, required=True)
    parser.add_argument("--baseline-csv", type=Path, required=True, help="cached baseline per_frame.csv")
    parser.add_argument("--out", type=Path, required=True, help="sweep_accuracy.csv")
    parser.add_argument("--tolerance", type=float, default=TOLERANCE)
    args = parser.parse_args(argv)

    subsets = load_subsets(args.subsets)
    per_config = {}
    for config in args.configs:
        check_evaluates(args.eval_root / config, config)
        per_config[config] = read_metric(args.eval_root / config / "per_frame.csv")
    rows = accuracy_rows(per_config, subsets)
    write_rows(args.out, rows)
    print(f"{'config':<22} {'frames':>6} {'rel_d':>7} {'diff':>8} {'95% CI':>19}")
    for r in rows:
        print(f"{r['config']:<22} {r['frames']:>6} {r['median_rel_d']:>7.3f} {r['median_rel_d_diff']:>+8.3f} "
              f"[{r['diff_ci_low']:+.3f}, {r['diff_ci_high']:+.3f}]")
    print(f"wrote {args.out}")

    keys = [k for ks in subset_keys(subsets).values() for k in ks]
    largest, failures = regression(per_config[REFERENCE], read_metric(args.baseline_csv), keys, args.tolerance)
    if failures:
        print(f"\n{BANNER}\n  REGRESSION CHECK FAILED  {REFERENCE} {METRIC} differs from {args.baseline_csv} on "
              f"{len(failures)} of {len(keys)} frames (tolerance {args.tolerance:g}).")
        for failure in failures[:20]:
            print(f"  {failure}")
        print(f"  {args.out} is written.\n{BANNER}")
        sys.exit(MISMATCH)
    print(f"regression: {REFERENCE} {METRIC} matches {args.baseline_csv} on all {len(keys)} frames "
          f"(largest difference {largest:.2e})")


if __name__ == "__main__":
    main()
