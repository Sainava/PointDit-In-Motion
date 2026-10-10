"""Temporal consistency of cached predictions over all frame pairs at fixed time gaps: pairs_<sequence>.csv,
summary.csv and consistency.json, then a regression check against the subtask-1 development values."""

from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pim.consistency.evaluation import (  # noqa: E402
    MAPS_GAP_MS, METRICS, PAIR_TOLERANCE_MS, PIXEL_SETS, PairTask, SequenceJob, SequenceOutcome, atomic_write,
    cell_value, evaluate_sequence, incomplete_reason, init_worker, load_window, pairs_csv, plan_pairs, read_pairs,
)
from pim.consistency.metrics import ALIGNMENT_GRID, MIN_VISIBLE_PIXELS, VISIBILITY_THRESHOLD  # noqa: E402
from pim.evaluation import check_rgb_file, prediction_path  # noqa: E402
from pim.metrics.geometry import MAX_ALIGNMENT_PIXELS  # noqa: E402
from pim.paths import FORK_DIR, REPO_ROOT  # noqa: E402
from pim.runs import available_cpus, check_window, git_commit, now, read_run, select_sequences  # noqa: E402

DEFAULT_GAPS_MS = (33.0, 100.0, 333.0)
EXPECTED = REPO_ROOT / "configs" / "consistency_expected.json"
RATIOS = {"m1_over_gt_floor_depth": ("m1", "gt_floor_depth"), "m2_over_m2_floor": ("m2", "m2_floor")}
SUMMARY_COLUMNS = (
    "sequence", "gap_ms", "pixels", "pairs", "too_few_pairs", "frac_median",
    *(f"{m}_{s}" for m in METRICS for s in ("median", "iqr", "n")),
    *RATIOS,
)
PROTOCOL = {
    "pairing": f"frame j is the window frame closest in time to t_i + gap, kept if within {PAIR_TOLERANCE_MS:g} ms",
    "visibility": f"GT_i forward-warped into j (z-buffer) with depth-timestamp poses; visible where hit, GT_j valid "
                  f"and |z_i->j - z_j| <= {VISIBILITY_THRESHOLD} z_j; interior = visible minus the GT_j depth-edge "
                  f"band; fewer than {MIN_VISIBLE_PIXELS} visible pixels sets too_few_pixels and NaN metrics",
    "pose_timing": {
        "gt_floor_depth": "depth timestamps (also visibility)",
        "gt_floor_rgb": "RGB timestamps, scored on the depth-timestamp visible pixels that the RGB-pose warp hits",
        "m1": "RGB timestamps",
        "m2_floor": "RGB timestamps",
        "m2": "RGB timestamps",
    },
    "alignment": "M1 and the M2 anchor use upstream compute_metrics' per-frame depth scale + shift to GT; M2 fits "
                 "frame j's raw z to frame i's aligned depth warped into j with align_depth_affine (weights 1/anchor); "
                 "the M2 floor does the same with GT_j as the raw depth and GT_i as the anchor",
    "subsample": f"{ALIGNMENT_GRID[0]}x{ALIGNMENT_GRID[1]} masked nearest subsample, at most {MAX_ALIGNMENT_PIXELS} "
                 "pixels per fit",
    "units": "relative depth error x100; per-pair mean, median, p90 and median within near/mid/far GT-depth thirds",
    "maps": f"--save-maps sequences at {MAPS_GAP_MS:g} ms: per-pixel M2 (relative, not x100) and frame i's aligned "
            "depth warped into j, float16, NaN where not scored or not hit",
}


@dataclass(frozen=True)
class Check:
    table: str
    sequence: str
    gap: str
    pixels: str
    metric: str
    expected: float
    actual: float
    skipped: str | None = None
    """Why the value could not be checked (sequence or gap not in this run)."""

    @property
    def difference(self) -> float:
        return abs(self.actual - self.expected)

    def passed(self, tolerance: float) -> bool:
        if self.skipped is not None:
            return True
        if self.metric == "pairs":
            return self.actual == self.expected
        return math.isfinite(self.actual) and self.difference <= tolerance


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True, help="run directory holding run.json")
    parser.add_argument("--data-root", type=Path, required=True, help="folder containing the sequence folders")
    parser.add_argument("--sequences-config", type=Path, default=Path("configs/sequences.yaml"))
    parser.add_argument("--sequences", nargs="+", help="default: every sequence in run.json")
    parser.add_argument("--gaps-ms", type=float, nargs="+", default=list(DEFAULT_GAPS_MS))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1,
                        help="one sequence per worker process; each needs ~4 GB of RAM")
    parser.add_argument("--threads", type=int, default=1, help="torch/OpenMP threads per worker")
    parser.add_argument("--save-maps", nargs="+", default=[], metavar="SEQUENCE",
                        help=f"save per-pixel M2 maps at the {MAPS_GAP_MS:g} ms gap for these sequences")
    parser.add_argument("--expected", type=Path, default=EXPECTED, help="regression values")
    return parser.parse_args(argv)


def validate(args: argparse.Namespace) -> None:
    if args.workers < 1:
        raise ValueError(f"--workers must be at least 1, got {args.workers}")
    if args.threads < 1:
        raise ValueError(f"--threads must be at least 1, got {args.threads}")
    if any(not (math.isfinite(g) and g > 0) for g in args.gaps_ms) or len(set(args.gaps_ms)) != len(args.gaps_ms):
        raise ValueError(f"--gaps-ms must be distinct positive values, got {args.gaps_ms}")
    if args.save_maps and MAPS_GAP_MS not in args.gaps_ms:
        raise ValueError(f"--save-maps needs the {MAPS_GAP_MS:g} ms gap in --gaps-ms, got {args.gaps_ms}")
    if not args.data_root.is_dir():
        raise FileNotFoundError(f"data root {args.data_root} does not exist")
    if not args.expected.is_file():
        raise FileNotFoundError(f"regression values {args.expected} not found")


def sequence_job(args: argparse.Namespace, name: str) -> SequenceJob:
    return SequenceJob(name, args.data_root, args.sequences_config, args.predictions, tuple(args.gaps_ms),
                       args.out, name in args.save_maps, args.threads)


def plan_sequence(job: SequenceJob, run: dict[str, Any]) -> list[PairTask]:
    sequence, frames = load_window(job)
    check_window(job.name, run, frames, job.sequences_config)
    for frame in frames:
        check_rgb_file(prediction_path(job.run_dir, job.name, frame.index), frame, sequence.root)
    return plan_pairs(frames, job.gaps_ms)


def run_jobs(jobs: list[SequenceJob], workers: int) -> dict[str, SequenceOutcome]:
    if workers == 1 or len(jobs) <= 1:
        return {job.name: evaluate_sequence(job) for job in jobs}
    # spawn: torch's thread pools are not fork-safe. One task per child, so each peak memory is one sequence's.
    executor = ProcessPoolExecutor(
        max_workers=min(workers, len(jobs)), mp_context=multiprocessing.get_context("spawn"),
        initializer=init_worker, initargs=(jobs[0].threads,), max_tasks_per_child=1,
    )
    outcomes = {}
    try:
        futures = {executor.submit(evaluate_sequence, job): job.name for job in jobs}
        for future in as_completed(futures):
            outcome = future.result()
            outcomes[outcome.name] = outcome
            print(f"{outcome.name}: done in {outcome.seconds / 60:.1f} min, peak {outcome.peak_mem_mib:.0f} MiB",
                  flush=True)
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    return outcomes


def finite(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return array[np.isfinite(array)]


def median(values: list[float]) -> float:
    kept = finite(values)
    return float(np.median(kept)) if kept.size else math.nan


def iqr(values: list[float]) -> float:
    kept = finite(values)
    if not kept.size:
        return math.nan
    q1, q3 = np.percentile(kept, [25, 75])
    return float(q3 - q1)


def ratio_median(rows: list[dict[str, str]], numerator: str, denominator: str) -> float:
    ratios = []
    for row in rows:
        n, d = cell_value(row[numerator]), cell_value(row[denominator])
        if math.isfinite(n) and math.isfinite(d) and d > 0:
            ratios.append(n / d)
    return median(ratios)


def summarize(rows_by_sequence: dict[str, list[dict[str, str]]]) -> list[dict[str, Any]]:
    summary = []
    for name, rows in rows_by_sequence.items():
        for gap in sorted({float(r["gap_ms"]) for r in rows}):
            at_gap = [r for r in rows if float(r["gap_ms"]) == gap]
            for pixels in PIXEL_SETS:
                entry: dict[str, Any] = {
                    "sequence": name, "gap_ms": f"{gap:g}", "pixels": pixels, "pairs": len(at_gap),
                    "too_few_pairs": sum(r["too_few_pixels"] == "True" for r in at_gap),
                    "frac_median": median([cell_value(r[f"frac_{pixels}"]) for r in at_gap]),
                }
                for metric in METRICS:
                    values = [cell_value(r[f"{metric}_{pixels}_median"]) for r in at_gap]
                    entry.update({f"{metric}_median": median(values), f"{metric}_iqr": iqr(values),
                                  f"{metric}_n": int(finite(values).size)})
                for column, (numerator, denominator) in RATIOS.items():
                    entry[column] = ratio_median(at_gap, f"{numerator}_{pixels}_median",
                                                 f"{denominator}_{pixels}_median")
                summary.append(entry)
    return summary


def write_summary(path: Path, summary: list[dict[str, Any]]) -> None:
    def write(f) -> None:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for entry in summary:
            writer.writerow({k: ("" if isinstance(v, float) and math.isnan(v) else v) for k, v in entry.items()})

    atomic_write(path, write, newline="")


def regression_checks(rows_by_sequence: dict[str, list[dict[str, str]]], expected: dict[str, Any]) -> list[Check]:
    checks = []
    for table_name, table in expected["tables"].items():
        frame_ranges = table.get("frames") or {}
        for name, by_pixels in table["values"].items():
            rows = rows_by_sequence.get(name)
            gaps = {float(r["gap_ms"]) for r in rows or []}
            if rows is not None and name in frame_ranges:
                lo, hi = frame_ranges[name]
                rows = [r for r in rows if lo <= int(r["frame_i"]) <= hi and lo <= int(r["frame_j"]) <= hi]
            for gap, count in table.get("pairs", {}).get(name, {}).items():
                checks.append(_check(table_name, name, gap, "visible", "pairs", count, rows, gaps,
                                     lambda at_gap, _: float(len(at_gap))))
            for pixels, by_gap in by_pixels.items():
                for gap, values in by_gap.items():
                    for metric, value in values.items():
                        checks.append(_check(table_name, name, gap, pixels, metric, value, rows, gaps,
                                             _median_of(pixels)))
    return checks


def _median_of(pixels: str):
    return lambda at_gap, metric: median([cell_value(r[f"{metric}_{pixels}_median"]) for r in at_gap])


def _check(table: str, name: str, gap: str, pixels: str, metric: str, expected: float,
           rows: list[dict[str, str]] | None, evaluated_gaps: set[float], compute) -> Check:
    if rows is None:
        return Check(table, name, gap, pixels, metric, expected, math.nan, "sequence not evaluated")
    if float(gap) not in evaluated_gaps:
        return Check(table, name, gap, pixels, metric, expected, math.nan, "gap not evaluated")
    at_gap = [r for r in rows if float(r["gap_ms"]) == float(gap)]
    return Check(table, name, gap, pixels, metric, float(expected), compute(at_gap, metric))


def report_regression(checks: list[Check], tolerance: float) -> bool:
    print(f"\nregression check against subtask-1 values (tolerance {tolerance:g}; pair counts must match exactly)")
    print(f"  {'table':<20} {'sequence':<45} {'gap':>4} {'pixels':<8} {'metric':<15} {'expected':>9} "
          f"{'actual':>9} {'|diff|':>8}")
    for c in checks:
        status = f"skipped: {c.skipped}" if c.skipped else ("ok" if c.passed(tolerance) else "FAIL")
        print(f"  {c.table:<20} {c.sequence:<45} {c.gap:>4} {c.pixels:<8} {c.metric:<15} {c.expected:>9.3f} "
              f"{c.actual:>9.4f} {c.difference:>8.5f}  {status}")
    failed = [c for c in checks if not c.passed(tolerance)]
    checked = [c for c in checks if c.skipped is None]
    print(f"regression: {len(checked) - len(failed)}/{len(checked)} passed, {len(checks) - len(checked)} skipped"
          + (f", {len(failed)} FAILED" if failed else ""))
    return not failed


def print_table(summary: list[dict[str, Any]]) -> None:
    metrics = ("gt_floor_depth", "gt_floor_rgb", "m1", "m2_floor", "m2")
    print("\nvisible pixels: median over pairs of each pair's median relative error x100")
    for name in dict.fromkeys(e["sequence"] for e in summary):
        print(name)
        print(f"  {'gap_ms':>6} {'pairs':>5} {'frac':>6}" + "".join(f" {m:>14}" for m in metrics))
        for e in (e for e in summary if e["sequence"] == name and e["pixels"] == "visible"):
            print(f"  {e['gap_ms']:>6} {e['pairs']:>5} {e['frac_median']:>6.3f}"
                  + "".join(f" {e[f'{m}_median']:>14.3f}" for m in metrics))


def consistency_record(
    args: argparse.Namespace, run: dict[str, Any], outcomes: dict[str, SequenceOutcome], reused: list[str],
    previous: dict[str, Any], start_time: str, seconds: float,
) -> dict[str, Any]:
    sequences = {name: {**asdict(o), "reused": False} for name, o in outcomes.items()}
    for name in reused:
        sequences[name] = {**previous.get(name, {}), "reused": True}
    return {
        "predictions": {"dir": str(args.predictions), "run": run},
        "commits": {"project": git_commit(REPO_ROOT), "pointdit": git_commit(FORK_DIR)},
        "arguments": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "protocol": PROTOCOL,
        "sequences_config": yaml.safe_load(args.sequences_config.read_text()),
        "sequences": sequences,
        "cpus": available_cpus(),
        "start_time": start_time,
        "end_time": now(),
        "seconds": round(seconds, 1),
    }


def previous_sequences(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text()).get("sequences", {})


def run_consistency(args: argparse.Namespace) -> bool:
    """True if the regression check passed."""
    validate(args)
    start_time, started = now(), time.perf_counter()
    run = read_run(args.predictions)
    names = select_sequences(run, args.sequences_config, args.sequences)
    unknown_maps = [name for name in args.save_maps if name not in names]
    if unknown_maps:
        raise ValueError(f"--save-maps names sequences that are not evaluated: {unknown_maps}; evaluating {names}")
    jobs = {name: sequence_job(args, name) for name in names}
    plans = {name: plan_sequence(job, run) for name, job in jobs.items()}

    args.out.mkdir(parents=True, exist_ok=True)
    todo, reused = [], []
    for name, job in jobs.items():
        reason = incomplete_reason(args.out, name, plans[name], job.save_maps)
        (todo if reason else reused).append(name)
        print(f"{name}: {len(plans[name])} pairs, " + (f"to evaluate ({reason})" if reason else "complete, skipping"))
    outcomes = run_jobs([jobs[name] for name in todo], args.workers)

    rows = {name: read_pairs(pairs_csv(args.out, name)) for name in names}
    summary = summarize(rows)
    write_summary(args.out / "summary.csv", summary)
    record_path = args.out / "consistency.json"
    record = consistency_record(args, run, outcomes, reused, previous_sequences(record_path), start_time,
                                time.perf_counter() - started)
    atomic_write(record_path, lambda f: f.write(json.dumps(record, indent=2) + "\n"))
    print_table(summary)
    for name, o in outcomes.items():
        print(f"{name}: {o.seconds / 60:.1f} min ({o.frame_seconds / 60:.1f} frames, {o.pair_seconds / 60:.1f} "
              f"pairs), peak {o.peak_mem_mib:.0f} MiB")

    expected = json.loads(args.expected.read_text())
    passed = report_regression(regression_checks(rows, expected), expected["tolerance"])
    print(f"wrote {args.out}")
    return passed


def main() -> None:
    args = parse_args()
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    if not run_consistency(args):
        sys.exit(1)


if __name__ == "__main__":
    main()
