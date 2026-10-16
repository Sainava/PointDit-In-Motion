"""Persistence decomposition of cached predictions over all frame pairs at fixed time gaps: per pair, frame i's and
frame j's relative depth errors against GT_j on the M1 pixels, their correlation and the persistent fraction.
Writes persistence_<sequence>.csv, persistence_summary.csv and persistence.json, then checks every pair's median
|E_i − E_j| against the M1 median of the consistency run."""

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
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pim.consistency.evaluation import (  # noqa: E402
    PAIR_TOLERANCE_MS, PairTask, SequenceJob, atomic_write, cell_value, init_worker, load_window, pairs_csv,
    plan_pairs, read_pairs,
)
from pim.consistency.persistence import (  # noqa: E402
    REGIONS, REGRESSION_TOLERANCE, STATS, SequenceOutcome, complete, evaluate_sequence, m1_regression,
    persistence_csv, read_rows,
)
from pim.evaluation import check_rgb_file, prediction_path  # noqa: E402
from pim.paths import FORK_DIR, REPO_ROOT  # noqa: E402
from pim.runs import available_cpus, check_window, git_commit, now, read_run, select_sequences  # noqa: E402

DEFAULT_GAPS_MS = (33.0, 100.0, 333.0)
REFERENCE = REPO_ROOT / "results" / "consistency" / "baseline"
SUMMARY_STATS = STATS[1:]
SUMMARY_COLUMNS = ("sequence", "gap_ms", "region", "pairs", "too_few_pairs",
                   *(f"{s}_{q}" for s in SUMMARY_STATS for q in ("median", "iqr", "n")))
PROTOCOL = {
    "pairs_and_pixels": "as consistency_eval.py's M1: same pairing, visibility (depth-timestamp poses) and frame i's "
                        "GT-aligned depth warped into j with RGB-timestamp poses; scored where visible and hit",
    "errors": "E_i = (z_i->j - z_gt_j) / z_gt_j, E_j = (z_j - z_gt_j) / z_gt_j, both GT-aligned per frame by "
              "upstream compute_metrics",
    "statistics": "median |E_i|, |E_j|, |E_i - E_j| (x100); Pearson and Spearman of E_i and E_j; persistent fraction "
                  "P = 1 - Var(E_i - E_j) / (Var(E_i) + Var(E_j))",
    "regions": "visible, interior (visible minus the GT_j depth-edge band), and near/mid/far thirds of GT_j depth "
               "over the visible pixels; a region with fewer than 1000 scored pixels is NaN",
    "regression": f"visible and interior median |E_i - E_j| equal m1_visible_median and m1_interior_median of the "
                  f"consistency pairs CSV within {REGRESSION_TOLERANCE:g}; pair keys must match exactly",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True, help="run directory holding run.json")
    parser.add_argument("--data-root", type=Path, required=True, help="folder containing the sequence folders")
    parser.add_argument("--sequences-config", type=Path, default=Path("configs/sequences.yaml"))
    parser.add_argument("--sequences", nargs="+", help="default: every sequence in run.json")
    parser.add_argument("--gaps-ms", type=float, nargs="+", default=list(DEFAULT_GAPS_MS))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1,
                        help="one sequence per worker process; each needs ~3.5 GB of RAM")
    parser.add_argument("--threads", type=int, default=1, help="torch/OpenMP threads per worker")
    parser.add_argument("--reference", type=Path, default=REFERENCE,
                        help="consistency_eval.py output holding pairs_<sequence>.csv for the regression check")
    return parser.parse_args(argv)


def validate(args: argparse.Namespace) -> None:
    if args.workers < 1:
        raise ValueError(f"--workers must be at least 1, got {args.workers}")
    if args.threads < 1:
        raise ValueError(f"--threads must be at least 1, got {args.threads}")
    if any(not (math.isfinite(g) and g > 0) for g in args.gaps_ms) or len(set(args.gaps_ms)) != len(args.gaps_ms):
        raise ValueError(f"--gaps-ms must be distinct positive values, got {args.gaps_ms}")
    if not args.data_root.is_dir():
        raise FileNotFoundError(f"data root {args.data_root} does not exist")
    if not args.reference.is_dir():
        raise FileNotFoundError(f"reference consistency results {args.reference} not found")


def check_reference(reference: Path, names: list[str]) -> None:
    missing = [str(pairs_csv(reference, name)) for name in names if not pairs_csv(reference, name).is_file()]
    if missing:
        raise FileNotFoundError(f"reference pair CSVs not found: {missing}")


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


def summarize(rows_by_sequence: dict[str, list[dict[str, str]]]) -> list[dict[str, Any]]:
    summary = []
    for name, rows in rows_by_sequence.items():
        for gap in sorted({float(r["gap_ms"]) for r in rows}):
            at_gap = [r for r in rows if float(r["gap_ms"]) == gap]
            for region in REGIONS:
                entry: dict[str, Any] = {"sequence": name, "gap_ms": f"{gap:g}", "region": region,
                                         "pairs": len(at_gap),
                                         "too_few_pairs": sum(r["too_few_pixels"] == "True" for r in at_gap)}
                for stat in SUMMARY_STATS:
                    values = [cell_value(r[f"{region}_{stat}"]) for r in at_gap]
                    entry.update({f"{stat}_median": median(values), f"{stat}_iqr": iqr(values),
                                  f"{stat}_n": int(finite(values).size)})
                summary.append(entry)
    return summary


def write_summary(path: Path, summary: list[dict[str, Any]]) -> None:
    def write(f) -> None:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for entry in summary:
            writer.writerow({k: ("" if isinstance(v, float) and math.isnan(v) else v) for k, v in entry.items()})

    atomic_write(path, write, newline="")


def print_table(summary: list[dict[str, Any]]) -> None:
    stats = ("median_abs_e_i", "median_abs_e_j", "median_abs_diff", "pearson", "spearman", "persistent_fraction")
    print("\nmedian over pairs (visible pixels; medians x100)")
    for name in dict.fromkeys(e["sequence"] for e in summary):
        print(name)
        print(f"  {'gap_ms':>6} {'pairs':>5}" + "".join(f" {s:>20}" for s in stats))
        for e in (e for e in summary if e["sequence"] == name and e["region"] == "visible"):
            print(f"  {e['gap_ms']:>6} {e['pairs']:>5}" + "".join(f" {e[f'{s}_median']:>20.4f}" for s in stats))


def report_regression(rows: dict[str, list[dict[str, str]]], reference: Path) -> bool:
    print(f"\nregression: median |E_i - E_j| vs M1 medians in {reference} (tolerance {REGRESSION_TOLERANCE:g}, pair "
          "keys exact)")
    passed = True
    for name, sequence_rows in rows.items():
        largest, failures = m1_regression(sequence_rows, read_pairs(pairs_csv(reference, name)))
        print(f"  {name}: {len(sequence_rows)} pairs, largest difference {largest:.2e}, "
              + (f"{len(failures)} FAILED" if failures else "ok"))
        for failure in failures[:10]:
            print(f"    {failure}")
        passed &= not failures
    print("regression: " + ("passed" if passed else "FAILED"))
    return passed


def persistence_record(args: argparse.Namespace, run: dict[str, Any], outcomes: dict[str, SequenceOutcome],
                       reused: list[str], start_time: str, seconds: float) -> dict[str, Any]:
    return {
        "predictions": {"dir": str(args.predictions), "run": run},
        "commits": {"project": git_commit(REPO_ROOT), "pointdit": git_commit(FORK_DIR)},
        "arguments": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "protocol": {"pairing": f"frame j is the window frame closest in time to t_i + gap, kept if within "
                                f"{PAIR_TOLERANCE_MS:g} ms", **PROTOCOL},
        "sequences": {**{name: asdict(o) for name, o in outcomes.items()}, **{name: "reused" for name in reused}},
        "cpus": available_cpus(),
        "start_time": start_time,
        "end_time": now(),
        "seconds": round(seconds, 1),
    }


def run_persistence(args: argparse.Namespace) -> bool:
    """True if the regression check passed."""
    validate(args)
    start_time, started = now(), time.perf_counter()
    run = read_run(args.predictions)
    names = select_sequences(run, args.sequences_config, args.sequences)
    check_reference(args.reference, names)
    jobs = {name: SequenceJob(name, args.data_root, args.sequences_config, args.predictions, tuple(args.gaps_ms),
                              args.out, False, args.threads) for name in names}
    plans = {name: plan_sequence(job, run) for name, job in jobs.items()}

    args.out.mkdir(parents=True, exist_ok=True)
    todo = [name for name in names if not complete(args.out, name, plans[name])]
    reused = [name for name in names if name not in todo]
    for name in names:
        print(f"{name}: {len(plans[name])} pairs, " + ("complete, skipping" if name in reused else "to evaluate"))
    outcomes = run_jobs([jobs[name] for name in todo], args.workers)

    rows = {name: read_rows(persistence_csv(args.out, name)) for name in names}
    summary = summarize(rows)
    write_summary(args.out / "persistence_summary.csv", summary)
    record = persistence_record(args, run, outcomes, reused, start_time, time.perf_counter() - started)
    atomic_write(args.out / "persistence.json", lambda f: f.write(json.dumps(record, indent=2) + "\n"))
    print_table(summary)
    for name, o in outcomes.items():
        print(f"{name}: {o.seconds / 60:.1f} min ({o.frame_seconds / 60:.1f} frames), peak {o.peak_mem_mib:.0f} MiB")
    print(f"wrote {args.out}")
    return report_regression(rows, args.reference)


def main() -> None:
    args = parse_args()
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    if not run_persistence(args):
        sys.exit(1)


if __name__ == "__main__":
    main()
