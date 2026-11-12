"""ORACLE-SCALE fusion, step 2. Fuse the GT-aligned depth cache of fusion_prepare.py into target frames for every
configuration and score each fused depth against GT. This is an upper bound, since every source was aligned to its
own GT first. Writes oracle_fusion_<sequence>.csv, oracle_fusion_summary.csv, oracle_fusion.json and
regression_status.json, and fails if the baseline_self rows or the cache do not reproduce per_frame.csv. The
baseline_self check runs after every output is written and exits with code 3 on a mismatch; any other failure exits
with code 1. --preflight scores baseline_self on the first targets of each sequence and writes nothing.
--summarize-only rebuilds the summary and json from existing per-sequence CSVs and reruns the baseline_self check."""

from __future__ import annotations

import argparse
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

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pim.consistency.evaluation import atomic_write, init_worker  # noqa: E402
from pim.evaluation import set_torch_threads  # noqa: E402
from pim.fusion.oracle import (  # noqa: E402
    BASELINE_SELF_TOLERANCE, DIVERSE_STRIDE_MS, PREPARE_RECORD, REGRESSION_TOLERANCE, SCALE, SCORE_THREADS, SEED,
    STORED_REL_D_TOLERANCE, TARGET_COUNT, TARGET_FIRST, WORST_SHOWN, EvalJob, EvalOutcome, Mismatch, RegressionError,
    SequenceContext, ViewStore, baseline_self_comparison, cache_failures, check_targets, complete, default_targets,
    evaluate_sequence, evaluate_target, expected_keys, fusion_csv, load_window, read_baseline, read_rows,
    regression_positions, resolve_specs, summarize, write_summary,
)
from pim.paths import FORK_DIR, REPO_ROOT  # noqa: E402
from pim.runs import available_cpus, git_commit, now  # noqa: E402

PROTOCOL = {
    "scale": f"{SCALE}. ORACLE. Every source and the target's own layer are the frame's prediction aligned to that "
             "frame's GT, so all results are an upper bound for fusion with estimated scale",
    "targets": f"kept-frame indices at window positions {TARGET_FIRST} to {TARGET_FIRST + TARGET_COUNT - 1} unless "
               "--targets or --all-targets is given; sources come from the whole window",
    "sources": f"chosen by timestamp with the engine's select_sources over the whole window; diverse stride "
               f"{DIVERSE_STRIDE_MS:g} ms",
    "source_validity": "finite and greater than 0 in the aligned depth; the GT mask is never used for sources",
    "poses": "RGB-timestamp GT poses; the GT ceiling uses depth-timestamp poses for sources and target",
    "scoring": "rel_d and delta1_d x100 with upstream rel_depth and delta1_depth over GT-valid target pixels, no "
               "realignment; holes take the target's own aligned depth; rel_d near, mid and far over GT depth "
               "thirds; rel_d_covered over GT-valid pixels that are not holes; median_n_sources counts every layer "
               "that survived aggregation over GT-valid pixels, the target's own layer included",
    "gt_ceiling": "GT depth (valid where GT is valid) fused with the causal_diverse_w4 settings with and without "
                  "the target's own GT layer, and with the rung d settings without it; rel_d_covered is also split "
                  "into interior pixels and pixels inside depth_edge_band of the target GT",
    "controls": "shuffled_poses gives each source the pose of a different random window frame; pose_noise adds "
                "per-axis Gaussian noise to source translations (mm) and rotation vectors (deg); both use --seed",
    "regression": f"baseline_self rel_d equals per_frame.csv within {BASELINE_SELF_TOLERANCE:g} (absolute, percent) "
                  f"for every target; the cached aligned depth of a few frames per sequence rescored directly "
                  f"equals it within {REGRESSION_TOLERANCE:g}, and its stored rel_d within "
                  f"{STORED_REL_D_TOLERANCE:g}",
}
SUMMARY_CSV = "oracle_fusion_summary.csv"
RECORD_JSON = "oracle_fusion.json"
STATUS_JSON = "regression_status.json"
PREFLIGHT_TARGETS = 3
EXIT_BASELINE_MISMATCH = 3


class BaselineSelfMismatch(RegressionError):
    """baseline_self differs from per_frame.csv after every output was written."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, help="aligned depth cache written by fusion_prepare.py")
    parser.add_argument("--data-root", type=Path, help="folder containing the sequence folders")
    parser.add_argument("--sequences-config", type=Path, default=Path("configs/sequences.yaml"))
    parser.add_argument("--baseline-csv", type=Path, required=True, help="per_frame.csv of the baseline evaluation")
    parser.add_argument("--sequences", nargs="+",
                        help="sequences to evaluate, default every sequence in the cache; the summary and json "
                             "still cover every oracle_fusion_<sequence>.csv in --out. With --summarize-only, the "
                             "sequences to summarize")
    parser.add_argument("--targets", type=Path,
                        help="YAML mapping targets to {sequence: [kept-frame indices]}; default is window positions "
                             f"{TARGET_FIRST} to {TARGET_FIRST + TARGET_COUNT - 1}")
    parser.add_argument("--all-targets", action="store_true", help="use every window frame as a target")
    parser.add_argument("--configs", nargs="+", help="configuration names; default is all, baseline_self is always run")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1, help="one sequence per worker process")
    parser.add_argument("--threads", type=int, default=1, help="OpenMP threads per worker")
    parser.add_argument("--score-threads", type=int, default=SCORE_THREADS,
                        help="torch threads for the float32 score reduction; must match the baseline evaluation run")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--regression-frames", type=int, default=5, help="cached frames per sequence to rescore")
    parser.add_argument("--summarize-only", action="store_true",
                        help="rebuild the summary and json from the oracle_fusion_<sequence>.csv files in --out and "
                             "rerun the baseline_self check; needs no cache, data or predictions")
    parser.add_argument("--preflight", action="store_true",
                        help=f"score only baseline_self on the first {PREFLIGHT_TARGETS} targets of each sequence "
                             "against per_frame.csv and write nothing")
    args = parser.parse_args(argv)
    if args.preflight and args.summarize_only:
        parser.error("give --preflight or --summarize-only, not both")
    if not args.summarize_only and (args.cache is None or args.data_root is None):
        parser.error("--cache and --data-root are required unless --summarize-only is given")
    return args


def validate(args: argparse.Namespace) -> None:
    for name in ("workers", "threads", "score_threads", "regression_frames"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be at least 1, got {getattr(args, name)}")
    if args.targets and args.all_targets:
        raise ValueError("give --targets or --all-targets, not both")
    for path in (args.data_root, args.cache):
        if not path.is_dir():
            raise FileNotFoundError(f"{path} does not exist")
    if not args.baseline_csv.is_file():
        raise FileNotFoundError(f"baseline per_frame.csv {args.baseline_csv} not found")


def read_prepare_record(cache: Path) -> dict[str, Any]:
    path = cache / PREPARE_RECORD
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; run fusion_prepare.py into {cache} first")
    record = json.loads(path.read_text())
    if record.get("scale") != SCALE:
        raise ValueError(f"{path} has scale {record.get('scale')!r}, expected {SCALE}")
    return record


def requested_targets(args: argparse.Namespace) -> dict[str, list[int]] | None:
    if args.targets is None:
        return None
    raw = yaml.safe_load(args.targets.read_text())
    entries = raw.get("targets") if isinstance(raw, dict) else None
    if not isinstance(entries, dict) or not entries:
        raise ValueError(f"{args.targets} needs a non-empty targets mapping of sequence to kept-frame indices")
    return {name: [int(i) for i in indices] for name, indices in entries.items()}


def run_jobs(jobs: list[EvalJob], workers: int) -> dict[str, EvalOutcome]:
    if workers == 1 or len(jobs) <= 1:
        return {job.name: evaluate_sequence(job) for job in jobs}
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
            print(f"{outcome.name} done in {outcome.seconds / 60:.1f} min, peak {outcome.peak_mem_mib:.0f} MiB",
                  flush=True)
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    return outcomes


def baseline_self_report(comparison: list[Mismatch], tolerance: float = BASELINE_SELF_TOLERANCE) -> dict[str, Any]:
    over = [m for m in comparison if m.diff > tolerance]
    max_diff = max((m.diff for m in comparison), default=0.0)
    print(f"\nbaseline_self against per_frame.csv. {len(over)} of {len(comparison)} frames over {tolerance:g}, "
          f"max |diff| {max_diff:.3g}")
    for m in comparison[:WORST_SHOWN]:
        print(f"  {'FAIL' if m.diff > tolerance else 'ok  '} {m.describe(tolerance)}")
    return {"baseline_self_rows": len(comparison), "baseline_self_tolerance": tolerance,
            "frames_over_tolerance": len(over), "max_abs_diff": _json_float(max_diff),
            "worst": [{"sequence": m.sequence, "frame_index": m.frame_index, "rel_d": _json_float(m.actual),
                       "per_frame_rel_d": _json_float(m.expected), "abs_diff": _json_float(m.diff)}
                      for m in comparison[:WORST_SHOWN]],
            "passed": not over}


def write_status(out: Path, regression: dict[str, Any]) -> None:
    status = {"status": "pass" if regression["passed"] else "fail",
              "frames_over_tolerance": regression["frames_over_tolerance"],
              "frames_compared": regression["baseline_self_rows"],
              "tolerance": regression["baseline_self_tolerance"],
              "max_abs_diff": regression["max_abs_diff"],
              "worst": regression["worst"]}
    atomic_write(out / STATUS_JSON, lambda f: f.write(json.dumps(status, indent=2) + "\n"))


def check_baseline_self(regression: dict[str, Any], out: Path) -> None:
    if not regression["passed"]:
        raise BaselineSelfMismatch(
            f"regression check of baseline_self against per_frame.csv FAILED for "
            f"{regression['frames_over_tolerance']} of {regression['baseline_self_rows']} frames (max |diff| "
            f"{regression['max_abs_diff']}, tolerance {regression['baseline_self_tolerance']:g}); every output is "
            f"already written to {out}")
    print(f"regression passed. {regression['baseline_self_rows']} baseline_self rows match per_frame.csv")


def _json_float(value: float) -> float | None:
    return value if math.isfinite(value) else None


def present_sequences(out: Path) -> list[str]:
    """Sequences with an oracle_fusion_<sequence>.csv in out."""
    prefix = fusion_csv(out, "").name.removesuffix(".csv")
    return sorted(p.stem.removeprefix(prefix) for p in out.glob(f"{prefix}*.csv") if p.name != SUMMARY_CSV)


def row_targets(rows: list[dict[str, str]], name: str) -> list[int]:
    return sorted({int(r["target_index"]) for r in rows if r["sequence"] == name})


def write_outputs(out: Path, summary: list[dict[str, Any]], record: dict[str, Any]) -> None:
    write_summary(out / SUMMARY_CSV, summary)
    atomic_write(out / RECORD_JSON, lambda f: f.write(json.dumps(record, indent=2) + "\n"))
    write_status(out, record["regression"])


def fail(title: str, failures: list[str]) -> None:
    print(f"\n{title} FAILED", flush=True)
    for line in failures:
        print(f"  {line}")
    raise RegressionError(f"{title} failed for {len(failures)} value(s), first {failures[0]}")


def print_table(summary: list[dict[str, Any]]) -> None:
    print("\nORACLE scale (upper bound). median per-target rel_d x100, median change against baseline_self")
    for name in dict.fromkeys(e["sequence"] for e in summary):
        print(name)
        print(f"  {'config':<58} {'rel_d':>7} {'diff':>8} {'change%':>8} {'improved':>8} {'holes':>6}")
        for e in (e for e in summary if e["sequence"] == name):
            print(f"  {e['config']:<58} {e['rel_d_median']:>7.3f} {e['delta_vs_baseline_median']:>8.3f} "
                  f"{e['rel_change_pct_median']:>8.2f} {e['share_improved']:>8.2f} {e['hole_fraction_median']:>6.3f}")


def cached_sequences(args: argparse.Namespace, prepared: dict[str, Any]) -> tuple[list[str], list[str]]:
    available = list(prepared["sequences"])
    names = args.sequences or available
    missing = [n for n in names if n not in available]
    if missing:
        raise ValueError(f"sequences {missing} are not in the cache {args.cache}, which holds {available}")
    return available, names


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    """baseline_self on the first targets of each sequence against per_frame.csv; writes nothing."""
    validate(args)
    _, names = cached_sequences(args, read_prepare_record(args.cache))
    specs = resolve_specs(["baseline_self"])
    baseline = read_baseline(args.baseline_csv)
    explicit = requested_targets(args)
    set_torch_threads(args.score_threads)
    rows = []
    for name in names:
        sequence, frames = load_window(name, args.data_root, args.sequences_config)
        chosen = (explicit[name] if explicit is not None else default_targets(frames, args.all_targets))[
            :PREFLIGHT_TARGETS]
        check_targets(name, frames, chosen)
        ctx = SequenceContext(name, sequence.K, frames, args.seed)
        store = ViewStore(args.cache, name, frames)
        position = {f.index: k for k, f in enumerate(frames)}
        rows.extend(r for t in chosen for r in evaluate_target(ctx, store, specs, position[t]))
    regression = baseline_self_report(baseline_self_comparison(rows, baseline))
    if not regression["passed"]:
        raise RegressionError(f"preflight FAILED. baseline_self differs from per_frame.csv for "
                              f"{regression['frames_over_tolerance']} of {regression['baseline_self_rows']} frames "
                              f"(max |diff| {regression['max_abs_diff']})")
    print(f"preflight passed. {regression['baseline_self_rows']} baseline_self rows match per_frame.csv")
    return regression


def run_fusion_eval(args: argparse.Namespace) -> dict[str, Any]:
    validate(args)
    start_time, started = now(), time.perf_counter()
    prepared = read_prepare_record(args.cache)
    available, names = cached_sequences(args, prepared)
    specs = resolve_specs(args.configs)
    baseline = read_baseline(args.baseline_csv)
    explicit = requested_targets(args)
    set_torch_threads(args.score_threads)

    jobs, targets = [], {}
    for name in names:
        _, frames = load_window(name, args.data_root, args.sequences_config)
        window = {"start": frames[0].index, "count": len(frames)}
        if prepared["sequences"][name]["window"] != window:
            raise ValueError(f"{name} cache covers window {prepared['sequences'][name]['window']}, but "
                             f"{args.sequences_config} gives {window}")
        chosen = explicit[name] if explicit is not None else default_targets(frames, args.all_targets)
        check_targets(name, frames, chosen)
        targets[name] = chosen
        failures = cache_failures(args.cache, name, frames, regression_positions(len(frames), args.regression_frames),
                                  baseline)
        if failures:
            fail("regression check of the aligned depth cache against per_frame.csv", failures)
        print(f"{name} cache reproduces per_frame.csv rel_d on {args.regression_frames} frames", flush=True)
        jobs.append(EvalJob(name, args.data_root, args.sequences_config, args.cache, args.out, tuple(chosen),
                            tuple(s.name for s in specs), {i: v for (n, i), v in baseline.items() if n == name},
                            args.seed, args.threads, args.score_threads))

    args.out.mkdir(parents=True, exist_ok=True)
    todo = [j for j in jobs if not complete(fusion_csv(args.out, j.name), expected_keys(j.name, j.targets, specs))]
    reused = [j.name for j in jobs if j not in todo]
    for name in reused:
        print(f"{name} {fusion_csv(args.out, name).name} is complete, skipping")
    outcomes = run_jobs(todo, args.workers)

    present = present_sequences(args.out)
    earlier = [n for n in available if n in present and n not in names] + [n for n in present if n not in available]
    summarized = [n for n in available if n in names or n in earlier] + [n for n in earlier if n not in available]
    for name in earlier:
        print(f"{name} {fusion_csv(args.out, name).name} was written by an earlier run, included in the summary")
    rows = [r for name in summarized for r in read_rows(fusion_csv(args.out, name))]
    summary = summarize(rows)
    regression = baseline_self_report(baseline_self_comparison(rows, baseline))
    record = {
        "scale": SCALE,
        "cache": {"dir": str(args.cache), "prepare": prepared},
        "baseline_csv": str(args.baseline_csv),
        "commits": {"project": git_commit(REPO_ROOT), "pointdit": git_commit(FORK_DIR)},
        "arguments": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "protocol": PROTOCOL,
        "configs": {s.name: s.describe() for s in specs},
        "targets": {**targets, **{n: row_targets(rows, n) for n in earlier}},
        "summarized_sequences": summarized,
        "sequences": {**{n: {"reused": True} for n in reused},
                      **{n: {"reused": True, "requested": False} for n in earlier},
                      **{n: {**asdict(o), "reused": False} for n, o in outcomes.items()}},
        "regression": {**regression, "cache_frames_per_sequence": args.regression_frames},
        "cpus": available_cpus(),
        "start_time": start_time,
        "end_time": now(),
        "seconds": round(time.perf_counter() - started, 1),
    }
    write_outputs(args.out, summary, record)
    print_table(summary)
    print(f"\nwrote {args.out}")
    check_baseline_self(record["regression"], args.out)
    return record


def summarize_existing(args: argparse.Namespace) -> dict[str, Any]:
    """Rebuild the summary and json from the per-sequence CSVs in --out, then rerun the baseline_self check."""
    if not args.baseline_csv.is_file():
        raise FileNotFoundError(f"baseline per_frame.csv {args.baseline_csv} not found")
    names = args.sequences or present_sequences(args.out)
    missing = [n for n in names if not fusion_csv(args.out, n).is_file()]
    if not names or missing:
        raise FileNotFoundError(f"no oracle_fusion_<sequence>.csv in {args.out} for {missing or 'any sequence'}")
    rows = [r for name in names for r in read_rows(fusion_csv(args.out, name))]
    summary = summarize(rows)
    regression = baseline_self_report(baseline_self_comparison(rows, read_baseline(args.baseline_csv)))

    path = args.out / RECORD_JSON
    record = json.loads(path.read_text()) if path.is_file() else {"scale": SCALE, "protocol": PROTOCOL}
    cache_frames = record.get("regression", {}).get("cache_frames_per_sequence")
    record["regression"] = {**regression, "cache_frames_per_sequence": cache_frames}
    record["summarized"] = {
        "time": now(), "baseline_csv": str(args.baseline_csv), "sequences": names,
        "configs": list(dict.fromkeys(r["config"] for r in rows)),
        "targets": {n: row_targets(rows, n) for n in names},
    }
    write_outputs(args.out, summary, record)
    print_table(summary)
    print(f"\nwrote {args.out / SUMMARY_CSV} and {path}")
    check_baseline_self(record["regression"], args.out)
    return record


def main() -> None:
    args = parse_args()
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    try:
        if args.preflight:
            run_preflight(args)
        elif args.summarize_only:
            summarize_existing(args)
        else:
            run_fusion_eval(args)
    except BaselineSelfMismatch as error:
        banner = "=" * 30 + " FAILED " + "=" * 30
        print(f"\n{banner}\n{error}\nsee {args.out / STATUS_JSON}\n{banner}", flush=True)
        sys.exit(EXIT_BASELINE_MISMATCH)
    except RegressionError as error:
        print(f"\n{error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
