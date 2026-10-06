"""Evaluate cached predictions frame by frame against TUM GT depth; write per_frame.csv and eval.json."""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pim.data.tum import apply_window, load_sequence  # noqa: E402
from pim.evaluation import (  # noqa: E402
    COLUMNS, METRICS, FrameTask, build_tasks, check_rgb_file, evaluate_task, set_torch_threads,
)
from pim.metrics.geometry import MIN_VALID_PIXELS  # noqa: E402
from pim.paths import FORK_DIR, REPO_ROOT  # noqa: E402

PROTOCOL = {
    "mask": "GT depth > 0 only; the prediction's sky mask is not applied (pred_valid=None), as upstream",
    "alignment": "upstream metrics.compute_metrics: points scale + xyz shift, depth scale + shift, "
                 "each fitted on a 64x64 masked nearest subsample and applied to the full frame",
    "resolution": "full 640x480 frames, no resize or crop",
    "metrics": "rel_p, delta1_p, rel_d, delta1_d from upstream metrics/metrics.py, x100",
    "min_valid_pixels": MIN_VALID_PIXELS,
}
PROGRESS_EVERY = 50


@dataclass(frozen=True)
class SequenceResult:
    frames: int
    skipped: int
    seconds: float


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def git_commit(repo: Path) -> dict[str, Any]:
    def git(*args: str) -> str:
        done = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
        if done.returncode != 0:
            raise RuntimeError(f"git {' '.join(args)} failed in {repo}: {done.stderr.strip()}")
        return done.stdout.strip()

    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}


def read_run(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "run.json"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; --predictions must be a run directory written by predict.py")
    return json.loads(path.read_text())


def select_sequences(run: dict[str, Any], sequences_config: Path, requested: list[str] | None) -> list[str]:
    predicted = list(run.get("sequences") or {})
    if not predicted:
        raise ValueError("run.json lists no predicted sequences")
    known = list((yaml.safe_load(sequences_config.read_text()) or {}).get("sequences") or {})
    unknown = [name for name in predicted if name not in known]
    if unknown:
        raise ValueError(f"run.json sequences {unknown} are not in {sequences_config}")
    if requested is None:
        return [name for name in known if name in predicted]
    missing = [name for name in requested if name not in predicted]
    if missing:
        raise ValueError(f"sequences {missing} were not predicted in this run; run.json has {predicted}")
    if len(set(requested)) != len(requested):
        raise ValueError(f"--sequences lists a sequence twice: {requested}")
    return requested


def sequence_tasks(
    name: str, run: dict[str, Any], run_dir: Path, data_root: Path, sequences_config: Path
) -> list[FrameTask]:
    sequence = load_sequence(data_root / name, sequences_config)
    frames = apply_window(sequence)
    window = {"start": frames[0].index, "count": len(frames)}
    if run["sequences"][name]["window"] != window:
        raise ValueError(f"{name}: predictions cover window {run['sequences'][name]['window']}, "
                         f"but {sequences_config} now gives {window}")
    return build_tasks(name, sequence.root, sequence.K, frames, run_dir)


def check_predictions(tasks: list[FrameTask]) -> None:
    for task in tasks:
        check_rgb_file(task.prediction, task.frame, task.sequence_root)


def available_cpus() -> int:
    return len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1


def evaluate_tasks(tasks: list[FrameTask], workers: int) -> Iterator[dict[str, Any]]:
    """Rows in task order; with workers > 1 the CPUs are split evenly between worker processes."""
    if workers == 1:
        yield from map(evaluate_task, tasks)
        return
    threads = max(1, available_cpus() // workers)
    # spawn rather than fork: torch's thread pools are not fork-safe.
    executor = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                                   initializer=set_torch_threads, initargs=(threads,))
    try:
        yield from executor.map(evaluate_task, tasks, chunksize=1)
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


def run_evaluation(
    run_dir: Path, data_root: Path, sequences_config: Path, requested: list[str] | None, out: Path, workers: int
) -> dict[str, Any]:
    if workers < 1:
        raise ValueError(f"--workers must be at least 1, got {workers}")
    if not data_root.is_dir():
        raise FileNotFoundError(f"data root {data_root} does not exist")
    run = read_run(run_dir)
    start_time = now()
    names = select_sequences(run, sequences_config, requested)
    tasks = {name: sequence_tasks(name, run, run_dir, data_root, sequences_config) for name in names}
    for sequence_list in tasks.values():
        check_predictions(sequence_list)

    out.mkdir(parents=True, exist_ok=True)
    partial = out / "per_frame.csv.partial"
    rows: dict[str, list[dict[str, Any]]] = {name: [] for name in names}
    results: dict[str, SequenceResult] = {}
    all_tasks = [task for sequence_list in tasks.values() for task in sequence_list]
    print(f"evaluating {len(all_tasks)} frames of {len(names)} sequences from {run_dir} "
          f"with {workers} worker(s)", flush=True)
    with open(partial, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        started = time.perf_counter()
        for row in evaluate_tasks(all_tasks, workers):
            writer.writerow(row)
            f.flush()
            name = row["sequence"]
            rows[name].append(row)
            done, count = len(rows[name]), len(tasks[name])
            elapsed = time.perf_counter() - started
            if done % PROGRESS_EVERY == 0 or done == count:
                print(f"  {name}: {done}/{count} frames, {elapsed / done:.2f} s/frame", flush=True)
            if done == count:
                results[name] = SequenceResult(count, sum(r["skipped"] for r in rows[name]), elapsed)
                started = time.perf_counter()
    os.replace(partial, out / "per_frame.csv")

    record = eval_record(run, run_dir, data_root, sequences_config, workers, results, start_time)
    (out / "eval.json").write_text(json.dumps(record, indent=2) + "\n")
    print_summary(rows)
    print(f"{record['seconds_per_frame']:.2f} s/frame overall; wrote {out / 'per_frame.csv'} and {out / 'eval.json'}")
    return record


def eval_record(
    run: dict[str, Any], run_dir: Path, data_root: Path, sequences_config: Path, workers: int,
    results: dict[str, SequenceResult], start_time: str,
) -> dict[str, Any]:
    frames = sum(r.frames for r in results.values())
    seconds = sum(r.seconds for r in results.values())
    return {
        "predictions": {"dir": str(run_dir), "run": run},
        "commits": {"project": git_commit(REPO_ROOT), "pointdit": git_commit(FORK_DIR)},
        "protocol": PROTOCOL,
        "data_root": str(data_root),
        "sequences_config": yaml.safe_load(sequences_config.read_text()),
        "sequences": {name: {"window": run["sequences"][name]["window"], "frames": r.frames,
                             "skipped": r.skipped, "seconds": round(r.seconds, 1)}
                      for name, r in results.items()},
        "frames": frames,
        "skipped": sum(r.skipped for r in results.values()),
        "workers": workers,
        "cpus": available_cpus(),
        "seconds_per_frame": seconds / frames,
        "start_time": start_time,
        "end_time": now(),
    }


def print_summary(rows: dict[str, list[dict[str, Any]]]) -> None:
    header = f"{'sequence':<45} {'frames':>6} {'skipped':>7}" + "".join(f" {m + ' mean/median':>22}" for m in METRICS)
    print(header)
    for name, sequence_rows in rows.items():
        kept = [r for r in sequence_rows if not r["skipped"]]
        cells = []
        for metric in METRICS:
            values = np.array([float(r[metric]) for r in kept])
            cells.append(f"{values.mean():>10.2f} /{np.median(values):>9.2f}" if len(values) else f"{'n/a':>22}")
        skipped = len(sequence_rows) - len(kept)
        print(f"{name:<45} {len(sequence_rows):>6} {skipped:>7}" + "".join(f" {c:>22}" for c in cells))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True, help="run directory holding run.json")
    parser.add_argument("--data-root", type=Path, required=True, help="folder containing the sequence folders")
    parser.add_argument("--sequences-config", type=Path, default=Path("configs/sequences.yaml"))
    parser.add_argument("--sequences", nargs="+", help="default: every sequence in run.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1,
                        help="worker processes; the CPUs are split between them and each needs ~3.5 GB of RAM")
    args = parser.parse_args()
    run_evaluation(args.predictions, args.data_root, args.sequences_config, args.sequences, args.out, args.workers)


if __name__ == "__main__":
    main()
