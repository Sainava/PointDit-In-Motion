"""ORACLE-SCALE fusion, step 1. For every window frame, align the cached prediction to that frame's own GT with
upstream compute_metrics (as evaluate.py does) and save the aligned depth as float32 npz with the frame index and the
frame's baseline rel_d. Frames already cached are skipped. Writes oracle_prepare.json with provenance."""

from __future__ import annotations

import argparse
import json
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
from pim.evaluation import check_rgb_file, prediction_path  # noqa: E402
from pim.fusion.oracle import (  # noqa: E402
    PREPARE_RECORD, SCALE, Aligner, PrepareJob, PrepareOutcome, align_frame, load_window, prepare_sequence,
)
from pim.metrics.geometry import MAX_ALIGNMENT_PIXELS  # noqa: E402
from pim.paths import FORK_DIR, REPO_ROOT  # noqa: E402
from pim.runs import available_cpus, check_window, git_commit, now, read_run, select_sequences  # noqa: E402

PROTOCOL = {
    "scale": f"{SCALE}. ORACLE. Each prediction is aligned to its own frame's GT, so fusion of these depths is an "
             "upper bound for fusion with estimated scale",
    "alignment": "upstream metrics.compute_metrics depth scale and shift fitted on the 64x64 masked nearest "
                 f"subsample (at most {MAX_ALIGNMENT_PIXELS} pixels per fit) over GT-valid pixels, the same call "
                 "as evaluate.py, so rel_d reproduces per_frame.csv",
    "cache": "<out>/<sequence>/<frame index>.npz with aligned_depth (H, W) float32, frame_index, rel_d, delta1_d, "
             "num_valid, skipped, rgb_file and scale",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True, help="run directory holding run.json")
    parser.add_argument("--data-root", type=Path, required=True, help="folder containing the sequence folders")
    parser.add_argument("--sequences-config", type=Path, default=Path("configs/sequences.yaml"))
    parser.add_argument("--sequences", nargs="+", help="default is every sequence in run.json")
    parser.add_argument("--out", type=Path, required=True, help="aligned depth cache directory")
    parser.add_argument("--workers", type=int, default=1,
                        help="one sequence per worker process, each needs about 3.5 GB of RAM")
    parser.add_argument("--threads", type=int, default=1, help="torch and OpenMP threads per worker")
    return parser.parse_args(argv)


def validate(args: argparse.Namespace) -> None:
    if args.workers < 1:
        raise ValueError(f"--workers must be at least 1, got {args.workers}")
    if args.threads < 1:
        raise ValueError(f"--threads must be at least 1, got {args.threads}")
    if not args.data_root.is_dir():
        raise FileNotFoundError(f"data root {args.data_root} does not exist")


def run_jobs(jobs: list[PrepareJob], workers: int, align: Aligner) -> dict[str, PrepareOutcome]:
    if workers == 1 or len(jobs) <= 1:
        return {job.name: prepare_sequence(job, align) for job in jobs}
    # spawn because torch's thread pools are not fork-safe, one sequence per child so each peak is one sequence.
    executor = ProcessPoolExecutor(
        max_workers=min(workers, len(jobs)), mp_context=multiprocessing.get_context("spawn"),
        initializer=init_worker, initargs=(jobs[0].threads,), max_tasks_per_child=1,
    )
    outcomes = {}
    try:
        futures = {executor.submit(prepare_sequence, job, align): job.name for job in jobs}
        for future in as_completed(futures):
            outcome = future.result()
            outcomes[outcome.name] = outcome
            print(f"{outcome.name} done in {outcome.seconds / 60:.1f} min, peak {outcome.peak_mem_mib:.0f} MiB",
                  flush=True)
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    return outcomes


def previous_record(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.is_file() else {}


def run_prepare(args: argparse.Namespace, align: Aligner = align_frame) -> dict[str, Any]:
    validate(args)
    start_time, started = now(), time.perf_counter()
    run = read_run(args.predictions)
    names = select_sequences(run, args.sequences_config, args.sequences)
    jobs, windows = [], {}
    for name in names:
        sequence, frames = load_window(name, args.data_root, args.sequences_config)
        check_window(name, run, frames, args.sequences_config)
        for frame in frames:
            check_rgb_file(prediction_path(args.predictions, name, frame.index), frame, sequence.root)
        windows[name] = {"start": frames[0].index, "count": len(frames)}
        jobs.append(PrepareJob(name, args.data_root, args.sequences_config, args.predictions, args.out, args.threads))
    record_path = args.out / PREPARE_RECORD
    previous = previous_record(record_path)
    if previous and previous.get("predictions", {}).get("run") != run:
        raise ValueError(f"{args.out} was prepared from a different prediction run; use a new --out")
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"ORACLE scale. aligning {len(names)} sequences into {args.out} with {args.workers} worker(s)", flush=True)
    outcomes = run_jobs(jobs, args.workers, align)

    sequences = {**previous.get("sequences", {}),
                 **{name: {**asdict(outcomes[name]), "window": windows[name]} for name in names}}
    record = {
        "scale": SCALE,
        "predictions": {"dir": str(args.predictions), "run": run},
        "commits": {"project": git_commit(REPO_ROOT), "pointdit": git_commit(FORK_DIR)},
        "arguments": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "protocol": PROTOCOL,
        "sequences_config": yaml.safe_load(args.sequences_config.read_text()),
        "sequences": sequences,
        "cpus": available_cpus(),
        "start_time": start_time,
        "end_time": now(),
        "seconds": round(time.perf_counter() - started, 1),
    }
    atomic_write(record_path, lambda f: f.write(json.dumps(record, indent=2) + "\n"))
    for name, o in outcomes.items():
        print(f"{name} {o.written} aligned, {o.existing} already cached, {o.seconds / 60:.1f} min, "
              f"peak {o.peak_mem_mib:.0f} MiB")
    print(f"wrote {args.out}")
    return record


def main() -> None:
    args = parse_args()
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    run_prepare(args)


if __name__ == "__main__":
    main()
