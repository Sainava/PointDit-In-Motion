"""Run PointDiT over each sequence's window and cache one compressed point map per frame."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import platform
import re
import subprocess
import sys
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Protocol

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pim.data.tum import Frame, TUMSequence, apply_window, load_sequence  # noqa: E402
from pim.paths import DINOV3_REPO_DIR, FORK_DIR, REPO_ROOT  # noqa: E402

WARMUP_RUNS = 2
CONFIG_KEYS = {"name", "inference"}
INFERENCE_KEYS = {"init", "seed"}
INITS = ("zeros", "noise")
RUN_NAME = re.compile(r"^[A-Za-z0-9._-]+$")


class Prediction(Protocol):
    points: np.ndarray
    valid_mask: np.ndarray
    latency_ms: float

    @property
    def peak_memory_gb(self) -> float: ...


class Runner(Protocol):
    def predict(self, image: np.ndarray, init: str, seed: int | None) -> Prediction: ...


@dataclass(frozen=True)
class RunConfig:
    name: str
    init: str
    seed: int | None

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "inference": {"init": self.init, "seed": self.seed}}


@dataclass(frozen=True)
class SequenceStats:
    frames: int
    predicted: int
    skipped: int
    mean_latency_ms: float
    peak_mem_gb: float


def load_run_config(path: Path) -> RunConfig:
    if not path.is_file():
        raise FileNotFoundError(f"run config {path} not found")
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a mapping at the top level")
    _check_keys(raw, CONFIG_KEYS, str(path))
    inference = raw["inference"]
    if not isinstance(inference, dict):
        raise ValueError(f"{path}: `inference` must be a mapping")
    _check_keys(inference, INFERENCE_KEYS, f"{path}: inference")

    name, init, seed = raw["name"], inference["init"], inference["seed"]
    if not isinstance(name, str) or not RUN_NAME.match(name):
        raise ValueError(f"{path}: name must match {RUN_NAME.pattern}, got {name!r}")
    if init not in INITS:
        raise ValueError(f"{path}: inference.init must be one of {INITS}, got {init!r}")
    if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
        raise ValueError(f"{path}: inference.seed must be an integer or null, got {seed!r}")
    return RunConfig(name, init, seed)


def _check_keys(mapping: dict, expected: set[str], where: str) -> None:
    unknown = sorted(set(mapping) - expected)
    missing = sorted(expected - set(mapping))
    if unknown or missing:
        raise ValueError(f"{where}: unknown keys {unknown}, missing keys {missing}; expected {sorted(expected)}")


def select_sequences(sequences_config: Path, requested: list[str] | None) -> list[str]:
    known = list((yaml.safe_load(sequences_config.read_text()) or {}).get("sequences") or {})
    if not known:
        raise ValueError(f"{sequences_config} lists no sequences")
    if requested is None:
        return known
    unknown = [name for name in requested if name not in known]
    if unknown:
        raise ValueError(f"sequences {unknown} are not in {sequences_config}; known: {known}")
    if len(set(requested)) != len(requested):
        raise ValueError(f"--sequences lists a sequence twice: {requested}")
    return requested


def prediction_path(sequence_dir: Path, frame: Frame) -> Path:
    return sequence_dir / f"{frame.index:05d}.npz"


def save_prediction(path: Path, frame: Frame, sequence_root: Path, result: Prediction) -> None:
    points = result.points.astype(np.float16)
    if not np.isfinite(points).all():
        raise ValueError(f"{path}: points are non-finite after the float16 cast "
                         f"(float32 max |value| {np.abs(result.points).max()})")
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "wb") as f:
            np.savez_compressed(
                f,
                points=points,
                valid=result.valid_mask.astype(bool),
                rgb_timestamp=np.float64(frame.timestamp),
                rgb_file=np.str_(frame.rgb_path.relative_to(sequence_root).as_posix()),
                latency_ms=np.float64(result.latency_ms),
                peak_mem_gb=np.float64(result.peak_memory_gb),
            )
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def predict_sequence(
    runner: Runner | None, config: RunConfig, sequence: TUMSequence, frames: list[Frame], run_dir: Path
) -> SequenceStats:
    sequence_dir = run_dir / sequence.name
    sequence_dir.mkdir(parents=True, exist_ok=True)
    # Left behind only if a process was killed mid-write; this process owns the sequence.
    for stale in sequence_dir.glob(".*.tmp"):
        stale.unlink()

    predicted = skipped = 0
    for frame in frames:
        path = prediction_path(sequence_dir, frame)
        if path.is_file():
            skipped += 1
            continue
        if runner is None:
            raise RuntimeError(f"{path} needs predicting but no model was loaded")
        result = runner.predict(frame.load_rgb(), init=config.init, seed=config.seed)
        save_prediction(path, frame, sequence.root, result)
        predicted += 1

    latencies, peaks = read_timings([prediction_path(sequence_dir, f) for f in frames])
    return SequenceStats(len(frames), predicted, skipped, float(np.mean(latencies)), float(np.max(peaks)))


def read_timings(paths: list[Path]) -> tuple[list[float], list[float]]:
    latencies, peaks = [], []
    for path in paths:
        with np.load(path) as data:
            latencies.append(float(data["latency_ms"]))
            peaks.append(float(data["peak_mem_gb"]))
    return latencies, peaks


def window_summary(frames: list[Frame]) -> dict[str, int]:
    return {"start": frames[0].index, "count": len(frames)}


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def run_record(run_dir: Path) -> Iterator[dict[str, Any]]:
    """Read-modify-write run.json under an exclusive lock shared by all processes of the run."""
    with open(run_dir / "run.json.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = run_dir / "run.json"
        record = json.loads(path.read_text()) if path.is_file() else {}
        yield record
        tmp = path.with_name(f".run.json.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(record, indent=2) + "\n")
        os.replace(tmp, path)


def register_invocation(
    record: dict[str, Any], config: RunConfig, windows: dict[str, dict[str, int]], invocation: dict[str, Any]
) -> None:
    if not record:
        record.update(name=config.name, config=config.as_dict(), start_time=None, end_time=None,
                      frames_predicted=0, frames_skipped=0, sequences={}, invocations=[])
    elif record["config"] != config.as_dict():
        raise ValueError(f"run.json holds config {record['config']} but this run uses {config.as_dict()}; "
                         "change the run name or remove the run directory")
    for name, window in windows.items():
        finished = record["sequences"].get(name)
        if finished and finished["window"] != window:
            raise ValueError(f"{name}: run.json holds window {finished['window']} but the config now gives "
                             f"{window}; change the run name or remove the run directory")
    record["invocations"].append(invocation)
    _refresh_totals(record)


def record_sequence(
    record: dict[str, Any], invocation_id: str, name: str, window: dict[str, int], stats: SequenceStats
) -> None:
    record["sequences"][name] = {
        "window": window,
        "frames": stats.frames,
        "mean_latency_ms": stats.mean_latency_ms,
        "peak_mem_gb": stats.peak_mem_gb,
        "invocation": invocation_id,
        "end_time": now(),
    }
    _invocation(record, invocation_id)["sequences"][name] = {
        "predicted": stats.predicted, "skipped": stats.skipped}
    _refresh_totals(record)


def finish_invocation(record: dict[str, Any], invocation_id: str) -> None:
    _invocation(record, invocation_id)["end_time"] = now()
    _refresh_totals(record)


def _invocation(record: dict[str, Any], invocation_id: str) -> dict[str, Any]:
    matches = [inv for inv in record["invocations"] if inv["id"] == invocation_id]
    if len(matches) != 1:
        raise RuntimeError(f"run.json has {len(matches)} invocations with id {invocation_id}")
    return matches[0]


def _refresh_totals(record: dict[str, Any]) -> None:
    invocations = record["invocations"]
    ends = [inv["end_time"] for inv in invocations if inv["end_time"]]
    record["start_time"] = min(inv["start_time"] for inv in invocations)
    record["end_time"] = max(ends) if ends else None
    counts = [seq for inv in invocations for seq in inv["sequences"].values()]
    record["frames_predicted"] = sum(c["predicted"] for c in counts)
    record["frames_skipped"] = sum(c["skipped"] for c in counts)


def git_commit(repo: Path) -> dict[str, Any]:
    def git(*args: str) -> str:
        done = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
        if done.returncode != 0:
            raise RuntimeError(f"git {' '.join(args)} failed in {repo}: {done.stderr.strip()}")
        return done.stdout.strip()

    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}


def environment(device: str) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; enable a GPU accelerator")
    return {
        "host": platform.node(),
        "device": device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu": torch.cuda.get_device_name(torch.device(device)),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "commits": {
            "project": git_commit(REPO_ROOT),
            "pointdit": git_commit(FORK_DIR),
            "dinov3": git_commit(Path(os.environ.get("DINOV3_REPO", DINOV3_REPO_DIR))),
        },
    }


def load_runner(device: str, warmup_image: np.ndarray, config: RunConfig) -> Runner:
    from pim.model.runner import PointDiTRunner

    runner = PointDiTRunner(device=device)
    for _ in range(WARMUP_RUNS):
        runner.predict(warmup_image, init=config.init, seed=config.seed)
    return runner


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="run config, e.g. configs/baseline.yaml")
    parser.add_argument("--sequences-config", type=Path, default=Path("configs/sequences.yaml"))
    parser.add_argument("--data-root", type=Path, required=True, help="folder containing the sequence folders")
    parser.add_argument("--sequences", nargs="+", help="default: every sequence in --sequences-config")
    parser.add_argument("--out", type=Path, default=Path("/kaggle/working/predictions"))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    config = load_run_config(args.config)
    if not args.data_root.is_dir():
        raise FileNotFoundError(f"data root {args.data_root} does not exist")
    names = select_sequences(args.sequences_config, args.sequences)
    sequences = [load_sequence(args.data_root / name, args.sequences_config) for name in names]
    windows = {s.name: apply_window(s) for s in sequences}
    summaries = {name: window_summary(frames) for name, frames in windows.items()}

    run_dir = args.out / config.name
    run_dir.mkdir(parents=True, exist_ok=True)
    invocation = {"id": uuid.uuid4().hex[:8], **environment(args.device), "start_time": now(),
                  "end_time": None, "sequences": {}}
    with run_record(run_dir) as record:
        register_invocation(record, config, summaries, invocation)

    pending = sum(not prediction_path(run_dir / name, f).is_file()
                  for name, frames in windows.items() for f in frames)
    print(f"{config.name} on {invocation['gpu']} ({args.device}): {len(names)} sequences, "
          f"{pending} frames to predict, run {invocation['id']} in {run_dir}", flush=True)
    runner = load_runner(args.device, windows[names[0]][0].load_rgb(), config) if pending else None

    for sequence in sequences:
        stats = predict_sequence(runner, config, sequence, windows[sequence.name], run_dir)
        with run_record(run_dir) as record:
            record_sequence(record, invocation["id"], sequence.name, summaries[sequence.name], stats)
        print(f"{sequence.name}: {stats.frames} frames done ({stats.predicted} predicted, "
              f"{stats.skipped} skipped), mean latency {stats.mean_latency_ms:.0f} ms, "
              f"peak memory {stats.peak_mem_gb:.2f} GB", flush=True)

    with run_record(run_dir) as record:
        finish_invocation(record, invocation["id"])


if __name__ == "__main__":
    main()
