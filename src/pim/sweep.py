"""Compute sweep bookkeeping with nvidia-smi readings, per-config sweep_status.json and the latency summary.

Standard library only, so the summary runs anywhere without a GPU or numpy.
"""

from __future__ import annotations

import csv
import json
import os
import statistics
import subprocess
from pathlib import Path
from typing import Any, Callable, Sequence

STATUS_FILE = "sweep_status.json"
TIMINGS_FILE = "sweep_timings.csv"
# nvidia-smi --query-gpu fields and the keys they are stored under.
GPU_FIELDS = {"clocks.sm": "sm_clock_mhz", "clocks.mem": "mem_clock_mhz", "temperature.gpu": "temperature_c",
              "power.draw": "power_w"}
SUMMARY_COLUMNS = ("config", "status", "frames", "mean_model_latency_ms", "median_model_latency_ms",
                   "end_to_end_latency_ms", "peak_mem_gb", "gpu_clock_start", "gpu_clock_end")


def parse_nvidia_smi(text: str) -> dict[str, float | str] | None:
    """The first GPU's line of `--format=csv,noheader,nounits` output; None if it does not have every field."""
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return None
    values = [v.strip() for v in lines[0].split(",")]
    if len(values) != len(GPU_FIELDS):
        return None
    reading: dict[str, float | str] = {}
    for key, value in zip(GPU_FIELDS.values(), values):
        try:
            reading[key] = float(value)
        except ValueError:
            reading[key] = value
    return reading


def nvidia_smi(gpu: int = 0, run: Callable[..., Any] = subprocess.run) -> dict[str, float | str] | None:
    """Clocks, temperature and power of one GPU; None when nvidia-smi is missing or fails."""
    command = ["nvidia-smi", f"--query-gpu={','.join(GPU_FIELDS)}", "--format=csv,noheader,nounits", "-i", str(gpu)]
    try:
        done = run(command, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    if done.returncode != 0:
        return None
    return parse_nvidia_smi(done.stdout)


def read_timings(path: Path) -> list[dict[str, float]]:
    if not path.is_file():
        return []
    with open(path, newline="") as f:
        return [{"model_latency_ms": float(r["model_latency_ms"]), "latency_ms": float(r["latency_ms"])}
                for r in csv.DictReader(f)]


def mean_median(values: Sequence[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    return statistics.fmean(values), statistics.median(values)


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def run_peak_mem_gb(run: dict[str, Any] | None) -> float | None:
    peaks = [s["peak_mem_gb"] for s in ((run or {}).get("sequences") or {}).values() if "peak_mem_gb" in s]
    return max(peaks) if peaks else None


def run_mean_latency_ms(run: dict[str, Any] | None) -> float | None:
    """Frame-weighted mean of the per-sequence mean latency_ms (end to end) in run.json."""
    entries = [s for s in ((run or {}).get("sequences") or {}).values() if s.get("frames")]
    frames = sum(s["frames"] for s in entries)
    return sum(s["mean_latency_ms"] * s["frames"] for s in entries) / frames if frames else None


def status_record(
    config: str, run_name: str | None, status: str, returncode: int | None, seconds: float,
    run_dir: Path | None, expected: dict[str, int], gpu_before: dict | None, gpu_after: dict | None,
    error: str | None = None,
) -> dict[str, Any]:
    run = read_json(run_dir / "run.json") if run_dir else None
    timings = read_timings(run_dir / TIMINGS_FILE) if run_dir else []
    model_mean, model_median = mean_median([t["model_latency_ms"] for t in timings])
    latency_mean, latency_median = mean_median([t["latency_ms"] for t in timings])
    on_disk = {name: len(list((run_dir / name).glob("*.npz"))) if run_dir and (run_dir / name).is_dir() else 0
               for name in expected}
    if status == "ok" and on_disk != expected:
        status = "incomplete"
    return {
        "config": config,
        "run_name": run_name,
        "status": status,
        "returncode": returncode,
        "seconds": round(seconds, 1),
        "frames_predicted": len(timings),
        "frames_expected": sum(expected.values()),
        "frames_on_disk": sum(on_disk.values()),
        "mean_model_latency_ms": model_mean,
        "median_model_latency_ms": model_median,
        "mean_latency_ms": latency_mean,
        "median_latency_ms": latency_median,
        "peak_mem_gb": run_peak_mem_gb(run),
        "options": (run or {}).get("options"),
        "gpu_before": gpu_before,
        "gpu_after": gpu_after,
        "error": error,
    }


def write_json(path: Path, record: dict[str, Any]) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(record, indent=2) + "\n")
    os.replace(tmp, path)


def _clock(reading: dict | None) -> float | str | None:
    return (reading or {}).get("sm_clock_mhz")


def summary_rows(sweep_dir: Path) -> list[dict[str, Any]]:
    """One row per run folder holding a sweep_status.json or a run.json."""
    run_dirs = sorted({p.parent for name in (STATUS_FILE, "run.json") for p in sweep_dir.glob(f"*/{name}")})
    rows = []
    for run_dir in run_dirs:
        status = read_json(run_dir / STATUS_FILE) or {}
        run = read_json(run_dir / "run.json")
        end_to_end = status.get("mean_latency_ms")
        rows.append({
            "config": status.get("run_name") or (run or {}).get("name") or run_dir.name,
            "status": status.get("status", "unknown"),
            "frames": status.get("frames_predicted", (run or {}).get("frames_predicted")),
            "mean_model_latency_ms": status.get("mean_model_latency_ms"),
            "median_model_latency_ms": status.get("median_model_latency_ms"),
            "end_to_end_latency_ms": end_to_end if end_to_end is not None else run_mean_latency_ms(run),
            "peak_mem_gb": status.get("peak_mem_gb", run_peak_mem_gb(run)),
            "gpu_clock_start": _clock(status.get("gpu_before")),
            "gpu_clock_end": _clock(status.get("gpu_after")),
        })
    return rows


def write_summary(rows: Sequence[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows({k: ("" if v is None else v) for k, v in row.items()} for row in rows)
