"""Run the compute sweep unattended, each config in its own predict.py process on GPU 0, one after another.

A failing config is recorded in its sweep_status.json and the next config runs. After the loop it prints a status
table and writes sweep_latency.csv; only then does it check sweep_baseline_check against the cached baseline.
A determinism mismatch prints a banner and exits 3; any other error exits non-zero as usual.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pim.paths import REPO_ROOT  # noqa: E402
from pim.subsets import load_subsets  # noqa: E402
from pim.sweep import (  # noqa: E402
    STATUS_FILE, TIMINGS_FILE, nvidia_smi, status_record, summary_rows, write_json, write_summary,
)

DETERMINISM_MISMATCH = 3
CHECK_RUN = "sweep_baseline_check"
TAIL_LINES = 20
BANNER = "!" * 78


@dataclass(frozen=True)
class Outcome:
    returncode: int | None
    tail: list[str]
    timed_out: bool = False


Launch = Callable[[list[str], dict[str, str], str, float | None], Outcome]


def launch(command: list[str], env: dict[str, str], prefix: str, timeout: float | None) -> Outcome:
    """Run a command, streaming its output with a prefix; killed after timeout seconds if one is given."""
    proc = subprocess.Popen(command, env=env, cwd=REPO_ROOT, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    timed_out = threading.Event()

    def kill() -> None:
        timed_out.set()
        proc.kill()

    timer = threading.Timer(timeout, kill) if timeout else None
    tail: deque[str] = deque(maxlen=TAIL_LINES)
    if timer:
        timer.start()
    try:
        for line in proc.stdout:
            print(f"{prefix}{line}", end="", flush=True)
            tail.append(line.rstrip("\n"))
        code = proc.wait()
    finally:
        if timer:
            timer.cancel()
    return Outcome(code, list(tail), timed_out.is_set())


def gpu_env(gpu: int) -> dict[str, str]:
    return {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONUNBUFFERED": "1"}


def predict_command(config: Path, args: argparse.Namespace, timings: Path) -> list[str]:
    return [sys.executable, str(REPO_ROOT / "scripts" / "predict.py"), "--config", str(config),
            "--sequences-config", str(args.sequences_config), "--data-root", str(args.data_root),
            "--out", str(args.out), "--device", "cuda:0", "--subsets", str(args.subsets), "--timings", str(timings)]


def last_error(tail: list[str]) -> str | None:
    lines = [line for line in tail if line.strip()]
    return lines[-1] if lines else None


def run_config(
    config: Path, out: Path, command: Callable[[Path, Path], list[str]], expected: dict[str, int],
    run: Launch = launch, smi: Callable[[], dict | None] = nvidia_smi, timeout: float | None = None, gpu: int = 0,
) -> dict[str, Any]:
    """Run one config and write its sweep_status.json; never raises."""
    run_name, run_dir, gpu_before, gpu_after, returncode, tail = None, None, None, None, None, []
    start = time.monotonic()
    try:
        run_name = yaml.safe_load(config.read_text())["name"]
        run_dir = out / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        timings = run_dir / TIMINGS_FILE
        timings.unlink(missing_ok=True)
        gpu_before = smi()
        outcome = run(command(config, timings), gpu_env(gpu), f"[{config.stem}] ", timeout)
        gpu_after = smi()
        returncode, tail = outcome.returncode, outcome.tail
        status = "timeout" if outcome.timed_out else "ok" if returncode == 0 else "failed"
        error = None if status == "ok" else last_error(tail) or f"exit code {returncode}"
    except Exception as exc:
        status, error = "failed", f"{type(exc).__name__}: {exc}"
    seconds = time.monotonic() - start
    try:
        record = status_record(config.stem, run_name, status, returncode, seconds, run_dir, expected,
                               gpu_before, gpu_after, error)
    except Exception as exc:
        record = {"config": config.stem, "run_name": run_name, "status": "failed", "returncode": returncode,
                  "seconds": round(seconds, 1), "gpu_before": gpu_before, "gpu_after": gpu_after,
                  "error": f"{error}; status record failed with {type(exc).__name__}: {exc}"}
    record["output_tail"] = tail if record["status"] != "ok" else []
    try:
        folder = run_dir or out / config.stem
        folder.mkdir(parents=True, exist_ok=True)
        write_json(folder / STATUS_FILE, record)
    except Exception as exc:
        print(f"[{config.stem}] could not write {STATUS_FILE}: {exc}", flush=True)
    return record


def run_configs(
    configs: list[Path], out: Path, command: Callable[[Path, Path], list[str]], expected: dict[str, int],
    run: Launch = launch, smi: Callable[[], dict | None] = nvidia_smi, timeout: float | None = None,
) -> list[dict[str, Any]]:
    records = []
    for k, config in enumerate(configs, start=1):
        print(f"\n=== {k}/{len(configs)} {config.stem} ===", flush=True)
        record = run_config(config, out, command, expected, run, smi, timeout)
        print(f"=== {config.stem}: {record['status']}"
              + (f" ({record['error']})" if record.get("error") else "") + " ===", flush=True)
        records.append(record)
    return records


def _fmt(value: Any, digits: int = 0) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def status_table(records: list[dict[str, Any]]) -> str:
    header = (f"{'config':<16} {'status':<10} {'frames':>6} {'model ms':>9} {'median':>7} {'e2e ms':>7} "
              f"{'peak GB':>7} {'MHz':>11} {'temp C':>9} {'seconds':>7}  error")
    lines = [header, "-" * len(header)]
    for r in records:
        before, after = r.get("gpu_before") or {}, r.get("gpu_after") or {}
        clocks = f"{_fmt(before.get('sm_clock_mhz'))}>{_fmt(after.get('sm_clock_mhz'))}"
        temps = f"{_fmt(before.get('temperature_c'))}>{_fmt(after.get('temperature_c'))}"
        lines.append(f"{r['config']:<16} {r['status']:<10} {_fmt(r.get('frames_predicted')):>6} "
                     f"{_fmt(r.get('mean_model_latency_ms')):>9} {_fmt(r.get('median_model_latency_ms')):>7} "
                     f"{_fmt(r.get('mean_latency_ms')):>7} {_fmt(r.get('peak_mem_gb'), 2):>7} {clocks:>11} "
                     f"{temps:>9} {_fmt(r.get('seconds')):>7}  {r.get('error') or ''}")
    return "\n".join(lines)


def determinism_check(
    records: list[dict[str, Any]], out: Path, reference: Path | None, subsets: Path, csv_out: Path,
    run: Launch = launch,
) -> int:
    check = next((r for r in records if r.get("run_name") == CHECK_RUN), None)
    if check is None or reference is None:
        print(f"\n{CHECK_RUN} not in this sweep or no --reference given; skipping the determinism check")
        return 0
    if check["status"] != "ok":
        print(f"\n{CHECK_RUN} did not finish ({check['status']}); the determinism check cannot run")
        return 1
    print(f"\n{CHECK_RUN} vs {reference}", flush=True)
    command = [sys.executable, str(REPO_ROOT / "scripts" / "check_determinism.py"), "--predictions",
               str(out / CHECK_RUN), "--reference", str(reference), "--subsets", str(subsets), "--out", str(csv_out),
               "--mismatch-exit-code", str(DETERMINISM_MISMATCH)]
    code = run(command, {**os.environ, "PYTHONUNBUFFERED": "1"}, "", None).returncode
    if code == DETERMINISM_MISMATCH:
        print(f"\n{BANNER}\n  DETERMINISM CHECK FAILED  {CHECK_RUN} does not reproduce the cached baseline.\n"
              f"  All sweep outputs are written. Latencies are kept, accuracy comparisons need care.\n"
              f"  Details in {csv_out}\n{BANNER}", flush=True)
    return code if code is not None else 1


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", type=Path, nargs="+", required=True, help="run configs, in order")
    parser.add_argument("--subsets", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--sequences-config", type=Path, default=REPO_ROOT / "configs" / "sequences.yaml")
    parser.add_argument("--out", type=Path, required=True, help="sweep folder; each run goes to <out>/<run name>")
    parser.add_argument("--reference", type=Path, help="cached baseline run folder for the determinism check")
    parser.add_argument("--summary-out", type=Path, required=True, help="sweep_latency.csv")
    parser.add_argument("--determinism-out", type=Path, required=True, help="determinism.csv")
    parser.add_argument("--timeout-minutes", type=float, default=90.0, help="per config; 0 for none")
    args = parser.parse_args(argv)

    expected = {name: r.last - r.first + 1 for name, r in load_subsets(args.subsets).items()}
    args.out.mkdir(parents=True, exist_ok=True)
    records = run_configs(args.configs, args.out, lambda config, timings: predict_command(config, args, timings),
                          expected, timeout=args.timeout_minutes * 60 or None)

    write_summary(summary_rows(args.out), args.summary_out)
    print(f"\n{status_table(records)}\n\nwrote {args.summary_out}", flush=True)
    args.determinism_out.parent.mkdir(parents=True, exist_ok=True)
    sys.exit(determinism_check(records, args.out, args.reference, args.subsets, args.determinism_out))


if __name__ == "__main__":
    main()
