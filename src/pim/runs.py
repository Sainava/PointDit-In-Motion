"""Prediction-run records, sequence selection and provenance shared by the evaluation scripts."""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from pim.data.tum import Frame

IGNORED_INPUT_PARTS = ("/Pointdit-In-Motion/",)
"""Paths containing any of these are never discovered, so a project checkout bundled into an attached output cannot
stand in for a run's own files."""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def git_commit(repo: Path) -> dict[str, Any]:
    def git(*args: str) -> str:
        done = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
        if done.returncode != 0:
            raise RuntimeError(f"git {' '.join(args)} failed in {repo}: {done.stderr.strip()}")
        return done.stdout.strip()

    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}


# Compute options every run used before predict.py recorded them. A run.json without an `options`
# entry was predicted with exactly these, so it reads back as them.
PRE_SWEEP_OPTIONS: dict[str, Any] = {"steps": 3, "dtype": "bf16", "target_tokens": 1024, "model": "L",
                                     "input_downscale": 1}


def run_options(run: dict[str, Any]) -> dict[str, Any]:
    return dict(run.get("options") or PRE_SWEEP_OPTIONS)


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


def check_window(name: str, run: dict[str, Any], frames: list[Frame], sequences_config: Path) -> None:
    window = {"start": frames[0].index, "count": len(frames)}
    if run["sequences"][name]["window"] != window:
        raise ValueError(f"{name}: predictions cover window {run['sequences'][name]['window']}, "
                         f"but {sequences_config} now gives {window}")


def available_cpus() -> int:
    return len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1


def find_input_files(root: Path, name: str, ignored: tuple[str, ...] = IGNORED_INPUT_PARTS) -> list[Path]:
    """Every file called name under root, following symlinked folders and skipping any path that contains one of
    ignored."""
    found = []
    for folder, dirs, files in os.walk(root, followlinks=True):
        dirs[:] = [d for d in dirs if not any(part in f"{folder}/{d}/" for part in ignored)]
        path = Path(folder) / name
        if name in files and not any(part in str(path) for part in ignored):
            found.append(path)
    return sorted(found)
