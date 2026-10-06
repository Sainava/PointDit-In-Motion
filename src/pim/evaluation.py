"""Per-frame evaluation of cached predictions against TUM ground truth."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pim.data.tum import Frame
from pim.metrics.geometry import FrameMetrics, backproject, evaluate_frame

COLUMNS = (
    "sequence", "frame_index", "rgb_timestamp", "rgb_file", "gap_ms", "depth_offset_ms",
    "gt_valid_fraction", "num_valid", "skipped", "rel_p", "delta1_p", "rel_d", "delta1_d",
    "latency_ms", "peak_mem_gb",
)
METRICS = ("rel_p", "delta1_p", "rel_d", "delta1_d")


@dataclass(frozen=True)
class CachedPrediction:
    points: np.ndarray
    rgb_file: str
    latency_ms: float
    peak_mem_gb: float


@dataclass(frozen=True)
class FrameTask:
    sequence: str
    sequence_root: Path
    K: np.ndarray
    frame: Frame
    prediction: Path
    gap_ms: float | None
    """Milliseconds since the previous frame of the window; None for the window's first frame."""


def prediction_path(run_dir: Path, sequence: str, frame_index: int) -> Path:
    return run_dir / sequence / f"{frame_index:05d}.npz"


def expected_rgb_file(frame: Frame, sequence_root: Path) -> str:
    return frame.rgb_path.relative_to(sequence_root).as_posix()


def check_rgb_file(path: Path, frame: Frame, sequence_root: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"prediction {path} not found")
    with np.load(path) as data:
        rgb_file = str(data["rgb_file"])
    _require_rgb_file(path, rgb_file, frame, sequence_root)


def load_prediction(path: Path, frame: Frame, sequence_root: Path) -> CachedPrediction:
    if not path.is_file():
        raise FileNotFoundError(f"prediction {path} not found")
    with np.load(path) as data:
        prediction = CachedPrediction(
            points=data["points"].astype(np.float32),
            rgb_file=str(data["rgb_file"]),
            latency_ms=float(data["latency_ms"]),
            peak_mem_gb=float(data["peak_mem_gb"]),
        )
    _require_rgb_file(path, prediction.rgb_file, frame, sequence_root)
    return prediction


def _require_rgb_file(path: Path, rgb_file: str, frame: Frame, sequence_root: Path) -> None:
    expected = expected_rgb_file(frame, sequence_root)
    if rgb_file != expected:
        raise ValueError(f"{path} was predicted from {rgb_file}, but frame {frame.index} is {expected}")


def build_tasks(
    sequence: str, sequence_root: Path, K: np.ndarray, frames: list[Frame], run_dir: Path
) -> list[FrameTask]:
    tasks = []
    for i, frame in enumerate(frames):
        gap_ms = None if i == 0 else (frame.timestamp - frames[i - 1].timestamp) * 1000.0
        tasks.append(FrameTask(sequence, sequence_root, K, frame,
                               prediction_path(run_dir, sequence, frame.index), gap_ms))
    return tasks


def set_torch_threads(threads: int) -> None:
    import torch

    torch.set_num_threads(threads)


def evaluate_task(task: FrameTask) -> dict[str, Any]:
    prediction = load_prediction(task.prediction, task.frame, task.sequence_root)
    depth, depth_valid = task.frame.load_depth()
    gt_points, gt_valid = backproject(depth, depth_valid, task.K)
    metrics = evaluate_frame(prediction.points, gt_points, gt_valid, pred_valid=None)
    return frame_row(task, prediction, float(gt_valid.mean()), metrics)


def frame_row(
    task: FrameTask, prediction: CachedPrediction, gt_valid_fraction: float, metrics: FrameMetrics
) -> dict[str, Any]:
    return {
        "sequence": task.sequence,
        "frame_index": task.frame.index,
        "rgb_timestamp": f"{task.frame.timestamp:.6f}",
        "rgb_file": prediction.rgb_file,
        "gap_ms": "" if task.gap_ms is None else f"{task.gap_ms:.3f}",
        "depth_offset_ms": f"{abs(task.frame.depth_offset) * 1000.0:.3f}",
        "gt_valid_fraction": f"{gt_valid_fraction:.6f}",
        "num_valid": metrics.num_valid,
        "skipped": metrics.skipped,
        **{name: _metric_cell(getattr(metrics, name)) for name in METRICS},
        "latency_ms": f"{prediction.latency_ms:.3f}",
        "peak_mem_gb": f"{prediction.peak_mem_gb:.4f}",
    }


def _metric_cell(value: float) -> str:
    return "" if math.isnan(value) else repr(float(value))
