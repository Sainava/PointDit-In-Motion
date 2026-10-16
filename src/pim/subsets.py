"""Contiguous frame subsets (inclusive kept-frame index ranges) that restrict prediction and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import yaml

from pim.data.tum import Frame


@dataclass(frozen=True)
class FrameRange:
    first: int
    last: int

    def contains(self, index: int) -> bool:
        return self.first <= index <= self.last

    def as_dict(self) -> dict[str, int]:
        return {"first": self.first, "last": self.last}


Subsets = dict[str, FrameRange]


def load_subsets(path: Path) -> Subsets:
    if not path.is_file():
        raise FileNotFoundError(f"subsets file {path} not found")
    raw = yaml.safe_load(path.read_text())
    entries = raw.get("subsets") if isinstance(raw, dict) else None
    if not isinstance(entries, dict) or not entries:
        raise ValueError(f"{path}: expected a non-empty `subsets` mapping of sequence -> {{first, last}}")
    return {name: _frame_range(entry, f"{path}: {name}") for name, entry in entries.items()}


def _frame_range(entry: Any, where: str) -> FrameRange:
    if not isinstance(entry, dict) or set(entry) != {"first", "last"}:
        raise ValueError(f"{where}: expected {{first, last}}, got {entry!r}")
    first, last = entry["first"], entry["last"]
    for value in (first, last):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{where}: first and last must be non-negative integers, got {entry!r}")
    if last < first:
        raise ValueError(f"{where}: last {last} is before first {first}")
    return FrameRange(first, last)


def subsets_record(subsets: Subsets | None) -> dict[str, dict[str, int]] | None:
    return None if subsets is None else {name: r.as_dict() for name, r in subsets.items()}


def subset_sequences(subsets: Subsets, requested: list[str] | None) -> list[str]:
    """--sequences restricted to the subsets' sequences; all of them when none are requested."""
    if requested is None:
        return list(subsets)
    outside = [name for name in requested if name not in subsets]
    if outside:
        raise ValueError(f"sequences {outside} have no subset; subsets cover {list(subsets)}")
    return requested


def check_inside_window(name: str, frames: Sequence[Frame], frame_range: FrameRange) -> None:
    first, last = frames[0].index, frames[-1].index
    if frame_range.first < first or frame_range.last > last:
        raise ValueError(f"{name}: subset {frame_range.first}-{frame_range.last} is not inside the window "
                         f"{first}-{last}")


def restrict_frames(name: str, frames: Sequence[Frame], frame_range: FrameRange) -> list[Frame]:
    """The window frames inside frame_range, which must lie entirely within the window."""
    check_inside_window(name, frames, frame_range)
    return [f for f in frames if frame_range.contains(f.index)]


def check_run_covers(run: dict[str, Any], subsets: Subsets | None) -> None:
    """A run predicted on subsets only can be evaluated only on frames inside them."""
    predicted = run.get("subsets")
    if predicted is None:
        return
    if subsets is None:
        raise ValueError(f"the run predicted only the subsets {predicted}; pass the same --subsets")
    for name, frame_range in subsets.items():
        held = predicted.get(name)
        if held is None or frame_range.first < held["first"] or frame_range.last > held["last"]:
            raise ValueError(f"{name}: subset {frame_range.as_dict()} is not inside the predicted subset {held}")
