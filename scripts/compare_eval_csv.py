"""Check that a new per_frame.csv reproduces an older one: same rows and keys, and every shared column within tolerance."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

KEYS = ("sequence", "frame_index", "rgb_timestamp")
DEFAULT_TOLERANCE = 1e-6


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found")
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        columns = list(reader.fieldnames or [])
    missing = [key for key in KEYS if key not in columns]
    if missing:
        raise ValueError(f"{path} has no key columns {missing}")
    return columns, rows


def as_number(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def column_difference(new: list[str], old: list[str]) -> float:
    """Max absolute difference; inf when the cells are not both numbers (or both empty) or differ as text."""
    worst = 0.0
    for a, b in zip(new, old):
        if a == b:
            continue
        x, y = as_number(a), as_number(b)
        if x is None or y is None or math.isnan(x) != math.isnan(y):
            return math.inf
        if not math.isnan(x):
            worst = max(worst, abs(x - y))
    return worst


def compare(new_path: Path, old_path: Path, tolerance: float) -> bool:
    new_columns, new_rows = read_rows(new_path)
    old_columns, old_rows = read_rows(old_path)
    new_keys = [tuple(r[k] for k in KEYS) for r in new_rows]
    old_keys = [tuple(r[k] for k in KEYS) for r in old_rows]
    if new_keys != old_keys:
        only_new, only_old = set(new_keys) - set(old_keys), set(old_keys) - set(new_keys)
        print(f"FAIL keys differ: {len(new_keys)} new rows vs {len(old_keys)} old rows; "
              f"{len(only_new)} only in new (e.g. {sorted(only_new)[:3]}), "
              f"{len(only_old)} only in old (e.g. {sorted(only_old)[:3]})"
              + ("; same keys in a different order" if not only_new and not only_old else ""))
        return False

    shared = [c for c in old_columns if c in new_columns and c not in KEYS]
    print(f"{len(new_rows)} rows with identical keys; {len(shared)} shared columns")
    print(f"  only in new: {[c for c in new_columns if c not in old_columns]}")
    print(f"  only in old: {[c for c in old_columns if c not in new_columns]}")
    ok = True
    for column in shared:
        difference = column_difference([r[column] for r in new_rows], [r[column] for r in old_rows])
        passed = difference <= tolerance
        ok &= passed
        shown = "text or empty-cell mismatch" if math.isinf(difference) else f"{difference:.3g}"
        print(f"  {'ok  ' if passed else 'FAIL'} {column:<20} max |diff| {shown}")
    print("PASS" if ok else f"FAIL: some shared column differs by more than {tolerance:g}")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("new", type=Path, help="newly written per_frame.csv")
    parser.add_argument("old", type=Path, help="per_frame.csv to reproduce")
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    args = parser.parse_args()
    sys.exit(0 if compare(args.new, args.old, args.tolerance) else 1)


if __name__ == "__main__":
    main()
