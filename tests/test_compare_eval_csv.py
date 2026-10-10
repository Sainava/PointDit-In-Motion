from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "compare_eval_csv.py"
OLD_COLUMNS = ("sequence", "frame_index", "rgb_timestamp", "rgb_file", "gap_ms", "skipped", "rel_d")


def old_rows() -> list[dict[str, str]]:
    return [
        {"sequence": "seq", "frame_index": "1", "rgb_timestamp": "1000.000000", "rgb_file": "rgb/1.png",
         "gap_ms": "", "skipped": "False", "rel_d": "5.123456789"},
        {"sequence": "seq", "frame_index": "2", "rgb_timestamp": "1000.033333", "rgb_file": "rgb/2.png",
         "gap_ms": "33.333", "skipped": "True", "rel_d": ""},
        {"sequence": "seq", "frame_index": "3", "rgb_timestamp": "1000.066667", "rgb_file": "rgb/3.png",
         "gap_ms": "33.334", "skipped": "False", "rel_d": "7.5"},
    ]


def write(path: Path, rows: list[dict[str, str]], columns: tuple[str, ...]) -> Path:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return path


def compare(new: Path, old: Path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPT), str(new), str(old)], capture_output=True, text=True)


def extended(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [{**r, "focal_ratio": "1.1"} for r in rows]


def test_identical_shared_columns_pass(tmp_path: Path) -> None:
    old = write(tmp_path / "old.csv", old_rows(), OLD_COLUMNS)
    new = write(tmp_path / "new.csv", extended(old_rows()), (*OLD_COLUMNS, "focal_ratio"))

    result = compare(new, old)

    assert result.returncode == 0, result.stdout
    assert "only in new: ['focal_ratio']" in result.stdout


@pytest.mark.parametrize("column, value", [("rel_d", "5.12346"), ("rel_d", ""), ("skipped", "False"),
                                           ("rgb_file", "rgb/x.png")])
def test_perturbed_cell_fails(tmp_path: Path, column: str, value: str) -> None:
    old = write(tmp_path / "old.csv", old_rows(), OLD_COLUMNS)
    rows = old_rows()
    rows[0 if column == "rel_d" else 1][column] = value
    new = write(tmp_path / "new.csv", extended(rows), (*OLD_COLUMNS, "focal_ratio"))

    result = compare(new, old)

    assert result.returncode == 1
    assert "FAIL" in result.stdout


def test_difference_within_tolerance_passes(tmp_path: Path) -> None:
    old = write(tmp_path / "old.csv", old_rows(), OLD_COLUMNS)
    rows = old_rows()
    rows[0]["rel_d"] = "5.1234572"
    new = write(tmp_path / "new.csv", rows, OLD_COLUMNS)

    assert compare(new, old).returncode == 0


def test_missing_row_fails(tmp_path: Path) -> None:
    old = write(tmp_path / "old.csv", old_rows(), OLD_COLUMNS)
    new = write(tmp_path / "new.csv", old_rows()[:2], OLD_COLUMNS)

    result = compare(new, old)

    assert result.returncode == 1
    assert "keys differ" in result.stdout
