from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from pim.consistency.evaluation import PAIR_COLUMNS
from pim.consistency.persistence import PAIR_COLUMNS as PERSISTENCE_COLUMNS


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_script",
                                                  Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


analyze = load_script("analyze_consistency")

SEQUENCES = ("rgbd_dataset_freiburg1_a", "rgbd_dataset_freiburg1_b", analyze.FR3)
FRAMES = 60
STEPS = {"33": 1, "100": 3, "333": 10}


def test_flicker_events_include_the_edges_and_respect_both_thresholds() -> None:
    values = np.full(40, 1.0)
    values[[0, 20, 39]] = 3.0
    values[10] = 1.8
    values[30] = np.nan

    assert np.flatnonzero(analyze.flicker_events(values)).tolist() == [0, 20, 39]


def test_flicker_events_use_the_absolute_floor_for_small_values() -> None:
    values = np.full(20, 0.1)
    values[5] = 0.55
    values[12] = 0.65

    assert np.flatnonzero(analyze.flicker_events(values)).tolist() == [12]


def pair_rows(name: str, rng: np.random.Generator) -> list[dict[str, str]]:
    rows = []
    for gap, step in STEPS.items():
        for i in range(FRAMES - step):
            row = {c: repr(float(rng.uniform(0.2, 1.0))) for c in PAIR_COLUMNS}
            row |= {"sequence": name, "gap_ms": gap, "frame_i": str(i), "frame_j": str(i + step),
                    "t_i": f"{i / 30:.6f}", "t_j": f"{(i + step) / 30:.6f}",
                    "actual_gap_ms": f"{1000 * step / 30:.3f}", "too_few_pixels": "False"}
            if i == 7:
                row["m2_visible_far_median"] = "5.0"
            rows.append(row)
    rows[3] |= {c: "" for c in PAIR_COLUMNS if c.endswith(("mean", "median", "p90"))} | {"too_few_pixels": "True"}
    return rows


def write(path: Path, rows: list[dict[str, str]], columns: tuple[str, ...] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_inputs(root: Path, drop_frame: int | None = None) -> dict[str, Path]:
    rng = np.random.default_rng(0)
    paths = {name: root / name for name in ("consistency", "persistence")}
    paths |= {"covariates": root / "covariates.csv", "per_frame": root / "per_frame.csv"}
    covariates, per_frame = [], []
    for name in SEQUENCES:
        pairs = pair_rows(name, rng)
        write(paths["consistency"] / f"pairs_{name}.csv", pairs, PAIR_COLUMNS)
        persistence = [{c: repr(float(rng.uniform(0, 1))) for c in PERSISTENCE_COLUMNS}
                       | {k: p[k] for k in ("sequence", "gap_ms", "frame_i", "frame_j", "actual_gap_ms",
                                            "too_few_pixels")} for p in pairs]
        write(paths["persistence"] / f"persistence_{name}.csv", persistence, PERSISTENCE_COLUMNS)
        for f in range(FRAMES):
            if f == drop_frame:
                continue
            covariates.append({"sequence": name, "frame_index": str(f),
                               **{c: repr(float(rng.uniform(0, 1))) for c in analyze.FRAME_COVARIATES}})
            per_frame.append({"sequence": name, "frame_index": str(f), "rel_d": repr(float(rng.uniform(2, 6)))})
    write(paths["covariates"], covariates)
    write(paths["per_frame"], per_frame)
    return paths


def run_main(paths: dict[str, Path], root: Path, monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", [
        "analyze_consistency.py", "--consistency", str(paths["consistency"]), "--covariates", str(paths["covariates"]),
        "--per-frame", str(paths["per_frame"]), "--persistence", str(paths["persistence"]),
        "--out", str(root / "out"), "--figures", str(root / "figures"), "--resamples", "100"])
    analyze.main()


def test_end_to_end_on_tiny_synthetic_csvs(tmp_path: Path, monkeypatch) -> None:
    paths = write_inputs(tmp_path)

    run_main(paths, tmp_path, monkeypatch)

    out = tmp_path / "out"
    text = (out / "summary.md").read_text()
    for heading in ("## 1. Headline", "## 2. Persistence", "## 3. Drivers", "## 4. M2 by depth third",
                    "## 5. Region-level flicker"):
        assert heading in text
    assert "Not available" not in text and "pooled_fr1" in text
    with open(out / "drivers.csv", newline="") as f:
        drivers = list(csv.DictReader(f))
    assert {r["sequence"] for r in drivers} == {*SEQUENCES, analyze.POOLED}
    assert {r["block_length"] for r in drivers} == {"8", "15", "30"}
    with open(out / "flicker_events.csv", newline="") as f:
        events = list(csv.DictReader(f))
    model = [r for r in events if r["metric"] == "m2"]
    assert {(r["sequence"], r["third"], r["frame_i"]) for r in model if r["frame_i"] == "7"} == \
        {(name, "far", "7") for name in SEQUENCES}
    for figure in ("headline_ratio", "headline_metrics", "thirds_m2", "persistence_p", "flicker_fr3"):
        assert (tmp_path / "figures" / f"{figure}.png").stat().st_size > 0


def test_pair_covariate_join_fails_on_missing_frames(tmp_path: Path, monkeypatch) -> None:
    paths = write_inputs(tmp_path, drop_frame=12)

    with pytest.raises(ValueError, match=r"covariates.csv lacks 1 frames .* \(e.g. \[12\]\)"):
        run_main(paths, tmp_path, monkeypatch)


def test_persistence_join_fails_on_pair_mismatch(tmp_path: Path, monkeypatch) -> None:
    paths = write_inputs(tmp_path)
    path = paths["persistence"] / f"persistence_{SEQUENCES[0]}.csv"
    lines = path.read_text().splitlines(keepends=True)
    path.write_text("".join(lines[:-1]))

    with pytest.raises(ValueError, match="1 consistency pairs missing"):
        run_main(paths, tmp_path, monkeypatch)


def test_near_any_matches_within_one_position_including_the_edges() -> None:
    model = np.zeros(12, bool)
    model[[0, 3, 6, 9, 11]] = True
    floor = np.zeros(12, bool)
    floor[[1, 6, 7, 10]] = True

    # 0: floor at 1; 3: none within 2..4; 6: same pair; 9: floor at 10; 11: floor at 10.
    assert np.flatnonzero(analyze.near_any(model, floor)).tolist() == [0, 6, 9, 11]
    assert not analyze.near_any(model, np.zeros(12, bool)).any()


def test_floor_events_split_model_events_into_coinciding_and_unexplained() -> None:
    n = 30
    s = {"frame_i": np.arange(n, dtype=float), "frame_j": np.arange(1, n + 1, dtype=float)}
    for third in analyze.THIRDS:
        s[f"m2_visible_{third}_median"] = np.full(n, 1.0)
        s[f"m2_floor_visible_{third}_median"] = np.full(n, 0.3)
    s["m2_visible_far_median"][[5, 15, 25]] = 4.0
    s["m2_floor_visible_far_median"][[6, 25]] = 2.0

    events, counts = analyze.event_tables({("seq", analyze.EVENT_GAP): s, ("seq", 100.0): s})

    far = next(c for c in counts if c["third"] == "far")
    assert (far["events"], far["floor_events"], far["coincident"], far["not_explained_by_floor"]) == (3, 2, 2, 1)
    assert sorted((e["frame_i"], e["floor_event_nearby"]) for e in events if e["metric"] == "m2") == \
        [(5, True), (15, False), (25, True)]
    assert sorted(e["frame_i"] for e in events if e["metric"] == "m2_floor") == [6, 25]
    assert all(c["events"] == 0 for c in counts if c["third"] != "far") and len(counts) == 3
