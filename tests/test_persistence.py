from __future__ import annotations

import csv
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from pim.consistency.metrics import FrameInputs, pair_consistency
from pim.consistency.persistence import (
    PAIR_COLUMNS, REGIONS, m1_regression, pair_errors, persistence_stats, read_rows, region_stats,
)

TESTS = Path(__file__).resolve().parent


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


persistence_eval = load(TESTS.parent / "scripts" / "persistence_eval.py", "persistence_eval_script")
fixtures = load(TESTS / "test_consistency_eval.py", "consistency_eval_fixtures")
consistency_eval = fixtures.consistency_eval

H, W = 40, 52
K = np.array([[45.0, 0.0, 25.5], [0.0, 45.0, 19.5], [0.0, 0.0, 1.0]])


def test_identical_errors_are_fully_persistent() -> None:
    e = np.random.default_rng(0).normal(0.0, 0.05, 5000)

    stats = persistence_stats(e, e.copy())

    assert stats["persistent_fraction"] == pytest.approx(1.0)
    assert stats["pearson"] == pytest.approx(1.0)
    assert stats["spearman"] == pytest.approx(1.0)
    assert stats["median_abs_diff"] == 0.0
    assert stats["median_abs_e_i"] == stats["median_abs_e_j"] == pytest.approx(100 * np.median(np.abs(e)))


def test_independent_errors_are_not_persistent() -> None:
    rng = np.random.default_rng(1)

    stats = persistence_stats(rng.normal(0.0, 0.05, 50_000), rng.normal(0.0, 0.03, 50_000))

    assert abs(stats["persistent_fraction"]) < 0.02
    assert abs(stats["pearson"]) < 0.02 and abs(stats["spearman"]) < 0.02


def test_partly_shared_errors_give_the_shared_variance_fraction() -> None:
    rng = np.random.default_rng(2)
    shared = rng.normal(0.0, 0.04, 200_000)

    stats = persistence_stats(shared + rng.normal(0.0, 0.02, shared.size), shared + rng.normal(0.0, 0.02, shared.size))

    # P = 2 Var(shared) / (Var(E_i) + Var(E_j)) = 2 * 16 / (2 * 20).
    assert stats["persistent_fraction"] == pytest.approx(0.8, abs=0.01)


def test_too_few_pixels_give_nan() -> None:
    stats = persistence_stats(np.zeros(999), np.zeros(999))

    assert stats["n"] == 999 and all(math.isnan(v) for k, v in stats.items() if k != "n")


def test_unequal_inputs_are_rejected() -> None:
    with pytest.raises(ValueError, match="equal length"):
        persistence_stats(np.zeros(10), np.zeros(11))


def synthetic_pair() -> tuple[FrameInputs, FrameInputs, np.ndarray]:
    v, u = np.indices((H, W))
    gt = (2.0 + 0.01 * v + 0.005 * u).astype(np.float32)
    gt[12:22, 18:30] = 1.4
    rng = np.random.default_rng(3)
    frames = []
    for k in range(2):
        aligned = (gt * (1 + 0.03 * np.sin(u / 5.0 + k) + 0.01 * rng.standard_normal(gt.shape))).astype(np.float32)
        frames.append(FrameInputs(gt, gt > 0, (0.5 * aligned + 0.2).astype(np.float32), aligned))
    T = np.eye(4)
    T[0, 3] = 0.02
    return frames[0], frames[1], T


def test_median_abs_difference_equals_m1_on_a_synthetic_pair() -> None:
    frame_i, frame_j, T = synthetic_pair()

    stats = region_stats(pair_errors(frame_i, frame_j, K, T, T))
    reference = pair_consistency(frame_i, frame_j, K, T, T)

    assert stats["visible"]["median_abs_diff"] == pytest.approx(reference.visible.m1.overall.median, abs=1e-9)
    assert stats["interior"]["median_abs_diff"] == pytest.approx(reference.interior.m1.overall.median, abs=1e-9)
    assert sum(stats[r]["n"] for r in REGIONS[2:]) == stats["visible"]["n"]
    assert 0.0 < stats["visible"]["persistent_fraction"] < 1.0


def regression_rows(reference: bool, shift: float = 0.0) -> list[dict[str, str]]:
    rows = []
    for k, value in enumerate((0.5, 0.7, "")):
        cells = {"gap_ms": "33", "frame_i": str(k), "frame_j": str(k + 1), "too_few_pixels": str(value == "")}
        if reference:
            cells |= {"m1_visible_median": str(value), "m1_interior_median": str(value)}
        else:
            moved = "" if value == "" else repr(value + shift)
            cells |= {"visible_median_abs_diff": moved, "interior_median_abs_diff": str(value)}
        rows.append(cells)
    return rows


def test_regression_passes_on_matching_values() -> None:
    largest, failures = m1_regression(regression_rows(False, 5e-7), regression_rows(True))

    assert failures == [] and largest == pytest.approx(5e-7)


def test_regression_fails_on_a_perturbed_value() -> None:
    _, failures = m1_regression(regression_rows(False, 2e-6), regression_rows(True))

    assert len(failures) == 2 and all("visible_median_abs_diff" in f for f in failures)


def test_regression_fails_on_a_key_mismatch() -> None:
    rows = regression_rows(False)
    rows[1]["frame_j"] = "5"

    _, failures = m1_regression(rows, regression_rows(True))

    assert any("pair keys differ: 1 only in the persistence rows" in f and "1 only in the reference" in f
               for f in failures)


def test_regression_fails_when_only_one_side_is_nan() -> None:
    rows = regression_rows(False)
    rows[0]["visible_median_abs_diff"] = ""

    _, failures = m1_regression(rows, regression_rows(True))

    assert len(failures) == 1 and "(33.0, 0, 1) visible_median_abs_diff: nan" in failures[0]


@pytest.fixture(scope="module")
def runs(tmp_path_factory) -> tuple[Path, Path]:
    root = tmp_path_factory.mktemp("persistence")
    run_dir, data_root, config = write_inputs(root)
    common = ["--predictions", str(run_dir), "--data-root", str(data_root), "--sequences-config", str(config),
              "--gaps-ms", *fixtures.GAPS]
    reference = root / "consistency"
    assert consistency_eval.run_consistency(consistency_eval.parse_args([*common, "--out", str(reference)]))
    out = root / "persistence"
    args = persistence_eval.parse_args([*common, "--out", str(out), "--reference", str(reference)])
    assert persistence_eval.run_persistence(args)
    return out, reference


def write_inputs(root: Path) -> tuple[Path, Path, Path]:
    data_root, run_dir = root / "data", root / "predictions" / "test"
    config = root / "sequences.yaml"
    count = len(fixtures.SLOTS)
    entries = "".join(f"  {n}:\n    camera: cam\n    window: {{start: 0, count: {count}}}\n" for n in fixtures.SEQUENCES)
    config.write_text(f"cameras:\n  cam: {{fx: {fixtures.FX}, fy: {fixtures.FY}, cx: {fixtures.CX}, "
                      f"cy: {fixtures.CY}}}\nsequences:\n{entries}")
    for name in fixtures.SEQUENCES:
        fixtures.write_sequence(data_root, name)
        fixtures.write_predictions(run_dir, name)
    window = {"start": 0, "count": count}
    run = {"name": "test", "sequences": {n: {"window": window, "frames": count} for n in fixtures.SEQUENCES}}
    (run_dir / "run.json").write_text(json.dumps(run))
    return run_dir, data_root, config


def test_end_to_end_matches_the_consistency_run(runs: tuple[Path, Path]) -> None:
    out, reference = runs
    for name in fixtures.SEQUENCES:
        rows = read_rows(out / f"persistence_{name}.csv")
        reference_rows = consistency_eval.read_pairs(reference / f"pairs_{name}.csv")
        assert [(r["gap_ms"], r["frame_i"], r["frame_j"]) for r in rows] == \
            [(r["gap_ms"], r["frame_i"], r["frame_j"]) for r in reference_rows]
        too_few = [r for r in rows if r["too_few_pixels"] == "True"]
        assert too_few and all(r["visible_persistent_fraction"] == "" for r in too_few)
    with open(out / "persistence_summary.csv", newline="") as f:
        summary = list(csv.DictReader(f))
    assert len(summary) == len(fixtures.SEQUENCES) * len(fixtures.GAPS) * len(REGIONS)
    assert set(read_rows(out / "persistence_seq_a.csv")[0]) == set(PAIR_COLUMNS)


def test_end_to_end_regression_fails_against_a_perturbed_reference(runs: tuple[Path, Path], tmp_path: Path) -> None:
    out, reference = runs
    for name in fixtures.SEQUENCES:
        rows = consistency_eval.read_pairs(reference / f"pairs_{name}.csv")
        scored = next(r for r in rows if r["m1_visible_median"])
        if name == "seq_a":
            scored["m1_visible_median"] = repr(float(scored["m1_visible_median"]) + 1e-4)
        with open(tmp_path / f"pairs_{name}.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    assert not persistence_eval.report_regression(
        {name: read_rows(out / f"persistence_{name}.csv") for name in fixtures.SEQUENCES}, tmp_path)
