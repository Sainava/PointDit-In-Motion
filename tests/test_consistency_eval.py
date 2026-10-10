from __future__ import annotations

import csv
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pim.consistency import evaluation
from pim.consistency.evaluation import (
    PAIR_COLUMNS, FrameStore, PairTask, atomic_write, evaluation_order, plan_pairs, read_pairs,
)
from pim.data.tum import Frame
from pim.metrics.geometry import backproject


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_script",
                                                  Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


consistency_eval = load_script("consistency_eval")

# Small frames keep upstream's quadratic alignment cheap: at most H * W pixels per fit.
H, W = 40, 52
FX, FY, CX, CY = 45.0, 45.0, 25.5, 19.5
K = np.array([[FX, 0.0, CX], [0.0, FY, CY], [0.0, 0.0, 1.0]])
SEQUENCES = ("seq_a", "seq_b")
# Slot 3 is a hole, so frames 2 and 3 are 2/30 s apart.
SLOTS = (0, 1, 2, 4, 5, 6, 7)
SPARSE_FRAME = 4
GAPS = ["33", "100"]


def depth_image(index: int) -> np.ndarray:
    v, u = np.indices((H, W))
    depth = 2.0 + 0.004 * v + 0.003 * u + 0.01 * index
    depth[14:20, 20:28] = 1.5
    if index == SPARSE_FRAME:
        depth[10:] = 0.0
    return depth


def write_sequence(data_root: Path, name: str) -> None:
    seq = data_root / name
    (seq / "rgb").mkdir(parents=True)
    (seq / "depth").mkdir()
    rgb_lines, depth_lines = [], []
    for i, slot in enumerate(SLOTS):
        t = 1000.0 + slot / 30
        Image.fromarray(np.full((H, W, 3), 10 * i, np.uint8)).save(seq / "rgb" / f"{i}.png")
        Image.fromarray(np.round(depth_image(i) * 5000).astype(np.uint16)).save(seq / "depth" / f"{i}.png")
        rgb_lines.append(f"{t:.6f} rgb/{i}.png")
        depth_lines.append(f"{t + 0.002 * (i % 3):.6f} depth/{i}.png")
    (seq / "rgb.txt").write_text("\n".join(rgb_lines) + "\n")
    (seq / "depth.txt").write_text("\n".join(depth_lines) + "\n")
    gt = [f"{999.9 + 0.01 * k:.6f} {0.001 * k:.6f} 0 0 0 0 0 1" for k in range(60)]
    (seq / "groundtruth.txt").write_text("\n".join(gt) + "\n")


def write_predictions(run_dir: Path, name: str) -> None:
    (run_dir / name).mkdir(parents=True)
    for i in range(len(SLOTS)):
        depth = depth_image(i).astype(np.float32)
        gt_points, _ = backproject(depth, depth > 0, K)
        rng = np.random.default_rng(i)
        points = 0.5 * gt_points + np.array([0.1, -0.2, 0.3]) + 0.01 * rng.standard_normal(gt_points.shape)
        np.savez_compressed(run_dir / name / f"{i:05d}.npz", points=points.astype(np.float16),
                            rgb_file=np.str_(f"rgb/{i}.png"), latency_ms=np.float64(1.0),
                            peak_mem_gb=np.float64(1.0))


@pytest.fixture(scope="module")
def setup(tmp_path_factory) -> tuple[Path, Path, Path]:
    root = tmp_path_factory.mktemp("consistency")
    data_root, run_dir = root / "data", root / "predictions" / "test"
    config = root / "sequences.yaml"
    entries = "".join(f"  {n}:\n    camera: cam\n    window: {{start: 0, count: {len(SLOTS)}}}\n" for n in SEQUENCES)
    config.write_text(f"cameras:\n  cam: {{fx: {FX}, fy: {FY}, cx: {CX}, cy: {CY}}}\nsequences:\n{entries}")
    for name in SEQUENCES:
        write_sequence(data_root, name)
        write_predictions(run_dir, name)
    window = {"start": 0, "count": len(SLOTS)}
    run = {"name": "test", "sequences": {n: {"window": window, "frames": len(SLOTS)} for n in SEQUENCES}}
    (run_dir / "run.json").write_text(json.dumps(run))
    return run_dir, data_root, config


def args_for(setup, out: Path, *extra: str):
    run_dir, data_root, config = setup
    return consistency_eval.parse_args(["--predictions", str(run_dir), "--data-root", str(data_root),
                                        "--sequences-config", str(config), "--out", str(out),
                                        "--gaps-ms", *GAPS, *extra])


@pytest.fixture(scope="module")
def single_run(setup, tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("single")
    assert consistency_eval.run_consistency(args_for(setup, out, "--save-maps", "seq_a"))
    return out


def test_pair_csv_has_all_columns_and_too_few_rows_are_nan(single_run: Path) -> None:
    rows = read_pairs(single_run / "pairs_seq_a.csv")
    assert set(rows[0]) == set(PAIR_COLUMNS)
    for column in ("m2_visible_median", "m2_interior_far_median", "gt_floor_rgb_visible_p90",
                   "gt_floor_depth_interior_near_median", "m1_visible_mean", "m2_floor_interior_mid_median"):
        assert column in PAIR_COLUMNS
    metric_columns = [c for c in PAIR_COLUMNS if c.endswith(("mean", "median", "p90"))]
    assert len(metric_columns) == 5 * 2 * 6

    too_few = [r for r in rows if r["too_few_pixels"] == "True"]
    scored = [r for r in rows if r["too_few_pixels"] == "False"]
    assert too_few and scored
    assert all(SPARSE_FRAME in (int(r["frame_i"]), int(r["frame_j"])) for r in too_few)
    assert all(r[c] == "" for r in too_few for c in metric_columns)
    assert all(r[f"{m}_visible_median"] != "" for r in scored for m in evaluation.METRICS)
    assert [(r["gap_ms"], r["frame_i"]) for r in rows] == sorted(
        ((r["gap_ms"], r["frame_i"]) for r in rows), key=lambda k: (float(k[0]), int(k[1])))


def test_maps_are_saved_at_33_ms_for_named_sequences_only(single_run: Path) -> None:
    maps = sorted((single_run / "maps" / "seq_a").glob("*.npz"))
    rows = [r for r in read_pairs(single_run / "pairs_seq_a.csv") if r["gap_ms"] == "33"]
    assert len(maps) == len(rows)
    assert not (single_run / "maps" / "seq_b").exists()
    with np.load(maps[0]) as data:
        assert data["m2"].dtype == np.float16 and data["m2"].shape == (H, W)
        assert data["anchor_depth"].dtype == np.float16
        assert np.isfinite(data["m2"]).all() and int(data["frame_i"]) == 0
    with np.load(next(m for m in maps if m.stem.endswith(f"_{SPARSE_FRAME:05d}"))) as data:
        assert np.isnan(data["m2"][10:]).all()


def test_summary_and_record_are_written(single_run: Path) -> None:
    with open(single_run / "summary.csv", newline="") as f:
        summary = list(csv.DictReader(f))
    assert len(summary) == len(SEQUENCES) * len(GAPS) * 2
    record = json.loads((single_run / "consistency.json").read_text())
    assert set(record["sequences"]) == set(SEQUENCES)
    assert record["sequences"]["seq_a"]["peak_mem_mib"] > 0
    assert "visibility" in record["protocol"] and record["arguments"]["gaps_ms"] == [33.0, 100.0]


def test_workers_give_identical_csvs(setup, single_run: Path, tmp_path: Path) -> None:
    out = tmp_path / "pool"
    assert consistency_eval.run_consistency(args_for(setup, out, "--workers", "2", "--save-maps", "seq_a"))
    for name in SEQUENCES:
        assert (out / f"pairs_{name}.csv").read_bytes() == (single_run / f"pairs_{name}.csv").read_bytes()
    assert (out / "summary.csv").read_bytes() == (single_run / "summary.csv").read_bytes()


def test_rerun_skips_completed_sequences(setup, single_run: Path, tmp_path: Path, monkeypatch) -> None:
    out = tmp_path / "rerun"
    out.mkdir()
    (out / "pairs_seq_a.csv").write_bytes((single_run / "pairs_seq_a.csv").read_bytes())
    # seq_b's CSV is missing a row, so it is incomplete and recomputed.
    lines = (single_run / "pairs_seq_b.csv").read_text().splitlines(keepends=True)
    (out / "pairs_seq_b.csv").write_text("".join(lines[:-1]))
    evaluated = []
    real = consistency_eval.evaluate_sequence
    monkeypatch.setattr(consistency_eval, "evaluate_sequence", lambda job: evaluated.append(job.name) or real(job))

    assert consistency_eval.run_consistency(args_for(setup, out))

    assert evaluated == ["seq_b"]
    assert (out / "pairs_seq_b.csv").read_bytes() == (single_run / "pairs_seq_b.csv").read_bytes()
    assert json.loads((out / "consistency.json").read_text())["sequences"]["seq_a"]["reused"] is True


def test_requesting_maps_makes_a_sequence_without_maps_incomplete(setup, single_run: Path) -> None:
    job = consistency_eval.sequence_job(args_for(setup, single_run, "--save-maps", "seq_b"), "seq_b")
    pairs = consistency_eval.plan_sequence(job, json.loads((setup[0] / "run.json").read_text()))
    assert evaluation.incomplete_reason(single_run, "seq_b", pairs, save_maps=False) is None
    assert "maps missing" in evaluation.incomplete_reason(single_run, "seq_b", pairs, save_maps=True)


def test_failure_mid_sequence_leaves_no_partial_file(setup, tmp_path: Path) -> None:
    out = tmp_path / "failed"
    job = consistency_eval.sequence_job(args_for(setup, out), "seq_a")
    calls = []

    def failing(*args, **kwargs):
        calls.append(1)
        if len(calls) == 3:
            raise RuntimeError("boom")
        return evaluation.pair_consistency(*args, **kwargs)

    with pytest.raises(RuntimeError, match="boom"):
        evaluation.evaluate_sequence(job, compute=failing)
    assert not out.exists() or not any(out.rglob("*"))


def test_atomic_write_removes_partial_file_on_error(tmp_path: Path) -> None:
    path = tmp_path / "pairs_x.csv"

    def write(f) -> None:
        f.write("sequence,gap_ms\n")
        raise OSError("disk full")

    with pytest.raises(OSError):
        atomic_write(path, write)
    assert list(tmp_path.iterdir()) == []


def row(gap: str, frame_i: int, **metrics: float | str) -> dict[str, str]:
    cells = dict.fromkeys(PAIR_COLUMNS, "")
    cells.update(sequence="s", gap_ms=gap, frame_i=str(frame_i), frame_j=str(frame_i + 1), too_few_pixels="False")
    cells.update({k: str(v) for k, v in metrics.items()})
    return cells


def test_summary_medians_iqr_and_ratios_match_hand_computation() -> None:
    m1 = (1.0, 2.0, 4.0, 10.0)
    floor = (0.5, 0.5, 1.0, 2.0)
    rows = [row("33", i, frac_visible=0.9, m1_visible_median=a, gt_floor_depth_visible_median=b)
            for i, (a, b) in enumerate(zip(m1, floor))]
    rows.append(row("33", 9, too_few_pixels="True"))

    summary = consistency_eval.summarize({"s": rows})
    visible = next(e for e in summary if e["pixels"] == "visible")

    assert visible["pairs"] == 5 and visible["too_few_pairs"] == 1
    assert visible["m1_median"] == pytest.approx(3.0)
    # numpy linear percentiles of (1, 2, 4, 10): q1 = 1.75, q3 = 5.5.
    assert visible["m1_iqr"] == pytest.approx(3.75)
    assert visible["m1_n"] == 4
    # Per-pair ratios 2, 4, 4, 5.
    assert visible["m1_over_gt_floor_depth"] == pytest.approx(4.0)
    assert visible["gt_floor_depth_median"] == pytest.approx(0.75)
    assert math.isnan(visible["m2_median"]) and math.isnan(visible["m2_over_m2_floor"])


EXPECTED = {
    "tolerance": 1e-3,
    "tables": {
        "full": {"pairs": {"s": {"33": 3}}, "values": {"s": {"visible": {"33": {"m2_floor": 0.25}}}}},
        "sample": {
            "frames": {"s": [0, 2]},
            "pairs": {"s": {"33": 2}},
            "values": {"s": {"visible": {"33": {"m2": 1.5}}}, "missing": {"visible": {"33": {"m2": 1.0}}}},
        },
    },
}


def regression_rows(m2_shift: float = 0.0) -> list[dict[str, str]]:
    return [row("33", 0, m2_floor_visible_median=0.2, m2_visible_median=1.0 + m2_shift),
            row("33", 1, m2_floor_visible_median=0.25, m2_visible_median=2.0 + m2_shift),
            row("33", 2, m2_floor_visible_median=0.3, m2_visible_median=9.0)]


def test_regression_passes_on_matching_values() -> None:
    checks = consistency_eval.regression_checks({"s": regression_rows()}, EXPECTED)

    assert consistency_eval.report_regression(checks, EXPECTED["tolerance"])
    assert {(c.table, c.metric): c.actual for c in checks if c.skipped is None} == {
        ("full", "pairs"): 3, ("full", "m2_floor"): 0.25, ("sample", "pairs"): 2, ("sample", "m2"): 1.5}
    assert [c.sequence for c in checks if c.skipped] == ["missing"]


def test_regression_fails_on_a_perturbed_value() -> None:
    checks = consistency_eval.regression_checks({"s": regression_rows(m2_shift=0.002)}, EXPECTED)

    assert not consistency_eval.report_regression(checks, EXPECTED["tolerance"])
    assert [c.metric for c in checks if not c.passed(EXPECTED["tolerance"])] == ["m2"]


def test_regression_fails_on_a_pair_count_mismatch() -> None:
    checks = consistency_eval.regression_checks({"s": regression_rows()[:2]}, EXPECTED)
    assert [(c.table, c.metric) for c in checks if not c.passed(1e-3)] == [("full", "pairs"), ("full", "m2_floor")]


def test_expected_values_file_has_both_subtask1_tables() -> None:
    expected = json.loads(consistency_eval.EXPECTED.read_text())
    full, sample = expected["tables"]["gt_only_full_window"], expected["tables"]["prediction_sample"]
    fr3, fr1_360 = "rgbd_dataset_freiburg3_long_office_household", "rgbd_dataset_freiburg1_360"
    assert full["values"][fr3]["visible"]["33"] == {"gt_floor_depth": 0.261, "m2_floor": 0.246}
    assert full["values"]["rgbd_dataset_freiburg1_desk"]["visible"]["333"]["gt_floor_depth"] == 0.964
    assert sample["values"][fr3]["visible"]["33"]["m1"] == 0.756
    assert sample["values"][fr1_360]["visible"]["33"]["m2"] == 0.463
    assert sample["frames"] == {fr1_360: [0, 19], fr3: [637, 656]}


def frames(times: list[float]) -> list[Frame]:
    return [Frame(k, t, Path(f"{k}.png"), Path(f"{k}.png"), 0.0, np.eye(4)) for k, t in enumerate(times)]


def test_frames_are_released_after_their_last_pair() -> None:
    sequence = frames([k / 30 for k in range(30)])
    pairs = plan_pairs(sequence, [33.0, 100.0, 333.0])
    loaded = []
    store = FrameStore(lambda f: loaded.append(f.index) or f.index, [(p.frame_i, p.frame_j) for p in pairs])
    order = evaluation_order(pairs)
    max_cached = 0

    for step, pair in enumerate(order):
        assert store.get(pair.frame_i) == pair.frame_i.index and store.get(pair.frame_j) == pair.frame_j.index
        store.release(pair.frame_i, pair.frame_j)
        still_needed = {f.index for p in order[step + 1:] for f in (p.frame_i, p.frame_j)}
        assert store.cached <= still_needed
        max_cached = max(max_cached, len(store.cached))

    assert sorted(loaded) == list(range(30)), "every frame is loaded exactly once"
    assert store.cached == set()
    # A 333 ms gap spans 10 frames, so about 11 frames are held at once, not the whole sequence.
    assert max_cached <= 12


def test_frame_store_rejects_use_after_release() -> None:
    a, b = frames([0.0, 0.033])
    store = FrameStore(lambda f: f.index, [(a, b)])
    store.get(a), store.get(b)
    store.release(a, b)
    with pytest.raises(RuntimeError, match="already released"):
        store.get(a)
    with pytest.raises(RuntimeError, match="released more often"):
        store.release(a)


def test_save_maps_requires_the_33_ms_gap(setup, tmp_path: Path) -> None:
    run_dir, data_root, config = setup
    args = consistency_eval.parse_args(["--predictions", str(run_dir), "--data-root", str(data_root),
                                        "--sequences-config", str(config), "--out", str(tmp_path),
                                        "--gaps-ms", "100", "--save-maps", "seq_a"])
    with pytest.raises(ValueError, match="33 ms gap"):
        consistency_eval.run_consistency(args)


def test_pair_task_key_identifies_gap_and_frames() -> None:
    a, b = frames([0.0, 0.033])
    assert PairTask(33.0, a, b).key == (33.0, 0, 1)
