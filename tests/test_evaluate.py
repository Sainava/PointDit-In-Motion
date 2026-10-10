from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pim.evaluation import COLUMNS, EXTRA_METRICS, extra_metrics
from pim.metrics.error_maps import rel_d_by_depth_tertile, relative_depth_error
from pim.metrics.geometry import backproject, evaluate_frame

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_script", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


evaluate = load_script("evaluate")
check_frame = load_script("check_frame")

H, W = 60, 80
SEQUENCE = "seq_a"
FX, FY, CX, CY = 70.0, 72.0, 39.5, 29.5
K = np.array([[FX, 0.0, CX], [0.0, FY, CY], [0.0, 0.0, 1.0]])
# Frame 3's slot is a hole: frames 2 and 3 are 2/30 s apart.
RGB_TIMES = [1000.0 + k / 30 for k in (0, 1, 2, 4, 5, 6)]
DEPTH_OFFSETS = [0.004, -0.010, 0.0, 0.015, -0.002, 0.008]
WINDOW = {"start": 1, "count": 5}
SKIPPED_FRAME = 3


def depth_image(index: int) -> np.ndarray:
    v, u = np.indices((H, W))
    depth = 1.0 + 0.01 * v + 0.005 * u + 0.05 * index
    depth[10:25, 30:50] = 0.8
    depth[::6, ::7] = 0.0
    if index == SKIPPED_FRAME:
        depth[:] = 0.0
        depth[0, :50] = 1.0
    return depth


def write_sequence(root: Path) -> Path:
    seq = root / "data" / SEQUENCE
    (seq / "rgb").mkdir(parents=True)
    (seq / "depth").mkdir()
    rgb_lines, depth_lines = [], []
    for i, (t, offset) in enumerate(zip(RGB_TIMES, DEPTH_OFFSETS)):
        Image.fromarray(np.full((H, W, 3), 10 * i, np.uint8)).save(seq / "rgb" / f"{i}.png")
        Image.fromarray(np.round(depth_image(i) * 5000).astype(np.uint16)).save(seq / "depth" / f"{i}.png")
        rgb_lines.append(f"{t:.6f} rgb/{i}.png")
        depth_lines.append(f"{t + offset:.6f} depth/{i}.png")
    (seq / "rgb.txt").write_text("# rgb\n" + "\n".join(rgb_lines) + "\n")
    (seq / "depth.txt").write_text("# depth\n" + "\n".join(depth_lines) + "\n")
    gt = [f"{999.9 + 0.01 * k:.6f} 0 0 0 0 0 0 1" for k in range(50)]
    (seq / "groundtruth.txt").write_text("# gt\n" + "\n".join(gt) + "\n")
    config = root / "sequences.yaml"
    config.write_text(
        f"cameras:\n  cam: {{fx: {FX}, fy: {FY}, cx: {CX}, cy: {CY}}}\n"
        f"sequences:\n  {SEQUENCE}:\n    camera: cam\n"
        f"    window: {{start: {WINDOW['start']}, count: {WINDOW['count']}}}\n"
    )
    return config


def write_prediction(run_dir: Path, index: int, rgb_file: str) -> None:
    depth = depth_image(index).astype(np.float32)
    gt_points, _ = backproject(depth, depth > 0, K)
    rng = np.random.default_rng(index)
    points = 0.5 * gt_points + np.array([0.1, -0.2, 0.3]) + 0.01 * rng.standard_normal(gt_points.shape)
    np.savez_compressed(
        run_dir / SEQUENCE / f"{index:05d}.npz",
        points=points.astype(np.float16),
        valid=np.ones((H, W), bool),
        rgb_timestamp=np.float64(RGB_TIMES[index]),
        rgb_file=np.str_(rgb_file),
        latency_ms=np.float64(1000.0 + index),
        peak_mem_gb=np.float64(2.5),
    )


def write_run(root: Path, window: dict[str, int] = WINDOW) -> Path:
    run_dir = root / "predictions" / "test"
    (run_dir / SEQUENCE).mkdir(parents=True)
    for i in range(window["start"], window["start"] + window["count"]):
        write_prediction(run_dir, i, f"rgb/{i}.png")
    run = {"name": "test", "sequences": {SEQUENCE: {"window": window, "frames": window["count"]}}}
    (run_dir / "run.json").write_text(json.dumps(run))
    return run_dir


@pytest.fixture
def setup(tmp_path: Path) -> tuple[Path, Path, Path]:
    config = write_sequence(tmp_path)
    return write_run(tmp_path), tmp_path / "data", config


def run(setup: tuple[Path, Path, Path], out: Path, workers: int = 1) -> list[dict[str, str]]:
    run_dir, data_root, config = setup
    evaluate.run_evaluation(run_dir, data_root, config, None, out, workers)
    with open(out / "per_frame.csv", newline="") as f:
        return list(csv.DictReader(f))


def test_one_row_per_window_frame_with_all_columns(setup, tmp_path: Path) -> None:
    rows = run(setup, tmp_path / "eval")

    with open(tmp_path / "eval" / "per_frame.csv", newline="") as f:
        assert tuple(next(csv.reader(f))) == COLUMNS
    assert [int(r["frame_index"]) for r in rows] == [1, 2, 3, 4, 5]
    assert [r["rgb_file"] for r in rows] == [f"rgb/{i}.png" for i in range(1, 6)]
    assert [r["rgb_timestamp"] for r in rows] == [f"{RGB_TIMES[i]:.6f}" for i in range(1, 6)]
    assert [float(r["latency_ms"]) for r in rows] == [1001.0, 1002.0, 1003.0, 1004.0, 1005.0]
    assert not (tmp_path / "eval" / "per_frame.csv.partial").exists()

    record = json.loads((tmp_path / "eval" / "eval.json").read_text())
    assert record["sequences"][SEQUENCE]["frames"] == 5
    assert record["sequences"][SEQUENCE]["skipped"] == 1
    assert record["predictions"]["run"]["name"] == "test"
    assert set(record["commits"]) == {"project", "pointdit"}


def test_gap_and_depth_offset_in_milliseconds(setup, tmp_path: Path) -> None:
    rows = run(setup, tmp_path / "eval")

    written = [float(f"{t:.6f}") for t in RGB_TIMES]
    expected_gaps = [(written[i] - written[i - 1]) * 1000 for i in range(2, 6)]
    assert rows[0]["gap_ms"] == ""
    assert [float(r["gap_ms"]) for r in rows[1:]] == pytest.approx(expected_gaps, abs=1e-3)
    assert float(rows[2]["gap_ms"]) == pytest.approx(66.666, abs=1e-3)
    assert [float(r["depth_offset_ms"]) for r in rows] == pytest.approx(
        [abs(o) * 1000 for o in DEPTH_OFFSETS[1:]], abs=2e-3)


def test_frame_with_too_few_valid_pixels_has_empty_metrics(setup, tmp_path: Path) -> None:
    rows = run(setup, tmp_path / "eval")

    skipped = rows[SKIPPED_FRAME - 1]
    assert skipped["skipped"] == "True"
    assert int(skipped["num_valid"]) == 50
    assert float(skipped["gt_valid_fraction"]) == pytest.approx(50 / (H * W), abs=1e-6)
    assert all(skipped[m] == "" for m in ("rel_p", "delta1_p", "rel_d", "delta1_d", *EXTRA_METRICS))
    for r in rows[:SKIPPED_FRAME - 1] + rows[SKIPPED_FRAME:]:
        assert r["skipped"] == "False"
        assert 0.0 <= float(r["rel_d"]) < 5.0 and float(r["delta1_d"]) > 95.0
        assert all(0.0 <= float(r[m]) < 5.0 for m in ("rel_d_near", "rel_d_mid", "rel_d_far"))
        # No pixel of an 80 px wide frame is more than 50 px from cx.
        assert r["focal_ratio"] == ""


def test_rgb_file_mismatch_fails_before_writing_anything(setup, tmp_path: Path) -> None:
    run_dir = setup[0]
    write_prediction(run_dir, 4, "rgb/5.png")

    with pytest.raises(ValueError, match="00004.npz was predicted from rgb/5.png, but frame 4 is rgb/4.png"):
        run(setup, tmp_path / "eval")
    assert not (tmp_path / "eval").exists()


def test_window_changed_since_prediction_fails(tmp_path: Path) -> None:
    config = write_sequence(tmp_path)
    run_dir = write_run(tmp_path, {"start": 0, "count": 5})

    with pytest.raises(ValueError, match="predictions cover window"):
        evaluate.run_evaluation(run_dir, tmp_path / "data", config, None, tmp_path / "eval", 1)


def test_worker_pool_writes_the_same_rows_in_frame_order(setup, tmp_path: Path) -> None:
    single = run(setup, tmp_path / "one")
    pooled = run(setup, tmp_path / "two", workers=2)

    assert pooled == single


def test_check_frame_numpy_recomputation_matches_evaluate_frame() -> None:
    depth = depth_image(1)
    mask = depth > 0
    gt_points = check_frame.gt_points_numpy(depth, K)
    rng = np.random.default_rng(0)
    pred = (0.4 * gt_points + 0.3 + 0.05 * rng.standard_normal(gt_points.shape)).astype(np.float32)

    result = evaluate_frame(pred, gt_points.astype(np.float32), mask)
    recomputed = check_frame.numpy_metrics(result.aligned_points, result.aligned_depth, gt_points, mask)

    assert not result.skipped
    assert 1.0 < result.rel_p < 50.0
    for name in ("rel_p", "delta1_p", "rel_d", "delta1_d"):
        assert recomputed[name] == pytest.approx(getattr(result, name), abs=check_frame.TOLERANCE)


def three_level_depth() -> np.ndarray:
    depth = np.full((H, W), 1.0, np.float32)
    depth[:, W // 3:2 * W // 3] = 2.0
    depth[:, 2 * W // 3:] = 4.0
    return depth


def test_error_only_in_the_far_third() -> None:
    depth = three_level_depth()
    valid = np.ones((H, W), bool)
    aligned = np.where(depth == 4.0, depth * 1.1, depth)

    thirds = rel_d_by_depth_tertile(relative_depth_error(aligned, depth, valid), depth, valid)

    assert thirds["rel_d_near"] == 0.0
    assert thirds["rel_d_mid"] == 0.0
    assert thirds["rel_d_far"] == pytest.approx(10.0, rel=1e-5)


def test_pixel_weighted_thirds_reproduce_rel_d() -> None:
    depth = depth_image(1).astype(np.float32)
    valid = depth > 0
    gt_points, valid = backproject(depth, valid, K)
    rng = np.random.default_rng(1)
    pred = 0.7 * gt_points + 0.2 + 0.05 * rng.standard_normal(gt_points.shape).astype(np.float32)

    metrics = evaluate_frame(pred, gt_points, valid)
    thirds = extra_metrics(pred, depth, valid, K, metrics)

    low, high = np.percentile(depth[valid], [100 / 3, 200 / 3])
    d = depth[valid]
    counts = np.array([(d < low).sum(), ((d >= low) & (d < high)).sum(), (d >= high).sum()])
    values = np.array([thirds["rel_d_near"], thirds["rel_d_mid"], thirds["rel_d_far"]])
    assert counts.sum() == metrics.num_valid
    assert float(counts @ values / counts.sum()) == pytest.approx(metrics.rel_d, abs=1e-4)


@pytest.mark.parametrize("focal_scale", [1.0, 1.2])
def test_focal_ratio_of_a_prediction_made_with_scaled_focal(focal_scale: float) -> None:
    v, u = np.indices((180, 240))
    depth = (1.2 + 0.01 * v + 0.004 * u).astype(np.float32)
    depth[30:90, 60:135] = 0.9
    K = np.array([[210.0, 0.0, 119.5], [0.0, 216.0, 89.5], [0.0, 0.0, 1.0]])
    gt_points, valid = backproject(depth, np.ones(depth.shape, bool), K)
    K_pred = K.copy()
    K_pred[0, 0] *= focal_scale
    K_pred[1, 1] *= focal_scale
    pred_points, _ = backproject(depth, valid, K_pred)
    pred = (0.4 * pred_points + np.array([0.1, -0.3, 0.6])).astype(np.float32)

    metrics = evaluate_frame(pred, gt_points, valid)
    ratio = extra_metrics(pred, depth, valid, K, metrics)["focal_ratio"]

    assert ratio == pytest.approx(focal_scale, abs=1e-3)
