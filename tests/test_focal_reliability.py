from __future__ import annotations

import csv
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pim.metrics.geometry import backproject, implied_focal

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_script", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


focal_reliability = load_script("focal_reliability")

H, W = 180, 240
SEQUENCE = "seq_a"
K = np.array([[210.0, 0.0, 119.5], [0.0, 216.0, 89.5], [0.0, 0.0, 1.0]])
FRAMES = 3


def depth_image(index: int) -> np.ndarray:
    v, u = np.indices((H, W))
    depth = 1.2 + 0.01 * v + 0.004 * u + 0.05 * index
    depth[30:90, 20:60] = 0.9
    depth[100:160, 170:220] = 1.6
    return depth.astype(np.float32)


def prediction(depth: np.ndarray, focal_scale: float) -> np.ndarray:
    K_pred = K.copy()
    K_pred[0, 0] *= focal_scale
    K_pred[1, 1] *= focal_scale
    points, _ = backproject(depth, depth > 0, K_pred)
    return (0.4 * points + np.array([0.1, -0.3, 0.6])).astype(np.float32)


@pytest.mark.parametrize("focal_scale", [1.0, 1.2, 0.8])
def test_full_left_and_right_fits_recover_a_known_focal(focal_scale: float) -> None:
    depth = depth_image(0)

    result = focal_reliability.focal_split(prediction(depth, focal_scale), depth > 0, K)

    for column in ("focal_full", "focal_left", "focal_right"):
        assert result[column] == pytest.approx(focal_scale, abs=1e-3)
    assert result["focal_split_logdiff"] < 1e-3


def test_side_with_too_few_valid_pixels_is_empty() -> None:
    depth = depth_image(0)
    valid = depth > 0
    valid[:, :67] = False

    result = focal_reliability.focal_split(prediction(depth, 1.2), valid, K)

    assert math.isnan(result["focal_left"]) and math.isnan(result["focal_split_logdiff"])
    assert result["focal_right"] == pytest.approx(1.2, abs=1e-3)
    assert result["focal_full"] == pytest.approx(1.2, abs=1e-3)


def test_frame_with_too_few_valid_pixels_is_all_empty() -> None:
    depth = depth_image(0)
    valid = np.zeros((H, W), bool)
    valid[0, :50] = True

    result = focal_reliability.focal_split(prediction(depth, 1.2), valid, K)

    assert all(math.isnan(value) for value in result.values())


def write_sequence(root: Path) -> tuple[Path, Path, Path]:
    seq = root / "data" / SEQUENCE
    (seq / "rgb").mkdir(parents=True)
    (seq / "depth").mkdir()
    run_dir = root / "predictions" / "test"
    (run_dir / SEQUENCE).mkdir(parents=True)
    rgb_lines, depth_lines = [], []
    for i in range(FRAMES):
        t = 1000.0 + i / 30
        Image.fromarray(np.zeros((H, W, 3), np.uint8)).save(seq / "rgb" / f"{i}.png")
        Image.fromarray(np.round(depth_image(i) * 5000).astype(np.uint16)).save(seq / "depth" / f"{i}.png")
        rgb_lines.append(f"{t:.6f} rgb/{i}.png")
        depth_lines.append(f"{t + 0.001:.6f} depth/{i}.png")
        np.savez_compressed(run_dir / SEQUENCE / f"{i:05d}.npz", points=prediction(depth_image(i), 1.2).astype(np.float16),
                            rgb_file=np.str_(f"rgb/{i}.png"), latency_ms=np.float64(1.0), peak_mem_gb=np.float64(1.0))
    (seq / "rgb.txt").write_text("# rgb\n" + "\n".join(rgb_lines) + "\n")
    (seq / "depth.txt").write_text("# depth\n" + "\n".join(depth_lines) + "\n")
    gt = [f"{999.9 + 0.01 * k:.6f} 0 0 0 0 0 0 1" for k in range(50)]
    (seq / "groundtruth.txt").write_text("# gt\n" + "\n".join(gt) + "\n")
    config = root / "sequences.yaml"
    config.write_text(f"cameras:\n  cam: {{fx: {K[0, 0]}, fy: {K[1, 1]}, cx: {K[0, 2]}, cy: {K[1, 2]}}}\n"
                      f"sequences:\n  {SEQUENCE}:\n    camera: cam\n    window: {{start: 0, count: {FRAMES}}}\n")
    return run_dir, root / "data", config


def write_per_frame(path: Path, run_dir: Path, offset: float = 0.0) -> Path:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=("sequence", "frame_index", "focal_ratio"))
        writer.writeheader()
        for i in range(FRAMES):
            with np.load(run_dir / SEQUENCE / f"{i:05d}.npz") as data:
                points = data["points"].astype(np.float32)
            ratio = implied_focal(points, depth_image(i) > 0, K[0, 2]) / K[0, 0]
            writer.writerow({"sequence": SEQUENCE, "frame_index": i, "focal_ratio": repr(float(ratio + offset * (i == 1)))})
    return path


def test_run_writes_one_row_per_frame_and_matches_per_frame_csv(tmp_path: Path) -> None:
    run_dir, data_root, config = write_sequence(tmp_path)
    out = tmp_path / "eval" / "focal_split.csv"

    focal_reliability.run_focal_split(run_dir, data_root, config, write_per_frame(tmp_path / "pf.csv", run_dir), out)

    with open(out, newline="") as f:
        rows = list(csv.DictReader(f))
    assert tuple(rows[0]) == focal_reliability.COLUMNS
    assert [int(r["frame_index"]) for r in rows] == list(range(FRAMES))
    for r in rows:
        for column in ("focal_full", "focal_left", "focal_right"):
            assert float(r[column]) == pytest.approx(1.2, abs=5e-3)


def test_run_fails_when_focal_full_differs_from_focal_ratio(tmp_path: Path) -> None:
    run_dir, data_root, config = write_sequence(tmp_path)
    per_frame = write_per_frame(tmp_path / "pf.csv", run_dir, offset=2e-6)
    out = tmp_path / "focal_split.csv"

    with pytest.raises(ValueError, match="by more than 1e-06 on 1 frames:\n  seq_a frame 1"):
        focal_reliability.run_focal_split(run_dir, data_root, config, per_frame, out)
    assert not out.exists()


def test_consistency_accepts_empty_on_both_sides_and_rejects_empty_on_one() -> None:
    rows = [{"sequence": "s", "frame_index": 0, "focal_full": math.nan}]

    focal_reliability.check_consistency(rows, {"s": {0: math.nan}})
    with pytest.raises(ValueError, match="s frame 0"):
        focal_reliability.check_consistency(rows, {"s": {0: 1.0}})
