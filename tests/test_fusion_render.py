from __future__ import annotations

import csv
import importlib.util
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pim.fusion.joint_scale import JointScaleConfig
from pim.fusion.oracle import PREPARE_RECORD, resolve_cache_dir
from pim.metrics.geometry import backproject


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_script",
                                                  Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


render = load_script("fusion_render_prepare")

H, W = 120, 160
F = 132.0
K = np.array([[F, 0.0, (W - 1) / 2], [0.0, F, (H - 1) / 2], [0.0, 0.0, 1.0]])
N = 5
NAME = "synthetic"


def camera_pose(k: int) -> np.ndarray:
    c, s = math.cos(0.01 * k), math.sin(0.01 * k)
    T = np.eye(4)
    T[:3, :3] = [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]
    T[:3, 3] = [0.04 * k, 0.01 * k, 0.03 * k]
    return T


def render_depth(T: np.ndarray) -> np.ndarray:
    """Camera z of the nearest of a back wall, a floor and a side wall."""
    v, u = np.indices((H, W), dtype=np.float64)
    d = np.stack([(u - K[0, 2]) / F, (v - K[1, 2]) / F, np.ones_like(u)], axis=-1) @ T[:3, :3].T
    c = T[:3, 3]
    depth = np.full((H, W), np.inf)
    for axis, value in ((2, 4.0), (1, 1.0), (0, -1.5)):
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (value - c[axis]) / d[..., axis]
        depth = np.where((t > 0) & (t < depth), t, depth)
    return depth


class FakeStore:
    def __init__(self, raw: list[np.ndarray], gt: list[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> None:
        self.raw, self._gt = raw, gt

    def retain(self, positions: set[int]) -> None:
        pass

    def raw_z(self, position: int) -> np.ndarray:
        return self.raw[position]

    def gt(self, position: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._gt[position]


@pytest.fixture(scope="module")
def scene():
    rng = np.random.default_rng(0)
    poses = [camera_pose(k) for k in range(N)]
    depths = [render_depth(T) for T in poses]
    raw = [((d * (1 + 0.002 * rng.standard_normal(d.shape)) - 0.2) / 0.8).astype(np.float32) for d in depths]
    gt = []
    for d in depths:
        gt_depth = d.astype(np.float32)
        gt_points, gt_valid = backproject(gt_depth, np.isfinite(gt_depth), K)
        gt.append((gt_depth, gt_valid, gt_points))
    frames = [SimpleNamespace(index=10 + k, timestamp=1000.0 + k / 30, T_world_cam=poses[k]) for k in range(N)]
    return frames, FakeStore(raw, gt)


def test_frames_without_four_earlier_sources_keep_the_raw_depth_only(scene, tmp_path: Path) -> None:
    frames, store = scene
    values, written, existing = render.write_outputs(NAME, store, frames, K, range(N), tmp_path,
                                                     render.fusion_config(), JointScaleConfig())
    assert (written, existing) == (N, 0)
    for k in range(N):
        with np.load(render.frame_path(tmp_path, NAME, 10 + k)) as data:
            fused = k == N - 1
            assert bool(data["has_fused"]) is fused
            assert ("fused_aligned" in data.files) is fused
            assert int(data["n_sources"]) == min(k, 4)
            assert data["raw_aligned"].dtype == np.float16 and data["raw_aligned"].shape == (H, W)
            assert int(data["index"]) == 10 + k and str(data["sequence"]) == NAME
            assert float(data["timestamp"]) == frames[k].timestamp
            # rel_d_raw is what baseline_self scores.
            gt_depth, gt_valid, gt_points = store.gt(k)
            expected = render.fusion_joint.score_depth(store.raw_z(k), gt_depth, gt_valid, gt_points)["rel_d"]
            assert float(data["rel_d_raw"]) == expected
            assert math.isfinite(float(data["rel_d_fused"])) is fused
            if fused:
                assert data["fused_aligned"].dtype == np.float16
    assert values[-1][0] == 10 + N - 1 and math.isfinite(values[-1][2])


def test_existing_files_are_skipped_and_writes_are_atomic(scene, tmp_path: Path, monkeypatch) -> None:
    frames, store = scene
    kept = render.frame_path(tmp_path, NAME, 11)
    kept.parent.mkdir(parents=True)
    kept.write_bytes(b"already here")
    _, written, existing = render.write_outputs(NAME, store, frames, K, range(3), tmp_path, render.fusion_config(),
                                                JointScaleConfig())
    assert (written, existing) == (2, 1)
    assert kept.read_bytes() == b"already here"

    def broken(f, **arrays) -> None:
        f.write(b"partial")
        raise OSError("disk full")

    monkeypatch.setattr(np, "savez", broken)
    with pytest.raises(OSError, match="disk full"):
        render.write_outputs(NAME, store, frames, K, [3], tmp_path, render.fusion_config(), JointScaleConfig())
    assert not render.frame_path(tmp_path, NAME, 13).exists()
    assert not list(tmp_path.rglob("*.partial"))


def synthetic_output(index: int, rel_d_raw: float, rel_d_fused: float) -> object:
    depth = np.full((4, 5), 2.0, dtype=np.float32)
    return render.FrameOutput(NAME, index, float(index), depth, depth, rel_d_raw, rel_d_fused, False, 4)


@pytest.mark.parametrize("perturb, code", [(None, None), ("raw", 3), ("fused", 3), ("fused_small", None)])
def test_regression_check_exits_with_code_3_after_outputs_exist(tmp_path: Path, monkeypatch, perturb, code) -> None:
    checked = {NAME: [100, 101, 102]}
    baseline = {(NAME, i): 1.0 + i / 1000 for i in checked[NAME]}
    joint = {(NAME, i): 0.9 + i / 1000 for i in checked[NAME]}
    for i in checked[NAME]:
        render.write_frame(render.frame_path(tmp_path, NAME, i),
                           synthetic_output(i, baseline[(NAME, i)], joint[(NAME, i)]))
    if perturb == "raw":
        baseline[(NAME, 101)] += 1e-3
    elif perturb == "fused":
        joint[(NAME, 102)] += 1e-2
    elif perturb == "fused_small":
        joint[(NAME, 102)] += 5e-4

    monkeypatch.setattr(render, "run_render", lambda args: render.finish(tmp_path, checked, baseline, joint, {}))
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    argv = ["--predictions", "p", "--cache", "c", "--data-root", "d", "--baseline-csv", "b", "--joint-csv", "j",
            "--out", str(tmp_path)]
    if code is None:
        render.main(argv)
    else:
        with pytest.raises(SystemExit) as raised:
            render.main(argv)
        assert raised.value.code == code
    assert all(render.frame_path(tmp_path, NAME, i).is_file() for i in checked[NAME])
    status = json.loads((tmp_path / render.STATUS_JSON).read_text())
    assert status["status"] == ("pass" if code is None else "fail")
    assert status["scale"] == "joint_solved" and status["poses"] == "oracle_gt"
    assert status["rel_d_raw"]["rows"] == status["rel_d_fused"]["rows"] == 3
    assert (tmp_path / render.RECORD_JSON).is_file()


def test_missing_checked_file_is_a_mismatch(tmp_path: Path) -> None:
    report = render.regression(render.scores_from_files(tmp_path, {NAME: [100]}), {(NAME, 100): 1.0},
                               {(NAME, 100): 1.0})
    assert not report["passed"]


def test_read_joint_takes_only_the_deployable_config(tmp_path: Path) -> None:
    columns = render.fusion_joint.ROW_COLUMNS
    path = render.fusion_joint.fusion_csv(tmp_path, NAME)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for config, value in (("baseline_self", "1.5"), (render.CONFIG, "1.2"), ("joint_causal_diverse_w4", "1.9")):
            writer.writerow({**dict.fromkeys(columns, ""), "sequence": NAME, "target_index": "100", "config": config,
                             "scale": "joint_solved", "poses": "oracle_gt", "rel_d": value})
    assert render.read_joint(tmp_path, [NAME]) == {(NAME, 100): 1.2}
    assert render.read_joint(path, [NAME]) == {(NAME, 100): 1.2}


def test_cache_path_helper_accepts_both_forms(tmp_path: Path) -> None:
    root = tmp_path / "notebook-output"
    cache = root / "fusion" / "oracle_aligned" / "baseline"
    cache.mkdir(parents=True)
    (cache / PREPARE_RECORD).write_text("{}")
    assert resolve_cache_dir(root, "baseline") == cache
    assert resolve_cache_dir(cache, "baseline") == cache
    assert resolve_cache_dir(cache.parent, "baseline") == cache
    with pytest.raises(FileNotFoundError):
        resolve_cache_dir(root, "other_run")
