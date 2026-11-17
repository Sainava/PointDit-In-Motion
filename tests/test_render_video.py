from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pim import video as pim_video
from pim.video import colorize

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_script", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


render_video = load_script("render_video")
export_map = load_script("export_fused_map")
render_orbit = load_script("render_orbit")

H, W = 48, 64
F = 50.0
K = np.array([[F, 0.0, (W - 1) / 2], [0.0, F, (H - 1) / 2], [0.0, 0.0, 1.0]])
DT = 1 / 30
WALL_Z = 2.0


def pose(x: float) -> np.ndarray:
    T = np.eye(4)
    T[0, 3] = x
    return T


def wall_depth(shape: tuple[int, int] = (H, W)) -> np.ndarray:
    depth = np.full(shape, WALL_Z, dtype=np.float32)
    depth[: shape[0] // 3, : shape[1] // 3] = 1.4
    return depth


def view(t: float, x: float, depth: np.ndarray) -> object:
    gt = wall_depth()
    return render_video.View(t, pose(x), pose(x), gt, np.ones(gt.shape, dtype=bool), depth)


# Colour ranges.

def test_colour_of_a_value_is_fixed_across_frames():
    low, high = 0.5, 4.0
    first = np.array([[1.0, 2.0], [3.0, np.nan]])
    second = np.array([[2.0, 0.6], [0.7, 0.8]])
    a, b = colorize(first, low, high, "turbo"), colorize(second, low, high, "turbo")
    assert np.array_equal(a[0, 1], b[0, 0])
    assert tuple(a[1, 1]) == pim_video.INVALID_RGB
    assert not np.array_equal(a[0, 0], a[0, 1])
    clipped = colorize(np.array([[-5.0, 0.5, 9.0, 4.0]]), low, high, "turbo")
    assert np.array_equal(clipped[0, 0], clipped[0, 1]) and np.array_equal(clipped[0, 2], clipped[0, 3])


def test_depth_range_is_one_window_percentile():
    samples = np.linspace(1.0, 2.0, 1001)
    low, high = render_video.depth_range(samples)
    assert low == pytest.approx(1.02) and high == pytest.approx(1.98)


# Disagreement.

def test_disagreement_is_zero_for_identical_frames():
    depth = wall_depth()
    error = render_video.disagreement(view(0.0, 0.0, depth), view(DT, 0.0, depth.copy()), K)
    scored = np.isfinite(error)
    assert scored.mean() > 0.9
    assert np.all(error[scored] == 0.0)


def test_disagreement_measures_warped_difference():
    depth = wall_depth()
    error = render_video.disagreement(view(0.0, 0.0, depth), view(DT, 0.0, depth * 1.02), K)
    assert np.nanmedian(error) == pytest.approx(0.02, abs=1e-5)


def test_disagreement_is_blank_beyond_50_ms_or_without_previous():
    depth = wall_depth()
    assert np.isnan(render_video.disagreement(view(0.0, 0.0, depth), view(0.0501, 0.0, depth), K)).all()
    assert np.isnan(render_video.disagreement(None, view(DT, 0.0, depth), K)).all()
    assert np.isfinite(render_video.disagreement(view(0.0, 0.0, depth), view(0.05, 0.0, depth), K)).any()


def test_replay_interval_picks_largest_difference():
    difference = np.full(100, 0.001)
    difference[10:15] = 0.004
    difference[60:70] = 0.01
    difference[[3, 80]] = np.nan
    assert render_video.replay_interval(difference, 10) == (60, 70)
    assert render_video.replay_interval(difference, 4) in {(60, 64), (61, 65), (62, 66), (63, 67), (64, 68),
                                                            (65, 69), (66, 70)}
    assert render_video.replay_interval(difference[:5], 10) == (0, 5)


# Maps.

def plane(z: float, n: int = 100) -> np.ndarray:
    u, v = np.meshgrid(np.linspace(-1, 1, n), np.linspace(-1, 1, n))
    return np.stack([u.ravel(), v.ravel(), np.full(u.size, z)], axis=1).astype(np.float32)


def test_ply_reads_back_with_point_count(tmp_path):
    points = plane(2.0, 11)
    colors = np.random.default_rng(0).integers(0, 256, points.shape, dtype=np.uint8)
    export_map.write_ply(tmp_path / "a.ply", points, colors)
    read_points, read_colors = export_map.read_ply(tmp_path / "a.ply")
    assert len(read_points) == 121
    assert np.array_equal(read_points, points) and np.array_equal(read_colors, colors)


def test_ghosting_is_larger_for_a_doubled_plane():
    gt = plane(2.0)
    planes = export_map.voxel_planes(gt, toward=np.zeros(3), voxel=0.1)
    single = export_map.ghosting(plane(2.0) + np.float32(0.001), planes)
    doubled = export_map.ghosting(np.concatenate([plane(2.0), plane(1.97)]), planes)
    assert single["covered_fraction"] > 0.9 and doubled["covered_fraction"] > 0.9
    assert single["std_m"] < 1e-3
    assert doubled["std_m"] == pytest.approx(0.015, abs=2e-3)
    assert doubled["mean_signed_m"] > 0


def test_orbit_renderer_draws_tiny_cloud():
    points = plane(0.0, 20)
    colors = np.full(points.shape, 200, dtype=np.uint8)
    orbit = {"centre": [0.0, 0.0, 0.0], "up": [0.0, 0.0, 1.0], "radius_m": 1.0, "start_direction": [1.0, 0.0, 0.0]}
    poses = render_orbit.orbit_poses(orbit, 4, 360.0, 35.0, 1.0, 60.0)
    size = (64, 48)
    for T in poses:
        frame = render_orbit.splat(points, colors, T, render_orbit.intrinsics(*size), size)
        assert frame.shape == (48, 64, 3)
        assert (frame == 200).all(axis=2).sum() > 50


def test_splat_keeps_nearest_point():
    points = np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    colors = np.array([[255, 0, 0], [0, 255, 0]], dtype=np.uint8)
    frame = render_orbit.splat(points, colors, np.eye(4), render_orbit.intrinsics(9, 9), (9, 9), 0.01, 1)
    assert tuple(frame[4, 4]) == (0, 255, 0)


def test_flythrough_continues_one_second_sideways():
    path = [pose(0.01 * k) for k in range(40)]
    poses, recorded = render_orbit.flythrough_poses(path, 30.0, 1.0, 0.2)
    assert recorded == 40 and len(poses) == 70
    assert poses[-1][0, 3] == pytest.approx(0.39 + 0.3 + 0.2)


# End to end on a tiny synthetic sequence.

def write_sequence(root: Path, frames: int = 8) -> tuple[Path, Path, Path]:
    name = "synthetic_seq"
    seq = root / "data" / name
    (seq / "rgb").mkdir(parents=True)
    (seq / "depth").mkdir()
    rgb_lines, depth_lines = [], []
    for k in range(frames):
        t = 100.0 + k * DT + (0.03 if k == 5 else 0.0)
        Image.fromarray(np.full((H, W, 3), 40 * k % 255, dtype=np.uint8)).save(seq / "rgb" / f"{t:.6f}.png")
        Image.fromarray((wall_depth() * 5000).astype(np.uint16)).save(seq / "depth" / f"{t + 0.005:.6f}.png")
        rgb_lines.append(f"{t:.6f} rgb/{t:.6f}.png")
        depth_lines.append(f"{t + 0.005:.6f} depth/{t + 0.005:.6f}.png")
    (seq / "rgb.txt").write_text("\n".join(rgb_lines) + "\n")
    (seq / "depth.txt").write_text("\n".join(depth_lines) + "\n")
    times = np.arange(99.9, 100.0 + frames * DT + 0.2, 0.01)
    (seq / "groundtruth.txt").write_text("".join(f"{t:.6f} {0.01 * (t - 100):.6f} 0 0 0 0 0 1\n" for t in times))
    config = root / "sequences.yaml"
    config.write_text(f"cameras:\n  synthetic: {{fx: {F}, fy: {F}, cx: {(W - 1) / 2}, cy: {(H - 1) / 2}}}\n"
                      f"sequences:\n  {name}:\n    camera: synthetic\n    window: {{start: 0, count: {frames}}}\n")
    render = root / "render"
    (render / name).mkdir(parents=True)
    for k in range(frames):
        gt = wall_depth()
        arrays = {"raw_aligned": (gt * (1 + 0.03 * (-1) ** k)).astype(np.float16)}
        if k >= 2:
            arrays["fused_aligned"] = (gt * (1 + 0.003 * (-1) ** k)).astype(np.float16)
        np.savez(render / name / f"{k:05d}.npz", **arrays, rel_d_raw=np.float64(3.0),
                 rel_d_fused=np.float64(0.5 if k >= 2 else np.nan), low_parallax=np.bool_(False),
                 n_sources=np.int64(min(k, 4)), has_fused=np.bool_(k >= 2), sequence=np.str_(name),
                 index=np.int64(k), timestamp=np.float64(100.0 + k * DT), scale=np.str_("joint_solved"),
                 poses=np.str_("oracle_gt"))
    return render, root / "data", config


def test_end_to_end_tiny(tmp_path, monkeypatch):
    monkeypatch.setattr(pim_video.shutil, "which", lambda _: None)
    render, data, config = write_sequence(tmp_path)
    out = tmp_path / "videos"
    record = render_video.main([str(render), str(data), str(config), "synthetic_seq", str(out),
                                "--replay-seconds", "0.1"])
    frames = sorted((out / "synthetic_seq").glob("*.png"))
    assert len(frames) == 8 + 3 * 3
    assert Image.open(frames[0]).size == (1920, 1080)
    assert (out / "synthetic_seq_poster.png").is_file() and (out / "synthetic_seq_preview.gif").is_file()
    stats = record["sequences"]["synthetic_seq"]
    assert stats["paired_frames"] >= 3
    assert stats["median_disagreement_fused_pct"] < stats["median_disagreement_raw_pct"]
    assert record["per_frame"][5]["median_raw_pct"] is None
    assert json.loads((out / "synthetic_seq_video.json").read_text())["poses"] == "oracle_gt"

    maps = tmp_path / "map"
    map_record = export_map.main([str(render), str(data), str(config), "synthetic_seq", str(maps), "--stride", "2",
                                  "--max-points", "500", "--voxel", "0.5"])
    assert map_record["points"] == {"gt": 500, "raw": 500, "fused": 500}
    assert len(export_map.read_ply(maps / "fused.ply")[0]) == 500
    assert map_record["ghosting"]["raw"]["std_m"] > map_record["ghosting"]["fused"]["std_m"]

    render_orbit.main([str(maps), str(tmp_path / "orbit"), "--seconds", "0.1"])
    orbit_frames = sorted((tmp_path / "orbit" / "orbit").glob("*.png"))
    assert len(orbit_frames) == 3 and (tmp_path / "orbit" / "orbit.gif").is_file()
    assert np.asarray(Image.open(orbit_frames[0]))[200:].std() > 0
    render_orbit.main([str(maps), str(tmp_path / "orbit"), "--flythrough", "--fps", "10"])
    assert len(list((tmp_path / "orbit" / "flythrough").glob("*.png"))) == 8 + 10
