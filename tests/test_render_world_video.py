from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pim import video as pim_video

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_script", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


world = load_script("render_world_video")

H, W = 48, 64
F = 50.0
DT = 1 / 30
FRAMES = 14
NAME = "synthetic_seq"


def wall_depth() -> np.ndarray:
    depth = np.full((H, W), 2.0, dtype=np.float32)
    depth[: H // 3, : W // 3] = 1.4
    return depth


def write_sequence(root: Path) -> tuple[Path, Path, Path]:
    """A camera sliding 1.5 m/s sideways past a wall with a box, fused depth from frame 2 on."""
    seq = root / "data" / NAME
    (seq / "rgb").mkdir(parents=True)
    (seq / "depth").mkdir()
    rgb_lines, depth_lines = [], []
    for k in range(FRAMES):
        t = 100.0 + k * DT
        Image.fromarray(np.full((H, W, 3), 15 * k % 255, dtype=np.uint8)).save(seq / "rgb" / f"{t:.6f}.png")
        Image.fromarray((wall_depth() * 5000).astype(np.uint16)).save(seq / "depth" / f"{t + 0.005:.6f}.png")
        rgb_lines.append(f"{t:.6f} rgb/{t:.6f}.png")
        depth_lines.append(f"{t + 0.005:.6f} depth/{t + 0.005:.6f}.png")
    (seq / "rgb.txt").write_text("\n".join(rgb_lines) + "\n")
    (seq / "depth.txt").write_text("\n".join(depth_lines) + "\n")
    times = np.arange(99.9, 100.0 + FRAMES * DT + 0.2, 0.01)
    (seq / "groundtruth.txt").write_text("".join(f"{t:.6f} {1.5 * (t - 100):.6f} 0 0 0 0 0 1\n" for t in times))
    config = root / "sequences.yaml"
    config.write_text(f"cameras:\n  synthetic: {{fx: {F}, fy: {F}, cx: {(W - 1) / 2}, cy: {(H - 1) / 2}}}\n"
                      f"sequences:\n  {NAME}:\n    camera: synthetic\n    window: {{start: 0, count: {FRAMES}}}\n")
    render = root / "render"
    (render / NAME).mkdir(parents=True)
    for k in range(FRAMES):
        gt = wall_depth()
        arrays = {"raw_aligned": (gt * (1 + 0.03 * (-1) ** k)).astype(np.float16)}
        if k >= 2:
            arrays["fused_aligned"] = (gt * (1 + 0.003 * (-1) ** k)).astype(np.float16)
        np.savez(render / NAME / f"{k:05d}.npz", **arrays, rel_d_raw=np.float64(3.0),
                 rel_d_fused=np.float64(0.5 if k >= 2 else np.nan), low_parallax=np.bool_(False),
                 n_sources=np.int64(min(k, 4)), has_fused=np.bool_(k >= 2), sequence=np.str_(NAME),
                 index=np.int64(k), timestamp=np.float64(100.0 + k * DT), scale=np.str_("joint_solved"),
                 poses=np.str_("oracle_gt"))
    return render, root / "data", config


@pytest.fixture()
def sequence(tmp_path, monkeypatch):
    monkeypatch.setattr(pim_video.shutil, "which", lambda _: None)
    return write_sequence(tmp_path)


def run(sequence: tuple[Path, Path, Path], out: Path, *extra: str) -> dict:
    render, data, config = sequence
    return world.main([str(render), str(data), str(config), NAME, str(out), "--stride", "1", "--frame-points", "300",
                       "--voxel", "0.5", *extra])


def frames_of(out: Path, mode: str) -> list[Path]:
    return sorted((out / f"{NAME}_world_{mode}").glob("*.png"))


def store_points(position: int, n: int) -> tuple[dict[str, np.ndarray], np.ndarray]:
    points = np.full((n, 3), float(position), dtype=np.float32)
    return {"raw": points, "fused": points + 1}, np.full((n, 3), position, dtype=np.uint8)


def test_store_keeps_newest_frame_and_thins_oldest_under_cap():
    store = world.PointStore(("raw", "fused"), 1000, np.random.default_rng(0))
    for position in range(30):
        store.add(position, *store_points(position, 300))
        assert len(store) <= 1000
        assert all(len(p) == len(store) for p in store.points.values())
        assert (store.frame == position).sum() == 300
    counts = np.bincount(store.frame, minlength=30)
    assert counts[:29].sum() == 700 and counts[0] < counts[28]
    assert np.all(store.points["raw"][:, 0] == store.frame) and np.all(store.colors[:, 0] == store.frame)
    store.add(30, *store_points(30, 5000))
    assert len(store) == 1000 and (store.frame == 30).all()


def test_store_drop_before_removes_old_frames():
    store = world.PointStore(("raw",), 10_000, np.random.default_rng(0))
    for position in range(10):
        store.add(position, {"raw": np.zeros((50, 3), dtype=np.float32)}, np.zeros((50, 3), dtype=np.uint8))
        store.drop_before(position - 3 + 1)
        assert store.frame.min() == max(0, position - 2) and len(store) == 50 * min(position + 1, 3)


def test_accumulate_map_grows_monotonically(sequence, tmp_path):
    out = tmp_path / "acc"
    record = run(sequence, out, "--mode", "accumulate")
    shown = record["points"]["shown_per_frame"]
    assert len(shown) == FRAMES - 2 and len(frames_of(out, "accumulate")) == FRAMES - 2
    assert all(b > a for a, b in zip(shown, shown[1:]))
    assert shown == record["points"]["accumulated_per_frame"]
    assert record["ghosting"]["raw"]["std_m"] > record["ghosting"]["fused"]["std_m"]
    frame = np.asarray(Image.open(frames_of(out, "accumulate")[-1]))
    assert frame.shape == (1080, 1920, 3) and frame[world.HEADER_PX + world.LABEL_PX:-world.FOOTER_PX].std() > 0
    assert Image.open(out / f"{NAME}_world_accumulate.gif").size == (960, 540)
    capped = run(sequence, tmp_path / "cap", "--max-points", "1000")["points"]["shown_per_frame"]
    assert all(b >= a for a, b in zip(capped, capped[1:])) and capped[-1] == 1000


def test_sliding_mode_drops_old_points(sequence, tmp_path):
    record = run(sequence, tmp_path / "slide", "--mode", "sliding", "--window", "3")
    shown, total = record["points"]["shown_per_frame"], record["points"]["accumulated_per_frame"]
    assert max(shown) <= 3 * 300 and shown[-1] < total[-1]
    assert shown[-1] == shown[-2] and total[-1] > total[-2]
    assert record["points"]["final_accumulated"]["fused"] == total[-1]


def test_panels_share_one_camera(sequence, tmp_path, monkeypatch):
    calls = []
    original = world.splat

    def recording(points, colors, T, K, size, point_size):
        calls.append((T, K, size))
        return original(points, colors, T, K, size, point_size)

    monkeypatch.setattr(world, "splat", recording)
    record = run(sequence, tmp_path / "gt", "--with-gt", "--max-frames", "6")
    frames = len(record["frames_used"])
    assert record["settings"]["panels"] == ["raw", "fused", "gt"] and len(calls) == 3 * frames
    for k in range(frames):
        group = calls[3 * k:3 * k + 3]
        assert all(c[0] is group[0][0] and np.array_equal(c[1], group[0][1]) and c[2] == (640, group[0][2][1])
                   for c in group)
    assert not np.allclose(calls[0][0], calls[-1][0])
    assert all(np.array_equal(c[1], calls[0][1]) for c in calls)


def test_framing_uses_the_whole_clip(sequence, tmp_path):
    full = run(sequence, tmp_path / "a")["framing"]
    short = run(sequence, tmp_path / "b", "--max-frames", "5")["framing"]
    assert full["centre"][0] > short["centre"][0] + 0.1


def test_frustum_is_visible_from_behind_and_dropped_when_behind_view():
    K = np.array([[F, 0.0, (W - 1) / 2], [0.0, F, (H - 1) / 2], [0.0, 0.0, 1.0]])
    view = np.eye(4)
    view[2, 3] = -2.0
    segments = world.frustum_segments(np.eye(4), K, (W, H), view, K, 0.3)
    assert len(segments) == 8
    behind = np.eye(4)
    behind[2, 3] = -2.0
    assert world.frustum_segments(behind, K, (W, H), np.eye(4), K, 0.3) == []
