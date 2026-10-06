from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pim.data.tum import (
    ListedImage,
    associate,
    build_frames,
    interpolate_pose,
    load_depth,
    quaternion_to_rotation,
    read_groundtruth,
    read_rgb_list,
    slerp,
)

IDENTITY = np.array([0.0, 0.0, 0.0, 1.0])
Z_90 = np.array([0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)])


def rotation_z(degrees: float) -> np.ndarray:
    c, s = np.cos(np.radians(degrees)), np.sin(np.radians(degrees))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def listed(*timestamps: float) -> list[ListedImage]:
    return [ListedImage(t, Path(f"{t:.3f}.png")) for t in timestamps]


def write_groundtruth(folder: Path, rows: list[tuple[float, ...]]) -> None:
    lines = ["# ground truth trajectory", "# timestamp tx ty tz qx qy qz qw"]
    lines += [" ".join(str(v) for v in row) for row in rows]
    (folder / "groundtruth.txt").write_text("\n".join(lines) + "\n")


def test_comment_lines_are_ignored(tmp_path: Path) -> None:
    (tmp_path / "rgb.txt").write_text(
        "# color images\n# timestamp filename\n1.0 rgb/1.png\n\n# trailing\n2.0 rgb/2.png\n"
    )
    write_groundtruth(tmp_path, [(1.0, 0, 0, 0, 0, 0, 0, 1), (2.0, 1, 0, 0, 0, 0, 0, 1)])

    frames = read_rgb_list(tmp_path)
    times, poses = read_groundtruth(tmp_path)

    assert [f.timestamp for f in frames] == [1.0, 2.0]
    assert frames[0].path == tmp_path / "rgb" / "1.png"
    assert times.tolist() == [1.0, 2.0]
    assert poses.shape == (2, 7)


def test_association_prefers_closest_pair() -> None:
    rgb = listed(10.000, 10.010)
    depth = listed(10.012)

    pairs = associate(rgb, depth)

    assert [(r.timestamp, d.timestamp) for r, d in pairs] == [(10.010, 10.012)]


def test_association_is_one_to_one_and_falls_back_to_next_closest() -> None:
    rgb = listed(10.000, 10.010)
    depth = listed(10.008, 10.015)

    pairs = associate(rgb, depth)

    assert [(r.timestamp, d.timestamp) for r, d in pairs] == [(10.000, 10.015), (10.010, 10.008)]


def test_association_respects_tolerance_and_sorts_by_rgb_time() -> None:
    rgb = listed(20.000, 10.000, 30.000)
    depth = listed(10.019, 20.021, 30.000)

    pairs = associate(rgb, depth)

    assert [(r.timestamp, d.timestamp) for r, d in pairs] == [(10.000, 10.019), (30.000, 30.000)]


def test_depth_scale_and_missing_values(tmp_path: Path) -> None:
    path = tmp_path / "depth.png"
    Image.fromarray(np.array([[5000, 0], [10000, 2500]], dtype=np.uint16)).save(path)

    depth, valid = load_depth(path)

    assert depth.dtype == np.float32 and valid.dtype == bool
    np.testing.assert_allclose(depth, [[1.0, 0.0], [2.0, 0.5]])
    assert valid.tolist() == [[True, False], [True, True]]


def test_quaternion_to_rotation() -> None:
    np.testing.assert_allclose(quaternion_to_rotation(IDENTITY), np.eye(3), atol=1e-12)
    np.testing.assert_allclose(quaternion_to_rotation(Z_90), rotation_z(90), atol=1e-12)


def test_slerp_midpoint_and_sign_ambiguity() -> None:
    midpoint = quaternion_to_rotation(slerp(IDENTITY, Z_90, 0.5))
    flipped = quaternion_to_rotation(slerp(IDENTITY, -Z_90, 0.5))

    np.testing.assert_allclose(midpoint, rotation_z(45), atol=1e-12)
    np.testing.assert_allclose(flipped, rotation_z(45), atol=1e-12)


def test_slerp_nearly_parallel_is_finite_and_unit() -> None:
    q = slerp(IDENTITY, np.array([0.0, 0.0, 1e-6, 1.0]), 0.5)

    assert np.isfinite(q).all()
    assert np.linalg.norm(q) == pytest.approx(1.0)


def test_interpolate_pose_translation_and_rotation() -> None:
    times = np.array([0.0, 0.04])
    poses = np.array([[0.0, 0.0, 0.0, *IDENTITY], [2.0, 4.0, 0.0, *Z_90]])

    pose = interpolate_pose(times, poses, 0.02)

    assert pose.shape == (4, 4) and pose.dtype == np.float64
    np.testing.assert_allclose(pose[:3, 3], [1.0, 2.0, 0.0])
    np.testing.assert_allclose(pose[:3, :3], rotation_z(45), atol=1e-12)
    np.testing.assert_allclose(pose[3], [0.0, 0.0, 0.0, 1.0])


def test_frames_outside_gt_range_or_across_gap_are_dropped() -> None:
    times = np.array([1.00, 1.04, 1.20, 1.24])
    poses = np.tile(np.array([0.0, 0.0, 0.0, *IDENTITY]), (4, 1))
    rgb = listed(0.99, 1.02, 1.10, 1.22, 1.25)
    pairs = list(zip(rgb, listed(0.99, 1.02, 1.10, 1.22, 1.25)))

    frames = build_frames(pairs, times, poses)

    assert [f.timestamp for f in frames] == [1.02, 1.22]
    assert [f.index for f in frames] == [0, 1]


def test_gap_of_exactly_max_is_kept_despite_float_error() -> None:
    times = np.array([1305031467.4592, 1305031467.5092])
    poses = np.tile(np.array([0.0, 0.0, 0.0, *IDENTITY]), (2, 1))

    assert interpolate_pose(times, poses, 1305031467.48) is not None
