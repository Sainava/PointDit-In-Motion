from __future__ import annotations

import math

import numpy as np
import pytest

from pim.analysis.covariates import (
    border_strip,
    camera_motion,
    depth_geometry,
    image_motion_px_per_s,
    near_border_fraction,
    relative_sharpness,
    sharpness,
)

RATE_DPS = 30.0
VELOCITY = np.array([0.3, -0.4, 0.0])
GT_HZ = 100.0


def constant_motion_groundtruth(duration_s: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    times = 1000.0 + np.arange(round(duration_s * GT_HZ) + 1) / GT_HZ
    half_angles = np.radians(RATE_DPS * (times - times[0])) / 2
    quaternions = np.stack([np.zeros_like(times), np.sin(half_angles), np.zeros_like(times), np.cos(half_angles)], 1)
    translations = (times - times[0])[:, None] * VELOCITY
    return times, np.hstack([translations, quaternions])


def textured_image(seed: int = 0, size: tuple[int, int] = (120, 160)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (*size, 3), dtype=np.uint8)


def box_blur(rgb: np.ndarray, radius: int = 2) -> np.ndarray:
    image = rgb.astype(np.float64)
    for axis in (0, 1):
        image = sum(np.roll(image, shift, axis=axis) for shift in range(-radius, radius + 1)) / (2 * radius + 1)
    return np.round(image).astype(np.uint8)


def test_constant_rotation_and_translation_give_their_rates() -> None:
    times, poses = constant_motion_groundtruth()

    for t in (times[50] + 0.003, times[100], times[150] - 0.0071):
        angular, linear = camera_motion(times, poses, t)
        assert angular == pytest.approx(RATE_DPS, rel=1e-6)
        assert linear == pytest.approx(np.linalg.norm(VELOCITY), rel=1e-6)


def test_motion_is_nan_when_the_window_leaves_the_groundtruth() -> None:
    times, poses = constant_motion_groundtruth()

    for t in (times[0] + 0.01, times[-1] - 0.01, times[0] - 1.0):
        assert all(math.isnan(v) for v in camera_motion(times, poses, t))
    assert not any(math.isnan(v) for v in camera_motion(times, poses, times[0] + 0.021))


def test_image_motion_adds_rotation_and_translation_terms() -> None:
    assert image_motion_px_per_s(math.degrees(1.0), 0.0, 500.0, 2.0) == pytest.approx(500.0)
    assert image_motion_px_per_s(0.0, 1.0, 500.0, 2.0) == pytest.approx(250.0)


def test_blurred_copy_is_less_sharp() -> None:
    image = textured_image()

    assert sharpness(box_blur(image)) < 0.2 * sharpness(image)
    assert sharpness(np.full((40, 40, 3), 128, np.uint8)) == 0.0


def test_relative_sharpness_is_one_for_constant_sharpness_sequence() -> None:
    base = textured_image()
    times = 1000.0 + np.arange(60) / 30.0
    values = np.array([sharpness(np.roll(base, 3 * k, axis=1)) for k in range(len(times))])

    relative = relative_sharpness(times, values)

    np.testing.assert_allclose(relative, 1.0, atol=0.02)


def test_relative_sharpness_uses_neighbours_within_the_radius() -> None:
    times = np.arange(10, dtype=np.float64)
    values = np.ones(10)
    values[5] = 4.0

    relative = relative_sharpness(times, values, radius_s=1.0)

    assert relative[5] == 4.0
    assert relative[4] == relative[6] == 1.0


def test_depth_percentiles_and_edge_density_of_a_step() -> None:
    depth = np.full((100, 200), 2.0, np.float32)
    depth[:, 100:] = 3.0

    stats = depth_geometry(depth, np.ones_like(depth, dtype=bool))

    assert stats["depth_p5"] == pytest.approx(2.0)
    assert stats["depth_p50"] == pytest.approx(2.5)
    assert stats["depth_p95"] == pytest.approx(3.0)
    assert stats["depth_range_ratio"] == pytest.approx(1.5)
    assert stats["edge_density"] == pytest.approx(12 / 200)
    assert stats["near_border_fraction"] == 0.0


def test_near_object_at_the_bottom_border() -> None:
    depth = np.full((100, 200), 2.0, np.float32)
    depth[90:, 50:150] = 1.0
    valid = np.ones_like(depth, dtype=bool)
    strip_pixels = 100 * 200 - 80 * 160

    assert border_strip(depth.shape).sum() == strip_pixels
    assert near_border_fraction(depth, valid) == pytest.approx(1000 / strip_pixels)

    valid[:10] = False
    assert near_border_fraction(depth, valid) == pytest.approx(1000 / (strip_pixels - 2000))


def test_object_nearer_but_above_the_threshold_does_not_count() -> None:
    depth = np.full((100, 200), 2.0, np.float32)
    depth[90:, 50:150] = 1.3

    assert near_border_fraction(depth, np.ones_like(depth, dtype=bool)) == 0.0
