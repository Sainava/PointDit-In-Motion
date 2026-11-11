from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_script",
                                                  Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ff = load_script("fusion_forecast")
P = (0.9, 0.8, 0.5)
FLOOR = (0.3, 0.5, 1.0)
SIGMA = (2.0, 2.1, 2.2)


@pytest.mark.parametrize("n", [1, 2, 3, 5, 9])
def test_independent_sources_give_inverse_sqrt_n(n: int) -> None:
    assert ff.std_ratio(range(n), np.eye(n), np.zeros(n)) == pytest.approx(1 / math.sqrt(n))


@pytest.mark.parametrize("n", [1, 2, 5, 9])
def test_fully_correlated_sources_give_one(n: int) -> None:
    assert ff.std_ratio(range(n), np.ones((n, n)), np.zeros(n)) == pytest.approx(1.0, abs=1e-12)


def test_self_only_gives_one() -> None:
    assert ff.forecast_ratio((0.0,), P, FLOOR, SIGMA) == 1.0
    assert ff.forecast_ratio((0.0,), P, None, None) == 1.0


def ratios_by_n(rho: float, warp: float, n_max: int = 12) -> list[float]:
    ratios = []
    for n in range(1, n_max):
        corr = np.full((n, n), rho)
        np.fill_diagonal(corr, 1.0)
        ratios.append(ff.std_ratio(range(n), corr, np.r_[0.0, np.full(n - 1, warp)]))
    return ratios


@pytest.mark.parametrize("rho,warp", [(0.0, 0.0), (0.5, 0.0), (0.7, 0.3), (0.9, 0.3), (0.5, 0.7), (1.0, 0.0)])
def test_ratio_non_increasing_in_n(rho: float, warp: float) -> None:
    assert warp ** 2 <= 1 - rho + 1e-12
    ratios = ratios_by_n(rho, warp)
    assert all(b <= a + 1e-12 for a, b in zip(ratios, ratios[1:]))


def test_warp_noise_above_decorrelation_makes_first_source_hurt() -> None:
    ratios = ratios_by_n(0.9, 0.5)
    assert ratios[1] == pytest.approx(math.sqrt(4.05 / 4))
    assert ratios[1] > ratios[0]


def test_interpolation_hits_anchors() -> None:
    for gap, p, floor, sigma in zip(ff.ANCHOR_GAPS, P, FLOOR, SIGMA):
        assert ff.rho_at(gap, P) == pytest.approx(p)
        assert ff.rho_at(-gap, P) == pytest.approx(p)
        assert ff.floor_at(gap, FLOOR) == pytest.approx(floor)
        assert ff.sigma_at(gap, SIGMA) == pytest.approx(sigma)


def test_interpolation_edges() -> None:
    assert ff.rho_at(0.0, P) == 1.0
    assert ff.rho_at(16.5, P) == pytest.approx(0.95)
    assert ff.floor_at(0.0, FLOOR) == 0.0
    assert ff.rho_at(400.0, P) == P[-1]
    expected = np.interp(math.log(200.0), np.log(ff.ANCHOR_GAPS), P)
    assert ff.rho_at(200.0, P) == pytest.approx(expected)
