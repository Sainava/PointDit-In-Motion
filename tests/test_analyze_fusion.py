from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_script", ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


af = load_script("analyze_fusion")
MEAN = af.STATISTICS["mean_change_pct"]


def ar1(n: int, phi: float, rng: np.random.Generator) -> np.ndarray:
    x = np.zeros(n)
    noise = rng.normal(size=n) * np.sqrt(1 - phi ** 2)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + noise[i]
    return x


def test_block_interval_wider_for_autocorrelated_data() -> None:
    rng = np.random.default_rng(1)
    independent, correlated = rng.normal(size=400), ar1(400, 0.9, rng)
    for block in af.BLOCKS:
        lo_i, hi_i = af.block_intervals([independent], MEAN, blocks=[block])[block]
        lo_c, hi_c = af.block_intervals([correlated], MEAN, blocks=[block])[block]
        assert hi_c - lo_c > 1.5 * (hi_i - lo_i)


def test_robust_rule_needs_every_block_length() -> None:
    rng = np.random.default_rng(2)
    shifted = rng.normal(size=100) - 2.0
    assert af.robust(af.block_intervals([shifted], MEAN))
    assert not af.robust(af.block_intervals([rng.normal(size=100)], MEAN))
    assert not af.robust({8: (-2.0, -1.0), 15: (-2.0, -0.5), 30: (-2.0, 0.1)})
    assert af.robust({8: (0.2, 1.0), 15: (0.1, 1.2), 30: (0.05, 1.5)})


def test_pooled_point_is_mean_of_group_means() -> None:
    a, b = np.full(50, -2.0), np.full(80, -4.0)
    row = af.describe([(1 + a / 100, np.ones(50)), (1 + b / 100, np.ones(80))])
    assert row["mean_change_pct"] == pytest.approx(-3.0)
    assert row["mean_change_pct_ci_low"] == pytest.approx(-3.0)


def test_forecast_mapping_finds_every_forecast_config() -> None:
    forecast = af.read_forecast()
    with open(str(af.SOURCES["oracle"]).format(next(iter(af.SEQUENCES))), newline="") as f:
        measured = {row["config"] for row in csv.DictReader(f)}
    names = {row["config"] for row in forecast}
    mapping = af.forecast_mapping(names, measured)
    assert set(mapping) == names
    assert mapping["self_only"] == af.BASELINE
    assert all(mapping[c] == c for c in names - {"self_only"})
    with pytest.raises(KeyError):
        af.forecast_mapping(["causal_adjacent_w9"], measured)
