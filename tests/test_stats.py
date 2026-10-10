from __future__ import annotations

import math

import numpy as np
import pytest

from pim.analysis.stats import (
    average_ranks, bootstrap, detect_spikes, fast_slow_median_difference, moving_block_indices, neighbour_medians,
    partial_spearman, percentile_ci, pooled_partial_spearman, spearman,
)

X = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
Y = np.array([2.0, 1.0, 4.0, 3.0, 5.0])
Z = np.array([1.0, 3.0, 2.0, 5.0, 4.0])


def test_average_ranks_share_the_mean_rank_on_ties() -> None:
    assert average_ranks(np.array([10.0, 20.0, 20.0, 30.0, 20.0])).tolist() == [1.0, 3.0, 3.0, 5.0, 3.0]


def test_spearman_without_ties_matches_the_d_squared_formula() -> None:
    # sum d^2 = 4, so rho = 1 - 6 * 4 / (5 * 24)
    assert spearman(X, Y) == pytest.approx(0.8)


def test_spearman_with_ties() -> None:
    # ranks x = [1, 2.5, 2.5, 4], y = [1, 3, 2, 4]: rho = 4.5 / sqrt(4.5 * 5)
    assert spearman(np.array([1.0, 2.0, 2.0, 3.0]), np.array([1.0, 3.0, 2.0, 4.0])) == pytest.approx(3 / math.sqrt(10))


def test_spearman_is_invariant_to_monotone_transforms() -> None:
    assert spearman(np.exp(X), Y ** 3) == pytest.approx(spearman(X, Y))


def test_partial_spearman_with_one_control_matches_the_closed_form() -> None:
    # rho_xy = 0.8, rho_xz = 0.8 (sum d^2 = 4), rho_yz = 0.3 (sum d^2 = 14)
    expected = (0.8 - 0.8 * 0.3) / math.sqrt((1 - 0.8 ** 2) * (1 - 0.3 ** 2))

    assert partial_spearman(X, Y, Z[:, None]) == pytest.approx(expected)


def test_partial_spearman_removes_a_shared_driver() -> None:
    rng = np.random.default_rng(0)
    driver = rng.standard_normal(500)
    x = driver + 0.3 * rng.standard_normal(500)
    y = driver + 0.3 * rng.standard_normal(500)

    assert spearman(x, y) > 0.8
    assert abs(partial_spearman(x, y, driver[:, None])) < 0.15


def test_pooled_ranks_within_groups_so_level_differences_do_not_count() -> None:
    x_a, y_a = np.arange(5.0), np.arange(5.0)
    x_b, y_b = np.arange(5.0) + 100.0, np.arange(5.0) - 100.0
    empty = np.empty((5, 0))

    assert pooled_partial_spearman([(x_a, y_a, empty), (x_b, y_b, empty)]) == pytest.approx(1.0)
    assert spearman(np.r_[x_a, x_b], np.r_[y_a, y_b]) < 0.0


def test_pooled_over_one_group_equals_spearman_and_drops_nan_rows() -> None:
    x, y = np.r_[X, np.nan], np.r_[Y, 7.0]

    assert pooled_partial_spearman([(x, y, np.empty((6, 0)))]) == pytest.approx(spearman(X, Y))


def test_moving_blocks_are_contiguous_and_truncated_to_n() -> None:
    indices = moving_block_indices(23, 5, 50, np.random.default_rng(0))

    assert indices.shape == (50, 23)
    assert indices.min() >= 0 and indices.max() <= 22
    blocks = indices[:, :20].reshape(50, 4, 5)
    assert np.all(np.diff(blocks, axis=2) == 1)


def test_block_length_must_fit_the_series() -> None:
    with pytest.raises(ValueError, match="block length must be in"):
        moving_block_indices(10, 11, 5, np.random.default_rng(0))


def ar1(n: int, phi: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    series = np.zeros(n)
    for t in range(1, n):
        series[t] = phi * series[t - 1] + rng.standard_normal()
    return series


def mean_ci(series: np.ndarray, block_length: int, seed: int = 0) -> tuple[float, float]:
    indices = moving_block_indices(len(series), block_length, 2000, np.random.default_rng(seed))
    return percentile_ci(bootstrap(lambda idx: float(series[idx[0]].mean()), [indices]))


def test_block_bootstrap_ci_is_wider_than_iid_on_an_autocorrelated_series() -> None:
    series = ar1(300, 0.95, seed=1)

    iid_low, iid_high = mean_ci(series, block_length=1)
    block_low, block_high = mean_ci(series, block_length=15)

    assert block_high - block_low > 2.0 * (iid_high - iid_low)


def test_block_bootstrap_is_reproducible_with_a_fixed_seed() -> None:
    series = ar1(300, 0.95, seed=1)

    assert mean_ci(series, 15, seed=7) == mean_ci(series, 15, seed=7)
    assert mean_ci(series, 15, seed=7) != mean_ci(series, 15, seed=8)


def test_bootstrap_resamples_groups_jointly() -> None:
    a = moving_block_indices(10, 2, 30, np.random.default_rng(0))
    b = moving_block_indices(8, 2, 30, np.random.default_rng(1))

    values = bootstrap(lambda idx: float(len(idx[0]) + len(idx[1])), [a, b])

    assert values.tolist() == [18.0] * 30
    with pytest.raises(ValueError, match="same number of resamples"):
        bootstrap(lambda idx: 0.0, [a, b[:5]])


def test_percentile_ci_ignores_nan_resamples() -> None:
    assert percentile_ci(np.r_[np.arange(101.0), np.nan]) == pytest.approx((2.5, 97.5))


def test_fast_slow_median_difference() -> None:
    speed = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, np.nan])
    value = np.array([1.0, 3.0, 9.0, 9.0, 5.0, 7.0, 100.0])

    # slowest third (speed 1, 2): median 2; fastest third (speed 5, 6): median 6
    assert fast_slow_median_difference(speed, value) == 4.0


def test_neighbour_medians_exclude_the_frame_and_are_one_sided_at_the_edges() -> None:
    values = np.array([100.0, 1.0, 2.0, 3.0, 4.0])

    medians = neighbour_medians(values, radius=2)

    assert medians.tolist() == [1.5, 3.0, 3.5, 2.0, 2.5]


def test_spikes_including_window_edges() -> None:
    values = np.full(40, 3.0)
    values[[0, 20, 39]] = 6.0
    values[10] = 4.4
    values[30] = 4.6

    spikes = detect_spikes(values, radius=5, min_excess=1.5, min_relative_excess=0.5)

    assert np.flatnonzero(spikes).tolist() == [0, 20, 30, 39]


def test_relative_threshold_applies_at_high_error() -> None:
    values = np.full(30, 10.0)
    values[10] = 14.9
    values[20] = 15.1

    spikes = detect_spikes(values, radius=5, min_excess=1.5, min_relative_excess=0.5)

    assert np.flatnonzero(spikes).tolist() == [20]


def test_nan_frames_are_never_spikes_and_are_left_out_of_neighbourhoods() -> None:
    values = np.full(20, 3.0)
    values[[4, 6]] = np.nan
    values[5] = 6.0

    spikes = detect_spikes(values, radius=1, min_excess=1.5, min_relative_excess=0.5)

    assert np.flatnonzero(spikes).tolist() == []
    assert np.flatnonzero(detect_spikes(values, radius=2, min_excess=1.5, min_relative_excess=0.5)).tolist() == [5]
