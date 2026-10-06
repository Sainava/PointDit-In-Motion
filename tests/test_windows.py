from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pim.data.tum import Frame, TUMSequence, Window, apply_window
from pim.data.windows import WindowChoice, count_holes, hole_gaps, select_window


def sequence_with(n: int, window: Window) -> TUMSequence:
    frames = [Frame(i, float(i), Path(f"rgb/{i}.png"), Path(f"depth/{i}.png"), 0.0, np.eye(4))
              for i in range(n)]
    return TUMSequence("seq", Path("."), np.eye(3), window, frames)


def timestamps_with_holes(n: int, hole_after: list[int], period: float = 0.033) -> np.ndarray:
    gaps = np.full(n - 1, period)
    gaps[hole_after] = 2 * period
    return 1305031000.0 + np.concatenate([[0.0], np.cumsum(gaps)])


def test_null_window_keeps_every_frame() -> None:
    sequence = sequence_with(5, Window(None, None))

    assert [f.index for f in apply_window(sequence)] == [0, 1, 2, 3, 4]


def test_window_is_a_contiguous_slice_of_kept_frames() -> None:
    sequence = sequence_with(10, Window(3, 4))

    assert [f.index for f in apply_window(sequence)] == [3, 4, 5, 6]


def test_window_may_end_on_the_last_frame() -> None:
    sequence = sequence_with(10, Window(7, 3))

    assert [f.index for f in apply_window(sequence)] == [7, 8, 9]


@pytest.mark.parametrize("window", [
    Window(8, 3), Window(-1, 3), Window(0, 0), Window(2, None), Window(None, 2), Window(1.0, 2), Window(True, 2),
])
def test_invalid_windows_are_rejected(window: Window) -> None:
    with pytest.raises(ValueError):
        apply_window(sequence_with(10, window))


def test_holes_are_gaps_above_one_and_a_half_median_gaps() -> None:
    times = 100.0 + np.cumsum([0.0, 0.033, 0.033, 0.033, 0.049, 0.051, 0.033])

    assert hole_gaps(times).tolist() == [False, False, False, False, True, False]
    assert count_holes(times) == 1


def test_hole_gaps_rejects_unsorted_or_short_lists() -> None:
    with pytest.raises(ValueError):
        hole_gaps([1.0])
    with pytest.raises(ValueError):
        hole_gaps([1.0, 1.0, 2.0])


def test_window_selection_finds_the_hole_free_stretch() -> None:
    times = timestamps_with_holes(20, hole_after=[2, 4, 15])

    choice = select_window(times, 10)

    # A 10-frame window spans 9 gaps; starts 5 (gaps 5..13) and 6 (gaps 6..14) are hole-free.
    assert (choice.start, choice.count, choice.holes) == (5, 10, 0)


def test_window_selection_ties_go_to_the_earliest_start() -> None:
    times = timestamps_with_holes(20, hole_after=[0, 9, 18])

    choice = select_window(times, 9)

    assert (choice.start, choice.holes) == (1, 0)


def test_window_selection_minimises_holes_when_none_is_clean() -> None:
    times = timestamps_with_holes(12, hole_after=[1, 2, 3, 7, 9])

    choice = select_window(times, 6)

    assert (choice.start, choice.holes) == (4, 1)


def test_hole_at_the_window_boundary_is_outside_the_window() -> None:
    times = timestamps_with_holes(6, hole_after=[2])

    assert select_window(times, 3).holes == 0
    assert select_window(times, 3).start == 0
    assert select_window(times, 4).holes == 1


def test_whole_sequence_window_and_oversized_window() -> None:
    times = timestamps_with_holes(8, hole_after=[3])

    assert select_window(times, 8) == WindowChoice(0, 8, 1)
    with pytest.raises(ValueError):
        select_window(times, 9)
