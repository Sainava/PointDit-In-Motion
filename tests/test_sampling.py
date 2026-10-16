from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from pim.data.tum import Frame
from pim.sampling import (
    comparisons, disagreement, median_estimate, median_ratio_estimate, prediction_difference,
)
from pim.subsets import FrameRange, check_run_covers, load_subsets, restrict_frames, subset_sequences


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_script",
                                                  Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


check_determinism = load_script("check_determinism")

# Small frames keep upstream's quadratic alignment cheap: at most H * W pixels per fit.
H, W = 40, 52


def test_load_subsets_reads_inclusive_ranges(tmp_path: Path) -> None:
    path = tmp_path / "subsets.yaml"
    path.write_text("subsets:\n  a: {first: 2, last: 4}\n  b: {first: 0, last: 0}\n")

    subsets = load_subsets(path)

    assert subsets == {"a": FrameRange(2, 4), "b": FrameRange(0, 0)}
    assert [subsets["a"].contains(i) for i in range(1, 6)] == [False, True, True, True, False]


@pytest.mark.parametrize("body", ["subsets: {}", "subsets:\n  a: {first: 4, last: 2}", "subsets:\n  a: {first: 1}",
                                  "subsets:\n  a: {first: 1, last: 2.5}", "frames: {}"])
def test_load_subsets_rejects_malformed_files(tmp_path: Path, body: str) -> None:
    path = tmp_path / "subsets.yaml"
    path.write_text(body + "\n")
    with pytest.raises(ValueError):
        load_subsets(path)


def test_the_repository_subsets_lie_inside_the_windows() -> None:
    root = Path(__file__).resolve().parents[1] / "configs"
    import yaml

    windows = {n: s["window"] for n, s in yaml.safe_load((root / "sequences.yaml").read_text())["sequences"].items()}
    subsets = load_subsets(root / "sampling_subsets.yaml")

    assert subsets == {"rgbd_dataset_freiburg1_360": FrameRange(200, 299),
                       "rgbd_dataset_freiburg3_long_office_household": FrameRange(740, 839)}
    for name, r in subsets.items():
        assert windows[name]["start"] <= r.first and r.last < windows[name]["start"] + windows[name]["count"]


def frames(indices: range) -> list[Frame]:
    return [Frame(i, 1000.0 + i / 30, Path(f"{i}.png"), Path(f"{i}.png"), 0.0, np.eye(4)) for i in indices]


def test_restrict_frames_and_sequence_selection() -> None:
    window = frames(range(10, 20))

    assert [f.index for f in restrict_frames("s", window, FrameRange(12, 14))] == [12, 13, 14]
    with pytest.raises(ValueError, match="not inside the window"):
        restrict_frames("s", window, FrameRange(15, 20))
    assert subset_sequences({"a": FrameRange(0, 1), "b": FrameRange(0, 1)}, None) == ["a", "b"]
    with pytest.raises(ValueError, match="have no subset"):
        subset_sequences({"a": FrameRange(0, 1)}, ["a", "c"])


def test_check_run_covers() -> None:
    check_run_covers({}, None)
    check_run_covers({}, {"a": FrameRange(0, 5)})
    run = {"subsets": {"a": {"first": 2, "last": 6}}}
    check_run_covers(run, {"a": FrameRange(3, 6)})
    with pytest.raises(ValueError, match="pass the same --subsets"):
        check_run_covers(run, None)
    with pytest.raises(ValueError, match="not inside the predicted subset"):
        check_run_covers(run, {"a": FrameRange(1, 6)})
    with pytest.raises(ValueError, match="not inside the predicted subset"):
        check_run_covers(run, {"b": FrameRange(2, 6)})


def test_comparisons_pair_every_seed_then_zeros_against_each() -> None:
    assert comparisons(["s0", "s1", "s2"], "z") == [
        ("s0", "s1"), ("s0", "s2"), ("s1", "s2"), ("z", "s0"), ("z", "s1"), ("z", "s2")]


def scene_depth() -> np.ndarray:
    v, u = np.indices((H, W))
    depth = 2.0 + 0.004 * v + 0.003 * u
    depth[14:20, 20:28] = 1.5
    return depth


def test_disagreement_is_zero_for_identical_and_positive_for_perturbed_predictions() -> None:
    depth = scene_depth()
    mask = np.ones((H, W), bool)
    region = depth.copy()
    region[:, : W // 3] *= 1.1
    noisy = depth * (1.0 + 0.02 * np.random.default_rng(1).standard_normal((H, W)))

    same = disagreement(depth, depth, mask)

    assert same.pixels == H * W
    assert same.median < 1e-3 and same.p90 < 1e-3
    # The L1 fit matches the unperturbed two thirds exactly, so only p90 sees the region.
    assert disagreement(depth, region, mask).p90 > 5.0
    assert disagreement(depth, noisy, mask).median > 0.5


def test_disagreement_is_invariant_to_an_affine_change_of_the_aligned_prediction() -> None:
    depth = scene_depth()
    mask = np.ones((H, W), bool)
    rng = np.random.default_rng(0)
    other = depth * (1.0 + 0.02 * rng.standard_normal((H, W)))

    direct = disagreement(depth, other, mask)
    affine = disagreement(depth, 0.3 * other - 0.4, mask)

    assert direct.median > 0.5
    # In float32 the exact L1 solver can pick a different, near-tied anchor pixel after the affine change.
    assert affine.median == pytest.approx(direct.median, rel=1e-2)
    assert affine.p90 == pytest.approx(direct.p90, rel=1e-2)
    assert disagreement(depth, -2.0 * depth + 9.0, mask).median < 1e-3


def test_disagreement_uses_only_masked_pixels_with_positive_reference_and_needs_enough_of_them() -> None:
    depth = scene_depth()
    corrupted = depth.copy()
    corrupted[:5] = 50.0
    mask = np.ones((H, W), bool)
    mask[:5] = False
    reference = depth.copy()
    reference[5:8] = 0.0

    result = disagreement(reference, corrupted, mask)
    assert result.pixels == H * W - 8 * W
    assert result.median < 1e-3
    small = np.zeros((H, W), bool)
    small[:10] = True
    assert math.isnan(disagreement(depth, depth, small).median)


def write_prediction(path: Path, points: np.ndarray, rgb_file: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, points=points.astype(np.float16), valid=np.ones(points.shape[:2], bool),
                        rgb_file=np.str_(rgb_file), latency_ms=np.float64(1.0), peak_mem_gb=np.float64(1.0))


def write_run(run_dir: Path, init: str = "zeros", subsets: dict | None = None) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    run = {"name": run_dir.name, "config": {"name": run_dir.name, "inference": {"init": init, "seed": None}},
           "sequences": {}}
    if subsets is not None:
        run["subsets"] = subsets
    (run_dir / "run.json").write_text(json.dumps(run))


SUBSETS = {"seq": FrameRange(3, 5)}


def point_map(index: int) -> np.ndarray:
    return np.random.default_rng(index).uniform(-1.5, 1.8, (8, 10, 3))


@pytest.fixture
def runs(tmp_path: Path) -> tuple[Path, Path]:
    rerun, reference = tmp_path / "zeros_rerun", tmp_path / "baseline"
    write_run(rerun, subsets={"seq": {"first": 3, "last": 5}})
    write_run(reference)
    for index in range(0, 8):
        write_prediction(reference / "seq" / f"{index:05d}.npz", point_map(index), f"rgb/{index}.png")
    for index in range(3, 6):
        write_prediction(rerun / "seq" / f"{index:05d}.npz", point_map(index), f"rgb/{index}.png")
    return rerun, reference


def test_determinism_passes_on_identical_predictions(runs: tuple[Path, Path]) -> None:
    rows = check_determinism.compare(*runs, SUBSETS, 2e-3)

    assert [r["frame_index"] for r in rows] == [3, 4, 5]
    assert all(r["max_abs_diff"] == 0.0 and r["passed"] for r in rows)
    assert check_determinism.report(rows, 2e-3)


@pytest.mark.parametrize("shift, passed", [(5e-4, True), (4e-3, False)])
def test_determinism_tolerance_on_a_perturbed_prediction(runs: tuple[Path, Path], shift: float, passed: bool) -> None:
    rerun, reference = runs
    points = point_map(4)
    points[2, 3, 1] += shift
    write_prediction(rerun / "seq" / "00004.npz", points, "rgb/4.png")

    rows = check_determinism.compare(rerun, reference, SUBSETS, 2e-3)

    assert [r["passed"] for r in rows] == [True, passed, True]
    assert check_determinism.report(rows, 2e-3) is passed


def test_determinism_fails_on_an_rgb_file_mismatch(runs: tuple[Path, Path]) -> None:
    rerun, reference = runs
    write_prediction(rerun / "seq" / "00005.npz", point_map(5), "rgb/6.png")

    diff = prediction_difference(rerun / "seq" / "00005.npz", reference / "seq" / "00005.npz")
    assert diff.max_abs_diff == 0.0 and not diff.rgb_match and not diff.passed()
    assert not check_determinism.report(check_determinism.compare(rerun, reference, SUBSETS, 2e-3), 2e-3)


def test_determinism_fails_on_a_missing_frame_or_a_noise_run(runs: tuple[Path, Path]) -> None:
    rerun, reference = runs
    (reference / "seq" / "00004.npz").unlink()
    with pytest.raises(FileNotFoundError, match="00004.npz"):
        check_determinism.compare(rerun, reference, SUBSETS, 2e-3)
    write_run(rerun, init="noise", subsets={"seq": {"first": 3, "last": 5}})
    with pytest.raises(ValueError, match="must use zeros"):
        check_determinism.compare(rerun, reference, SUBSETS, 2e-3)


def test_median_estimate_and_ratio_bootstrap() -> None:
    values = np.r_[np.linspace(1.0, 3.0, 40), np.nan]

    estimate = median_estimate(values)
    ratio = median_ratio_estimate(2.0 * values, values)

    assert estimate.value == pytest.approx(2.0)
    assert estimate.ci_low < 2.0 < estimate.ci_high
    assert (ratio.value, ratio.ci_low, ratio.ci_high) == pytest.approx((2.0, 2.0, 2.0))
    assert math.isnan(median_estimate(values[:5]).ci_low)
    with pytest.raises(ValueError, match="differ in length"):
        median_ratio_estimate(values, values[:-1])
