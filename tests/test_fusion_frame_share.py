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


fs = load_script("fusion_frame_share")


def write_frame(folder: Path, index: int, raw: float, fused: float, has_fused: bool = True) -> None:
    # Depth arrays are object dtype, so reading them with allow_pickle False raises
    poison = np.empty(2, dtype=object)
    folder.mkdir(parents=True, exist_ok=True)
    np.savez(folder / f"{index:05d}.npz", raw_aligned=poison, fused_aligned=poison,
             rel_d_raw=np.float64(raw), rel_d_fused=np.float64(fused), has_fused=np.bool_(has_fused),
             index=np.int64(index), timestamp=np.float64(index / 30))


@pytest.fixture
def render_dir(tmp_path: Path) -> Path:
    rng = np.random.default_rng(1)
    seq = tmp_path / "render" / "rgbd_dataset_freiburg1_360"
    for i in range(40):
        raw = 0.1 + 0.01 * rng.random()
        fused = raw if i % 10 == 0 else raw * (1 + 0.05 * rng.standard_normal())
        if 23 <= i <= 26:
            fused = raw * 1.5 + 0.01 * i
        write_frame(seq, i, raw, fused)
    write_frame(seq, 40, 0.1, float("nan"), has_fused=False)
    write_frame(tmp_path / "render" / "other", 0, 0.2, 0.1)
    return tmp_path / "render"


def test_reading_frames_never_touches_the_depth_arrays(render_dir: Path) -> None:
    path = render_dir / "rgbd_dataset_freiburg1_360" / "00000.npz"
    with np.load(path, allow_pickle=False) as data, pytest.raises(ValueError):
        data["raw_aligned"]

    assert fs.read_scalars(path)["index"] == 0
    assert fs.read_scalars(render_dir / "rgbd_dataset_freiburg1_360" / "00040.npz") is None


def test_shares_sum_to_one_and_frame_23_is_flagged(render_dir: Path, tmp_path: Path) -> None:
    seqs = fs.read_render_dir(render_dir)
    row = next(fs.summarise(s, None) for s in seqs if s.name == "fr1_360")

    assert row["fused_frames"] == 40 and row["frames_without_fused"] == 1
    assert row["share_lower"] + row["share_equal"] + row["share_higher"] == pytest.approx(1.0)
    assert row["share_equal"] == pytest.approx(4 / 40)
    assert set(row["worst_indices"].split()[:4]) == {"23", "24", "25", "26"}
    assert row["frame_23_worst_rank"] <= 4 and row["frame_23_in_worst_5"]
    assert row["longest_worse_run"] >= 4
    assert row["longest_worse_run_start"] <= 23 and row["longest_worse_run_end"] >= 26


def test_bootstrap_gives_ordered_intervals_around_the_point_value(render_dir: Path) -> None:
    seq = next(s for s in fs.read_render_dir(render_dir) if s.name == "fr1_360")
    row = fs.summarise(seq, None)

    for metric in ("mean_change_pct", "share_lower"):
        for block in fs.BLOCKS:
            lo, hi = row[f"{metric}_ci_low_b{block}"], row[f"{metric}_ci_high_b{block}"]
            assert lo <= hi
        assert row[f"{metric}_ci_low"] <= row[metric] <= row[f"{metric}_ci_high"]
    assert fs.summarise(seq, None) == row


def test_bootstrap_skips_blocks_longer_than_the_series(render_dir: Path) -> None:
    row = next(fs.summarise(s, None) for s in fs.read_render_dir(render_dir) if s.name == "other")

    assert row["share_lower"] == 1.0
    assert np.isnan(row["mean_change_pct_ci_low_b8"])
    assert np.isnan(row["mean_change_pct_ci_low"])


def test_longest_worse_run() -> None:
    frames = [fs.Frame(i, 0.0, 1.0, f) for i, f in enumerate([2, 2, 0.5, 2, 2, 2, 1.0, 2])]

    assert fs.longest_worse_run(frames) == (3, 3, 5)


def test_main_writes_csvs_and_tables_only_markdown(render_dir: Path, tmp_path: Path) -> None:
    disagreement = tmp_path / "dis.csv"
    with open(disagreement, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sequence", "index", "disagreement_raw_pct", "disagreement_fused_pct"])
        writer.writerows([["rgbd_dataset_freiburg1_360", i, 2.0, 1.0 if i < 30 else 3.0] for i in range(40)])
    out = tmp_path / "out"

    fs.main([str(render_dir), str(out), "--disagreement-csv", str(disagreement)])

    with open(out / "frame_share.csv") as f:
        frames = list(csv.DictReader(f))
    assert len(frames) == 41
    assert next(r for r in frames if r["sequence"] == "fr1_360")["disagreement_raw_pct"] == "2.0"
    with open(out / "frame_share_summary.csv") as f:
        summary = {r["sequence"]: r for r in csv.DictReader(f)}
    assert float(summary["fr1_360"]["share_disagreement_lower"]) == pytest.approx(0.75)
    md = (out / "summary.md").read_text()
    assert "oracle poses" in md and "joint solved scale" in md
    body = [line for line in md.splitlines() if line and not line.startswith("#")]
    assert all(line.startswith("|") for line in body)
