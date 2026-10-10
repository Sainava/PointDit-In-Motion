from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_script", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


analyze = load_script("analyze")


def per_frame(keys: list[tuple[str, int]]) -> list[dict[str, str]]:
    return [{"sequence": s, "frame_index": str(i), "rgb_timestamp": f"{1000 + i / 30:.6f}", "rgb_file": f"rgb/{i}.png",
             "skipped": "False", "depth_offset_ms": "12.0", "rel_d": str(3.0 + i), "focal_ratio": "1.1"}
            for s, i in keys]


def covariates(keys: list[tuple[str, int]]) -> list[dict[str, str]]:
    return [{"sequence": s, "frame_index": str(i), "rgb_timestamp": f"{1000 + i / 30:.6f}", "rgb_file": f"rgb/{i}.png",
             "depth_offset_ms": "11.99", "angular_speed_dps": str(10.0 * i), "sharpness_rel": ""}
            for s, i in keys]


KEYS = [("seq_b", 2), ("seq_a", 1), ("seq_b", 1), ("seq_a", 0)]


def test_join_sorts_frames_and_keeps_per_frame_columns_on_clashes() -> None:
    tables = analyze.join(per_frame(KEYS), covariates(KEYS[::-1]))

    assert list(tables) == ["seq_b", "seq_a"]
    assert tables["seq_a"]["frame_index"].tolist() == [0, 1]
    assert tables["seq_a"]["rel_d"].tolist() == [3.0, 4.0]
    assert tables["seq_a"]["angular_speed_dps"].tolist() == [0.0, 10.0]
    assert tables["seq_b"]["depth_offset_ms"].tolist() == [12.0, 12.0]
    assert np.isnan(tables["seq_b"]["sharpness_rel"]).all()


def test_join_fails_on_mismatched_keys() -> None:
    with pytest.raises(ValueError, match=r"key mismatch .* 1 keys only in per_frame.csv .*\('seq_b', 2\).* 1 only in "
                                         r"covariates.csv .*\('seq_b', 3\)"):
        analyze.join(per_frame(KEYS), covariates(KEYS[1:] + [("seq_b", 3)]))


def test_join_fails_on_mismatched_focal_split_keys() -> None:
    focal = [{"sequence": s, "frame_index": str(i), "focal_split_logdiff": "0.01"} for s, i in KEYS[:3]]

    with pytest.raises(ValueError, match="key mismatch between per_frame.csv and focal_split.csv"):
        analyze.join(per_frame(KEYS), covariates(KEYS), focal)


def test_join_fails_on_duplicate_keys() -> None:
    with pytest.raises(ValueError, match=r"covariates.csv has duplicate key \('seq_a', 0\)"):
        analyze.join(per_frame(KEYS), covariates(KEYS + [("seq_a", 0)]))


def test_join_fails_when_timestamps_disagree() -> None:
    shifted = covariates(KEYS)
    shifted[0]["rgb_timestamp"] = "999.000000"

    with pytest.raises(ValueError, match="rgb_timestamp"):
        analyze.join(per_frame(KEYS), shifted)


def test_join_adds_abs_log_focal_ratio_with_focal_split() -> None:
    focal = [{"sequence": s, "frame_index": str(i), "focal_split_logdiff": "0.01"} for s, i in KEYS]

    tables = analyze.join(per_frame(KEYS), covariates(KEYS), focal)

    assert tables["seq_a"]["abs_log_focal_ratio"] == pytest.approx([np.log(1.1)] * 2)
    assert tables["seq_a"]["focal_split_logdiff"].tolist() == [0.01, 0.01]
