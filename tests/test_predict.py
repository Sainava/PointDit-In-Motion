from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pim.data.tum import Frame, TUMSequence, Window

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "predict.py"
spec = importlib.util.spec_from_file_location("predict_script", SCRIPT)
predict = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = predict
spec.loader.exec_module(predict)

H, W = 4, 6
CONFIG = predict.RunConfig("test", "zeros", None)


@dataclass
class StubResult:
    points: np.ndarray
    valid_mask: np.ndarray
    latency_ms: float
    peak_memory_gb: float


class StubRunner:
    def __init__(self, fail_on_call: int | None = None) -> None:
        self.calls = 0
        self.fail_on_call = fail_on_call

    def predict(self, image: np.ndarray, init: str, seed: int | None) -> StubResult:
        self.calls += 1
        if self.calls == self.fail_on_call:
            raise KeyboardInterrupt
        value = float(image[0, 0, 0])
        return StubResult(np.full((H, W, 3), value, np.float32), np.ones((H, W), bool), 100.0 + value, 1.5)


def make_sequence(root: Path, n: int = 5) -> TUMSequence:
    (root / "rgb").mkdir(parents=True)
    frames = []
    for i in range(n):
        path = root / "rgb" / f"{i}.png"
        Image.fromarray(np.full((H, W, 3), 10 * i, np.uint8)).save(path)
        frames.append(Frame(i, 1000.0 + i / 30, path, root / "depth" / f"{i}.png", 0.0, np.eye(4)))
    return TUMSequence("seq", root, np.eye(3), Window(None, None), frames)


def saved_files(run_dir: Path) -> list[str]:
    return sorted(p.name for p in (run_dir / "seq").iterdir())


def test_every_frame_is_saved_with_its_metadata(tmp_path: Path) -> None:
    sequence = make_sequence(tmp_path / "data")
    runner = StubRunner()

    stats = predict.predict_sequence(runner, CONFIG, sequence, sequence.frames[1:4], tmp_path / "run")

    assert saved_files(tmp_path / "run") == ["00001.npz", "00002.npz", "00003.npz"]
    assert (stats.frames, stats.predicted, stats.skipped) == (3, 3, 0)
    assert stats.mean_latency_ms == pytest.approx(120.0)
    with np.load(tmp_path / "run" / "seq" / "00002.npz") as data:
        assert data["points"].dtype == np.float16
        assert data["points"].shape == (H, W, 3)
        assert np.all(data["points"] == 20.0)
        assert data["valid"].dtype == bool
        assert str(data["rgb_file"]) == "rgb/2.png"
        assert float(data["rgb_timestamp"]) == sequence.frames[2].timestamp
        assert float(data["latency_ms"]) == 120.0
        assert float(data["peak_mem_gb"]) == 1.5


def test_existing_frames_are_skipped_without_predicting(tmp_path: Path) -> None:
    sequence = make_sequence(tmp_path / "data")
    predict.predict_sequence(StubRunner(), CONFIG, sequence, sequence.frames[:2], tmp_path / "run")
    before = (tmp_path / "run" / "seq" / "00000.npz").read_bytes()
    runner = StubRunner()

    stats = predict.predict_sequence(runner, CONFIG, sequence, sequence.frames, tmp_path / "run")

    assert runner.calls == 3
    assert (stats.predicted, stats.skipped) == (3, 2)
    assert (tmp_path / "run" / "seq" / "00000.npz").read_bytes() == before


def test_nothing_is_predicted_when_every_file_exists(tmp_path: Path) -> None:
    sequence = make_sequence(tmp_path / "data")
    predict.predict_sequence(StubRunner(), CONFIG, sequence, sequence.frames, tmp_path / "run")

    stats = predict.predict_sequence(None, CONFIG, sequence, sequence.frames, tmp_path / "run")

    assert (stats.predicted, stats.skipped) == (0, 5)


def test_interrupted_run_leaves_only_whole_files_and_resumes(tmp_path: Path) -> None:
    sequence = make_sequence(tmp_path / "data")
    with pytest.raises(KeyboardInterrupt):
        predict.predict_sequence(StubRunner(fail_on_call=3), CONFIG, sequence, sequence.frames, tmp_path / "run")
    assert saved_files(tmp_path / "run") == ["00000.npz", "00001.npz"]

    runner = StubRunner()
    stats = predict.predict_sequence(runner, CONFIG, sequence, sequence.frames, tmp_path / "run")

    assert runner.calls == 3
    assert (stats.predicted, stats.skipped) == (3, 2)


def test_failed_write_leaves_no_partial_or_temporary_file(tmp_path: Path, monkeypatch) -> None:
    sequence = make_sequence(tmp_path / "data")
    real_savez = np.savez_compressed
    calls = 0

    def savez_then_crash(file, **arrays) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            file.write(b"PK\x03\x04 partial")
            raise OSError("disk full")
        real_savez(file, **arrays)

    monkeypatch.setattr(predict.np, "savez_compressed", savez_then_crash)
    with pytest.raises(OSError, match="disk full"):
        predict.predict_sequence(StubRunner(), CONFIG, sequence, sequence.frames, tmp_path / "run")

    assert saved_files(tmp_path / "run") == ["00000.npz"]


def test_stale_temporary_files_are_removed(tmp_path: Path) -> None:
    sequence = make_sequence(tmp_path / "data")
    (tmp_path / "run" / "seq").mkdir(parents=True)
    (tmp_path / "run" / "seq" / ".00000.npz.123.tmp").write_bytes(b"partial")

    predict.predict_sequence(StubRunner(), CONFIG, sequence, sequence.frames[:1], tmp_path / "run")

    assert saved_files(tmp_path / "run") == ["00000.npz"]


@pytest.mark.filterwarnings("ignore:overflow encountered in cast")
def test_non_finite_points_are_rejected(tmp_path: Path) -> None:
    sequence = make_sequence(tmp_path / "data")
    result = StubResult(np.full((H, W, 3), 1e6, np.float32), np.ones((H, W), bool), 1.0, 1.0)

    with pytest.raises(ValueError, match="non-finite"):
        predict.save_prediction(tmp_path / "00000.npz", sequence.frames[0], sequence.root, result)
    assert list(tmp_path.glob("*.npz")) == []


def test_run_config_rejects_unknown_and_missing_keys(tmp_path: Path) -> None:
    path = tmp_path / "run.yaml"
    path.write_text("name: a\ninference: {init: zeros, seed: null, steps: 3}\n")
    with pytest.raises(ValueError, match="unknown keys \\['steps'\\]"):
        predict.load_run_config(path)
    path.write_text("name: a\ninference: {init: zeros}\n")
    with pytest.raises(ValueError, match="missing keys \\['seed'\\]"):
        predict.load_run_config(path)


def test_baseline_config_loads() -> None:
    config = predict.load_run_config(SCRIPT.parents[1] / "configs" / "baseline.yaml")

    assert config == predict.RunConfig("baseline", "zeros", None)


def invocation(invocation_id: str) -> dict:
    return {"id": invocation_id, "start_time": f"2026-10-06T10:00:0{invocation_id}+00:00",
            "end_time": None, "sequences": {}}


def test_run_record_merges_two_processes_and_rejects_config_changes(tmp_path: Path) -> None:
    windows = {"a": {"start": 0, "count": 3}, "b": {"start": 2, "count": 3}}
    stats = predict.SequenceStats(3, 2, 1, 100.0, 1.5)
    for process, name in (("1", "a"), ("2", "b")):
        with predict.run_record(tmp_path) as record:
            predict.register_invocation(record, CONFIG, {name: windows[name]}, invocation(process))
    for process, name in (("2", "b"), ("1", "a")):
        with predict.run_record(tmp_path) as record:
            predict.record_sequence(record, process, name, windows[name], stats)
            predict.finish_invocation(record, process)

    record = json.loads((tmp_path / "run.json").read_text())
    assert sorted(record["sequences"]) == ["a", "b"]
    assert [inv["id"] for inv in record["invocations"]] == ["1", "2"]
    assert (record["frames_predicted"], record["frames_skipped"]) == (4, 2)
    assert record["start_time"] == "2026-10-06T10:00:01+00:00"
    assert record["end_time"] is not None

    with pytest.raises(ValueError, match="config"):
        with predict.run_record(tmp_path) as record:
            predict.register_invocation(record, predict.RunConfig("test", "noise", 0), {}, invocation("3"))
    with pytest.raises(ValueError, match="window"):
        with predict.run_record(tmp_path) as record:
            predict.register_invocation(record, CONFIG, {"a": {"start": 1, "count": 3}}, invocation("3"))
    assert len(json.loads((tmp_path / "run.json").read_text())["invocations"]) == 2
