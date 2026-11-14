from __future__ import annotations

import csv
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pim import sweep
from pim.subsets import FrameRange

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {"seq_a": 3, "seq_b": 2}
GPU = {"sm_clock_mhz": 1590.0, "mem_clock_mhz": 5000.0, "temperature_c": 45.0, "power_w": 30.5}


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_runner_test", ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


run_sweep = load_script("run_sweep")
sweep_summary = load_script("sweep_summary")
sweep_accuracy = load_script("sweep_accuracy")


def write_config(folder: Path, stem: str) -> Path:
    path = folder / f"{stem}.yaml"
    path.write_text(f"name: sweep_{stem}\ninference: {{init: zeros, seed: null}}\n")
    return path


def fake_prediction(out: Path, run_name: str, timings: Path, frames: dict[str, int], model_ms: list[float]) -> None:
    """The npz files, run.json and timings CSV that predict.py leaves behind."""
    run_dir = out / run_name
    for name, count in frames.items():
        (run_dir / name).mkdir(parents=True, exist_ok=True)
        for i in range(count):
            (run_dir / name / f"{i:05d}.npz").write_bytes(b"")
    run = {"name": run_name, "frames_predicted": sum(frames.values()), "options": {"steps": 3},
           "sequences": {name: {"frames": count, "mean_latency_ms": 100.0 + k, "peak_mem_gb": 2.0 + k}
                         for k, (name, count) in enumerate(frames.items())}}
    (run_dir / "run.json").write_text(json.dumps(run))
    with open(timings, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("sequence", "frame_index", "model_latency_ms", "latency_ms"))
        writer.writerows(("seq_a", i, ms, ms + 50) for i, ms in enumerate(model_ms))


class StubLaunch:
    """Plays predict.py, failing for stems in `fail`, raising for `crash`, else writing a full run."""

    def __init__(self, out: Path, fail: set[str] = frozenset(), crash: set[str] = frozenset(),
                 timeout: set[str] = frozenset()) -> None:
        self.out, self.fail, self.crash, self.timeout = out, fail, crash, timeout
        self.calls: list[tuple[list[str], dict[str, str]]] = []

    def __call__(self, command: list[str], env: dict[str, str], prefix: str, timeout: float | None):
        self.calls.append((command, env))
        stem = Path(command[1]).stem
        if stem in self.crash:
            raise OSError("cannot start")
        if stem in self.timeout:
            return run_sweep.Outcome(-9, ["frame 3"], timed_out=True)
        if stem in self.fail:
            return run_sweep.Outcome(1, ["Traceback (most recent call last):",
                                         "FloatingPointError: fp16 prediction has 12 non-finite values"])
        fake_prediction(self.out, f"sweep_{stem}", Path(command[2]), EXPECTED, [10.0, 20.0, 60.0])
        return run_sweep.Outcome(0, ["done"])


def command(config: Path, timings: Path) -> list[str]:
    return ["predict", str(config), str(timings)]


def test_one_failing_config_does_not_stop_the_others(tmp_path: Path) -> None:
    configs = [write_config(tmp_path, s) for s in ("baseline_check", "fp16", "model_b", "steps_1")]
    out = tmp_path / "sweep"
    launch = StubLaunch(out, fail={"fp16"}, crash={"model_b"})

    records = run_sweep.run_configs(configs, out, command, EXPECTED, run=launch, smi=lambda: GPU)

    assert [r["status"] for r in records] == ["ok", "failed", "failed", "ok"]
    assert len(launch.calls) == 4
    assert all(env["CUDA_VISIBLE_DEVICES"] == "0" for _, env in launch.calls)
    assert records[1]["error"] == "FloatingPointError: fp16 prediction has 12 non-finite values"
    assert records[1]["output_tail"][-1] == records[1]["error"]
    assert records[2]["error"] == "OSError: cannot start"
    for stem in ("baseline_check", "fp16", "model_b", "steps_1"):
        status = json.loads((out / f"sweep_{stem}" / sweep.STATUS_FILE).read_text())
        assert status["config"] == stem
    table = run_sweep.status_table(records)
    assert "fp16" in table and "failed" in table and "1590>1590" in table


def test_timeouts_and_unreadable_configs_are_recorded(tmp_path: Path) -> None:
    bad = tmp_path / "broken.yaml"
    bad.write_text("inference: {}\n")
    configs = [bad, write_config(tmp_path, "tokens_2048"), write_config(tmp_path, "steps_2")]
    out = tmp_path / "sweep"

    records = run_sweep.run_configs(configs, out, command, EXPECTED, run=StubLaunch(out, timeout={"tokens_2048"}),
                                    smi=lambda: None)

    assert [r["status"] for r in records] == ["failed", "timeout", "ok"]
    assert "KeyError" in records[0]["error"]
    assert json.loads((out / "broken" / sweep.STATUS_FILE).read_text())["status"] == "failed"
    assert records[2]["gpu_before"] is None


def test_status_json_holds_timings_memory_and_gpu_readings(tmp_path: Path) -> None:
    out = tmp_path / "sweep"
    readings = iter([{**GPU, "temperature_c": 40.0}, {**GPU, "sm_clock_mhz": 1200.0}])

    run_sweep.run_config(write_config(tmp_path, "steps_4"), out, command, EXPECTED, run=StubLaunch(out),
                         smi=lambda: next(readings))

    status = json.loads((out / "sweep_steps_4" / sweep.STATUS_FILE).read_text())
    assert status["status"] == "ok" and status["run_name"] == "sweep_steps_4"
    assert status["frames_predicted"] == 3
    assert (status["frames_expected"], status["frames_on_disk"]) == (5, 5)
    assert status["mean_model_latency_ms"] == pytest.approx(30.0)
    assert status["median_model_latency_ms"] == 20.0
    assert status["mean_latency_ms"] == pytest.approx(80.0)
    assert status["peak_mem_gb"] == 3.0
    assert status["gpu_before"]["temperature_c"] == 40.0
    assert status["gpu_after"]["sm_clock_mhz"] == 1200.0
    assert status["output_tail"] == [] and status["error"] is None


def test_missing_frames_mark_a_clean_exit_incomplete(tmp_path: Path) -> None:
    out = tmp_path / "sweep"
    run_dir = out / "sweep_x"
    (run_dir / "seq_a").mkdir(parents=True)
    (run_dir / "seq_a" / "00000.npz").write_bytes(b"")

    record = sweep.status_record("x", "sweep_x", "ok", 0, 1.0, run_dir, EXPECTED, None, None)

    assert record["status"] == "incomplete" and record["frames_on_disk"] == 1
    assert record["mean_model_latency_ms"] is None and record["peak_mem_gb"] is None


def test_nvidia_smi_is_optional() -> None:
    def missing(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi")

    def hangs(*args, **kwargs):
        raise subprocess.TimeoutExpired("nvidia-smi", 30)

    assert sweep.nvidia_smi(run=missing) is None
    assert sweep.nvidia_smi(run=hangs) is None
    assert sweep.nvidia_smi(run=lambda *a, **k: SimpleNamespace(returncode=9, stdout="")) is None
    ok = sweep.nvidia_smi(run=lambda *a, **k: SimpleNamespace(returncode=0, stdout="1590, 5000, 45, 30.50\n"))
    assert ok == GPU
    assert sweep.parse_nvidia_smi("1590, 5000, 45, [N/A]")["power_w"] == "[N/A]"
    assert sweep.parse_nvidia_smi("1590, 5000") is None


def stub_check(code: int):
    calls = []

    def run(command, env, prefix, timeout):
        calls.append(command)
        return run_sweep.Outcome(code, [])

    return run, calls


def test_determinism_mismatch_exits_3_with_a_banner(tmp_path: Path, capsys) -> None:
    records = [{"run_name": "sweep_baseline_check", "status": "ok"}]
    run, calls = stub_check(3)

    code = run_sweep.determinism_check(records, tmp_path, tmp_path / "baseline", tmp_path / "s.yaml",
                                       tmp_path / "determinism.csv", run=run)

    assert code == 3
    assert calls[0][calls[0].index("--mismatch-exit-code") + 1] == "3"
    assert "DETERMINISM CHECK FAILED" in capsys.readouterr().out


@pytest.mark.parametrize("records, code, expected", [
    ([{"run_name": "sweep_baseline_check", "status": "ok"}], 0, 0),
    ([{"run_name": "sweep_baseline_check", "status": "ok"}], 1, 1),
    ([{"run_name": "sweep_baseline_check", "status": "failed"}], 0, 1),
    ([{"run_name": "sweep_fp16", "status": "ok"}], 0, 0),
])
def test_determinism_exit_codes(tmp_path: Path, records: list, code: int, expected: int) -> None:
    run, calls = stub_check(code)

    assert run_sweep.determinism_check(records, tmp_path, tmp_path, tmp_path, tmp_path / "d.csv", run=run) == expected


def test_summary_writes_one_row_per_config(tmp_path: Path) -> None:
    out = tmp_path / "sweep"
    configs = [write_config(tmp_path, s) for s in ("baseline_check", "fp16")]
    run_sweep.run_configs(configs, out, command, EXPECTED, run=StubLaunch(out, fail={"fp16"}), smi=lambda: GPU)
    # A run folder from an older session with run.json only.
    fake_prediction(out, "sweep_old", tmp_path / "old.csv", {"seq_a": 2}, [5.0])
    (out / "sweep_old" / "run.json").write_text(json.dumps(
        {"name": "sweep_old", "frames_predicted": 2, "sequences": {"seq_a": {"frames": 2, "mean_latency_ms": 90.0,
                                                                              "peak_mem_gb": 1.5}}}))

    sweep_summary.main(["--sweep-dir", str(out)])

    with open(out / "sweep_latency.csv", newline="") as f:
        reader = csv.DictReader(f)
        rows = {r["config"]: r for r in reader}
    assert tuple(reader.fieldnames) == sweep.SUMMARY_COLUMNS
    assert set(rows) == {"sweep_baseline_check", "sweep_fp16", "sweep_old"}
    ok, failed, old = rows["sweep_baseline_check"], rows["sweep_fp16"], rows["sweep_old"]
    assert (ok["status"], ok["frames"], ok["median_model_latency_ms"]) == ("ok", "3", "20.0")
    assert (ok["end_to_end_latency_ms"], ok["peak_mem_gb"], ok["gpu_clock_start"]) == ("80.0", "3.0", "1590.0")
    assert (failed["status"], failed["frames"], failed["mean_model_latency_ms"]) == ("failed", "0", "")
    assert (old["status"], old["frames"], old["end_to_end_latency_ms"], old["gpu_clock_end"]) == \
        ("unknown", "2", "90.0", "")


SUBSETS = {"seq_a": FrameRange(10, 29), "seq_b": FrameRange(0, 19)}


def synthetic_rel_d(offset: float, seed: int) -> dict[tuple[str, int], float]:
    rng = np.random.default_rng(seed)
    return {(name, i): 5.0 + offset + float(rng.normal(0, 0.01)) * (seed > 0)
            for name, r in SUBSETS.items() for i in range(r.first, r.last + 1)}


def test_accuracy_table_on_a_tiny_synthetic_input() -> None:
    check = synthetic_rel_d(0.0, 0)
    shifted = {k: v + 0.5 for k, v in check.items()}
    shifted["seq_b", 3] = math.nan
    noisy = synthetic_rel_d(0.2, 1)

    rows = {r["config"]: r for r in sweep_accuracy.accuracy_rows(
        {"sweep_baseline_check": check, "sweep_steps_1": shifted, "sweep_fp16": noisy}, SUBSETS, resamples=200)}

    assert rows["sweep_baseline_check"]["median_rel_d_diff"] == 0.0
    assert (rows["sweep_baseline_check"]["diff_ci_low"], rows["sweep_baseline_check"]["diff_ci_high"]) == (0.0, 0.0)
    steps = rows["sweep_steps_1"]
    assert steps["frames"] == 39
    assert steps["median_rel_d"] == pytest.approx(5.5)
    assert steps["median_rel_d_diff"] == pytest.approx(0.5)
    assert steps["diff_ci_low"] == pytest.approx(0.5) and steps["diff_ci_high"] == pytest.approx(0.5)
    fp16 = rows["sweep_fp16"]
    assert fp16["diff_ci_low"] <= fp16["median_rel_d_diff"] <= fp16["diff_ci_high"]
    assert 0.15 < fp16["diff_ci_low"] and fp16["diff_ci_high"] < 0.25


def test_accuracy_needs_the_reference_and_every_subset_frame() -> None:
    check = synthetic_rel_d(0.0, 0)
    with pytest.raises(ValueError, match="sweep_baseline_check has no evaluation"):
        sweep_accuracy.accuracy_rows({"sweep_fp16": check}, SUBSETS)
    partial = dict(check)
    del partial["seq_a", 12]
    with pytest.raises(ValueError, match="sweep_fp16 lacks 1 subset frames"):
        sweep_accuracy.accuracy_rows({"sweep_baseline_check": check, "sweep_fp16": partial}, SUBSETS)


def test_short_series_get_no_interval() -> None:
    short = {"seq_a": FrameRange(0, 4)}
    values = {("seq_a", i): 1.0 for i in range(5)}

    row = sweep_accuracy.accuracy_rows({"sweep_baseline_check": values}, short)[0]

    assert math.isnan(row["diff_ci_low"]) and row["median_rel_d_diff"] == 0.0


def test_regression_tolerance_and_nans() -> None:
    keys = [("s", 0), ("s", 1), ("s", 2)]
    baseline = {("s", 0): 1.0, ("s", 1): math.nan, ("s", 2): 3.0}

    largest, failures = sweep_accuracy.regression({("s", 0): 1.0005, ("s", 1): math.nan, ("s", 2): 3.0}, baseline, keys)
    assert failures == [] and largest == pytest.approx(5e-4)
    _, failures = sweep_accuracy.regression({("s", 0): 1.002, ("s", 1): 2.0}, baseline, keys)
    assert len(failures) == 3


def write_eval(root: Path, config: str, values: dict[tuple[str, int], float]) -> None:
    folder = root / config
    folder.mkdir(parents=True)
    (folder / "eval.json").write_text(json.dumps({"predictions": {"run": {"name": config}}}))
    write_per_frame(folder / "per_frame.csv", values)


def write_per_frame(path: Path, values: dict[tuple[str, int], float]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("sequence", "frame_index", "rel_d", "delta1_d"))
        writer.writerows((s, i, "" if math.isnan(v) else v, 99.0) for (s, i), v in values.items())


def test_accuracy_script_writes_the_csv_before_a_failed_regression(tmp_path: Path) -> None:
    (tmp_path / "subsets.yaml").write_text("subsets:\n  seq_a: {first: 10, last: 29}\n  seq_b: {first: 0, last: 19}\n")
    check = synthetic_rel_d(0.0, 0)
    write_eval(tmp_path / "eval", "sweep_baseline_check", check)
    write_eval(tmp_path / "eval", "sweep_steps_2", synthetic_rel_d(0.1, 2))
    baseline = {**check, ("seq_a", 99): 7.0}
    write_per_frame(tmp_path / "baseline.csv", baseline)
    args = ["--eval-root", str(tmp_path / "eval"), "--configs", "sweep_baseline_check", "sweep_steps_2",
            "--subsets", str(tmp_path / "subsets.yaml"), "--baseline-csv", str(tmp_path / "baseline.csv"),
            "--out", str(tmp_path / "out" / "sweep_accuracy.csv")]

    sweep_accuracy.main(args)
    with open(tmp_path / "out" / "sweep_accuracy.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [r["config"] for r in rows] == ["sweep_baseline_check", "sweep_steps_2"]
    assert tuple(rows[0]) == sweep_accuracy.COLUMNS

    write_per_frame(tmp_path / "baseline.csv", {**check, ("seq_b", 5): check["seq_b", 5] + 0.01})
    (tmp_path / "out" / "sweep_accuracy.csv").unlink()
    with pytest.raises(SystemExit) as exit_info:
        sweep_accuracy.main(args)
    assert exit_info.value.code == 3
    assert (tmp_path / "out" / "sweep_accuracy.csv").is_file()


def test_predict_appends_per_frame_timings_after_warmup(tmp_path: Path) -> None:
    from PIL import Image

    from pim.data.tum import Frame, TUMSequence, Window

    predict = load_script("predict")

    class Result(SimpleNamespace):
        peak_memory_gb = 1.0

    class Runner:
        calls = 0

        def predict(self, image, init, seed):
            Runner.calls += 1
            return Result(points=np.zeros((4, 6, 3), np.float32), valid_mask=np.ones((4, 6), bool),
                          latency_ms=100.0 + Runner.calls, model_latency_ms=float(Runner.calls))

    (tmp_path / "data" / "rgb").mkdir(parents=True)
    frames = []
    for i in range(3):
        path = tmp_path / "data" / "rgb" / f"{i}.png"
        Image.fromarray(np.zeros((4, 6, 3), np.uint8)).save(path)
        frames.append(Frame(i, 1000.0 + i, path, tmp_path / "depth.png", 0.0, np.eye(4)))
    sequence = TUMSequence("seq", tmp_path / "data", np.eye(3), Window(None, None), frames)
    runner = Runner()
    for _ in range(predict.WARMUP_RUNS):
        runner.predict(None, "zeros", None)

    stats = predict.predict_sequence(runner, predict.RunConfig("t", "zeros", None), sequence, frames[:2],
                                     tmp_path / "run")
    predict.append_timings(tmp_path / "t.csv", "seq", stats)
    stats = predict.predict_sequence(runner, predict.RunConfig("t", "zeros", None), sequence, frames,
                                     tmp_path / "run")
    predict.append_timings(tmp_path / "t.csv", "seq", stats)

    with open(tmp_path / "t.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [(r["frame_index"], r["model_latency_ms"]) for r in rows] == [("0", "3.0"), ("1", "4.0"), ("2", "5.0")]
    assert stats.mean_model_latency_ms == 5.0
