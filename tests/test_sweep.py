from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pim.model import runner
from pim.runs import PRE_SWEEP_OPTIONS, run_options
from pim.subsets import FrameRange, load_subsets

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_sweep_script", ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


predict = load_script("predict")
check_determinism = load_script("check_determinism")

SWEEP = {
    "steps_1": {"steps": 1},
    "steps_2": {"steps": 2},
    "steps_4": {"steps": 4},
    "fp16": {"dtype": "fp16"},
    "tokens_512": {"target_tokens": 512},
    "tokens_2048": {"target_tokens": 2048},
    "model_b": {"model": "B"},
    "downscale_2": {"input_downscale": 2},
    "baseline_check": {},
}
OLD_NPZ_KEYS = {"points", "valid", "rgb_timestamp", "rgb_file", "latency_ms", "peak_mem_gb"}


def write_config(tmp_path: Path, inference: str) -> Path:
    path = tmp_path / "run.yaml"
    path.write_text(f"name: a\ninference: {{init: zeros, seed: null{inference}}}\n")
    return path


def invocation(invocation_id: str) -> dict:
    return {"id": invocation_id, "start_time": "2026-11-12T10:00:00+00:00", "end_time": None, "sequences": {}}


@pytest.mark.parametrize("name", sorted(SWEEP))
def test_sweep_configs_change_one_option_each(name: str) -> None:
    config = predict.load_run_config(CONFIGS / "sweep" / f"{name}.yaml")

    assert config.name == f"sweep_{name}"
    assert (config.init, config.seed, config.seed_mode) == ("zeros", None, "per_call")
    assert config.options() == {**PRE_SWEEP_OPTIONS, **SWEEP[name]}
    assert config.as_dict()["inference"] == {"init": "zeros", "seed": None, **SWEEP[name]}


def test_defaults_reproduce_the_old_run_config() -> None:
    baseline = predict.load_run_config(CONFIGS / "baseline.yaml")
    check = predict.load_run_config(CONFIGS / "sweep" / "baseline_check.yaml")

    assert baseline == predict.RunConfig("baseline", "zeros", None)
    assert baseline.options() == check.options() == PRE_SWEEP_OPTIONS
    assert baseline.as_dict() == {"name": "baseline", "inference": {"init": "zeros", "seed": None}}
    assert check.as_dict()["inference"] == baseline.as_dict()["inference"]


@pytest.mark.parametrize("inference, key", [
    (", steps: 0", "steps"), (", steps: 2.5", "steps"), (", steps: true", "steps"), (", steps: '3'", "steps"),
    (", dtype: fp32", "dtype"), (", dtype: float16", "dtype"),
    (", target_tokens: -1", "target_tokens"), (", target_tokens: 1024.0", "target_tokens"),
    (", model: H", "model"), (", model: l", "model"),
    (", input_downscale: 0", "input_downscale"), (", input_downscale: 1.5", "input_downscale"),
])
def test_invalid_option_values_are_rejected(tmp_path: Path, inference: str, key: str) -> None:
    with pytest.raises(ValueError, match=f"inference.{key}"):
        predict.load_run_config(write_config(tmp_path, inference))


def test_unknown_inference_keys_are_still_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown keys \\['precision'\\]"):
        predict.load_run_config(write_config(tmp_path, ", precision: fp16"))


def stub_runner(target_tokens: int) -> SimpleNamespace:
    return SimpleNamespace(target_tokens=target_tokens, model=SimpleNamespace(patch_size=16))


def test_default_token_budget_gives_the_baseline_inference_size() -> None:
    assert runner.PointDiTRunner._inference_size(stub_runner(runner.DEFAULT_TARGET_TOKENS), 480, 640) == (448, 592)
    assert runner.resize_for_tokens(480, 640) == (448, 592)


@pytest.mark.parametrize("target_tokens", [256, 512, 2048, 4096])
def test_other_token_budgets_give_patch_multiples_near_the_target(target_tokens: int) -> None:
    h, w = runner.PointDiTRunner._inference_size(stub_runner(target_tokens), 480, 640)

    assert h % 16 == 0 and w % 16 == 0
    assert abs((h // 16) * (w // 16) / target_tokens - 1) < 0.05
    assert abs((w / h) / (640 / 480) - 1) < 0.05


def test_downscale_factor_one_returns_the_input_unchanged() -> None:
    image = np.random.default_rng(0).integers(0, 256, (480, 640, 3), dtype=np.uint8)
    copy = image.copy()

    out = runner.downscale_rgb(image, 1)

    assert out is image
    assert np.array_equal(out, copy)


def test_downscale_factor_two_is_a_box_average_and_maps_back_to_the_original_grid() -> None:
    image = np.random.default_rng(1).integers(0, 256, (480, 640, 3), dtype=np.uint8)

    small = runner.downscale_rgb(image, 2)

    assert small.shape == (240, 320, 3) and small.dtype == np.uint8
    block = image[10:12, 20:22].astype(np.float64).mean(axis=(0, 1))
    assert np.array_equal(small[5, 10], np.rint(block).astype(np.uint8))
    size = runner.resize_for_tokens(*small.shape[:2])
    assert size == (448, 592)
    points = torch.randn(1, 3, *size)
    restored = runner.restore_size(points, (480, 640))
    assert restored.shape == (1, 3, 480, 640)
    assert runner.restore_size(restored, (480, 640)) is restored


def test_downscale_rejects_factors_that_do_not_divide_the_image() -> None:
    with pytest.raises(ValueError, match="does not divide"):
        runner.downscale_rgb(np.zeros((480, 640, 3), np.uint8), 7)
    with pytest.raises(ValueError, match="at least 1"):
        runner.downscale_rgb(np.zeros((480, 640, 3), np.uint8), 0)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_fp16_finite_check_raises_on_nan_and_inf(bad: float) -> None:
    points = torch.zeros(1, 3, 4, 6)
    runner.require_finite(points, "fp16 prediction")
    points[0, 1, 2, 3] = bad
    with pytest.raises(FloatingPointError, match="1 non-finite"):
        runner.require_finite(points, "fp16 prediction")


def test_run_json_records_every_option_and_reads_back(tmp_path: Path) -> None:
    config = predict.RunConfig("sweep", "zeros", None, steps=2, dtype="fp16", target_tokens=512, model="B",
                               input_downscale=2)
    with predict.run_record(tmp_path) as record:
        predict.register_invocation(record, config, {"seq": {"start": 0, "count": 5}}, invocation("1"))

    record = json.loads((tmp_path / "run.json").read_text())
    assert record["options"] == {"steps": 2, "dtype": "fp16", "target_tokens": 512, "model": "B",
                                 "input_downscale": 2}
    assert run_options(record) == record["options"]
    assert record["config"]["inference"] == {"init": "zeros", "seed": None, **record["options"]}
    with pytest.raises(ValueError, match="config"):
        with predict.run_record(tmp_path) as record:
            predict.register_invocation(record, predict.RunConfig("sweep", "zeros", None), {}, invocation("2"))


def test_default_config_keeps_the_old_run_json_config_and_records_default_options(tmp_path: Path) -> None:
    config = predict.load_run_config(CONFIGS / "baseline.yaml")
    with predict.run_record(tmp_path) as record:
        predict.register_invocation(record, config, {"seq": {"start": 0, "count": 5}}, invocation("1"))

    record = json.loads((tmp_path / "run.json").read_text())
    assert record["config"] == {"name": "baseline", "inference": {"init": "zeros", "seed": None}}
    assert record["options"] == PRE_SWEEP_OPTIONS


def test_old_run_json_without_options_reads_as_the_pre_sweep_defaults_and_resumes(tmp_path: Path) -> None:
    config = predict.RunConfig("baseline", "zeros", None)
    with predict.run_record(tmp_path) as record:
        predict.register_invocation(record, config, {"seq": {"start": 0, "count": 5}}, invocation("1"))
    old = json.loads((tmp_path / "run.json").read_text())
    del old["options"]
    (tmp_path / "run.json").write_text(json.dumps(old))

    assert run_options(old) == PRE_SWEEP_OPTIONS
    with predict.run_record(tmp_path) as record:
        predict.register_invocation(record, config, {"seq": {"start": 0, "count": 5}}, invocation("2"))
    assert "options" not in json.loads((tmp_path / "run.json").read_text())


def test_fork_args_for_l_are_unchanged() -> None:
    assert runner.fork_args() == runner.DEMO_L_512_ARGS == [
        "--model", "PointDiT-L/16", "--feature_embedding_type", "dinov3_vitl16", "--proj_dropout", "0.0",
        "--evaluate_gen", "--eval_wild_images", "--img_size", "512", "--num_sampling_steps", "3",
    ]


def test_model_b_args_differ_only_in_model_and_encoder() -> None:
    large, base = runner.fork_args("L"), runner.fork_args("B")

    assert len(large) == len(base)
    changed = {large[i - 1]: (large[i], base[i]) for i in range(len(large)) if large[i] != base[i]}
    assert changed == {"--model": ("PointDiT-L/16", "PointDiT-B/16"),
                       "--feature_embedding_type": ("dinov3_vitl16", "dinov3_vitb16")}
    steps = runner.fork_args("L", 2)
    assert [a for a, b in zip(large, steps) if a != b] == ["3"]
    assert steps[steps.index("--num_sampling_steps") + 1] == "2"


def test_model_b_weights_follow_the_fork_naming() -> None:
    from pim.paths import checkpoint_path, dinov3_weights_path, DINOV3_WEIGHTS, POINTDIT_CHECKPOINT

    assert checkpoint_path("L") == POINTDIT_CHECKPOINT
    assert dinov3_weights_path("L") == DINOV3_WEIGHTS
    assert checkpoint_path("B").name == "pointditb-512-mixdata-nodinov3-1d42aacf.pth"
    assert dinov3_weights_path("B").name == "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
    assert dinov3_weights_path("B").parent == DINOV3_WEIGHTS.parent


class TimedResult(SimpleNamespace):
    @property
    def peak_memory_gb(self) -> float:
        return 1.5


class TimedRunner:
    def __init__(self) -> None:
        self.calls = 0

    def predict(self, image: np.ndarray, init: str, seed: int | None) -> TimedResult:
        self.calls += 1
        return TimedResult(points=np.zeros((4, 6, 3), np.float32), valid_mask=np.ones((4, 6), bool),
                           latency_ms=100.0, model_latency_ms=10.0 * self.calls)


def test_model_latency_goes_to_run_json_and_npz_keys_stay_the_same(tmp_path: Path) -> None:
    from PIL import Image

    from pim.data.tum import Frame, TUMSequence, Window

    (tmp_path / "data" / "rgb").mkdir(parents=True)
    frames = []
    for i in range(3):
        path = tmp_path / "data" / "rgb" / f"{i}.png"
        Image.fromarray(np.zeros((4, 6, 3), np.uint8)).save(path)
        frames.append(Frame(i, 1000.0 + i, path, tmp_path / "depth.png", 0.0, np.eye(4)))
    sequence = TUMSequence("seq", tmp_path / "data", np.eye(3), Window(None, None), frames)
    config = predict.RunConfig("test", "zeros", None)

    stats = predict.predict_sequence(TimedRunner(), config, sequence, frames, tmp_path / "run")

    assert (stats.mean_model_latency_ms, stats.median_model_latency_ms) == (20.0, 20.0)
    with np.load(tmp_path / "run" / "seq" / "00001.npz") as data:
        assert set(data.files) == OLD_NPZ_KEYS
    with predict.run_record(tmp_path / "run") as record:
        predict.register_invocation(record, config, {"seq": {"start": 0, "count": 3}}, invocation("1"))
        predict.record_sequence(record, "1", "seq", {"start": 0, "count": 3}, stats)
    entry = json.loads((tmp_path / "run" / "run.json").read_text())["invocations"][0]["sequences"]["seq"]
    assert entry == {"predicted": 3, "skipped": 0, "mean_model_latency_ms": 20.0, "median_model_latency_ms": 20.0}

    resumed = predict.predict_sequence(None, config, sequence, frames, tmp_path / "run")
    assert resumed.mean_model_latency_ms is None


def test_sweep_subsets_match_the_sampling_study() -> None:
    subsets = load_subsets(CONFIGS / "sweep_subsets.yaml")

    assert subsets == load_subsets(CONFIGS / "sampling_subsets.yaml")
    assert subsets == {"rgbd_dataset_freiburg1_360": FrameRange(200, 299),
                       "rgbd_dataset_freiburg3_long_office_household": FrameRange(740, 839)}


def test_determinism_check_rejects_runs_with_different_options(tmp_path: Path) -> None:
    for name, options in (("a", None), ("b", {**PRE_SWEEP_OPTIONS, "steps": 2})):
        (tmp_path / name).mkdir()
        run = {"name": name, "config": {"name": name, "inference": {"init": "zeros", "seed": None}}}
        if options is not None:
            run["options"] = options
        (tmp_path / name / "run.json").write_text(json.dumps(run))

    with pytest.raises(ValueError, match="same options"):
        check_determinism.compare(tmp_path / "a", tmp_path / "b", {"seq": FrameRange(0, 0)}, 2e-3)
