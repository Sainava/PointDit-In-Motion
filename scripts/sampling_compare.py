"""Compare sampler initialisations on frame subsets: same-frame disagreement between noise seeds and zero init,
per-frame accuracy and temporal consistency under noise vs zeros, and a regression check of the zero-init
consistency against the full baseline run. Writes CSVs, summary.md and sampling.json to --out."""

from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pim.consistency.evaluation import cell_value, pairs_csv, read_pairs  # noqa: E402
from pim.consistency.metrics import frame_inputs  # noqa: E402
from pim.data.tum import Frame, apply_window, load_sequence  # noqa: E402
from pim.evaluation import expected_rgb_file, prediction_path, set_torch_threads  # noqa: E402
from pim.paths import FORK_DIR, REPO_ROOT  # noqa: E402
from pim.runs import git_commit, now, read_run  # noqa: E402
from pim.sampling import (  # noqa: E402
    BLOCK_LENGTH, BOOTSTRAP_SEED, REGRESSION_TOLERANCE, RESAMPLES, Estimate, comparisons, consistency_regression,
    disagreement, median_estimate, median_ratio_estimate, pair_key,
)
from pim.subsets import Subsets, load_subsets, restrict_frames, subsets_record  # noqa: E402

ACCURACY_METRICS = ("rel_d", "rel_p", "delta1_d")
CONSISTENCY_METRICS = {"m1": "m1_visible_median", "m2": "m2_visible_median"}
DISAGREEMENT_COLUMNS = ("sequence", "frame_index", "comparison", "reference", "aligned", "pixels", "median", "p90")
PROTOCOL = {
    "seed_disagreement": "per frame, the reference prediction's depth aligned to GT by upstream compute_metrics "
                         "(as the M1/M2 anchor), and the other prediction's raw z fitted to it with "
                         "align_depth_affine (scale + shift, weights 1/reference, 64x64 masked nearest subsample), "
                         "as M2 fits frame j; relative difference |aligned - reference| / reference x100 over pixels "
                         "valid in both predictions, GT-valid and with positive reference depth",
    "accuracy": "evaluate.py per-frame rel_d, rel_p, delta1_d; per-frame difference config - zeros",
    "consistency": "consistency_eval.py visible-pixel per-pair medians; median over pairs per gap",
    "bootstrap": f"moving-block bootstrap over frames (accuracy, disagreement) or pairs ordered by frame i "
                 f"(consistency), block {BLOCK_LENGTH}, {RESAMPLES} resamples, seed {BOOTSTRAP_SEED}, 95% percentile "
                 "CI; M2 ratio resamples both configs' pairs jointly",
    "regression": f"zero-init pairs on the subsets vs the same pairs of the full baseline run, every metric column "
                  f"within {REGRESSION_TOLERANCE:g}",
}


@dataclass(frozen=True)
class Config:
    name: str
    run_dir: Path
    run: dict[str, Any]

    @property
    def init(self) -> str:
        return self.run["config"]["inference"]["init"]

    @property
    def seed(self) -> int | None:
        return self.run["config"]["inference"]["seed"]

    @property
    def seed_mode(self) -> str:
        return self.run["config"]["inference"].get("seed_mode", "per_call")


@dataclass(frozen=True)
class FrameTask:
    sequence: str
    sequence_root: Path
    K: np.ndarray
    frame: Frame
    predictions: dict[str, Path]
    comparisons: tuple[tuple[str, str], ...]
    threads: int


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--noise", type=Path, nargs="+", required=True, help="noise-init run directories")
    parser.add_argument("--zeros", type=Path, required=True, help="zero-init run directory on the same subsets")
    parser.add_argument("--eval-root", type=Path, required=True, help="holds <run name>/per_frame.csv")
    parser.add_argument("--consistency-root", type=Path, required=True,
                        help="holds <run name>/pairs_<sequence>.csv")
    parser.add_argument("--baseline-consistency", type=Path, required=True,
                        help="full-window zero-init consistency run, e.g. results/consistency/baseline")
    parser.add_argument("--subsets", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True, help="folder containing the sequence folders")
    parser.add_argument("--sequences-config", type=Path, default=Path("configs/sequences.yaml"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1,
                        help="worker processes for seed disagreement; each needs ~3.5 GB of RAM")
    parser.add_argument("--threads", type=int, default=1, help="torch/OpenMP threads per worker")
    return parser.parse_args(argv)


def load_configs(noise_dirs: Sequence[Path], zeros_dir: Path, subsets: Subsets) -> tuple[list[Config], Config]:
    noise = [Config(read_run(d)["name"], d, read_run(d)) for d in noise_dirs]
    zeros = Config(read_run(zeros_dir)["name"], zeros_dir, read_run(zeros_dir))
    names = [c.name for c in (*noise, zeros)]
    if len(set(names)) != len(names):
        raise ValueError(f"run names must be distinct, got {names}")
    wrong = [c.name for c in noise if c.init != "noise"] + ([zeros.name] if zeros.init != "zeros" else [])
    if wrong:
        raise ValueError(f"runs {wrong} have the wrong init for their role (--noise needs noise, --zeros zeros)")
    seeds = [c.seed for c in noise]
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"noise runs must use distinct seeds, got {seeds}")
    expected = subsets_record(subsets)
    for c in (*noise, zeros):
        if c.run.get("subsets") != expected:
            raise ValueError(f"{c.run_dir} was predicted on subsets {c.run.get('subsets')}, expected {expected}")
    return noise, zeros


def check_evaluates(record_path: Path, config: Config, subsets: Subsets) -> dict[str, Any]:
    if not record_path.is_file():
        raise FileNotFoundError(f"{record_path} not found; evaluate run {config.name!r} on the subsets first")
    record = json.loads(record_path.read_text())
    if record["predictions"]["run"] != config.run:
        raise ValueError(f"{record_path} evaluated run {record['predictions']['run'].get('name')!r} started "
                         f"{record['predictions']['run'].get('start_time')}, not the run in {config.run_dir}")
    if record.get("subsets") != subsets_record(subsets):
        raise ValueError(f"{record_path} covers subsets {record.get('subsets')}, expected {subsets_record(subsets)}")
    return record


def subset_frames(args: argparse.Namespace, subsets: Subsets) -> dict[str, tuple[Any, list[Frame]]]:
    out = {}
    for name, frame_range in subsets.items():
        sequence = load_sequence(args.data_root / name, args.sequences_config)
        out[name] = (sequence, restrict_frames(name, apply_window(sequence), frame_range))
    return out


def read_accuracy(eval_root: Path, configs: Sequence[Config], subsets: Subsets) -> dict[str, dict[tuple[str, int], dict]]:
    by_config = {}
    for config in configs:
        directory = eval_root / config.name
        check_evaluates(directory / "eval.json", config, subsets)
        with open(directory / "per_frame.csv", newline="") as f:
            rows = {(r["sequence"], int(r["frame_index"])): r for r in csv.DictReader(f)}
        expected = {(name, i) for name, r in subsets.items() for i in range(r.first, r.last + 1)}
        if set(rows) != expected:
            raise ValueError(f"{directory / 'per_frame.csv'} holds {len(rows)} frames, expected exactly the "
                             f"{len(expected)} subset frames")
        by_config[config.name] = rows
    return by_config


def accuracy_tables(
    accuracy: dict[str, dict[tuple[str, int], dict]], noise: Sequence[Config], zeros: Config, subsets: Subsets
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    per_frame, summary = [], []
    for name, frame_range in subsets.items():
        indices = range(frame_range.first, frame_range.last + 1)
        values = {c: {m: np.array([cell_value(accuracy[c][name, i][m]) for i in indices]) for m in ACCURACY_METRICS}
                  for c in accuracy}
        for config in (zeros, *noise):
            for k, i in enumerate(indices):
                per_frame.append({"sequence": name, "frame_index": i, "config": config.name,
                                  **{m: values[config.name][m][k] for m in ACCURACY_METRICS}})
        for config in noise:
            for metric in ACCURACY_METRICS:
                difference = median_estimate(values[config.name][metric] - values[zeros.name][metric])
                summary.append({"sequence": name, "config": config.name, "metric": metric, "frames": len(indices),
                                "zeros_median": float(np.nanmedian(values[zeros.name][metric])),
                                "config_median": float(np.nanmedian(values[config.name][metric])),
                                **estimate_columns("difference", difference)})
    return per_frame, summary


def read_consistency(
    root: Path, configs: Sequence[Config], subsets: Subsets
) -> dict[str, dict[str, list[dict[str, str]]]]:
    by_config = {}
    for config in configs:
        directory = root / config.name
        check_evaluates(directory / "consistency.json", config, subsets)
        by_config[config.name] = {name: read_pairs(pairs_csv(directory, name)) for name in subsets}
    keys = {c: {name: [pair_key(r) for r in rows] for name, rows in by_name.items()} for c, by_name in
            by_config.items()}
    first = next(iter(keys.values()))
    for name, config_keys in keys.items():
        if config_keys != first:
            raise ValueError(f"{name}'s consistency pairs differ from {next(iter(keys))}'s")
    return by_config


def regression(
    zeros_rows: dict[str, list[dict[str, str]]], baseline_dir: Path, subsets: Subsets
) -> tuple[list[dict[str, Any]], list[str]]:
    table, failures = [], []
    for name, frame_range in subsets.items():
        baseline = [r for r in read_pairs(pairs_csv(baseline_dir, name))
                    if frame_range.contains(int(r["frame_i"])) and frame_range.contains(int(r["frame_j"]))]
        rows = zeros_rows[name]
        largest, problems = consistency_regression(rows, baseline)
        missing = {pair_key(r) for r in baseline} - {pair_key(r) for r in rows}
        if missing:
            problems.append(f"{len(missing)} baseline pairs inside the subset are missing, e.g. {min(missing)}")
        failures += [f"{name}: {p}" for p in problems]
        table.append({"sequence": name, "pairs": len(rows), "baseline_pairs": len(baseline),
                      "max_abs_diff": largest, "failures": len(problems)})
    return table, failures


def consistency_summary(
    by_config: dict[str, dict[str, list[dict[str, str]]]], noise: Sequence[Config], zeros: Config
) -> list[dict[str, Any]]:
    summary = []
    for name, zeros_rows in by_config[zeros.name].items():
        for gap in sorted({float(r["gap_ms"]) for r in zeros_rows}):
            series = {}
            for config in (zeros, *noise):
                rows = sorted((r for r in by_config[config.name][name] if float(r["gap_ms"]) == gap),
                              key=lambda r: int(r["frame_i"]))
                series[config.name] = {m: np.array([cell_value(r[column]) for r in rows])
                                       for m, column in CONSISTENCY_METRICS.items()}
            for config in (zeros, *noise):
                values = series[config.name]
                ratio = (Estimate(1.0, math.nan, math.nan) if config is zeros
                         else median_ratio_estimate(values["m2"], series[zeros.name]["m2"]))
                summary.append({"sequence": name, "gap_ms": f"{gap:g}", "config": config.name,
                                "pairs": len(values["m2"]),
                                **estimate_columns("m1", median_estimate(values["m1"])),
                                **estimate_columns("m2", median_estimate(values["m2"])),
                                **estimate_columns("m2_ratio_to_zeros", ratio)})
    return summary


def estimate_columns(prefix: str, estimate: Estimate) -> dict[str, float]:
    return {prefix: estimate.value, f"{prefix}_ci_low": estimate.ci_low, f"{prefix}_ci_high": estimate.ci_high}


def init_worker(threads: int) -> None:
    os.environ["OMP_NUM_THREADS"] = str(threads)
    set_torch_threads(threads)


def load_points(path: Path, frame: Frame, sequence_root: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"prediction {path} not found")
    with np.load(path) as data:
        points, valid, rgb_file = data["points"].astype(np.float32), data["valid"].astype(bool), str(data["rgb_file"])
    expected = expected_rgb_file(frame, sequence_root)
    if rgb_file != expected:
        raise ValueError(f"{path} was predicted from {rgb_file}, but frame {frame.index} is {expected}")
    return points, valid


def frame_disagreements(task: FrameTask) -> list[dict[str, Any]]:
    depth, gt_valid = task.frame.load_depth()
    loaded = {name: load_points(path, task.frame, task.sequence_root) for name, path in task.predictions.items()}
    references = dict.fromkeys(reference for reference, _ in task.comparisons)
    anchors = {name: frame_inputs(depth, gt_valid, task.K, loaded[name][0]) for name in references}
    rows = []
    for reference, other in task.comparisons:
        inputs = anchors[reference]
        mask = inputs.gt_valid & loaded[reference][1] & loaded[other][1]
        result = disagreement(inputs.aligned_depth, loaded[other][0][..., 2], mask)
        rows.append({"sequence": task.sequence, "frame_index": task.frame.index,
                     "comparison": f"{reference}-{other}", "reference": reference, "aligned": other,
                     "pixels": result.pixels, "median": result.median, "p90": result.p90})
    return rows


def run_disagreements(tasks: list[FrameTask], workers: int) -> Iterator[list[dict[str, Any]]]:
    if workers == 1:
        init_worker(tasks[0].threads)
        yield from map(frame_disagreements, tasks)
        return
    executor = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                                   initializer=init_worker, initargs=(tasks[0].threads,))
    try:
        yield from executor.map(frame_disagreements, tasks, chunksize=1)
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


def disagreement_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary = []
    for name in dict.fromkeys(r["sequence"] for r in rows):
        for comparison in dict.fromkeys(r["comparison"] for r in rows):
            selected = sorted((r for r in rows if r["sequence"] == name and r["comparison"] == comparison),
                              key=lambda r: r["frame_index"])
            medians = np.array([r["median"] for r in selected], dtype=np.float64)
            p90s = np.array([r["p90"] for r in selected], dtype=np.float64)
            summary.append({"sequence": name, "comparison": comparison, "frames": len(selected),
                            "frames_nan": int(np.isnan(medians).sum()),
                            **estimate_columns("median_of_medians", median_estimate(medians)),
                            "median_of_p90": float(np.nanmedian(p90s)), "max_of_medians": float(np.nanmax(medians))})
    return summary


def write_csv(path: Path, rows: Sequence[dict[str, Any]], columns: Sequence[str] | None = None) -> None:
    if not rows and columns is None:
        raise ValueError(f"{path}: no rows to write")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns or rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if isinstance(v, float) and math.isnan(v) else v) for k, v in row.items()})


def fmt(value: float, digits: int = 3) -> str:
    return "n/a" if math.isnan(value) else f"{value:.{digits}f}"


def with_ci(row: dict[str, Any], prefix: str, digits: int = 3) -> str:
    low, high = row[f"{prefix}_ci_low"], row[f"{prefix}_ci_high"]
    ci = "" if math.isnan(low) else f" [{fmt(low, digits)}, {fmt(high, digits)}]"
    return f"{fmt(row[prefix], digits)}{ci}"


def markdown_table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header),
            *("| " + " | ".join(row) + " |" for row in rows), ""]


def summary_markdown(
    noise: Sequence[Config], zeros: Config, subsets: Subsets, disagreement_rows: list[dict[str, Any]],
    accuracy_rows: list[dict[str, Any]], consistency_rows: list[dict[str, Any]], regression_rows: list[dict[str, Any]],
) -> str:
    lines = ["# Sampler initialisation comparison", ""]
    lines += ["## Runs", ""]
    lines += markdown_table(("run", "init", "seed", "seed mode"),
                            [(c.name, c.init, str(c.seed), c.seed_mode) for c in (zeros, *noise)])
    lines += markdown_table(("sequence", "frames"),
                            [(name, f"{r.first}-{r.last}") for name, r in subsets.items()])
    lines += ["## Seed disagreement (x100, per-frame median relative difference)", ""]
    lines += markdown_table(
        ("sequence", "comparison (reference-aligned)", "frames", "median of medians [95% CI]", "median of p90",
         "max of medians"),
        [(r["sequence"], r["comparison"], str(r["frames"]), with_ci(r, "median_of_medians"), fmt(r["median_of_p90"]),
          fmt(r["max_of_medians"])) for r in disagreement_rows])
    lines += [f"## Accuracy: median per-frame difference {{noise}} - {zeros.name} [95% CI]", ""]
    lines += markdown_table(
        ("sequence", "metric", f"{zeros.name} median", *(c.name for c in noise)),
        [(name, metric, fmt(next(r for r in accuracy_rows if r["sequence"] == name and r["metric"] == metric)
                            ["zeros_median"]),
          *(with_ci(next(r for r in accuracy_rows if r["sequence"] == name and r["metric"] == metric
                         and r["config"] == c.name), "difference") for c in noise))
         for name in subsets for metric in ACCURACY_METRICS])
    lines += ["## Consistency: median over pairs of per-pair median x100, visible pixels [95% CI]", ""]
    lines += markdown_table(
        ("sequence", "gap (ms)", "run", "pairs", "M1", "M2", f"M2 / M2({zeros.name})"),
        [(r["sequence"], r["gap_ms"], r["config"], str(r["pairs"]), with_ci(r, "m1"), with_ci(r, "m2"),
          "-" if r["config"] == zeros.name else with_ci(r, "m2_ratio_to_zeros")) for r in consistency_rows])
    lines += ["## Regression: zero-init subset pairs vs the full baseline run", ""]
    lines += markdown_table(("sequence", "pairs", "baseline pairs in subset", "max |diff|", "failures"),
                            [(r["sequence"], str(r["pairs"]), str(r["baseline_pairs"]), f"{r['max_abs_diff']:.2e}",
                              str(r["failures"])) for r in regression_rows])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.workers < 1 or args.threads < 1:
        raise ValueError(f"--workers and --threads must be at least 1, got {args.workers}, {args.threads}")
    if not args.data_root.is_dir():
        raise FileNotFoundError(f"data root {args.data_root} does not exist")
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    start_time, started = now(), time.perf_counter()
    subsets = load_subsets(args.subsets)
    noise, zeros = load_configs(args.noise, args.zeros, subsets)
    configs = [zeros, *noise]
    args.out.mkdir(parents=True, exist_ok=True)

    consistency = read_consistency(args.consistency_root, configs, subsets)
    regression_rows, failures = regression(consistency[zeros.name], args.baseline_consistency, subsets)
    write_csv(args.out / "consistency_regression.csv", regression_rows)
    for row in regression_rows:
        print(f"regression {row['sequence']}: {row['pairs']} pairs (baseline {row['baseline_pairs']}), "
              f"max |diff| {row['max_abs_diff']:.2e}, {row['failures']} failures")
    if failures:
        for failure in failures[:20]:
            print(f"  FAIL {failure}")
        raise SystemExit(f"regression check failed: {len(failures)} problems; see above")

    accuracy_frames, accuracy_rows = accuracy_tables(read_accuracy(args.eval_root, configs, subsets), noise, zeros,
                                                     subsets)
    consistency_rows = consistency_summary(consistency, noise, zeros)

    frames = subset_frames(args, subsets)
    pairs = tuple(comparisons([c.name for c in noise], zeros.name))
    tasks = [FrameTask(name, sequence.root, sequence.K, frame,
                       {c.name: prediction_path(c.run_dir, name, frame.index) for c in configs}, pairs, args.threads)
             for name, (sequence, sequence_frames) in frames.items() for frame in sequence_frames]
    print(f"seed disagreement: {len(tasks)} frames x {len(pairs)} comparisons with {args.workers} worker(s)",
          flush=True)
    disagreement_rows: list[dict[str, Any]] = []
    disagreement_started = time.perf_counter()
    for done, rows in enumerate(run_disagreements(tasks, args.workers), start=1):
        disagreement_rows += rows
        if done % 25 == 0 or done == len(tasks):
            seconds = (time.perf_counter() - disagreement_started) / done
            print(f"  {done}/{len(tasks)} frames, {seconds:.1f} s/frame", flush=True)
    disagreement_table = disagreement_summary(disagreement_rows)

    write_csv(args.out / "seed_disagreement.csv", disagreement_rows, DISAGREEMENT_COLUMNS)
    write_csv(args.out / "seed_disagreement_summary.csv", disagreement_table)
    write_csv(args.out / "accuracy_per_frame.csv", accuracy_frames)
    write_csv(args.out / "accuracy_summary.csv", accuracy_rows)
    write_csv(args.out / "consistency_summary.csv", consistency_rows)
    (args.out / "summary.md").write_text(summary_markdown(noise, zeros, subsets, disagreement_table, accuracy_rows,
                                                          consistency_rows, regression_rows))
    record = {
        "runs": {c.name: {"dir": str(c.run_dir), "run": c.run} for c in configs},
        "commits": {"project": git_commit(REPO_ROOT), "pointdit": git_commit(FORK_DIR)},
        "arguments": {k: ([str(p) for p in v] if isinstance(v, list) else str(v) if isinstance(v, Path) else v)
                      for k, v in vars(args).items()},
        "subsets": subsets_record(subsets),
        "sequences_config": yaml.safe_load(args.sequences_config.read_text()),
        "protocol": PROTOCOL,
        "regression": {"tolerance": REGRESSION_TOLERANCE, "sequences": regression_rows},
        "start_time": start_time,
        "end_time": now(),
        "seconds": round(time.perf_counter() - started, 1),
    }
    (args.out / "sampling.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
