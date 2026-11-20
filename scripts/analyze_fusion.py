"""Block-bootstrap intervals for the per target fusion results and a measured versus forecast comparison.
Reads the per sequence CSVs of results/fusion/{oracle_eval,oracle_eval_edge,joint_eval}/baseline and
results/forecast/fusion_forecast.csv, writes CSV tables and summary.md to results/fusion_analysis and figures to
report/figures/fusion."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pim.analysis.stats import moving_block_indices, percentile_ci  # noqa: E402

SEQUENCES = {
    "rgbd_dataset_freiburg1_desk": "fr1_desk",
    "rgbd_dataset_freiburg1_room": "fr1_room",
    "rgbd_dataset_freiburg1_360": "fr1_360",
    "rgbd_dataset_freiburg3_long_office_household": "fr3",
}
BLOCKS = (8, 15, 30)
RESAMPLES = 2000
SEED = 0
BASELINE = "baseline_self"
FUSION_DIR = ROOT / "results" / "fusion"
SOURCES = {
    "oracle": FUSION_DIR / "oracle_eval" / "baseline" / "oracle_fusion_{}.csv",
    "edge": FUSION_DIR / "oracle_eval_edge" / "baseline" / "oracle_fusion_{}.csv",
    "joint": FUSION_DIR / "joint_eval" / "baseline" / "joint_fusion_{}.csv",
}
FORECAST_CSV = ROOT / "results" / "forecast" / "fusion_forecast.csv"
OUT_DIR = ROOT / "results" / "fusion_analysis"
FIG_DIR = ROOT / "report" / "figures" / "fusion"

LADDER = [f"causal_adjacent_w{w}" for w in (1, 2, 4)] + [f"causal_diverse_w{w}" for w in (1, 2, 3, 4)] + \
         [f"centred_adjacent_w{w}" for w in (1, 2, 4)] + [f"centred_diverse_w{w}" for w in (1, 2)] + \
         [f"{base}_rung_{r}" for base in ("causal_diverse_w4", "centred_adjacent_w4") for r in "abcdef"]
JOINT_BASES = ("causal_adjacent_w4", "causal_diverse_w4", "centred_adjacent_w4")
JOINT_CONTROL_BASE = "causal_adjacent_w4"
JOINT = [f"joint_{b}" for b in JOINT_BASES] + [f"nofusion_joint_{JOINT_CONTROL_BASE}",
                                               f"shuffled_poses_joint_{JOINT_CONTROL_BASE}"]
DEPLOYABLE = f"joint_{JOINT_CONTROL_BASE}"
ORACLE_CONTROL_REFERENCE = "causal_diverse_w4_rung_d"
NOFUSION = f"nofusion_joint_{JOINT_CONTROL_BASE}"
FORECAST_SELF = {"self_only": BASELINE}

Statistic = Callable[[np.ndarray], np.ndarray]
"""Maps (resamples, n) values to (resamples,) statistics; a 1-D input gives a scalar."""
STATISTICS: dict[str, Statistic] = {
    "mean_change_pct": lambda x: np.mean(x, axis=-1),
    "median_change_pct": lambda x: np.median(x, axis=-1),
    "share_improved": lambda x: np.mean(x < 0, axis=-1),
}
ABS_DIFF = "mean_abs_diff"
METRICS = (*STATISTICS, ABS_DIFF)

Table = dict[str, dict[str, dict[int, float]]]
"""source -> sequence -> config -> {target_index: rel_d}."""


def read_source(pattern: str) -> dict[str, dict[str, dict[int, float]]]:
    out: dict[str, dict[str, dict[int, float]]] = {}
    for sequence in SEQUENCES:
        configs: dict[str, dict[int, float]] = {}
        with open(pattern.format(sequence), newline="") as f:
            for row in csv.DictReader(f):
                configs.setdefault(row["config"], {})[int(row["target_index"])] = float(row["rel_d"])
        out[sequence] = configs
    return out


def paired(configs: dict[str, dict[int, float]], config: str, reference: str = BASELINE
           ) -> tuple[np.ndarray, np.ndarray]:
    """rel_d of config and reference on their shared targets, ordered by target index."""
    targets = sorted(set(configs[config]) & set(configs[reference]))
    if len(targets) != len(configs[reference]):
        raise ValueError(f"{config} misses targets of {reference}")
    return (np.array([configs[config][t] for t in targets]), np.array([configs[reference][t] for t in targets]))


def percent_change(value: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return 100.0 * (value - reference) / reference


def block_intervals(groups: Sequence[np.ndarray], statistic: Statistic, blocks: Iterable[int] = BLOCKS,
                    resamples: int = RESAMPLES, seed: int = SEED) -> dict[int, tuple[float, float]]:
    """Percentile intervals of the mean over groups of statistic(group); each group is resampled on its own with a
    moving block bootstrap in its given order. The same seed gives the same indices for groups of equal length."""
    out = {}
    for block in blocks:
        boot = np.zeros(resamples)
        for g, values in enumerate(groups):
            rng = np.random.default_rng([seed, block, g])
            boot += statistic(values[moving_block_indices(len(values), block, resamples, rng)])
        out[block] = percentile_ci(boot / len(groups))
    return out


def widest(intervals: dict[int, tuple[float, float]]) -> tuple[float, float]:
    return min(lo for lo, _ in intervals.values()), max(hi for _, hi in intervals.values())


def robust(intervals: dict[int, tuple[float, float]], null: float = 0.0) -> bool:
    """True only when every interval excludes the null value."""
    return all(lo > null or hi < null for lo, hi in intervals.values())


def describe(groups: Sequence[tuple[np.ndarray, np.ndarray]]) -> dict[str, object]:
    """Point values, per block intervals, widest interval and robustness for every metric. Each group is a
    (value, reference) pair; the point value is the mean over groups of the per group statistic."""
    changes = [percent_change(v, r) for v, r in groups]
    diffs = [v - r for v, r in groups]
    row: dict[str, object] = {"targets": sum(len(c) for c in changes)}
    for metric in METRICS:
        data, stat = (diffs, STATISTICS["mean_change_pct"]) if metric == ABS_DIFF else (changes, STATISTICS[metric])
        null = 0.5 if metric == "share_improved" else 0.0
        intervals = block_intervals(data, stat)
        row[metric] = float(np.mean([stat(d) for d in data]))
        for block, (lo, hi) in intervals.items():
            row[f"{metric}_ci_low_b{block}"], row[f"{metric}_ci_high_b{block}"] = lo, hi
        row[f"{metric}_ci_low"], row[f"{metric}_ci_high"] = widest(intervals)
        row[f"{metric}_robust"] = robust(intervals, null)
    return row


def metric_columns() -> list[str]:
    cols = ["targets"]
    for metric in METRICS:
        cols += [metric, *(f"{metric}_ci_{s}_b{b}" for b in BLOCKS for s in ("low", "high")),
                 f"{metric}_ci_low", f"{metric}_ci_high", f"{metric}_robust"]
    return cols


def forecast_mapping(forecast_configs: Iterable[str], measured_configs: Iterable[str]) -> dict[str, str]:
    """Forecast config to the measured config with the same setting and the plain weighted mean aggregator."""
    measured = set(measured_configs)
    mapping = {}
    for config in forecast_configs:
        target = FORECAST_SELF.get(config, config)
        if target not in measured:
            raise KeyError(f"no measured row for forecast config {config}")
        mapping[config] = target
    return mapping


def read_forecast(path: Path = FORECAST_CSV) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], columns: Sequence[str]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in row.items()})


def check_baselines(data: dict[str, Table]) -> None:
    for sequence in SEQUENCES:
        ref = data["oracle"][sequence][BASELINE]
        for source in ("edge", "joint"):
            other = data[source][sequence][BASELINE]
            if set(other) != set(ref) or max(abs(other[t] - ref[t]) for t in ref) > 1e-9:
                raise ValueError(f"{source} baseline_self differs from oracle on {sequence}")


def target_rows(data: dict[str, Table]) -> list[dict[str, object]]:
    rows = []
    for sequence, short in SEQUENCES.items():
        groups = {"ladder": ("oracle", LADDER),
                  "edge": ("edge", [c for c in data["edge"][sequence] if c != BASELINE]),
                  "joint": ("joint", JOINT)}
        for group, (source, configs) in groups.items():
            for config in configs:
                pair = paired(data[source][sequence], config)
                rows.append({"sequence": short, "group": group, "config": config, **describe([pair])})
    return rows


def forecast_rows(forecast: list[dict[str, str]], by_key: dict[tuple[str, str], dict[str, object]]
                  ) -> list[dict[str, object]]:
    rows = []
    for sequence, short in SEQUENCES.items():
        entries = [f for f in forecast if f["sequence"] == sequence]
        oracle_names = [k[1] for k in by_key if k[0] == short] + [BASELINE]
        mapping = forecast_mapping([f["config"] for f in entries], oracle_names)
        for f in entries:
            if mapping[f["config"]] == BASELINE:
                continue
            m = by_key[(short, mapping[f["config"]])]
            pred = float(f["pred_rel_d_change_pct"])
            lo, hi = m["mean_change_pct_ci_low"], m["mean_change_pct_ci_high"]
            rows.append({"sequence": short, "config": f["config"], "measured_config": mapping[f["config"]],
                         "forecast_change_pct": pred, "forecast_low": float(f["pred_rel_d_change_pct_low"]),
                         "forecast_high": float(f["pred_rel_d_change_pct_high"]),
                         "measured_mean_change_pct": m["mean_change_pct"], "measured_ci_low": lo,
                         "measured_ci_high": hi, "measured_robust": m["mean_change_pct_robust"],
                         "ratio_measured_over_forecast": m["mean_change_pct"] / pred,
                         "forecast_outside_ci": not lo <= pred <= hi})
    return rows


def control_rows(data: dict[str, Table]) -> list[dict[str, object]]:
    rows = []
    for sequence, short in SEQUENCES.items():
        oracle = data["oracle"][sequence]
        controls = [("oracle", c, ORACLE_CONTROL_REFERENCE) for c in oracle
                    if c.startswith(("shuffled_poses_", "pose_noise_"))]
        controls.append(("joint", f"shuffled_poses_joint_{JOINT_CONTROL_BASE}", DEPLOYABLE))
        for source, config, reference in controls:
            configs = data[source][sequence]
            row: dict[str, object] = {"sequence": short, "source": source, "config": config, "reference": reference}
            for label, ref in (("vs_baseline", BASELINE), ("vs_reference", reference)):
                d = describe([paired(configs, config, ref)])
                row[f"{label}_mean_change_pct"] = d["mean_change_pct"]
                row[f"{label}_ci_low"], row[f"{label}_ci_high"] = d["mean_change_pct_ci_low"], d["mean_change_pct_ci_high"]
                row[f"{label}_robust"] = d["mean_change_pct_robust"]
            rows.append(row)
    return rows


def nofusion_rows(data: dict[str, Table]) -> list[dict[str, object]]:
    rows = []
    for sequence, short in SEQUENCES.items():
        value, reference = paired(data["joint"][sequence], NOFUSION)
        d = describe([(value, reference)])
        rows.append({"sequence": short, "config": NOFUSION, "targets": len(value),
                     "mean_abs_diff": d[ABS_DIFF], "ci_low": d[f"{ABS_DIFF}_ci_low"],
                     "ci_high": d[f"{ABS_DIFF}_ci_high"], "max_abs_diff": float(np.max(np.abs(value - reference)))})
    return rows


def deployable_rows(data: dict[str, Table]) -> list[dict[str, object]]:
    pairs = {short: paired(data["joint"][seq], DEPLOYABLE) for seq, short in SEQUENCES.items()}
    rows = [{"sequence": short, "config": DEPLOYABLE, **describe([pair])} for short, pair in pairs.items()]
    rows.append({"sequence": "pooled", "config": DEPLOYABLE, **describe(list(pairs.values()))})
    return rows


def fmt_ci(value: float, lo: float, hi: float, digits: int = 1) -> str:
    return f"{value:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def yes(flag: object) -> str:
    return "yes" if flag else "no"


def markdown_table(header: Sequence[str], body: Iterable[Sequence[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return lines + ["| " + " | ".join(r) + " |" for r in body]


def metric_cells(r: dict[str, object]) -> list[str]:
    return [fmt_ci(r["mean_change_pct"], r["mean_change_pct_ci_low"], r["mean_change_pct_ci_high"]),
            yes(r["mean_change_pct_robust"]),
            fmt_ci(r["median_change_pct"], r["median_change_pct_ci_low"], r["median_change_pct_ci_high"]),
            yes(r["median_change_pct_robust"]),
            fmt_ci(100 * r["share_improved"], 100 * r["share_improved_ci_low"], 100 * r["share_improved_ci_high"], 0),
            fmt_ci(r[ABS_DIFF], r[f"{ABS_DIFF}_ci_low"], r[f"{ABS_DIFF}_ci_high"], 3)]


METRIC_HEADER = ["Mean change % [CI]", "Mean robust", "Median change % [CI]", "Median robust",
                 "Improved % [CI]", "Mean abs diff [CI]"]


def summary_markdown(targets: list[dict[str, object]], forecast: list[dict[str, object]],
                     controls: list[dict[str, object]], nofusion: list[dict[str, object]],
                     deployable: list[dict[str, object]]) -> str:
    lines = ["# Fusion analysis", "",
             f"Per target change in rel_d against baseline_self. Moving block bootstrap over targets in index order, "
             f"{RESAMPLES} resamples, seed {SEED}. Each interval is the widest 95 percent interval over block lengths "
             f"{', '.join(map(str, BLOCKS))}. Robust means every one of these intervals excludes zero (0.5 for the "
             f"improved share). Mean abs diff is the paired mean of fused minus baseline rel_d in rel_d points.", ""]
    lines += ["## Deployable headline (joint solved scale, causal adjacent window 4)", ""]
    lines += markdown_table(["Sequence", *METRIC_HEADER],
                            ([str(r["sequence"]), *metric_cells(r)] for r in deployable))
    lines += ["", "Pooled is the mean of the per sequence values, bootstrapped within each sequence.", ""]
    lines += ["## Measured versus forecast (oracle scale, plain weighted mean)", ""]
    lines += markdown_table(
        ["Sequence", "Config", "Forecast % [band]", "Measured mean % [CI]", "Robust", "Measured over forecast",
         "Forecast outside CI"],
        ([str(r["sequence"]), str(r["config"]),
          fmt_ci(r["forecast_change_pct"], r["forecast_low"], r["forecast_high"]),
          fmt_ci(r["measured_mean_change_pct"], r["measured_ci_low"], r["measured_ci_high"]),
          yes(r["measured_robust"]), f"{r['ratio_measured_over_forecast']:.2f}", yes(r["forecast_outside_ci"])]
         for r in forecast))
    lines += ["", "## Controls", ""]
    lines += markdown_table(
        ["Sequence", "Control", "Reference", "Mean change % vs baseline [CI]", "Robust",
         "Mean change % vs reference [CI]", "Robust"],
        ([str(r["sequence"]), str(r["config"]), str(r["reference"]),
          fmt_ci(r["vs_baseline_mean_change_pct"], r["vs_baseline_ci_low"], r["vs_baseline_ci_high"]),
          yes(r["vs_baseline_robust"]),
          fmt_ci(r["vs_reference_mean_change_pct"], r["vs_reference_ci_low"], r["vs_reference_ci_high"]),
          yes(r["vs_reference_robust"])] for r in controls))
    lines += ["", "## No fusion control minus baseline_self", ""]
    lines += markdown_table(
        ["Sequence", "Targets", "Mean diff [CI]", "Max abs diff"],
        ([str(r["sequence"]), str(r["targets"]), f"{r['mean_abs_diff']:.2e} [{r['ci_low']:.2e}, {r['ci_high']:.2e}]",
          f"{r['max_abs_diff']:.2e}"] for r in nofusion))
    for group, title in (("ladder", "Oracle ladder"), ("edge", "Edge group (oracle scale)"),
                         ("joint", "Joint solved scale")):
        lines += ["", f"## {title}", ""]
        lines += markdown_table(["Sequence", "Config", *METRIC_HEADER],
                                ([str(r["sequence"]), str(r["config"]), *metric_cells(r)]
                                 for r in targets if r["group"] == group))
    return "\n".join(lines) + "\n"


FAMILY_COLORS = {"causal_adjacent": "#2a78d6", "centred_adjacent": "#eb6834", "causal_diverse": "#1baf7a",
                 "centred_diverse": "#eda100"}
SEQUENCE_COLORS = dict(zip(SEQUENCES.values(), FAMILY_COLORS.values()))
INK, MUTED = "#3d3d3a", "#8c8b85"


def _style(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=8)
    ax.grid(color="#e6e5e0", linewidth=0.6)
    ax.set_axisbelow(True)


def figures(forecast: list[dict[str, object]], controls: list[dict[str, object]],
            deployable: list[dict[str, object]], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(9, 8), sharex=True, sharey=True)
    for ax, short in zip(axes.flat, SEQUENCES.values()):
        for r in (r for r in forecast if r["sequence"] == short):
            family = str(r["config"]).rsplit("_w", 1)[0]
            x, y = r["forecast_change_pct"], r["measured_mean_change_pct"]
            ax.errorbar(x, y, xerr=[[x - r["forecast_low"]], [r["forecast_high"] - x]],
                        yerr=[[y - r["measured_ci_low"]], [r["measured_ci_high"] - y]], fmt="o", ms=5,
                        color=FAMILY_COLORS[family], elinewidth=1, capsize=0, label=family.replace("_", " "))
        ax.axline((0, 0), slope=1, color=MUTED, linewidth=1, linestyle="--")
        ax.axhline(0, color=MUTED, linewidth=0.8)
        ax.set_title(short, fontsize=10, color=INK)
        _style(ax)
    for ax in axes[1]:
        ax.set_xlabel("Forecast rel_d change %", fontsize=9, color=INK)
    for ax in axes[:, 0]:
        ax.set_ylabel("Measured mean rel_d change %", fontsize=9, color=INK)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.legend(unique.values(), unique.keys(), loc="upper center", ncol=4, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out / "measured_vs_forecast.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.2))
    for i, r in enumerate(deployable):
        color = SEQUENCE_COLORS.get(str(r["sequence"]), INK)
        lo, hi = r["mean_change_pct_ci_low"], r["mean_change_pct_ci_high"]
        lo15, hi15 = r["mean_change_pct_ci_low_b15"], r["mean_change_pct_ci_high_b15"]
        ax.plot([lo, hi], [i, i], color=color, linewidth=1.5)
        ax.plot([lo15, hi15], [i, i], color=color, linewidth=4, solid_capstyle="round")
        ax.plot(r["mean_change_pct"], i, "o", ms=8, color=color, markeredgecolor="white", markeredgewidth=1.5)
        ax.annotate(f"{r['mean_change_pct']:.1f}", (r["mean_change_pct"], i), xytext=(0, 8),
                    textcoords="offset points", ha="center", fontsize=8, color=INK)
    ax.axvline(0, color=MUTED, linewidth=1)
    ax.set_yticks(range(len(deployable)), [str(r["sequence"]) for r in deployable])
    ax.set_ylim(len(deployable) - 0.5, -0.9)
    ax.set_xlabel("Mean rel_d change % vs baseline_self (thick block 15, thin widest over 8, 15, 30)",
                  fontsize=8, color=INK)
    ax.set_title("Joint solved causal adjacent window 4", fontsize=10, color=INK)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out / "deployable_gain.png", dpi=150)
    plt.close(fig)

    noise = sorted({str(r["config"]) for r in controls if str(r["config"]).startswith("pose_noise_")},
                   key=lambda c: [r["config"] for r in controls].index(c))
    shuffled = [c for c in dict.fromkeys(str(r["config"]) for r in controls) if c.startswith("shuffled_")]
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [2, 1]})
    for k, short in enumerate(SEQUENCES.values()):
        by_config = {r["config"]: r for r in controls if r["sequence"] == short}
        offset = (k - 1.5) * 0.15
        for ax, configs in ((left, noise), (right, shuffled)):
            for i, c in enumerate(configs):
                r = by_config[c]
                y = r["vs_baseline_mean_change_pct"]
                ax.errorbar(i + offset, y, yerr=[[y - r["vs_baseline_ci_low"]], [r["vs_baseline_ci_high"] - y]],
                            fmt="o", ms=5, color=SEQUENCE_COLORS[short], elinewidth=1, capsize=0,
                            label=short if i == 0 else None)
    left.set_xticks(range(len(noise)), [c.removeprefix("pose_noise_").split("_causal")[0] for c in noise],
                    fontsize=8, rotation=30)
    right.set_xticks(range(len(shuffled)), ["oracle rung d", "joint causal adjacent w4"], fontsize=8)
    left.set_title(f"Pose noise on {ORACLE_CONTROL_REFERENCE} (oracle scale)", fontsize=10, color=INK)
    right.set_title("Shuffled poses", fontsize=10, color=INK)
    for ax in (left, right):
        ax.axhline(0, color=MUTED, linewidth=1)
        ax.set_ylabel("Mean rel_d change % vs baseline_self", fontsize=9, color=INK)
        _style(ax)
    left.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "controls.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--figures", type=Path, default=FIG_DIR)
    args = parser.parse_args()
    data = {name: read_source(str(pattern)) for name, pattern in SOURCES.items()}
    check_baselines(data)
    targets = target_rows(data)
    by_key = {(r["sequence"], r["config"]): r for r in targets if r["group"] == "ladder"}
    forecast = forecast_rows(read_forecast(), by_key)
    controls = control_rows(data)
    nofusion = nofusion_rows(data)
    deployable = deployable_rows(data)

    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "fusion_targets.csv", targets, ["sequence", "group", "config", *metric_columns()])
    write_csv(args.out / "forecast_vs_measured.csv", forecast, list(forecast[0]))
    write_csv(args.out / "controls.csv", controls, list(controls[0]))
    write_csv(args.out / "nofusion.csv", nofusion, list(nofusion[0]))
    write_csv(args.out / "deployable.csv", deployable, ["sequence", "config", *metric_columns()])
    (args.out / "summary.md").write_text(summary_markdown(targets, forecast, controls, nofusion, deployable))
    figures(forecast, controls, deployable, args.figures)
    print(f"wrote tables to {args.out} and figures to {args.figures}")


if __name__ == "__main__":
    main()
