"""Rank correlations of per-frame error with motion, blur, scene content and implied focal length, with moving-block
bootstrap CIs, plus temporal spike detection; reads per_frame.csv, covariates.csv and optionally focal_split.csv."""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pim.analysis.stats import (  # noqa: E402
    average_ranks, bootstrap, detect_spikes, fast_slow_median_difference, moving_block_indices, neighbour_medians,
    percentile_ci, pooled_partial_spearman,
)

Table = dict[str, np.ndarray]
KEYS = ("sequence", "frame_index")
DEFAULT_BLOCK_LENGTH = 15
ROBUSTNESS_BLOCK_LENGTHS = (8, 30)
RESAMPLES = 2000
SEED = 20261010
MIN_SUBSET_N = 40
POOLED = "pooled_fr1"
SPIKE_RADIUS = 5
SPIKE_MIN_EXCESS = 1.5
SPIKE_MIN_RELATIVE_EXCESS = 0.5
THIRDS = ("near", "mid", "far")
NON_NUMERIC = {"sequence", "rgb_file", "skipped"}
ALL, SMALL_OFFSET, RELIABLE_FOCAL = "all", "depth_offset_ms<5", "focal_split_logdiff<ln(1.1)"
SUBSETS: dict[str, Callable[[Table], np.ndarray]] = {
    ALL: lambda t: np.ones(len(t["frame_index"]), bool),
    SMALL_OFFSET: lambda t: t["depth_offset_ms"] < 5.0,
    RELIABLE_FOCAL: lambda t: np.nan_to_num(t["focal_split_logdiff"], nan=np.inf) < math.log(1.1),
}
CORRELATION_COLUMNS = ("sequence", "question", "x", "y", "subset", "controls", "n", "rho", "ci_low", "ci_high",
                       "block_length")
TERTILE_COLUMNS = ("sequence", "question", "subset", "n", "n_fast", "n_slow", "median_fast", "median_slow",
                   "difference", "ci_low", "ci_high", "block_length")
SPIKE_COLUMNS = ("sequence", "frame_index", "rel_d", "neighbour_median", "excess", "near_increase", "mid_increase",
                 "far_increase", "largest_increase", "angular_speed_dps", "sharpness_rel")
SPIKE_COMPARISON_COLUMNS = ("sequence", "covariate", "n_spikes", "n_other", "median_spikes", "median_other",
                            "difference", "ci_low", "ci_high", "block_length")
LINE_COLOR = "#2a78d6"
SPIKE_COLOR = "#eb6834"
INK = "#52514e"
GRID = "#e4e3df"


@dataclass(frozen=True)
class Correlation:
    question: str
    x: str
    y: str
    subset: str = ALL
    controls: tuple[str, ...] = ()


@dataclass(frozen=True)
class Unit:
    """One sequence, or the fr1 sequences pooled."""
    label: str
    sequences: tuple[str, ...]


def correlations(with_focal: bool) -> list[Correlation]:
    specs = []
    for question, subset in (("Q1", ALL), ("Q2", SMALL_OFFSET)):
        specs += [Correlation(question, "angular_speed_dps", "rel_d", subset),
                  Correlation(question, "sharpness_rel", "rel_d", subset)]
    specs.append(Correlation("Q2", "sharpness_rel", "rel_d", ALL, ("expected_misalignment_px", "edge_density")))
    specs += [Correlation("Q3", x, "rel_d") for x in ("edge_density", "depth_range_ratio", "near_border_fraction")]
    specs.append(Correlation("Q3", "near_border_fraction", "rel_d_near"))
    if with_focal:
        specs.append(Correlation("Q4", "abs_log_focal_ratio", "rel_d", RELIABLE_FOCAL))
    return specs


def short(name: str) -> str:
    return name.replace("rgbd_dataset_freiburg", "fr")


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found")
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{path} has no rows")
    missing = [key for key in KEYS if key not in rows[0]]
    if missing:
        raise ValueError(f"{path} lacks key columns {missing}")
    return rows


def keyed(rows: list[dict[str, str]], source: str) -> dict[tuple[str, int], dict[str, str]]:
    by_key: dict[tuple[str, int], dict[str, str]] = {}
    for row in rows:
        key = (row["sequence"], int(row["frame_index"]))
        if key in by_key:
            raise ValueError(f"{source} has duplicate key {key}")
        by_key[key] = row
    return by_key


def require_same_keys(left: dict, right: dict, left_name: str, right_name: str) -> None:
    only_left, only_right = sorted(left.keys() - right.keys()), sorted(right.keys() - left.keys())
    if only_left or only_right:
        raise ValueError(f"key mismatch between {left_name} and {right_name}: {len(only_left)} keys only in "
                         f"{left_name} (e.g. {only_left[:5]}), {len(only_right)} only in {right_name} "
                         f"(e.g. {only_right[:5]})")


def join(per_frame: list[dict[str, str]], covariates: list[dict[str, str]],
         focal_split: list[dict[str, str]] | None = None) -> dict[str, Table]:
    """Tables per sequence (in per_frame order), sorted by frame_index; per_frame's columns win on name clashes."""
    sources = {"per_frame.csv": keyed(per_frame, "per_frame.csv"), "covariates.csv": keyed(covariates, "covariates.csv")}
    if focal_split is not None:
        sources["focal_split.csv"] = keyed(focal_split, "focal_split.csv")
    names = list(sources)
    base = sources[names[0]]
    for name in names[1:]:
        require_same_keys(base, sources[name], names[0], name)
    for key, row in base.items():
        other = sources["covariates.csv"][key]["rgb_timestamp"]
        if abs(float(row["rgb_timestamp"]) - float(other)) > 1e-6:
            raise ValueError(f"{key}: rgb_timestamp {row['rgb_timestamp']} in per_frame.csv but {other} in covariates.csv")

    merged = {key: {k: v for source in reversed(names) for k, v in sources[source][key].items()} for key in base}
    tables = {}
    for sequence in dict.fromkeys(row["sequence"] for row in per_frame):
        rows = sorted((row for (s, _), row in merged.items() if s == sequence), key=lambda r: int(r["frame_index"]))
        tables[sequence] = {column: np.array([float(r[column]) if r[column] else np.nan for r in rows])
                            for column in rows[0] if column not in NON_NUMERIC}
        if focal_split is not None:
            tables[sequence]["abs_log_focal_ratio"] = np.abs(np.log(tables[sequence]["focal_ratio"]))
    return tables


def units(tables: dict[str, Table]) -> list[Unit]:
    fr1 = tuple(name for name in tables if "freiburg1" in name)
    return [Unit(name, (name,)) for name in tables] + ([Unit(POOLED, fr1)] if fr1 else [])


def block_indices(tables: dict[str, Table], block_length: int) -> dict[str, np.ndarray]:
    indices = {}
    for name, table in tables.items():
        rng = np.random.default_rng([SEED, block_length, zlib.crc32(name.encode())])
        indices[name] = moving_block_indices(len(table["frame_index"]), block_length, RESAMPLES, rng)
    return indices


def complete(table: Table, subset: str, columns: tuple[str, ...]) -> np.ndarray:
    mask = SUBSETS[subset](table)
    for column in columns:
        mask &= ~np.isnan(table[column])
    return mask


def correlation_statistic(tables: list[Table], spec: Correlation) -> Callable[[list[np.ndarray]], float]:
    masks = [complete(t, spec.subset, (spec.x, spec.y, *spec.controls)) for t in tables]

    def statistic(indices: list[np.ndarray]) -> float:
        groups = []
        for table, mask, idx in zip(tables, masks, indices):
            rows = idx[mask[idx]]
            controls = np.column_stack([table[c][rows] for c in spec.controls]) if spec.controls else \
                np.empty((len(rows), 0))
            groups.append((table[spec.x][rows], table[spec.y][rows], controls))
        return pooled_partial_spearman(groups)

    return statistic


def tertile_statistic(tables: list[Table], subset: str) -> Callable[[list[np.ndarray]], float]:
    masks = [complete(t, subset, ("angular_speed_dps", "rel_d")) for t in tables]

    def statistic(indices: list[np.ndarray]) -> float:
        return float(np.mean([fast_slow_median_difference(t["angular_speed_dps"][idx[m[idx]]], t["rel_d"][idx[m[idx]]])
                              for t, m, idx in zip(tables, masks, indices)]))

    return statistic


def eligible(unit: Unit, tables: dict[str, Table], subset: str, columns: tuple[str, ...],
             notes: list[str], what: str) -> tuple[str, ...]:
    counts = {name: int(complete(tables[name], subset, columns).sum()) for name in unit.sequences}
    kept = tuple(name for name, n in counts.items() if n >= MIN_SUBSET_N)
    if unit.label == POOLED:
        if len(kept) < 2:
            notes.append(f"{what}, {POOLED}: skipped, fewer than 2 fr1 sequences have n >= {MIN_SUBSET_N}")
            return ()
        if len(kept) < len(unit.sequences):
            notes.append(f"{what}, {POOLED}: pools only {', '.join(map(short, kept))}")
    elif not kept and subset != ALL:
        notes.append(f"{what}, {short(unit.label)}: skipped, n = {counts[unit.label]} < {MIN_SUBSET_N}")
    return kept


def estimate(statistic: Callable[[list[np.ndarray]], float], names: tuple[str, ...], tables: dict[str, Table],
             indices: dict[int, dict[str, np.ndarray]]) -> tuple[float, dict[int, tuple[float, float]]]:
    point = statistic([np.arange(len(tables[name]["frame_index"])) for name in names])
    cis = {L: percentile_ci(bootstrap(statistic, [by_name[name] for name in names])) for L, by_name in indices.items()}
    return point, cis


def run_correlations(tables: dict[str, Table], specs: list[Correlation], indices: dict[int, dict[str, np.ndarray]],
                     notes: list[str]) -> list[dict[str, Any]]:
    rows = []
    for spec in specs:
        for unit in units(tables):
            what = f"{spec.question} {spec.y} vs {spec.x} [{spec.subset}]"
            names = eligible(unit, tables, spec.subset, (spec.x, spec.y, *spec.controls), notes, what)
            if not names:
                continue
            statistic = correlation_statistic([tables[name] for name in names], spec)
            rho, cis = estimate(statistic, names, tables, indices)
            n = sum(int(complete(tables[name], spec.subset, (spec.x, spec.y, *spec.controls)).sum()) for name in names)
            rows += [{"sequence": unit.label, "question": spec.question, "x": spec.x, "y": spec.y,
                      "subset": spec.subset, "controls": " ".join(spec.controls), "n": n, "rho": rho,
                      "ci_low": low, "ci_high": high, "block_length": L} for L, (low, high) in cis.items()]
    return rows


def run_tertiles(tables: dict[str, Table], indices: dict[int, dict[str, np.ndarray]],
                 notes: list[str]) -> list[dict[str, Any]]:
    rows = []
    for question, subset in (("Q1", ALL), ("Q2", SMALL_OFFSET)):
        for unit in units(tables):
            names = eligible(unit, tables, subset, ("angular_speed_dps", "rel_d"), notes,
                             f"{question} fast vs slow third [{subset}]")
            if not names:
                continue
            difference, cis = estimate(tertile_statistic([tables[n] for n in names], subset), names, tables, indices)
            thirds = [n // 3 for n in (int(complete(tables[name], subset, ("angular_speed_dps", "rel_d")).sum())
                                       for name in names)]
            medians = tertile_medians(tables[names[0]], subset) if len(names) == 1 else (math.nan, math.nan)
            rows += [{"sequence": unit.label, "question": question, "subset": subset,
                      "n": sum(int(complete(tables[n], subset, ("angular_speed_dps", "rel_d")).sum()) for n in names),
                      "n_fast": sum(thirds), "n_slow": sum(thirds), "median_fast": medians[0],
                      "median_slow": medians[1], "difference": difference, "ci_low": low, "ci_high": high,
                      "block_length": L} for L, (low, high) in cis.items()]
    return rows


def tertile_medians(table: Table, subset: str) -> tuple[float, float]:
    mask = complete(table, subset, ("angular_speed_dps", "rel_d"))
    speed, rel_d = table["angular_speed_dps"][mask], table["rel_d"][mask]
    order, third = np.argsort(speed, kind="stable"), len(speed) // 3
    return float(np.median(rel_d[order[-third:]])), float(np.median(rel_d[order[:third]]))


def find_spikes(table: Table) -> np.ndarray:
    return detect_spikes(table["rel_d"], SPIKE_RADIUS, SPIKE_MIN_EXCESS, SPIKE_MIN_RELATIVE_EXCESS)


def spike_rows(name: str, table: Table) -> list[dict[str, Any]]:
    spikes = np.flatnonzero(find_spikes(table))
    medians = neighbour_medians(table["rel_d"], SPIKE_RADIUS)
    increases = {third: table[f"rel_d_{third}"] - neighbour_medians(table[f"rel_d_{third}"], SPIKE_RADIUS)
                 for third in THIRDS}
    rows = []
    for i in spikes:
        by_third = {third: float(increases[third][i]) for third in THIRDS}
        finite = {k: v for k, v in by_third.items() if not math.isnan(v)}
        rows.append({"sequence": name, "frame_index": int(table["frame_index"][i]), "rel_d": float(table["rel_d"][i]),
                     "neighbour_median": float(medians[i]), "excess": float(table["rel_d"][i] - medians[i]),
                     **{f"{t}_increase": v for t, v in by_third.items()},
                     "largest_increase": max(finite, key=finite.get) if finite else "",
                     "angular_speed_dps": float(table["angular_speed_dps"][i]),
                     "sharpness_rel": float(table["sharpness_rel"][i])})
    return rows


def spike_difference_statistic(table: Table, spikes: np.ndarray, column: str) -> Callable[[list[np.ndarray]], float]:
    # Spike labels are fixed from the original series and carried along with each resampled frame.
    def statistic(indices: list[np.ndarray]) -> float:
        values, labels = table[column][indices[0]], spikes[indices[0]]
        keep = ~np.isnan(values)
        if not (labels & keep).any() or not (~labels & keep).any():
            return math.nan
        return float(np.median(values[labels & keep]) - np.median(values[~labels & keep]))

    return statistic


def run_spike_comparisons(tables: dict[str, Table], indices: dict[int, dict[str, np.ndarray]]) -> list[dict[str, Any]]:
    rows = []
    for name, table in tables.items():
        spikes = find_spikes(table)
        for column in ("angular_speed_dps", "sharpness_rel"):
            difference, cis = estimate(spike_difference_statistic(table, spikes, column), (name,), tables, indices)
            values = table[column]
            spiked, other = values[spikes & ~np.isnan(values)], values[~spikes & ~np.isnan(values)]
            rows += [{"sequence": name, "covariate": column, "n_spikes": len(spiked), "n_other": len(other),
                      "median_spikes": float(np.median(spiked)) if len(spiked) else math.nan,
                      "median_other": float(np.median(other)), "difference": difference, "ci_low": low,
                      "ci_high": high, "block_length": L} for L, (low, high) in cis.items()]
    return rows


def ci_class(low: float, high: float) -> str:
    if math.isnan(low) or math.isnan(high):
        return "n/a"
    return "> 0" if low > 0 else "< 0" if high < 0 else "spans 0"


def robustness(rows: list[dict[str, Any]], identity: tuple[str, ...], what: str) -> tuple[int, list[str]]:
    by_estimate: dict[tuple, dict[int, str]] = {}
    for row in rows:
        by_estimate.setdefault(tuple(row[k] for k in identity), {})[row["block_length"]] = ci_class(row["ci_low"],
                                                                                                    row["ci_high"])
    changed = []
    for key, classes in by_estimate.items():
        if any(classes[L] != classes[DEFAULT_BLOCK_LENGTH] for L in ROBUSTNESS_BLOCK_LENGTHS):
            label = ", ".join(short(str(k)) for k in key if k != "")
            changed.append(f"| {what} | {label} | " + " | ".join(classes[L] for L in sorted(classes)) + " |")
    return len(by_estimate), changed


def cell(value: Any) -> Any:
    if isinstance(value, float):
        return "" if math.isnan(value) else repr(value)
    return value


def write_csv(path: Path, columns: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".csv.partial")
    with open(partial, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: cell(row[key]) for key in columns} for row in rows)
    os.replace(partial, path)


def fmt(value: float, digits: int = 3) -> str:
    return "n/a" if math.isnan(value) else f"{value:.{digits}f}"


def ci(row: dict[str, Any], digits: int = 3) -> str:
    return f"[{fmt(row['ci_low'], digits)}, {fmt(row['ci_high'], digits)}]"


def default(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row["block_length"] == DEFAULT_BLOCK_LENGTH]


def correlation_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = ["| unit | y | x | subset | controls | n | rho | 95% CI |", "|---|---|---|---|---|---|---|---|"]
    lines += [f"| {short(r['sequence'])} | {r['y']} | {r['x']} | {r['subset']} | {r['controls'] or '-'} | {r['n']} | "
              f"{fmt(r['rho'])} | {ci(r)} |" for r in rows]
    return lines


def tertile_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = ["| unit | subset | n | n per third | median rel_d fast | median rel_d slow | fast − slow | 95% CI |",
             "|---|---|---|---|---|---|---|---|"]
    lines += [f"| {short(r['sequence'])} | {r['subset']} | {r['n']} | {r['n_fast']} | {fmt(r['median_fast'], 2)} | "
              f"{fmt(r['median_slow'], 2)} | {fmt(r['difference'], 2)} | {ci(r, 2)} |" for r in rows]
    return lines


def subset_counts(tables: dict[str, Table], subset: str) -> str:
    return ", ".join(f"{short(name)} {int(SUBSETS[subset](t).sum())}/{len(t['frame_index'])}"
                     for name, t in tables.items())


def summary(tables: dict[str, Table], correlation_rows: list[dict[str, Any]], tertile_rows: list[dict[str, Any]],
            spikes: list[dict[str, Any]], spike_comparisons: list[dict[str, Any]], notes: list[str],
            focal_path: Path | None, inputs: list[Path]) -> str:
    corr, tert = default(correlation_rows), default(tertile_rows)

    def notes_for(question: str) -> list[str]:
        return [f"- {note}" for note in notes if note.startswith(question)]

    lines = [
        "# Analysis summary", "",
        "Inputs: " + ", ".join(f"`{p}`" for p in inputs) + ".", "",
        f"Spearman rho with average ranks. Partial rho: correlation of rank residuals after linear regression on the "
        f"controls' ranks. {POOLED}: ranks and residuals computed within each fr1 sequence, then pooled; for fast vs "
        f"slow, the mean of the per-sequence differences. 95% percentile CIs from a moving-block bootstrap within each "
        f"sequence (block {DEFAULT_BLOCK_LENGTH} frames, {RESAMPLES} resamples, seed {SEED}; subsets are applied "
        f"after resampling). Fast / slow: fastest / slowest third of frames by angular_speed_dps. Subsets with "
        f"n < {MIN_SUBSET_N} are skipped.", "",
        "## Q1 Motion and blur", "", *correlation_table([r for r in corr if r["question"] == "Q1"]), "",
        *tertile_table([r for r in tert if r["question"] == "Q1"]), "",
        "## Q2 Blur vs GT timing", "", f"Frames with depth_offset_ms < 5: {subset_counts(tables, SMALL_OFFSET)}.", "",
        *notes_for("Q2"), "", *correlation_table([r for r in corr if r["question"] == "Q2"]), "",
        *tertile_table([r for r in tert if r["question"] == "Q2"]), "",
        "## Q3 Scene content", "", *correlation_table([r for r in corr if r["question"] == "Q3"]), "",
        "## Q4 Implied focal length", "",
    ]
    if focal_path is None:
        lines += ["Skipped: focal_split.csv not found.", ""]
    else:
        lines += [f"From `{focal_path}`. Frames with focal_split_logdiff < ln(1.1): "
                  f"{subset_counts(tables, RELIABLE_FOCAL)}.", "", *notes_for("Q4"),
                  "", *correlation_table([r for r in corr if r["question"] == "Q4"]), ""]
    lines += spike_section(tables, spikes, default(spike_comparisons))
    lines += robustness_section(correlation_rows, tertile_rows, spike_comparisons)
    return "\n".join(line for i, line in enumerate(lines) if line or (i and lines[i - 1])) + "\n"


def spike_section(tables: dict[str, Table], spikes: list[dict[str, Any]],
                  comparisons: list[dict[str, Any]]) -> list[str]:
    lines = ["## Q5 Temporal spikes", "",
             f"Spike: rel_d exceeds the median rel_d of the frames within ±{SPIKE_RADIUS} positions (itself excluded) "
             f"by more than max({SPIKE_MIN_EXCESS} points, {SPIKE_MIN_RELATIVE_EXCESS:.0%}). Largest increase: the "
             "depth third whose rel_d exceeds its own neighbour median the most. Differences use spike labels fixed "
             "from the original series.", "",
             "| sequence | frames | spikes | rate | largest increase near / mid / far | spike frames |",
             "|---|---|---|---|---|---|"]
    for name, table in tables.items():
        rows = [r for r in spikes if r["sequence"] == name]
        frames = int((~np.isnan(table["rel_d"])).sum())
        counts = " / ".join(str(sum(r["largest_increase"] == t for r in rows)) for t in THIRDS)
        indices = ", ".join(str(r["frame_index"]) for r in rows) or "-"
        lines.append(f"| {short(name)} | {frames} | {len(rows)} | {len(rows) / frames:.1%} | {counts} | {indices} |")
    lines += ["", "| sequence | covariate | median at spikes | median elsewhere | spikes − elsewhere | 95% CI |",
              "|---|---|---|---|---|---|"]
    lines += [f"| {short(r['sequence'])} | {r['covariate']} | {fmt(r['median_spikes'])} | {fmt(r['median_other'])} | "
              f"{fmt(r['difference'])} | {ci(r)} |" for r in comparisons]
    return lines + [""]


def robustness_section(correlation_rows: list[dict[str, Any]], tertile_rows: list[dict[str, Any]],
                       spike_comparisons: list[dict[str, Any]]) -> list[str]:
    checks = [robustness(correlation_rows, ("question", "sequence", "y", "x", "subset", "controls"), "correlation"),
              robustness(tertile_rows, ("question", "sequence", "subset"), "fast − slow"),
              robustness(spike_comparisons, ("sequence", "covariate"), "spikes − elsewhere")]
    total, changed = sum(n for n, _ in checks), [line for _, lines in checks for line in lines]
    lengths = sorted((DEFAULT_BLOCK_LENGTH, *ROBUSTNESS_BLOCK_LENGTHS))
    lines = ["## Robustness to block length", "",
             f"CI position relative to zero at block lengths {', '.join(map(str, lengths))}: {len(changed)} of {total} "
             f"estimates differ from block {DEFAULT_BLOCK_LENGTH} at block "
             f"{' or '.join(map(str, ROBUSTNESS_BLOCK_LENGTHS))}.", ""]
    if changed:
        lines += ["| estimate | identity | " + " | ".join(f"block {L}" for L in lengths) + " |",
                  "|---|---|" + "---|" * len(lengths), *changed, ""]
    return lines


def style(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=INK, labelsize=8)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_scatter(name: str, table: Table, correlation_rows: list[dict[str, Any]], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    for ax, x in zip(axes, ("angular_speed_dps", "sharpness_rel")):
        keep = ~(np.isnan(table[x]) | np.isnan(table["rel_d"]))
        n = int(keep.sum())
        ax.scatter(average_ranks(table[x][keep]) / n, average_ranks(table["rel_d"][keep]) / n, s=10,
                   color=LINE_COLOR, alpha=0.6, linewidths=0)
        row = next(r for r in default(correlation_rows) if r["sequence"] == name and r["x"] == x
                   and r["question"] == "Q1")
        ax.set_title(f"rho {fmt(row['rho'], 2)} {ci(row, 2)}", fontsize=9, color=INK)
        ax.set_xlabel(f"rank of {x} / n", color=INK)
        style(ax)
    axes[0].set_ylabel("rank of rel_d / n", color=INK)
    fig.suptitle(f"{short(name)}: rel_d vs motion and sharpness (rank-transformed, n = {n})", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_forest(correlation_rows: list[dict[str, Any]], path: Path) -> None:
    rows = default(correlation_rows)
    labels, y, positions, headers = [], 0.0, [], []
    for question in dict.fromkeys(r["question"] for r in rows):
        headers.append((y, question))
        y += 1
        for r in (r for r in rows if r["question"] == question):
            controls = f" | ctrl {', '.join(r['controls'].split())}" if r["controls"] else ""
            subset = "" if r["subset"] == ALL else f" | {r['subset']}"
            labels.append(f"{r['y']} vs {r['x']}{subset}{controls} | {short(r['sequence'])}")
            positions.append(y)
            y += 1
        y += 0.5
    fig, ax = plt.subplots(figsize=(10, 0.24 * y + 1))
    rho = np.array([r["rho"] for r in rows])
    low, high = np.array([r["ci_low"] for r in rows]), np.array([r["ci_high"] for r in rows])
    pooled = np.array([r["sequence"] == POOLED for r in rows])
    ax.hlines(positions, low, high, color=LINE_COLOR, linewidth=2)
    for mask, marker in ((~pooled, "o"), (pooled, "D")):
        ax.scatter(rho[mask], np.array(positions)[mask], marker=marker, s=28, color=LINE_COLOR, zorder=3,
                   edgecolors="white", linewidths=1.0, label=POOLED if marker == "D" else "single sequence")
    ax.axvline(0, color=INK, linewidth=0.8)
    ax.set_yticks(positions, labels, fontsize=7)
    for header_y, question in headers:
        ax.text(-1.0, header_y, question, fontsize=9, fontweight="bold", color=INK, va="center")
    ax.set_xlim(-1, 1)
    ax.set_ylim(y, -1)
    ax.set_xlabel(f"Spearman rho with 95% block-bootstrap CI (block {DEFAULT_BLOCK_LENGTH})", color=INK)
    ax.legend(loc="lower left", frameon=False, fontsize=8)
    style(ax)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_spikes(name: str, table: Table, path: Path) -> None:
    spikes = find_spikes(table)
    fig, ax = plt.subplots(figsize=(12, 3.2))
    frames, rel_d = table["frame_index"], table["rel_d"]
    ax.plot(frames, rel_d, color=LINE_COLOR, linewidth=1.2, label="rel_d")
    ax.plot(frames, neighbour_medians(rel_d, SPIKE_RADIUS), color=INK, linewidth=0.8, alpha=0.6,
            label=f"median of ±{SPIKE_RADIUS} neighbours")
    ax.scatter(frames[spikes], rel_d[spikes], s=30, color=SPIKE_COLOR, zorder=3, edgecolors="white",
               linewidths=1.0, label=f"spike ({int(spikes.sum())})")
    ax.set_xlabel("frame index", color=INK)
    ax.set_ylabel("Rel_d (x100)", color=INK)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    ax.set_title(short(name), fontsize=10)
    style(ax)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-frame", type=Path, default=Path("results/eval/baseline/per_frame.csv"))
    parser.add_argument("--covariates", type=Path, default=Path("results/covariates/covariates.csv"))
    parser.add_argument("--focal-split", type=Path, default=Path("results/eval/baseline/focal_split.csv"),
                        help="optional; Q4 is skipped if the file does not exist")
    parser.add_argument("--out", type=Path, default=Path("results/analysis"))
    parser.add_argument("--figures", type=Path, default=Path("report/figures/analysis"))
    args = parser.parse_args()

    focal_path = args.focal_split if args.focal_split.is_file() else None
    print(f"focal split: {focal_path or f'{args.focal_split} not found, skipping Q4'}")
    tables = join(read_rows(args.per_frame), read_rows(args.covariates),
                  read_rows(focal_path) if focal_path else None)
    lengths = sorted((DEFAULT_BLOCK_LENGTH, *ROBUSTNESS_BLOCK_LENGTHS))
    indices = {L: block_indices(tables, L) for L in lengths}
    notes: list[str] = []

    correlation_rows = run_correlations(tables, correlations(focal_path is not None), indices, notes)
    tertile_rows = run_tertiles(tables, indices, notes)
    spikes = [row for name, table in tables.items() for row in spike_rows(name, table)]
    spike_comparisons = run_spike_comparisons(tables, indices)

    write_csv(args.out / "correlations.csv", CORRELATION_COLUMNS, correlation_rows)
    write_csv(args.out / "tertile_comparison.csv", TERTILE_COLUMNS, tertile_rows)
    write_csv(args.out / "spikes.csv", SPIKE_COLUMNS, spikes)
    write_csv(args.out / "spike_comparison.csv", SPIKE_COMPARISON_COLUMNS, spike_comparisons)
    inputs = [args.per_frame, args.covariates, *([focal_path] if focal_path else [])]
    text = summary(tables, correlation_rows, tertile_rows, spikes, spike_comparisons, notes, focal_path, inputs)
    (args.out / "summary.md").write_text(text)

    args.figures.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        plot_scatter(name, table, correlation_rows, args.figures / f"{short(name)}_scatter.png")
        plot_spikes(name, table, args.figures / f"{short(name)}_spikes.png")
    plot_forest(correlation_rows, args.figures / "correlations_forest.png")
    print(f"wrote {args.out} and {args.figures}")


if __name__ == "__main__":
    main()
