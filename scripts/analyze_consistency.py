"""Analysis of the temporal-consistency pairs: headline M2 / M2 floor curves, error persistence (if available), drivers
of excess inconsistency, M2 by depth third and region-level flicker events, with moving-block bootstrap CIs over pairs.
Reads pairs_<sequence>.csv, covariates.csv, per_frame.csv and optionally persistence_<sequence>.csv."""

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
    detect_spikes, moving_block_indices, neighbour_medians, percentile_ci, pooled_partial_spearman,
)
from pim.consistency.evaluation import read_pairs  # noqa: E402
from pim.consistency.persistence import REGIONS as PERSISTENCE_REGIONS  # noqa: E402
from pim.consistency.persistence import read_rows as read_persistence_rows  # noqa: E402

Series = dict[str, np.ndarray]
SeriesKey = tuple[str, float]
GAPS = (33.0, 100.0, 333.0)
DRIVER_GAPS = (33.0, 100.0)
DEFAULT_BLOCK_LENGTH = 15
BLOCK_LENGTHS = (8, 15, 30)
RESAMPLES = 2000
SEED = 20261013
POOLED = "pooled_fr1"
THIRDS = ("near", "mid", "far")
HEADLINE = {"m2": "m2_visible_median", "m2_floor": "m2_floor_visible_median", "m1": "m1_visible_median",
            "gt_floor": "gt_floor_depth_visible_median"}
FRAME_COVARIATES = ("sharpness_rel", "angular_speed_dps", "edge_density", "depth_range_ratio",
                    "expected_misalignment_px")
DRIVER_XS = (*FRAME_COVARIATES, "actual_gap_ms", "rel_d_j")
PARTIAL = ("sharpness_rel", ("edge_density", "expected_misalignment_px"))
PERSISTENCE_STATS = ("median_abs_e_i", "median_abs_e_j", "median_abs_diff", "pearson", "spearman",
                     "persistent_fraction")
EVENT_RADIUS = 5
EVENT_MIN_EXCESS = 0.5
EVENT_MIN_RELATIVE_EXCESS = 1.0
EVENT_GAP = 33.0
COINCIDENCE_PAIRS = 1
FR3 = "rgbd_dataset_freiburg3_long_office_household"
FR3_SPIKE_RANGE = (758, 776)
FR3_SPIKE_FRAMES = (758, 761, 767, 768, 770, 776)
ESTIMATE_COLUMNS = ("sequence", "gap_ms", "quantity", "n", "value", "ci_low", "ci_high", "block_length")
DRIVER_COLUMNS = ("sequence", "gap_ms", "y", "x", "controls", "n", "rho", "ci_low", "ci_high", "block_length")
EVENT_METRICS = ("m2", "m2_floor")
EVENT_COLUMNS = ("sequence", "metric", "third", "frame_i", "frame_j", "value", "neighbour_median", "excess",
                 "floor_event_nearby")
EVENT_COUNT_COLUMNS = ("sequence", "third", "pairs", "events", "rate", "floor_events", "coincident",
                       "not_explained_by_floor")
PAIR_DRIVER_COLUMNS = ("sequence", "gap_ms", "frame_i", "frame_j", "excess", "ratio",
                       *(f"{x}" for x in DRIVER_XS))
SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
THIRD_COLORS = dict(zip(THIRDS, ("#2a78d6", "#eb6834", "#1baf7a")))
METRIC_COLORS = dict(zip(HEADLINE, ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")))
INK = "#52514e"
MUTED = "#8a8984"
GRID = "#e4e3df"


@dataclass(frozen=True)
class Estimate:
    n: int
    value: float
    cis: dict[int, tuple[float, float]]


def short(name: str) -> str:
    return name.replace("rgbd_dataset_freiburg", "fr").replace("_long_office_household", "")


def number(cell: str) -> float:
    return float(cell) if cell else math.nan


def read_csv(path: Path, keys: tuple[str, ...]) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found")
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{path} has no rows")
    missing = [key for key in keys if key not in rows[0]]
    if missing:
        raise ValueError(f"{path} lacks columns {missing}")
    return rows


def frame_table(rows: list[dict[str, str]], columns: tuple[str, ...], source: str) -> dict[tuple[str, int], dict]:
    table: dict[tuple[str, int], dict[str, float]] = {}
    for row in rows:
        key = (row["sequence"], int(row["frame_index"]))
        if key in table:
            raise ValueError(f"{source} has duplicate key {key}")
        table[key] = {column: number(row[column]) for column in columns}
    return table


def load_pairs(consistency_dir: Path) -> dict[str, list[dict[str, str]]]:
    paths = sorted(consistency_dir.glob("pairs_*.csv"))
    if not paths:
        raise FileNotFoundError(f"no pairs_<sequence>.csv in {consistency_dir}")
    return {path.stem.removeprefix("pairs_"): read_pairs(path) for path in paths}


def pair_series(pairs: dict[str, list[dict[str, str]]]) -> dict[SeriesKey, Series]:
    """One series per sequence and gap, ordered by frame i, with excess and ratio of M2 over its floor."""
    series = {}
    for name, rows in pairs.items():
        for gap in sorted({float(r["gap_ms"]) for r in rows}):
            at_gap = sorted((r for r in rows if float(r["gap_ms"]) == gap), key=lambda r: int(r["frame_i"]))
            columns = {c: np.array([number(r[c]) for r in at_gap]) for c in at_gap[0]
                       if c not in ("sequence", "gap_ms", "too_few_pixels")}
            with np.errstate(divide="ignore", invalid="ignore"):
                columns["excess"] = columns["m2_visible_median"] - columns["m2_floor_visible_median"]
                floor = columns["m2_floor_visible_median"]
                columns["ratio"] = np.where(floor > 0, columns["m2_visible_median"] / floor, np.nan)
                for third in THIRDS:
                    floor = columns[f"m2_floor_visible_{third}_median"]
                    columns[f"ratio_{third}"] = np.where(floor > 0, columns[f"m2_visible_{third}_median"] / floor,
                                                         np.nan)
            series[(name, gap)] = columns
    return series


def join_frame_values(series: dict[SeriesKey, Series], covariates: dict[tuple[str, int], dict],
                      per_frame: dict[tuple[str, int], dict]) -> None:
    """Adds the mean over frames i and j of each frame covariate, and frame j's rel_d; fails on any missing frame."""
    for (name, gap), s in series.items():
        frames = {int(f) for f in np.r_[s["frame_i"], s["frame_j"]]}
        for source, table in (("covariates.csv", covariates), ("per_frame.csv", per_frame)):
            missing = sorted(f for f in frames if (name, f) not in table)
            if missing:
                raise ValueError(f"{source} lacks {len(missing)} frames of {name} pairs at {gap:g} ms "
                                 f"(e.g. {missing[:5]})")
        for column in FRAME_COVARIATES:
            values_i = np.array([covariates[(name, int(f))][column] for f in s["frame_i"]])
            values_j = np.array([covariates[(name, int(f))][column] for f in s["frame_j"]])
            s[column] = (values_i + values_j) / 2
        s["rel_d_j"] = np.array([per_frame[(name, int(f))]["rel_d"] for f in s["frame_j"]])


def join_persistence(series: dict[SeriesKey, Series], persistence_dir: Path) -> None:
    for name in dict.fromkeys(name for name, _ in series):
        path = persistence_dir / f"persistence_{name}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"{path} not found, but {persistence_dir} exists")
        rows = {(float(r["gap_ms"]), int(r["frame_i"]), int(r["frame_j"])): r for r in read_persistence_rows(path)}
        keys = {(gap, int(i), int(j)) for (n, gap), s in series.items() if n == name
                for i, j in zip(s["frame_i"], s["frame_j"])}
        if keys != set(rows):
            raise ValueError(f"{path.name}: {len(set(rows) - keys)} pairs not in the consistency pairs and "
                             f"{len(keys - set(rows))} consistency pairs missing")
        for (n, gap), s in series.items():
            if n != name:
                continue
            for region in PERSISTENCE_REGIONS:
                for stat in PERSISTENCE_STATS:
                    column = f"{region}_{stat}"
                    s[f"p_{column}"] = np.array([number(rows[(gap, int(i), int(j))][column])
                                                 for i, j in zip(s["frame_i"], s["frame_j"])])


def check_against_summary(series: dict[SeriesKey, Series], summary_path: Path) -> None:
    """The point medians must reproduce consistency_eval.py's summary.csv."""
    if not summary_path.is_file():
        return
    with open(summary_path, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["pixels"] == "visible"]
    for row in rows:
        s = series.get((row["sequence"], float(row["gap_ms"])))
        if s is None:
            continue
        for column, value in (("m2_median", nanmedian(s["m2_visible_median"])),
                              ("m2_over_m2_floor", nanmedian(s["ratio"]))):
            if abs(value - float(row[column])) > 1e-9:
                raise ValueError(f"{row['sequence']} {row['gap_ms']} ms: {column} {value} here but {row[column]} in "
                                 f"{summary_path}")


def nanmedian(values: np.ndarray) -> float:
    kept = values[~np.isnan(values)]
    return float(np.median(kept)) if kept.size else math.nan


class Bootstrap:
    def __init__(self, series: dict[SeriesKey, Series], resamples: int) -> None:
        self.indices: dict[int, dict[SeriesKey, np.ndarray]] = {}
        for L in BLOCK_LENGTHS:
            self.indices[L] = {}
            for (name, gap), s in series.items():
                rng = np.random.default_rng([SEED, L, zlib.crc32(name.encode()), int(round(gap))])
                n = len(s["frame_i"])
                self.indices[L][(name, gap)] = moving_block_indices(n, min(L, n), resamples, rng)

    def estimate(self, keys: list[SeriesKey], series: dict[SeriesKey, Series], n: int,
                 statistic: Callable[[list[np.ndarray]], float]) -> Estimate:
        value = statistic([np.arange(len(series[k]["frame_i"])) for k in keys])
        cis = {}
        for L, by_key in self.indices.items():
            groups = [by_key[k] for k in keys]
            cis[L] = percentile_ci(np.array([statistic([g[b] for g in groups]) for b in range(len(groups[0]))]))
        return Estimate(n, value, cis)


def median_estimate(boot: Bootstrap, series: dict[SeriesKey, Series], key: SeriesKey, column: str) -> Estimate:
    values = series[key][column]
    return boot.estimate([key], series, int((~np.isnan(values)).sum()), lambda idx: nanmedian(values[idx[0]]))


def estimate_rows(sequence: str, gap: float, quantity: str, est: Estimate) -> list[dict[str, Any]]:
    return [{"sequence": sequence, "gap_ms": f"{gap:g}", "quantity": quantity, "n": est.n, "value": est.value,
             "ci_low": low, "ci_high": high, "block_length": L} for L, (low, high) in est.cis.items()]


def medians_by_quantity(boot: Bootstrap, series: dict[SeriesKey, Series],
                        quantities: dict[str, str]) -> list[dict[str, Any]]:
    rows = []
    for key in series:
        for quantity, column in quantities.items():
            rows += estimate_rows(*key, quantity, median_estimate(boot, series, key, column))
    return rows


def headline_rows(boot: Bootstrap, series: dict[SeriesKey, Series]) -> list[dict[str, Any]]:
    return medians_by_quantity(boot, series, {**HEADLINE, "ratio_m2_over_m2_floor": "ratio"})


def thirds_rows(boot: Bootstrap, series: dict[SeriesKey, Series]) -> list[dict[str, Any]]:
    return medians_by_quantity(boot, series, {**{f"{metric}_{third}": f"{metric}_visible_{third}_median"
                                                 for metric in ("m2", "m2_floor") for third in THIRDS},
                                              **{f"ratio_{third}": f"ratio_{third}" for third in THIRDS}})


def persistence_rows(boot: Bootstrap, series: dict[SeriesKey, Series]) -> list[dict[str, Any]]:
    return medians_by_quantity(boot, series, {f"{region}_{stat}": f"p_{region}_{stat}"
                                              for region in PERSISTENCE_REGIONS for stat in PERSISTENCE_STATS})


def units(names: list[str]) -> list[tuple[str, tuple[str, ...]]]:
    fr1 = tuple(name for name in names if "freiburg1" in name)
    return [(name, (name,)) for name in names] + ([(POOLED, fr1)] if len(fr1) >= 2 else [])


def driver_specs() -> list[tuple[str, tuple[str, ...]]]:
    return [(x, ()) for x in DRIVER_XS] + [(PARTIAL[0], PARTIAL[1])]


def driver_rows(boot: Bootstrap, series: dict[SeriesKey, Series]) -> list[dict[str, Any]]:
    names = list(dict.fromkeys(name for name, _ in series))
    rows = []
    for gap in DRIVER_GAPS:
        for label, members in units(names):
            keys = [(name, gap) for name in members if (name, gap) in series]
            if len(keys) != len(members):
                continue
            for x, controls in driver_specs():
                columns = ("excess", x, *controls)
                n = sum(int(np.all([~np.isnan(series[k][c]) for c in columns], axis=0).sum()) for k in keys)

                def statistic(indices: list[np.ndarray], x: str = x, controls: tuple[str, ...] = controls) -> float:
                    groups = []
                    for key, idx in zip(keys, indices):
                        s = series[key]
                        control_values = np.column_stack([s[c][idx] for c in controls]) if controls else \
                            np.empty((len(idx), 0))
                        groups.append((s[x][idx], s["excess"][idx], control_values))
                    return pooled_partial_spearman(groups)

                est = boot.estimate(keys, series, n, statistic)
                rows += [{"sequence": label, "gap_ms": f"{gap:g}", "y": "excess", "x": x,
                          "controls": " ".join(controls), "n": n, "rho": est.value, "ci_low": low, "ci_high": high,
                          "block_length": L} for L, (low, high) in est.cis.items()]
    return rows


def flicker_events(values: np.ndarray) -> np.ndarray:
    return detect_spikes(values, EVENT_RADIUS, EVENT_MIN_EXCESS, EVENT_MIN_RELATIVE_EXCESS)


def near_any(flags: np.ndarray, reference: np.ndarray, radius: int = COINCIDENCE_PAIRS) -> np.ndarray:
    """flags that have a reference flag within ±radius positions."""
    if flags.shape != reference.shape:
        raise ValueError(f"series differ in length: {flags.shape} vs {reference.shape}")
    padded = np.r_[np.zeros(radius, bool), reference.astype(bool), np.zeros(radius, bool)]
    nearby = np.any([padded[k:k + len(reference)] for k in range(2 * radius + 1)], axis=0)
    return flags.astype(bool) & nearby


def event_tables(series: dict[SeriesKey, Series]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Events on M2 and, as a control, on the M2 floor of each third; a model event is explained by the floor when a
    floor event of the same third lies within ±COINCIDENCE_PAIRS pairs."""
    events, counts = [], []
    for (name, gap), s in series.items():
        if gap != EVENT_GAP:
            continue
        for third in THIRDS:
            values = {m: s[f"{m}_visible_{third}_median"] for m in EVENT_METRICS}
            flags = {m: flicker_events(v) for m, v in values.items()}
            explained = near_any(flags["m2"], flags["m2_floor"])
            pairs = int((~np.isnan(values["m2"])).sum())
            counts.append({"sequence": name, "third": third, "pairs": pairs, "events": int(flags["m2"].sum()),
                           "rate": flags["m2"].sum() / pairs if pairs else math.nan,
                           "floor_events": int(flags["m2_floor"].sum()), "coincident": int(explained.sum()),
                           "not_explained_by_floor": int((flags["m2"] & ~explained).sum())})
            for metric in EVENT_METRICS:
                medians = neighbour_medians(values[metric], EVENT_RADIUS)
                events += [{"sequence": name, "metric": metric, "third": third, "frame_i": int(s["frame_i"][k]),
                            "frame_j": int(s["frame_j"][k]), "value": float(values[metric][k]),
                            "neighbour_median": float(medians[k]), "excess": float(values[metric][k] - medians[k]),
                            "floor_event_nearby": bool(explained[k]) if metric == "m2" else ""}
                           for k in np.flatnonzero(flags[metric])]
    return events, counts


def pair_driver_rows(series: dict[SeriesKey, Series]) -> list[dict[str, Any]]:
    rows = []
    for (name, gap), s in series.items():
        for k in range(len(s["frame_i"])):
            rows.append({"sequence": name, "gap_ms": f"{gap:g}", "frame_i": int(s["frame_i"][k]),
                         "frame_j": int(s["frame_j"][k]),
                         **{c: float(s[c][k]) for c in ("excess", "ratio", *DRIVER_XS)}})
    return rows


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


def fmt(value: float, digits: int = 2) -> str:
    return "n/a" if math.isnan(value) else f"{value:.{digits}f}"


def lookup(rows: list[dict[str, Any]], **match: Any) -> dict[int, dict[str, Any]]:
    """Rows of one estimate, by block length."""
    found = {r["block_length"]: r for r in rows if all(r[k] == v for k, v in match.items())}
    if set(found) != set(BLOCK_LENGTHS):
        raise KeyError(f"no complete estimate for {match}")
    return found


def with_ci(by_length: dict[int, dict[str, Any]], value_key: str = "value", digits: int = 2) -> str:
    row = by_length[DEFAULT_BLOCK_LENGTH]
    return f"{fmt(row[value_key], digits)} [{fmt(row['ci_low'], digits)}, {fmt(row['ci_high'], digits)}]"


def excludes_all(by_length: dict[int, dict[str, Any]], reference: float) -> str:
    sides = {("above" if r["ci_low"] > reference else "below" if r["ci_high"] < reference else "spans")
             for r in by_length.values()}
    return "yes" if len(sides) == 1 and "spans" not in sides else "no"


def table(header: list[str], rows: list[list[str]]) -> list[str]:
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header),
            *("| " + " | ".join(row) + " |" for row in rows), ""]


def summary_text(names: list[str], headline: list[dict[str, Any]], persistence: list[dict[str, Any]] | None,
                 drivers: list[dict[str, Any]], thirds: list[dict[str, Any]], events: list[dict[str, Any]],
                 counts: list[dict[str, Any]], inputs: list[Path], resamples: int) -> str:
    lengths = "/".join(map(str, BLOCK_LENGTHS))
    gaps = [f"{g:g}" for g in GAPS]
    lines = [
        "# Consistency analysis summary", "",
        "Inputs: " + ", ".join(f"`{p}`" for p in inputs) + ".", "",
        f"Per-pair values are each pair's median over visible pixels (x100). Median over pairs with 95% percentile CIs "
        f"from a moving-block bootstrap over pairs ordered by frame i, within each sequence and gap (block "
        f"{DEFAULT_BLOCK_LENGTH} shown, {resamples} resamples, seed {SEED}). 'All {lengths}': the CI lies on the same "
        f"side of the reference value (1 for ratios, 0 for correlations) at block lengths {lengths}. Ratio: median of "
        f"per-pair M2 / M2 floor. {POOLED}: ranks and residuals within each fr1 sequence, then pooled.", "",
        "## 1. Headline", "",
    ]
    rows = []
    for name in names:
        for gap in gaps:
            cells = [with_ci(lookup(headline, sequence=name, gap_ms=gap, quantity=q)) for q in HEADLINE]
            ratio = lookup(headline, sequence=name, gap_ms=gap, quantity="ratio_m2_over_m2_floor")
            rows.append([short(name), gap, *cells, with_ci(ratio), excludes_all(ratio, 1.0)])
    lines += table(["sequence", "gap ms", "M2", "M2 floor", "M1", "GT floor", "M2 / M2 floor",
                    f"ratio CI excl. 1, all {lengths}"], rows)

    lines += ["## 2. Persistence", ""]
    if persistence is None:
        lines += ["Not available: no persistence results.", ""]
    else:
        rows = []
        for name in names:
            for gap in gaps:
                get = lambda q: lookup(persistence, sequence=name, gap_ms=gap, quantity=f"visible_{q}")  # noqa: E731
                rows.append([short(name), gap, *(with_ci(get(q)) for q in PERSISTENCE_STATS[:3]),
                             *(with_ci(get(q), digits=3) for q in PERSISTENCE_STATS[3:])])
        lines += table(["sequence", "gap ms", "median abs E_i", "median abs E_j", "median abs (E_i − E_j)",
                        "Pearson", "Spearman", "P"], rows)
        rows = [[short(name), gap, *(with_ci(lookup(persistence, sequence=name, gap_ms=gap,
                                                    quantity=f"{region}_persistent_fraction"), digits=3)
                                     for region in PERSISTENCE_REGIONS)]
                for name in names for gap in gaps]
        lines += ["Persistent fraction P by region (thirds of GT_j depth over visible pixels).", "",
                  *table(["sequence", "gap ms", "visible", "interior", "near", "mid", "far"], rows)]

    lines += ["## 3. Drivers of excess inconsistency", "",
              "Excess = M2 − M2 floor per pair. Covariates: mean over frames i and j; rel_d_j: per-frame Rel_d of "
              f"frame j. Partial: controlling for {' and '.join(PARTIAL[1])}.", ""]
    for gap in (f"{g:g}" for g in DRIVER_GAPS):
        rows = []
        for unit in [*names, POOLED]:
            for x, controls in driver_specs():
                match = {"sequence": unit, "gap_ms": gap, "x": x, "controls": " ".join(controls)}
                if not any(all(r[k] == v for k, v in match.items()) for r in drivers):
                    continue
                by_length = lookup(drivers, **match)
                rows.append([short(unit), x + (" (partial)" if controls else ""),
                             str(by_length[DEFAULT_BLOCK_LENGTH]["n"]), with_ci(by_length, "rho"),
                             excludes_all(by_length, 0.0)])
        lines += [f"Gap {gap} ms.", "", *table(["unit", "x", "n", "rho", f"CI excl. 0, all {lengths}"], rows)]

    lines += ["## 4. M2 by depth third", ""]
    rows = [[short(name), gap,
             *(with_ci(lookup(thirds, sequence=name, gap_ms=gap, quantity=f"m2_{t}")) for t in THIRDS),
             *(fmt(lookup(thirds, sequence=name, gap_ms=gap, quantity=f"m2_floor_{t}")[DEFAULT_BLOCK_LENGTH]["value"])
               for t in THIRDS)]
            for name in names for gap in gaps]
    lines += table(["sequence", "gap ms", "M2 near", "M2 mid", "M2 far", "floor near", "floor mid", "floor far"], rows)
    rows = []
    for name in names:
        for gap in gaps:
            ratios = [lookup(thirds, sequence=name, gap_ms=gap, quantity=f"ratio_{t}") for t in THIRDS]
            rows.append([short(name), gap, *(f"{with_ci(r)} {excludes_all(r, 1.0)}" for r in ratios)])
    lines += [f"Median of per-pair M2 / M2 floor within each third; yes / no: CI excludes 1 at all of {lengths}.", "",
              *table(["sequence", "gap ms", "near", "mid", "far"], rows)]

    lines += ["## 5. Region-level flicker events at 33 ms", "",
              f"Event: a third's per-pair M2 median exceeds the median of the ±{EVENT_RADIUS} neighbouring pairs "
              f"(ordered by time, itself excluded) by more than max({EVENT_MIN_EXCESS} points, "
              f"{EVENT_MIN_RELATIVE_EXCESS:.0%}). Floor control: the same rule on the M2 floor of the same third; "
              f"a model event coincides when a floor event lies on the same pair or within ±{COINCIDENCE_PAIRS} "
              "pair.", ""]
    rows = []
    for name in names:
        for t in THIRDS:
            c = next(c for c in counts if c["sequence"] == name and c["third"] == t)
            rows.append([short(name), t, str(c["pairs"]), f"{c['events']} ({c['rate']:.1%})", str(c["floor_events"]),
                         str(c["coincident"]), str(c["not_explained_by_floor"])])
    lines += table(["sequence", "third", "pairs", "model events", "floor events", "model events coinciding",
                    "model events not explained by floor"], rows)
    model = [e for e in events if e["metric"] == "m2"]

    def pair_list(hits: list[dict[str, Any]]) -> str:
        return ", ".join(f"{e['frame_i']}→{e['frame_j']}" + ("*" if e["floor_event_nearby"] else "")
                         for e in hits) or "-"

    rows = [[short(name), t, pair_list([e for e in model if e["sequence"] == name and e["third"] == t])]
            for name in names for t in THIRDS]
    lines += ["Model event pairs; * coincides with a floor event.", "",
              *table(["sequence", "third", "event pairs (i→j)"], rows)]
    if FR3 in names:
        lo, hi = FR3_SPIKE_RANGE
        rows = []
        for t in THIRDS:
            hits = [e for e in model if e["sequence"] == FR3 and e["third"] == t
                    and (lo <= e["frame_i"] <= hi or lo <= e["frame_j"] <= hi)]
            covered = sorted({f for e in hits for f in (e["frame_i"], e["frame_j"])} & set(FR3_SPIKE_FRAMES))
            rows.append([t, pair_list(hits), ", ".join(map(str, covered)) or "-"])
        lines += [f"fr3 events with frame i or j in {lo}–{hi}; earlier per-frame far-third spike frames: "
                  f"{', '.join(map(str, FR3_SPIKE_FRAMES))}.", "",
                  *table(["third", "event pairs (i→j)", "spike frames in an event pair"], rows)]
    return "\n".join(lines).rstrip() + "\n"


def style(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=8)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def gap_axis(ax: plt.Axes) -> None:
    ax.set_xscale("log")
    ax.set_xticks(GAPS, [f"{g:g}" for g in GAPS])
    ax.minorticks_off()
    ax.set_xlim(26, 420)
    ax.set_xlabel("time gap (ms)", color=INK, fontsize=9)


def curve(rows: list[dict[str, Any]], name: str, quantity: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    found = [lookup(rows, sequence=name, gap_ms=f"{g:g}", quantity=quantity)[DEFAULT_BLOCK_LENGTH] for g in GAPS]
    return tuple(np.array([r[k] for r in found]) for k in ("value", "ci_low", "ci_high"))


def plot_headline(names: list[str], rows: list[dict[str, Any]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    offsets = np.linspace(-0.035, 0.035, len(names))
    ends = []
    for name, color, offset in zip(names, SERIES_COLORS, offsets):
        value, low, high = curve(rows, name, "ratio_m2_over_m2_floor")
        x = np.array(GAPS) * 10 ** offset
        ax.errorbar(x, value, yerr=[value - low, high - value], color=color, linewidth=2, marker="o",
                    markersize=7, markeredgecolor="white", markeredgewidth=1.5, capsize=0, elinewidth=1.5,
                    label=short(name))
        ends.append([value[-1], short(name), color])
    ends.sort()
    for k in range(1, len(ends)):
        ends[k][0] = max(ends[k][0], ends[k - 1][0] + 0.045)
    for y, label, color in ends:
        ax.text(GAPS[-1] * 1.3, y, label, color=INK, fontsize=8, va="center")
        ax.plot([GAPS[-1] * 1.22], [y], marker="s", markersize=5, color=color, linestyle="none")
    ax.axhline(1.0, color=MUTED, linewidth=1, linestyle="--")
    ax.text(28, 1.02, "M2 = M2 floor", color=MUTED, fontsize=8, va="bottom")
    gap_axis(ax)
    ax.set_xlim(26, 720)
    ax.set_ylabel("M2 / M2 floor (median over pairs)", color=INK, fontsize=9)
    ax.set_ylim(0.9, max(2.1, ax.get_ylim()[1]))
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    ax.set_title(f"Frame-to-frame inconsistency above the GT noise floor (95% CI, block {DEFAULT_BLOCK_LENGTH})",
                 fontsize=9, color=INK)
    style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_panels(names: list[str], rows: list[dict[str, Any]], quantities: dict[str, str],
                colors: dict[str, str], ylabel: str, title: str, path: Path) -> None:
    fig, axes = plt.subplots(1, len(names), figsize=(3.0 * len(names), 3.4), sharey=True, squeeze=False)
    for ax, name in zip(axes[0], names):
        for quantity, label in quantities.items():
            value, low, high = curve(rows, name, quantity)
            ax.fill_between(GAPS, low, high, color=colors[quantity], alpha=0.18, linewidth=0)
            ax.plot(GAPS, value, color=colors[quantity], linewidth=2, marker="o", markersize=6,
                    markeredgecolor="white", markeredgewidth=1.2, label=label)
        gap_axis(ax)
        ax.set_title(short(name), fontsize=9, color=INK)
        style(ax)
    axes[0][0].set_ylabel(ylabel, color=INK, fontsize=9)
    axes[0][0].set_ylim(bottom=0)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(quantities), frameon=False, fontsize=8)
    fig.suptitle(title, fontsize=9, color=INK, y=0.02, va="bottom")
    fig.tight_layout(rect=(0, 0.05, 1, 0.92))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_events(name: str, s: Series, events: list[dict[str, Any]], path: Path) -> None:
    fig, axes = plt.subplots(len(THIRDS), 1, figsize=(11, 6.4), sharex=True)
    frames = s["frame_i"]
    for ax, third in zip(axes, THIRDS):
        values = s[f"m2_visible_{third}_median"]
        ax.plot(frames, values, color=THIRD_COLORS[third], linewidth=1.4)
        ax.plot(frames, neighbour_medians(values, EVENT_RADIUS), color=MUTED, linewidth=0.9,
                label=f"median of ±{EVENT_RADIUS} neighbouring pairs")
        hits = [e for e in events if e["sequence"] == name and e["third"] == third and e["metric"] == "m2"]
        for explained in (False, True):
            group = [e for e in hits if e["floor_event_nearby"] == explained]
            ax.scatter([e["frame_i"] for e in group], [e["value"] for e in group], s=40, zorder=3,
                       color="white" if explained else THIRD_COLORS[third], edgecolors=INK, linewidths=1.0,
                       label=f"{'event with floor event ±1 pair' if explained else 'event'} ({len(group)})")
        ax.set_ylabel(f"M2 {third} (x100)", color=INK, fontsize=9)
        ax.set_ylim(bottom=0)
        ax.legend(loc="upper right", frameon=False, fontsize=8)
        style(ax)
        ax.grid(axis="x", visible=False)
    axes[-1].set_xlabel("frame i (pairs at 33 ms)", color=INK, fontsize=9)
    axes[0].set_title(f"{short(name)}: M2 per depth third over time", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--consistency", type=Path, default=Path("results/consistency/baseline"))
    parser.add_argument("--covariates", type=Path, default=Path("results/covariates/covariates.csv"))
    parser.add_argument("--per-frame", type=Path, default=Path("results/eval/baseline/per_frame.csv"))
    parser.add_argument("--persistence", type=Path, default=Path("results/persistence"),
                        help="optional; the persistence section is skipped if the folder does not exist")
    parser.add_argument("--out", type=Path, default=Path("results/consistency_analysis"))
    parser.add_argument("--figures", type=Path, default=Path("report/figures/consistency"))
    parser.add_argument("--resamples", type=int, default=RESAMPLES)
    args = parser.parse_args()
    if args.resamples < 100:
        raise ValueError(f"--resamples must be at least 100, got {args.resamples}")

    series = pair_series(load_pairs(args.consistency))
    missing_gaps = [(n, g) for n in dict.fromkeys(n for n, _ in series) for g in GAPS if (n, g) not in series]
    if missing_gaps:
        raise ValueError(f"pairs CSVs lack sequence/gap combinations {missing_gaps}")
    check_against_summary(series, args.consistency / "summary.csv")
    join_frame_values(series, frame_table(read_csv(args.covariates, ("sequence", "frame_index", *FRAME_COVARIATES)),
                                          FRAME_COVARIATES, "covariates.csv"),
                      frame_table(read_csv(args.per_frame, ("sequence", "frame_index", "rel_d")), ("rel_d",),
                                  "per_frame.csv"))
    with_persistence = args.persistence.is_dir()
    print(f"persistence: {args.persistence if with_persistence else f'{args.persistence} not found, skipping'}")
    if with_persistence:
        join_persistence(series, args.persistence)
    names = list(dict.fromkeys(n for n, _ in series))
    boot = Bootstrap(series, args.resamples)

    headline = headline_rows(boot, series)
    persistence = persistence_rows(boot, series) if with_persistence else None
    drivers = driver_rows(boot, series)
    thirds = thirds_rows(boot, series)
    events, counts = event_tables(series)

    write_csv(args.out / "headline.csv", ESTIMATE_COLUMNS, headline)
    if persistence is not None:
        write_csv(args.out / "persistence.csv", ESTIMATE_COLUMNS, persistence)
    write_csv(args.out / "drivers.csv", DRIVER_COLUMNS, drivers)
    write_csv(args.out / "pair_drivers.csv", PAIR_DRIVER_COLUMNS, pair_driver_rows(series))
    write_csv(args.out / "thirds.csv", ESTIMATE_COLUMNS, thirds)
    write_csv(args.out / "flicker_events.csv", EVENT_COLUMNS, events)
    write_csv(args.out / "flicker_counts.csv", EVENT_COUNT_COLUMNS, counts)
    inputs = [args.consistency, args.covariates, args.per_frame, *([args.persistence] if with_persistence else [])]
    (args.out / "summary.md").write_text(summary_text(names, headline, persistence, drivers, thirds, events, counts,
                                                      inputs, args.resamples))

    args.figures.mkdir(parents=True, exist_ok=True)
    plot_headline(names, headline, args.figures / "headline_ratio.png")
    plot_panels(names, headline, {"m2": "M2", "m2_floor": "M2 floor", "m1": "M1", "gt_floor": "GT floor"},
                METRIC_COLORS, "median over pairs (x100)", f"95% CI band, block {DEFAULT_BLOCK_LENGTH}",
                args.figures / "headline_metrics.png")
    plot_panels(names, thirds, {f"m2_{t}": t for t in THIRDS}, {f"m2_{t}": THIRD_COLORS[t] for t in THIRDS},
                "M2, median over pairs (x100)", "M2 by depth third of GT_j", args.figures / "thirds_m2.png")
    if persistence is not None:
        plot_panels(names, persistence, {f"{r}_persistent_fraction": r.replace("visible_", "")
                                         for r in ("visible", "visible_near", "visible_mid", "visible_far")},
                    {"visible_persistent_fraction": SERIES_COLORS[3],
                     **{f"visible_{t}_persistent_fraction": THIRD_COLORS[t] for t in THIRDS}},
                    "persistent fraction P", "P = 1 − Var(E_i − E_j) / (Var(E_i) + Var(E_j))",
                    args.figures / "persistence_p.png")
    for name in names:
        plot_events(name, series[(name, EVENT_GAP)], events, args.figures / f"flicker_{short(name)}.png")
    print(f"wrote {args.out} and {args.figures}")


if __name__ == "__main__":
    main()
