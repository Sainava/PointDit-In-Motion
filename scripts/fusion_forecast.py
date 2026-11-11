"""Pre-registered forecast of the per-frame error reduction from temporal fusion, derived only from measured persistence.
Reads persistence.csv (P with block-bootstrap CIs, median |E_j|) and the baseline GT floors, writes
results/forecast/fusion_forecast.csv and report/figures/forecast/forecast.png."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ANCHOR_GAPS = (33.0, 100.0, 333.0)
REGION = "visible"
FRAME_MS = 33.0
DIVERSE_MS = 100.0
SHORT_NAMES = {
    "rgbd_dataset_freiburg1_desk": "fr1_desk",
    "rgbd_dataset_freiburg1_room": "fr1_room",
    "rgbd_dataset_freiburg1_360": "fr1_360",
    "rgbd_dataset_freiburg3_long_office_household": "fr3",
}
COLUMNS = ("sequence", "config", "n", "offsets_ms", "pred_std_ratio", "pred_std_ratio_low", "pred_std_ratio_high",
           "pred_rel_d_change_pct", "pred_rel_d_change_pct_low", "pred_rel_d_change_pct_high",
           "pred_std_ratio_no_extra_warp")


def interp_log_gap(gap: float, anchors: Sequence[float], at_zero: float | None) -> float:
    """Linear in log(gap) between anchors, flat beyond the last anchor; below the first anchor linear in gap from
    at_zero at 0 ms, or flat if at_zero is None."""
    gap = abs(float(gap))
    if gap >= ANCHOR_GAPS[-1]:
        return float(anchors[-1])
    if gap <= ANCHOR_GAPS[0]:
        if at_zero is None:
            return float(anchors[0])
        return float(at_zero + (anchors[0] - at_zero) * gap / ANCHOR_GAPS[0])
    return float(np.interp(math.log(gap), np.log(ANCHOR_GAPS), anchors))


def rho_at(gap: float, p_anchors: Sequence[float]) -> float:
    return interp_log_gap(gap, p_anchors, at_zero=1.0)


def floor_at(gap: float, floor_anchors: Sequence[float]) -> float:
    return interp_log_gap(gap, floor_anchors, at_zero=0.0)


def sigma_at(gap: float, sigma_anchors: Sequence[float]) -> float:
    return interp_log_gap(gap, sigma_anchors, at_zero=None)


def std_ratio(offsets: Sequence[float], rho: np.ndarray, warp_rel: np.ndarray) -> float:
    """sqrt((sum_kl rho_kl + sum_k warp_rel_k^2) / N^2) for equal weights; warp_rel is 0 for the target."""
    n = len(offsets)
    return math.sqrt((float(np.sum(rho)) + float(np.sum(np.square(warp_rel)))) / n ** 2)


def forecast_ratio(offsets: Sequence[float], p_anchors: Sequence[float], floor_anchors: Sequence[float] | None,
                   sigma_anchors: Sequence[float] | None) -> float:
    t = np.asarray(offsets, dtype=float)
    rho = np.array([[rho_at(a - b, p_anchors) for b in t] for a in t])
    if floor_anchors is None or sigma_anchors is None:
        warp = np.zeros(len(t))
    else:
        warp = np.array([0.0 if x == 0 else floor_at(x, floor_anchors) / sigma_at(x, sigma_anchors) for x in t])
    return std_ratio(t, rho, warp)


def configurations() -> dict[str, tuple[float, ...]]:
    configs: dict[str, tuple[float, ...]] = {"self_only": (0.0,)}
    for w in (1, 2, 4):
        configs[f"causal_adjacent_w{w}"] = (0.0, *(-FRAME_MS * k for k in range(1, w + 1)))
    for w in (1, 2, 4):
        configs[f"centred_adjacent_w{w}"] = (0.0, *(s * FRAME_MS * k for k in range(1, w + 1) for s in (-1, 1)))
    for w in (1, 2, 3, 4):
        configs[f"causal_diverse_w{w}"] = (0.0, *(-DIVERSE_MS * k for k in range(1, w + 1)))
    for w in (1, 2):
        configs[f"centred_diverse_w{w}"] = (0.0, *(s * DIVERSE_MS * k for k in range(1, w + 1) for s in (-1, 1)))
    return configs


@dataclass(frozen=True)
class SequenceInputs:
    p: tuple[float, ...]
    p_low: tuple[float, ...]
    p_high: tuple[float, ...]
    sigma: tuple[float, ...]
    floor: tuple[float, ...]


def read_inputs(persistence_csv: Path, summary_csv: Path) -> dict[str, SequenceInputs]:
    p: dict[tuple[str, float], list[tuple[float, float, float]]] = {}
    sigma: dict[tuple[str, float], float] = {}
    with persistence_csv.open() as f:
        for row in csv.DictReader(f):
            key = (row["sequence"], float(row["gap_ms"]))
            if row["quantity"] == f"{REGION}_persistent_fraction":
                p.setdefault(key, []).append((float(row["value"]), float(row["ci_low"]), float(row["ci_high"])))
            elif row["quantity"] == f"{REGION}_median_abs_e_j":
                sigma[key] = float(row["value"])
    floor: dict[tuple[str, float], float] = {}
    with summary_csv.open() as f:
        for row in csv.DictReader(f):
            if row["pixels"] == REGION:
                floor[(row["sequence"], float(row["gap_ms"]))] = float(row["gt_floor_depth_median"])
    out = {}
    for seq in sorted({s for s, _ in p}):
        rows = [p[(seq, g)] for g in ANCHOR_GAPS]
        out[seq] = SequenceInputs(
            p=tuple(r[0][0] for r in rows),
            p_low=tuple(min(x[1] for x in r) for r in rows),
            p_high=tuple(max(x[2] for x in r) for r in rows),
            sigma=tuple(sigma[(seq, g)] for g in ANCHOR_GAPS),
            floor=tuple(floor[(seq, g)] for g in ANCHOR_GAPS),
        )
    return out


def build_rows(inputs: dict[str, SequenceInputs]) -> list[dict[str, object]]:
    rows = []
    for seq, x in inputs.items():
        for name, offsets in configurations().items():
            mid = forecast_ratio(offsets, x.p, x.floor, x.sigma)
            low = forecast_ratio(offsets, x.p_low, x.floor, x.sigma)
            high = forecast_ratio(offsets, x.p_high, x.floor, x.sigma)
            rows.append({
                "sequence": seq, "config": name, "n": len(offsets),
                "offsets_ms": " ".join(f"{o:g}" for o in offsets),
                "pred_std_ratio": round(mid, 4), "pred_std_ratio_low": round(low, 4),
                "pred_std_ratio_high": round(high, 4),
                "pred_rel_d_change_pct": round(100 * (mid - 1), 2),
                "pred_rel_d_change_pct_low": round(100 * (low - 1), 2),
                "pred_rel_d_change_pct_high": round(100 * (high - 1), 2),
                "pred_std_ratio_no_extra_warp": round(forecast_ratio(offsets, x.p, None, None), 4),
            })
    return rows


def plot(rows: list[dict[str, object]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(configurations())
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for seq in sorted({str(r["sequence"]) for r in rows}):
        by = {r["config"]: r for r in rows if r["sequence"] == seq}
        y = np.array([by[c]["pred_std_ratio"] for c in names], dtype=float)
        lo = np.array([by[c]["pred_std_ratio_low"] for c in names], dtype=float)
        hi = np.array([by[c]["pred_std_ratio_high"] for c in names], dtype=float)
        line, = ax.plot(range(len(names)), y, marker="o", label=SHORT_NAMES.get(seq, seq))
        ax.fill_between(range(len(names)), lo, hi, color=line.get_color(), alpha=0.15, linewidth=0)
    ax.axhline(1.0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_xticks(range(len(names)), names, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("predicted error std ratio (fused / single)")
    ax.set_title("Pre-registered fusion forecast (band from P confidence limits)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--persistence", type=Path, default=ROOT / "results/consistency_analysis/persistence.csv")
    parser.add_argument("--summary", type=Path, default=ROOT / "results/consistency/baseline/summary.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "results/forecast/fusion_forecast.csv")
    parser.add_argument("--figure", type=Path, default=ROOT / "report/figures/forecast/forecast.png")
    args = parser.parse_args()
    rows = build_rows(read_inputs(args.persistence, args.summary))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    plot(rows, args.figure)
    print(f"wrote {len(rows)} rows to {args.out} and {args.figure}")


if __name__ == "__main__":
    main()
