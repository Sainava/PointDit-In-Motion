"""Plot rel_d and rel_p over time per sequence from per_frame.csv and list the worst frames."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HOLE_FACTOR = 1.5
WORST = 5
LINE_COLOR = "#2a78d6"
HOLE_COLOR = "#eb6834"
INK = "#52514e"


def read_rows(csv_path: Path) -> dict[str, list[dict[str, str]]]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"{csv_path} not found")
    by_sequence: dict[str, list[dict[str, str]]] = defaultdict(list)
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            by_sequence[row["sequence"]].append(row)
    if not by_sequence:
        raise ValueError(f"{csv_path} has no rows")
    return {name: sorted(rows, key=lambda r: int(r["frame_index"])) for name, rows in by_sequence.items()}


def column(rows: list[dict[str, str]], name: str) -> np.ndarray:
    return np.array([float(r[name]) if r[name] else np.nan for r in rows])


def after_hole(gap_ms: np.ndarray) -> np.ndarray:
    """Frames whose gap to the previous kept frame exceeds HOLE_FACTOR x the median gap."""
    median = np.nanmedian(gap_ms)
    return np.nan_to_num(gap_ms, nan=0.0) > HOLE_FACTOR * median


def plot_sequence(name: str, rows: list[dict[str, str]], path: Path) -> None:
    t = column(rows, "rgb_timestamp")
    t = t - t[0]
    holes = after_hole(column(rows, "gap_ms"))
    fig, axes = plt.subplots(2, 1, figsize=(12, 5.5), sharex=True)
    for ax, metric, label in ((axes[0], "rel_d", "Rel_d (x100)"), (axes[1], "rel_p", "Rel_p (x100)")):
        values = column(rows, metric)
        ax.plot(t, values, color=LINE_COLOR, linewidth=1.2)
        ax.scatter(t[holes], values[holes], s=28, color=HOLE_COLOR, zorder=3,
                   edgecolors="white", linewidths=1.0, label=f"after a hole (gap > {HOLE_FACTOR}x median)")
        for x in t[holes]:
            ax.axvline(x, color=HOLE_COLOR, linewidth=0.6, alpha=0.35, zorder=1)
        ax.set_ylabel(label, color=INK)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", color="#e4e3df", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(colors=INK)
    axes[0].legend(loc="upper right", frameon=False, fontsize=9)
    axes[1].set_xlabel("seconds from window start", color=INK)
    skipped = sum(r["skipped"] == "True" for r in rows)
    fig.suptitle(f"{name}: {len(rows)} frames, {int(holes.sum())} after a hole, {skipped} skipped", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def print_worst(name: str, rows: list[dict[str, str]]) -> None:
    scored = [r for r in rows if r["rel_d"]]
    worst = sorted(scored, key=lambda r: float(r["rel_d"]), reverse=True)[:WORST]
    print(f"{name}: {WORST} worst frames by rel_d")
    print(f"  {'frame':>6} {'rel_d':>8} {'rel_p':>8} {'depth_offset_ms':>16} {'gt_valid_fraction':>18}")
    for r in worst:
        print(f"  {int(r['frame_index']):>6} {float(r['rel_d']):>8.2f} {float(r['rel_p']):>8.2f} "
              f"{float(r['depth_offset_ms']):>16.1f} {float(r['gt_valid_fraction']):>18.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    for name, rows in read_rows(args.csv).items():
        path = args.out / f"{name}_metrics.png"
        plot_sequence(name, rows, path)
        print_worst(name, rows)
        print(f"  saved {path}")


if __name__ == "__main__":
    main()
