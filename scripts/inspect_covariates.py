"""Plot covariates against rel_d per sequence, save sharpest/blurriest contact sheets, print the fr1_room 366-380 table."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pim.data.tum import load_rgb  # noqa: E402

ROOM = "rgbd_dataset_freiburg1_room"
CLUSTERS = {"rgbd_dataset_freiburg1_360": (233, 243), ROOM: (366, 380)}
ROOM_TABLE = range(366, 381)
SERIES = (
    ("angular_speed_dps", "angular speed\n(deg/s)"),
    ("sharpness_rel", "sharpness_rel"),
    ("expected_misalignment_px", "expected GT\nmisalignment (px)"),
    ("rel_d", "Rel_d (x100)"),
)
CONTACT_COUNT = 5
LINE_COLOR = "#2a5d8f"
SHADE_COLOR = "0.85"

Key = tuple[str, int]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found")
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def number(value: str) -> float:
    return float(value) if value else math.nan


def join(covariates: list[dict[str, str]], per_frame: list[dict[str, str]], per_frame_path: Path) -> dict[str, list[dict[str, str]]]:
    evaluated: dict[Key, dict[str, str]] = {(r["sequence"], int(r["frame_index"])): r for r in per_frame}
    joined: dict[str, list[dict[str, str]]] = {}
    for row in covariates:
        key = (row["sequence"], int(row["frame_index"]))
        other = evaluated.get(key)
        if other is None:
            raise ValueError(f"{per_frame_path} has no row for {key[0]} frame {key[1]}")
        if other["rgb_timestamp"] != row["rgb_timestamp"]:
            raise ValueError(f"{key[0]} frame {key[1]}: rgb_timestamp {row['rgb_timestamp']} in covariates but "
                             f"{other['rgb_timestamp']} in {per_frame_path}")
        joined.setdefault(key[0], []).append({**row, "rel_d": other["rel_d"],
                                              "eval_depth_offset_ms": other["depth_offset_ms"]})
    return joined


def save_time_series(name: str, rows: list[dict[str, str]], path: Path) -> None:
    times = np.array([float(r["rgb_timestamp"]) for r in rows])
    seconds = times - times[0]
    index = {int(r["frame_index"]): s for r, s in zip(rows, seconds)}
    cluster = CLUSTERS.get(name)
    fig, axes = plt.subplots(len(SERIES), 1, figsize=(12, 9), sharex=True)
    for ax, (column, label) in zip(axes, SERIES):
        ax.plot(seconds, [number(r[column]) for r in rows], color=LINE_COLOR, linewidth=1.2)
        if cluster and cluster[0] in index and cluster[1] in index:
            ax.axvspan(index[cluster[0]], index[cluster[1]], color=SHADE_COLOR, zorder=0)
        ax.set_ylabel(label, fontsize=9)
        ax.grid(axis="y", color="0.92", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xlabel("seconds from window start")
    title = name
    if cluster:
        title += f"  (shaded: frames {cluster[0]}-{cluster[1]})"
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def save_contact_sheet(name: str, rows: list[dict[str, str]], data_root: Path, path: Path) -> None:
    ranked = sorted((r for r in rows if r["sharpness_rel"]), key=lambda r: float(r["sharpness_rel"]))
    if len(ranked) < 2 * CONTACT_COUNT:
        raise ValueError(f"{name}: {len(ranked)} frames with sharpness_rel, need {2 * CONTACT_COUNT}")
    picks = [("sharpest", list(reversed(ranked[-CONTACT_COUNT:]))), ("least sharp", ranked[:CONTACT_COUNT])]
    fig, axes = plt.subplots(2, CONTACT_COUNT, figsize=(3.2 * CONTACT_COUNT, 6.2))
    for row_axes, (label, chosen) in zip(axes, picks):
        for ax, r in zip(row_axes, chosen):
            ax.imshow(load_rgb(data_root / name / r["rgb_file"])[::2, ::2])
            ax.set_title(f"{label}: frame {r['frame_index']}\nsharpness_rel {float(r['sharpness_rel']):.2f}",
                         fontsize=9)
            ax.axis("off")
    fig.suptitle(name)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def print_room_table(rows: list[dict[str, str]]) -> None:
    by_index = {int(r["frame_index"]): r for r in rows}
    missing = [i for i in ROOM_TABLE if i not in by_index]
    if missing:
        raise ValueError(f"{ROOM}: frames {missing} are not in the covariates")
    print(f"\n{ROOM} frames {ROOM_TABLE.start}-{ROOM_TABLE.stop - 1}")
    print(f"{'frame':>6} {'rel_d':>7} {'depth_offset_ms':>15} {'angular_speed_dps':>17} {'expected_misalignment_px':>24}")
    for i in ROOM_TABLE:
        r = by_index[i]
        print(f"{i:>6} {number(r['rel_d']):>7.2f} {float(r['eval_depth_offset_ms']):>15.1f} "
              f"{number(r['angular_speed_dps']):>17.1f} {number(r['expected_misalignment_px']):>24.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--covariates", type=Path, required=True, help="covariates.csv from compute_covariates.py")
    parser.add_argument("--per-frame", type=Path, required=True, help="per_frame.csv from evaluate.py")
    parser.add_argument("--data-root", type=Path, required=True, help="folder containing the sequence folders (RGB thumbnails)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    joined = join(read_csv(args.covariates), read_csv(args.per_frame), args.per_frame)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, rows in joined.items():
        save_time_series(name, rows, args.out / f"{name}_timeseries.png")
        save_contact_sheet(name, rows, args.data_root, args.out / f"{name}_sharpness_extremes.png")
        print(f"saved {name} time series and contact sheet to {args.out}")
    if ROOM in joined:
        print_room_table(joined[ROOM])


if __name__ == "__main__":
    main()
