"""Per frame share of fused frames where fusion lowers, keeps or raises rel_d, from the render preparation output.

Reads only the scalar fields of each <render dir>/<sequence>/<index>.npz, one file at a time, and keeps frames with
has_fused true. The change is 100 x (rel_d_fused - rel_d_raw) / rel_d_raw, so a negative change means fusion helps.
These are rel_d changes under oracle poses and the joint solved scale. Writes frame_share.csv, frame_share_summary.csv
and summary.md to the output dir. An optional --disagreement-csv with sequence and index columns is joined onto the
frame rows by (sequence, index), and when it has disagreement_raw_pct and disagreement_fused_pct the summary also
reports the share of frames where fused disagreement is lower.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pim.analysis.stats import moving_block_indices, percentile_ci  # noqa: E402

SHORT_NAMES = {
    "rgbd_dataset_freiburg1_desk": "fr1_desk",
    "rgbd_dataset_freiburg1_room": "fr1_room",
    "rgbd_dataset_freiburg1_360": "fr1_360",
    "rgbd_dataset_freiburg3_long_office_household": "fr3",
}
SCALAR_KEYS = ("has_fused", "rel_d_raw", "rel_d_fused", "index", "timestamp")
TOL = 1e-6
BLOCKS = (8, 15, 30)
RESAMPLES = 2000
SEED = 0
EXTREMES = 5
FLAG_SEQUENCE, FLAG_INDEX = "fr1_360", 23
DISAGREEMENT_RAW, DISAGREEMENT_FUSED = "disagreement_raw_pct", "disagreement_fused_pct"
SCOPE = (("quantity", "per frame rel_d change in percent of raw, negative means fusion helps"),
         ("poses", "oracle poses"),
         ("scale", "joint solved scale"),
         ("frames", "has_fused true only"),
         ("equal", f"rel_d_fused within {TOL:g} of rel_d_raw"),
         ("bootstrap", f"moving block over frames in index order, blocks {', '.join(map(str, BLOCKS))}, "
                       f"{RESAMPLES} resamples, seed {SEED}, widest 95 percent interval over blocks"),
         ("worse run", "consecutive fused frames in index order"))


@dataclass
class Frame:
    index: int
    timestamp: float
    rel_d_raw: float
    rel_d_fused: float

    @property
    def change_pct(self) -> float:
        return 100.0 * (self.rel_d_fused - self.rel_d_raw) / self.rel_d_raw

    @property
    def outcome(self) -> str:
        diff = self.rel_d_fused - self.rel_d_raw
        return "equal" if abs(diff) <= TOL else ("lower" if diff < 0 else "higher")


@dataclass
class SequenceFrames:
    name: str
    frames: list[Frame] = field(default_factory=list)
    skipped: int = 0


def read_scalars(path: Path) -> dict[str, float | int | bool] | None:
    """Scalar fields of one frame, or None when it has no fused depth. Depth arrays are never read."""
    with np.load(path, allow_pickle=False) as data:
        if not bool(data["has_fused"]):
            return None
        return {"index": int(data["index"]), "timestamp": float(data["timestamp"]),
                "rel_d_raw": float(data["rel_d_raw"]), "rel_d_fused": float(data["rel_d_fused"])}


def short_name(folder: str) -> str:
    return SHORT_NAMES.get(folder, folder)


def read_sequence(folder: Path) -> SequenceFrames:
    seq = SequenceFrames(short_name(folder.name))
    for path in sorted(folder.glob("*.npz")):
        values = read_scalars(path)
        if values is None:
            seq.skipped += 1
            continue
        seq.frames.append(Frame(**values))
    seq.frames.sort(key=lambda f: f.index)
    return seq


def read_render_dir(render_dir: Path) -> list[SequenceFrames]:
    folders = sorted(p for p in render_dir.iterdir() if p.is_dir() and any(p.glob("*.npz")))
    if not folders:
        raise SystemExit(f"no sequence folders with npz files under {render_dir}")
    return [read_sequence(f) for f in folders]


def read_disagreement(path: Path) -> tuple[dict[tuple[str, int], dict[str, str]], list[str]]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        columns = [c for c in reader.fieldnames or [] if c not in ("sequence", "index")]
        rows = {(short_name(r["sequence"]), int(r["index"])): {c: r[c] for c in columns} for r in reader}
    return rows, columns


Statistic = Callable[[np.ndarray], np.ndarray]


def mean_rows(x: np.ndarray) -> np.ndarray:
    return x.mean(axis=-1)


def block_intervals(values: np.ndarray, statistic: Statistic) -> dict[int, tuple[float, float]]:
    out = {}
    for block in BLOCKS:
        if block > len(values):
            out[block] = (math.nan, math.nan)
            continue
        rng = np.random.default_rng([SEED, block])
        out[block] = percentile_ci(statistic(values[moving_block_indices(len(values), block, RESAMPLES, rng)]))
    return out


def widest(intervals: dict[int, tuple[float, float]]) -> tuple[float, float]:
    finite = [iv for iv in intervals.values() if not math.isnan(iv[0])]
    if not finite:
        return math.nan, math.nan
    return min(lo for lo, _ in finite), max(hi for _, hi in finite)


def longest_worse_run(frames: Sequence[Frame]) -> tuple[int, int | None, int | None]:
    best, start, best_span = 0, 0, (None, None)
    for k, frame in enumerate(frames):
        if frame.outcome != "higher":
            start = k + 1
            continue
        if k - start + 1 > best:
            best, best_span = k - start + 1, (frames[start].index, frame.index)
    return best, *best_span


def summarise(seq: SequenceFrames, disagreement: dict[tuple[str, int], dict[str, str]] | None) -> dict[str, object]:
    frames = seq.frames
    n = len(frames)
    row: dict[str, object] = {"sequence": seq.name, "fused_frames": n, "frames_without_fused": seq.skipped,
                              "basis": "rel_d change under oracle poses and joint solved scale"}
    if n == 0:
        return row
    change = np.array([f.change_pct for f in frames])
    outcomes = [f.outcome for f in frames]
    for kind in ("lower", "equal", "higher"):
        row[f"share_{kind}"] = outcomes.count(kind) / n
    row.update(median_change_pct=float(np.median(change)), mean_change_pct=float(change.mean()),
               p10_change_pct=float(np.percentile(change, 10)), p90_change_pct=float(np.percentile(change, 90)))
    improved = np.array([o == "lower" for o in outcomes], dtype=np.float64)
    for metric, values in (("mean_change_pct", change), ("share_lower", improved)):
        intervals = block_intervals(values, mean_rows)
        for block, (lo, hi) in intervals.items():
            row[f"{metric}_ci_low_b{block}"], row[f"{metric}_ci_high_b{block}"] = lo, hi
        row[f"{metric}_ci_low"], row[f"{metric}_ci_high"] = widest(intervals)
    order = np.argsort(change, kind="stable")
    worst, best = order[::-1][:EXTREMES], order[:EXTREMES]
    row["worst_indices"] = " ".join(str(frames[k].index) for k in worst)
    row["worst_change_pct"] = " ".join(f"{change[k]:.3f}" for k in worst)
    row["best_indices"] = " ".join(str(frames[k].index) for k in best)
    row["best_change_pct"] = " ".join(f"{change[k]:.3f}" for k in best)
    if seq.name == FLAG_SEQUENCE:
        rank = {frames[k].index: r + 1 for r, k in enumerate(order[::-1])}
        row[f"frame_{FLAG_INDEX}_worst_rank"] = rank.get(FLAG_INDEX, "")
        row[f"frame_{FLAG_INDEX}_in_worst_{EXTREMES}"] = FLAG_INDEX in {frames[k].index for k in worst}
    row["longest_worse_run"], row["longest_worse_run_start"], row["longest_worse_run_end"] = longest_worse_run(frames)
    if disagreement is not None:
        pairs = [disagreement.get((seq.name, f.index), {}) for f in frames]
        pairs = [(float(p[DISAGREEMENT_RAW]), float(p[DISAGREEMENT_FUSED])) for p in pairs
                 if p.get(DISAGREEMENT_RAW) and p.get(DISAGREEMENT_FUSED)]
        row["disagreement_frames"] = len(pairs)
        if pairs:
            row["share_disagreement_lower"] = sum(fu < ra for ra, fu in pairs) / len(pairs)
    return row


def frame_rows(seqs: Iterable[SequenceFrames], disagreement: dict[tuple[str, int], dict[str, str]] | None
               ) -> list[dict[str, object]]:
    rows = []
    for seq in seqs:
        for f in seq.frames:
            row: dict[str, object] = {"sequence": seq.name, "index": f.index, "timestamp": f.timestamp,
                                      "rel_d_raw": f.rel_d_raw, "rel_d_fused": f.rel_d_fused,
                                      "change_pct": f.change_pct, "outcome": f.outcome}
            if disagreement is not None:
                row.update(disagreement.get((seq.name, f.index), {}))
            rows.append(row)
    return rows


def summary_columns(rows: Sequence[dict[str, object]]) -> list[str]:
    columns: list[str] = []
    for row in rows:
        columns += [c for c in row if c not in columns]
    return columns


def write_csv(path: Path, rows: Sequence[dict[str, object]], columns: Sequence[str]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in row.items()})


def table(header: Sequence[str], body: Iterable[Sequence[object]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return lines + ["| " + " | ".join(str(c) for c in r) + " |" for r in body] + [""]


def pct(value: object, digits: int = 1) -> str:
    return "n/a" if value is None or (isinstance(value, float) and math.isnan(value)) else \
        f"{100 * float(value):.{digits}f} %"


def num(value: object, digits: int = 2) -> str:
    return "n/a" if value is None or (isinstance(value, float) and math.isnan(value)) else \
        f"{float(value):+.{digits}f}"


def interval(row: dict[str, object], metric: str, fmt: Callable[[object], str]) -> str:
    return f"{fmt(row.get(metric))} [{fmt(row.get(metric + '_ci_low'))}, {fmt(row.get(metric + '_ci_high'))}]"


def summary_markdown(rows: Sequence[dict[str, object]]) -> str:
    lines = ["# Fusion frame share", ""]
    lines += table(("scope", "value"), SCOPE)
    lines += table(("sequence", "fused frames", "lower", "equal", "higher"),
                   [(r["sequence"], r["fused_frames"], pct(r.get("share_lower")), pct(r.get("share_equal")),
                     pct(r.get("share_higher"))) for r in rows])
    lines += table(("sequence", "median change %", "mean change %", "p10 change %", "p90 change %"),
                   [(r["sequence"], num(r.get("median_change_pct")), num(r.get("mean_change_pct")),
                     num(r.get("p10_change_pct")), num(r.get("p90_change_pct"))) for r in rows])
    lines += table(("sequence", "mean change % [95 % CI]", "share lower [95 % CI]"),
                   [(r["sequence"], interval(r, "mean_change_pct", num), interval(r, "share_lower", pct))
                    for r in rows])
    lines += table(("sequence", "worst indices", "worst change %", "best indices", "best change %"),
                   [(r["sequence"], r.get("worst_indices", ""), r.get("worst_change_pct", ""),
                     r.get("best_indices", ""), r.get("best_change_pct", "")) for r in rows])
    lines += table(("sequence", "longest worse run", "run start index", "run end index"),
                   [(r["sequence"], r.get("longest_worse_run", 0), r.get("longest_worse_run_start") or "",
                     r.get("longest_worse_run_end") or "") for r in rows])
    flag = [r for r in rows if r["sequence"] == FLAG_SEQUENCE]
    if flag:
        r = flag[0]
        rank = r.get(f"frame_{FLAG_INDEX}_worst_rank", "")
        lines += table(("sequence", "frame", "has fused", "rank from worst", f"in worst {EXTREMES}"),
                       [(FLAG_SEQUENCE, FLAG_INDEX, "yes" if rank != "" else "no", rank or "n/a",
                         "yes" if r.get(f"frame_{FLAG_INDEX}_in_worst_{EXTREMES}") else "no")])
    if any("disagreement_frames" in r for r in rows):
        lines += table(("sequence", "frames with disagreement", "share fused disagreement lower"),
                       [(r["sequence"], r.get("disagreement_frames", 0), pct(r.get("share_disagreement_lower")))
                        for r in rows])
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("render_dir", type=Path, help="folder holding the sequence folders of npz frames")
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--disagreement-csv", type=Path,
                        help="per frame CSV with sequence and index columns, joined onto frame_share.csv")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    seqs = read_render_dir(args.render_dir)
    disagreement, extra = read_disagreement(args.disagreement_csv) if args.disagreement_csv else (None, [])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    frames = frame_rows(seqs, disagreement)
    write_csv(args.out_dir / "frame_share.csv", frames,
              ["sequence", "index", "timestamp", "rel_d_raw", "rel_d_fused", "change_pct", "outcome", *extra])
    rows = [summarise(s, disagreement) for s in seqs]
    write_csv(args.out_dir / "frame_share_summary.csv", rows, summary_columns(rows))
    (args.out_dir / "summary.md").write_text(summary_markdown(rows))
    for r in rows:
        print(f"{r['sequence']} fused frames {r['fused_frames']} lower {pct(r.get('share_lower'))} "
              f"higher {pct(r.get('share_higher'))} mean change {interval(r, 'mean_change_pct', num)}")
    print(f"wrote {args.out_dir}")


if __name__ == "__main__":
    main()
