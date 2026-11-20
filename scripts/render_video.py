"""Side by side video of raw and fused depth from fusion_render_prepare.py, with JOINT-SOLVED scale and ORACLE GT poses.

Row one shows RGB, raw, fused and GT depth on one turbo range per sequence, the GT 2nd to 98th percentile over the
window. Row two shows the temporal disagreement of raw and of fused depth with the previous kept frame (within 50 ms),
the previous depth warped into the frame by the GT poses and divided by GT depth, on a fixed 0 to 5 percent scale
and only where the project's GT visibility rule holds, beside a text and running plot panel. The full window plays at
--fps, then the --replay-seconds stretch with the largest raw minus fused disagreement replays at a third of the speed.
Also writes a 12 second half size gif, a poster png and a json of settings and statistics."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from pim.consistency.metrics import FrameInputs, visibility  # noqa: E402
from pim.consistency.warp import depth_pose, relative_pose, warp_depth  # noqa: E402
from pim.data.tum import TIMESTAMP_EPS, Frame, apply_window, load_sequence, read_groundtruth  # noqa: E402
from pim.video import VideoWriter, colorize, half_size, write_gif  # noqa: E402

SCALE = "joint_solved"
POSES = "oracle_gt"
FOOTER = "oracle poses, upper bound, joint solved scale, 4 earlier frames"
SIZE = (1920, 1080)
MAX_GAP_S = 0.05
ERROR_MAX = 0.05
HISTORY = 30
SLOW_FACTOR = 3
GIF_SECONDS = 12.0
GIF_FPS = 10.0
DEPTH_CMAP = "turbo"
ERROR_CMAP = "viridis"
DEPTH_PERCENTILES = (2.0, 98.0)
GT_SAMPLE_STRIDE = 16
BACKGROUND = "#101010"
RAW_COLOUR = "#f28e2b"
FUSED_COLOUR = "#4e9fe5"


@dataclass(frozen=True)
class View:
    timestamp: float
    T_rgb: np.ndarray
    """T_world_cam at the RGB timestamp, which the predictions are warped with."""
    T_depth: np.ndarray
    """T_world_cam at the depth timestamp, which the GT visibility rule is evaluated with."""
    gt_depth: np.ndarray
    gt_valid: np.ndarray
    depth: np.ndarray
    """Metric depth after the per frame GT alignment, NaN where missing."""


@dataclass(frozen=True)
class Loaded:
    index: int
    timestamp: float
    raw: View
    fused: View
    """The fused depth, or the raw depth on frames with has_fused false."""
    has_fused: bool
    rel_d_raw: float
    rel_d_fused: float
    n_sources: int
    low_parallax: bool
    rgb: np.ndarray | None


@dataclass(frozen=True)
class FrameData:
    cur: Loaded
    error_raw: np.ndarray
    error_fused: np.ndarray
    """The raw map unless both frames have fused depth."""
    has_previous: bool
    paired: bool
    """The frame and its previous kept frame within MAX_GAP_S both have fused depth."""


def within_gap(previous_timestamp: float, timestamp: float, max_gap: float = MAX_GAP_S) -> bool:
    return 0 < timestamp - previous_timestamp <= max_gap + TIMESTAMP_EPS


def disagreement(prev: View | None, cur: View, K: np.ndarray, max_gap: float = MAX_GAP_S) -> np.ndarray:
    """|prev depth warped into cur minus cur depth| / cur GT depth on visible pixels, NaN elsewhere and everywhere when
    prev is None or more than max_gap seconds earlier."""
    error = np.full(cur.gt_depth.shape, np.nan, dtype=np.float32)
    if prev is None or not within_gap(prev.timestamp, cur.timestamp, max_gap):
        return error
    visible = visibility(FrameInputs(prev.gt_depth, prev.gt_valid), FrameInputs(cur.gt_depth, cur.gt_valid), K,
                         relative_pose(prev.T_depth, cur.T_depth)).visible
    source_valid = prev.gt_valid.astype(bool) & np.isfinite(prev.depth)
    warped, hit = warp_depth(prev.depth, source_valid, K, relative_pose(prev.T_rgb, cur.T_rgb), cur.gt_depth.shape)
    scored = visible & hit & np.isfinite(cur.depth)
    gt = cur.gt_depth[scored].astype(np.float64)
    error[scored] = np.abs(warped[scored].astype(np.float64) - cur.depth[scored].astype(np.float64)) / gt
    return error


def map_median(error: np.ndarray) -> float:
    values = error[np.isfinite(error)]
    return float(np.median(values)) if values.size else math.nan


def compare(prev: Loaded | None, cur: Loaded, K: np.ndarray) -> FrameData:
    error_raw = disagreement(None if prev is None else prev.raw, cur.raw, K)
    has_previous = prev is not None and within_gap(prev.timestamp, cur.timestamp)
    paired = has_previous and prev.has_fused and cur.has_fused
    error_fused = disagreement(prev.fused, cur.fused, K) if paired else error_raw
    return FrameData(cur, error_raw, error_fused, has_previous, paired)


def replay_interval(difference: np.ndarray, length: int) -> tuple[int, int]:
    """[start, stop) of the length long stretch with the largest summed difference, NaN counting as zero."""
    n = len(difference)
    if length >= n:
        return 0, n
    if length < 1:
        raise ValueError(f"replay length must be positive, got {length}")
    cumulative = np.concatenate([[0.0], np.cumsum(np.nan_to_num(np.asarray(difference, dtype=np.float64)))])
    start = int(np.argmax(cumulative[length:] - cumulative[:-length]))
    return start, start + length


def depth_range(samples: np.ndarray) -> tuple[float, float]:
    samples = samples[np.isfinite(samples)]
    if not samples.size:
        raise ValueError("no valid GT depth in the window")
    low, high = (float(v) for v in np.percentile(samples, DEPTH_PERCENTILES))
    return (low, high) if high > low else (low, low + 1e-3)


# Inputs.

class SequenceInputs:
    def __init__(self, render_dir: Path, data_root: Path, sequences_config: Path, name: str,
                 max_frames: int | None = None) -> None:
        sequence = load_sequence(data_root / name, sequences_config)
        by_index = {f.index: f for f in apply_window(sequence)}
        self.name, self.K = name, sequence.K
        self.gt_trajectory = read_groundtruth(sequence.root)
        folder = render_dir / name
        paths = sorted(folder.glob("*.npz"), key=lambda p: int(p.stem))
        if not paths:
            raise FileNotFoundError(f"no npz under {folder}")
        self.paths = paths[:max_frames] if max_frames else paths
        missing = [int(p.stem) for p in self.paths if int(p.stem) not in by_index]
        if missing:
            raise ValueError(f"{folder} has frames {missing[:5]} outside the window of {sequences_config}")
        self.frames: list[Frame] = [by_index[int(p.stem)] for p in self.paths]
        with np.load(self.paths[0]) as data:
            labels = (str(data["scale"]), str(data["poses"]))
        if labels != (SCALE, POSES):
            raise ValueError(f"{self.paths[0]} has scale and poses {labels}, expected {(SCALE, POSES)}")

    def __len__(self) -> int:
        return len(self.paths)

    def load(self, position: int, with_rgb: bool = False) -> Loaded:
        frame = self.frames[position]
        with np.load(self.paths[position]) as data:
            if int(data["index"]) != frame.index:
                raise ValueError(f"{self.paths[position]} holds index {int(data['index'])}")
            raw = data["raw_aligned"].astype(np.float32)
            has_fused = bool(data["has_fused"])
            fused = data["fused_aligned"].astype(np.float32) if has_fused else raw
            meta = (float(data["rel_d_raw"]), float(data["rel_d_fused"]), int(data["n_sources"]),
                    bool(data["low_parallax"]))
        gt_depth, gt_valid = frame.load_depth()
        T_depth = depth_pose(frame, *self.gt_trajectory)

        def view(depth: np.ndarray) -> View:
            return View(frame.timestamp, frame.T_world_cam, T_depth, gt_depth, gt_valid, depth)

        return Loaded(frame.index, frame.timestamp, view(raw), view(fused), has_fused, *meta,
                      frame.load_rgb() if with_rgb else None)

    def iterate(self, start: int, stop: int, with_rgb: bool = False) -> Iterator[FrameData]:
        prev = self.load(start - 1) if start > 0 else None
        for position in range(start, stop):
            cur = self.load(position, with_rgb)
            yield compare(prev, cur, self.K)
            prev = cur


@dataclass(frozen=True)
class Series:
    index: np.ndarray
    timestamp: np.ndarray
    median_raw: np.ndarray
    median_fused: np.ndarray
    """NaN on frames that are not paired."""
    rel_d_raw: np.ndarray
    rel_d_fused: np.ndarray
    has_fused: np.ndarray
    paired: np.ndarray
    gt_samples: np.ndarray

    @property
    def difference(self) -> np.ndarray:
        return self.median_raw - self.median_fused


def scan(inputs: SequenceInputs) -> Series:
    rows, samples = [], []
    for data in inputs.iterate(0, len(inputs)):
        cur = data.cur
        samples.append(cur.raw.gt_depth[cur.raw.gt_valid][::GT_SAMPLE_STRIDE])
        rows.append((cur.index, cur.timestamp, map_median(data.error_raw),
                     map_median(data.error_fused) if data.paired else math.nan, cur.rel_d_raw, cur.rel_d_fused,
                     cur.has_fused, data.paired))
    columns = list(zip(*rows))
    return Series(np.array(columns[0], dtype=np.int64), *(np.array(c, dtype=np.float64) for c in columns[1:6]),
                  np.array(columns[6], dtype=bool), np.array(columns[7], dtype=bool), np.concatenate(samples))


# Drawing.

def _rect(x: float, y: float, w: float, h: float) -> list[float]:
    """Figure fraction rectangle of a pixel box given from the top left."""
    return [x / SIZE[0], 1 - (y + h) / SIZE[1], w / SIZE[0], h / SIZE[1]]


class Renderer:
    def __init__(self, name: str, depth_limits: tuple[float, float], series: Series) -> None:
        self.name, self.depth_limits, self.series = name, depth_limits, series
        self.t0 = float(series.timestamp[0])
        self.figure = Figure(figsize=(SIZE[0] / 100, SIZE[1] / 100), dpi=100, facecolor=BACKGROUND)
        self.canvas = FigureCanvasAgg(self.figure)
        fig = self.figure
        text = {"color": "white"}
        fig.text(0.006, 0.985, f"{name}   scale {SCALE}, poses {POSES} (ORACLE GT poses, an upper bound)",
                 fontsize=18, va="top", weight="bold", **text)
        self.slow = fig.text(0.994, 0.985, "SLOW MOTION  x1/3", fontsize=20, va="top", ha="right", weight="bold",
                             color="black", bbox={"facecolor": "#ffd166", "edgecolor": "none", "pad": 5})

        width, height, gap = 460, 345, 20
        titles = ("RGB", "raw depth", "fused depth", "GT depth")
        self.top = []
        for k, title in enumerate(titles):
            ax = fig.add_axes(_rect(10 + k * (width + gap), 80, width, height))
            self.top.append(self._image_axes(ax, title, (480, 640)))
        self.no_fusion_depth = self._label(10 + 2 * (width + gap) + width / 2, 80 + 22, "no fusion")
        cax = fig.add_axes(_rect(10 + width + gap, 440, 3 * width + 2 * gap, 16))
        self._colorbar(cax, DEPTH_CMAP, depth_limits,
                       "depth (m), fixed range per sequence from the GT 2nd to 98th percentile over the window")

        width, height, gap, top = 560, 420, 20, 560
        self.bottom = []
        for k, title in enumerate(("raw temporal disagreement", "fused temporal disagreement")):
            ax = fig.add_axes(_rect(10 + k * (width + gap), top, width, height))
            self.bottom.append(self._image_axes(ax, title, (480, 640)))
        self.no_fusion_error = self._label(10 + width + gap + width / 2, top + 22, "no fusion")
        self.no_previous = [self._label(10 + k * (width + gap) + width / 2, top + height / 2,
                                        "no previous frame within 50 ms") for k in range(2)]
        cax = fig.add_axes(_rect(10, 995, 2 * width + gap, 16))
        self._colorbar(cax, ERROR_CMAP, (0.0, 100 * ERROR_MAX),
                       "|previous depth warped in minus depth| / GT depth (%), grey is not visible or invalid")

        left = 2 * (width + gap) + 30
        fig.text(left / SIZE[0], 1 - 540 / SIZE[1], "frame summary", fontsize=18, weight="bold", va="top", **text)
        self.info = fig.text(left / SIZE[0], 1 - 578 / SIZE[1], "", fontsize=18, va="top", linespacing=1.5, **text)
        self.plot = fig.add_axes(_rect(left + 50, 800, SIZE[0] - left - 80, 175), facecolor="#1c1c1c")
        self.plot.set_title(f"median disagreement, last {HISTORY} frames (%)", fontsize=16, **text)
        self.plot.tick_params(colors="white", labelsize=13)
        for spine in self.plot.spines.values():
            spine.set_color("#777777")
        self.plot.grid(color="#444444", linewidth=0.6)
        self.line_raw, = self.plot.plot([], [], color=RAW_COLOUR, linewidth=2.5, marker="o", markersize=3,
                                        label="raw")
        self.line_fused, = self.plot.plot([], [], color=FUSED_COLOUR, linewidth=2.5, marker="o", markersize=3,
                                          label="fused")
        self.plot.legend(loc="upper left", fontsize=13, facecolor="#1c1c1c", labelcolor="white", edgecolor="#777777")
        finite = np.concatenate([series.median_raw, series.median_fused])
        finite = finite[np.isfinite(finite)]
        y_max = 100 * float(np.percentile(finite, 99)) * 1.15 if finite.size else 100 * ERROR_MAX
        self.plot.set_ylim(0.0, max(y_max, 0.1))
        self.plot.set_xlabel("frame index", fontsize=13, **text)
        fig.text(left / SIZE[0], 1 - 1052 / SIZE[1], FOOTER, fontsize=14, weight="bold", color="#ffd166",
                 va="center")

    def _image_axes(self, ax, title: str, shape: tuple[int, int]):
        ax.set_title(title, fontsize=18, color="white", pad=6)
        ax.set_axis_off()
        return ax.imshow(np.zeros((*shape, 3), dtype=np.uint8), interpolation="antialiased", aspect="auto")

    def _label(self, x: float, y: float, text: str):
        return self.figure.text(x / SIZE[0], 1 - y / SIZE[1], text, fontsize=18, ha="center", va="center",
                                weight="bold", color="black", visible=False,
                                bbox={"facecolor": "#ffd166", "edgecolor": "none", "pad": 4})

    def _colorbar(self, cax, cmap: str, limits: tuple[float, float], label: str) -> None:
        bar = self.figure.colorbar(ScalarMappable(Normalize(*limits), cmap), cax=cax, orientation="horizontal")
        bar.ax.tick_params(colors="white", labelsize=13)
        bar.outline.set_edgecolor("#777777")
        bar.set_label(label, color="white", fontsize=14)

    def draw(self, data: FrameData, slow: bool = False) -> np.ndarray:
        cur = data.cur
        low, high = self.depth_limits
        gt = np.where(cur.raw.gt_valid, cur.raw.gt_depth, np.nan)
        images = (cur.rgb, colorize(cur.raw.depth, low, high, DEPTH_CMAP),
                  colorize(cur.fused.depth, low, high, DEPTH_CMAP), colorize(gt, low, high, DEPTH_CMAP))
        for artist, image in zip(self.top, images):
            artist.set_data(image)
        for artist, error in zip(self.bottom, (data.error_raw, data.error_fused)):
            artist.set_data(colorize(error, 0.0, ERROR_MAX, ERROR_CMAP))
        self.no_fusion_depth.set_visible(not cur.has_fused)
        self.no_fusion_error.set_visible(data.has_previous and not data.paired)
        for label in self.no_previous:
            label.set_visible(not data.has_previous)
        self.slow.set_visible(slow)
        fused = f"{cur.rel_d_fused:.2f}" if cur.has_fused else "n/a (no fusion)"
        parallax = ", low parallax" if cur.low_parallax else ""
        self.info.set_text("\n".join([
            self.name,
            f"frame {cur.index}    t = {cur.timestamp - self.t0:.2f} s",
            f"rel_d x100  raw {cur.rel_d_raw:.2f}   fused {fused}",
            f"{cur.n_sources} earlier source frames{parallax}",
        ]))
        position = int(np.searchsorted(self.series.index, cur.index))
        first = max(0, position - HISTORY + 1)
        x = self.series.index[first:position + 1]
        self.line_raw.set_data(x, 100 * self.series.median_raw[first:position + 1])
        self.line_fused.set_data(x, 100 * self.series.median_fused[first:position + 1])
        start = max(cur.index - HISTORY + 1, int(self.series.index[0]))
        self.plot.set_xlim(start - 0.5, start + HISTORY - 0.5)
        self.canvas.draw()
        return np.asarray(self.canvas.buffer_rgba())[..., :3].copy()


# Command line.

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("render_dir", type=Path, help="fusion/render/<run name>, the folder holding sequence folders")
    parser.add_argument("data_root", type=Path, help="folder containing the sequence folders")
    parser.add_argument("sequences_config", type=Path)
    parser.add_argument("sequence")
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--replay-seconds", type=float, default=2.0)
    return parser.parse_args(argv)


def summary(series: Series) -> dict[str, object]:
    def median(values: np.ndarray) -> float | None:
        values = values[np.isfinite(values)]
        return round(100 * float(np.median(values)), 4) if values.size else None

    paired = series.paired
    return {
        "frames": int(len(series.index)),
        "fused_frames": int(series.has_fused.sum()),
        "paired_frames": int(paired.sum()),
        "median_disagreement_raw_pct": median(series.median_raw[paired]),
        "median_disagreement_fused_pct": median(series.median_fused[paired]),
        "median_disagreement_raw_all_frames_pct": median(series.median_raw),
        "statistic": "median over frames of the per frame median disagreement in percent; raw and fused on the "
                     "paired frames, where the frame and its previous kept frame both have fused depth",
    }


def main(argv: list[str] | None = None) -> dict[str, object]:
    args = parse_args(argv)
    inputs = SequenceInputs(args.render_dir, args.data_root, args.sequences_config, args.sequence, args.max_frames)
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    print(f"{args.sequence} {len(inputs)} frames, scanning", flush=True)
    series = scan(inputs)
    limits = depth_range(series.gt_samples)
    replay_length = max(1, round(args.replay_seconds * args.fps))
    replay = replay_interval(series.difference, replay_length)
    difference = np.where(series.paired, series.difference, -np.inf)
    poster = int(np.argmax(difference)) if np.isfinite(difference).any() else len(series.index) // 2
    gif_step = max(1, round(args.fps / GIF_FPS))
    gif_length = round(GIF_SECONDS * args.fps)
    centre = (replay[0] + replay[1]) // 2
    gif_start = int(np.clip(centre - gif_length // 2, 0, max(0, len(inputs) - gif_length)))
    gif_positions = set(range(gif_start, min(len(inputs), gif_start + gif_length), gif_step))

    renderer = Renderer(args.sequence, limits, series)
    video = VideoWriter(out / f"{args.sequence}.mp4", SIZE, args.fps)
    gif_frames = []
    poster_path = out / f"{args.sequence}_poster.png"
    for position, data in enumerate(inputs.iterate(0, len(inputs), with_rgb=True)):
        frame = renderer.draw(data)
        video.write(frame)
        if position in gif_positions:
            gif_frames.append(half_size(frame))
        if position == poster:
            Image.fromarray(frame).save(poster_path)
    for data in inputs.iterate(*replay, with_rgb=True):
        frame = renderer.draw(data, slow=True)
        for _ in range(SLOW_FACTOR):
            video.write(frame)
    video_path = video.close()
    gif_path = write_gif(out / f"{args.sequence}_preview.gif", gif_frames, args.fps / gif_step)

    t0 = float(series.timestamp[0])
    record = {
        "scale": SCALE,
        "poses": POSES,
        "label": FOOTER,
        "settings": {
            **{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
            "size": list(SIZE),
            "depth_colormap": DEPTH_CMAP,
            "depth_range_m": [round(v, 4) for v in limits],
            "depth_range_rule": f"GT depth percentiles {DEPTH_PERCENTILES} over the window, every "
                                f"{GT_SAMPLE_STRIDE}th valid pixel",
            "error_colormap": ERROR_CMAP,
            "error_range": [0.0, ERROR_MAX],
            "max_gap_ms": 1000 * MAX_GAP_S,
            "history_frames": HISTORY,
            "replay_slow_factor": SLOW_FACTOR,
            "gif": {"seconds": GIF_SECONDS, "fps": args.fps / gif_step, "half_size": True},
        },
        "replay": {
            "start_index": int(series.index[replay[0]]),
            "stop_index": int(series.index[replay[1] - 1]),
            "start_s": round(float(series.timestamp[replay[0]]) - t0, 3),
            "frames": replay[1] - replay[0],
            "mean_raw_minus_fused_pct": round(100 * float(np.nanmean(series.difference[replay[0]:replay[1]])), 4)
            if np.isfinite(series.difference[replay[0]:replay[1]]).any() else None,
        },
        "poster": {"index": int(series.index[poster]),
                   "raw_minus_fused_pct": round(100 * float(series.difference[poster]), 4)
                   if np.isfinite(series.difference[poster]) else None,
                   "rule": "paired frame with the largest raw minus fused median disagreement"},
        "sequences": {args.sequence: summary(series)},
        "per_frame": [
            {"index": int(i), "t_s": round(float(t) - t0, 3), "median_raw_pct": _pct(r), "median_fused_pct": _pct(f),
             "rel_d_raw": _round(a), "rel_d_fused": _round(b), "has_fused": bool(h)}
            for i, t, r, f, a, b, h in zip(series.index, series.timestamp, series.median_raw, series.median_fused,
                                           series.rel_d_raw, series.rel_d_fused, series.has_fused)
        ],
        "outputs": {"video": str(video_path), "gif": str(gif_path), "poster": str(poster_path)},
    }
    record_path = out / f"{args.sequence}_video.json"
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    print(f"wrote {video_path}, {gif_path}, {poster_path} and {record_path}")
    return record


def _pct(value: float) -> float | None:
    return round(100 * float(value), 4) if math.isfinite(value) else None


def _round(value: float) -> float | None:
    return round(float(value), 4) if math.isfinite(value) else None


if __name__ == "__main__":
    main()
