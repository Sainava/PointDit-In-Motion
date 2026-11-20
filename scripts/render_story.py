"""Simple story clips of the render preparation output beside the dashboard of render_video.py, from the same inputs,
with JOINT-SOLVED scale and ORACLE GT poses.

--mode hero          RGB, raw and fused depth on one fixed depth range, the first and last frame held.
--mode flicker       raw and fused temporal disagreement over the dimmed RGB with a stable to flickers legend.
--mode toggle        raw and fused depth alternating every --toggle-seconds while the clip plays.
--mode stillflip     one frame with raw and fused alternating, plus a poster of its disagreement overlay.
--mode cloud         one frame as coloured point clouds of raw, fused and GT depth from a swaying virtual camera.
--mode cloud_moving  every clip frame as point clouds from a virtual camera at a fixed yaw offset.

Every mode writes a 1920x1080 H.264 mp4 (PNG frames without ffmpeg), a half size gif of at most --gif-seconds and a
json of the settings, the frame choice and the clip medians. --start and --max-frames pick the clip by position in the
render folder. --frame picks one frame by its index, by default the clip frame of median fusion gain, the raw minus
fused median disagreement on frames paired with fused depth. The depth range is the one render_video.py uses, the GT
2nd to 98th percentile over the whole window."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from export_fused_map import frame_mask  # noqa: E402
from render_orbit import splat  # noqa: E402
from render_video import (  # noqa: E402
    DEPTH_CMAP, DEPTH_PERCENTILES, ERROR_CMAP, ERROR_MAX, FOOTER, GT_SAMPLE_STRIDE, POSES, SCALE, FrameData, Loaded,
    SequenceInputs, Series, depth_range, map_median, summary,
)

from pim.video import VideoWriter, colorize, font, half_size, hold_repeats, write_gif  # noqa: E402

MODES = ("hero", "flicker", "toggle", "stillflip", "cloud", "cloud_moving")
SIZE = (1920, 1080)
DEFAULT_FPS = 10.0
HERO_HOLD_S = 1.0
TOGGLE_S = 0.5
STILLFLIP_CYCLES = 4
GIF_SECONDS = 12.0
GIF_MAX_FPS = 10.0
OVERLAY_ALPHA = 0.6
RGB_DIM = 0.4
CLOUD_SECONDS = 6.0
SWAY_DEG = 15.0
PITCH_DEG = 5.0
MOVING_YAW_DEG = 10.0
SPLAT_FACTOR = 2.0
"""Splat edge as a multiple of the source pixel footprint, so a surface seen up to SWAY_DEG off axis has no holes."""

DARK = (16, 16, 16)
WHITE = (255, 255, 255)
LIGHT_TEXT = (245, 245, 245)
DARK_TEXT = (25, 25, 25)
YELLOW = (255, 209, 102)
COLOURS = {"raw": (242, 142, 43), "fused": (78, 159, 229), "gt": (70, 70, 70)}
CLOUD_PANELS = (("raw", "raw"), ("fused", "fused"), ("gt", "GT"))

Box = tuple[int, int, int, int]
"""x, y, width, height in pixels from the top left."""

THREE = [(24 + k * (608 + 24), 215, 608, 456) for k in range(3)]
TWO = [(24 + k * (912 + 48), 200, 912, 684) for k in range(2)]
BIG: Box = (440, 110, 1120, 840)


# Clip and frame choice.

def clip_bounds(count: int, start: int, max_frames: int | None) -> tuple[int, int]:
    if not 0 <= start < count:
        raise ValueError(f"--start {start} is outside the {count} render frames")
    if max_frames is not None and max_frames < 1:
        raise ValueError(f"--max-frames must be positive, got {max_frames}")
    return start, count if max_frames is None else min(count, start + max_frames)


def scan_clip(inputs: SequenceInputs, start: int, stop: int) -> Series:
    rows = []
    for data in inputs.iterate(start, stop):
        cur = data.cur
        rows.append((cur.index, cur.timestamp, map_median(data.error_raw),
                     map_median(data.error_fused) if data.paired else math.nan, cur.rel_d_raw, cur.rel_d_fused,
                     cur.has_fused, data.paired))
    columns = list(zip(*rows))
    return Series(np.array(columns[0], dtype=np.int64), *(np.array(c, dtype=np.float64) for c in columns[1:6]),
                  np.array(columns[6], dtype=bool), np.array(columns[7], dtype=bool), np.empty(0))


def window_depth_range(inputs: SequenceInputs) -> tuple[float, float]:
    samples = []
    for frame in inputs.frames:
        depth, valid = frame.load_depth()
        samples.append(depth[valid.astype(bool)][::GT_SAMPLE_STRIDE])
    return depth_range(np.concatenate(samples))


def median_gain_position(series: Series) -> int | None:
    """Clip position of the paired frame whose raw minus fused disagreement is the median, the lower one of an even
    count, or None without a paired frame."""
    gain = series.difference
    candidates = np.flatnonzero(series.paired & np.isfinite(gain))
    if not candidates.size:
        return None
    ordered = candidates[np.argsort(gain[candidates], kind="stable")]
    return int(ordered[(len(ordered) - 1) // 2])


def frame_data(inputs: SequenceInputs, position: int) -> FrameData:
    return next(inputs.iterate(position, position + 1, with_rgb=True))


def gain_of(data: FrameData) -> float:
    return map_median(data.error_raw) - map_median(data.error_fused) if data.paired else math.nan


def choose_frame(inputs: SequenceInputs, start: int, series: Series,
                 frame_index: int | None) -> tuple[int, dict[str, object]]:
    gains = series.difference[series.paired & np.isfinite(series.difference)]
    choice: dict[str, object] = {"clip_median_gain_pct": _pct(float(np.median(gains))) if gains.size else None}
    if frame_index is not None:
        positions = {f.index: p for p, f in enumerate(inputs.frames)}
        if frame_index not in positions:
            raise ValueError(f"--frame {frame_index} is not a render frame of {inputs.name}")
        position = positions[frame_index]
        choice.update(chosen_by="--frame", rule="frame index given on the command line")
    else:
        relative = median_gain_position(series)
        if relative is None:
            relative = len(series.index) // 2
            choice.update(chosen_by="default", rule="no paired frame in the clip, so the middle clip frame")
        else:
            choice.update(chosen_by="default",
                          rule="clip frame of median fusion gain, the raw minus fused median disagreement over "
                               "paired frames, lower median for an even count, not the best case")
        position = start + relative
    choice.update(index=int(inputs.frames[position].index), position=position)
    return position, choice


# Drawing.

def short_name(name: str) -> str:
    return name.replace("rgbd_dataset_freiburg", "fr")


def canvas(background: tuple[int, int, int] = DARK) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", SIZE, background)
    return image, ImageDraw.Draw(image)


def place(image: Image.Image, array: np.ndarray, box: Box) -> None:
    x, y, w, h = box
    image.paste(Image.fromarray(array).resize((w, h), Image.Resampling.BILINEAR), (x, y))


def text(draw: ImageDraw.ImageDraw, xy: tuple[float, float], value: str, size: int,
         fill: tuple[int, int, int] = LIGHT_TEXT, anchor: str = "la") -> None:
    draw.text(xy, value, font=font(size), fill=fill, anchor=anchor)


def badge(draw: ImageDraw.ImageDraw, centre: tuple[float, float], value: str, size: int = 34) -> None:
    left, top, right, bottom = draw.textbbox(centre, value, font=font(size), anchor="mm")
    draw.rectangle((left - 12, top - 8, right + 12, bottom + 8), fill=YELLOW)
    text(draw, centre, value, size, (0, 0, 0), "mm")


def colour_bar(image: Image.Image, draw: ImageDraw.ImageDraw, box: Box, limits: tuple[float, float], cmap: str,
               ticks: list[tuple[float, str]], tick_size: int = 28, fill: tuple[int, int, int] = LIGHT_TEXT) -> None:
    x, y, w, h = box
    ramp = np.repeat(np.linspace(*limits, w)[None], h, axis=0)
    image.paste(Image.fromarray(colorize(ramp, *limits, cmap)), (x, y))
    for value, label in ticks:
        u = x + (value - limits[0]) / (limits[1] - limits[0]) * (w - 1)
        draw.line((u, y + h, u, y + h + 8), fill=fill, width=2)
        half = draw.textlength(label, font=font(tick_size)) / 2
        text(draw, (min(max(u, x + half), x + w - half), y + h + 12), label, tick_size, fill, "ma")


def depth_ticks(limits: tuple[float, float]) -> list[tuple[float, str]]:
    """Round metre values inside the range, three to seven of them."""
    low, high = limits
    step = next((s for s in (0.1, 0.2, 0.25, 0.5, 1.0, 2.0, 5.0) if (high - low) / s <= 6), 10.0)
    values = np.arange(math.ceil(low / step - 1e-9), math.floor(high / step + 1e-9) + 1) * step
    return [(float(v), f"{v:g} m") for v in values]


def depth_image(depth: np.ndarray, limits: tuple[float, float]) -> np.ndarray:
    return colorize(depth, *limits, DEPTH_CMAP)


def overlay(rgb: np.ndarray, error: np.ndarray, alpha: float = OVERLAY_ALPHA, dim: float = RGB_DIM) -> np.ndarray:
    """The RGB dimmed to dim, with the disagreement colours at alpha opacity where it is finite."""
    base = rgb.astype(np.float64) * dim
    colours = colorize(error, 0.0, ERROR_MAX, ERROR_CMAP).astype(np.float64)
    finite = np.isfinite(error)[..., None]
    return np.rint(np.where(finite, (1 - alpha) * base + alpha * colours, base)).astype(np.uint8)


def frame_time(cur: Loaded, t0: float) -> str:
    return f"frame {cur.index}   t {cur.timestamp - t0:.2f} s"


def draw_hero(cur: Loaded, limits: tuple[float, float], name: str, t0: float) -> np.ndarray:
    image, draw = canvas()
    text(draw, (24, 30), f"{short_name(name)}   one depth network, raw per frame and fused over time", 44)
    panels = (("RGB", cur.rgb), ("raw depth", depth_image(cur.raw.depth, limits)),
              ("fused depth", depth_image(cur.fused.depth, limits)))
    for (title, array), box in zip(panels, THREE):
        place(image, array, box)
        text(draw, (box[0] + box[2] / 2, box[1] - 18), title, 46, anchor="ms")
    if not cur.has_fused:
        badge(draw, (THREE[2][0] + THREE[2][2] / 2, THREE[2][1] + 40), "no fusion yet, raw shown")
    x = THREE[1][0]
    width = THREE[2][0] + THREE[2][2] - x
    colour_bar(image, draw, (x, 720, width, 34), limits, DEPTH_CMAP, depth_ticks(limits))
    text(draw, (x + width / 2, 840), "depth in metres, one fixed range for every frame", 30, anchor="ma")
    text(draw, (SIZE[0] / 2, 950), "fused depth combines 4 earlier frames, oracle GT poses (an upper bound), joint "
                                   "solved scale", 34, YELLOW, "ma")
    text(draw, (SIZE[0] - 24, 1050), frame_time(cur, t0), 26, anchor="rs")
    return np.asarray(image)


def flicker_legend(image: Image.Image, draw: ImageDraw.ImageDraw, y: int) -> None:
    box = (480, y, 960, 34)
    colour_bar(image, draw, box, (0.0, ERROR_MAX), ERROR_CMAP,
               [(0.0, "0 %"), (ERROR_MAX / 2, f"{50 * ERROR_MAX:g} %"), (ERROR_MAX, f"{100 * ERROR_MAX:g} % or more")],
               26)
    text(draw, (box[0] - 24, y + 17), "stable", 40, anchor="rm")
    text(draw, (box[0] + box[2] + 24, y + 17), "flickers", 40, anchor="lm")


def draw_flicker(data: FrameData, name: str, t0: float) -> np.ndarray:
    cur = data.cur
    image, draw = canvas()
    text(draw, (24, 30), f"{short_name(name)}   how much the depth changes from one frame to the next", 40)
    text(draw, (SIZE[0] - 24, 40), frame_time(cur, t0), 26, anchor="ra")
    panels = (("raw", data.error_raw, data.has_previous), ("fused", data.error_fused, data.paired))
    for (label, error, shown), box in zip(panels, TWO):
        x, y, w, h = box
        place(image, overlay(cur.rgb, error if shown else np.full(error.shape, np.nan)), box)
        text(draw, (x, y - 20), label, 50, COLOURS[label], "ls")
        value = f"median {100 * map_median(error):.2f} %" if shown else "median n/a"
        text(draw, (x + w, y - 20), value, 56, anchor="rs")
        if not data.has_previous:
            badge(draw, (x + w / 2, y + h / 2), "no previous frame within 50 ms")
        elif label == "fused" and not shown:
            badge(draw, (x + w / 2, y + h / 2), "no fusion on this frame or the previous one")
    flicker_legend(image, draw, 930)
    text(draw, (SIZE[0] / 2, 1052), "previous depth warped in by oracle GT poses minus this depth, over GT depth, "
                                    "shown at 60 % over the dimmed RGB", 28, YELLOW, "ms")
    return np.asarray(image)


def draw_single(cur: Loaded, source: str, limits: tuple[float, float], name: str, t0: float, title: str,
                note: str) -> np.ndarray:
    image, draw = canvas()
    depth = cur.raw.depth if source == "raw" else cur.fused.depth
    place(image, depth_image(depth, limits), BIG)
    text(draw, (24, 30), f"{short_name(name)}   {title}", 40)
    text(draw, (SIZE[0] - 24, 40), "oracle GT poses, joint solved scale", 26, YELLOW, "ra")
    text(draw, (40, 400), "on screen", 36)
    text(draw, (40, 450), source.upper(), 96, COLOURS[source])
    draw.multiline_text((40, 600), note, font=font(28), fill=LIGHT_TEXT, spacing=10)
    if source == "fused" and not cur.has_fused:
        badge(draw, (BIG[0] + BIG[2] / 2, BIG[1] + 50), "no fusion on this frame, raw shown")
    text(draw, (BIG[0] + BIG[2] + 40, 400), f"frame {cur.index}", 30)
    text(draw, (BIG[0] + BIG[2] + 40, 445), f"t {cur.timestamp - t0:.2f} s", 30)
    colour_bar(image, draw, (BIG[0], 970, BIG[2], 28), limits, DEPTH_CMAP, depth_ticks(limits), 24)
    return np.asarray(image)


def toggle_sources(count: int, fps: float, seconds: float) -> list[str]:
    """raw or fused for each of count frames, switching every seconds of playback at fps, starting with raw."""
    per_state = max(1, round(seconds * fps))
    return ["raw" if (k // per_state) % 2 == 0 else "fused" for k in range(count)]


# Point clouds.

def rotation_y(degrees: float) -> np.ndarray:
    c, s = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rotation_x(degrees: float) -> np.ndarray:
    c, s = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def virtual_pose(yaw_deg: float, pitch_deg: float, pivot_m: float) -> np.ndarray:
    """T_cam_virtual, the original camera turned by yaw about its y axis and pitch about its x axis around the point
    pivot_m ahead of it on the optical axis, still looking at that point. The identity at zero angles."""
    R = rotation_y(yaw_deg) @ rotation_x(pitch_deg)
    pivot = np.array([0.0, 0.0, pivot_m])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = pivot - pivot_m * R[:, 2]
    return T


def sway_angles(t: float, period: float, yaw_deg: float, pitch_deg: float) -> tuple[float, float]:
    """Yaw and pitch at time t of a figure eight sway that starts and ends each period at zero."""
    phase = 2 * math.pi * t / period
    return yaw_deg * math.sin(phase), pitch_deg * math.sin(2 * phase)


def panel_intrinsics(K: np.ndarray, source_shape: tuple[int, int], panel: tuple[int, int]) -> np.ndarray:
    sx, sy = panel[0] / source_shape[1], panel[1] / source_shape[0]
    return np.array([[K[0, 0] * sx, 0.0, (K[0, 2] + 0.5) * sx - 0.5],
                     [0.0, K[1, 1] * sy, (K[1, 2] + 0.5) * sy - 0.5], [0.0, 0.0, 1.0]])


@dataclass(frozen=True)
class Cloud:
    points: np.ndarray
    colours: np.ndarray
    size_m: np.ndarray


def make_cloud(depth: np.ndarray, rgb: np.ndarray, mask: np.ndarray, K: np.ndarray) -> Cloud:
    """The masked pixels back projected into the camera frame with their RGB colour, each splat about SPLAT_FACTOR
    times its pixel footprint."""
    v, u = np.nonzero(mask & np.isfinite(depth) & (depth > 0))
    z = depth[v, u].astype(np.float64)
    points = np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], axis=1)
    return Cloud(points, rgb[v, u], SPLAT_FACTOR * z / K[0, 0])


def render_cloud(cloud: Cloud, T: np.ndarray, K_panel: np.ndarray, K_source: np.ndarray,
                 panel: tuple[int, int]) -> np.ndarray:
    max_px = max(2, math.ceil(2 * SPLAT_FACTOR * K_panel[0, 0] / K_source[0, 0]))
    return splat(cloud.points, cloud.colours, T, K_panel, panel, cloud.size_m, max_px, WHITE)


def cloud_mask(cur: Loaded) -> tuple[np.ndarray, int]:
    """The pixels shown in every cloud panel, and how many valid pixels the depth edge band removed."""
    gt_valid = cur.raw.gt_valid.astype(bool)
    raw, fused = cur.raw.depth, cur.fused.depth
    base = gt_valid & np.isfinite(raw) & np.isfinite(fused) & (raw > 0) & (fused > 0)
    mask = frame_mask(cur.raw.gt_depth, gt_valid, raw, fused)
    return mask, int((base & ~mask).sum())


def make_clouds(cur: Loaded, K: np.ndarray) -> tuple[dict[str, Cloud], int]:
    mask, dropped = cloud_mask(cur)
    gt = np.where(cur.raw.gt_valid, cur.raw.gt_depth, np.nan).astype(np.float64)
    depths = {"raw": cur.raw.depth, "fused": cur.fused.depth, "gt": gt}
    return {s: make_cloud(d.astype(np.float64), cur.rgb, mask, K) for s, d in depths.items()}, dropped


def cloud_panels(clouds: dict[str, Cloud], T: np.ndarray, K_source: np.ndarray,
                 source_shape: tuple[int, int], panel: tuple[int, int]) -> list[np.ndarray]:
    """One image per source of CLOUD_PANELS, all from the same virtual camera T."""
    K_panel = panel_intrinsics(K_source, source_shape, panel)
    return [render_cloud(clouds[s], T, K_panel, K_source, panel) for s, _ in CLOUD_PANELS]


def draw_cloud(panels: list[np.ndarray], name: str, title: str, status: str, has_fused: bool) -> np.ndarray:
    image, draw = canvas(WHITE)
    text(draw, (24, 30), f"{short_name(name)}   {title}", 40, DARK_TEXT)
    for (source, label), array, box in zip(CLOUD_PANELS, panels, THREE):
        x, y, w, h = box
        image.paste(Image.fromarray(array), (x, y))
        draw.rectangle((x - 1, y - 1, x + w, y + h), outline=(200, 200, 200))
        text(draw, (x + w / 2, y - 22), label, 56, COLOURS[source], "ms")
    if not has_fused:
        badge(draw, (THREE[1][0] + THREE[1][2] / 2, THREE[1][1] + 40), "no fusion on this frame, raw shown")
    text(draw, (SIZE[0] / 2, 760), status, 34, DARK_TEXT, "ma")
    text(draw, (SIZE[0] / 2, 840), "every pixel back projected with its RGB colour, depth edge band dropped, white "
                                   "is no data", 30, DARK_TEXT, "ma")
    text(draw, (SIZE[0] / 2, 1040), "oracle GT poses, an upper bound, joint solved scale", 28, (150, 110, 0), "ms")
    return np.asarray(image)


def cloud_panel_size() -> tuple[int, int]:
    return THREE[0][2], THREE[0][3]


# Output.

class Sink:
    """The mp4 and the first gif_seconds of it as a half size gif at no more than GIF_MAX_FPS."""

    def __init__(self, out_dir: Path, stem: str, fps: float, gif_seconds: float) -> None:
        self.out_dir, self.stem, self.fps = out_dir, stem, fps
        self.video = VideoWriter(out_dir / f"{stem}.mp4", SIZE, fps)
        self.gif_step = max(1, round(fps / GIF_MAX_FPS))
        self.gif_limit = max(1, round(gif_seconds * fps))
        self.gif_frames: list[np.ndarray] = []
        self.count = 0

    def write(self, frame: np.ndarray, repeat: int = 1) -> None:
        half = None
        for _ in range(repeat):
            self.video.write(frame)
            if self.count < self.gif_limit and self.count % self.gif_step == 0:
                half = half_size(frame) if half is None else half
                self.gif_frames.append(half)
            self.count += 1

    def close(self) -> dict[str, object]:
        video = self.video.close()
        gif = write_gif(self.out_dir / f"{self.stem}.gif", self.gif_frames, self.fps / self.gif_step)
        return {"video": str(video), "gif": str(gif), "video_frames": self.count,
                "gif_frames": len(self.gif_frames), "gif_fps": self.fps / self.gif_step}


@dataclass(frozen=True)
class Context:
    args: argparse.Namespace
    inputs: SequenceInputs
    start: int
    stop: int
    series: Series
    limits: tuple[float, float]
    fps: float
    hold_seconds: float
    t0: float


def run_hero(ctx: Context, sink: Sink) -> dict[str, object]:
    positions = range(ctx.start, ctx.stop)
    for position, repeat in zip(positions, hold_repeats(len(positions), ctx.fps, ctx.hold_seconds)):
        sink.write(draw_hero(ctx.inputs.load(position, with_rgb=True), ctx.limits, ctx.inputs.name, ctx.t0), repeat)
    return {}


def run_flicker(ctx: Context, sink: Sink) -> dict[str, object]:
    datas = ctx.inputs.iterate(ctx.start, ctx.stop, with_rgb=True)
    for data, repeat in zip(datas, hold_repeats(ctx.stop - ctx.start, ctx.fps, ctx.hold_seconds)):
        sink.write(draw_flicker(data, ctx.inputs.name, ctx.t0), repeat)
    return {}


def run_toggle(ctx: Context, sink: Sink) -> dict[str, object]:
    seconds = ctx.args.toggle_seconds
    sources = toggle_sources(ctx.stop - ctx.start, ctx.fps, seconds)
    note = f"switches every\n{seconds:g} s"
    repeats = hold_repeats(len(sources), ctx.fps, ctx.hold_seconds)
    for position, source, repeat in zip(range(ctx.start, ctx.stop), sources, repeats):
        cur = ctx.inputs.load(position, with_rgb=True)
        sink.write(draw_single(cur, source, ctx.limits, ctx.inputs.name, ctx.t0, "raw and fused depth taking turns",
                               note), repeat)
    return {"toggle": {"seconds": seconds, "frames_per_state": max(1, round(seconds * ctx.fps)),
                       "sources": sources}}


def run_stillflip(ctx: Context, sink: Sink) -> dict[str, object]:
    position, choice = choose_frame(ctx.inputs, ctx.start, ctx.series, ctx.args.frame)
    data = frame_data(ctx.inputs, position)
    choice["gain_pct"] = _pct(gain_of(data))
    seconds = ctx.args.toggle_seconds
    count = 2 * STILLFLIP_CYCLES * max(1, round(seconds * ctx.fps))
    note = f"one frame,\nswitches every\n{seconds:g} s"
    title = "the same frame, raw and fused depth taking turns"
    frames = {s: draw_single(data.cur, s, ctx.limits, ctx.inputs.name, ctx.t0, title, note) for s in ("raw", "fused")}
    for source in toggle_sources(count, ctx.fps, seconds):
        sink.write(frames[source])
    poster = ctx.args.out_dir / f"{ctx.inputs.name}_stillflip_poster.png"
    Image.fromarray(draw_flicker(data, ctx.inputs.name, ctx.t0)).save(poster)
    return {"frame": choice, "stillflip": {"seconds": seconds, "cycles": STILLFLIP_CYCLES, "frames": count,
                                           "paired": data.paired}, "outputs": {"poster": str(poster)}}


def edge_record(dropped: int | list[int]) -> dict[str, object]:
    return {"dropped": True, "dropped_pixels": dropped,
            "rule": "valid pixels within the depth_edge_band of the GT, raw or fused depth (relative jump 0.04, "
                    "width 5 px) are left out of every panel, as export_fused_map.py does, so no stretched "
                    "triangles or flying pixels appear"}


def cloud_settings(ctx: Context) -> dict[str, object]:
    pivot = sum(ctx.limits) / 2
    return {"pivot_m": round(pivot, 4),
            "pivot_rule": "the camera turns about the point on the optical axis at the middle of the fixed depth "
                          "range, so the pivot is the same on every frame",
            "splat_factor": SPLAT_FACTOR, "panel_size": list(cloud_panel_size()), "background": "white",
            "camera": "one virtual camera shared by the raw, fused and GT panels",
            "frame_space": "camera frame of each frame, GT depth unwarped from the depth time"}


def run_cloud(ctx: Context, sink: Sink) -> dict[str, object]:
    args = ctx.args
    position, choice = choose_frame(ctx.inputs, ctx.start, ctx.series, args.frame)
    data = frame_data(ctx.inputs, position)
    choice["gain_pct"] = _pct(gain_of(data))
    cur = data.cur
    clouds, dropped = make_clouds(cur, ctx.inputs.K)
    settings = cloud_settings(ctx)
    count = max(1, round(args.cloud_seconds * ctx.fps))
    title = f"frame {cur.index} as a point cloud, seen from a swaying camera"
    for k, repeat in zip(range(count), hold_repeats(count, ctx.fps, ctx.hold_seconds)):
        yaw, pitch = sway_angles(k / ctx.fps, args.cloud_seconds, args.sway_degrees, args.pitch_degrees)
        T = virtual_pose(yaw, pitch, settings["pivot_m"])
        panels = cloud_panels(clouds, T, ctx.inputs.K, cur.raw.depth.shape, cloud_panel_size())
        sink.write(draw_cloud(panels, ctx.inputs.name, title, f"yaw {yaw:+.1f} deg   pitch {pitch:+.1f} deg",
                              cur.has_fused), repeat)
    sway = {"yaw_degrees": args.sway_degrees, "pitch_degrees": args.pitch_degrees, "period_s": args.cloud_seconds,
            "rule": "yaw = A sin(2 pi t / period), pitch = B sin(4 pi t / period), a figure eight that starts and "
                    "ends at the original camera"}
    return {"frame": choice, "cloud": {**settings, "sway": sway, "points": len(clouds["raw"].points)},
            "edge_band": edge_record(dropped)}


def run_cloud_moving(ctx: Context, sink: Sink) -> dict[str, object]:
    args = ctx.args
    settings = cloud_settings(ctx)
    T = virtual_pose(args.yaw_offset, 0.0, settings["pivot_m"])
    dropped = []
    title = f"every frame as a point cloud, camera turned {args.yaw_offset:g} degrees"
    positions = range(ctx.start, ctx.stop)
    for position, repeat in zip(positions, hold_repeats(len(positions), ctx.fps, ctx.hold_seconds)):
        cur = ctx.inputs.load(position, with_rgb=True)
        clouds, n = make_clouds(cur, ctx.inputs.K)
        dropped.append(n)
        panels = cloud_panels(clouds, T, ctx.inputs.K, cur.raw.depth.shape, cloud_panel_size())
        sink.write(draw_cloud(panels, ctx.inputs.name, title, frame_time(cur, ctx.t0), cur.has_fused), repeat)
    frames = [int(ctx.inputs.frames[p].index) for p in positions]
    return {"frame": {"chosen_by": "--start and --max-frames", "indices": frames,
                      "rule": "every clip frame, each with its own RGB and depth"},
            "cloud": {**settings, "yaw_degrees": args.yaw_offset, "pitch_degrees": 0.0},
            "edge_band": edge_record(dropped)}


RUNNERS = {"hero": run_hero, "flicker": run_flicker, "toggle": run_toggle, "stillflip": run_stillflip,
           "cloud": run_cloud, "cloud_moving": run_cloud_moving}


# Command line.

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("render_dir", type=Path, help="fusion/render/<run name>, the folder holding sequence folders")
    parser.add_argument("data_root", type=Path, help="folder containing the sequence folders")
    parser.add_argument("sequences_config", type=Path)
    parser.add_argument("sequence")
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument("--start", type=int, default=0, help="first clip frame as a position in the render folder")
    parser.add_argument("--max-frames", type=int, help="clip length, the rest of the window by default")
    parser.add_argument("--hold-seconds", type=float,
                        help=f"extra time on the first and last frame, {HERO_HOLD_S:g} for hero and 0 otherwise")
    parser.add_argument("--frame", type=int, help="frame index for stillflip and cloud, the median gain by default")
    parser.add_argument("--toggle-seconds", type=float, default=TOGGLE_S)
    parser.add_argument("--gif-seconds", type=float, default=GIF_SECONDS)
    parser.add_argument("--cloud-seconds", type=float, default=CLOUD_SECONDS, help="cloud length and sway period")
    parser.add_argument("--sway-degrees", type=float, default=SWAY_DEG, help="largest yaw of the cloud sway")
    parser.add_argument("--pitch-degrees", type=float, default=PITCH_DEG, help="largest pitch of the cloud sway")
    parser.add_argument("--yaw-offset", type=float, default=MOVING_YAW_DEG, help="fixed yaw of cloud_moving")
    args = parser.parse_args(argv)
    if args.fps <= 0:
        parser.error("--fps must be positive")
    return args


def clip_medians(series: Series) -> dict[str, object]:
    fused = series.has_fused

    def median(values: np.ndarray) -> float | None:
        values = values[np.isfinite(values)]
        return round(float(np.median(values)), 4) if values.size else None

    return {**summary(series), "median_rel_d_raw_x100": median(series.rel_d_raw),
            "median_rel_d_raw_on_fused_frames_x100": median(series.rel_d_raw[fused]),
            "median_rel_d_fused_x100": median(series.rel_d_fused[fused])}


def main(argv: list[str] | None = None) -> dict[str, object]:
    args = parse_args(argv)
    inputs = SequenceInputs(args.render_dir, args.data_root, args.sequences_config, args.sequence)
    start, stop = clip_bounds(len(inputs), args.start, args.max_frames)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{args.sequence} {args.mode}, clip positions {start} to {stop - 1}, scanning", flush=True)
    series = scan_clip(inputs, start, stop)
    limits = window_depth_range(inputs)
    hold = args.hold_seconds if args.hold_seconds is not None else (HERO_HOLD_S if args.mode == "hero" else 0.0)
    ctx = Context(args, inputs, start, stop, series, limits, args.fps, hold, float(inputs.frames[0].timestamp))
    stem = f"{args.sequence}_{args.mode}"
    sink = Sink(args.out_dir, stem, args.fps, args.gif_seconds)
    extra = RUNNERS[args.mode](ctx, sink)
    outputs = {**sink.close(), **extra.pop("outputs", {})}

    record = {
        "scale": SCALE,
        "poses": POSES,
        "label": FOOTER,
        "mode": args.mode,
        "sequence": args.sequence,
        "settings": {
            **{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
            "hold_seconds": hold,
            "size": list(SIZE),
            "depth_colormap": DEPTH_CMAP,
            "depth_range_m": [round(v, 4) for v in limits],
            "depth_range_rule": f"GT depth percentiles {DEPTH_PERCENTILES} over the whole window, every "
                                f"{GT_SAMPLE_STRIDE}th valid pixel, as render_video.py",
            "error_colormap": ERROR_CMAP,
            "error_range": [0.0, ERROR_MAX],
            "overlay_alpha": OVERLAY_ALPHA,
            "rgb_dim": RGB_DIM,
        },
        "clip": {"start_position": start, "stop_position": stop, "start_index": int(series.index[0]),
                 "last_index": int(series.index[-1]), "frames": stop - start},
        "clip_medians": clip_medians(series),
        **extra,
        "outputs": outputs,
    }
    path = args.out_dir / f"{stem}.json"
    path.write_text(json.dumps(record, indent=2) + "\n")
    print(f"wrote {outputs['video']}, {outputs['gif']} and {path}")
    return record


def _pct(value: float) -> float | None:
    return round(100 * float(value), 4) if math.isfinite(value) else None


if __name__ == "__main__":
    main()
