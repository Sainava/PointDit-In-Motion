"""World frame video of raw and fused depth (optionally GT) from fusion_render_prepare.py, with JOINT-SOLVED scale and
ORACLE GT poses, side by side from one virtual camera on a slow fixed orbit around the scene centre.

Every --stride th frame with fused depth adds --frame-points points per source, on the same pixels for every source
and with the depth edge band dropped as in export_fused_map.py. --mode accumulate shows every point added so far, so
the map grows, the oldest points thinned uniformly to keep at most --max-points per source. --mode sliding shows only
the points of the last --window dataset frames. The framing comes from GT depth over the whole clip before rendering.
Writes a 1920x1080 mp4, a half size gif and a json of settings, point counts and the ghosting statistic of the final
accumulated map per source."""

from __future__ import annotations

import argparse
import json
import math
import resource
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from export_fused_map import VOXEL_M, frame_mask, ghosting, voxel_planes, world_points  # noqa: E402
from render_orbit import BACKGROUND, DIVIDER, SIDE_COLOURS, draw_text, intrinsics, orbit_poses, splat  # noqa: E402
from render_video import POSES, SCALE, SequenceInputs  # noqa: E402

from pim.consistency.warp import depth_pose  # noqa: E402
from pim.video import VideoWriter, font, half_size, write_gif  # noqa: E402

MODES = ("accumulate", "sliding")
SIZE = (1920, 1080)
HEADER_PX = 112
LABEL_PX = 60
FOOTER_PX = 48
DISCLAIMER = "oracle GT poses, upper bound, joint solved scale, not dynamic 4D reconstruction"
DEFAULT_STRIDE = 3
DEFAULT_WINDOW = 30
MAX_POINTS = 400_000
FRAME_POINTS = 40_000
DEFAULT_FPS = 10.0
GIF_MAX_FRAMES = 150
FRAMING_PIXEL_STRIDE = 37
FRUSTUM_M = 0.3
FRUSTUM_COLOUR = (255, 209, 102)
FOV_DEG = 60.0
POINT_SIZE_M = 0.012
SEED = 0


class PointStore:
    """Points of several sources on shared pixels, one colour and one dataset frame position per point, never more
    than cap per source. Adding past the cap thins the older points uniformly at random and keeps the new frame."""

    def __init__(self, sources: tuple[str, ...], cap: int, rng: np.random.Generator) -> None:
        self.sources, self.cap, self.rng = sources, cap, rng
        self.points = {s: np.empty((0, 3), dtype=np.float32) for s in sources}
        self.colors = np.empty((0, 3), dtype=np.uint8)
        self.frame = np.empty(0, dtype=np.int32)

    def __len__(self) -> int:
        return len(self.frame)

    def _take(self, keep: np.ndarray) -> None:
        self.points = {s: p[keep] for s, p in self.points.items()}
        self.colors, self.frame = self.colors[keep], self.frame[keep]

    def add(self, position: int, points: dict[str, np.ndarray], colors: np.ndarray) -> None:
        if len(colors) > self.cap:
            pick = np.sort(self.rng.choice(len(colors), self.cap, replace=False))
            points, colors = {s: p[pick] for s, p in points.items()}, colors[pick]
        room = self.cap - len(colors)
        if len(self) > room:
            self._take(np.sort(self.rng.choice(len(self), room, replace=False)))
        self.points = {s: np.concatenate([self.points[s], points[s]]) for s in self.sources}
        self.colors = np.concatenate([self.colors, colors])
        self.frame = np.concatenate([self.frame, np.full(len(colors), position, dtype=np.int32)])

    def drop_before(self, position: int) -> None:
        self._take(self.frame >= position)


def framing(inputs: SequenceInputs, positions: list[int], pixel_stride: int = FRAMING_PIXEL_STRIDE) -> dict:
    """Orbit centre, up, radius and start direction from sparse GT depth of every chosen frame, as export_fused_map.py
    frames its orbit."""
    samples = []
    for position in positions:
        frame = inputs.frames[position]
        depth, valid = frame.load_depth()
        T = depth_pose(frame, *inputs.gt_trajectory)
        pixels = np.flatnonzero(valid)[::pixel_stride]
        samples.append(world_points(depth, inputs.K, T, pixels))
    gt = np.concatenate(samples)
    path = [inputs.frames[p].T_world_cam for p in positions]
    up = -np.mean([T[:3, 1] for T in path], axis=0)
    up /= np.linalg.norm(up)
    centre = np.median(gt, axis=0)
    radius = float(np.percentile(np.linalg.norm(gt - centre, axis=1), 90))
    start = np.mean([T[:3, 3] for T in path], axis=0) - centre
    return {"centre": centre.round(4).tolist(), "up": up.round(6).tolist(), "radius_m": round(max(radius, 1e-3), 4),
            "start_direction": start.round(4).tolist()}


def frustum_segments(T_cam: np.ndarray, K_cam: np.ndarray, image_size: tuple[int, int], T_view: np.ndarray,
                     K_view: np.ndarray, depth_m: float = FRUSTUM_M) -> list[tuple[float, float, float, float]]:
    """Pixel segments of the real camera's frustum to depth_m seen from the virtual camera, dropping any segment
    with an end behind it."""
    width, height = image_size
    corners = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float64)
    rays = np.stack([(corners[:, 0] - K_cam[0, 2]) / K_cam[0, 0], (corners[:, 1] - K_cam[1, 2]) / K_cam[1, 1],
                     np.ones(4)], axis=1) * depth_m
    local = np.vstack([np.zeros(3), rays])
    world = local @ T_cam[:3, :3].T + T_cam[:3, 3]
    p = (world - T_view[:3, 3]) @ T_view[:3, :3]
    front = p[:, 2] > 1e-3
    z = np.where(front, p[:, 2], 1.0)
    uv = np.stack([K_view[0, 0] * p[:, 0] / z + K_view[0, 2], K_view[1, 1] * p[:, 1] / z + K_view[1, 2]], axis=1)
    edges = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1)]
    return [(*uv[a], *uv[b]) for a, b in edges if front[a] and front[b]]


def draw_frustum(image: np.ndarray, segments: list[tuple[float, float, float, float]]) -> np.ndarray:
    canvas = Image.fromarray(image)
    draw = ImageDraw.Draw(canvas)
    for segment in segments:
        draw.line(segment, fill=FRUSTUM_COLOUR, width=2)
    return np.asarray(canvas)


def compose(panels: list[np.ndarray], labels: list[tuple[str, str]], header: list[str], footer: str,
            panel: tuple[int, int]) -> np.ndarray:
    frame = np.empty((SIZE[1], SIZE[0], 3), dtype=np.uint8)
    frame[:] = BACKGROUND
    top = HEADER_PX + LABEL_PX
    canvas = Image.fromarray(frame)
    draw = ImageDraw.Draw(canvas)
    for p, (image, (source, label)) in enumerate(zip(panels, labels)):
        x0 = p * panel[0]
        canvas.paste(Image.fromarray(image), (x0, top))
        draw.text((x0 + panel[0] // 2, HEADER_PX + LABEL_PX // 2), label, font=font(44), fill=SIDE_COLOURS[source],
                  anchor="mm")
        if p:
            draw.rectangle((x0 - 2, HEADER_PX, x0 + 1, SIZE[1] - FOOTER_PX), fill=DIVIDER)
    frame = np.asarray(canvas)
    return draw_text(frame, [((16, 12), header[0], 32, (255, 255, 255)), ((16, 60), header[1], 26, FRUSTUM_COLOUR),
                             ((16, SIZE[1] - FOOTER_PX + 10), footer, 24, (220, 220, 220))])


def peak_rss_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(peak / (1e6 if sys.platform == "darwin" else 1e3), 1)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("render_dir", type=Path, help="fusion/render/<run name>, the folder holding sequence folders")
    parser.add_argument("data_root", type=Path, help="folder containing the sequence folders")
    parser.add_argument("sequences_config", type=Path)
    parser.add_argument("sequence")
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--mode", choices=MODES, default="accumulate")
    parser.add_argument("--with-gt", action="store_true", help="add GT as a third panel")
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE, help="every stride th frame with fused depth")
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW, help="sliding mode, in dataset frames")
    parser.add_argument("--max-points", type=int, default=MAX_POINTS, help="per source")
    parser.add_argument("--frame-points", type=int, default=FRAME_POINTS, help="points each frame adds per source")
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument("--arc", type=float, default=90.0, help="orbit arc in degrees over the whole clip")
    parser.add_argument("--elevation", type=float, default=35.0, help="orbit elevation in degrees")
    parser.add_argument("--distance-scale", type=float, default=1.0)
    parser.add_argument("--fov", type=float, default=FOV_DEG, help="horizontal field of view of each panel")
    parser.add_argument("--point-size", type=float, default=POINT_SIZE_M, help="splat edge in metres")
    parser.add_argument("--voxel", type=float, default=VOXEL_M, help="ghosting voxel edge in metres")
    parser.add_argument("--max-frames", type=int)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, object]:
    args = parse_args(argv)
    inputs = SequenceInputs(args.render_dir, args.data_root, args.sequences_config, args.sequence, args.max_frames)
    fused_positions = []
    for position, path in enumerate(inputs.paths):
        with np.load(path) as data:
            if bool(data["has_fused"]):
                fused_positions.append(position)
    chosen = fused_positions[::args.stride]
    if not chosen:
        raise ValueError(f"{args.sequence} has no frame with fused depth")
    panels = [("raw", "raw"), ("fused", "fused")] + ([("gt", "GT")] if args.with_gt else [])
    panel = (SIZE[0] // len(panels), SIZE[1] - HEADER_PX - LABEL_PX - FOOTER_PX)
    K_view = intrinsics(*panel, args.fov)
    orbit = framing(inputs, chosen)
    poses = orbit_poses(orbit, len(chosen), args.arc, args.elevation, args.distance_scale, args.fov)

    sources = ("raw", "fused", "gt")
    rng = np.random.default_rng(SEED)
    accumulated = PointStore(sources, args.max_points, rng)
    window = PointStore(sources, args.max_points, rng) if args.mode == "sliding" else None
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{args.sequence}_world_{args.mode}"
    video = VideoWriter(out / f"{stem}.mp4", SIZE, args.fps)
    gif_step = math.ceil(len(chosen) / GIF_MAX_FRAMES)
    gif_frames = []
    name = args.sequence.replace("rgbd_dataset_freiburg", "fr")
    what = ("every point so far" if args.mode == "accumulate"
            else f"points of the last {args.window} dataset frames")
    title = f"{name}   world frame map, {what}, fused frames at stride {args.stride}"
    shown_counts, total_counts = [], []
    for k, (position, T_view) in enumerate(zip(chosen, poses)):
        loaded = inputs.load(position, with_rgb=True)
        raw, fused = loaded.raw, loaded.fused
        pixels = np.flatnonzero(frame_mask(raw.gt_depth, raw.gt_valid, raw.depth, fused.depth))
        pixels = np.sort(rng.choice(pixels, min(args.frame_points, pixels.size), replace=False))
        points = {"raw": world_points(raw.depth, inputs.K, raw.T_rgb, pixels),
                  "fused": world_points(fused.depth, inputs.K, fused.T_rgb, pixels),
                  "gt": world_points(raw.gt_depth, inputs.K, raw.T_depth, pixels)}
        colors = loaded.rgb.reshape(-1, 3)[pixels]
        accumulated.add(position, points, colors)
        shown = accumulated
        if window is not None:
            window.add(position, points, colors)
            window.drop_before(position - args.window + 1)
            shown = window
        shown_counts.append(len(shown))
        total_counts.append(len(accumulated))
        segments = frustum_segments(raw.T_rgb, inputs.K, (raw.gt_depth.shape[1], raw.gt_depth.shape[0]), T_view,
                                    K_view)
        images = [draw_frustum(splat(shown.points[s], shown.colors, T_view, K_view, panel, args.point_size),
                               segments) for s, _ in panels]
        t = loaded.timestamp - inputs.frames[chosen[0]].timestamp
        footer = (f"frame {k + 1} / {len(chosen)}   dataset frame {loaded.index}   t = {t:.2f} s   "
                  f"{len(shown):,} points per panel   yellow is the current real camera")
        frame = compose(images, panels, [title, DISCLAIMER], footer, panel)
        video.write(frame)
        if k % gif_step == 0:
            gif_frames.append(half_size(frame))
    video_path = video.close()
    gif_path = write_gif(out / f"{stem}.gif", gif_frames, args.fps / gif_step)

    centres = np.array([inputs.frames[p].T_world_cam[:3, 3] for p in chosen])
    planes = voxel_planes(accumulated.points["gt"], centres.mean(axis=0), args.voxel)
    record = {
        "scale": SCALE,
        "poses": POSES,
        "label": DISCLAIMER,
        "sequence": args.sequence,
        "mode": args.mode,
        "coordinates": "TUM world frame in metres, cameras in the OpenCV convention",
        "settings": {**{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
                     "size": list(SIZE), "panels": [s for s, _ in panels], "seed": SEED,
                     "gif": {"fps": args.fps / gif_step, "half_size": True, "frame_step": gif_step},
                     "pixels": "GT valid, finite predictions, outside depth_edge_band of GT, raw and fused, the "
                               "same pixels for every source",
                     "cap_rule": "past max_points the older points are thinned uniformly at random, the newest "
                                 "frame is kept whole"},
        "framing": {**orbit, "rule": "GT depth of every chosen frame, every "
                                     f"{FRAMING_PIXEL_STRIDE}th valid pixel; median centre, 90th percentile radius"},
        "frames_used": [inputs.frames[p].index for p in chosen],
        "points": {"shown_per_frame": shown_counts, "accumulated_per_frame": total_counts,
                   "final_shown": {s: len(shown) for s in sources},
                   "final_accumulated": {s: len(accumulated) for s in sources}},
        "ghosting": {s: ghosting(accumulated.points[s], planes) for s in sources},
        "ghosting_rule": f"export_fused_map.py ghosting on the final accumulated map, signed distance to the GT "
                         f"plane of the {args.voxel:g} m voxel a point falls in; std over covered points",
        "peak_rss_mb": peak_rss_mb(),
        "outputs": {"video": str(video_path), "gif": str(gif_path)},
    }
    record_path = out / f"{stem}.json"
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    for source in sources:
        g = record["ghosting"][source]
        print(f"{source} ghosting std {g['std_m']} m, covered {g['covered_fraction']}")
    print(f"wrote {video_path}, {gif_path} and {record_path}, peak RSS {record['peak_rss_mb']} MB")
    return record


if __name__ == "__main__":
    main()
