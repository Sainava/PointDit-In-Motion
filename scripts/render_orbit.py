"""Orbit video of the raw, fused and GT point maps of export_fused_map.py side by side, with JOINT-SOLVED scale and
ORACLE GT poses, by numpy point splatting with a z-buffer on the CPU.

The camera circles the map centre at an elevation, looking at the centre, for --seconds at --fps into an mp4 and a
half size gif, the first and last frame held for --hold-seconds. --side-by-side draws the three maps under large
labels with a one line header instead of the default layout. --flythrough instead renders the fused map alone from
the recorded camera path, then one second beyond it with a growing sideways offset."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from export_fused_map import MAP_JSON, read_ply  # noqa: E402

from pim.video import VideoWriter, font, half_size, hold_repeats, write_gif  # noqa: E402

SIZE = (1920, 1080)
HEADER_PX = 110
PANELS = (("raw", "raw"), ("fused", "fused"), ("gt", "GT"))
BACKGROUND = (16, 16, 16)
FOV_DEG = 60.0
NEAR_M = 0.05
POINT_SIZE_M = 0.012
MAX_SPLAT_PX = 4
GIF_FPS = 10.0
FLY_EXTRA_S = 1.0
LATERAL_OFFSET_M = 0.2
DEFAULT_FPS = 10.0
SIDE_HEADER_PX = 70
SIDE_LABEL_PX = 100
SIDE_FOOTER_PX = 60
SIDE_COLOURS = {"raw": (242, 142, 43), "fused": (78, 159, 229), "gt": (235, 235, 235)}
DIVIDER = (90, 90, 90)


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """T_world_cam of an OpenCV camera (x right, y down, z forward) at eye looking at target."""
    z = target - eye
    z = z / np.linalg.norm(z)
    x = np.cross(-up, z)
    x = x / np.linalg.norm(x)
    T = np.eye(4)
    T[:3, :3] = np.stack([x, np.cross(z, x), z], axis=1)
    T[:3, 3] = eye
    return T


def intrinsics(width: int, height: int, fov_deg: float = FOV_DEG) -> np.ndarray:
    f = (width / 2) / math.tan(math.radians(fov_deg) / 2)
    return np.array([[f, 0.0, (width - 1) / 2], [0.0, f, (height - 1) / 2], [0.0, 0.0, 1.0]])


def splat(points: np.ndarray, colors: np.ndarray, T_world_cam: np.ndarray, K: np.ndarray, size: tuple[int, int],
          point_size_m: float | np.ndarray = POINT_SIZE_M, max_px: int = MAX_SPLAT_PX,
          background: tuple[int, int, int] = BACKGROUND) -> np.ndarray:
    """HxWx3 uint8 image of the points as squares of about point_size_m (one value or one per point), nearest point
    winning each pixel."""
    width, height = size
    R, t = T_world_cam[:3, :3], T_world_cam[:3, 3]
    p = (points.astype(np.float64) - t) @ R
    front = p[:, 2] > NEAR_M
    p, index = p[front], np.flatnonzero(front)
    size_m = np.broadcast_to(np.asarray(point_size_m, dtype=np.float64), (len(points),))[front]
    z = p[:, 2]
    u = K[0, 0] * p[:, 0] / z + K[0, 2]
    v = K[1, 1] * p[:, 1] / z + K[1, 2]
    side = np.clip(np.ceil(K[0, 0] * size_m / z), 1, max_px).astype(np.int64)
    pixel_parts, z_parts, index_parts = [], [], []
    for s in np.unique(side):
        pick = side == s
        offsets = np.arange(s) - (s - 1) // 2
        du, dv = (a.ravel() for a in np.meshgrid(offsets, offsets))
        uu = (np.rint(u[pick]).astype(np.int64)[:, None] + du[None]).ravel()
        vv = (np.rint(v[pick]).astype(np.int64)[:, None] + dv[None]).ravel()
        inside = (uu >= 0) & (uu < width) & (vv >= 0) & (vv < height)
        pixel_parts.append((vv * width + uu)[inside])
        z_parts.append(np.repeat(z[pick], s * s)[inside])
        index_parts.append(np.repeat(index[pick], s * s)[inside])
    image = np.empty((height * width, 3), dtype=np.uint8)
    image[:] = background
    if pixel_parts:
        pixel, depth, source = (np.concatenate(a) for a in (pixel_parts, z_parts, index_parts))
        order = np.lexsort((depth, pixel))
        pixel, source = pixel[order], source[order]
        first = np.ones(pixel.size, dtype=bool)
        first[1:] = pixel[1:] != pixel[:-1]
        image[pixel[first]] = colors[source[first]]
    return image.reshape(height, width, 3)


def orbit_poses(orbit: dict, frames: int, arc_deg: float, elevation_deg: float, distance_scale: float,
                fov_deg: float) -> list[np.ndarray]:
    centre, up = np.asarray(orbit["centre"], float), np.asarray(orbit["up"], float)
    start = np.asarray(orbit["start_direction"], float)
    start = start - (start @ up) * up
    if np.linalg.norm(start) < 1e-6:
        start = np.cross(up, [1.0, 0.0, 0.0]) if abs(up[0]) < 0.9 else np.cross(up, [0.0, 1.0, 0.0])
    e1 = start / np.linalg.norm(start)
    e2 = np.cross(up, e1)
    distance = distance_scale * orbit["radius_m"] / math.sin(math.radians(fov_deg) / 2)
    elevation = math.radians(elevation_deg)
    poses = []
    for k in range(frames):
        theta = math.radians(arc_deg) * k / frames
        direction = math.cos(theta) * e1 + math.sin(theta) * e2
        eye = centre + distance * (math.cos(elevation) * direction + math.sin(elevation) * up)
        poses.append(look_at(eye, centre, up))
    return poses


def flythrough_poses(path: list[np.ndarray], fps: float, extra_s: float = FLY_EXTRA_S,
                     lateral_m: float = LATERAL_OFFSET_M) -> tuple[list[np.ndarray], int]:
    """(the recorded poses followed by extra_s seconds beyond the last one, the count of recorded poses). Beyond the
    path the camera keeps its last orientation and velocity and slides sideways up to lateral_m."""
    last = path[-1]
    back = max(0, len(path) - 1 - round(fps))
    velocity = (last[:3, 3] - path[back][:3, 3]) / max(1, len(path) - 1 - back)
    extra = max(1, round(extra_s * fps))
    poses = list(path)
    for k in range(1, extra + 1):
        T = last.copy()
        T[:3, 3] = last[:3, 3] + velocity * k + last[:3, 0] * lateral_m * k / extra
        poses.append(T)
    return poses, len(path)


def draw_text(image: np.ndarray, items: list[tuple[tuple[int, int], str, int, tuple[int, int, int]]]) -> np.ndarray:
    canvas = Image.fromarray(image)
    draw = ImageDraw.Draw(canvas)
    for xy, text, size, colour in items:
        draw.text(xy, text, font=font(size), fill=colour)
    return np.asarray(canvas)


def ghost_text(record: dict, source: str) -> str:
    g = record["ghosting"][source]
    if g["std_m"] is None:
        return "ghosting n/a"
    return f"ghosting std {100 * g['std_m']:.1f} cm, covered {100 * g['covered_fraction']:.0f} %"


def header(record: dict, title: str) -> list[tuple[tuple[int, int], str, int, tuple[int, int, int]]]:
    return [((16, 12), f"{record['sequence']}   {title}", 30, (255, 255, 255)),
            ((16, 58), f"{record['label']}   (scale {record['scale']}, poses {record['poses']}, ORACLE GT poses)", 24,
             (255, 209, 102))]


def draw_centred(image: np.ndarray, items: list[tuple[tuple[int, int], str, int, tuple[int, int, int]]]) -> np.ndarray:
    canvas = Image.fromarray(image)
    draw = ImageDraw.Draw(canvas)
    for xy, text, size, colour in items:
        draw.text(xy, text, font=font(size), fill=colour, anchor="mm")
    return np.asarray(canvas)


def side_by_side_frame(record: dict, clouds: dict[str, tuple[np.ndarray, np.ndarray]], T: np.ndarray, K: np.ndarray,
                       panel: tuple[int, int], point_size: float) -> np.ndarray:
    top = SIDE_HEADER_PX + SIDE_LABEL_PX
    frame = np.empty((SIZE[1], SIZE[0], 3), dtype=np.uint8)
    frame[:] = BACKGROUND
    centred = []
    for p, (source, label) in enumerate(PANELS):
        x0 = p * panel[0]
        frame[top:top + panel[1], x0:x0 + panel[0]] = splat(*clouds[source], T, K, panel, point_size)
        centred += [((x0 + panel[0] // 2, SIDE_HEADER_PX + SIDE_LABEL_PX // 2), label, 64, SIDE_COLOURS[source]),
                    ((x0 + panel[0] // 2, SIZE[1] - SIDE_FOOTER_PX // 2), ghost_text(record, source), 26,
                     (220, 220, 220))]
        if p:
            frame[SIDE_HEADER_PX:, x0 - 2:x0 + 2] = DIVIDER
    frame = draw_centred(frame, centred)
    name = record["sequence"].replace("rgbd_dataset_freiburg", "fr")
    return draw_text(frame, [((16, 18), f"{name}   point maps of raw, fused and GT depth   oracle GT poses, an upper "
                                         f"bound, joint solved scale", 30, (255, 255, 255))])


def render_orbit(record: dict, clouds: dict[str, tuple[np.ndarray, np.ndarray]], args: argparse.Namespace) -> None:
    top, bottom = (SIDE_HEADER_PX + SIDE_LABEL_PX, SIDE_FOOTER_PX) if args.side_by_side else (HEADER_PX, 0)
    panel = (SIZE[0] // len(PANELS), SIZE[1] - top - bottom)
    K = intrinsics(*panel, args.fov)
    frames = round(args.seconds * args.fps)
    poses = orbit_poses(record["orbit"], frames, args.arc, args.elevation, args.distance_scale, args.fov)
    stem = "orbit_side_by_side" if args.side_by_side else "orbit"
    video = VideoWriter(args.out_dir / f"{stem}.mp4", SIZE, args.fps)
    gif_step = max(1, round(args.fps / GIF_FPS))
    gif_frames = []
    written = 0
    title = f"{args.arc:g} degree orbit of the point maps, fused frames at stride {record['settings']['stride']}"
    for T, repeat in zip(poses, hold_repeats(len(poses), args.fps, args.hold_seconds)):
        if args.side_by_side:
            frame = side_by_side_frame(record, clouds, T, K, panel, args.point_size)
        else:
            frame = np.empty((SIZE[1], SIZE[0], 3), dtype=np.uint8)
            frame[:] = BACKGROUND
            labels = header(record, title)
            for p, (source, label) in enumerate(PANELS):
                x0 = p * panel[0]
                frame[HEADER_PX:, x0:x0 + panel[0]] = splat(*clouds[source], T, K, panel, args.point_size)
                labels += [((x0 + 16, HEADER_PX + 10), label, 34, (255, 255, 255)),
                           ((x0 + 16, SIZE[1] - 40), ghost_text(record, source), 22, (220, 220, 220))]
            frame = draw_text(frame, labels)
        half = None
        for _ in range(repeat):
            video.write(frame)
            if written % gif_step == 0:
                half = half_size(frame) if half is None else half
                gif_frames.append(half)
            written += 1
    video_path = video.close()
    gif_path = write_gif(args.out_dir / f"{stem}.gif", gif_frames, args.fps / gif_step)
    print(f"wrote {video_path} and {gif_path}")


def render_flythrough(record: dict, clouds: dict[str, tuple[np.ndarray, np.ndarray]],
                      args: argparse.Namespace) -> None:
    camera = record["camera"]
    scale = SIZE[0] / camera["image_size"][0]
    K = np.array(camera["K"], dtype=np.float64)
    K = np.array([[K[0, 0] * scale, 0.0, (SIZE[0] - 1) / 2], [0.0, K[1, 1] * scale, (SIZE[1] - 1) / 2],
                  [0.0, 0.0, 1.0]])
    path = [np.array(T, dtype=np.float64) for T in camera["path_T_world_cam"]]
    poses, recorded = flythrough_poses(path, args.fps, FLY_EXTRA_S, args.lateral)
    video = VideoWriter(args.out_dir / "flythrough.mp4", SIZE, args.fps)
    for k, (T, repeat) in enumerate(zip(poses, hold_repeats(len(poses), args.fps, args.hold_seconds))):
        frame = splat(*clouds["fused"], T, K, SIZE, args.point_size)
        where = ("recorded camera path" if k < recorded
                 else f"beyond the recorded path, sideways offset up to {args.lateral:.2f} m")
        frame = draw_text(frame, [*header(record, "fused map flythrough"),
                                  ((16, SIZE[1] - 50), f"fused   {where}   {ghost_text(record, 'fused')}", 28,
                                   (255, 255, 255))])
        for _ in range(repeat):
            video.write(frame)
    print(f"wrote {video.close()}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("map_dir", type=Path, help=f"output folder of export_fused_map.py holding {MAP_JSON}")
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument("--seconds", type=float, default=12.0, help="orbit length before the holds")
    parser.add_argument("--hold-seconds", type=float, default=0.0, help="extra time on the first and last frame")
    parser.add_argument("--side-by-side", action="store_true",
                        help="raw, fused and GT under large labels with a one line header")
    parser.add_argument("--arc", type=float, default=360.0, help="orbit arc in degrees")
    parser.add_argument("--elevation", type=float, default=35.0, help="orbit elevation in degrees")
    parser.add_argument("--distance-scale", type=float, default=1.0)
    parser.add_argument("--fov", type=float, default=FOV_DEG, help="horizontal field of view of each panel")
    parser.add_argument("--point-size", type=float, default=POINT_SIZE_M, help="splat edge in metres")
    parser.add_argument("--flythrough", action="store_true", help="render the fused map along the recorded path")
    parser.add_argument("--lateral", type=float, default=LATERAL_OFFSET_M)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    record = json.loads((args.map_dir / MAP_JSON).read_text())
    sources = ["fused"] if args.flythrough else [s for s, _ in PANELS]
    clouds = {s: read_ply(args.map_dir / record["files"][s]) for s in sources}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.flythrough:
        render_flythrough(record, clouds, args)
    else:
        render_orbit(record, clouds, args)


if __name__ == "__main__":
    main()
