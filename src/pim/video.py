"""Frame sinks for the render scripts. H.264 mp4 through an ffmpeg pipe when ffmpeg is on PATH, otherwise PNG frames
and the ffmpeg command that would encode them, plus a GIF writer and colour mapping on fixed ranges."""

from __future__ import annotations

import functools
import shlex
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

INVALID_RGB = (48, 48, 48)
FRAME_PATTERN = "%05d.png"


def colorize(values: np.ndarray, vmin: float, vmax: float, cmap_name: str,
             invalid_rgb: tuple[int, int, int] = INVALID_RGB) -> np.ndarray:
    """HxWx3 uint8 colours of values on the fixed range [vmin, vmax], clipped at both ends, invalid_rgb where not
    finite. The mapping depends on the range only, never on the frame."""
    from matplotlib import colormaps

    if not vmax > vmin:
        raise ValueError(f"empty colour range [{vmin}, {vmax}]")
    lut = (colormaps[cmap_name](np.linspace(0.0, 1.0, 256))[:, :3] * 255).round().astype(np.uint8)
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    scaled = np.clip((np.where(finite, values, vmin) - vmin) / (vmax - vmin), 0.0, 1.0)
    rgb = lut[np.rint(scaled * 255).astype(np.int64)]
    rgb[~finite] = invalid_rgb
    return rgb


def ffmpeg_command(size: tuple[int, int], fps: float, out: Path, source: str = "-") -> list[str]:
    width, height = size
    inputs = (["-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", f"{fps:g}", "-i", "-"]
              if source == "-" else ["-framerate", f"{fps:g}", "-i", source])
    return ["ffmpeg", "-y", "-loglevel", "error", *inputs, "-c:v", "libx264", "-preset", "medium", "-crf", "20",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)]


class VideoWriter:
    """Writes HxWx3 uint8 frames of one size. With ffmpeg they stream into out (an .mp4), otherwise they go to
    out.with_suffix('') / 00000.png and so on, and close() prints the encode command."""

    def __init__(self, out: Path, size: tuple[int, int], fps: float, use_ffmpeg: bool | None = None) -> None:
        self.out, self.size, self.fps = Path(out), size, fps
        self.out.parent.mkdir(parents=True, exist_ok=True)
        self.count = 0
        self.ffmpeg = shutil.which("ffmpeg") is not None if use_ffmpeg is None else use_ffmpeg
        self.frame_dir = self.out.with_suffix("")
        self._proc = None
        if self.ffmpeg:
            self._proc = subprocess.Popen(ffmpeg_command(size, fps, self.out), stdin=subprocess.PIPE)
        else:
            self.frame_dir.mkdir(parents=True, exist_ok=True)

    def write(self, frame: np.ndarray) -> None:
        if frame.shape != (self.size[1], self.size[0], 3) or frame.dtype != np.uint8:
            raise ValueError(f"expected a {self.size[1]}x{self.size[0]}x3 uint8 frame, got {frame.shape} {frame.dtype}")
        if self._proc is not None:
            self._proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        else:
            Image.fromarray(frame).save(self.frame_dir / (FRAME_PATTERN % self.count))
        self.count += 1

    def close(self) -> Path:
        """The mp4, or the PNG folder without ffmpeg."""
        if self._proc is None:
            command = ffmpeg_command(self.size, self.fps, self.out, str(self.frame_dir / FRAME_PATTERN))
            print(f"ffmpeg not found; wrote {self.count} PNG frames to {self.frame_dir}. Encode them with\n  "
                  f"{shlex.join(command)}")
            return self.frame_dir
        self._proc.stdin.close()
        if self._proc.wait() != 0:
            raise RuntimeError(f"ffmpeg exited with code {self._proc.returncode} writing {self.out}")
        return self.out


def write_gif(path: Path, frames: list[np.ndarray], fps: float) -> Path:
    if not frames:
        raise ValueError("no frames for the gif")
    images = [Image.fromarray(f).quantize(colors=256, method=Image.Quantize.MEDIANCUT) for f in frames]
    path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(path, save_all=True, append_images=images[1:], duration=round(1000 / fps), loop=0,
                   optimize=False, disposal=1)
    return path


def half_size(frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    return np.asarray(Image.fromarray(frame).resize((width // 2, height // 2), Image.Resampling.BILINEAR))


@functools.lru_cache(maxsize=None)
def font(size: int):
    from matplotlib import font_manager
    from PIL import ImageFont

    return ImageFont.truetype(font_manager.findfont("DejaVu Sans"), size)
