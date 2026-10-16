"""Check GT reprojection between two frames visually and print the GT floor, M1 and M2 for a pair, or sweep gaps."""

from __future__ import annotations

import argparse
import resource
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pim.consistency.metrics import (  # noqa: E402
    FrameInputs, PairConsistency, PixelSetMetrics, frame_inputs, pair_consistency,
)
from pim.consistency.warp import depth_pose, pairs_by_gap, relative_pose, warp_image  # noqa: E402
from pim.data.tum import Frame, TUMSequence, apply_window, load_sequence, read_groundtruth  # noqa: E402
from pim.evaluation import load_prediction, prediction_path  # noqa: E402

SWEEP_GAPS_MS = (33.0, 100.0, 333.0)


class InputCache:
    def __init__(self, sequence: TUMSequence, predictions: Path | None, gt_floor_poses: str) -> None:
        self.sequence = sequence
        self.predictions = predictions
        self.gt_floor_poses = gt_floor_poses
        self._gt_trajectory = read_groundtruth(sequence.root) if gt_floor_poses == "depth" else None
        self._inputs: dict[int, FrameInputs] = {}

    def __call__(self, frame: Frame) -> FrameInputs:
        if frame.index not in self._inputs:
            depth, valid = frame.load_depth()
            points = None
            if self.predictions is not None:
                path = prediction_path(self.predictions, self.sequence.name, frame.index)
                points = load_prediction(path, frame, self.sequence.root).points
            self._inputs[frame.index] = frame_inputs(depth, valid, self.sequence.K, points)
        return self._inputs[frame.index]

    def pair(self, frame_i: Frame, frame_j: Frame) -> PairConsistency:
        T_j_from_i = relative_pose(frame_i.T_world_cam, frame_j.T_world_cam)
        T_gt_j_from_i = None
        if self._gt_trajectory is not None:
            T_gt_j_from_i = relative_pose(depth_pose(frame_i, *self._gt_trajectory),
                                          depth_pose(frame_j, *self._gt_trajectory))
        return pair_consistency(self(frame_i), self(frame_j), self.sequence.K, T_j_from_i, T_gt_j_from_i)


def find_pair(frames: list[Frame], index: int, gap_ms: float) -> tuple[Frame, Frame]:
    matches = [(i, j) for i, j in pairs_by_gap(frames, gap_ms) if i.index == index]
    if not matches:
        raise SystemExit(f"frame {index} has no partner {gap_ms:g} ms later within tolerance")
    return matches[0]


def save_warp_figure(sequence: TUMSequence, frame_i: Frame, frame_j: Frame, gap_ms: float, out_dir: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    depth_i, valid_i = frame_i.load_depth()
    rgb_i, rgb_j = frame_i.load_rgb(), frame_j.load_rgb()
    warped, hit = warp_image(rgb_i, depth_i, valid_i, sequence.K, relative_pose(frame_i.T_world_cam, frame_j.T_world_cam))
    blend = rgb_j.copy()
    blend[hit] = (0.5 * rgb_j[hit].astype(np.float32) + 0.5 * warped[hit].astype(np.float32)).astype(np.uint8)
    difference = np.full(hit.shape, np.nan)
    difference[hit] = np.abs(rgb_j[hit].astype(np.float32) - warped[hit].astype(np.float32)).mean(axis=1)

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.4))
    panels = (
        (rgb_j, f"RGB j (frame {frame_j.index})"),
        (warped, f"RGB i (frame {frame_i.index}) warped into j"),
        (blend, "50/50 blend"),
        (difference, "|RGB j − warped|, hit pixels"),
    )
    for ax, (image, title) in zip(axes, panels):
        shown = ax.imshow(image, cmap="magma", vmin=0, vmax=80) if image.ndim == 2 else ax.imshow(image)
        ax.set_title(title)
        ax.axis("off")
    fig.colorbar(shown, ax=axes[-1], fraction=0.035)
    actual_gap = (frame_j.timestamp - frame_i.timestamp) * 1000.0
    fig.suptitle(f"{sequence.name}: frame {frame_i.index} → {frame_j.index}, gap {actual_gap:.1f} ms "
                 f"(hit {hit.mean():.1%})")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{sequence.name}_{frame_i.index:05d}_{frame_j.index:05d}_{gap_ms:g}ms.png"
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return path


def print_pair(label: str, result: PairConsistency) -> None:
    print(f"{label}: visible fraction {result.visible_fraction:.3f}"
          + ("  [TOO FEW VISIBLE PIXELS]" if result.too_few else ""))
    print(f"  {'pixels':<9} {'metric':<9} {'n':>7} {'mean':>7} {'median':>7} {'p90':>7} "
          f"{'near':>7} {'mid':>7} {'far':>7}   (x100, near/mid/far are means)")
    for name, pixels in (("visible", result.visible), ("interior", result.interior)):
        for metric, summary in _named_metrics(pixels):
            o = summary.overall
            print(f"  {name:<9} {metric:<9} {pixels.num_pixels:>7} {o.mean:>7.3f} {o.median:>7.3f} {o.p90:>7.3f} "
                  f"{summary.near.mean:>7.3f} {summary.mid.mean:>7.3f} {summary.far.mean:>7.3f}")


def _named_metrics(pixels: PixelSetMetrics):
    yield "gt_floor", pixels.gt_floor
    yield "m2_floor", pixels.m2_floor
    if pixels.m1 is not None:
        yield "m1", pixels.m1
        yield "m2", pixels.m2


def sweep(sequence: TUMSequence, frames: list[Frame], cache: InputCache) -> None:
    has_predictions = cache.predictions is not None
    print(f"{sequence.name}: {len(frames)} frames ({frames[0].index}–{frames[-1].index}); GT floor and visibility "
          f"from {cache.gt_floor_poses}-timestamp poses; median over pairs of each pair's median pixel error, x100")
    columns = ("gt_floor", "m1", "m2_floor", "m2") if has_predictions else ("gt_floor", "m2_floor")
    print(f"  {'pixels':<9} {'gap_ms':>6} {'pairs':>5} {'too_few':>7} {'frac':>7}"
          + "".join(f" {c:>8}" for c in columns))
    results = {gap: [cache.pair(i, j) for i, j in pairs_by_gap(frames, gap)] for gap in SWEEP_GAPS_MS}
    for name in ("visible", "interior"):
        for gap, pairs in results.items():
            sets = [getattr(r, name) for r in pairs]
            print(f"  {name:<9} {gap:>6g} {len(pairs):>5} {sum(s.too_few for s in sets):>7} "
                  f"{_median([s.fraction for s in sets]):>7.3f}"
                  + "".join(f" {_median([getattr(s, c).overall.median for s in sets]):>8.3f}" for c in columns))


def _median(values: list[float]) -> float:
    finite = [v for v in values if np.isfinite(v)]
    return float(np.median(finite)) if finite else float("nan")


def predicted_frames(frames: list[Frame], sequence: TUMSequence, predictions: Path) -> list[Frame]:
    available = [f for f in frames if prediction_path(predictions, sequence.name, f.index).is_file()]
    if not available:
        raise SystemExit(f"no predictions for {sequence.name} window frames under {predictions / sequence.name}")
    return available


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--sequences-config", type=Path, default=Path("configs/sequences.yaml"))
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--frame", type=int, help="frame index of frame i (required without --sweep)")
    parser.add_argument("--gap-ms", type=float, default=100.0)
    parser.add_argument("--predictions", type=Path, help="run directory with <sequence>/<frame:05d>.npz")
    parser.add_argument("--out", type=Path, default=Path("outputs/consistency"), help="figure directory")
    parser.add_argument("--gt-floor-poses", choices=("depth", "rgb"), default="depth",
                        help="poses for the GT floor and visibility, interpolated at depth or RGB timestamps")
    parser.add_argument("--sweep", action="store_true", help=f"table over gaps {SWEEP_GAPS_MS} ms instead of one pair")
    parser.add_argument("--threads", type=int, help="cap torch intra-op threads (pair with OMP_NUM_THREADS)")
    args = parser.parse_args()
    if args.threads is not None:
        import torch
        torch.set_num_threads(args.threads)
    try:
        run(args)
    finally:
        print_peak_memory()


def print_peak_memory() -> None:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is bytes on macOS and KiB on Linux.
    peak_mib = peak / 2**20 if sys.platform == "darwin" else peak / 2**10
    print(f"peak memory (ru_maxrss): {peak_mib:.0f} MiB")


def run(args: argparse.Namespace) -> None:
    sequence = load_sequence(args.data_root / args.sequence, args.sequences_config)
    window = apply_window(sequence)
    cache = InputCache(sequence, args.predictions, args.gt_floor_poses)
    if args.sweep:
        frames = window if args.predictions is None else predicted_frames(window, sequence, args.predictions)
        sweep(sequence, frames, cache)
        return

    if args.frame is None:
        raise SystemExit("--frame is required without --sweep")
    frames = sequence.frames if args.predictions is None else predicted_frames(sequence.frames, sequence, args.predictions)
    frame_i, frame_j = find_pair(frames, args.frame, args.gap_ms)
    print(f"figure: {save_warp_figure(sequence, frame_i, frame_j, args.gap_ms, args.out)}")
    gap = (frame_j.timestamp - frame_i.timestamp) * 1000.0
    print_pair(f"pair {frame_i.index} → {frame_j.index} ({gap:.1f} ms)", cache.pair(frame_i, frame_j))
    print_pair(f"self-pair {frame_i.index} → {frame_i.index}", cache.pair(frame_i, frame_i))


if __name__ == "__main__":
    main()
