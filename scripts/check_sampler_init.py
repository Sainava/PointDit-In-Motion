"""Check PointDiT sampler initialisation: zeros is deterministic, seeded noise is reproducible."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pim.data.tum import load_rgb  # noqa: E402
from pim.model.runner import PointDiTRunner  # noqa: E402

TOLERANCE = 1e-4


def load_reference(folder: Path) -> tuple[np.ndarray, np.ndarray]:
    arrays = []
    for name in ("points.npy", "valid_mask.npy"):
        path = folder / name
        if not path.is_file():
            raise FileNotFoundError(f"reference file {path} not found")
        arrays.append(np.load(path))
    points, valid_mask = arrays
    if points.shape[:2] != valid_mask.shape or points.shape[2:] != (3,):
        raise ValueError(f"reference shapes do not match: points {points.shape}, mask {valid_mask.shape}")
    return points.astype(np.float32), valid_mask.astype(bool)


def max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a - b).max())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True, help="RGB frame")
    parser.add_argument("--reference", type=Path, required=True,
                        help="folder with points.npy and valid_mask.npy from the smoke test")
    args = parser.parse_args()

    ref_points, ref_mask = load_reference(args.reference)
    image = load_rgb(args.image)
    if image.shape[:2] != ref_mask.shape:
        raise ValueError(f"image {image.shape[:2]} and reference {ref_mask.shape} sizes differ")

    runner = PointDiTRunner()
    default = runner.predict(image)
    both_valid = default.valid_mask & ref_mask
    if not both_valid.any():
        raise ValueError("default prediction and reference share no valid pixels")
    default_diff = np.abs(default.points[both_valid] - ref_points[both_valid])
    zeros_a = runner.predict(image, init="zeros").points
    zeros_b = runner.predict(image, init="zeros").points
    noise_0a = runner.predict(image, init="noise", seed=0).points
    noise_1 = runner.predict(image, init="noise", seed=1).points
    noise_0b = runner.predict(image, init="noise", seed=0).points

    zeros_repeat = max_abs_diff(zeros_a, zeros_b)
    seed_change = max_abs_diff(noise_0a, noise_1)
    seed_repeat = max_abs_diff(noise_0a, noise_0b)
    failures = []
    if zeros_repeat > TOLERANCE:
        failures.append(f"check 2: zeros repeat {zeros_repeat:.3e} > {TOLERANCE:g}")
    if seed_repeat > TOLERANCE:
        failures.append(f"check 4: noise seed 0 repeat {seed_repeat:.3e} > {TOLERANCE:g}")
    if not seed_change > zeros_repeat:
        failures.append(f"check 3: seed 0 vs 1 {seed_change:.3e} not larger than check 2 {zeros_repeat:.3e}")

    rows = [
        ("1 default vs reference", f"max {default_diff.max():.3e}  mean {default_diff.mean():.3e}  "
                                   f"masks identical: {np.array_equal(default.valid_mask, ref_mask)}"),
        ("2 zeros, run twice", f"max {zeros_repeat:.3e}"),
        ("3 noise, seed 0 vs 1", f"max {seed_change:.3e}"),
        ("4 noise, seed 0 twice", f"max {seed_repeat:.3e}"),
    ]
    width = max(len(name) for name, _ in rows)
    print(f"image: {args.image}  (model ran at {default.inference_size})")
    for name, value in rows:
        print(f"{name:<{width}}  {value}")
    for failure in failures:
        print(f"FAIL {failure}", file=sys.stderr)
    print("FAIL" if failures else "PASS")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
