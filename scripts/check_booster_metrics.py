"""Cross-check evaluate_frame against upstream's evaluation on the 10 Booster scenes of the original Kaggle run."""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pim.metrics.geometry import evaluate_frame  # noqa: E402
from pim.model.runner import PointDiTRunner  # noqa: E402
from pim.paths import configure_fork  # noqa: E402

configure_fork()
from dataloader.eval_depth import EvalDepthDataset  # noqa: E402
from metrics.compute_metrics import compute_metrics  # noqa: E402

# The SUBSET of the fork's PointDiT_Kaggle.ipynb, and what its evaluation cell printed.
SUBSET = ["Motorcycle", "Microwave", "Bedroom", "Mirror", "Fridge", "DogHouse", "Shower", "Door", "TV", "Bathroom"]
ORIGINAL_RUN = {"rel_p": 3.001, "delta1_p": 99.939, "rel_d": 1.995, "delta1_d": 99.937}
# scripts/eval_l_512.sh: --eval_depth_resize_height 512, center crop to --img_size 512.
RESIZE_HEIGHT = 512
METRICS = list(ORIGINAL_RUN)
SCENE_TOLERANCE = 1e-4
ORIGINAL_TOLERANCE = 0.05


def build_subset(booster_dir: Path, root: Path) -> None:
    for name in SUBSET:
        scene = booster_dir / name
        if not (scene / "depth.png").is_file():
            raise FileNotFoundError(f"{scene}/depth.png not found; is {booster_dir} the extracted Booster folder?")
        shutil.copytree(scene, root / "Booster" / name)


def predict_like_engine(runner: PointDiTRunner, image_01: torch.Tensor) -> torch.Tensor:
    # engine.evaluate_img2point: a 512x512 eval input is used as is (no token resize).
    image = image_01[None].to(runner.device).to(torch.float32) * 2.0 - 1.0
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        points = runner.model.generate(image)
    return points.float()


def upstream_metrics(points: torch.Tensor, gt_points: torch.Tensor, gt_valid: torch.Tensor) -> dict[str, float]:
    pred = {"points_affine_invariant": points.permute(0, 2, 3, 1), "depth_affine_invariant": points[:, 2]}
    gt = {"points": gt_points.permute(0, 2, 3, 1), "depth": gt_points[:, 2], "depth_mask": gt_valid}
    with torch.no_grad():
        metrics, _ = compute_metrics(pred, gt, vis=True)
    point, depth = metrics["points_affine_invariant"], metrics["depth_affine_invariant"]
    return {
        "rel_p": point["rel"] * 100.0,
        "delta1_p": point["delta1"] * 100.0,
        "rel_d": depth["rel"] * 100.0,
        "delta1_d": depth["delta1"] * 100.0,
    }


def wrapper_metrics(points: torch.Tensor, gt_points: torch.Tensor, gt_valid: torch.Tensor) -> dict[str, float]:
    pred_np = points[0].permute(1, 2, 0).cpu().numpy()
    gt_np = gt_points[0].permute(1, 2, 0).cpu().numpy()
    valid_np = gt_valid[0].cpu().numpy()
    result = evaluate_frame(pred_np, gt_np, valid_np, device=str(points.device))
    return {name: getattr(result, name) for name in METRICS}


def columns(values: dict[str, float], spec: str = ">10.3f") -> str:
    return "".join(f"{values[name]:{spec}}" for name in METRICS)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--booster-dir", type=Path, required=True, help="extracted Booster folder (one dir per scene)")
    args = parser.parse_args()

    runner = PointDiTRunner()
    upstream_rows: dict[str, dict[str, float]] = {}
    wrapper_rows: dict[str, dict[str, float]] = {}
    with tempfile.TemporaryDirectory() as root:
        build_subset(args.booster_dir, Path(root))
        dataset = EvalDepthDataset(root, "Booster", crop_size=runner.args.img_size, resize_height=RESIZE_HEIGHT)
        for sample in dataset:
            gt_points = sample["pointcloud"][None].to(runner.device)
            gt_valid = sample["valid_mask"][None].to(runner.device)
            if gt_valid.sum().item() < 100:
                raise ValueError(f"{sample['sample_id']}: fewer than 100 valid GT pixels; upstream would skip it")
            points = predict_like_engine(runner, sample["image"])
            upstream_rows[sample["sample_id"]] = upstream_metrics(points, gt_points, gt_valid)
            wrapper_rows[sample["sample_id"]] = wrapper_metrics(points, gt_points, gt_valid)

    if sorted(upstream_rows) != sorted(SUBSET):
        raise RuntimeError(f"evaluated {sorted(upstream_rows)}, expected {sorted(SUBSET)}")

    upstream_mean = {m: float(np.mean([row[m] for row in upstream_rows.values()])) for m in METRICS}
    wrapper_mean = {m: float(np.mean([row[m] for row in wrapper_rows.values()])) for m in METRICS}
    max_diff = {m: max(abs(upstream_rows[s][m] - wrapper_rows[s][m]) for s in upstream_rows) for m in METRICS}

    header = "".join(f"{name:>10}" for name in METRICS)
    print(f"{'':<12}{header}  |{header}")
    print(f"{'':<12}{'upstream':>40}  |{'wrapper':>40}")
    for scene in upstream_rows:
        print(f"{scene:<12}{columns(upstream_rows[scene])}  |{columns(wrapper_rows[scene])}")
    print(f"{'mean':<12}{columns(upstream_mean)}  |{columns(wrapper_mean)}")
    print(f"{'original':<12}{columns(ORIGINAL_RUN)}")
    print(f"{'max |diff|':<12}{columns(max_diff, '>10.1e')}")

    failures = [f"{m}: upstream vs wrapper differ by {max_diff[m]:.2e} > {SCENE_TOLERANCE:g}"
                for m in METRICS if max_diff[m] > SCENE_TOLERANCE]
    for label, mean in (("upstream", upstream_mean), ("wrapper", wrapper_mean)):
        failures += [f"{label} mean {m} {mean[m]:.3f} vs original {ORIGINAL_RUN[m]:.3f} (> {ORIGINAL_TOLERANCE})"
                     for m in METRICS if abs(mean[m] - ORIGINAL_RUN[m]) > ORIGINAL_TOLERANCE]
    for failure in failures:
        print(f"FAIL {failure}", file=sys.stderr)
    print("FAIL" if failures else "PASS")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
