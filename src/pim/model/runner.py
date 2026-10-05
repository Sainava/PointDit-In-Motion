"""Single-image PointDiT-L/16 512 inference, matching the fork's wild-image path."""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F

from pim.paths import DINOV3_WEIGHTS, POINTDIT_CHECKPOINT, configure_fork

# Flags of the fork's scripts/demo_l_512.sh; everything else keeps main.py's defaults.
DEMO_L_512_ARGS = [
    "--model", "PointDiT-L/16",
    "--feature_embedding_type", "dinov3_vitl16",
    "--proj_dropout", "0.0",
    "--evaluate_gen",
    "--eval_wild_images",
    "--img_size", "512",
    "--num_sampling_steps", "3",
]

SamplerInit = Literal["noise", "zeros"]


@dataclass(frozen=True)
class PredictionResult:
    points: np.ndarray
    """(H, W, 3) float32 point map at the input resolution, in the model's normalised space."""
    valid_mask: np.ndarray
    """(H, W) bool, False where upstream's sky removal flags the point."""
    inference_size: tuple[int, int]
    """(h, w) the model actually ran at."""
    latency_ms: float
    """Whole predict() call: upload, resize, sampling, resize back, download."""
    model_latency_ms: float
    """Sampling only, the window upstream's evaluate_img2point times."""
    peak_memory_bytes: int

    @property
    def peak_memory_gb(self) -> float:
        return self.peak_memory_bytes / 1024**3


class PointDiTRunner:
    def __init__(
        self,
        device: str | torch.device = "cuda",
        checkpoint: Path = POINTDIT_CHECKPOINT,
        dinov3_weights: Path = DINOV3_WEIGHTS,
    ) -> None:
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError(f"PointDiTRunner needs a CUDA device, got {self.device}")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available; enable a GPU accelerator")
        for path in (checkpoint, dinov3_weights):
            if not Path(path).is_file():
                raise FileNotFoundError(f"{path} not found; run scripts/setup_weights.py first")

        configure_fork()
        from main import get_args_parser

        self.args = get_args_parser().parse_args([*DEMO_L_512_ARGS, "--pretrained", str(checkpoint)])
        # main() sets this globally before evaluation.
        torch.backends.cudnn.benchmark = True
        self.model = _load_model(self.args, Path(checkpoint), Path(dinov3_weights), self.device)

    @torch.no_grad()
    def predict(
        self, image: np.ndarray, init: SamplerInit = "zeros", seed: int | None = None
    ) -> PredictionResult:
        _check_image(image)
        # Denoiser.generate starts from generate_noise_scale * randn; upstream's 0.0 is zeros.
        self.model.args.generate_noise_scale = _noise_scale(init, self.args.noise_scale)
        height, width = image.shape[:2]

        torch.cuda.reset_peak_memory_stats(self.device)
        torch.cuda.synchronize(self.device)
        start = time.perf_counter()

        pixels = torch.from_numpy(image).to(self.device).permute(2, 0, 1)[None].float() / 255.0
        pixels = pixels * 2.0 - 1.0
        inference_size = self._inference_size(height, width)
        if inference_size != (height, width):
            pixels = F.interpolate(pixels, size=inference_size, mode="bilinear", align_corners=False)

        if seed is not None:
            torch.manual_seed(seed)
        torch.cuda.synchronize(self.device)
        model_start = time.perf_counter()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            points = self.model.generate(pixels)
        torch.cuda.synchronize(self.device)
        model_latency_ms = (time.perf_counter() - model_start) * 1e3

        points = points.float()
        if inference_size != (height, width):
            points = F.interpolate(points, size=(height, width), mode="nearest")
        valid = ~self._sky_mask(points)
        points_np = points[0].permute(1, 2, 0).cpu().numpy()
        valid_np = valid[0].cpu().numpy()

        torch.cuda.synchronize(self.device)
        latency_ms = (time.perf_counter() - start) * 1e3
        return PredictionResult(
            points=np.ascontiguousarray(points_np, dtype=np.float32),
            valid_mask=valid_np,
            inference_size=inference_size,
            latency_ms=latency_ms,
            model_latency_ms=model_latency_ms,
            peak_memory_bytes=torch.cuda.max_memory_allocated(self.device),
        )

    def _inference_size(self, height: int, width: int) -> tuple[int, int]:
        # Mirrors the is_wild_images branch of engine.evaluate_img2point.
        patch_size = getattr(self.model, "patch_size", 16)
        target_tokens = self.args.eval_wild_target_tokens or self.args.eval_num_tokens
        if target_tokens <= 0:
            target_tokens = (self.args.img_size // patch_size) ** 2
        resize_factor = ((target_tokens * patch_size**2) / (height * width)) ** 0.5
        new_h = max(patch_size, int(round(height * resize_factor / patch_size)) * patch_size)
        new_w = max(patch_size, int(round(width * resize_factor / patch_size)) * patch_size)
        return new_h, new_w

    def _sky_mask(self, points: torch.Tensor) -> torch.Tensor:
        # Mirrors upstream's remove_sky detection on the raw output (on by default).
        if not self.args.remove_sky:
            return torch.zeros_like(points[:, 0], dtype=torch.bool)
        metric = points[:, 2] if self.args.remove_sky_metric == "depth" else points.norm(dim=1)
        return metric > self.args.remove_sky_threshold


def _noise_scale(init: SamplerInit, training_noise_scale: float) -> float:
    if init == "zeros":
        return 0.0
    if init == "noise":
        return training_noise_scale
    raise ValueError(f"init must be 'noise' or 'zeros', got {init!r}")


def _check_image(image: np.ndarray) -> None:
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected an HxWx3 uint8 RGB image, got {image.dtype} {image.shape}")


def _load_model(args, checkpoint: Path, dinov3_weights: Path, device: torch.device):
    """The evaluate_gen + --pretrained loading sequence of the fork's main()."""
    from denoiser import Denoiser

    model = Denoiser(args).to(device)
    encoder_state = torch.load(dinov3_weights, map_location="cpu")
    model.net.y_embedder.load_state_dict(encoder_state, strict=True)
    del encoder_state

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    stripped = state.get("dinov3_stripped")
    if stripped is not None:
        prefix = stripped.get("prefix", "net.y_embedder.")
        incompatible = model.load_state_dict(state["model"], strict=False)
        missing = [k for k in incompatible.missing_keys if not k.startswith(prefix)]
        if missing or incompatible.unexpected_keys:
            raise RuntimeError(
                f"checkpoint mismatch: missing {missing}, unexpected {incompatible.unexpected_keys}"
            )
    else:
        model.load_state_dict(state["model"], strict=True)

    if "model_ema1" in state:
        param_names = {name for name, _ in model.named_parameters()}
        ema = {k: v for k, v in state["model_ema1"].items() if k in param_names}
        model.load_state_dict(ema, strict=False)
    model.ema_params1 = None
    model.ema_params2 = None

    del state
    gc.collect()
    return model.eval()
