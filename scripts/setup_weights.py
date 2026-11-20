"""Download the PointDiT checkpoint and the converted DINOv3 encoder for each model (default L, ViT-L/16).

Idempotent: each step is skipped when its output already exists. The gated DINOv3
weights need HF_TOKEN in the environment.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pim.paths import (  # noqa: E402
    CHECKPOINT_NAMES,
    DINOV3_REPO_DIR,
    FORK_KAGGLE_NOTEBOOK,
    WEIGHTS_DIR,
    checkpoint_path,
    dinov3_weights_path,
)

POINTDIT_HF_REPO = "haofeixu/pointdit"
DINOV3_HF_REPOS = {"L": "facebook/dinov3-vitl16-pretrain-lvd1689m", "B": "facebook/dinov3-vitb16-pretrain-lvd1689m"}
DINOV3_HUB_ENTRIES = {"L": "dinov3_vitl16", "B": "dinov3_vitb16"}
CONVERTER_NAME = "hf_dinov3_to_hub"
HF_DOWNLOAD_DIR = WEIGHTS_DIR / ".hf_download"


def download_checkpoint(model: str) -> None:
    from huggingface_hub import hf_hub_download

    checkpoint = checkpoint_path(model)
    if checkpoint.is_file():
        print(f"PointDiT checkpoint present: {checkpoint}")
        return
    path = hf_hub_download(POINTDIT_HF_REPO, CHECKPOINT_NAMES[model], local_dir=WEIGHTS_DIR)
    print(f"Downloaded {path}")


def load_fork_converter() -> Callable:
    """The converter defined in the fork's Kaggle notebook, which ships no importable module."""
    import torch

    cells = json.loads(FORK_KAGGLE_NOTEBOOK.read_text())["cells"]
    for cell in cells:
        source = "".join(cell["source"])
        if cell["cell_type"] != "code" or f"def {CONVERTER_NAME}(" not in source:
            continue
        for node in ast.parse(source).body:
            if isinstance(node, ast.FunctionDef) and node.name == CONVERTER_NAME:
                namespace = {"re": re, "torch": torch}
                exec(ast.get_source_segment(source, node), namespace)
                return namespace[CONVERTER_NAME]
    raise RuntimeError(f"{CONVERTER_NAME} not found in {FORK_KAGGLE_NOTEBOOK}")


def require_hf_token(repo: str) -> str:
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise RuntimeError(
            f"HF_TOKEN is not set; it must belong to an account granted access to {repo}"
        )
    return token


def require_dinov3_repo() -> None:
    if not (DINOV3_REPO_DIR / "hubconf.py").is_file():
        raise FileNotFoundError(
            f"DINOv3 code not found at {DINOV3_REPO_DIR}. Clone it with: "
            f"git clone https://github.com/facebookresearch/dinov3.git {DINOV3_REPO_DIR}"
        )


def build_dinov3_template(model: str):
    import torch

    return torch.hub.load(str(DINOV3_REPO_DIR), DINOV3_HUB_ENTRIES[model], source="local", pretrained=False)


def convert_dinov3(model: str) -> None:
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    weights = dinov3_weights_path(model)
    if weights.is_file():
        print(f"DINOv3 weights present: {weights}")
        return
    repo = DINOV3_HF_REPOS[model]
    token = require_hf_token(repo)
    converter = load_fork_converter()

    safetensors_path = hf_hub_download(repo, "model.safetensors", local_dir=HF_DOWNLOAD_DIR, token=token)
    template = build_dinov3_template(model)
    converted = converter(load_file(safetensors_path), template.state_dict())
    del template

    weights.parent.mkdir(parents=True, exist_ok=True)
    partial = weights.with_suffix(".pth.partial")
    torch.save(converted, partial)
    partial.replace(weights)
    shutil.rmtree(HF_DOWNLOAD_DIR, ignore_errors=True)
    print(f"Converted DINOv3 weights: {weights}")


def verify_dinov3(model: str) -> None:
    """Strict, weights_only load into the DINOv3 model, as the fork's notebook does."""
    import torch

    encoder = build_dinov3_template(model)
    encoder.load_state_dict(torch.load(dinov3_weights_path(model), map_location="cpu", weights_only=True),
                            strict=True)
    print(f"DINOv3 {DINOV3_HUB_ENTRIES[model]} weights verified")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=sorted(CHECKPOINT_NAMES), default=["L"])
    args = parser.parse_args()

    require_dinov3_repo()
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    for model in dict.fromkeys(args.models):
        download_checkpoint(model)
        convert_dinov3(model)
        verify_dinov3(model)


if __name__ == "__main__":
    main()
