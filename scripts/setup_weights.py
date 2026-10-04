"""Download the PointDiT checkpoint and the converted DINOv3 ViT-L/16 encoder.

Idempotent: each step is skipped when its output already exists. The gated DINOv3
weights need HF_TOKEN in the environment.
"""

from __future__ import annotations

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
    DINOV3_REPO_DIR,
    DINOV3_WEIGHTS,
    FORK_KAGGLE_NOTEBOOK,
    POINTDIT_CHECKPOINT,
    POINTDIT_CHECKPOINT_NAME,
    WEIGHTS_DIR,
)

POINTDIT_HF_REPO = "haofeixu/pointdit"
DINOV3_HF_REPO = "facebook/dinov3-vitl16-pretrain-lvd1689m"
DINOV3_HUB_ENTRY = "dinov3_vitl16"
CONVERTER_NAME = "hf_dinov3_to_hub"
HF_DOWNLOAD_DIR = WEIGHTS_DIR / ".hf_download"


def download_checkpoint() -> None:
    from huggingface_hub import hf_hub_download

    if POINTDIT_CHECKPOINT.is_file():
        print(f"PointDiT checkpoint present: {POINTDIT_CHECKPOINT}")
        return
    path = hf_hub_download(POINTDIT_HF_REPO, POINTDIT_CHECKPOINT_NAME, local_dir=WEIGHTS_DIR)
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


def require_hf_token() -> str:
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise RuntimeError(
            f"HF_TOKEN is not set; it must belong to an account granted access to {DINOV3_HF_REPO}"
        )
    return token


def require_dinov3_repo() -> None:
    if not (DINOV3_REPO_DIR / "hubconf.py").is_file():
        raise FileNotFoundError(
            f"DINOv3 code not found at {DINOV3_REPO_DIR}. Clone it with: "
            f"git clone https://github.com/facebookresearch/dinov3.git {DINOV3_REPO_DIR}"
        )


def build_dinov3_template():
    import torch

    return torch.hub.load(str(DINOV3_REPO_DIR), DINOV3_HUB_ENTRY, source="local", pretrained=False)


def convert_dinov3() -> None:
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    if DINOV3_WEIGHTS.is_file():
        print(f"DINOv3 weights present: {DINOV3_WEIGHTS}")
        return
    token = require_hf_token()
    converter = load_fork_converter()

    safetensors_path = hf_hub_download(
        DINOV3_HF_REPO, "model.safetensors", local_dir=HF_DOWNLOAD_DIR, token=token
    )
    template = build_dinov3_template()
    converted = converter(load_file(safetensors_path), template.state_dict())
    del template

    DINOV3_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    partial = DINOV3_WEIGHTS.with_suffix(".pth.partial")
    torch.save(converted, partial)
    partial.replace(DINOV3_WEIGHTS)
    shutil.rmtree(HF_DOWNLOAD_DIR, ignore_errors=True)
    print(f"Converted DINOv3 weights: {DINOV3_WEIGHTS}")


def verify_dinov3() -> None:
    """Strict, weights_only load into the DINOv3 model, as the fork's notebook does."""
    import torch

    encoder = build_dinov3_template()
    encoder.load_state_dict(torch.load(DINOV3_WEIGHTS, map_location="cpu", weights_only=True), strict=True)
    print("DINOv3 weights verified")


def main() -> None:
    require_dinov3_repo()
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    download_checkpoint()
    convert_dinov3()
    verify_dinov3()


if __name__ == "__main__":
    main()
