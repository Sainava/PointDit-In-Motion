"""Repository paths, resolved from this file rather than the working directory."""

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FORK_DIR = REPO_ROOT / "third_party" / "pointdit"
FORK_KAGGLE_NOTEBOOK = FORK_DIR / "PointDiT_Kaggle.ipynb"
# The fork's default DINOv3 location; the fork's own .gitignore excludes it.
DINOV3_REPO_DIR = FORK_DIR / "third_party" / "dinov3"

WEIGHTS_DIR = REPO_ROOT / "weights"
POINTDIT_CHECKPOINT_NAME = "pointditl-512-mixdata-nodinov3-240c1a4f.pth"
POINTDIT_CHECKPOINT = WEIGHTS_DIR / POINTDIT_CHECKPOINT_NAME
DINOV3_WEIGHTS_DIR = WEIGHTS_DIR / "dinov3"
DINOV3_WEIGHTS = DINOV3_WEIGHTS_DIR / "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"


def configure_fork() -> None:
    """Make the fork importable and point it at our DINOv3 checkout and weights.

    The fork reads DINOV3_REPO and DINOV3_WEIGHTS_DIR from the environment; values
    already set by the caller take precedence.
    """
    if not (FORK_DIR / "model.py").is_file():
        raise FileNotFoundError(
            f"PointDiT fork not found at {FORK_DIR}. "
            "Run: git submodule update --init --recursive"
        )
    os.environ.setdefault("DINOV3_REPO", str(DINOV3_REPO_DIR))
    os.environ.setdefault("DINOV3_WEIGHTS_DIR", str(DINOV3_WEIGHTS_DIR))
    fork = str(FORK_DIR)
    if fork not in sys.path:
        sys.path.insert(0, fork)
