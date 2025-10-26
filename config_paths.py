from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path.cwd()

DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"

ESSENTIA_MUSICNN_EMBED_PB = MODELS_DIR / "msd-musicnn-1.pb"
ESSENTIA_DEAM_HEAD_PB = MODELS_DIR / "deam-msd-musicnn-2.pb"
