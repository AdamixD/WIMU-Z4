from pathlib import Path

import torch
from dotenv import load_dotenv
from loguru import logger
from tqdm import tqdm

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

# https://github.com/Delgan/loguru/issues/135

logger.remove(0)
logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)

SAMPLE_RATE = 16000
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
