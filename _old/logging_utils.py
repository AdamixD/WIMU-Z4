from __future__ import annotations

import logging
import sys


def setup_logger(level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("mer")
    if logger.handlers:
        return logger
    lvl = getattr(logging, str(level).upper(), logging.INFO)
    logger.setLevel(lvl)
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s", "%H:%M:%S"))
    logger.addHandler(h)
    return logger
