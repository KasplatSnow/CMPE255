"""Misc helpers: reproducibility, logging, and metrics persistence."""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np

try:
    import torch
except ModuleNotFoundError:  # CPU‑only box is fine
    torch = None  # type: ignore

def set_seed(seed: int = 42) -> None:
    """Fix every RNG we care about for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch is not None:  # pragma: no cover – optional GPU
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_logger(name: str = "cmpe255") -> logging.Logger:
    """Return a singleton console logger with ISO timestamps."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
    return logger

def save_metrics(row: Dict[str, Any], path: str | Path) -> None:
    """Append one metrics row into a CSV, creating the file if missing."""
    import pandas as pd

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])
    if path.exists():
        df_existing = pd.read_csv(path)
        df = pd.concat([df_existing, df], ignore_index=True)
    df.to_csv(path, index=False)