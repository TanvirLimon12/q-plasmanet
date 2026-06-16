"""Seed control. Every result is mean +/- std over seeds [0,1,2,3,4]."""
from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Fix all RNGs we touch. torch is seeded only if importable (kept optional)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def log_device() -> str:
    """Return a one-line device string; log it in every run."""
    try:
        import torch

        if torch.cuda.is_available():
            return f"cuda:{torch.cuda.current_device()} ({torch.cuda.get_device_name(0)})"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps (Apple Silicon)"
        return "cpu"
    except ImportError:
        return "cpu (torch not installed)"
