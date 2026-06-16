"""YAML config loading with deep-merge and dotted access.

Kept dependency-light (no torch) so data-audit scripts stay fast to import.
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into a copy of ``base``."""
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


class Config(dict):
    """dict with attribute access; nested dicts are wrapped lazily."""

    def __getattr__(self, name: str) -> Any:
        try:
            v = self[name]
        except KeyError as e:
            raise AttributeError(name) from e
        return Config(v) if isinstance(v, dict) else v

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def get_path(self, dotted: str, default: Any = None) -> Any:
        """Fetch a nested value by dotted key, e.g. ``cfg.get_path('data.raw_root')``."""
        cur: Any = self
        for part in dotted.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur


def load_config(paths: str | list[str] | None, overrides: dict | None = None) -> Config:
    """Load one or more YAML files, deep-merged left to right, then apply overrides.

    The repo's ``configs/base.yaml`` is always loaded first when present.
    """
    merged: dict = {}
    files: list[str] = []

    here = Path(__file__).resolve()
    repo_root = here.parents[2]          # src/utils/config.py -> repo root
    base = repo_root / "configs" / "base.yaml"
    if base.is_file():
        files.append(str(base))

    if paths:
        files += [paths] if isinstance(paths, str) else list(paths)

    for f in files:
        with open(f, "r") as fh:
            data = yaml.safe_load(fh) or {}
        merged = _deep_merge(merged, data)

    if overrides:
        merged = _deep_merge(merged, overrides)

    cfg = Config(merged)
    cfg["_repo_root"] = str(repo_root)
    cfg["_config_files"] = files
    return cfg


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Standard CLI surface every phase script shares: --config and --smoke."""
    parser.add_argument(
        "--config",
        nargs="*",
        default=None,
        help="One or more YAML configs, merged on top of configs/base.yaml.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke mode: 1 seed, tiny subset, <=2 min. For fast verification.",
    )
    return parser
