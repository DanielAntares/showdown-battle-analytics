"""Shared config loading and path resolution."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config() -> dict:
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # resolve paths relative to the repo root so scripts work from anywhere
    cfg["paths"] = {k: ROOT / v for k, v in cfg["paths"].items()}
    return cfg
