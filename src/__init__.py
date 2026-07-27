"""Shared path helpers for the sector PE pipeline."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
SECTORS_DIR = DATA_DIR / "sectors"
SP500_PATH = ROOT / "SP500.csv"


def sector_slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def sector_dir(name: str) -> Path:
    return SECTORS_DIR / sector_slug(name)
