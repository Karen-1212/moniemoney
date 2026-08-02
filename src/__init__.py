"""Shared path helpers for the Consumer Discretionary PE pipeline."""

from __future__ import annotations

import re
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Iterable, Iterator

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
SECTORS_DIR = DATA_DIR / "sectors"
SP500_PATH = ROOT / "SP500.csv"

# Train / test protocol used by the factsheet
TRAIN_DATA_START = "2022-01-01"
TRAIN_END = "2024-12-31"
TEST_START = "2025-01-01"

DEFAULT_SECTOR = "Consumer Discretionary"

_SUBINDUSTRY_COLS = ("GICS Sub-Industry", "subindustry", "Sub-Industry")


def sector_slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def sectors_root(metric: str = "pe") -> Path:
    del metric  # PE-only book
    return SECTORS_DIR


def sector_dir(name: str = DEFAULT_SECTOR, metric: str = "pe") -> Path:
    return sectors_root(metric) / sector_slug(name)


def load_subindustry_map(members: pd.DataFrame | None = None) -> dict[str, str]:
    """Symbol → GICS Sub-Industry (from members if present, else SP500.csv)."""
    if members is not None and not members.empty:
        col = next((c for c in _SUBINDUSTRY_COLS if c in members.columns), None)
        if col is not None:
            out: dict[str, str] = {}
            for sym, si in zip(members["Symbol"], members[col]):
                if pd.isna(si) or not str(si).strip():
                    continue
                out[str(sym).strip()] = str(si).strip()
            if out:
                return out
    sp = pd.read_csv(SP500_PATH)
    return {
        str(s).strip(): str(si).strip()
        for s, si in zip(sp["Symbol"], sp["GICS Sub-Industry"])
        if pd.notna(si) and str(si).strip()
    }


def same_subindustry_combinations(tickers: Iterable[str], sub_map: dict[str, str]) -> Iterator[tuple[str, str]]:
    """Yield unordered pairs (a, b) that share the same GICS Sub-Industry."""
    by_sub: dict[str, list[str]] = defaultdict(list)
    for t in tickers:
        si = sub_map.get(str(t), "")
        if si:
            by_sub[si].append(str(t))
    for ticks in by_sub.values():
        yield from combinations(sorted(ticks), 2)
