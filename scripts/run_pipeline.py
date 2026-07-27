#!/usr/bin/env python3
"""Run sector PE pipeline. Default: Health Care only."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import CACHE_DIR, SECTORS_DIR, SP500_PATH, sector_dir, sector_slug
from src.eps import build_eps_panel
from src.pe_stats import significant_pe_pairs


def load_universe() -> pd.DataFrame:
    df = pd.read_csv(SP500_PATH)
    df["Symbol"] = df["Symbol"].astype(str).str.strip()
    return df


def write_members(universe: pd.DataFrame, sector: str, force: bool = False) -> pd.DataFrame:
    out_dir = sector_dir(sector)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "members.csv"
    if path.exists() and not force:
        print(f"members exist: {path}")
        return pd.read_csv(path)

    members = (
        universe.loc[universe["GICS Sector"] == sector, ["Symbol", "Security"]]
        .drop_duplicates(subset=["Symbol"])
        .sort_values("Symbol")
        .reset_index(drop=True)
    )
    members.to_csv(path, index=False)
    print(f"Wrote {len(members)} members -> {path}")
    return members


def fetch_close(members: pd.DataFrame, sector: str, force: bool = False) -> pd.DataFrame:
    out_dir = sector_dir(sector)
    path = out_dir / "close.csv"
    cache_path = CACHE_DIR / f"{sector_slug(sector)}_close.csv"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if path.exists() and not force:
        print(f"close exists: {path}")
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        return df

    symbols = members["Symbol"].tolist()
    print(f"Downloading 5y Close for {len(symbols)} tickers...")
    raw = yf.download(
        symbols,
        period="5y",
        auto_adjust=True,
        group_by="column",
        threads=True,
        progress=True,
    )

    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.get_level_values(0):
            close = raw["Close"].copy()
        else:
            # sometimes (ticker, OHLCV)
            close = raw.xs("Close", axis=1, level=-1).copy()
    else:
        # single ticker edge case
        close = raw[["Close"]].copy()
        close.columns = symbols[:1]

    close = close.sort_index()
    close.index = pd.to_datetime(close.index).tz_localize(None)
    # ensure all symbols present
    for sym in symbols:
        if sym not in close.columns:
            close[sym] = pd.NA
    close = close[symbols]
    close.to_csv(cache_path)
    close.to_csv(path)
    print(f"Wrote close -> {path}  shape={close.shape}")
    return close


def build_eps(members: pd.DataFrame, close: pd.DataFrame, sector: str, force: bool = False) -> pd.DataFrame:
    out_dir = sector_dir(sector)
    path = out_dir / "eps_365.csv"
    if path.exists() and not force:
        print(f"eps exists: {path}")
        return pd.read_csv(path, index_col=0, parse_dates=True)

    symbols = members["Symbol"].tolist()
    print(f"Building EPS_365 for {len(symbols)} tickers...")
    eps = build_eps_panel(symbols, close.index)
    eps.to_csv(path)
    print(f"Wrote eps -> {path}  shape={eps.shape}")
    return eps


def build_pe(close: pd.DataFrame, eps: pd.DataFrame, sector: str, force: bool = False) -> pd.DataFrame:
    out_dir = sector_dir(sector)
    path = out_dir / "pe.csv"
    if path.exists() and not force:
        print(f"pe exists: {path}")
        return pd.read_csv(path, index_col=0, parse_dates=True)

    # align
    common_cols = [c for c in close.columns if c in eps.columns]
    pe = close[common_cols] / eps[common_cols]
    pe = pe.where(eps[common_cols] > 0)
    pe.to_csv(path)
    print(f"Wrote pe -> {path}  shape={pe.shape}")
    return pe


def correlate(pe: pd.DataFrame, members: pd.DataFrame, sector: str, force: bool = False) -> pd.DataFrame:
    out_dir = sector_dir(sector)
    path = out_dir / "significant_pe_pairs.csv"
    if path.exists() and not force:
        print(f"pairs exist: {path}")
        return pd.read_csv(path)

    print("Computing significant PE pairs...")
    pairs = significant_pe_pairs(pe, members)
    pairs.insert(0, "sector", sector)
    pairs.to_csv(path, index=False)
    rollup = SECTORS_DIR / "all_significant_pe_pairs.csv"
    pairs.to_csv(rollup, index=False)
    print(f"Wrote {len(pairs)} significant pairs -> {path}")
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Sector PE correlation pipeline")
    parser.add_argument(
        "--sector",
        default="Health Care",
        help='GICS sector name (default: "Health Care")',
    )
    parser.add_argument("--force", action="store_true", help="Rebuild all outputs")
    parser.add_argument(
        "--skip-eps",
        action="store_true",
        help="Skip EPS rebuild even with --force (use existing eps_365.csv)",
    )
    args = parser.parse_args()

    t0 = time.time()
    universe = load_universe()
    sectors = sorted(universe["GICS Sector"].dropna().unique())
    if args.sector not in sectors:
        raise SystemExit(f"Unknown sector {args.sector!r}. Available: {sectors}")

    print(f"=== Sector: {args.sector} ===")
    members = write_members(universe, args.sector, force=args.force)
    close = fetch_close(members, args.sector, force=args.force)
    if args.skip_eps and (sector_dir(args.sector) / "eps_365.csv").exists():
        eps = pd.read_csv(sector_dir(args.sector) / "eps_365.csv", index_col=0, parse_dates=True)
    else:
        eps = build_eps(members, close, args.sector, force=args.force)
    pe = build_pe(close, eps, args.sector, force=args.force)
    pairs = correlate(pe, members, args.sector, force=args.force)

    print(f"\nDone in {time.time() - t0:.1f}s")
    print(f"Members: {len(members)}")
    print(f"Close shape: {close.shape}")
    print(f"EPS non-null (any): {(eps.notna().sum() > 0).sum()} tickers")
    print(f"Significant pairs: {len(pairs)}")
    if len(pairs):
        print(pairs.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
