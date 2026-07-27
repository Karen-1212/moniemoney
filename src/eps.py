"""Yahoo-style trailing twelve-month (TTM) diluted EPS.

EPS_365 is a step series: at each fiscal period-end,
TTM = sum of the last 4 reported Diluted EPS quarters
(matching Yahoo's trailingDilutedEPS / ttm_income_stmt Diluted EPS).

No 0q/+1q forecast fill — Yahoo's published TTM uses reported quarters only.
"""

from __future__ import annotations

import json
import time
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
from yfinance.data import YfData


def _extract_diluted_eps_stmt(ticker: yf.Ticker) -> pd.Series:
    """Diluted EPS from quarterly income statement, indexed by fiscal period-end."""
    stmt = None
    for getter in (
        lambda: ticker.quarterly_income_stmt,
        lambda: ticker.quarterly_financials,
    ):
        try:
            stmt = getter()
        except Exception:
            stmt = None
        if stmt is not None and not stmt.empty:
            break
    if stmt is None or stmt.empty:
        return pd.Series(dtype=float)

    keys = ("Diluted EPS", "DilutedEPS", "Basic EPS", "BasicEPS")
    row = None
    lower_map = {str(i).lower(): i for i in stmt.index}
    for key in keys:
        if key in stmt.index:
            row = stmt.loc[key]
            break
        if key.lower() in lower_map:
            row = stmt.loc[lower_map[key.lower()]]
            break
    if row is None:
        return pd.Series(dtype=float)

    s = pd.to_numeric(row, errors="coerce").dropna()
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    return s.sort_index()


def _fetch_ttm_diluted_from_stmt(ticker: yf.Ticker) -> Optional[tuple[pd.Timestamp, float]]:
    """Latest TTM Diluted EPS from ttm_income_stmt, if available."""
    try:
        ttm = ticker.ttm_income_stmt
    except Exception:
        return None
    if ttm is None or ttm.empty:
        return None

    row = None
    for key in ("Diluted EPS", "DilutedEPS"):
        if key in ttm.index:
            row = ttm.loc[key]
            break
    if row is None:
        return None

    s = pd.to_numeric(row, errors="coerce").dropna()
    if s.empty:
        return None
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    s = s.sort_index()
    return pd.Timestamp(s.index[-1]).normalize(), float(s.iloc[-1])


def _fetch_yahoo_trailing_diluted_eps(symbol: str) -> pd.Series:
    """
    Yahoo fundamentals-timeseries trailingDilutedEPS (filing TTM snapshots).

    These asOfDate values are the closest match to Yahoo Finance TTM Diluted EPS.
    """
    try:
        data = YfData(session=None)
        end = pd.Timestamp.now("UTC").ceil("D")
        url = (
            f"https://query2.finance.yahoo.com/ws/fundamentals-timeseries/v1/finance/"
            f"timeseries/{symbol}?symbol={symbol}&type=trailingDilutedEPS"
            f"&period1=0&period2={int(end.timestamp())}"
        )
        js = json.loads(data.cache_get(url=url).text)
        results = (js.get("timeseries") or {}).get("result") or []
        points: dict[pd.Timestamp, float] = {}
        for item in results:
            for key, vals in item.items():
                if key in ("meta", "timestamp") or not vals:
                    continue
                if "trailingDilutedEPS" not in key and key != "trailingDilutedEPS":
                    # keys look like trailingDilutedEPS
                    if "DilutedEPS" not in key:
                        continue
                for v in vals:
                    try:
                        as_of = pd.Timestamp(v["asOfDate"]).normalize()
                        raw = float(v["reportedValue"]["raw"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    if np.isfinite(raw):
                        points[as_of] = raw
        if not points:
            return pd.Series(dtype=float)
        s = pd.Series(points, dtype=float).sort_index()
        s.name = "trailing_diluted_eps"
        return s
    except Exception:
        return pd.Series(dtype=float)


def _fetch_screener_eps(ticker: yf.Ticker) -> pd.Series:
    """Reported EPS from earnings screener (announcement dates). History fallback only."""
    ed = None
    try:
        ed = ticker._get_earnings_dates_using_screener(limit=80)
    except Exception:
        ed = None
    if ed is None or (isinstance(ed, pd.DataFrame) and ed.empty):
        try:
            ed = ticker.get_earnings_dates(limit=80)
        except Exception:
            ed = None
    if ed is None or ed.empty:
        return pd.Series(dtype=float)

    cols = {c.lower().replace(" ", "_"): c for c in ed.columns}
    reported_col = cols.get("reported_eps") or cols.get("reportedeps")
    event_col = cols.get("event_type") or cols.get("eventtype")
    if reported_col is None:
        return pd.Series(dtype=float)

    df = ed.copy()
    if event_col is not None:
        df = df[df[event_col].astype(str).str.lower().eq("earnings")]

    eps = pd.to_numeric(df[reported_col], errors="coerce")
    idx = pd.to_datetime(df.index, utc=True).tz_convert(None).normalize()
    s = pd.Series(eps.values, index=idx, dtype=float).dropna()
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s


def _fiscal_end_months(period_ends: pd.DatetimeIndex) -> list[int]:
    months = sorted({pd.Timestamp(d).month for d in period_ends})
    return months if months else [3, 6, 9, 12]


def _month_end(year: int, month: int) -> pd.Timestamp:
    return (pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)).normalize()


def _quarter_end_on_or_before(date: pd.Timestamp, fiscal_months: list[int]) -> pd.Timestamp | None:
    date = pd.Timestamp(date).normalize()
    candidates: list[pd.Timestamp] = []
    for year in range(date.year - 2, date.year + 1):
        for month in fiscal_months:
            qe = _month_end(year, month)
            if qe <= date:
                candidates.append(qe)
    return max(candidates) if candidates else None


def _map_screener_to_fiscal_ends(
    screener: pd.Series,
    fiscal_months: list[int],
    existing_ends: set[pd.Timestamp],
) -> pd.Series:
    """Map announcement-date EPS to estimated fiscal period-ends (older history only)."""
    mapped: dict[pd.Timestamp, float] = {}
    for ann, eps in screener.items():
        pe = _quarter_end_on_or_before(pd.Timestamp(ann), fiscal_months)
        if pe is None or not np.isfinite(eps):
            continue
        if pe in existing_ends:
            continue
        mapped[pe] = float(eps)
    if not mapped:
        return pd.Series(dtype=float)
    s = pd.Series(mapped, dtype=float).sort_index()
    s.index = pd.to_datetime(s.index).normalize()
    return s


def load_fiscal_quarters(ticker: yf.Ticker) -> pd.Series:
    """
    Fiscal-quarter Diluted EPS indexed by period-end.

    Prefer income-statement Diluted EPS. Older gaps filled from screener Reported EPS
    mapped to fiscal ends (less precise; used only when statement history is short).
    """
    stmt = _extract_diluted_eps_stmt(ticker)
    if stmt.empty:
        screener = _fetch_screener_eps(ticker)
        if screener.empty:
            return pd.Series(dtype=float)
        return _map_screener_to_fiscal_ends(screener, [3, 6, 9, 12], set())

    fiscal_months = _fiscal_end_months(stmt.index)
    existing = {pd.Timestamp(d).normalize() for d in stmt.index}
    screener = _fetch_screener_eps(ticker)
    older = _map_screener_to_fiscal_ends(screener, fiscal_months, existing)

    merged = pd.concat([older, stmt])
    merged = merged[~merged.index.duplicated(keep="last")].sort_index().dropna()
    return merged


def ttm_from_quarters(quarterly: pd.Series) -> pd.Series:
    """At each period-end with ≥4 prior quarters, TTM = sum of last 4 Diluted EPS."""
    if len(quarterly) < 4:
        return pd.Series(dtype=float)
    ends = [pd.Timestamp(d).normalize() for d in quarterly.index]
    vals = quarterly.to_numpy(dtype=float)
    out: dict[pd.Timestamp, float] = {}
    for i in range(3, len(ends)):
        out[ends[i]] = float(vals[i - 3 : i + 1].sum())
    return pd.Series(out, dtype=float).sort_index()


def build_yahoo_ttm_steps(ticker: yf.Ticker, symbol: str) -> pd.Series:
    """
    Build TTM Diluted EPS steps keyed by fiscal period-end.

    Priority at each date (highest wins):
      1. Yahoo trailingDilutedEPS timeseries
      2. ttm_income_stmt Diluted EPS (latest point)
      3. Sum of last 4 quarterly Diluted EPS
    """
    yahoo_ttm = _fetch_yahoo_trailing_diluted_eps(symbol)
    quarterly = load_fiscal_quarters(ticker)
    computed = ttm_from_quarters(quarterly)

    # Start with computed 4Q sums, overlay Yahoo timeseries (more accurate)
    steps = computed.copy()
    for dt, val in yahoo_ttm.items():
        steps.loc[pd.Timestamp(dt).normalize()] = float(val)

    ttm_stmt = _fetch_ttm_diluted_from_stmt(ticker)
    if ttm_stmt is not None:
        dt, val = ttm_stmt
        steps.loc[dt] = val

    if steps.empty:
        return pd.Series(dtype=float)
    steps = steps[~steps.index.duplicated(keep="last")].sort_index()
    steps = steps.replace([np.inf, -np.inf], np.nan).dropna()
    steps.name = "ttm_diluted_eps"
    return steps


def build_eps_365_series(
    symbol: str,
    dates: pd.DatetimeIndex,
    sleep_s: float = 0.25,
    window_days: int = 365,  # kept for API compatibility; unused (Yahoo TTM is 4Q)
) -> pd.Series:
    """
    Yahoo-style TTM diluted EPS aligned to trading dates.

    Step function: TTM updates at fiscal period-ends and is forward-filled
    until the next filing. Matches Yahoo trailingDilutedEPS as closely as possible.
    """
    del window_days  # unused
    time.sleep(sleep_s)
    ticker = yf.Ticker(symbol)
    dates = pd.DatetimeIndex(pd.to_datetime(dates)).tz_localize(None).normalize()
    out = pd.Series(np.nan, index=dates, name=symbol, dtype=float)

    try:
        steps = build_yahoo_ttm_steps(ticker, symbol)
    except Exception:
        return out

    if steps.empty:
        return out

    # Forward-fill steps onto the trading calendar (asOfDate = period-end)
    step_dates = steps.index.to_numpy(dtype="datetime64[ns]")
    step_vals = steps.to_numpy(dtype=float)
    date_vals = dates.values.astype("datetime64[ns]")
    idx = np.searchsorted(step_dates, date_vals, side="right") - 1
    valid = idx >= 0
    result = np.full(len(dates), np.nan, dtype=float)
    result[valid] = step_vals[idx[valid]]
    return pd.Series(result, index=dates, name=symbol, dtype=float)


def build_eps_panel(
    symbols: list[str],
    dates: pd.DatetimeIndex,
    sleep_s: float = 0.25,
) -> pd.DataFrame:
    cols = {}
    for i, sym in enumerate(symbols, 1):
        print(f"  EPS [{i}/{len(symbols)}] {sym}", flush=True)
        try:
            cols[sym] = build_eps_365_series(sym, dates, sleep_s=sleep_s)
        except Exception as exc:
            print(f"    failed {sym}: {exc}", flush=True)
            cols[sym] = pd.Series(np.nan, index=dates, name=sym)
    return pd.DataFrame(cols, index=dates)
