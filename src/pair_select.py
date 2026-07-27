"""Relative-PE pair selection with locked Health Care thresholds."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Locked default Health Care selection rule (do not change lightly)
# ---------------------------------------------------------------------------
THRESHOLDS = {
    "n_days_min": 924,
    "corr_min": 0.65,
    "adf_t_max": -2.20,
    "half_life_min": 20.0,
    "half_life_max": 250.0,
    "exc_ge_1_5_min": 3,
    "revert_rate_min": 0.60,
    "z_entry": 1.5,
    "z_exit": 0.5,
    "max_pairs_per_ticker": 2,
}

SCORE_WEIGHTS = {
    "corr": 0.30,
    "adf": 0.25,
    "hl": 0.20,
    "exc": 0.15,
    "rev": 0.10,
}

HALF_LIFE_TARGET = 45.0


@dataclass
class PairMetrics:
    symbol_a: str
    symbol_b: str
    n_days: int
    pearson_r: float
    adf_t: float
    half_life: float
    exc_ge_1_5: int
    revert_rate: float
    mean_log_rel_pe: float
    spread_sd: float
    score: float = np.nan


def robust_z(series: pd.Series) -> pd.Series:
    x = series.to_numpy(dtype=float)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    scale = mad * 1.4826 if mad > 1e-12 else (float(np.std(x)) if float(np.std(x)) > 1e-12 else 1.0)
    return (series - med) / scale


def _pair_metrics(a: str, b: str, pe: pd.DataFrame) -> PairMetrics | None:
    sub = pe[[a, b]].dropna()
    sub = sub[(sub[a] > 0) & (sub[b] > 0)]
    n = len(sub)
    if n < THRESHOLDS["n_days_min"]:
        return None

    x = sub[a].to_numpy(dtype=float)
    y = sub[b].to_numpy(dtype=float)
    if np.std(x) == 0 or np.std(y) == 0:
        return None

    r = float(np.corrcoef(x, y)[0, 1])
    s = np.log(x / y)
    mu = float(np.mean(s))
    sd = float(np.std(s, ddof=1))
    if not np.isfinite(sd) or sd <= 1e-8:
        return None

    # ADF(0)-style: Δs_t = α + β s_{t-1} + ε
    s_lag = s[:-1]
    ds = np.diff(s)
    X = np.column_stack([np.ones_like(s_lag), s_lag])
    beta_hat = np.linalg.lstsq(X, ds, rcond=None)[0]
    resid = ds - X.dot(beta_hat)
    dof = len(ds) - 2
    if dof <= 0:
        return None
    s2 = float(np.sum(resid**2) / dof)
    cov = s2 * np.linalg.inv(X.T @ X)
    se = float(np.sqrt(cov[1, 1])) if cov[1, 1] > 0 else np.nan
    beta = float(beta_hat[1])
    adf_t = float(beta / se) if np.isfinite(se) and se > 0 else np.nan

    phi = 1.0 + beta
    if phi <= 0 or phi >= 0.9995:
        half_life = np.inf
    else:
        half_life = float(-np.log(2.0) / np.log(phi))

    z = (s - mu) / sd
    z_entry = THRESHOLDS["z_entry"]
    z_exit = THRESHOLDS["z_exit"]
    in_exc = False
    exc = 0
    rev = 0
    for zi in z:
        if (not in_exc) and abs(zi) >= z_entry:
            in_exc = True
            exc += 1
        elif in_exc and abs(zi) <= z_exit:
            in_exc = False
            rev += 1
    revert_rate = (rev / exc) if exc > 0 else 0.0

    return PairMetrics(
        symbol_a=a,
        symbol_b=b,
        n_days=n,
        pearson_r=r,
        adf_t=adf_t,
        half_life=half_life,
        exc_ge_1_5=exc,
        revert_rate=revert_rate,
        mean_log_rel_pe=mu,
        spread_sd=sd,
    )


def passes_thresholds(m: PairMetrics) -> bool:
    t = THRESHOLDS
    return (
        m.n_days >= t["n_days_min"]
        and m.pearson_r >= t["corr_min"]
        and np.isfinite(m.adf_t)
        and m.adf_t <= t["adf_t_max"]
        and np.isfinite(m.half_life)
        and t["half_life_min"] <= m.half_life <= t["half_life_max"]
        and m.exc_ge_1_5 >= t["exc_ge_1_5_min"]
        and m.revert_rate >= t["revert_rate_min"]
    )


def score_candidates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["corr_z"] = robust_z(out["pearson_r"])
    out["adf_z"] = robust_z(-out["adf_t"])
    out["hl_z"] = robust_z(-np.abs(np.log(out["half_life"] / HALF_LIFE_TARGET)))
    out["exc_z"] = robust_z(out["exc_ge_1_5"].astype(float))
    out["rev_z"] = robust_z(out["revert_rate"])
    w = SCORE_WEIGHTS
    out["score"] = (
        w["corr"] * out["corr_z"]
        + w["adf"] * out["adf_z"]
        + w["hl"] * out["hl_z"]
        + w["exc"] * out["exc_z"]
        + w["rev"] * out["rev_z"]
    )
    return out.sort_values("score", ascending=False).reset_index(drop=True)


def diversify_top_n(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    max_per = THRESHOLDS["max_pairs_per_ticker"]
    selected: list[pd.Series] = []
    use: dict[str, int] = {}
    for _, row in df.iterrows():
        a, b = row["symbol_a"], row["symbol_b"]
        if use.get(a, 0) >= max_per or use.get(b, 0) >= max_per:
            continue
        selected.append(row)
        use[a] = use.get(a, 0) + 1
        use[b] = use.get(b, 0) + 1
        if len(selected) >= n:
            break
    return pd.DataFrame(selected).reset_index(drop=True)


def select_relative_pe_pairs(
    pe: pd.DataFrame,
    members: pd.DataFrame | None = None,
    top_n: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return (all_passing_candidates_scored, top_n_diversified).

    Uses locked THRESHOLDS / SCORE_WEIGHTS.
    """
    cols = [c for c in pe.columns if pe[c].notna().any()]
    rows: list[dict] = []
    for a, b in combinations(sorted(cols), 2):
        m = _pair_metrics(a, b, pe)
        if m is None or not passes_thresholds(m):
            continue
        rows.append(asdict(m))

    if not rows:
        empty = pd.DataFrame()
        return empty, empty

    cand = pd.DataFrame(rows)
    if members is not None and not members.empty:
        name = dict(zip(members["Symbol"], members["Security"]))
        cand["security_a"] = cand["symbol_a"].map(name)
        cand["security_b"] = cand["symbol_b"].map(name)

    scored = score_candidates(cand)
    top = diversify_top_n(scored, n=top_n)
    return scored, top


def spread_z_series(pe: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    """Daily spread and z-score for pair (A, B), using full-sample μ, σ."""
    sub = pe[[a, b]].dropna()
    sub = sub[(sub[a] > 0) & (sub[b] > 0)].copy()
    s = np.log(sub[a] / sub[b])
    mu = float(s.mean())
    sd = float(s.std(ddof=1))
    out = pd.DataFrame({"spread": s, "z": (s - mu) / sd}, index=sub.index)
    out["pe_a"] = sub[a]
    out["pe_b"] = sub[b]
    return out
