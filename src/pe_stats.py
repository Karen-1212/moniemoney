"""PE pair correlation and relative-multiple statistics (numpy-only)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src import load_subindustry_map, same_subindustry_combinations


def pearsonr_pvalue(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Pearson r and two-sided p-value via Student-t approximation."""
    n = len(x)
    if n < 3:
        return np.nan, np.nan
    x = x - x.mean()
    y = y - y.mean()
    ssx = np.dot(x, x)
    ssy = np.dot(y, y)
    if ssx <= 0 or ssy <= 0:
        return np.nan, np.nan
    r = float(np.dot(x, y) / np.sqrt(ssx * ssy))
    r = float(np.clip(r, -1.0 + 1e-15, 1.0 - 1e-15))
    df = n - 2
    t_stat = r * np.sqrt(df / (1.0 - r * r))
    # two-sided p from regularized incomplete beta / erfc approximation via t CDF
    p = float(2.0 * student_t_sf(abs(t_stat), df))
    return r, p


def student_t_sf(t: float, df: float) -> float:
    """Survival function P(T > t) for Student-t; accurate enough for screening."""
    # Use relationship with regularized incomplete beta
    x = df / (df + t * t)
    return 0.5 * incomplete_beta_reg(df / 2.0, 0.5, x)


def incomplete_beta_reg(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a,b) via continued fraction (Lentz)."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    # Symmetry for numerical stability
    if x > (a + 1.0) / (a + b + 2.0):
        return 1.0 - incomplete_beta_reg(b, a, 1.0 - x)

    ln_beta = log_gamma(a) + log_gamma(b) - log_gamma(a + b)
    front = np.exp(np.log(x) * a + np.log(1.0 - x) * b - ln_beta) / a
    return front * betacf(a, b, x)


def betacf(a: float, b: float, x: float, max_iter: int = 200, eps: float = 1e-10) -> float:
    am, bm = 1.0, 1.0
    az = 1.0
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    bz = 1.0 - qab * x / qap
    for m in range(1, max_iter + 1):
        em = float(m)
        tem = em + em
        d = em * (b - em) * x / ((qam + tem) * (a + tem))
        ap = az + d * am
        bp = bz + d * bm
        d = -(a + em) * (qab + em) * x / ((a + tem) * (qap + tem))
        app = ap + d * az
        bpp = bp + d * bz
        am, bm = ap / bpp, bp / bpp
        az, bz = app / bpp, 1.0
        if abs(az - am) < eps * abs(az):
            return az
    return az


def log_gamma(z: float) -> float:
    return float(math.lgamma(z))


def student_t_ppf(p: float, df: float) -> float:
    """Inverse CDF of Student-t via binary search on the survival function."""
    lo, hi = 0.0, 100.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        sf = student_t_sf(mid, df)
        if sf > (1.0 - p):
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def mean_ci_99(x: np.ndarray) -> tuple[float, float, float]:
    """Return mean, ci_low, ci_high for 99% CI of the mean."""
    n = len(x)
    mean = float(np.mean(x))
    if n < 2:
        return mean, np.nan, np.nan
    s = float(np.std(x, ddof=1))
    se = s / np.sqrt(n)
    tcrit = student_t_ppf(0.995, n - 1)
    return mean, mean - tcrit * se, mean + tcrit * se


def significant_valuation_pairs(
    panel: pd.DataFrame,
    members: pd.DataFrame,
    metric: str = "pe",
    min_days: int = 252,
    p_threshold: float = 0.01,
) -> pd.DataFrame:
    """Find same-subindustry pairs with significant correlation and log relative-multiple stats."""
    metric = metric.lower()
    mean_col = f"mean_log_rel_{metric}"
    exp_col = f"exp_mean_log_rel_{metric}"
    name_map = dict(zip(members["Symbol"], members["Security"]))
    sub_map = load_subindustry_map(members)
    tickers = [c for c in panel.columns if panel[c].notna().sum() >= min_days]
    rows = []

    pairs = list(same_subindustry_combinations(tickers, sub_map))
    n_subs = len({sub_map[t] for t in tickers if t in sub_map})
    print(
        f"  Evaluating {len(pairs)} same-subindustry pairs "
        f"among {len(tickers)} tickers ({n_subs} sub-industries)...",
        flush=True,
    )

    for i, (a, b) in enumerate(pairs, 1):
        if i % 500 == 0:
            print(f"    pair {i}/{len(pairs)}", flush=True)
        sub = panel[[a, b]].dropna()
        # require positive multiples for relative ratio and correlation on same mask
        sub = sub[(sub[a] > 0) & (sub[b] > 0)]
        n = len(sub)
        if n < min_days:
            continue
        xa = sub[a].to_numpy(dtype=float)
        xb = sub[b].to_numpy(dtype=float)
        r, p = pearsonr_pvalue(xa, xb)
        if not np.isfinite(p) or p >= p_threshold:
            continue
        rel = np.log(xa / xb)
        mean_rel, lo, hi = mean_ci_99(rel)
        si = sub_map.get(a, "")
        rows.append(
            {
                "symbol_a": a,
                "security_a": name_map.get(a, ""),
                "symbol_b": b,
                "security_b": name_map.get(b, ""),
                "subindustry": si,
                "n_days": n,
                "pearson_r": r,
                "p_value": p,
                mean_col: mean_rel,
                "ci99_low": lo,
                "ci99_high": hi,
                exp_col: float(np.exp(mean_rel)),
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["p_value", "pearson_r"], ascending=[True, False]).reset_index(drop=True)
    return out


def significant_pe_pairs(
    pe: pd.DataFrame,
    members: pd.DataFrame,
    min_days: int = 252,
    p_threshold: float = 0.01,
) -> pd.DataFrame:
    """Find within-panel pairs with significant PE correlation and log relative PE stats."""
    return significant_valuation_pairs(
        pe, members, metric="pe", min_days=min_days, p_threshold=p_threshold
    )
