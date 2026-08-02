"""Relative-PE pair selection with locked thresholds (same GICS sub-industry)."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from src import load_subindustry_map, same_subindustry_combinations

# Locked selection rule (do not change lightly)
THRESHOLDS = {
    "n_days_min": 924,
    "corr_min": 0.65,
    "adf_t_max": -2.20,
    "half_life_min": 20.0,
    "half_life_max": 250.0,
    "exc_ge_1_5_min": 3,
    "revert_rate_min": 0.60,
    "z_entry": 1.2,
    "z_exit": 0.75,
    "max_pairs_per_ticker": 2,
}


def thresholds_for(metric: str = "pe") -> dict:
    del metric  # PE-only book
    return THRESHOLDS

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
    mean_log_rel: float
    spread_sd: float
    score: float = np.nan


def robust_z(series: pd.Series) -> pd.Series:
    x = series.to_numpy(dtype=float)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    scale = mad * 1.4826 if mad > 1e-12 else (float(np.std(x)) if float(np.std(x)) > 1e-12 else 1.0)
    return (series - med) / scale


def _pair_metrics(
    a: str, b: str, panel: pd.DataFrame, thresholds: dict | None = None
) -> PairMetrics | None:
    t = thresholds or THRESHOLDS
    sub = panel[[a, b]].dropna()
    sub = sub[(sub[a] > 0) & (sub[b] > 0)]
    n = len(sub)
    if n < t["n_days_min"]:
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
    z_entry = t["z_entry"]
    z_exit = t["z_exit"]
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
        mean_log_rel=mu,
        spread_sd=sd,
    )


def passes_thresholds(m: PairMetrics, thresholds: dict | None = None) -> bool:
    t = thresholds or THRESHOLDS
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


def select_relative_valuation_pairs(
    panel: pd.DataFrame,
    members: pd.DataFrame | None = None,
    top_n: int = 10,
    metric: str = "pe",
    n_days_min: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return (all_passing_candidates_scored, top_n_diversified).

    Only same-GICS-Sub-Industry pairs are considered.
    Uses locked THRESHOLDS / SCORE_WEIGHTS (PB: n_days_min=800) unless n_days_min overrides.
    Column mean_log_rel_{metric}.
    """
    metric = metric.lower()
    t = dict(thresholds_for(metric))
    if n_days_min is not None:
        t["n_days_min"] = int(n_days_min)
    mean_col = f"mean_log_rel_{metric}"
    cols = [c for c in panel.columns if panel[c].notna().any()]
    sub_map = load_subindustry_map(members)
    rows: list[dict] = []
    for a, b in same_subindustry_combinations(cols, sub_map):
        m = _pair_metrics(a, b, panel, thresholds=t)
        if m is None or not passes_thresholds(m, thresholds=t):
            continue
        d = asdict(m)
        d[mean_col] = d.pop("mean_log_rel")
        d["subindustry"] = sub_map.get(a, "")
        rows.append(d)

    if not rows:
        empty = pd.DataFrame(
            columns=[
                "symbol_a",
                "symbol_b",
                "subindustry",
                "n_days",
                "pearson_r",
                "adf_t",
                "half_life",
                "exc_ge_1_5",
                "revert_rate",
                mean_col,
                "spread_sd",
                "score",
            ]
        )
        return empty, empty

    cand = pd.DataFrame(rows)
    if members is not None and not members.empty:
        name = dict(zip(members["Symbol"], members["Security"]))
        cand["security_a"] = cand["symbol_a"].map(name)
        cand["security_b"] = cand["symbol_b"].map(name)

    scored = score_candidates(cand)
    top = diversify_top_n(scored, n=top_n)
    return scored, top


def select_top_significant_pairs(
    panel: pd.DataFrame,
    significant: pd.DataFrame,
    metric: str = "pe",
    top_n: int = 10,
    score_min_days: int = 252,
    members: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Rank significant (p-value) same-subindustry pairs by mean-reversion score and take top_n.

    If there are ≤ top_n scorable pairs, return all of them (still sorted by score).
    Score uses the same ingredients as candidate scoring; n_days floor for scoring
    is score_min_days (default 252) so significant pairs can be ranked even when
    they are shorter than the locked selection n_days_min.
    """
    metric = metric.lower()
    t = {**thresholds_for(metric), "n_days_min": int(score_min_days)}
    mean_col = f"mean_log_rel_{metric}"
    rows: list[dict] = []
    if significant is None or significant.empty:
        return pd.DataFrame()

    sub_map = load_subindustry_map(members)
    for _, row in significant.iterrows():
        a, b = str(row["symbol_a"]), str(row["symbol_b"])
        if a not in panel.columns or b not in panel.columns:
            continue
        sa, sb = sub_map.get(a, ""), sub_map.get(b, "")
        if not sa or sa != sb:
            continue
        m = _pair_metrics(a, b, panel, thresholds=t)
        if m is None:
            continue
        d = asdict(m)
        d[mean_col] = d.pop("mean_log_rel")
        d["subindustry"] = sa
        if "security_a" in row.index:
            d["security_a"] = row.get("security_a", "")
            d["security_b"] = row.get("security_b", "")
        rows.append(d)

    if not rows:
        return pd.DataFrame()

    scored = score_candidates(pd.DataFrame(rows))
    return scored.head(top_n).reset_index(drop=True)


def select_relative_pe_pairs(
    pe: pd.DataFrame,
    members: pd.DataFrame | None = None,
    top_n: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (all_passing_candidates_scored, top_n_diversified) for PE."""
    return select_relative_valuation_pairs(pe, members=members, top_n=top_n, metric="pe")


def spread_z_series(panel: pd.DataFrame, a: str, b: str, metric: str = "pe") -> pd.DataFrame:
    """Daily spread and z-score for pair (A, B), using full-sample μ, σ."""
    metric = metric.lower()
    sub = panel[[a, b]].dropna()
    sub = sub[(sub[a] > 0) & (sub[b] > 0)].copy()
    s = np.log(sub[a] / sub[b])
    mu = float(s.mean())
    sd = float(s.std(ddof=1))
    out = pd.DataFrame({"spread": s, "z": (s - mu) / sd}, index=sub.index)
    out[f"{metric}_a"] = sub[a]
    out[f"{metric}_b"] = sub[b]
    return out
