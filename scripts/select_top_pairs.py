#!/usr/bin/env python3
"""Select top relative-PE pairs, review them, and backtest z-score rules."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import DEFAULT_SECTOR, load_subindustry_map, sector_dir
from src.pair_select import (
    HALF_LIFE_TARGET,
    SCORE_WEIGHTS,
    THRESHOLDS,
    select_relative_valuation_pairs,
    spread_z_series,
    thresholds_for,
)

# Selection is restricted to same GICS Sub-Industry; labels kept for review CSVs.
COMPAT_SCORE = {"same_subindustry": 2}


def compatibility(a: str, b: str, sub: dict[str, str]) -> tuple[str, int, str]:
    sa, sb = sub.get(a, ""), sub.get(b, "")
    if sa and sb and sa == sb:
        return "same_subindustry", COMPAT_SCORE["same_subindustry"], f"Both: {sa}"
    # Should not appear when selection filters to same sub-industry.
    return "mismatch", 0, f"{sa} vs {sb}"


def fundamental_event_risk(
    panel: pd.DataFrame,
    fundamental: pd.DataFrame,
    a: str,
    b: str,
    metric: str = "pe",
) -> dict:
    """
    Flag pairs where large relative-multiple spread jumps coincide with
    fundamental steps (EPS or BVPS) rather than gradual price-driven moves.
    """
    step_key = "eps" if metric == "pe" else "bvps"
    zdf = spread_z_series(panel, a, b, metric=metric)
    if len(zdf) < 60 or a not in fundamental.columns or b not in fundamental.columns:
        return {
            "event_risk": "unknown",
            "large_spread_jumps": 0,
            f"jump_with_{step_key}_step": 0,
            f"jump_{step_key}_step_share": np.nan,
        }

    s = zdf["spread"]
    ds = s.diff().abs()
    thresh = float(ds.quantile(0.99))
    jump_days = ds[ds >= thresh].dropna().index

    fund_ab = fundamental[[a, b]].reindex(zdf.index).ffill()
    fund_step = (fund_ab[a].diff().abs() > 1e-9) | (fund_ab[b].diff().abs() > 1e-9)

    coincident = 0
    for d in jump_days:
        loc = zdf.index.get_loc(d)
        lo = max(0, loc - 3)
        hi = min(len(zdf) - 1, loc + 3)
        if bool(fund_step.iloc[lo : hi + 1].any()):
            coincident += 1

    n_jumps = int(len(jump_days))
    share = (coincident / n_jumps) if n_jumps else 0.0
    if share >= 0.40:
        risk = "high"
    elif share >= 0.20:
        risk = "medium"
    else:
        risk = "low"

    return {
        "event_risk": risk,
        "large_spread_jumps": n_jumps,
        f"jump_with_{step_key}_step": coincident,
        f"jump_{step_key}_step_share": round(share, 3),
    }


def backtest_pair(
    panel: pd.DataFrame, close: pd.DataFrame, a: str, b: str, metric: str = "pe"
) -> dict:
    """
    Illustrative single-pair backtest (review CSVs only).

    Portfolio capital sizing is NOT applied here — see portfolio_backtest.py
    simulate_sector_portfolio (equal dynamic capital among active trades).

    Dollar-neutral relative-multiple pair backtest on prices:
      enter when |z| >= z_entry
        z>0: short A / long B  (A rich vs B)
        z<0: long A / short B
      exit when |z| <= z_exit
    PnL uses equal-dollar legs on daily close-to-close returns.
    Uses expanding mean/std for z to reduce look-ahead (min 126 days warm-up).
    """
    z_entry = THRESHOLDS["z_entry"]
    z_exit = THRESHOLDS["z_exit"]
    suf = f"_{metric}"

    common = panel[[a, b]].join(close[[a, b]], lsuffix=suf, rsuffix="_px", how="inner")
    common = common.dropna()
    common = common[(common[f"{a}{suf}"] > 0) & (common[f"{b}{suf}"] > 0)]
    if len(common) < 200:
        return {"n_trades": 0, "total_return": np.nan, "avg_hold_days": np.nan, "hit_rate": np.nan}

    spread = np.log(common[f"{a}{suf}"] / common[f"{b}{suf}"])
    mu = spread.expanding(min_periods=126).mean()
    sd = spread.expanding(min_periods=126).std(ddof=1)
    z = (spread - mu) / sd

    ret_a = common[f"{a}_px"].pct_change()
    ret_b = common[f"{b}_px"].pct_change()

    pos_a = 0.0
    pos_b = 0.0
    entry_i = None
    trades = []
    daily_pnl = []

    for i in range(1, len(common)):
        zi = z.iloc[i]
        if not np.isfinite(zi):
            daily_pnl.append(0.0)
            continue

        pnl = 0.5 * pos_a * float(ret_a.iloc[i]) + 0.5 * pos_b * float(ret_b.iloc[i])
        if not np.isfinite(pnl):
            pnl = 0.0
        daily_pnl.append(pnl)

        if pos_a == 0:
            if abs(zi) >= z_entry:
                if zi > 0:
                    pos_a, pos_b = -1.0, 1.0
                else:
                    pos_a, pos_b = 1.0, -1.0
                entry_i = i
        else:
            if abs(zi) <= z_exit:
                hold = i - (entry_i or i)
                trade_ret = float(np.nansum(daily_pnl[entry_i + 1 : i + 1])) if entry_i is not None else 0.0
                trades.append({"hold_days": hold, "return": trade_ret})
                pos_a = pos_b = 0.0
                entry_i = None

    if pos_a != 0 and entry_i is not None:
        hold = len(common) - 1 - entry_i
        trade_ret = float(np.nansum(daily_pnl[entry_i + 1 :]))
        trades.append({"hold_days": hold, "return": trade_ret})

    if not trades:
        return {
            "n_trades": 0,
            "total_return": 0.0,
            "avg_hold_days": np.nan,
            "hit_rate": np.nan,
            "avg_trade_return": np.nan,
            "max_drawdown": np.nan,
        }

    rets = np.array([t["return"] for t in trades], dtype=float)
    holds = np.array([t["hold_days"] for t in trades], dtype=float)
    equity = np.cumsum(daily_pnl)
    peak = np.maximum.accumulate(equity)
    dd = equity - peak
    max_dd = float(dd.min()) if len(dd) else np.nan

    return {
        "n_trades": len(trades),
        "total_return": float(np.nansum(rets)),
        "avg_hold_days": float(np.nanmean(holds)),
        "hit_rate": float(np.mean(rets > 0)),
        "avg_trade_return": float(np.nanmean(rets)),
        "max_drawdown": max_dd,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Relative-PE top-10 selection + review + backtest")
    parser.add_argument("--sector", default=DEFAULT_SECTOR)
    parser.add_argument("--top-n", type=int, default=10)
    args = parser.parse_args()
    metric = "pe"
    label = "PE"

    out_dir = sector_dir(args.sector)
    out_dir.mkdir(parents=True, exist_ok=True)

    panel = pd.read_csv(out_dir / "pe.csv", index_col=0, parse_dates=True)
    close = pd.read_csv(out_dir / "close.csv", index_col=0, parse_dates=True)
    fundamental = pd.read_csv(out_dir / "eps_365.csv", index_col=0, parse_dates=True)
    members = pd.read_csv(out_dir / "members.csv")
    panel.index = pd.to_datetime(panel.index).tz_localize(None).normalize()
    close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
    fundamental.index = pd.to_datetime(fundamental.index).tz_localize(None).normalize()
    sub = load_subindustry_map(members)

    thr = dict(thresholds_for(metric))
    rule_path = out_dir / "selection_thresholds.json"
    rule_path.write_text(
        json.dumps(
            {
                "metric": metric,
                "mode": "full_sample",
                "thresholds": thr,
                "score_weights": SCORE_WEIGHTS,
                "half_life_target": HALF_LIFE_TARGET,
                "sector": args.sector,
                "pair_universe": "same_gics_subindustry",
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Locked thresholds -> {rule_path}")

    scored, top = select_relative_valuation_pairs(
        panel,
        members=members,
        top_n=args.top_n,
        metric=metric,
    )
    scored_path = out_dir / f"relative_{metric}_candidates.csv"
    top_path = out_dir / f"top10_relative_{metric}_pairs.csv"
    scored.to_csv(scored_path, index=False)
    print(f"Candidates passing filters: {len(scored)} -> {scored_path}")

    reviews = []
    for rank, row in top.iterrows():
        a, b = row["symbol_a"], row["symbol_b"]
        compat_label, compat_score, compat_detail = compatibility(a, b, sub)
        risk = fundamental_event_risk(panel, fundamental, a, b, metric=metric)
        reviews.append(
            {
                "rank": rank + 1,
                "symbol_a": a,
                "security_a": row.get("security_a", ""),
                "symbol_b": b,
                "security_b": row.get("security_b", ""),
                "subindustry_a": sub.get(a, ""),
                "subindustry_b": sub.get(b, ""),
                "compat_label": compat_label,
                "compat_score": compat_score,
                "compat_detail": compat_detail,
                **risk,
                "n_days": row["n_days"],
                "pearson_r": row["pearson_r"],
                "adf_t": row["adf_t"],
                "half_life": row["half_life"],
                "exc_ge_1_5": row["exc_ge_1_5"],
                "revert_rate": row["revert_rate"],
                "score": row["score"],
            }
        )

    review_df = pd.DataFrame(reviews)
    if review_df.empty:
        review_df = pd.DataFrame(
            columns=[
                "rank",
                "symbol_a",
                "security_a",
                "symbol_b",
                "security_b",
                "subindustry_a",
                "subindustry_b",
                "compat_label",
                "compat_score",
                "compat_detail",
                "event_risk",
                "n_days",
                "pearson_r",
                "adf_t",
                "half_life",
                "exc_ge_1_5",
                "revert_rate",
                "score",
            ]
        )
    review_path = out_dir / "top10_pair_review.csv"
    review_df.to_csv(review_path, index=False)
    print(f"Review written -> {review_path}")

    bt_rows = []
    for _, row in review_df.iterrows():
        a, b = row["symbol_a"], row["symbol_b"]
        bt = backtest_pair(panel, close, a, b, metric=metric)
        bt_rows.append(
            {
                "rank": row["rank"],
                "symbol_a": a,
                "symbol_b": b,
                "compat_label": row["compat_label"],
                "event_risk": row["event_risk"],
                **bt,
            }
        )
    bt_df = pd.DataFrame(bt_rows)
    if bt_df.empty:
        bt_df = pd.DataFrame(
            columns=[
                "rank",
                "symbol_a",
                "symbol_b",
                "compat_label",
                "event_risk",
                "n_trades",
                "total_return",
                "avg_hold_days",
                "hit_rate",
                "avg_trade_return",
                "max_drawdown",
            ]
        )
    bt_path = out_dir / "top10_pair_backtest.csv"
    bt_df.to_csv(bt_path, index=False)
    print(f"Backtest written -> {bt_path}")

    top_out = review_df.copy()
    if not bt_df.empty and not review_df.empty:
        top_out = top_out.merge(
            bt_df[
                ["symbol_a", "symbol_b", "n_trades", "total_return", "hit_rate", "avg_hold_days", "max_drawdown"]
            ],
            on=["symbol_a", "symbol_b"],
            how="left",
        )
    top_out.to_csv(top_path, index=False)
    print(f"Top {args.top_n} written -> {top_path}")

    print(f"\n=== Top relative-{label} pairs ===")
    if top_out.empty:
        print("(none)")
    else:
        cols = ["rank", "symbol_a", "symbol_b", "compat_label", "event_risk", "pearson_r", "half_life"]
        if "n_trades" in top_out.columns:
            cols += ["n_trades", "total_return", "hit_rate"]
        print(top_out[[c for c in cols if c in top_out.columns]].to_string(index=False))


if __name__ == "__main__":
    main()
