#!/usr/bin/env python3
"""$100k relative-PE portfolio backtest vs S&P 500 Health Care; Excel deliverable."""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from openpyxl import Workbook
from openpyxl.chart import LineChart, Reference
from openpyxl.drawing.image import Image as XLImage
from openpyxl.styles import Font
from openpyxl.utils.dataframe import dataframe_to_rows

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import sector_dir
from src.pair_select import THRESHOLDS


INITIAL_CAPITAL = 100_000.0
SCORE_MIN = 0.5
WARMUP = 126
BENCH_LABEL = "S&P 500 Health Care"
BENCH_PRIMARY = "^SP500-35"
BENCH_FALLBACK = "XLV"


def load_eligible_pairs(cand_path: Path) -> pd.DataFrame:
    cand = pd.read_csv(cand_path)
    elig = cand.loc[cand["score"] > SCORE_MIN].copy()
    elig = elig.sort_values("score", ascending=False).reset_index(drop=True)
    return elig


def generate_pair_trades(
    pe: pd.DataFrame,
    close: pd.DataFrame,
    a: str,
    b: str,
    start: pd.Timestamp,
    slot_notional: float,
) -> tuple[list[dict], pd.Series]:
    """Expanding-z pair trades from `start`."""
    z_entry = THRESHOLDS["z_entry"]
    z_exit = THRESHOLDS["z_exit"]

    common = pe[[a, b]].join(close[[a, b]], lsuffix="_pe", rsuffix="_px", how="inner")
    common = common.dropna()
    common = common[(common[f"{a}_pe"] > 0) & (common[f"{b}_pe"] > 0)]
    if common.empty:
        return [], pd.Series(dtype=float)

    spread = np.log(common[f"{a}_pe"] / common[f"{b}_pe"])
    mu = spread.expanding(min_periods=WARMUP).mean()
    sd = spread.expanding(min_periods=WARMUP).std(ddof=1)
    z = (spread - mu) / sd

    ret_a = common[f"{a}_px"].pct_change()
    ret_b = common[f"{b}_px"].pct_change()

    daily_pnl = pd.Series(0.0, index=common.index, dtype=float)
    trades: list[dict] = []

    pos_a = 0.0
    pos_b = 0.0

    dates = common.index
    for i in range(1, len(dates)):
        dt = dates[i]
        zi = z.iloc[i]
        ra = float(ret_a.iloc[i]) if np.isfinite(ret_a.iloc[i]) else 0.0
        rb = float(ret_b.iloc[i]) if np.isfinite(ret_b.iloc[i]) else 0.0

        if pos_a != 0:
            day_ret = 0.5 * pos_a * ra + 0.5 * pos_b * rb
            daily_pnl.iloc[i] = slot_notional * day_ret

        if not np.isfinite(zi):
            continue

        if pos_a == 0:
            if dt >= start and abs(zi) >= z_entry:
                if zi > 0:
                    pos_a, pos_b = -1.0, 1.0
                    direction = f"short {a} / long {b}"
                else:
                    pos_a, pos_b = 1.0, -1.0
                    direction = f"long {a} / short {b}"
                trades.append(
                    {
                        "symbol_a": a,
                        "symbol_b": b,
                        "pair": f"{a}-{b}",
                        "open_date": dt,
                        "close_date": pd.NaT,
                        "direction": direction,
                        "entry_z": float(zi),
                        "exit_z": np.nan,
                        "hold_days": np.nan,
                        "slot_notional": slot_notional,
                        "pnl_usd": np.nan,
                        "return": np.nan,
                        "_entry_i": i,
                        "_open": True,
                    }
                )
        else:
            if abs(zi) <= z_exit:
                for t in reversed(trades):
                    if t.get("_open") and t["symbol_a"] == a and t["symbol_b"] == b:
                        ei = t["_entry_i"]
                        pnl = float(daily_pnl.iloc[ei + 1 : i + 1].sum())
                        t["close_date"] = dt
                        t["exit_z"] = float(zi)
                        t["hold_days"] = i - ei
                        t["pnl_usd"] = pnl
                        t["return"] = pnl / slot_notional if slot_notional else np.nan
                        t["_open"] = False
                        break
                pos_a = pos_b = 0.0

    if pos_a != 0:
        last_i = len(dates) - 1
        last_dt = dates[last_i]
        for t in reversed(trades):
            if t.get("_open") and t["symbol_a"] == a and t["symbol_b"] == b:
                ei = t["_entry_i"]
                pnl = float(daily_pnl.iloc[ei + 1 :].sum())
                t["close_date"] = last_dt
                t["exit_z"] = float(z.iloc[last_i]) if np.isfinite(z.iloc[last_i]) else np.nan
                t["hold_days"] = last_i - ei
                t["pnl_usd"] = pnl
                t["return"] = pnl / slot_notional if slot_notional else np.nan
                t["_open"] = False
                break

    daily_pnl = daily_pnl.loc[daily_pnl.index >= start]
    clean_trades = []
    for t in trades:
        if t.get("_open"):
            continue
        clean_trades.append({k: v for k, v in t.items() if not k.startswith("_")})

    return clean_trades, daily_pnl


def _download_close(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    raw = yf.download(
        ticker,
        start=(start - pd.Timedelta(days=5)).strftime("%Y-%m-%d"),
        end=(end + pd.Timedelta(days=5)).strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
    )
    if raw is None or raw.empty:
        raise RuntimeError(f"empty download for {ticker}")
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
        if ticker in close.columns:
            px = close[ticker]
        else:
            px = close.iloc[:, 0]
    else:
        px = raw["Close"]
    px = px.copy()
    px.index = pd.to_datetime(px.index).tz_localize(None).normalize()
    px = px.sort_index().dropna()
    if px.empty:
        raise RuntimeError(f"no close prices for {ticker}")
    return px


def fetch_healthcare_equity(
    start: pd.Timestamp,
    end: pd.Timestamp,
    calendar: pd.DatetimeIndex,
) -> tuple[pd.Series, str]:
    """
    $100k buy-and-hold in S&P 500 Health Care.
    Prefer ^SP500-35; fall back to XLV. Series always labeled as S&P 500 Health Care.
    """
    used = BENCH_PRIMARY
    try:
        px = _download_close(BENCH_PRIMARY, start, end)
        print(f"Benchmark: using {BENCH_PRIMARY}")
    except Exception as exc:
        print(f"Benchmark: {BENCH_PRIMARY} failed ({exc}); falling back to {BENCH_FALLBACK}")
        px = _download_close(BENCH_FALLBACK, start, end)
        used = BENCH_FALLBACK

    aligned = px.reindex(calendar).ffill().bfill()
    base = float(aligned.iloc[0])
    equity = INITIAL_CAPITAL * (aligned / base)
    equity.name = "healthcare_value"
    return equity, used


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min()) if len(dd) else np.nan


def render_performance_history_png(equity_df: pd.DataFrame) -> bytes:
    """Performance History–style cumulative % chart with labeled axes."""
    dates = pd.to_datetime(equity_df["date"])
    strat_pct = equity_df["strategy_return_pct"].to_numpy(dtype=float) * 100.0
    hc_pct = equity_df["healthcare_return_pct"].to_numpy(dtype=float) * 100.0

    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=140)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.plot(dates, strat_pct, color="#2F6FED", linewidth=1.8, label="Relative-PE Strategy")
    ax.plot(dates, hc_pct, color="#2C3E50", linewidth=1.8, label=BENCH_LABEL)

    ax.axhline(0.0, color="#9AA0A6", linewidth=0.9)
    ax.set_title("PERFORMANCE HISTORY", fontsize=14, fontweight="bold", color="#1A73E8", loc="left", pad=12)
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Cumulative return (%)", fontsize=11)

    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1f}%"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator((1, 4, 7, 10)))
    fig.autofmt_xdate(rotation=0, ha="center")

    ax.grid(True, which="major", color="#E6E8EB", linewidth=0.8)
    ax.grid(True, which="minor", color="#F3F4F6", linewidth=0.5)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#CFD4DA")
    ax.spines["bottom"].set_color("#CFD4DA")

    ax.legend(loc="lower right", frameon=True, fancybox=False, edgecolor="#E6E8EB")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def write_excel(
    path: Path,
    summary_rows: list[tuple[str, object]],
    pairs_df: pd.DataFrame,
    equity_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    chart_png: bytes,
) -> None:
    wb = Workbook()

    # --- Summary ---
    ws = wb.active
    ws.title = "Summary"
    ws["A1"] = "Relative-PE Portfolio Backtest"
    ws["A1"].font = Font(bold=True, size=14)
    ws["A2"] = f"Strategy vs {BENCH_LABEL} buy-and-hold"
    row = 4
    for label, value in summary_rows:
        ws.cell(row=row, column=1, value=label).font = Font(bold=True)
        cell = ws.cell(row=row, column=2, value=value)
        if isinstance(value, float) and ("return" in label.lower() or "drawdown" in label.lower()):
            cell.number_format = "0.00%"
        elif isinstance(value, float) and (
            "value" in label.lower() or "capital" in label.lower() or "pnl" in label.lower() or "notional" in label.lower()
        ):
            cell.number_format = "#,##0.00"
        row += 1

    row += 1
    ws.cell(row=row, column=1, value="Eligible pairs (score > 0.5)").font = Font(bold=True)
    row += 1
    headers = list(pairs_df.columns)
    for c, h in enumerate(headers, 1):
        ws.cell(row=row, column=c, value=h).font = Font(bold=True)
    for i, rec in enumerate(pairs_df.itertuples(index=False), 1):
        for c, val in enumerate(rec, 1):
            ws.cell(row=row + i, column=c, value=val)

    ws.column_dimensions["A"].width = 40
    ws.column_dimensions["B"].width = 22

    # Embed Performance History PNG
    img_path = path.parent / "_performance_history_tmp.png"
    img_path.write_bytes(chart_png)
    img = XLImage(str(img_path))
    img.width = 880
    img.height = 440
    ws.add_image(img, "D4")

    # --- Equity ---
    ws_eq = wb.create_sheet("Equity")
    eq = equity_df.copy()
    eq["date"] = pd.to_datetime(eq["date"]).dt.tz_localize(None)
    # Friendly headers for Excel chart
    eq_out = eq.rename(
        columns={
            "strategy_return_pct": "Relative-PE Strategy (%)",
            "healthcare_return_pct": f"{BENCH_LABEL} (%)",
        }
    )
    # Store % as percentage points * 100 for readable chart? Better store as Excel % (0.1 = 10%)
    display = eq_out.copy()
    for r_idx, row_data in enumerate(dataframe_to_rows(display, index=False, header=True), 1):
        for c_idx, val in enumerate(row_data, 1):
            cell = ws_eq.cell(row=r_idx, column=c_idx, value=val)
            if r_idx == 1:
                cell.font = Font(bold=True)
                continue
            col_name = display.columns[c_idx - 1]
            if col_name == "date":
                cell.number_format = "YYYY-MM-DD"
            elif col_name in ("strategy_value", "healthcare_value"):
                cell.number_format = "#,##0.00"
            elif "(%)" in str(col_name) or col_name.endswith("_pct"):
                cell.number_format = "0.00%"

    n_eq = len(display) + 1
    # Interactive % chart on Equity (cols for strategy/healthcare return %)
    # Column order: date, strategy_value, healthcare_value, Relative-PE Strategy (%), S&P 500 Health Care (%)
    chart = LineChart()
    chart.title = "PERFORMANCE HISTORY"
    chart.style = 10
    chart.y_axis.title = "Cumulative return (%)"
    chart.x_axis.title = "Date"
    chart.height = 12
    chart.width = 22
    data = Reference(ws_eq, min_col=4, min_row=1, max_col=5, max_row=n_eq)
    cats = Reference(ws_eq, min_col=1, min_row=2, max_row=n_eq)
    chart.add_data(data, titles_from_data=True)
    chart.set_categories(cats)
    ws_eq.add_chart(chart, "G2")

    for col, width in zip("ABCDE", [14, 16, 18, 24, 26]):
        ws_eq.column_dimensions[col].width = width

    # --- Trades ---
    ws_tr = wb.create_sheet("Trades")
    tr = trades_df.copy()
    if not tr.empty:
        for col in ("open_date", "close_date"):
            if col in tr.columns:
                tr[col] = pd.to_datetime(tr[col]).dt.tz_localize(None)
    for r_idx, row_data in enumerate(dataframe_to_rows(tr, index=False, header=True), 1):
        for c_idx, val in enumerate(row_data, 1):
            cell = ws_tr.cell(row=r_idx, column=c_idx, value=val)
            if r_idx == 1:
                cell.font = Font(bold=True)
                continue
            col_name = tr.columns[c_idx - 1] if c_idx - 1 < len(tr.columns) else ""
            if col_name in ("open_date", "close_date"):
                cell.number_format = "YYYY-MM-DD"
            elif col_name in ("pnl_usd", "slot_notional"):
                cell.number_format = "#,##0.00"
            elif col_name in ("return",):
                cell.number_format = "0.00%"
            elif col_name in ("entry_z", "exit_z"):
                cell.number_format = "0.000"

    for col in ws_tr.columns:
        ws_tr.column_dimensions[col[0].column_letter].width = 16

    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)
    try:
        img_path.unlink(missing_ok=True)
    except Exception:
        pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sector", default="Health Care")
    parser.add_argument("--capital", type=float, default=INITIAL_CAPITAL)
    parser.add_argument("--score-min", type=float, default=SCORE_MIN)
    args = parser.parse_args()

    out_dir = sector_dir(args.sector)
    pe = pd.read_csv(out_dir / "pe.csv", index_col=0, parse_dates=True)
    close = pd.read_csv(out_dir / "close.csv", index_col=0, parse_dates=True)
    pe.index = pd.to_datetime(pe.index).tz_localize(None).normalize()
    close.index = pd.to_datetime(close.index).tz_localize(None).normalize()

    elig = load_eligible_pairs(out_dir / "relative_pe_candidates.csv")
    elig = elig.loc[elig["score"] > args.score_min].copy()
    if elig.empty:
        raise SystemExit("No pairs with score > threshold")

    n_pairs = len(elig)
    slot = args.capital / n_pairs
    start_req = pd.Timestamp("2021-01-01")
    start = max(start_req, close.index.min())
    end = close.index.max()
    calendar = close.loc[close.index >= start].index

    print(f"Eligible pairs: {n_pairs}; slot=${slot:,.2f}; start={start.date()}; end={end.date()}")

    all_trades: list[dict] = []
    slot_pnls: list[pd.Series] = []

    for _, row in elig.iterrows():
        a, b = row["symbol_a"], row["symbol_b"]
        print(f"  Backtesting {a}-{b} (score={row['score']:.3f})...")
        trades, daily = generate_pair_trades(pe, close, a, b, start, slot)
        for t in trades:
            t["score"] = float(row["score"])
        all_trades.extend(trades)
        daily = daily.reindex(calendar).fillna(0.0)
        slot_pnls.append(daily)

    pnl_matrix = pd.concat(slot_pnls, axis=1).fillna(0.0)
    daily_total = pnl_matrix.sum(axis=1)
    strategy = args.capital + daily_total.cumsum()
    strategy.name = "strategy_value"

    healthcare, bench_ticker = fetch_healthcare_equity(start, end, calendar)

    strat_vals = strategy.reindex(calendar).to_numpy(dtype=float)
    hc_vals = healthcare.reindex(calendar).to_numpy(dtype=float)
    equity_df = pd.DataFrame(
        {
            "date": calendar,
            "strategy_value": strat_vals,
            "healthcare_value": hc_vals,
            "strategy_return_pct": strat_vals / args.capital - 1.0,
            "healthcare_return_pct": hc_vals / args.capital - 1.0,
        }
    )

    trades_df = pd.DataFrame(all_trades)
    if not trades_df.empty:
        trades_df = trades_df.sort_values(["open_date", "pair"]).reset_index(drop=True)
        cols = [
            "pair",
            "symbol_a",
            "symbol_b",
            "score",
            "open_date",
            "close_date",
            "direction",
            "entry_z",
            "exit_z",
            "hold_days",
            "slot_notional",
            "pnl_usd",
            "return",
        ]
        trades_df = trades_df[cols]

    strat_final = float(equity_df["strategy_value"].iloc[-1])
    hc_final = float(equity_df["healthcare_value"].iloc[-1])
    strat_ret = strat_final / args.capital - 1.0
    hc_ret = hc_final / args.capital - 1.0
    mdd = max_drawdown(equity_df.set_index("date")["strategy_value"])

    pairs_out = elig[
        ["symbol_a", "symbol_b", "score", "pearson_r", "adf_t", "half_life", "exc_ge_1_5", "revert_rate"]
    ].copy()
    if "security_a" in elig.columns:
        pairs_out.insert(2, "security_a", elig["security_a"])
        pairs_out.insert(3, "security_b", elig["security_b"])

    summary_rows = [
        ("Initial capital (USD)", float(args.capital)),
        ("Score filter", f"> {args.score_min}"),
        ("Number of pairs / slots", n_pairs),
        ("Slot notional (USD)", float(slot)),
        ("Requested start", "2021-01-01"),
        ("Actual start (data)", start.strftime("%Y-%m-%d")),
        ("End date", end.strftime("%Y-%m-%d")),
        ("Entry rule", f"|z| >= {THRESHOLDS['z_entry']}"),
        ("Exit rule", f"|z| <= {THRESHOLDS['z_exit']}"),
        ("Z warm-up (days)", WARMUP),
        ("Benchmark label", BENCH_LABEL),
        ("Benchmark ticker used", bench_ticker),
        ("Final strategy value (USD)", strat_final),
        (f"Final {BENCH_LABEL} value (USD)", hc_final),
        ("Strategy total return", strat_ret),
        (f"{BENCH_LABEL} total return", hc_ret),
        ("Strategy max drawdown", mdd),
        ("Number of trades", int(len(trades_df))),
        ("Total trade PnL (USD)", float(trades_df["pnl_usd"].sum()) if not trades_df.empty else 0.0),
    ]

    chart_png = render_performance_history_png(equity_df)
    # Also save a standalone PNG next to the workbook for easy viewing
    (out_dir / "performance_history.png").write_bytes(chart_png)

    out_path = out_dir / "portfolio_backtest.xlsx"
    write_excel(out_path, summary_rows, pairs_out, equity_df, trades_df, chart_png)
    print(f"\nWrote {out_path}")
    print(f"Also wrote {out_dir / 'performance_history.png'}")
    print(
        f"Strategy: ${strat_final:,.2f} ({strat_ret:.1%}) | "
        f"{BENCH_LABEL} ({bench_ticker}): ${hc_final:,.2f} ({hc_ret:.1%})"
    )
    print(f"Trades: {len(trades_df)} | Max DD: {mdd:.1%}")


if __name__ == "__main__":
    main()
