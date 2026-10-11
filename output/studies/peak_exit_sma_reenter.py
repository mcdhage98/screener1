#!/usr/bin/env python3
"""Counterfactual: Long from 2024 → exit all at Sep peak → cash → re-enter on SMA200.

Story
-----
1. Trade live Long from --start (default 2024-01-01) until the September peak.
2. Force 100% cash at the peak (foresight exit).
3. Stay in cash until Nifty 500 has gone below SMA200 and then recovered above it.
4. After that, re-enter with the same rules as long_sma200_block_main.py (live default)
   (new buys only while index > SMA200; holdings kept otherwise).

Compared to:
  - Live Long the whole way (no foresight)
  - Main / long_sma200_block_main the whole way (no foresight flatten)
  - Peak exit then immediately allow long_sma buys (may rebuy next month-end)

Usage:
  python output/studies/peak_exit_sma_reenter.py
  python output/studies/peak_exit_sma_reenter.py --start 2024-01-01 --top-n 750
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import main as m
from long_sma200_block_main import sma200_block_kwargs, sep_peak_date, ungated_long_kwargs

OUT = ROOT / "output" / "studies"
TAG = "peak_exit_sma_reenter"


def peak_exit_then_sma_reenter_kwargs(P, bench, peak: str, *, wait_for_dip: bool):
    """Force flatten at peak; then long_sma re-entry rules.

    wait_for_dip=True: stay cash from peak until index has been below SMA200
    at least once after the peak and is back above (recommended reading of the story).
    wait_for_dip=False: after peak day flatten, allow buys whenever index > SMA200
    (may redeploy at the next month-end if still above).
    """
    kw = m.adopted_long_kwargs(P)
    dates = P["Close"].index
    peak_ts = pd.Timestamp(peak)
    below = np.asarray(m.index_below_sma200(bench, dates, lag=1), dtype=bool)
    above = np.logical_not(below)
    post = dates >= peak_ts

    if wait_for_dip:
        # Once we've observed below after peak, allow buys only when above again.
        seen_below = (pd.Series(below & post, index=dates).cumsum() > 0).to_numpy()
        # Cash window: from peak until (seen_below and above)
        in_cash = post & ~(seen_below & above)
        crash_off = in_cash
        reentry_ok = ~in_cash & above
    else:
        # Flatten on peak session only; thereafter block buys when below SMA200.
        crash_off = post & (dates <= peak_ts)
        # ensure sell executes: keep crash_off for peak day; next opens clear
        reentry_ok = above
        # still block buys before peak? no — normal before peak
        reentry_ok = np.where(post, above, True)

    kw["crash_off"] = np.asarray(crash_off, dtype=bool)
    kw["crash_scale"] = 1.0
    kw["reentry_ok"] = np.asarray(reentry_ok, dtype=bool)
    return kw


def _run(name, tickers, P, sectors, bench, start, end, kw):
    kw = dict(kw)
    kw["start"] = start
    kw["end"] = end
    eq, tr = m.backtest(tickers, P, sectors=sectors, **kw)
    eq = eq.dropna().loc[start:end]
    n500 = bench.reindex(eq.index).ffill().dropna()
    common = eq.index.intersection(n500.index)
    eq_n = eq.loc[common] / eq.loc[common].iloc[0] * m.INITIAL_CAPITAL
    n500_n = n500.loc[common] / n500.loc[common].iloc[0] * m.INITIAL_CAPITAL
    st = m._period_stats(eq_n, name)
    sn = m._period_stats(n500_n, "Nifty 500")
    if tr is not None and len(tr) and "entry_date" in tr.columns:
        tr = tr.copy()
        tr["entry_date"] = pd.to_datetime(tr["entry_date"])
        tr = tr[(tr["entry_date"] >= start) & (tr["entry_date"] <= end)]
    ts = m.trade_stats(tr) if tr is not None and len(tr) else {}
    allow = 100.0
    if "reentry_ok" in kw:
        re = pd.Series(np.asarray(kw["reentry_ok"], dtype=bool), index=P["Close"].index)
        allow = float(re.loc[start:end].mean() * 100)
    reasons = {}
    if tr is not None and len(tr) and "reason" in tr.columns:
        reasons = tr["reason"].value_counts().to_dict()
    dd = eq_n / eq_n.cummax() - 1.0

    # leg stats: pre-peak / post-peak
    peak = None
    for k, v in kw.items():
        pass
    row = {
        "Strategy": name,
        **st,
        "Allow new buys %": round(allow, 1),
        "Trades": ts.get("Trades"),
        "Win %": round(ts.get("Win rate %", float("nan")), 1),
        "PF": round(ts.get("Profit factor", float("nan")), 2),
        "N500 Total %": round((n500_n.iloc[-1] / n500_n.iloc[0] - 1) * 100, 2),
        "N500 Max DD %": sn["Max DD %"],
        "Excess CAGR pts": round(st["CAGR %"] - sn["CAGR %"], 2),
        "Trough date": str(pd.Timestamp(dd.idxmin()).date()),
        "exit_reasons": reasons,
    }
    return row, eq_n, n500_n, tr


def leg_return(eq: pd.Series, a: str, b: str) -> float:
    s = eq.loc[a:b].dropna()
    if len(s) < 2:
        return float("nan")
    return float(s.iloc[-1] / s.iloc[0] - 1) * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--top-n", type=int, default=750)
    ap.add_argument("--peak-year", type=int, default=2024)
    args = ap.parse_args()

    m.set_data_mode("cache")
    OUT.mkdir(parents=True, exist_ok=True)

    data, universe, ranked = m.load_mcap_floor_universe(
        min_mcap_cr=m.MIN_MCAP_CR, top_n=args.top_n, force=False, cache_only=True
    )
    bench, bname = m._load_nifty500_close(None)
    peak = sep_peak_date(bench, args.peak_year)
    tickers, P = m.build_panels(data, universe=universe, bench_close=bench)
    sectors = [m.load_sectors(tickers).get(t, "") for t in tickers]
    end = args.end or str(P["Close"].index.max().date())
    start = args.start
    print(
        f"=== Peak-exit + SMA re-enter | top {args.top_n} | {start} → {end} ===\n"
        f"peak flatten date={peak} bench={bname} symbols={len(tickers)}",
        flush=True,
    )

    # Diagnose cash window for wait_for_dip
    dates = P["Close"].index
    below = np.asarray(m.index_below_sma200(bench, dates, lag=1), dtype=bool)
    post = dates >= pd.Timestamp(peak)
    seen = (pd.Series(below & post, index=dates).cumsum() > 0).to_numpy()
    above = ~below
    in_cash = post & ~(seen & above)
    cash_days = dates[in_cash]
    if len(cash_days):
        print(
            f"Foresight cash window (wait-for-dip): {cash_days[0].date()} → {cash_days[-1].date()} "
            f"({len(cash_days)} sessions)",
            flush=True,
        )
        reenter_from = dates[seen & above & post]
        if len(reenter_from):
            print(f"First SMA re-entry allowed: {reenter_from[0].date()}", flush=True)

    variants = [
        ("Main / SMA200 block (no foresight)", sma200_block_kwargs(P, bench)),
        ("Long ungated (no foresight)", ungated_long_kwargs(P)),
        (
            "Peak-exit Sep → long_sma (may rebuy next month)",
            peak_exit_then_sma_reenter_kwargs(P, bench, peak, wait_for_dip=False),
        ),
        (
            "Peak-exit Sep → cash until SMA recovery → long_sma",
            peak_exit_then_sma_reenter_kwargs(P, bench, peak, wait_for_dip=True),
        ),
    ]

    rows, eqs = [], {}
    n500_ref = None
    for name, kw in variants:
        print(f"\nRunning {name}…", flush=True)
        row, eq_n, n500_n, tr = _run(name, tickers, P, sectors, bench, start, end, kw)
        # legs
        row["Pre-peak Total %"] = round(leg_return(eq_n, start, peak), 2)
        row["Post-peak Total %"] = round(leg_return(eq_n, peak, end), 2)
        print(
            f"  Total {row['Total %']}%  (pre-peak {row['Pre-peak Total %']}% / "
            f"post-peak {row['Post-peak Total %']}%)  "
            f"CAGR {row['CAGR %']}%  MaxDD {row['Max DD %']}%  "
            f"trades {row['Trades']}  exits {row['exit_reasons']}",
            flush=True,
        )
        rows.append({k: v for k, v in row.items() if k != "exit_reasons"})
        eqs[name] = eq_n
        n500_ref = n500_n
        safe = (
            name.replace(" ", "_")
            .replace("→", "to")
            .replace(",", "")
            .replace("(", "")
            .replace(")", "")
            .replace("/", "_")
        )
        eq_n.to_csv(OUT / f"{TAG}_equity_{safe}.csv", header=["equity"])
        if tr is not None and len(tr):
            tr.to_csv(OUT / f"{TAG}_trades_{safe}.csv", index=False)

    headline = pd.DataFrame(rows).sort_values("Total %", ascending=False)
    headline.to_csv(OUT / f"{TAG}_compare.csv", index=False)
    n500_ref.to_csv(OUT / f"{TAG}_equity_nifty500.csv", header=["equity"])

    # Equity at peak / end for the foresight path
    key = "Peak-exit Sep → cash until SMA recovery → long_sma"
    eq = eqs[key]
    print("\n=== FORESIGHT PATH DETAIL ===")
    print(f"Start equity {eq.iloc[0]:,.0f}")
    print(f"At peak {peak}: {eq.loc[:peak].iloc[-1]:,.0f}  "
          f"(+{(eq.loc[:peak].iloc[-1]/eq.iloc[0]-1)*100:.1f}% from start)")
    print(f"Final {end}: {eq.iloc[-1]:,.0f}  "
          f"(+{(eq.iloc[-1]/eq.iloc[0]-1)*100:.1f}% from start)")
    print(f"From peak→end: {(eq.iloc[-1]/eq.loc[:peak].iloc[-1]-1)*100:.1f}%")

    print("\n=== COMPARE ===")
    show = [
        "Strategy", "Total %", "Pre-peak Total %", "Post-peak Total %",
        "CAGR %", "Max DD %", "Sharpe", "Calmar", "Trades", "Win %",
        "N500 Total %", "Allow new buys %", "Trough date",
    ]
    print(headline[show].to_string(index=False))
    print(f"\nWrote {TAG}_* under {OUT}")


if __name__ == "__main__":
    main()
