#!/usr/bin/env python3
"""Default live Long study: Index>SMA200 block-new-buys (now main.py default).

Same book as main.py adopted Long (peak 30% stop, equal-weight slots, cash sleeve)
with new month-end buys paused while Nifty 500 is below its SMA200 (lagged 1d).
Existing positions still exit on SMA200 / peak stop as usual.

This gate is the live default via main.adopted_long_kwargs(). This file compares
that default against the ungated baseline (index_sma200_block=False).

Usage:
  python long_sma200_block_main.py                  # Sep-2024 peak → latest, top 750
  python long_sma200_block_main.py --start 2010-01-01
  python long_sma200_block_main.py --top-n 800
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import main as m

OUT = ROOT / "output" / "studies"
TAG = "long_sma200_block_main"


def sma200_block_kwargs(P, bench=None) -> dict:
    """Adopted Long kwargs with live default gate (Nifty 500 > SMA200)."""
    return m.adopted_long_kwargs(
        P, bench=bench, index_sma200_block=True, index_gate="nifty500",
    )


def ungated_long_kwargs(P) -> dict:
    """Former live Long: peak stop + cash sleeve, no index buy gate."""
    return m.adopted_long_kwargs(P, index_sma200_block=False)


def sep_peak_date(bench: pd.Series, year: int = 2024) -> str:
    c = bench.dropna()
    c.index = pd.to_datetime(c.index).tz_localize(None)
    sep = c.loc[f"{year}-09-01" : f"{year}-09-30"]
    if sep.empty:
        raise SystemExit(f"No September {year} bars on Nifty 500")
    return sep.idxmax().strftime("%Y-%m-%d")


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
    row = {
        "Strategy": name,
        **st,
        "Allow new buys %": round(allow, 1),
        "Trades": ts.get("Trades"),
        "Win %": round(ts.get("Win rate %", float("nan")), 1),
        "PF": round(ts.get("Profit factor", float("nan")), 2),
        "Avg hold days": round(ts.get("Avg days held", float("nan")), 0),
        "N500 Total %": round((n500_n.iloc[-1] / n500_n.iloc[0] - 1) * 100, 2),
        "N500 Max DD %": sn["Max DD %"],
        "Excess CAGR pts": round(st["CAGR %"] - sn["CAGR %"], 2),
        "Trough date": str(pd.Timestamp(dd.idxmin()).date()),
        "exit_reasons": reasons,
    }
    return row, eq_n, n500_n, tr


def load_top_n(top_n: int):
    m.set_data_mode("cache")
    data, universe, ranked = m.load_mcap_floor_universe(
        min_mcap_cr=m.MIN_MCAP_CR,
        top_n=top_n,
        force=False,
        cache_only=True,
    )
    return data, universe, ranked


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default=None, help="Default: Sep peak of --peak-year")
    ap.add_argument("--end", default=None)
    ap.add_argument("--peak-year", type=int, default=2024)
    ap.add_argument("--top-n", type=int, default=750)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    print(
        f"=== Main (SMA200 block = default) vs ungated Long ===\n"
        f"top_n={args.top_n} MIN_MCAP_CR={m.MIN_MCAP_CR} slots={m.LONG_POSITIONS}",
        flush=True,
    )

    data, universe, ranked = load_top_n(args.top_n)
    print(f"Loaded {len(data)}/{args.top_n} symbols (ranked≥floor={len(ranked)})", flush=True)

    bench, bname = m._load_nifty500_close(None)
    if bench is None:
        raise SystemExit("Need Nifty 500 benchmark in data_cache")
    start = args.start or sep_peak_date(bench, args.peak_year)
    tickers, P = m.build_panels(data, universe=universe, bench_close=bench)
    sectors = [m.load_sectors(tickers).get(t, "") for t in tickers]
    end = args.end or str(P["Close"].index.max().date())
    tag = args.tag or f"{TAG}_{start.replace('-', '')}_{end.replace('-', '')}"
    print(f"bench={bname} symbols={len(tickers)} {start} → {end}", flush=True)

    # Default main first (baseline for Δ columns)
    variants = [
        ("Main (SMA200 block = default)", sma200_block_kwargs(P, bench)),
        ("Long ungated (no index buy gate)", ungated_long_kwargs(P)),
    ]

    rows, eqs = [], {}
    n500_ref = None
    base = None
    for name, kw in variants:
        print(f"\nRunning {name}…", flush=True)
        row, eq_n, n500_n, tr = _run(name, tickers, P, sectors, bench, start, end, kw)
        if base is None:
            base = row
        row["Δ Total pts vs Main"] = round(row["Total %"] - base["Total %"], 2)
        row["Δ MaxDD pts vs Main"] = round(row["Max DD %"] - base["Max DD %"], 2)
        print(
            f"  Total {row['Total %']}%  CAGR {row['CAGR %']}%  MaxDD {row['Max DD %']}%  "
            f"Sharpe {row['Sharpe']}  allow_buys {row['Allow new buys %']}%  "
            f"trades {row['Trades']}  exits {row['exit_reasons']}",
            flush=True,
        )
        rows.append({k: v for k, v in row.items() if k != "exit_reasons"})
        eqs[name] = eq_n
        n500_ref = n500_n
        safe = "main" if "default" in name else "ungated"
        eq_n.to_csv(OUT / f"{tag}_equity_{safe}.csv", header=["equity"])
        if tr is not None and len(tr):
            tr.to_csv(OUT / f"{tag}_trades_{safe}.csv", index=False)

    headline = pd.DataFrame(rows)
    headline.to_csv(OUT / f"{tag}_compare.csv", index=False)
    n500_ref.to_csv(OUT / f"{tag}_equity_nifty500.csv", header=["equity"])

    def _month_last(eq: pd.Series) -> pd.Series:
        try:
            return eq.resample("ME").last()
        except ValueError:
            return eq.resample("M").last()

    months = sorted(
        set().union(*[set(_month_last(eq).dropna().index) for eq in eqs.values()])
    )
    monthly = pd.DataFrame({"Month": [d.strftime("%Y-%m") for d in months]})
    for name, eq in eqs.items():
        r = _month_last(eq).pct_change() * 100
        monthly[name] = [
            float(r[d]) if d in r.index and pd.notna(r[d]) else np.nan for d in months
        ]
    n_m = _month_last(n500_ref).pct_change() * 100
    monthly["Nifty 500"] = [
        float(n_m[d]) if d in n_m.index and pd.notna(n_m[d]) else np.nan for d in months
    ]
    monthly.to_csv(OUT / f"{tag}_monthly.csv", index=False)

    main_tr = pd.read_csv(OUT / f"{tag}_trades_main.csv")
    ungated_tr = pd.read_csv(OUT / f"{tag}_trades_ungated.csv")
    main_entries = set(
        zip(main_tr["ticker"], pd.to_datetime(main_tr["entry_date"]).astype(str))
    )
    ungated_entries = set(
        zip(ungated_tr["ticker"], pd.to_datetime(ungated_tr["entry_date"]).astype(str))
    )
    only_main = sorted(main_entries - ungated_entries)
    only_ungated = sorted(ungated_entries - main_entries)
    print("\n=== COMPARE ===")
    print(headline.to_string(index=False))
    print(f"\nEntries only in MAIN / SMA200-block ({len(only_main)}):")
    for t, d in only_main:
        print(f"  {d}  {t}")
    print(f"\nEntries only in UNGATED ({len(only_ungated)}):")
    for t, d in only_ungated:
        print(f"  {d}  {t}")
    print(f"\nWrote {tag}_* under {OUT}")
    print(f"Default strategy: adopted_long_kwargs() in main.py / {Path(__file__).name}")


if __name__ == "__main__":
    main()
