#!/usr/bin/env python3
"""Sep-2024-peak analysis using main.py live Long book on Yahoo top-750 by mcap.

Universe: top 750 NSE names by today's Yahoo mcap among ≥ MIN_MCAP_CR (main.py).
Prices: data_cache Yahoo only (cache mode).
Window: Nifty 500 September peak → latest close.

Usage:
  python output/studies/main_yahoo_top750_seppeak.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import main as m

OUT = ROOT / "output" / "studies"
TAG = "main_yahoo_top750_seppeak"
TOP_N = 750
PEAK_YEAR = 2024


def sep_peak_date(bench: pd.Series, year: int = PEAK_YEAR) -> str:
    c = bench.dropna()
    c.index = pd.to_datetime(c.index).tz_localize(None)
    sep = c.loc[f"{year}-09-01" : f"{year}-09-30"]
    if sep.empty:
        raise SystemExit(f"No September {year} bars on Nifty 500")
    return sep.idxmax().strftime("%Y-%m-%d")


def run_one(name, tickers, P, sectors, bench, start, end, kw):
    kw = dict(kw)
    kw["start"] = start
    kw["end"] = end
    eq, tr = m.backtest(tickers, P, sectors=sectors, **kw)
    eq = eq.dropna().loc[start:end]
    if len(eq) < 2:
        return None, None, None, None
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
    reasons = {}
    if tr is not None and len(tr) and "reason" in tr.columns:
        reasons = tr["reason"].value_counts().to_dict()
    dd = eq_n / eq_n.cummax() - 1.0
    row = {
        "Strategy": name,
        **st,
        "Trades": ts.get("Trades"),
        "Win %": round(ts.get("Win rate %", float("nan")), 1),
        "PF": round(ts.get("Profit factor", float("nan")), 2),
        "Avg hold days": round(ts.get("Avg days held", float("nan")), 0),
        "N500 Total %": round((n500_n.iloc[-1] / n500_n.iloc[0] - 1) * 100, 2),
        "N500 CAGR %": sn["CAGR %"],
        "N500 Max DD %": sn["Max DD %"],
        "Excess CAGR pts": round(st["CAGR %"] - sn["CAGR %"], 2),
        "Trough date": str(pd.Timestamp(dd.idxmin()).date()),
        "exit_reasons": reasons,
    }
    return row, eq_n, n500_n, tr


def main():
    m.set_data_mode("cache")
    OUT.mkdir(parents=True, exist_ok=True)
    print(
        f"=== main.py Yahoo top-{TOP_N} | Sep {PEAK_YEAR} peak → latest ===\n"
        f"MIN_MCAP_CR={m.MIN_MCAP_CR} LIVE_UNIVERSE_TOP_N(main)={m.LIVE_UNIVERSE_TOP_N} "
        f"slots={m.LONG_POSITIONS} sector_cap={m.SECTOR_CAP}",
        flush=True,
    )

    data, universe, ranked = m.load_mcap_floor_universe(
        min_mcap_cr=m.MIN_MCAP_CR,
        top_n=TOP_N,
        force=False,
        cache_only=True,
    )
    print(f"Loaded {len(data)} symbols (ranked table {len(ranked)})", flush=True)

    bench, bname = m._load_nifty500_close(None)
    if bench is None:
        raise SystemExit("Need Nifty 500 in data_cache")
    start = sep_peak_date(bench, PEAK_YEAR)
    end = str(pd.to_datetime(bench.dropna().index).max().date())
    # Prefer panel end if later/earlier
    print(f"bench={bname} start={start} (Sep peak) end={end}", flush=True)

    tickers, P = m.build_panels(data, universe=universe, bench_close=bench)
    sectors = [m.load_sectors(tickers).get(t, "") for t in tickers]
    end = str(P["Close"].index.max().date())
    print(f"Panel symbols={len(tickers)} end={end}", flush=True)

    m.augment_long_entries(P, sectors)
    dates = P["Close"].index
    below200 = m.index_below_sma200(bench, dates, lag=1)
    choppy = m.index_choppy(bench, dates, lag=1)
    bad = below200 & choppy
    risk_on = np.logical_not(below200)

    breadth_on = m.breadth_regime(P)
    index_on = m.index_crash_regime(bench, dates)
    combined_on = breadth_on & index_on

    # Cast gates to bool ndarray (avoid ~ on 0/1 ints)
    breadth_on = np.asarray(breadth_on, dtype=bool)
    index_on = np.asarray(index_on, dtype=bool)
    combined_on = np.asarray(combined_on, dtype=bool)
    below200 = np.asarray(below200, dtype=bool)
    bad = np.asarray(bad, dtype=bool)
    risk_on = np.asarray(risk_on, dtype=bool)

    adopted = m.adopted_long_kwargs(P)
    peak_only = dict(m.LONG_BOOK)
    legacy = dict(m.LONG_BOOK_LEGACY)

    variants = [
        ("Long (adopted = live)", adopted),
        ("Long (peak stop, no cash sleeve)", peak_only),
        ("Long (no peak stop)", legacy),
        ("Block new buys: index < SMA200", {**adopted, "reentry_ok": risk_on}),
        (
            "Flatten: index < SMA200",
            {
                **adopted,
                "reentry_ok": risk_on,
                "crash_off": below200,
                "crash_scale": 1.0,
            },
        ),
        (
            "Flatten: index < SMA200 + choppy",
            {
                **adopted,
                "reentry_ok": np.logical_not(bad),
                "crash_off": bad,
                "crash_scale": 1.0,
            },
        ),
        (
            "Index crash regime: flatten + re-enter",
            {
                **adopted,
                "reentry_ok": index_on,
                "crash_off": np.logical_not(index_on),
                "crash_scale": 1.0,
            },
        ),
        (
            "Breadth crash regime: flatten + re-enter",
            {
                **adopted,
                "reentry_ok": breadth_on,
                "crash_off": np.logical_not(breadth_on),
                "crash_scale": 1.0,
            },
        ),
        (
            "Combined crash (breadth & index): flatten",
            {
                **adopted,
                "reentry_ok": combined_on,
                "crash_off": np.logical_not(combined_on),
                "crash_scale": 1.0,
            },
        ),
        ("Block new buys: breadth risk-off", {**adopted, "reentry_ok": breadth_on}),
        ("Entry: industry momentum top half", {**adopted, "entry_col": "entry_long_ind"}),
        ("Rank: roc_12_1", {**adopted, "rank_col": "roc_12_1"}),
        ("Rank: prox", {**adopted, "rank_col": "prox"}),
        ("No cash sleeve (peak stop kept)", {**adopted, "cash_ret": None}),
    ]
    if "entry_long_rs" in P:
        variants.append(
            ("Entry: RS vs N500", {**adopted, "entry_col": "entry_long_rs"})
        )
    if "mom_adj" in P:
        variants.append(("Rank: mom_adj", {**adopted, "rank_col": "mom_adj"}))

    rows, eqs = [], {}
    n500_ref = None
    base = None
    for name, kw in variants:
        print(f"\nRunning {name}…", flush=True)
        row, eq_n, n500_n, tr = run_one(name, tickers, P, sectors, bench, start, end, kw)
        if row is None:
            print("  skipped (no equity)")
            continue
        if base is None:
            base = row
        row["Δ Total pts vs Adopted"] = round(row["Total %"] - base["Total %"], 2)
        row["Δ MaxDD pts vs Adopted"] = round(row["Max DD %"] - base["Max DD %"], 2)
        print(
            f"  Total {row['Total %']}%  CAGR {row['CAGR %']}%  MaxDD {row['Max DD %']}%  "
            f"Sharpe {row['Sharpe']}  trades {row['Trades']}  "
            f"exits {row['exit_reasons']}",
            flush=True,
        )
        rows.append({k: v for k, v in row.items() if k != "exit_reasons"})
        eqs[name] = eq_n
        n500_ref = n500_n
        safe = (
            name.replace(" ", "_")
            .replace("<", "lt")
            .replace(">", "gt")
            .replace("&", "and")
            .replace(":", "")
            .replace("+", "plus")
            .replace("(", "")
            .replace(")", "")
            .replace("/", "_")
            .replace("=", "")
        )
        eq_n.to_csv(OUT / f"{TAG}_equity_{safe}.csv", header=["equity"])
        if tr is not None and len(tr):
            tr.to_csv(OUT / f"{TAG}_trades_{safe}.csv", index=False)

    headline = pd.DataFrame(rows).sort_values("Calmar", ascending=False)
    headline.to_csv(OUT / f"{TAG}_headline.csv", index=False)
    n500_ref.to_csv(OUT / f"{TAG}_equity_nifty500.csv", header=["equity"])

    # Universe snapshot
    uni = universe.reindex(tickers).copy()
    uni["ticker"] = uni.index
    uni = uni.sort_values("mcap_cr", ascending=False)
    uni[["mcap_cr"]].to_csv(OUT / f"{TAG}_universe.csv")

    meta = {
        "top_n": TOP_N,
        "min_mcap_cr": m.MIN_MCAP_CR,
        "symbols": len(tickers),
        "start": start,
        "end": end,
        "bench": bname,
        "long_positions": m.LONG_POSITIONS,
        "sector_cap": m.SECTOR_CAP,
        "n500_total_pct": round((n500_ref.iloc[-1] / n500_ref.iloc[0] - 1) * 100, 2),
    }
    pd.Series(meta).to_json(OUT / f"{TAG}_meta.json")

    print("\n=== HEADLINE (by Calmar) ===")
    show = [
        "Strategy",
        "Total %",
        "CAGR %",
        "Max DD %",
        "Sharpe",
        "Calmar",
        "Trades",
        "Win %",
        "N500 Total %",
        "Δ Total pts vs Adopted",
        "Δ MaxDD pts vs Adopted",
        "Trough date",
    ]
    print(headline[show].to_string(index=False))
    print(f"\nWrote {TAG}_* under {OUT}")


if __name__ == "__main__":
    main()
