#!/usr/bin/env python3
"""Long + Breadth≥65%×3d: block new buys (holdings kept).

Same live Long book as main.py (peak 30% stop, equal-weight slots, cash sleeve),
but new month-end buys are paused while market breadth (% of names with
Close > SMA200) is below the threshold for a confirm streak (default 65% × 3d,
lagged 1 session). Existing positions still exit on SMA200 / peak stop as usual.

Also compares vs Main (SMA200 block = default) and the ungated Long baseline.

Usage:
  python long_breadth_block.py                       # Sep-2024 peak → latest, top 750
  python long_breadth_block.py --start 2009-12-31 --end 2018-01-23
  python long_breadth_block.py --threshold 70 --confirm 3
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
from long_sma200_block_main import sma200_block_kwargs, ungated_long_kwargs

OUT = ROOT / "output" / "studies"
TAG = "long_breadth_block"

DEFAULT_THRESHOLD = 65  # % of names above SMA200 required to allow new buys
DEFAULT_CONFIRM = 3     # consecutive sessions at/above threshold


def pct_above_sma200(P, member_mask=None) -> pd.Series:
    """Fraction of names with Close > SMA200 (NaNs excluded). Optional PIT mask."""
    c, s = P["Close"], P["sma200"]
    valid = c.notna() & s.notna()
    if member_mask is not None:
        valid = valid & member_mask.reindex(index=c.index, columns=c.columns).fillna(False)
    return (c > s).where(valid).mean(axis=1)


def confirm_days(raw: pd.Series, n: int) -> pd.Series:
    """True only after n consecutive True sessions (n<=1 → passthrough)."""
    if n is None or int(n) <= 1:
        return raw.fillna(False).astype(bool)
    x = raw.fillna(False).astype(int)
    return x.rolling(int(n), min_periods=int(n)).sum().eq(int(n))


def breadth_reentry_ok(
    pct_above: pd.Series,
    threshold_pct: float = DEFAULT_THRESHOLD,
    lag: int = 1,
    confirm: int = DEFAULT_CONFIRM,
) -> np.ndarray:
    """True = allow new entries. Optional N-day confirm; then lag (no same-bar look-ahead)."""
    thr = float(threshold_pct) / 100.0
    raw = pct_above >= thr
    sig = confirm_days(raw, confirm)
    return sig.shift(lag).fillna(False).to_numpy(dtype=bool)


def breadth_block_kwargs(
    P,
    threshold_pct: float = DEFAULT_THRESHOLD,
    confirm: int = DEFAULT_CONFIRM,
    member_mask=None,
) -> dict:
    """Adopted Long kwargs + reentry_ok when breadth ≥ threshold × confirm (lag 1)."""
    kw = m.adopted_long_kwargs(P)
    pct = pct_above_sma200(P, member_mask=member_mask)
    kw["reentry_ok"] = breadth_reentry_ok(
        pct, threshold_pct=threshold_pct, lag=1, confirm=confirm
    )
    return kw


def sep_peak_date(bench: pd.Series, year: int = 2024) -> str:
    c = bench.dropna()
    c.index = pd.to_datetime(c.index).tz_localize(None)
    sep = c.loc[f"{year}-09-01" : f"{year}-09-30"]
    if sep.empty:
        raise SystemExit(f"No September {year} bars on Nifty 500")
    return sep.idxmax().strftime("%Y-%m-%d")


def _run(name, tickers, P, sectors, bench, start, end, kw):
    kw = dict(kw)
    # drop any reporting-only keys if present
    for k in ("_breadth_pct_above", "_breadth_threshold", "_breadth_confirm"):
        kw.pop(k, None)
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


def _safe_tag(name: str) -> str:
    low = name.lower()
    if "default" in low or ("sma200" in low and "ungated" not in low):
        return "main"
    if "ungated" in low:
        return "ungated"
    if "breadth" in low:
        return "breadth_block"
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)[:48]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default=None, help="Default: Sep peak of --peak-year")
    ap.add_argument("--end", default=None)
    ap.add_argument("--peak-year", type=int, default=2024)
    ap.add_argument("--top-n", type=int, default=750)
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                    help="%% of names above SMA200 to allow new buys (default 65)")
    ap.add_argument("--confirm", type=int, default=DEFAULT_CONFIRM,
                    help="Consecutive sessions at/above threshold (default 3)")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    print(
        f"=== Main (SMA200) vs ungated vs Breadth≥{args.threshold:g}%×{args.confirm}d ===\n"
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

    # Main default = SMA200 block; also show ungated baseline + breadth gate
    main_kw = sma200_block_kwargs(P, bench)
    ungated_kw = ungated_long_kwargs(P)
    br_kw = breadth_block_kwargs(
        P, threshold_pct=args.threshold, confirm=args.confirm
    )

    b_label = f"Long + Breadth≥{args.threshold:g}%×{args.confirm}d: block new buys"
    variants = [
        ("Main (SMA200 block = default)", main_kw),
        ("Long ungated (no index buy gate)", ungated_kw),
        (b_label, br_kw),
    ]

    rows, eqs = [], {}
    n500_ref = None
    base = None
    for name, kw in variants:
        print(f"\nRunning {name}…", flush=True)
        row, eq_n, n500_n, tr = _run(name, tickers, P, sectors, bench, start, end, kw)
        if base is None:
            base = row
        row["Δ Total pts vs Live"] = round(row["Total %"] - base["Total %"], 2)
        row["Δ MaxDD pts vs Live"] = round(row["Max DD %"] - base["Max DD %"], 2)
        print(
            f"  Total {row['Total %']}%  CAGR {row['CAGR %']}%  MaxDD {row['Max DD %']}%  "
            f"Sharpe {row['Sharpe']}  allow_buys {row['Allow new buys %']}%  "
            f"trades {row['Trades']}  exits {row['exit_reasons']}",
            flush=True,
        )
        rows.append({k: v for k, v in row.items() if k != "exit_reasons"})
        eqs[name] = eq_n
        n500_ref = n500_n
        safe = _safe_tag(name)
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

    print("\n=== COMPARE (3 strategies) ===")
    cols = [
        "Strategy", "Total %", "CAGR %", "Max DD %", "Sharpe",
        "Allow new buys %", "Trades", "Win %", "PF",
        "N500 Total %", "Excess CAGR pts",
        "Δ Total pts vs Live", "Δ MaxDD pts vs Live", "Trough date",
    ]
    show = [c for c in cols if c in headline.columns]
    print(headline[show].to_string(index=False))
    print(f"\nWrote {tag}_* under {OUT}")
    print(f"Helpers: breadth_block_kwargs() / sma200_block_kwargs()")


if __name__ == "__main__":
    main()
