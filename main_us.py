"""
US Monthly Long book: download + portfolio backtest (Yahoo US stocks, 2010 onwards).

Same Long rules as India live book in main.py / LONG_STRATEGY.md:
  month-end signals, next-open fills, equal-weight 15 slots, SMA200 exit, 30% peak stop,
  idle cash in GLD (above 200dma) else ~T-bill proxy.

Setup:   pip install yfinance pandas numpy matplotlib
Run:     python main_us.py help

Cache:   data_cache_us/
Output:  output/backtest/  (and output/studies/ for mcap sweep)
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

import main as m


# ----------------------------- US helpers ------------------------------------
SKIP_CACHE = {"universe_meta_us", "us_top_by_mcap", "SPY", "GLD", "QQQ"}


def nasdaq_listed_symbols(exclude_etf=True, exclude_test=True):
    """NASDAQ common stocks from nasdaqtrader.com (not NYSE/AMEX otherlisted)."""
    url = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; screener/1.0)"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        text = resp.read().decode("utf-8", errors="ignore")
    lines = text.strip().splitlines()
    header = lines[0].split("|")
    sym_i = header.index("Symbol")
    etf_i = header.index("ETF") if "ETF" in header else None
    ti_i = header.index("Test Issue") if "Test Issue" in header else None
    out = set()
    for line in lines[1:]:
        if line.startswith("File Creation"):
            continue
        parts = line.split("|")
        if len(parts) <= sym_i:
            continue
        if exclude_test and ti_i is not None and ti_i < len(parts) and parts[ti_i].upper() == "Y":
            continue
        if exclude_etf and etf_i is not None and etf_i < len(parts) and parts[etf_i].upper() == "Y":
            continue
        sym = m._yahoo_symbol_us(parts[sym_i])
        if sym and 1 <= len(sym) <= 5 and "-" not in sym:
            out.add(sym)
    return sorted(out)


def cached_us_symbols(through=None):
    """Equity tickers present in data_cache_us (optional alphabetical cap, e.g. through='ROK')."""
    syms = []
    for p in m.US_CACHE_DIR.glob("*.csv"):
        name = p.stem
        if name in SKIP_CACHE or name.startswith("IDX_"):
            continue
        if through is not None and name > through:
            continue
        syms.append(name)
    return sorted(syms)


def load_us_from_cache(symbols=None, nasdaq_only=False, through=None):
    """
    Load price panels + meta from cache.
    symbols=None -> all cached equities (optionally NASDAQ-only / through= ticker).
    """
    m._ensure_output_dirs()
    if symbols is None:
        symbols = cached_us_symbols(through=through)
        if nasdaq_only:
            nd = set(nasdaq_listed_symbols())
            symbols = [s for s in symbols if s in nd]
    else:
        symbols = [m._yahoo_symbol_us(s) for s in symbols]

    print(f"US symbols requested: {len(symbols)}"
          + (" (NASDAQ only)" if nasdaq_only else "")
          + (f" (through {through})" if through else ""))
    meta = m.get_us_universe_meta(symbols, refresh=False)
    data = m.download_many_us(symbols, start=m.US_PRICE_START, force=False, sequential=False)
    meta = meta.reindex(data.keys())
    print(f"Usable price histories: {len(data)}")
    return data, meta


def _us_long_kwargs(dates):
    kw = dict(m.LONG_BOOK)
    sleeve = m.cash_sleeve_returns_us(dates)
    if sleeve is not None:
        kw["cash_ret"] = sleeve
        kw["cash_ret_idle_only"] = False
    return kw


def _bh_equity(close, index, capital=None):
    capital = m.INITIAL_CAPITAL if capital is None else capital
    s = close.reindex(index).ffill().dropna()
    common = index.intersection(s.index)
    s = s.loc[common]
    return s / s.iloc[0] * capital


def _print_perf(label, eq):
    p = m.perf_stats(eq.dropna(), label)
    print(f"{label:24s}  CAGR {p['CAGR %']:6.2f}%  MaxDD {p['Max DD %']:7.2f}%  "
          f"Calmar {p['Calmar']:5.3f}  Sharpe {p['Sharpe']:5.2f}  Final ${p['Final value']:,.0f}")
    return p


def _window_table(series_map):
    windows = [
        ("2010-2018", "2010-01-01", "2018-12-31"),
        ("2019+", "2019-01-01", None),
        ("Full", "2010-01-01", None),
    ]
    print(f"\n{'Window':10s}", end="")
    for name in series_map:
        print(f"  {name[:12]:>12s}", end="")
    print()
    for label, start, end in windows:
        print(f"{label:10s}", end="")
        for s in series_map.values():
            x = s.loc[s.index >= pd.Timestamp(start)]
            if end:
                x = x.loc[: pd.Timestamp(end)]
            x = x.dropna()
            if len(x) < 2:
                print(f"  {'n/a':>12s}", end="")
                continue
            x = x / x.iloc[0] * m.INITIAL_CAPITAL
            st = m.perf_stats(x, "")
            print(f"  {st['CAGR %']:10.1f}%", end="")
        print()


def run_us_backtest(nasdaq_only=False, through=None, compare_ndx=True, tag=None):
    """
    Run Long on cached US names.
      nasdaq_only=True  -> NASDAQ-listed only
      through='ROK'     -> alphabetical cache subset
      compare_ndx=True  -> print QQQ / ^IXIC / SPY alongside
    """
    m._ensure_output_dirs()
    data, meta = load_us_from_cache(nasdaq_only=nasdaq_only, through=through)
    if not data:
        raise SystemExit("No US price data in data_cache_us/. Run: python main_us.py import")

    bench, bname = m._load_us_sp500_close(None)
    ind_kw = dict(
        tradable_mcap_min=m.US_MIN_MCAP_USD,
        turnover_min=m.US_MIN_TURNOVER,
        mcap_in_usd=True,
    )
    print("Building indicator panels...")
    tickers, P = m.build_panels(data, universe=meta, bench_close=bench, indicator_kw=ind_kw)
    sectors = [""] * len(tickers)
    kw = _us_long_kwargs(P["Close"].index)

    print(f"Running Long from {m.US_PRICE_START} | {len(tickers)} names | bench {bname}...")
    eq, tr = m.backtest(tickers, P, sectors=sectors, start=m.US_PRICE_START, **kw)

    if tag is None:
        if nasdaq_only:
            tag = "nasdaq"
        elif through:
            tag = f"through_{through.lower()}"
        else:
            tag = "cache"
    eq_path = m.OUTPUT_BACKTEST / f"equity_curve_us_long_{tag}.csv"
    tr_path = m.OUTPUT_BACKTEST / f"trades_us_long_{tag}.csv"
    eq.to_csv(eq_path, header=["equity"])
    tr.to_csv(tr_path, index=False)

    eqn = eq.dropna()
    print(f"\n=== US Long ({tag}) | {len(tickers)} names | "
          f"mcap ≥ ${m.US_MIN_MCAP_USD/1e6:.0f}M | {eqn.index[0].date()} → {eqn.index[-1].date()} ===")
    _print_perf("Long", eqn)
    t = m.trade_stats(tr)
    print(f"  Trades {t.get('Trades', len(tr))}  Win rate {t.get('Win rate %', float('nan')):.1f}%  "
          f"PF {t.get('Profit factor', float('nan')):.2f}")

    series = {"Long": eqn / eqn.iloc[0] * m.INITIAL_CAPITAL}
    if compare_ndx:
        for ticker, label in (("QQQ", "QQQ"), ("^IXIC", "NASDAQ Comp"), ("SPY", "SPY")):
            df = m.download_one_us(ticker)
            if df is None or len(df) < 200:
                print(f"  (skip {label}: no data)")
                continue
            bh = _bh_equity(df.Close, eqn.index)
            common = eqn.index.intersection(bh.index)
            series[label] = bh.loc[common] / bh.loc[common].iloc[0] * m.INITIAL_CAPITAL
            series["Long"] = eqn.loc[common] / eqn.loc[common].iloc[0] * m.INITIAL_CAPITAL
            _print_perf(label, series[label])

        print("\n--- CAGR by window ---")
        _window_table(series)

        # Yearly Long vs QQQ if present
        if "QQQ" in series:
            print("\n--- Calendar year returns (Long vs QQQ) ---")
            long_yr = series["Long"].resample("Y").last().pct_change().dropna() * 100
            qqq_yr = series["QQQ"].resample("Y").last().pct_change().dropna() * 100
            print(f"{'Year':6s} {'Long':>8} {'QQQ':>8} {'vs QQQ':>8}")
            for dt, r in long_yr.items():
                q = qqq_yr.loc[dt] if dt in qqq_yr.index else np.nan
                print(f"{dt.year:6d} {r:+7.1f}% {q:+7.1f}% {r-q:+7.1f}%")

            # Chart
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                L = series["Long"]
                Q = series["QQQ"]
                dd_l = L / L.cummax() - 1
                dd_q = Q / Q.cummax() - 1
                fig, ax = plt.subplots(2, 1, figsize=(11, 8), sharex=True,
                                       gridspec_kw={"height_ratios": [3, 1]})
                ax[0].plot(L.index, L.values, label="Long", color="#c62828", lw=1.4)
                ax[0].plot(Q.index, Q.values, label="QQQ", color="#1565c0", lw=1.1)
                if "NASDAQ Comp" in series:
                    I = series["NASDAQ Comp"]
                    ax[0].plot(I.index, I.values, label="NASDAQ Comp", color="#00838f", lw=1.0, alpha=0.85)
                ax[0].set_yscale("log")
                ax[0].set_ylabel("Equity ($, log)")
                ax[0].legend(loc="upper left", fontsize=8)
                ax[0].set_title(f"US Long ({tag}) vs NASDAQ benchmarks")
                ax[0].grid(True, alpha=0.25)
                ax[1].plot(dd_l.index, dd_l.values * 100, color="#c62828", lw=1.0, label="Long DD")
                ax[1].plot(dd_q.index, dd_q.values * 100, color="#1565c0", lw=0.9, label="QQQ DD")
                ax[1].set_ylabel("Drawdown %")
                ax[1].legend(loc="lower left", fontsize=8)
                ax[1].grid(True, alpha=0.25)
                plt.tight_layout()
                chart = m.OUTPUT_BACKTEST / f"equity_curve_us_long_{tag}_vs_ndx.png"
                plt.savefig(chart, dpi=120)
                print(f"\nSaved {chart}")
            except Exception as e:
                print(f"Chart skipped: {e}")

    print(f"\nSaved {eq_path}")
    print(f"Saved {tr_path}")
    return eq, tr


def run_us_backtest_top_mcap():
    """Same as main.py backtest-us (top-by-mcap list)."""
    return m.run_us_backtest()


def run_us_mcap_study():
    return m.run_us_long_study()


def run_import(argv):
    return m.run_import_us(argv)


def run_download(argv):
    return m.run_download_us(argv)


MAJOR_NASDAQ_EXTRA = [
    "TSLA", "META", "GOOGL", "GOOG", "AMZN", "MSFT", "NVDA", "AAPL", "NFLX", "AMD", "INTC",
    "CSCO", "ADBE", "COST", "PEP", "AVGO", "QCOM", "TXN", "AMAT", "MU", "SBUX", "ISRG", "BKNG",
    "MELI", "PDD", "ABNB", "DASH", "COIN", "HOOD", "SMCI", "ARM", "APP", "PLTR", "CRWD", "PANW",
    "DDOG", "MDB", "SNOW", "ZS", "NET", "OKTA", "FTNT", "TEAM", "WDAY", "INTU", "ADP", "ROST",
    "ORLY", "LULU", "FAST", "PAYX", "VRSK", "CTAS", "PCAR", "ODFL", "IDXX", "DXCM", "GILD",
    "BIIB", "REGN", "VRTX", "AMGN", "MRNA", "ILMN", "ALGN", "PYPL", "EBAY", "ETSY", "LCID",
    "RIVN", "SOFI", "ZG", "Z", "ROKU", "TTD", "SHOP", "ZM", "CDNS", "SNPS", "KLAC", "LRCX",
    "ASML", "MRVL", "ADI", "NXPI", "ON", "MCHP", "CHTR", "CMCSA", "TMUS", "EA", "TTWO",
    "BIDU", "JD", "NTES", "BILI", "LI", "FUTU", "TIGR", "IQ", "GRAB", "AFRM", "RIOT",
]


def run_download_majors():
    """Ensure major NASDAQ names (curated list ∩ NASDAQ listed) are cached from 2010."""
    m._ensure_output_dirs()
    nd = set(nasdaq_listed_symbols())
    major = sorted({m._yahoo_symbol_us(s) for s in MAJOR_NASDAQ_EXTRA if m._yahoo_symbol_us(s) in nd})
    print(f"Major NASDAQ targets: {len(major)}")
    need = [s for s in major if not m._us_cache_is_usable(m._us_cache_file(s))]
    print(f"Already usable: {len(major) - len(need)} | downloading: {len(need)}")
    if need:
        print(need)
        m.download_many_us(need, start=m.US_PRICE_START, force=False, sequential=True)
    m.get_us_universe_meta(major, refresh=False)
    ok = [s for s in major if m._us_cache_is_usable(m._us_cache_file(s))]
    out = m.OUTPUT_BACKTEST / "nasdaq_major_list.csv"
    pd.DataFrame({"ticker": ok}).to_csv(out, index=False)
    print(f"Usable majors: {len(ok)}/{len(major)}")
    print(f"Saved {out}")
    return ok


def run_coverage():
    """Print NASDAQ / cache coverage stats."""
    nd = set(nasdaq_listed_symbols())
    cached = set(cached_us_symbols())
    usable = {s for s in cached if m._us_cache_is_usable(m._us_cache_file(s))}
    have = sorted(nd & usable)
    missing = sorted(nd - usable)
    print(f"NASDAQ listed commons: {len(nd)}")
    print(f"In cache (usable):     {len(have)} ({100 * len(have) / len(nd):.1f}%)")
    print(f"Missing:               {len(missing)}")
    print(f"Total cache CSVs:      {len(list(m.US_CACHE_DIR.glob('*.csv')))}")
    out = m.OUTPUT_BACKTEST / "nasdaq_missing_from_cache.csv"
    pd.DataFrame({"ticker": missing}).to_csv(out, index=False)
    print(f"Wrote {out}")
    return have, missing


def cli_help():
    print(
        """
US Long screener — same strategy as main.py, US data only.

Usage:
  python main_us.py help

Backtest (from 2010, writes output/backtest/):
  python main_us.py backtest              All cached US equities + QQQ/IXIC compare
  python main_us.py backtest-nasdaq       NASDAQ-listed cached names only + QQQ/IXIC
  python main_us.py backtest-top          Top-by-mcap list (same as main.py backtest-us)
  python main_us.py backtest --through ROK   Alphabetical subset through ROK.csv

Data:
  python main_us.py import                S&P 1500 + extra NASDAQ/NYSE into data_cache_us/
  python main_us.py import --extra 3000
  python main_us.py download --top 1000   Top N by Yahoo mcap
  python main_us.py download-majors       Curated major NASDAQ names
  python main_us.py coverage              NASDAQ vs cache coverage report

Research:
  python main_us.py study-mcap            Market-cap floor sweep -> output/studies/

Strategy rules: LONG_STRATEGY.md (shared with India live book).
Cache: data_cache_us/   |   prices from 2010-01-01 (or IPO if later).
"""
    )


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    cmd = argv[0] if argv else "help"
    rest = argv[1:]

    if cmd in ("help", "-h", "--help"):
        cli_help()
    elif cmd in ("backtest", "bt"):
        nasdaq_only = False
        through = None
        i = 0
        while i < len(rest):
            a = rest[i]
            if a in ("--nasdaq", "-n"):
                nasdaq_only = True
                i += 1
            elif a in ("--through", "-t") and i + 1 < len(rest):
                through = rest[i + 1].upper().replace(".CSV", "")
                i += 2
            else:
                i += 1
        run_us_backtest(nasdaq_only=nasdaq_only, through=through, compare_ndx=True)
    elif cmd in ("backtest-nasdaq", "nasdaq", "bt-nasdaq"):
        run_us_backtest(nasdaq_only=True, compare_ndx=True, tag="nasdaq")
    elif cmd in ("backtest-top", "top", "backtest-us"):
        run_us_backtest_top_mcap()
    elif cmd in ("import", "import-us"):
        run_import(rest)
    elif cmd in ("download", "download-us"):
        run_download(rest)
    elif cmd in ("download-majors", "majors"):
        run_download_majors()
    elif cmd in ("coverage", "nasdaq-coverage"):
        run_coverage()
    elif cmd in ("study-mcap", "study", "mcap"):
        run_us_mcap_study()
    else:
        print(f"Unknown command: {cmd}\n")
        cli_help()
        raise SystemExit(2)


if __name__ == "__main__":
    main()
