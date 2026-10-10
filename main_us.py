"""
US Monthly Long book: download + portfolio backtest (Yahoo US stocks, 2010 onwards).

Adopted US book (NASDAQ studies; India LONG_BOOK in main.py unchanged):
  month-end signals, next-open fills, equal-weight 12 slots, sector_cap=4,
  SMA200 exit, month-end 20% peak stop, QQQ-trend idle cash,
  PIT dollar-volume top ~700 universe + GICS sector labels.

Setup:   pip install yfinance pandas numpy matplotlib
Run:     python main_us.py help

Cache:   data_cache_us/
Output:  output/backtest/  (and output/studies/ for grids)
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

import main as m


# ----------------------------- US helpers ------------------------------------
SKIP_CACHE = {"universe_meta_us", "us_top_by_mcap", "SPY", "GLD", "QQQ"}
US_SECTOR_CACHE = m.US_CACHE_DIR / "us_sectors.csv"
# PIT dollar-volume universe (NASDAQ size study sweet spot ~700).
US_PIT_DV_TOP_N = 700
US_PIT_DV_WINDOW = 60

# Finalized US Long book (does not mutate India LONG_BOOK).
US_LONG_BOOK = {
    **m.LONG_BOOK,
    "max_positions": 12,
    "sector_cap": 4,
    "dd_stop": 0.20,  # month-end peak stop (NASDAQ; India stays 30%)
}


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
        if name in SKIP_CACHE or name.startswith("IDX_") or name == "us_sectors":
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


def apply_pit_universe(P, top_n=US_PIT_DV_TOP_N, window=US_PIT_DV_WINDOW):
    """
    Point-in-time tradable universe: top_n names by rolling median dollar volume.
    Needs only price/volume (no today's market cap). Masks entry_* / signal_bo.
    """
    dv = (P["Close"] * P["Volume"]).rolling(window, min_periods=max(10, window // 2)).median()
    in_u = dv.rank(axis=1, ascending=False) <= top_n
    in_u = in_u.fillna(False)
    for c in [k for k in P if str(k).startswith("entry_long") or k == "signal_bo"]:
        P[c] = (P[c].astype(bool) & in_u).fillna(False).astype(bool)
    return in_u


def ew_universe_equity(P, in_u, capital=None):
    """Equal-weight daily B&H of the same PIT universe (bias gauge vs SPY)."""
    capital = m.INITIAL_CAPITAL if capital is None else capital
    ew = P["Close"].pct_change().where(in_u).mean(axis=1).fillna(0)
    return (1 + ew).cumprod() * capital


def cash_sleeve_trend_us(dates, ticker="QQQ", filter_ticker=None, debt_yield=0.045):
    """
    Idle cash sleeve aligned to NASDAQ:
      - Invest in `ticker` (default QQQ) when the trend filter is on
      - Else earn ~T-bill yield
    filter_ticker: optional separate trend series (e.g. '^IXIC'); default = ticker itself.
    """
    px = m.download_one_us(ticker)
    if px is None or len(px) < 252:
        return None
    invest = px.Close.reindex(dates).ffill()
    if filter_ticker and filter_ticker != ticker:
        ft = m.download_one_us(filter_ticker)
        if ft is None or len(ft) < 252:
            filt = invest
        else:
            filt = ft.Close.reindex(dates).ffill()
    else:
        filt = invest
    on = (filt > filt.rolling(200).mean()).shift(1).fillna(False).to_numpy()
    return np.where(on, invest.pct_change().fillna(0).to_numpy(), debt_yield / 252).astype(float)


def load_us_sectors(tickers, refresh=False):
    """
    GICS sector labels from Wikipedia S&P 500/400/600 tables (cached).
    Survivorship-biased to current constituents — still better than blank sectors.
    """
    m.US_CACHE_DIR.mkdir(exist_ok=True)
    if US_SECTOR_CACHE.exists() and not refresh:
        cached = pd.read_csv(US_SECTOR_CACHE, index_col=0)["sector"]
    else:
        cached = pd.Series(dtype=object)
    need = [t for t in tickers if t not in cached.index]
    if need or refresh:
        rows = {}
        urls = [
            ("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
             ("Symbol",), ("GICS Sector", "Sector")),
            ("https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
             ("Symbol", "Ticker symbol"), ("GICS Sector", "Sector", "GICS sector")),
            ("https://en.wikipedia.org/wiki/List_of_S%26P_600_companies",
             ("Symbol", "Ticker symbol"), ("GICS Sector", "Sector", "GICS sector")),
        ]
        for url, sym_hints, sec_hints in urls:
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; screener/1.0)"})
                with urllib.request.urlopen(req, timeout=30) as resp:
                    tables = pd.read_html(resp.read())
            except Exception as e:
                print(f"  sector fetch skipped ({url.split('/')[-1]}): {e}")
                continue
            for tbl in tables:
                cols = list(tbl.columns)
                sym_col = next((c for c in cols if str(c).strip() in sym_hints or "Symbol" in str(c)), None)
                sec_col = next((c for c in cols if str(c).strip() in sec_hints or "Sector" in str(c)), None)
                if sym_col is None or sec_col is None:
                    continue
                for _, r in tbl.iterrows():
                    sym = m._yahoo_symbol_us(str(r[sym_col]).split("\n")[0].strip())
                    sec = str(r[sec_col]).strip()
                    if sym and sec and sec.lower() != "nan":
                        rows[sym] = sec
                if rows:
                    break
        if rows:
            new = pd.Series(rows, name="sector")
            cached = pd.concat([cached, new])
            cached = cached[~cached.index.duplicated(keep="last")]
            cached.to_csv(US_SECTOR_CACHE, header=["sector"])
            print(f"Cached GICS sectors for {len(cached)} names -> {US_SECTOR_CACHE.name}")
    return [str(cached.get(t, "") or "") for t in tickers]


def _us_long_kwargs(dates, cash_mode="qqq"):
    """
    Adopted US kwargs: 12 slots, sector_cap=4, peak 20%, QQQ cash sleeve by default.
    cash_mode: 'qqq' | 'ndx' | 'spy' | 'gld' | None
    """
    kw = dict(US_LONG_BOOK)
    sleeve = None
    if cash_mode == "qqq":
        sleeve = cash_sleeve_trend_us(dates, ticker="QQQ")
    elif cash_mode == "ndx":
        sleeve = cash_sleeve_trend_us(dates, ticker="QQQ", filter_ticker="^IXIC")
    elif cash_mode == "spy":
        sleeve = cash_sleeve_trend_us(dates, ticker="SPY")
    elif cash_mode == "gld":
        sleeve = m.cash_sleeve_returns_us(dates)
    if sleeve is not None:
        kw["cash_ret"] = sleeve
        kw["cash_ret_idle_only"] = False
    return kw


def _ind_kw_pit():
    """No look-ahead mcap; liquidity floor still applied; PIT DV rank applied after panels."""
    return dict(
        tradable_mcap_min=0,
        turnover_min=m.US_MIN_TURNOVER,
        mcap_in_usd=True,
        require_mcap=False,
    )


def _ind_kw_mcap():
    return dict(
        tradable_mcap_min=m.US_MIN_MCAP_USD,
        turnover_min=m.US_MIN_TURNOVER,
        mcap_in_usd=True,
        require_mcap=True,
    )


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


def _prepare_us_panels(nasdaq_only=False, through=None, use_pit_dv=True, pit_top_n=US_PIT_DV_TOP_N,
                       with_sectors=True):
    data, meta = load_us_from_cache(nasdaq_only=nasdaq_only, through=through)
    if not data:
        raise SystemExit("No US price data in data_cache_us/. Run: python main_us.py import")
    bench, bname = m._load_us_sp500_close(None)
    ind_kw = _ind_kw_pit() if use_pit_dv else _ind_kw_mcap()
    print(f"Building panels (pit_dv={use_pit_dv}"
          + (f", top_n={pit_top_n}" if use_pit_dv else f", mcap≥${m.US_MIN_MCAP_USD/1e6:.0f}M")
          + f") | bench {bname}...")
    tickers, P = m.build_panels(data, universe=meta, bench_close=bench, indicator_kw=ind_kw)
    in_u = None
    if use_pit_dv:
        in_u = apply_pit_universe(P, top_n=pit_top_n)
        med = float(in_u.sum(axis=1).median())
        print(f"  PIT DV universe: median {med:.0f} names/day (top_n={pit_top_n})")
    sectors = load_us_sectors(tickers) if with_sectors else [""] * len(tickers)
    n_labeled = sum(1 for s in sectors if s)
    print(f"  Sector labels: {n_labeled}/{len(tickers)}")
    return tickers, P, sectors, in_u, bench, bname


def run_us_backtest(nasdaq_only=True, through=None, compare_ndx=True, tag=None,
                    use_pit_dv=True, pit_top_n=US_PIT_DV_TOP_N, with_sectors=True,
                    cash_mode="qqq", extra_kw=None):
    """
    Run adopted US Long book.

    Defaults (finalized):
      - NASDAQ-listed cache, PIT DV top 700
      - 12 equal slots, GICS sector_cap=4, month-end peak 20%
      - QQQ-trend cash sleeve
    """
    m._ensure_output_dirs()
    tickers, P, sectors, in_u, bench, bname = _prepare_us_panels(
        nasdaq_only=nasdaq_only, through=through, use_pit_dv=use_pit_dv,
        pit_top_n=pit_top_n, with_sectors=with_sectors,
    )
    kw = _us_long_kwargs(P["Close"].index, cash_mode=cash_mode)
    if extra_kw:
        kw.update(extra_kw)

    print(f"Running Long from {m.US_PRICE_START} | {len(tickers)} names | bench {bname}...")
    print(f"  Book: {kw.get('max_positions')} slots | sector_cap={kw.get('sector_cap')} | "
          f"peak={kw.get('dd_stop')} | cash={cash_mode}")
    eq, tr = m.backtest(tickers, P, sectors=sectors, start=m.US_PRICE_START, **kw)

    if tag is None:
        tag = "nasdaq_adopted"
    eq_path = m.OUTPUT_BACKTEST / f"equity_curve_us_long_{tag}.csv"
    tr_path = m.OUTPUT_BACKTEST / f"trades_us_long_{tag}.csv"
    eq.to_csv(eq_path, header=["equity"])
    tr.to_csv(tr_path, index=False)

    # Persist adopted defaults for reference
    rec = {
        "max_positions": US_LONG_BOOK["max_positions"],
        "sector_cap": US_LONG_BOOK["sector_cap"],
        "dd_stop": US_LONG_BOOK["dd_stop"],
        "cash_mode": cash_mode,
        "nasdaq_only": nasdaq_only,
        "use_pit_dv": use_pit_dv,
        "pit_top_n": pit_top_n,
        "with_sectors": with_sectors,
    }
    (m.OUTPUT_STUDIES / "us_adopted_defaults.json").write_text(json.dumps(rec, indent=2))

    eqn = eq.dropna()
    univ = f"PIT DV top {pit_top_n}" if use_pit_dv else f"mcap ≥ ${m.US_MIN_MCAP_USD/1e6:.0f}M"
    print(f"\n=== US Long ({tag}) | {len(tickers)} names | {univ} | "
          f"{kw.get('max_positions')} slots | sector_cap={kw.get('sector_cap')} | "
          f"peak {int(kw.get('dd_stop', 0)*100)}% | "
          f"{eqn.index[0].date()} → {eqn.index[-1].date()} ===")
    _print_perf("Long", eqn)
    t = m.trade_stats(tr)
    print(f"  Trades {t.get('Trades', len(tr))}  Win rate {t.get('Win rate %', float('nan')):.1f}%  "
          f"PF {t.get('Profit factor', float('nan')):.2f}")

    series = {"Long": eqn / eqn.iloc[0] * m.INITIAL_CAPITAL}
    if in_u is not None:
        ew = ew_universe_equity(P, in_u).reindex(eqn.index).ffill()
        ew = ew / ew.iloc[0] * m.INITIAL_CAPITAL
        series["EW univ"] = ew
        _print_perf("EW univ (bias)", ew)

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
            if "EW univ" in series:
                series["EW univ"] = series["EW univ"].loc[common] / series["EW univ"].loc[common].iloc[0] * m.INITIAL_CAPITAL
            _print_perf(label, series[label])

        if "EW univ" in series and "SPY" in series:
            ew_st = m.perf_stats(series["EW univ"], "")
            sp_st = m.perf_stats(series["SPY"], "")
            gap = ew_st["CAGR %"] - sp_st["CAGR %"]
            print(f"\n  Bias gauge: EW universe CAGR − SPY CAGR = {gap:+.2f} pp "
                  f"(large positive ⇒ leftover survivorship/liquidity bias)")

        print("\n--- CAGR by window ---")
        _window_table(series)

        if "QQQ" in series:
            print("\n--- Calendar year returns (Long vs QQQ) ---")
            long_yr = series["Long"].resample("Y").last().pct_change().dropna() * 100
            qqq_yr = series["QQQ"].resample("Y").last().pct_change().dropna() * 100
            print(f"{'Year':6s} {'Long':>8} {'QQQ':>8} {'vs QQQ':>8}")
            for dt, r in long_yr.items():
                q = qqq_yr.loc[dt] if dt in qqq_yr.index else np.nan
                print(f"{dt.year:6d} {r:+7.1f}% {q:+7.1f}% {r-q:+7.1f}%")

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
                if "EW univ" in series:
                    ax[0].plot(series["EW univ"].index, series["EW univ"].values,
                               label="EW univ", color="#6a1b9a", lw=1.0, alpha=0.8)
                if "NASDAQ Comp" in series:
                    I = series["NASDAQ Comp"]
                    ax[0].plot(I.index, I.values, label="NASDAQ Comp", color="#00838f", lw=1.0, alpha=0.85)
                ax[0].set_yscale("log")
                ax[0].set_ylabel("Equity ($, log)")
                ax[0].legend(loc="upper left", fontsize=8)
                ax[0].set_title(f"US Long ({tag}) vs benchmarks")
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


def tune_pit_top_n(tickers, P_mcap, P_raw, top_ns=(500, 800, 1000, 1200, 1500)):
    """Pick top_n whose median daily count is closest to the old $240M mcap universe size."""
    # Count how many names pass entry_long under mcap panels (proxy for prior breadth)
    el = P_mcap["entry_long"]
    # Also count tradable-ish via mcap entry candidates more loosely: any finite close with liquidity
    # Better: rebuild isn't available — use entry_long True count median under mcap book
    target = float(el.sum(axis=1).replace(0, np.nan).median())
    if not np.isfinite(target) or target < 10:
        # fallback: count names with Close notna
        target = 400.0
    print(f"\nTuning PIT top_n to match mcap-era entry breadth (median entry_long count ≈ {target:.0f})")
    best_n, best_gap = US_PIT_DV_TOP_N, 1e18
    # Rank by DV on raw Close/Volume (same P)
    rows = []
    for n in top_ns:
        # don't mutate caller's P — work on a copy of entry mask size via DV only
        dv = (P_raw["Close"] * P_raw["Volume"]).rolling(
            US_PIT_DV_WINDOW, min_periods=max(10, US_PIT_DV_WINDOW // 2)
        ).median()
        in_u = (dv.rank(axis=1, ascending=False) <= n).fillna(False)
        med = float(in_u.sum(axis=1).median())
        # entry breadth under PIT: entry_long from raw panels & in_u
        ent = (P_raw["entry_long"].astype(bool) & in_u).sum(axis=1)
        ent_med = float(ent.replace(0, np.nan).median())
        gap = abs(ent_med - target) if np.isfinite(ent_med) else abs(med - 800)
        rows.append({"top_n": n, "median_in_u": round(med, 1), "median_entry_long": round(ent_med, 1), "gap": round(gap, 1)})
        if gap < best_gap:
            best_gap, best_n = gap, n
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"Chosen top_n={best_n}")
    return best_n


def run_us_study(nasdaq_only=False, through=None, pit_top_n=None):
    """
    Test honesty fixes + new ideas on US data. Adopt into defaults only if both
    2010-18 and 2019+ Calmar improve (and CAGR within 5pts) vs baseline — same as India _helps.
    Holdout note: 2023+ is reported but not used for adoption.
    """
    m._ensure_output_dirs()
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)

    data, meta = load_us_from_cache(nasdaq_only=nasdaq_only, through=through)
    if not data:
        raise SystemExit("No US price data.")
    bench, bname = m._load_us_sp500_close(None)

    # --- A) mcap baseline vs PIT DV (honesty) ---
    print("\n=== A) Universe: today-mcap look-ahead vs PIT dollar-volume ===")
    t_m, P_m = m.build_panels(data, universe=meta, bench_close=bench, indicator_kw=_ind_kw_mcap())
    t_p, P_p = m.build_panels(data, universe=meta, bench_close=bench, indicator_kw=_ind_kw_pit())
    sectors_blank = [""] * len(t_p)
    sectors = load_us_sectors(t_p)

    if pit_top_n is None:
        pit_top_n = tune_pit_top_n(t_m, P_m, P_p)
    in_u = apply_pit_universe(P_p, top_n=pit_top_n)
    print(f"PIT DV median names/day: {float(in_u.sum(axis=1).median()):.0f}")

    dates = P_p["Close"].index
    # Study baseline keeps GLD sleeve so SPY-sleeve is a true one-change test.
    base_kw = _us_long_kwargs(dates, cash_mode="gld")

    def _half_stats(eq, tr):
        return m._window_stats(eq, tr)

    windows = [
        ("2010-2018", dict(start="2010-01-01", end="2018-12-31")),
        ("2019+", dict(start="2019-01-01")),
        ("Full", {}),
        ("2023+ holdout", dict(start="2023-01-01")),
    ]

    univ_specs = [
        ("mcap floor (look-ahead)", t_m, P_m, sectors_blank[:len(t_m)], dict(base_kw)),
        ("PIT DV + blank sectors", t_p, P_p, sectors_blank, dict(base_kw)),
        ("PIT DV + GICS sectors", t_p, P_p, sectors, dict(base_kw)),
    ]
    univ_rows = []
    stored_univ = {}
    for name, tickers, P, secs, kw in univ_specs:
        row = {"Variant": name}
        halves = []
        for wlabel, wkw in windows:
            eq, tr = m.backtest(tickers, P, sectors=secs, **{**kw, **wkw})
            st = _half_stats(eq, tr)
            row[f"{wlabel} CAGR"] = round(st["CAGR"], 2)
            row[f"{wlabel} DD"] = round(st["DD"], 2)
            row[f"{wlabel} Calmar"] = round(st["Calmar"], 3)
            if wlabel in ("2010-2018", "2019+"):
                halves.append(st)
            if wlabel == "Full":
                stored_univ[name] = (eq, tr, st)
                row["Trades"] = len(tr)
        univ_rows.append(row)
        print(f"  {name}: Full CAGR {row['Full CAGR']}%  DD {row['Full DD']}%  Calmar {row['Full Calmar']}")

    univ_tbl = pd.DataFrame(univ_rows).set_index("Variant")
    print("\n", univ_tbl.to_string())

    # Bias gauge on PIT universe
    ew = ew_universe_equity(P_p, in_u)
    spy = m.download_one_us("SPY")
    spy_eq = _bh_equity(spy.Close, ew.index)
    common = ew.index.intersection(spy_eq.index)
    ew_a = ew.loc[common] / ew.loc[common].iloc[0] * m.INITIAL_CAPITAL
    sp_a = spy_eq.loc[common] / spy_eq.loc[common].iloc[0] * m.INITIAL_CAPITAL
    ew_p = m.perf_stats(ew_a, "")
    sp_p = m.perf_stats(sp_a, "")
    print(f"\n=== D) Bias gauge (PIT EW B&H vs SPY) ===")
    print(f"  EW univ CAGR {ew_p['CAGR %']:.2f}%  |  SPY {sp_p['CAGR %']:.2f}%  |  "
          f"gap {ew_p['CAGR %'] - sp_p['CAGR %']:+.2f} pp")

    # Choose universe baseline for idea tests: prefer PIT+sectors if Full Calmar >= mcap - small epsilon
    # Adoption for honesty: PIT is preferred when Full Calmar not worse by >0.05 AND CAGR within 5pts,
    # else keep reporting but default stays mcap until PIT wins on profitability.
    mcap_full = stored_univ["mcap floor (look-ahead)"][2]
    pit_sec_full = stored_univ["PIT DV + GICS sectors"][2]
    pit_blank_full = stored_univ["PIT DV + blank sectors"][2]
    use_pit = (
        pit_sec_full["Calmar"] >= mcap_full["Calmar"] - 0.02
        and pit_sec_full["CAGR"] >= mcap_full["CAGR"] - 5
    ) or (
        pit_blank_full["Calmar"] >= mcap_full["Calmar"] - 0.02
        and pit_blank_full["CAGR"] >= mcap_full["CAGR"] - 5
    )
    # Stricter user rule: only include if increases profitability
    use_pit_strict = (
        pit_sec_full["Calmar"] > mcap_full["Calmar"]
        and pit_sec_full["CAGR"] >= mcap_full["CAGR"] - 5
    ) or (
        pit_blank_full["Calmar"] > mcap_full["Calmar"]
        and pit_blank_full["CAGR"] >= mcap_full["CAGR"] - 5
    )
    use_sectors = pit_sec_full["Calmar"] > pit_blank_full["Calmar"] and pit_sec_full["CAGR"] >= pit_blank_full["CAGR"] - 5

    print(f"\nUniverse adoption (profitability): PIT={'YES' if use_pit_strict else 'NO'}  "
          f"sectors={'YES' if use_sectors else 'NO'}")

    # Work on PIT panels for idea grid regardless (honest OOS methodology), but mark adoption separately
    tickers, P, secs = t_p, P_p, (sectors if use_sectors else sectors_blank)
    # Ensure PIT mask applied (already applied to P_p)
    spy_c, _ = m._load_us_sp500_close(None)
    idx_on = m.index_crash_regime(spy_c, dates)
    spy_sleeve = cash_sleeve_trend_us(dates)

    print("\n=== B/C ideas on PIT DV universe (grid) ===")
    base = dict(base_kw)
    # If sectors adopted for testing path, keep them in base sectors list `secs`
    specs = [
        ("baseline (PIT + GLD sleeve)", {}),
        ("daily peak 25%", dict(daily_dd_stop=0.25)),
        ("daily peak 30%", dict(daily_dd_stop=0.30)),
        ("daily peak 35%", dict(daily_dd_stop=0.35)),
        ("index crash exit", dict(crash_off=~idx_on, reentry_ok=idx_on)),
        ("SPY-trend cash sleeve", dict(cash_ret=spy_sleeve, cash_ret_idle_only=False)),
        ("20 slots", dict(max_positions=20)),
        ("daily peak 30% + crash", dict(daily_dd_stop=0.30, crash_off=~idx_on, reentry_ok=idx_on)),
        ("daily peak 30% + SPY sleeve", dict(daily_dd_stop=0.30, cash_ret=spy_sleeve, cash_ret_idle_only=False)),
    ]

    # Half baselines for _helps
    base_halves = []
    for wlabel, wkw in windows:
        if wlabel not in ("2010-2018", "2019+"):
            continue
        eq_b, tr_b = m.backtest(tickers, P, sectors=secs, **{**base, **wkw})
        base_halves.append(_half_stats(eq_b, tr_b))
    b0, b1 = base_halves

    idea_rows = []
    stored_ideas = {}
    for name, extra in specs:
        row = {"Variant": name}
        halves = []
        for wlabel, wkw in windows:
            kw = {**base, **extra, **wkw}
            # None cash_ret override: if key present as None, drop
            if kw.get("cash_ret") is None and "cash_ret" in extra and extra["cash_ret"] is None:
                kw.pop("cash_ret", None)
            eq, tr = m.backtest(tickers, P, sectors=secs, **kw)
            st = _half_stats(eq, tr)
            row[f"{wlabel} CAGR"] = round(st["CAGR"], 2)
            row[f"{wlabel} DD"] = round(st["DD"], 2)
            row[f"{wlabel} Calmar"] = round(st["Calmar"], 3)
            if wlabel in ("2010-2018", "2019+"):
                halves.append(st)
            if wlabel == "Full":
                stored_ideas[name] = (eq, tr, st)
                row["Trades"] = len(tr)
        keep = ""
        if name != "baseline (PIT + GLD sleeve)" and len(halves) == 2:
            if m._helps((b0, halves[0]), (b1, halves[1])):
                keep = "yes"
        row["Adopt?"] = keep
        idea_rows.append(row)
        print(f"  {name}: Full Calmar {row['Full Calmar']}  Adopt={keep or '—'}")

    idea_tbl = pd.DataFrame(idea_rows).set_index("Variant")
    print("\n", idea_tbl.to_string())

    out = m.OUTPUT_STUDIES / "us_honesty_idea_study.csv"
    # combine
    u = univ_tbl.reset_index()
    u.insert(0, "Study", "universe")
    i = idea_tbl.reset_index()
    i.insert(0, "Study", "ideas")
    pd.concat([u, i], ignore_index=True).to_csv(out, index=False)
    print(f"\nSaved {out}")

    adopted = [r["Variant"] for r in idea_rows if r.get("Adopt?") == "yes"]
    print("\n=== Adoption summary (US only; India defaults unchanged) ===")
    print(f"  Tick scrub (_scrub_bad_ticks): ALWAYS ON (data honesty)")
    print(f"  PIT dollar-volume universe:   {'ADOPT' if use_pit_strict else 'KEEP TESTING ONLY (did not raise profitability vs mcap)'}")
    print(f"  GICS sector cap labels:       {'ADOPT' if use_sectors else 'skip'}")
    print(f"  Idea variants to adopt:       {adopted or '(none)'}")
    print("  Note: treat US as OOS for India-tuned rules; holdout 2023+ shown but not used for Adopt?.")

    # Write recommended US kwargs for reference (does not mutate LONG_BOOK in main.py)
    rec = {"use_pit_dv": bool(use_pit_strict), "pit_top_n": int(pit_top_n),
           "with_sectors": bool(use_sectors), "extra_kw": {}}
    # Map adopted idea names to kwargs (first winning single change; prefer daily peak 30 if present)
    for name in adopted:
        if name.startswith("daily peak") and "crash" not in name and "SPY" not in name:
            pct = float(name.split()[-1].replace("%", "")) / 100.0
            rec["extra_kw"]["daily_dd_stop"] = pct
        elif name == "index crash exit":
            rec["extra_kw"]["use_crash"] = True
        elif name == "SPY-trend cash sleeve":
            rec["extra_kw"]["use_spy_sleeve"] = True
        elif name == "20 slots":
            rec["extra_kw"]["max_positions"] = 20
    rec_path = m.OUTPUT_STUDIES / "us_adopted_defaults.json"
    import json
    rec_path.write_text(json.dumps(rec, indent=2))
    print(f"  Wrote {rec_path}")

    # Update module defaults only for profitability wins
    global US_PIT_DV_TOP_N
    US_PIT_DV_TOP_N = int(pit_top_n)
    return univ_tbl, idea_tbl, rec


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

Adopted book: NASDAQ PIT top 700 | 12 slots | sector_cap=4 | peak 20% | QQQ sleeve

Backtest (from 2010, writes output/backtest/):
  python main_us.py backtest              Adopted NASDAQ Long + QQQ/IXIC compare
  python main_us.py backtest-nasdaq       Same
  python main_us.py backtest-top          Legacy top-mcap (India-like 15/30% via main.py)
  python main_us.py backtest --all-exchanges
  python main_us.py backtest --ndx        ^IXIC filter → QQQ sleeve
  python main_us.py backtest --gld        GLD cash sleeve (legacy)

Study:
  python main_us.py study                 Research grids -> output/studies/

Data:
  python main_us.py import
  python main_us.py import --extra 3000
  python main_us.py download --top 1000
  python main_us.py download-majors
  python main_us.py coverage

Research:
  python main_us.py study-mcap

India LONG_BOOK unchanged in main.py. Cache: data_cache_us/ from 2010-01-01.
"""
    )


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    cmd = argv[0] if argv else "help"
    rest = argv[1:]

    if cmd in ("help", "-h", "--help"):
        cli_help()
    elif cmd in ("backtest", "bt"):
        nasdaq_only = True
        through = None
        use_pit = True
        cash_mode = "qqq"
        i = 0
        while i < len(rest):
            a = rest[i]
            if a in ("--nasdaq", "-n"):
                nasdaq_only = True
                i += 1
            elif a in ("--all-exchanges", "--all"):
                nasdaq_only = False
                i += 1
            elif a in ("--through", "-t") and i + 1 < len(rest):
                through = rest[i + 1].upper().replace(".CSV", "")
                i += 2
            elif a in ("--pit", "--pit-dv"):
                use_pit = True
                i += 1
            elif a in ("--mcap",):
                use_pit = False
                i += 1
            elif a in ("--gld",):
                cash_mode = "gld"
                i += 1
            elif a in ("--ndx", "--ixic"):
                cash_mode = "ndx"
                i += 1
            elif a in ("--spy",):
                cash_mode = "spy"
                i += 1
            elif a in ("--qqq",):
                cash_mode = "qqq"
                i += 1
            else:
                i += 1
        run_us_backtest(
            nasdaq_only=nasdaq_only, through=through, compare_ndx=True,
            use_pit_dv=use_pit, with_sectors=True, cash_mode=cash_mode,
            tag="nasdaq_adopted",
        )
    elif cmd in ("backtest-nasdaq", "nasdaq", "bt-nasdaq"):
        run_us_backtest(
            nasdaq_only=True, compare_ndx=True, tag="nasdaq_adopted",
            use_pit_dv=True, with_sectors=True, cash_mode="qqq",
        )
    elif cmd in ("backtest-top", "top", "backtest-us"):
        run_us_backtest_top_mcap()
    elif cmd in ("study", "study-us", "honesty"):
        nasdaq_only = "--nasdaq" in rest
        run_us_study(nasdaq_only=nasdaq_only)
    elif cmd in ("import", "import-us"):
        run_import(rest)
    elif cmd in ("download", "download-us"):
        run_download(rest)
    elif cmd in ("download-majors", "majors"):
        run_download_majors()
    elif cmd in ("coverage", "nasdaq-coverage"):
        run_coverage()
    elif cmd in ("study-mcap", "mcap"):
        run_us_mcap_study()
    else:
        print(f"Unknown command: {cmd}\n")
        cli_help()
        raise SystemExit(2)


if __name__ == "__main__":
    main()
