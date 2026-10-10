"""
Monthly Long book: data download + portfolio backtest (NSE stocks, 2010 onwards)

Setup:   pip install yfinance pandas numpy matplotlib
         pip install nsepython   # optional: python main.py check-data
Run:     python main.py backtest   |   python main.py live   (see RUN.md)

Long book (live): month-end signals; orders at the next open (no look-ahead). See LONG_STRATEGY.md.
Optional `python main.py full` also runs breakout reference strategies (not live).
"""

from pathlib import Path
import numpy as np
import pandas as pd

# ----------------------------- CONFIG ---------------------------------------
START = "2010-01-01"
END = None                      # None = today
BENCHMARK = "^NSEI"             # Nifty 50 index
CACHE_DIR = Path("data_cache")
OUTPUT_BACKTEST = Path("output/backtest")
OUTPUT_LIVE = Path("output/live")
OUTPUT_STUDIES = Path("output/studies")


def _ensure_output_dirs():
    for d in (OUTPUT_BACKTEST, OUTPUT_LIVE, OUTPUT_STUDIES):
        d.mkdir(parents=True, exist_ok=True)

# --- Universe ---
MIN_MCAP_CR = 2000              # tradable only when ESTIMATED mcap on that date >= this (INR crore)
# "point_in_time": tradable when est. mcap + liquidity + listing cooldown pass (see estimate_mcap_cr)
# "today":         stocks with mcap >= MIN_MCAP_CR today are tradable for all history (upper-bound bias)
MCAP_MODE = "point_in_time"
# Download price history for symbols with current mcap >= this (can be below MIN_MCAP_CR so names that
# later grew into the book are not missing). Delisted/extra symbols in delisted_symbols.txt are always tried.
PIT_UNIVERSE_FLOOR_CR = 500
LISTING_COOLDOWN_DAYS = 126     # no entries until this many calendar days after NSE listing date
DELISTED_SYMBOLS_FILE = Path("delisted_symbols.txt")  # optional, one NSE symbol per line (no .NS)
MIN_TURNOVER = 2e7              # INR 2 crore: 20-day median daily traded value needed to enter (liquidity)
MAX_ABS_DAILY_MOVE = 0.60       # drop symbols with a >60% one-day move (usually bad Yahoo data)

# --- Strategy / portfolio ---
INITIAL_CAPITAL = 1_000_000     # INR
MAX_POSITIONS = 10              # breakout reference book only
LONG_POSITIONS = 15             # monthly book: one fifteenth of equity each
RISK_PER_TRADE = 0.01           # 1% of equity risked per trade
MAX_WEIGHT = 0.15               # no position above 15% of equity
ATR_MULT = 3.0                  # trailing stop distance in ATRs
HIGH_WINDOW = 252               # trading days in the 52-week high
BETA_LOOKBACK = 252             # rolling beta vs Nifty 500 for entry filter
BETA_MAX = 1.25                 # require beta < this at signal (when filter enabled)
ATH_PROX_MIN = 0.75             # close >= 75% of series high-water mark (within 25% of ATH)
SECTOR_CAP = 3                  # max open positions in one industry
LONG_BOOK_LEGACY = dict(        # monthly book without the peak stop (for comparison)
    entry_col="entry_long", rank_col="roc126", exit_mode="sma200",
    size_mode="equal", sector_cap=SECTOR_CAP, max_positions=LONG_POSITIONS,
)
LONG_BOOK = {**LONG_BOOK_LEGACY, "dd_stop": 0.30}   # equity rules + peak stop (no cash sleeve)
VOL_LOOKBACK = 20               # days of index realized vol used to scale risk
BENCH_TICKERS = {
    "Nifty 50": ["^NSEI"],
    "Nifty 500": ["^CRSLDX", "^CNX500"],
    "Nifty Smallcap 250": ["NIFTYSMLCAP250.NS", "^CNXSC"],
    "Midcap 100": ["NIFTY_MIDCAP_100.NS", "^CRSMID"],
}
CHART_BENCHMARKS = ("Nifty 500", "Nifty Smallcap 250")
# Fixed colors so equity and drawdown panels match line-for-line.
CHART_SERIES_COLORS = {
    "Long (live)": "#c62828",
    "Long (peak only)": "#ef6c00",
    "Long (no peak stop)": "#6a1b9a",
}
CHART_BENCHMARK_COLORS = {
    "Nifty 500": "#00838f",
    "Nifty Smallcap 250": "#9e9d24",
}
DEFAULT_BO_CHART_COLOR = "#2e7d32"
SECTOR_LIST_URLS = [
    "https://nsearchives.nseindia.com/content/indices/ind_niftytotalmarket_list.csv",
    "https://nsearchives.nseindia.com/content/indices/ind_niftymicrocap250_list.csv",
]
COST_PER_SIDE = 0.0015          # 0.15% per side: brokerage + STT/charges + slippage (small/mid caps cost more)
RISK_FREE = 0.06                # for Sharpe

NSE_LIST_URL = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"


# ----------------------------- UNIVERSE --------------------------------------
def _read_equity_l():
    local = Path("EQUITY_L.csv")
    if not local.exists():
        import requests
        try:
            r = requests.get(NSE_LIST_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
            r.raise_for_status()
            local.write_bytes(r.content)
        except Exception as e:
            raise SystemExit(
                f"Could not download the NSE symbol list ({e}).\n"
                f"Download it manually from {NSE_LIST_URL} and save as EQUITY_L.csv next to this script.")
    df = pd.read_csv(local)
    df.columns = df.columns.str.strip()
    df = df[df["SERIES"].astype(str).str.strip() == "EQ"].copy()
    df["SYMBOL"] = df["SYMBOL"].astype(str).str.strip()
    list_col = next((c for c in df.columns if "DATE OF LISTING" in c.upper()), None)
    if list_col:
        df["listing"] = pd.to_datetime(df[list_col], format="%d-%b-%Y", errors="coerce")
    else:
        df["listing"] = pd.NaT
    return df


def get_nse_symbols():
    """All NSE EQ symbols from EQUITY_L.csv."""
    return sorted(_read_equity_l()["SYMBOL"].unique())


def load_delisted_symbol_list():
    """Optional extra symbols (merged/delisted) to attempt downloading; one symbol per line."""
    if not DELISTED_SYMBOLS_FILE.exists():
        return []
    lines = DELISTED_SYMBOLS_FILE.read_text(encoding="utf-8").splitlines()
    out = []
    for line in lines:
        s = line.split("#")[0].strip().upper()
        if s:
            out.append(s.replace(".NS", ""))
    return sorted(set(out))


def _fetch_yahoo_meta(symbol):
    """Current market cap (INR) and shares outstanding from Yahoo, if available."""
    import yfinance as yf
    t = yf.Ticker(symbol + ".NS")
    mcap, shares = np.nan, np.nan
    try:
        fi = t.fast_info
        for key in ("market_cap", "marketCap"):
            try:
                v = fi[key]
                if v:
                    mcap = float(v)
                    break
            except Exception:
                pass
        for key in ("shares", "sharesOutstanding"):
            try:
                v = fi[key]
                if v:
                    shares = float(v)
                    break
            except Exception:
                pass
    except Exception:
        pass
    try:
        info = t.info
        if not np.isfinite(mcap):
            mcap = float(info.get("marketCap") or np.nan)
        if not np.isfinite(shares):
            shares = float(info.get("sharesOutstanding") or np.nan)
    except Exception:
        pass
    return mcap, shares


def get_universe_meta(symbols, refresh=False):
    """Per-symbol mcap (₹ cr), shares, listing date. Cached in data_cache/universe_meta.csv."""
    from concurrent.futures import ThreadPoolExecutor
    CACHE_DIR.mkdir(exist_ok=True)
    f = CACHE_DIR / "universe_meta.csv"
    eq = _read_equity_l().set_index("SYMBOL")
    listing = eq["listing"].reindex(symbols)
    legacy = CACHE_DIR / "mcap_now.csv"
    if f.exists() and not refresh:
        cached = pd.read_csv(f, index_col=0)
        for col in ("mcap_cr", "shares"):
            if col not in cached.columns:
                cached[col] = np.nan
    elif legacy.exists() and not refresh:
        old = pd.read_csv(legacy, index_col=0).iloc[:, 0]
        cached = pd.DataFrame({"mcap_cr": old, "shares": np.nan})
    else:
        cached = pd.DataFrame(index=symbols, columns=["mcap_cr", "shares"], dtype=float)
    todo = [s for s in symbols if s not in cached.index or not np.isfinite(cached.loc[s, "mcap_cr"])]
    if todo:
        print(f"Fetching market cap / shares for {len(todo)} symbols (cached in universe_meta.csv)...")
        with ThreadPoolExecutor(max_workers=8) as ex:
            rows = list(ex.map(_fetch_yahoo_meta, todo))
        new = pd.DataFrame(
            [(m / 1e7 if np.isfinite(m) else np.nan, sh) for m, sh in rows],
            index=todo, columns=["mcap_cr", "shares"],
        )
        cached = cached.drop(index=[s for s in todo if s in cached.index], errors="ignore")
        cached = pd.concat([cached, new])
        cached.to_csv(f)
    meta = cached.reindex(symbols)
    meta["listing"] = listing.values
    return meta


def get_current_mcaps(symbols, refresh=False):
    """Backward-compatible: mcap in ₹ crore only."""
    return get_universe_meta(symbols, refresh=refresh)["mcap_cr"]


def estimate_mcap_cr(close, mcap_now_cr, shares=None):
    """
    Rough historical market cap (INR crore) without a point-in-time shares database.
    Uses price-scaled today's mcap; when shares are known, also caps with shares×price
    (conservative when today's share count overstates past capitalization).
    """
    if mcap_now_cr is None or not np.isfinite(mcap_now_cr) or mcap_now_cr <= 0:
        return pd.Series(np.nan, index=close.index)
    last = close.iloc[-1]
    if not np.isfinite(last) or last <= 0:
        return pd.Series(np.nan, index=close.index)
    scaled = mcap_now_cr * close / last
    if shares is not None and np.isfinite(shares) and shares > 0:
        from_shares = shares * close / 1e7
        return pd.concat([scaled, from_shares], axis=1).min(axis=1)
    return scaled


# ----------------------------- DATA -----------------------------------------
def _clean(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df[~df.index.duplicated()].sort_index()


def _cache_file(symbol):
    return CACHE_DIR / (symbol.replace("^", "IDX_").replace("&", "_") + ".csv")


def download_one(symbol, start=START, end=END):
    """Single symbol (used for the benchmark). Cached as CSV."""
    import yfinance as yf
    CACHE_DIR.mkdir(exist_ok=True)
    f = _cache_file(symbol)
    if f.exists():
        return pd.read_csv(f, index_col=0, parse_dates=True)
    df = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False)
    if df is None or df.empty:
        return None
    df = _clean(df)
    df.to_csv(f)
    return df


def download_many(symbols, batch=40):
    """Download adjusted daily OHLCV for many NSE symbols in batches; cached per symbol."""
    import yfinance as yf
    CACHE_DIR.mkdir(exist_ok=True)
    yf_syms = {s: s + ".NS" for s in symbols}
    need = [s for s in symbols if not _cache_file(yf_syms[s]).exists()]
    print(f"{len(symbols) - len(need)} cached, downloading {len(need)}...")
    for k in range(0, len(need), batch):
        chunk = need[k:k + batch]
        try:
            raw = yf.download([yf_syms[s] for s in chunk], start=START, end=END, auto_adjust=True,
                              progress=False, group_by="ticker", threads=True)
        except Exception as e:
            print(f"  batch failed: {e}")
            continue
        for s in chunk:
            try:
                df = _clean(raw[yf_syms[s]].copy())
                if len(df):
                    df.to_csv(_cache_file(yf_syms[s]))
            except Exception:
                pass
        print(f"  {min(k + batch, len(need))}/{len(need)}")
    data = {}
    for s in symbols:
        f = _cache_file(yf_syms[s])
        if not f.exists():
            continue
        df = pd.read_csv(f, index_col=0, parse_dates=True)
        if len(df) < 300:
            continue
        if df.Close.pct_change().abs().max() > MAX_ABS_DAILY_MOVE:
            print(f"  dropped {s}: >{MAX_ABS_DAILY_MOVE:.0%} one-day move (likely bad data)")
            continue
        data[s] = df
    print(f"Loaded {len(data)} symbols with usable history")
    return data


def load_universe():
    """
    Download OHLCV for NSE EQ (+ optional delisted_symbols.txt).
    Returns (data, universe_meta DataFrame: mcap_cr, shares, listing).
    Tradable size filter (MIN_MCAP_CR) is applied later in add_indicators, not at download time.
    """
    eq = _read_equity_l()
    symbols = sorted(eq["SYMBOL"].unique())
    extra = load_delisted_symbol_list()
    if extra:
        print(f"Extra delisted / manual symbols: {len(extra)}")
    symbols = sorted(set(symbols) | set(extra))
    print(f"{len(symbols)} NSE EQ-series symbols (+ extras)")
    meta = get_universe_meta(symbols)
    if MCAP_MODE == "today":
        dl_floor = MIN_MCAP_CR
    else:
        dl_floor = PIT_UNIVERSE_FLOOR_CR
    by_mcap = meta["mcap_cr"].dropna()
    download_syms = set(by_mcap[by_mcap >= dl_floor].index)
    download_syms.update(extra)
    print(f"{len(download_syms)} symbols to download (mcap >= {dl_floor:,.0f} cr and/or delisted list)")
    data = download_many(sorted(download_syms))
    meta = meta.reindex(list(data))
    print(f"Universe bias: survivors only (no full NSE delist history). "
          f"PIT mcap uses price-scaled cap + listing dates from EQUITY_L; "
          f"add delisted tickers to {DELISTED_SYMBOLS_FILE.name} if you have price files.")
    return data, meta


# ----------------------------- INDICATORS / SIGNALS --------------------------
def _rolling_sharpe(close, window, rf_daily=None):
    """Annualized Sharpe from daily returns over `window` sessions."""
    rf_daily = RISK_FREE / 252 if rf_daily is None else rf_daily
    r = close.pct_change()
    min_p = max(21, window // 2)
    mu = r.rolling(window, min_periods=min_p).mean()
    sig = r.rolling(window, min_periods=min_p).std()
    return (mu - rf_daily) / sig.replace(0, np.nan) * np.sqrt(252)


def add_indicators(df, mcap_now_cr=None, shares=None, listed_on=None):
    d = df.copy()
    d["sma50"] = d.Close.rolling(50).mean()
    d["sma100"] = d.Close.rolling(100).mean()
    d["sma150"] = d.Close.rolling(150).mean()
    d["sma200"] = d.Close.rolling(200).mean()
    d["mom_smooth"] = (d.Close.pct_change() > 0).rolling(252).mean()
    prev_close = d.Close.shift()
    tr = pd.concat([d.High - d.Low,
                    (d.High - prev_close).abs(),
                    (d.Low - prev_close).abs()], axis=1).max(axis=1)
    d["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    d["roc126"] = d.Close.pct_change(126)
    trend = (d.Close > d.sma200) & (d.sma50 > d.sma200)
    liquid = (d.Close * d.Volume).rolling(20).median() >= MIN_TURNOVER
    ok = liquid
    if listed_on is not None and pd.notna(listed_on):
        cool = pd.Timestamp(listed_on) + pd.Timedelta(days=LISTING_COOLDOWN_DAYS)
        ok = ok & (d.index >= cool)
    if mcap_now_cr is not None and MCAP_MODE == "point_in_time":
        est_mcap = estimate_mcap_cr(d.Close, mcap_now_cr, shares=shares)
        ok = ok & (est_mcap >= MIN_MCAP_CR)
    elif mcap_now_cr is not None and MCAP_MODE == "today":
        ok = ok & (float(mcap_now_cr) >= MIN_MCAP_CR)
    d["high252"] = d.Close.rolling(HIGH_WINDOW).max()
    d["prox"] = d.Close / d["high252"]
    d["prox_ath"] = d.Close / d.Close.cummax()
    d["roc63"] = d.Close.pct_change(63)
    d["roc252"] = d.Close.pct_change(252)
    d["roc_12_1"] = d.Close.shift(21) / d.Close.shift(252) - 1
    for w in (63, 126, 252):
        d[f"sharpe{w}"] = _rolling_sharpe(d.Close, w)
    d["vol126"] = d.Close.pct_change().rolling(126).std() * np.sqrt(252)
    d["mom_adj"] = d["roc_12_1"] / d["vol126"].replace(0, np.nan)
    d["sma200_up"] = d["sma200"] > d["sma200"].shift(63)
    d["sma200_up126"] = d["sma200"] > d["sma200"].shift(126)
    d["tradable"] = ok.fillna(False)
    # Month-end book: buy strength and let it run. The slow exit is applied in the backtest.
    long_base = trend & (d["roc252"] > 0) & d["tradable"]
    d["entry_long"] = long_base & d["sma200_up"] & (d["prox"] >= 0.85)
    d["entry_long90"] = long_base & d["sma200_up"] & (d["prox"] >= 0.90)
    d["entry_long95"] = long_base & d["sma200_up"] & (d["prox"] >= 0.95)
    d["entry_long_s126"] = long_base & d["sma200_up126"] & (d["prox"] >= 0.85)
    vol_surge = d.Volume >= 1.5 * d.Volume.rolling(50).mean()
    d["entry_long_vol"] = d["entry_long"] & vol_surge.fillna(False)
    d["entry_long_smooth"] = d["entry_long"] & (d["mom_smooth"] >= 0.55)
    d["entry_long_quality"] = d["entry_long"] & (d["vol126"] <= d["vol126"].rolling(252).median())
    # Breakout book: a new 52-week closing high, stacked moving averages, and
    # minimum 6- and 12-month returns, but not if price is already 30% above the 50-day.
    prior_high = d.Close.rolling(HIGH_WINDOW).max().shift(1)
    d["signal_bo"] = (
        (d.Close > prior_high)
        & (d.Close > d["sma150"]) & (d["sma150"] > d["sma200"])
        & (d["sma200"] > d["sma200"].shift(20))
        & (d["roc126"] >= 0.15) & (d["roc252"] >= 0.20)
        & (d.Close <= d["sma50"] * 1.30)
        & d["tradable"]
    )
    return d


def rolling_beta_panel(close, bench, lookback=BETA_LOOKBACK):
    """Per-stock rolling beta vs benchmark (same index for all names on each date)."""
    m = bench.reindex(close.index).ffill().pct_change()
    m_var = m.rolling(lookback, min_periods=lookback // 2).var()
    betas = {}
    for col in close.columns:
        r = close[col].pct_change()
        betas[col] = r.rolling(lookback, min_periods=lookback // 2).cov(m) / m_var
    return pd.DataFrame(betas, index=close.index)


def residual_momentum_panel(close, bench, lookback=252, skip=21):
    """Stock return minus beta×index return, summed over lookback-skip, vol-scaled."""
    r = close.pct_change()
    m = bench.reindex(close.index).ffill().pct_change()
    m_var = m.rolling(lookback).var()
    beta = r.apply(lambda s: s.rolling(lookback).cov(m)).div(m_var, axis=0)
    resid = r.sub(beta.shift(1).mul(m, axis=0))
    x = resid.shift(skip)
    n = lookback - skip
    return x.rolling(n).sum() / (x.rolling(n).std() * np.sqrt(n))


def build_panels(data, universe=None, mcaps=None, bench_close=None):
    """universe: DataFrame indexed by symbol with mcap_cr, shares, listing (from load_universe)."""
    if universe is not None:
        meta = universe
    else:
        mcaps = mcaps or {}
        meta = pd.DataFrame({"mcap_cr": pd.Series(mcaps, dtype=float)})
    ind = {}
    for t, df in data.items():
        row = meta.loc[t] if t in meta.index else {}
        if hasattr(row, "get"):
            m = row.get("mcap_cr", np.nan)
            sh = row.get("shares", np.nan)
            listed = row.get("listing", pd.NaT)
        else:
            m, sh, listed = np.nan, np.nan, pd.NaT
        ind[t] = add_indicators(df, m, shares=sh, listed_on=listed)
    tickers = list(ind)
    panel = lambda col: pd.DataFrame({t: ind[t][col] for t in tickers}).sort_index()
    entry_cols = ["entry_long", "entry_long90", "entry_long95", "entry_long_s126", "entry_long_vol",
                  "entry_long_smooth", "entry_long_quality", "signal_bo"]
    P = {c: panel(c) for c in
         ["Open", "Close", "Volume", "sma50", "sma100", "sma150", "sma200", "atr", "roc63", "roc126",
          "roc252", "roc_12_1", "vol126", "mom_adj", "mom_smooth", "prox", "prox_ath",
          "sharpe63", "sharpe126", "sharpe252"] + entry_cols}
    for c in entry_cols:
        P[c] = P[c].fillna(False).astype(bool)
    rank3 = P["roc63"].rank(axis=1, pct=True)
    rank6 = P["roc126"].rank(axis=1, pct=True)
    rank12 = P["roc252"].rank(axis=1, pct=True)
    P["score"] = (rank6 + rank12) / 2.0
    P["score_3_6_12"] = (rank3 + rank6 + rank12) / 3.0
    rs3 = P["sharpe63"].rank(axis=1, pct=True)
    rs6 = P["sharpe126"].rank(axis=1, pct=True)
    rs12 = P["sharpe252"].rank(axis=1, pct=True)
    P["score_sharpe_3_6_12"] = (rs3 + rs6 + rs12) / 3.0
    el = P["entry_long"]
    P["entry_long_ath25"] = el & (P["prox_ath"] >= ATH_PROX_MIN)
    P["entry_long_ath25"] = P["entry_long_ath25"].fillna(False).astype(bool)
    if bench_close is not None:
        b = bench_close.reindex(P["Close"].index).ffill()
        P["beta252"] = rolling_beta_panel(P["Close"], b, BETA_LOOKBACK)
        low_beta = P["beta252"] < BETA_MAX
        P["entry_long_beta125"] = el & low_beta.fillna(False)
        P["entry_long_beta125"] = P["entry_long_beta125"].fillna(False).astype(bool)
        P["entry_long_ath25_beta125"] = (
            el & (P["prox_ath"] >= ATH_PROX_MIN) & low_beta.fillna(False)
        )
        P["entry_long_ath25_beta125"] = P["entry_long_ath25_beta125"].fillna(False).astype(bool)
        rs = P["Close"].div(b, axis=0)
        rs_high = rs.rolling(HIGH_WINDOW).max()
        rs_prox = rs / rs_high.replace(0, np.nan)
        P["entry_long_rs"] = el & (rs_prox >= 0.85)
        P["entry_long_rs"] = P["entry_long_rs"].fillna(False).astype(bool)
        P["resid_mom"] = residual_momentum_panel(P["Close"], b)
        rm = P["resid_mom"].rank(axis=1, pct=True)
        ms = P["mom_smooth"].rank(axis=1, pct=True)
        P["score_resid"] = 0.7 * rm + 0.3 * ms
    else:
        P["entry_long_beta125"] = el & False
        P["entry_long_ath25_beta125"] = P["entry_long_ath25"]
    return tickers, P


# ----------------------------- BACKTEST --------------------------------------
def month_end_mask(dates, rebal_offset_days=0):
    """rebal_offset_days=0 is the last session of the month; 3 is three sessions before that."""
    s = pd.Series(False, index=dates)
    for _, grp in pd.Series(0, index=dates).groupby(dates.to_period("M")):
        ix = grp.index.sort_values()
        k = min(rebal_offset_days, len(ix) - 1)
        s.loc[ix[-(1 + k)]] = True
    return s.to_numpy()


def long_pick_at_signal(i, tickers, sectors, P, entry_col, rank_col, held_j, n_pos, sector_cap):
    """Same candidate walk as backtest() step 4 on month-end index i (signal close)."""
    ENT = P[entry_col].to_numpy()
    RANK = P[rank_col].to_numpy()
    held_j = set(held_j)
    slots = n_pos - len(held_j)
    if slots <= 0:
        return []
    cands = [j for j in range(len(tickers)) if ENT[i, j] and j not in held_j and np.isfinite(RANK[i, j])]
    cands.sort(key=lambda j: -RANK[i, j])
    held_sec = {}
    for j in held_j:
        sec = sectors[j]
        held_sec[sec] = held_sec.get(sec, 0) + 1
    picked = []
    for j in cands:
        if len(picked) >= slots:
            break
        sec = sectors[j]
        if sector_cap and sec and held_sec.get(sec, 0) >= sector_cap:
            continue
        picked.append(j)
        held_sec[sec] = held_sec.get(sec, 0) + 1
    close = P["Close"].iloc[i]
    rows = []
    for j in picked:
        rows.append(dict(
            ticker=tickers[j],
            sector=sectors[j],
            rank_col=rank_col,
            rank_value=float(RANK[i, j]),
            close=float(close.iloc[j]) if j < len(close) else np.nan,
        ))
    return rows


def official_long_next_buys(fresh_account=False, save_csv=True):
    """
    Long rebalance: month-end close -> next session open.
    If the next calendar month-end is not in the price panel yet, prints a provisional
    pick list using the latest close (clearly labelled).
    """
    data, universe = load_universe()
    bench, _ = _load_nifty500_close(None)
    tickers, P = build_panels(data, universe=universe, bench_close=bench)
    sectors = [load_sectors(tickers).get(t, "") for t in tickers]
    kw = adopted_long_kwargs(P)
    entry_col = kw.get("entry_col", "entry_long")
    rank_col = kw.get("rank_col", "roc126")
    n_pos = kw.get("max_positions", LONG_POSITIONS)
    sector_cap = kw.get("sector_cap", SECTOR_CAP)

    dates = P["Close"].index
    me = month_end_mask(dates)
    me_idx = np.flatnonzero(me)
    i_last = len(dates) - 1
    d_last = dates[i_last]

    if fresh_account:
        held_j = []
    else:
        _, tr = backtest(tickers, P, sectors=sectors, **kw)
        held_j = []
        if not tr.empty:
            open_t = set(tr.loc[tr.reason == "open_at_end", "ticker"])
            held_j = [j for j, t in enumerate(tickers) if t in open_t]

    last_me_i = int(me_idx[me_idx <= i_last][-1])
    future_me = me_idx[me_idx > i_last]
    if len(future_me):
        signal_i = int(future_me[0])
        exec_i = signal_i + 1
        exec_date = dates[exec_i] if exec_i < len(dates) else pd.Timestamp(dates[signal_i]) + pd.offsets.BDay(1)
        official = True
        preview_i = signal_i
    else:
        # Current month-end not in panel yet (month still open).
        signal_i = None
        exec_date = None
        official = False
        preview_i = i_last

    picks = long_pick_at_signal(
        preview_i, tickers, sectors, P, entry_col, rank_col, held_j, n_pos, sector_cap,
    )
    slots = n_pos - len(held_j)

    names = _load_nse_company_names()
    for row in picks:
        row["company"] = names.get(row["ticker"], "")

    df = pd.DataFrame(picks)
    print("\n=== Long next buys ===")
    print(f"Live book: {entry_col}, rank {rank_col}, {n_pos} slots, sector cap {sector_cap}")
    print(f"Data through: {d_last.date()}  |  {'Fresh account' if fresh_account else f'Incumbent holdings: {len(held_j)}'}")
    print(f"Free slots at next rebalance: {slots}")
    print(f"Last completed month-end signal: {dates[last_me_i].date()}  (buys filled next open)")
    if official:
        print(f"Next official signal close: {dates[signal_i].date()}  ->  buy open: {exec_date.date()}")
    else:
        month_end_pending = dates[dates.to_period("M") == d_last.to_period("M")][-1]
        print(
            f"Next official signal: last session of {d_last.strftime('%B %Y')} "
            f"(not finalized yet; latest bar {month_end_pending.date()})"
        )
        print("Next official buys: following session open (usually early next month).")
        print(f"Provisional preview below uses close of {dates[preview_i].date()} (NOT the official month-end).")
    if df.empty:
        print("No new buys (no free slots or no entry_long qualifiers).")
    else:
        show = df[["ticker", "company", "sector", "rank_value", "close"]].copy()
        show["rank_value"] = (show["rank_value"] * 100).round(2)
        show = show.rename(columns={"rank_value": f"{rank_col} %", "close": "signal_close"})
        print(show.to_string(index=False))
    if save_csv and not df.empty:
        sig_str = str(dates[signal_i].date()) if signal_i is not None else ""
        exec_str = str(exec_date.date()) if exec_date is not None and hasattr(exec_date, "date") else ""
        df.assign(
            signal_date=sig_str or str(dates[preview_i].date()) + " (preview)",
            exec_date=exec_str or "after month-end",
            official=official,
            fresh_account=fresh_account,
            free_slots=slots,
        ).to_csv(OUTPUT_LIVE / "long_next_buys.csv", index=False)
        print(f"\nSaved {OUTPUT_LIVE / 'long_next_buys.csv'}")

    sig_dt = dates[signal_i] if signal_i is not None else dates[preview_i]
    return df, dict(signal_date=sig_dt, exec_date=exec_date, official=official)


def run_long_next(fresh_account=None):
    import sys
    if fresh_account is None:
        fresh_account = "--fresh" in sys.argv
    _ensure_output_dirs()
    official_long_next_buys(fresh_account=fresh_account)


def run_backtest():
    """Long backtest; writes output/backtest/* (trades, equity curve, chart)."""
    _ensure_output_dirs()
    refresh_equity_chart()


def cli_help():
    print(
        """
Usage:
  python main.py backtest          Run Long backtest -> output/backtest/
  python main.py rolling           Rolling 1–5y return stats -> output/backtest/rolling_returns_summary.csv
  python main.py live            Next Long buy list -> output/live/long_next_buys.csv
  python main.py live --fresh    Same, assuming empty 15-slot book

Research (optional, -> output/studies/):
  python main.py study rank | entry | validate | atr | compare | universe | regime | sep2024

Full strategy grid + breakout sweep (slow):
  python main.py full

Help:
  python main.py help
"""
    )


def backtest(tickers, P, entry_col="entry_long", rank_col="roc126", start=None,
             atr_mult=None, risk_mult=None, slot_cap=None, time_stop=None,
             sectors=None, sector_cap=None, exit_mode="trend_pullback", size_mode="risk",
             max_positions=None, end=None, dd_stop=None, rank_pct=None, rs_min=None,
             regime_off=None, tighten=False, cash_ret=None, vol=None,
             also_trail=False, atr_after=0.0, eq_filter=None, equity_vol_scale=False,
             crash_off=None, reentry_ok=None, crash_scale=1.0,
             rebal_offset_days=0, tiered_exit=False, holdings_vol_target=False,
             cash_ret_idle_only=False, cost_per_side=None, exit_stress_slippage=False,
             risk_per_trade=None):
    """risk_mult and slot_cap are per-date arrays aligned to P['Close'].index.

    risk_mult scales RISK_PER_TRADE (0 blocks the entry). slot_cap replaces
    MAX_POSITIONS on the signal day. Both are read on the signal close, and the
    order fills at the next open.
    crash_off: daily flatten (or partial trim via crash_scale) on held positions.
    reentry_ok: when False, block new entries even on rebalance days.
    """
    dates = P["Close"].index
    O = P["Open"].to_numpy()
    C = P["Close"].ffill().to_numpy()          # forward-filled for valuation
    S50 = P["sma50"].to_numpy()
    S200 = P["sma200"].to_numpy()
    S100 = P["sma100"].to_numpy() if tiered_exit else None
    ATR = P["atr"].ffill().to_numpy()
    RANK = P[rank_col].to_numpy()
    ENT = P[entry_col].to_numpy()
    atr_mult = ATR_MULT if atr_mult is None else atr_mult
    n_pos = MAX_POSITIONS if max_positions is None else max_positions
    sectors = sectors or [""] * len(tickers)
    VOL = None if vol is None else np.asarray(vol, dtype=float)
    month_end = month_end_mask(dates, rebal_offset_days)
    crash_off = None if crash_off is None else np.asarray(crash_off, dtype=bool)
    reentry_ok = None if reentry_ok is None else np.asarray(reentry_ok, dtype=bool)
    cps = COST_PER_SIDE if cost_per_side is None else float(cost_per_side)
    rpt = RISK_PER_TRADE if risk_per_trade is None else float(risk_per_trade)

    n, m = C.shape
    cash = float(INITIAL_CAPITAL)
    pos = {}                 # j -> dict
    pending_entries = []     # list of (j, risk multiplier from the signal day)
    pending_trims = []       # (j, shares, reason) filled at next open
    equity = np.full(n, np.nan)
    equity_liquidation = np.full(n, np.nan)
    trades = []

    i_first = 1
    if start is not None:
        i_first = max(1, int(dates.searchsorted(pd.Timestamp(start))))
    i_last = n - 1
    if end is not None:
        i_last = min(n - 1, int(dates.searchsorted(pd.Timestamp(end), side="right")) - 1)
    equity[i_first - 1] = cash
    equity_liquidation[i_first - 1] = cash
    cash_hist = []
    eq_rets, rv_hist = [], []
    vol_scale_now = 1.0
    h_ewma_var = 0.0
    h_rv_hist = []
    last_vol_trim_i = -999

    def px_open(i, j):
        v = O[i, j]
        return v if not np.isnan(v) else C[i - 1, j]

    def exit_proceeds(i, j, price, shares):
        if exit_stress_slippage and i > i_first:
            prev = C[i - 1, j]
            if np.isfinite(prev) and prev > 0 and np.isfinite(price) and price / prev - 1 < -0.08:
                price *= 1 - 0.03
        return shares * price * (1 - cps)

    def record_exit(j, p, price, shares, reason, exit_i):
        proceeds = exit_proceeds(exit_i, j, price, shares)
        trades.append(dict(
            ticker=tickers[j], entry_date=dates[p["entry_i"]], exit_date=dates[exit_i],
            entry_price=p["entry_price"], exit_price=price, shares=shares,
            pnl=proceeds - shares * p["entry_price"] * (1 + cps),
            reason=reason, sector=sectors[j]))

    for i in range(i_first, i_last + 1):
        if cash_ret is not None and np.isfinite(cash_ret[i]):
            if not cash_ret_idle_only or len(pos) == 0:
                cash *= 1.0 + cash_ret[i]
        for j, n_sell, reason in pending_trims:
            if j not in pos:
                continue
            p = pos[j]
            n_sell = min(int(n_sell), p["shares"])
            if n_sell < 1:
                continue
            price = px_open(i, j)
            cash += exit_proceeds(i, j, price, n_sell)
            record_exit(j, p, price, n_sell, reason, i)
            p["shares"] -= n_sell
            if p["shares"] < 1:
                del pos[j]
        pending_trims = []
        # 1) execute exits flagged yesterday, at today's open
        for j in list(pos):
            p = pos[j]
            if p["exit_reason"]:
                price = px_open(i, j)
                cash += exit_proceeds(i, j, price, p["shares"])
                record_exit(j, p, price, p["shares"], p["exit_reason"], i)
                del pos[j]

        # 2) execute entries flagged yesterday, at today's open
        eq_prev = equity[i - 1]
        for j, mult in pending_entries:
            if j in pos or len(pos) >= n_pos or np.isnan(O[i, j]):
                continue
            if mult <= 0 or not np.isfinite(mult):
                continue
            price = O[i, j]
            atr = ATR[i - 1, j]
            if size_mode in ("equal", "invvol"):
                if size_mode == "invvol" and VOL is not None and np.isfinite(VOL[i - 1, j]) and VOL[i - 1, j] > 0:
                    med = np.nanmedian(VOL[i - 1])
                    raw = (1.0 / n_pos) * (med / VOL[i - 1, j]) if np.isfinite(med) and med > 0 else 1.0 / n_pos
                    budget = min(raw, 0.15) * mult * eq_prev
                else:
                    budget = min(1.0 / n_pos, MAX_WEIGHT) * mult * eq_prev
                shares = int(min(budget, cash) / (price * (1 + cps)))
            else:
                if not np.isfinite(atr) or atr <= 0:
                    continue
                risk_shares = rpt * mult * eq_prev / (atr_mult * atr)
                cap_shares = MAX_WEIGHT * eq_prev / price
                shares = int(min(risk_shares, cap_shares, cash / (price * (1 + cps))))
            if shares < 1:
                continue
            cash -= shares * price * (1 + cps)
            if also_trail and np.isfinite(atr) and atr > 0:
                stop = price if atr_after > 0 else price - atr_mult * atr
            elif size_mode in ("equal", "invvol"):
                stop = price
            else:
                stop = price - atr_mult * atr
            pos[j] = dict(shares=shares, entry_price=price, entry_i=i,
                          high=price, stop=stop, exit_reason=None)
        pending_entries = []

        # 3) end-of-day: update trailing stops, flag exits, mark to market
        for j, p in pos.items():
            c = C[i, j]
            p["high"] = max(p["high"], c)
            if exit_mode != "sma200" and np.isfinite(ATR[i, j]):
                p["stop"] = max(p["stop"], p["high"] - atr_mult * ATR[i, j])
            if exit_mode == "sma200":
                if also_trail and np.isfinite(ATR[i, j]) and ATR[i, j] > 0:
                    if p["high"] / p["entry_price"] - 1 >= atr_after:
                        p["stop"] = max(p["stop"], p["high"] - atr_mult * ATR[i, j])
                        if c < p["stop"]:
                            p["exit_reason"] = "trail_stop"
                if (not p["exit_reason"] and time_stop and (i - p["entry_i"]) >= time_stop
                        and c <= p["entry_price"]):
                    p["exit_reason"] = "time_stop"
                if month_end[i] and not p["exit_reason"]:
                    rank_now = rank_pct[i, j] if rank_pct is not None else np.nan
                    if not np.isnan(S200[i, j]) and c < S200[i, j]:
                        p["exit_reason"] = "below_sma200"
                    elif tiered_exit and S100 is not None and not np.isnan(S100[i, j]) and c < S100[i, j]:
                        sell = p["shares"] // 2
                        if sell >= 1:
                            pending_trims.append((j, sell, "tier_sma100"))
                    elif dd_stop and c < p["high"] * (1 - dd_stop):
                        p["exit_reason"] = "peak_dd"
                    elif rs_min is not None and np.isfinite(rank_now) and rank_now < rs_min:
                        p["exit_reason"] = "rank_drop"
                    elif (tighten and regime_off is not None and regime_off[i]
                          and not np.isnan(S50[i, j]) and c < S50[i, j]):
                        p["exit_reason"] = "regime_sma50"
            elif c < p["stop"]:
                p["exit_reason"] = "trail_stop"
            elif not np.isnan(S50[i, j]) and c < S50[i, j]:
                p["exit_reason"] = "below_sma50"
            elif time_stop and (i - p["entry_i"]) >= time_stop and c <= p["entry_price"]:
                p["exit_reason"] = "time_stop"
        if crash_off is not None and crash_off[i]:
            for j, p in list(pos.items()):
                if p["exit_reason"]:
                    continue
                if crash_scale >= 1.0 - 1e-9:
                    p["exit_reason"] = "regime_exit"
                else:
                    sell = int(p["shares"] * (1.0 - crash_scale))
                    if sell >= p["shares"]:
                        p["exit_reason"] = "regime_exit"
                    elif sell >= 1:
                        pending_trims.append((j, sell, "regime_trim"))
        equity[i] = cash + sum(p["shares"] * C[i, j] for j, p in pos.items())
        liq = cash
        for j, p in pos.items():
            cj = C[i, j]
            if not np.isfinite(cj):
                cj = C[i - 1, j] if i > 0 else p["entry_price"]
            liq += exit_proceeds(i, j, cj, p["shares"])
        equity_liquidation[i] = liq
        if holdings_vol_target and len(pos) > 0 and i > i_first:
            rets = []
            for j in pos:
                if i > 0 and np.isfinite(C[i - 1, j]) and C[i - 1, j] > 0 and np.isfinite(C[i, j]):
                    rets.append(C[i, j] / C[i - 1, j] - 1.0)
            if rets:
                port_r = float(np.mean(rets))
                h_ewma_var = 0.94 * h_ewma_var + 0.06 * port_r ** 2
                rv = float(np.sqrt(h_ewma_var * 252))
                h_rv_hist.append(rv)
                if len(h_rv_hist) >= 252:
                    target = float(np.median(h_rv_hist))
                    w = min(1.0, target / rv) if rv > 0 else 1.0
                    if w < 0.8 and i - last_vol_trim_i >= 5:
                        for j, p in list(pos.items()):
                            sell = int(p["shares"] * (1.0 - w))
                            if sell >= 1:
                                pending_trims.append((j, sell, "vol_trim"))
                        last_vol_trim_i = i
        if equity[i] > 0:
            cash_hist.append(cash / equity[i])
        block_new = False
        if equity_vol_scale and i > i_first and equity[i - 1] > 0 and np.isfinite(equity[i]):
            eq_rets.append(equity[i] / equity[i - 1] - 1)
            if len(eq_rets) >= 20:
                rv = float(np.std(eq_rets[-20:], ddof=1) * np.sqrt(252))
                rv_hist.append(rv)
                vol_scale_now = min(1.0, float(np.median(rv_hist)) / rv) if len(rv_hist) >= 252 and rv > 0 else 1.0
        if eq_filter and i - (i_first - 1) >= 199:
            window = equity[i - 199:i + 1]
            if np.isfinite(window).all() and equity[i] < window.mean():
                block_new = True
                if eq_filter == "flatten":
                    for p in pos.values():
                        if not p["exit_reason"]:
                            p["exit_reason"] = "equity_sma"

        # 4) find new entry candidates for tomorrow
        day_cap = n_pos if slot_cap is None else int(slot_cap[i])
        slots = day_cap - len(pos)
        mult = 1.0 if risk_mult is None else float(risk_mult[i])
        if equity_vol_scale:
            mult *= vol_scale_now
        if exit_mode == "sma200" and not month_end[i]:
            mult = 0.0
        if reentry_ok is not None and not reentry_ok[i]:
            mult = 0.0
        if block_new:
            mult = 0.0
        if slots > 0 and mult > 0:
            cands = [j for j in np.flatnonzero(ENT[i]) if j not in pos and np.isfinite(RANK[i, j])]
            cands.sort(key=lambda j: -RANK[i, j])
            held = {}
            for j in pos:
                held[sectors[j]] = held.get(sectors[j], 0) + 1
            picked = []
            for j in cands:
                if len(picked) >= slots:
                    break
                sec = sectors[j]
                if sector_cap and sec and held.get(sec, 0) >= sector_cap:
                    continue
                picked.append(j)
                held[sec] = held.get(sec, 0) + 1
            pending_entries = [(j, mult) for j in picked]

    for j, p in pos.items():
        price = C[i_last, j]
        if not np.isfinite(price):
            price = p["entry_price"]
        record_exit(j, p, price, p["shares"], "open_at_end", i_last)

    eq = pd.Series(equity, index=dates).iloc[:i_last + 1].ffill()
    eq_liq = pd.Series(equity_liquidation, index=dates).iloc[:i_last + 1].ffill()
    eq.attrs["avg_cash"] = float(np.mean(cash_hist)) if cash_hist else np.nan
    eq.attrs["equity_liquidation"] = eq_liq
    tr = pd.DataFrame(trades)
    if not tr.empty:
        basis = tr.shares * tr.entry_price * (1 + cps)
        tr["ret_pct"] = np.where(basis > 0, tr.pnl / basis * 100, np.nan)
        tr["days_held"] = (tr.exit_date - tr.entry_date).dt.days
    return eq, tr


def backtest_breakout(tickers, P, sectors, regime_ok, atr_mult, exit_mode, market_filter=True,
                       start=None, risk_per_trade=0.005, max_weight=0.10, cooldown=20,
                       size_mode="risk"):
    """52-week-high breakout. Signal at the close, fill at the next open.

    exit_mode "atr" trails. "sma150" uses the initial ATR stop plus two closes
    under the 150-day average. "hybrid" uses both. "sma200" sells only when a
    month ends under the 200-day average. "sma200_hard" adds the initial ATR stop.
    size_mode "risk" risks 0.5% to that stop. "equal" buys about 10% of equity.
    """
    dates = P["Close"].index
    O = P["Open"].to_numpy(dtype=float)
    C = P["Close"].to_numpy(dtype=float)
    Cmark = P["Close"].ffill().to_numpy(dtype=float)
    SMA150 = P["sma150"].to_numpy(dtype=float)
    SMA200 = P["sma200"].to_numpy(dtype=float)
    ATR = P["atr"].to_numpy(dtype=float)
    ENT = P["signal_bo"].to_numpy(dtype=bool)
    SCORE = P["score"].to_numpy(dtype=float)
    n, _ = C.shape
    sectors = sectors or [""] * len(tickers)

    i_first = 1
    if start is not None:
        i_first = max(1, int(dates.searchsorted(pd.Timestamp(start))))
    cash = float(INITIAL_CAPITAL)
    pos = {}
    pending = []
    last_exit = np.full(len(tickers), -10**9, dtype=int)
    equity = np.full(n, np.nan)
    exposure = np.full(n, np.nan)
    equity[i_first - 1] = cash
    exposure[i_first - 1] = 0.0
    trades = []
    month_end = np.zeros(n, dtype=bool)
    month_end[pd.Series(np.arange(n), index=dates).groupby(dates.to_period("M")).max().to_numpy()] = True

    def shut(j, p, i, price, reason, charge_exit):
        nonlocal cash
        px = price * (1 - COST_PER_SIDE) if charge_exit else price
        cash += p["shares"] * px
        entry_cost = p["entry_price"] * (1 + COST_PER_SIDE)
        net_pnl = p["shares"] * (px - entry_cost) if charge_exit else p["shares"] * (price - entry_cost)
        trades.append(dict(
            ticker=tickers[j], entry_date=dates[p["entry_i"]], exit_date=dates[i],
            entry_price=p["entry_price"], exit_price=price, shares=p["shares"],
            pnl=net_pnl, reason=reason,
            max_multiple=p["max_close"] / p["entry_price"]))

    for i in range(i_first, n):
        for j in list(pos):
            p = pos[j]
            if not p["exit_reason"]:
                continue
            price = O[i, j]
            if not np.isfinite(price):
                continue
            shut(j, p, i, price, p["exit_reason"], True)
            del pos[j]
            last_exit[j] = i

        eq_prev = equity[i - 1]
        for j in pending:
            if j in pos or len(pos) >= MAX_POSITIONS or not np.isfinite(O[i, j]):
                continue
            sec = sectors[j]
            if sec and sum(sectors[k] == sec for k in pos) >= SECTOR_CAP:
                continue
            if i - last_exit[j] < cooldown:
                continue
            price = O[i, j]
            atr = ATR[i - 1, j]
            if not np.isfinite(atr) or atr <= 0:
                continue
            if size_mode == "equal":
                budget = min(1.0 / MAX_POSITIONS, max_weight) * eq_prev
                shares = int(min(budget / price, cash / (price * (1 + COST_PER_SIDE))))
            else:
                risk_shares = risk_per_trade * eq_prev / (atr_mult * atr)
                weight_shares = max_weight * eq_prev / price
                shares = int(min(risk_shares, weight_shares, cash / (price * (1 + COST_PER_SIDE))))
            if shares < 1:
                continue
            cash -= shares * price * (1 + COST_PER_SIDE)
            pos[j] = dict(shares=shares, entry_price=price, entry_i=i, max_close=price,
                          stop=price - atr_mult * atr, hard_stop=price - atr_mult * atr,
                          below150=0, exit_reason=None)
        pending = []

        for j, p in pos.items():
            c = C[i, j]
            if not np.isfinite(c):
                continue
            p["max_close"] = max(p["max_close"], c)
            atr_now = ATR[i, j]
            if exit_mode in ("atr", "hybrid"):
                if np.isfinite(atr_now) and atr_now > 0:
                    p["stop"] = max(p["stop"], p["max_close"] - atr_mult * atr_now)
                if c < p["stop"]:
                    p["exit_reason"] = "atr_trail"
                    continue
            elif exit_mode in ("sma150", "sma200_hard") and c < p["hard_stop"]:
                p["exit_reason"] = "hard_stop"
                continue
            sma = SMA150[i, j]
            p["below150"] = p["below150"] + 1 if np.isfinite(sma) and c < sma else 0
            if exit_mode in ("sma150", "hybrid") and p["below150"] >= 2:
                p["exit_reason"] = "below_sma150"
            elif exit_mode in ("sma200", "sma200_hard") and month_end[i]:
                sma200 = SMA200[i, j]
                if np.isfinite(sma200) and c < sma200:
                    p["exit_reason"] = "below_sma200_month_end"

        mtm = sum(p["shares"] * (C[i, j] if np.isfinite(C[i, j]) else Cmark[i, j]) for j, p in pos.items())
        equity[i] = cash + mtm
        if equity[i] > 0:
            exposure[i] = mtm / equity[i]

        if market_filter and not regime_ok[i]:
            continue
        exits_pending = sum(p["exit_reason"] is not None for p in pos.values())
        slots = MAX_POSITIONS - (len(pos) - exits_pending)
        if slots <= 0:
            continue
        cands = [j for j in np.flatnonzero(ENT[i])
                 if j not in pos and i - last_exit[j] >= cooldown and np.isfinite(SCORE[i, j])]
        cands.sort(key=lambda j: -SCORE[i, j])
        held = {}
        for j, p in pos.items():
            if sectors[j] and p["exit_reason"] is None:
                held[sectors[j]] = held.get(sectors[j], 0) + 1
        picked = []
        for j in cands:
            if len(picked) >= slots:
                break
            sec = sectors[j]
            if sec and held.get(sec, 0) >= SECTOR_CAP:
                continue
            picked.append(j)
            if sec:
                held[sec] = held.get(sec, 0) + 1
        pending = picked

    for j, p in pos.items():
        price = C[-1, j] if np.isfinite(C[-1, j]) else Cmark[-1, j]
        shut(j, p, n - 1, price, "open_at_end", True)
    eq = pd.Series(equity, index=dates).ffill()
    eq.attrs["exposure"] = pd.Series(exposure, index=dates).ffill()
    tr = pd.DataFrame(trades)
    if not tr.empty:
        basis = tr.shares * tr.entry_price * (1 + COST_PER_SIDE)
        tr["ret_pct"] = np.where(basis > 0, tr.pnl / basis * 100, np.nan)
        tr["days_held"] = (tr.exit_date - tr.entry_date).dt.days
        tr["reached_2x"] = tr["max_multiple"] >= 2
        tr["reached_3x"] = tr["max_multiple"] >= 3
    return eq, tr


# ----------------------------- METRICS ---------------------------------------
def perf_stats(eq, name="Strategy"):
    eq = eq.dropna()
    r = eq.pct_change().dropna()
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1
    vol = r.std() * np.sqrt(252)
    sharpe = (r.mean() * 252 - RISK_FREE) / vol if vol > 0 else np.nan
    dd = eq / eq.cummax() - 1
    return {"name": name, "CAGR %": cagr * 100, "Vol %": vol * 100, "Sharpe": sharpe,
            "Max DD %": dd.min() * 100, "Calmar": cagr / abs(dd.min()) if dd.min() < 0 else np.nan,
            "Final value": eq.iloc[-1]}


def trade_stats(tr):
    if tr.empty:
        return {}
    wins, losses = tr[tr.pnl > 0], tr[tr.pnl <= 0]
    return {"Trades": len(tr),
            "Win rate %": len(wins) / len(tr) * 100,
            "Avg win %": wins.ret_pct.mean() if len(wins) else 0,
            "Avg loss %": losses.ret_pct.mean() if len(losses) else 0,
            "Profit factor": wins.pnl.sum() / abs(losses.pnl.sum()) if len(losses) and losses.pnl.sum() != 0 else np.nan,
            "Avg days held": tr.days_held.mean()}


def yearly_returns(eq):
    return (eq.resample("A").last().pct_change() * 100).round(1)


TRADING_DAYS_PER_YEAR = 252


def rolling_window_returns(eq, years):
    """Total return % over each rolling N-year window (252 sessions per year)."""
    w = int(TRADING_DAYS_PER_YEAR * years)
    s = eq.dropna()
    if len(s) <= w:
        return pd.Series(dtype=float)
    return (s / s.shift(w) - 1.0) * 100.0


def _annualized_from_total_pct(total_pct, years):
    return ((1.0 + total_pct / 100.0) ** (1.0 / years) - 1.0) * 100.0


def summarize_rolling_windows(eq, years, name):
    raw = rolling_window_returns(eq, years)
    if raw.empty:
        return None
    ann = raw.apply(lambda x: _annualized_from_total_pct(x, years))
    pct = lambda q: float(np.nanpercentile(raw, q))
    return {
        "Strategy": name,
        "Horizon (y)": years,
        "Windows": len(raw),
        "Total % median": round(raw.median(), 2),
        "Total % mean": round(raw.mean(), 2),
        "Total % p10": round(pct(10), 2),
        "Total % p25": round(pct(25), 2),
        "Total % p75": round(pct(75), 2),
        "Total % p90": round(pct(90), 2),
        "Total % best": round(raw.max(), 2),
        "Total % worst": round(raw.min(), 2),
        "Win %": round((raw > 0).mean() * 100, 1),
        "CAGR % median": round(ann.median(), 2),
        "CAGR % mean": round(ann.mean(), 2),
    }


def run_rolling_returns():
    """Rolling 1–5y total and annualized returns for Long (live) and Nifty 500."""
    _ensure_output_dirs()
    data, universe = load_universe()
    bench, _ = _load_nifty500_close(None)
    tickers, P = build_panels(data, universe=universe, bench_close=bench)
    sectors = [load_sectors(tickers).get(t, "") for t in tickers]
    eq_long, _ = backtest(tickers, P, sectors=sectors, **adopted_long_kwargs(P))
    eq_long = eq_long.dropna()
    idx = eq_long.index
    benches = load_benchmarks()
    n500 = benches.get("Nifty 500")
    if n500 is not None:
        n500 = n500.reindex(idx).ffill()
        n500_eq = n500 / n500.dropna().iloc[0] * INITIAL_CAPITAL
    else:
        n500_eq = None

    horizons = (1, 2, 3, 4, 5)
    rows = []
    series_map = {"Long (live)": eq_long}
    if n500_eq is not None:
        series_map["Nifty 500"] = n500_eq
    for name, eq in series_map.items():
        for y in horizons:
            row = summarize_rolling_windows(eq, y, name)
            if row:
                rows.append(row)
    table = pd.DataFrame(rows)
    out = OUTPUT_BACKTEST / "rolling_returns_summary.csv"
    table.to_csv(out, index=False)
    print("\n=== Rolling holding-period returns (252 trading days per year) ===")
    print("Total return % over each overlapping window; CAGR columns annualize that total.")
    print(table.to_string(index=False))
    print(f"\nSaved {out}")
    # Latest window ending on last data date
    print("\n=== Latest window (ends on last backtest date) ===")
    last_rows = []
    for name, eq in series_map.items():
        for y in horizons:
            raw = rolling_window_returns(eq, y)
            if raw.empty:
                continue
            v = raw.iloc[-1]
            last_rows.append({
                "Strategy": name, "Horizon (y)": y,
                "Total %": round(v, 2),
                "CAGR %": round(_annualized_from_total_pct(v, y), 2),
            })
    print(pd.DataFrame(last_rows).to_string(index=False))
    return table


def hold_profile(tr, years):
    """How long capital sits, and how many closed trades actually multiplied."""
    if tr.empty:
        return {}
    closed = tr[tr.reason != "open_at_end"]
    d = closed.days_held if len(closed) else tr.days_held
    return {"Trades": len(tr),
            "Per year": len(closed) / years if years else np.nan,
            "Median days": d.median() if len(d) else np.nan,
            "Avg days": d.mean() if len(d) else np.nan,
            "Held >= 1y %": (d >= 365).mean() * 100 if len(d) else np.nan,
            "2x closed": int((closed.ret_pct >= 100).sum()),
            "3x closed": int((closed.ret_pct >= 200).sum()),
            "Best closed %": closed.ret_pct.max() if len(closed) else np.nan}


# ----------------------------- MAIN ------------------------------------------
def _summary_row(name, eq, tr):
    p = perf_stats(eq, name)
    t = trade_stats(tr)
    exp = eq.attrs.get("exposure")
    avg_exp = exp.dropna().mean() * 100 if isinstance(exp, pd.Series) and not exp.dropna().empty else np.nan
    med_exp = exp.dropna().median() * 100 if isinstance(exp, pd.Series) and not exp.dropna().empty else np.nan
    return {"CAGR %": p["CAGR %"], "Max DD %": p["Max DD %"], "Sharpe": p["Sharpe"],
            "Calmar": p["Calmar"], "Final": p["Final value"], "Trades": t.get("Trades", 0),
            "Win %": t.get("Win rate %", np.nan), "PF": t.get("Profit factor", np.nan),
            "Avg hold": t.get("Avg days held", np.nan),
            "Avg invested %": avg_exp, "Median invested %": med_exp}


def _jackpot(tr, k=10):
    top = tr.nlargest(min(k, len(tr)), "pnl")
    rest = tr.drop(top.index)
    win = rest.loc[rest.pnl > 0, "pnl"].sum()
    loss = rest.loc[rest.pnl <= 0, "pnl"].sum()
    return {"PnL all": tr.pnl.sum(), "Top 10 PnL": top.pnl.sum(),
            "PnL ex top 10": rest.pnl.sum(),
            "PF ex top 10": win / abs(loss) if loss else np.nan}


def load_sectors(tickers):
    """Industry from NSE index constituent files, cached. Missing symbols stay uncapped."""
    CACHE_DIR.mkdir(exist_ok=True)
    cache = CACHE_DIR / "sectors.csv"
    if not cache.exists():
        import io
        import requests
        frames = []
        for url in SECTOR_LIST_URLS:
            try:
                r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
                r.raise_for_status()
                df = pd.read_csv(io.BytesIO(r.content))
                df.columns = [str(c).strip() for c in df.columns]
                sym = next(c for c in df.columns if c.lower().replace(" ", "") == "symbol")
                ind = next(c for c in df.columns if "industry" in c.lower())
                part = df[[sym, ind]].dropna()
                part.columns = ["symbol", "industry"]
                frames.append(part)
            except Exception as e:
                print(f"  sector list skipped ({url.split('/')[-1]}): {e}")
        if not frames:
            return {t: "" for t in tickers}
        out = pd.concat(frames, ignore_index=True)
        out["symbol"] = out["symbol"].astype(str).str.strip()
        out["industry"] = out["industry"].astype(str).str.strip()
        out.drop_duplicates("symbol").to_csv(cache, index=False)
    df = pd.read_csv(cache)
    mapping = dict(zip(df.symbol.astype(str), df.industry.astype(str)))
    return {t: mapping.get(t, "") for t in tickers}


def load_benchmarks():
    """Close series for Nifty 50, Nifty 500 and Midcap 100. First ticker with history wins."""
    out = {}
    for name, tickers in BENCH_TICKERS.items():
        for ticker in tickers:
            try:
                df = download_one(ticker)
            except Exception as e:
                print(f"  {name}: {ticker} failed ({e})")
                continue
            if df is None or len(df) < 252:
                print(f"  {name}: {ticker} unavailable")
                continue
            out[name] = df.Close.copy()
            print(f"  {name} ({ticker}): {df.index.min().date()} → {df.index.max().date()} ({len(df)} bars)")
            break
    return out


def breadth_regime(P, off=0.35, on=0.50, smooth=5):
    """True = risk on: fraction of names above their 200-day average, with hysteresis."""
    c, s = P["Close"], P["sma200"]
    valid = c.notna() & s.notna()
    b = (c > s).where(valid).mean(axis=1).rolling(smooth, min_periods=1).mean()
    state, out = True, []
    for v in b.to_numpy():
        if np.isfinite(v):
            if state and v < off:
                state = False
            elif not state and v > on:
                state = True
        out.append(state)
    return np.array(out, dtype=bool)


def index_crash_regime(bench_close, dates, dd_off=0.15, dd_on=0.10):
    """Risk off when index is >dd_off below 252d high and under 50-day SMA; back on above SMA50 and DD < dd_on."""
    s = bench_close.reindex(dates).ffill()
    sma50 = s.rolling(50).mean()
    high252 = s.rolling(252).max()
    dd = 1.0 - s / high252
    state, out = True, []
    for c, ma, d in zip(s.to_numpy(), sma50.to_numpy(), dd.to_numpy()):
        if not (np.isfinite(c) and np.isfinite(ma) and np.isfinite(d)):
            out.append(state)
            continue
        if state and d > dd_off and c < ma:
            state = False
        elif not state and c > ma and d < dd_on:
            state = True
        out.append(state)
    return np.array(out, dtype=bool)


def _load_nifty500_close(dates=None):
    for ticker in ("^CRSLDX", "^CNX500", "^NSEI"):
        df = download_one(ticker)
        if df is not None and len(df) >= 252:
            return df.Close, ticker
    return None, ""


def _slice_dd(eq, start, end):
    sub = eq.loc[start:end].dropna()
    if len(sub) < 2:
        return np.nan
    return float((sub / sub.cummax() - 1).min() * 100)


def cash_sleeve_returns(dates, debt_yield=0.065):
    """Gold when GOLDBEES > 200dma (signal lagged 1 day), else flat debt yield."""
    gold = download_one("GOLDBEES.NS")
    if gold is None or len(gold) < 252:
        return None
    g = gold.Close.reindex(dates).ffill()
    gold_ret = g.pct_change().fillna(0).to_numpy()
    gold_on = (g > g.rolling(200).mean()).shift(1).fillna(False).to_numpy()
    debt_ret = np.full(len(dates), debt_yield / 252)
    return np.where(gold_on, gold_ret, debt_ret).astype(float)


def index_below_sma200(bench_close, dates, lag=1):
    """True when benchmark close was below its 200-day SMA (signal lagged `lag` sessions)."""
    s = bench_close.reindex(dates).ffill()
    sma200 = s.rolling(200, min_periods=200).mean()
    return (s < sma200).shift(lag).fillna(False).to_numpy(dtype=bool)


def index_choppy(bench_close, dates, window=20, lookback=126, lag=1):
    """True when recent index vol exceeds its trailing median (lagged)."""
    s = bench_close.reindex(dates).ffill()
    r = s.pct_change()
    vol = r.rolling(window, min_periods=window).std() * np.sqrt(252)
    med = vol.rolling(lookback, min_periods=lookback // 2).median()
    return (vol > med).shift(lag).fillna(False).to_numpy(dtype=bool)


def cash_sleeve_with_regime(dates, force_gold_mask, debt_yield=0.065):
    """Base gold/debt sleeve; on force_gold days cash earns GOLDBEES return only."""
    base = cash_sleeve_returns(dates, debt_yield)
    if base is None:
        return None
    if force_gold_mask is None:
        return base
    gold = download_one("GOLDBEES.NS")
    if gold is None or len(gold) < 2:
        return base
    g = gold.Close.reindex(dates).ffill()
    gold_ret = g.pct_change().fillna(0).to_numpy()
    mask = np.asarray(force_gold_mask, dtype=bool)
    return np.where(mask, gold_ret, base).astype(float)


def adopted_long_kwargs(P):
    """Live Long: peak stop + gold/debt return on all cash daily."""
    kw = dict(LONG_BOOK)
    sleeve = cash_sleeve_returns(P["Close"].index)
    if sleeve is not None:
        kw["cash_ret"] = sleeve
        kw["cash_ret_idle_only"] = False
    return kw


def industry_rank_panel(P, sectors, lookback=126):
    """Per-stock percentile rank of its industry's median lookback return."""
    r = P["Close"].pct_change(lookback)
    sec = pd.Series(sectors, index=r.columns)
    med = r.T.groupby(sec).median().T
    rk = med.rank(axis=1, pct=True)
    out = pd.DataFrame(np.nan, index=r.index, columns=r.columns)
    for s in rk.columns:
        if not s:
            continue
        cols = sec.index[sec == s]
        out.loc[:, cols] = rk[s].to_numpy()[:, None]
    return out


def augment_long_entries(P, sectors):
    """Add entry_long_ind (industry momentum filter) to panel dict P."""
    ir = industry_rank_panel(P, sectors)
    P["entry_long_ind"] = P["entry_long"] & (ir.isna() | (ir >= 0.5))
    P["entry_long_ind"] = P["entry_long_ind"].fillna(False).astype(bool)


def tranched_long(tickers, P, sectors, offsets=(0, 5, 10, 15), **kw):
    """Same book on staggered month-end offsets; return equal-weight of normalized curves."""
    curves = []
    for off in offsets:
        eq, _ = backtest(tickers, P, sectors=sectors, rebal_offset_days=off, **kw)
        eq = eq.dropna()
        curves.append(eq / eq.iloc[0])
        print(f"  tranche offset {off}: {perf_stats(eq, str(off))['CAGR %']:.1f}% CAGR, "
              f"{perf_stats(eq, str(off))['Max DD %']:.1f}% DD")
    avg = pd.concat(curves, axis=1).mean(axis=1) * INITIAL_CAPITAL
    return avg, perf_stats(avg, "tranched")


def hedge_crash_returns(eq, bench_close, index_risk_on):
    """Post-hoc: short index beta when crash regime off; +0.05% turnover cost on hedge changes."""
    r = eq.pct_change().fillna(0)
    m = bench_close.reindex(eq.index).ffill().pct_change().fillna(0)
    h = pd.Series(~index_risk_on, index=eq.index).shift(1).fillna(False).astype(float)
    beta = (r.rolling(126).cov(m) / m.rolling(126).var()).shift(1).clip(0.5, 1.5).fillna(1.0)
    hedged_r = r - h * beta * m - h.diff().abs().fillna(0) * 0.0005
    out = (1 + hedged_r).cumprod() * eq.iloc[0]
    out.iloc[0] = eq.iloc[0]
    return out


def _row_from_backtest(label, tickers, P, sectors, kw):
    eq, tr = backtest(tickers, P, sectors=sectors, **kw)
    p = perf_stats(eq.dropna(), label)
    return dict(
        Scenario=label,
        Symbols=len(tickers),
        **{k: round(v, 2) if isinstance(v, (float, np.floating)) else v for k, v in p.items() if k != "name"},
        Trades=len(tr),
    )


def survivorship_sanity(tickers, P, sectors, data=None, universe=None):
    """
    Compare live Long under PIT rules vs biased 'today' universe on the same prices,
    plus a random symbol subsample. Delisted names are still missing unless listed in
    delisted_symbols.txt with cached prices.
    """
    global MCAP_MODE
    rows = []
    kw = adopted_long_kwargs(P)
    rows.append(_row_from_backtest("Long PIT (live rules)", tickers, P, sectors, kw))

    if data is not None and universe is not None:
        saved = MCAP_MODE
        try:
            MCAP_MODE = "today"
            _, P_today = build_panels(data, universe=universe)
            t2 = list(P_today["Close"].columns)
            s2 = [load_sectors(t2).get(t, "") for t in t2]
            rows.append(_row_from_backtest(
                "Long MCAP_MODE=today (biased upper bound)", t2, P_today, s2, adopted_long_kwargs(P_today),
            ))
        finally:
            MCAP_MODE = saved

        import random
        rng = random.Random(42)
        cagrs = []
        for _ in range(3):
            sub = rng.sample(tickers, max(1, int(len(tickers) * 0.75)))
            cols = [c for c in P["Close"].columns if c in sub]
            Psub = {k: (v[cols] if hasattr(v, "columns") else v) for k, v in P.items()}
            sec_map = dict(zip(tickers, sectors))
            sec_sub = [sec_map.get(c, "") for c in cols]
            eq, _ = backtest(cols, Psub, sectors=sec_sub, **kw)
            cagrs.append(perf_stats(eq.dropna(), "")["CAGR %"])
        rows.append({
            "Scenario": "Long PIT, random 75% symbols (median CAGR, 3 seeds)",
            "Symbols": int(len(tickers) * 0.75),
            "CAGR %": round(float(np.median(cagrs)), 2),
            "Max DD %": np.nan,
            "Calmar": np.nan,
            "Sharpe": np.nan,
            "Final value": np.nan,
            "Trades": np.nan,
        })

    benches = load_benchmarks()
    ref_eq = backtest(tickers, P, sectors=sectors, **kw)[0].dropna()
    for bname in ("Nifty 500", "Nifty 50"):
        if bname not in benches:
            continue
        pair = _same_window(ref_eq, benches[bname])
        if not pair:
            continue
        _, b = pair
        p = perf_stats(b, bname)
        rows.append(dict(
            Scenario=f"{bname} B&H (same window as Long)",
            Symbols=np.nan,
            **{k: round(v, 2) if isinstance(v, (float, np.floating)) else v for k, v in p.items() if k != "name"},
            Trades=np.nan,
        ))

    table = pd.DataFrame(rows)
    print(f"\n=== Universe / survivorship check (download floor {PIT_UNIVERSE_FLOOR_CR} cr, "
          f"tradable >= {MIN_MCAP_CR} cr, listing cooldown {LISTING_COOLDOWN_DAYS}d) ===")
    print(table.to_string(index=False))
    _ensure_output_dirs()
    table.to_csv(OUTPUT_STUDIES / "universe_bias_report.csv", index=False)
    print(f"Saved {OUTPUT_STUDIES / 'universe_bias_report.csv'}")
    print("Limitations: no historical NSE delist database; mcap is estimated (not true PIT shares); "
          "today-universe row is an optimistic upper bound.")
    return backtest(tickers, P, sectors=sectors, **kw)


def regime_cash_study(tickers, P, sectors):
    """
    Test Nifty 500 below 200 SMA and choppy regimes: force gold on cash, block entries, or flatten.
    Adopt only if _helps vs live Long on 2010-18 and 2019+.
    """
    dates = P["Close"].index
    bench, bench_name = _load_nifty500_close(dates)
    if bench is None:
        raise SystemExit("Need Nifty 500 benchmark.")
    below200 = index_below_sma200(bench, dates, lag=1)
    choppy = index_choppy(bench, dates, lag=1)
    bad = below200 & choppy
    risk_on = ~below200

    print(f"\n=== Index regime ({bench_name}, lag 1d) ===")
    print(f"  Days below 200 SMA: {below200.mean():.1%}")
    print(f"  Days choppy (20d vol > 126d median): {choppy.mean():.1%}")
    print(f"  Days below 200 AND choppy: {bad.mean():.1%}")
    i_last = len(dates) - 1
    print(f"  As of {dates[i_last].date()}: below 200={bool(below200[i_last])}  "
          f"choppy={bool(choppy[i_last])}  bad={bool(bad[i_last])}")
    print("  Live book: stocks unchanged by index; idle cash uses GOLDBEES/debt sleeve only.")

    base_kw = adopted_long_kwargs(P)
    eq_base, _ = backtest(tickers, P, sectors=sectors, **base_kw)
    eq_base = eq_base.dropna()
    br = eq_base.pct_change().fillna(0)
    m = bench.reindex(eq_base.index).ffill().pct_change().reindex(eq_base.index).fillna(0)
    b200 = pd.Series(below200, index=dates).reindex(eq_base.index).fillna(False)
    bch = pd.Series(choppy, index=dates).reindex(eq_base.index).fillna(False)
    bb = b200 & bch
    print("\n=== Live Long today (no index rule on stocks) — daily return context ===")
    for label, mask in [
        ("Index above 200 SMA", ~b200),
        ("Index below 200 SMA", b200),
        ("Below 200 + choppy", bb),
    ]:
        if mask.sum() < 5:
            continue
        print(f"  {label}: n={int(mask.sum())}  Long avg daily {br[mask].mean()*100:.3f}%  "
              f"N500 avg daily {m[mask].mean()*100:.3f}%")

    flat = np.full(len(dates), 0.065 / 252)
    sleeve_base = cash_sleeve_returns(dates)
    specs = [
        ("Long (live baseline)", {}),
        ("cash: force gold when below 200", dict(
            cash_ret=cash_sleeve_with_regime(dates, below200),
            cash_ret_idle_only=False)),
        ("cash: force gold when below 200 + choppy", dict(
            cash_ret=cash_sleeve_with_regime(dates, bad),
            cash_ret_idle_only=False)),
        ("cash: force gold when below 200 OR choppy", dict(
            cash_ret=cash_sleeve_with_regime(dates, below200 | choppy),
            cash_ret_idle_only=False)),
        ("cash: debt only on bad (below+choppy), else live sleeve", dict(
            cash_ret=np.where(bad, flat, sleeve_base) if sleeve_base is not None else flat,
            cash_ret_idle_only=False)),
        ("block new buys when below 200", dict(reentry_ok=risk_on)),
        ("block new buys when below 200 + choppy", dict(reentry_ok=~bad)),
        ("flatten stocks when below 200 + choppy", dict(
            crash_off=bad, reentry_ok=~bad, crash_scale=1.0)),
        ("flatten when below 200 (any)", dict(
            crash_off=below200, reentry_ok=risk_on, crash_scale=1.0)),
    ]
    specs.append(("flatten bad + gold cash on bad days", dict(
        crash_off=bad, reentry_ok=~bad,
        cash_ret=np.where(bad, cash_sleeve_with_regime(dates, np.ones(len(dates), dtype=bool)),
                          sleeve_base) if sleeve_base is not None else flat,
        cash_ret_idle_only=False)))

    windows = [
        ("2010-2018", dict(start="2010-01-01", end="2018-12-31")),
        ("2019+", dict(start="2019-01-01")),
        ("Full", {}),
    ]
    base_half = []
    rows = []
    for wlabel, wkw in windows:
        if wlabel == "Full":
            continue
        eq, tr = backtest(tickers, P, sectors=sectors, **{**base_kw, **wkw})
        base_half.append(_window_stats(eq, tr))
    for name, extra in specs:
        row = {"Variant": name}
        for wlabel, wkw in windows:
            kw = {**base_kw, **extra, **wkw}
            eq, tr = backtest(tickers, P, sectors=sectors, **kw)
            st = _window_stats(eq, tr)
            row[f"{wlabel} CAGR"] = round(st["CAGR"], 2)
            row[f"{wlabel} DD"] = round(st["DD"], 2)
            row[f"{wlabel} Calmar"] = round(st["Calmar"], 3)
            if wlabel == "Full":
                row["Final"] = round(perf_stats(eq, "")["Final value"])
        ok = _helps(
            (base_half[0], {"Calmar": row["2010-2018 Calmar"], "CAGR": row["2010-2018 CAGR"]}),
            (base_half[1], {"Calmar": row["2019+ Calmar"], "CAGR": row["2019+ CAGR"]}),
        ) if name != "Long (live baseline)" else False
        row["Keep vs live"] = "yes" if ok else ""
        rows.append(row)
    table = pd.DataFrame(rows)
    _ensure_output_dirs()
    path = OUTPUT_STUDIES / "regime_cash_study.csv"
    table.to_csv(path, index=False)
    print("\n=== Regime / gold / cash variants (one change vs live Long) ===")
    print(table.to_string(index=False))
    print(f"\nSaved {path}")
    best = table[table["Keep vs live"] == "yes"]
    if not best.empty:
        i = best["Full Calmar"].idxmax() if "Full Calmar" in best.columns else best.index[0]
        winner = best.loc[i, "Variant"] if len(best) else None
        print(f"\nCandidate passing both halves: {winner}")
    else:
        print("\nNo variant beat live Long on both halves (Calmar up, CAGR within 5pts). Live book unchanged.")
    return table


def run_regime_cash_study():
    pd.set_option("display.width", 240)
    data, universe = load_universe()
    bench, _ = _load_nifty500_close(None)
    tickers, P = build_panels(data, universe=universe, bench_close=bench)
    sectors = [load_sectors(tickers).get(t, "") for t in tickers]
    return regime_cash_study(tickers, P, sectors)


def compare_long_improvements(tickers, P, sectors, save_csv=True):
    """Adopted Long (LONG_BOOK) vs one-change variants. Prints table and optional CSV."""
    augment_long_entries(P, sectors)
    dates = P["Close"].index
    bench, bench_name = _load_nifty500_close(dates)
    if bench is None:
        raise SystemExit("Need a benchmark index for regime signals.")
    breadth_on = breadth_regime(P)
    index_on = index_crash_regime(bench, dates)
    combined_on = breadth_on & index_on
    crash_breadth = ~breadth_on
    crash_index = ~index_on
    crash_combined = ~(breadth_on & index_on)
    sleeve = cash_sleeve_returns(dates)
    flat_ret = np.full(len(dates), 0.065 / 252)

    base = adopted_long_kwargs(P)
    variants = [
        ("Long (adopted)", {}),
        ("0 exit stress gap>8%", dict(exit_stress_slippage=True)),
        ("1 breadth crash exit", dict(crash_off=crash_breadth, reentry_ok=breadth_on)),
        ("1 index crash exit", dict(crash_off=crash_index, reentry_ok=index_on)),
        ("1 index crash zero cost", dict(crash_off=crash_index, reentry_ok=index_on, cost_per_side=0)),
        ("1 combined crash exit", dict(crash_off=crash_combined, reentry_ok=combined_on)),
        ("2 rank roc_12_1", dict(rank_col="roc_12_1")),
        ("2 rank prox", dict(rank_col="prox")),
        ("2 rank resid_mom", dict(rank_col="resid_mom") if "resid_mom" in P else None),
        ("2 rank mom_adj", dict(rank_col="mom_adj")),
        ("3 holdings vol trim", dict(holdings_vol_target=True)),
        ("4 no cash sleeve", {"cash_ret": None, "cash_ret_idle_only": False}),
        ("4 cash 6.5% only (no gold)", dict(cash_ret=flat_ret, cash_ret_idle_only=False)),
        ("5 entry industry top half", dict(entry_col="entry_long_ind")),
        ("5 entry RS vs N500", dict(entry_col="entry_long_rs") if "entry_long_rs" in P else None),
        ("6 tiered SMA100/200", dict(tiered_exit=True)),
    ]
    windows = [("Full", {}), ("2010-18", dict(start="2010-01-01", end="2018-12-31")), ("2019+", dict(start="2019-01-01"))]

    rows = []
    stored_full = {}
    for name, extra in variants:
        if extra is None:
            continue
        kw = {**base, **extra}
        stats = {}
        for wlabel, wkw in windows:
            eq, tr = backtest(tickers, P, sectors=sectors, **{**kw, **wkw})
            p = perf_stats(eq, name)
            closed = tr[tr.reason != "open_at_end"] if not tr.empty else tr
            n_reg = int((closed.reason == "regime_exit").sum()) if not closed.empty else 0
            stats[wlabel] = {
                "CAGR": p["CAGR %"], "DD": p["Max DD %"], "Calmar": p["Calmar"],
                "Final": p["Final value"], "Trades": len(tr),
                "regime_exits": n_reg, "avg_cash": eq.attrs.get("avg_cash", np.nan),
            }
            if wlabel == "Full":
                stored_full[name] = eq
                stats[wlabel]["DD_2018"] = _slice_dd(eq, "2018-01-01", "2019-12-31")
                stats[wlabel]["DD_2020"] = _slice_dd(eq, "2020-01-01", "2020-12-31")
        b18 = stats["2010-18"]
        b19 = stats["2019+"]
        rows.append({
            "Variant": name,
            "CAGR": stats["Full"]["CAGR"], "Max DD": stats["Full"]["DD"],
            "Calmar": stats["Full"]["Calmar"], "Final": stats["Full"]["Final"],
            "DD 18-19": stats["Full"]["DD_2018"], "DD 2020": stats["Full"]["DD_2020"],
            "2010-18 CAGR": b18["CAGR"], "2010-18 DD": b18["DD"], "2010-18 Calmar": b18["Calmar"],
            "2019+ CAGR": b19["CAGR"], "2019+ DD": b19["DD"], "2019+ Calmar": b19["Calmar"],
            "Trades": stats["Full"]["Trades"], "Regime exits": stats["Full"]["regime_exits"],
            "Avg cash": stats["Full"]["avg_cash"],
        })

    # fix Keep column vs baseline
    base_stats = None
    for r in rows:
        if r["Variant"] == "Long (adopted)":
            base_stats = r
            r["Keep vs baseline"] = ""
            break
    if base_stats:
        for r in rows:
            if r["Variant"] == "Long (adopted)":
                continue
            ok = _helps(
                ({"Calmar": base_stats["2010-18 Calmar"], "CAGR": base_stats["2010-18 CAGR"]},
                 {"Calmar": r["2010-18 Calmar"], "CAGR": r["2010-18 CAGR"]}),
                ({"Calmar": base_stats["2019+ Calmar"], "CAGR": base_stats["2019+ CAGR"]},
                 {"Calmar": r["2019+ Calmar"], "CAGR": r["2019+ CAGR"]}),
            )
            r["Keep vs baseline"] = "yes" if ok else ""

    table = pd.DataFrame(rows)
    print(f"\n=== Long (adopted) vs one-change tests (regime index {bench_name}) ===")
    print(table.round(2).to_string(index=False))
    kept = table.loc[table["Keep vs baseline"] == "yes", "Variant"].tolist()
    print("\nPasses _helps vs baseline (both halves):", ", ".join(kept) if kept else "none")

    if save_csv:
        _ensure_output_dirs()
        table.to_csv(OUTPUT_STUDIES / "long_comparison.csv", index=False)
        print(f"\nSaved {OUTPUT_STUDIES / 'long_comparison.csv'}")

    # equity chart: baseline vs best Calmar among kept (or best DD if none kept)
    print("\n=== Tranched rebalance offsets (timing luck) ===")
    tranched_long(tickers, P, sectors, **base)

    print("\n=== Crash hedge (post-hoc, index short when regime off) ===")
    eq0, _ = backtest(tickers, P, sectors=sectors, **base)
    hedged = hedge_crash_returns(eq0.dropna(), bench, index_on)
    print(perf_stats(hedged, "hedged post-hoc"))

    if "Long (adopted)" in stored_full:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        base_eq = stored_full["Long (adopted)"]
        alt_name = None
        if kept:
            sub = table[table["Variant"].isin(kept)].sort_values("Calmar", ascending=False)
            alt_name = sub.iloc[0]["Variant"]
        else:
            sub = table[table["Variant"] != "Long (adopted)"].sort_values("Max DD", ascending=False)
            if len(sub):
                alt_name = sub.iloc[0]["Variant"]
        fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
        ax[0].plot(base_eq, label="Long baseline")
        if alt_name and alt_name in stored_full:
            ax[0].plot(stored_full[alt_name], label=alt_name)
        ax[0].set_yscale("log")
        ax[0].set_title("Long baseline vs best variant (log)")
        ax[0].legend(fontsize=8)
        ax[1].plot(base_eq.index, (base_eq / base_eq.cummax() - 1) * 100, label="baseline")
        if alt_name and alt_name in stored_full:
            ae = stored_full[alt_name]
            ax[1].plot(ae.index, (ae / ae.cummax() - 1) * 100, label=alt_name)
        ax[1].set_ylabel("Drawdown %")
        ax[1].legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(OUTPUT_STUDIES / "long_comparison.png", dpi=120)
        print(f"Saved {OUTPUT_STUDIES / 'long_comparison.png'}")
    return table, stored_full


def index_risk_controls(close, dates, n_positions=None):
    """Graded regime, binary regime, and a half-slot cap, all from one index.

    Graded: full risk above the 200-day average, half risk when price is below it
    but the 50-day is still above the 200-day, no new entries once the 50-day
    is also below. Unknown (not enough history) stays at full risk.
    """
    s = close.reindex(dates).ffill()
    sma50 = s.rolling(50).mean()
    sma200 = s.rolling(200).mean()
    known = (s.notna() & sma50.notna() & sma200.notna()).to_numpy()
    above = (s > sma200).to_numpy()
    confirmed = ((s < sma200) & (sma50 < sma200)).to_numpy()
    graded = np.ones(len(dates))
    graded[known & ~above] = 0.5
    graded[known & confirmed] = 0.0
    binary = np.ones(len(dates))
    binary[known & ~above] = 0.0
    n_positions = MAX_POSITIONS if n_positions is None else n_positions
    slots = np.full(len(dates), n_positions)
    slots[known & ~above] = max(n_positions // 2, 1)
    slots[known & confirmed] = 0
    return graded, binary, slots


def vol_risk_scale(close, dates, lookback=VOL_LOOKBACK):
    """Cut risk when 20-day realized vol is above its own history. Never above 1."""
    s = close.reindex(dates).ffill()
    rv = s.pct_change().rolling(lookback).std() * np.sqrt(252)
    target = rv.expanding(min_periods=252).median()
    return (target / rv).clip(upper=1.0).fillna(1.0).to_numpy()


def _run_table(tickers, P, specs, sectors, start=None):
    rows, stored = [], {}
    for name, kw in specs:
        eq, tr = backtest(tickers, P, start=start, sectors=sectors, **kw)
        rows.append(_summary_row(name, eq, tr))
        stored[name] = (eq, tr)
    table = pd.DataFrame(rows, index=[n for n, _ in specs])
    return table, stored


def _same_window(eq, close):
    """Strategy equity and a buy-and-hold index, scaled together from the first shared date."""
    e = eq.dropna()
    b = close.reindex(e.index)
    first = b.first_valid_index()
    if first is None:
        return None
    b = b.loc[first:].ffill()
    e = e.reindex(b.index).dropna()
    b = (b.reindex(e.index).ffill() / b.reindex(e.index).ffill().iloc[0]) * e.iloc[0]
    if len(e) < 252:
        return None
    return e, b


def _equity_drawdown_pct(eq):
    e = eq.dropna()
    return (e / e.cummax() - 1) * 100


def _plot_equity_and_drawdown(ax0, ax1, eq, label, **plot_kw):
    """Same label, color, and linestyle on both panels."""
    (line,) = ax0.plot(eq, label=label, **plot_kw)
    dd = _equity_drawdown_pct(eq)
    ax1.plot(
        dd.index,
        dd.values,
        label=label,
        color=line.get_color(),
        linestyle=line.get_linestyle(),
        alpha=line.get_alpha(),
        linewidth=line.get_linewidth(),
    )


def build_live_equity_chart_strategies(eq, eq_peak, eq_legacy, bo_eq, bo_label):
    return [
        ("Long (live)", eq, {"color": CHART_SERIES_COLORS["Long (live)"], "linewidth": 2.2}),
        (
            "Long (peak only)",
            eq_peak,
            {"color": CHART_SERIES_COLORS["Long (peak only)"], "linestyle": "--", "alpha": 0.9},
        ),
        (
            "Long (no peak stop)",
            eq_legacy,
            {"color": CHART_SERIES_COLORS["Long (no peak stop)"], "linestyle": ":", "alpha": 0.9},
        ),
        (bo_label, bo_eq, {"color": DEFAULT_BO_CHART_COLOR, "linewidth": 1.3}),
    ]


def plot_equity_drawdown_panels(ax0, ax1, strategies, benches, align_eq, bench_labels=CHART_BENCHMARKS):
    """Equity (log) + drawdown % with matched colors and identical legend labels."""
    for label, eq, kw in strategies:
        _plot_equity_and_drawdown(ax0, ax1, eq.dropna(), label, **kw)
    ref = align_eq.dropna()
    for label in bench_labels:
        if label not in benches:
            continue
        pair = _same_window(ref, benches[label])
        if pair is None:
            continue
        _, b = pair
        bc = CHART_BENCHMARK_COLORS.get(label, "#17becf")
        _plot_equity_and_drawdown(ax0, ax1, b, label, alpha=0.75, linewidth=1.2, color=bc)
    ax0.set_yscale("log")
    ax0.set_title("Equity (log scale)")
    ax1.set_title("Drawdown %")


def plot_chart_benchmarks(ax0, ax1, eq, benches, labels=CHART_BENCHMARKS):
    """Overlay buy-and-hold index curves (log equity + drawdown) aligned to strategy dates."""
    if ax1 is None:
        ax1 = ax0
    ref = eq.dropna()
    for label in labels:
        if label not in benches:
            continue
        pair = _same_window(ref, benches[label])
        if pair is None:
            continue
        _, b = pair
        bc = CHART_BENCHMARK_COLORS.get(label, "#17becf")
        _plot_equity_and_drawdown(ax0, ax1, b, label, alpha=0.75, linewidth=1.2, color=bc)


SEP2024_START = "2024-09-02"
EXIT_REASON_LABEL = {
    "below_sma200": "Month closed under the 200-day average",
    "peak_dd": "Month-end: more than 30% below high since entry",
    "below_sma50": "Close under the 50-day average",
    "trail_stop": "Close under the 2.5 ATR trailing stop",
    "open_at_end": "Still open, marked at the last close",
    "below_sma200_month_end": "Month closed under the 200-day average",
    "hard_stop": "Hard stop",
    "below_sma150": "Close under the 150-day average",
    "time_stop": "Time stop",
    "rank_drop": "Rank drop",
    "regime_sma50": "Regime exit",
}


def _load_nse_company_names():
    local = Path("EQUITY_L.csv")
    if not local.exists():
        return {}
    df = pd.read_csv(local)
    df.columns = df.columns.str.strip()
    sym = df["SYMBOL"].astype(str).str.strip()
    name = df["NAME OF COMPANY"].astype(str).str.strip()
    return dict(zip(sym, name))


def _fmt_inr_rupees(x):
    x = float(x)
    body = f"₹{abs(x):,.0f}"
    return f"-{body}" if x < 0 else body


def _fmt_inr_price(x):
    x = float(x)
    if x >= 1000:
        return f"₹{x:,.2f}"
    if x >= 100:
        return f"₹{x:,.2f}"
    return f"₹{x:.2f}"


def _sym_display(ticker):
    return str(ticker).replace(".NS", "")


def _trade_book_section(title, blurb, eq, tr, names):
    eq = eq.dropna()
    closed = tr[tr.reason != "open_at_end"] if not tr.empty else tr
    open_at = tr[tr.reason == "open_at_end"] if not tr.empty else tr
    closed_pnl = float(closed.pnl.sum()) if len(closed) else 0.0
    open_pnl = float(open_at.pnl.sum()) if len(open_at) else 0.0
    lines = [
        f"## {title}",
        "",
        blurb,
        "",
        f"Account went from ₹{INITIAL_CAPITAL:,.0f} to ₹{eq.iloc[-1]:,.0f} between "
        f"{eq.index[0].date()} and {eq.index[-1].date()}.",
        f"Distinct tickers traded: {tr['ticker'].nunique() if not tr.empty else 0}. "
        f"Closed trades: {len(closed)}. Still open: {len(open_at)}. "
        f"Closed profit {_fmt_inr_rupees(closed_pnl)}. "
        f"Open profit, before a sell cost, {_fmt_inr_rupees(open_pnl)}.",
        "",
        "| # | Ticker | Company | Bought | Buy price | Shares | Sold | Sell price | Days | Return | Profit | Exit |",
        "|---:|---|---|---|---:|---:|---|---:|---:|---:|---:|---|",
    ]
    if tr.empty:
        lines.append("| — | — | — | — | — | — | — | — | — | — | — | — |")
    else:
        tr_sorted = tr.sort_values(["entry_date", "ticker"]).reset_index(drop=True)
        for i, r in tr_sorted.iterrows():
            sym = _sym_display(r.ticker)
            co = names.get(sym, "")
            bought = pd.Timestamp(r.entry_date).strftime("%Y-%m-%d")
            if r.reason == "open_at_end":
                sold, exit_l = "Still open", EXIT_REASON_LABEL["open_at_end"]
            else:
                sold = pd.Timestamp(r.exit_date).strftime("%Y-%m-%d")
                exit_l = EXIT_REASON_LABEL.get(r.reason, str(r.reason))
            lines.append(
                f"| {i + 1} | {sym} | {co} | {bought} | {_fmt_inr_price(r.entry_price)} | "
                f"{int(r.shares)} | {sold} | {_fmt_inr_price(r.exit_price)} | {int(r.days_held)} | "
                f"{r.ret_pct:+.1f}% | {_fmt_inr_rupees(r.pnl)} | {exit_l} |"
            )
    if len(open_at):
        open_sorted = open_at.sort_values("ret_pct", ascending=False)
        lines.extend([
            "",
            "### Still open",
            "",
            "| Ticker | Company | Bought | Buy price | Last price | Days | Return | Unrealized profit |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ])
        for _, r in open_sorted.iterrows():
            sym = _sym_display(r.ticker)
            co = names.get(sym, "")
            bought = pd.Timestamp(r.entry_date).strftime("%Y-%m-%d")
            lines.append(
                f"| {sym} | {co} | {bought} | {_fmt_inr_price(r.entry_price)} | "
                f"{_fmt_inr_price(r.exit_price)} | {int(r.days_held)} | {r.ret_pct:+.1f}% | "
                f"{_fmt_inr_rupees(r.pnl)} |"
            )
    lines.append("")
    return lines


def write_trades_from_sep2024(path="TRADES_FROM_SEP2024.md", start=SEP2024_START):
    """Regenerate Sep-2024-forward blotter markdown (Long live + benchmarks)."""
    data, universe = load_universe()
    bench, _ = _load_nifty500_close(None)
    tickers, P = build_panels(data, universe=universe, bench_close=bench)
    sectors = [load_sectors(tickers).get(t, "") for t in tickers]
    names = _load_nse_company_names()
    n_sym = len(tickers)

    eq_long, tr_long = backtest(tickers, P, sectors=sectors, start=start, **adopted_long_kwargs(P))
    benches = load_benchmarks()
    ref = eq_long.dropna()
    last = ref.index[-1].date()

    result_rows = []
    for label, eq, tr in (("Long (live)", eq_long, tr_long),):
        p = perf_stats(eq.dropna(), label)
        result_rows.append((
            label,
            eq.dropna().iloc[-1],
            p["CAGR %"],
            p["Max DD %"],
            len(tr),
            tr["ticker"].nunique() if not tr.empty else 0,
        ))
    for label in ("Nifty 500", "Nifty 50"):
        if label not in benches:
            continue
        pair = _same_window(ref, benches[label])
        if pair is None:
            continue
        _, b = pair
        p = perf_stats(b, label)
        result_rows.append((label, b.iloc[-1], p["CAGR %"], p["Max DD %"], None, None))

    out = [
        "# Buys and sells from September 2024",
        "",
        f"Universe: {n_sym} symbols with usable daily prices (download floor ₹{PIT_UNIVERSE_FLOOR_CR:,.0f} cr "
        f"or delisted list). Tradable only when estimated point-in-time market cap ≥ ₹{MIN_MCAP_CR:,.0f} cr, "
        f"liquidity ≥ ₹2 crore median turnover, and after {LISTING_COOLDOWN_DAYS} days from NSE listing.",
        "",
        f"Fresh account of ₹{INITIAL_CAPITAL:,.0f}, from 2 September 2024 through the latest close in the "
        f"cache ({last}). Nothing held before September 2024 is carried in.",
        "",
        "A buy fills at the next session’s open after the signal close. A sell fills at the next session’s "
        "open after the exit close. There is no clock time in the daily data. An open trade is valued at "
        "the last close, before the 0.15% sell cost. Closed trades deduct 0.15% on the buy and 0.15% on the sell.",
        "",
        "## Result",
        "",
        "| Book | Final value | CAGR % | Max drawdown % | Trades | Tickers traded |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, final, cagr, dd, ntr, ntick in result_rows:
        tr_s = "—" if ntr is None else str(int(ntr))
        tick_s = "—" if ntick is None else str(int(ntick))
        out.append(
            f"| {label} | {_fmt_inr_rupees(final)} | {cagr:.1f} | {dd:.1f} | {tr_s} | {tick_s} |"
        )
    out.append("")
    long_blurb = (
        "Live Long book: orders only after the last session of a month. Buy strength above a rising "
        "200-day average (see LONG_STRATEGY.md). Up to 15 equal-weight slots (~6.7% each). Sell when a "
        "month closes under the 200-day average, or when price is more than 30% below the high since entry. "
        "Idle cash earns GOLDBEES when gold is above its 200-day average (lagged one day), else ~6.5% annual."
    )
    out.extend(_trade_book_section("Long book (live)", long_blurb, eq_long, tr_long, names))

    mcap_s = universe["mcap_cr"] if universe is not None else pd.Series(dtype=float)
    mcap_rows = sorted(
        ((t, names.get(t, ""), mcap_s.get(t)) for t in tickers if np.isfinite(mcap_s.get(t, np.nan))),
        key=lambda x: -float(x[2]),
    )
    out.extend([
        "## Tickers included",
        "",
        f"{n_sym} stocks with current market cap above ₹{MIN_MCAP_CR:,.0f} crore. Market cap is today’s value, in ₹ crore.",
        "",
        "| Ticker | Company | Market cap, ₹ cr |",
        "|---|---|---:|",
    ])
    for sym, co, cap in mcap_rows:
        out.append(f"| {sym} | {co} | {int(round(float(cap))):,} |")
    out.append("")

    Path(path).write_text("\n".join(out), encoding="utf-8")
    print(f"Wrote {path} ({len(out)} lines)")
    return path


def _window_stats(eq, tr):
    p = perf_stats(eq, "")
    t = trade_stats(tr)
    return {"CAGR": p["CAGR %"], "DD": p["Max DD %"], "Calmar": p["Calmar"],
            "PF": t.get("Profit factor", np.nan), "Sharpe": p["Sharpe"]}


def _helps(base, trial):
    """Keep a change when both halves improve Calmar and neither gives up more than 5 points of CAGR."""
    for b, v in (base, trial):
        if not (np.isfinite(v["Calmar"]) and np.isfinite(b["Calmar"])):
            return False
        if v["Calmar"] <= b["Calmar"] or v["CAGR"] < b["CAGR"] - 5:
            return False
    return True


def simulate_exit(close, opn, ma, vol, atr, i0, entry, rule, chk_ma, chk_dd, i_last,
                  month_end=None, cost_per_side=COST_PER_SIDE):
    """Replay one position from entry index i0 with layered exit rules (daily ATR + optional month-end)."""
    peak = float(entry)
    stop = -np.inf
    peak_dd = rule.get("peak_dd")
    atr_mult = rule.get("atr_mult")
    atr_after = rule.get("atr_after", 0.0)
    hard = rule.get("hard_stop")

    for i in range(i0, i_last + 1):
        c = close[i]
        if not np.isfinite(c):
            continue
        peak = max(peak, c)
        reason = None
        if hard is not None and c <= entry * (1 - hard):
            reason = "hard_stop"
        if (reason is None and atr_mult and np.isfinite(atr[i]) and atr[i] > 0
                and peak / entry - 1 >= atr_after and c < peak - atr_mult * atr[i]):
            reason = "atr_trail"
        if reason is None and month_end is not None and month_end[i]:
            if chk_ma and np.isfinite(ma[i]) and c < ma[i]:
                reason = "below_sma200"
            elif chk_dd and peak_dd and c < peak * (1 - peak_dd):
                reason = "peak_dd"
        if reason:
            fill_i = min(i + 1, i_last)
            px = opn[fill_i] if np.isfinite(opn[fill_i]) else c
            return fill_i, px, reason, peak
    last_c = close[i_last]
    if not np.isfinite(last_c):
        last_c = entry
    return i_last, last_c, "open_at_end", peak


def _exit_lab_rules():
    return {
        "baseline (SMA200 + 30% peak, month-end)": dict(peak_dd=0.30),
        "ATR 3 daily + peak": dict(peak_dd=0.30, atr_mult=3),
        "ATR 5 daily + peak": dict(peak_dd=0.30, atr_mult=5),
        "ATR 8 daily + peak": dict(peak_dd=0.30, atr_mult=8),
        "ATR 10 daily + peak": dict(peak_dd=0.30, atr_mult=10),
        "ATR 6 after +30% + peak": dict(peak_dd=0.30, atr_mult=6, atr_after=0.30),
        "ATR 8 after +50% + peak": dict(peak_dd=0.30, atr_mult=8, atr_after=0.50),
    }


def exit_lab(tickers, P, trades, rules=None, cost_per_side=COST_PER_SIDE):
    """Counterfactual exits on the same entries as `trades` (from the live Long book)."""
    if trades.empty:
        return pd.DataFrame()
    rules = rules or _exit_lab_rules()
    dates = P["Close"].index
    month_end = month_end_mask(dates)
    C = P["Close"].ffill().to_numpy()
    O = P["Open"].to_numpy()
    MA = P["sma200"].to_numpy()
    A = P["atr"].ffill().to_numpy()
    tix = {t: j for j, t in enumerate(tickers)}
    rows = []
    cps = cost_per_side
    for _, e in trades.iterrows():
        j = tix.get(e.ticker)
        if j is None:
            continue
        i0 = int(dates.searchsorted(pd.Timestamp(e.entry_date)))
        if e.reason == "open_at_end":
            i_last = len(dates) - 1
        else:
            i_last = int(dates.searchsorted(pd.Timestamp(e.exit_date)))
        i_last = max(i0, min(i_last, len(dates) - 1))
        entry = float(e.entry_price)
        base_peak = float(e.exit_price) if e.reason != "open_at_end" else np.nan
        for name, rule in rules.items():
            i1, px, why, peak = simulate_exit(
                C[:, j], O[:, j], MA[:, j], None, A[:, j], i0, entry, rule,
                chk_ma=True, chk_dd=True, i_last=i_last, month_end=month_end, cost_per_side=cps,
            )
            ret = (px / entry - 1) * 100
            gross_pnl = (px * (1 - cps) - entry * (1 + cps)) * int(e.shares)
            days = (dates[i1] - dates[i0]).days
            cap = max(peak / entry - 1, 1e-9) if peak > entry else np.nan
            capture = ret / (cap * 100) if np.isfinite(cap) and cap > 0 else np.nan
            rows.append(dict(
                rule=name, ticker=e.ticker, entry_date=e.entry_date, exit_reason=why,
                ret_pct=ret, days=days, capture=capture, pnl=gross_pnl,
            ))
    return pd.DataFrame(rows)


def _exit_lab_summary(lab_df):
    if lab_df.empty:
        return pd.DataFrame()
    slot_years = lab_df.groupby("rule").apply(
        lambda g: g["ret_pct"].sum() / (g["days"].sum() / 365.25) if g["days"].sum() > 0 else np.nan
    )
    out = lab_df.groupby("rule").agg(
        n=("ret_pct", "size"),
        mean_ret=("ret_pct", "mean"),
        median_ret=("ret_pct", "median"),
        worst10_mean=("ret_pct", lambda s: s.quantile(0.10) if len(s) >= 10 else s.min()),
        mean_capture=("capture", "mean"),
        mean_days=("days", "mean"),
    )
    out["ret_per_slot_year"] = slot_years
    return out.round(2)


def atr_exit_study(tickers, P, sectors):
    """Exit lab on live Long entries + full-portfolio tests with daily ATR chandelier on top of SMA200/peak."""
    base_kw = adopted_long_kwargs(P)
    _, tr_base = backtest(tickers, P, sectors=sectors, **base_kw)
    print("\n=== Exit lab (same entries as live Long, counterfactual exits) ===")
    lab = exit_lab(tickers, P, tr_base)
    lab_summary = _exit_lab_summary(lab)
    print(lab_summary.to_string())
    _ensure_output_dirs()
    lab_summary.to_csv(OUTPUT_STUDIES / "exit_lab_summary.csv")

    windows = [
        ("2010-2018", dict(start="2010-01-01", end="2018-12-31")),
        ("2019+", dict(start="2019-01-01")),
        ("Full", {}),
    ]
    specs = [("Long (live) baseline", {})]
    for k in (3, 5, 6, 8, 10):
        specs.append((f"ATR {k} daily + peak", {"also_trail": True, "atr_mult": float(k)}))
    specs.extend([
        ("ATR 6 after +30% + peak", {"also_trail": True, "atr_mult": 6.0, "atr_after": 0.30}),
        ("ATR 8 after +50% + peak", {"also_trail": True, "atr_mult": 8.0, "atr_after": 0.50}),
    ])

    print("\n=== Portfolio: daily ATR trail on adopted Long (SMA200 + 30% peak unchanged) ===")
    base_half = []
    for wlabel, wkw in windows:
        if wlabel == "Full":
            continue
        eq_b, tr_b = backtest(tickers, P, sectors=sectors, **{**base_kw, **wkw})
        base_half.append(_window_stats(eq_b, tr_b))
    b0, b1 = base_half

    port_rows = []
    stored = {}
    for name, extra in specs:
        row = {"Variant": name}
        halves = []
        for wlabel, wkw in windows:
            kw = dict(base_kw)
            kw.update(extra)
            kw.update(wkw)
            eq, tr = backtest(tickers, P, sectors=sectors, **kw)
            if wlabel == "Full":
                stored[name] = (eq, tr)
            st = _window_stats(eq, tr)
            row[f"{wlabel} CAGR"] = round(st["CAGR"], 2)
            row[f"{wlabel} DD"] = round(st["DD"], 2)
            row[f"{wlabel} Calmar"] = round(st["Calmar"], 3)
            if wlabel in ("2010-2018", "2019+"):
                halves.append(st)
            if wlabel == "Full":
                row["Trades"] = trade_stats(tr).get("Trades", 0)
                ts = tr.groupby("reason").size() if not tr.empty else pd.Series(dtype=int)
                row["trail_exits"] = int(ts.get("trail_stop", 0))
        row["Keep vs baseline"] = (
            "yes" if name != "Long (live) baseline" and _helps((b0, halves[0]), (b1, halves[1])) else ""
        )
        port_rows.append(row)
    port = pd.DataFrame(port_rows).set_index("Variant")
    print(port.to_string())
    _ensure_output_dirs()
    port.to_csv(OUTPUT_STUDIES / "atr_exit_portfolio.csv")

    base_full = _window_stats(stored["Long (live) baseline"][0], stored["Long (live) baseline"][1])
    best_name, best_cal = None, base_full["Calmar"]
    for name in port.index:
        if name == "Long (live) baseline":
            continue
        eq, tr = stored[name]
        st = _window_stats(eq, tr)
        if st["Calmar"] > best_cal and st["CAGR"] >= base_full["CAGR"] - 5:
            best_cal = st["Calmar"]
            best_name = name
    print(f"\nBest Calmar (CAGR within 5pts of baseline): {best_name} (Calmar {best_cal:.3f} vs "
          f"baseline {base_full['Calmar']:.3f})")
    return lab_summary, port, best_name


def market_correlation(close):
    """Median stock volatility, equal-weight portfolio volatility, and the correlation that links them."""
    rets = close.pct_change()
    keep = rets.columns[rets.notna().mean() > 0.8]
    sub = rets[keep]
    sigma = float(sub.std().median() * np.sqrt(252))
    port = sub.mean(axis=1)
    sig_p = float(port.std() * np.sqrt(252))
    n = sub.shape[1]
    rho = ((sig_p / sigma) ** 2 - 1 / n) / (1 - 1 / n) if sigma > 0 and n > 1 else np.nan
    return sigma, sig_p, rho, n


def long_study(tickers, P, sectors):
    """One change at a time versus the monthly book. A change is kept only if both halves improve."""
    dates = P["Close"].index
    bench = None
    for ticker in ("^CRSLDX", "^NSEI"):
        downloaded = download_one(ticker)
        if downloaded is not None and len(downloaded) >= 252:
            bench = downloaded.Close
            bench_name = ticker
            break
    graded, binary, _ = index_risk_controls(bench, dates)
    nifty_ret = bench.pct_change().reindex(dates).fillna(0).to_numpy()
    flat_ret = np.full(len(dates), 0.06 / 252)
    pct_roc = P["roc252"].rank(axis=1, pct=True).to_numpy()
    pct_adj = P["mom_adj"].rank(axis=1, pct=True).to_numpy()
    vol = P["vol126"].to_numpy(dtype=float)
    base = dict(entry_col="entry_long", rank_col="roc252", exit_mode="sma200",
                size_mode="equal", sector_cap=SECTOR_CAP, max_positions=LONG_POSITIONS)
    tests = [
        ("A binary regime", {"risk_mult": binary}),
        ("A graded regime", {"risk_mult": graded}),
        ("A binary + sell below SMA50", {"risk_mult": binary, "tighten": True, "regime_off": binary == 0}),
        ("B vol-adjusted 12-1", {"rank_col": "mom_adj"}),
        ("B 6m+12m score", {"rank_col": "score"}),
        ("C peak stop 25%", {"dd_stop": 0.25}),
        ("C peak stop 30%", {"dd_stop": 0.30}),
        ("C peak stop 35%", {"dd_stop": 0.35}),
        ("C drop out of top 30%", {"rank_pct": pct_roc, "rs_min": 0.70}),
        ("D cash in Nifty", {"cash_ret": nifty_ret}),
        ("D cash at 6%", {"cash_ret": flat_ret}),
        ("E inverse-vol weight", {"size_mode": "invvol", "vol": vol}),
        ("F 10 positions", {"max_positions": 10}),
        ("F 20 positions", {"max_positions": 20}),
        ("G within 10% of high", {"entry_col": "entry_long90"}),
        ("G within 5% of high", {"entry_col": "entry_long95"}),
        ("G 200-day rising 6m", {"entry_col": "entry_long_s126"}),
    ]
    windows = {"2010-2018": dict(start="2010-01-01", end="2018-12-31"),
               "2019+": dict(start="2019-01-01")}

    def run(extra, window):
        kw = dict(base)
        kw.update(extra)
        kw.update(window)
        return backtest(tickers, P, sectors=sectors, **kw)

    print(f"\n=== Long-book changes, one at a time (regime index {bench_name}) ===")
    sigma, sig_p, rho, n_names = market_correlation(P["Close"])
    print(f"Measured on {n_names} stocks: median vol {sigma:.0%}, equal-weight vol {sig_p:.0%}, "
          f"average correlation {rho:.2f}.")
    print("At that correlation, going from 10 to 20 names barely changes portfolio volatility.")

    base_stats = {}
    rows = []
    stored = {}
    for label, window in windows.items():
        eq, tr = run({}, window)
        base_stats[label] = _window_stats(eq, tr)
    full_eq, full_tr = run({}, {})
    print(f"Baseline average cash {full_eq.attrs.get('avg_cash', np.nan):.0%}")
    print("Baseline, drop the 10 largest trades:")
    print(pd.Series(_jackpot(full_tr)).round(2).to_string())

    results = {}
    for name, extra in tests:
        stats = {}
        for label, window in windows.items():
            eq, tr = run(extra, window)
            stats[label] = _window_stats(eq, tr)
        ok = _helps((base_stats["2010-2018"], stats["2010-2018"]), (base_stats["2019+"], stats["2019+"]))
        results[name] = (extra, stats, ok)
        rows.append({
            "2010-18 CAGR": stats["2010-2018"]["CAGR"], "2010-18 DD": stats["2010-2018"]["DD"],
            "2010-18 Calmar": stats["2010-2018"]["Calmar"], "2010-18 PF": stats["2010-2018"]["PF"],
            "2019+ CAGR": stats["2019+"]["CAGR"], "2019+ DD": stats["2019+"]["DD"],
            "2019+ Calmar": stats["2019+"]["Calmar"], "2019+ PF": stats["2019+"]["PF"],
            "Keep": "yes" if ok else "",
        })
    b = base_stats
    base_row = {
        "2010-18 CAGR": b["2010-2018"]["CAGR"], "2010-18 DD": b["2010-2018"]["DD"],
        "2010-18 Calmar": b["2010-2018"]["Calmar"], "2010-18 PF": b["2010-2018"]["PF"],
        "2019+ CAGR": b["2019+"]["CAGR"], "2019+ DD": b["2019+"]["DD"],
        "2019+ Calmar": b["2019+"]["Calmar"], "2019+ PF": b["2019+"]["PF"],
        "Keep": "",
    }
    table = pd.DataFrame([base_row] + rows, index=["Long baseline"] + [n for n, _ in tests])
    print(table.round(2).to_string())

    def best_of(prefix, require_count=1):
        hits = [(n, extra, st) for n, (extra, st, ok) in results.items() if ok and n.startswith(prefix)]
        if len(hits) < require_count:
            return None
        hits.sort(key=lambda item: item[2]["2019+"]["Calmar"], reverse=True)
        return hits[0]

    dd_hits = [n for n, (_, _, ok) in results.items() if ok and n.startswith("C peak")]
    chosen = []
    regime = best_of("A ")
    if regime:
        chosen.append(regime)
    rank = best_of("B ")
    if rank:
        chosen.append(rank)
    if len(dd_hits) >= 2:
        chosen.append(best_of("C peak"))
    rs = best_of("C drop")
    if rs:
        chosen.append(rs)
    cash6 = results.get("D cash at 6%")
    cash_etf = results.get("D cash in Nifty")
    if cash6 and cash6[2]:
        chosen.append(("D cash at 6%", cash6[0], cash6[1]))
    elif cash_etf and cash_etf[2]:
        chosen.append(("D cash in Nifty", cash_etf[0], cash_etf[1]))
    inv = best_of("E ")
    if inv:
        chosen.append(inv)
    breadth = best_of("F ")
    if breadth:
        chosen.append(breadth)
    # A tighter high-proximity rule counts only when the next tighter step also holds.
    near = [n for n, (_, _, ok) in results.items() if ok and n.startswith("G within")]
    if "G within 10% of high" in near:
        chosen.append(("G within 10% of high",) + results["G within 10% of high"][:2])
    slope = best_of("G 200")
    if slope:
        chosen.append(slope)

    print("\nKept, because Calmar rose in both 2010-2018 and 2019+ without giving up more than 5 points of CAGR:")
    if not chosen:
        print("  none")
        return None
    for name, _, _ in chosen:
        print(f"  {name}")
    combo = dict(base)
    for name, extra, _ in chosen:
        combo.update(extra)
    if combo.get("rank_col") == "mom_adj" and combo.get("rs_min") is not None:
        combo["rank_pct"] = pct_adj
    print("\n=== Combined book ===")
    combo_rows = []
    for label, window in list(windows.items()) + [("Full", {})]:
        eq, tr = backtest(tickers, P, sectors=sectors, **{**combo, **window})
        st = _window_stats(eq, tr)
        combo_rows.append(st)
        if label == "Full":
            print("Drop the 10 largest trades:")
            print(pd.Series(_jackpot(tr)).round(2).to_string())
            print(f"Average cash {eq.attrs.get('avg_cash', np.nan):.0%}")
            hold = hold_profile(tr, (eq.index[-1] - eq.index[0]).days / 365.25)
            print(pd.Series(hold).round(1).to_string())
    show = pd.DataFrame(combo_rows, index=["2010-2018", "2019+", "Full"])
    base_full = _window_stats(full_eq, full_tr)
    show.loc["Long full"] = base_full
    print(show.round(2).to_string())
    return combo


def control_study(tickers, P, sectors):
    """The four portfolio controls, one at a time, on the monthly Long book."""
    dates = P["Close"].index
    bench = None
    for ticker in ("^CRSLDX", "^NSEI"):
        downloaded = download_one(ticker)
        if downloaded is not None and len(downloaded) >= 252:
            bench = downloaded.Close
            break
    graded, _, slots = index_risk_controls(bench, dates, LONG_POSITIONS)
    nifty_vol = vol_risk_scale(bench, dates)
    base = dict(entry_col="entry_long", rank_col="roc252", exit_mode="sma200",
                size_mode="equal", sector_cap=SECTOR_CAP, max_positions=LONG_POSITIONS)
    tests = [
        ("1 graded, half size", {"risk_mult": graded}),
        ("1 graded, fewer slots", {"slot_cap": slots}),
        ("1 equity curve < its 200dma, stop buying", {"eq_filter": "block"}),
        ("1 equity curve < its 200dma, go to cash", {"eq_filter": "flatten"}),
        ("2 scale size by Nifty 20d vol", {"risk_mult": nifty_vol}),
        ("2 scale size by equity-curve vol", {"equity_vol_scale": True}),
        ("3 sector cap 2", {"sector_cap": 2}),
        ("3 no sector cap", {"sector_cap": 0}),
        ("4 time stop 20d", {"time_stop": 20}),
        ("4 time stop 30d", {"time_stop": 30}),
        ("4 also trail 2.5 ATR", {"also_trail": True, "atr_mult": 2.5}),
        ("4 also trail 3 ATR", {"also_trail": True, "atr_mult": 3.0}),
        ("4 also trail 4 ATR", {"also_trail": True, "atr_mult": 4.0}),
    ]
    windows = [("2010-2018", dict(start="2010-01-01", end="2018-12-31")),
               ("2019+", dict(start="2019-01-01")),
               ("Full", {})]

    def run(extra, window):
        kw = dict(base)
        kw.update(extra)
        kw.update(window)
        return backtest(tickers, P, sectors=sectors, **kw)

    print("\n=== Portfolio controls on the Long book, one at a time ===")
    base_stats = {label: _window_stats(*run({}, window)) for label, window in windows}
    rows = []
    for name, extra in tests:
        stats = {label: _window_stats(*run(extra, window)) for label, window in windows}
        halves = [(base_stats[k], stats[k]) for k in ("2010-2018", "2019+")]
        row = {}
        for label, st in stats.items():
            row[f"{label} CAGR"] = st["CAGR"]
            row[f"{label} DD"] = st["DD"]
            row[f"{label} Calmar"] = st["Calmar"]
        row["Keep"] = "yes" if _helps(halves[0], halves[1]) else ""
        rows.append(row)
    table = pd.DataFrame(rows, index=[n for n, _ in tests])
    base_row = {}
    for label, st in base_stats.items():
        base_row[f"{label} CAGR"] = st["CAGR"]
        base_row[f"{label} DD"] = st["DD"]
        base_row[f"{label} Calmar"] = st["Calmar"]
    table.loc["Long baseline"] = base_row
    print(table.round(2).to_string())
    kept = [n for n, _ in tests if table.loc[n, "Keep"] == "yes"]
    print("Kept:", ", ".join(kept) if kept else "none")
    return table


def main():
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 20)
    data, universe = load_universe()
    tickers, P = build_panels(data, universe=universe)
    sectors = [load_sectors(tickers).get(t, "") for t in tickers]
    covered = sum(bool(s) for s in sectors)
    print(f"Sector labels: {covered}/{len(tickers)} symbols "
          f"(unlabeled names are not subject to the cap of {SECTOR_CAP})")
    pd.set_option("display.max_rows", 40)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 20)
    long_study(tickers, P, sectors)
    control_study(tickers, P, sectors)

    specs = [
        ("Long", adopted_long_kwargs(P)),
        ("Long (peak, no sleeve)", dict(LONG_BOOK)),
        ("Long (no peak stop)", dict(LONG_BOOK_LEGACY)),
    ]
    print("\n=== Full sample ===")
    full, runs = _run_table(tickers, P, specs, sectors)
    bench_px = None
    for ticker in ("^CRSLDX", "^NSEI"):
        downloaded = download_one(ticker)
        if downloaded is not None and len(downloaded) >= 200:
            bench_px = downloaded.Close
            print(f"Breakout market filter uses {ticker}")
            break
    regime_s = bench_px.reindex(P["Close"].index).ffill()
    regime_sma = regime_s.rolling(200, min_periods=200).mean()
    regime_ok = ((regime_s > regime_sma) | regime_sma.isna()).fillna(True).to_numpy()
    bo_grid = []
    for gate, gate_on in (("mkt", True), ("no mkt", False)):
        for label, mult, mode in (
            ("3.5 ATR", 3.5, "atr"),
            ("5 ATR", 5.0, "atr"),
            ("SMA150+5", 5.0, "sma150"),
            ("Hybrid 4", 4.0, "hybrid"),
        ):
            bo_grid.append((f"BO {label}, {gate}", mult, mode, gate_on))
    print(f"Breakout signals: {int(P['signal_bo'].to_numpy().sum()):,}")
    for name, mult, mode, gate_on in bo_grid:
        runs[name] = backtest_breakout(tickers, P, sectors, regime_ok, mult, mode, gate_on)
        full.loc[name] = _summary_row(name, *runs[name])
    # Same breakout entry. Only the hold and the position size change.
    hold_grid = []
    for gate, gate_on in (("mkt", True), ("no mkt", False)):
        hold_grid.extend([
            (f"BO SMA200 equal, {gate}", 5.0, "sma200", gate_on, "equal"),
            (f"BO SMA200+hard equal, {gate}", 5.0, "sma200_hard", gate_on, "equal"),
            (f"BO SMA200+hard risk, {gate}", 5.0, "sma200_hard", gate_on, "risk"),
        ])
    for name, mult, mode, gate_on, sizing in hold_grid:
        runs[name] = backtest_breakout(tickers, P, sectors, regime_ok, mult, mode, gate_on, size_mode=sizing)
        full.loc[name] = _summary_row(name, *runs[name])
    print(full.round(2))
    print("\n=== 2019 onwards ===")
    recent, recent_runs = _run_table(tickers, P, specs, sectors, start="2019-01-01")
    for name, mult, mode, gate_on in bo_grid:
        eq_i, tr_i = backtest_breakout(
            tickers, P, sectors, regime_ok, mult, mode, gate_on, start="2019-01-01")
        recent_runs[name] = (eq_i, tr_i)
        recent.loc[name] = _summary_row(name, eq_i, tr_i)
    for name, mult, mode, gate_on, sizing in hold_grid:
        eq_i, tr_i = backtest_breakout(
            tickers, P, sectors, regime_ok, mult, mode, gate_on, start="2019-01-01", size_mode=sizing)
        recent_runs[name] = (eq_i, tr_i)
        recent.loc[name] = _summary_row(name, eq_i, tr_i)
    print(recent.round(2))

    print("\n=== Holding period and big winners (closed trades) ===")
    hold_rows = []
    for name, (eq_i, tr) in runs.items():
        span = eq_i.dropna()
        years = (span.index[-1] - span.index[0]).days / 365.25
        hold_rows.append(hold_profile(tr, years))
    print(pd.DataFrame(hold_rows, index=list(runs)).round(1))

    print("\n=== Did the trade ever double, and was that gain still there at the exit? ===")
    for name in ("Long", "Long (no peak stop)") + tuple(n for n, *_ in hold_grid):
        tr = runs[name][1]
        closed = tr[tr.reason != "open_at_end"] if not tr.empty else tr
        if closed.empty or "max_multiple" not in closed.columns:
            if name.startswith("Long"):
                touched2 = int((closed.ret_pct >= 100).sum()) if not closed.empty else 0
                touched3 = int((closed.ret_pct >= 200).sum()) if not closed.empty else 0
                print(f"{name:32} closed 2x {touched2:3}  closed 3x {touched3:3}  (exit price only)")
            continue
        print(f"{name:32} touched 2x {int(closed.reached_2x.sum()):3}  still 2x at exit {int((closed.ret_pct >= 100).sum()):3}"
              f"   touched 3x {int(closed.reached_3x.sum()):3}  still 3x at exit {int((closed.ret_pct >= 200).sum()):3}")

    bo_names = [name for name, *_ in bo_grid] + [name for name, *_ in hold_grid]
    best_name = full.loc[bo_names, "CAGR %"].idxmax()
    calm_name = full.loc[bo_names, "Calmar"].idxmax()
    print(f"\nHighest-return breakout: {best_name}")
    print(f"Highest-Calmar breakout: {calm_name}")
    for name in ("Long", best_name):
        tr = runs[name][1]
        print(f"\n=== {name} exits ===")
        print(tr.groupby("reason").agg(n=("pnl", "size"), pnl=("pnl", "sum"), avg_ret=("ret_pct", "mean")).round(2))

    eq, trades = runs["Long"]  # peak stop + gold/debt cash sleeve
    eq_peak_only, _ = runs["Long (peak, no sleeve)"]
    eq_legacy, _ = runs["Long (no peak stop)"]
    bo_eq, bo_tr = runs[best_name]

    print("\nBenchmarks:")
    benches = load_benchmarks()
    print("\n=== Same dates as the proposed book ===")
    rows = []
    for name, close in benches.items():
        pair = _same_window(eq, close)
        if pair is None:
            continue
        e, b = pair
        rows.append(perf_stats(e, f"Long ({name} window)"))
        rows.append(perf_stats(b, name))
    if rows:
        print(pd.DataFrame(rows).set_index("name").round(2))

    yearly = {"Long": yearly_returns(eq),
              "Long (peak, no sleeve)": yearly_returns(eq_peak_only),
              "Long (no peak stop)": yearly_returns(eq_legacy), best_name: yearly_returns(bo_eq)}
    for label in ("Nifty 500", "Nifty 50"):
        if label not in benches:
            continue
        px = benches[label].reindex(eq.dropna().index).ffill()
        yearly[label] = yearly_returns(px / px.dropna().iloc[0] * INITIAL_CAPITAL)
    print("\n=== Calendar-year returns % ===")
    print(pd.DataFrame(yearly))

    _ensure_output_dirs()
    trades.to_csv(OUTPUT_BACKTEST / "trades_long.csv", index=False)
    bo_tr.to_csv(OUTPUT_BACKTEST / "trades_breakout.csv", index=False)
    eq.to_csv(OUTPUT_BACKTEST / "equity_curve_long.csv", header=["equity"])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, figsize=(11, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    chart_specs = build_live_equity_chart_strategies(eq, eq_peak_only, eq_legacy, bo_eq, best_name)
    plot_equity_drawdown_panels(ax[0], ax[1], chart_specs, benches, eq)
    ax[0].legend(fontsize=8, loc="upper left")
    ax[1].legend(fontsize=8, loc="lower left")
    plt.tight_layout()
    plt.savefig(OUTPUT_BACKTEST / "equity_curve.png", dpi=120)
    print(f"\nSaved output/backtest/ (trades, equity curves, equity_curve.png)")


def refresh_equity_chart():
    """Regenerate equity_curve.png and trades_long.csv without the full study grid."""
    data, universe = load_universe()
    tickers, P = build_panels(data, universe=universe)
    sectors = [load_sectors(tickers).get(t, "") for t in tickers]
    eq, trades = backtest(tickers, P, sectors=sectors, **adopted_long_kwargs(P))
    eq_peak, _ = backtest(tickers, P, sectors=sectors, **LONG_BOOK)
    eq_legacy, _ = backtest(tickers, P, sectors=sectors, **LONG_BOOK_LEGACY)
    bench_px = None
    for ticker in ("^CRSLDX", "^NSEI"):
        downloaded = download_one(ticker)
        if downloaded is not None and len(downloaded) >= 200:
            bench_px = downloaded.Close
            break
    regime_s = bench_px.reindex(P["Close"].index).ffill()
    regime_sma = regime_s.rolling(200, min_periods=200).mean()
    regime_ok = ((regime_s > regime_sma) | regime_sma.isna()).fillna(True).to_numpy()
    bo_eq, bo_tr = backtest_breakout(tickers, P, sectors, regime_ok, 5.0, "sma200", True, size_mode="equal")
    benches = load_benchmarks()
    _ensure_output_dirs()
    trades.to_csv(OUTPUT_BACKTEST / "trades_long.csv", index=False)
    eq.to_csv(OUTPUT_BACKTEST / "equity_curve_long.csv", header=["equity"])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, figsize=(11, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    chart_specs = build_live_equity_chart_strategies(
        eq, eq_peak, eq_legacy, bo_eq, "BO SMA200 equal, mkt"
    )
    plot_equity_drawdown_panels(ax[0], ax[1], chart_specs, benches, eq)
    ax[0].legend(fontsize=8, loc="upper left")
    ax[1].legend(fontsize=8, loc="lower left")
    plt.tight_layout()
    plt.savefig(OUTPUT_BACKTEST / "equity_curve.png", dpi=120)
    p0 = perf_stats(eq_legacy, "legacy")
    p1 = perf_stats(eq_peak, "peak")
    p2 = perf_stats(eq, "Long")
    print(f"Long no peak: {p0['CAGR %']:.1f}% CAGR, {p0['Max DD %']:.1f}% DD, Calmar {p0['Calmar']:.2f}")
    print(f"Long peak only: {p1['CAGR %']:.1f}% CAGR, {p1['Max DD %']:.1f}% DD, Calmar {p1['Calmar']:.2f}")
    print(f"Long live (peak+cash sleeve): {p2['CAGR %']:.1f}% CAGR, {p2['Max DD %']:.1f}% DD, "
          f"Calmar {p2['Calmar']:.2f}, final {p2['Final value']:,.0f}")
    print("Saved output/backtest/ (equity_curve.png, trades_long.csv, …)")


def rank_momentum_study(tickers, P, sectors):
    """Compare rank columns for live Long (same entries/exits/cash sleeve)."""
    base_kw = adopted_long_kwargs(P)
    specs = [
        ("rank roc126 (live)", {}),
        ("rank roc126 (6m raw)", dict(rank_col="roc126")),
        ("rank roc63 (3m raw)", dict(rank_col="roc63")),
        ("rank 6m+12m score", dict(rank_col="score")),
        ("rank 3+6+12m score", dict(rank_col="score_3_6_12")),
        ("rank mom_adj", dict(rank_col="mom_adj")),
        ("rank roc_12_1", dict(rank_col="roc_12_1")),
        ("rank sharpe63", dict(rank_col="sharpe63")),
        ("rank sharpe126", dict(rank_col="sharpe126")),
        ("rank sharpe252", dict(rank_col="sharpe252")),
        ("rank sharpe 3+6+12 score", dict(rank_col="score_sharpe_3_6_12")),
    ]
    windows = [
        ("2010-2018", dict(start="2010-01-01", end="2018-12-31")),
        ("2019+", dict(start="2019-01-01")),
        ("Full", {}),
    ]
    base_half = []
    for wlabel, wkw in windows:
        if wlabel == "Full":
            continue
        eq_b, tr_b = backtest(tickers, P, sectors=sectors, **{**base_kw, **wkw})
        base_half.append(_window_stats(eq_b, tr_b))
    b0, b1 = base_half

    rows = []
    stored = {}
    for name, extra in specs:
        row = {"Variant": name}
        halves = []
        for wlabel, wkw in windows:
            kw = {**base_kw, **extra, **wkw}
            eq, tr = backtest(tickers, P, sectors=sectors, **kw)
            if wlabel == "Full":
                stored[name] = (eq, tr)
            st = _window_stats(eq, tr)
            p = perf_stats(eq, name)
            t = trade_stats(tr)
            row[f"{wlabel} CAGR"] = round(st["CAGR"], 2)
            row[f"{wlabel} DD"] = round(st["DD"], 2)
            row[f"{wlabel} Calmar"] = round(st["Calmar"], 3)
            if wlabel in ("2010-2018", "2019+"):
                halves.append(st)
            if wlabel == "Full":
                row["Sharpe"] = round(p["Sharpe"], 2)
                row["Trades"] = t.get("Trades", 0)
                row["PF"] = round(t.get("Profit factor", np.nan), 2)
        row["Keep vs live"] = (
            "yes" if name != "rank roc126 (live)" and _helps((b0, halves[0]), (b1, halves[1])) else ""
        )
        rows.append(row)

    table = pd.DataFrame(rows).set_index("Variant")
    print("\n=== Rank / momentum / Sharpe study (live Long book, one rank change at a time) ===")
    print(table.to_string())
    _ensure_output_dirs()
    table.to_csv(OUTPUT_STUDIES / "rank_momentum_study.csv")

    base_full = _window_stats(*stored["rank roc126 (live)"])
    best_name, best_cal = None, base_full["Calmar"]
    for name in table.index:
        if name == "rank roc126 (live)":
            continue
        st = _window_stats(*stored[name])
        if st["Calmar"] > best_cal and st["CAGR"] >= base_full["CAGR"] - 5:
            best_cal = st["Calmar"]
            best_name = name
    print(f"\nBest Full Calmar with CAGR within 5pts of live: {best_name} "
          f"(Calmar {best_cal:.3f} vs live {base_full['Calmar']:.3f})")
    return table, best_name


def entry_filter_study(tickers, P, sectors):
    """Test ATH proximity and low-beta entry filters on live Long (exits/cash sleeve unchanged)."""
    if "entry_long_ath25_beta125" not in P:
        raise SystemExit("Need Nifty 500 benchmark in build_panels for beta filter.")
    base_kw = adopted_long_kwargs(P)
    specs = [
        ("entry_long (live baseline)", dict(entry_col="entry_long")),
        ("+ within 25% of ATH", dict(entry_col="entry_long_ath25")),
        ("+ beta < 1.25 vs N500", dict(entry_col="entry_long_beta125")),
        ("+ ATH 25% and beta < 1.25", dict(entry_col="entry_long_ath25_beta125")),
    ]
    windows = [
        ("2010-2018", dict(start="2010-01-01", end="2018-12-31")),
        ("2019+", dict(start="2019-01-01")),
        ("Full", {}),
    ]
    base_half = []
    for wlabel, wkw in windows:
        if wlabel == "Full":
            continue
        eq_b, tr_b = backtest(tickers, P, sectors=sectors, **{**base_kw, **wkw})
        base_half.append(_window_stats(eq_b, tr_b))
    b0, b1 = base_half

    rows = []
    stored = {}
    for name, extra in specs:
        row = {"Variant": name}
        halves = []
        for wlabel, wkw in windows:
            kw = {**base_kw, **extra, **wkw}
            eq, tr = backtest(tickers, P, sectors=sectors, **kw)
            if wlabel == "Full":
                stored[name] = (eq, tr)
            st = _window_stats(eq, tr)
            t = trade_stats(tr)
            row[f"{wlabel} CAGR"] = round(st["CAGR"], 2)
            row[f"{wlabel} DD"] = round(st["DD"], 2)
            row[f"{wlabel} Calmar"] = round(st["Calmar"], 3)
            if wlabel in ("2010-2018", "2019+"):
                halves.append(st)
            if wlabel == "Full":
                row["Trades"] = t.get("Trades", 0)
                row["PF"] = round(t.get("Profit factor", np.nan), 2)
        row["Keep vs live"] = (
            "yes" if name != "entry_long (live baseline)" and _helps((b0, halves[0]), (b1, halves[1])) else ""
        )
        rows.append(row)

    table = pd.DataFrame(rows).set_index("Variant")
    print("\n=== Entry filters (live Long; ATH = high-water close in Yahoo history; "
          f"beta = {BETA_LOOKBACK}d vs Nifty 500) ===")
    print(f"Baseline already requires within 15% of 52-week high. ATH filter: prox_ath >= {ATH_PROX_MIN}. "
          f"Beta filter: beta < {BETA_MAX}.")
    print(table.to_string())
    _ensure_output_dirs()
    table.to_csv(OUTPUT_STUDIES / "entry_filter_study.csv")

    base_full = _window_stats(*stored["entry_long (live baseline)"])
    best_name, best_cal = None, base_full["Calmar"]
    for name in table.index:
        if name == "entry_long (live baseline)":
            continue
        st = _window_stats(*stored[name])
        if st["Calmar"] > best_cal and st["CAGR"] >= base_full["CAGR"] - 5:
            best_cal = st["Calmar"]
            best_name = name
    print(f"\nBest Full Calmar (CAGR within 5pts of live): {best_name} "
          f"(Calmar {best_cal:.3f} vs live {base_full['Calmar']:.3f})")
    return table, best_name


def _study_windows():
    return [
        ("2010-2018", dict(start="2010-01-01", end="2018-12-31")),
        ("2019+", dict(start="2019-01-01")),
        ("Full", {}),
    ]


def _run_spec_grid(tickers, P, sectors, specs, base_kw, label):
    """Run variants across train/test/full; return DataFrame."""
    rows = []
    for name, extra in specs:
        row = {"Study": label, "Variant": name}
        for wlabel, wkw in _study_windows():
            kw = {**base_kw, **extra, **wkw}
            eq, tr = backtest(tickers, P, sectors=sectors, **kw)
            st = _window_stats(eq, tr)
            row[f"{wlabel} CAGR"] = round(st["CAGR"], 2)
            row[f"{wlabel} DD"] = round(st["DD"], 2)
            row[f"{wlabel} Calmar"] = round(st["Calmar"], 3)
            if wlabel == "Full":
                row["Trades"] = len(tr)
                liq = eq.attrs.get("equity_liquidation")
                row["Final MTM"] = round(eq.dropna().iloc[-1])
                row["Final liq est"] = round(liq.dropna().iloc[-1]) if liq is not None else np.nan
                if not tr.empty:
                    open_p = tr[tr.reason == "open_at_end"]
                    row["Open PnL net"] = round(open_p.pnl.sum()) if len(open_p) else 0
        rows.append(row)
    return pd.DataFrame(rows)


def validation_suite(tickers, P, sectors):
    """Execution realism + peak stop + costs on live Long (no auto-adopt)."""
    parts = []
    base_long = adopted_long_kwargs(P)

    print("\n=== 1) Mark-to-market vs estimated liquidation (live Long, full sample) ===")
    eq, tr = backtest(tickers, P, sectors=sectors, **base_long)
    liq = eq.attrs.get("equity_liquidation")
    p = perf_stats(eq.dropna(), "MTM")
    pl = perf_stats(liq.dropna(), "Liquidation est") if liq is not None else {}
    print(f"  Final MTM equity: ₹{p['Final value']:,.0f}  |  After exit costs on open legs: ₹{pl.get('Final value', np.nan):,.0f}")
    print(f"  Gap: ₹{p['Final value'] - pl.get('Final value', p['Final value']):,.0f} "
          f"({(1 - pl.get('Final value', p['Final value']) / p['Final value']) * 100:.2f}% of MTM)")
    if not tr.empty:
        op = tr[tr.reason == "open_at_end"]
        print(f"  Open positions: {len(op)} | Net open PnL (incl. costs): ₹{op.pnl.sum():,.0f}")

    print("\n=== 2) Peak drawdown stop grid (live Long) ===")
    peak_specs = [
        ("no peak stop", dict(dd_stop=None)),
        ("peak 20%", dict(dd_stop=0.20)),
        ("peak 25%", dict(dd_stop=0.25)),
        ("peak 30% (live)", dict(dd_stop=0.30)),
        ("peak 35%", dict(dd_stop=0.35)),
    ]
    peak_df = _run_spec_grid(tickers, P, sectors, peak_specs, base_long, "peak_stop")
    print(peak_df.to_string(index=False))
    parts.append(peak_df)
    eq30, tr30 = backtest(tickers, P, sectors=sectors, **{**base_long, "dd_stop": 0.30})
    print("  Top-10 trade share (30% peak):", pd.Series(_jackpot(tr30)).round(2).to_string())

    print("\n=== 3) Cost stress per side (live Long) ===")
    cost_specs = [(f"cost {int(c*10000)/100}%/side", dict(cost_per_side=c)) for c in (0.0015, 0.0025, 0.004)]
    cost_df = _run_spec_grid(tickers, P, sectors, cost_specs, base_long, "cost")
    print(cost_df.to_string(index=False))
    parts.append(cost_df)

    rank_path = OUTPUT_STUDIES / "rank_momentum_study.csv"
    if rank_path.exists():
        print("\n=== 4) Rank momentum study (saved) — prefer both halves Calmar up ===")
        print(pd.read_csv(rank_path).to_string(index=False))
    else:
        print("\n=== 4) Rank momentum — run: python main.py study rank ===")

    out = pd.concat(parts, ignore_index=True)
    _ensure_output_dirs()
    out.to_csv(OUTPUT_STUDIES / "validation_suite.csv", index=False)
    print(f"\nSaved {OUTPUT_STUDIES / 'validation_suite.csv'} (no live parameters changed).")
    return out


def run_validation_suite():
    pd.set_option("display.width", 240)
    data, universe = load_universe()
    bench, _ = _load_nifty500_close(None)
    tickers, P = build_panels(data, universe=universe, bench_close=bench)
    sectors = [load_sectors(tickers).get(t, "") for t in tickers]
    return validation_suite(tickers, P, sectors)


def run_entry_filter_study():
    pd.set_option("display.width", 220)
    data, universe = load_universe()
    bench, _ = _load_nifty500_close(None)
    if bench is None:
        raise SystemExit("Nifty 500 history required for beta filter.")
    tickers, P = build_panels(data, universe=universe, bench_close=bench)
    sectors = [load_sectors(tickers).get(t, "") for t in tickers]
    table, best = entry_filter_study(tickers, P, sectors)
    if best and "live baseline" not in best:
        print(f"\nCandidate entry: {best} — review output/studies/entry_filter_study.csv")
    else:
        print("\nNo entry filter beat live baseline on Full Calmar (within CAGR guard). LONG_BOOK unchanged.")
    return table, best


def run_rank_momentum_study():
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    data, universe = load_universe()
    bench, _ = _load_nifty500_close(None)
    tickers, P = build_panels(data, universe=universe, bench_close=bench)
    sectors = [load_sectors(tickers).get(t, "") for t in tickers]
    table, best = rank_momentum_study(tickers, P, sectors)
    if best and best != "rank roc126 (live)":
        print(f"\nCandidate rank: {best} — review output/studies/rank_momentum_study.csv")
    else:
        print("\nNo rank variant beat live roc252 on Full Calmar (within CAGR guard). LONG_BOOK unchanged.")
    return table, best


def run_atr_exit_study():
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    data, universe = load_universe()
    bench, _ = _load_nifty500_close(None)
    tickers, P = build_panels(data, universe=universe, bench_close=bench)
    sectors = [load_sectors(tickers).get(t, "") for t in tickers]
    _, port, best = atr_exit_study(tickers, P, sectors)
    if best:
        print(f"\nCandidate to adopt: {best} — review atr_exit_portfolio.csv before changing live book.")
    else:
        print("\nNo ATR variant beat baseline Calmar (within CAGR guard). Live book unchanged.")
    return port, best


def run_compare_long():
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    data, universe = load_universe()
    bench, _ = _load_nifty500_close(None)
    tickers, P = build_panels(data, universe=universe, bench_close=bench)
    sectors = [load_sectors(tickers).get(t, "") for t in tickers]
    survivorship_sanity(tickers, P, sectors, data=data, universe=universe)
    compare_long_improvements(tickers, P, sectors)


if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"
    # Primary commands
    if cmd in ("help", "-h", "--help"):
        cli_help()
    elif cmd in ("backtest", "bt"):
        run_backtest()
    elif cmd in ("rolling", "rolling-returns"):
        run_rolling_returns()
    elif cmd in ("live", "long-next", "--long-next"):
        run_long_next()
    elif cmd == "full":
        main()
    elif cmd == "study" and len(sys.argv) > 2:
        sub = sys.argv[2]
        if sub == "rank":
            run_rank_momentum_study()
        elif sub in ("entry", "entry-filter"):
            run_entry_filter_study()
        elif sub in ("validate", "validation"):
            run_validation_suite()
        elif sub == "atr":
            run_atr_exit_study()
        elif sub == "compare":
            run_compare_long()
        elif sub == "universe":
            data, universe = load_universe()
            bench, _ = _load_nifty500_close(None)
            tickers, P = build_panels(data, universe=universe, bench_close=bench)
            sectors = [load_sectors(tickers).get(t, "") for t in tickers]
            survivorship_sanity(tickers, P, sectors, data=data, universe=universe)
        elif sub == "sep2024":
            _ensure_output_dirs()
            write_trades_from_sep2024(path=str(OUTPUT_STUDIES / "TRADES_FROM_SEP2024.md"))
        elif sub in ("regime", "regime-cash", "gold-regime"):
            run_regime_cash_study()
        else:
            cli_help()
    # Legacy flags (same as above)
    elif cmd in ("--refresh-chart", "refresh-chart"):
        run_backtest()
    elif cmd in ("--compare-long", "compare-long"):
        run_compare_long()
    elif cmd in ("--trades-sep2024", "trades-sep2024"):
        _ensure_output_dirs()
        write_trades_from_sep2024(path=str(OUTPUT_STUDIES / "TRADES_FROM_SEP2024.md"))
    elif cmd in ("--atr-exit-study", "atr-exit-study"):
        run_atr_exit_study()
    elif cmd in ("--rank-study", "rank-study"):
        run_rank_momentum_study()
    elif cmd in ("--entry-filter-study", "entry-filter-study"):
        run_entry_filter_study()
    elif cmd in ("--validate", "validate", "--validation-suite"):
        run_validation_suite()
    elif cmd in ("--universe-report", "universe-report"):
        data, universe = load_universe()
        bench, _ = _load_nifty500_close(None)
        tickers, P = build_panels(data, universe=universe, bench_close=bench)
        sectors = [load_sectors(tickers).get(t, "") for t in tickers]
        survivorship_sanity(tickers, P, sectors, data=data, universe=universe)
    else:
        cli_help()

# ----------------------------- NOTES -----------------------------------------
# * Survivorship: delisted names are still missing unless listed in delisted_symbols.txt with cached
#   prices. PIT mode uses listing dates, price-scaled mcap (+ shares cap when known), and a lower
#   download floor (PIT_UNIVERSE_FLOOR_CR) than the tradable MIN_MCAP_CR. Run --universe-report.
# * Prices are dividend/split adjusted by Yahoo; Yahoo's Indian data has occasional errors, so
#   cross-check with a broker API (Kite Connect, Upstox, Angel One) before trusting results.
# * Taxes are not modelled (STCG on equity delivery is taxed in India).
# * Don't tune parameters until the equity curve looks good: that is overfitting. Use a
#   train/test split (e.g. tune on 2010-2018, validate on 2019 onwards).