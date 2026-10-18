"""
Monthly Long book: data download + portfolio backtest (India, 2000 onwards)

Live defaults (mcap-grid study): top 800 by today's mcap among ≥ ₹2,000 cr
from NSE EQUITY_L series EQ + BE + BZ (Trade-for-Trade kept — large names often
sit in BE/BZ around corporate actions), 10 slots, industry cap 3, Nifty500 Equal
Weight > SMA200 block-new-buys (CNX500 / ^CRSLDX before Equal Weight history,
~2024-06). Every load_universe() / live / backtest / study pick rebuilds that list.
See LONG_STRATEGY.md / RUN.md.

Setup:   pip install yfinance pandas numpy matplotlib
Run:     python main.py download | python main.py fix-data | python main.py backtest | live
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd

# ----------------------------- CONFIG ---------------------------------------
START = "2000-01-01"
END = None                      # None = today
BENCHMARK = "^NSEI"             # Nifty 50 index
CACHE_DIR = Path("data_cache")
OUTPUT_BACKTEST = Path("output/backtest")
OUTPUT_LIVE = Path("output/live")
OUTPUT_STUDIES = Path("output/studies")

# Data source mode (prices + mcap meta):
#   "cache" — local CSVs only (default for live / backtest / study)
#   "sync"  — keep cache; Yahoo only for missing files / truncated starts / stale ends
#             (end updates always append recent bars — never re-pull full history)
#   "force" — full Yahoo re-download from START (escape hatch for broken caches)
DATA_MODE = "cache"
_DATA_MODE_CACHE = frozenset({"cache", "cached", "cache-only", "offline"})
_DATA_MODE_SYNC = frozenset({"sync", "yahoo", "from-yahoo", "pull"})
_DATA_MODE_FORCE = frozenset({"force", "resync", "refresh"})


def set_data_mode(mode="cache"):
    """Set global DATA_MODE to cache | sync | force."""
    global DATA_MODE
    m = str(mode).strip().lower().lstrip("-")
    if m in _DATA_MODE_FORCE or m in ("f",):
        DATA_MODE = "force"
    elif m in _DATA_MODE_SYNC:
        DATA_MODE = "sync"
    else:
        DATA_MODE = "cache"
    return DATA_MODE


def parse_data_mode(argv, default="cache"):
    """
    Parse CLI data-source flags from argv.
      --cache / --cached / --cache-only  → cache (local only)
      --sync / --yahoo / --from-yahoo    → sync (incremental end-fill + missing)
      --force / -f / --resync            → force (full history re-pull)
    """
    argv = list(argv or [])
    mode = default
    if any(a in argv for a in ("--force", "-f", "--resync")):
        mode = "force"
    elif any(a in argv for a in ("--sync", "--yahoo", "--from-yahoo", "--refresh-mcap")):
        mode = "sync"
    elif any(a in argv for a in ("--cache", "--cached", "--cache-only", "--offline")):
        mode = "cache"
    return set_data_mode(mode)


def _ensure_output_dirs():
    for d in (OUTPUT_BACKTEST, OUTPUT_LIVE, OUTPUT_STUDIES):
        d.mkdir(parents=True, exist_ok=True)

# --- Universe (live India book) ---
# Chosen from study-mcap-grid --from-2000: top 800 by today's mcap among ≥ ₹2,000 cr,
# 10 slots; uniform NSE sector_cap=3 (2015+ rolling + from-2021/2024 favor cap3 over adaptive).
# Symbol list = NSE EQUITY_L filtered by NSE_UNIVERSE_SERIES (EQ + BE + BZ).
MIN_MCAP_CR = 2000              # floor; tradable only when ESTIMATED mcap on that date >= this
LIVE_UNIVERSE_TOP_N = 800       # keep largest N names by today's mcap among the floor set
LIVE_UNIVERSE_LIST_FILE = OUTPUT_LIVE / "universe_top800_by_mcap.csv"
# "point_in_time": tradable when est. mcap + liquidity + listing cooldown pass (see estimate_mcap_cr)
# "today":         stocks with mcap >= MIN_MCAP_CR today are tradable for all history (upper-bound bias)
MCAP_MODE = "point_in_time"
PIT_UNIVERSE_FLOOR_CR = 500     # legacy floor (kept for US mapping)
LISTING_COOLDOWN_DAYS = 126     # no entries until this many calendar days after NSE listing date
DELISTED_SYMBOLS_FILE = Path("delisted_symbols.txt")  # optional extras (merged/delisted) to also download
MIN_TURNOVER = 2e7              # INR 2 crore: 20-day median daily traded value needed to enter (liquidity)
MAX_ABS_DAILY_MOVE = 0.60       # drop symbol if a >60% one-day move remains after repair
SPIKE_REPAIR_MOVE = 0.35        # null day t when |r_t| exceeds this and next day reverts
SPIKE_REVERT_MAX = 0.08         # |(1+r_t)*(1+r_{t+1})-1| below this => treat as spike-and-revert
ISOLATED_SPIKE_MOVE = 0.50      # null isolated bar if |r_t| exceeds this and neighbors look normal
MIN_BARS = 300                  # minimum history to keep a symbol
CACHE_START_SLACK_DAYS = 30     # cache needs full re-pull if first bar is later than START + this (and listing allows)
CACHE_END_SLACK_DAYS = 3        # end is current if last bar within this many calendar days of target (weekends/holidays)
CACHE_END_OVERLAP_DAYS = 7      # re-fetch this many days before last bar when appending (corp-action adjust)

# --- Strategy / portfolio ---
INITIAL_CAPITAL = 1_000_000     # INR
MAX_POSITIONS = 10              # breakout reference book only
LONG_POSITIONS = 10             # monthly live book: equal-weight tenths of equity
RISK_PER_TRADE = 0.01           # 1% of equity risked per trade
MAX_WEIGHT = 0.15               # no position above 15% of equity
ATR_MULT = 3.0                  # trailing stop distance in ATRs
HIGH_WINDOW = 252               # trading days in the 52-week high
BETA_LOOKBACK = 252             # rolling beta vs Nifty 500 for entry filter
BETA_MAX = 1.25                 # require beta < this at signal (when filter enabled)
ATH_PROX_MIN = 0.75             # close >= 75% of series high-water mark (within 25% of ATH)
SECTOR_CAP = 3                  # default max open positions in one NSE industry
SECTOR_CAP_LARGE = 4            # cap for large NSE industries (see LARGE_INDUSTRY_MIN_NAMES)
LARGE_INDUSTRY_MIN_NAMES = 50   # tag count in sectors.csv → treat as large
USE_ADAPTIVE_SECTOR_CAP = False  # live uses SECTOR_CAP=3; set True for large=4 / others=3
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
NIFTY500_LIST_URL = "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv"
NIFTY500_LIST_FILE = Path("ind_nifty500list.csv")
# Official Nifty500 Equal Weight daily closes from NSE index archives (launched ~2024-05-31).
NIFTY500_EW_INDEX_NAME = "Nifty500 Equal Weight"
NIFTY500_EW_CACHE = CACHE_DIR / "nifty500_equal_weight.csv"
NSE_IND_CLOSE_URL = (
    "https://nsearchives.nseindia.com/content/indices/ind_close_all_{ddmmyyyy}.csv"
)
NIFTY500_EW_ARCHIVE_START = "2024-06-03"  # first archive days with this index name
# Live / Nifty-500 universe: normal EQ plus Trade-for-Trade (BE/BZ). NSE parks
# large liquid names in BE/BZ around corporate actions; excluding them dropped
# STLTECH / HFCL / MTARTECH from the top-800 mcap book.
NSE_UNIVERSE_SERIES = frozenset({"EQ", "BE", "BZ"})


def _series_in_universe(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    return s.isin(NSE_UNIVERSE_SERIES) | s.isin(["", "nan"]) | series.isna()


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
    df = df[_series_in_universe(df["SERIES"])].copy()
    df["SYMBOL"] = df["SYMBOL"].astype(str).str.strip()
    df = df.drop_duplicates(subset=["SYMBOL"], keep="first")
    list_col = next((c for c in df.columns if "DATE OF LISTING" in c.upper()), None)
    if list_col:
        df["listing"] = pd.to_datetime(df[list_col], format="%d-%b-%Y", errors="coerce")
    else:
        df["listing"] = pd.NaT
    return df


def _read_nifty500_list(refresh=False):
    """Today's Nifty 500 constituents from NSE archives (cached as ind_nifty500list.csv)."""
    local = NIFTY500_LIST_FILE
    if refresh or not local.exists():
        import requests
        try:
            r = requests.get(NIFTY500_LIST_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
            r.raise_for_status()
            local.write_bytes(r.content)
        except Exception as e:
            if local.exists():
                print(f"Warning: could not refresh Nifty 500 list ({e}); using cached {local}")
            else:
                raise SystemExit(
                    f"Could not download the Nifty 500 list ({e}).\n"
                    f"Download it manually from {NIFTY500_LIST_URL} and save as {local.name}.")
    df = pd.read_csv(local)
    df.columns = df.columns.str.strip()
    sym_col = next((c for c in df.columns if c.upper() == "SYMBOL"), None)
    if sym_col is None:
        raise SystemExit(f"{local} has no Symbol column: {list(df.columns)}")
    df["SYMBOL"] = df[sym_col].astype(str).str.strip()
    series_col = next((c for c in df.columns if c.upper() == "SERIES"), None)
    if series_col is not None:
        df = df[_series_in_universe(df[series_col])]
    return df


def get_nifty500_symbols(refresh=False):
    """Symbols in today's Nifty 500 index list (skips NSE dummy/test rows)."""
    df = _read_nifty500_list(refresh=refresh)
    isin_col = next((c for c in df.columns if "ISIN" in c.upper()), None)
    mask = ~df["SYMBOL"].str.upper().str.startswith("DUMMY")
    if isin_col is not None:
        isin = df[isin_col].astype(str).str.upper()
        mask &= ~isin.str.startswith("DUM")
    return sorted(df.loc[mask, "SYMBOL"].unique())


def get_nse_symbols():
    """NSE symbols from EQUITY_L.csv in NSE_UNIVERSE_SERIES (EQ + BE + BZ)."""
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


def get_universe_meta(symbols, refresh=False, fetch_missing=None):
    """Per-symbol mcap (₹ cr), shares, listing date. Cached in data_cache/universe_meta.csv.

    Respects DATA_MODE unless overridden:
      cache → CSV only
      sync  → Yahoo for symbols missing mcap
      force / refresh=True → Yahoo for missing (and rewrite cache)
    """
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
    if fetch_missing is None:
        fetch_missing = bool(refresh) or DATA_MODE in ("sync", "force")
    if todo and fetch_missing:
        print(f"Fetching market cap / shares for {len(todo)} symbols via Yahoo "
              f"(mode={DATA_MODE})...")
        with ThreadPoolExecutor(max_workers=8) as ex:
            rows = list(ex.map(_fetch_yahoo_meta, todo))
        new = pd.DataFrame(
            [(m / 1e7 if np.isfinite(m) else np.nan, sh) for m, sh in rows],
            index=todo, columns=["mcap_cr", "shares"],
        )
        cached = cached.drop(index=[s for s in todo if s in cached.index], errors="ignore")
        cached = pd.concat([cached, new])
        cached.to_csv(f)
    elif todo:
        print(f"Meta cache-only: {len(todo)} symbols missing mcap "
              f"(use --sync / --yahoo to pull)")
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


def estimate_mcap_usd(close, mcap_now_usd, shares=None):
    """Point-in-time market cap in USD (same logic as estimate_mcap_cr, US units)."""
    if mcap_now_usd is None or not np.isfinite(mcap_now_usd) or mcap_now_usd <= 0:
        return pd.Series(np.nan, index=close.index)
    last = close.iloc[-1]
    if not np.isfinite(last) or last <= 0:
        return pd.Series(np.nan, index=close.index)
    scaled = mcap_now_usd * close / last
    if shares is not None and np.isfinite(shares) and shares > 0:
        from_shares = shares * close
        return pd.concat([scaled, from_shares], axis=1).min(axis=1)
    return scaled


# ----------------------------- DATA -----------------------------------------
def _clean(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    need = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        return pd.DataFrame(columns=need)
    df = df[need].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df[~df.index.duplicated()].sort_index()
    return df


def fix_ohlcv(df, spike_move=SPIKE_REPAIR_MOVE, revert_max=SPIKE_REVERT_MAX,
              isolated_move=ISOLATED_SPIKE_MOVE, drop_move=MAX_ABS_DAILY_MOVE,
              junk_gap_days=60):
    """
    Post-Yahoo cleanup: drop zero-volume bars, OHLC clamp, spike-and-revert nulling,
    isolated spike nulling. Non-reverting mega-gaps: trim only after a long calendar
    hole (junk early history); otherwise back-adjust the prefix (demerger / missed adjust).
    Returns (cleaned_df, stats_dict).
    """
    stats = {
        "bars_nulled": 0, "ohlc_clamped": 0, "rows_bad_price": 0,
        "zero_vol_dropped": 0, "history_trimmed": 0, "history_backadjusted": 0,
    }
    if df is None or df.empty:
        return df, stats
    out = _clean(df)
    if out.empty:
        return out, stats

    px = out[["Open", "High", "Low", "Close"]]
    bad = (px <= 0).any(axis=1) | px.isna().any(axis=1)
    stats["rows_bad_price"] = int(bad.sum())
    out = out.loc[~bad].copy()
    if out.empty:
        return out, stats

    zv = out["Volume"].fillna(0) <= 0
    stats["zero_vol_dropped"] = int(zv.sum())
    out = out.loc[~zv].copy()
    if out.empty:
        return out, stats

    hi = out[["Open", "Close", "High"]].max(axis=1)
    lo = out[["Open", "Close", "Low"]].min(axis=1)
    clamp = (out["High"] < hi) | (out["Low"] > lo)
    stats["ohlc_clamped"] = int(clamp.sum())
    out["High"] = hi
    out["Low"] = lo

    close = out["Close"].astype(float)
    ret = close.pct_change()
    null_mask = pd.Series(False, index=out.index)
    # Spike day t that mostly reverts on t+1 → null day t
    for i in range(1, len(out) - 1):
        r0 = ret.iloc[i]
        r1 = ret.iloc[i + 1]
        if not (np.isfinite(r0) and np.isfinite(r1)):
            continue
        if abs(r0) < spike_move:
            continue
        two_day = (1.0 + r0) * (1.0 + r1) - 1.0
        if abs(two_day) <= revert_max:
            null_mask.iloc[i] = True
    # Isolated extreme spike with calm neighbors
    for i in range(1, len(out) - 1):
        if null_mask.iloc[i]:
            continue
        r0 = ret.iloc[i]
        if not np.isfinite(r0) or abs(r0) < isolated_move:
            continue
        prev_r = ret.iloc[i - 1] if i > 0 else np.nan
        next_r = ret.iloc[i + 1]
        calm_prev = (not np.isfinite(prev_r)) or abs(prev_r) < spike_move * 0.5
        calm_next = (not np.isfinite(next_r)) or abs(next_r) < spike_move * 0.5
        if calm_prev and calm_next:
            null_mask.iloc[i] = True

    stats["bars_nulled"] = int(null_mask.sum())
    if null_mask.any():
        out.loc[null_mask, ["Open", "High", "Low", "Close", "Volume"]] = np.nan
        out = out.dropna(subset=["Open", "High", "Low", "Close"]).copy()

    # Non-reverting mega-gaps: trim junk prefix after long calendar holes; else back-adjust
    trimmed = 0
    backadjusted = 0
    guard = 0
    while len(out) > 2 and guard < 8:
        guard += 1
        ret = out["Close"].astype(float).pct_change()
        mx = float(ret.abs().max()) if len(ret) else 0.0
        if not np.isfinite(mx) or mx <= drop_move:
            break
        i = int(np.nanargmax(ret.abs().to_numpy()))
        if i <= 0:
            break
        r0 = float(ret.iloc[i])
        r1 = float(ret.iloc[i + 1]) if i + 1 < len(out) else np.nan
        two_day = (1.0 + r0) * (1.0 + r1) - 1.0 if np.isfinite(r1) else np.nan
        if np.isfinite(two_day) and abs(two_day) <= revert_max:
            out = out.drop(out.index[i]).copy()
            trimmed += 1
            continue
        gap_days = (out.index[i] - out.index[i - 1]).days
        prev_close = float(out["Close"].iloc[i - 1])
        open_i = float(out["Open"].iloc[i])
        if not (np.isfinite(prev_close) and prev_close > 0):
            break
        # Long hole before the jump → unusable early Yahoo stub; keep from gap bar
        if gap_days >= junk_gap_days:
            trimmed += i
            out = out.iloc[i:].copy()
            continue
        # Demerger / missed corporate adjust: scale prefix so the gap disappears
        factor = open_i / prev_close if np.isfinite(open_i) and open_i > 0 else float(out["Close"].iloc[i]) / prev_close
        if not np.isfinite(factor) or factor <= 0:
            break
        cols = ["Open", "High", "Low", "Close"]
        out.loc[out.index[:i], cols] = out.loc[out.index[:i], cols].to_numpy() * factor
        backadjusted += 1
    stats["history_trimmed"] = trimmed
    stats["history_backadjusted"] = backadjusted
    return out, stats


def _cache_file(symbol):
    return CACHE_DIR / (symbol.replace("^", "IDX_").replace("&", "_") + ".csv")


def _yf_equity(symbol):
    """Map bare NSE symbol or index ticker to Yahoo symbol used in cache filenames."""
    if symbol.startswith("^") or symbol.endswith(".NS") or symbol.endswith(".BO"):
        return symbol
    return symbol + ".NS"


def _cache_first_date(path):
    try:
        idx = pd.read_csv(path, usecols=[0], index_col=0, parse_dates=True).index
        if len(idx) == 0:
            return None
        return pd.Timestamp(idx.min()).tz_localize(None)
    except Exception:
        return None


def _cache_target_end(end=END):
    """Inclusive last session we want cached (normalize to midnight)."""
    ts = pd.Timestamp(end) if end else pd.Timestamp.now()
    if getattr(ts, "tzinfo", None) is not None:
        ts = ts.tz_localize(None)
    return ts.normalize()


def _yahoo_end_exclusive(end=END):
    """yfinance `end` is exclusive — pass the day after the target session."""
    target = _cache_target_end(end)
    return (target + pd.Timedelta(days=1)).strftime("%Y-%m-%d")


def _india_cache_last_date(path):
    try:
        idx = pd.read_csv(path, usecols=[0], index_col=0, parse_dates=True).index
        if len(idx) == 0:
            return None
        return pd.Timestamp(idx.max()).tz_localize(None).normalize()
    except Exception:
        return None


def _india_cache_needs_end_fill(path, end=END, slack_days=None):
    """True if last cached bar is behind the target session (minus slack)."""
    slack = CACHE_END_SLACK_DAYS if slack_days is None else int(slack_days)
    last = _india_cache_last_date(path)
    if last is None:
        return True
    target = _cache_target_end(end)
    return last < target - pd.Timedelta(days=slack)


def _incremental_start(path, fallback_start=START, overlap_days=None):
    """Yahoo start date for end-fill: last bar minus overlap (corp-action adjust)."""
    overlap = CACHE_END_OVERLAP_DAYS if overlap_days is None else int(overlap_days)
    last = _india_cache_last_date(path)
    if last is None:
        return fallback_start
    start = last - pd.Timedelta(days=overlap)
    return start.strftime("%Y-%m-%d")


def _merge_ohlcv_cache(existing, new_df):
    """Append/override overlapping dates; keep newest Yahoo bars on collide."""
    if existing is None or getattr(existing, "empty", True):
        return new_df
    if new_df is None or getattr(new_df, "empty", True):
        return existing
    a = _clean(existing)
    b = _clean(new_df)
    if a.empty:
        return b
    if b.empty:
        return a
    combined = pd.concat([a, b])
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    return combined


def _india_cache_is_usable(path, start=START, listing=None, min_bars=MIN_BARS):
    """True if cache CSV has enough history / start coverage (end freshness separate).

    Rules:
    - File must exist with >= min_bars.
    - If listing date is known: first bar should reach max(start, listing) (± slack).
      Used to detect truncated caches that need a deeper re-download.
    - If listing is unknown: trust any adequate cache. Requiring START=2000 for
      post-IPO names with no listing caused endless re-downloads of the same CSVs.
    End-of-series freshness is handled by _india_cache_needs_end_fill (sync appends).
    """
    if not path.exists():
        return False
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
    except Exception:
        return False
    if len(df) < min_bars:
        return False
    first = pd.Timestamp(df.index.min()).tz_localize(None)
    if listing is None or (isinstance(listing, float) and not np.isfinite(listing)) or pd.isna(listing):
        return True
    start_ts = pd.Timestamp(start)
    slack = pd.Timedelta(days=CACHE_START_SLACK_DAYS)
    expect = max(start_ts, pd.Timestamp(listing))
    return first <= expect + slack


def _yf_download_ohlcv(ysym, start=START, end=END):
    """Yahoo download with repair=True; fall back to repair=False on Yahoo bugs.

    yfinance often prints a false 'Failed download' / IntCastingNaNError for Indian
    names when repair=True; stderr is silenced and we retry without repair.
    `end` is treated as an inclusive session date (converted to exclusive for Yahoo).
    """
    import yfinance as yf
    import warnings
    import os
    from contextlib import redirect_stderr

    y_end = _yahoo_end_exclusive(end) if end else None

    def _one(repair):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with open(os.devnull, "w") as devnull, redirect_stderr(devnull):
                try:
                    return yf.download(
                        ysym, start=start, end=y_end, auto_adjust=True, repair=repair,
                        progress=False,
                    )
                except Exception:
                    return None

    df = _one(True)
    if df is None or df.empty:
        df = _one(False)
    return df


def download_one(symbol, start=START, end=END, force=False, listing=None,
                 end_slack_days=None):
    """Single symbol (equities use .NS suffix; indices like ^NSEI as-is). Cached as CSV.

    Honours DATA_MODE:
      cache → never Yahoo
      sync  → full pull only if missing/unusable; else incremental end-fill if stale
      force → full Yahoo re-download from start
    """
    CACHE_DIR.mkdir(exist_ok=True)
    ysym = _yf_equity(symbol)
    f = _cache_file(ysym)
    do_force = force or DATA_MODE == "force"
    if DATA_MODE == "cache" and not do_force:
        if f.exists():
            try:
                df = pd.read_csv(f, index_col=0, parse_dates=True)
                return df if len(df) else None
            except Exception:
                return None
        return None

    if do_force or not f.exists() or not _india_cache_is_usable(f, start=start, listing=listing):
        raw = _yf_download_ohlcv(ysym, start=start, end=end)
        if raw is None or raw.empty:
            if f.exists():
                try:
                    return pd.read_csv(f, index_col=0, parse_dates=True)
                except Exception:
                    return None
            return None
        df, _ = fix_ohlcv(raw)
        if df is None or df.empty:
            return None
        df.to_csv(f)
        return df

    # Structurally OK — append only if end is stale
    if not _india_cache_needs_end_fill(f, end=end, slack_days=end_slack_days):
        try:
            return pd.read_csv(f, index_col=0, parse_dates=True)
        except Exception:
            return None

    try:
        existing = pd.read_csv(f, index_col=0, parse_dates=True)
    except Exception:
        existing = None
    raw = _yf_download_ohlcv(ysym, start=_incremental_start(f, fallback_start=start), end=end)
    if raw is None or raw.empty:
        return existing
    merged = _merge_ohlcv_cache(existing, raw)
    df, _ = fix_ohlcv(merged)
    if df is None or df.empty:
        return existing
    df.to_csv(f)
    return df


def download_many(symbols, batch=20, start=START, end=END, force=False, listings=None,
                  sequential=True, end_slack_days=None):
    """
    Download adjusted daily OHLCV for NSE symbols with yfinance repair=True,
    then custom fix_ohlcv. Cached per symbol as SYMBOL.NS.csv.

    sync: missing/truncated → full history; end-stale → incremental append only.
    force: full history re-pull for every symbol.
    """
    import yfinance as yf
    CACHE_DIR.mkdir(exist_ok=True)
    _ensure_output_dirs()
    listings = listings if listings is not None else {}
    yf_syms = {s: _yf_equity(s) for s in symbols}
    do_force = force or DATA_MODE == "force"
    if DATA_MODE == "cache" and not do_force:
        print(f"Data mode=cache: loading {len(symbols)} symbols from disk (no Yahoo)")
        return _load_cached_ohlcv(symbols, listings=listings)

    need_full, need_end = [], []
    for s in symbols:
        f = _cache_file(yf_syms[s])
        if do_force or not _india_cache_is_usable(f, start=start, listing=listings.get(s)):
            need_full.append(s)
        elif _india_cache_needs_end_fill(f, end=end, slack_days=end_slack_days):
            need_end.append(s)

    print(
        f"{len(symbols) - len(need_full) - len(need_end)} current, "
        f"{len(need_end)} end-fill, {len(need_full)} full "
        f"(start={start}, end={_cache_target_end(end).date()}, "
        f"mode={DATA_MODE}, force={do_force})..."
    )

    repair_rows = []

    def _save_fixed(s, raw_df, merge_existing=False):
        try:
            f = _cache_file(yf_syms[s])
            if merge_existing and f.exists():
                try:
                    existing = pd.read_csv(f, index_col=0, parse_dates=True)
                except Exception:
                    existing = None
                raw_df = _merge_ohlcv_cache(existing, raw_df)
            df, stats = fix_ohlcv(raw_df)
            if df is None or df.empty:
                repair_rows.append({"symbol": s, "bars_nulled": stats.get("bars_nulled", 0),
                                    "ohlc_clamped": stats.get("ohlc_clamped", 0),
                                    "status": "empty_after_fix"})
                return
            df.to_csv(f)
            repair_rows.append({"symbol": s, "bars_nulled": stats["bars_nulled"],
                                "ohlc_clamped": stats["ohlc_clamped"],
                                "status": "ok_endfill" if merge_existing else "ok"})
        except Exception as e:
            repair_rows.append({"symbol": s, "bars_nulled": 0, "ohlc_clamped": 0,
                                "status": f"fix_error:{e}"})

    def _pull_list(syms, full):
        if not syms:
            return
        label = "full" if full else "end-fill"
        if sequential:
            for i, s in enumerate(syms, 1):
                try:
                    y_start = start if full else _incremental_start(
                        _cache_file(yf_syms[s]), fallback_start=start)
                    raw = _yf_download_ohlcv(yf_syms[s], start=y_start, end=end)
                    if raw is None or raw.empty:
                        repair_rows.append({"symbol": s, "bars_nulled": 0, "ohlc_clamped": 0,
                                            "status": "download_empty"})
                    else:
                        _save_fixed(s, raw, merge_existing=not full)
                except Exception as e:
                    repair_rows.append({"symbol": s, "bars_nulled": 0, "ohlc_clamped": 0,
                                        "status": f"download_error:{e}"})
                if i % 25 == 0 or i == len(syms):
                    print(f"  {label} {i}/{len(syms)}", flush=True)
        else:
            y_end = _yahoo_end_exclusive(end) if end else None
            for k in range(0, len(syms), batch):
                chunk = syms[k:k + batch]
                # Batch end-fill uses the oldest incremental start in the chunk
                if full:
                    y_start = start
                else:
                    starts = [
                        pd.Timestamp(_incremental_start(_cache_file(yf_syms[s]), fallback_start=start))
                        for s in chunk
                    ]
                    y_start = min(starts).strftime("%Y-%m-%d")
                try:
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        raw = yf.download(
                            [yf_syms[s] for s in chunk], start=y_start, end=y_end,
                            auto_adjust=True, repair=True, progress=False,
                            group_by="ticker", threads=True,
                        )
                except Exception as e:
                    print(f"  {label} batch failed: {e}", flush=True)
                    continue
                for s in chunk:
                    try:
                        piece = raw[yf_syms[s]].copy() if len(chunk) > 1 else raw.copy()
                        if piece is None or piece.empty:
                            piece = _yf_download_ohlcv(
                                yf_syms[s],
                                start=(start if full else _incremental_start(
                                    _cache_file(yf_syms[s]), fallback_start=start)),
                                end=end,
                            )
                        _save_fixed(s, piece, merge_existing=not full)
                    except Exception as e:
                        repair_rows.append({"symbol": s, "bars_nulled": 0, "ohlc_clamped": 0,
                                            "status": f"parse_error:{e}"})
                print(f"  {label} {min(k + batch, len(syms))}/{len(syms)}", flush=True)

    _pull_list(need_full, full=True)
    _pull_list(need_end, full=False)

    data = {}
    for s in symbols:
        f = _cache_file(yf_syms[s])
        if not f.exists():
            repair_rows.append({"symbol": s, "bars_nulled": 0, "ohlc_clamped": 0,
                                "status": "missing_cache"})
            continue
        df = pd.read_csv(f, index_col=0, parse_dates=True)
        if len(df) < MIN_BARS:
            repair_rows.append({"symbol": s, "bars_nulled": 0, "ohlc_clamped": 0,
                                "status": f"too_short:{len(df)}"})
            continue
        mx = df.Close.pct_change().abs().max()
        if np.isfinite(mx) and mx > MAX_ABS_DAILY_MOVE:
            print(f"  dropped {s}: >{MAX_ABS_DAILY_MOVE:.0%} one-day move after repair (max={mx:.0%})")
            repair_rows.append({"symbol": s, "bars_nulled": 0, "ohlc_clamped": 0,
                                "status": f"dropped_spike:{mx:.2%}"})
            continue
        data[s] = df

    if repair_rows:
        rep = pd.DataFrame(repair_rows).drop_duplicates("symbol", keep="last")
        rep_path = OUTPUT_STUDIES / "nifty500_data_repair_report.csv"
        rep.to_csv(rep_path, index=False)
        print(f"Repair report -> {rep_path} ({len(rep)} rows)")
    print(f"Loaded {len(data)} symbols with usable history")
    return data


def sync_all_cached_prices(end=None, end_slack_days=0, sequential=True, batch=40):
    """Incremental end-fill for every SYMBOL.NS.csv already in data_cache/.

    Only pulls missing recent bars (plus short overlap). Does not re-download
    full history unless a file is missing/unusable (handled inside download_many).
    """
    set_data_mode("sync")
    target = _cache_target_end(end)
    files = sorted(CACHE_DIR.glob("*.NS.csv"))
    symbols = [p.name[:-7] for p in files]  # strip .NS.csv
    print(f"End-fill sync: {len(symbols)} cached equities → through {target.date()} "
          f"(slack={end_slack_days}d)")
    data = download_many(
        symbols, start=START, end=target.strftime("%Y-%m-%d"), force=False,
        sequential=sequential, batch=batch, end_slack_days=end_slack_days,
    )
    # Benchmarks / indices commonly used
    for idx in ("^NSEI", "^CRSLDX", "^CNX500", "NIFTYSMLCAP250.NS"):
        try:
            download_one(idx, end=target.strftime("%Y-%m-%d"), end_slack_days=end_slack_days)
        except Exception as e:
            print(f"  index {idx}: {e}")
    try:
        refresh_nifty500_equal_weight_cache(end=target.strftime("%Y-%m-%d"))
    except Exception as e:
        print(f"  EW refresh: {e}")
    return data


def fix_cached_symbols(symbols, listings=None):
    """Re-run fix_ohlcv on existing cache files; rewrite CSVs; return loadable data dict."""
    _ensure_output_dirs()
    listings = listings if listings is not None else {}
    repair_rows = []
    data = {}
    for s in symbols:
        ysym = _yf_equity(s)
        f = _cache_file(ysym)
        if not f.exists():
            repair_rows.append({"symbol": s, "bars_nulled": 0, "ohlc_clamped": 0,
                                "status": "missing_cache"})
            continue
        raw = pd.read_csv(f, index_col=0, parse_dates=True)
        df, stats = fix_ohlcv(raw)
        if df is None or df.empty or len(df) < MIN_BARS:
            repair_rows.append({"symbol": s, "bars_nulled": stats.get("bars_nulled", 0),
                                "ohlc_clamped": stats.get("ohlc_clamped", 0),
                                "status": "empty_or_short_after_fix"})
            continue
        df.to_csv(f)
        mx = df.Close.pct_change().abs().max()
        if np.isfinite(mx) and mx > MAX_ABS_DAILY_MOVE:
            repair_rows.append({"symbol": s, "bars_nulled": stats["bars_nulled"],
                                "ohlc_clamped": stats["ohlc_clamped"],
                                "status": f"dropped_spike:{mx:.2%}"})
            print(f"  dropped {s}: >{MAX_ABS_DAILY_MOVE:.0%} one-day move after fix (max={mx:.0%})")
            continue
        repair_rows.append({"symbol": s, "bars_nulled": stats["bars_nulled"],
                            "ohlc_clamped": stats["ohlc_clamped"], "status": "ok"})
        data[s] = df
    rep = pd.DataFrame(repair_rows)
    rep_path = OUTPUT_STUDIES / "nifty500_data_repair_report.csv"
    rep.to_csv(rep_path, index=False)
    print(f"Fixed {len(data)}/{len(symbols)} symbols; report -> {rep_path}")
    return data


def load_universe(force=False, refresh_list=False, cache_only=None, data_mode=None):
    """
    Live India universe: top LIVE_UNIVERSE_TOP_N by today's mcap among ≥ MIN_MCAP_CR
    (+ optional delisted_symbols.txt). Returns (data, universe_meta).

    data_mode / DATA_MODE: cache | sync | force (see parse_data_mode).
    Legacy: force=True → force; cache_only=True → cache; cache_only=False → sync.
    """
    if data_mode is not None:
        set_data_mode(data_mode)
    elif force:
        set_data_mode("force")
    elif cache_only is True:
        set_data_mode("cache")
    elif cache_only is False:
        set_data_mode("sync")
    # else keep current DATA_MODE
    print(f"Data mode: {DATA_MODE}")
    data, meta, _ranked = load_mcap_floor_universe(
        min_mcap_cr=MIN_MCAP_CR,
        top_n=LIVE_UNIVERSE_TOP_N,
        force=(DATA_MODE == "force"),
        cache_only=(DATA_MODE == "cache"),
    )
    extra = load_delisted_symbol_list()
    if extra:
        missing = [s for s in extra if s not in data]
        if missing:
            print(f"Extra delisted / manual symbols: {len(missing)}")
            listings = {}
            eq = _read_equity_l().set_index("SYMBOL")
            for s in missing:
                if s in eq.index:
                    listings[s] = eq.loc[s, "listing"]
            if DATA_MODE == "force":
                more = download_many(missing, start=START, force=True, listings=listings, sequential=True)
            elif DATA_MODE == "cache":
                more = _load_cached_ohlcv(missing, listings=listings)
            else:
                more = download_many(missing, start=START, force=False, listings=listings, sequential=True)
            data.update(more)
            if len(more):
                extra_meta = get_universe_meta(list(more))
                meta = pd.concat([meta, extra_meta.reindex(list(more))]).loc[
                    lambda m: ~m.index.duplicated(keep="first")
                ]
    print(
        f"Live universe: top {LIVE_UNIVERSE_TOP_N} by mcap (≥ ₹{MIN_MCAP_CR:,.0f} cr) "
        f"from {'+'.join(sorted(NSE_UNIVERSE_SERIES))} → "
        f"{len(data)} symbols | book {LONG_POSITIONS} slots | "
        f"sector_cap {format_sector_cap(live_sector_cap())}"
    )
    return data, meta


def run_download_nifty500(argv=None):
    """CLI: sync/force live-universe Yahoo history from START into data_cache/."""
    import sys
    argv = list(argv if argv is not None else sys.argv[2:])
    # download defaults to sync (fill gaps); --force / --cache override
    parse_data_mode(argv, default="sync")
    data, meta = load_universe()
    print(f"Done: {len(data)} symbols loaded, meta rows={len(meta)} (mode={DATA_MODE})")
    return data, meta


def run_fix_data(argv=None):
    """CLI: re-apply fix_ohlcv on cached live-universe files without re-download."""
    import sys
    argv = list(argv if argv is not None else sys.argv[2:])
    _ = argv  # reserved (--refresh-list accepted as no-op for compat)
    syms, _table = nse_symbols_by_mcap_rank(1, LIVE_UNIVERSE_TOP_N)
    # Restrict to the mcap floor used by the live book
    meta_all = get_universe_meta(syms)
    syms = [s for s in syms
            if s in meta_all.index and np.isfinite(meta_all.loc[s, "mcap_cr"])
            and meta_all.loc[s, "mcap_cr"] >= MIN_MCAP_CR]
    extra = load_delisted_symbol_list()
    symbols = sorted(set(syms) | set(extra))
    meta = get_universe_meta(symbols)
    listings = meta["listing"].to_dict() if "listing" in meta.columns else {}
    return fix_cached_symbols(symbols, listings=listings)


NSE_TOP_LIST_FILE = CACHE_DIR / "nse_top_by_mcap.csv"


def nse_symbols_by_mcap_rank(rank_from=1, rank_to=1000, refresh_mcap=False):
    """
    Rank all NSE EQ symbols by today's Yahoo market cap.
    Returns (symbols, table) for ranks rank_from..rank_to inclusive (1-based).
    """
    rank_from = max(1, int(rank_from))
    rank_to = max(rank_from, int(rank_to))
    symbols = get_nse_symbols()
    meta = get_universe_meta(symbols, refresh=refresh_mcap)
    ranked = meta.dropna(subset=["mcap_cr"]).sort_values("mcap_cr", ascending=False).copy()
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    CACHE_DIR.mkdir(exist_ok=True)
    ranked.to_csv(NSE_TOP_LIST_FILE)
    table = ranked.iloc[rank_from - 1:rank_to].copy()
    if len(table) == 0:
        raise SystemExit(
            f"No symbols in ranks {rank_from}-{rank_to} "
            f"(only {len(ranked)} names with known mcap).")
    print(f"NSE mcap ranks {rank_from}-{rank_to}: {len(table)} symbols "
          f"(of {len(ranked)} with mcap) -> {NSE_TOP_LIST_FILE}")
    print(f"  Rank {rank_from}: {table.index[0]} (₹{table.iloc[0]['mcap_cr']:,.0f} cr)  "
          f"Rank {int(table.iloc[-1]['rank'])}: {table.index[-1]} "
          f"(₹{table.iloc[-1]['mcap_cr']:,.0f} cr)")
    return list(table.index), table


def download_nse_mcap_ranks(rank_from=500, rank_to=1000, start=START, force=False,
                            refresh_mcap=False):
    """Download/recheck Yahoo OHLCV from `start` for NSE mcap ranks rank_from..rank_to."""
    syms, table = nse_symbols_by_mcap_rank(
        rank_from=rank_from, rank_to=rank_to, refresh_mcap=refresh_mcap)
    listings = table["listing"].to_dict() if "listing" in table.columns else {}
    # Listing dates for names missing from the ranked meta slice
    eq = _read_equity_l().set_index("SYMBOL")
    for s in syms:
        if s not in listings or pd.isna(listings.get(s)):
            if s in eq.index:
                listings[s] = eq.loc[s, "listing"]
    print(f"\nDownloading NSE ranks {rank_from}-{rank_to} from {start} "
          f"({len(syms)} symbols) -> {CACHE_DIR}/")
    data = download_many(syms, start=start, force=force, listings=listings, sequential=True)
    ok = [s for s in syms if s in data]
    print(f"Done ranks {rank_from}-{rank_to}: {len(ok)}/{len(syms)} usable "
          f"(≥{MIN_BARS} bars after repair)")
    return data, table


def run_download_nse_ranks(argv=None):
    """CLI: python main.py download-ranks [from] [to] [--force] [--refresh-mcap]"""
    import sys
    argv = list(argv if argv is not None else sys.argv[2:])
    force = "--force" in argv or "-f" in argv
    refresh_mcap = "--refresh-mcap" in argv
    nums = [a for a in argv if a.lstrip("-").isdigit()]
    rank_from = int(nums[0]) if len(nums) >= 1 else 500
    rank_to = int(nums[1]) if len(nums) >= 2 else 1000
    return download_nse_mcap_ranks(
        rank_from=rank_from, rank_to=rank_to, force=force, refresh_mcap=refresh_mcap)


def _load_cached_ohlcv(symbols, listings=None, min_bars=MIN_BARS):
    """Read existing data_cache CSVs only — never hits Yahoo."""
    listings = listings if listings is not None else {}
    data = {}
    missing = 0
    short = 0
    spiked = 0
    for s in symbols:
        f = _cache_file(_yf_equity(s))
        if not f.exists():
            missing += 1
            continue
        try:
            df = pd.read_csv(f, index_col=0, parse_dates=True)
        except Exception:
            missing += 1
            continue
        if len(df) < min_bars:
            short += 1
            continue
        mx = df.Close.pct_change().abs().max()
        if np.isfinite(mx) and mx > MAX_ABS_DAILY_MOVE:
            spiked += 1
            continue
        data[s] = df
    print(
        f"Cache-only load: {len(data)} OK, {missing} missing file, "
        f"{short} <{min_bars} bars, {spiked} spike-dropped (no download)"
    )
    return data


def _persist_live_universe_list(take):
    """
    Write the exact symbol list the live book will consider (rank, mcap, series).
    Called on every load_mcap_floor_universe / load_universe so picks always
    come from the current top-N EQ+BE+BZ ranking.
    """
    _ensure_output_dirs()
    eq = _read_equity_l().set_index("SYMBOL")
    rows = []
    series_counts = {"EQ": 0, "BE": 0, "BZ": 0, "other": 0}
    for sym, r in take.iterrows():
        series = ""
        if sym in eq.index:
            series = eq.loc[sym, "SERIES"]
            if isinstance(series, pd.Series):
                series = series.iloc[0]
            series = str(series).strip()
        bucket = series if series in series_counts else "other"
        series_counts[bucket] = series_counts.get(bucket, 0) + 1
        rows.append({
            "rank": int(r["rank"]),
            "symbol": sym,
            "mcap_cr": float(r["mcap_cr"]),
            "series": series,
        })
    df = pd.DataFrame(rows)
    path = OUTPUT_LIVE / f"universe_top{len(df)}_by_mcap.csv"
    df.to_csv(path, index=False)
    if len(df) == int(LIVE_UNIVERSE_TOP_N):
        df.to_csv(LIVE_UNIVERSE_LIST_FILE, index=False)
        path = LIVE_UNIVERSE_LIST_FILE
    be_names = [r["symbol"] for r in rows if r["series"] == "BE"]
    bz_names = [r["symbol"] for r in rows if r["series"] == "BZ"]
    print(
        f"Universe series in pick list: EQ={series_counts.get('EQ', 0)} "
        f"BE={series_counts.get('BE', 0)} BZ={series_counts.get('BZ', 0)}"
        + (f" other={series_counts.get('other', 0)}" if series_counts.get("other") else "")
        + f" → {path}"
    )
    if be_names:
        print(f"  BE in list: {', '.join(be_names)}")
    if bz_names:
        print(f"  BZ in list: {', '.join(bz_names)}")
    return df


def load_mcap_floor_universe(min_mcap_cr=MIN_MCAP_CR, top_n=None, force=False, cache_only=True):
    """
    Load prices for today's names with mcap >= min_mcap_cr (optionally top_n by mcap).
    Symbol pool is get_nse_symbols() = NSE EQUITY_L series EQ + BE + BZ.
    Returns (data, meta_ranked, ranked_all).

    Honours DATA_MODE; legacy force/cache_only still work when set explicitly.
    """
    if force:
        set_data_mode("force")
    elif cache_only is False:
        set_data_mode("sync")
    elif cache_only is True:
        set_data_mode("cache")
    symbols = get_nse_symbols()
    meta = get_universe_meta(symbols)
    ranked = meta.dropna(subset=["mcap_cr"]).sort_values("mcap_cr", ascending=False).copy()
    ranked = ranked[ranked["mcap_cr"] >= float(min_mcap_cr)]
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    CACHE_DIR.mkdir(exist_ok=True)
    ranked.to_csv(NSE_TOP_LIST_FILE)
    take = ranked if top_n is None else ranked.head(int(top_n))
    syms = list(take.index)
    print(
        f"Mcap ≥ ₹{min_mcap_cr:,.0f} cr: {len(ranked)} names; loading top {len(syms)} "
        f"(series={'+'.join(sorted(NSE_UNIVERSE_SERIES))}, mode={DATA_MODE})"
    )
    _persist_live_universe_list(take)
    listings = take["listing"].to_dict() if "listing" in take.columns else {}
    eq = _read_equity_l().set_index("SYMBOL")
    for s in syms:
        if s not in listings or pd.isna(listings.get(s)):
            if s in eq.index:
                listings[s] = eq.loc[s, "listing"]
    if DATA_MODE == "force":
        data = download_many(syms, start=START, force=True, listings=listings, sequential=True)
    elif DATA_MODE == "cache":
        data = _load_cached_ohlcv(syms, listings=listings)
    else:
        data = download_many(syms, start=START, force=False, listings=listings, sequential=True)
    meta_ok = take.reindex(list(data))
    print(f"Usable history: {len(data)}/{len(syms)}")
    return data, meta_ok, ranked


def _subset_long_panels(tickers, P, keep_set):
    """Keep columns in keep_set and recompute cross-sectional ranks inside that universe."""
    keep = [t for t in tickers if t in keep_set and t in P["Close"].columns]
    P2 = {}
    for k, v in P.items():
        if isinstance(v, pd.DataFrame):
            cols = [c for c in keep if c in v.columns]
            P2[k] = v[cols]
        else:
            P2[k] = v
    rank3 = P2["roc63"].rank(axis=1, pct=True)
    rank6 = P2["roc126"].rank(axis=1, pct=True)
    rank12 = P2["roc252"].rank(axis=1, pct=True)
    P2["score"] = (rank6 + rank12) / 2.0
    P2["score_3_6_12"] = (rank3 + rank6 + rank12) / 3.0
    rs3 = P2["sharpe63"].rank(axis=1, pct=True)
    rs6 = P2["sharpe126"].rank(axis=1, pct=True)
    rs12 = P2["sharpe252"].rank(axis=1, pct=True)
    P2["score_sharpe_3_6_12"] = (rs3 + rs6 + rs12) / 3.0
    return keep, P2


# Default start years for long-history mcap grid (Nifty 500 Yahoo from ~2007;
# earlier starts still run Long, vs-index metrics begin when ^CRSLDX exists).
MCAP_GRID_STARTS_FROM_2000 = (
    "2000-01-01",
    "2003-01-01",
    "2006-01-01",
    "2009-01-01",
    "2012-01-01",
    "2015-01-01",
    "2017-12-01",
    "2020-01-01",
)


def run_mcap_grid_rolling_study(
    start="2017-12-01",
    starts=None,
    min_mcap_cr=MIN_MCAP_CR,
    top_ns=None,
    sector_caps=(2, 3, 4),
    position_counts=(10, 15, 20),
    horizons=(1, 2, 3, 5),
    out_tag=None,
):
    """
    Sweep universe size (top N by mcap among ≥ min_mcap_cr), industry cap, and book size.
    Backtests live Long from each start; writes detailed rolling vs Nifty 500 under output/studies/.

    If `starts` is set (list of YYYY-MM-DD), runs the full grid for each start with one panel build.
    Comparison vs Nifty 500 uses the overlap only (index history begins ~2007-03).
    """
    _ensure_output_dirs()
    start_list = list(starts) if starts else [start]
    data, meta, ranked_all = load_mcap_floor_universe(min_mcap_cr=min_mcap_cr, top_n=None)
    max_n = len(data)
    if top_ns is None:
        # Include 300 for early-history thin coverage; 500..1200 step 100 clipped to usable
        cand = [300] + list(range(500, 1201, 100))
        top_ns = sorted({min(n, max_n) for n in cand if n <= max_n or n == 300})
        if max_n not in top_ns and max_n >= 300:
            top_ns.append(max_n)
            top_ns = sorted(set(top_ns))
    multi = len(start_list) > 1
    if out_tag is None:
        out_tag = "multistart_from_2000" if multi else "from_2017dec"
    print(
        f"Grid top_n={top_ns} sector_caps={sector_caps} positions={position_counts} "
        f"starts={start_list} horizons={horizons}"
    )

    bench, bench_ticker = _load_nifty500_close(None)
    print("Building full panel (largest mcap floor set)...")
    tickers_all, P_all = build_panels(data, universe=meta, bench_close=bench)
    sectors_map = load_sectors(tickers_all)
    ranked_syms = [s for s in ranked_all.index if s in data]
    # How many top-N names have any bars by each calendar year (coverage hint)
    close_all = P_all["Close"]
    for s0 in start_list:
        ts = pd.Timestamp(s0)
        if ts in close_all.index or len(close_all.loc[:ts]):
            slice_c = close_all.loc[:ts]
            if len(slice_c):
                n_live = int(slice_c.iloc[-1].notna().sum()) if len(slice_c) else 0
            else:
                n_live = 0
        else:
            n_live = 0
        # Prefer first available row on/after start
        after = close_all.loc[ts:]
        n_at = int(after.iloc[0].notna().sum()) if len(after) else 0
        print(f"  coverage @ {s0}: {n_at}/{len(close_all.columns)} panel names have a print")

    benches = load_benchmarks()
    n500 = benches.get("Nifty 500", bench)
    if n500 is None:
        raise SystemExit("Need Nifty 500 benchmark for rolling comparison.")
    n500_start = str(pd.Series(n500).dropna().index.min().date()) if n500 is not None else "?"
    print(f"Nifty 500 series starts {n500_start} — vs-index / rolling use overlap only.")

    rows = []
    roll_rows = []
    n_jobs = len(start_list) * len(top_ns) * len(sector_caps) * len(position_counts)
    done = 0
    for start_req in start_list:
        for top_n in top_ns:
            keep = set(ranked_syms[: int(top_n)])
            tickers, P = _subset_long_panels(tickers_all, P_all, keep)
            sectors = [sectors_map.get(t, "") for t in tickers]
            sleeve = cash_sleeve_returns(P["Close"].index)
            for sector_cap in sector_caps:
                for n_pos in position_counts:
                    done += 1
                    kw = dict(LONG_BOOK)
                    kw.update(
                        sector_cap=int(sector_cap),
                        max_positions=int(n_pos),
                        start=start_req,
                    )
                    if sleeve is not None:
                        kw["cash_ret"] = sleeve
                        kw["cash_ret_idle_only"] = False
                    eq, tr = backtest(tickers, P, sectors=sectors, **kw)
                    eq = eq.dropna()
                    if len(eq) < 252:
                        print(
                            f"  [{done}/{n_jobs}] start{start_req[:4]} top{top_n} "
                            f"sec{sector_cap} pos{n_pos}: too short, skip"
                        )
                        continue
                    # Align Nifty 500 on same window, rebase
                    n5 = n500.reindex(eq.index).ffill().dropna()
                    both = eq.index.intersection(n5.index)
                    if len(both) < 252:
                        print(
                            f"  [{done}/{n_jobs}] start{start_req[:4]} top{top_n} "
                            f"sec{sector_cap} pos{n_pos}: <1y overlap with N500, skip"
                        )
                        continue
                    eq_a = eq.loc[both]
                    n5_a = _equity_from_price(n5.loc[both])
                    eq_a = eq_a / eq_a.iloc[0] * INITIAL_CAPITAL
                    p_long = perf_stats(eq_a, "Long")
                    p_n5 = perf_stats(n5_a, "N500")
                    tstats = trade_stats(tr)
                    row = {
                        "start_requested": start_req,
                        "top_n": int(top_n),
                        "symbols_panel": len(tickers),
                        "sector_cap": int(sector_cap),
                        "max_positions": int(n_pos),
                        "start": str(eq_a.index[0].date()),
                        "end": str(eq_a.index[-1].date()),
                        "Long CAGR %": round(p_long["CAGR %"], 2),
                        "N500 CAGR %": round(p_n5["CAGR %"], 2),
                        "Excess CAGR pts": round(p_long["CAGR %"] - p_n5["CAGR %"], 2),
                        "Long Max DD %": round(p_long["Max DD %"], 2),
                        "N500 Max DD %": round(p_n5["Max DD %"], 2),
                        "Long Calmar": round(p_long["Calmar"], 3) if np.isfinite(p_long["Calmar"]) else np.nan,
                        "N500 Calmar": round(p_n5["Calmar"], 3) if np.isfinite(p_n5["Calmar"]) else np.nan,
                        "Long Sharpe": round(p_long["Sharpe"], 3) if np.isfinite(p_long["Sharpe"]) else np.nan,
                        "Long Total %": round((eq_a.iloc[-1] / eq_a.iloc[0] - 1) * 100, 2),
                        "N500 Total %": round((n5_a.iloc[-1] / n5_a.iloc[0] - 1) * 100, 2),
                        "Long Final": round(p_long["Final value"], 0),
                        "Trades": tstats.get("Trades", 0),
                        "Win %": round(tstats.get("Win rate %", np.nan), 1) if tstats else np.nan,
                        "PF": round(tstats.get("Profit factor", np.nan), 2) if tstats else np.nan,
                        "Avg hold days": round(tstats.get("Avg days held", np.nan), 0) if tstats else np.nan,
                    }
                    # Rolling excess vs Nifty 500
                    for y in horizons:
                        rl = rolling_window_returns(eq_a, y)
                        rn = rolling_window_returns(n5_a, y)
                        if rl.empty or rn.empty:
                            continue
                        ix = rl.index.intersection(rn.index)
                        # need enough finite overlapping windows
                        rl_i = rl.loc[ix].dropna()
                        rn_i = rn.loc[ix].dropna()
                        ix2 = rl_i.index.intersection(rn_i.index)
                        if len(ix2) < 20:
                            continue
                        ex = rl_i.loc[ix2] - rn_i.loc[ix2]
                        beat = float((rl_i.loc[ix2] > rn_i.loc[ix2]).mean() * 100)
                        row[f"roll{y}y Long med CAGR"] = round(
                            _annualized_from_total_pct(float(rl_i.loc[ix2].median()), y), 2)
                        row[f"roll{y}y N500 med CAGR"] = round(
                            _annualized_from_total_pct(float(rn_i.loc[ix2].median()), y), 2)
                        row[f"roll{y}y beat %"] = round(beat, 1)
                        row[f"roll{y}y excess med %"] = round(float(ex.median()), 2)
                        row[f"roll{y}y excess p10 %"] = round(float(np.nanpercentile(ex, 10)), 2)
                        row[f"roll{y}y Long win%"] = round(float((rl_i.loc[ix2] > 0).mean() * 100), 1)
                        roll_rows.append({
                            "start_requested": start_req,
                            "top_n": int(top_n),
                            "sector_cap": int(sector_cap),
                            "max_positions": int(n_pos),
                            "Horizon (y)": y,
                            "Windows": len(ix2),
                            "Long beat N500 %": round(beat, 1),
                            "Excess total % median": round(float(ex.median()), 2),
                            "Excess total % mean": round(float(ex.mean()), 2),
                            "Excess total % p10": round(float(np.nanpercentile(ex, 10)), 2),
                            "Excess total % p90": round(float(np.nanpercentile(ex, 90)), 2),
                            "Long CAGR % median": round(
                                _annualized_from_total_pct(float(rl_i.loc[ix2].median()), y), 2),
                            "N500 CAGR % median": round(
                                _annualized_from_total_pct(float(rn_i.loc[ix2].median()), y), 2),
                            "Long total % median": round(float(rl_i.loc[ix2].median()), 2),
                            "N500 total % median": round(float(rn_i.loc[ix2].median()), 2),
                            "Long win %": round(float((rl_i.loc[ix2] > 0).mean() * 100), 1),
                        })
                    rows.append(row)
                    print(
                        f"  [{done}/{n_jobs}] {start_req[:4]} top{top_n} sec{sector_cap} pos{n_pos}: "
                        f"CAGR {row['Long CAGR %']:.1f}% vs N500 {row['N500 CAGR %']:.1f}% "
                        f"(+{row['Excess CAGR pts']:.1f}) DD {row['Long Max DD %']:.1f}% "
                        f"Calmar {row['Long Calmar']:.2f} "
                        f"roll1y beat {row.get('roll1y beat %', float('nan'))}% "
                        f"[{row['start']}→{row['end']}]"
                    )

    grid = pd.DataFrame(rows)
    roll = pd.DataFrame(roll_rows)
    out_grid = OUTPUT_STUDIES / f"mcap_grid_{out_tag}.csv"
    out_roll = OUTPUT_STUDIES / f"mcap_grid_rolling_{out_tag}.csv"
    grid.to_csv(out_grid, index=False)
    roll.to_csv(out_roll, index=False)

    # Leaderboards
    if len(grid):
        by_calmar = grid.sort_values("Long Calmar", ascending=False).head(20)
        by_excess = grid.sort_values("Excess CAGR pts", ascending=False).head(20)
        by_roll = (
            grid.sort_values("roll1y beat %", ascending=False).head(20)
            if "roll1y beat %" in grid else grid.head(0)
        )
        print("\n=== Top by Calmar ===")
        cols = [c for c in [
            "start_requested", "top_n", "sector_cap", "max_positions",
            "Long CAGR %", "N500 CAGR %", "Excess CAGR pts", "Long Max DD %",
            "Long Calmar", "roll1y beat %", "roll2y beat %", "roll3y beat %",
            "roll5y beat %", "Trades",
        ] if c in grid.columns]
        print(by_calmar[cols].to_string(index=False))
        print("\n=== Top by Excess CAGR ===")
        print(by_excess[cols].to_string(index=False))
        if len(by_roll):
            print("\n=== Top by 1y rolling beat rate ===")
            print(by_roll[cols].to_string(index=False))
        # Slice views
        slice_keys = [
            ("start year", "start_requested"),
            ("universe size", "top_n"),
            ("sector_cap", "sector_cap"),
            ("max_positions", "max_positions"),
        ]
        for label, key in slice_keys:
            if key not in grid.columns:
                continue
            agg = {
                "Long CAGR %": "median",
                "Excess CAGR pts": "median",
                "Long Calmar": "median",
                "Long Max DD %": "median",
                **({k: "median" for k in grid.columns if k.startswith("roll") and k.endswith("beat %")}),
            }
            g = grid.groupby(key).agg(agg).round(2)
            print(f"\n=== Median by {label} ===")
            print(g.to_string())
            safe = key.replace(" ", "_")
            g.to_csv(OUTPUT_STUDIES / f"mcap_grid_by_{safe}_{out_tag}.csv")

        # Best config per start (by Calmar)
        if "start_requested" in grid.columns:
            print("\n=== Best Calmar per start ===")
            idx = grid.groupby("start_requested")["Long Calmar"].idxmax()
            print(grid.loc[idx, cols].to_string(index=False))

    print(f"\nSaved {out_grid}")
    print(f"Saved {out_roll}")
    print(
        f"Universe available: {max_n} usable names with mcap ≥ ₹{min_mcap_cr:,.0f} cr "
        f"(requested up to 1200). Bench: {bench_ticker or 'Nifty 500'} (from {n500_start})"
    )
    return grid, roll


def run_mcap_grid_cli(argv=None):
    """
    CLI:
      python main.py study-mcap-grid [--start YYYY-MM-DD]
      python main.py study-mcap-grid --from-2000
      python main.py study-mcap-grid --starts 2000-01-01,2009-01-01,2017-12-01
    """
    import sys
    argv = list(argv if argv is not None else sys.argv[2:])
    starts = None
    start = "2017-12-01"
    out_tag = None
    if "--from-2000" in argv:
        starts = list(MCAP_GRID_STARTS_FROM_2000)
        out_tag = "multistart_from_2000"
    if "--starts" in argv:
        i = argv.index("--starts")
        if i + 1 < len(argv):
            starts = [s.strip() for s in argv[i + 1].split(",") if s.strip()]
            out_tag = out_tag or "multistart_custom"
    if "--start" in argv:
        i = argv.index("--start")
        if i + 1 < len(argv):
            start = argv[i + 1]
    return run_mcap_grid_rolling_study(start=start, starts=starts, out_tag=out_tag)


# ----------------------------- US UNIVERSE (Long backtest) --------------------
# India live book: tradable ≥ ₹2,000 cr (~$240M USD at ~₹83/$). Liquidity ~₹2 cr/day (~$240k).
US_CACHE_DIR = Path("data_cache_us")
US_META_FILE = US_CACHE_DIR / "universe_meta_us.csv"
# FX reference for mapping India rules → USD (update occasionally).
INR_PER_USD = 83.0
US_MIN_MCAP_INR_CR = MIN_MCAP_CR
US_MIN_MCAP_USD = US_MIN_MCAP_INR_CR * 1e7 / INR_PER_USD          # ~₹2,000 cr
US_PIT_FLOOR_USD = PIT_UNIVERSE_FLOOR_CR * 1e7 / INR_PER_USD       # ~₹500 cr
US_MIN_TURNOVER = MIN_TURNOVER / INR_PER_USD                       # ~₹2 cr/day in USD
US_GOLD_TICKER = "GLD"
US_BENCH_TICKERS = ("^GSPC", "SPY")  # S&P 500
US_TOP_LIST_FILE = US_CACHE_DIR / "us_top_by_mcap.csv"
US_PRICE_START = "2010-01-01"


def _yahoo_symbol_us(sym):
    return sym.replace(".", "-").strip().upper()


def _us_cache_file(symbol):
    US_CACHE_DIR.mkdir(exist_ok=True)
    safe = symbol.replace("^", "IDX_").replace("&", "_")
    return US_CACHE_DIR / f"{safe}.csv"


def _fetch_yahoo_meta_us(symbol):
    import time
    import yfinance as yf
    time.sleep(0.25)
    t = yf.Ticker(_yahoo_symbol_us(symbol))
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
    return mcap, shares


def _wiki_index_symbols(url, symbol_col_hint=("Symbol", "Ticker symbol")):
    """S&P index constituents from Wikipedia."""
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; screener/1.0)"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        html = resp.read()
    tables = pd.read_html(html)
    for tbl in tables:
        cols = [c for c in tbl.columns if str(c).strip() in symbol_col_hint or "Symbol" in str(c)]
        if not cols:
            continue
        col = cols[0]
        syms = tbl[col].astype(str).str.strip()
        syms = syms[syms.str.len() > 0]
        out = []
        for s in syms:
            s = s.split("\n")[0].strip()
            if s and s.lower() != "nan":
                out.append(_yahoo_symbol_us(s))
        if out:
            return sorted(set(out))
    return []


def us_nasdaq_nyse_symbols():
    """NASDAQ + NYSE/AMEX common stocks (no test issues; skip ETFs when flagged)."""
    import urllib.request
    urls = [
        "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
        "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
    ]
    out = set()
    for url in urls:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; screener/1.0)"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            text = resp.read().decode("utf-8", errors="ignore")
        lines = text.strip().splitlines()
        header = lines[0].split("|")
        if "Symbol" in header:
            sym_i = header.index("Symbol")
        elif "ACT Symbol" in header:
            sym_i = header.index("ACT Symbol")
        else:
            sym_i = 0
        for line in lines[1:]:
            if line.startswith("File Creation"):
                continue
            parts = line.split("|")
            if len(parts) <= sym_i:
                continue
            sym = _yahoo_symbol_us(parts[sym_i])
            if not sym or sym == "SYMBOL":
                continue
            if "Test Issue" in header:
                ti = header.index("Test Issue")
                if ti < len(parts) and parts[ti].upper() == "Y":
                    continue
            if "ETF" in header:
                ei = header.index("ETF")
                if ei < len(parts) and parts[ei].upper() == "Y":
                    continue
            if len(sym) > 6 or sym.endswith("W") or sym.endswith("R"):
                continue
            out.add(sym)
    return sorted(out)


def us_equity_symbol_list():
    """S&P 500 + S&P 400 MidCap + S&P 600 SmallCap (survivorship-biased, current constituents)."""
    syms = set()
    try:
        sp500 = pd.read_csv(
            "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
        )
        syms.update(_yahoo_symbol_us(s) for s in sp500["Symbol"].astype(str))
        print(f"  S&P 500 list: {len(sp500)} symbols (GitHub datasets)")
    except Exception as e:
        print(f"  S&P 500 GitHub list failed: {e}")
    urls = [
        "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
        "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies",
    ]
    for url in urls:
        try:
            part = _wiki_index_symbols(url)
            syms.update(part)
            print(f"  {url.split('/')[-1]}: {len(part)} symbols")
        except Exception as e:
            print(f"  US list skipped ({url.split('/')[-1]}): {e}")
    return sorted(syms)


def get_us_universe_meta(symbols, refresh=False):
    from concurrent.futures import ThreadPoolExecutor
    US_CACHE_DIR.mkdir(exist_ok=True)
    if US_META_FILE.exists():
        cached = pd.read_csv(US_META_FILE, index_col=0)
    else:
        cached = pd.DataFrame(columns=["mcap_usd", "shares"])
    for col in ("mcap_usd", "shares"):
        if col not in cached.columns:
            cached[col] = np.nan
    if refresh:
        todo = [s for s in symbols]
    else:
        todo = [s for s in symbols if s not in cached.index or not np.isfinite(cached.loc[s, "mcap_usd"])]
    if todo:
        print(f"Fetching US market cap for {len(todo)} symbols (cached in {US_META_FILE})...")
        with ThreadPoolExecutor(max_workers=4) as ex:
            rows = list(ex.map(_fetch_yahoo_meta_us, todo))
        new = pd.DataFrame(rows, index=todo, columns=["mcap_usd", "shares"])
        cached = cached.drop(index=[s for s in todo if s in cached.index], errors="ignore")
        cached = pd.concat([cached, new])
        cached.to_csv(US_META_FILE)
    return cached.reindex(symbols)


def _us_cache_is_usable(path, start=US_PRICE_START, min_bars=100):
    """True if cache has enough history (from `start` or IPO after that)."""
    if not path.exists():
        return False
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        if len(df) < min_bars:
            return False
        # Prefer files that reach back near start, but accept later IPOs with enough bars.
        first = pd.Timestamp(df.index.min())
        if first <= pd.Timestamp(start) + pd.Timedelta(days=30):
            return True
        return len(df) >= min_bars
    except Exception:
        return False


def _us_cache_covers_start(path, start=US_PRICE_START):
    return _us_cache_is_usable(path, start=start)


def _download_one_us_stooq(symbol, start=US_PRICE_START):
    """Fallback daily OHLCV from Stooq (symbol.US)."""
    import urllib.request
    ysym = _yahoo_symbol_us(symbol).lower()
    if ysym.startswith("^"):
        return None
    url = f"https://stooq.com/q/d/l/?s={ysym}.us&i=d"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            text = resp.read().decode("utf-8", errors="ignore")
        from io import StringIO
        raw = pd.read_csv(StringIO(text))
        if raw.empty or "Date" not in raw.columns:
            return None
        raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce")
        raw = raw.dropna(subset=["Date"]).set_index("Date").sort_index()
        cols = {c: c.title() if c.lower() != "volume" else "Volume" for c in raw.columns}
        raw = raw.rename(columns=cols)
        need = ["Open", "High", "Low", "Close", "Volume"]
        if not all(c in raw.columns for c in need):
            return None
        df = raw[need].dropna(how="all")
        df = df.loc[df.index >= pd.Timestamp(start)]
        return df if len(df) else None
    except Exception:
        return None


def download_one_us(symbol, start=US_PRICE_START, end=END, force=False, retries=3):
    import time
    import yfinance as yf
    ysym = _yahoo_symbol_us(symbol)
    f = _us_cache_file(ysym)
    if not force and _us_cache_is_usable(f, start):
        return pd.read_csv(f, index_col=0, parse_dates=True)
    last_err = None
    for attempt in range(retries):
        try:
            t = yf.Ticker(ysym)
            df = t.history(start=start, end=end, auto_adjust=True)
            if (df is None or df.empty) and attempt == retries - 1:
                df = yf.download(ysym, start=start, end=end, auto_adjust=True, progress=False)
            if df is not None and not df.empty:
                df = _clean(df)
                if len(df):
                    df.to_csv(f)
                    return df
        except Exception as e:
            last_err = e
            msg = str(e).lower()
            if "rate" in msg or "too many" in msg or "401" in msg:
                wait = 20 * (attempt + 1)
                print(f"    rate-limit on {ysym}, sleep {wait}s...", flush=True)
                time.sleep(wait)
                continue
        time.sleep(2.0 * (attempt + 1))
    # Stooq fallback when Yahoo is blocked / empty
    st = _download_one_us_stooq(symbol, start=start)
    if st is not None and len(st):
        st.to_csv(f)
        return st
    if last_err:
        print(f"    fail {ysym}: {last_err}", flush=True)
    return None


def download_many_us(symbols, batch=20, start=US_PRICE_START, force=False, sequential=True):
    """Download US OHLCV. Default sequential=True to avoid Yahoo rate limits."""
    import time
    import yfinance as yf
    US_CACHE_DIR.mkdir(exist_ok=True)
    yf_syms = {s: _yahoo_symbol_us(s) for s in symbols}
    need = []
    for s in symbols:
        f = _us_cache_file(yf_syms[s])
        if force or not _us_cache_is_usable(f, start):
            need.append(s)
    print(f"{len(symbols) - len(need)} already cached, downloading {len(need)}...")
    failed = []
    if sequential and need:
        print("  (sequential — slower, fewer Yahoo rate limits)")
        for i, s in enumerate(need, 1):
            if download_one_us(s, start=start, force=force) is None:
                failed.append(s)
            if i % 10 == 0 or i == len(need):
                n_ok = len(list(US_CACHE_DIR.glob("*.csv")))
                print(f"  {i}/{len(need)} this run | ~{n_ok} csv files in cache", flush=True)
            time.sleep(2.0)
    elif need:
        for k in range(0, len(need), batch):
            chunk = need[k:k + batch]
            try:
                raw = yf.download([yf_syms[s] for s in chunk], start=start, end=END, auto_adjust=True,
                                  progress=False, group_by="ticker", threads=True)
            except Exception as e:
                print(f"  batch failed: {e}")
                failed.extend(chunk)
                time.sleep(5)
                continue
            for s in chunk:
                y = yf_syms[s]
                try:
                    part = raw.copy() if len(chunk) == 1 else raw[y].copy()
                    df = _clean(part)
                    if len(df):
                        df.to_csv(_us_cache_file(y))
                    else:
                        failed.append(s)
                except Exception:
                    failed.append(s)
            print(f"  {min(k + batch, len(need))}/{len(need)}")
            time.sleep(2)
        if failed:
            print(f"  Retrying {len(failed)} symbols one-by-one...")
            for s in failed:
                if download_one_us(s, start=start, force=True) is None:
                    print(f"    skip {s}")
                time.sleep(1.2)
            failed = []
    if failed:
        print(f"  Failed: {len(failed)} symbols (rate limit or no data)")
    data = {}
    for s in symbols:
        f = _us_cache_file(yf_syms[s])
        if not f.exists():
            continue
        df = pd.read_csv(f, index_col=0, parse_dates=True)
        if len(df) < 300:
            continue
        if df.Close.pct_change().abs().max() > MAX_ABS_DAILY_MOVE:
            print(f"  dropped {s}: >{MAX_ABS_DAILY_MOVE:.0%} one-day move (likely bad data)")
            continue
        data[s] = df
    print(f"Loaded {len(data)} US symbols with usable history")
    return data


def us_top_symbols_by_mcap(top_n=1000, refresh_mcap=False):
    """
    Rank US common stocks by today's Yahoo market cap; return top_n symbols + meta table.
    Starts with S&P index lists, then scans NASDAQ/NYSE in chunks until top_n names have mcap.
    """
    US_CACHE_DIR.mkdir(exist_ok=True)
    candidates = set(us_equity_symbol_list())
    print(f"{len(candidates)} US ticker candidates (S&P 500/400/600 lists)")
    meta = get_us_universe_meta(sorted(candidates), refresh=refresh_mcap)
    ranked = meta.dropna(subset=["mcap_usd"]).sort_values("mcap_usd", ascending=False)

    if len(ranked) < top_n:
        have = set(ranked.index)
        nasdaq = [s for s in us_nasdaq_nyse_symbols() if s not in have]
        print(f"Need more names for top {top_n}: scanning NASDAQ/NYSE in batches of 400...")
        for i in range(0, len(nasdaq), 400):
            chunk = nasdaq[i:i + 400]
            extra = get_us_universe_meta(chunk, refresh=True)
            meta = pd.concat([meta, extra])
            meta = meta[~meta.index.duplicated(keep="last")]
            ranked = meta.dropna(subset=["mcap_usd"]).sort_values("mcap_usd", ascending=False)
            print(f"  mcap known: {len(ranked)} (batch {i // 400 + 1})")
            if len(ranked) >= top_n + 50:
                break

    ranked = ranked[~ranked.index.str.startswith("^") & (ranked.index != US_GOLD_TICKER)]
    ranked = ranked[(ranked["mcap_usd"] >= 50e6) & (ranked["mcap_usd"] <= 4e12)]
    top = ranked.head(int(top_n)).copy()
    top["rank"] = np.arange(1, len(top) + 1)
    top.to_csv(US_TOP_LIST_FILE)
    print(f"Top {len(top)} by market cap saved to {US_TOP_LIST_FILE}")
    if len(top):
        print(f"  Largest: {top.index[0]} (${top.iloc[0]['mcap_usd']/1e9:.1f}B)  "
              f"Smallest in list: {top.index[-1]} (${top.iloc[-1]['mcap_usd']/1e9:.2f}B)")
    return list(top.index), top


def download_us_top_mcap(top_n=1000, start=US_PRICE_START, force=False, refresh_mcap=False):
    """Download adjusted daily OHLCV from `start` for the top `top_n` US stocks by market cap."""
    syms, table = us_top_symbols_by_mcap(top_n=top_n, refresh_mcap=refresh_mcap)
    extras = [US_GOLD_TICKER, "SPY", "^GSPC"]
    for e in extras:
        if e not in syms:
            syms.append(e)
    print(f"\nDownloading US prices from {start} for {len(syms)} symbols -> {US_CACHE_DIR}/")
    data = download_many_us(syms, start=start, force=force, sequential=force)
    ok = []
    for s in syms:
        f = _us_cache_file(_yahoo_symbol_us(s))
        if not f.exists():
            continue
        try:
            df = pd.read_csv(f, index_col=0, parse_dates=True)
            if len(df) >= 100:
                ok.append(s)
        except Exception:
            pass
    summary = {
        "requested_top": top_n,
        "downloaded_ok": len(ok),
        "start": start,
        "cache_dir": str(US_CACHE_DIR),
    }
    print(f"\nDone: {len(ok)}/{len(syms)} files with history from ~{start}.")
    return data, table, summary


def download_us_sp1500_history(start=US_PRICE_START, force=False):
    """Download full Yahoo history for all S&P 500/400/600 symbols (~1500 names)."""
    symbols = sorted(set(us_equity_symbol_list()))
    extras = [US_GOLD_TICKER, "SPY", "^GSPC"]
    for e in extras:
        if e not in symbols:
            symbols.append(e)
    print(f"Downloading {len(symbols)} S&P index names from {start} (force={force})...")
    return download_many_us(symbols, start=start, force=force, sequential=True)


def download_us_nasdaq_extra(extra_n=2000, start=US_PRICE_START, force=False):
    """
    Import more US common stocks beyond S&P 1500 from NASDAQ/NYSE directories.
    Skips symbols already cached. Does not require market-cap ranking first.
    """
    have = set()
    for p in US_CACHE_DIR.glob("*.csv"):
        name = p.stem
        if name.startswith("IDX_"):
            continue
        if name in ("universe_meta_us", "us_top_by_mcap"):
            continue
        have.add(name)
    sp = set(us_equity_symbol_list())
    nasdaq = [s for s in us_nasdaq_nyse_symbols() if s not in have and s not in sp]
    # Prefer shorter tickers (common stocks); skip obvious warrants/units
    nasdaq = [s for s in nasdaq if 1 <= len(s) <= 5 and "-" not in s]
    take = nasdaq[: int(extra_n)]
    print(f"Importing {len(take)} extra NASDAQ/NYSE names (have {len(have)} cached, "
          f"{len(nasdaq)} candidates left)...")
    return download_many_us(take, start=start, force=force, sequential=True)


def import_us_prices(sp1500=True, extra_n=2000, top_n=1000, force=False, start=US_PRICE_START):
    """
    Broad US history import:
      1) all S&P 500/400/600 (~1500)
      2) next `extra_n` NASDAQ/NYSE common stocks not already cached
      3) refresh top-`top_n` mcap list from what we have
    """
    _ensure_output_dirs()
    if sp1500:
        download_us_sp1500_history(start=start, force=force)
    if extra_n > 0:
        download_us_nasdaq_extra(extra_n=extra_n, start=start, force=False)
    # Rank from cached meta + S&P list; skip full NASDAQ mcap scan if slow
    try:
        syms, table = us_top_symbols_by_mcap(top_n=top_n, refresh_mcap=False)
        print(f"Top mcap list: {len(syms)} names -> {US_TOP_LIST_FILE}")
    except Exception as e:
        print(f"Top mcap list skipped: {e}")
        table = None
    n_files = len(list(US_CACHE_DIR.glob("*.csv")))
    print(f"\nUS cache now has ~{n_files} CSV files under {US_CACHE_DIR}/")
    return table


def run_import_us(argv=None):
    import sys
    argv = list(argv if argv is not None else sys.argv[2:])
    extra_n = 2000
    top_n = 1000
    force = False
    no_sp = False
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in ("--extra", "-e") and i + 1 < len(argv):
            extra_n = int(argv[i + 1])
            i += 2
            continue
        if a in ("--top", "-n") and i + 1 < len(argv):
            top_n = int(argv[i + 1])
            i += 2
            continue
        if a in ("--force", "-f"):
            force = True
            i += 1
            continue
        if a in ("--no-sp",):
            no_sp = True
            i += 1
            continue
        i += 1
    return import_us_prices(sp1500=not no_sp, extra_n=extra_n, top_n=top_n, force=force)


def run_download_us(argv=None):
    import sys
    argv = list(argv if argv is not None else sys.argv[2:])
    top_n = 1000
    force = False
    refresh_mcap = False
    sp1500_all = False
    import_all = False
    extra_n = 2000
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in ("--top", "-n") and i + 1 < len(argv):
            top_n = int(argv[i + 1])
            i += 2
            continue
        if a in ("--extra", "-e") and i + 1 < len(argv):
            extra_n = int(argv[i + 1])
            i += 2
            continue
        if a in ("--force", "-f", "regenerate"):
            force = True
            i += 1
            continue
        if a in ("--refresh-mcap",):
            refresh_mcap = True
            i += 1
            continue
        if a in ("--sp1500", "--all-sp"):
            sp1500_all = True
            i += 1
            continue
        if a in ("--import", "--broad"):
            import_all = True
            i += 1
            continue
        i += 1
    if import_all:
        return import_us_prices(sp1500=True, extra_n=extra_n, top_n=top_n, force=force)
    if sp1500_all:
        download_us_sp1500_history(force=force)
        return download_us_top_mcap(top_n=top_n, force=force, refresh_mcap=refresh_mcap)
    return download_us_top_mcap(top_n=top_n, force=force, refresh_mcap=refresh_mcap)


def load_us_universe(download_floor_usd=None, top_n=1000):
    """Top US names by mcap with cached prices; PIT mcap/liquidity in add_indicators."""
    if US_TOP_LIST_FILE.exists():
        top_df = pd.read_csv(US_TOP_LIST_FILE, index_col=0)
        symbols = list(top_df.index[:top_n])
        print(f"Using {len(symbols)} symbols from {US_TOP_LIST_FILE.name}")
    else:
        symbols, _ = us_top_symbols_by_mcap(top_n=top_n)
    meta = get_us_universe_meta(symbols)
    floor = US_PIT_FLOOR_USD if download_floor_usd is None else download_floor_usd
    by_mcap = meta["mcap_usd"].dropna()
    download_syms = sorted(by_mcap[by_mcap >= floor].index)
    if len(download_syms) < len(symbols) * 0.5:
        download_syms = symbols
    print(f"{len(download_syms)} symbols to load from cache (mcap floor ${floor/1e6:,.0f}M)")
    data = download_many_us(download_syms, start=US_PRICE_START, force=False)
    meta = meta.reindex(data.keys())
    print("US universe: top-by-mcap list; survivorship bias (no delist history yet).")
    return data, meta


def load_us_universe_sp1500(download_floor_usd=None):
    """US S&P 1500-ish list; PIT mcap/liquidity applied in add_indicators."""
    symbols = us_equity_symbol_list()
    print(f"{len(symbols)} US symbols (S&P 500/400/600 lists from Wikipedia)")
    meta = get_us_universe_meta(symbols)
    floor = US_PIT_FLOOR_USD if download_floor_usd is None else download_floor_usd
    by_mcap = meta["mcap_usd"].dropna()
    download_syms = sorted(by_mcap[by_mcap >= floor].index)
    print(f"{len(download_syms)} symbols to download (mcap >= ${floor/1e6:,.0f}M USD, INR mapping)")
    data = download_many_us(download_syms)
    meta = meta.reindex(data.keys())
    print("US universe: current S&P constituents only (survivorship bias). No US delist file yet.")
    return data, meta


def _load_us_sp500_close(dates=None):
    for ticker in US_BENCH_TICKERS:
        df = download_one_us(ticker)
        if df is not None and len(df) >= 252:
            s = df.Close
            if dates is not None:
                s = s.reindex(dates).ffill()
            return s, ticker
    return None, ""


def cash_sleeve_returns_us(dates, debt_yield=0.045):
    """US idle cash: GLD when above 200dma (lag 1d), else ~4.5% p.a. (T-bill proxy)."""
    gold = download_one_us(US_GOLD_TICKER)
    if gold is None or len(gold) < 252:
        return None
    g = gold.Close.reindex(dates).ffill()
    gold_ret = g.pct_change().fillna(0).to_numpy()
    gold_on = (g > g.rolling(200).mean()).shift(1).fillna(False).to_numpy()
    debt_ret = np.full(len(dates), debt_yield / 252)
    return np.where(gold_on, gold_ret, debt_ret).astype(float)


def us_long_mcap_study(mcap_thresholds_usd=None):
    """
    Backtest live Long on US universe across mcap floors comparable to India ₹2,000 cr.
    Saves output/studies/us_long_mcap_study.csv
    """
    _ensure_output_dirs()
    if mcap_thresholds_usd is None:
        mcap_thresholds_usd = [
            (US_MIN_MCAP_USD, f"~INR {US_MIN_MCAP_INR_CR:,} cr (${US_MIN_MCAP_USD/1e6:.0f}M)"),
            (500e6, "$500M"),
            (1e9, "$1B"),
            (2e9, "$2B"),
            (5e9, "$5B"),
        ]
    data, meta = load_us_universe()
    if not data:
        raise SystemExit("No US price data loaded. Check network or data_cache_us/.")
    bench, bname = _load_us_sp500_close(None)
    ind_kw_base = dict(tradable_mcap_min=US_MIN_MCAP_USD, turnover_min=US_MIN_TURNOVER, mcap_in_usd=True)
    tickers, P = build_panels(data, universe=meta, bench_close=bench, indicator_kw=ind_kw_base)
    sectors = [""] * len(tickers)
    print(f"\nBuilt panels: {len(tickers)} names | benchmark {bname}")

    sp_eq = None
    if bench is not None and len(tickers):
        sp = bench.reindex(P["Close"].index).ffill()
        base = sp.dropna()
        if len(base):
            sp_eq = sp / base.iloc[0] * INITIAL_CAPITAL

    rows = []
    windows = [
        ("Full", {}),
        ("2010-2018", dict(start="2010-01-01", end="2018-12-31")),
        ("2019+", dict(start="2019-01-01")),
    ]
    for floor_usd, label in mcap_thresholds_usd:
        ind_kw = dict(tradable_mcap_min=floor_usd, turnover_min=US_MIN_TURNOVER, mcap_in_usd=True)
        t2, P2 = build_panels(data, universe=meta, bench_close=bench, indicator_kw=ind_kw)
        kw = dict(LONG_BOOK)
        sleeve = cash_sleeve_returns_us(P2["Close"].index)
        if sleeve is not None:
            kw["cash_ret"] = sleeve
            kw["cash_ret_idle_only"] = False
        row = {"mcap_floor": label, "mcap_usd": floor_usd, "symbols": len(t2)}
        for wlabel, wkw in windows:
            eq, tr = backtest(t2, P2, sectors=[""] * len(t2), **kw, **wkw)
            st = perf_stats(eq.dropna(), "")
            row[f"{wlabel} CAGR"] = round(st["CAGR %"], 2)
            row[f"{wlabel} DD"] = round(st["Max DD %"], 2)
            row[f"{wlabel} Calmar"] = round(st["Calmar"], 3)
            if wlabel == "Full":
                row["Final"] = round(st["Final value"])
                row["Trades"] = len(tr)
        rows.append(row)

    if sp_eq is not None:
        brow = {"mcap_floor": "S&P 500 B&H", "mcap_usd": np.nan, "symbols": 1}
        for wlabel, wkw in windows:
            sub = sp_eq.loc[sp_eq.index >= pd.Timestamp(wkw.get("start", sp_eq.index[0]))]
            if "end" in wkw:
                sub = sub.loc[: pd.Timestamp(wkw["end"])]
            sub = sub.dropna()
            if len(sub) < 2:
                continue
            st = perf_stats(sub, "")
            brow[f"{wlabel} CAGR"] = round(st["CAGR %"], 2)
            brow[f"{wlabel} DD"] = round(st["Max DD %"], 2)
            brow[f"{wlabel} Calmar"] = round(st["Calmar"], 3)
        rows.append(brow)

    table = pd.DataFrame(rows)
    path = OUTPUT_STUDIES / "us_long_mcap_study.csv"
    table.to_csv(path, index=False)
    print("\n=== US Long book — market cap floor sweep (same rules as India live Long) ===")
    print(f"Comparable India tradable size: ₹{US_MIN_MCAP_INR_CR:,} cr ≈ ${US_MIN_MCAP_USD/1e6:.0f}M USD @ ₹{INR_PER_USD}/$")
    print(f"Liquidity floor ≈ ${US_MIN_TURNOVER:,.0f}/day median (≈ ₹2 cr/day)\n")
    print(table.to_string(index=False))
    print(f"\nSaved {path}")
    best = table[table["mcap_floor"].str.contains("INR", na=False)]
    if not best.empty:
        print("\nIndia-comparable floor row is the INR-labelled mcap line (not necessarily best Calmar).")
    return table


def run_us_long_study():
    pd.set_option("display.width", 240)
    return us_long_mcap_study()


def run_us_backtest():
    """Single US Long backtest at India-comparable mcap (~$240M USD)."""
    _ensure_output_dirs()
    data, meta = load_us_universe()
    bench, bname = _load_us_sp500_close(None)
    ind_kw = dict(tradable_mcap_min=US_MIN_MCAP_USD, turnover_min=US_MIN_TURNOVER, mcap_in_usd=True)
    tickers, P = build_panels(data, universe=meta, bench_close=bench, indicator_kw=ind_kw)
    sectors = [""] * len(tickers)
    kw = dict(LONG_BOOK)
    sleeve = cash_sleeve_returns_us(P["Close"].index)
    if sleeve is not None:
        kw["cash_ret"] = sleeve
        kw["cash_ret_idle_only"] = False
    eq, tr = backtest(tickers, P, sectors=sectors, **kw)
    eq.to_csv(OUTPUT_BACKTEST / "equity_curve_us_long.csv", header=["equity"])
    tr.to_csv(OUTPUT_BACKTEST / "trades_us_long.csv", index=False)
    p = perf_stats(eq.dropna(), "US Long")
    print(f"\nUS Long (≥${US_MIN_MCAP_USD/1e6:.0f}M mcap, {len(tickers)} names, bench {bname})")
    print(f"  CAGR {p['CAGR %']:.1f}%  Max DD {p['Max DD %']:.1f}%  Calmar {p['Calmar']:.2f}  "
          f"Final ${p['Final value']:,.0f}")
    print(f"Saved output/backtest/equity_curve_us_long.csv, trades_us_long.csv")
    return eq, tr


# ----------------------------- INDICATORS / SIGNALS --------------------------
def _rolling_sharpe(close, window, rf_daily=None):
    """Annualized Sharpe from daily returns over `window` sessions."""
    rf_daily = RISK_FREE / 252 if rf_daily is None else rf_daily
    r = close.pct_change()
    min_p = max(21, window // 2)
    mu = r.rolling(window, min_periods=min_p).mean()
    sig = r.rolling(window, min_periods=min_p).std()
    return (mu - rf_daily) / sig.replace(0, np.nan) * np.sqrt(252)


def add_indicators(df, mcap_now_cr=None, shares=None, listed_on=None,
                   tradable_mcap_min=None, turnover_min=None, mcap_in_usd=False,
                   require_mcap=True):
    d = df.copy()
    t_mcap = MIN_MCAP_CR if tradable_mcap_min is None else tradable_mcap_min
    t_turn = MIN_TURNOVER if turnover_min is None else turnover_min
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
    liquid = (d.Close * d.Volume).rolling(20).median() >= t_turn
    ok = liquid
    if listed_on is not None and pd.notna(listed_on):
        cool = pd.Timestamp(listed_on) + pd.Timedelta(days=LISTING_COOLDOWN_DAYS)
        ok = ok & (d.index >= cool)
    if require_mcap and mcap_now_cr is not None and MCAP_MODE == "point_in_time":
        if mcap_in_usd:
            est_mcap = estimate_mcap_usd(d.Close, mcap_now_cr, shares=shares)
        else:
            est_mcap = estimate_mcap_cr(d.Close, mcap_now_cr, shares=shares)
        ok = ok & (est_mcap >= t_mcap)
    elif require_mcap and mcap_now_cr is not None and MCAP_MODE == "today":
        ok = ok & (float(mcap_now_cr) >= t_mcap)
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


def build_panels(data, universe=None, mcaps=None, bench_close=None, indicator_kw=None):
    """universe: DataFrame indexed by symbol with mcap_cr, shares, listing (from load_universe)."""
    indicator_kw = indicator_kw or {}
    if universe is not None:
        meta = universe
    else:
        mcaps = mcaps or {}
        meta = pd.DataFrame({"mcap_cr": pd.Series(mcaps, dtype=float)})
    ind = {}
    for t, df in data.items():
        row = meta.loc[t] if t in meta.index else {}
        if hasattr(row, "get"):
            m = row.get("mcap_usd", row.get("mcap_cr", np.nan))
            sh = row.get("shares", np.nan)
            listed = row.get("listing", pd.NaT)
        else:
            m, sh, listed = np.nan, np.nan, pd.NaT
        ind[t] = add_indicators(df, m, shares=sh, listed_on=listed, **indicator_kw)
    tickers = list(ind)
    panel = lambda col: pd.DataFrame({t: ind[t][col] for t in tickers}).sort_index()
    entry_cols = ["entry_long", "entry_long90", "entry_long95", "entry_long_s126", "entry_long_vol",
                  "entry_long_smooth", "entry_long_quality", "signal_bo"]
    P = {c: panel(c) for c in
         ["Open", "Close", "Volume", "sma50", "sma100", "sma150", "sma200", "atr", "roc63", "roc126",
          "roc252", "roc_12_1", "vol126", "mom_adj", "mom_smooth", "prox", "prox_ath",
          "sharpe63", "sharpe126", "sharpe252", "tradable"] + entry_cols}
    for c in entry_cols:
        P[c] = P[c].fillna(False).astype(bool)
    P["tradable"] = P["tradable"].fillna(False).astype(bool)
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
        P["bench_close"] = b
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


def long_pick_at_signal(i, tickers, sectors, P, entry_col, rank_col, held_j, n_pos, sector_cap,
                        clusters=None):
    """Same candidate walk as backtest() step 4 on month-end index i (signal close)."""
    ENT = P[entry_col].to_numpy()
    RANK = P[rank_col].to_numpy()
    held_j = set(held_j)
    slots = n_pos - len(held_j)
    if slots <= 0:
        return []
    cap_sec = clusters if clusters is not None else sectors
    cands = [j for j in range(len(tickers)) if ENT[i, j] and j not in held_j and np.isfinite(RANK[i, j])]
    cands.sort(key=lambda j: -RANK[i, j])
    held_sec = {}
    for j in held_j:
        _sector_counts_add(held_sec, cap_sec[j], 1)
    picked = []
    for j in cands:
        if len(picked) >= slots:
            break
        sec = cap_sec[j]
        if sector_cap_blocks(held_sec, sec, sector_cap):
            continue
        picked.append(j)
        _sector_counts_add(held_sec, sec, 1)
    close = P["Close"].iloc[i]
    rows = []
    for j in picked:
        rows.append(dict(
            ticker=tickers[j],
            sector=sectors[j],
            cluster=cap_sec[j],
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
    sectors, clusters = aligned_sector_lists(tickers)
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

    # Index buy-gate status (lag-1), same series as adopted_long_kwargs default
    gate_allows = True
    if "reentry_ok" in kw:
        gate_allows = bool(np.asarray(kw["reentry_ok"], dtype=bool)[i_last])
    gate_series, gate_meta = load_nifty500_equal_weight_close(refresh=False)
    gate_px = gate_sma = np.nan
    if gate_series is not None and len(gate_series):
        gs = gate_series.reindex(dates).ffill()
        sma = gs.rolling(200, min_periods=200).mean()
        # lag-1 signal uses prior session; show that print for clarity
        lag_i = max(0, i_last - 1)
        gate_px = float(gs.iloc[lag_i]) if np.isfinite(gs.iloc[lag_i]) else np.nan
        gate_sma = float(sma.iloc[lag_i]) if np.isfinite(sma.iloc[lag_i]) else np.nan

    bt_kw = dict(kw)
    if USE_SECTOR_CLUSTERS:
        bt_kw["clusters"] = clusters
    if fresh_account:
        held_j = []
    else:
        _, tr = backtest(tickers, P, sectors=sectors, **bt_kw)
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
        clusters=clusters if USE_SECTOR_CLUSTERS else None,
    )
    slots = n_pos - len(held_j)

    names = _load_nse_company_names()
    for row in picks:
        row["company"] = names.get(row["ticker"], "")

    df = pd.DataFrame(picks)
    print("\n=== Long next buys ===")
    print(
        f"Universe: top {LIVE_UNIVERSE_TOP_N} by mcap (≥ ₹{MIN_MCAP_CR:,.0f} cr) "
        f"series {'+'.join(sorted(NSE_UNIVERSE_SERIES))} · {len(tickers)} with prices "
        f"· list {LIVE_UNIVERSE_LIST_FILE}"
    )
    print(f"Live book: {entry_col}, rank {rank_col}, {n_pos} slots, "
          f"sector cap {format_sector_cap(sector_cap)}"
          f"{' (risk clusters)' if USE_SECTOR_CLUSTERS else ' (NSE industry)'}")
    print(
        f"Index gate: Nifty500 Equal Weight > SMA200 (lag 1"
        f"{'; CNX500 before ' + gate_meta.get('ew_start', '2024-06') if gate_meta else ''}"
        f") → {'ALLOWS new buys' if gate_allows else 'BLOCKS new buys'}"
    )
    if np.isfinite(gate_px) and np.isfinite(gate_sma):
        print(
            f"  Gate print (lag 1): EW {gate_px:.2f} vs SMA200 {gate_sma:.2f} "
            f"({'above' if gate_px > gate_sma else 'below'})"
        )
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
    if not gate_allows:
        print("\n*** Market gate OFF — do not buy the list below until Nifty500 Equal Weight > SMA200. ***")
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
  python main.py download          Sync live universe (incremental end-fill + missing)
  python main.py sync-prices [--end YYYY-MM-DD]   End-fill ALL cached equities (missing dates only)
  python main.py download --force  Full-history resync (escape hatch)
  python main.py download --cache  Load cache only (no Yahoo)
  python main.py download-ranks 500 1000 [--sync|--force|--cache]
  python main.py fix-data          Re-run spike/OHLC cleanup on cached files

  python main.py backtest [--cache|--sync|--force]
  python main.py rolling  [--cache|--sync|--force]
  python main.py live [--fresh] [--cache|--sync|--force]
  python main.py study-mcap-grid [--from-2000] [--cache|--sync|--force]

Data source flags (prices + mcap meta):
  --cache / --cache-only   local CSVs only (default for live/backtest/study)
  --sync  / --yahoo        keep cache; Yahoo for missing / truncated / stale ends (append only)
  --force / --resync / -f  full-history re-download from Yahoo

  python main.py backtest-us       US Long backtest (top mcap list, ~$240M floor)
  python main.py download-us       Top 1000 US by mcap, Yahoo history from 2010
  python main.py import-us         S&P 1500 + 2000 more NASDAQ/NYSE (fill missing, from 2010)
Research (optional, -> output/studies/):
  python main.py study rank | entry | validate | atr | compare | universe | regime | us | sep2024 | top30-band | mom-hybrid | cg-taxonomy | sector-clusters

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
             risk_per_trade=None, daily_dd_stop=None, rank_abs=None, rank_keep=None,
             sma_exit=True, clusters=None):
    """risk_mult and slot_cap are per-date arrays aligned to P['Close'].index.

    risk_mult scales RISK_PER_TRADE (0 blocks the entry). slot_cap replaces
    MAX_POSITIONS on the signal day. Both are read on the signal close, and the
    order fills at the next open.
    crash_off: daily flatten (or partial trim via crash_scale) on held positions.
    reentry_ok: when False, block new entries even on rebalance days.
    daily_dd_stop: check peak drawdown every day (sma200 book); dd_stop stays month-end.
    rank_abs / rank_keep: month-end exit when absolute momentum rank (1=best) > rank_keep.
    sma_exit: if False, skip the stock below-SMA200 month-end exit (studies / mom band).
    sectors: fine industry labels logged on trades.
    clusters: optional cap labels (risk clusters); when None, cap uses sectors.
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
    # Cap uses risk clusters when provided; blotter still logs fine industry in sectors.
    cap_sec = clusters if clusters is not None else sectors
    if len(cap_sec) != len(tickers):
        raise ValueError("clusters/sectors length must match tickers")
    VOL = None if vol is None else np.asarray(vol, dtype=float)
    month_end = month_end_mask(dates, rebal_offset_days)
    crash_off = None if crash_off is None else np.asarray(crash_off, dtype=bool)
    reentry_ok = None if reentry_ok is None else np.asarray(reentry_ok, dtype=bool)
    rank_abs_arr = None if rank_abs is None else np.asarray(rank_abs, dtype=float)
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
                if daily_dd_stop and not p["exit_reason"] and c < p["high"] * (1 - daily_dd_stop):
                    p["exit_reason"] = "peak_dd_daily"
                if (not p["exit_reason"] and time_stop and (i - p["entry_i"]) >= time_stop
                        and c <= p["entry_price"]):
                    p["exit_reason"] = "time_stop"
                if month_end[i] and not p["exit_reason"]:
                    rank_now = rank_pct[i, j] if rank_pct is not None else np.nan
                    abs_now = (rank_abs_arr[i, j] if rank_abs_arr is not None else np.nan)
                    if sma_exit and not np.isnan(S200[i, j]) and c < S200[i, j]:
                        p["exit_reason"] = "below_sma200"
                    elif tiered_exit and S100 is not None and not np.isnan(S100[i, j]) and c < S100[i, j]:
                        sell = p["shares"] // 2
                        if sell >= 1:
                            pending_trims.append((j, sell, "tier_sma100"))
                    elif dd_stop and c < p["high"] * (1 - dd_stop):
                        p["exit_reason"] = "peak_dd"
                    elif (rank_keep is not None and rank_abs_arr is not None
                          and (not np.isfinite(abs_now) or abs_now > float(rank_keep))):
                        p["exit_reason"] = "rank_drop_topN"
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
                _sector_counts_add(held, cap_sec[j], 1)
            picked = []
            for j in cands:
                if len(picked) >= slots:
                    break
                sec = cap_sec[j]
                if sector_cap_blocks(held, sec, sector_cap):
                    continue
                picked.append(j)
                _sector_counts_add(held, sec, 1)
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
            held_now = {}
            for k in pos:
                _sector_counts_add(held_now, sectors[k], 1)
            if sector_cap_blocks(held_now, sectors[j], SECTOR_CAP):
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
            if p["exit_reason"] is None:
                _sector_counts_add(held, sectors[j], 1)
        picked = []
        for j in cands:
            if len(picked) >= slots:
                break
            sec = sectors[j]
            if sector_cap_blocks(held, sec, SECTOR_CAP):
                continue
            picked.append(j)
            _sector_counts_add(held, sec, 1)
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


def _equity_from_price(px, start_equity=INITIAL_CAPITAL):
    s = px.dropna()
    if s.empty:
        return s
    return s / s.iloc[0] * start_equity


def _period_stats(eq, name, start=None, end=None):
    s = eq.dropna()
    if start is not None:
        s = s.loc[pd.Timestamp(start):]
    if end is not None:
        s = s.loc[:pd.Timestamp(end)]
    if len(s) < 2:
        return None
    s = s / s.iloc[0] * INITIAL_CAPITAL
    p = perf_stats(s, name)
    return {
        "name": name,
        "start": str(s.index[0].date()),
        "end": str(s.index[-1].date()),
        "CAGR %": round(p["CAGR %"], 2),
        "Vol %": round(p["Vol %"], 2),
        "Sharpe": round(p["Sharpe"], 3) if np.isfinite(p["Sharpe"]) else np.nan,
        "Max DD %": round(p["Max DD %"], 2),
        "Calmar": round(p["Calmar"], 3) if np.isfinite(p["Calmar"]) else np.nan,
        "Final": round(p["Final value"], 0),
        "Total %": round((s.iloc[-1] / s.iloc[0] - 1) * 100, 2),
    }


def run_rolling_returns():
    """Deep Long (Nifty 500 universe) vs Nifty 500 index comparison."""
    return run_deep_nifty500_compare()


def run_deep_nifty500_compare():
    """
    Complete Long-vs-Nifty-500 comparison on today's Nifty 500 equity universe.
    Writes CSVs under output/backtest/ and prints the key tables.
    """
    _ensure_output_dirs()
    print("Loading Nifty 500 universe + running live Long book...")
    data, universe = load_universe()
    bench, bench_ticker = _load_nifty500_close(None)
    tickers, P = build_panels(data, universe=universe, bench_close=bench)
    eq_long, trades = backtest(
        tickers, P, **backtest_sector_kwargs(tickers), **adopted_long_kwargs(P),
    )
    eq_long = eq_long.dropna()
    idx = eq_long.index

    benches = load_benchmarks()
    n500 = benches.get("Nifty 500")
    if n500 is None and bench is not None:
        n500 = bench
        print(f"  Using fallback Nifty 500 proxy ({bench_ticker})")
    if n500 is None:
        raise SystemExit("Could not load Nifty 500 benchmark (^CRSLDX / ^CNX500).")
    n500 = n500.reindex(idx).ffill()
    n500_eq = _equity_from_price(n500.loc[idx])

    eq_long.to_csv(OUTPUT_BACKTEST / "equity_curve_long.csv", header=["equity"])
    n500_eq.to_csv(OUTPUT_BACKTEST / "equity_curve_nifty500.csv", header=["equity"])
    trades.to_csv(OUTPUT_BACKTEST / "trades_long.csv", index=False)

    long_full = _period_stats(eq_long, "Long (live)")
    n500_full = _period_stats(n500_eq, "Nifty 500")
    tstats = trade_stats(trades)
    years = (eq_long.index[-1] - eq_long.index[0]).days / 365.25
    hold = hold_profile(trades, years)
    headline = pd.DataFrame([
        {**long_full, "Trades": tstats.get("Trades", np.nan),
         "Win %": round(tstats.get("Win rate %", np.nan), 1) if tstats else np.nan,
         "PF": round(tstats.get("Profit factor", np.nan), 2) if tstats else np.nan,
         "Avg hold days": round(tstats.get("Avg days held", np.nan), 0) if tstats else np.nan,
         "Symbols in panel": len(tickers)},
        {**n500_full, "Trades": np.nan, "Win %": np.nan, "PF": np.nan,
         "Avg hold days": np.nan, "Symbols in panel": np.nan},
    ])
    headline["Excess CAGR vs N500 (pts)"] = [
        round(long_full["CAGR %"] - n500_full["CAGR %"], 2), 0.0]
    # Positive = Long had a shallower (less negative) max drawdown than Nifty 500
    headline["DD improvement vs N500 (pts)"] = [
        round(long_full["Max DD %"] - n500_full["Max DD %"], 2), 0.0]
    headline.to_csv(OUTPUT_BACKTEST / "long_vs_n500_headline.csv", index=False)

    periods = [
        ("2000-2009", "2000-01-01", "2009-12-31"),
        ("2010-2018", "2010-01-01", "2018-12-31"),
        ("2019+", "2019-01-01", None),
        ("Full", None, None),
    ]
    sub_rows = []
    for label, a, b in periods:
        lp = _period_stats(eq_long, "Long (live)", a, b)
        np_ = _period_stats(n500_eq, "Nifty 500", a, b)
        if lp is None or np_ is None:
            continue
        sub_rows.append({
            "Period": label,
            "Long start": lp["start"], "Long end": lp["end"],
            "Long CAGR %": lp["CAGR %"], "N500 CAGR %": np_["CAGR %"],
            "Excess CAGR pts": round(lp["CAGR %"] - np_["CAGR %"], 2),
            "Long Max DD %": lp["Max DD %"], "N500 Max DD %": np_["Max DD %"],
            "Long Calmar": lp["Calmar"], "N500 Calmar": np_["Calmar"],
            "Long Sharpe": lp["Sharpe"], "N500 Sharpe": np_["Sharpe"],
            "Long Total %": lp["Total %"], "N500 Total %": np_["Total %"],
            "Long Final": lp["Final"], "N500 Final": np_["Final"],
        })
    sub = pd.DataFrame(sub_rows)
    sub.to_csv(OUTPUT_BACKTEST / "long_vs_n500_subperiods.csv", index=False)

    y_long = yearly_returns(eq_long).dropna()
    y_n500 = yearly_returns(n500_eq).dropna()
    y_long.index = pd.to_datetime(y_long.index).year
    y_n500.index = pd.to_datetime(y_n500.index).year
    years_ix = sorted(set(y_long.index) | set(y_n500.index))
    yearly = pd.DataFrame({
        "Year": years_ix,
        "Long %": [float(y_long[y]) if y in y_long.index else np.nan for y in years_ix],
        "Nifty 500 %": [float(y_n500[y]) if y in y_n500.index else np.nan for y in years_ix],
    })
    yearly["Excess pts"] = yearly["Long %"] - yearly["Nifty 500 %"]
    yearly["Long wins"] = yearly["Excess pts"] > 0
    yearly.to_csv(OUTPUT_BACKTEST / "long_vs_n500_yearly.csv", index=False)

    horizons = (1, 2, 3, 4, 5)
    roll_rows, excess_rows, latest_rows = [], [], []
    series_map = {"Long (live)": eq_long, "Nifty 500": n500_eq}
    for name, eq in series_map.items():
        for y in horizons:
            row = summarize_rolling_windows(eq, y, name)
            if row:
                roll_rows.append(row)
    for y in horizons:
        rl = rolling_window_returns(eq_long, y)
        rn = rolling_window_returns(n500_eq, y)
        if rl.empty or rn.empty:
            continue
        both = rl.index.intersection(rn.index)
        if len(both) == 0:
            continue
        ex = rl.loc[both] - rn.loc[both]
        beat = (rl.loc[both] > rn.loc[both]).mean() * 100
        excess_rows.append({
            "Horizon (y)": y,
            "Windows": len(both),
            "Long beat N500 %": round(beat, 1),
            "Excess total % median": round(ex.median(), 2),
            "Excess total % mean": round(ex.mean(), 2),
            "Excess total % p10": round(float(np.nanpercentile(ex, 10)), 2),
            "Excess total % p90": round(float(np.nanpercentile(ex, 90)), 2),
            "Excess total % worst": round(ex.min(), 2),
            "Excess total % best": round(ex.max(), 2),
            "Long total % median": round(rl.loc[both].median(), 2),
            "N500 total % median": round(rn.loc[both].median(), 2),
            "Long CAGR % median": round(_annualized_from_total_pct(rl.loc[both].median(), y), 2),
            "N500 CAGR % median": round(_annualized_from_total_pct(rn.loc[both].median(), y), 2),
        })
        latest_rows.append({
            "Horizon (y)": y,
            "Long Total %": round(rl.iloc[-1], 2),
            "N500 Total %": round(rn.iloc[-1], 2),
            "Excess pts": round(rl.iloc[-1] - rn.iloc[-1], 2),
            "Long CAGR %": round(_annualized_from_total_pct(rl.iloc[-1], y), 2),
            "N500 CAGR %": round(_annualized_from_total_pct(rn.iloc[-1], y), 2),
        })
    roll = pd.DataFrame(roll_rows)
    excess = pd.DataFrame(excess_rows)
    latest = pd.DataFrame(latest_rows)
    roll.to_csv(OUTPUT_BACKTEST / "rolling_returns_summary.csv", index=False)
    excess.to_csv(OUTPUT_BACKTEST / "long_vs_n500_rolling_excess.csv", index=False)
    latest.to_csv(OUTPUT_BACKTEST / "long_vs_n500_latest_windows.csv", index=False)

    jack = _jackpot(trades) if not trades.empty else {}
    meta = pd.DataFrame([{
        "universe": "Nifty 500 (today membership)",
        "symbols_panel": len(tickers),
        "bench": bench_ticker or "Nifty 500",
        "start": str(eq_long.index[0].date()),
        "end": str(eq_long.index[-1].date()),
        "years": round(years, 2),
        "median_hold_days": hold.get("Median days", np.nan),
        "held_ge_1y_pct": hold.get("Held >= 1y %", np.nan),
        "trades_per_year": hold.get("Per year", np.nan),
        "2x_closed": hold.get("2x closed", np.nan),
        "3x_closed": hold.get("3x closed", np.nan),
        "pnl_all": jack.get("PnL all", np.nan),
        "top10_pnl_share": (
            jack["Top 10 PnL"] / jack["PnL all"] if jack.get("PnL all") else np.nan
        ),
        "yearly_long_win_rate_%": round(
            float(yearly.loc[yearly["Excess pts"].notna(), "Long wins"].mean() * 100), 1),
        "yearly_avg_excess_pts": round(float(yearly["Excess pts"].mean()), 2),
    }])
    meta.to_csv(OUTPUT_BACKTEST / "long_vs_n500_meta.csv", index=False)

    y_both = yearly.dropna(subset=["Excess pts"])
    y_win = float(y_both["Long wins"].mean() * 100) if len(y_both) else np.nan
    y_ex = float(y_both["Excess pts"].mean()) if len(y_both) else np.nan

    print("\n" + "=" * 72)
    print("LONG (Nifty 500 universe) vs NIFTY 500 — deep comparison")
    print("=" * 72)
    print(f"Panel: {len(tickers)} symbols | {eq_long.index[0].date()} → {eq_long.index[-1].date()} "
          f"({years:.1f}y) | bench {bench_ticker or 'Nifty 500'}")
    print("Note: ^CRSLDX (Nifty 500) history starts ~2007; earlier Long years have no index comparison.")
    print("\n--- Headline ---")
    print(headline.to_string(index=False))
    print("\n--- Subperiods ---")
    print(sub.to_string(index=False))
    print("\n--- Rolling windows (summary) ---")
    print(roll.to_string(index=False))
    print("\n--- Rolling excess (Long − Nifty 500) ---")
    print(excess.to_string(index=False))
    print("\n--- Latest window (ends on last date) ---")
    print(latest.to_string(index=False))
    print(f"\n--- Calendar years (overlap only): Long win rate {y_win:.0f}% | "
          f"avg excess {y_ex:+.1f} pts ---")
    print(yearly.to_string(index=False))
    print(f"\nTrade profile: {tstats.get('Trades', 0)} trades, "
          f"win {tstats.get('Win rate %', float('nan')):.1f}%, "
          f"PF {tstats.get('Profit factor', float('nan')):.2f}, "
          f"avg hold {tstats.get('Avg days held', float('nan')):.0f}d")
    print("Saved under output/backtest/: long_vs_n500_*.csv, rolling_returns_summary.csv, "
          "equity_curve_long.csv, equity_curve_nifty500.csv, trades_long.csv")
    return {
        "headline": headline, "subperiods": sub, "yearly": yearly,
        "rolling": roll, "excess": excess, "latest": latest, "meta": meta,
        "eq_long": eq_long, "eq_n500": n500_eq, "trades": trades,
    }


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


def sector_labels(sec):
    """Split a sector field into industry labels. Multi-tags use '|', ';', or '+'."""
    if sec is None:
        return []
    if isinstance(sec, (list, tuple)):
        return [str(s).strip() for s in sec if str(s).strip() and str(s).strip().lower() != "nan"]
    if isinstance(sec, float) and not np.isfinite(sec):
        return []
    s = str(sec).strip()
    if not s or s.lower() in ("nan", "none", ""):
        return []
    parts = re.split(r"[|;+]+", s)
    return [p.strip() for p in parts if p.strip() and p.strip().lower() != "nan"]


def _sector_counts_add(held, sec, delta=1, mode=None):
    """Increment / decrement industry labels on a name (multi-industry aware)."""
    labs = sector_labels(sec)
    mode = mode or MULTI_SECTOR_CAP_MODE
    if mode == "primary_only" and labs:
        labs = labs[:1]
    for lab in labs:
        held[lab] = held.get(lab, 0) + delta


# Multi-industry sector cap:
#   "any_room" — allow a name if at least one of its industries is still under the
#                cap (e.g. Capital Goods full but Chemicals open → CG|Chemicals OK).
#                Buying still increments every label (CG can go above the cap via
#                dual-tagged names). From-2015 retest: weaker than all_clear.
#   "all_clear" — block if any industry is already at the cap (stricter; live default).
#   "primary_only" — only the first tag counts toward the cap (hybrids softer).
MULTI_SECTOR_CAP_MODE = "all_clear"

# Risk clusters used for sector_cap when USE_SECTOR_CLUSTERS is True.
# Fine industry stays on the blotter; cap counts exactly one cluster per name.
# sector-clusters study (2010 + peaks): median Calmar did not beat NSE industry cap → False.
USE_SECTOR_CLUSTERS = False

CAP_CLUSTER_HIGHLIGHTS = (
    "Industrials Electrical",
    "Industrials Mechanical",
    "Industrials Defense",
    "Banks",
    "NBFC Lending",
    "Insurance",
    "Capital Markets",
    "Pharma",
    "Hospitals Diagnostics",
)


def sector_cap_limit(lab, sector_cap):
    """Per-industry limit. sector_cap may be an int or {industry: cap, '_default': n}."""
    if isinstance(sector_cap, dict):
        return int(sector_cap.get(lab, sector_cap.get("_default", SECTOR_CAP)))
    return int(sector_cap)


def sector_cap_blocks(held, sec, sector_cap, mode=None):
    """True if sector_cap forbids buying this name given current held counts."""
    if not sector_cap:
        return False
    labs = sector_labels(sec)
    if not labs:
        return False  # unlabeled still uncapped
    mode = mode or MULTI_SECTOR_CAP_MODE
    if mode == "primary_only":
        labs = labs[:1]
    at_cap = [held.get(lab, 0) >= sector_cap_limit(lab, sector_cap) for lab in labs]
    if mode == "all_clear":
        return any(at_cap)  # block if any industry is full
    if mode == "primary_only":
        return at_cap[0] if at_cap else False
    # any_room: block only if every industry is already full
    return all(at_cap)


def nse_adaptive_sector_cap(min_names=None, large_cap=None, default_cap=None,
                            cache_path=None, overrides_path=None):
    """NSE industry caps: large buckets get large_cap, others default_cap.

    Size is counted from data_cache/sectors.csv tag frequency (primary+secondary).
    Returns dict suitable for backtest(sector_cap=...).
    """
    min_names = LARGE_INDUSTRY_MIN_NAMES if min_names is None else int(min_names)
    large_cap = SECTOR_CAP_LARGE if large_cap is None else int(large_cap)
    default_cap = SECTOR_CAP if default_cap is None else int(default_cap)
    from collections import Counter
    ind_map, _cl = _read_sectors_frame(cache_path, overrides_path)
    counts = Counter()
    for ind in ind_map.values():
        for lab in sector_labels(ind):
            counts[lab] += 1
    out = {"_default": default_cap}
    for lab, n in counts.items():
        out[lab] = large_cap if n >= min_names else default_cap
    return out


def live_sector_cap():
    """Sector cap used by the live Long book (adaptive NSE or plain SECTOR_CAP)."""
    if USE_ADAPTIVE_SECTOR_CAP:
        return nse_adaptive_sector_cap()
    return SECTOR_CAP


def format_sector_cap(sector_cap):
    """Short string for logs / live banner."""
    if isinstance(sector_cap, dict):
        large = sorted(k for k, v in sector_cap.items()
                       if k != "_default" and v >= SECTOR_CAP_LARGE)
        d = sector_cap.get("_default", SECTOR_CAP)
        return (f"adaptive large={SECTOR_CAP_LARGE} else={d} "
                f"({len(large)} large: {', '.join(large[:6])}"
                f"{'…' if len(large) > 6 else ''})")
    return str(sector_cap)


def _sector_tag_counts(mapping):
    """Count primary+secondary tags across a symbol→industry mapping."""
    from collections import Counter
    c = Counter()
    for ind in mapping.values():
        for lab in sector_labels(ind):
            c[lab] += 1
    return c


def _read_sectors_frame(cache_path=None, overrides_path=None):
    """Load symbol→industry and symbol→cluster maps from cache + overrides."""
    CACHE_DIR.mkdir(exist_ok=True)
    cache = Path(cache_path) if cache_path else (CACHE_DIR / "sectors.csv")
    overrides = Path(overrides_path) if overrides_path else (CACHE_DIR / "sector_overrides.csv")
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
            return {}, {}
        out = pd.concat(frames, ignore_index=True)
        out["symbol"] = out["symbol"].astype(str).str.strip()
        out["industry"] = out["industry"].astype(str).str.strip()
        out["cluster"] = out["industry"]
        out.drop_duplicates("symbol").to_csv(cache, index=False)
    df = pd.read_csv(cache)
    df.columns = [str(c).strip().lower() for c in df.columns]
    if "symbol" not in df.columns or "industry" not in df.columns:
        return {}, {}
    # Live/backtest use a single NSE industry per name (first tag if multi-tag sneaks in).
    def _one_nse(ind):
        labs = sector_labels(ind)
        return labs[0] if labs else ""

    ind_map = {str(sym).strip(): _one_nse(ind)
               for sym, ind in zip(df.symbol.astype(str), df.industry.astype(str))}
    if "cluster" in df.columns:
        cl_map = {str(sym).strip(): _one_nse(cl)
                  for sym, cl in zip(df.symbol.astype(str), df.cluster.fillna("").astype(str))}
    else:
        cl_map = dict(ind_map)
    if overrides.exists():
        ov = pd.read_csv(overrides)
        ov.columns = [str(c).strip().lower() for c in ov.columns]
        if {"symbol", "industry"}.issubset(set(ov.columns)):
            for _, r in ov.iterrows():
                sym = str(r["symbol"]).strip()
                ind = _one_nse(r["industry"])
                if sym and ind:
                    ind_map[sym] = ind
                if "cluster" in ov.columns:
                    cl = _one_nse(r["cluster"])
                    if sym and cl:
                        cl_map[sym] = cl
                elif sym and ind:
                    cl_map[sym] = ind
    for sym, ind in ind_map.items():
        cl = cl_map.get(sym, "")
        if not cl or cl.lower() in ("nan", "none"):
            cl_map[sym] = ind
    return ind_map, cl_map


def load_sectors(tickers, cache_path=None, overrides_path=None, quiet=False):
    """
    Fine industry labels for the blotter / display.
    Cached in data_cache/sectors.csv (symbol, industry[, cluster]).
    Multi-tag OK: \"Electrical Equipment | Engineering\" or \"Textiles + Engineering\".
    Optional sector_overrides.csv wins per symbol. Missing symbols stay unlabeled.
    """
    ind_map, cl_map = _read_sectors_frame(cache_path, overrides_path)
    if not quiet and not getattr(load_sectors, "_logged", False):
        n_cl = sum(1 for v in cl_map.values() if v)
        bits = []
        from collections import Counter
        cc = Counter(cl_map.values())
        for lab in CAP_CLUSTER_HIGHLIGHTS:
            bits.append(f"{lab}={cc.get(lab, 0)}")
        print(f"  Sectors: {len(ind_map)} industry | {n_cl} clusters | "
              f"{', '.join(bits)}")
        load_sectors._logged = True
    return {t: ind_map.get(t, "") for t in tickers}


def load_sector_clusters(tickers, cache_path=None, overrides_path=None, quiet=True):
    """Single risk-cluster label per name for sector_cap (see USE_SECTOR_CLUSTERS)."""
    _ind_map, cl_map = _read_sectors_frame(cache_path, overrides_path)
    if not quiet:
        load_sectors(tickers, cache_path=cache_path, overrides_path=overrides_path, quiet=False)
    return {t: cl_map.get(t, "") for t in tickers}


def aligned_sector_lists(tickers, use_clusters=None, cache_path=None, overrides_path=None):
    """Return (industries, cap_labels) lists aligned to tickers.

    industries — fine labels for trade logs.
    cap_labels — clusters when use_clusters else industries (legacy NSE cap).
    """
    if use_clusters is None:
        use_clusters = USE_SECTOR_CLUSTERS
    ind_map, cl_map = _read_sectors_frame(cache_path, overrides_path)
    industries = [ind_map.get(t, "") for t in tickers]
    if use_clusters:
        cap_labels = [cl_map.get(t, "") or (sector_labels(industries[i])[0]
                      if sector_labels(industries[i]) else "")
                      for i, t in enumerate(tickers)]
    else:
        cap_labels = list(industries)
    return industries, cap_labels


def backtest_sector_kwargs(tickers, use_clusters=None, cache_path=None, overrides_path=None):
    """kwargs to pass into backtest(): sectors (+ clusters when enabled)."""
    industries, clusters = aligned_sector_lists(
        tickers, use_clusters=use_clusters,
        cache_path=cache_path, overrides_path=overrides_path,
    )
    kw = {"sectors": industries}
    if (USE_SECTOR_CLUSTERS if use_clusters is None else use_clusters):
        kw["clusters"] = clusters
    return kw


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


def _load_nifty_smallcap_close(dates=None):
    """Nifty Smallcap 250 / CNX Smallcap close (Yahoo)."""
    for ticker in ("NIFTYSMLCAP250.NS", "^CNXSC"):
        df = download_one(ticker)
        if df is not None and len(df) >= 252:
            s = df.Close.copy()
            s.index = pd.to_datetime(s.index).tz_localize(None)
            return s.dropna(), ticker
    return None, ""


def load_n500_smallcap_ew_close():
    """Equal-weight Nifty 500 + Nifty Smallcap 250 composite level series.

    Daily returns of each leg are averaged 50/50 (on days both exist), then
    compounded. Study gate only (not live default).
    Returns (composite_close, meta_dict) or (None, meta).
    """
    n500, t500 = _load_nifty500_close(None)
    small, tsc = _load_nifty_smallcap_close(None)
    meta = {"nifty500": t500 or None, "smallcap": tsc or None}
    if n500 is None or small is None:
        meta["error"] = "missing leg"
        return None, meta
    a = n500.copy()
    a.index = pd.to_datetime(a.index).tz_localize(None)
    b = small.copy()
    b.index = pd.to_datetime(b.index).tz_localize(None)
    ret = pd.DataFrame({"n500": a.pct_change(), "small": b.pct_change()}).dropna(how="any")
    if len(ret) < 252:
        meta["error"] = "overlap too short"
        return None, meta
    lvl = (1.0 + ret.mean(axis=1)).cumprod() * 1000.0
    meta.update({
        "start": str(lvl.index.min().date()),
        "end": str(lvl.index.max().date()),
        "bars": int(len(lvl)),
    })
    return lvl, meta


def _nse_headers():
    return {
        "User-Agent": "Mozilla/5.0",
        "Accept": "text/csv,*/*",
        "Referer": "https://www.nseindia.com/",
    }


def _parse_nse_ind_close_csv(text, index_name=NIFTY500_EW_INDEX_NAME):
    """Return (date, close) from one NSE ind_close_all CSV, or (None, None)."""
    from io import StringIO
    df = pd.read_csv(StringIO(text))
    df.columns = df.columns.str.strip()
    if "Index Name" not in df.columns or "Closing Index Value" not in df.columns:
        return None, None
    m = df[df["Index Name"].astype(str).str.strip() == index_name]
    if m.empty:
        return None, None
    row = m.iloc[0]
    close = pd.to_numeric(row["Closing Index Value"], errors="coerce")
    if not np.isfinite(close):
        return None, None
    d = pd.to_datetime(str(row.get("Index Date", "")), dayfirst=True, errors="coerce")
    if pd.isna(d):
        return None, None
    return pd.Timestamp(d).normalize(), float(close)


def _read_nifty500_ew_cache():
    path = NIFTY500_EW_CACHE
    if not path.exists():
        return pd.Series(dtype=float)
    df = pd.read_csv(path)
    if df.empty or "Date" not in df.columns or "Close" not in df.columns:
        return pd.Series(dtype=float)
    idx = pd.DatetimeIndex(pd.to_datetime(df["Date"], errors="coerce"))
    if idx.tz is not None:
        idx = idx.tz_convert(None)
    s = pd.Series(
        pd.to_numeric(df["Close"], errors="coerce").to_numpy(),
        index=idx.normalize(),
        dtype=float,
    ).dropna().sort_index()
    s = s[~s.index.duplicated(keep="last")]
    return s


def _write_nifty500_ew_cache(series):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    s = series.dropna().sort_index()
    s = s[~s.index.duplicated(keep="last")]
    pd.DataFrame({"Date": s.index.strftime("%Y-%m-%d"), "Close": s.values}).to_csv(
        NIFTY500_EW_CACHE, index=False,
    )


def refresh_nifty500_equal_weight_cache(start=None, end=None, max_days=None):
    """Pull Nifty500 Equal Weight closes from NSE ind_close_all archives into cache.

    Incremental: only fetches calendar days after the last cached bar (or from
    NIFTY500_EW_ARCHIVE_START). Holidays/404s are skipped. Returns cached Series.
    """
    import requests

    cached = _read_nifty500_ew_cache()
    start = pd.Timestamp(start or (
        (cached.index.max() + pd.Timedelta(days=1)) if len(cached)
        else NIFTY500_EW_ARCHIVE_START
    )).normalize()
    end = pd.Timestamp(end or pd.Timestamp.today()).normalize()
    if start > end:
        return cached

    days = pd.bdate_range(start, end)
    if max_days is not None:
        days = days[: int(max_days)]
    got = {}
    miss = 0
    for d in days:
        url = NSE_IND_CLOSE_URL.format(ddmmyyyy=d.strftime("%d%m%Y"))
        try:
            r = requests.get(url, headers=_nse_headers(), timeout=25)
            if r.status_code != 200 or b"<" in r.content[:80]:
                miss += 1
                continue
            dt, close = _parse_nse_ind_close_csv(r.text)
            if dt is None:
                miss += 1
                continue
            got[pd.Timestamp(dt).normalize()] = close
        except Exception:
            miss += 1
            continue
    if got:
        add = pd.Series(got, dtype=float).sort_index()
        cached = pd.concat([cached, add]).sort_index()
        cached = cached[~cached.index.duplicated(keep="last")]
        _write_nifty500_ew_cache(cached)
        print(
            f"  Nifty500 Equal Weight cache: +{len(add)} days "
            f"({cached.index.min().date()} → {cached.index.max().date()}, "
            f"{len(cached)} bars; {miss} archive misses)"
        )
    elif len(cached):
        print(
            f"  Nifty500 Equal Weight cache unchanged "
            f"({cached.index.min().date()} → {cached.index.max().date()}, {len(cached)} bars)"
        )
    else:
        print("  Nifty500 Equal Weight: no archive rows fetched")
    return cached


def load_nifty500_equal_weight_close(refresh=False):
    """Official Nifty500 Equal Weight close (NSE archives), with CNX500 before it exists.

    Live SMA200 buy-gate series: use Equal Weight when available (~2024-06+);
    splice Yahoo ^CRSLDX / ^CNX500 levels for earlier dates (return-matched at join).
    Returns (close_series, meta_dict).
    """
    meta = {"index": NIFTY500_EW_INDEX_NAME, "source": "nse_archives"}
    ew = _read_nifty500_ew_cache()
    need_refresh = refresh or len(ew) < 200
    if not need_refresh and len(ew):
        # Refresh if cache is more than 2 calendar days behind a typical last session.
        lag = (pd.Timestamp.today().normalize() - ew.index.max()).days
        need_refresh = lag > 3
    if need_refresh:
        try:
            ew = refresh_nifty500_equal_weight_cache()
        except Exception as e:
            meta["refresh_error"] = str(e)
            if not len(ew):
                meta["error"] = f"ew unavailable ({e})"
                return None, meta

    if len(ew) < 200:
        meta["error"] = f"ew too short ({len(ew)} bars; need 200)"
        return None, meta

    n500, t500 = _load_nifty500_close(None)
    if n500 is None:
        meta["pre_ew"] = None
        meta.update({
            "start": str(ew.index.min().date()),
            "end": str(ew.index.max().date()),
            "bars": int(len(ew)),
            "ew_start": str(ew.index.min().date()),
        })
        return ew.sort_index(), meta

    a = n500.copy()
    a.index = pd.to_datetime(a.index).tz_localize(None)
    a = a.dropna().sort_index()
    ew = ew.sort_index()
    join = ew.index.min()
    # Cap-weighted Nifty 500 (CNX500/^CRSLDX) until Equal Weight begins.
    pre = a.loc[a.index < join]
    if len(pre):
        # Level-match at the last pre day that also has a CNX500 print on/near join.
        join_n500 = a.asof(join)
        join_ew = float(ew.iloc[0])
        if np.isfinite(join_n500) and join_n500 > 0:
            pre = pre * (join_ew / float(join_n500))
        combined = pd.concat([pre, ew]).sort_index()
        combined = combined[~combined.index.duplicated(keep="last")]
        meta["pre_ew"] = t500 or "CNX500"
        meta["spliced_at"] = str(join.date())
    else:
        combined = ew
        meta["pre_ew"] = None
    meta.update({
        "start": str(combined.index.min().date()),
        "end": str(combined.index.max().date()),
        "bars": int(len(combined)),
        "ew_start": str(ew.index.min().date()),
        "ew_bars": int(len(ew)),
    })
    return combined, meta


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


def adopted_long_kwargs(P, bench=None, *, index_sma200_block=True,
                        index_gate="nifty500_ew"):
    """Live Long default: peak stop + cash sleeve + Nifty500 EW > SMA200 block new buys.

    index_gate (when index_sma200_block=True):
      "nifty500_ew"      — Nifty500 Equal Weight > SMA200 (live default; CNX500 before ~2024-06)
      "nifty500"         — cap-weighted Nifty 500 (^CRSLDX / CNX500) > SMA200
      "n500_smallcap_ew" — equal-weight Nifty 500 + Smallcap 250 > its SMA200 (study)
    Signal lagged 1 session. Holdings still exit on stock SMA200 / peak stop.
    Pass index_sma200_block=False for the ungated baseline (studies only).
    """
    kw = dict(LONG_BOOK)
    kw["sector_cap"] = live_sector_cap()
    sleeve = cash_sleeve_returns(P["Close"].index)
    if sleeve is not None:
        kw["cash_ret"] = sleeve
        kw["cash_ret_idle_only"] = False
    if index_sma200_block:
        gate = str(index_gate or "nifty500_ew").lower()
        series = None
        if gate in ("nifty500_ew", "n500_ew", "equal_weight", "nifty500_equal_weight"):
            series, _meta = load_nifty500_equal_weight_close(refresh=False)
        elif gate in ("n500_smallcap_ew", "dual", "n500_sc", "smallcap_ew"):
            series, _meta = load_n500_smallcap_ew_close()
        if series is None:
            if bench is None:
                bench = P.get("bench_close")
            if bench is None:
                bench, _ = _load_nifty500_close(None)
            series = bench
        if series is not None:
            below = index_below_sma200(series, P["Close"].index, lag=1)
            kw["reentry_ok"] = np.logical_not(np.asarray(below, dtype=bool))
    return kw


def industry_rank_panel(P, sectors, lookback=126):
    """Per-stock percentile rank of its (primary) industry's median lookback return."""
    r = P["Close"].pct_change(lookback)
    # primary = first tag when multi-industry
    primary = [sector_labels(s)[0] if sector_labels(s) else "" for s in sectors]
    sec = pd.Series(primary, index=r.columns)
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
    sec_kw = backtest_sector_kwargs(tickers)
    sectors = sec_kw["sectors"]
    eq, trades = backtest(tickers, P, **sec_kw, **adopted_long_kwargs(P))
    eq_peak, _ = backtest(tickers, P, **sec_kw, **LONG_BOOK)
    eq_legacy, _ = backtest(tickers, P, **sec_kw, **LONG_BOOK_LEGACY)
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


def _open_industry_mix(tr):
    """Industry counts for open_at_end rows (end-of-book concentration)."""
    if tr is None or tr.empty or "reason" not in tr.columns:
        return 0, np.nan, ""
    open_tr = tr[tr["reason"] == "open_at_end"]
    if open_tr.empty:
        return 0, np.nan, ""
    held = {}
    for sec in open_tr["sector"].fillna(""):
        _sector_counts_add(held, sec, 1)
    if not held:
        return int(len(open_tr)), np.nan, ""
    return int(len(open_tr)), float(max(held.values())), str(dict(sorted(held.items(), key=lambda kv: -kv[1])))


def run_cg_taxonomy_study(
    starts=("2010-01-01", "2021-10-18", "2022-12-01", "2024-09-26"),
):
    """Baseline (pre-CG-split) vs new industrial taxonomy under sector-cap modes."""
    global MULTI_SECTOR_CAP_MODE
    _ensure_output_dirs()
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)

    baseline_path = CACHE_DIR / "sectors_backup_pre_cg_split.csv"
    if not baseline_path.exists():
        raise FileNotFoundError(
            f"Missing {baseline_path} — restore pre-split sectors before studying"
        )

    data, universe = load_universe()
    bench, _ = _load_nifty500_close(None)
    tickers, P = build_panels(data, universe=universe, bench_close=bench)

    # Baseline = pre-split file as-is (no overrides). New = live sectors + overrides.
    load_sectors._logged = True
    dfb = pd.read_csv(baseline_path)
    map_base = {t: dict(zip(dfb.symbol.astype(str).str.strip(),
                            dfb.industry.astype(str).str.strip())).get(t, "")
                for t in tickers}
    map_new = load_sectors(tickers, quiet=True)

    variants = [
        ("A baseline all_clear", map_base, "all_clear"),
        ("B new taxonomy all_clear", map_new, "all_clear"),
        ("C new primary_only", map_new, "primary_only"),
        ("D new any_room", map_new, "any_room"),
    ]

    rows = []
    prev_mode = MULTI_SECTOR_CAP_MODE
    try:
        for start in starts:
            for label, smap, mode in variants:
                MULTI_SECTOR_CAP_MODE = mode
                secs = [smap.get(t, "") for t in tickers]
                kw = adopted_long_kwargs(P)
                eq, tr = backtest(tickers, P, sectors=secs, start=start, **kw)
                row = _after_tax_row(label, eq, tr)
                open_n, max_ind, mix = _open_industry_mix(tr)
                row.update({
                    "WindowStart": start,
                    "Mode": mode,
                    "Open names": open_n,
                    "Max industry count (open)": max_ind,
                    "Open industry mix": mix,
                })
                # Keep Trades / yr alias used in older sector CSVs
                row["Trades / yr"] = row.get("Turnover / yr", np.nan)
                rows.append(row)
                print(
                    f"  [{start}] {label}: CAGR {row['CAGR %']}%  "
                    f"DD {row['Max DD %']}%  Calmar {row['Calmar']}  "
                    f"max_ind={max_ind}"
                )
    finally:
        MULTI_SECTOR_CAP_MODE = prev_mode

    out = pd.DataFrame(rows)
    path = OUTPUT_STUDIES / "cg_taxonomy_split_compare.csv"
    out.to_csv(path, index=False)
    print(f"\nWrote {path}")

    # Adoption gate: across windows, does B (or C) beat A on median Calmar
    # without a large median CAGR give-up (>3 pts)?
    def _med(df, strat, col):
        sub = df[df["Strategy"] == strat]
        return float(sub[col].median()) if len(sub) else np.nan

    a_cal, b_cal, c_cal = (_med(out, s, "Calmar") for s in (
        "A baseline all_clear", "B new taxonomy all_clear", "C new primary_only"))
    a_cagr, b_cagr, c_cagr = (_med(out, s, "CAGR %") for s in (
        "A baseline all_clear", "B new taxonomy all_clear", "C new primary_only"))

    adopt = None
    if np.isfinite(b_cal) and b_cal >= a_cal and (b_cagr >= a_cagr - 3.0):
        adopt = "B"
    elif np.isfinite(c_cal) and c_cal >= a_cal and (c_cagr >= a_cagr - 3.0):
        adopt = "C"
    print("\n=== Median across windows ===")
    print(f"  A Calmar {a_cal:.3f} CAGR {a_cagr:.2f}")
    print(f"  B Calmar {b_cal:.3f} CAGR {b_cagr:.2f}")
    print(f"  C Calmar {c_cal:.3f} CAGR {c_cagr:.2f}")
    if adopt == "B":
        print("Verdict: ADOPT B (new taxonomy, all_clear) — keep sectors.csv as live.")
    elif adopt == "C":
        print("Verdict: ADOPT C (new taxonomy, primary_only) — set MULTI_SECTOR_CAP_MODE.")
    else:
        print("Verdict: KEEP baseline labels (restore from sectors_backup_pre_cg_split.csv).")
    return out, adopt


def _open_label_mix(tr, label_by_ticker):
    """Industry/cluster counts for open_at_end rows using an external label map."""
    if tr is None or tr.empty or "reason" not in tr.columns:
        return 0, np.nan, ""
    open_tr = tr[tr["reason"] == "open_at_end"]
    if open_tr.empty:
        return 0, np.nan, ""
    held = {}
    for t in open_tr["ticker"].astype(str):
        _sector_counts_add(held, label_by_ticker.get(t, ""), 1)
    if not held:
        return int(len(open_tr)), np.nan, ""
    return (
        int(len(open_tr)),
        float(max(held.values())),
        str(dict(sorted(held.items(), key=lambda kv: -kv[1]))),
    )


def run_sector_clusters_study(
    starts=("2010-01-01", "2021-10-18", "2022-12-01", "2024-09-26"),
):
    """NSE industry cap (baseline) vs two-level risk-cluster cap."""
    global USE_SECTOR_CLUSTERS
    _ensure_output_dirs()
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)

    baseline_path = CACHE_DIR / "sectors_backup_pre_clusters.csv"
    if not baseline_path.exists():
        baseline_path = CACHE_DIR / "sectors_backup_pre_cg_split.csv"
    if not baseline_path.exists():
        raise FileNotFoundError("Missing sectors backup for baseline industry cap")

    data, universe = load_universe()
    bench, _ = _load_nifty500_close(None)
    tickers, P = build_panels(data, universe=universe, bench_close=bench)
    load_sectors._logged = True

    dfb = pd.read_csv(baseline_path)
    dfb.columns = [str(c).strip().lower() for c in dfb.columns]
    map_base = dict(zip(dfb.symbol.astype(str).str.strip(),
                        dfb.industry.astype(str).str.strip()))
    base_secs = [map_base.get(t, "") for t in tickers]

    industries, clusters = aligned_sector_lists(tickers, use_clusters=True)
    cl_by_t = dict(zip(tickers, clusters))

    variants = [
        # A: legacy NSE industry labels for both blotter and cap
        ("A baseline industry cap3", base_secs, None, 3),
        # B/C: fine industry on blotter, risk cluster for cap
        ("B cluster cap3", industries, clusters, 3),
        ("C cluster cap4", industries, clusters, 4),
        ("D no cap", industries, None, 0),
    ]

    rows = []
    prev = USE_SECTOR_CLUSTERS
    USE_SECTOR_CLUSTERS = False  # study passes clusters explicitly
    try:
        for start in starts:
            for label, secs, cls, cap in variants:
                kw = adopted_long_kwargs(P)
                kw["sector_cap"] = cap
                bt = dict(sectors=secs, start=start, **kw)
                if cls is not None:
                    bt["clusters"] = cls
                eq, tr = backtest(tickers, P, **bt)
                row = _after_tax_row(label, eq, tr)
                if cls is not None:
                    open_n, max_ind, mix = _open_label_mix(tr, cl_by_t)
                else:
                    open_n, max_ind, mix = _open_industry_mix(tr)
                row.update({
                    "WindowStart": start,
                    "sector_cap": cap,
                    "Open names": open_n,
                    "Max industry count (open)": max_ind,
                    "Open industry mix": mix,
                    "Trades / yr": row.get("Turnover / yr", np.nan),
                })
                rows.append(row)
                print(
                    f"  [{start}] {label}: CAGR {row['CAGR %']}%  "
                    f"DD {row['Max DD %']}%  Calmar {row['Calmar']}  "
                    f"max_ind={max_ind}"
                )
    finally:
        USE_SECTOR_CLUSTERS = prev

    out = pd.DataFrame(rows)
    path = OUTPUT_STUDIES / "sector_clusters_compare.csv"
    out.to_csv(path, index=False)
    print(f"\nWrote {path}")

    def _med(df, strat, col):
        sub = df[df["Strategy"] == strat]
        return float(sub[col].median()) if len(sub) else np.nan

    a_cal, b_cal, c_cal = (_med(out, s, "Calmar") for s in (
        "A baseline industry cap3", "B cluster cap3", "C cluster cap4"))
    a_cagr, b_cagr, c_cagr = (_med(out, s, "CAGR %") for s in (
        "A baseline industry cap3", "B cluster cap3", "C cluster cap4"))

    adopt = None
    if np.isfinite(b_cal) and b_cal >= a_cal and (b_cagr >= a_cagr - 3.0):
        adopt = "B"
    elif np.isfinite(c_cal) and c_cal >= a_cal and (c_cagr >= a_cagr - 3.0):
        adopt = "C"
    print("\n=== Median across windows ===")
    print(f"  A Calmar {a_cal:.3f} CAGR {a_cagr:.2f}")
    print(f"  B Calmar {b_cal:.3f} CAGR {b_cagr:.2f}")
    print(f"  C Calmar {c_cal:.3f} CAGR {c_cagr:.2f}")
    if adopt == "B":
        print("Verdict: ADOPT B (cluster cap 3) — set USE_SECTOR_CLUSTERS=True.")
    elif adopt == "C":
        print("Verdict: ADOPT C (cluster cap 4) — set USE_SECTOR_CLUSTERS=True, SECTOR_CAP=4.")
    else:
        print("Verdict: KEEP baseline industry cap — set USE_SECTOR_CLUSTERS=False.")
    return out, adopt


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


def build_mom_band_panels(P, rank_col="roc126", band_n=30):
    """Absolute roc rank among tradable names; entry only inside top band_n."""
    tradable = P["tradable"].astype(bool)
    mom = P[rank_col].where(tradable)
    rank_abs = mom.rank(axis=1, ascending=False, method="first")
    entry = tradable & mom.notna() & (rank_abs <= float(band_n))
    return rank_abs, entry.fillna(False).astype(bool)


def _strategy_study_row(label, eq, tr, start_capital=INITIAL_CAPITAL):
    eq = eq.dropna()
    p = perf_stats(eq, label)
    ts = trade_stats(tr)
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    closed = tr[tr["reason"] != "open_at_end"] if not tr.empty and "reason" in tr.columns else tr
    n_closed = int(len(closed))
    total_ret = (float(eq.iloc[-1]) / float(start_capital) - 1.0) * 100.0
    row = {
        "Strategy": label,
        "Start": str(eq.index[0].date()),
        "End": str(eq.index[-1].date()),
        "Total %": round(total_ret, 2),
        "CAGR %": round(float(p["CAGR %"]), 2),
        "Max DD %": round(float(p["Max DD %"]), 2),
        "Sharpe": round(float(p["Sharpe"]), 2) if np.isfinite(p["Sharpe"]) else np.nan,
        "Calmar": round(float(p["Calmar"]), 2) if np.isfinite(p.get("Calmar", np.nan)) else np.nan,
        "Final ₹": round(float(p["Final value"]), 0),
        "Trades": int(ts.get("Trades", len(tr))),
        "Closed": n_closed,
        "Win %": round(float(ts.get("Win rate %", np.nan)), 1) if ts else np.nan,
        "Avg days held": round(float(ts.get("Avg days held", np.nan)), 1) if ts else np.nan,
        "Turnover / yr": round(n_closed / years, 2),
    }
    return row, p, ts


# India listed equity (delivery) post-Jul 2024 Budget rates used for study tax drag.
STCG_RATE = 0.20                # hold <= 12 months
LTCG_RATE = 0.125               # hold > 12 months
LTCG_EXEMPT_ANNUAL = 125_000    # ₹1.25L exemption per FY on LTCG (simplified)


def trade_tax_stats(tr, stcg_rate=STCG_RATE, ltcg_rate=LTCG_RATE,
                    ltcg_exempt_annual=LTCG_EXEMPT_ANNUAL, hold_days_lt=365):
    """Estimate STCG/LTCG tax on trade PnL (gains only; annual LTCG exemption).

    Open-at-end rows are treated as if realized on their exit_date (mark-to-market).
    Losses do not create a tax credit here (no loss-setoff modelling).
    """
    empty = {
        "ST trades": 0, "LT trades": 0,
        "ST share %": np.nan, "LT share %": np.nan,
        "ST gains ₹": 0.0, "LT gains ₹": 0.0,
        "ST tax ₹": 0.0, "LT tax ₹": 0.0, "Total tax ₹": 0.0,
        "Tax / gross profit %": np.nan,
    }
    if tr is None or len(tr) == 0:
        return empty
    t = tr.copy()
    if "days_held" not in t.columns:
        t["days_held"] = (pd.to_datetime(t["exit_date"]) - pd.to_datetime(t["entry_date"])).dt.days
    t["exit_date"] = pd.to_datetime(t["exit_date"])
    t["is_lt"] = t["days_held"] > hold_days_lt
    t["gain"] = t["pnl"].clip(lower=0.0)
    st = t.loc[~t["is_lt"]]
    lt = t.loc[t["is_lt"]]
    st_gains = float(st["gain"].sum()) if len(st) else 0.0
    st_tax = st_gains * float(stcg_rate)
    # Annual LTCG exemption by exit calendar year
    lt_tax = 0.0
    lt_gains = float(lt["gain"].sum()) if len(lt) else 0.0
    if len(lt):
        for _, g in lt.groupby(lt["exit_date"].dt.year):
            taxable = max(float(g["gain"].sum()) - float(ltcg_exempt_annual), 0.0)
            lt_tax += taxable * float(ltcg_rate)
    total_tax = st_tax + lt_tax
    gross = float(t["gain"].sum())
    n = len(t)
    return {
        "ST trades": int((~t["is_lt"]).sum()),
        "LT trades": int(t["is_lt"].sum()),
        "ST share %": round(100.0 * (~t["is_lt"]).sum() / n, 1),
        "LT share %": round(100.0 * t["is_lt"].sum() / n, 1),
        "ST gains ₹": round(st_gains, 0),
        "LT gains ₹": round(lt_gains, 0),
        "ST tax ₹": round(st_tax, 0),
        "LT tax ₹": round(lt_tax, 0),
        "Total tax ₹": round(total_tax, 0),
        "Tax / gross profit %": round(100.0 * total_tax / gross, 2) if gross > 0 else np.nan,
    }


def _after_tax_row(label, eq, tr, start_capital=INITIAL_CAPITAL):
    row, p, _ = _strategy_study_row(label, eq, tr, start_capital=start_capital)
    tax = trade_tax_stats(tr)
    final_pre = float(p["Final value"])
    final_post = final_pre - float(tax["Total tax ₹"])
    years = max((eq.dropna().index[-1] - eq.dropna().index[0]).days / 365.25, 1e-9)
    cagr_post = ((final_post / start_capital) ** (1 / years) - 1.0) * 100.0 if final_post > 0 else np.nan
    row.update({
        "ST share %": tax["ST share %"],
        "LT share %": tax["LT share %"],
        "ST tax ₹": tax["ST tax ₹"],
        "LT tax ₹": tax["LT tax ₹"],
        "Total tax ₹": tax["Total tax ₹"],
        "Tax / gross profit %": tax["Tax / gross profit %"],
        "Final after tax ₹": round(final_post, 0),
        "CAGR after tax %": round(float(cagr_post), 2) if np.isfinite(cagr_post) else np.nan,
        "CAGR tax drag pts": round(float(row["CAGR %"]) - float(cagr_post), 2)
        if np.isfinite(cagr_post) else np.nan,
    })
    return row


def run_top30_band_study(start="2010-01-01", band_n=None, bands=(20, 30, 40, 50)):
    """Hold 10; sell if abs roc126 rank outside top band. Sweep bands vs live Long + tax drag."""
    _ensure_output_dirs()
    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 50)
    if band_n is not None:
        bands = (int(band_n),)
    bands = tuple(int(b) for b in bands)

    data, universe = load_universe()
    bench, _ = _load_nifty500_close(None)
    tickers, P = build_panels(data, universe=universe, bench_close=bench)
    sectors = [load_sectors(tickers).get(t, "") for t in tickers]

    print(f"\n=== Momentum band sweep {list(bands)} vs live Long (from {start}) ===")
    print(f"Universe: {len(tickers)} names | 10 slots | sector_cap {SECTOR_CAP} | "
          f"costs {COST_PER_SIDE:.2%}/side")
    print("Band: tradable only, rank roc126, sell if abs rank > N; no SMA200/peak/index gate")
    print(f"Tax model: STCG {STCG_RATE:.0%} (≤12m), LTCG {LTCG_RATE:.1%} (>12m), "
          f"LTCG exempt ₹{LTCG_EXEMPT_ANNUAL:,.0f}/yr; open positions taxed as if sold at end")

    live_kw = dict(adopted_long_kwargs(P), start=start)
    eq_live, tr_live = backtest(tickers, P, sectors=sectors, **live_kw)
    rows = [_after_tax_row("Long (live)", eq_live, tr_live)]
    eq_live.dropna().to_csv(OUTPUT_STUDIES / "mom_band_sweep_2010_equity_live.csv", header=["equity"])
    tr_live.to_csv(OUTPUT_STUDIES / "mom_band_sweep_2010_trades_live.csv", index=False)

    for n in bands:
        rank_abs, entry_band = build_mom_band_panels(P, rank_col="roc126", band_n=n)
        P[f"entry_mom_band_{n}"] = entry_band
        band_kw = dict(
            entry_col=f"entry_mom_band_{n}",
            rank_col="roc126",
            exit_mode="sma200",
            size_mode="equal",
            sector_cap=SECTOR_CAP,
            max_positions=LONG_POSITIONS,
            dd_stop=None,
            sma_exit=False,
            rank_abs=rank_abs.to_numpy(dtype=float),
            rank_keep=n,
            start=start,
        )
        eq_b, tr_b = backtest(tickers, P, sectors=sectors, **band_kw)
        rows.append(_after_tax_row(f"Top-{n} mom band", eq_b, tr_b))
        eq_b.dropna().to_csv(OUTPUT_STUDIES / f"mom_band_sweep_2010_equity_top{n}.csv", header=["equity"])
        tr_b.to_csv(OUTPUT_STUDIES / f"mom_band_sweep_2010_trades_top{n}.csv", index=False)

    compare = pd.DataFrame(rows)
    print("\n--- Pre-tax performance ---")
    pre_cols = ["Strategy", "CAGR %", "Max DD %", "Sharpe", "Calmar", "Final ₹",
                "Win %", "Avg days held", "Turnover / yr"]
    print(compare[pre_cols].to_string(index=False))
    print("\n--- Tax mix & after-tax ---")
    tax_cols = ["Strategy", "ST share %", "LT share %", "ST tax ₹", "LT tax ₹",
                "Total tax ₹", "Tax / gross profit %", "CAGR %", "CAGR after tax %",
                "CAGR tax drag pts", "Final after tax ₹"]
    print(compare[tax_cols].to_string(index=False))

    compare_path = OUTPUT_STUDIES / "mom_band_sweep_2010_compare.csv"
    compare.to_csv(compare_path, index=False)

    # Keep legacy single-band filename when only one band requested
    if len(bands) == 1:
        n = bands[0]
        legacy = compare.iloc[[0, 1]].copy()
        legacy.to_csv(OUTPUT_STUDIES / f"top{n}_band_vs_live_2010_compare.csv", index=False)

    best_pre = compare.loc[compare["CAGR %"].idxmax(), "Strategy"]
    best_post = compare.loc[compare["CAGR after tax %"].idxmax(), "Strategy"]
    print(f"\nVerdict pre-tax CAGR: {best_pre}")
    print(f"Verdict after-tax CAGR: {best_post}")
    print(f"Saved {compare_path}")

    cost_df = run_mom_band_cost_stress(
        tickers, P, sectors, start=start, bands=bands,
        costs=(0.0, 0.0015, 0.0025, 0.004, 0.005),
    )
    slots_df = run_mom_band_slots_sweep(
        tickers, P, sectors, start=start, bands=bands, slots=(10, 15, 20),
    )
    return compare, cost_df, slots_df


def run_mom_band_slots_sweep(tickers, P, sectors, start="2010-01-01",
                             bands=(20, 30, 40, 50), slots=(10, 15, 20)):
    """Equal-weight book size 10/15/20 for live Long and each momentum band."""
    _ensure_output_dirs()
    print(f"\n=== Slots sweep {list(slots)} × bands {list(bands)} (from {start}) ===")
    print(f"sector_cap={SECTOR_CAP}, cost={COST_PER_SIDE:.2%}/side; live uses adopted gate/exits")
    rows = []

    for n_pos in slots:
        live_kw = dict(adopted_long_kwargs(P), start=start, max_positions=n_pos)
        eq_l, tr_l = backtest(tickers, P, sectors=sectors, **live_kw)
        row = _after_tax_row(f"Long (live) {n_pos}sl", eq_l, tr_l)
        row["Slots"] = n_pos
        row["Band"] = "live"
        rows.append(row)

        for band_n in bands:
            rank_abs, entry_band = build_mom_band_panels(P, rank_col="roc126", band_n=band_n)
            col = f"entry_mom_band_{band_n}"
            P[col] = entry_band
            band_kw = dict(
                entry_col=col,
                rank_col="roc126",
                exit_mode="sma200",
                size_mode="equal",
                sector_cap=SECTOR_CAP,
                max_positions=n_pos,
                dd_stop=None,
                sma_exit=False,
                rank_abs=rank_abs.to_numpy(dtype=float),
                rank_keep=band_n,
                start=start,
            )
            eq_b, tr_b = backtest(tickers, P, sectors=sectors, **band_kw)
            row = _after_tax_row(f"Top-{band_n} band {n_pos}sl", eq_b, tr_b)
            row["Slots"] = n_pos
            row["Band"] = band_n
            rows.append(row)

    slots_df = pd.DataFrame(rows)
    print("\n--- Pre-tax by slots × strategy ---")
    show = slots_df[["Strategy", "Slots", "CAGR %", "Max DD %", "Sharpe", "Calmar",
                     "Final ₹", "Win %", "Avg days held", "Turnover / yr",
                     "CAGR after tax %", "CAGR tax drag pts"]]
    print(show.to_string(index=False))

    print("\n--- CAGR % pivot (rows=strategy family, cols=slots) ---")
    # Normalize label for pivot: live / Top-N
    fam = slots_df["Strategy"].str.replace(r" \d+sl$", "", regex=True)
    piv = pd.DataFrame({"Family": fam, "Slots": slots_df["Slots"], "CAGR %": slots_df["CAGR %"]})
    print(piv.pivot(index="Family", columns="Slots", values="CAGR %").to_string())

    print("\n--- After-tax CAGR % pivot ---")
    piv_t = pd.DataFrame({
        "Family": fam, "Slots": slots_df["Slots"],
        "CAGR after tax %": slots_df["CAGR after tax %"],
    })
    print(piv_t.pivot(index="Family", columns="Slots", values="CAGR after tax %").to_string())

    out = OUTPUT_STUDIES / "mom_band_sweep_2010_slots.csv"
    slots_df.to_csv(out, index=False)
    best = slots_df.loc[slots_df["CAGR %"].idxmax()]
    best_tax = slots_df.loc[slots_df["CAGR after tax %"].idxmax()]
    print(f"\nBest pre-tax: {best['Strategy']}  CAGR {best['CAGR %']}%  DD {best['Max DD %']}%")
    print(f"Best after-tax: {best_tax['Strategy']}  CAGR {best_tax['CAGR after tax %']}%")
    print(f"Saved {out}")
    return slots_df


def run_mom_hybrid_study(start="2010-01-01", band_n=30):
    """Top-30 mom band + live risk overlays grid; pick Adopt by Calmar with CAGR floor."""
    _ensure_output_dirs()
    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 50)
    data, universe = load_universe()
    bench, _ = _load_nifty500_close(None)
    tickers, P = build_panels(data, universe=universe, bench_close=bench)
    sectors = [load_sectors(tickers).get(t, "") for t in tickers]

    rank_abs, entry_band = build_mom_band_panels(P, rank_col="roc126", band_n=band_n)
    P["entry_mom_band"] = entry_band
    rank_abs_np = rank_abs.to_numpy(dtype=float)

    live_full = dict(adopted_long_kwargs(P), start=start)
    gate = live_full.get("reentry_ok")
    sleeve = live_full.get("cash_ret")

    def band_base(**extra):
        kw = dict(
            entry_col="entry_mom_band",
            rank_col="roc126",
            exit_mode="sma200",
            size_mode="equal",
            sector_cap=SECTOR_CAP,
            max_positions=LONG_POSITIONS,
            dd_stop=None,
            sma_exit=False,
            rank_abs=rank_abs_np,
            rank_keep=band_n,
            start=start,
        )
        kw.update(extra)
        return kw

    scenarios = [
        ("A Long (live)", dict(live_full), "benchmark"),
        ("B Top-30 raw band", band_base(), "benchmark"),
        # One-change overlays
        ("1 Top-30 + index gate", band_base(reentry_ok=gate), "one_change"),
        ("2 Top-30 + SMA200 exit", band_base(sma_exit=True), "one_change"),
        ("3a Top-30 + peak 30%", band_base(dd_stop=0.30), "one_change"),
        ("3b Top-30 + peak 25%", band_base(dd_stop=0.25), "one_change"),
        ("3c Top-30 + peak 20%", band_base(dd_stop=0.20), "one_change"),
        ("4 Top-30 + equity vol scale", band_base(equity_vol_scale=True), "one_change"),
        # Stacked hybrids
        ("5 Top-30 + SMA + peak30 + gate",
         band_base(sma_exit=True, dd_stop=0.30, reentry_ok=gate), "hybrid"),
        ("6 Top-30 + SMA + peak25 + gate",
         band_base(sma_exit=True, dd_stop=0.25, reentry_ok=gate), "hybrid"),
        ("7 Top-30 + SMA + peak30 + gate + vol",
         band_base(sma_exit=True, dd_stop=0.30, reentry_ok=gate, equity_vol_scale=True),
         "hybrid"),
        ("8 Top-30 + SMA + peak30 + gate + sleeve",
         band_base(sma_exit=True, dd_stop=0.30, reentry_ok=gate,
                   cash_ret=sleeve, cash_ret_idle_only=False),
         "hybrid"),
    ]

    print(f"\n=== Mom hybrid risk grid (Top-{band_n} band) from {start} ===")
    print(f"Universe: {len(tickers)} | 10 slots | sector_cap {SECTOR_CAP} | "
          f"cost {COST_PER_SIDE:.2%}/side")
    print("Score: max Calmar among rows with CAGR >= live CAGR − 1.0 pt; "
          "then Sharpe, then shallower DD, then after-tax CAGR")

    rows = []
    equity_map = {}
    for label, kw, kind in scenarios:
        # Drop None overlays so backtest defaults apply cleanly
        kw = {k: v for k, v in kw.items() if v is not None or k in ("dd_stop",)}
        if kw.get("dd_stop") is None and "dd_stop" in kw:
            # explicit None means no peak stop
            pass
        eq, tr = backtest(tickers, P, sectors=sectors, **kw)
        row = _after_tax_row(label, eq, tr)
        row["Kind"] = kind
        rows.append(row)
        equity_map[label] = eq.dropna()
        print(f"  done: {label}")

    compare = pd.DataFrame(rows)
    print("\n--- Full grid ---")
    cols = ["Strategy", "Kind", "CAGR %", "Max DD %", "Sharpe", "Calmar",
            "Final ₹", "Turnover / yr", "CAGR after tax %", "CAGR tax drag pts"]
    print(compare[cols].to_string(index=False))

    live_cagr = float(compare.loc[compare["Strategy"] == "A Long (live)", "CAGR %"].iloc[0])
    floor = live_cagr - 1.0
    eligible = compare[compare["CAGR %"] >= floor].copy()
    # Prefer hybrids/one_change over raw band when scores tie; keep live eligible
    kind_rank = {"hybrid": 0, "one_change": 1, "benchmark": 2}
    eligible["_kind_rank"] = eligible["Kind"].map(kind_rank).fillna(9)
    # Max DD % is negative; larger (less negative) is better
    eligible = eligible.sort_values(
        by=["Calmar", "Sharpe", "Max DD %", "CAGR after tax %", "_kind_rank"],
        ascending=[False, False, False, False, True],
    )
    adopt = eligible.iloc[0]
    adopt_label = adopt["Strategy"]

    compare["Adopt"] = compare["Strategy"] == adopt_label
    out = OUTPUT_STUDIES / "mom_hybrid_2010_compare.csv"
    compare.drop(columns=["_kind_rank"], errors="ignore").to_csv(out, index=False)

    for label, eq in equity_map.items():
        safe = label.split(" ", 1)[0].replace("/", "_")
        # clearer filenames for key legs
        slug = (
            "live" if label.startswith("A ") else
            "raw_band" if label.startswith("B ") else
            "adopt" if label == adopt_label else
            label.lower().replace(" ", "_").replace("+", "plus").replace("%", "pct")[:48]
        )
        eq.to_csv(OUTPUT_STUDIES / f"mom_hybrid_2010_equity_{slug}.csv", header=["equity"])

    print(f"\nLive CAGR floor: {floor:.2f}% (live {live_cagr:.2f}% − 1.0)")
    print(f"Eligible configs: {len(eligible)} / {len(compare)}")
    print(f"\n*** ADOPT: {adopt_label} ***")
    print(
        f"  CAGR {adopt['CAGR %']}%  Max DD {adopt['Max DD %']}%  "
        f"Sharpe {adopt['Sharpe']}  Calmar {adopt['Calmar']}  "
        f"After-tax CAGR {adopt['CAGR after tax %']}%  "
        f"Turnover/yr {adopt['Turnover / yr']}"
    )
    hybrids = eligible[eligible["Kind"] == "hybrid"]
    best_hybrid_label = None
    if len(hybrids):
        best_hybrid = hybrids.iloc[0]
        best_hybrid_label = best_hybrid["Strategy"]
        print(f"\nBest eligible hybrid: {best_hybrid_label}")
        print(
            f"  CAGR {best_hybrid['CAGR %']}%  Max DD {best_hybrid['Max DD %']}%  "
            f"Sharpe {best_hybrid['Sharpe']}  Calmar {best_hybrid['Calmar']}  "
            f"After-tax CAGR {best_hybrid['CAGR after tax %']}%"
        )
    key_labels = ["A Long (live)", "B Top-30 raw band", adopt_label]
    if best_hybrid_label and best_hybrid_label not in key_labels:
        key_labels.append(best_hybrid_label)
    key = compare[compare["Strategy"].isin(key_labels)]
    print("\n--- Live vs raw Top-30 vs ADOPT vs best hybrid ---")
    print(key[cols].to_string(index=False))
    print(f"\nSaved {out}")
    return compare, adopt_label


def run_mom_band_cost_stress(tickers, P, sectors, start="2010-01-01",
                             bands=(20, 30, 40, 50),
                             costs=(0.0, 0.0015, 0.0025, 0.004, 0.005)):
    """CAGR / final wealth vs cost_per_side (brokerage+STT+slippage) for live + bands."""
    _ensure_output_dirs()
    print(f"\n=== Cost / slippage stress (cost per side; round-trip ≈ 2×) from {start} ===")
    print("Baseline live book already uses 0.15%/side. Higher = worse fills / thicker books.")
    rows = []
    base_live = dict(adopted_long_kwargs(P), start=start)
    # baseline CAGR at 0.15% for drag calc
    base_cagrs = {}

    def _run(label, kw, cps):
        eq, tr = backtest(tickers, P, sectors=sectors, **{**kw, "cost_per_side": cps})
        row, p, ts = _strategy_study_row(label, eq, tr)
        cagr = float(row["CAGR %"])
        if cps == COST_PER_SIDE:
            base_cagrs[label] = cagr
        closed = int(row["Closed"])
        years = max((eq.dropna().index[-1] - eq.dropna().index[0]).days / 365.25, 1e-9)
        # Rough ₹ friction: each closed round-trip pays ~2*cps on position notional;
        # report CAGR drag vs 0.15% baseline when available.
        return {
            "Strategy": label,
            "Cost %/side": round(cps * 100, 2),
            "Round-trip %": round(cps * 200, 2),
            "CAGR %": row["CAGR %"],
            "Max DD %": row["Max DD %"],
            "Final ₹": row["Final ₹"],
            "Closed": closed,
            "Turnover / yr": round(closed / years, 2),
            "Avg days held": row["Avg days held"],
        }

    for cps in costs:
        rows.append(_run("Long (live)", base_live, cps))
        for n in bands:
            rank_abs, entry_band = build_mom_band_panels(P, rank_col="roc126", band_n=n)
            P[f"entry_mom_band_{n}"] = entry_band
            band_kw = dict(
                entry_col=f"entry_mom_band_{n}",
                rank_col="roc126",
                exit_mode="sma200",
                size_mode="equal",
                sector_cap=SECTOR_CAP,
                max_positions=LONG_POSITIONS,
                dd_stop=None,
                sma_exit=False,
                rank_abs=rank_abs.to_numpy(dtype=float),
                rank_keep=n,
                start=start,
            )
            rows.append(_run(f"Top-{n} mom band", band_kw, cps))

    cost_df = pd.DataFrame(rows)
    # Drag vs 0.15%/side baseline
    drag = []
    for _, r in cost_df.iterrows():
        base = base_cagrs.get(r["Strategy"])
        drag.append(round(base - r["CAGR %"], 2) if base is not None else np.nan)
    cost_df["CAGR drag vs 0.15% pts"] = drag

    # Pivot-friendly print: CAGR by strategy × cost
    print("\n--- CAGR % by cost/side ---")
    piv = cost_df.pivot(index="Strategy", columns="Cost %/side", values="CAGR %")
    print(piv.to_string())
    print("\n--- CAGR drag vs 0.15%/side baseline (pts) ---")
    piv_d = cost_df.pivot(index="Strategy", columns="Cost %/side", values="CAGR drag vs 0.15% pts")
    print(piv_d.to_string())
    print("\n--- Detail ---")
    print(cost_df.to_string(index=False))

    out = OUTPUT_STUDIES / "mom_band_sweep_2010_cost_stress.csv"
    cost_df.to_csv(out, index=False)
    print(f"Saved {out}")
    return cost_df


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
    # Data source: download* defaults to sync (fill gaps); everything else cache-only
    _dl_cmds = {
        "download", "download-nifty500", "download-n500",
        "download-ranks", "download-nse-ranks", "download-mcap-ranks",
        "sync-prices", "end-fill", "sync-cache",
        "download-us", "us-download", "import-us", "us-import", "regenerate-us",
    }
    parse_data_mode(sys.argv[2:], default="sync" if cmd in _dl_cmds else "cache")
    # Primary commands
    if cmd in ("help", "-h", "--help"):
        cli_help()
    elif cmd in ("download", "download-nifty500", "download-n500"):
        run_download_nifty500(sys.argv[2:])
    elif cmd in ("download-ranks", "download-nse-ranks", "download-mcap-ranks"):
        run_download_nse_ranks(sys.argv[2:])
    elif cmd in ("sync-prices", "end-fill", "sync-cache"):
        # Incremental: only missing recent bars for every *.NS.csv in data_cache/
        _argv = list(sys.argv[2:])
        _end = None
        _slack = 0
        for i, a in enumerate(_argv):
            if a in ("--end", "-e") and i + 1 < len(_argv):
                _end = _argv[i + 1]
            if a in ("--slack",) and i + 1 < len(_argv):
                _slack = int(_argv[i + 1])
        if _end is None:
            _end = pd.Timestamp.now().strftime("%Y-%m-%d")
        sync_all_cached_prices(end=_end, end_slack_days=_slack, sequential=False, batch=40)
    elif cmd in ("fix-data", "fixdata", "repair-data"):
        run_fix_data(sys.argv[2:])
    elif cmd in ("backtest", "bt"):
        run_backtest()
    elif cmd in ("rolling", "rolling-returns", "compare-n500", "deep-compare"):
        run_deep_nifty500_compare()
    elif cmd in ("study-mcap-grid", "mcap-grid"):
        run_mcap_grid_cli(sys.argv[2:])
    elif cmd in ("live", "long-next", "--long-next"):
        run_long_next()
    elif cmd == "full":
        main()
    elif cmd in ("backtest-us", "us-backtest"):
        run_us_backtest()
    elif cmd in ("download-us", "us-download"):
        run_download_us(sys.argv[2:])
    elif cmd in ("import-us", "us-import"):
        run_import_us(sys.argv[2:])
    elif cmd == "regenerate-us":
        # Fill gaps (no force): S&P 1500 + extra NASDAQ, then top-mcap list
        run_import_us(["--extra", "2000"] + sys.argv[2:])
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
        elif sub in ("us", "us-long", "us-mcap"):
            run_us_long_study()
        elif sub in ("top30-band", "top30_band", "mom-band", "momentum-band"):
            run_top30_band_study()
        elif sub in ("mom-hybrid", "mom_hybrid", "hybrid-band", "top30-hybrid"):
            run_mom_hybrid_study()
        elif sub in ("cg-taxonomy", "cg_taxonomy", "sector-taxonomy", "cg-split"):
            run_cg_taxonomy_study()
        elif sub in ("sector-clusters", "sector_clusters", "clusters", "risk-clusters"):
            run_sector_clusters_study()
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
# * Universe is today's Nifty 500 (survivorship vs true PIT index membership). Optional extras in
#   delisted_symbols.txt. PIT tradable filters still use listing dates + price-scaled mcap.
# * Prices: Yahoo auto_adjust + repair=True, then custom spike/OHLC fix (python main.py fix-data).
#   Symbols with a remaining >60% one-day move are dropped. Report: output/studies/nifty500_data_repair_report.csv
# * Taxes are not modelled (STCG on equity delivery is taxed in India).
# * Don't tune parameters until the equity curve looks good: that is overfitting. Use a
#   train/test split (e.g. tune on 2010-2018, validate on 2019 onwards).