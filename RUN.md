# How to run

India entry point: **`main.py`** · US entry point: **`main_us.py`**

```bash
python main.py help
python main_us.py help
```

## Live (what to buy next)

```bash
python main.py live              # continuing book (refill empty slots)
python main.py live --fresh      # new ₹10L, up to 10 names
```

Output: **`output/live/long_next_buys.csv`**

Live book defaults: **top 800** by mcap (≥ ₹2,000 cr) · **10 slots** · **sector_cap 3**.

## Data source flags (prices + mcap meta)

| Flag | Mode | Behaviour |
|------|------|-----------|
| `--cache` / `--cache-only` | **cache** (default for live/backtest/study) | Local `data_cache/` only — no Yahoo |
| `--sync` / `--yahoo` | **sync** (default for `download*`) | Use cache; Yahoo only for missing gaps |
| `--force` / `--resync` / `-f` | **force** | Re-download from Yahoo, ignore usable cache |

```bash
python main.py live --fresh              # cache only
python main.py live --fresh --sync       # fill any missing prices/meta from Yahoo
python main.py download                  # sync gaps
python main.py download --force          # full resync
python main.py backtest --cache          # explicit cache-only
```

## Data (India — top 800 mcap from 2000)

```bash
python main.py download              # sync gaps from Yahoo -> data_cache/
python main.py download --force      # full resync
python main.py download --cache      # read cache only
python main.py fix-data              # re-apply cleanup on cache (no re-download)
python main.py download-ranks 1 1200 # optional: wider mcap band
```

Universe is **top 800 by today’s mcap** among names ≥ ₹2,000 cr (`nse_top_by_mcap.csv` / `universe_meta.csv`). Repair report: `output/studies/nifty500_data_repair_report.csv`.

## Backtest

```bash
python main.py backtest
python main.py rolling         # deep Long vs Nifty 500 (rolling / yearly / subperiods)
python main.py compare-n500    # same as rolling
```

Output: **`output/backtest/`**

| File | Content |
|------|---------|
| `trades_long.csv` | Monthly Long trades |
| `equity_curve_long.csv` | Long equity |
| `equity_curve.png` | Chart |
| `rolling_returns_summary.csv` | Rolling 1–5y windows (from `python main.py rolling`) |
| `trades_breakout.csv` | Breakout sleeve (reference) |

Slow full grid (breakout sweep + tables):

```bash
python main.py full
```

## Optional research

Writes to **`output/studies/`** only when you run these:

```bash
python main.py study rank
python main.py study validate
python main.py study entry
python main.py study atr
python main.py study compare
python main.py study universe
python main.py study sep2024
python main.py study us         # also: python main_us.py study-mcap
python main.py backtest-us      # prefer: python main_us.py backtest
python main.py import-us                 # also: python main_us.py import
python main.py download-us --top 1000
```

## US (`main_us.py`)

Adopted US book (India `LONG_BOOK` unchanged): **NASDAQ PIT top 700**, **12 slots**, **sector_cap=4**, **peak 20%**, QQQ sleeve. Cache: **`data_cache_us/`**.

```bash
python main_us.py backtest              # adopted NASDAQ Long + QQQ/IXIC
python main_us.py backtest-nasdaq       # same
python main_us.py backtest --ndx        # ^IXIC filter → QQQ sleeve
python main_us.py backtest-top          # legacy top-mcap path
python main_us.py study
python main_us.py import
python main_us.py import --extra 3000
python main_us.py download --top 1000
python main_us.py download-majors
python main_us.py coverage
python main_us.py study-mcap
```

## Data (keep)

- **`data_cache/`** — India price history + `universe_meta.csv` + `nse_top_by_mcap.csv`
- **`data_cache_us/`** — US price history + `universe_meta_us.csv`
- **`ind_nifty500list.csv`** — Nifty 500 list (benchmark / optional)
- **`EQUITY_L.csv`** — NSE listing dates / metadata
- **`LONG_STRATEGY.md`** — Long book rules (India + US)
