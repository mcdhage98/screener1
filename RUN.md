# How to run

India entry point: **`main.py`** · US entry point: **`main_us.py`**

```bash
python main.py help
python main_us.py help
```

## Live (what to buy next)

```bash
python main.py live              # continuing book (refill empty slots)
python main.py live --fresh      # new ₹10L, up to 15 names
```

Output: **`output/live/long_next_buys.csv`**

## Backtest

```bash
python main.py backtest
python main.py rolling    # 1–5 year rolling return stats (Long, Nifty 500)
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

Same Long rules as India (`LONG_STRATEGY.md`). Cache: **`data_cache_us/`** from **2010-01-01**.

```bash
python main_us.py backtest              # all cached US + QQQ / NASDAQ Comp compare
python main_us.py backtest-nasdaq       # NASDAQ-listed cache only + QQQ/IXIC
python main_us.py backtest-top          # top-by-mcap list (~$240M floor)
python main_us.py backtest --through ROK
python main_us.py import                # S&P 1500 + extra NASDAQ/NYSE
python main_us.py import --extra 3000
python main_us.py download --top 1000
python main_us.py download-majors       # curated large NASDAQ names
python main_us.py coverage              # NASDAQ vs cache gaps
python main_us.py study-mcap            # mcap floor sweep -> output/studies/
```

## Data (keep)

- **`data_cache/`** — India price history + `universe_meta.csv`
- **`data_cache_us/`** — US price history + `universe_meta_us.csv`
- **`EQUITY_L.csv`** — NSE symbol list
- **`LONG_STRATEGY.md`** — Long book rules (India + US)
