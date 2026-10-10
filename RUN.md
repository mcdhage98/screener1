# How to run

Single entry point: **`main.py`**

```bash
python main.py help
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
python main.py rolling    # 1–5 year rolling return stats (Long, Current, Nifty 500)
```

Output: **`output/backtest/`**

| File | Content |
|------|---------|
| `trades_long.csv` | Monthly Long trades |
| `trades.csv` | Daily Current trades |
| `equity_curve_long.csv` | Long equity |
| `equity_curve.csv` | Current equity |
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
```

## Data (keep)

- **`data_cache/`** — price history + `universe_meta.csv`
- **`EQUITY_L.csv`** — NSE symbol list
- **`LONG_STRATEGY.md`** — Long book rules
