# My running portfolio (master)

Personal live Long book — **separate from** `main.py` backtests / `live --fresh` previews.

| Field | Value |
|-------|--------|
| Started investing | **2026-10-15** |
| Signal basis | **2026-10-14** (`python main.py live --fresh` preview) |
| Capital | ₹5,00,000 · 10 equal slots |
| Master file | `MASTER.json` |

## Holdings (all OPEN)

CUPID · CYIENTDLM · FCL · IBULLSLTD · MARINE · RAYMOND · RPEL · SKYGOLD · WELCORP · MOREPENLAB

Entry prices = Oct 14 closes. Share counts = floor(₹50,000 / price). Leftover sits as cash in `MASTER.json`.

## Track / analyse

```bash
python track_my_portfolio.py           # mark-to-market from data_cache
python track_my_portfolio.py --sync    # sync prices first, then MTM
```

Writes:

- `STATUS.txt` — human blotter
- `mtm.csv` — per-name P&L
- refreshes `holdings.csv` from `MASTER.json`
