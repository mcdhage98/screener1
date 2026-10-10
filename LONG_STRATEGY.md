# The Long strategy

This is the monthly book in `main.py`. It tries to buy stocks that are already in a strong uptrend and hold them for months, until that trend breaks. It does not trade dips, and it does not use a stop.

The other book in the same file, Current, is a daily pullback trader. This note is only about Long.

Prices are daily bars from Yahoo Finance, adjusted for splits and dividends. A decision is made only on the last NSE session of each calendar month. The order is filled at the next session’s open, so the signal never uses a price that was not yet known.

## What it is trying to do

A stock that goes on to multiply usually spends a long time trending: price above a rising 200-day average, and not far from its 52-week high. The Long book buys that condition and then gets out of the way. It will sit through ordinary pullbacks. It sells only when a month actually closes below the 200-day average.

That is why holds are measured in months. The Current book sells on the 50-day average, so its typical trade lasts about two weeks. Long has no 50-day exit and no daily trailing stop. It does use a **30% peak stop** checked at month-end (see Sell rules).

## The stock list

The starting list is every NSE equity-series symbol in `EQUITY_L.csv`. Two filters cut it down before any signal is considered.

**Size today, used only to decide what to download.** A symbol is downloaded if its current market cap is at least ₹2,000 crore. After dropping series with a one-day move above 60% (usually a bad print), the test used 999 stocks.

**Size on the date of the trade.** A stock can be bought on a past date only if an estimate of its market cap that day is at least ₹2,000 crore. The estimate is today’s market cap scaled by the price change since then. It ignores share issues and buybacks, so it is a rough size check, not a real historical market cap.

**Liquidity.** The median daily traded value over the last 20 sessions must be at least ₹2 crore. Traded value is close times volume.

Both the size check and the liquidity check have to pass on the signal day. A name that is too small or too thin that month is skipped even if the chart looks perfect.

Industry labels come from NSE index constituent files (Nifty Total Market and Nifty Microcap 250). Stocks with no label are not subject to the industry cap.

## When it looks

Indicators are calculated every day. The book only acts on the last trading day of the month.

On that close it either marks a holding for sale, or lines up new buys. Those orders execute at the next trading day’s open. A signal on 30 September is bought or sold at the open of the first session of October. Nothing is done on the other days of the month, except marking the portfolio to market.

If a stock has no open print the next morning, the backtest uses the previous close.

## Buy rules

All of the following must be true on the month-end close. The averages are simple moving averages of the daily close.

1. **Close is above the 200-day average.** The long trend is up.
2. **The 50-day average is above the 200-day average.** The medium trend agrees with the long one. This is an entry filter only. The 50-day average is not used to sell.
3. **The 200-day average is higher than it was 63 trading days ago**, about three months. A flat or falling 200-day average does not qualify, even if price is still above it.
4. **Close is at least 85% of the highest close of the last 252 trading days.** The stock is still within 15% of its 52-week high. The book buys strength, not a stock that has already fallen a long way.
5. **The close is higher than it was 252 trading days ago.** The past year is positive.
6. **The size and liquidity checks above pass.**

There is no pullback rule, no RSI, no ADX, and no requirement of a fresh 52-week high. A stock can qualify for many months in a row. If it is already held, it is not bought again.

## If several stocks qualify

The account holds at most 15 positions. Free slots are the difference between 15 and the number already held, after that month’s sales have been flagged.

Holding 15 names, rather than 10, is what keeps the result intact once the list grows to every stock above ₹2,000 crore. With only 10 slots, the ranking fills the book with whichever names just had the strongest recent run, and a longer list changes those names. Fifteen slots rank by **6-month** price momentum (126 trading days), largest first — chosen after comparing 3/6/12-month momentum, blended scores, and rolling Sharpe ranks on the same live book (`python main.py --rank-study`).

Candidates are sorted by that 6-month return. The book walks down the list and takes names until the free slots are full.

**Industry cap.** If a stock has an industry label, it is skipped when that industry already has 3 open positions. Stocks with no label are not capped. The cap is checked against positions that are still held, including ones bought in the same month.

## How much it buys

Each new position is one fifteenth of the account’s equity at the previous close, about 6.7%. Fifteen positions would use the whole account. Equity is cash plus the market value of open holdings. The weight is also capped at 15% of equity, which does not bind at this size.

The number of shares is that rupee amount divided by the next day’s open, after a 0.15% entry cost, and then rounded down to a whole share. If cash is short, the position is smaller. A buy that cannot afford one share is dropped.

There is no stop, so position size is not based on volatility. A wild stock and a quiet stock get the same weight.

## Sell rules

On a later month-end, two checks run (in this order):

1. **Below the 200-day average:** if the close is below the 200-day average, the position is marked for sale at the next open (`below_sma200`).
2. **30% off the high since entry:** if the close is more than 30% below the highest close since you bought, sell at the next open (`peak_dd`), even if price is still above the 200-day average.

Both checks happen only on the last trading day of the month. There is no profit target, no time stop, and no “sell when it doubles” rule between month-ends.

A sale frees a slot. The same month-end can sell one name and buy another. The sale is filled first, at the next open, and the new buy is filled at that same open if the slot math on the signal day already counted the sale as still occupying the slot. In this backtest the slot is freed only after the sale actually fills, so a name sold at the start of the month does not free a slot until the following month’s signal. Turnover is slower than “sell and replace the same morning.”

## Costs

Each buy and each completed sale pays 0.15% of the trade value. That is a flat allowance for brokerage, STT, other charges, and slippage. It is not a line-by-line tax invoice.

Positions still open on the last day of the test are marked at that day’s close and are not charged an exit cost. In the trade file those rows have reason `open_at_end`.

Taxes are not deducted. In India, equity delivery held for 12 months or less is taxed as a short-term gain. Many closed trades in this book last under a year, so a winner can still be short-term.

## A month, start to finish

Suppose the last session of the month is 31 March, and the account already holds 12 stocks.

1. For each holding, compare that day’s close with its own 200-day average. Any holding below its average is queued for sale.
2. Count free slots: 15 minus 12. Sales queued today do not free a slot until they fill, so if none were already empty, there are 3 slots.
3. Build the list of stocks that pass all six buy rules and are not already held. Sort by 12-month return. Drop any labeled industry that already has 3 holdings.
4. Queue the top 3 remaining names.
5. On the next session’s open, sell the queued holdings and buy the queued names, each for about 6.7% of equity, paying 0.15% each way.
6. Until the last session of April, do nothing except watch prices.

## Cash not in stocks

Cash in the account (the part not invested in the 15 names) earns a daily return each session:

- If **GOLDBEES** closed above its 200-day average **yesterday**, cash gets that day’s gold ETF return.
- Otherwise cash earns a **6.5%** annual rate (flat daily accrual in the backtest).

This applies to whatever cash is idle that day, not only when the book is 100% in cash.

## What the test showed

Same stocks, same costs, from the start of 2010 through 9 October 2026. Starting capital ₹10 lakh. **999** names (download and tradable floor **₹2,000 crore**).

| | Live Long (peak + cash sleeve) | Peak only, no sleeve | Nifty 500 | Nifty Smallcap 250 |
|---|---:|---:|---:|---:|
| Annual return | **~31.3%** | ~27.2% | 10.1% | 11.9% |
| Worst decline from a peak | **~−30%** | −32% | −38% | −61% |
| Calmar (CAGR ÷ |drawdown|) | **~1.04** | ~0.85 | ~0.26 | ~0.20 |
| Final from ₹10 lakh | **~₹9.55 cr** | ~₹5.9 cr | ~₹50 lakh | ~₹66 lakh |

The live book is: 15 names, original buys, rank by **6-month return (roc126)**, SMA200 + 30% peak stop at month-end, plus the gold/debt cash sleeve above.

## What this test cannot prove

The universe is still **survivorship-biased**: delisted and merged companies are missing unless you add them to `delisted_symbols.txt` and have price history in `data_cache/`. Treat headline CAGR as an **upper-ish** reading, not a forecast.

**What the code does for data quality (not a full NSE point-in-time database):**

- **Download** symbols with current mcap ≥ **₹500 cr** (`PIT_UNIVERSE_FLOOR_CR`) plus any `delisted_symbols.txt` entries — so names that later grew into the ₹2,000 cr book can appear in price history.
- **Tradable** only when estimated mcap on that date ≥ **₹2,000 cr**, liquidity passes, and the date is ≥ **126 days** after the NSE **listing date** in `EQUITY_L.csv`.
- **Estimated mcap** = price-scaled today’s cap, capped with **shares×price** when Yahoo reports shares (still not true historical shares outstanding).

Run `python main.py --universe-report` for PIT vs biased “today mcap” Long and vs Nifty on the same window (`universe_bias_report.csv`).

Optional stress: `exit_stress_slippage=True` adds 3% exit slippage when the next open gaps down more than 8%.

Yahoo’s adjusted Indian prices have bad prints. Symbols with a one-day move over 60% were dropped, which does not catch every error.

No earnings, return on capital, or debt filter is applied. The book can buy a stock that is expensive or deteriorating, as long as the price trend is intact.

## Files

Backtest outputs live under **`output/backtest/`** (`trades_long.csv`, `equity_curve_long.csv`, `equity_curve.png`, …). Regenerate with `python main.py backtest`.

**Next rebalance candidates:** `python main.py live` or `python main.py live --fresh` → **`output/live/long_next_buys.csv`**. Before the current month-end is in the price file, the list is a **provisional preview** only; official buys use the **last session of the month** close.

See **`RUN.md`** for all commands.
