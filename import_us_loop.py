#!/usr/bin/env python3
"""Resilient US price importer — continues until S&P 1500 + extras are cached."""
from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

# Ensure project root on path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from main import (  # noqa: E402
    US_CACHE_DIR,
    US_GOLD_TICKER,
    US_PRICE_START,
    _us_cache_file,
    _us_cache_is_usable,
    download_one_us,
    us_equity_symbol_list,
    us_nasdaq_nyse_symbols,
    _yahoo_symbol_us,
)

LOG = ROOT / "output" / "us_download.log"
PROGRESS = ROOT / "output" / "us_import_progress.txt"


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def cached_count() -> int:
    return len(list(US_CACHE_DIR.glob("*.csv")))


def build_queue(extra_n: int = 2500) -> list[str]:
    sp = sorted(set(us_equity_symbol_list()))
    for e in (US_GOLD_TICKER, "SPY", "^GSPC"):
        if e not in sp:
            sp.append(e)
    have = {p.stem for p in US_CACHE_DIR.glob("*.csv") if not p.stem.startswith("IDX_")}
    nasdaq = [
        s for s in us_nasdaq_nyse_symbols()
        if s not in have and s not in set(sp) and 1 <= len(s) <= 5 and "-" not in s
    ]
    extras = nasdaq[:extra_n]
    return sp + extras


def need_download(symbols: list[str]) -> list[str]:
    out = []
    for s in symbols:
        y = _yahoo_symbol_us(s)
        if not _us_cache_is_usable(_us_cache_file(y), US_PRICE_START):
            out.append(s)
    return out


def main() -> None:
    extra_n = 2500
    if "--extra" in sys.argv:
        i = sys.argv.index("--extra")
        if i + 1 < len(sys.argv):
            extra_n = int(sys.argv[i + 1])

    log(f"=== import_us_loop start (extra={extra_n}) cache={cached_count()} ===")
    symbols = build_queue(extra_n=extra_n)
    need = need_download(symbols)
    log(f"queue={len(symbols)} need={len(need)} already_ok={len(symbols) - len(need)}")

    ok = fail = 0
    for i, s in enumerate(need, 1):
        try:
            df = download_one_us(s, start=US_PRICE_START, force=False)
            if df is not None and len(df) > 0:
                ok += 1
                status = f"ok bars={len(df)} from={df.index.min().date()}"
            else:
                fail += 1
                status = "empty/fail"
        except Exception as e:
            fail += 1
            status = f"EXC {e}"
            log(traceback.format_exc())

        if i % 5 == 0 or i == len(need):
            msg = f"{i}/{len(need)} ok={ok} fail={fail} cache={cached_count()} last={s} {status}"
            log(msg)
            PROGRESS.write_text(msg + "\n", encoding="utf-8")
        time.sleep(2.5)

    log(f"=== import_us_loop done ok={ok} fail={fail} cache={cached_count()} ===")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log(traceback.format_exc())
        raise
