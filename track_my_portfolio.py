#!/usr/bin/env python3
"""Track the personal live Long book in output/my_portfolio/MASTER.json.

Usage:
  python track_my_portfolio.py              # mark-to-market from data_cache
  python track_my_portfolio.py --sync       # end-fill prices via main.py, then MTM
  python track_my_portfolio.py --status     # print summary only (default)

Does not change main.py live/backtest logic. MASTER.json is the source of truth.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MASTER_PATH = ROOT / "output" / "my_portfolio" / "MASTER.json"
HOLDINGS_CSV = ROOT / "output" / "my_portfolio" / "holdings.csv"
STATUS_TXT = ROOT / "output" / "my_portfolio" / "STATUS.txt"
STATUS_CSV = ROOT / "output" / "my_portfolio" / "mtm.csv"
CACHE = ROOT / "data_cache"


def load_master() -> dict:
    if not MASTER_PATH.exists():
        raise SystemExit(f"Missing master book: {MASTER_PATH}")
    return json.loads(MASTER_PATH.read_text())


def save_master(master: dict) -> None:
    MASTER_PATH.write_text(json.dumps(master, indent=2) + "\n")
    rows = []
    for h in master.get("holdings", []):
        rows.append({
            "ticker": h["ticker"],
            "company": h.get("company") or "",
            "sector": h.get("sector") or "",
            "status": h.get("status", "OPEN"),
            "signal_date": h.get("signal_date", ""),
            "entry_date": h.get("entry_date", ""),
            "entry_price": h.get("entry_price", ""),
            "shares": h.get("shares", ""),
            "cost_value": h.get("cost_value", ""),
            "target_slot_inr": h.get("target_slot_inr", ""),
            "notes": h.get("notes", ""),
        })
    with HOLDINGS_CSV.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "ticker", "company", "sector", "status", "signal_date", "entry_date",
                "entry_price", "shares", "cost_value", "target_slot_inr", "notes",
            ],
        )
        w.writeheader()
        w.writerows(rows)


def latest_bar(ticker: str) -> dict | None:
    path = CACHE / f"{ticker}.NS.csv"
    if not path.exists():
        return None
    with path.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None

    def get(r, *keys):
        lower = {k.lower(): v for k, v in r.items()}
        for k in keys:
            if k.lower() in lower and lower[k.lower()] not in ("", None):
                return lower[k.lower()]
        return None

    last = rows[-1]
    return {
        "date": str(get(last, "Date", "date"))[:10],
        "open": float(get(last, "Open")),
        "high": float(get(last, "High")),
        "low": float(get(last, "Low")),
        "close": float(get(last, "Close")),
    }


def fmt_inr(x: float) -> str:
    return f"₹{x:,.0f}"


def fmt_pct(x: float) -> str:
    return f"{x:+.2f}%"


def mark_to_market(master: dict) -> tuple[list[dict], dict]:
    cash = float(master.get("cash_inr", 0))
    capital = float(master.get("starting_capital_inr", 0))
    lines = []
    invested_cost = 0.0
    mtm_value = 0.0
    asof = None

    for h in master.get("holdings", []):
        if h.get("status", "OPEN") != "OPEN":
            continue
        t = h["ticker"]
        bar = latest_bar(t)
        shares = int(h["shares"])
        entry = float(h["entry_price"])
        cost = float(h.get("cost_value") or shares * entry)
        invested_cost += cost
        if bar is None:
            lines.append({
                "ticker": t,
                "company": h.get("company", ""),
                "entry_date": h.get("entry_date", ""),
                "entry_price": entry,
                "shares": shares,
                "cost_value": cost,
                "asof": "",
                "last_close": "",
                "market_value": "",
                "pnl": "",
                "ret_pct": "",
                "weight_pct": "",
                "error": "missing cache",
            })
            continue
        px = bar["close"]
        mv = shares * px
        pnl = mv - cost
        ret = (px / entry - 1.0) * 100.0 if entry else 0.0
        mtm_value += mv
        asof = bar["date"] if asof is None or bar["date"] > asof else asof
        lines.append({
            "ticker": t,
            "company": h.get("company", ""),
            "entry_date": h.get("entry_date", ""),
            "entry_price": round(entry, 4),
            "shares": shares,
            "cost_value": round(cost, 2),
            "asof": bar["date"],
            "last_close": round(px, 4),
            "market_value": round(mv, 2),
            "pnl": round(pnl, 2),
            "ret_pct": round(ret, 2),
            "weight_pct": None,
            "error": "",
        })

    equity = mtm_value + cash
    for row in lines:
        if row["market_value"] != "" and equity > 0:
            row["weight_pct"] = round(100.0 * float(row["market_value"]) / equity, 2)

    summary = {
        "asof": asof or "",
        "starting_capital": capital,
        "cash": cash,
        "invested_cost": round(invested_cost, 2),
        "market_value": round(mtm_value, 2),
        "equity": round(equity, 2),
        "pnl": round(equity - capital, 2),
        "ret_pct": round((equity / capital - 1.0) * 100.0, 2) if capital else 0.0,
        "open_positions": sum(1 for h in master.get("holdings", []) if h.get("status") == "OPEN"),
        "closed_trades": len(master.get("closed_trades", [])),
    }
    return lines, summary


def write_reports(master: dict, lines: list[dict], summary: dict) -> None:
    with STATUS_CSV.open("w", newline="") as f:
        fields = [
            "ticker", "company", "entry_date", "entry_price", "shares", "cost_value",
            "asof", "last_close", "market_value", "pnl", "ret_pct", "weight_pct", "error",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(lines)

    started = master.get("started_investing", "")
    signal = master.get("signal_basis", "")
    name = master.get("name", "My portfolio")
    out = []
    out.append(f"{name.upper()}")
    out.append(f"Master: {MASTER_PATH}")
    out.append(f"Status: {master.get('status', 'RUNNING')}  ·  Started {started} (signal {signal})")
    out.append(f"As of: {summary['asof'] or 'n/a'}  ·  Generated {datetime.now():%Y-%m-%d %H:%M}")
    out.append("")
    out.append("SUMMARY")
    out.append(f"• Equity: {fmt_inr(summary['equity'])}  ({fmt_pct(summary['ret_pct'])})")
    out.append(f"• PnL: {fmt_inr(summary['pnl'])}  on capital {fmt_inr(summary['starting_capital'])}")
    out.append(
        f"• Invested (cost): {fmt_inr(summary['invested_cost'])}  ·  "
        f"MTM: {fmt_inr(summary['market_value'])}  ·  Cash: {fmt_inr(summary['cash'])}"
    )
    out.append(
        f"• Open: {summary['open_positions']}  ·  Closed: {summary['closed_trades']}"
    )
    out.append("")
    out.append("OPEN POSITIONS")
    out.append("─" * 72)
    ranked = sorted(
        [r for r in lines if r.get("ret_pct") != ""],
        key=lambda r: float(r["ret_pct"]),
        reverse=True,
    )
    for i, r in enumerate(ranked, 1):
        out.append(
            f"{i:2}. {r['ticker']:<12} {r['company'][:32]}"
        )
        out.append(
            f"    Bought {r['entry_date']} @ ₹{float(r['entry_price']):,.2f}  × {r['shares']}"
        )
        out.append(
            f"    Now ₹{float(r['last_close']):,.2f}  ·  {fmt_pct(float(r['ret_pct']))}  ·  "
            f"PnL {fmt_inr(float(r['pnl']))}  ·  ~{r['weight_pct']}%"
        )
        out.append("")
    missing = [r for r in lines if r.get("error")]
    if missing:
        out.append("MISSING PRICE CACHE")
        for r in missing:
            out.append(f"  - {r['ticker']}: {r['error']}")
        out.append("")
    out.append("Files: holdings.csv · mtm.csv · MASTER.json")
    STATUS_TXT.write_text("\n".join(out) + "\n")


def print_summary(summary: dict, lines: list[dict]) -> None:
    print(f"\n=== My portfolio MTM ({summary['asof'] or 'n/a'}) ===")
    print(
        f"Equity {fmt_inr(summary['equity'])}  {fmt_pct(summary['ret_pct'])}  "
        f"PnL {fmt_inr(summary['pnl'])}"
    )
    print(
        f"Cash {fmt_inr(summary['cash'])}  ·  "
        f"Open {summary['open_positions']}  ·  Closed {summary['closed_trades']}"
    )
    print("")
    for r in sorted(
        [x for x in lines if x.get("ret_pct") != ""],
        key=lambda x: float(x["ret_pct"]),
        reverse=True,
    ):
        print(
            f"  {r['ticker']:<12} {fmt_pct(float(r['ret_pct'])):>8}  "
            f"{fmt_inr(float(r['pnl'])):>10}  close ₹{float(r['last_close']):,.2f}"
        )
    print(f"\nSaved {STATUS_TXT}")
    print(f"Saved {STATUS_CSV}")


def maybe_sync() -> None:
    cmd = [sys.executable, str(ROOT / "main.py"), "sync-prices"]
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(ROOT))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Track personal Long portfolio (MASTER.json)")
    ap.add_argument("--sync", action="store_true", help="End-fill cache prices before MTM")
    ap.add_argument("--status", action="store_true", default=True, help="Print status (default)")
    args = ap.parse_args(argv)

    if args.sync:
        maybe_sync()

    master = load_master()
    if not master.get("is_master"):
        print("Warning: MASTER.json is_master is not true — still tracking.")
    lines, summary = mark_to_market(master)
    write_reports(master, lines, summary)
    # keep holdings.csv in sync
    save_master(master)
    print_summary(summary, lines)


if __name__ == "__main__":
    main()
