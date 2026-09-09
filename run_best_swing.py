#!/usr/bin/env python3
"""Run the best discovered swing rule and print monthly stats."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from tabulate import tabulate

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backtest.optimize_rules import (
    RuleSet,
    portfolio_monthly_returns,
    simulate_trades,
    trade_stats,
)
from run_optimize_monthly import load_features, monthly_stats
from backtest.universe import fetch_top_trading_stocks


BEST = RuleSet(
    name="squeeze_mr_mfi6_pb015_mid",
    style="mean_rev",
    mfi_period=6,
    mfi_buy_max=20.0,
    pb_buy_max=0.15,
    require_squeeze=True,
    stop_pct=0.07,
    max_hold=20,
    exit_to_mid=True,
    pb_sell_min=0.85,
    mfi_sell_min=70.0,
)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2019-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--positions", type=int, default=1)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--cache", default="cache_ohlcv_v2")
    p.add_argument("--trades", action="store_true")
    args = p.parse_args()

    if Path("out_top100_universe.csv").exists():
        universe = pd.read_csv("out_top100_universe.csv")
    else:
        universe = fetch_top_trading_stocks(100)
    if args.limit:
        universe = universe.head(args.limit)

    data = load_features(universe, args.start, args.end, Path(args.cache))
    trades: list[dict] = []
    for yahoo, df in data.items():
        for t in simulate_trades(df, BEST):
            t = dict(t)
            t["ticker"] = yahoo
            trades.append(t)

    ts = trade_stats(trades)
    monthly = portfolio_monthly_returns(
        trades, max_positions=args.positions, start=args.start, end=args.end
    )
    ms = monthly_stats(monthly)

    print("BEST RULE: Bandwidth squeeze + MFI(6)<20 + %b<=0.15 → exit mid / -7% / 20d")
    print(tabulate(ts.items(), headers=["trade_metric", "value"], tablefmt="github"))
    print(tabulate(ms.items(), headers=["monthly_metric", "value"], tablefmt="github"))
    print(f"\nPositions={args.positions} | months hitting >=20%: {ms['months_ge_20']}/{ms['months']}")

    if args.trades and trades:
        tdf = pd.DataFrame(trades).sort_values("entry_date")
        print(tabulate(tdf.tail(20), headers="keys", tablefmt="simple", showindex=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
