#!/usr/bin/env python3
"""Most certain BB/%b/MFI/Bandwidth uptrend rule (Friday weekly)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tabulate import tabulate

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backtest.optimize_rules import trade_stats
from run_bb_uptrend_certain import (
    ExitSpec,
    friday_bb_pack,
    simulate_with_mask,
    uptrend_hit_rate,
    with_exit,
)
from run_friday_daily_best import weekly_portfolio_returns
from run_optimize_monthly import monthly_stats
from run_weekly_best import WeeklyRule


# Most certain (win≈62%, uptrend-hit≈71%, worst-month≈-13%)
BEST_ENTRY = WeeklyRule(
    name="certain_mfi20_pb25_mfi_up",
    style="mean_rev",
    mfi_period=14,
    mfi_buy_max=20.0,
    pb_buy_max=0.25,
    require_mfi_up=True,
    require_pb_up=False,
    require_squeeze=False,
    bw_pct_max=1.0,
    stop_pct=0.10,
    max_hold=3,
    exit_to_mid=False,
    pb_sell_min=0.85,
    mfi_sell_min=80.0,
    trail_pb_drop=0.20,
)

BEST_EXIT = ExitSpec(
    name="oh_trail",
    pb_sell_min=0.85,
    mfi_sell_min=80.0,
    exit_to_mid=False,
    stop_pct=0.10,
    max_hold=3,
    trail_pb_drop=0.20,
)

# More signals, still high certainty (win≈63%, n≈280)
BALANCED_ENTRY = WeeklyRule(
    name="balanced_mfi30_pb25_both",
    style="mean_rev",
    mfi_period=14,
    mfi_buy_max=30.0,
    pb_buy_max=0.25,
    require_mfi_up=True,
    require_pb_up=True,
    require_squeeze=False,
    bw_pct_max=1.0,
    stop_pct=0.08,
    max_hold=3,
    exit_to_mid=True,
    pb_sell_min=0.85,
    mfi_sell_min=70.0,
    trail_pb_drop=9.0,
)

BALANCED_EXIT = ExitSpec(
    name="mid_s8",
    pb_sell_min=0.85,
    mfi_sell_min=70.0,
    exit_to_mid=True,
    stop_pct=0.08,
    max_hold=3,
    trail_pb_drop=9.0,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["certain", "balanced"], default="certain")
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--cache", default="cache_ohlcv_v2")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    entry = BEST_ENTRY if args.mode == "certain" else BALANCED_ENTRY
    ex = BEST_EXIT if args.mode == "certain" else BALANCED_EXIT
    rule = with_exit(entry, ex)

    universe = pd.read_csv("out_top100_universe.csv")
    if args.limit:
        universe = universe.head(args.limit)

    trades: list[dict] = []
    cache = Path(args.cache)
    for _, row in universe.iterrows():
        path = cache / f"{row['yahoo']}.pkl"
        if not path.exists():
            continue
        daily = pd.read_pickle(path)
        daily = daily[daily.index >= pd.Timestamp(args.start)]
        if len(daily) < 200:
            continue
        fri = friday_bb_pack(daily[["Open", "High", "Low", "Close", "Volume"]])
        for t in simulate_with_mask(fri, rule):
            t = dict(t)
            t["ticker"] = row["yahoo"]
            t["name"] = row["name"]
            t["return"] = float(min(max(t["return"], -0.5), 1.0))
            trades.append(t)

    ts = trade_stats(trades)
    ms = monthly_stats(weekly_portfolio_returns(trades, 1))
    print(f"MODE={args.mode}")
    if args.mode == "certain":
        print("BUY: %b<=0.25 & MFI(14)<20 & MFI rising | SELL: %b>=0.85 & MFI>=80 / trail / -10% / 3w")
    else:
        print("BUY: %b<=0.25 & MFI(14)<30 & %b↑ & MFI↑ | SELL: mid-band / overheat / -8% / 3w")
    print(tabulate(ts.items(), headers=["trade", "value"], tablefmt="github"))
    print(tabulate(ms.items(), headers=["month", "value"], tablefmt="github"))
    print(f"uptrend_hit%: {uptrend_hit_rate(trades):.1f}")
    print(f"avg hold weeks: {np.mean([t['bars'] for t in trades]):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
