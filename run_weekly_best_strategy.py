#!/usr/bin/env python3
"""Run the recommended once-per-week squeeze-break swing method."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tabulate import tabulate

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backtest.optimize_rules import trade_stats
from run_optimize_monthly import monthly_stats
from run_weekly_best import (
    WeeklyRule,
    add_weekly_features,
    simulate_weekly,
    to_weekly,
)


BEST = WeeklyRule(
    name="weekly_squeeze_break",
    style="squeeze_break",
    mfi_period=6,
    mfi_buy_min=50.0,
    pb_buy_min=0.8,
    require_break_upper=False,
    require_mfi_up=True,
    stop_pct=0.10,
    max_hold=8,
    exit_to_mid=False,
    pb_sell_min=1.05,
    mfi_sell_min=85.0,
    trail_pb_drop=0.2,
)


def weekly_portfolio_returns(trades: list[dict], max_positions: int = 1) -> pd.Series:
    if not trades:
        return pd.Series(dtype=float)
    tdf = pd.DataFrame(trades).sort_values("entry_date")
    open_pos: list[dict] = []
    accepted: list[dict] = []
    for _, t in tdf.iterrows():
        open_pos = [p for p in open_pos if p["exit_date"] > t["entry_date"]]
        if len(open_pos) >= max_positions:
            continue
        accepted.append(t.to_dict())
        open_pos.append(t.to_dict())
    if not accepted:
        return pd.Series(dtype=float)

    start = min(a["entry_date"] for a in accepted)
    end = max(a["exit_date"] for a in accepted)
    week_index = pd.date_range(start, end, freq="W-FRI")
    weekly_ret = pd.Series(0.0, index=week_index)
    slot_w = 1.0 / max_positions
    for a in accepted:
        bars = max(int(a["bars"]), 1)
        per_week = (1.0 + float(a["return"])) ** (1.0 / bars) - 1.0
        held = pd.date_range(a["entry_date"], periods=bars, freq="W-FRI")
        for d in held:
            if d in weekly_ret.index:
                weekly_ret.loc[d] += slot_w * per_week
    equity = (1.0 + weekly_ret).cumprod()
    return equity.resample("ME").last().pct_change().dropna()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2019-01-01")
    p.add_argument("--positions", type=int, default=1)
    p.add_argument("--cache", default="cache_ohlcv_v2")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    universe = pd.read_csv("out_top100_universe.csv")
    if args.limit:
        universe = universe.head(args.limit)
    cache = Path(args.cache)

    trades: list[dict] = []
    for _, row in universe.iterrows():
        path = cache / f"{row['yahoo']}.pkl"
        if not path.exists():
            continue
        daily = pd.read_pickle(path)
        daily = daily[daily.index >= pd.Timestamp(args.start)]
        if len(daily) < 120:
            continue
        weekly = add_weekly_features(to_weekly(daily))
        for t in simulate_weekly(weekly, BEST):
            t = dict(t)
            t["ticker"] = row["yahoo"]
            t["name"] = row["name"]
            t["return"] = float(min(max(t["return"], -0.5), 1.0))
            trades.append(t)

    ts = trade_stats(trades)
    monthly = weekly_portfolio_returns(trades, args.positions)
    ms = monthly_stats(monthly)
    print("WEEKLY BEST: squeeze → expand + %b>=0.8 + MFI(6)>=50 rising")
    print(tabulate(ts.items(), headers=["trade", "value"], tablefmt="github"))
    print(tabulate(ms.items(), headers=["month", "value"], tablefmt="github"))
    print(f"avg hold weeks: {np.mean([t['bars'] for t in trades]):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
