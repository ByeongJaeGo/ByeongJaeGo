#!/usr/bin/env python3
"""Daily-chart, once-per-week (Friday decision) max-return swing."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tabulate import tabulate

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backtest.features import add_swing_features
from backtest.indicators import money_flow_index
from backtest.optimize_rules import trade_stats
from run_optimize_monthly import monthly_stats
from run_weekly_best import WeeklyRule, simulate_weekly


BEST = WeeklyRule(
    name="friday_daily_mr",
    style="mean_rev",
    mfi_period=14,
    mfi_buy_max=30.0,
    pb_buy_max=0.15,
    stop_pct=0.10,
    max_hold=2,
    exit_to_mid=False,
    pb_sell_min=0.85,
    mfi_sell_min=70.0,
)


def friday_pack(daily: pd.DataFrame) -> pd.DataFrame:
    feat = add_swing_features(daily)
    fri = feat.loc[feat.index.dayofweek == 4].copy()
    fri["pb"] = feat.loc[fri.index, "pb_20"].astype(float)
    fri["mfi_6"] = feat.loc[fri.index, "mfi_6"].astype(float)
    fri["mfi_14"] = feat.loc[fri.index, "mfi_14"].astype(float)
    fri["mfi_9"] = money_flow_index(fri["High"], fri["Low"], fri["Close"], fri["Volume"], 9)
    fri["pb_up"] = fri["pb"] > fri["pb"].shift(1)
    fri["mfi_up"] = fri["mfi_6"] > fri["mfi_6"].shift(1)
    return fri


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
    slot = 1.0 / max_positions
    for a in accepted:
        bars = max(int(a["bars"]), 1)
        per = (1.0 + float(a["return"])) ** (1.0 / bars) - 1.0
        for d in pd.date_range(a["entry_date"], periods=bars, freq="W-FRI"):
            if d in weekly_ret.index:
                weekly_ret.loc[d] += slot * per
    return (1.0 + weekly_ret).cumprod().resample("ME").last().pct_change().dropna()


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

    trades: list[dict] = []
    cache = Path(args.cache)
    for _, row in universe.iterrows():
        path = cache / f"{row['yahoo']}.pkl"
        if not path.exists():
            continue
        daily = pd.read_pickle(path)
        daily = daily[daily.index >= pd.Timestamp(args.start)]
        if len(daily) < 150:
            continue
        fri = friday_pack(daily[["Open", "High", "Low", "Close", "Volume"]])
        for t in simulate_weekly(fri, BEST):
            t = dict(t)
            t["ticker"] = row["yahoo"]
            t["name"] = row["name"]
            t["return"] = float(min(max(t["return"], -0.5), 1.0))
            trades.append(t)

    ts = trade_stats(trades)
    ms = monthly_stats(weekly_portfolio_returns(trades, args.positions))
    print("DAILY chart + Friday-only trading (max profit mode)")
    print("Rule: %b<=0.15 & MFI(14)<30 → next open | stop -10% | max 2 weeks")
    print(tabulate(ts.items(), headers=["trade", "value"], tablefmt="github"))
    print(tabulate(ms.items(), headers=["month", "value"], tablefmt="github"))
    print(f"avg hold weeks: {np.mean([t['bars'] for t in trades]):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
