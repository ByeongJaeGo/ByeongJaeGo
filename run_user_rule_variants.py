#!/usr/bin/env python3
"""
Backtest the user's exact entry + nested variants (full return report).

Exact BUY:
  %b(20) < 0
  MFI(6)  <= 20
  MFI(16) <= 20
  Bandwidth at 60-day low
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tabulate import tabulate

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backtest.optimize_rules import portfolio_monthly_returns, trade_stats
from run_friday_daily_best import weekly_portfolio_returns
from run_optimize_monthly import monthly_stats
from run_user_rule_pb_mfi_bw60 import ExitCfg, add_user_features, simulate


VARIANTS = {
    "exact_pb_m6_m16_bw60": lambda d: (
        (d["pb"] < 0) & (d["mfi_6"] <= 20) & (d["mfi_16"] <= 20) & d["bw_at_60low"].fillna(False)
    ),
    "pb_m6_m16": lambda d: (d["pb"] < 0) & (d["mfi_6"] <= 20) & (d["mfi_16"] <= 20),
    "pb_m6": lambda d: (d["pb"] < 0) & (d["mfi_6"] <= 20),
    "pb_m16": lambda d: (d["pb"] < 0) & (d["mfi_16"] <= 20),
    "pb_m6_m16_bw20low": lambda d: (
        (d["pb"] < 0)
        & (d["mfi_6"] <= 20)
        & (d["mfi_16"] <= 20)
        & (d["bandwidth"] <= d["bw_low20"] * 1.0001).fillna(False)
    ),
    "pb_m6_m16_bw60_near5": lambda d: (
        (d["pb"] < 0)
        & (d["mfi_6"] <= 20)
        & (d["mfi_16"] <= 20)
        & (d["bandwidth"] <= d["bw_low60"] * 1.05).fillna(False)
    ),
    "pb_m6_m16_bw60_near10": lambda d: (
        (d["pb"] < 0)
        & (d["mfi_6"] <= 20)
        & (d["mfi_16"] <= 20)
        & (d["bandwidth"] <= d["bw_low60"] * 1.10).fillna(False)
    ),
    "pb_neg_only": lambda d: d["pb"] < 0,
    "m6_m16_bw60": lambda d: (
        (d["mfi_6"] <= 20) & (d["mfi_16"] <= 20) & d["bw_at_60low"].fillna(False)
    ),
}


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    out = add_user_features(df)
    out["bw_low20"] = out["bandwidth"].rolling(20, min_periods=20).min()
    return out


EXITS = [
    ExitCfg("mid_h20_s7", 0.07, 20, 0.85, 70.0, True),
    ExitCfg("oh_h20_s10", 0.10, 20, 0.85, 80.0, False),
    ExitCfg("timeout_h10_s10", 0.10, 10, 9.0, 200.0, False),
    ExitCfg("timeout_h20_s10", 0.10, 20, 9.0, 200.0, False),
    ExitCfg("fri_oh_h3_s10", 0.10, 3, 0.85, 80.0, False),
    ExitCfg("fri_mid_h3_s8", 0.08, 3, 0.85, 70.0, True),
]


def load(cache: Path, universe: pd.DataFrame, start: str, limit: int | None):
    rows = universe if limit is None else universe.head(limit)
    daily, friday = [], []
    for _, r in rows.iterrows():
        path = cache / f"{r['yahoo']}.pkl"
        if not path.exists():
            continue
        raw = pd.read_pickle(path)
        raw = raw[raw.index >= pd.Timestamp(start)]
        if len(raw) < 120:
            continue
        feat = enrich(raw[["Open", "High", "Low", "Close", "Volume"]])
        daily.append((r["yahoo"], r["name"], feat))
        fri = feat.loc[feat.index.dayofweek == 4].copy()
        if len(fri) >= 40:
            friday.append((r["yahoo"], r["name"], fri))
    return daily, friday


def eval_variant(frames, mask_fn, ex: ExitCfg, mode: str, positions: int) -> dict:
    trades = []
    n_sig = 0
    for yahoo, name, df in frames:
        buys = mask_fn(df).fillna(False)
        n_sig += int(buys.sum())
        for t in simulate(df, buys, ex):
            t = dict(t)
            t["ticker"] = yahoo
            t["name"] = name
            trades.append(t)
    ts = trade_stats(trades)
    if mode == "friday":
        monthly = weekly_portfolio_returns(trades, positions)
    else:
        monthly = portfolio_monthly_returns(trades, max_positions=positions)
    ms = monthly_stats(monthly)
    split = pd.Timestamp("2022-01-01")
    tr = trade_stats([t for t in trades if t["entry_date"] < split])
    te = trade_stats([t for t in trades if t["entry_date"] >= split])
    return {
        "mode": mode,
        "entry": None,  # filled by caller
        "exit": ex.name,
        "signals": n_sig,
        "trades": ts["trades"],
        "win_rate": ts["win_rate"],
        "avg_return": ts["avg_return"],
        "median_return": ts["median_return"],
        "avg_month_pct": ms.get("avg_month_pct"),
        "median_month_pct": ms.get("median_month_pct"),
        "pct_months_gt_0": ms.get("pct_months_gt_0"),
        "best_month_pct": ms.get("best_month_pct"),
        "worst_month_pct": ms.get("worst_month_pct"),
        "train_n": tr["trades"],
        "train_wr": tr["win_rate"],
        "train_avg": tr["avg_return"],
        "test_n": te["trades"],
        "test_wr": te["win_rate"],
        "test_avg": te["avg_return"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--cache", default="cache_ohlcv_v2")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--positions", type=int, default=1)
    ap.add_argument("--out", default="out_user_rule_variants.csv")
    args = ap.parse_args()

    universe = pd.read_csv("out_top100_universe.csv")
    daily, friday = load(Path(args.cache), universe, args.start, args.limit)
    print(f"loaded daily={len(daily)} friday={len(friday)}")

    rows = []
    for vname, fn in VARIANTS.items():
        for mode, frames in (("daily", daily), ("friday", friday)):
            exits = [e for e in EXITS if (e.name.startswith("fri_") == (mode == "friday")) or (mode == "daily" and not e.name.startswith("fri_"))]
            if mode == "friday":
                exits = [e for e in EXITS if e.name.startswith("fri_") or e.max_hold <= 3]
            for ex in exits:
                row = eval_variant(frames, fn, ex, mode, args.positions)
                row["entry"] = vname
                rows.append(row)
                print(f"done {mode} {vname} {ex.name} trades={row['trades']}")

    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)

    cols = [
        "mode", "entry", "exit", "signals", "trades", "win_rate", "avg_return",
        "avg_month_pct", "median_month_pct", "pct_months_gt_0",
        "best_month_pct", "worst_month_pct", "train_wr", "train_avg", "test_wr", "test_avg",
    ]

    print("\n========== EXACT RULE ONLY ==========")
    exact = out[out["entry"] == "exact_pb_m6_m16_bw60"]
    print(tabulate(exact[cols], headers="keys", tablefmt="github", showindex=False, floatfmt=".2f"))

    print("\n========== ALL VARIANTS (sorted by avg monthly, trades>=20) ==========")
    usable = out[(out["trades"] >= 20) & out["avg_month_pct"].notna()].sort_values(
        "avg_month_pct", ascending=False
    )
    print(tabulate(usable[cols].head(25), headers="keys", tablefmt="github", showindex=False, floatfmt=".2f"))

    print("\n========== BEST PER ENTRY (daily, best exit by avg month) ==========")
    daily_u = out[(out["mode"] == "daily") & (out["trades"] >= 5) & out["avg_month_pct"].notna()]
    if len(daily_u):
        best = daily_u.sort_values("avg_month_pct", ascending=False).groupby("entry", as_index=False).head(1)
        print(tabulate(best[cols], headers="keys", tablefmt="github", showindex=False, floatfmt=".2f"))

    print(f"\nsaved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
