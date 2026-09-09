#!/usr/bin/env python3
"""
Find the most certain buy/sell rules using ONLY:
  Bollinger Bands, MFI, %b, Bandwidth
that tend to lead into an uptrend.

Cadence: daily chart, Friday decision → next open (once/week).
Certainty = win rate + walk-forward both sides + uptrend hit + stability.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from tabulate import tabulate

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backtest.features import add_swing_features
from backtest.optimize_rules import trade_stats
from run_friday_daily_best import weekly_portfolio_returns
from run_optimize_monthly import monthly_stats
from run_weekly_best import WeeklyRule, entry_mask, simulate_weekly


@dataclass(frozen=True)
class ExitSpec:
    name: str
    pb_sell_min: float
    mfi_sell_min: float
    exit_to_mid: bool
    stop_pct: float
    max_hold: int
    trail_pb_drop: float


def friday_bb_pack(daily: pd.DataFrame) -> pd.DataFrame:
    feat = add_swing_features(daily)
    fri = feat.loc[feat.index.dayofweek == 4].copy()
    fri["pb"] = feat.loc[fri.index, "pb_20"].astype(float)
    fri["pb_8"] = feat.loc[fri.index, "pb_8"].astype(float)
    fri["mfi_6"] = feat.loc[fri.index, "mfi_6"].astype(float)
    fri["mfi_14"] = feat.loc[fri.index, "mfi_14"].astype(float)
    fri["pb_up"] = fri["pb"] > fri["pb"].shift(1)
    fri["mfi_up"] = fri["mfi_14"] > fri["mfi_14"].shift(1)
    fri["mfi6_up"] = fri["mfi_6"] > fri["mfi_6"].shift(1)
    # squeeze releasing: was squeeze recently, bandwidth expanding now
    fri["sq_release"] = (
        fri["squeeze"].rolling(3, min_periods=1).max().astype(bool)
        & fri["bw_expand"].fillna(False)
    )
    return fri


def build_entry_grid() -> list[WeeklyRule]:
    """BB-family-only entries aimed at uptrend continuation / turn."""
    rules: list[WeeklyRule] = []

    # --- A) Oversold turn → uptrend (mean_rev + turn confirms) ---
    for mfi_p, mfi_max, pb_max, bw, turn, hold in itertools.product(
        (6, 14),
        (20, 30),
        (0.15, 0.25),
        ("none", "squeeze", "bw40", "sq_rel"),
        ("none", "pb_up", "mfi_up", "both"),
        (2, 3),
    ):
        sq = bw in ("squeeze", "sq_rel")
        bw_max = 0.40 if bw == "bw40" else 1.0
        pb_up = turn in ("pb_up", "both")
        mfi_up = turn in ("mfi_up", "both")
        name = f"MR_m{mfi_p}<{mfi_max}_pb{pb_max}_{bw}_{turn}_h{hold}"
        rules.append(
            WeeklyRule(
                name=name,
                style="mean_rev",
                mfi_period=mfi_p,
                mfi_buy_max=float(mfi_max),
                pb_buy_max=float(pb_max),
                require_squeeze=sq and bw != "sq_rel",
                bw_pct_max=float(bw_max),
                require_pb_up=pb_up,
                require_mfi_up=mfi_up,
                stop_pct=0.10,
                max_hold=int(hold),
                exit_to_mid=False,
                pb_sell_min=0.85,
                mfi_sell_min=70.0,
            )
        )

    # Dedicated squeeze-release mean-rev (oversold + expand)
    for mfi_p, mfi_max, pb_max, turn, hold in itertools.product(
        (6, 14),
        (20, 30),
        (0.15, 0.25),
        ("pb_up", "mfi_up", "both"),
        (2, 3),
    ):
        rules.append(
            WeeklyRule(
                name=f"REL_m{mfi_p}<{mfi_max}_pb{pb_max}_{turn}_h{hold}",
                style="mean_rev",
                mfi_period=mfi_p,
                mfi_buy_max=float(mfi_max),
                pb_buy_max=float(pb_max),
                require_squeeze=False,
                bw_pct_max=1.0,
                require_pb_up=turn in ("pb_up", "both"),
                require_mfi_up=turn in ("mfi_up", "both"),
                stop_pct=0.10,
                max_hold=int(hold),
                exit_to_mid=False,
                pb_sell_min=0.85,
                mfi_sell_min=70.0,
            )
        )

    # --- B) Squeeze break → uptrend ---
    for mfi_p, mfi_min, pb_min, brk, mfi_up, hold in itertools.product(
        (6, 14),
        (45, 50, 55),
        (0.55, 0.70, 0.80),
        (False, True),
        (False, True),
        (2, 3, 4),
    ):
        rules.append(
            WeeklyRule(
                name=f"SQ_m{mfi_p}>={mfi_min}_pb{pb_min}_brk{int(brk)}_up{int(mfi_up)}_h{hold}",
                style="squeeze_break",
                mfi_period=mfi_p,
                mfi_buy_min=float(mfi_min),
                pb_buy_min=float(pb_min),
                require_break_upper=bool(brk),
                require_mfi_up=bool(mfi_up),
                stop_pct=0.10,
                max_hold=int(hold),
                exit_to_mid=False,
                pb_sell_min=1.05,
                mfi_sell_min=85.0,
                trail_pb_drop=0.20,
            )
        )

    # --- C) Band walk (already rising) ---
    for mfi_p, mfi_min, pb_min, mfi_up, hold in itertools.product(
        (6, 14),
        (50, 55, 60),
        (0.70, 0.80, 0.85),
        (False, True),
        (2, 3, 4),
    ):
        rules.append(
            WeeklyRule(
                name=f"WALK_m{mfi_p}>={mfi_min}_pb{pb_min}_up{int(mfi_up)}_h{hold}",
                style="band_walk",
                mfi_period=mfi_p,
                mfi_buy_min=float(mfi_min),
                pb_buy_min=float(pb_min),
                require_mfi_up=bool(mfi_up),
                stop_pct=0.10,
                max_hold=int(hold),
                exit_to_mid=False,
                pb_sell_min=1.10,
                mfi_sell_min=90.0,
                trail_pb_drop=0.25,
            )
        )

    # Deduplicate by name
    uniq: dict[str, WeeklyRule] = {r.name: r for r in rules}
    return list(uniq.values())


def build_exit_grid() -> list[ExitSpec]:
    exits: list[ExitSpec] = []
    for mid, pb_s, mfi_s, stop, hold, trail in itertools.product(
        (False, True),
        (0.85, 0.95, 1.05),
        (70.0, 80.0, 85.0),
        (0.08, 0.10),
        (2, 3),
        (0.20, 9.0),  # 9.0 = trail off
    ):
        if mid and pb_s >= 1.0:
            continue
        name = f"x_mid{int(mid)}_pb{pb_s}_mfi{mfi_s}_s{stop}_h{hold}_tr{trail}"
        exits.append(
            ExitSpec(
                name=name,
                pb_sell_min=float(pb_s),
                mfi_sell_min=float(mfi_s),
                exit_to_mid=bool(mid),
                stop_pct=float(stop),
                max_hold=int(hold),
                trail_pb_drop=float(trail),
            )
        )
    uniq = {e.name: e for e in exits}
    return list(uniq.values())


def with_exit(rule: WeeklyRule, ex: ExitSpec) -> WeeklyRule:
    return WeeklyRule(
        name=f"{rule.name}__{ex.name}",
        style=rule.style,
        mfi_period=rule.mfi_period,
        mfi_buy_max=rule.mfi_buy_max,
        mfi_buy_min=rule.mfi_buy_min,
        pb_buy_max=rule.pb_buy_max,
        pb_buy_min=rule.pb_buy_min,
        require_squeeze=rule.require_squeeze,
        bw_pct_max=rule.bw_pct_max,
        require_mfi_up=rule.require_mfi_up,
        require_pb_up=rule.require_pb_up,
        require_break_upper=rule.require_break_upper,
        pb_sell_min=ex.pb_sell_min,
        mfi_sell_min=ex.mfi_sell_min,
        exit_to_mid=ex.exit_to_mid,
        stop_pct=ex.stop_pct,
        max_hold=ex.max_hold,
        trail_pb_drop=ex.trail_pb_drop,
    )


def entry_mask_extra(df: pd.DataFrame, rule: WeeklyRule) -> pd.Series:
    """
    Squeeze-release hook + intentional MFI turn confirm:
    when buy uses MFI(6), turn confirm uses MFI(14) rising (cross-horizon).
    when buy uses MFI(14), turn confirm uses MFI(14) rising.
    """
    work = df.copy()
    if rule.require_mfi_up:
        # Cross-horizon: short oversold + medium MFI turning up is more certain
        turn_p = 14 if rule.mfi_period <= 6 else rule.mfi_period
        mfi = work[f"mfi_{turn_p}"]
        work["mfi_up"] = mfi > mfi.shift(1)
    base = entry_mask(work, rule)
    if rule.name.startswith("REL_") or "_sq_rel_" in rule.name:
        return base & work["sq_release"].fillna(False)
    return base


def simulate_with_mask(df: pd.DataFrame, rule: WeeklyRule) -> list[dict]:
    """Like simulate_weekly but uses entry_mask_extra."""
    buys = entry_mask_extra(df, rule).to_numpy()
    # temporarily patch by cloning rule into simulate via local loop
    close = df["Close"].to_numpy(float)
    open_ = df["Open"].to_numpy(float)
    pb = df["pb"].to_numpy(float)
    mfi = df[f"mfi_{rule.mfi_period}"].to_numpy(float)
    mid = df["bb_mid"].to_numpy(float)
    idx = df.index
    n = len(df)
    trades: list[dict] = []
    i = 1
    while i < n - 1:
        if not buys[i]:
            i += 1
            continue
        entry_i = i + 1
        entry_px = open_[entry_i] if np.isfinite(open_[entry_i]) else close[entry_i]
        if not np.isfinite(entry_px) or entry_px <= 0:
            i += 1
            continue
        stop_px = entry_px * (1.0 - rule.stop_pct)
        peak_pb = pb[entry_i] if np.isfinite(pb[entry_i]) else 0.0
        exit_i = None
        reason = "eod"
        for j in range(entry_i, min(entry_i + rule.max_hold, n - 1)):
            if np.isfinite(pb[j]):
                peak_pb = max(peak_pb, pb[j])
            if close[j] < stop_px:
                exit_i, reason = j + 1, "stop"
                break
            sell = False
            if np.isfinite(pb[j]) and np.isfinite(mfi[j]):
                if pb[j] >= rule.pb_sell_min and mfi[j] >= rule.mfi_sell_min:
                    sell, reason = True, "overheat"
                if rule.exit_to_mid and np.isfinite(mid[j]) and close[j] >= mid[j] and j > entry_i:
                    sell, reason = True, "mid"
                if rule.trail_pb_drop < 5 and peak_pb - pb[j] >= rule.trail_pb_drop and peak_pb >= 0.7:
                    sell, reason = True, "trail"
            if sell:
                exit_i = j + 1
                break
        if exit_i is None:
            exit_i = min(entry_i + rule.max_hold, n - 1)
            reason = "timeout" if exit_i < n - 1 else "eod"
        exit_px = open_[exit_i] if np.isfinite(open_[exit_i]) else close[exit_i]
        if exit_i == n - 1:
            exit_px = close[exit_i]
        trades.append(
            {
                "entry_date": idx[entry_i],
                "exit_date": idx[exit_i],
                "entry_px": float(entry_px),
                "exit_px": float(exit_px),
                "return": float(exit_px / entry_px - 1.0),
                "bars": int(exit_i - entry_i),
                "reason": reason,
                "peak_pb": float(peak_pb),
            }
        )
        i = exit_i
    return trades


def uptrend_hit_rate(trades: list[dict], pb_target: float = 0.50) -> float:
    """Share of trades that reached mid/upper band territory (proxy for uptrend)."""
    if not trades:
        return 0.0
    hits = sum(1 for t in trades if t.get("peak_pb", 0) >= pb_target or t["return"] > 0.03)
    return 100.0 * hits / len(trades)


def certainty_score(row: dict) -> float:
    """Higher = more certain uptrend-leading rule (win/stability first)."""
    if row["trades"] < 80:
        return -1e9
    if row["win_rate"] is None or row["avg_return"] is None:
        return -1e9
    if row["avg_return"] <= 0:
        return -1e9
    if row.get("train_wr", 0) < 55 or row.get("test_wr", 0) < 55:
        return -1e9
    if (row.get("train_avg") or 0) <= 0 or (row.get("test_avg") or 0) <= 0:
        return -1e9
    wr = row["win_rate"]
    hit = row.get("uptrend_hit", 0) or 0
    worst = row.get("worst_month_pct") or -50
    n = min(row["trades"], 250) / 250.0
    gap = abs((row.get("train_wr") or 0) - (row.get("test_wr") or 0))
    return (
        wr * 2.0
        + hit * 1.2
        + min(row.get("train_wr") or 0, row.get("test_wr") or 0) * 1.0
        + max(worst, -40) * 0.8
        + n * 8
        + min(row["avg_return"], 6) * 1.0
        - gap * 0.5
    )


def load_frames(cache: Path, universe: pd.DataFrame, start: str, limit: int | None) -> list[tuple[str, str, pd.DataFrame]]:
    rows = universe if limit is None else universe.head(limit)
    out = []
    for _, r in rows.iterrows():
        path = cache / f"{r['yahoo']}.pkl"
        if not path.exists():
            continue
        daily = pd.read_pickle(path)
        daily = daily[daily.index >= pd.Timestamp(start)]
        if len(daily) < 200:
            continue
        fri = friday_bb_pack(daily[["Open", "High", "Low", "Close", "Volume"]])
        if len(fri) < 80:
            continue
        out.append((r["yahoo"], r["name"], fri))
    return out


def eval_rule(frames, rule: WeeklyRule, split: str = "2022-01-01") -> dict:
    trades: list[dict] = []
    for yahoo, name, fri in frames:
        for t in simulate_with_mask(fri, rule):
            t = dict(t)
            t["ticker"] = yahoo
            t["name"] = name
            t["return"] = float(min(max(t["return"], -0.5), 1.0))
            trades.append(t)
    ts = trade_stats(trades)
    ms = monthly_stats(weekly_portfolio_returns(trades, 1))
    split_ts = pd.Timestamp(split)
    train = [t for t in trades if t["entry_date"] < split_ts]
    test = [t for t in trades if t["entry_date"] >= split_ts]
    tr = trade_stats(train)
    te = trade_stats(test)
    row = {
        "name": rule.name,
        "style": rule.style,
        **{f"buy_{k}": v for k, v in asdict(rule).items() if k != "name"},
        "trades": ts["trades"],
        "win_rate": ts["win_rate"],
        "avg_return": ts["avg_return"],
        "median_return": ts["median_return"],
        "worst_trade": ts.get("worst"),
        "uptrend_hit": round(uptrend_hit_rate(trades), 1),
        "avg_month_pct": ms.get("avg_month_pct"),
        "median_month_pct": ms.get("median_month_pct"),
        "pct_months_gt_0": ms.get("pct_months_gt_0"),
        "worst_month_pct": ms.get("worst_month_pct"),
        "best_month_pct": ms.get("best_month_pct"),
        "train_wr": tr["win_rate"],
        "train_avg": tr["avg_return"],
        "train_n": tr["trades"],
        "test_wr": te["win_rate"],
        "test_avg": te["avg_return"],
        "test_n": te["trades"],
    }
    row["certainty"] = certainty_score(row)
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--cache", default="cache_ohlcv_v2")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--phase", choices=["entry", "exit", "both"], default="both")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default="out_bb_uptrend_certain.csv")
    args = ap.parse_args()

    universe = pd.read_csv("out_top100_universe.csv")
    frames = load_frames(Path(args.cache), universe, args.start, args.limit)
    print(f"loaded {len(frames)} tickers")

    entries = build_entry_grid()
    print(f"entry candidates: {len(entries)}")

    def make_entry_rule(er: WeeklyRule) -> WeeklyRule:
        pb_sell = 0.85 if er.style == "mean_rev" else (1.05 if er.style == "squeeze_break" else 1.10)
        mfi_sell = 70.0 if er.style == "mean_rev" else (85.0 if er.style == "squeeze_break" else 90.0)
        trail = 9.0 if er.style == "mean_rev" else 0.20
        return WeeklyRule(
            name=er.name,
            style=er.style,
            mfi_period=er.mfi_period,
            mfi_buy_max=er.mfi_buy_max,
            mfi_buy_min=er.mfi_buy_min,
            pb_buy_max=er.pb_buy_max,
            pb_buy_min=er.pb_buy_min,
            require_squeeze=er.require_squeeze,
            bw_pct_max=er.bw_pct_max,
            require_mfi_up=er.require_mfi_up,
            require_pb_up=er.require_pb_up,
            require_break_upper=er.require_break_upper,
            pb_sell_min=pb_sell,
            mfi_sell_min=mfi_sell,
            exit_to_mid=False,
            stop_pct=0.10,
            max_hold=er.max_hold,
            trail_pb_drop=trail,
        )

    entry_rows = []
    for i, er in enumerate(entries, 1):
        entry_rows.append(eval_rule(frames, make_entry_rule(er)))
        if i % 50 == 0 or i == len(entries):
            print(f"  entry {i}/{len(entries)}")

    edf = pd.DataFrame(entry_rows).sort_values("certainty", ascending=False)
    edf.to_csv("out_bb_uptrend_entries.csv", index=False)
    print("\n=== TOP ENTRIES (certainty) ===")
    cols = [
        "name", "style", "trades", "win_rate", "avg_return", "uptrend_hit",
        "avg_month_pct", "pct_months_gt_0", "worst_month_pct",
        "train_wr", "test_wr", "certainty",
    ]
    print(tabulate(edf[cols].head(args.top), headers="keys", tablefmt="github", showindex=False, floatfmt=".2f"))

    if args.phase == "entry":
        edf.to_csv(args.out, index=False)
        return 0

    top_entries = edf.head(10)["name"].tolist()
    entry_map = {r.name: r for r in entries}
    exits = build_exit_grid()
    print(f"\nexit candidates: {len(exits)} × top {len(top_entries)} entries")

    combo_rows = []
    combo_rules = [with_exit(entry_map[ename], ex) for ename in top_entries for ex in exits]
    for i, rule in enumerate(combo_rules, 1):
        combo_rows.append(eval_rule(frames, rule))
        if i % 100 == 0 or i == len(combo_rules):
            print(f"  combo {i}/{len(combo_rules)}")

    cdf = pd.DataFrame(combo_rows).sort_values("certainty", ascending=False)
    cdf.to_csv(args.out, index=False)
    print("\n=== TOP BUY+SELL COMBOS (certainty) ===")
    print(tabulate(cdf[cols].head(args.top), headers="keys", tablefmt="github", showindex=False, floatfmt=".2f"))

    best = cdf.iloc[0]
    print("\n=== MOST CERTAIN RULE ===")
    print(best[["name", "style", "trades", "win_rate", "avg_return", "uptrend_hit",
                "avg_month_pct", "worst_month_pct", "train_wr", "test_wr", "certainty"]].to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
