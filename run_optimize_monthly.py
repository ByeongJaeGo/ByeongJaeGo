#!/usr/bin/env python3
"""
Search best swing rules using MFI + %b + Bandwidth + Bollinger Bands.

Optimizes for monthly return statistics on KOSPI/KOSDAQ trading-value top N.
Target reference: +20%/month (reported honestly vs achievable).

Example:
  python3 run_optimize_monthly.py --limit 40 --max-positions 5
  python3 run_optimize_monthly.py --limit 100 --top 15
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tabulate import tabulate

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backtest.data import download_ohlcv
from backtest.features import add_swing_features
from backtest.optimize_rules import (
    RuleSet,
    portfolio_monthly_returns,
    simulate_trades,
    trade_stats,
)
from backtest.universe import fetch_top_trading_stocks


def build_rule_grid() -> list[RuleSet]:
    """Focused ~400-rule grid: mean-rev / squeeze-break / band-walk."""
    rules: list[RuleSet] = []

    for mfi_p, mfi_max, pb_max, pb_n, filt, confirm, hold, mid_exit in itertools.product(
        (6, 14),
        (20, 30),
        (0.0, 0.15),
        (0, 2),
        ("none", "squeeze", "bw40", "week60"),
        ("none", "pb_up", "both"),
        (12, 20),
        (True, False),
    ):
        if pb_n == 2 and pb_max > 0.05:
            continue
        sq = filt == "squeeze"
        bw = 0.40 if filt == "bw40" else 1.0
        wmax = 60.0 if filt == "week60" else 100.0
        pb_up = confirm in ("pb_up", "both")
        mfi_up = confirm == "both"
        name = (
            f"MR_mfi{mfi_p}<{mfi_max}_pb<={pb_max}_n{pb_n}"
            f"_{filt}_{confirm}_h{hold}_mid{int(mid_exit)}"
        )
        rules.append(
            RuleSet(
                name=name,
                style="mean_rev",
                mfi_period=mfi_p,
                mfi_buy_max=float(mfi_max),
                pb_buy_max=float(pb_max),
                pb_periods_lt0_min=int(pb_n),
                require_squeeze=sq,
                bw_pct_max=float(bw),
                require_pb_up=pb_up,
                require_mfi_up=mfi_up,
                weekly_mfi_max=float(wmax),
                stop_pct=0.07,
                max_hold=int(hold),
                exit_to_mid=bool(mid_exit),
                pb_sell_min=0.85,
                mfi_sell_min=70.0 if mfi_p <= 13 else 80.0,
            )
        )

    for mfi_p, mfi_min, brk, mfi_up, hold in itertools.product(
        (6, 14), (50, 55), (True, False), (True, False), (10, 15)
    ):
        name = f"SQ_mfi{mfi_p}>={mfi_min}_brk{int(brk)}_mup{int(mfi_up)}_h{hold}"
        rules.append(
            RuleSet(
                name=name,
                style="squeeze_break",
                mfi_period=mfi_p,
                mfi_buy_min=float(mfi_min),
                require_break_upper=bool(brk),
                require_mfi_up=bool(mfi_up),
                stop_pct=0.08,
                max_hold=int(hold),
                pb_sell_min=1.05,
                mfi_sell_min=85.0,
                trail_pb_drop=0.2,
            )
        )

    for mfi_p, mfi_min, pb_min, mfi_up, hold in itertools.product(
        (6, 14), (55, 60), (0.75, 0.85), (True, False), (12, 20)
    ):
        name = f"WALK_mfi{mfi_p}>={mfi_min}_pb>={pb_min}_mup{int(mfi_up)}_h{hold}"
        rules.append(
            RuleSet(
                name=name,
                style="band_walk",
                mfi_period=mfi_p,
                mfi_buy_min=float(mfi_min),
                pb_buy_max=float(pb_min),
                require_mfi_up=bool(mfi_up),
                stop_pct=0.10,
                max_hold=int(hold),
                pb_sell_min=1.1,
                mfi_sell_min=90.0,
                trail_pb_drop=0.25,
            )
        )

    return rules


def monthly_stats(monthly: pd.Series) -> dict:
    if monthly is None or len(monthly) == 0:
        return {
            "months": 0,
            "avg_month_pct": None,
            "median_month_pct": None,
            "pct_months_ge_20": None,
            "pct_months_gt_0": None,
            "best_month_pct": None,
            "worst_month_pct": None,
            "months_ge_20": 0,
        }
    m = monthly.astype(float)
    return {
        "months": int(len(m)),
        "avg_month_pct": round(float(m.mean() * 100), 2),
        "median_month_pct": round(float(m.median() * 100), 2),
        "pct_months_ge_20": round(float((m >= 0.20).mean() * 100), 1),
        "pct_months_gt_0": round(float((m > 0).mean() * 100), 1),
        "best_month_pct": round(float(m.max() * 100), 2),
        "worst_month_pct": round(float(m.min() * 100), 2),
        "months_ge_20": int((m >= 0.20).sum()),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--start", default="2019-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--max-positions", type=int, default=3)
    p.add_argument("--also-pos", type=str, default="1,5", help="Extra position sizes to score, comma-separated")
    p.add_argument("--cache", default="cache_ohlcv_v2")
    p.add_argument("--top", type=int, default=20)
    p.add_argument("--max-rules", type=int, default=None, help="Debug: cap rule count")
    p.add_argument("--out", default="out_optimize_monthly.csv")
    return p.parse_args()


def load_features(universe: pd.DataFrame, start: str, end: str | None, cache_dir: Path) -> dict[str, pd.DataFrame]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, pd.DataFrame] = {}
    for i, row in universe.iterrows():
        yahoo = row["yahoo"]
        path = cache_dir / f"{yahoo}.pkl"
        try:
            if path.exists():
                df = pd.read_pickle(path)
            else:
                raw = download_ohlcv(yahoo, start=start, end=end)
                df = add_swing_features(raw)
                df.to_pickle(path)
            # ensure date filter
            if start:
                df = df[df.index >= pd.Timestamp(start)]
            if end:
                df = df[df.index <= pd.Timestamp(end)]
            if len(df) > 60:
                out[yahoo] = df
                print(f"[{i+1}/{len(universe)}] cached {yahoo} {row['name']} bars={len(df)}")
            else:
                print(f"[{i+1}/{len(universe)}] skip short {yahoo}")
        except Exception as exc:  # noqa: BLE001
            print(f"[ERROR] {yahoo}: {exc}", file=sys.stderr)
    return out


def main() -> int:
    args = parse_args()
    t0 = time.time()

    print("Fetching universe...")
    if Path("out_top100_universe.csv").exists() and args.n == 100 and not args.limit:
        universe = pd.read_csv("out_top100_universe.csv")
        print(f"Loaded saved universe ({len(universe)})")
    else:
        universe = fetch_top_trading_stocks(n=args.n)
        universe.to_csv("out_top100_universe.csv", index=False)
    if args.limit:
        universe = universe.head(args.limit).reset_index(drop=True)

    data = load_features(universe, args.start, args.end, Path(args.cache))
    print(f"Loaded {len(data)} tickers in {time.time()-t0:.1f}s")

    rules = build_rule_grid()
    if args.max_rules:
        rules = rules[: args.max_rules]
    print(f"Evaluating {len(rules)} rule sets...")

    name_map = dict(zip(universe["yahoo"], universe["name"]))
    rows = []
    for ri, rule in enumerate(rules):
        all_trades: list[dict] = []
        for yahoo, df in data.items():
            trades = simulate_trades(df, rule)
            for t in trades:
                t = dict(t)
                t["ticker"] = yahoo
                t["name"] = name_map.get(yahoo, yahoo)
                all_trades.append(t)

        ts = trade_stats(all_trades)
        pos_sizes = sorted({args.max_positions, *[int(x) for x in args.also_pos.split(",") if x.strip()]})
        best_score = -1e9
        best_ms = monthly_stats(pd.Series(dtype=float))
        best_pos = args.max_positions
        for pos in pos_sizes:
            monthly = portfolio_monthly_returns(
                all_trades, max_positions=pos, start=args.start, end=args.end
            )
            ms = monthly_stats(monthly)
            score = -1e9
            if ts["trades"] >= 30 and ms["months"] >= 12 and ms["avg_month_pct"] is not None:
                score = (
                    ms["avg_month_pct"] * 2.0
                    + (ms["pct_months_ge_20"] or 0) * 0.8
                    + (ms["median_month_pct"] or 0) * 1.5
                    + (ts["win_rate"] or 0) * 0.05
                    + min(ts["trades"], 200) * 0.01
                )
                if ms["worst_month_pct"] is not None and ms["worst_month_pct"] < -35:
                    score -= 25
            if score > best_score:
                best_score, best_ms, best_pos = score, ms, pos

        rows.append(
            {
                "rule": rule.name,
                "style": rule.style,
                "max_positions": best_pos,
                "score": round(best_score, 2),
                **{f"t_{k}": v for k, v in ts.items()},
                **{f"m_{k}": v for k, v in best_ms.items()},
                "mfi_period": rule.mfi_period,
                "mfi_buy_max": rule.mfi_buy_max,
                "mfi_buy_min": rule.mfi_buy_min,
                "pb_buy_max": rule.pb_buy_max,
                "pb_n": rule.pb_periods_lt0_min,
                "squeeze": rule.require_squeeze,
                "bw_pct_max": rule.bw_pct_max,
                "pb_up": rule.require_pb_up,
                "mfi_up": rule.require_mfi_up,
                "weekly_mfi_max": rule.weekly_mfi_max,
                "stop_pct": rule.stop_pct,
                "max_hold": rule.max_hold,
                "exit_to_mid": rule.exit_to_mid,
                "break_upper": rule.require_break_upper,
            }
        )
        if (ri + 1) % 50 == 0 or ri == 0:
            print(f"  rules {ri+1}/{len(rules)} best_score_so_far={max(r['score'] for r in rows):.1f}")

    res = pd.DataFrame(rows).sort_values("score", ascending=False)
    res.to_csv(args.out, index=False)
    print(f"\nSaved {args.out}")

    top = res.head(args.top)
    cols = [
        "style",
        "max_positions",
        "t_trades",
        "t_win_rate",
        "t_avg_return",
        "m_avg_month_pct",
        "m_median_month_pct",
        "m_pct_months_ge_20",
        "m_months_ge_20",
        "m_best_month_pct",
        "m_worst_month_pct",
        "score",
        "rule",
    ]
    print("\n=== TOP RULES ===")
    print(tabulate(top[cols].values, headers=cols, tablefmt="github", floatfmt=".2f"))

    best = res.iloc[0]
    print("\n=== BEST RULE DETAIL ===")
    print(best[["rule", "style", "score", "t_trades", "t_win_rate", "t_avg_return",
                "m_avg_month_pct", "m_median_month_pct", "m_pct_months_ge_20",
                "m_best_month_pct", "m_worst_month_pct"]].to_string())

    # Feasibility note vs 20%/mo
    print("\n=== vs 20%/month target ===")
    print(
        f"Best avg month: {best['m_avg_month_pct']}% | "
        f"months >=20%: {best['m_pct_months_ge_20']}% "
        f"({best['m_months_ge_20']}/{best['m_months']})"
    )
    hit = res[res["m_avg_month_pct"] >= 20]
    print(f"Rules with avg month >=20%: {len(hit)} / {len(res)}")

    summary = {
        "tickers": len(data),
        "rules_tested": len(res),
        "max_positions": args.max_positions,
        "best_rule": best["rule"],
        "best_avg_month_pct": best["m_avg_month_pct"],
        "best_pct_months_ge_20": best["m_pct_months_ge_20"],
        "rules_avg_month_ge_20": int(len(hit)),
    }
    Path("out_optimize_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Elapsed {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
