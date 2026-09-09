#!/usr/bin/env python3
"""
Backtest on current KOSPI+KOSDAQ trading-value top 100.

Entry (daily bar):
  - MFI(6) and MFI(13) below threshold  (request: <0; standard MFI can't go
    negative, so default operable threshold is 20 = near-zero oversold)
  - Weekly MFI(6) < 60
  - At least 2 of %%b(8), %%b(14), %%b(20) are < 0

Exit:
  - Weekly MFI(6) >= 60, or daily overheat rollover, or setup-low stop, or max hold

Examples:
  python3 run_top100_backtest.py
  python3 run_top100_backtest.py --mfi-max 0
  python3 run_top100_backtest.py --limit 30 --start 2020-01-01
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tabulate import tabulate

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backtest.data import download_ohlcv
from backtest.engine import aggregate_summaries, run_top100_backtest
from backtest.indicators import add_v2_indicators
from backtest.strategy import Top100Params
from backtest.universe import fetch_top_trading_stocks


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Top100 MFI/%b swing backtest")
    p.add_argument("--n", type=int, default=100, help="Universe size (default 100)")
    p.add_argument("--limit", type=int, default=None, help="Only first N of the universe")
    p.add_argument("--start", default="2019-01-01")
    p.add_argument("--end", default=None)
    p.add_argument(
        "--mfi-max",
        type=float,
        default=20.0,
        help="Daily MFI(6)&MFI(13) must be below this (0 = literal floor; default 20)",
    )
    p.add_argument("--weekly-mfi-max", type=float, default=60.0)
    p.add_argument("--min-pb", type=int, default=2, help="Min count of %%b periods < 0")
    p.add_argument("--max-hold", type=int, default=60)
    p.add_argument("--no-stop", action="store_true")
    p.add_argument("--trades", action="store_true")
    p.add_argument(
        "--also-literal-zero",
        action="store_true",
        help="Also print aggregate for literal MFI<=0 (usually ~0 trades)",
    )
    p.add_argument(
        "--universe-csv",
        default="out_top100_universe.csv",
        help="Where to save the fetched universe",
    )
    p.add_argument(
        "--results-csv",
        default="out_top100_results.csv",
        help="Where to save per-ticker results",
    )
    return p.parse_args()


def run_universe(
    universe,
    *,
    start: str,
    end: str | None,
    params: Top100Params,
    max_hold: int,
    show_trades: bool,
) -> list[dict]:
    rows: list[dict] = []
    for i, row in universe.iterrows():
        yahoo = row["yahoo"]
        name = row["name"]
        try:
            ohlcv = download_ohlcv(yahoo, start=start, end=end)
            data = add_v2_indicators(ohlcv)
            result = run_top100_backtest(
                data,
                ticker=yahoo,
                params=params,
                max_hold_bars=max_hold,
                name=name,
            )
            summary = result.summary()
            summary["market"] = row["market"]
            rows.append(summary)

            n_tr = summary["trades"]
            wr = summary["win_rate_pct"]
            ar = summary["avg_return_pct"]
            tr = summary["total_return_pct"]
            print(
                f"[{i+1}/{len(universe)}] {yahoo} {name}: "
                f"trades={n_tr} win={wr}% avg={ar}% total={tr}%"
            )

            if show_trades and result.closed_trades:
                trade_rows = [
                    [
                        t.entry_date.date(),
                        f"{t.entry_price:.0f}",
                        t.exit_date.date() if t.exit_date is not None else "",
                        f"{t.exit_price:.0f}" if t.exit_price is not None else "",
                        f"{t.return_pct:.2f}%" if t.return_pct is not None else "",
                        t.bars_held,
                        t.reason,
                    ]
                    for t in result.closed_trades
                ]
                print(
                    tabulate(
                        trade_rows,
                        headers=["Entry", "Px", "Exit", "Px", "Ret", "Bars", "Why"],
                        tablefmt="simple",
                    )
                )
        except Exception as exc:  # noqa: BLE001
            print(f"[ERROR] {yahoo} {name}: {exc}", file=sys.stderr)
            rows.append(
                {
                    "ticker": yahoo,
                    "name": name,
                    "market": row["market"],
                    "trades": 0,
                    "wins": 0,
                    "losses": 0,
                    "win_rate_pct": None,
                    "avg_return_pct": None,
                    "total_return_pct": None,
                    "error": str(exc),
                }
            )
    return rows


def print_report(title: str, rows: list[dict]) -> None:
    print(f"\n=== {title}: per-ticker (trades > 0) ===")
    traded = [r for r in rows if r.get("trades")]
    traded = sorted(traded, key=lambda r: r.get("total_return_pct") or -999, reverse=True)
    display_cols = [
        "ticker",
        "name",
        "market",
        "trades",
        "win_rate_pct",
        "avg_return_pct",
        "total_return_pct",
        "sum_return_pct",
        "max_drawdown_pct",
        "avg_bars_held",
    ]
    table = [[r.get(c) for c in display_cols] for r in traded]
    print(tabulate(table, headers=display_cols, tablefmt="github"))

    agg = aggregate_summaries(rows)
    print(f"\n=== {title}: AGGREGATE ===")
    print(tabulate([[k, v] for k, v in agg.items()], headers=["metric", "value"], tablefmt="github"))


def main() -> int:
    args = parse_args()

    print("Fetching KOSPI+KOSDAQ trading-value top universe from Naver...")
    universe = fetch_top_trading_stocks(n=args.n)
    if args.limit:
        universe = universe.head(args.limit).reset_index(drop=True)
    universe.to_csv(args.universe_csv, index=False)
    print(f"Universe {len(universe)} names → {args.universe_csv}")
    print(universe[["code", "name", "market"]].head(10).to_string(index=False))
    print("...")

    print(
        "\nRules:\n"
        f"  Daily MFI(6) & MFI(13) < {args.mfi_max} "
        f"{'(literal floor ≤0)' if args.mfi_max <= 0 else '(near-zero oversold; request <0 is impossible on standard MFI)'}\n"
        f"  Weekly MFI(6) < {args.weekly_mfi_max}\n"
        f"  %b(8/14/20): at least {args.min_pb} periods < 0\n"
        f"  Period: {args.start} → {args.end or 'today'} | max_hold={args.max_hold}\n"
    )

    params = Top100Params(
        daily_mfi_max=args.mfi_max,
        weekly_mfi_max=args.weekly_mfi_max,
        min_pb_lt0=args.min_pb,
        stop_below_entry_low=not args.no_stop,
    )

    rows = run_universe(
        universe,
        start=args.start,
        end=args.end,
        params=params,
        max_hold=args.max_hold,
        show_trades=args.trades,
    )

    import pandas as pd

    pd.DataFrame(rows).to_csv(args.results_csv, index=False)
    print(f"\nSaved {args.results_csv}")
    print_report(f"MFI_max={args.mfi_max}", rows)

    if args.also_literal_zero and args.mfi_max != 0:
        print("\n--- Literal MFI<=0 check (request text) ---")
        lit = Top100Params(
            daily_mfi_max=0.0,
            weekly_mfi_max=args.weekly_mfi_max,
            min_pb_lt0=args.min_pb,
            stop_below_entry_low=not args.no_stop,
        )
        lit_rows = run_universe(
            universe,
            start=args.start,
            end=args.end,
            params=lit,
            max_hold=args.max_hold,
            show_trades=False,
        )
        print_report("MFI_max=0 (literal)", lit_rows)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
