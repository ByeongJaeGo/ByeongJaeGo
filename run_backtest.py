#!/usr/bin/env python3
"""
CLI: backtest MFI + Bollinger %b mean-reversion swing on daily bars.

Examples:
  python run_backtest.py 005930 035420.KQ AAPL
  python run_backtest.py 005930 --start 2020-01-01 --no-trend-filter
  python run_backtest.py 005930 --trades
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tabulate import tabulate

# Allow running without install
sys.path.insert(0, str(Path(__file__).resolve().parent))

from backtest import StrategyParams, add_indicators, download_ohlcv, run_backtest


DEFAULT_TICKERS = [
    "005930.KS",  # Samsung Electronics
    "000660.KS",  # SK hynix
    "035420.KS",  # NAVER
    "005380.KS",  # Hyundai Motor
    "AAPL",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Backtest daily swing: buy undervalued (%b+MFI), sell overheated."
    )
    p.add_argument(
        "tickers",
        nargs="*",
        default=DEFAULT_TICKERS,
        help="Tickers (KR 6-digit → .KS). Default: sample KR megacaps + AAPL",
    )
    p.add_argument("--start", default="2018-01-01", help="Start date YYYY-MM-DD")
    p.add_argument("--end", default=None, help="End date YYYY-MM-DD")
    p.add_argument("--buy-pb", type=float, default=0.0, help="Buy if %%b <= this")
    p.add_argument("--buy-mfi", type=float, default=20.0, help="Buy if MFI <= this")
    p.add_argument("--sell-pb", type=float, default=0.85, help="Sell zone %%b >=")
    p.add_argument("--sell-mfi", type=float, default=80.0, help="Sell zone MFI >=")
    p.add_argument("--max-hold", type=int, default=60, help="Max holding days")
    p.add_argument(
        "--no-trend-filter",
        action="store_true",
        help="Allow buys in hard downtrends",
    )
    p.add_argument(
        "--no-stop",
        action="store_true",
        help="Disable stop under setup-bar low (close-based)",
    )
    p.add_argument(
        "--relaxed",
        action="store_true",
        help="Wider zone: buy %%b<=0.2 & MFI<=30, sell %%b>=0.8 & MFI>=70",
    )
    p.add_argument("--trades", action="store_true", help="Print each trade")
    p.add_argument(
        "--export",
        type=str,
        default=None,
        help="Export per-ticker signal CSV directory",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    buy_pb, buy_mfi = args.buy_pb, args.buy_mfi
    sell_pb, sell_mfi = args.sell_pb, args.sell_mfi
    if args.relaxed:
        buy_pb, buy_mfi = 0.2, 30.0
        sell_pb, sell_mfi = 0.8, 70.0

    params = StrategyParams(
        buy_percent_b=buy_pb,
        buy_mfi=buy_mfi,
        sell_percent_b=sell_pb,
        sell_mfi=sell_mfi,
        require_trend_filter=not args.no_trend_filter,
        stop_below_entry_low=not args.no_stop,
    )

    print("Strategy: MFI + Bollinger %b daily swing (mean reversion)")
    print(
        f"  Buy:  %b<={params.buy_percent_b} & MFI<={params.buy_mfi} + reversal"
        f" | trend_filter={params.require_trend_filter}"
    )
    print(
        f"  Sell: %b>={params.sell_percent_b} & MFI>={params.sell_mfi} + rollover"
        f" | stop={params.stop_below_entry_low} | max_hold={args.max_hold}"
    )
    print(f"  Period: {args.start} → {args.end or 'today'}\n")

    rows = []
    export_dir = Path(args.export) if args.export else None
    if export_dir:
        export_dir.mkdir(parents=True, exist_ok=True)

    for ticker in args.tickers:
        try:
            ohlcv = download_ohlcv(ticker, start=args.start, end=args.end)
            data = add_indicators(ohlcv)
            result = run_backtest(data, ticker=ticker, params=params, max_hold_bars=args.max_hold)
            summary = result.summary()
            rows.append(summary)

            if args.trades and result.closed_trades:
                print(f"\n=== Trades: {ticker} ===")
                trade_rows = [
                    [
                        t.entry_date.date(),
                        f"{t.entry_price:.2f}",
                        t.exit_date.date() if t.exit_date is not None else "",
                        f"{t.exit_price:.2f}" if t.exit_price is not None else "",
                        f"{t.return_pct:.2f}%" if t.return_pct is not None else "",
                        t.bars_held,
                        t.reason,
                    ]
                    for t in result.closed_trades
                ]
                print(
                    tabulate(
                        trade_rows,
                        headers=["Entry", "Px", "Exit", "Px", "Return", "Bars", "Why"],
                        tablefmt="simple",
                    )
                )

            if export_dir is not None:
                from backtest.strategy import generate_signals

                sig = generate_signals(data, params)
                safe = ticker.replace("/", "_")
                path = export_dir / f"{safe}_signals.csv"
                cols = [
                    "Open",
                    "High",
                    "Low",
                    "Close",
                    "Volume",
                    "percent_b",
                    "mfi",
                    "sma_fast",
                    "sma_slow",
                    "entry_ready",
                    "buy",
                    "sell_zone",
                    "sell",
                    "bull_div",
                    "bear_div",
                ]
                sig[cols].to_csv(path)
                print(f"Exported {path}")

        except Exception as exc:  # noqa: BLE001 — surface per-ticker failures
            print(f"[ERROR] {ticker}: {exc}", file=sys.stderr)
            rows.append(
                {
                    "ticker": ticker,
                    "trades": 0,
                    "win_rate_pct": None,
                    "avg_return_pct": None,
                    "total_return_pct": None,
                    "max_drawdown_pct": None,
                    "buy_hold_return_pct": None,
                    "vs_buy_hold_pct": None,
                    "error": str(exc),
                }
            )

    if not rows:
        return 1

    display_cols = [
        "ticker",
        "trades",
        "win_rate_pct",
        "avg_return_pct",
        "total_return_pct",
        "max_drawdown_pct",
        "avg_bars_held",
        "buy_hold_return_pct",
        "vs_buy_hold_pct",
    ]
    table = []
    for r in rows:
        table.append([r.get(c) for c in display_cols])

    print("\n=== Summary ===")
    print(tabulate(table, headers=display_cols, tablefmt="github"))
    print(
        "\nNote: total_return_pct compounds sequential full-capital trades "
        "(no position sizing). Past results ≠ future performance."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
