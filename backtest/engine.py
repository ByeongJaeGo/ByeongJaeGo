"""Backtest engine for daily swing trades."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .strategy import StrategyParams, generate_signals


@dataclass
class Trade:
    entry_date: pd.Timestamp
    entry_price: float
    exit_date: pd.Timestamp | None = None
    exit_price: float | None = None
    reason: str = ""
    bars_held: int = 0

    @property
    def return_pct(self) -> float | None:
        if self.exit_price is None:
            return None
        return (self.exit_price / self.entry_price - 1.0) * 100.0


@dataclass
class BacktestResult:
    ticker: str
    trades: list[Trade] = field(default_factory=list)
    equity_curve: pd.Series | None = None
    buy_hold_return_pct: float = 0.0
    params: StrategyParams | None = None

    @property
    def closed_trades(self) -> list[Trade]:
        return [t for t in self.trades if t.exit_price is not None]

    def summary(self) -> dict[str, Any]:
        closed = self.closed_trades
        rets = [t.return_pct for t in closed if t.return_pct is not None]
        if not rets:
            return {
                "ticker": self.ticker,
                "trades": 0,
                "win_rate_pct": None,
                "avg_return_pct": None,
                "total_return_pct": 0.0,
                "max_drawdown_pct": None,
                "avg_bars_held": None,
                "buy_hold_return_pct": round(self.buy_hold_return_pct, 2),
                "vs_buy_hold_pct": None,
            }

        wins = sum(1 for r in rets if r > 0)
        # Compound sequential full-position trades
        equity = 1.0
        peak = 1.0
        max_dd = 0.0
        curve = []
        for r in rets:
            equity *= 1.0 + r / 100.0
            peak = max(peak, equity)
            dd = (equity / peak - 1.0) * 100.0
            max_dd = min(max_dd, dd)
            curve.append(equity)

        total_return = (equity - 1.0) * 100.0
        return {
            "ticker": self.ticker,
            "trades": len(rets),
            "win_rate_pct": round(100.0 * wins / len(rets), 1),
            "avg_return_pct": round(float(np.mean(rets)), 2),
            "median_return_pct": round(float(np.median(rets)), 2),
            "best_trade_pct": round(float(np.max(rets)), 2),
            "worst_trade_pct": round(float(np.min(rets)), 2),
            "total_return_pct": round(total_return, 2),
            "max_drawdown_pct": round(max_dd, 2),
            "avg_bars_held": round(float(np.mean([t.bars_held for t in closed])), 1),
            "buy_hold_return_pct": round(self.buy_hold_return_pct, 2),
            "vs_buy_hold_pct": round(total_return - self.buy_hold_return_pct, 2),
        }


def run_backtest(
    df: pd.DataFrame,
    ticker: str,
    params: StrategyParams | None = None,
    max_hold_bars: int = 60,
) -> BacktestResult:
    """
    Long-only: enter at next open after buy signal, exit at next open after sell/stop.
    Falls back to same-bar close if Open is missing.
    """
    params = params or StrategyParams()
    signalled = generate_signals(df, params)
    result = BacktestResult(ticker=ticker, params=params)

    if len(signalled) < 2:
        return result

    first = float(signalled["Close"].iloc[0])
    last = float(signalled["Close"].iloc[-1])
    result.buy_hold_return_pct = (last / first - 1.0) * 100.0 if first else 0.0

    position: Trade | None = None
    entry_low = np.nan
    bars_in_trade = 0

    for i in range(len(signalled) - 1):
        row = signalled.iloc[i]
        nxt = signalled.iloc[i + 1]
        next_open = float(nxt["Open"]) if pd.notna(nxt["Open"]) else float(nxt["Close"])

        if position is None:
            if bool(row["buy"]):
                position = Trade(
                    entry_date=signalled.index[i + 1],
                    entry_price=next_open,
                )
                entry_low = float(row["Low"])
                bars_in_trade = 0
            continue

        bars_in_trade += 1
        # Swing stop: close below setup low (ignore intraday wicks)
        stop_hit = False
        if params.stop_below_entry_low and pd.notna(entry_low) and bars_in_trade >= 1:
            if float(row["Close"]) < entry_low:
                stop_hit = True

        timed_out = bars_in_trade >= max_hold_bars
        sell_hit = bool(row["sell"])

        if stop_hit or sell_hit or timed_out:
            reason = "sell"
            if stop_hit:
                reason = "stop"
            elif timed_out:
                reason = "timeout"
            position.exit_date = signalled.index[i + 1]
            position.exit_price = next_open
            position.reason = reason
            position.bars_held = bars_in_trade
            result.trades.append(position)
            position = None
            entry_low = np.nan
            bars_in_trade = 0

    # Force close open position at last close
    if position is not None:
        position.exit_date = signalled.index[-1]
        position.exit_price = float(signalled["Close"].iloc[-1])
        position.reason = "eod"
        position.bars_held = bars_in_trade
        result.trades.append(position)

    result.equity_curve = None
    return result
