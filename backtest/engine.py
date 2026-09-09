"""Backtest engine for daily swing trades."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from .strategy import StrategyParams, Top100Params, generate_signals, generate_top100_signals


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
    params: Any = None
    name: str = ""

    @property
    def closed_trades(self) -> list[Trade]:
        return [t for t in self.trades if t.exit_price is not None]

    def summary(self) -> dict[str, Any]:
        closed = self.closed_trades
        rets = [t.return_pct for t in closed if t.return_pct is not None]
        base = {
            "ticker": self.ticker,
            "name": self.name,
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate_pct": None,
            "avg_return_pct": None,
            "median_return_pct": None,
            "total_return_pct": 0.0,
            "sum_return_pct": 0.0,
            "max_drawdown_pct": None,
            "avg_bars_held": None,
            "buy_hold_return_pct": round(self.buy_hold_return_pct, 2),
            "vs_buy_hold_pct": None,
        }
        if not rets:
            return base

        wins = sum(1 for r in rets if r > 0)
        equity = 1.0
        peak = 1.0
        max_dd = 0.0
        for r in rets:
            equity *= 1.0 + r / 100.0
            peak = max(peak, equity)
            dd = (equity / peak - 1.0) * 100.0
            max_dd = min(max_dd, dd)

        total_return = (equity - 1.0) * 100.0
        return {
            "ticker": self.ticker,
            "name": self.name,
            "trades": len(rets),
            "wins": wins,
            "losses": len(rets) - wins,
            "win_rate_pct": round(100.0 * wins / len(rets), 1),
            "avg_return_pct": round(float(np.mean(rets)), 2),
            "median_return_pct": round(float(np.median(rets)), 2),
            "best_trade_pct": round(float(np.max(rets)), 2),
            "worst_trade_pct": round(float(np.min(rets)), 2),
            "total_return_pct": round(total_return, 2),
            "sum_return_pct": round(float(np.sum(rets)), 2),
            "max_drawdown_pct": round(max_dd, 2),
            "avg_bars_held": round(float(np.mean([t.bars_held for t in closed])), 1),
            "buy_hold_return_pct": round(self.buy_hold_return_pct, 2),
            "vs_buy_hold_pct": round(total_return - self.buy_hold_return_pct, 2),
        }


def _run_on_signals(
    signalled: pd.DataFrame,
    ticker: str,
    stop_below_entry_low: bool,
    max_hold_bars: int,
    name: str = "",
    params: Any = None,
) -> BacktestResult:
    result = BacktestResult(ticker=ticker, params=params, name=name)
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
        stop_hit = False
        if stop_below_entry_low and pd.notna(entry_low) and bars_in_trade >= 1:
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

    if position is not None:
        position.exit_date = signalled.index[-1]
        position.exit_price = float(signalled["Close"].iloc[-1])
        position.reason = "eod"
        position.bars_held = bars_in_trade
        result.trades.append(position)

    return result


def run_backtest(
    df: pd.DataFrame,
    ticker: str,
    params: StrategyParams | None = None,
    max_hold_bars: int = 60,
) -> BacktestResult:
    params = params or StrategyParams()
    signalled = generate_signals(df, params)
    return _run_on_signals(
        signalled,
        ticker=ticker,
        stop_below_entry_low=params.stop_below_entry_low,
        max_hold_bars=max_hold_bars,
        params=params,
    )


def run_top100_backtest(
    df: pd.DataFrame,
    ticker: str,
    params: Top100Params | None = None,
    max_hold_bars: int = 60,
    name: str = "",
) -> BacktestResult:
    params = params or Top100Params()
    signalled = generate_top100_signals(df, params)
    return _run_on_signals(
        signalled,
        ticker=ticker,
        stop_below_entry_low=params.stop_below_entry_low,
        max_hold_bars=max_hold_bars,
        name=name,
        params=params,
    )


def aggregate_summaries(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Pool all per-ticker trade stats into one portfolio-style summary."""
    traded = [r for r in rows if r.get("trades")]
    if not traded:
        return {
            "tickers_tested": len(rows),
            "tickers_with_trades": 0,
            "trades": 0,
            "win_rate_pct": None,
            "avg_return_pct": None,
            "median_ticker_total_return_pct": None,
            "avg_ticker_total_return_pct": None,
        }

    total_trades = sum(int(r["trades"]) for r in traded)
    total_wins = sum(int(r.get("wins") or 0) for r in traded)
    # Average of per-trade means weighted by trade count
    weighted_avg = sum(float(r["avg_return_pct"]) * int(r["trades"]) for r in traded) / total_trades
    ticker_totals = [float(r["total_return_pct"]) for r in traded]
    return {
        "tickers_tested": len(rows),
        "tickers_with_trades": len(traded),
        "trades": total_trades,
        "wins": total_wins,
        "losses": total_trades - total_wins,
        "win_rate_pct": round(100.0 * total_wins / total_trades, 1),
        "avg_return_pct": round(weighted_avg, 2),
        "avg_ticker_total_return_pct": round(float(np.mean(ticker_totals)), 2),
        "median_ticker_total_return_pct": round(float(np.median(ticker_totals)), 2),
        "best_ticker_total_return_pct": round(float(np.max(ticker_totals)), 2),
        "worst_ticker_total_return_pct": round(float(np.min(ticker_totals)), 2),
    }
