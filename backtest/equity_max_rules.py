"""Account-equity-maximizing swing rules (KOSDAQ, sequential all-in).

WARNING: This maximizes compounded equity (can be luck/regime heavy).
For highest *probability* / expectancy among the user's filters, use
`prob_max_rules.PROB_MAX` instead.

Optimized on wide KOSDAQ OHLCV (2020-01 ~ 2026-10) for **sequential
compounded equity** (one position at a time, full capital), not average
trade return.

Selection notes
---------------
- Daily %b cross is the true threshold cross: prev <= th and pb > th.
- Same-day ties broken by stock name (alphabetical).
- Must be profitable in both 2020-2023 and 2024-present.
- Among name-pick configs with TP >= 15%, pick highest geo-mean of the
  two period multiples.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EquityMaxParams:
    """Buy / sell thresholds for equity-max sequential backtest."""

    # --- Buy (signal-day close) ---
    pb_cross: float = 0.4  # daily %b crosses above this
    require_weekly_mfi10_up: bool = True
    weekly_pb_lo: float | None = None  # no weekly %b band
    weekly_pb_hi: float | None = None
    bb_width_min: float | None = 0.20  # BandWidth >= 20%
    weekly_mfi6_max: float | None = None
    close_loc_min: float | None = 0.7  # (C-L)/(H-L)
    ret1_min: float | None = 0.05  # +5% day

    # --- Sell ---
    stop_pct: float = 0.15  # -15% intraday low
    take_profit_pct: float = 0.30  # +30% intraday high
    day3_no_up_cut: bool = False
    max_hold_days: int = 5

    market: str = "KOSDAQ"


# Sequential name-pick results (run_equity_max.py on cached KOSDAQ):
# 2020~2026-10: ~440x | 2020-2023: ~38x | 2024-now: ~12x | ~213 trades | WR ~57%
EQUITY_MAX = EquityMaxParams()


@dataclass(frozen=True)
class PriorRulesParams:
    """Previous user 'final' rules (kept for comparison; weaker on equity)."""

    pb_cross: float = 0.3
    require_weekly_mfi10_up: bool = True
    weekly_pb_lo: float | None = 0.15
    weekly_pb_hi: float | None = 0.30
    bb_width_min: float | None = 0.25
    weekly_mfi6_max: float | None = 45.0
    close_loc_min: float | None = 0.8
    ret1_min: float | None = 0.05
    stop_pct: float = 0.10
    take_profit_pct: float = 0.20
    day3_no_up_cut: bool = True
    max_hold_days: int = 20
    market: str = "KOSDAQ"


PRIOR_RULES = PriorRulesParams()
