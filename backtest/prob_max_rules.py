"""Highest-probability rules in the user's MFI/%b condition neighborhood.

Objective (NOT max compounded equity)
-------------------------------------
Maximize trade-level reliability among variants of the user's filters:
  - mean return > 0 with strong t-test
  - high win rate
  - positive mean in BOTH 2020-2023 and 2024-present
  - as many calendar years as possible with mean > 0

Evaluated on **all matching signals** (equal trade weight), KOSDAQ 2020~.
Weekly MFI(10) rising is kept (user core).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProbMaxParams:
    """Buy/sell thresholds for highest-probability variant."""

    # --- Buy ---
    pb_cross: float = 0.2  # daily %b crosses above 0.2
    require_weekly_mfi10_up: bool = True
    weekly_pb_lo: float | None = 0.15
    weekly_pb_hi: float | None = 0.30
    bb_width_min: float | None = 0.20
    weekly_mfi6_max: float | None = 40.0
    close_loc_min: float | None = 0.7
    ret1_min: float | None = 0.03

    # --- Sell ---
    stop_pct: float = 0.15
    take_profit_pct: float = 0.20
    day3_no_up_cut: bool = False
    max_hold_days: int = 20

    market: str = "KOSDAQ"


# Pool stats (all signals, not sequential): n≈243, mean≈+6.3%, WR≈62.6%,
# all 7 years mean>0, 2020-23≈+7.0%, 2024-≈+6.0%, t-test p≈2e-10
PROB_MAX = ProbMaxParams()


@dataclass(frozen=True)
class ProbMaxTightParams:
    """Closer to user's original filters; slightly lower score, higher selectivity."""

    pb_cross: float = 0.3
    require_weekly_mfi10_up: bool = True
    weekly_pb_lo: float | None = 0.15
    weekly_pb_hi: float | None = 0.30
    bb_width_min: float | None = 0.15
    weekly_mfi6_max: float | None = 40.0
    close_loc_min: float | None = 0.7
    ret1_min: float | None = 0.05
    stop_pct: float = 0.15
    take_profit_pct: float = 0.20
    day3_no_up_cut: bool = False
    max_hold_days: int = 20
    market: str = "KOSDAQ"


# From top table near user pb=0.3: mean≈+5.4%, WR≈60%, both periods +
PROB_MAX_TIGHT = ProbMaxTightParams()
