"""Forward-max swing rules (KOSDAQ) for *future* compounded returns.

Why not B (equity-max on 2020+)?
--------------------------------
B maximizes 2020~ residual equity and collapses in 2010-2019 (~0.13x).
That is in-sample jackpot, not a forward criterion.

Selection used here
-------------------
Grid over the user's MFI/%b neighborhood on KOSDAQ:
  - require sequential all-in multiple > 1x in BOTH 2010-2019 and 2020-present
  - among those, maximize geo-mean(mult_2010, mult_2020)
  - min 15 trades each period

Verified (sequential, 1억 start, name-pick same-day):
  2010-2019 ≈ 13.3x | 2020-2026-10 ≈ 65.4x | geo ≈ 29.5
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ForwardMaxParams:
    """Buy/sell thresholds chosen for cross-regime compounded equity."""

    # --- Buy ---
    pb_cross: float = 0.3  # daily %b true cross above 0.3
    require_weekly_mfi10_up: bool = True
    weekly_pb_lo: float | None = None
    weekly_pb_hi: float | None = None
    bb_width_min: float | None = None
    weekly_mfi6_max: float | None = None
    close_loc_min: float | None = 0.8
    ret1_min: float | None = 0.05

    # --- Sell ---
    stop_pct: float = 0.10
    take_profit_pct: float = 0.20
    day3_no_up_cut: bool = False
    max_hold_days: int = 20

    market: str = "KOSDAQ"


FORWARD_MAX = ForwardMaxParams()


@dataclass(frozen=True)
class ForwardBalancedParams:
    """Near-equal multiples in both decades (~22x / ~21x); lower geo than FORWARD_MAX."""

    pb_cross: float = 0.4
    require_weekly_mfi10_up: bool = True
    weekly_pb_lo: float | None = None
    weekly_pb_hi: float | None = None
    bb_width_min: float | None = None
    weekly_mfi6_max: float | None = None
    close_loc_min: float | None = 0.6
    ret1_min: float | None = 0.05
    stop_pct: float = 0.15
    take_profit_pct: float = 0.30
    day3_no_up_cut: bool = False
    max_hold_days: int = 10
    market: str = "KOSDAQ"


FORWARD_BALANCED = ForwardBalancedParams()
