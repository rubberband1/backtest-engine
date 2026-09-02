"""Cost model, fully parametric on the SymbolSpec.

No per-point value hardcoded anywhere: everything goes through the
instrument's `point`, `tick_size`, `tick_value` and `contract_size`. Changing
symbol changes the costs without touching a line.

A note on currency: MT5's `tick_value` is expressed in the **account
currency** (on this installation the account is in EUR and gold quotes in
USD, which is why tick_value is ~0.86 and not 1.00). Converting via
`contract_size` would instead give the instrument's profit currency. We use
`tick_value` when available, because the account equity is what matters.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, tzinfo
from typing import Literal

import pandas as pd

from core.data.provider import SymbolSpec

logger = logging.getLogger(__name__)

SpreadMode = Literal["per_bar", "fixed", "quantile"]
SwapMode = Literal["points", "money", "none"]

# 0 = Monday. The Wednesday-to-Thursday night carries the triple swap because
# it settles the weekend (two-day value date).
WEDNESDAY = 2


def money_per_point(spec: SymbolSpec, lots: float = 1.0) -> float:
    """Currency value of a 1-point move, for `lots` lots."""
    if spec.tick_size > 0 and spec.tick_value > 0:
        return (spec.point / spec.tick_size) * spec.tick_value * lots
    logger.warning(
        "%s: tick_value/tick_size unusable, falling back to contract_size "
        "(result in the profit currency, not the account currency)",
        spec.name,
    )
    return spec.point * spec.contract_size * lots


def points_to_money(points: float, spec: SymbolSpec, lots: float) -> float:
    return points * money_per_point(spec, lots)


def price_to_points(price_delta: float, spec: SymbolSpec) -> float:
    return price_delta / spec.point


def points_to_price(points: float, spec: SymbolSpec) -> float:
    return points * spec.point


@dataclass(frozen=True)
class SpreadPolicy:
    """Where the spread applied to each fill comes from.

    `per_bar` uses the feed's 'spread' column, which is the cost actually
    quoted bar by bar. The other two are for stress tests: a fixed spread, or
    a quantile of the observed distribution (to ask "what if I had always
    entered at the 90th percentile?").
    """

    mode: SpreadMode = "per_bar"
    value: float | None = None

    def series(self, bars: pd.DataFrame) -> pd.Series:
        if self.mode == "per_bar":
            if "spread" not in bars.columns:
                raise ValueError("policy 'per_bar' but the data has no 'spread' column")
            return bars["spread"].astype("float64")
        if self.mode == "fixed":
            if self.value is None:
                raise ValueError("policy 'fixed' requires value (points)")
            return pd.Series(float(self.value), index=bars.index, dtype="float64")
        if self.mode == "quantile":
            if self.value is None or not 0.0 <= self.value <= 1.0:
                raise ValueError("policy 'quantile' requires value between 0 and 1")
            level = float(bars["spread"].quantile(self.value))
            logger.info("spread at quantile %.2f = %.1f points", self.value, level)
            return pd.Series(level, index=bars.index, dtype="float64")
        raise ValueError(f"unknown spread policy: {self.mode}")


@dataclass(frozen=True)
class CommissionModel:
    """Commission per lot per side, in the account currency."""

    per_lot_per_side: float = 0.0

    def round_turn(self, lots: float) -> float:
        return self.per_lot_per_side * lots * 2.0


@dataclass(frozen=True)
class SwapModel:
    """Swap charged at every server midnight crossed.

    `mode='points'` interprets the SymbolSpec's swap_long/swap_short as points
    per lot per night (SYMBOL_SWAP_MODE_POINTS, the most common case);
    `mode='money'` as an amount in the account currency. The phase 1
    SymbolSpec does not expose swap_mode, so the interpretation is declared
    here.
    """

    mode: SwapMode = "points"
    triple_weekday: int = WEDNESDAY

    def nights(self, entry: datetime, exit_: datetime, server_tz: tzinfo) -> float:
        """Nights crossed, weighted (x3 on the triple-rollover night)."""
        if self.mode == "none" or exit_ <= entry:
            return 0.0
        local_entry = entry.astimezone(server_tz)
        local_exit = exit_.astimezone(server_tz)

        midnight = (local_entry + timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        total = 0.0
        while midnight <= local_exit:
            closing_day = (midnight - timedelta(minutes=1)).weekday()
            total += 3.0 if closing_day == self.triple_weekday else 1.0
            midnight += timedelta(days=1)
        return total

    def charge(
        self,
        spec: SymbolSpec,
        direction: int,
        lots: float,
        entry: datetime,
        exit_: datetime,
        server_tz: tzinfo,
    ) -> float:
        """Signed amount: negative = cost. Direction +1 long, -1 short."""
        nights = self.nights(entry, exit_, server_tz)
        if not nights:
            return 0.0
        rate = spec.swap_long if direction > 0 else spec.swap_short
        if self.mode == "money":
            return rate * lots * nights
        return points_to_money(rate, spec, lots) * nights


@dataclass(frozen=True)
class CostModel:
    """The three costs together."""

    spread: SpreadPolicy = SpreadPolicy()
    commission: CommissionModel = CommissionModel()
    swap: SwapModel = SwapModel()

    @classmethod
    def zero(cls) -> "CostModel":
        """No costs: isolates the gross edge in tests and comparisons."""
        return cls(
            spread=SpreadPolicy(mode="fixed", value=0.0),
            commission=CommissionModel(0.0),
            swap=SwapModel(mode="none"),
        )
