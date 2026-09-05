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
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, tzinfo
from typing import Literal

import pandas as pd

from core.data.provider import MIN_SPREAD_M1_COLUMN, SymbolSpec, Timeframe

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


class AggregatedSpreadRefused(ValueError):
    """A per-bar spread was requested where the column is not a spread.

    Its own type so the API can turn it into a status a caller can act on,
    and so that no `except ValueError` anywhere quietly swallows the one
    refusal that protects every result above M1.
    """


@dataclass(frozen=True)
class SpreadPolicy:
    """Where the spread applied to each fill comes from.

    `per_bar` reads the bars' `spread` column. **On M1 that column is a
    spread. Above M1 the broker's field is the minimum spread inside the
    bar** - verified at 100% on three instruments over four thousand periods
    each (see `core.data.spread`) - and since phase 7 it is not called
    `spread` at all: it arrives as `min_spread_m1`, and this policy will not
    read it under any name.

    Above M1, therefore, `per_bar` requires a `spread` column that
    `core.data.spread.attach` reconstructed from the M1 sample of the same
    period. Where that sample does not exist the run is refused
    (`AggregatedSpreadRefused`) rather than served the aggregated column: the
    fallback is what made every high-timeframe backtest of phase 5 trade for
    free on most bars, and a refusal is the only version of that story that
    cannot happen quietly.

    `fixed` charges a constant, which is the honest choice when the number
    comes from a measurement. `quantile` charges one level taken from the
    observed distribution, for stress tests ("what if I had always entered at
    the 90th percentile?") - over that same reconstructed column, so it
    inherits the requirement instead of the old caveat.
    """

    mode: SpreadMode = "per_bar"
    value: float | None = None

    def _column(
        self, bars: pd.DataFrame, timeframe: Timeframe | str | None
    ) -> pd.Series:
        """The per-bar spread, or the reason there is none that can be charged."""
        if "spread" in bars.columns:
            return bars["spread"].astype("float64")
        if MIN_SPREAD_M1_COLUMN in bars.columns:
            label = (
                Timeframe.parse(timeframe).name
                if timeframe is not None
                else "this timeframe"
            )
            raise AggregatedSpreadRefused(
                f"spread_mode {self.mode!r} on {label}: the only spread field "
                f"these bars carry is {MIN_SPREAD_M1_COLUMN!r}, the MINIMUM of "
                f"the M1 spreads inside each bar, which is not the cost of any "
                f"fill. Charging it understates the cost by up to 7x on this "
                f"broker, and by all of it on the FX hours whose minimum is "
                f"zero. Reconstruct the spread from the M1 bars of the same "
                f"period (core.data.spread.attach), or charge a measured "
                f"constant with spread_mode 'fixed'"
            )
        raise AggregatedSpreadRefused(
            f"spread_mode {self.mode!r} but the data carries no spread field at all"
        )

    def series(
        self, bars: pd.DataFrame, timeframe: Timeframe | str | None = None
    ) -> pd.Series:
        if self.mode == "per_bar":
            return self._column(bars, timeframe)
        if self.mode == "fixed":
            if self.value is None:
                raise ValueError("policy 'fixed' requires value (points)")
            return pd.Series(float(self.value), index=bars.index, dtype="float64")
        if self.mode == "quantile":
            if self.value is None or not 0.0 <= self.value <= 1.0:
                raise ValueError("policy 'quantile' requires value between 0 and 1")
            level = float(self._column(bars, timeframe).quantile(self.value))
            logger.info("spread at quantile %.2f = %.1f points", self.value, level)
            return pd.Series(level, index=bars.index, dtype="float64")
        raise ValueError(f"unknown spread policy: {self.mode}")

    def realism(self, bars: pd.DataFrame, timeframe: Timeframe) -> SpreadRealism:
        """Whether the charged spread can be the cost of a real fill.

        This exists so that a run cannot come back with a number without the
        engine having said, next to it, what that number charged. It measures
        rather than judges: the share of bars charged nothing, and whether the
        column being read is a spread at all at this timeframe.
        """
        charged = self.series(bars, timeframe)
        zero_share = float((charged <= 0).mean()) if len(charged) else 0.0
        # reaching here above M1 with a per-bar policy means the column was
        # reconstructed from M1: `series` refuses the aggregated field, so it
        # can no longer be what a run charged
        reconstructed = self.mode in ("per_bar", "quantile") and timeframe.minutes > 1

        warnings: list[str] = []
        if zero_share > 0:
            warnings.append(
                f"{zero_share:.1%} of the bars were charged a zero spread. A "
                f"zero spread does not exist: those fills were free"
            )

        return SpreadRealism(
            mode=self.mode,
            value=self.value,
            timeframe=timeframe.name,
            median_charged_points=float(charged.median()) if len(charged) else 0.0,
            mean_charged_points=float(charged.mean()) if len(charged) else 0.0,
            zero_charged_share=zero_share,
            reads_aggregated_column=False,
            reconstructed_from_m1=reconstructed,
            trustworthy=not warnings,
            warnings=warnings,
        )


@dataclass(frozen=True)
class SpreadRealism:
    """What a run's spread policy actually charged, and whether it can be true."""

    mode: str
    value: float | None
    timeframe: str
    median_charged_points: float
    mean_charged_points: float
    zero_charged_share: float
    # kept, and False on every run this engine can still produce: it is what
    # distinguishes a stored run from before the refusal existed
    reads_aggregated_column: bool
    # above M1, a per-bar spread taken from the M1 bars of the same period
    reconstructed_from_m1: bool = False
    trustworthy: bool = True
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


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
    def zero(cls) -> CostModel:
        """No costs: isolates the gross edge in tests and comparisons."""
        return cls(
            spread=SpreadPolicy(mode="fixed", value=0.0),
            commission=CommissionModel(0.0),
            swap=SwapModel(mode="none"),
        )
