"""Is this instrument, at this timeframe, worth testing anything on?

Phase 5 established that ATR-sized exits do not reduce the dispersion of
results across instruments: the standard deviation of mean R went *up*, from
0.066 with fixed point exits to 0.164 with ATR exits. The reason is that the
ATR normalizes exposure to volatility but not the ratio the strategy actually
pays, which is spread over volatility - and that ratio varies by more than a
factor of thirteen across this broker's instruments. On M1 and M5 the spread
reaches 47% of the stop distance on some of them.

A strategy on such a pair is not a bad strategy. It is a bet that cannot be
won: the broker takes a fixed share of every move the instrument makes, and
no realistic edge covers it. Backtesting it produces a number, and the number
is noise around a known negative constant.

So this module runs *before* the edge gate and answers one question per cell:

    median spread / median ATR  <=  threshold ?

Cells that fail are not tested. That is not the same as testing them and
finding nothing, and the distinction matters for the trial count: a cell that
was refused was never an attempt, and counting it in the multiple-testing
correction would inflate N with experiments nobody ran.

The headline ratio is stated against one ATR because that is the unit the
data comes in. The stop distance a strategy actually places is a multiple of
it - every strategy in this library uses 2.0 x ATR - so the share of the stop
the spread eats is also reported, at that multiple, to avoid the reader
having to do it in their head and get it wrong.

**The spread does not come from the bars being judged.** The `spread` field
of a bar above M1 is the minimum spread inside it, not a spread (see
`core.data.spread`): on this broker's H1 bars four FX instruments have a
median of exactly zero, which would make every one of their cells look free
to trade. The numerator here is therefore the instrument's spread measured on
M1, where the field means what it says, and each cell records which source it
used. Where no M1 sample exists the cell is not judged at all: an unknown
cost is reported as unknown, never as zero.
"""
from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core.data.cache import ParquetCache
from core.data.provider import Timeframe
from core.data.spread import (
    MEANINGFUL_TIMEFRAME,
    SpreadReference,
    measure_from_cache,
)
from core.indicators import functions as f
from core.serialization import json_safe

logger = logging.getLogger(__name__)

# Above this share of one ATR, the spread is not a cost the strategy pays: it
# is the strategy. Chosen as the default, not derived - it is a judgement
# about how much of the move a system may hand over before the exercise is
# pointless, and it is exposed so it can be argued with.
DEFAULT_MAX_SPREAD_ATR = 0.15

# The stop multiple every strategy in `strategies/` uses. Only used to restate
# the same ratio as a share of the stop actually placed.
LIBRARY_STOP_ATR_MULT = 2.0

DEFAULT_ATR_PERIOD = 14

# Below this many bars the medians are not medians of anything.
MIN_BARS = 200


@dataclass(frozen=True)
class TradabilityCell:
    """One instrument at one timeframe, and whether it can be traded at all."""

    symbol: str
    timeframe: str
    bars: int
    first_bar: datetime | None
    last_bar: datetime | None
    median_spread_points: float | None
    p90_spread_points: float | None
    spread_source: str
    median_atr_points: float | None
    spread_atr_ratio: float | None
    p90_spread_atr_ratio: float | None
    spread_stop_share: float | None
    stop_atr_mult: float
    max_ratio: float
    tradable: bool
    judged: bool
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return json_safe(asdict(self))


@dataclass
class TradabilityTable:
    """Every cell, plus the threshold they were judged against."""

    cells: list[TradabilityCell]
    max_ratio: float
    atr_period: int
    stop_atr_mult: float = LIBRARY_STOP_ATR_MULT
    warnings: list[str] = field(default_factory=list)

    def get(self, symbol: str, timeframe: str) -> TradabilityCell | None:
        return next(
            (c for c in self.cells if c.symbol == symbol and c.timeframe == timeframe),
            None,
        )

    def is_tradable(self, symbol: str, timeframe: str) -> bool:
        """Unknown pairs are tradable: absence of a measurement is not a verdict."""
        cell = self.get(symbol, timeframe)
        return True if cell is None else cell.tradable

    @property
    def excluded(self) -> list[TradabilityCell]:
        return [c for c in self.cells if not c.tradable]

    def as_dict(self) -> dict[str, Any]:
        return json_safe(
            {
                "cells": [cell.as_dict() for cell in self.cells],
                "max_ratio": self.max_ratio,
                "atr_period": self.atr_period,
                "stop_atr_mult": self.stop_atr_mult,
                "tradable": sum(1 for c in self.cells if c.tradable and c.judged),
                "excluded": len(self.excluded),
                "unjudged": sum(1 for c in self.cells if not c.judged),
                "warnings": self.warnings,
            }
        )

    def as_text(self) -> str:
        lines = [
            f"Tradability - spread must stay under {self.max_ratio:.0%} of one "
            f"ATR({self.atr_period})",
            f"the spread is measured on {MEANINGFUL_TIMEFRAME.name}, never on "
            f"the bar being judged",
            f"{'symbol':<10} {'tf':<4} {'bars':>7} {'spread pt':>10} "
            f"{'ATR pt':>10} {'sprd/ATR':>9} {'of stop':>8}  verdict",
            "-" * 80,
        ]
        for cell in self.cells:
            if cell.spread_atr_ratio is None:
                lines.append(
                    f"{cell.symbol:<10} {cell.timeframe:<4} {cell.bars:>7} "
                    f"{'-':>10} {'-':>10} {'-':>9} {'-':>8}  not judged: "
                    f"{cell.reason}"
                )
                continue
            lines.append(
                f"{cell.symbol:<10} {cell.timeframe:<4} {cell.bars:>7} "
                f"{cell.median_spread_points:>10.1f} {cell.median_atr_points:>10.1f} "
                f"{cell.spread_atr_ratio:>8.1%} {cell.spread_stop_share:>7.1%}  "
                f"{'testable' if cell.tradable else 'EXCLUDED'}"
            )
        excluded = self.excluded
        lines.append("")
        unjudged_cells = [c for c in self.cells if not c.judged]
        lines.append(
            f"{sum(1 for c in self.cells if c.tradable and c.judged)} of "
            f"{len(self.cells)} cells are testable, {len(excluded)} are refused "
            f"before any strategy is run, {len(unjudged_cells)} could not be "
            f"judged and are neither"
        )
        for warning in self.warnings:
            lines.append(f"  ! {warning}")
        return "\n".join(lines)


def assess(
    bars: pd.DataFrame,
    symbol: str,
    timeframe: str | Timeframe,
    point: float,
    spread: SpreadReference | None,
    atr_period: int = DEFAULT_ATR_PERIOD,
    max_ratio: float = DEFAULT_MAX_SPREAD_ATR,
    stop_atr_mult: float = LIBRARY_STOP_ATR_MULT,
) -> TradabilityCell:
    """Measures one cell against an independently measured spread.

    `spread` is the instrument's spread reference, measured on M1. Passing
    None means the cost is unknown, and an unknown cost produces an unjudged
    cell - not a permissive one and not a zero.
    """
    tf = Timeframe.parse(timeframe)
    blank = TradabilityCell(
        symbol=symbol,
        timeframe=tf.name,
        bars=int(len(bars)),
        first_bar=bars.index[0].to_pydatetime() if len(bars) else None,
        last_bar=bars.index[-1].to_pydatetime() if len(bars) else None,
        median_spread_points=None,
        p90_spread_points=None,
        spread_source="none",
        median_atr_points=None,
        spread_atr_ratio=None,
        p90_spread_atr_ratio=None,
        spread_stop_share=None,
        stop_atr_mult=stop_atr_mult,
        max_ratio=max_ratio,
        tradable=True,
        judged=False,
        reason="",
    )

    if len(bars) < MIN_BARS:
        return unjudged(
            blank,
            f"{len(bars)} bars is below the {MIN_BARS} needed for a median: "
            f"not measured, and therefore not refused",
        )
    if spread is None or not np.isfinite(spread.median_points):
        return unjudged(
            blank,
            f"no spread measured on {MEANINGFUL_TIMEFRAME.name} for this "
            f"instrument: the cost is unknown, and the bar column at this "
            f"timeframe is a minimum of spreads, not a spread",
        )

    atr_price = f.atr(bars["high"], bars["low"], bars["close"], atr_period)
    atr_points = (atr_price / point).replace([np.inf, -np.inf], np.nan).dropna()
    if atr_points.empty or float(atr_points.median()) <= 0:
        return unjudged(
            blank, "ATR is zero or undefined over this series: nothing to compare"
        )

    median_spread = float(spread.median_points)
    p90_spread = float(spread.p90_points)
    median_atr = float(atr_points.median())
    ratio = median_spread / median_atr
    p90_ratio = p90_spread / median_atr
    stop_share = ratio / stop_atr_mult
    tradable = ratio <= max_ratio

    reason = (
        f"spread is {ratio:.1%} of one ATR ({median_spread:.0f} of "
        f"{median_atr:.0f} points), {stop_share:.1%} of a {stop_atr_mult:g}x ATR "
        f"stop"
    )
    # the spread sample and the bars being judged rarely cover the same period:
    # applying a 2025 spread to 1999 bars is an extrapolation, and saying so is
    # cheaper than having someone discover it later
    if spread.window_start is not None and bars.index[0] < spread.window_start:
        reason += (
            f". The spread was measured from {spread.window_start:%Y-%m-%d}, "
            f"after this series begins ({bars.index[0]:%Y-%m-%d}): for the "
            f"earlier bars it is an assumption, not a measurement"
        )
    if not tradable:
        reason += (
            f" - above the {max_ratio:.0%} threshold: no realistic edge covers "
            f"this, so nothing is tested here"
        )

    return TradabilityCell(
        symbol=symbol,
        timeframe=tf.name,
        bars=int(len(bars)),
        first_bar=bars.index[0].to_pydatetime(),
        last_bar=bars.index[-1].to_pydatetime(),
        median_spread_points=median_spread,
        p90_spread_points=p90_spread,
        spread_source=(
            f"{spread.source_timeframe}, {spread.usable_bars} bars, "
            f"{spread.window_start:%Y-%m-%d} to {spread.window_end:%Y-%m-%d}"
            if spread.window_start and spread.window_end
            else spread.source_timeframe
        ),
        median_atr_points=median_atr,
        spread_atr_ratio=ratio,
        p90_spread_atr_ratio=p90_ratio,
        spread_stop_share=stop_share,
        stop_atr_mult=stop_atr_mult,
        max_ratio=max_ratio,
        tradable=tradable,
        judged=True,
        reason=reason,
    )


def unjudged(cell: TradabilityCell, reason: str) -> TradabilityCell:
    """A cell no verdict could be reached on, carrying why.

    It stays `tradable` so the screening does not silently drop it, and
    `judged` is False so nobody reads the permissiveness as an endorsement.
    """
    return TradabilityCell(
        **{**asdict(cell), "tradable": True, "judged": False, "reason": reason}
    )


def build_table(
    cache: ParquetCache,
    symbols: Sequence[str],
    timeframes: Sequence[str],
    points: dict[str, float],
    start: datetime | None = None,
    end: datetime | None = None,
    atr_period: int = DEFAULT_ATR_PERIOD,
    max_ratio: float = DEFAULT_MAX_SPREAD_ATR,
) -> TradabilityTable:
    """The full instrument x timeframe grid, read from the local cache."""
    from core.runs.runner import DataUnavailable, load_bars

    cells: list[TradabilityCell] = []
    warnings: list[str] = []
    for symbol in symbols:
        point = points.get(symbol)
        if not point:
            warnings.append(
                f"{symbol}: no point size available, every timeframe left unjudged"
            )
            continue

        spread = measure_from_cache(cache, symbol)
        if spread is None:
            warnings.append(
                f"{symbol}: no {MEANINGFUL_TIMEFRAME.name} bars cached, so the "
                f"spread is unknown and no cell can be refused on cost grounds"
            )

        for name in timeframes:
            tf = Timeframe.parse(name)
            blank = TradabilityCell(
                symbol=symbol, timeframe=tf.name, bars=0,
                first_bar=None, last_bar=None,
                median_spread_points=None, p90_spread_points=None,
                spread_source="none", median_atr_points=None,
                spread_atr_ratio=None, p90_spread_atr_ratio=None,
                spread_stop_share=None, stop_atr_mult=LIBRARY_STOP_ATR_MULT,
                max_ratio=max_ratio, tradable=True, judged=False, reason="",
            )
            try:
                bars = load_bars(cache, symbol, tf, start, end)
            except DataUnavailable as exc:
                cells.append(unjudged(blank, f"no cached data: {exc}"))
                continue
            cells.append(
                assess(bars, symbol, tf, point, spread, atr_period, max_ratio)
            )

    table = TradabilityTable(
        cells=cells, max_ratio=max_ratio, atr_period=atr_period, warnings=warnings
    )
    logger.info(
        "tradability: %d of %d cells testable, %d refused, %d unjudged "
        "(threshold %.0f%% spread/ATR)",
        sum(1 for c in cells if c.tradable and c.judged),
        len(cells),
        len(table.excluded),
        sum(1 for c in cells if not c.judged),
        max_ratio * 100,
    )
    return table


def as_frame(table: TradabilityTable) -> pd.DataFrame:
    if not table.cells:
        return pd.DataFrame()
    return pd.DataFrame([cell.as_dict() for cell in table.cells])


def save(table: TradabilityTable, path: Path | str) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    import json

    target.write_text(
        json.dumps(table.as_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return target
