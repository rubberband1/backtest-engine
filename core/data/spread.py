"""What the spread on a bar actually is, and why it must not be trusted.

Measured on this broker's feed, on XAUUSD.r, GBPUSD.r and XTIUSD, over more
than four thousand hours each:

    the `spread` field of an H1 bar equals the MINIMUM of the spreads of the
    sixty M1 bars inside it, in 100.0% of the hours compared.

It is not an average and it is not the spread at any particular moment of the
bar. It is the best price of the hour. Charging it as the transaction cost of
a fill is charging the luckiest second of the period, and the size of the
error is not small:

    XTIUSD   M1 median 29 points -> H1 column median  4 points   (7x cheap)
    GBPUSD.r M1 median  3 points -> H1 column median  0 points   (free)
    USDJPY.r M1 median  3 points -> H1 column median  0 points   (free)
    XAUUSD.r M1 median  7 points -> H1 column median  2 points

On four FX instruments the median H1 spread is exactly zero, and 67-76% of
their H1 bars carry a zero spread. A zero spread does not exist. A backtest
that runs on `per_bar` above M1 therefore trades for free on most bars, and
every result it produces is optimistic by an amount nobody measured.

This module gives the honest number instead: the spread distribution measured
where it is meaningful, on M1, per instrument. `SpreadPolicy(mode="fixed",
value=<median>)` then charges a cost that was actually observed, and the run
records which value it used so the choice stays auditable.

Since phase 7 the bars do not even keep the name: above M1 the raw field is
called `min_spread_m1`, because two different quantities sharing one name is
how it got charged in the first place. A `spread` column above M1 exists only
where `reconstruct_from_m1` below put it, from the M1 sample of the same
period - and where that sample is missing, the run is refused rather than
served the aggregated column.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core.data.cache import ParquetCache
from core.data.provider import MIN_SPREAD_M1_COLUMN, Timeframe
from core.serialization import json_safe

logger = logging.getLogger(__name__)

REFERENCE_FILE = "spread_reference.json"

# The timeframe on which a bar's spread field is a spread and not a minimum of
# spreads. One minute is the finest bar the terminal stores.
MEANINGFUL_TIMEFRAME = Timeframe.M1

# Bars whose spread field is zero are dropped from the measurement rather than
# averaged in: zero is the feed saying nothing, not a free trade.
DROP_ZERO_SPREAD = True


class SpreadUnavailable(RuntimeError):
    """A per-bar spread was asked for and cannot be produced honestly.

    Raised instead of falling back to the aggregated column. The fallback is
    the bug: it charges the minimum spread of the period as if it were the
    cost of a fill, and it does so silently.
    """


@dataclass(frozen=True)
class SpreadReference:
    """The spread distribution of an instrument, measured on M1 bars."""

    symbol: str
    source_timeframe: str
    bars: int
    usable_bars: int
    zero_share: float
    median_points: float
    mean_points: float
    p90_points: float
    p99_points: float
    window_start: datetime | None
    window_end: datetime | None
    measured_at: datetime

    def as_dict(self) -> dict[str, Any]:
        return json_safe(asdict(self))

    @property
    def verdict(self) -> str:
        return (
            f"{self.symbol}: median {self.median_points:.0f} points on "
            f"{self.usable_bars} usable {self.source_timeframe} bars "
            f"({self.zero_share:.1%} of the sample carried a zero spread and was "
            f"dropped), p90 {self.p90_points:.0f}, p99 {self.p99_points:.0f}"
        )


def measure(bars: pd.DataFrame, symbol: str, timeframe: Timeframe) -> SpreadReference:
    """Summarizes the spread column of a frame. Zeros are excluded, not zeroed.

    Refuses above M1 rather than measuring the aggregated field: a reference
    built from minima is a reference to the wrong quantity, and it would be
    charged as a constant by every `fixed` policy that reads it.
    """
    if timeframe.minutes > MEANINGFUL_TIMEFRAME.minutes and (
        MIN_SPREAD_M1_COLUMN in bars.columns
    ):
        raise SpreadUnavailable(
            f"a spread reference cannot be measured on {timeframe.name}: the "
            f"field these bars carry is {MIN_SPREAD_M1_COLUMN!r}, a minimum of "
            f"spreads. Measure on {MEANINGFUL_TIMEFRAME.name}"
        )
    spread = bars["spread"].astype("float64") if "spread" in bars.columns else pd.Series(
        dtype="float64"
    )
    zero_share = float((spread == 0).mean()) if len(spread) else 0.0
    usable = spread[spread > 0] if DROP_ZERO_SPREAD else spread

    if usable.empty:
        return SpreadReference(
            symbol=symbol,
            source_timeframe=timeframe.name,
            bars=int(len(bars)),
            usable_bars=0,
            zero_share=zero_share,
            median_points=float("nan"),
            mean_points=float("nan"),
            p90_points=float("nan"),
            p99_points=float("nan"),
            window_start=bars.index[0].to_pydatetime() if len(bars) else None,
            window_end=bars.index[-1].to_pydatetime() if len(bars) else None,
            measured_at=datetime.now(timezone.utc),
        )

    return SpreadReference(
        symbol=symbol,
        source_timeframe=timeframe.name,
        bars=int(len(bars)),
        usable_bars=int(len(usable)),
        zero_share=zero_share,
        median_points=float(usable.median()),
        mean_points=float(usable.mean()),
        p90_points=float(usable.quantile(0.90)),
        p99_points=float(usable.quantile(0.99)),
        window_start=bars.index[0].to_pydatetime(),
        window_end=bars.index[-1].to_pydatetime(),
        measured_at=datetime.now(timezone.utc),
    )


def path_for(cache: ParquetCache, symbol: str) -> Path:
    return cache.root / cache._slug(symbol) / REFERENCE_FILE


def load(cache: ParquetCache, symbol: str) -> SpreadReference | None:
    target = path_for(cache, symbol)
    if not target.exists():
        return None
    payload = json.loads(target.read_text(encoding="utf-8"))
    for key in ("window_start", "window_end", "measured_at"):
        if payload.get(key):
            payload[key] = datetime.fromisoformat(payload[key])
    return SpreadReference(**payload)


def store(cache: ParquetCache, reference: SpreadReference) -> Path:
    target = path_for(cache, reference.symbol)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(reference.as_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return target


def measure_from_cache(
    cache: ParquetCache,
    symbol: str,
    timeframe: Timeframe = MEANINGFUL_TIMEFRAME,
    refresh: bool = False,
) -> SpreadReference | None:
    """The instrument spread reference, measured once and then reused.

    Returns None when there are no M1 bars to measure on: the caller must
    then say it does not know the spread, not substitute the aggregated
    column for it.
    """
    if not refresh:
        existing = load(cache, symbol)
        if existing is not None:
            return existing

    from core.runs.runner import DataUnavailable, load_bars

    try:
        bars = load_bars(cache, symbol, timeframe)
    except DataUnavailable:
        logger.warning(
            "%s: no %s bars cached, the spread cannot be measured where it is "
            "meaningful",
            symbol,
            timeframe.name,
        )
        return None
    reference = measure(bars, symbol, timeframe)
    store(cache, reference)
    logger.info(reference.verdict)
    return reference


# -- the claim this module rests on, re-checkable on demand ---------------


@dataclass
class AggregationCheck:
    """Whether a high-timeframe spread column is the min of its sub-bars."""

    symbol: str
    low_timeframe: str
    high_timeframe: str
    periods_compared: int
    share_equal_to_min: float
    share_equal_to_median: float
    share_equal_to_max: float
    low_median_points: float
    high_median_points: float
    verdict: str

    def as_dict(self) -> dict[str, Any]:
        return json_safe(asdict(self))


def check_aggregation(
    low: pd.DataFrame,
    high: pd.DataFrame,
    symbol: str,
    low_timeframe: Timeframe,
    high_timeframe: Timeframe,
    min_sub_bars: int = 0,
) -> AggregationCheck:
    """Compares a high-timeframe spread column with its sub-bar distribution.

    `min_sub_bars` drops periods that are only partly covered by the
    low-timeframe sample: a period holding three M1 bars out of sixty says
    nothing about aggregation, only about coverage.
    """
    freq = f"{high_timeframe.minutes}min"
    column = MIN_SPREAD_M1_COLUMN if MIN_SPREAD_M1_COLUMN in high.columns else "spread"
    grouped = low["spread"].astype("float64").resample(freq).agg(
        ["min", "median", "max", "count"]
    )
    joined = grouped.join(high[column].astype("float64").rename("column"), how="inner")
    joined = joined.dropna()
    if min_sub_bars:
        joined = joined[joined["count"] >= min_sub_bars]

    if joined.empty:
        return AggregationCheck(
            symbol=symbol,
            low_timeframe=low_timeframe.name,
            high_timeframe=high_timeframe.name,
            periods_compared=0,
            share_equal_to_min=float("nan"),
            share_equal_to_median=float("nan"),
            share_equal_to_max=float("nan"),
            low_median_points=float("nan"),
            high_median_points=float("nan"),
            verdict=(
                f"no {high_timeframe.name} period is covered by enough "
                f"{low_timeframe.name} bars: the claim cannot be checked on "
                f"this cache"
            ),
        )

    equal_min = float((joined["column"] == joined["min"]).mean())
    equal_median = float((joined["column"] == joined["median"]).mean())
    equal_max = float((joined["column"] == joined["max"]).mean())
    low_median = float(joined["median"].median())
    high_median = float(joined["column"].median())

    if equal_min > 0.99:
        verdict = (
            f"the {high_timeframe.name} spread column is the minimum of its "
            f"{low_timeframe.name} sub-bars in {equal_min:.1%} of "
            f"{len(joined)} periods: as a fill cost it charges the best price "
            f"of the bar ({high_median:.0f} points against a sub-bar median of "
            f"{low_median:.0f})"
        )
    else:
        verdict = (
            f"over {len(joined)} periods the {high_timeframe.name} column "
            f"matches the sub-bar minimum {equal_min:.1%} of the time, the "
            f"median {equal_median:.1%}, the maximum {equal_max:.1%}"
        )

    return AggregationCheck(
        symbol=symbol,
        low_timeframe=low_timeframe.name,
        high_timeframe=high_timeframe.name,
        periods_compared=int(len(joined)),
        share_equal_to_min=equal_min,
        share_equal_to_median=equal_median,
        share_equal_to_max=equal_max,
        low_median_points=low_median,
        high_median_points=high_median,
        verdict=verdict,
    )


# -- reconstruction: the only honest way to charge per-bar above M1 ------


# Which point of the M1 distribution inside a bar stands for "the spread a
# fill would have paid". The median is the default because it is the level
# half the minutes of the bar were at or below; the minimum is not offered,
# because the minimum is the thing this module exists to stop charging.
DEFAULT_PER_BAR_QUANTILE = 0.5
MIN_PER_BAR_QUANTILE = 0.5


def reconstruct_from_m1(
    m1: pd.DataFrame,
    index: pd.DatetimeIndex,
    timeframe: Timeframe,
    quantile: float = DEFAULT_PER_BAR_QUANTILE,
) -> pd.Series:
    """Per-bar spread for `index`, taken from the M1 spreads inside each bar.

    Every bar of `index` must be covered by at least one M1 bar carrying a
    positive spread. A bar that is not covered has an unknown cost, and an
    unknown cost is refused rather than filled in: interpolating it would put
    a number nobody measured into a fill.
    """
    if not MIN_PER_BAR_QUANTILE <= quantile <= 1.0:
        raise ValueError(
            f"per-bar spread quantile must be between {MIN_PER_BAR_QUANTILE} and "
            f"1.0, got {quantile}: below the median the reconstruction drifts "
            f"back towards the minimum it exists to replace"
        )
    if timeframe.minutes <= MEANINGFUL_TIMEFRAME.minutes:
        raise ValueError(
            f"{timeframe.name} needs no reconstruction: its spread column is a "
            f"spread"
        )
    if "spread" not in m1.columns:
        raise SpreadUnavailable("the M1 sample has no spread column")

    source = m1["spread"].astype("float64")
    if DROP_ZERO_SPREAD:
        # a zero in the M1 feed is the feed saying nothing, and averaging it
        # in would drag the reconstruction towards free
        source = source[source > 0]
    if source.empty:
        raise SpreadUnavailable(
            "every M1 spread in the sample is zero: nothing to reconstruct from"
        )

    # Each M1 bar is assigned to the bar of `index` that CONTAINS it, rather
    # than to an epoch-aligned resample bin. The two are not the same thing:
    # this broker stamps its H4 bars at 21:00, 01:00, 05:00 server-side, and
    # a resample on the 4-hour epoch grid would line up with none of them and
    # report the whole series as uncovered. Session gaps are handled by the
    # same rule - an M1 bar past the end of the bar before it belongs to no
    # bar and is dropped, instead of being folded into the previous one.
    # in integer nanoseconds: a tz-aware index converts to an object array of
    # Timestamps, which searchsorted would compare one Python object at a time
    span = int(pd.Timedelta(minutes=timeframe.minutes).value)
    edges = index.asi8
    stamps = pd.DatetimeIndex(source.index).asi8
    position = np.searchsorted(edges, stamps, side="right") - 1
    inside = position >= 0
    inside[inside] &= stamps[inside] < edges[position[inside]] + span

    owner = position[inside]
    values = source.to_numpy(dtype="float64")[inside]
    grouped = pd.Series(values).groupby(owner)
    level = grouped.quantile(quantile)
    covered = grouped.count()

    aligned = pd.Series(np.nan, index=np.arange(len(index)), dtype="float64")
    aligned.loc[level.index] = level.to_numpy(dtype="float64")
    counts = pd.Series(0, index=np.arange(len(index)), dtype="int64")
    counts.loc[covered.index] = covered.to_numpy(dtype="int64")

    uncovered = aligned.isna() | (counts <= 0)
    if bool(uncovered.any()):
        missing = index[uncovered.to_numpy()]
        raise SpreadUnavailable(
            f"{int(uncovered.sum())} of {len(index)} {timeframe.name} bars have "
            f"no M1 spread behind them (first {missing[0]}, last {missing[-1]}). "
            f"The cost of those fills is unknown: download the M1 history for "
            f"this period, or charge a measured constant with spread_mode "
            f"'fixed'"
        )
    return pd.Series(
        aligned.to_numpy(dtype="float64"), index=index, name="spread", dtype="float64"
    )


def reconstructed_series(
    cache: ParquetCache,
    symbol: str,
    timeframe: Timeframe,
    index: pd.DatetimeIndex,
    quantile: float = DEFAULT_PER_BAR_QUANTILE,
) -> pd.Series:
    """`reconstruct_from_m1` reading the M1 sample from the local cache."""
    from core.runs.runner import DataUnavailable, load_bars

    start = index[0].to_pydatetime()
    end = (index[-1] + pd.Timedelta(minutes=timeframe.minutes)).to_pydatetime()
    try:
        m1 = load_bars(cache, symbol, MEANINGFUL_TIMEFRAME, start, end)
    except DataUnavailable as exc:
        raise SpreadUnavailable(
            f"a per-bar spread on {symbol} {timeframe.name} has to come from "
            f"the M1 bars of the same period, and there are none cached "
            f"({exc}). Download them, or charge a measured constant with "
            f"spread_mode 'fixed' - the {timeframe.name} spread column is the "
            f"minimum of the M1 spreads inside the bar, not a spread"
        ) from exc
    return reconstruct_from_m1(m1, index, timeframe, quantile)


# -- how much of a run's spread was measured -------------------------------


@dataclass(frozen=True)
class SpreadCoverage:
    """What share of a run's bars had an M1 sample underneath them.

    The cost model charges a spread on every bar. Where M1 exists for the
    period, that spread is measured; where it does not, it is a constant
    taken from a *different* period - usually a recent one - and applied
    backwards. Both are reported as "the spread charged", and until this
    existed nothing said which of the two a given run had been given.

    It matters most exactly where it is least visible. The oldest bars in a
    sample are the ones with no M1 behind them, and they are also the ones
    added to buy statistical power, so the part of a result resting on an
    assumed cost is systematically the part nobody looks at.
    """

    symbol: str
    timeframe: str
    bars: int
    measured_bars: int
    assumed_bars: int
    m1_window_start: datetime | None
    m1_window_end: datetime | None

    @property
    def measured_share(self) -> float:
        return self.measured_bars / self.bars if self.bars else 0.0

    @property
    def fully_measured(self) -> bool:
        return self.bars > 0 and self.assumed_bars == 0

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(
            measured_share=self.measured_share,
            fully_measured=self.fully_measured,
            verdict=self.verdict,
        )
        return json_safe(payload)

    @property
    def verdict(self) -> str:
        if self.bars == 0:
            return "no bars"
        if self.fully_measured:
            return (
                "every bar has an M1 sample behind it: the spread charged is "
                "measured over the whole period"
            )
        if self.measured_bars == 0:
            return (
                f"no bar has an M1 sample behind it: the spread charged is an "
                f"assumption over the whole period, carried in from "
                f"{_window(self.m1_window_start, self.m1_window_end)}"
            )
        return (
            f"{self.measured_share:.1%} of bars have an M1 sample behind them "
            f"({self.measured_bars} of {self.bars}); on the remaining "
            f"{self.assumed_bars} the spread is an assumption carried in from "
            f"{_window(self.m1_window_start, self.m1_window_end)}"
        )


def _window(start: datetime | None, end: datetime | None) -> str:
    if start is None or end is None:
        return "no measured window"
    return f"{start:%Y-%m-%d} to {end:%Y-%m-%d}"


def coverage(
    cache: ParquetCache,
    symbol: str,
    timeframe: Timeframe,
    index: pd.DatetimeIndex,
) -> SpreadCoverage:
    """Which of `index`'s bars have M1 bars inside them.

    A bar counts as measured when at least one M1 bar falls inside it, not
    when it merely falls between the first and last M1 timestamps: the M1
    sample has holes, and a bar inside a hole is exactly as assumed as one
    outside the range entirely.
    """
    from core.runs.runner import DataUnavailable, load_bars

    if len(index) == 0:
        return SpreadCoverage(symbol, timeframe.name, 0, 0, 0, None, None)

    start = index[0].to_pydatetime()
    end = (index[-1] + pd.Timedelta(minutes=timeframe.minutes)).to_pydatetime()
    try:
        m1 = load_bars(cache, symbol, MEANINGFUL_TIMEFRAME, start, end)
    except DataUnavailable:
        m1 = None

    if m1 is None or m1.empty:
        stored = load(cache, symbol)
        return SpreadCoverage(
            symbol=symbol,
            timeframe=timeframe.name,
            bars=int(len(index)),
            measured_bars=0,
            assumed_bars=int(len(index)),
            m1_window_start=stored.window_start if stored else None,
            m1_window_end=stored.window_end if stored else None,
        )

    covered = set(
        pd.DatetimeIndex(m1.index).floor(f"{timeframe.minutes}min").unique()
    )
    measured = int(sum(1 for moment in index if moment in covered))
    return SpreadCoverage(
        symbol=symbol,
        timeframe=timeframe.name,
        bars=int(len(index)),
        measured_bars=measured,
        assumed_bars=int(len(index)) - measured,
        m1_window_start=m1.index[0].to_pydatetime(),
        m1_window_end=m1.index[-1].to_pydatetime(),
    )


def attach(
    cache: ParquetCache,
    symbol: str,
    timeframe: Timeframe,
    bars: pd.DataFrame,
    quantile: float = DEFAULT_PER_BAR_QUANTILE,
) -> pd.DataFrame:
    """`bars` with a `spread` column reconstructed from M1. Never in place.

    Above M1 a `spread` column exists only because this function put it
    there. That is the invariant the cost model relies on: it refuses to
    charge per-bar when the column is absent, so no path can charge the
    aggregated minimum by accident.
    """
    if timeframe.minutes <= MEANINGFUL_TIMEFRAME.minutes or bars.empty:
        return bars
    series = reconstructed_series(
        cache, symbol, timeframe, pd.DatetimeIndex(bars.index), quantile
    )
    out = bars.copy()
    out["spread"] = series
    logger.info(
        "%s %s: per-bar spread reconstructed from M1 at quantile %.2f "
        "(median %.1f points against a column median of %.1f)",
        symbol,
        timeframe.name,
        quantile,
        float(series.median()),
        float(bars[MIN_SPREAD_M1_COLUMN].median())
        if MIN_SPREAD_M1_COLUMN in bars.columns
        else float("nan"),
    )
    return out
