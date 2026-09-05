"""Quality checks on historical series.

Ground rule: nothing is corrected and nothing is discarded here. Measure and
report. Deciding what to do with a hole or a malformed bar is the job of the
layer that uses the data, not the one that reads it.

The session calendar is not hardcoded: it is derived from the data itself by
counting, slot by slot of the week, how often that minute is quoted. A market
that never has bars on Saturday will produce inactive slots without anyone
writing "Saturday" anywhere.

**Which clock that counting runs on matters, and on long history it matters a
lot.** A broker session is fixed in the server's local time: the daily bar
opens at 00:00 server, every day of the year. In UTC that same bar moves by
an hour twice a year, so a weekly grid built in UTC finds the winter slot
active and declares the whole summer missing. On EURUSD D1 over 27 years that
reported 40.8% completeness for a series with no real holes at all. Passing
`server_tz` makes the grid follow the server clock and the artefact
disappears; leaving it out keeps the previous UTC behaviour. Either way the
report states which clock it used, because a completeness figure without that
is not interpretable.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, tzinfo

import numpy as np
import pandas as pd

from core.data.provider import MIN_SPREAD_M1_COLUMN, Timeframe

logger = logging.getLogger(__name__)

MINUTES_PER_WEEK = 7 * 24 * 60


@dataclass(frozen=True)
class Gap:
    """Interval of expected but absent bars, within the session."""

    start: datetime
    end: datetime
    missing_bars: int

    @property
    def duration(self) -> timedelta:
        return self.end - self.start


@dataclass
class QualityReport:
    """Outcome of the checks. No data is modified."""

    symbol: str
    timeframe: str
    rows: int
    first_bar: datetime | None
    last_bar: datetime | None
    duplicate_timestamps: int
    unsorted: bool
    nan_counts: dict[str, int]
    invalid_high_low: int
    invalid_range: int
    zero_tick_volume: int
    non_positive_prices: int
    zero_spread: int
    active_slots: int
    total_slots: int
    session_confidence: str
    session_clock: str = "UTC"
    gaps: list[Gap] = field(default_factory=list)
    missing_bars: int = 0
    expected_bars: int = 0
    samples: dict[str, list[str]] = field(default_factory=dict)

    @property
    def completeness(self) -> float:
        if not self.expected_bars:
            return 0.0
        return 1.0 - self.missing_bars / self.expected_bars

    @property
    def is_clean(self) -> bool:
        return (
            self.duplicate_timestamps == 0
            and not self.unsorted
            and sum(self.nan_counts.values()) == 0
            and self.invalid_high_low == 0
            and self.invalid_range == 0
            and self.non_positive_prices == 0
            and not self.gaps
        )

    def as_text(self, max_gaps: int = 10) -> str:
        lines = [
            f"Data quality - {self.symbol} {self.timeframe}",
            f"  bars                : {self.rows}",
            f"  range (UTC)         : {self.first_bar} -> {self.last_bar}",
            f"  inferred session    : {self.active_slots}/{self.total_slots} weekly slots "
            f"on the {self.session_clock} clock "
            f"(confidence: {self.session_confidence})",
            f"  expected in session : {self.expected_bars}",
            f"  missing             : {self.missing_bars} "
            f"(completeness {self.completeness:.4%})",
            f"  gaps                : {len(self.gaps)}",
            f"  duplicate timestamps: {self.duplicate_timestamps}",
            f"  unsorted index      : {self.unsorted}",
            f"  NaN per column      : "
            f"{ {k: v for k, v in self.nan_counts.items() if v} or 'none'}",
            f"  high < low          : {self.invalid_high_low}",
            f"  OHLC out of range   : {self.invalid_range}",
            f"  prices <= 0         : {self.non_positive_prices}",
            f"  tick_volume == 0    : {self.zero_tick_volume}",
            f"  spread == 0         : {self.zero_spread}",
        ]
        for gap in self.gaps[:max_gaps]:
            lines.append(
                f"    gap {gap.start} -> {gap.end} "
                f"({gap.missing_bars} bars, {gap.duration})"
            )
        if len(self.gaps) > max_gaps:
            lines.append(f"    ... {len(self.gaps) - max_gaps} more gaps")
        return "\n".join(lines)


def _zero_spread(bars: pd.DataFrame) -> int:
    """Bars whose spread field is zero, under either of the names it has.

    Above M1 the field is `min_spread_m1` and a zero there means the best
    minute of the bar was quoted at zero - which is the feed saying nothing,
    not a free trade. Counting it is the point; charging it is what
    `core.engine.costs` refuses.
    """
    for column in ("spread", MIN_SPREAD_M1_COLUMN):
        if column in bars.columns:
            return int((bars[column] == 0).sum())
    return 0


def _slot_ids(index: pd.DatetimeIndex, timeframe: Timeframe) -> np.ndarray:
    """Weekly slot identifier of each timestamp, on the index's own clock."""
    minute_of_week = index.dayofweek * 24 * 60 + index.hour * 60 + index.minute
    return (minute_of_week // timeframe.minutes).to_numpy()


def on_session_clock(
    index: pd.DatetimeIndex, server_tz: tzinfo | None
) -> pd.DatetimeIndex:
    """The index as naive wall-clock time on the clock the session lives on.

    Dropping the offset after converting is what removes DST from the
    picture: on the server clock a session slot is the same slot all year,
    which is the whole point of using it.
    """
    if server_tz is None:
        return index.tz_convert("UTC").tz_localize(None)
    return index.tz_convert(server_tz).tz_localize(None)


def infer_session_slots(
    index: pd.DatetimeIndex,
    timeframe: Timeframe,
    threshold: float = 0.5,
    server_tz: tzinfo | None = None,
) -> tuple[np.ndarray, str]:
    """Weekly slots in which the instrument is quoted.

    A slot counts as active if it appears in at least `threshold` of the
    weeks covered by the sample. Also returns a confidence rating: with few
    weeks the inferred calendar is not worth much.

    `server_tz` is the clock the slots are counted on. Left out, UTC is used.
    """
    total_slots = MINUTES_PER_WEEK // timeframe.minutes
    if len(index) == 0:
        return np.zeros(0, dtype="int64"), "no data"

    span_weeks = (index[-1] - index[0]).total_seconds() / (7 * 86400)
    weeks = max(1, int(np.ceil(span_weeks)))
    counts = np.bincount(
        _slot_ids(on_session_clock(index, server_tz), timeframe), minlength=total_slots
    )
    active = np.flatnonzero(counts >= max(1.0, threshold * weeks))

    if weeks < 4:
        confidence = "low (less than 4 weeks of history)"
    elif weeks < 12:
        confidence = "medium"
    else:
        confidence = "high"
    return active, confidence


def _find_gaps(
    index: pd.DatetimeIndex,
    timeframe: Timeframe,
    active_slots: np.ndarray,
    server_tz: tzinfo | None = None,
) -> tuple[list[Gap], int, int]:
    """Bars the session says should exist and the data does not have.

    The grid is laid out on the session clock, as naive timestamps, so that a
    DST change does not shift every slot by an hour and turn half a year of
    perfectly good bars into a gap. The reported gap boundaries are converted
    back, so the caller still sees real instants.
    """
    local = on_session_clock(index, server_tz)
    step = pd.Timedelta(minutes=timeframe.minutes)
    grid = pd.date_range(local[0], local[-1], freq=step)
    in_session = np.isin(_slot_ids(grid, timeframe), active_slots)
    expected = grid[in_session]
    missing = expected.difference(local)
    if len(missing) == 0:
        return [], 0, len(expected)

    # Group consecutive missing timestamps on the grid into a single gap.
    positions = expected.get_indexer(missing)
    breaks = np.flatnonzero(np.diff(positions) != 1)
    starts = np.concatenate(([0], breaks + 1))
    ends = np.concatenate((breaks, [len(missing) - 1]))

    def back(moment: pd.Timestamp) -> datetime:
        if server_tz is None:
            return moment.tz_localize("UTC").to_pydatetime()
        return (
            moment.tz_localize(server_tz, ambiguous=True, nonexistent="shift_forward")
            .tz_convert("UTC")
            .to_pydatetime()
        )

    gaps = [
        Gap(
            start=back(missing[s]),
            end=back(missing[e] + step),
            missing_bars=int(e - s + 1),
        )
        for s, e in zip(starts, ends)
    ]
    return gaps, len(missing), len(expected)


def check_quality(
    bars: pd.DataFrame,
    symbol: str,
    timeframe: Timeframe | str,
    session_threshold: float = 0.5,
    max_samples: int = 5,
    server_tz: tzinfo | None = None,
) -> QualityReport:
    """Analyzes a frame of bars and returns the structured report.

    `server_tz` is the clock the trading session runs on. Without it the
    session grid is built in UTC, which on history spanning a DST change
    reports large gaps that are not there.
    """
    tf = Timeframe.parse(timeframe)
    total_slots = MINUTES_PER_WEEK // tf.minutes
    clock = "UTC" if server_tz is None else f"server ({server_tz})"

    if bars.empty:
        logger.warning("no bars to analyze for %s %s", symbol, tf.name)
        return QualityReport(
            symbol=symbol,
            timeframe=tf.name,
            rows=0,
            first_bar=None,
            last_bar=None,
            duplicate_timestamps=0,
            unsorted=False,
            nan_counts={},
            invalid_high_low=0,
            invalid_range=0,
            zero_tick_volume=0,
            non_positive_prices=0,
            zero_spread=0,
            active_slots=0,
            total_slots=total_slots,
            session_confidence="no data",
            session_clock=clock,
        )

    index = pd.DatetimeIndex(bars.index)
    if index.tz is None:
        raise ValueError("the bar index must be tz-aware in UTC")

    unsorted = not index.is_monotonic_increasing
    duplicates = int(index.duplicated().sum())

    ohlc = [c for c in ("open", "high", "low", "close") if c in bars.columns]
    high_low = (
        bars["high"] < bars["low"]
        if {"high", "low"} <= set(bars.columns)
        else pd.Series(False, index=bars.index)
    )
    if {"open", "high", "low", "close"} <= set(bars.columns):
        body_max = bars[["open", "close"]].max(axis=1)
        body_min = bars[["open", "close"]].min(axis=1)
        out_of_range = (bars["high"] < body_max) | (bars["low"] > body_min)
    else:
        out_of_range = pd.Series(False, index=bars.index)
    non_positive = (bars[ohlc] <= 0).any(axis=1) if ohlc else pd.Series(False, index=bars.index)

    # Gap analysis needs a sorted, deduplicated index; the anomaly counts
    # above stay on the raw data instead.
    clean_index = index[~index.duplicated(keep="last")].sort_values()
    active_slots, confidence = infer_session_slots(
        clean_index, tf, session_threshold, server_tz
    )
    gaps, missing_bars, expected_bars = _find_gaps(
        clean_index, tf, active_slots, server_tz
    )

    samples = {
        "duplicates": [str(t) for t in index[index.duplicated()][:max_samples]],
        "high_lt_low": [str(t) for t in bars.index[high_low][:max_samples]],
        "ohlc_out_of_range": [str(t) for t in bars.index[out_of_range][:max_samples]],
        "non_positive_prices": [str(t) for t in bars.index[non_positive][:max_samples]],
    }

    report = QualityReport(
        symbol=symbol,
        timeframe=tf.name,
        rows=int(len(bars)),
        first_bar=index.min().to_pydatetime(),
        last_bar=index.max().to_pydatetime(),
        duplicate_timestamps=duplicates,
        unsorted=unsorted,
        nan_counts={str(c): int(bars[c].isna().sum()) for c in bars.columns},
        invalid_high_low=int(high_low.sum()),
        invalid_range=int(out_of_range.sum()),
        zero_tick_volume=(
            int((bars["tick_volume"] == 0).sum()) if "tick_volume" in bars.columns else 0
        ),
        non_positive_prices=int(non_positive.sum()),
        zero_spread=_zero_spread(bars),
        active_slots=int(len(active_slots)),
        total_slots=total_slots,
        session_confidence=confidence,
        session_clock=clock,
        gaps=gaps,
        missing_bars=missing_bars,
        expected_bars=expected_bars,
        samples={k: v for k, v in samples.items() if v},
    )
    logger.info(
        "%s %s: %d bars, %d gaps, completeness %.4f",
        symbol,
        tf.name,
        report.rows,
        len(report.gaps),
        report.completeness,
    )
    return report
