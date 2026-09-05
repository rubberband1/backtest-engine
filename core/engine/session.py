"""Session bar counter.

A "60 bars" time stop must mean 60 bars of open market, not 60 array rows: if
two hours of quotes are missing in between, counting rows makes the trade
last much longer than the strategy intended. The session calendar comes from
phase 1, so it is derived from the data and not written by hand.

Derived from the data is not the same as derived from *this* data. Inferring
the calendar from whatever frame is in hand makes the ordinal of a bar depend
on which bars came with it: harmless in a backtest over a fixed period, wrong
the moment a live runner has to count the same session bars one bar at a
time. `SessionCalendar` therefore separates the inference, which needs a
sample, from the counting, which does not - a runner infers once from history,
pins the result, and then agrees with a backtest over the same period by
construction rather than by luck.

**Which clock the slots are counted on decides whether the count is right.**
A broker session is fixed in the server's local time: the day opens at 00:00
server, every day of the year. In UTC that same moment moves by an hour
twice a year, so a weekly grid laid out in UTC has slots that are active for
part of the year and inactive for the rest, and the threshold picks one of
the two regimes and calls the other a hole. Around a DST change the count of
"session bars since entry" then drifts by up to a full session of bars, and a
60-bar time stop fires on the wrong bar for the days on either side of it.

The grid is therefore laid out in UTC - which has no missing or repeated
instants - and each of its instants is assigned the weekly slot it falls in
**on the server clock**. The spring hour that never happens contributes no
expected bars, the autumn hour that happens twice contributes both, and a
slot is one slot all year. This is the same correction phase 6 applied to
the quality report, arriving where it decides a trade rather than a
completeness figure.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, tzinfo
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from core.data.provider import Timeframe
from core.data.quality import _slot_ids, infer_session_slots
from core.serialization import json_safe

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SessionCalendar:
    """Which weekly slots the market is open in, plus where counting starts.

    `origin` is the timestamp ordinal 0 is measured from. Only differences
    between ordinals are ever used, so the choice of origin cannot change a
    time stop - but it has to be the same on both sides of a comparison, so
    it is part of the pinned object rather than of the caller's memory.
    """

    timeframe: str
    active_slots: tuple[int, ...]
    origin: datetime
    threshold: float
    confidence: str = "pinned"
    # The clock the weekly slots are counted on, as an IANA name. None keeps
    # the pre-4.0 behaviour of counting in UTC, which a DST change shifts by
    # an hour twice a year; it survives only so that a calendar pinned by an
    # older runner still loads and still says what it did.
    server_timezone: str | None = None
    # The expected-bar grid, built once and extended by doubling. Excluded
    # from equality and from the serialized form: it is a memo of a pure
    # function of the fields above, and two calendars that agree on those
    # agree on every ordinal they will ever produce. It exists because a
    # streaming caller asks for one ordinal per bar, and rebuilding the grid
    # from the origin each time made a replay quadratic in its own history.
    _grid: dict[str, Any] = field(
        default_factory=dict, compare=False, repr=False, hash=False
    )

    @property
    def tf(self) -> Timeframe:
        return Timeframe.parse(self.timeframe)

    @property
    def clock(self) -> tzinfo | None:
        return ZoneInfo(self.server_timezone) if self.server_timezone else None

    @classmethod
    def infer(
        cls,
        index: pd.DatetimeIndex,
        timeframe: Timeframe,
        threshold: float = 0.5,
        server_tz: tzinfo | None = None,
    ) -> SessionCalendar:
        active, confidence = infer_session_slots(
            index, timeframe, threshold, server_tz
        )
        return cls(
            timeframe=timeframe.name,
            active_slots=tuple(int(slot) for slot in active),
            origin=index[0].to_pydatetime(),
            threshold=threshold,
            confidence=confidence,
            server_timezone=str(server_tz) if server_tz is not None else None,
        )

    def _expected(self, end: pd.Timestamp) -> np.ndarray:
        """UTC nanoseconds of every expected in-session bar up to `end`.

        Extending the horizon never changes an ordinal already computed: the
        count is of grid points at or before a timestamp, and points added
        beyond it are not at or before anything that was asked about. So the
        grid is grown by doubling and reused, which turns one rebuild per bar
        into a handful of rebuilds per run.
        """
        tf = self.tf
        origin = pd.Timestamp(self.origin)
        horizon: pd.Timestamp | None = self._grid.get("horizon")
        if horizon is not None and end <= horizon:
            return self._grid["expected"]

        step = pd.Timedelta(minutes=tf.minutes)
        span = max(end - origin, step)
        if horizon is not None:
            span = max(span, 2 * (horizon - origin))
        target = origin + span

        # the grid is in UTC because UTC has every instant exactly once; the
        # slot each instant belongs to is read off the session clock, which
        # is where the session is actually defined
        grid = pd.date_range(origin, target, freq=tf.pandas_freq, tz="UTC")
        clock = self.clock
        local = grid.tz_convert(clock) if clock is not None else grid
        expected = grid[np.isin(_slot_ids(local, tf), np.asarray(self.active_slots))]

        self._grid["horizon"] = target
        self._grid["expected"] = expected.asi8
        return self._grid["expected"]

    def ordinals(self, index: pd.DatetimeIndex) -> np.ndarray:
        """Position of every timestamp on the grid of expected in-session bars."""
        if len(index) == 0:
            return np.zeros(0, dtype="int64")
        if not self.active_slots:
            return np.arange(len(index), dtype="int64")

        expected = self._expected(max(index[-1], pd.Timestamp(self.origin)))
        if len(expected) == 0:
            return np.arange(len(index), dtype="int64")
        # searchsorted instead of get_indexer: out-of-session bars (rare but
        # they exist) are not on the grid and must map to the previous slot
        return np.searchsorted(expected, index.asi8, side="right").astype("int64")

    def ordinal(self, moment: datetime) -> int:
        """The ordinal of a single bar. What a streaming caller needs."""
        if not self.active_slots:
            return 0
        stamp = pd.Timestamp(moment)
        expected = self._expected(max(stamp, pd.Timestamp(self.origin)))
        if len(expected) == 0:
            return 0
        return int(np.searchsorted(expected, stamp.value, side="right"))

    def to_dict(self) -> dict[str, Any]:
        return json_safe(
            {
                "timeframe": self.timeframe,
                "active_slots": list(self.active_slots),
                "origin": self.origin,
                "threshold": self.threshold,
                "confidence": self.confidence,
                "server_timezone": self.server_timezone,
            }
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SessionCalendar:
        return cls(
            timeframe=str(payload["timeframe"]),
            active_slots=tuple(int(s) for s in payload["active_slots"]),
            origin=datetime.fromisoformat(str(payload["origin"])),
            threshold=float(payload["threshold"]),
            confidence=str(payload.get("confidence", "pinned")),
            server_timezone=(
                str(payload["server_timezone"])
                if payload.get("server_timezone")
                else None
            ),
        )


def session_ordinals(
    index: pd.DatetimeIndex,
    timeframe: Timeframe,
    threshold: float = 0.5,
    calendar: SessionCalendar | None = None,
    server_tz: tzinfo | None = None,
) -> np.ndarray:
    """Position of every bar on the grid of expected in-session bars.

    The difference between two ordinals is the number of open-market bars
    between them, including the ones missing from the data.

    Without a `calendar` the session is inferred from `index` itself on
    `server_tz`, and the grid starts at its first bar. With one, the counting
    is that calendar's - including the clock it was inferred on - and does not
    depend on the sample.
    """
    if len(index) == 0:
        return np.zeros(0, dtype="int64")
    if calendar is not None:
        if calendar.timeframe != timeframe.name:
            raise ValueError(
                f"session calendar is for {calendar.timeframe}, asked to count "
                f"{timeframe.name} bars"
            )
        return calendar.ordinals(index)
    return SessionCalendar.infer(index, timeframe, threshold, server_tz).ordinals(index)
