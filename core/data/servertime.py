"""Conversions between broker server time and UTC.

The critical point: MetaTrader (and several other feeds) exposes timestamps
as epoch seconds that actually encode **the server's wall clock** as if it
were UTC. If the server clock shows 10:00 and sits at UTC+3, the received
value is the epoch of 10:00 UTC, not of 07:00 UTC. The same holds in the
other direction: a naive `datetime` passed to the terminal is read as server
time.

This module imports no broker SDK: it is pure, testable logic.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone, tzinfo
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np
import pandas as pd
from pytz.exceptions import AmbiguousTimeError

logger = logging.getLogger(__name__)

# A set of candidates, not an imposed default: the actual zone is chosen by
# matching the offset measured on the feed. The caller may extend it.
DEFAULT_CANDIDATE_ZONES: tuple[str, ...] = (
    "UTC",
    "Europe/London",
    "Europe/Athens",
    "Europe/Helsinki",
    "Europe/Riga",
    "Europe/Nicosia",
    "Europe/Moscow",
    "Europe/Berlin",
    "America/New_York",
    "Asia/Dubai",
    "Australia/Sydney",
)

# Broker offsets are always multiples of half an hour.
_OFFSET_QUANTUM = timedelta(minutes=30)

# How far a raw measurement may sit from a half-hour multiple before it is
# refused. A live tick is a second or two behind the server clock, so the
# residual is negligible; a tick from a closed market is as old as the
# session has been shut, and rounding hides that age inside the quantum. On
# this broker, with the market closed for fifty minutes, the last tick made
# an Athens server (UTC+3) measure 2h09 and round to UTC+2 - Berlin, an hour
# out, silently, in the value the session calendar and the swap accounting
# are both built on.
MAX_OFFSET_RESIDUAL = timedelta(minutes=2)


def quantize_offset(delta: timedelta) -> timedelta:
    """Rounds a measured offset to the nearest multiple of 30 minutes."""
    quantum = _OFFSET_QUANTUM.total_seconds()
    return timedelta(seconds=round(delta.total_seconds() / quantum) * quantum)


def offset_residual(delta: timedelta) -> timedelta:
    """How far a raw offset sits from the nearest half hour, as a magnitude.

    Large means the two instants were not simultaneous, which means the
    measurement is of a tick's age and not of a timezone.
    """
    return abs(delta - quantize_offset(delta))


def offset_is_measurable(delta: timedelta) -> bool:
    """Whether a raw offset can be trusted to name a timezone."""
    return offset_residual(delta) <= MAX_OFFSET_RESIDUAL


def raw_offset(server_wall_clock: datetime, reference_utc: datetime) -> timedelta:
    """The unrounded difference between the two instants."""
    a = server_wall_clock.replace(tzinfo=None)
    b = reference_utc.astimezone(timezone.utc).replace(tzinfo=None)
    return a - b


def measure_offset(server_wall_clock: datetime, reference_utc: datetime) -> timedelta:
    """Server offset from UTC, derived from two simultaneous instants.

    `server_wall_clock` is the time shown by the server at the same moment
    the local clock shows `reference_utc`. Both are read as naive on their
    respective zone; the result is quantized to 30 minutes because network
    latency must not leak into the offset.

    **Simultaneous is a requirement, not a description.** Rounding a
    difference that is really the age of a stale quote produces a plausible
    offset for the wrong zone, so callers measuring against a feed should
    check `offset_is_measurable` on `raw_offset` before trusting this.
    """
    return quantize_offset(raw_offset(server_wall_clock, reference_utc))


def resolve_timezone(
    offset: timedelta,
    at: datetime | None = None,
    candidates: tuple[str, ...] = DEFAULT_CANDIDATE_ZONES,
) -> tzinfo:
    """Identifies the server timezone given the offset measured now.

    Prefers an IANA zone among the candidates, so that past DST transitions
    are reconstructed correctly over the history. If no candidate matches, it
    falls back to a fixed offset and says so: in that case history straddling
    a clock change will be off by one hour.
    """
    moment = (at or datetime.now(timezone.utc)).astimezone(timezone.utc)
    matches: list[str] = []
    for name in candidates:
        try:
            zone = ZoneInfo(name)
        except ZoneInfoNotFoundError:
            logger.debug("timezone %s not available on this system", name)
            continue
        if moment.astimezone(zone).utcoffset() == offset:
            matches.append(name)

    if not matches:
        logger.warning(
            "no candidate timezone with offset %s: using a fixed offset, "
            "historical DST transitions will not be correct",
            offset,
        )
        return timezone(offset)

    chosen = matches[0]
    if len(matches) > 1:
        logger.info("offset %s compatible with %s, choosing %s", offset, matches, chosen)
    return ZoneInfo(chosen)


def utc_to_server_naive(moment: datetime, server_tz: tzinfo) -> datetime:
    """UTC -> naive datetime in server time, ready for the broker SDK."""
    aware = moment if moment.tzinfo is not None else moment.replace(tzinfo=timezone.utc)
    return aware.astimezone(server_tz).replace(tzinfo=None)


def utc_to_broker_datetime(moment: datetime, server_tz: tzinfo) -> datetime:
    """UTC -> instant to pass to the broker SDK.

    The MetaTrader5 package converts naive `datetime`s using the local
    machine's timezone, not UTC: on a PC not set to UTC the bounds of
    historical requests end up hours off, silently. Passing a tz-aware UTC
    datetime that carries the server's wall clock makes the conversion
    independent of the machine settings.
    """
    return utc_to_server_naive(moment, server_tz).replace(tzinfo=timezone.utc)


def server_naive_to_utc(moment: datetime, server_tz: tzinfo) -> datetime:
    """Naive datetime in server time -> tz-aware UTC."""
    if moment.tzinfo is not None:
        raise ValueError("expected a naive datetime in server time")
    return moment.replace(tzinfo=server_tz).astimezone(timezone.utc)


def _localize(index: pd.DatetimeIndex, server_tz: tzinfo) -> pd.DatetimeIndex:
    try:
        localized = index.tz_localize(server_tz, ambiguous="infer", nonexistent="shift_forward")
    except (AmbiguousTimeError, ValueError):
        # Happens when the sample does not cover enough context around the
        # clock change: 'infer' needs to see the hour actually repeated, and
        # on a sparse or low-frequency series it never does. Markets are
        # closed at that moment, so the impact is marginal, but it must be
        # said. The exception is pytz's, which pandas raises from
        # tz_localize and which does not derive from ValueError.
        logger.warning("ambiguous time in the server DST change: assuming DST")
        localized = index.tz_localize(server_tz, ambiguous=True, nonexistent="shift_forward")
    return localized.tz_convert("UTC")


def server_epoch_to_utc_index(
    seconds: np.ndarray | pd.Series, server_tz: tzinfo
) -> pd.DatetimeIndex:
    """'Server-encoded' epoch -> real UTC DatetimeIndex.

    Values arrive as seconds representing the server clock: they are read as
    naive and then localized onto the server timezone.
    """
    naive = pd.to_datetime(np.asarray(seconds, dtype="int64"), unit="s")
    index = pd.DatetimeIndex(naive, name="time")
    return _localize(index, server_tz)


def server_epoch_ms_to_utc_index(
    millis: np.ndarray | pd.Series, server_tz: tzinfo
) -> pd.DatetimeIndex:
    naive = pd.to_datetime(np.asarray(millis, dtype="int64"), unit="ms")
    index = pd.DatetimeIndex(naive, name="time")
    return _localize(index, server_tz)
