from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from core.data.servertime import (
    measure_offset,
    resolve_timezone,
    server_epoch_to_utc_index,
    server_naive_to_utc,
    utc_to_broker_datetime,
    utc_to_server_naive,
)

ATHENS = ZoneInfo("Europe/Athens")


@pytest.mark.parametrize(
    "moment",
    [
        datetime(2025, 1, 15, 9, 30, tzinfo=timezone.utc),  # standard time, UTC+2
        datetime(2025, 7, 15, 9, 30, tzinfo=timezone.utc),  # daylight saving time, UTC+3
        datetime(2025, 3, 30, 5, 0, tzinfo=timezone.utc),  # right after the switch
        datetime(2025, 10, 26, 5, 0, tzinfo=timezone.utc),
    ],
)
def test_roundtrip_utc_server_utc(moment: datetime) -> None:
    server_naive = utc_to_server_naive(moment, ATHENS)
    assert server_naive.tzinfo is None
    assert server_naive_to_utc(server_naive, ATHENS) == moment


def test_offset_follows_dst() -> None:
    winter = datetime(2025, 1, 15, 12, 0, tzinfo=timezone.utc)
    summer = datetime(2025, 7, 15, 12, 0, tzinfo=timezone.utc)
    assert utc_to_server_naive(winter, ATHENS).hour == 14
    assert utc_to_server_naive(summer, ATHENS).hour == 15


def test_measure_offset_ignores_latency() -> None:
    reference = datetime(2025, 7, 15, 12, 0, 0, tzinfo=timezone.utc)
    # the server shows 15:00:03: three seconds of latency must not move the offset
    server_wall = datetime(2025, 7, 15, 15, 0, 3)
    assert measure_offset(server_wall, reference) == timedelta(hours=3)


def test_measure_offset_supports_half_hours() -> None:
    reference = datetime(2025, 7, 15, 12, 0, tzinfo=timezone.utc)
    assert measure_offset(datetime(2025, 7, 15, 17, 30), reference) == timedelta(hours=5, minutes=30)


def test_resolve_timezone_finds_iana_zone() -> None:
    summer = datetime(2025, 7, 15, 12, 0, tzinfo=timezone.utc)
    zone = resolve_timezone(timedelta(hours=3), at=summer, candidates=("UTC", "Europe/Athens"))
    assert zone == ZoneInfo("Europe/Athens")
    # the resolved zone must then reconstruct winter correctly
    winter = datetime(2025, 1, 15, 12, 0, tzinfo=timezone.utc)
    assert winter.astimezone(zone).utcoffset() == timedelta(hours=2)


def test_resolve_timezone_falls_back_to_fixed_offset() -> None:
    zone = resolve_timezone(timedelta(hours=7), at=datetime(2025, 7, 15, tzinfo=timezone.utc),
                            candidates=("UTC", "Europe/Athens"))
    assert zone.utcoffset(None) == timedelta(hours=7)


def test_server_encoded_epoch_becomes_real_utc() -> None:
    # the feed sends the 15:00 epoch "as if" it were UTC, but it is 15:00 in Athens
    server_wall = datetime(2025, 7, 15, 15, 0)
    epoch = np.array([int((server_wall - datetime(1970, 1, 1)).total_seconds())])
    index = server_epoch_to_utc_index(epoch, ATHENS)
    assert index[0] == pd.Timestamp("2025-07-15 12:00", tz="UTC")
    assert index.name == "time"


def test_broker_datetime_is_tz_aware_with_server_clock() -> None:
    """The bound passed to the SDK must not depend on the machine timezone.

    MetaTrader5 converts naive datetimes with the PC's local time: a naive
    bound would silently shift the requested window by hours.
    """
    moment = datetime(2026, 8, 31, 10, 44, tzinfo=timezone.utc)
    marker = utc_to_broker_datetime(moment, ATHENS)
    assert marker.tzinfo is timezone.utc
    assert marker.replace(tzinfo=None) == utc_to_server_naive(moment, ATHENS)
    assert marker.hour == 13  # server wall clock, not real UTC
