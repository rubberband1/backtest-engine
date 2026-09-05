from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from core.data.cache import (
    ParquetCache,
    merge_intervals,
    split_by_year,
    subtract_intervals,
)
from core.data.provider import (
    BAR_COLUMNS,
    MIN_SPREAD_M1_COLUMN,
    Timeframe,
    rename_aggregated_spread,
    spread_column_for,
)

TF = Timeframe.M1
SYMBOL = "TEST.SYM"


def utc(*args: int) -> datetime:
    return datetime(*args, tzinfo=timezone.utc)


def make_bars(start: datetime, end: datetime, timeframe: Timeframe = TF) -> pd.DataFrame:
    index = pd.date_range(start, end, freq=timeframe.pandas_freq, tz="UTC", inclusive="left")
    index.name = "time"
    n = len(index)
    return pd.DataFrame(
        {
            "open": [1.0] * n,
            "high": [2.0] * n,
            "low": [0.5] * n,
            "close": [1.5] * n,
            "tick_volume": [10.0] * n,
            "spread": [3.0] * n,
            "real_volume": [0.0] * n,
        },
        index=index,
    )


def test_subtract_intervals_leaves_the_holes() -> None:
    target = (utc(2024, 1, 1), utc(2024, 1, 10))
    covered = [(utc(2024, 1, 2), utc(2024, 1, 4)), (utc(2024, 1, 6), utc(2024, 1, 7))]
    assert subtract_intervals(target, covered) == [
        (utc(2024, 1, 1), utc(2024, 1, 2)),
        (utc(2024, 1, 4), utc(2024, 1, 6)),
        (utc(2024, 1, 7), utc(2024, 1, 10)),
    ]


def test_merge_intervals_joins_contiguous() -> None:
    merged = merge_intervals(
        [(utc(2024, 1, 3), utc(2024, 1, 5)), (utc(2024, 1, 1), utc(2024, 1, 3))]
    )
    assert merged == [(utc(2024, 1, 1), utc(2024, 1, 5))]


def test_split_by_year_cuts_on_the_boundary() -> None:
    pieces = split_by_year((utc(2023, 12, 30), utc(2024, 1, 3)))
    assert pieces == [
        (2023, (utc(2023, 12, 30), utc(2024, 1, 1))),
        (2024, (utc(2024, 1, 1), utc(2024, 1, 3))),
    ]


def test_get_or_fetch_downloads_only_the_holes(tmp_path: Path) -> None:
    cache = ParquetCache(tmp_path)
    calls: list[tuple[datetime, datetime]] = []

    def fetch(symbol: str, tf: Timeframe, start: datetime, end: datetime) -> pd.DataFrame:
        calls.append((start, end))
        return make_bars(start, end, tf)

    first = cache.get_or_fetch(SYMBOL, TF, utc(2024, 1, 1), utc(2024, 1, 2), fetch)
    assert len(first) == 24 * 60
    assert calls == [(utc(2024, 1, 1), utc(2024, 1, 2))]

    # the second request extends the range: it must only download the tail
    calls.clear()
    second = cache.get_or_fetch(SYMBOL, TF, utc(2024, 1, 1), utc(2024, 1, 3), fetch)
    assert calls == [(utc(2024, 1, 2), utc(2024, 1, 3))]
    assert len(second) == 2 * 24 * 60
    assert list(second.columns) == list(BAR_COLUMNS)
    assert second.index.is_monotonic_increasing
    assert not second.index.duplicated().any()

    # an already covered request does not touch the provider
    calls.clear()
    cached = cache.get_or_fetch(SYMBOL, TF, utc(2024, 1, 1, 12), utc(2024, 1, 2, 12), fetch)
    assert calls == []
    assert len(cached) == 24 * 60


def test_inner_hole_is_not_assumed_covered(tmp_path: Path) -> None:
    cache = ParquetCache(tmp_path)

    def fetch(symbol: str, tf: Timeframe, start: datetime, end: datetime) -> pd.DataFrame:
        return make_bars(start, end, tf)

    cache.get_or_fetch(SYMBOL, TF, utc(2024, 1, 1), utc(2024, 1, 2), fetch)
    cache.get_or_fetch(SYMBOL, TF, utc(2024, 1, 5), utc(2024, 1, 6), fetch)

    holes = cache.missing_ranges(SYMBOL, TF, utc(2024, 1, 1), utc(2024, 1, 6))
    assert holes == [(utc(2024, 1, 2), utc(2024, 1, 5))]


def test_request_straddling_two_years(tmp_path: Path) -> None:
    cache = ParquetCache(tmp_path)

    def fetch(symbol: str, tf: Timeframe, start: datetime, end: datetime) -> pd.DataFrame:
        return make_bars(start, end, tf)

    bars = cache.get_or_fetch(SYMBOL, Timeframe.H1, utc(2023, 12, 31), utc(2024, 1, 2), fetch)
    assert len(bars) == 48
    assert cache.paths(SYMBOL, Timeframe.H1, 2023)[0].exists()
    assert cache.paths(SYMBOL, Timeframe.H1, 2024)[0].exists()


def test_metadata_written_next_to_the_parquet(tmp_path: Path) -> None:
    cache = ParquetCache(tmp_path)
    cache.get_or_fetch(
        SYMBOL,
        TF,
        utc(2024, 1, 1),
        utc(2024, 1, 2),
        lambda s, tf, a, b: make_bars(a, b, tf),
        server_timezone="Europe/Athens",
    )
    meta = cache.read_meta(SYMBOL, TF, 2024)
    assert meta is not None
    assert meta.server_timezone == "Europe/Athens"
    assert meta.rows == 24 * 60
    assert meta.coverage == [(utc(2024, 1, 1), utc(2024, 1, 2))]
    assert meta.downloaded_at is not None
    assert meta.downloaded_at.tzinfo is not None


def test_explicit_invalidation(tmp_path: Path) -> None:
    cache = ParquetCache(tmp_path)
    fetch = lambda s, tf, a, b: make_bars(a, b, tf)  # noqa: E731
    cache.get_or_fetch(SYMBOL, TF, utc(2024, 1, 1), utc(2024, 1, 2), fetch)

    removed = cache.invalidate(SYMBOL, TF, 2024)
    assert len(removed) == 2
    assert cache.read_meta(SYMBOL, TF, 2024) is None
    assert cache.missing_ranges(SYMBOL, TF, utc(2024, 1, 1), utc(2024, 1, 2)) == [
        (utc(2024, 1, 1), utc(2024, 1, 2))
    ]


def test_empty_range_does_not_download(tmp_path: Path) -> None:
    cache = ParquetCache(tmp_path)
    calls: list[int] = []

    def fetch(symbol: str, tf: Timeframe, start: datetime, end: datetime) -> pd.DataFrame:
        calls.append(1)
        return make_bars(start, end, tf)

    out = cache.get_or_fetch(SYMBOL, TF, utc(2024, 1, 2), utc(2024, 1, 1), fetch)
    assert out.empty
    assert calls == []


def test_naive_input_treated_as_utc(tmp_path: Path) -> None:
    cache = ParquetCache(tmp_path)
    cache.get_or_fetch(
        SYMBOL,
        Timeframe.H1,
        datetime(2024, 1, 1),
        datetime(2024, 1, 2),
        lambda s, tf, a, b: make_bars(a, b, tf),
    )
    assert cache.missing_ranges(
        SYMBOL, Timeframe.H1, utc(2024, 1, 1), utc(2024, 1, 2)
    ) == []


def test_hole_without_data_stays_marked_covered(tmp_path: Path) -> None:
    """An interval genuinely empty on the broker must not be re-downloaded every time."""
    cache = ParquetCache(tmp_path)
    empty_calls: list[int] = []

    def fetch(symbol: str, tf: Timeframe, start: datetime, end: datetime) -> pd.DataFrame:
        empty_calls.append(1)
        return make_bars(start, start, tf)

    cache.get_or_fetch(SYMBOL, TF, utc(2024, 1, 6), utc(2024, 1, 7), fetch)
    cache.get_or_fetch(SYMBOL, TF, utc(2024, 1, 6), utc(2024, 1, 7), fetch)
    assert len(empty_calls) == 1


@pytest.mark.parametrize("timeframe", [Timeframe.M1, Timeframe.M15, Timeframe.H1])
def test_parquet_roundtrip_preserves_utc_and_spread(tmp_path: Path, timeframe: Timeframe) -> None:
    """The values survive the round trip; above M1 the name changes on purpose.

    A `spread` column at M15 or H1 would be the minimum of the M1 spreads
    inside the bar wearing the name of a cost. The cache is where that name
    is corrected, so the round trip is expected to rename and not expected to
    round-trip the header.
    """
    cache = ParquetCache(tmp_path)
    bars = make_bars(utc(2024, 3, 1), utc(2024, 3, 2), timeframe)
    cache.write_year(SYMBOL, timeframe, 2024, bars, [(utc(2024, 3, 1), utc(2024, 3, 2))])
    back = cache.read_year(SYMBOL, timeframe, 2024)
    assert str(back.index.tz) == "UTC"

    field = spread_column_for(timeframe)
    assert field == ("spread" if timeframe is Timeframe.M1 else MIN_SPREAD_M1_COLUMN)
    assert (back[field] == 3.0).all()
    pd.testing.assert_frame_equal(
        back, rename_aggregated_spread(bars, timeframe), check_freq=False
    )
