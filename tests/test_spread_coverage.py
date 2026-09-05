"""How much of a run's spread was measured, and how much was assumed.

The cost model charges a spread on every bar. Where M1 exists for the period
that spread is measured; where it does not, a constant taken from a different
period - usually a recent one - is charged backwards. Both come out of the
report as "the spread charged", and until this existed nothing said which of
the two a given run had been given.

The asymmetry is the reason it matters: the oldest bars in a sample are the
ones with no M1 behind them, and they are also the ones added to buy
statistical power. The part of a result standing on an assumed cost is
systematically the part nobody looks at.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from core.data.cache import ParquetCache
from core.data.provider import Timeframe
from core.data.spread import SpreadCoverage, coverage
from core.strategy.binding import bind_cell

SYMBOL = "COVER"
UTC = timezone.utc


def m1_frame(start: datetime, minutes: int, spread: float = 5.0) -> pd.DataFrame:
    index = pd.date_range(start, periods=minutes, freq="1min", tz="UTC", name="time")
    return pd.DataFrame(
        {
            "open": 1.0, "high": 1.1, "low": 0.9, "close": 1.0,
            "tick_volume": 1.0, "spread": spread, "real_volume": 0.0,
        },
        index=index,
    )


def h1_index(start: datetime, hours: int) -> pd.DatetimeIndex:
    return pd.date_range(start, periods=hours, freq="60min", tz="UTC", name="time")


@pytest.fixture
def cache(tmp_path) -> ParquetCache:
    return ParquetCache(tmp_path / "data_cache")


def write_m1(cache: ParquetCache, frame: pd.DataFrame) -> None:
    year = frame.index[0].year
    cache.write_year(
        SYMBOL, Timeframe.M1, year, frame,
        [(frame.index[0].to_pydatetime(), frame.index[-1].to_pydatetime())],
        server_timezone="UTC", source="test",
    )


def test_no_m1_at_all_is_reported_as_wholly_assumed(cache) -> None:
    report = coverage(
        cache, SYMBOL, Timeframe.H1, h1_index(datetime(2024, 1, 1, tzinfo=UTC), 100)
    )
    assert report.bars == 100
    assert report.measured_bars == 0
    assert report.assumed_bars == 100
    assert report.measured_share == 0.0
    assert report.fully_measured is False
    assert "no bar has an M1 sample" in report.verdict


def test_full_m1_cover_is_reported_as_measured(cache) -> None:
    start = datetime(2024, 3, 1, tzinfo=UTC)
    write_m1(cache, m1_frame(start, 10 * 60))

    report = coverage(cache, SYMBOL, Timeframe.H1, h1_index(start, 10))
    assert report.bars == 10
    assert report.measured_bars == 10
    assert report.assumed_bars == 0
    assert report.measured_share == 1.0
    assert report.fully_measured is True
    assert "every bar has an M1 sample" in report.verdict


def test_a_partly_covered_period_reports_the_fraction(cache) -> None:
    """The realistic case: recent M1, older bars with none."""
    start = datetime(2024, 5, 1, tzinfo=UTC)
    # M1 for the last 4 of 20 hours
    write_m1(cache, m1_frame(start + timedelta(hours=16), 4 * 60))

    report = coverage(cache, SYMBOL, Timeframe.H1, h1_index(start, 20))
    assert report.bars == 20
    assert report.measured_bars == 4
    assert report.assumed_bars == 16
    assert report.measured_share == pytest.approx(0.2)
    assert "20.0% of bars" in report.verdict
    assert "16" in report.verdict


def test_a_hole_in_the_m1_sample_counts_as_assumed(cache) -> None:
    """Inside the M1 range is not the same as having M1.

    Counting everything between the first and last M1 timestamp as measured
    would report full coverage over a sample that is mostly holes, which is
    the more flattering answer and the wrong one.
    """
    start = datetime(2024, 7, 1, tzinfo=UTC)
    first = m1_frame(start, 2 * 60)
    later = m1_frame(start + timedelta(hours=8), 2 * 60)
    write_m1(cache, pd.concat([first, later]))

    report = coverage(cache, SYMBOL, Timeframe.H1, h1_index(start, 10))
    assert report.measured_bars == 4, "only the hours with minutes inside them"
    assert report.assumed_bars == 6


def test_an_empty_index_is_not_an_error(cache) -> None:
    report = coverage(
        cache, SYMBOL, Timeframe.H1, pd.DatetimeIndex([], tz="UTC", name="time")
    )
    assert report.bars == 0
    assert report.measured_share == 0.0
    assert report.verdict == "no bars"


def test_the_report_serialises_with_its_verdict() -> None:
    report = SpreadCoverage(
        symbol="X", timeframe="H1", bars=10, measured_bars=3, assumed_bars=7,
        m1_window_start=datetime(2025, 1, 1, tzinfo=UTC),
        m1_window_end=datetime(2025, 2, 1, tzinfo=UTC),
    )
    payload = report.as_dict()
    assert payload["measured_share"] == pytest.approx(0.3)
    assert payload["fully_measured"] is False
    assert payload["verdict"] == report.verdict
    assert payload["m1_window_start"].startswith("2025-01-01")


def test_a_run_records_its_spread_coverage(tmp_path) -> None:
    """End to end: the number reaches the stored run, not just the function."""
    from zoneinfo import ZoneInfo

    from core.data.provider import SymbolSpecSnapshot
    from core.runs.runner import execute_run, load_bars_for_run
    from core.runs.store import RunConfig, RunStore
    from tests.conftest_engine import random_walk, spec_from, symbol_spec

    cache = ParquetCache(tmp_path / "data_cache")
    start = datetime(2024, 9, 2, tzinfo=UTC)

    # 200 H1 bars, with M1 behind only the first 20 of them
    hours = 200
    index = h1_index(start, hours)
    frame = random_walk(hours)
    frame.index = index
    frame["spread"] = 5.0
    cache.write_year(
        SYMBOL, Timeframe.H1, 2024, frame,
        [(index[0].to_pydatetime(), index[-1].to_pydatetime())],
        server_timezone="UTC", source="test",
    )
    write_m1(cache, m1_frame(start, 20 * 60))

    config = RunConfig(
        symbol=SYMBOL, timeframe="H1", spread_mode="fixed", spread_value=5.0
    )
    bars = load_bars_for_run(cache, config)
    store = RunStore(tmp_path / "runs")
    snapshot = SymbolSpecSnapshot(
        spec=symbol_spec(name=SYMBOL), read_at=datetime(2024, 1, 1, tzinfo=UTC)
    )
    meta = execute_run(
        store,
        bind_cell(spec_from(), config.symbol, config.timeframe),
        config, bars, snapshot, ZoneInfo("UTC"),
    )

    stored = (store.load_run(meta.run_id).metrics or {}).get("spread_coverage")
    assert stored is not None, "the run did not record its spread coverage"
    assert stored["bars"] == hours
    assert stored["measured_bars"] == 20
    assert stored["assumed_bars"] == hours - 20
    assert stored["measured_share"] == pytest.approx(0.1)
    assert stored["fully_measured"] is False
