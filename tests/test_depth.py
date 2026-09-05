"""Depth probing, provenance and cross-source continuity."""
from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from core.data.cache import UNRECORDED_SOURCE, ParquetCache
from core.data.continuity import compare_sources, stitch
from core.data.depth import DepthProbe, build_report, inception, probe_all
from core.data.provider import Timeframe
from core.data.quality import check_quality
from tests.conftest_engine import random_walk

POINT = 0.01


def probe(
    symbol: str,
    first: str,
    bars: int = 1000,
    truncated: bool = False,
    timeframe: str = "H1",
) -> DepthProbe:
    return DepthProbe(
        symbol=symbol,
        timeframe=timeframe,
        source="test_feed",
        bars=bars,
        first_bar=datetime.fromisoformat(first).replace(tzinfo=timezone.utc),
        last_bar=datetime(2026, 1, 1, tzinfo=timezone.utc),
        truncated=truncated,
        probed_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def test_a_full_probe_is_reported_as_a_lower_bound() -> None:
    report = build_report([probe("XAUUSD.r", "2020-01-01", truncated=True)])
    notes = report.note_lines(report.probes[0])
    assert any("lower bound" in note for note in notes)


def test_history_before_the_euro_existed_is_called_backfill() -> None:
    assert inception("EURUSD.r") == datetime(1999, 1, 1, tzinfo=timezone.utc)
    early = probe("EURUSD.r", "1971-01-04")
    report = build_report([early])
    assert early.predates_inception
    assert any("backfill" in note for note in report.note_lines(early))


def test_the_usable_start_is_floored_by_inception_not_by_the_cohort() -> None:
    report = build_report(
        [
            probe("EURUSD.r", "1971-01-04"),
            probe("GBPUSD.r", "2021-05-26"),
            probe("USDJPY.r", "2021-05-26"),
            probe("AUDUSD.r", "2021-05-26"),
        ]
    )
    assert report.usable_from("EURUSD.r", "H1") == datetime(
        1999, 1, 1, tzinfo=timezone.utc
    )
    assert report.usable_from("GBPUSD.r", "H1") == datetime(
        2021, 5, 26, tzinfo=timezone.utc
    )


def test_a_series_older_than_its_own_feed_is_flagged_as_an_outlier() -> None:
    """No hardcoded calendar: the rest of the feed is the reference."""
    odd = probe("XAUUSD.r", "2005-01-01")
    report = build_report(
        [
            odd,
            probe("XAGUSD.r", "2020-02-28"),
            probe("XTIUSD", "2020-02-24"),
            probe("XBRUSD", "2020-02-28"),
        ]
    )
    assert any("older than the feed itself" in n for n in report.note_lines(odd))


def test_a_probe_failure_on_one_pair_does_not_stop_the_others() -> None:
    class Flaky:
        source_name = "flaky"

        def probe_depth(self, symbol: str, timeframe: Timeframe) -> DepthProbe:
            if symbol == "BAD":
                raise RuntimeError("terminal said no")
            return probe(symbol, "2020-01-01", timeframe=timeframe.name)

    probes = probe_all(Flaky(), ["GOOD", "BAD"], ["H1"])
    assert len(probes) == 2
    assert probes[1].error is not None
    assert "terminal said no" in probes[1].error


# -- provenance ----------------------------------------------------------


def bars(n: int = 500, start: str = "2024-01-01", freq: str = "1h") -> pd.DataFrame:
    frame = random_walk(n, seed=7)
    frame.index = pd.date_range(start, periods=n, freq=freq, tz="UTC", name="time")
    frame["spread"] = 5.0
    return frame


def test_the_cache_records_which_source_filled_each_interval(tmp_path) -> None:
    cache = ParquetCache(tmp_path)
    first = bars(200)
    second = bars(200, start="2024-02-01")
    cache.write_year(
        "SYM", Timeframe.H1, 2024, first,
        [(first.index[0].to_pydatetime(), first.index[-1].to_pydatetime())],
        source="mt5_feed",
    )
    cache.write_year(
        "SYM", Timeframe.H1, 2024, second,
        [(second.index[0].to_pydatetime(), second.index[-1].to_pydatetime())],
        source="mt5_hc_cache",
    )
    meta = cache.read_meta("SYM", Timeframe.H1, 2024)
    assert meta is not None
    assert meta.sources == ["mt5_feed", "mt5_hc_cache"]
    assert len(meta.provenance) == 2


def test_coverage_without_provenance_is_reported_as_unrecorded(tmp_path) -> None:
    """A pre-provenance meta file must not be assumed to come from the feed."""
    cache = ParquetCache(tmp_path)
    frame = bars(100)
    cache.write_year(
        "SYM", Timeframe.H1, 2024, frame,
        [(frame.index[0].to_pydatetime(), frame.index[-1].to_pydatetime())],
    )
    meta = cache.read_meta("SYM", Timeframe.H1, 2024)
    assert meta is not None
    assert meta.sources == [UNRECORDED_SOURCE]


def test_get_or_fetch_takes_the_source_from_the_fetcher(tmp_path) -> None:
    cache = ParquetCache(tmp_path)
    frame = bars(100)

    class Provider:
        source_name = "named_provider"

        def get_bars(self, symbol, timeframe, start, end):
            return frame[(frame.index >= start) & (frame.index < end)]

    provider = Provider()
    cache.get_or_fetch(
        "SYM", Timeframe.H1, frame.index[0].to_pydatetime(),
        frame.index[-1].to_pydatetime(), fetch=provider.get_bars,
    )
    meta = cache.read_meta("SYM", Timeframe.H1, 2024)
    assert meta is not None
    assert meta.sources == ["named_provider"]


# -- continuity ----------------------------------------------------------


def test_two_identical_sources_agree() -> None:
    frame = bars(300)
    report = compare_sources(frame, frame.copy(), "SYM", "H1", POINT)
    assert report.agrees
    assert report.disagreeing_bars == 0
    assert "can be stitched" in report.verdict


def test_a_subset_is_not_a_disagreement() -> None:
    """One source holding fewer bars is a coverage fact, not a contradiction."""
    frame = bars(300)
    report = compare_sources(frame, frame.iloc[100:].copy(), "SYM", "H1", POINT)
    assert report.only_in_left == 100
    assert report.agrees


def test_a_disagreement_on_the_open_bar_is_excused_and_named() -> None:
    frame = bars(300)
    other = frame.copy()
    other.iloc[-1, other.columns.get_loc("close")] += 5.0
    report = compare_sources(frame, other, "SYM", "H1", POINT)
    assert report.disagreeing_bars == 1
    assert report.only_last_bar_disagrees
    assert report.agrees
    assert "still forming" in report.verdict


def test_a_disagreement_in_the_middle_is_not_excused() -> None:
    frame = bars(300)
    other = frame.copy()
    other.iloc[50, other.columns.get_loc("close")] += 5.0
    report = compare_sources(frame, other, "SYM", "H1", POINT)
    assert report.disagreeing_bars == 1
    assert not report.only_last_bar_disagrees
    assert not report.agrees
    assert "not only the last one" in report.verdict


def test_drift_is_measured_in_points_not_in_price() -> None:
    frame = bars(200)
    other = frame.copy()
    other.iloc[10, other.columns.get_loc("high")] += 0.50  # 50 points at 0.01
    report = compare_sources(frame, other, "SYM", "H1", POINT)
    high = next(d for d in report.drift if d.column == "high")
    assert high.max_points == pytest.approx(50.0)


def test_no_overlap_is_reported_as_uncheckable() -> None:
    left = bars(100)
    right = bars(100, start="2025-01-01")
    report = compare_sources(left, right, "SYM", "H1", POINT)
    assert not report.agrees
    assert report.overlap_bars == 0
    assert "cannot be checked" in report.verdict


def test_stitching_prefers_the_first_source_and_counts_contributions() -> None:
    preferred = bars(200)
    # same grid, different values: the overlap must resolve to `preferred`
    fallback = bars(400) + 1.0
    joined, counts = stitch(preferred, fallback)
    assert len(joined) == 400
    assert counts == {"preferred": 200, "fallback": 200}
    # the preferred values survive where both had a bar
    assert joined.loc[preferred.index[0], "close"] == preferred.iloc[0]["close"]


# -- the quality clock ---------------------------------------------------


def test_a_dst_change_is_not_reported_as_a_missing_summer() -> None:
    """The bug this guards: a UTC session grid loses an hour twice a year."""
    athens = ZoneInfo("Europe/Athens")
    # one daily bar at 00:00 server time for two years, across four DST changes
    local = pd.date_range("2024-01-01", periods=730, freq="1D")
    index = (
        pd.DatetimeIndex(local)
        .tz_localize(athens, nonexistent="shift_forward")
        .tz_convert("UTC")
    )
    frame = random_walk(len(index), seed=2)
    frame.index = index
    frame.index.name = "time"
    frame["spread"] = 5.0

    on_utc = check_quality(frame, "SYM", Timeframe.D1)
    on_server = check_quality(frame, "SYM", Timeframe.D1, server_tz=athens)

    assert on_server.completeness > 0.99
    assert on_server.completeness > on_utc.completeness
    assert on_server.session_clock == "server (Europe/Athens)"
    assert on_utc.session_clock == "UTC"
