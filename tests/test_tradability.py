"""Stage zero: what the broker's spread makes untestable, and what it does not."""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from core.data.cache import ParquetCache
from core.data.provider import Timeframe
from core.data.spread import SpreadReference, measure, store
from core.research.tradability import (
    DEFAULT_MAX_SPREAD_ATR,
    assess,
    build_table,
)
from tests.conftest_engine import random_walk

SYMBOL = "TRADE"
POINT = 0.01


def bars(n: int = 3000, spread: float = 5.0, seed: int = 3) -> pd.DataFrame:
    frame = random_walk(n, seed=seed)
    frame.index = pd.date_range(
        "2024-01-01", periods=n, freq="1min", tz="UTC", name="time"
    )
    frame["spread"] = spread
    return frame


def reference(points: float, symbol: str = SYMBOL) -> SpreadReference:
    return SpreadReference(
        symbol=symbol,
        source_timeframe="M1",
        bars=1000,
        usable_bars=1000,
        zero_share=0.0,
        median_points=points,
        mean_points=points,
        p90_points=points,
        p99_points=points,
        window_start=datetime(2023, 1, 1, tzinfo=timezone.utc),
        window_end=datetime(2025, 1, 1, tzinfo=timezone.utc),
        measured_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
    )


def test_a_cheap_spread_leaves_the_cell_testable() -> None:
    cell = assess(bars(), SYMBOL, Timeframe.M1, POINT, reference(2.0))
    assert cell.tradable
    assert cell.judged
    assert cell.spread_atr_ratio < DEFAULT_MAX_SPREAD_ATR


def test_a_spread_above_the_threshold_refuses_the_cell() -> None:
    cheap = assess(bars(), SYMBOL, Timeframe.M1, POINT, reference(2.0))
    wide = assess(bars(), SYMBOL, Timeframe.M1, POINT, reference(60.0))
    assert wide.judged
    assert not wide.tradable
    assert wide.spread_atr_ratio > cheap.spread_atr_ratio
    assert "no realistic edge covers this" in wide.reason


def test_the_stop_share_is_the_ratio_divided_by_the_stop_multiple() -> None:
    """The two numbers must never disagree: one is derived from the other."""
    cell = assess(bars(), SYMBOL, Timeframe.M1, POINT, reference(10.0))
    assert cell.spread_stop_share == pytest.approx(
        cell.spread_atr_ratio / cell.stop_atr_mult
    )


def test_an_unknown_spread_produces_an_unjudged_cell_not_a_free_one() -> None:
    """No measurement must never read as zero cost."""
    cell = assess(bars(), SYMBOL, Timeframe.M1, POINT, None)
    assert not cell.judged
    assert cell.spread_atr_ratio is None
    assert "the cost is unknown" in cell.reason
    # still admitted, so the screening does not silently drop it
    assert cell.tradable


def test_the_spread_never_comes_from_the_bars_being_judged() -> None:
    """A cheap column on the judged bars must not rescue a wide instrument."""
    cheap_column = bars(spread=0.0)
    cell = assess(cheap_column, SYMBOL, Timeframe.M1, POINT, reference(60.0))
    assert not cell.tradable
    assert cell.median_spread_points == 60.0


def test_extrapolating_the_spread_backwards_is_declared(tmp_path) -> None:
    old = bars()
    old.index = pd.date_range(
        "2010-01-01", periods=len(old), freq="1min", tz="UTC", name="time"
    )
    cell = assess(old, SYMBOL, Timeframe.M1, POINT, reference(2.0))
    assert "an assumption, not a measurement" in cell.reason


def test_too_few_bars_is_not_a_refusal() -> None:
    cell = assess(bars(n=50), SYMBOL, Timeframe.M1, POINT, reference(2.0))
    assert not cell.judged
    assert cell.tradable
    assert "below the" in cell.reason


def test_the_table_reads_the_spread_from_the_m1_cache(tmp_path) -> None:
    cache = ParquetCache(tmp_path / "cache")
    m1 = bars(spread=4.0)
    window = (m1.index[0].to_pydatetime(), m1.index[-1].to_pydatetime())
    cache.write_year(SYMBOL, Timeframe.M1, 2024, m1, [window], source="test")

    table = build_table(cache, [SYMBOL], ["M1"], {SYMBOL: POINT})
    cell = table.get(SYMBOL, "M1")
    assert cell is not None
    assert cell.judged
    assert cell.median_spread_points == 4.0
    assert cell.spread_source.startswith("M1")


def test_a_cell_with_no_data_is_unjudged(tmp_path) -> None:
    cache = ParquetCache(tmp_path / "cache")
    m1 = bars()
    cache.write_year(
        SYMBOL, Timeframe.M1, 2024, m1,
        [(m1.index[0].to_pydatetime(), m1.index[-1].to_pydatetime())], source="test",
    )
    table = build_table(cache, [SYMBOL], ["M1", "H1"], {SYMBOL: POINT})
    h1 = table.get(SYMBOL, "H1")
    assert h1 is not None
    assert not h1.judged
    assert "no cached data" in h1.reason


def test_an_unknown_pair_is_admitted_rather_than_refused(tmp_path) -> None:
    """Absence of a measurement is not a verdict."""
    table = build_table(ParquetCache(tmp_path), [], [], {})
    assert table.is_tradable("NEVER-MEASURED", "H1")


def test_the_measured_spread_drops_zeros_instead_of_averaging_them() -> None:
    frame = bars(n=1000, spread=4.0)
    frame.iloc[:500, frame.columns.get_loc("spread")] = 0.0
    measured = measure(frame, SYMBOL, Timeframe.M1)
    assert measured.zero_share == pytest.approx(0.5)
    assert measured.usable_bars == 500
    # a zero spread does not pull the median towards zero
    assert measured.median_points == 4.0


def test_a_reference_survives_a_round_trip_to_disk(tmp_path) -> None:
    cache = ParquetCache(tmp_path)
    original = reference(7.0)
    store(cache, original)
    from core.data.spread import load

    assert load(cache, SYMBOL) == original


def test_an_all_zero_spread_sample_reports_no_median() -> None:
    frame = bars(n=500, spread=0.0)
    measured = measure(frame, SYMBOL, Timeframe.M1)
    assert measured.usable_bars == 0
    assert np.isnan(measured.median_points)
    cell = assess(bars(), SYMBOL, Timeframe.M1, POINT, measured)
    assert not cell.judged
