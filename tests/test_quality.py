from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.data.provider import Timeframe
from core.data.quality import check_quality, infer_session_slots

TF = Timeframe.M15
SYMBOL = "SYNTH"


def synthetic_bars(
    start: str = "2024-01-01",
    weeks: int = 12,
    timeframe: Timeframe = TF,
    open_hour: int = 1,
    close_hour: int = 23,
) -> pd.DataFrame:
    """Synthetic series with a weekly session: Mon-Fri, weekend closed."""
    index = pd.date_range(
        start, periods=weeks * 7 * 24 * 60 // timeframe.minutes, freq=timeframe.pandas_freq, tz="UTC"
    )
    index = index[(index.dayofweek < 5) & (index.hour >= open_hour) & (index.hour < close_hour)]
    index.name = "time"
    n = len(index)
    rng = np.random.default_rng(0)
    close = 2000 + np.cumsum(rng.normal(0, 0.5, n))
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "tick_volume": rng.integers(1, 500, n).astype(float),
            "spread": np.full(n, 20.0),
            "real_volume": np.zeros(n),
        },
        index=index,
    )


def test_clean_series_produces_no_gaps() -> None:
    report = check_quality(synthetic_bars(), SYMBOL, TF)
    assert report.gaps == []
    assert report.missing_bars == 0
    assert report.completeness == pytest.approx(1.0)
    assert report.is_clean


def test_weekend_does_not_count_as_gap() -> None:
    bars = synthetic_bars()
    active, confidence = infer_session_slots(pd.DatetimeIndex(bars.index), TF)
    slots_per_day = 24 * 60 // TF.minutes
    # 5 days x 22 hours of session
    assert len(active) == 5 * 22 * 60 // TF.minutes
    assert confidence == "high"
    assert len(active) < 7 * slots_per_day


def test_intraweek_gap_detected() -> None:
    bars = synthetic_bars()
    hole_start = pd.Timestamp("2024-01-10 10:00", tz="UTC")
    hole_end = pd.Timestamp("2024-01-10 13:00", tz="UTC")
    holed = bars[(bars.index < hole_start) | (bars.index >= hole_end)]

    report = check_quality(holed, SYMBOL, TF)
    assert len(report.gaps) == 1
    gap = report.gaps[0]
    assert gap.start == hole_start.to_pydatetime()
    assert gap.end == hole_end.to_pydatetime()
    assert gap.missing_bars == 3 * 60 // TF.minutes
    assert not report.is_clean


def test_duplicates_and_nans_counted_without_dropping() -> None:
    bars = synthetic_bars(weeks=6)
    duplicated = pd.concat([bars, bars.iloc[[5, 6]]])
    duplicated.iloc[0, duplicated.columns.get_loc("close")] = np.nan

    report = check_quality(duplicated, SYMBOL, TF)
    assert report.rows == len(duplicated)
    assert report.duplicate_timestamps == 2
    assert report.nan_counts["close"] == 1
    assert report.unsorted
    assert len(report.samples["duplicates"]) == 2


def test_malformed_bars_detected() -> None:
    bars = synthetic_bars(weeks=6).copy()
    bars.iloc[10, bars.columns.get_loc("high")] = bars.iloc[10]["low"] - 5
    bars.iloc[20, bars.columns.get_loc("high")] = bars.iloc[20]["close"] - 1
    bars.iloc[30, bars.columns.get_loc("tick_volume")] = 0
    bars.iloc[40, bars.columns.get_loc("open")] = -1.0

    report = check_quality(bars, SYMBOL, TF)
    assert report.invalid_high_low == 1
    assert report.invalid_range >= 1
    assert report.zero_tick_volume == 1
    assert report.non_positive_prices == 1


def test_empty_frame_does_not_blow_up() -> None:
    empty = synthetic_bars(weeks=1).iloc[0:0]
    report = check_quality(empty, SYMBOL, TF)
    assert report.rows == 0
    assert report.gaps == []
    assert report.session_confidence == "no data"


def test_naive_index_rejected() -> None:
    bars = synthetic_bars(weeks=2)
    bars.index = bars.index.tz_localize(None)
    with pytest.raises(ValueError):
        check_quality(bars, SYMBOL, TF)


def test_short_history_reports_low_confidence() -> None:
    report = check_quality(synthetic_bars(weeks=2), SYMBOL, TF)
    assert "low" in report.session_confidence


def test_as_text_reports_the_key_numbers() -> None:
    report = check_quality(synthetic_bars(weeks=6), SYMBOL, TF)
    text = report.as_text()
    assert SYMBOL in text
    assert "completeness" in text
