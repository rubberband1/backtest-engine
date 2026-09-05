"""The engine must say what its spread policy charged, and when it cannot be true."""
from __future__ import annotations

import pandas as pd
import pytest

from core.data.provider import Timeframe, rename_aggregated_spread
from core.data.spread import check_aggregation
from core.engine.costs import SpreadPolicy
from tests.conftest_engine import random_walk


def frame(n: int = 600, spread: float | pd.Series = 5.0) -> pd.DataFrame:
    bars = random_walk(n, seed=11)
    bars.index = pd.date_range(
        "2024-01-01", periods=n, freq="1min", tz="UTC", name="time"
    )
    bars["spread"] = spread
    return bars


def test_per_bar_on_m1_is_trustworthy() -> None:
    realism = SpreadPolicy().realism(frame(), Timeframe.M1)
    assert realism.trustworthy
    assert not realism.reads_aggregated_column
    assert realism.median_charged_points == 5.0


def test_per_bar_above_m1_is_now_refused_rather_than_flagged() -> None:
    """Phase 6 warned about this. Phase 7 makes it impossible.

    A warning next to a number is only as good as the reader; the frame here
    carries the aggregated field, and asking to charge it raises instead of
    producing a cheap result with a footnote.
    """
    from core.engine.costs import AggregatedSpreadRefused

    aggregated = rename_aggregated_spread(frame(), Timeframe.H1)
    with pytest.raises(AggregatedSpreadRefused):
        SpreadPolicy().realism(aggregated, Timeframe.H1)


def test_a_zero_spread_is_reported_as_a_free_fill() -> None:
    bars = frame()
    bars.iloc[:300, bars.columns.get_loc("spread")] = 0.0
    realism = SpreadPolicy().realism(bars, Timeframe.M1)
    assert not realism.trustworthy
    assert realism.zero_charged_share == pytest.approx(0.5)
    assert any("zero spread does not exist" in w for w in realism.warnings)


def test_a_measured_fixed_spread_is_trustworthy_at_any_timeframe() -> None:
    realism = SpreadPolicy(mode="fixed", value=3.0).realism(frame(), Timeframe.D1)
    assert realism.trustworthy
    assert realism.median_charged_points == 3.0
    assert not realism.reads_aggregated_column


def test_a_zero_fixed_spread_is_still_a_free_fill() -> None:
    realism = SpreadPolicy(mode="fixed", value=0.0).realism(frame(), Timeframe.M1)
    assert not realism.trustworthy
    assert realism.zero_charged_share == 1.0


def test_the_aggregation_claim_is_re_checkable() -> None:
    """An H1 column built as the min of its M1 sub-bars must be detected."""
    m1 = frame(n=600)
    m1["spread"] = pd.Series(range(600), index=m1.index) % 7 + 1.0
    hourly = m1["spread"].resample("60min").min()
    h1 = m1.resample("60min").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last",
         "tick_volume": "sum", "real_volume": "sum", "spread": "min"}
    )
    assert (h1["spread"] == hourly).all()

    check = check_aggregation(m1, h1, "SYNTH", Timeframe.M1, Timeframe.H1)
    assert check.share_equal_to_min == 1.0
    assert "minimum of its M1 sub-bars" in check.verdict


def test_a_column_that_is_not_a_minimum_is_not_accused_of_being_one() -> None:
    m1 = frame(n=600)
    m1["spread"] = pd.Series(range(600), index=m1.index) % 7 + 1.0
    h1 = m1.resample("60min").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last",
         "tick_volume": "sum", "real_volume": "sum", "spread": "max"}
    )
    check = check_aggregation(m1, h1, "SYNTH", Timeframe.M1, Timeframe.H1)
    assert check.share_equal_to_max == 1.0
    assert "minimum of its" not in check.verdict


def test_the_realism_block_travels_with_the_backtest_result() -> None:
    from zoneinfo import ZoneInfo

    from core.engine.backtester import BacktestConfig, run_backtest
    from core.strategy.spec import StrategySpec
    from tests.conftest_engine import symbol_spec

    spec = StrategySpec.from_dict(
        {
            "schema_version": 1,
            "id": "realism",
            "name": "realism",
            "instrument": {"symbol": "SYNTH", "timeframe": "H1"},
            "indicators": [{"id": "rsi", "type": "rsi", "params": {"period": 9}}],
            "entry": {
                "long": {"op": "lt", "left": {"ref": "rsi"}, "right": {"const": 40}}
            },
            "exit": {"stop_loss": {"type": "points", "value": 100},
                     "take_profit": {"type": "points", "value": 100}},
            "sizing": {"type": "equity_per_step", "equity_per_001_lot": 100,
                       "min_lot": 0.01, "max_lot": 0.05},
        }
    )
    # the H1 frame here still carries a `spread` column, which above M1 only
    # exists where it was rebuilt from M1: the realism block then reports a
    # reconstruction, and never an aggregated column
    result = run_backtest(
        spec, frame(), symbol_spec(name="SYNTH"), ZoneInfo("Europe/Athens"),
        BacktestConfig(),
    )
    assert result.spread_realism is not None
    assert not result.spread_realism.reads_aggregated_column
    assert result.spread_realism.reconstructed_from_m1
    assert result.spread_realism.trustworthy
