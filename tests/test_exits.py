"""Normalized exits: points, percent and ATR, on trades computed by hand."""
from __future__ import annotations

from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from core.engine.backtester import BacktestConfig, run_backtest
from core.engine.costs import CostModel, SpreadPolicy
from core.indicators.functions import atr
from core.strategy.exits import distance_points, has_variable_exits
from tests.conftest_engine import bars_from, flat_bars, spec_from, symbol_spec

ATHENS = ZoneInfo("Europe/Athens")
EQUITY = 1000.0  # 0.10 lots, 0.10 currency per point with the test symbol


def _spec(exit_block: dict, indicators: list[dict] | None = None):
    return spec_from(
        exit_block=exit_block,
        indicators=indicators
        if indicators is not None
        else [{"id": "atr", "type": "atr", "params": {"period": 3}}],
    )


def _run(spec, bars, costs: CostModel | None = None):
    return run_backtest(
        spec,
        bars,
        symbol_spec(),
        ATHENS,
        BacktestConfig(
            initial_equity=EQUITY,
            costs=costs or CostModel(spread=SpreadPolicy(mode="fixed", value=0.0)),
        ),
    )


# -- percent -------------------------------------------------------------


def test_percent_levels_are_a_share_of_the_entry_price() -> None:
    """0.15% of an entry at 2000.00 is 3.00 in price, i.e. 300 points."""
    bars = bars_from(
        [
            {"open": 2000.00, "high": 2000.50, "low": 1999.50, "close": 2000.40, "spread": 0},
            *flat_bars(6, price=2000.00, spread=0),
        ]
    )
    spec = _spec(
        {
            "stop_loss": {"type": "percent", "value": 0.15},
            "take_profit": {"type": "percent", "value": 0.15},
            "time_stop": {"bars": 3},
            "signal_exit": None,
        },
        indicators=[],
    )
    trade = _run(spec, bars).trades.iloc[0]
    assert trade["entry_price"] == pytest.approx(2000.00)
    assert trade["stop_level"] == pytest.approx(1997.00)
    assert trade["target_level"] == pytest.approx(2003.00)


def test_percent_levels_scale_with_the_price_of_the_instrument() -> None:
    """The same spec on a 10x price places a 10x distance: that is the point."""
    spec = _spec(
        {
            "stop_loss": {"type": "percent", "value": 0.10},
            "take_profit": {"type": "percent", "value": 0.10},
            "time_stop": {"bars": 3},
            "signal_exit": None,
        },
        indicators=[],
    )
    distances = []
    for price in (200.0, 2000.0):
        bars = bars_from(
            [
                {"open": price, "high": price * 1.001, "low": price * 0.999,
                 "close": price * 1.0002, "spread": 0},
                *flat_bars(6, price=price, spread=0),
            ]
        )
        trade = _run(spec, bars).trades.iloc[0]
        distances.append(float(trade["entry_price"] - trade["stop_level"]))
    assert distances[1] == pytest.approx(distances[0] * 10.0)


# -- ATR -----------------------------------------------------------------


def _atr_bars(n: int = 40, seed: int = 3) -> pd.DataFrame:
    """Bars with a signal on the first usable bar and a wide, varying range."""
    rng = np.random.default_rng(seed)
    rows = []
    price = 2000.0
    for _ in range(n):
        span = float(rng.uniform(0.5, 1.5))
        rows.append(
            {
                "open": price,
                "high": price + span,
                "low": price - span,
                "close": price + span * 0.5,
                "spread": 0,
            }
        )
        price += 0.1
    return bars_from(rows)


def test_atr_levels_use_the_signal_bar_value_and_stay_frozen() -> None:
    bars = _atr_bars()
    spec = _spec(
        {
            "stop_loss": {"type": "atr", "indicator": "atr", "mult": 2.0},
            "take_profit": {"type": "atr", "indicator": "atr", "mult": 1.0},
            "time_stop": {"bars": 5},
            "signal_exit": None,
        }
    )
    result = _run(spec, bars)
    assert len(result.trades)

    reference = atr(bars["high"], bars["low"], bars["close"], period=3)
    for _, trade in result.trades.iterrows():
        entry_position = bars.index.get_loc(trade["entry_time"])
        # the value read is the one of the *signal* bar, the last closed one
        expected = float(reference.iloc[entry_position - 1])
        assert trade["entry_price"] - trade["stop_level"] == pytest.approx(expected * 2.0)
        assert trade["target_level"] - trade["entry_price"] == pytest.approx(expected * 1.0)


def test_an_entry_during_the_atr_warmup_is_skipped_not_guessed() -> None:
    """The exit indicator is not the entry one: the signal can fire while it is NaN."""
    bars = _atr_bars(n=20)
    spec = _spec(
        {
            "stop_loss": {"type": "atr", "indicator": "slow_atr", "mult": 2.0},
            "take_profit": {"type": "points", "value": 80},
            "time_stop": {"bars": 5},
            "signal_exit": None,
        },
        indicators=[{"id": "slow_atr", "type": "atr", "params": {"period": 14}}],
    )
    result = _run(spec, bars)
    assert result.blocked["exit_indicator_warmup"] > 0
    for _, trade in result.trades.iterrows():
        assert np.isfinite(trade["stop_level"])


def test_atr_stop_sizes_the_risk_and_therefore_the_r_multiple() -> None:
    bars = _atr_bars()
    spec = _spec(
        {
            "stop_loss": {"type": "atr", "indicator": "atr", "mult": 2.0},
            "take_profit": {"type": "atr", "indicator": "atr", "mult": 1.0},
            "time_stop": {"bars": 5},
            "signal_exit": None,
        }
    )
    trade = _run(spec, bars).trades.iloc[0]
    distance_price = abs(float(trade["entry_price"]) - float(trade["stop_level"]))
    # risk = distance in points x currency per point (0.10 per point here)
    assert trade["risk_money"] == pytest.approx(distance_price / 0.01 * 0.10)
    assert trade["r_multiple"] == pytest.approx(trade["net_pnl"] / trade["risk_money"])


# -- the shared distance helper ------------------------------------------


def test_distance_helper_agrees_with_the_engine_on_points() -> None:
    bars = _atr_bars(n=10)
    spec = _spec(
        {
            "stop_loss": {"type": "points", "value": 150},
            "take_profit": {"type": "points", "value": 80},
            "time_stop": {"bars": 5},
            "signal_exit": None,
        },
        indicators=[],
    )
    values = distance_points(spec.exit.stop_loss, bars, {}, 0.01)
    assert values is not None and np.allclose(values, 150.0)
    assert has_variable_exits(spec.exit) is False


def test_variable_exits_are_declared_as_such() -> None:
    spec = _spec(
        {
            "stop_loss": {"type": "atr", "indicator": "atr", "mult": 2.0},
            "take_profit": {"type": "percent", "value": 0.1},
            "time_stop": {"bars": 5},
            "signal_exit": None,
        }
    )
    assert has_variable_exits(spec.exit) is True
