"""Costs: currency conversion, spread policies, overnight swap."""
from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pytest

from core.engine.backtester import BacktestConfig, run_backtest
from core.engine.costs import (
    CommissionModel,
    CostModel,
    SpreadPolicy,
    SwapModel,
    money_per_point,
    points_to_money,
    points_to_price,
    price_to_points,
)
from core.engine.sizing import lots_for
from core.strategy.spec import Sizing
from tests.conftest_engine import bars_from, flat_bars, random_walk, spec_from, symbol_spec

ATHENS = ZoneInfo("Europe/Athens")


def utc(*args: int) -> datetime:
    return datetime(*args, tzinfo=timezone.utc)


def test_point_value_from_tick_value() -> None:
    """XAUUSD on an EUR account: tick_value is already in the account currency."""
    gold = symbol_spec(point=0.01, tick_size=0.01, tick_value=0.8622, contract_size=100.0)
    assert money_per_point(gold, 1.0) == pytest.approx(0.8622)
    assert money_per_point(gold, 0.01) == pytest.approx(0.008622)
    assert points_to_money(150, gold, 0.01) == pytest.approx(1.2933)


def test_point_value_scales_with_tick_size() -> None:
    """A 5-decimal forex pair: point 0.00001, tick_size 0.00001."""
    eurusd = symbol_spec(point=0.00001, tick_size=0.00001, tick_value=0.86, contract_size=100000.0)
    assert money_per_point(eurusd, 1.0) == pytest.approx(0.86)
    # 10 points = 1 pip
    assert points_to_money(10, eurusd, 0.1) == pytest.approx(0.86)


def test_fallback_to_contract_size() -> None:
    spec = symbol_spec(point=0.01, tick_size=0.0, tick_value=0.0, contract_size=100.0)
    assert money_per_point(spec, 1.0) == pytest.approx(1.0)


def test_points_price_conversion() -> None:
    spec = symbol_spec(point=0.01)
    assert points_to_price(150, spec) == pytest.approx(1.5)
    assert price_to_points(1.5, spec) == pytest.approx(150)


def test_per_bar_spread_policy() -> None:
    bars = random_walk(50, spread=17.0)
    bars.iloc[10, bars.columns.get_loc("spread")] = 99.0
    series = SpreadPolicy(mode="per_bar").series(bars)
    assert series.iloc[0] == 17.0
    assert series.iloc[10] == 99.0


def test_fixed_and_quantile_spread_policies() -> None:
    bars = random_walk(100)
    bars["spread"] = list(range(100))

    fixed = SpreadPolicy(mode="fixed", value=25.0).series(bars)
    assert fixed.nunique() == 1 and fixed.iloc[0] == 25.0

    quantile = SpreadPolicy(mode="quantile", value=0.9).series(bars)
    assert quantile.nunique() == 1
    assert quantile.iloc[0] == pytest.approx(bars["spread"].quantile(0.9))


def test_spread_policy_missing_parameters() -> None:
    bars = random_walk(10)
    with pytest.raises(ValueError, match="fixed"):
        SpreadPolicy(mode="fixed").series(bars)
    with pytest.raises(ValueError, match="quantile"):
        SpreadPolicy(mode="quantile", value=2.0).series(bars)
    with pytest.raises(ValueError, match="no spread field at all"):
        SpreadPolicy(mode="per_bar").series(bars.drop(columns="spread"))


def test_round_turn_commission() -> None:
    assert CommissionModel(3.5).round_turn(0.1) == pytest.approx(0.7)
    assert CommissionModel(0.0).round_turn(10.0) == 0.0


def test_swap_counts_server_midnights() -> None:
    swap = SwapModel(mode="points")
    # Monday 2024-01-01 12:00 server (10:00 UTC) -> Tuesday 12:00: one night
    assert swap.nights(utc(2024, 1, 1, 10), utc(2024, 1, 2, 10), ATHENS) == 1.0
    # within the same server day: no nights
    assert swap.nights(utc(2024, 1, 1, 6), utc(2024, 1, 1, 20), ATHENS) == 0.0


def test_triple_swap_on_wednesday() -> None:
    swap = SwapModel(mode="points")
    # Wednesday 2024-01-03 -> Thursday 2024-01-04, Athens midnight
    nights = swap.nights(utc(2024, 1, 3, 8), utc(2024, 1, 4, 8), ATHENS)
    assert nights == 3.0
    # Tuesday through Thursday: one normal + one triple
    assert swap.nights(utc(2024, 1, 2, 8), utc(2024, 1, 4, 8), ATHENS) == 4.0


def test_swap_uses_server_timezone_not_utc() -> None:
    """At 23:00 UTC it is already the next day in Athens: the night has passed."""
    swap = SwapModel(mode="points")
    entry = utc(2024, 1, 1, 21, 0)  # 23:00 in Athens
    exit_ = utc(2024, 1, 1, 23, 0)  # 01:00 on January 2nd in Athens
    assert swap.nights(entry, exit_, ATHENS) == 1.0
    assert swap.nights(entry, exit_, ZoneInfo("UTC")) == 0.0


def test_swap_in_points_becomes_currency() -> None:
    spec = symbol_spec(point=0.01, tick_size=0.01, tick_value=1.0, swap_long=-8.0, swap_short=2.0)
    swap = SwapModel(mode="points")
    charge = swap.charge(spec, 1, 0.10, utc(2024, 1, 1, 10), utc(2024, 1, 2, 10), ATHENS)
    assert charge == pytest.approx(-8.0 * 0.10)  # one night, 0.10 currency per point
    credit = swap.charge(spec, -1, 0.10, utc(2024, 1, 1, 10), utc(2024, 1, 2, 10), ATHENS)
    assert credit == pytest.approx(2.0 * 0.10)


def test_swap_in_money_skips_the_points_conversion() -> None:
    spec = symbol_spec(swap_long=-1.25)
    charge = SwapModel(mode="money").charge(
        spec, 1, 2.0, utc(2024, 1, 1, 10), utc(2024, 1, 2, 10), ATHENS
    )
    assert charge == pytest.approx(-2.5)


def test_swap_can_be_disabled() -> None:
    spec = symbol_spec(swap_long=-100.0)
    assert SwapModel(mode="none").charge(
        spec, 1, 1.0, utc(2024, 1, 1), utc(2024, 3, 1), ATHENS
    ) == 0.0


def test_zero_cost_model_costs_nothing() -> None:
    bars = random_walk(20, spread=30.0)
    zero = CostModel.zero()
    assert zero.spread.series(bars).eq(0.0).all()
    assert zero.commission.round_turn(1.0) == 0.0
    assert zero.swap.charge(
        symbol_spec(swap_long=-10), 1, 1.0, utc(2024, 1, 1), utc(2024, 2, 1), ATHENS
    ) == 0.0


def test_swap_only_charged_on_trades_crossing_server_midnight() -> None:
    """A1 audit: a 60-bar M1 time stop must pay swap only when the holding
    period actually crosses a server midnight, never merely because it is
    held for an "overnight-length" number of bars.

    Both trades below are held the same ~61 minutes by the same time stop;
    only the second one's window straddles the Athens midnight.
    """
    spec = spec_from(
        exit_block={"stop_loss": None, "take_profit": None,
                    "time_stop": {"bars": 60}, "signal_exit": None},
    )
    instrument = symbol_spec(point=0.01, tick_size=0.01, tick_value=1.0, swap_long=-8.0)
    costs = CostModel(spread=SpreadPolicy(mode="fixed", value=5.0), swap=SwapModel(mode="points"))
    rows = [
        {"open": 2000.0, "high": 2000.5, "low": 1999.5, "close": 2000.4, "spread": 5},
        *flat_bars(70, price=2000.4, spread=5),
    ]

    # entry 10:00 Athens, exit ~11:01 Athens: same server day
    no_crossing = run_backtest(
        spec, bars_from(rows, start="2024-01-15 07:59"), instrument, ATHENS,
        BacktestConfig(initial_equity=1000.0, costs=costs),
    )
    assert len(no_crossing.trades) == 1
    assert no_crossing.trades.iloc[0]["swap"] == 0.0

    # entry 23:30 Athens, exit ~00:31 Athens: exactly one midnight crossed
    crossing = run_backtest(
        spec, bars_from(rows, start="2024-01-15 21:29"), instrument, ATHENS,
        BacktestConfig(initial_equity=1000.0, costs=costs),
    )
    assert len(crossing.trades) == 1
    trade = crossing.trades.iloc[0]
    expected = points_to_money(-8.0, instrument, trade["lots"])  # one night
    assert trade["swap"] == pytest.approx(expected)
    assert trade["swap"] != 0.0


# -- sizing --------------------------------------------------------------


def sizing(equity_per_001_lot: float = 100, min_lot: float = 0.01, max_lot: float = 0.05) -> Sizing:
    return Sizing(
        type="equity_per_step",
        equity_per_001_lot=equity_per_001_lot,
        min_lot=min_lot,
        max_lot=max_lot,
    )


def test_stepped_sizing() -> None:
    spec = symbol_spec()
    assert lots_for(sizing(), 100.0, spec) == pytest.approx(0.01)
    assert lots_for(sizing(), 299.0, spec) == pytest.approx(0.02)
    assert lots_for(sizing(), 300.0, spec) == pytest.approx(0.03)


def test_sizing_respects_the_cap() -> None:
    assert lots_for(sizing(max_lot=0.05), 10_000.0, symbol_spec()) == pytest.approx(0.05)


def test_min_lot_is_a_floor_not_a_threshold() -> None:
    """Below one equity step the minimum lot is traded, as the spec says."""
    assert lots_for(sizing(), 99.0, symbol_spec()) == pytest.approx(0.01)
    assert lots_for(sizing(), 1.0, symbol_spec()) == pytest.approx(0.01)


def test_non_positive_equity_does_not_open() -> None:
    assert lots_for(sizing(), 0.0, symbol_spec()) == 0.0
    assert lots_for(sizing(), -50.0, symbol_spec()) == 0.0


def test_below_broker_minimum_does_not_open() -> None:
    spec = symbol_spec()
    spec = spec.__class__(**{**spec.__dict__, "volume_min": 0.5, "volume_step": 0.5})
    assert lots_for(sizing(max_lot=0.05), 1_000.0, spec) == 0.0


def test_sizing_rounds_to_broker_step() -> None:
    spec = symbol_spec()
    spec = spec.__class__(**{**spec.__dict__, "volume_step": 0.1, "volume_min": 0.1})
    assert lots_for(sizing(max_lot=1.0), 2_500.0, spec) == pytest.approx(0.2)
