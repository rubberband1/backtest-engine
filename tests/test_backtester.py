"""Engine execution rules, verified on trades with hand-computed outcomes."""
from __future__ import annotations

from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from core.engine.backtester import BacktestConfig, run_backtest
from core.engine.costs import CommissionModel, CostModel, SpreadPolicy, SwapModel
from tests.conftest_engine import bars_from, flat_bars, random_walk, spec_from, symbol_spec

ATHENS = ZoneInfo("Europe/Athens")
EQUITY = 1000.0  # 1000 / 100 per step => 0.10 lots => 0.10 currency per point


def run(bars: pd.DataFrame, strategy=None, costs: CostModel | None = None, equity=EQUITY,
        spec=None):
    return run_backtest(
        strategy or spec_from(),
        bars,
        spec or symbol_spec(),
        ATHENS,
        BacktestConfig(initial_equity=equity, costs=costs or CostModel()),
    )


def test_long_trade_with_hand_computed_pnl() -> None:
    """0.10 lots, tick_value 1.0 over tick_size 0.01 => 0.10 currency per point.

    Signal on bar 0 (close > open), entry at the open of bar 1 at
    2000.00 + 10 points of spread = 2000.10. Target at +80 points = 2000.90,
    touched by bar 2. Gross 90 points x 0.10 = 9.00, spread 10 points x 0.10
    = 1.00, net 8.00 (i.e. exactly the 80 points of target).
    """
    bars = bars_from(
        [
            {"open": 2000.00, "high": 2000.50, "low": 1999.50, "close": 2000.40, "spread": 10},
            {"open": 2000.00, "high": 2000.50, "low": 1999.90, "close": 2000.20, "spread": 10},
            # close < open: the bar that carries the target must not open a new trade
            {"open": 2000.20, "high": 2001.00, "low": 2000.05, "close": 2000.10, "spread": 10},
            *flat_bars(3, price=2000.10),
        ]
    )
    result = run(bars)

    assert len(result.trades) == 1
    trade = result.trades.iloc[0]
    assert trade["direction"] == 1
    assert trade["entry_time"] == bars.index[1]
    assert trade["entry_price"] == pytest.approx(2000.10)
    assert trade["exit_time"] == bars.index[2]
    assert trade["exit_price"] == pytest.approx(2000.90)
    assert trade["exit_reason"] == "take_profit"
    assert trade["lots"] == pytest.approx(0.10)
    assert trade["gross_pnl"] == pytest.approx(9.00)
    assert trade["spread_cost"] == pytest.approx(1.00)
    assert trade["net_pnl"] == pytest.approx(8.00)
    assert result.final_equity == pytest.approx(EQUITY + 8.00)


def test_signal_on_close_executed_on_next_open() -> None:
    bars = bars_from(
        [
            {"open": 2000.00, "high": 2000.50, "low": 1999.50, "close": 2000.40, "spread": 10},
            {"open": 2000.30, "high": 2000.60, "low": 2000.20, "close": 2000.35, "spread": 10},
            *flat_bars(4),
        ]
    )
    trade = run(bars).trades.iloc[0]
    # the entry uses the open of bar 1, never the close of bar 0
    assert trade["entry_price"] == pytest.approx(2000.30 + 10 * 0.01)


def test_gap_beyond_stop_fills_at_open() -> None:
    """The bar opens 5 dollars below: the fill is there, not at the stop level.

    Filling at the level (1998.60) would give -14.00. The real price is
    1995.00, i.e. -51.00. That is the difference between a real loss tail and
    one truncated by construction.
    """
    bars = bars_from(
        [
            {"open": 2000.00, "high": 2000.50, "low": 1999.50, "close": 2000.40, "spread": 10},
            {"open": 2000.00, "high": 2000.20, "low": 1999.80, "close": 2000.00, "spread": 10},
            {"open": 1995.00, "high": 1995.20, "low": 1994.00, "close": 1994.50, "spread": 10},
            *flat_bars(3, price=1994.5),
        ]
    )
    trade = run(bars).trades.iloc[0]

    assert trade["exit_reason"] == "gap_stop_loss"
    assert trade["exit_price"] == pytest.approx(1995.00)
    assert trade["net_pnl"] == pytest.approx(-51.00)
    assert trade["net_pnl"] < -14.00


def test_gap_beyond_target_fills_at_open() -> None:
    bars = bars_from(
        [
            {"open": 2000.00, "high": 2000.50, "low": 1999.50, "close": 2000.40, "spread": 10},
            {"open": 2000.00, "high": 2000.20, "low": 1999.80, "close": 2000.00, "spread": 10},
            {"open": 2005.00, "high": 2005.50, "low": 2004.80, "close": 2005.20, "spread": 10},
            *flat_bars(3, price=2005.0),
        ]
    )
    trade = run(bars).trades.iloc[0]

    assert trade["exit_reason"] == "gap_take_profit"
    assert trade["exit_price"] == pytest.approx(2005.00)
    # 500 gross points instead of the target's 80
    assert trade["gross_pnl"] == pytest.approx(50.00)


def test_stop_and_target_in_same_bar_assumes_stop_and_flags() -> None:
    bars = bars_from(
        [
            {"open": 2000.00, "high": 2000.50, "low": 1999.50, "close": 2000.40, "spread": 10},
            {"open": 2000.00, "high": 2000.20, "low": 1999.90, "close": 2000.00, "spread": 10},
            # touches both 2000.90 (target) and 1998.60 (stop)
            {"open": 2000.10, "high": 2001.50, "low": 1998.00, "close": 1999.00, "spread": 10},
            *flat_bars(3, price=1999.0),
        ]
    )
    result = run(bars)
    trade = result.trades.iloc[0]

    assert trade["exit_reason"] == "stop_loss"
    assert bool(trade["ambiguous"]) is True
    assert result.ambiguous_trades == 1
    assert trade["exit_price"] == pytest.approx(1998.60)


def test_short_uses_ask_for_stop_and_target() -> None:
    """A short enters on the bid and exits on the ask: touches are measured on the ask."""
    strategy = spec_from(
        entry_long=None,
        entry_short={"op": "lt", "left": {"bar": "close"}, "right": {"bar": "open"}},
    )
    bars = bars_from(
        [
            {"open": 2000.00, "high": 2000.50, "low": 1999.50, "close": 1999.60, "spread": 10},
            {"open": 2000.00, "high": 2000.20, "low": 1999.80, "close": 2000.00, "spread": 10},
            # bid low 1999.30 => ask low 1999.40, below the 1999.20 target? no.
            {"open": 1999.90, "high": 2000.00, "low": 1999.10, "close": 1999.30, "spread": 10},
            *flat_bars(3, price=1999.3),
        ]
    )
    result = run(bars, strategy)
    trade = result.trades.iloc[0]

    assert trade["direction"] == -1
    assert trade["entry_price"] == pytest.approx(2000.00)  # bid, no spread
    # short target = 2000.00 - 0.80 = 1999.20; ask low = 1999.10 + 0.10 = 1999.20 => touch
    assert trade["exit_reason"] == "take_profit"
    assert trade["exit_price"] == pytest.approx(1999.20)
    # gross on raw prices: entry 2000.00, exit ask 1999.20 -> bid 1999.10
    assert trade["gross_pnl"] == pytest.approx(90 * 0.10)
    assert trade["spread_cost"] == pytest.approx(1.00)
    assert trade["net_pnl"] == pytest.approx(8.00)


def test_scales_with_the_instrument_point_value() -> None:
    """Same spec, same signals, instrument whose point is worth twice as much."""
    bars = bars_from(
        [
            {"open": 2000.00, "high": 2000.50, "low": 1999.50, "close": 2000.40, "spread": 10},
            {"open": 2000.00, "high": 2000.50, "low": 1999.90, "close": 2000.20, "spread": 10},
            {"open": 2000.20, "high": 2001.00, "low": 2000.05, "close": 2000.10, "spread": 10},
            *flat_bars(3, price=2000.10),
        ]
    )
    base = run(bars, spec=symbol_spec(tick_value=1.0)).trades.iloc[0]
    doubled = run(bars, spec=symbol_spec(tick_value=2.0)).trades.iloc[0]

    assert doubled["net_pnl"] == pytest.approx(base["net_pnl"] * 2)
    assert doubled["gross_pnl"] == pytest.approx(base["gross_pnl"] * 2)
    assert doubled["spread_cost"] == pytest.approx(base["spread_cost"] * 2)


def test_falls_back_to_contract_size_without_tick_value() -> None:
    bars = bars_from(
        [
            {"open": 2000.00, "high": 2000.50, "low": 1999.50, "close": 2000.40, "spread": 10},
            {"open": 2000.00, "high": 2000.50, "low": 1999.90, "close": 2000.20, "spread": 10},
            {"open": 2000.20, "high": 2001.00, "low": 2000.05, "close": 2000.10, "spread": 10},
            *flat_bars(3, price=2000.10),
        ]
    )
    # without tick_value: 1 point = point * contract_size = 0.01 * 200 = 2.0 per lot
    trade = run(bars, spec=symbol_spec(tick_value=0.0, contract_size=200.0)).trades.iloc[0]
    assert trade["gross_pnl"] == pytest.approx(90 * 0.01 * 200.0 * 0.10)


def test_zero_costs_minus_real_costs_equals_exactly_spread_plus_commission() -> None:
    """Time-stop exits, so the triggers do not depend on the spread.

    With stop and target the difference would not be isolable: the spread
    shifts the touch levels and the two runs would diverge in exit prices too.
    """
    strategy = spec_from(
        exit_block={
            "stop_loss": None,
            "take_profit": None,
            "time_stop": {"bars": 3},
            "signal_exit": None,
        },
        # fixed lot: with stepped sizing the higher equity of the cost-free run
        # would buy bigger lots and the comparison would not be isolated
        sizing={"type": "equity_per_step", "equity_per_001_lot": 100,
                "min_lot": 0.01, "max_lot": 0.01},
    )
    bars = random_walk(600, spread=12.0)
    costs = CostModel(
        spread=SpreadPolicy(mode="per_bar"),
        commission=CommissionModel(per_lot_per_side=2.5),
        swap=SwapModel(mode="none"),
    )

    real = run(bars, strategy, costs)
    free = run(bars, strategy, CostModel.zero())

    assert len(real.trades) == len(free.trades) > 10
    # same entries and same exits: only what was paid changes
    assert (real.trades["entry_time"].to_numpy() == free.trades["entry_time"].to_numpy()).all()
    assert (real.trades["exit_time"].to_numpy() == free.trades["exit_time"].to_numpy()).all()
    assert real.trades["gross_pnl"].sum() == pytest.approx(free.trades["gross_pnl"].sum())

    paid = real.trades["spread_cost"].sum() + real.trades["commission"].sum()
    difference = free.trades["net_pnl"].sum() - real.trades["net_pnl"].sum()
    assert difference == pytest.approx(paid, abs=1e-9)
    assert free.final_equity - real.final_equity == pytest.approx(paid, abs=1e-9)


def test_time_stop_counts_session_bars_and_flags_gaps() -> None:
    """A hole in the data must not gift the position extra time.

    Take a real trade from the run on continuous data, delete bars while that
    trade is open and rerun: the entry stays identical (signals only look
    back), but the time stop fires counting the missing bars too, instead of
    waiting for 10 array rows.
    """
    strategy = spec_from(
        exit_block={
            "stop_loss": None,
            "take_profit": None,
            "time_stop": {"bars": 10},
            "signal_exit": None,
        }
    )
    bars = random_walk(3 * 7 * 24 * 60, seed=2)  # three continuous weeks
    continuous = run(bars, strategy)
    assert (continuous.trades["session_bars_held"] <= 11).all()
    assert not continuous.trades["crossed_gap"].any()

    reference = continuous.trades.iloc[500]
    inside = bars.index[
        (bars.index > reference["entry_time"]) & (bars.index < reference["exit_time"])
    ]
    assert len(inside) >= 4
    holed = bars.drop(inside[1:])  # hole while the position is open

    result = run(holed, strategy)
    same = result.trades[result.trades["entry_time"] == reference["entry_time"]]
    assert len(same) == 1
    trade = same.iloc[0]

    assert bool(trade["crossed_gap"]) is True
    assert trade["session_bars_held"] > trade["bars_held"]
    assert trade["bars_held"] < reference["bars_held"]
    # trades that do not cross the hole stay within the time stop
    clean = result.trades[~result.trades["crossed_gap"]]
    assert (clean["session_bars_held"] <= 11).all()


def test_position_open_at_end_of_data_is_closed_and_flagged() -> None:
    strategy = spec_from(
        exit_block={
            "stop_loss": None,
            "take_profit": None,
            "time_stop": {"bars": 10_000},
            "signal_exit": None,
        }
    )
    bars = random_walk(200)
    result = run(bars, strategy)
    assert result.trades.iloc[-1]["exit_reason"] == "end_of_data"


def test_signal_exit_closes_on_next_open() -> None:
    strategy = spec_from(
        exit_block={
            "stop_loss": None,
            "take_profit": None,
            "time_stop": None,
            "signal_exit": {"op": "lt", "left": {"bar": "close"}, "right": {"bar": "open"}},
        }
    )
    bars = bars_from(
        [
            {"open": 2000.00, "high": 2000.50, "low": 1999.50, "close": 2000.40, "spread": 10},
            {"open": 2000.40, "high": 2000.60, "low": 2000.30, "close": 2000.50, "spread": 10},
            {"open": 2000.50, "high": 2000.60, "low": 2000.10, "close": 2000.20, "spread": 10},
            {"open": 2000.25, "high": 2000.30, "low": 2000.20, "close": 2000.25, "spread": 10},
            *flat_bars(2, price=2000.25),
        ]
    )
    trade = run(bars, strategy).trades.iloc[0]
    assert trade["exit_reason"] == "signal_exit"
    assert trade["exit_time"] == bars.index[3]
    assert trade["exit_price"] == pytest.approx(2000.25)


def test_risk_gates_applied_in_the_engine() -> None:
    strategy = spec_from(
        risk={"max_open_positions": 1, "max_spread_points": 5, "cooldown_minutes": 0},
    )
    bars = random_walk(500, spread=30.0)  # always above the spread cap
    result = run(bars, strategy)

    assert len(result.trades) == 0
    assert sum(result.blocked.values()) > 0
    assert any("spread" in reason for reason in result.blocked)


def test_cooldown_spaces_out_trades() -> None:
    strategy = spec_from(
        exit_block={
            "stop_loss": None, "take_profit": None,
            "time_stop": {"bars": 1}, "signal_exit": None,
        },
        risk={"max_open_positions": 1, "cooldown_minutes": 30},
    )
    bars = random_walk(2000, seed=9)
    trades = run(bars, strategy).trades
    assert len(trades) > 2
    gaps = trades["entry_time"].diff().dropna()
    assert (gaps >= pd.Timedelta(minutes=30)).all()


def test_low_equity_trades_at_minimum_lot() -> None:
    bars = bars_from(
        [
            {"open": 2000.00, "high": 2000.50, "low": 1999.50, "close": 2000.40, "spread": 10},
            *flat_bars(4),
        ]
    )
    result = run(bars, equity=50.0)  # below equity_per_001_lot: falls to min_lot
    assert len(result.trades) == 1
    assert result.trades.iloc[0]["lots"] == pytest.approx(0.01)


def test_zero_equity_blocks_the_entry() -> None:
    bars = bars_from(
        [
            {"open": 2000.00, "high": 2000.50, "low": 1999.50, "close": 2000.40, "spread": 10},
            *flat_bars(4),
        ]
    )
    result = run(bars, equity=0.0)
    assert len(result.trades) == 0
    assert result.blocked["insufficient_equity"] == 1


def test_multiple_open_positions_not_supported() -> None:
    strategy = spec_from(risk={"max_open_positions": 3})
    with pytest.raises(NotImplementedError, match="max_open_positions"):
        run(random_walk(50), strategy)
