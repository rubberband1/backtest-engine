"""Metrics: hand-verifiable values and annualization derived from the data."""
from __future__ import annotations

from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from core.data.provider import Timeframe
from core.engine.backtester import TRADE_COLUMNS
from core.engine.costs import CostModel
from core.metrics.performance import buy_and_hold, compute_metrics, periods_per_year
from core.strategy.spec import Sizing
from tests.conftest_engine import random_walk, symbol_spec

ATHENS = ZoneInfo("Europe/Athens")


def trades_from(pnls: list[float], start: str = "2024-01-01") -> pd.DataFrame:
    times = pd.date_range(start, periods=len(pnls), freq="1h", tz="UTC")
    rows = []
    for moment, pnl in zip(times, pnls):
        rows.append(
            {
                "direction": 1,
                "entry_time": moment,
                "entry_price": 2000.0,
                "exit_time": moment + pd.Timedelta(minutes=30),
                "exit_price": 2000.0 + pnl,
                "exit_reason": "take_profit" if pnl > 0 else "stop_loss",
                "lots": 0.01,
                "bars_held": 30,
                "session_bars_held": 30,
                "gross_pnl": pnl,
                "spread_points": 10.0,
                "spread_cost": 0.0,
                "commission": 0.0,
                "swap": 0.0,
                "net_pnl": pnl,
                "risk_money": 10.0,
                "r_multiple": pnl / 10.0,
                "ambiguous": False,
                "crossed_gap": False,
            }
        )
    return pd.DataFrame(rows, columns=list(TRADE_COLUMNS))


def equity_from(pnls: list[float], initial: float = 100.0) -> pd.Series:
    index = pd.date_range("2024-01-01", periods=len(pnls) + 1, freq="1h", tz="UTC")
    return pd.Series(initial + np.cumsum([0.0, *pnls]), index=index, name="equity")


def test_profit_factor_expectancy_and_winrate() -> None:
    pnls = [10.0, -5.0, 8.0, -3.0, -2.0]
    report = compute_metrics(
        trades_from(pnls), equity_from(pnls), Timeframe.H1, 100.0, "test"
    )
    assert report.trades == 5
    assert report.win_rate == pytest.approx(2 / 5)
    assert report.profit_factor == pytest.approx(18.0 / 10.0)
    assert report.expectancy == pytest.approx(8.0 / 5)
    assert report.final_equity == pytest.approx(108.0)
    assert report.total_return == pytest.approx(0.08)


def test_max_drawdown_absolute_and_percent() -> None:
    # 100 -> 120 -> 90 -> 110: peak 120, trough 90
    pnls = [20.0, -30.0, 20.0]
    report = compute_metrics(
        trades_from(pnls), equity_from(pnls), Timeframe.H1, 100.0, "test"
    )
    assert report.max_drawdown_money == pytest.approx(30.0)
    assert report.max_drawdown_pct == pytest.approx(30.0 / 120.0)


def test_max_losing_streak() -> None:
    pnls = [-1.0, -1.0, 5.0, -1.0, -1.0, -1.0, 2.0]
    report = compute_metrics(
        trades_from(pnls), equity_from(pnls), Timeframe.H1, 100.0, "test"
    )
    assert report.max_losing_streak == 3


def test_t_stat_on_series_without_edge() -> None:
    """Zero mean: the t must be null and the p-value high."""
    pnls = [5.0, -5.0] * 25
    report = compute_metrics(
        trades_from(pnls), equity_from(pnls), Timeframe.H1, 100.0, "test"
    )
    assert abs(report.t_stat) < 1e-9
    assert report.p_value > 0.99


def test_t_stat_on_series_with_edge() -> None:
    rng = np.random.default_rng(1)
    pnls = list(rng.normal(2.0, 1.0, 200))
    report = compute_metrics(
        trades_from(pnls), equity_from(pnls), Timeframe.H1, 100.0, "test"
    )
    assert report.t_stat > 10
    assert report.p_value < 0.001


def test_r_multiples() -> None:
    pnls = [10.0, -10.0, 20.0, -5.0]
    report = compute_metrics(
        trades_from(pnls), equity_from(pnls), Timeframe.H1, 100.0, "test"
    )
    assert report.r_multiples["mean"] == pytest.approx(0.375)
    assert report.r_multiples["min"] == pytest.approx(-1.0)
    assert report.r_multiples["max"] == pytest.approx(2.0)


def test_annualization_derived_from_the_calendar() -> None:
    """A 24/5 instrument does not share the factor of a 24/7 one."""
    continuous = pd.date_range("2024-01-01", periods=6 * 7 * 24 * 60, freq="1min", tz="UTC")
    weekdays_only = continuous[continuous.dayofweek < 5]

    full = periods_per_year(continuous, Timeframe.M1)
    partial = periods_per_year(weekdays_only, Timeframe.M1)

    assert full == pytest.approx(7 * 24 * 60 * 365.25 / 7)
    assert partial == pytest.approx(5 * 24 * 60 * 365.25 / 7)
    assert partial < full


def test_sharpe_scales_with_annualization() -> None:
    rng = np.random.default_rng(7)
    pnls = list(rng.normal(0.05, 1.0, 500))
    report = compute_metrics(
        trades_from(pnls), equity_from(pnls), Timeframe.H1, 1000.0, "test"
    )
    assert np.isfinite(report.sharpe)
    assert report.sortino != 0.0


def test_empty_report_does_not_blow_up() -> None:
    empty = pd.DataFrame(columns=list(TRADE_COLUMNS))
    report = compute_metrics(empty, pd.Series(dtype="float64"), Timeframe.M1, 100.0, "empty")
    assert report.trades == 0
    assert report.final_equity == 100.0
    assert "empty" in report.as_text()


def test_buy_and_hold_pays_the_spread_once() -> None:
    bars = random_walk(2000, spread=10.0)
    spec = symbol_spec(tick_value=1.0, tick_size=0.01, point=0.01)
    sizing = Sizing(type="equity_per_step", equity_per_001_lot=100, min_lot=0.01, max_lot=1.0)

    report = buy_and_hold(bars, spec, sizing, Timeframe.M1, 1000.0, CostModel(), ATHENS)

    lots = 0.10
    value = 0.10
    expected_gross = (bars["close"].iloc[-1] - bars["open"].iloc[0]) / 0.01 * value
    assert report.trades == 1
    assert report.costs["gross"] == pytest.approx(expected_gross)
    assert report.costs["spread"] == pytest.approx(10 * value)
    assert report.exposure == pytest.approx(1.0)
    assert report.label == "buy & hold"
    assert lots > 0


def test_buy_and_hold_without_costs_is_just_the_price() -> None:
    bars = random_walk(500)
    spec = symbol_spec()
    sizing = Sizing(type="equity_per_step", equity_per_001_lot=100, min_lot=0.01, max_lot=1.0)
    report = buy_and_hold(bars, spec, sizing, Timeframe.M1, 1000.0, CostModel.zero(), ATHENS)
    assert report.costs["net"] == pytest.approx(report.costs["gross"])
