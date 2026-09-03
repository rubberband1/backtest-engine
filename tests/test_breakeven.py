"""Break-even win rate: realized and a-priori, with the validity gates."""
from __future__ import annotations

import pandas as pd
import pytest

from core.data.provider import SymbolSpec
from core.engine.costs import CommissionModel
from core.metrics.breakeven import (
    BreakevenPrior,
    breakeven_from_trades,
    breakeven_prior,
)
from core.strategy.spec import StrategySpec


def _trades(rows: list[tuple[str, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "exit_reason": [reason for reason, _ in rows],
            "net_pnl": [pnl for _, pnl in rows],
        }
    )


def _symbol_spec() -> SymbolSpec:
    return SymbolSpec(
        name="TEST",
        point=0.01,
        digits=2,
        contract_size=100.0,
        tick_value=0.01,
        tick_size=0.01,
        volume_min=0.01,
        volume_max=10.0,
        volume_step=0.01,
        swap_long=0.0,
        swap_short=0.0,
        currency_profit="USD",
        trade_mode="full",
    )


def _spec(exit_block: dict) -> StrategySpec:
    return StrategySpec.from_dict(
        {
            "schema_version": 1,
            "id": "t",
            "name": "t",
            "description": "",
            "instrument": {"symbol": "TEST", "timeframe": "M1"},
            "indicators": [
                {"id": "rsi", "type": "rsi", "params": {"period": 9, "source": "close"}}
            ],
            "entry": {
                "long": {"op": "lt", "left": {"ref": "rsi"}, "right": {"const": 20}},
                "short": None,
            },
            "exit": exit_block,
            "sizing": {
                "type": "equity_per_step",
                "equity_per_001_lot": 100,
                "min_lot": 0.01,
                "max_lot": 0.05,
            },
            "risk": {"max_open_positions": 1},
        }
    )


class TestRealized:
    def test_binary_distribution(self) -> None:
        rows = [("take_profit", 0.8)] * 60 + [("stop_loss", -1.5)] * 40
        report = breakeven_from_trades(_trades(rows))
        assert report.valid
        # 1.5 / (1.5 + 0.8) = 0.652...
        assert report.breakeven_win_rate == pytest.approx(1.5 / 2.3)
        assert report.realized_win_rate == pytest.approx(0.60)
        assert report.delta == pytest.approx(0.60 - 1.5 / 2.3)
        assert report.observations == 100

    def test_non_binary_exits_invalidate(self) -> None:
        rows = [("take_profit", 0.8)] * 50 + [("stop_loss", -1.5)] * 40
        rows += [("time_stop", 0.1)] * 10
        report = breakeven_from_trades(_trades(rows))
        assert not report.valid
        assert report.reason is not None and "not binary" in report.reason
        assert report.non_binary_reasons == {"time_stop": 10}

    def test_dispersed_amounts_invalidate(self) -> None:
        # exit reasons look binary but the win amounts are all over the place
        # (a trailing stop would do this)
        wins = [("take_profit", 0.2 + 0.05 * i) for i in range(40)]
        losses = [("stop_loss", -1.5)] * 40
        report = breakeven_from_trades(_trades(wins + losses))
        assert not report.valid
        assert report.reason is not None and "dispersed" in report.reason

    def test_tolerates_few_gap_fills(self) -> None:
        rows = [("take_profit", 0.8)] * 58 + [("stop_loss", -1.5)] * 40
        rows += [("gap_stop_loss", -1.9), ("gap_take_profit", 1.1)]
        report = breakeven_from_trades(_trades(rows))
        assert report.valid
        assert report.non_binary_trades == 2

    def test_empty(self) -> None:
        report = breakeven_from_trades(_trades([]))
        assert not report.valid
        assert report.observations == 0

    def test_variable_exits_keep_the_number_and_declare_the_dispersion(self) -> None:
        """With ATR or percent exits every trade has its own distance."""
        wins = [("take_profit", 0.2 + 0.05 * i) for i in range(40)]
        losses = [("stop_loss", -(1.0 + 0.05 * i)) for i in range(40)]
        report = breakeven_from_trades(_trades(wins + losses), variable_exits=True)

        assert report.valid
        assert report.variable_exits is True
        avg_win = sum(pnl for _, pnl in wins) / len(wins)
        avg_loss = -sum(pnl for _, pnl in losses) / len(losses)
        assert report.breakeven_win_rate == pytest.approx(avg_loss / (avg_loss + avg_win))
        assert report.win_relative_std is not None and report.win_relative_std > 0.05
        assert report.loss_relative_std is not None and report.loss_relative_std > 0
        assert report.reason is not None and "weighted average" in report.reason

    def test_variable_exits_do_not_excuse_a_non_binary_distribution(self) -> None:
        rows = [("take_profit", 0.8)] * 50 + [("stop_loss", -1.5)] * 40
        rows += [("time_stop", 0.1)] * 10
        report = breakeven_from_trades(_trades(rows), variable_exits=True)
        assert not report.valid
        assert report.reason is not None and "not binary" in report.reason


class TestPrior:
    def test_baseline_numbers(self) -> None:
        spec = _spec(
            {
                "stop_loss": {"type": "points", "value": 150},
                "take_profit": {"type": "points", "value": 80},
                "time_stop": None,
                "signal_exit": None,
            }
        )
        prior = breakeven_prior(spec, _symbol_spec(), avg_spread_points=7.5)
        assert prior.valid
        # 150 / (150 + 80) = 65.2%
        assert prior.breakeven_win_rate == pytest.approx(150 / 230, abs=1e-6)
        assert any("spread" in caveat for caveat in prior.caveats)

    def test_commission_raises_threshold(self) -> None:
        spec = _spec(
            {
                "stop_loss": {"type": "points", "value": 150},
                "take_profit": {"type": "points", "value": 80},
                "time_stop": None,
                "signal_exit": None,
            }
        )
        # 0.005 per lot per side -> 0.01 round turn; value per point at 1 lot
        # is (0.01/0.01)*0.01 = 0.01 -> 1 point of commission
        prior = breakeven_prior(spec, _symbol_spec(), CommissionModel(0.005))
        assert prior.valid
        assert prior.breakeven_win_rate == pytest.approx(151 / 230, abs=1e-6)

    def test_missing_target_invalidates(self) -> None:
        spec = _spec(
            {
                "stop_loss": {"type": "points", "value": 150},
                "take_profit": None,
                "time_stop": None,
                "signal_exit": None,
            }
        )
        prior = breakeven_prior(spec, _symbol_spec())
        assert not prior.valid
        assert prior.reason is not None

    def test_variable_exits_need_a_measured_distance(self) -> None:
        spec = _spec(
            {
                "stop_loss": {"type": "percent", "value": 0.15},
                "take_profit": {"type": "percent", "value": 0.08},
                "time_stop": None,
                "signal_exit": None,
            }
        )
        prior = breakeven_prior(spec, _symbol_spec())
        assert not prior.valid
        assert prior.variable_exits is True
        assert prior.reason is not None and "no average distance" in prior.reason

    def test_variable_exits_with_measured_distances_are_stated_as_an_average(self) -> None:
        spec = _spec(
            {
                "stop_loss": {"type": "percent", "value": 0.15},
                "take_profit": {"type": "percent", "value": 0.08},
                "time_stop": None,
                "signal_exit": None,
            }
        )
        prior = breakeven_prior(
            spec, _symbol_spec(),
            stop_points=150.0, target_points=80.0,
            stop_points_std=12.0, target_points_std=6.0,
        )
        assert prior.valid
        assert prior.variable_exits is True
        assert prior.breakeven_win_rate == pytest.approx(150 / 230, abs=1e-6)
        assert any("sized per trade" in caveat for caveat in prior.caveats)

    def test_time_stop_adds_caveat(self) -> None:
        spec = _spec(
            {
                "stop_loss": {"type": "points", "value": 150},
                "take_profit": {"type": "points", "value": 80},
                "time_stop": {"bars": 60},
                "signal_exit": None,
            }
        )
        prior: BreakevenPrior = breakeven_prior(spec, _symbol_spec())
        assert prior.valid
        assert any("time stop" in caveat for caveat in prior.caveats)
