"""The uncertainty band: what the stop-first assumption is worth."""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from core.metrics.ambiguity import net_pnl_at, uncertainty_band
from core.research.edge import ambiguity_prior
from tests.conftest_engine import bars_from, flat_bars, spec_from, symbol_spec

UTC = timezone.utc


def _trade(
    ambiguous: bool = True,
    direction: int = 1,
    entry_price: float = 2000.0,
    net_pnl: float = -1.0,
    stop_level: float = 1999.0,
    target_level: float = 2000.5,
) -> dict:
    return {
        "direction": direction,
        "entry_time": datetime(2024, 1, 1, tzinfo=UTC),
        "entry_price": entry_price,
        "stop_level": stop_level,
        "target_level": target_level,
        "exit_time": datetime(2024, 1, 1, 0, 5, tzinfo=UTC),
        "exit_price": stop_level,
        "exit_reason": "stop_loss",
        "lots": 0.1,
        "bars_held": 5,
        "session_bars_held": 5,
        "gross_pnl": net_pnl,
        "spread_points": 0.0,
        "spread_cost": 0.0,
        "commission": 0.0,
        "swap": 0.0,
        "net_pnl": net_pnl,
        "risk_money": 1.0,
        "r_multiple": -1.0,
        "ambiguous": ambiguous,
        "crossed_gap": False,
    }


def test_a_run_without_ambiguous_trades_has_a_zero_band() -> None:
    band = uncertainty_band(pd.DataFrame([_trade(ambiguous=False)]), symbol_spec(), 100.0)
    assert band.ambiguous_trades == 0
    assert band.band_money == pytest.approx(0.0)
    assert band.conservative_net_pnl == pytest.approx(band.optimistic_net_pnl)
    assert band.exceeds_threshold is False
    assert "No ambiguous trade" in band.verdict


def test_the_band_is_the_distance_between_all_stops_and_all_targets() -> None:
    trades = pd.DataFrame([_trade(ambiguous=True), _trade(ambiguous=False, net_pnl=2.0)])
    band = uncertainty_band(trades, symbol_spec(), 100.0)

    optimistic_pnl = net_pnl_at(pd.Series(_trade()), 2000.5, symbol_spec())
    assert band.conservative_net_pnl == pytest.approx(1.0)
    assert band.optimistic_net_pnl == pytest.approx(optimistic_pnl + 2.0)
    assert band.band_money == pytest.approx(optimistic_pnl - (-1.0))
    assert band.conservative_final_equity == pytest.approx(101.0)
    assert band.optimistic_final_equity == pytest.approx(100.0 + optimistic_pnl + 2.0)


def test_the_share_above_the_threshold_is_declared_not_conclusive() -> None:
    trades = pd.DataFrame([_trade(ambiguous=True)] + [_trade(ambiguous=False)] * 9)
    band = uncertainty_band(trades, symbol_spec(), 100.0)
    assert band.ambiguous_share == pytest.approx(0.10)
    assert band.exceeds_threshold is True
    assert "not conclusive" in band.verdict


def test_trades_without_a_target_level_narrow_the_band_and_say_so() -> None:
    """Runs written before the levels were recorded must not fake a band."""
    row = _trade(ambiguous=True)
    row["target_level"] = float("nan")
    band = uncertainty_band(pd.DataFrame([row]), symbol_spec(), 100.0)
    assert band.resolvable is False
    assert band.band_money == pytest.approx(0.0)
    assert any("no target level" in warning for warning in band.warnings)


def test_the_win_rate_moves_with_the_band() -> None:
    band = uncertainty_band(pd.DataFrame([_trade(ambiguous=True)]), symbol_spec(), 100.0)
    assert band.conservative_win_rate == pytest.approx(0.0)
    assert band.optimistic_win_rate == pytest.approx(1.0)


# -- the a-priori estimate -----------------------------------------------


def _prior_spec(stop: float, target: float):
    return spec_from(
        exit_block={
            "stop_loss": {"type": "points", "value": stop},
            "take_profit": {"type": "points", "value": target},
            "time_stop": {"bars": 60},
            "signal_exit": None,
        }
    )


def _bars_with_range(points: float, n: int = 200) -> pd.DataFrame:
    half = points * 0.01 / 2.0
    return bars_from(
        [
            {"open": 2000.0, "high": 2000.0 + half, "low": 2000.0 - half,
             "close": 2000.0, "spread": 1}
            for _ in range(n)
        ]
    )


def test_bars_that_never_reach_a_level_have_nothing_to_resolve() -> None:
    """20-point bars against an 80-point target: the levels are never touched."""
    prior = ambiguity_prior(_prior_spec(150, 80), _bars_with_range(20), 0.01, {})
    assert prior.applicable is False
    assert prior.level_distance_points == pytest.approx(230.0)
    assert prior.nearest_level_points == pytest.approx(80.0)
    assert "never reached" in prior.verdict


def test_bars_that_reach_only_the_nearer_level_are_conclusive() -> None:
    """100-point bars reach the 80-point target but never span both levels."""
    prior = ambiguity_prior(_prior_spec(150, 80), _bars_with_range(100), 0.01, {})
    assert prior.applicable is True
    assert prior.any_reachable_share == pytest.approx(1.0)
    assert prior.both_reachable_share == pytest.approx(0.0)
    assert prior.expected_ambiguous_share == pytest.approx(0.0)
    assert prior.exceeds_threshold is False


def test_bars_wider_than_the_two_levels_make_the_run_untestable_on_bars() -> None:
    prior = ambiguity_prior(_prior_spec(150, 80), _bars_with_range(400), 0.01, {})
    assert prior.both_reachable_share == pytest.approx(1.0)
    assert prior.any_reachable_share == pytest.approx(1.0)
    assert prior.expected_ambiguous_share == pytest.approx(1.0)
    assert prior.exceeds_threshold is True
    assert "stop-first assumption" in prior.verdict


def test_a_spec_without_both_levels_has_no_ambiguity_to_estimate() -> None:
    spec = spec_from(
        exit_block={
            "stop_loss": {"type": "points", "value": 150},
            "take_profit": None,
            "time_stop": {"bars": 60},
            "signal_exit": None,
        }
    )
    prior = ambiguity_prior(spec, _bars_with_range(20), 0.01, {})
    assert prior.applicable is False
    assert "no stop and target pair" in (prior.reason or "")


def test_the_engine_and_the_estimate_agree_on_a_run_with_ambiguity() -> None:
    """A bar wide enough to hold both levels produces exactly the ambiguous trades."""
    from zoneinfo import ZoneInfo

    from core.engine.backtester import BacktestConfig, run_backtest
    from core.engine.costs import CostModel, SpreadPolicy

    spec = _prior_spec(100, 100)
    bars = bars_from(
        [
            {"open": 2000.00, "high": 2000.50, "low": 1999.50, "close": 2000.40, "spread": 0},
            # entry at 2000.00: stop 1999.00, target 2001.00, both inside this bar
            {"open": 2000.00, "high": 2001.50, "low": 1998.50, "close": 2000.00, "spread": 0},
            *flat_bars(3, price=2000.0, spread=0),
        ]
    )
    result = run_backtest(
        spec, bars, symbol_spec(), ZoneInfo("Europe/Athens"),
        BacktestConfig(
            initial_equity=1000.0,
            costs=CostModel(spread=SpreadPolicy(mode="fixed", value=0.0)),
        ),
    )
    assert result.ambiguous_trades == 1
    band = uncertainty_band(result.trades, symbol_spec(), 1000.0)
    assert band.ambiguous_share == pytest.approx(1.0)
    # the conservative reading is a stop, the optimistic one a target: the band
    # is the whole distance between the two outcomes
    assert band.band_money > 0
    assert band.optimistic_net_pnl > band.conservative_net_pnl
