"""Permutation nulls: construction of the draws, and the empirical p-value."""
from __future__ import annotations

from datetime import timezone

import numpy as np
import pandas as pd
import pytest

from core.engine.backtester import BacktestConfig
from core.engine.costs import CostModel
from core.validation.permutation import (
    _empirical,
    _statistics,
    block_bootstrap_bars,
    permutation_test,
    random_entry_signals,
)
from tests.conftest_engine import random_walk, spec_from, symbol_spec

UTC = timezone.utc


def _spec():
    return spec_from(
        exit_block={
            "stop_loss": {"type": "points", "value": 40},
            "take_profit": {"type": "points", "value": 40},
            "time_stop": {"bars": 30},
            "signal_exit": None,
        }
    )


# -- random entries ------------------------------------------------------


def test_random_entries_place_exactly_the_requested_number_of_signals() -> None:
    bars = random_walk(5000)
    signals = random_entry_signals(bars.index, 30, 20, np.random.default_rng(1))
    assert int(signals.long.sum()) == 30
    assert int(signals.short.sum()) == 20


def test_random_entries_never_put_two_signals_on_the_same_bar() -> None:
    bars = random_walk(500)
    signals = random_entry_signals(bars.index, 100, 100, np.random.default_rng(2))
    assert not (signals.long & signals.short).any()


def test_random_entries_are_reproducible_from_the_seed() -> None:
    bars = random_walk(2000)
    first = random_entry_signals(bars.index, 25, 25, np.random.default_rng(7))
    second = random_entry_signals(bars.index, 25, 25, np.random.default_rng(7))
    assert first.long.equals(second.long)
    assert first.short.equals(second.short)


def test_random_entries_cannot_ask_for_more_signals_than_bars() -> None:
    bars = random_walk(10)
    signals = random_entry_signals(bars.index, 50, 50, np.random.default_rng(3))
    assert int(signals.long.sum()) + int(signals.short.sum()) <= 10


# -- block bootstrap -----------------------------------------------------


def test_bootstrapped_bars_keep_the_index_and_the_spread() -> None:
    bars = random_walk(3000)
    synthetic = block_bootstrap_bars(bars, 128, np.random.default_rng(1))
    assert synthetic.index.equals(bars.index)
    assert synthetic["spread"].equals(bars["spread"])
    assert list(synthetic.columns) == list(bars.columns)


def test_bootstrapped_bars_are_still_valid_bars() -> None:
    """High above open and close, low below: a shuffle must not break OHLC."""
    bars = random_walk(4000, seed=9)
    synthetic = block_bootstrap_bars(bars, 64, np.random.default_rng(5))
    assert (synthetic["high"] >= synthetic["open"] - 1e-9).all()
    assert (synthetic["high"] >= synthetic["close"] - 1e-9).all()
    assert (synthetic["low"] <= synthetic["open"] + 1e-9).all()
    assert (synthetic["low"] <= synthetic["close"] + 1e-9).all()


def test_bootstrapped_path_starts_at_the_real_first_close() -> None:
    bars = random_walk(2000)
    synthetic = block_bootstrap_bars(bars, 100, np.random.default_rng(4))
    assert synthetic["close"].iloc[0] == pytest.approx(bars["close"].iloc[0])


def test_bootstrap_changes_the_path_but_keeps_the_return_distribution() -> None:
    bars = random_walk(20000, seed=21)
    synthetic = block_bootstrap_bars(bars, 256, np.random.default_rng(6))
    assert not np.allclose(synthetic["close"].to_numpy(), bars["close"].to_numpy())
    real = np.diff(np.log(bars["close"].to_numpy()))
    fake = np.diff(np.log(synthetic["close"].to_numpy()))
    # the same returns are being reused, so the dispersion has to survive
    assert fake.std() == pytest.approx(real.std(), rel=0.15)


def test_bootstrap_is_reproducible_from_the_seed() -> None:
    bars = random_walk(3000)
    first = block_bootstrap_bars(bars, 64, np.random.default_rng(11))
    second = block_bootstrap_bars(bars, 64, np.random.default_rng(11))
    assert first["close"].equals(second["close"])


def test_a_block_longer_than_the_series_is_clamped_not_an_error() -> None:
    bars = random_walk(50)
    synthetic = block_bootstrap_bars(bars, 10_000, np.random.default_rng(1))
    assert len(synthetic) == len(bars)


def test_a_series_too_short_to_resample_comes_back_untouched() -> None:
    bars = random_walk(2)
    synthetic = block_bootstrap_bars(bars, 10, np.random.default_rng(1))
    assert synthetic["close"].equals(bars["close"])


# -- the empirical p-value -----------------------------------------------


def test_the_p_value_can_never_reach_zero() -> None:
    null = np.zeros(1000)
    _, p_value = _empirical(10.0, null)
    assert p_value == pytest.approx(1.0 / 1001.0)
    assert p_value > 0.0


def test_a_result_at_the_bottom_of_the_null_gets_a_p_value_of_one() -> None:
    null = np.ones(100)
    percentile, p_value = _empirical(0.0, null)
    assert percentile == 0.0
    assert p_value == pytest.approx(1.0)


def test_the_percentile_counts_the_draws_strictly_below() -> None:
    null = np.arange(100, dtype="float64")
    percentile, _ = _empirical(25.0, null)
    assert percentile == pytest.approx(25.0)


def test_statistics_of_an_empty_trade_list_are_zero_not_nan() -> None:
    values = _statistics(pd.DataFrame(), pd.Series(dtype="float64"), 100.0)
    assert values == {"net_pnl": 0.0, "expectancy": 0.0, "sharpe": 0.0, "trades": 0.0}


def test_expectancy_is_the_mean_and_net_pnl_the_sum() -> None:
    trades = pd.DataFrame({"net_pnl": [2.0, -1.0, 3.0]})
    values = _statistics(trades, pd.Series(dtype="float64"), 100.0)
    assert values["net_pnl"] == pytest.approx(4.0)
    assert values["expectancy"] == pytest.approx(4.0 / 3.0)
    assert values["trades"] == 3.0


# -- end to end ----------------------------------------------------------


@pytest.mark.parametrize("kind", ["random_entries", "permuted_returns"])
def test_a_small_permutation_run_produces_a_complete_report(kind: str) -> None:
    bars = random_walk(20000, seed=31)
    report = permutation_test(
        kind,
        _spec(),
        bars,
        symbol_spec(),
        UTC,
        BacktestConfig(initial_equity=1000.0, costs=CostModel.zero()),
        iterations=12,
        block_bars=256,
        seed=99,
        max_workers=2,
    )
    assert report.iterations == 12
    assert report.kind == kind
    assert {statistic.key for statistic in report.statistics} == {
        "net_pnl",
        "expectancy",
        "sharpe",
    }
    for statistic in report.statistics:
        assert 0.0 < statistic.p_value <= 1.0
        assert 0.0 <= statistic.percentile <= 100.0
        assert statistic.iterations <= 12
    assert report.histogram is not None
    assert len(report.histogram.counts) == len(report.histogram.edges) - 1
    assert report.verdict


def test_the_random_entry_null_reproduces_the_real_trade_count() -> None:
    """The null must trade about as often as the strategy, or the totals lie."""
    bars = random_walk(20000, seed=33)
    report = permutation_test(
        "random_entries",
        _spec(),
        bars,
        symbol_spec(),
        UTC,
        BacktestConfig(initial_equity=1000.0, costs=CostModel.zero()),
        iterations=10,
        seed=5,
        max_workers=2,
    )
    assert report.observed_trades > 0
    assert report.null_trades_mean == pytest.approx(report.observed_trades, rel=0.35)


def test_the_same_seed_gives_the_same_permutation_report() -> None:
    bars = random_walk(15000, seed=41)
    config = BacktestConfig(initial_equity=1000.0, costs=CostModel.zero())
    kwargs = {"iterations": 8, "seed": 1234, "max_workers": 2}
    first = permutation_test(
        "random_entries", _spec(), bars, symbol_spec(), UTC, config, **kwargs
    )
    second = permutation_test(
        "random_entries", _spec(), bars, symbol_spec(), UTC, config, **kwargs
    )
    assert [s.p_value for s in first.statistics] == [s.p_value for s in second.statistics]


def test_zero_iterations_is_rejected() -> None:
    bars = random_walk(1000)
    with pytest.raises(ValueError, match="at least 1"):
        permutation_test(
            "random_entries",
            _spec(),
            bars,
            symbol_spec(),
            UTC,
            BacktestConfig(),
            iterations=0,
        )


def test_the_report_serializes_to_json_safe_primitives() -> None:
    import json

    bars = random_walk(12000, seed=43)
    report = permutation_test(
        "random_entries",
        _spec(),
        bars,
        symbol_spec(),
        UTC,
        BacktestConfig(initial_equity=1000.0, costs=CostModel.zero()),
        iterations=6,
        max_workers=2,
    )
    json.dumps(report.as_dict())
