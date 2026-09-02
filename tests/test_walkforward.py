"""Walk-forward: windows, embargo, trade-count gate, and no leakage."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from core.engine.backtester import BacktestConfig
from core.engine.costs import CostModel
from core.validation.walkforward import (
    WalkForwardConfig,
    WalkForwardError,
    apply_params,
    build_windows,
    embargo_minutes,
    expand_grid,
    set_in,
    walk_forward,
)
from tests.conftest_engine import random_walk, spec_from, symbol_spec

UTC = timezone.utc


def _config(equity: float = 1000.0) -> BacktestConfig:
    return BacktestConfig(initial_equity=equity, costs=CostModel.zero())


def _spec():
    return spec_from(
        exit_block={
            "stop_loss": {"type": "points", "value": 40},
            "take_profit": {"type": "points", "value": 40},
            "time_stop": {"bars": 30},
            "signal_exit": None,
        }
    )


# -- grid ----------------------------------------------------------------


def test_expand_grid_is_a_cartesian_product_in_a_stable_order() -> None:
    grid = {"b": [1, 2], "a": ["x", "y"]}
    combinations = expand_grid(grid)
    assert len(combinations) == 4
    # keys sorted, so two runs of the same grid enumerate identically
    assert combinations[0] == {"a": "x", "b": 1}
    assert expand_grid(grid) == combinations


def test_empty_grid_is_a_single_candidate() -> None:
    assert expand_grid({}) == [{}]


def test_grid_with_an_empty_value_list_is_rejected() -> None:
    with pytest.raises(WalkForwardError, match="no value to try"):
        expand_grid({"exit.stop_loss.value": []})


def test_set_in_rejects_a_path_that_does_not_exist() -> None:
    with pytest.raises(WalkForwardError, match="does not exist"):
        set_in({"exit": {"stop_loss": {}}}, "exit.take_profit.value", 10)


def test_apply_params_returns_a_revalidated_copy() -> None:
    spec = _spec()
    changed = apply_params(spec, {"exit.stop_loss.value": 999})
    assert changed.exit.stop_loss is not None and changed.exit.stop_loss.value == 999
    # the original is untouched: a candidate must not mutate the base spec
    assert spec.exit.stop_loss is not None and spec.exit.stop_loss.value == 40


def test_apply_params_rejects_a_candidate_the_spec_refuses() -> None:
    with pytest.raises(WalkForwardError, match="invalid candidate"):
        apply_params(_spec(), {"exit.stop_loss.value": -5})


# -- embargo -------------------------------------------------------------


def test_embargo_comes_from_the_time_stop_when_there_are_no_trades() -> None:
    # 30 bars of M1
    assert embargo_minutes(_spec()) == pytest.approx(30.0)


def test_embargo_takes_the_longest_observed_holding_when_it_exceeds_the_time_stop() -> None:
    entry = pd.Timestamp("2024-01-01 00:00", tz="UTC")
    trades = pd.DataFrame(
        {
            "entry_time": [entry, entry],
            "exit_time": [entry + timedelta(minutes=10), entry + timedelta(minutes=120)],
        }
    )
    assert embargo_minutes(_spec(), trades) == pytest.approx(120.0)


def test_embargo_is_zero_without_a_time_stop_or_trades() -> None:
    spec = spec_from(
        exit_block={
            "stop_loss": {"type": "points", "value": 40},
            "take_profit": {"type": "points", "value": 40},
            "time_stop": None,
            "signal_exit": None,
        }
    )
    assert embargo_minutes(spec) == 0.0


# -- windows -------------------------------------------------------------


def test_windows_leave_the_embargo_between_the_two_legs() -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    windows = build_windows(
        start,
        start + timedelta(days=200),
        WalkForwardConfig(train_days=90, test_days=30),
        timedelta(minutes=60),
    )
    assert windows
    for train_start, train_end, test_start, test_end in windows:
        assert test_start - train_end == timedelta(minutes=60)
        assert train_end - train_start == timedelta(days=90)
        assert test_end - test_start == timedelta(days=30)


def test_anchored_windows_keep_the_same_train_start() -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    windows = build_windows(
        start,
        start + timedelta(days=200),
        WalkForwardConfig(mode="anchored", train_days=90, test_days=30),
        timedelta(0),
    )
    assert len({window[0] for window in windows}) == 1
    # the train leg grows one test period at a time
    assert windows[1][1] - windows[0][1] == timedelta(days=30)


def test_rolling_windows_slide_both_ends() -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    windows = build_windows(
        start,
        start + timedelta(days=200),
        WalkForwardConfig(mode="rolling", train_days=90, test_days=30),
        timedelta(0),
    )
    assert windows[1][0] - windows[0][0] == timedelta(days=30)


def test_a_truncated_final_window_is_not_produced() -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    windows = build_windows(
        start,
        start + timedelta(days=125),
        WalkForwardConfig(train_days=90, test_days=30),
        timedelta(0),
    )
    # 90 + 30 fits once; the next test leg would run past the data
    assert len(windows) == 1


def test_too_little_data_is_an_error_that_says_how_much_was_there() -> None:
    bars = random_walk(400)
    with pytest.raises(WalkForwardError, match="not enough for a"):
        walk_forward(
            _spec(),
            bars,
            symbol_spec(),
            UTC,
            _config(),
            WalkForwardConfig(train_days=90, test_days=30),
        )


# -- the trade-count gate ------------------------------------------------


def test_a_window_without_enough_train_trades_is_discarded_and_says_why() -> None:
    bars = random_walk(60 * 24 * 120, seed=3)
    report = walk_forward(
        _spec(),
        bars,
        symbol_spec(),
        UTC,
        _config(),
        WalkForwardConfig(train_days=30, test_days=15, min_train_trades=10**6),
    )
    assert report.windows
    assert all(window.skipped for window in report.windows)
    assert report.windows_evaluated == 0
    assert "nothing to select on" in (report.windows[0].skip_reason or "")
    # a walk-forward with no usable window must not read as a negative result
    assert "says nothing" in report.verdict


def test_a_permissive_gate_keeps_the_windows() -> None:
    bars = random_walk(60 * 24 * 120, seed=3)
    report = walk_forward(
        _spec(),
        bars,
        symbol_spec(),
        UTC,
        _config(),
        WalkForwardConfig(train_days=30, test_days=15, min_train_trades=0),
    )
    assert report.windows_evaluated > 0
    assert report.oos_metrics is not None


# -- no leakage ----------------------------------------------------------


def test_out_of_sample_results_ignore_data_after_the_window() -> None:
    """The core guarantee: a window may not see past its own test end.

    The tail of the series is replaced with a violent, completely different
    path. Every window that closes before the mutation must come back
    identical - same chosen parameters, same trades, same metrics.
    """
    bars = random_walk(60 * 24 * 120, seed=11)
    config = WalkForwardConfig(train_days=30, test_days=15, min_train_trades=0)
    grid = {"exit.stop_loss.value": [30, 40, 60]}

    original = walk_forward(_spec(), bars, symbol_spec(), UTC, _config(), config, grid=grid)
    assert original.windows_evaluated >= 3

    cut = original.windows[1].test_end
    mutated = bars.copy()
    after = mutated.index >= pd.Timestamp(cut).tz_convert("UTC")
    assert after.sum() > 0
    for column in ("open", "high", "low", "close"):
        mutated.loc[after, column] = mutated.loc[after, column] * 3.0 + 500.0

    changed = walk_forward(_spec(), mutated, symbol_spec(), UTC, _config(), config, grid=grid)

    for before_window, after_window in zip(original.windows, changed.windows):
        if before_window.test_end > cut:
            break
        assert after_window.chosen_params == before_window.chosen_params
        assert after_window.train_trades == before_window.train_trades
        assert after_window.test_trades == before_window.test_trades
        assert after_window.test_metrics == before_window.test_metrics


def test_the_test_leg_of_a_window_never_reuses_a_train_bar() -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    windows = build_windows(
        start,
        start + timedelta(days=300),
        WalkForwardConfig(train_days=90, test_days=30),
        timedelta(minutes=30),
    )
    for _, train_end, test_start, _ in windows:
        assert test_start > train_end


# -- outputs -------------------------------------------------------------


def test_equity_is_carried_from_one_window_to_the_next() -> None:
    bars = random_walk(60 * 24 * 120, seed=7)
    report = walk_forward(
        _spec(),
        bars,
        symbol_spec(),
        UTC,
        _config(),
        WalkForwardConfig(train_days=30, test_days=15, min_train_trades=0),
    )
    evaluated = [window for window in report.windows if not window.skipped]
    assert len(evaluated) >= 2
    for previous, following in zip(evaluated, evaluated[1:]):
        assert following.equity_start == pytest.approx(previous.equity_end)


def test_without_a_grid_the_report_says_it_optimized_nothing() -> None:
    bars = random_walk(60 * 24 * 120, seed=7)
    report = walk_forward(
        _spec(),
        bars,
        symbol_spec(),
        UTC,
        _config(),
        WalkForwardConfig(train_days=30, test_days=15, min_train_trades=0),
    )
    assert report.optimized is False
    assert report.grid_size == 1
    assert any("no parameter grid" in warning for warning in report.warnings)
    assert report.parameter_stability == []


def test_parameter_stability_reports_what_the_optimizer_kept_choosing() -> None:
    bars = random_walk(60 * 24 * 120, seed=13)
    report = walk_forward(
        _spec(),
        bars,
        symbol_spec(),
        UTC,
        _config(),
        WalkForwardConfig(train_days=30, test_days=15, min_train_trades=0),
        grid={"exit.stop_loss.value": [30, 40, 60]},
    )
    assert report.optimized is True
    stability = {row["parameter"]: row for row in report.parameter_stability}
    row = stability["exit.stop_loss.value"]
    assert row["observations"] == report.windows_evaluated
    assert len(row["chosen"]) == report.windows_evaluated
    assert 0.0 < row["mode_share"] <= 1.0


def test_degradation_hides_the_ratio_when_the_in_sample_mean_is_not_positive() -> None:
    bars = random_walk(60 * 24 * 120, seed=17)
    report = walk_forward(
        _spec(),
        bars,
        symbol_spec(),
        UTC,
        _config(),
        WalkForwardConfig(train_days=30, test_days=15, min_train_trades=0),
    )
    for row in report.degradation:
        assert row["observations"] == report.windows_evaluated
        if row["in_sample_mean"] is not None and row["in_sample_mean"] <= 0:
            assert row["ratio"] is None
            assert row["ratio_note"]


def test_the_verdict_flags_a_thin_out_of_sample_sample() -> None:
    bars = random_walk(60 * 24 * 120, seed=19)
    report = walk_forward(
        _spec(),
        bars,
        symbol_spec(),
        UTC,
        _config(),
        WalkForwardConfig(train_days=30, test_days=15, min_train_trades=0),
    )
    if report.oos_trades < 100:
        assert "no power" in report.verdict


def test_the_report_serializes_to_json_safe_primitives() -> None:
    import json

    bars = random_walk(60 * 24 * 120, seed=23)
    report = walk_forward(
        _spec(),
        bars,
        symbol_spec(),
        UTC,
        _config(),
        WalkForwardConfig(train_days=30, test_days=15, min_train_trades=0),
    )
    json.dumps(report.as_dict())
