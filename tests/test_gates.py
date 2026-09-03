"""Gate accounting: the signals that never became trades, counted."""
from __future__ import annotations

from collections import Counter
from zoneinfo import ZoneInfo

import pytest

from core.engine.backtester import BacktestConfig, run_backtest
from core.engine.costs import CostModel, SpreadPolicy
from core.metrics.gates import gate_accounting
from tests.conftest_engine import bars_from, flat_bars, spec_from, symbol_spec

ATHENS = ZoneInfo("Europe/Athens")


def _signal_bars(n: int, spread: float) -> list[dict]:
    """Bars that all fire the default long signal (close > open)."""
    return [
        {"open": 2000.0, "high": 2000.6, "low": 1999.4, "close": 2000.4, "spread": spread}
        for _ in range(n)
    ]


def _run(spec, bars):
    return run_backtest(
        spec, bars, symbol_spec(), ATHENS,
        BacktestConfig(
            initial_equity=1000.0,
            costs=CostModel(spread=SpreadPolicy(mode="per_bar")),
        ),
    )


def test_the_spread_gate_is_counted_once_per_rejection_not_once_per_value() -> None:
    """Counting on the human reason gives one bucket per spread value: it must not."""
    rows = [
        {"open": 2000.0, "high": 2000.6, "low": 1999.4, "close": 2000.4,
         "spread": 40 + index}
        for index in range(8)
    ]
    spec = spec_from(risk={"max_open_positions": 1, "max_spread_points": 30})
    result = _run(spec, bars_from(rows))

    assert set(result.blocked) == {"max_spread_points"}
    assert result.blocked["max_spread_points"] == result.entry_attempts


def test_a_gate_rejecting_most_signals_produces_a_warning() -> None:
    accounting = gate_accounting(
        Counter({"max_spread_points": 60, "cooldown": 10}),
        signals=100,
        executed=30,
        entry_attempts=100,
    )
    assert accounting.rejected == 70
    assert accounting.executed_share == pytest.approx(0.30)
    assert accounting.rows[0].code == "max_spread_points"
    assert accounting.rows[0].share == pytest.approx(0.60)
    assert accounting.rows[0].exceeds_threshold is True
    # the cooldown at 10% stays below the threshold and raises nothing
    assert len(accounting.warnings) == 1
    assert "spread" in accounting.warnings[0]


def test_a_clean_run_says_no_gate_rejected_anything() -> None:
    accounting = gate_accounting(Counter(), signals=12, executed=12)
    assert accounting.rejected == 0
    assert not accounting.warnings
    assert "no gate rejected" in accounting.verdict


def test_signals_arriving_while_in_a_position_are_counted_not_dropped() -> None:
    """Without this the engine's busiest filter is invisible in every report."""
    bars = bars_from(_signal_bars(12, spread=5))
    spec = spec_from(
        exit_block={
            "stop_loss": None, "take_profit": None,
            "time_stop": {"bars": 8}, "signal_exit": None,
        },
    )
    result = _run(spec, bars)

    signals = result.signals.counts["long"] + result.signals.counts["short"]
    accounted = int(len(result.trades)) + sum(result.blocked.values())
    assert result.blocked["position_open"] > 0
    assert accounted == signals


def test_the_accounting_adds_up_on_a_run_with_several_gates() -> None:
    bars = bars_from(_signal_bars(30, spread=5) + flat_bars(5, spread=5))
    spec = spec_from(
        risk={"max_open_positions": 1, "cooldown_minutes": 5, "max_trades_per_day": 2},
        exit_block={
            "stop_loss": {"type": "points", "value": 50},
            "take_profit": {"type": "points", "value": 50},
            "time_stop": {"bars": 3},
            "signal_exit": None,
        },
    )
    result = _run(spec, bars)
    signals = result.signals.counts["long"] + result.signals.counts["short"]
    accounting = gate_accounting(
        result.blocked, signals=signals, executed=int(len(result.trades)),
        entry_attempts=result.entry_attempts,
    )
    assert accounting.executed + accounting.rejected == signals
    assert {row.code for row in accounting.rows} <= {
        "position_open", "cooldown", "max_trades_per_day", "max_spread_points",
        "insufficient_equity", "exit_indicator_warmup", "max_open_positions",
    }
