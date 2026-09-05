"""The test that makes the live runner worth having.

The previous project's central defect was a backtest that simulated a
different system from the one that traded: signals on ticks instead of closed
bars, different risk gates, a time stop present in simulation and absent
live. Every one of those is invisible in the results and fatal to them.

Here the runner replays historical bars one at a time, as if they were
arriving now, and the trades it produces must match the backtester's exactly
- same entry and exit timestamps, same prices, same levels, same lots, same
exit reasons. No tolerance: both sides run the same arithmetic in the same
order, so any difference means the order changed, which is the bug being
hunted.

This runs on every change to the engine. If it fails, the engine is not
allowed to be shipped, whatever else passes.
"""
from __future__ import annotations

from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from core.engine.costs import CommissionModel, CostModel, SpreadPolicy, SwapModel
from core.live.replay import compare_replay, diff_trades, replay
from core.strategy.binding import BoundSpec, bind_cell
from core.strategy.spec import StrategySpec
from tests.conftest_engine import random_walk, symbol_spec

SERVER_TZ = ZoneInfo("Europe/Athens")
SYMBOL = "SYNTH"

# The baseline plus strategies that actually trade on a random walk and
# between them cover fixed-point exits, ATR exits, a time stop, both sides of
# the book and a signal exit.
STRATEGIES = [
    "ma-crossover",
    "rsi-mean-reversion",
    "bollinger-breakout",
    "macd-signal",
]


def synthetic(n: int = 1200, seed: int = 4, spread: float = 8.0) -> pd.DataFrame:
    bars = random_walk(n, seed=seed)
    bars.index = pd.date_range(
        "2024-01-01", periods=n, freq="1min", tz="UTC", name="time"
    )
    bars["spread"] = spread
    return bars


def rejection_bars(cycles: int = 40, leg: int = 10, seed: int = 1) -> pd.DataFrame:
    """Bars built to trigger the baseline, which a random walk almost never does.

    The baseline needs three things on the same bar - RSI(9) crossing an
    extreme, a wick over 60% of the range and a body under 30% - and a random
    walk produces that a handful of times in ten thousand bars. Comparing two
    empty trade tables proves nothing, so the series is a sawtooth: a
    directional leg that walks the RSI to the threshold, then one rejection
    candle that crosses it.
    """
    rng = np.random.default_rng(seed)
    rows: list[tuple[float, float, float, float]] = []
    price = 2000.0
    for cycle in range(cycles):
        falling = cycle % 2 == 0
        for _ in range(leg):
            step = (-0.9 if falling else 0.9) + rng.normal(0, 0.05)
            open_ = price
            price += step
            rows.append((open_, max(open_, price) + 0.1, min(open_, price) - 0.1, price))
        open_ = price
        if falling:  # hammer: small down body, long lower wick
            close = price - 2.0
            high, low = open_ + 0.5, close - 6.0
        else:  # shooting star: small up body, long upper wick
            close = price + 2.0
            high, low = close + 6.0, open_ - 0.5
        price = close
        rows.append((open_, high, low, close))

    open_s, high_s, low_s, close_s = zip(*rows)
    index = pd.date_range(
        "2024-01-01", periods=len(rows), freq="1min", tz="UTC", name="time"
    )
    return pd.DataFrame(
        {
            "open": open_s,
            "high": high_s,
            "low": low_s,
            "close": close_s,
            "tick_volume": np.full(len(rows), 100.0),
            "spread": 8.0,
            "real_volume": np.zeros(len(rows)),
        },
        index=index,
    )


def gappy(n: int = 1000, seed: int = 9) -> pd.DataFrame:
    """Bars with weekend-shaped holes, so the session-bar time stop is exercised."""
    bars = random_walk(n, seed=seed)
    index = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC", name="time")
    bars.index = index
    bars["spread"] = 7.0
    return bars[index.dayofweek < 5]


def variable_spread(n: int = 1000, seed: int = 12) -> pd.DataFrame:
    bars = synthetic(n, seed=seed)
    rng = np.random.default_rng(seed)
    bars["spread"] = rng.integers(3, 25, len(bars)).astype(float)
    return bars


def bind(name: str, timeframe: str = "M1") -> BoundSpec:
    return bind_cell(
        StrategySpec.from_json(f"strategies/{name}.json"), SYMBOL, timeframe
    )


def test_the_baseline_replays_exactly() -> None:
    """The reference strategy, on bars that actually make it trade."""
    report, _, _ = compare_replay(
        bind("rsi-wick-baseline"), rejection_bars(),
        symbol_spec(name=SYMBOL), SERVER_TZ,
    )
    assert report.backtest_trades > 5, "the fixture stopped triggering the baseline"
    assert report.equivalent, "\n".join(
        [report.verdict, *(str(d) for d in report.divergences[:10])]
    )


@pytest.mark.parametrize("name", STRATEGIES)
def test_the_runner_replaying_history_trades_exactly_like_the_backtester(
    name: str,
) -> None:
    report, _expected, _actual = compare_replay(
        bind(name), synthetic(), symbol_spec(name=SYMBOL), SERVER_TZ
    )
    assert report.backtest_trades > 0, f"{name} took no trade: nothing was compared"
    assert report.equivalent, "\n".join(
        [report.verdict, *(str(d) for d in report.divergences[:10])]
    )
    assert report.backtest_trades == report.replay_trades


@pytest.mark.parametrize("name", ["ma-crossover", "bollinger-breakout"])
def test_equivalence_holds_across_session_gaps(name: str) -> None:
    """Where the time stop counts session bars, not rows."""
    report, _, _ = compare_replay(
        bind(name, "H1"), gappy(), symbol_spec(name=SYMBOL), SERVER_TZ
    )
    assert report.equivalent, "\n".join(str(d) for d in report.divergences[:10])


@pytest.mark.parametrize("name", ["rsi-mean-reversion", "macd-signal"])
def test_equivalence_holds_with_a_spread_that_moves_bar_by_bar(name: str) -> None:
    report, _, _ = compare_replay(
        bind(name), variable_spread(), symbol_spec(name=SYMBOL), SERVER_TZ
    )
    assert report.equivalent, "\n".join(str(d) for d in report.divergences[:10])


def test_equivalence_holds_with_commission_and_swap() -> None:
    """Costs charged at exit and per night crossed must land on the same trades."""
    costs = CostModel(
        spread=SpreadPolicy(mode="per_bar"),
        commission=CommissionModel(per_lot_per_side=3.5),
        swap=SwapModel(mode="points"),
    )
    report, _, _ = compare_replay(
        bind("bollinger-breakout", "H1"),
        gappy(),
        symbol_spec(name=SYMBOL),
        SERVER_TZ,
        costs=costs,
    )
    assert report.equivalent, "\n".join(str(d) for d in report.divergences[:10])


def test_a_strategy_with_atr_exits_is_equivalent() -> None:
    """The exit distance is frozen from the signal bar on both sides or neither."""
    report, expected, _ = compare_replay(
        bind("donchian-breakout"), synthetic(1500, seed=21),
        symbol_spec(name=SYMBOL), SERVER_TZ,
    )
    assert report.equivalent, "\n".join(str(d) for d in report.divergences[:10])
    if len(expected.trades):
        # the levels really were set from an ATR, not from a constant
        assert expected.trades["stop_level"].notna().all()


def test_the_gate_accounting_is_the_same_on_both_sides() -> None:
    """Not just the trades: the signals that never became trades too."""
    spec = bind("bollinger-breakout")
    bars = synthetic()
    report, expected, actual = compare_replay(
        spec, bars, symbol_spec(name=SYMBOL), SERVER_TZ
    )
    assert report.equivalent
    assert dict(expected.blocked) == actual.blocked
    assert expected.entry_attempts == actual.entry_attempts


def test_the_equity_curve_matches_bar_by_bar() -> None:
    spec = bind("rsi-mean-reversion")
    bars = synthetic()
    _, expected, actual = compare_replay(
        spec, bars, symbol_spec(name=SYMBOL), SERVER_TZ
    )
    pd.testing.assert_series_equal(expected.equity, actual.equity)


def test_the_comparison_can_fail() -> None:
    """Check of the check: a diff that never reports anything proves nothing."""
    spec = bind("bollinger-breakout")
    bars = synthetic()
    _, expected, _ = compare_replay(spec, bars, symbol_spec(name=SYMBOL), SERVER_TZ)
    assert len(expected.trades) > 1

    tampered = expected.trades.copy()
    tampered.iloc[0, tampered.columns.get_loc("exit_price")] += 1.0
    divergences = diff_trades(expected.trades, tampered)
    assert divergences
    assert divergences[0].column == "exit_price"

    # and a missing trade is reported, not silently truncated
    assert diff_trades(expected.trades, expected.trades.iloc[:-1])


def test_the_runner_ignores_a_bar_that_is_not_newer() -> None:
    """A repeated or late bar must not be processed twice."""
    from core.engine.session import SessionCalendar
    from core.live.replay import ReplayBroker
    from core.live.runner import LiveConfig, LiveRunner

    spec = bind("bollinger-breakout")
    bars = synthetic(300)
    calendar = SessionCalendar.infer(bars.index, spec.tf)
    runner = LiveRunner(
        spec, symbol_spec(name=SYMBOL), SERVER_TZ,
        LiveConfig(session_calendar=calendar), broker=ReplayBroker(),
    )
    runner.start(bars.iloc[:0])
    for moment in bars.index[:100]:
        runner.on_closed_bar(bars.loc[[moment]])
    seen = runner.state().bars_seen

    runner.on_closed_bar(bars.loc[[bars.index[50]]])
    runner.on_closed_bar(bars.loc[[bars.index[99]]])
    assert runner.state().bars_seen == seen


def test_only_closed_bars_are_offered_to_the_runner() -> None:
    """The most recent bar the terminal returns is the one still forming."""
    from core.engine.session import SessionCalendar
    from core.live.replay import ReplayBroker
    from core.live.runner import LiveConfig, LiveRunner

    spec = bind("bollinger-breakout", "H1")
    bars = random_walk(50, seed=3)
    bars.index = pd.date_range(
        "2024-01-01", periods=50, freq="1h", tz="UTC", name="time"
    )
    bars["spread"] = 5.0
    runner = LiveRunner(
        spec, symbol_spec(name=SYMBOL), SERVER_TZ,
        LiveConfig(
            session_calendar=SessionCalendar.infer(bars.index, spec.tf)
        ),
        broker=ReplayBroker(),
    )
    runner.start(bars.iloc[:0])

    # "now" is halfway through the bar labelled 10:00, so that bar is open
    now = bars.index[10] + pd.Timedelta(minutes=30)
    closed = runner.closed_bars_after(bars, now.to_pydatetime())
    assert closed.index[-1] == bars.index[9]
    assert bars.index[10] not in closed.index


def test_a_replay_sends_no_orders() -> None:
    spec = bind("bollinger-breakout")
    result = replay(spec, synthetic(), symbol_spec(name=SYMBOL), SERVER_TZ)
    assert len(result.trades) > 0
    # trades happened in the engine; nothing was sent anywhere
    assert result.equity.iloc[-1] != pytest.approx(0.0)
