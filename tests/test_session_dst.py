"""Session bars are counted on the server clock, not on UTC. Blocking.

A broker session is fixed in the server's local time: it opens at the same
wall-clock hour every day of the year. In UTC that hour moves twice a year,
so a weekly grid laid out in UTC has slots that exist for part of the year
and not the rest. The threshold then picks one of the two regimes and treats
the other as a hole, and "session bars since entry" - which is what a time
stop counts - drifts by up to a full session around every DST change.

The property these tests pin is simple enough to be worth stating: over a
sample with no missing bars, the number of session bars between two bars must
equal the number of bars between them. On the server clock that holds through
a DST change. On UTC it does not, and the second test says so explicitly
rather than leaving the old behaviour undocumented.
"""
from __future__ import annotations

from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from core.data.provider import Timeframe
from core.engine.session import SessionCalendar, session_ordinals

ATHENS = ZoneInfo("Europe/Athens")
# EEST -> EET, the Sunday the server clock goes back an hour
AUTUMN_CHANGE = pd.Timestamp("2025-10-26", tz=ATHENS)
# EET -> EEST, the Sunday it goes forward
SPRING_CHANGE = pd.Timestamp("2025-03-30", tz=ATHENS)


def session_index(
    start: str, end: str, open_hour: int = 8, close_hour: int = 16
) -> pd.DatetimeIndex:
    """Hourly bars of a fixed server-clock session, Monday to Friday.

    Built in Athens local time and then expressed in UTC, which is how the
    terminal's bars actually arrive: the session is the same wall-clock hours
    all year, and their UTC offset is not.
    """
    days = pd.date_range(start, end, freq="D", tz=ATHENS)
    stamps: list[pd.Timestamp] = []
    for day in days:
        if day.dayofweek >= 5:
            continue
        for hour in range(open_hour, close_hour):
            stamps.append(
                pd.Timestamp(
                    year=day.year, month=day.month, day=day.day, hour=hour, tz=ATHENS
                )
            )
    return pd.DatetimeIndex(stamps).tz_convert("UTC").sort_values()


def counts_every_bar_exactly_once(
    index: pd.DatetimeIndex, ordinals: np.ndarray
) -> bool:
    """With no holes in the sample, ordinal distance must be bar distance."""
    steps = np.diff(ordinals)
    return bool(len(index) == len(ordinals) and np.all(steps == 1))


# -- the property ---------------------------------------------------------


def test_the_server_clock_counts_every_session_bar_once_through_autumn() -> None:
    index = session_index("2025-09-15", "2025-11-30")
    assert (index < AUTUMN_CHANGE).any() and (index > AUTUMN_CHANGE).any()

    calendar = SessionCalendar.infer(index, Timeframe.H1, 0.5, ATHENS)
    assert counts_every_bar_exactly_once(index, calendar.ordinals(index))


def test_the_server_clock_counts_every_session_bar_once_through_spring() -> None:
    index = session_index("2025-03-01", "2025-05-15")
    assert (index < SPRING_CHANGE).any() and (index > SPRING_CHANGE).any()

    calendar = SessionCalendar.infer(index, Timeframe.H1, 0.5, ATHENS)
    assert counts_every_bar_exactly_once(index, calendar.ordinals(index))


def test_utc_counting_is_what_used_to_break_and_still_would() -> None:
    """The bug, kept in a test so nobody re-introduces it by 'simplifying'."""
    index = session_index("2025-09-15", "2025-11-30")
    utc_ordinals = SessionCalendar.infer(index, Timeframe.H1, 0.5, None).ordinals(index)
    assert not counts_every_bar_exactly_once(index, utc_ordinals)

    steps = np.diff(utc_ordinals)
    # the two shapes of the defect, on a sample with no missing bars at all:
    # a pair of consecutive bars the count treats as the same bar, and a pair
    # it treats as two apart. The first is what makes a time stop fire late.
    assert (steps == 0).any()
    assert (steps > 1).any()

    server = SessionCalendar.infer(index, Timeframe.H1, 0.5, ATHENS).ordinals(index)
    assert (np.diff(server) == 1).all()


def test_a_sample_inside_one_regime_agrees_on_both_clocks() -> None:
    """No DST change in the window means no correction to make."""
    index = session_index("2025-06-02", "2025-07-31")
    utc = SessionCalendar.infer(index, Timeframe.H1, 0.5, None).ordinals(index)
    server = SessionCalendar.infer(index, Timeframe.H1, 0.5, ATHENS).ordinals(index)
    assert np.array_equal(utc - utc[0], server - server[0])


# -- the calendar as an object -------------------------------------------


def test_the_clock_travels_with_the_pinned_calendar() -> None:
    index = session_index("2025-09-15", "2025-11-30")
    calendar = SessionCalendar.infer(index, Timeframe.H1, 0.5, ATHENS)
    assert calendar.server_timezone == "Europe/Athens"

    restored = SessionCalendar.from_dict(calendar.to_dict())
    assert restored == calendar
    assert np.array_equal(restored.ordinals(index), calendar.ordinals(index))


def test_a_calendar_pinned_before_the_fix_still_loads_and_still_counts() -> None:
    """An old journal must not crash, and must not silently change meaning."""
    index = session_index("2025-09-15", "2025-11-30")
    legacy = SessionCalendar.infer(index, Timeframe.H1, 0.5, None)
    payload = legacy.to_dict()
    payload.pop("server_timezone")

    restored = SessionCalendar.from_dict(payload)
    assert restored.server_timezone is None
    assert np.array_equal(restored.ordinals(index), legacy.ordinals(index))


def test_session_ordinals_passes_the_clock_through() -> None:
    index = session_index("2025-09-15", "2025-11-30")
    direct = SessionCalendar.infer(index, Timeframe.H1, 0.5, ATHENS).ordinals(index)
    through = session_ordinals(index, Timeframe.H1, 0.5, None, ATHENS)
    assert np.array_equal(direct, through)


def test_a_pinned_calendar_wins_over_the_clock_argument() -> None:
    """A pinned calendar is an input: nothing may quietly re-infer it."""
    index = session_index("2025-09-15", "2025-11-30")
    pinned = SessionCalendar.infer(index, Timeframe.H1, 0.5, None)
    out = session_ordinals(index, Timeframe.H1, 0.5, pinned, ATHENS)
    assert np.array_equal(out, pinned.ordinals(index))


# -- what it does to a trade ---------------------------------------------


def test_the_time_stop_fires_on_the_right_bar_across_a_dst_change() -> None:
    """End to end: the count is only interesting because it closes trades."""
    from core.engine.backtester import BacktestConfig, run_backtest
    from core.engine.costs import CommissionModel, CostModel, SpreadPolicy, SwapModel
    from core.strategy.spec import StrategySpec
    from tests.conftest_engine import symbol_spec

    index = session_index("2025-10-06", "2025-11-14")
    price = pd.Series(2000.0, index=index)
    bars = pd.DataFrame(
        {
            "open": price,
            "high": price + 0.02,
            "low": price - 0.02,
            "close": price,
            "tick_volume": 100.0,
            "spread": 1.0,
            "real_volume": 0.0,
        },
        index=index,
    )
    # one long signal on the first bar and never again; a stop and target far
    # enough away that only the time stop can close the trade
    bars.iloc[0, bars.columns.get_loc("close")] = 2000.5

    spec = StrategySpec.from_dict(
        {
            "schema_version": 1,
            "id": "timestop",
            "name": "timestop",
            "instrument": {"symbol": "SYNTH", "timeframe": "H1"},
            "indicators": [],
            "entry": {
                "long": {
                    "op": "gt",
                    "left": {"bar": "close"},
                    "right": {"const": 2000.2},
                }
            },
            "exit": {
                "stop_loss": {"type": "points", "value": 100000},
                "take_profit": {"type": "points", "value": 100000},
                "time_stop": {"bars": 40},
            },
            "sizing": {
                "type": "equity_per_step",
                "equity_per_001_lot": 100,
                "min_lot": 0.01,
                "max_lot": 0.05,
            },
            "risk": {"max_open_positions": 1},
        }
    )
    costs = CostModel(
        spread=SpreadPolicy(mode="fixed", value=1.0),
        commission=CommissionModel(0.0),
        swap=SwapModel(mode="none"),
    )
    result = run_backtest(
        spec, bars, symbol_spec(name="SYNTH"), ATHENS,
        BacktestConfig(costs=costs),
    )
    assert len(result.trades) == 1
    trade = result.trades.iloc[0]
    assert trade["exit_reason"] == "time_stop"

    # The signal is on bar 0 and the fill on bar 1. The time stop is decided
    # on the close of the bar where 40 session bars have elapsed and executed
    # at the next open, so the exit lands 41 bars after the entry. With no
    # holes in this sample that is a fact about the data, and no DST change
    # may move it.
    entry_position = index.get_loc(trade["entry_time"])
    exit_position = index.get_loc(trade["exit_time"])
    assert exit_position - entry_position == 41
    assert int(trade["session_bars_held"]) == 41
    assert int(trade["bars_held"]) == 41
    assert not bool(trade["crossed_gap"])


def test_the_same_trade_times_out_late_on_the_utc_clock() -> None:
    """The size of the defect, stated as the thing it actually breaks."""
    from core.engine.backtester import BacktestConfig, run_backtest
    from core.engine.costs import CommissionModel, CostModel, SpreadPolicy, SwapModel
    from core.strategy.spec import StrategySpec
    from tests.conftest_engine import symbol_spec

    index = session_index("2025-10-06", "2025-11-14")
    price = pd.Series(2000.0, index=index)
    bars = pd.DataFrame(
        {
            "open": price, "high": price + 0.02, "low": price - 0.02,
            "close": price, "tick_volume": 100.0, "spread": 1.0,
            "real_volume": 0.0,
        },
        index=index,
    )
    bars.iloc[0, bars.columns.get_loc("close")] = 2000.5
    spec = StrategySpec.from_dict(
        {
            "schema_version": 1, "id": "timestop", "name": "timestop",
            "instrument": {"symbol": "SYNTH", "timeframe": "H1"},
            "indicators": [],
            "entry": {
                "long": {"op": "gt", "left": {"bar": "close"},
                         "right": {"const": 2000.2}}
            },
            "exit": {
                "stop_loss": {"type": "points", "value": 100000},
                "take_profit": {"type": "points", "value": 100000},
                "time_stop": {"bars": 40},
            },
            "sizing": {"type": "equity_per_step", "equity_per_001_lot": 100,
                       "min_lot": 0.01, "max_lot": 0.05},
            "risk": {"max_open_positions": 1},
        }
    )
    costs = CostModel(
        spread=SpreadPolicy(mode="fixed", value=1.0),
        commission=CommissionModel(0.0), swap=SwapModel(mode="none"),
    )
    result = run_backtest(
        spec, bars, symbol_spec(name="SYNTH"), ATHENS,
        BacktestConfig(
            costs=costs,
            session_calendar=SessionCalendar.infer(index, Timeframe.H1, 0.5, None),
        ),
    )
    trade = result.trades.iloc[0]
    held = index.get_loc(trade["exit_time"]) - index.get_loc(trade["entry_time"])
    assert held != 41, (
        "counting on UTC used to move this exit; if it no longer does, the "
        "sample stopped spanning a DST change and the test above proves less "
        "than it claims"
    )
    # Early here, and it could as easily be late: the UTC grid has slots with
    # no bar behind them (which make the count run ahead) and pairs of bars
    # sharing a slot (which make it lag). This trade sits after a run of the
    # first kind, so its 40-bar stop fires on the 37th bar of open market -
    # a 10% haircut on the holding time the spec asked for.
    assert held == 37
