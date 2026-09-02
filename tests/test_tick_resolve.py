"""Tick resolution of ambiguous trades, and its refusal to guess."""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from core.data.provider import Timeframe, empty_ticks
from core.validation.tick_resolve import (
    first_touch,
    levels_for,
    resolve_ambiguous,
)
from tests.conftest_engine import spec_from, symbol_spec

UTC = timezone.utc


def _spec():
    return spec_from(
        exit_block={
            "stop_loss": {"type": "points", "value": 100},
            "take_profit": {"type": "points", "value": 50},
            "time_stop": None,
            "signal_exit": None,
        }
    )


def _ticks(rows: list[tuple[float, float]]) -> pd.DataFrame:
    index = pd.date_range("2024-01-01 00:00", periods=len(rows), freq="1s", tz="UTC")
    return pd.DataFrame(
        {
            "bid": [bid for bid, _ in rows],
            "ask": [ask for _, ask in rows],
            "last": 0.0,
            "volume": 0.0,
        },
        index=index,
    )


def _trades(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _trade(
    direction: int = 1,
    entry_price: float = 2000.0,
    net_pnl: float = -1.0,
    ambiguous: bool = True,
    spread_points: float = 10.0,
) -> dict:
    return {
        "direction": direction,
        "entry_time": datetime(2024, 1, 1, tzinfo=UTC),
        "entry_price": entry_price,
        "exit_time": datetime(2024, 1, 1, 0, 5, tzinfo=UTC),
        "exit_price": entry_price,
        "exit_reason": "stop_loss",
        "lots": 0.1,
        "bars_held": 5,
        "session_bars_held": 5,
        "gross_pnl": net_pnl,
        "spread_points": spread_points,
        "spread_cost": 0.0,
        "commission": 0.0,
        "swap": 0.0,
        "net_pnl": net_pnl,
        "risk_money": 1.0,
        "r_multiple": -1.0,
        "ambiguous": ambiguous,
        "crossed_gap": False,
    }


class _Provider:
    """A provider that returns whatever the test wants, including nothing."""

    def __init__(self, ticks: pd.DataFrame | None = None, fail: bool = False) -> None:
        self._ticks = ticks if ticks is not None else empty_ticks()
        self._fail = fail
        self.calls = 0

    def get_ticks(self, symbol: str, start, end) -> pd.DataFrame:
        self.calls += 1
        if self._fail:
            raise RuntimeError("no tick history for this period")
        return self._ticks


# -- levels --------------------------------------------------------------


def test_levels_are_rebuilt_from_the_entry_price_as_the_engine_placed_them() -> None:
    spec = _spec()
    stop, target = levels_for(pd.Series(_trade(direction=1, entry_price=2000.0)), spec, symbol_spec())
    # point is 0.01: 100 points below, 50 above
    assert stop == pytest.approx(1999.0)
    assert target == pytest.approx(2000.5)


def test_a_short_has_its_levels_mirrored() -> None:
    stop, target = levels_for(
        pd.Series(_trade(direction=-1, entry_price=2000.0)), _spec(), symbol_spec()
    )
    assert stop == pytest.approx(2001.0)
    assert target == pytest.approx(1999.5)


def test_a_spec_without_fixed_levels_has_nothing_to_resolve() -> None:
    spec = spec_from(
        exit_block={
            "stop_loss": {"type": "points", "value": 100},
            "take_profit": None,
            "time_stop": {"bars": 10},
            "signal_exit": None,
        }
    )
    assert levels_for(pd.Series(_trade()), spec, symbol_spec()) == (None, None)


# -- first touch ---------------------------------------------------------


def test_a_long_is_read_on_the_bid() -> None:
    # the ask dives through the stop but the bid never does: no stop for a long
    ticks = _ticks([(2000.0, 1998.0), (2000.6, 2001.0)])
    assert first_touch(ticks, 1, stop_level=1999.0, target_level=2000.5) == "take_profit"


def test_a_short_is_read_on_the_ask() -> None:
    # the bid stays low, but the ask reaches the short's stop first
    ticks = _ticks([(1999.0, 2001.5), (1999.0, 1999.4)])
    assert first_touch(ticks, -1, stop_level=2001.0, target_level=1999.5) == "stop_loss"


def test_the_earlier_tick_wins() -> None:
    ticks = _ticks([(2000.5, 2000.6), (1999.0, 1999.1)])
    assert first_touch(ticks, 1, stop_level=1999.0, target_level=2000.5) == "take_profit"
    reversed_ticks = _ticks([(1999.0, 1999.1), (2000.5, 2000.6)])
    assert first_touch(reversed_ticks, 1, stop_level=1999.0, target_level=2000.5) == "stop_loss"


def test_one_tick_touching_both_keeps_the_conservative_stop() -> None:
    """At tick resolution there is no ordering to read, so nothing is invented."""
    ticks = _ticks([(1999.0, 2000.5)])
    assert first_touch(ticks, 1, stop_level=1999.0, target_level=2000.5) == "stop_loss"


def test_ticks_that_reach_neither_level_stay_unresolved() -> None:
    ticks = _ticks([(2000.0, 2000.1), (2000.1, 2000.2)])
    assert first_touch(ticks, 1, stop_level=1999.0, target_level=2000.5) == "unresolved"


def test_no_ticks_at_all_is_unresolved_not_a_stop() -> None:
    assert first_touch(empty_ticks(), 1, 1999.0, 2000.5) == "unresolved"


# -- the report ----------------------------------------------------------


def test_a_run_without_ambiguous_trades_says_so_plainly() -> None:
    trades = _trades([_trade(ambiguous=False, net_pnl=-1.0)])
    report = resolve_ambiguous(
        "r1", _spec(), trades, symbol_spec(), _Provider(), Timeframe.M1, 100.0
    )
    assert report.available is True
    assert report.ambiguous_trades == 0
    assert "No ambiguous trade" in report.verdict


def test_a_trade_resolved_to_the_target_moves_the_net_pnl_up() -> None:
    trades = _trades([_trade(direction=1, net_pnl=-1.0, ambiguous=True)])
    ticks = _ticks([(2000.5, 2000.6)])  # bid reaches the target, never the stop
    report = resolve_ambiguous(
        "r1", _spec(), trades, symbol_spec(), _Provider(ticks), Timeframe.M1, 100.0
    )
    assert report.resolved == 1
    assert report.resolved_to_take_profit == 1
    assert report.delta_net_pnl > 0
    assert report.resolved_net_pnl > report.original_net_pnl


def test_unresolvable_trades_are_counted_and_never_assumed() -> None:
    trades = _trades([_trade(ambiguous=True), _trade(ambiguous=True)])
    report = resolve_ambiguous(
        "r1", _spec(), trades, symbol_spec(), _Provider(), Timeframe.M1, 100.0
    )
    assert report.ambiguous_trades == 2
    assert report.unresolved == 2
    assert report.resolved == 0
    assert report.delta_net_pnl == pytest.approx(0.0)
    assert "None of the 2 ambiguous trades could be resolved" in report.verdict
    assert "not confirmed by this" in report.verdict


def test_a_provider_that_throws_does_not_lose_the_other_trades() -> None:
    trades = _trades([_trade(ambiguous=True)])
    report = resolve_ambiguous(
        "r1", _spec(), trades, symbol_spec(), _Provider(fail=True), Timeframe.M1, 100.0
    )
    assert report.unresolved == 1
    assert report.trades[0].resolution == "unresolved"


def test_a_partial_resolution_says_how_much_of_it_is_covered() -> None:
    ticks = _ticks([(2000.5, 2000.6)])
    trades = _trades([_trade(ambiguous=True), _trade(ambiguous=False, net_pnl=2.0)])
    report = resolve_ambiguous(
        "r1", _spec(), trades, symbol_spec(), _Provider(ticks), Timeframe.M1, 100.0
    )
    assert report.ambiguous_trades == 1
    # the non-ambiguous trade keeps its pnl untouched in the totals
    assert report.original_net_pnl == pytest.approx(1.0)


def test_the_win_rate_is_recomputed_over_every_trade() -> None:
    ticks = _ticks([(2000.5, 2000.6)])
    trades = _trades([_trade(ambiguous=True, net_pnl=-1.0)])
    report = resolve_ambiguous(
        "r1", _spec(), trades, symbol_spec(), _Provider(ticks), Timeframe.M1, 100.0
    )
    assert report.original_win_rate == pytest.approx(0.0)
    assert report.resolved_win_rate == pytest.approx(1.0)


def test_the_report_serializes_to_json_safe_primitives() -> None:
    import json

    ticks = _ticks([(2000.5, 2000.6)])
    trades = _trades([_trade(ambiguous=True)])
    report = resolve_ambiguous(
        "r1", _spec(), trades, symbol_spec(), _Provider(ticks), Timeframe.M1, 100.0
    )
    json.dumps(report.as_dict())
