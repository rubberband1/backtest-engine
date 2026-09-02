"""Integration tests: they require a running, logged-in MT5 terminal."""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from core.data.provider import BAR_COLUMNS, TICK_COLUMNS, SymbolSpec, Timeframe

pytestmark = pytest.mark.mt5

# No hardcoded symbol: taken from the Market Watch, or from MT5_TEST_SYMBOL.
SYMBOL_ENV = "MT5_TEST_SYMBOL"


@pytest.fixture(scope="module")
def provider():
    from core.data.mt5_provider import MT5Provider

    with MT5Provider() as p:
        yield p


@pytest.fixture(scope="module")
def symbol(provider) -> str:
    from_env = os.environ.get(SYMBOL_ENV)
    if from_env:
        return from_env
    specs = provider.list_symbols()
    if not specs:
        pytest.skip("no symbol listed by the terminal")
    return specs[0].name


def test_list_symbols(provider) -> None:
    specs = provider.list_symbols()
    assert specs and all(isinstance(s, SymbolSpec) for s in specs)


def test_symbol_spec_is_complete(provider, symbol: str) -> None:
    spec = provider.get_symbol_spec(symbol)
    assert spec.name == symbol
    assert spec.point > 0
    assert spec.digits >= 0
    assert spec.volume_min > 0
    assert spec.currency_profit


def test_server_timezone_detected(provider) -> None:
    tz = provider.server_timezone
    offset = datetime.now(timezone.utc).astimezone(tz).utcoffset()
    assert offset is not None
    assert timedelta(hours=-12) <= offset <= timedelta(hours=14)


def test_bars_in_utc_with_spread(provider, symbol: str) -> None:
    end = datetime.now(timezone.utc)
    bars = provider.get_bars(symbol, Timeframe.M1, end - timedelta(days=5), end)
    if bars.empty:
        pytest.skip("no bars returned in the requested window")

    assert list(bars.columns) == list(BAR_COLUMNS)
    assert str(bars.index.tz) == "UTC"
    assert bars.index.name == "time"
    assert bars.index.is_monotonic_increasing
    assert not bars.index.duplicated().any()
    assert (bars["spread"] >= 0).all()
    assert bars["spread"].max() > 0


def test_timestamp_consistent_with_live_price(provider, symbol: str) -> None:
    """The last M1 bar can be neither in the future nor hours away.

    This is the check that fails if the timezone conversion is wrong: a 2-3
    hour offset here means the whole history is being mislabeled.
    """
    now = datetime.now(timezone.utc)
    bars = provider.get_bars(symbol, Timeframe.M1, now - timedelta(days=5), now)
    if bars.empty:
        pytest.skip("no bars available")
    age = now - bars.index.max().to_pydatetime()
    assert age > -timedelta(minutes=1), "bars in the future: timezone conversion inverted"
    if age > timedelta(days=3):
        pytest.skip("market closed for days, the check is uninformative")
    assert age < timedelta(hours=1), f"last bar is {age} old: timezone probably wrong"


def test_requested_range_respected(provider, symbol: str) -> None:
    end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(days=2)
    bars = provider.get_bars(symbol, Timeframe.H1, start, end)
    if bars.empty:
        pytest.skip("no bars available")
    assert bars.index.min() >= pd.Timestamp(start)
    assert bars.index.max() < pd.Timestamp(end)


def test_ticks_in_utc(provider, symbol: str) -> None:
    end = datetime.now(timezone.utc)
    ticks = provider.get_ticks(symbol, end - timedelta(hours=2), end)
    if ticks.empty:
        pytest.skip("no ticks in the requested window (market closed?)")
    assert list(ticks.columns) == list(TICK_COLUMNS)
    assert str(ticks.index.tz) == "UTC"
    assert ticks.index.is_monotonic_increasing


def test_provider_does_not_expose_order_sending() -> None:
    import core.data.mt5_provider as module

    source = open(module.__file__, encoding="utf-8").read()
    for forbidden in ("order_send", "order_check", "positions_get", "TRADE_ACTION"):
        assert forbidden not in source, f"the data layer must not touch {forbidden}"
