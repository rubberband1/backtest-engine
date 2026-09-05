"""The synthetic dataset: deterministic, self-consistent, and clearly labelled.

The fixture is what makes the repository runnable without a broker, so it is
load-bearing: if it drifts, every golden number measured on it moves, and the
demo a stranger runs stops matching the one described in the README. These
tests pin the properties the rest of the project is allowed to rely on.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from core.data.cache import ParquetCache
from core.data.fixture_provider import (
    EPOCH,
    FIXTURE_CACHE,
    HORIZON,
    INSTRUMENTS,
    SOURCE_NAME,
    FixtureProvider,
    resolve_cache_dir,
)
from core.data.provider import Timeframe, bar_columns_for

SYMBOLS = tuple(INSTRUMENTS)


@pytest.fixture(scope="module")
def provider() -> FixtureProvider:
    return FixtureProvider()


# -- the contract every provider owes ------------------------------------


@pytest.mark.parametrize("symbol", SYMBOLS)
@pytest.mark.parametrize("timeframe", ["M1", "H1", "H4", "D1"])
def test_bars_honour_the_provider_contract(provider, symbol, timeframe) -> None:
    tf = Timeframe.parse(timeframe)
    bars = provider.get_bars(
        symbol, tf, datetime(2023, 6, 1, tzinfo=timezone.utc),
        datetime(2023, 7, 1, tzinfo=timezone.utc),
    )
    assert len(bars) > 0
    assert list(bars.columns) == list(bar_columns_for(tf))
    assert bars.index.name == "time"
    assert str(bars.index.tz) == "UTC"
    assert bars.index.is_monotonic_increasing
    assert not bars.index.has_duplicates


@pytest.mark.parametrize("symbol", SYMBOLS)
def test_ohlc_is_internally_consistent(provider, symbol) -> None:
    bars = provider.get_bars(symbol, "H1", EPOCH, HORIZON)
    body_high = bars[["open", "close"]].max(axis=1)
    body_low = bars[["open", "close"]].min(axis=1)
    assert (bars["high"] >= body_high).all()
    assert (bars["low"] <= body_low).all()
    assert (bars["high"] >= bars["low"]).all()


def test_an_unknown_symbol_is_refused(provider) -> None:
    with pytest.raises(ValueError, match="fixture instruments"):
        provider.get_symbol_spec("XAUUSD.r")


# -- determinism ---------------------------------------------------------


def test_the_same_period_gives_the_same_bars_every_time(provider) -> None:
    window = (
        datetime(2023, 3, 1, tzinfo=timezone.utc),
        datetime(2023, 3, 15, tzinfo=timezone.utc),
    )
    first = provider.get_bars("SYNTHGOLD", "H1", *window)
    second = FixtureProvider().get_bars("SYNTHGOLD", "H1", *window)
    pd.testing.assert_frame_equal(first, second)


def test_a_narrow_window_matches_the_wide_one_that_contains_it(provider) -> None:
    """The bug this guards is subtle and was live once.

    The generator draws several arrays sized by the number of bars asked for.
    Sized by the request, each array after the first starts at a different
    place in the random stream, so the same timestamp comes back with a
    different price depending on how wide the window around it was. Nothing
    raises; the fixture is simply not a fixture any more.
    """
    wide = provider.get_bars(
        "SYNTHGOLD", "H1", datetime(2023, 1, 1, tzinfo=timezone.utc),
        datetime(2023, 3, 1, tzinfo=timezone.utc),
    )
    narrow = provider.get_bars(
        "SYNTHGOLD", "H1", datetime(2023, 1, 20, tzinfo=timezone.utc),
        datetime(2023, 2, 10, tzinfo=timezone.utc),
    )
    shared = wide.index.intersection(narrow.index)
    assert len(shared) > 200
    pd.testing.assert_frame_equal(wide.loc[shared], narrow.loc[shared])


def test_the_two_instruments_are_not_the_same_series(provider) -> None:
    window = (
        datetime(2023, 5, 1, tzinfo=timezone.utc),
        datetime(2023, 6, 1, tzinfo=timezone.utc),
    )
    gold = provider.get_bars("SYNTHGOLD", "H1", *window)["close"]
    fx = provider.get_bars("SYNTHFX", "H1", *window)["close"]
    assert gold.index.equals(fx.index)
    # correlated by construction they are not, and identical they must not be
    assert abs(gold.pct_change().corr(fx.pct_change())) < 0.2


# -- one path, every timeframe -------------------------------------------


@pytest.mark.parametrize("symbol", SYMBOLS)
def test_h4_is_the_aggregation_of_the_h1_inside_it(provider, symbol) -> None:
    window = (
        datetime(2023, 4, 1, tzinfo=timezone.utc),
        datetime(2023, 5, 1, tzinfo=timezone.utc),
    )
    h1 = provider.get_bars(symbol, "H1", *window)
    h4 = provider.get_bars(symbol, "H4", *window)
    folded = (
        h1.resample("240min", label="left", closed="left")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
        .dropna()
    )
    shared = folded.index.intersection(h4.index)
    assert len(shared) > 100
    for column in ("open", "high", "low", "close"):
        pd.testing.assert_series_equal(
            folded.loc[shared, column], h4.loc[shared, column], check_names=False
        )


def test_above_m1_the_spread_field_is_renamed_to_what_it_is(provider) -> None:
    window = (
        datetime(2023, 12, 4, tzinfo=timezone.utc),
        datetime(2023, 12, 8, tzinfo=timezone.utc),
    )
    minutes = provider.get_bars("SYNTHGOLD", "M1", *window)
    hours = provider.get_bars("SYNTHGOLD", "H1", *window)
    assert "spread" in minutes.columns
    assert "spread" not in hours.columns
    assert "min_spread_m1" in hours.columns
    # and it really is the minimum, which is why it must not be charged
    folded = minutes["spread"].resample("60min", label="left", closed="left").min()
    shared = folded.dropna().index.intersection(hours.index)
    assert len(shared) > 20
    assert (hours.loc[shared, "min_spread_m1"] == folded.loc[shared]).all()


# -- the shape of the week -----------------------------------------------


def test_the_weekend_is_closed(provider) -> None:
    bars = provider.get_bars("SYNTHGOLD", "H1", EPOCH, HORIZON)
    weekday, hour = bars.index.weekday, bars.index.hour
    assert not (weekday == 5).any(), "Saturday has bars"
    assert not ((weekday == 4) & (hour >= 21)).any(), "Friday evening has bars"
    assert not ((weekday == 6) & (hour < 21)).any(), "Sunday daytime has bars"


def test_the_spread_is_not_constant(provider) -> None:
    """A flat spread would make the per-bar and fixed policies agree, and the
    cost model's whole reason to exist would go untested on this data."""
    minutes = provider.get_bars(
        "SYNTHGOLD", "M1", datetime(2023, 12, 1, tzinfo=timezone.utc),
        datetime(2023, 12, 15, tzinfo=timezone.utc),
    )
    spread = minutes["spread"]
    assert spread.min() > 0
    assert spread.nunique() > 5
    assert spread.quantile(0.9) > spread.quantile(0.1)


def test_there_are_no_ticks_rather_than_invented_ones(provider) -> None:
    ticks = provider.get_ticks("SYNTHGOLD", EPOCH, HORIZON)
    assert ticks.empty
    assert list(ticks.columns) == ["bid", "ask", "last", "volume"]


# -- the committed dataset -----------------------------------------------


@pytest.mark.skipif(not FIXTURE_CACHE.exists(), reason="fixture not built")
def test_the_committed_fixture_is_readable_and_says_it_is_synthetic() -> None:
    cache = ParquetCache(FIXTURE_CACHE)
    for symbol in SYMBOLS:
        meta = cache.read_meta(symbol, Timeframe.H1, 2023)
        assert meta is not None, f"{symbol} H1 2023 missing from the fixture"
        assert meta.rows > 1000
        # the one thing that must survive into every downstream report
        assert meta.sources == [SOURCE_NAME]
        assert meta.server_timezone == "Europe/Athens"


@pytest.mark.skipif(not FIXTURE_CACHE.exists(), reason="fixture not built")
def test_regenerating_the_fixture_reproduces_the_committed_bars(tmp_path) -> None:
    """The committed dataset is exactly what the generator produces today.

    If this fails, the generator changed, and so did every number in the
    README that was measured on it.
    """
    from scripts.make_fixture import build

    rebuilt = build(tmp_path / "data_cache")
    committed = ParquetCache(FIXTURE_CACHE)
    for symbol in SYMBOLS:
        for timeframe in (Timeframe.H1, Timeframe.H4, Timeframe.D1):
            for year in (2022, 2023):
                pd.testing.assert_frame_equal(
                    committed.read_year(symbol, timeframe, year),
                    rebuilt.read_year(symbol, timeframe, year),
                )


@pytest.mark.skipif(not FIXTURE_CACHE.exists(), reason="fixture not built")
def test_a_backtest_runs_end_to_end_on_the_fixture() -> None:
    """The demo a stranger runs after cloning, as a test."""
    from core.data.spread import measure_from_cache
    from core.engine.backtester import BacktestConfig, run_backtest
    from core.engine.costs import CostModel, SpreadPolicy
    from core.runs.runner import SymbolResolver, load_bars
    from core.strategy.binding import bind_cell
    from core.strategy.spec import StrategySpec

    cache = ParquetCache(FIXTURE_CACHE)
    resolver = SymbolResolver(cache)
    reference = measure_from_cache(cache, "SYNTHGOLD")
    assert reference is not None, "the M1 sample must be there to measure on"

    bound = bind_cell(
        StrategySpec.from_json("strategies/rsi-mean-reversion.json"),
        "SYNTHGOLD",
        "H1",
    )
    result = run_backtest(
        bound.spec,
        load_bars(cache, "SYNTHGOLD", Timeframe.H1),
        resolver.symbol_spec("SYNTHGOLD"),
        resolver.server_timezone(),
        BacktestConfig(
            initial_equity=100.0,
            costs=CostModel(
                spread=SpreadPolicy(mode="fixed", value=reference.median_points)
            ),
        ),
    )
    assert len(result.trades) > 50
    assert result.equity.notna().all()


def test_the_fixture_is_chosen_only_when_there_is_nothing_else(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    target, is_fixture = resolve_cache_dir(None)
    assert is_fixture is FIXTURE_CACHE.exists()

    real = tmp_path / "data_cache" / "EURUSD" / "H1"
    real.mkdir(parents=True)
    (real / "2024.parquet").write_bytes(b"not really a parquet, but it is there")
    target, is_fixture = resolve_cache_dir(None)
    assert not is_fixture, "real data must win over the fixture"
    assert target == Path("data_cache")


def test_the_depth_probe_reports_the_whole_fixture(provider) -> None:
    """Regression: this method named a class that does not exist.

    Nothing called it, so nothing failed, and it would have raised the first
    time the Screen page asked an instrument how much history it had.
    """
    probe = provider.probe_depth("SYNTHGOLD", "H1")
    assert probe.source == SOURCE_NAME
    assert probe.bars > 10_000
    assert probe.truncated is False
    assert probe.first_bar is not None and probe.last_bar is not None
    assert probe.first_bar >= EPOCH
    assert probe.last_bar < HORIZON
    assert probe.as_dict()["source"] == SOURCE_NAME
