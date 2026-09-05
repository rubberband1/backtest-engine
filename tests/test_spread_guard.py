"""The aggregated spread column can no longer become a cost. Blocking.

Phase 6 measured it: the `spread` field of a bar above M1 is the MINIMUM of
the spreads of the M1 bars inside it, in 100% of more than four thousand
periods on each of three instruments. Charging it as a fill cost charges the
best price of the period, and on this broker's D1 bars that means charging
nothing at all on 77-100% of the bars of four FX instruments.

Phase 6 fixed the one place that read it. These tests exist because a fix in
one place is not a guarantee: they check the mechanism that makes the mistake
unrepresentable rather than merely absent.

1. The raw column does not carry a name anyone can charge: above M1 it is
   `min_spread_m1` wherever bars come from.
2. Asking for a per-bar spread over such a frame raises, with a message that
   says what to do instead.
3. The honest per-bar spread is rebuilt from the M1 bars of the same period,
   at the median or above, never the minimum.
4. Where the M1 sample does not cover the period, the run is refused instead
   of served the column.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from core.data.cache import ParquetCache
from core.data.provider import (
    MIN_SPREAD_M1_COLUMN,
    Timeframe,
    bar_columns_for,
    rename_aggregated_spread,
    spread_column_for,
)
from core.data.spread import (
    SpreadUnavailable,
    attach,
    reconstruct_from_m1,
    reconstructed_series,
)
from core.engine.costs import AggregatedSpreadRefused, CostModel, SpreadPolicy

START = datetime(2024, 3, 4, tzinfo=timezone.utc)


def m1_frame(minutes: int = 600, spreads: np.ndarray | None = None) -> pd.DataFrame:
    index = pd.date_range(START, periods=minutes, freq="1min", tz="UTC", name="time")
    if spreads is None:
        # a spread that is 1 for one minute in every hour and 10 otherwise:
        # the minimum and the median are then unmistakably different numbers
        spreads = np.where(np.arange(minutes) % 60 == 0, 1.0, 10.0)
    price = 2000.0 + np.arange(minutes) * 0.01
    return pd.DataFrame(
        {
            "open": price,
            "high": price + 0.5,
            "low": price - 0.5,
            "close": price,
            "tick_volume": 100.0,
            "spread": spreads.astype("float64"),
            "real_volume": 0.0,
        },
        index=index,
    )


def h1_frame(hours: int = 10, minimum: float = 1.0) -> pd.DataFrame:
    """H1 bars carrying the raw field under its honest name."""
    index = pd.date_range(START, periods=hours, freq="1h", tz="UTC", name="time")
    price = 2000.0 + np.arange(hours)
    return pd.DataFrame(
        {
            "open": price,
            "high": price + 5.0,
            "low": price - 5.0,
            "close": price,
            "tick_volume": 6000.0,
            MIN_SPREAD_M1_COLUMN: minimum,
            "real_volume": 0.0,
        },
        index=index,
    )


# -- 1. the name -----------------------------------------------------------


def test_the_raw_column_is_only_called_spread_on_m1() -> None:
    assert spread_column_for(Timeframe.M1) == "spread"
    for name in ("M5", "M15", "H1", "H4", "D1"):
        assert spread_column_for(name) == MIN_SPREAD_M1_COLUMN
    assert bar_columns_for("H1")[5] == MIN_SPREAD_M1_COLUMN
    assert bar_columns_for("M1")[5] == "spread"


def test_rename_is_idempotent_and_never_touches_m1() -> None:
    raw = m1_frame(120)
    assert "spread" in rename_aggregated_spread(raw, Timeframe.M1).columns

    once = rename_aggregated_spread(raw, Timeframe.H1)
    twice = rename_aggregated_spread(once, Timeframe.H1)
    assert MIN_SPREAD_M1_COLUMN in once.columns and "spread" not in once.columns
    assert twice.columns.tolist() == once.columns.tolist()


def test_a_reconstructed_spread_is_not_overwritten_by_the_rename() -> None:
    """Both names present means somebody put an honest column there."""
    frame = h1_frame()
    frame["spread"] = 7.0
    out = rename_aggregated_spread(frame, Timeframe.H1)
    assert out["spread"].eq(7.0).all()
    assert out[MIN_SPREAD_M1_COLUMN].eq(1.0).all()


def test_the_cache_renames_on_the_way_out(tmp_path) -> None:
    """Whatever the file on disk says, above M1 the reader gets the truth."""
    cache = ParquetCache(tmp_path)
    hours = h1_frame(24).rename(columns={MIN_SPREAD_M1_COLUMN: "spread"})
    cache.write_year(
        "SYM", Timeframe.H1, 2024, hours, [(START, START + pd.Timedelta(days=1))]
    )
    read = cache.read_year("SYM", Timeframe.H1, 2024)
    assert MIN_SPREAD_M1_COLUMN in read.columns
    assert "spread" not in read.columns

    minutes = m1_frame(120)
    cache.write_year(
        "SYM", Timeframe.M1, 2024, minutes, [(START, START + pd.Timedelta(hours=2))]
    )
    assert "spread" in cache.read_year("SYM", Timeframe.M1, 2024).columns


def test_normalize_does_not_turn_a_renamed_column_into_a_zero_spread(
    tmp_path,
) -> None:
    """The failure this rename could have introduced, checked directly.

    `normalize_bars` fills a column it was asked for and cannot find with
    zero. Asked for `spread` on an H1 frame it would produce a free trade on
    every bar - which is worse than the bug being fixed.
    """
    cache = ParquetCache(tmp_path)
    hours = h1_frame(24)
    cache.write_year(
        "SYM", Timeframe.H1, 2024, hours, [(START, START + pd.Timedelta(days=1))]
    )
    read = cache.read_year("SYM", Timeframe.H1, 2024)
    assert not read.empty
    assert "spread" not in read.columns
    assert read[MIN_SPREAD_M1_COLUMN].gt(0).all()


# -- 2. the refusal --------------------------------------------------------


def test_per_bar_above_m1_is_refused_not_served() -> None:
    bars = h1_frame()
    with pytest.raises(AggregatedSpreadRefused) as exc:
        SpreadPolicy(mode="per_bar").series(bars, Timeframe.H1)

    message = str(exc.value)
    assert MIN_SPREAD_M1_COLUMN in message
    assert "MINIMUM" in message
    # the message has to say what to do, not only what went wrong
    assert "spread_mode 'fixed'" in message
    assert "H1" in message


def test_quantile_mode_is_refused_on_the_same_grounds() -> None:
    with pytest.raises(AggregatedSpreadRefused):
        SpreadPolicy(mode="quantile", value=0.9).series(h1_frame(), Timeframe.H1)


def test_fixed_mode_never_reads_the_bars() -> None:
    """A measured constant is the escape hatch, and it must stay open."""
    series = SpreadPolicy(mode="fixed", value=7.0).series(h1_frame(), Timeframe.H1)
    assert series.eq(7.0).all()
    assert CostModel.zero().spread.series(h1_frame(), Timeframe.H1).eq(0.0).all()


def test_per_bar_still_works_on_m1() -> None:
    bars = m1_frame(120)
    series = SpreadPolicy(mode="per_bar").series(bars, Timeframe.M1)
    assert series.equals(bars["spread"].astype("float64"))


def test_the_backtester_refuses_rather_than_charging_the_minimum() -> None:
    """End to end: the guard is not something a caller can route around."""
    from core.engine.backtester import BacktestConfig, run_backtest
    from tests.conftest_engine import spec_from, symbol_spec

    spec = spec_from()
    spec = spec.model_copy(
        update={"instrument": spec.instrument.model_copy(update={"timeframe": "H1"})}
    )
    with pytest.raises(AggregatedSpreadRefused):
        run_backtest(
            spec,
            h1_frame(50),
            symbol_spec(),
            timezone.utc,
            BacktestConfig(costs=CostModel(spread=SpreadPolicy(mode="per_bar"))),
        )


# -- 3. the reconstruction -------------------------------------------------


def test_reconstruction_takes_the_median_not_the_minimum() -> None:
    m1 = m1_frame(600)
    index = pd.date_range(START, periods=10, freq="1h", tz="UTC", name="time")

    rebuilt = reconstruct_from_m1(m1, index, Timeframe.H1)
    assert rebuilt.eq(10.0).all()  # the median of one 1 and fifty-nine 10s
    # the minimum is what the raw column would have charged
    assert m1["spread"].resample("1h").min().eq(1.0).all()


def test_reconstruction_honours_the_quantile() -> None:
    m1 = m1_frame(600, spreads=np.tile(np.arange(1.0, 61.0), 10))
    index = pd.date_range(START, periods=10, freq="1h", tz="UTC", name="time")

    median = reconstruct_from_m1(m1, index, Timeframe.H1, 0.5)
    p90 = reconstruct_from_m1(m1, index, Timeframe.H1, 0.9)
    assert float(median.iloc[0]) == pytest.approx(30.5)
    assert float(p90.iloc[0]) == pytest.approx(54.1)
    assert (p90 > median).all()


def test_the_quantile_cannot_walk_back_to_the_minimum() -> None:
    m1 = m1_frame(600)
    index = pd.date_range(START, periods=10, freq="1h", tz="UTC", name="time")
    for below in (0.0, 0.1, 0.49):
        with pytest.raises(ValueError, match=r"between 0\.5"):
            reconstruct_from_m1(m1, index, Timeframe.H1, below)


def test_reconstruction_follows_the_bars_it_is_given_not_an_epoch_grid() -> None:
    """This broker stamps H4 bars at 21:00, and a resample would miss them.

    The failure mode is not subtle - an epoch-aligned 4-hour grid lines up
    with none of the bars and reports the whole series as uncovered - but it
    only shows up on the timeframes the campaign actually ran on.
    """
    m1 = m1_frame(60 * 24)
    index = pd.DatetimeIndex(
        [START + pd.Timedelta(hours=h) for h in (1, 5, 9, 13, 17)],
        tz="UTC",
        name="time",
    )
    rebuilt = reconstruct_from_m1(m1, index, Timeframe.H4)
    assert len(rebuilt) == len(index)
    assert rebuilt.notna().all()
    assert rebuilt.index.equals(index)


def test_minutes_in_a_session_gap_belong_to_no_bar() -> None:
    """A weekend's worth of minutes must not be folded into the bar before it."""
    m1 = m1_frame(60 * 6)
    index = pd.DatetimeIndex(
        [START, START + pd.Timedelta(hours=5)], tz="UTC", name="time"
    )
    rebuilt = reconstruct_from_m1(m1, index, Timeframe.H1)
    assert len(rebuilt) == 2
    # hours 1 to 4 are in the gap: they contribute to neither bar
    assert rebuilt.notna().all()


def test_zero_m1_spreads_are_dropped_not_averaged_in() -> None:
    spreads = np.where(np.arange(120) % 2 == 0, 0.0, 8.0)
    m1 = m1_frame(120, spreads=spreads)
    index = pd.date_range(START, periods=2, freq="1h", tz="UTC", name="time")
    assert reconstruct_from_m1(m1, index, Timeframe.H1).eq(8.0).all()


def test_attach_puts_a_charged_column_next_to_the_raw_one(tmp_path) -> None:
    cache = ParquetCache(tmp_path)
    cache.write_year(
        "SYM", Timeframe.M1, 2024, m1_frame(600),
        [(START, START + pd.Timedelta(hours=10))],
    )
    hours = h1_frame(10)
    out = attach(cache, "SYM", Timeframe.H1, hours)

    assert out["spread"].eq(10.0).all()
    assert out[MIN_SPREAD_M1_COLUMN].eq(1.0).all()
    # never in place: the caller's frame keeps saying what it said
    assert "spread" not in hours.columns
    charged = SpreadPolicy(mode="per_bar").series(out, Timeframe.H1)
    assert charged.eq(10.0).all()


def test_m1_bars_are_returned_untouched_by_attach(tmp_path) -> None:
    cache = ParquetCache(tmp_path)
    minutes = m1_frame(60)
    assert attach(cache, "SYM", Timeframe.M1, minutes) is minutes


# -- 4. the refusal when the M1 sample is missing --------------------------


def test_uncovered_bars_refuse_rather_than_interpolate() -> None:
    m1 = m1_frame(120)  # two hours of minutes
    index = pd.date_range(START, periods=10, freq="1h", tz="UTC", name="time")

    with pytest.raises(SpreadUnavailable) as exc:
        reconstruct_from_m1(m1, index, Timeframe.H1)
    message = str(exc.value)
    assert "8 of 10" in message
    assert "spread_mode 'fixed'" in message


def test_no_m1_history_at_all_refuses_with_the_remedy(tmp_path) -> None:
    cache = ParquetCache(tmp_path)
    index = pd.date_range(START, periods=5, freq="1h", tz="UTC", name="time")
    with pytest.raises(SpreadUnavailable) as exc:
        reconstructed_series(cache, "SYM", Timeframe.H1, index)
    assert "M1" in str(exc.value)
    assert "minimum of the M1 spreads" in str(exc.value)


def test_a_run_config_asking_for_per_bar_above_m1_fails_loudly(tmp_path) -> None:
    """The loader every run goes through refuses before anything is computed."""
    from core.runs.runner import load_bars_for_run
    from core.runs.store import RunConfig

    cache = ParquetCache(tmp_path)
    cache.write_year(
        "SYM", Timeframe.H1, 2024, h1_frame(24),
        [(START, START + pd.Timedelta(days=1))],
    )
    config = RunConfig(symbol="SYM", timeframe="H1", spread_mode="per_bar")
    with pytest.raises(SpreadUnavailable):
        load_bars_for_run(cache, config)

    # with the M1 sample present the same call succeeds and carries a spread
    cache.write_year(
        "SYM", Timeframe.M1, 2024, m1_frame(60 * 24),
        [(START, START + pd.Timedelta(days=1))],
    )
    bars = load_bars_for_run(cache, config)
    assert bars["spread"].eq(10.0).all()


def test_a_fixed_config_needs_no_m1_at_all(tmp_path) -> None:
    from core.runs.runner import load_bars_for_run
    from core.runs.store import RunConfig

    cache = ParquetCache(tmp_path)
    cache.write_year(
        "SYM", Timeframe.H1, 2024, h1_frame(24),
        [(START, START + pd.Timedelta(days=1))],
    )
    config = RunConfig(
        symbol="SYM", timeframe="H1", spread_mode="fixed", spread_value=6.0
    )
    bars = load_bars_for_run(cache, config)
    assert "spread" not in bars.columns
    assert SpreadPolicy(mode="fixed", value=6.0).series(bars, Timeframe.H1).eq(6.0).all()


# -- 5. the mark on runs produced before any of this existed ---------------


def test_the_reconstruction_quantile_is_part_of_the_run_identity() -> None:
    from core.runs.store import RunConfig, compute_run_id
    from tests.conftest_engine import spec_from, symbol_spec

    spec = spec_from()
    base = RunConfig(symbol="SYM", timeframe="H1", spread_mode="per_bar")
    other = RunConfig(
        symbol="SYM",
        timeframe="H1",
        spread_mode="per_bar",
        per_bar_spread_quantile=0.9,
    )
    left = compute_run_id(spec, base, "fp", symbol_spec())
    right = compute_run_id(spec, other, "fp", symbol_spec())
    assert left != right


def test_a_reconstructed_frame_fingerprints_differently(tmp_path) -> None:
    """Two different cost inputs must not share a run id."""
    from core.runs.store import data_fingerprint

    cache = ParquetCache(tmp_path)
    cache.write_year(
        "SYM", Timeframe.M1, 2024, m1_frame(600),
        [(START, START + pd.Timedelta(hours=10))],
    )
    raw = h1_frame(10)
    rebuilt = attach(cache, "SYM", Timeframe.H1, raw)
    assert data_fingerprint(raw) != data_fingerprint(rebuilt)


def test_the_realism_report_no_longer_claims_to_read_the_column(tmp_path) -> None:
    cache = ParquetCache(tmp_path)
    cache.write_year(
        "SYM", Timeframe.M1, 2024, m1_frame(600),
        [(START, START + pd.Timedelta(hours=10))],
    )
    bars = attach(cache, "SYM", Timeframe.H1, h1_frame(10))
    realism = SpreadPolicy(mode="per_bar").realism(bars, Timeframe.H1)

    assert realism.reads_aggregated_column is False
    assert realism.reconstructed_from_m1 is True
    assert realism.trustworthy is True
    assert realism.median_charged_points == 10.0
