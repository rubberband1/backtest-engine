"""Indicators: warm-up genuinely NaN, and values that check out by hand."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.indicators import functions as f
from core.indicators import registry


@pytest.fixture
def series() -> pd.Series:
    rng = np.random.default_rng(4)
    return pd.Series(2000 + np.cumsum(rng.normal(0, 0.5, 300)))


@pytest.fixture
def bars() -> pd.DataFrame:
    rng = np.random.default_rng(4)
    close = 2000 + np.cumsum(rng.normal(0, 0.5, 300))
    open_ = np.concatenate(([close[0]], close[:-1]))
    high = np.maximum(open_, close) + np.abs(rng.normal(0, 0.4, 300))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, 0.4, 300))
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "tick_volume": 100.0,
            "spread": 10.0,
            "real_volume": 0.0,
        }
    )


@pytest.mark.parametrize(
    "name,params,expected_warmup",
    [
        ("sma", {"period": 20}, 19),
        ("ema", {"period": 20}, 19),
        ("rsi", {"period": 14}, 14),
        ("roc", {"period": 12}, 12),
        ("atr", {"period": 14}, 14),
        ("bollinger", {"period": 20}, 19),
        ("macd", {"fast": 12, "slow": 26, "signal": 9}, 33),
        ("stoch", {"period": 14, "smooth_k": 3, "smooth_d": 3}, 17),
        ("donchian", {"period": 20}, 20),
    ],
)
def test_warmup_stays_nan(
    bars: pd.DataFrame, name: str, params: dict[str, int], expected_warmup: int
) -> None:
    """No fillna: the first bars have no value and must say so."""
    values = registry.get(name).compute(bars, params)
    frame = values.to_frame() if isinstance(values, pd.Series) else values

    assert f.warmup_bars(values) == expected_warmup
    # during warm-up at least one output is undefined (macd and signal form
    # at different times), afterwards none remains
    assert frame.iloc[:expected_warmup].isna().any(axis=1).all()
    assert frame.iloc[expected_warmup:].notna().all().all()


def test_rsi_on_monotone_series_is_one_hundred(series: pd.Series) -> None:
    ascent = pd.Series(np.arange(100, dtype="float64"))
    values = f.rsi(ascent, period=14)
    assert values.iloc[:14].isna().all()
    assert values.iloc[14:].eq(100.0).all()


def test_rsi_uses_wilder_alpha() -> None:
    """The first defined value must match the hand-made recursive computation."""
    values = pd.Series([10.0, 11, 10.5, 11.5, 12, 11.8, 12.5, 12.2])
    period = 3
    delta = values.diff()
    alpha = 1.0 / period
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)

    avg_gain = gains.iloc[1]
    avg_loss = losses.iloc[1]
    for i in range(2, len(values)):
        avg_gain = avg_gain + alpha * (gains.iloc[i] - avg_gain)
        avg_loss = avg_loss + alpha * (losses.iloc[i] - avg_loss)
    expected = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)

    assert f.rsi(values, period).iloc[-1] == pytest.approx(expected)


def test_ema_recursive_without_rescaling() -> None:
    values = pd.Series([1.0, 2, 3, 4, 5, 6])
    period = 3
    alpha = 2.0 / (period + 1)
    seed = values.iloc[:period].mean()  # not the seed actually used: just a reference
    manual = values.iloc[0]
    for x in values.iloc[1:]:
        manual = manual + alpha * (x - manual)
    assert f.ema(values, period).iloc[-1] == pytest.approx(manual)
    assert seed != manual


def test_donchian_excludes_the_current_bar(bars: pd.DataFrame) -> None:
    channel = f.donchian(bars["high"], bars["low"], period=5)
    # if it included the current bar, high could never exceed upper
    breakouts = (bars["high"] > channel["upper"]).sum()
    assert breakouts > 0


def test_true_range_first_bar_undefined(bars: pd.DataFrame) -> None:
    tr = f.true_range(bars["high"], bars["low"], bars["close"])
    assert pd.isna(tr.iloc[0])
    assert tr.iloc[1:].notna().all()


def test_cross_does_not_fire_on_the_warmup_boundary() -> None:
    """A NaN turning into a number is not a crossing."""
    left = pd.Series([np.nan, np.nan, 30.0, 20.0, 26.0])
    right = pd.Series(25.0, index=left.index)

    below = f.crossed_below(left, right)
    assert not below.iloc[2]  # first defined value: no "before" to compare with
    assert below.iloc[3]
    assert not below.iloc[4]
    assert f.crossed_above(left, right).iloc[4]


def test_rising_and_falling_ignore_nans() -> None:
    values = pd.Series([np.nan, 1.0, 2.0, 2.0, 1.0])
    assert list(f.rising(values)) == [False, False, True, False, False]
    assert list(f.falling(values)) == [False, False, False, False, True]


def test_stoch_on_flat_range_does_not_divide_by_zero() -> None:
    flat = pd.Series(2000.0, index=range(30))
    result = f.stoch(flat, flat, flat, period=14, smooth_k=1, smooth_d=1)
    assert result["k"].iloc[14:].eq(50.0).all()


def test_registry_rejects_unknown_parameters(bars: pd.DataFrame) -> None:
    with pytest.raises(Exception, match=r"periodo|period|extra"):
        registry.get("rsi").compute(bars, {"periodo": 14})


def test_registry_lists_the_indicators() -> None:
    names = registry.available()
    assert {"sma", "ema", "rsi", "atr", "bollinger", "macd", "stoch", "donchian", "roc"} <= set(
        names
    )
    with pytest.raises(KeyError, match="unknown"):
        registry.get("supertrend")


def test_configurable_source(bars: pd.DataFrame) -> None:
    on_close = registry.get("sma").compute(bars, {"period": 5, "source": "close"})
    on_hl2 = registry.get("sma").compute(bars, {"period": 5, "source": "hl2"})
    assert not on_close.equals(on_hl2)
