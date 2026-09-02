"""Technical indicators as pure functions over pandas.

Non-negotiable rule: **no fillna on warm-up**. The first bars where the
indicator is not yet defined stay NaN and must propagate to the signals as
"no signal". Filling a not-yet-formed RSI with 50 fabricates crossings that
never existed in the market, and fabricates them right at the start of the
sample, where nobody looks.

Functions working on a single series have signature `(series, **params)`.
Those needing multiple columns (ATR, stochastic, Donchian) take the series
they need as explicit arguments: it is the only way to stay pure functions
without pretending the true range is computed on the close alone.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _wilder(series: pd.Series, period: int) -> pd.Series:
    """Wilder's exponential mean: alpha = 1/period, no rescaling.

    `min_periods` keeps the warm-up NaN instead of returning a partial mean.
    """
    return series.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI. The first defined value lands on bar `period`."""
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = _wilder(gain, period)
    avg_loss = _wilder(loss, period)
    rs = avg_gain / avg_loss
    out = 100.0 - 100.0 / (1.0 + rs)
    # zero average loss => RSI 100 by definition, not NaN from 0/0
    out = out.where(avg_loss != 0.0, 100.0)
    return out.where(avg_gain.notna() & avg_loss.notna())


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    previous_close = close.shift(1)
    ranges = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    )
    tr = ranges.max(axis=1)
    # the first bar has no previous close: it stays undefined
    return tr.where(previous_close.notna())


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    return _wilder(true_range(high, low, close), period)


def roc(series: pd.Series, period: int = 12) -> pd.Series:
    """Rate of change, in percent."""
    return series.pct_change(periods=period) * 100.0


def bollinger(
    series: pd.Series, period: int = 20, deviations: float = 2.0
) -> pd.DataFrame:
    middle = sma(series, period)
    # population deviation (ddof=0), as in Bollinger's definition
    sigma = series.rolling(window=period, min_periods=period).std(ddof=0)
    return pd.DataFrame(
        {
            "middle": middle,
            "upper": middle + deviations * sigma,
            "lower": middle - deviations * sigma,
            "width": 2.0 * deviations * sigma,
        }
    )


def macd(
    series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    if fast >= slow:
        raise ValueError(f"macd: fast ({fast}) must be less than slow ({slow})")
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = macd_line.ewm(
        span=signal, adjust=False, min_periods=signal
    ).mean()
    # the signal line's warm-up starts after the macd's: mask it
    signal_line = signal_line.where(macd_line.notna())
    return pd.DataFrame(
        {
            "macd": macd_line,
            "signal": signal_line,
            "histogram": macd_line - signal_line,
        }
    )


def stoch(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
    smooth_k: int = 3,
    smooth_d: int = 3,
) -> pd.DataFrame:
    highest = high.rolling(window=period, min_periods=period).max()
    lowest = low.rolling(window=period, min_periods=period).min()
    span = highest - lowest
    raw_k = 100.0 * (close - lowest) / span
    # flat range: the price is exactly mid-channel by convention
    raw_k = raw_k.where(span != 0.0, 50.0).where(span.notna())
    k = raw_k.rolling(window=smooth_k, min_periods=smooth_k).mean()
    d = k.rolling(window=smooth_d, min_periods=smooth_d).mean()
    return pd.DataFrame({"k": k, "d": d})


def donchian(high: pd.Series, low: pd.Series, period: int = 20) -> pd.DataFrame:
    """Donchian channel over the **previous** `period` bars.

    The current bar is excluded: including it would make the breakout true by
    construction (the channel high would contain today's high).
    """
    upper = high.shift(1).rolling(window=period, min_periods=period).max()
    lower = low.shift(1).rolling(window=period, min_periods=period).min()
    return pd.DataFrame({"upper": upper, "lower": lower, "middle": (upper + lower) / 2.0})


def crossed_above(left: pd.Series, right: pd.Series) -> pd.Series:
    """True on the bar where `left` moves above `right`.

    NaN on either side (warm-up included) means no crossing.
    """
    previous = left.shift(1) <= right.shift(1)
    now = left > right
    valid = left.notna() & right.notna() & left.shift(1).notna() & right.shift(1).notna()
    return (previous & now & valid).fillna(False)


def crossed_below(left: pd.Series, right: pd.Series) -> pd.Series:
    previous = left.shift(1) >= right.shift(1)
    now = left < right
    valid = left.notna() & right.notna() & left.shift(1).notna() & right.shift(1).notna()
    return (previous & now & valid).fillna(False)


def rising(series: pd.Series, periods: int = 1) -> pd.Series:
    previous = series.shift(periods)
    return ((series > previous) & series.notna() & previous.notna()).fillna(False)


def falling(series: pd.Series, periods: int = 1) -> pd.Series:
    previous = series.shift(periods)
    return ((series < previous) & series.notna() & previous.notna()).fillna(False)


def warmup_bars(values: pd.Series | pd.DataFrame) -> int:
    """How many leading bars are NaN. Used by the tests, not by the logic."""
    frame = values.to_frame() if isinstance(values, pd.Series) else values
    valid = frame.notna().all(axis=1)
    if not valid.any():
        return int(len(frame))
    return int(np.argmax(valid.to_numpy()))
