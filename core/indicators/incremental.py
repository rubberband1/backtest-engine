"""Bar-at-a-time indicator state, exact to the last bit.

The live runner used to recompute every indicator over its whole retained
history on every closed bar. Correct, and quadratic: replaying a year of H1
through it costs a hundred times what the backtest costs, and the cost grows
with the history the runner is asked to keep.

The obvious fix - recompute over a trailing window - is wrong, and measurably
so. A rolling mean in pandas is a Kahan-compensated running sum with an
add/remove pair, so its compensation term depends on every value that has
ever entered the window; recomputing the last 20, 500 or 5000 bars of an
`sma(20)` reproduces the full-series value on roughly two thirds of the bars
and never converges. `bollinger` and `stoch` miss on every single bar at
every window length tried. An exponentially weighted mean does converge, but
only because `(1-alpha)^k` eventually falls under the representation error -
which is a fact about the data, not a guarantee, and it is not one to hang a
zero-tolerance equivalence test on.

So this module does not approximate pandas. It **is** pandas' arithmetic,
transcribed one operation at a time from `pandas/_libs/window/aggregations.pyx`
(2.3.x): `add_mean`/`remove_mean` with their two separate compensation
accumulators, `add_var`/`remove_var` with Welford's update, `calc_mean` and
`calc_var` with their consecutive-same-value and all-negative branches, and
the `ewm` recursion in its `adjust=False` form - including the division by
`(old_wt + new_wt)`, which is not one and cannot be simplified away.

`tests/test_incremental.py` demands bit equality against a from-scratch
recomputation over five thousand bars for all nine registered indicators.
Anything that fails that stays recomputed and is declared there rather than
shipped as "close enough": a live runner that disagrees with the backtester
in the last decimal is a live runner nobody can compare with a backtest.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Mapping
from typing import Any

import numpy as np

NAN = float("nan")

# The bar fields a state may read, in the spelling the frames use.
BarView = Mapping[str, float]


def _is_nan(value: float) -> bool:
    return value != value


def _signbit(value: float) -> bool:
    """True for negatives and for -0.0, as C's signbit is."""
    return math.copysign(1.0, value) < 0.0


# -- the primitives, transcribed ------------------------------------------


class RollingMean:
    """`Series.rolling(window, min_periods).mean()`, one value at a time.

    Mirrors `roll_mean`: a fixed window over a contiguous series takes the
    incremental branch from the second bar onwards, removing the value that
    left and adding the one that arrived, each against its own Kahan
    compensation term. A window of one takes the setup branch every bar,
    which the loop below reproduces rather than special-cases.
    """

    def __init__(self, window: int, min_periods: int | None = None) -> None:
        self.window = int(window)
        self.min_periods = int(window if min_periods is None else min_periods)
        self._values: deque[float] = deque()
        self._nobs = 0
        self._neg_ct = 0
        self._sum_x = 0.0
        self._compensation_add = 0.0
        self._compensation_remove = 0.0
        self._num_consecutive_same_value = 0
        self._prev_value = 0.0

    def _add(self, val: float) -> None:
        if not _is_nan(val):
            self._nobs += 1
            y = val - self._compensation_add
            t = self._sum_x + y
            self._compensation_add = t - self._sum_x - y
            self._sum_x = t
            if _signbit(val):
                self._neg_ct += 1
            if val == self._prev_value:
                self._num_consecutive_same_value += 1
            else:
                self._num_consecutive_same_value = 1
            self._prev_value = val

    def _remove(self, val: float) -> None:
        if not _is_nan(val):
            self._nobs -= 1
            y = -val - self._compensation_remove
            t = self._sum_x + y
            self._compensation_remove = t - self._sum_x - y
            self._sum_x = t
            if _signbit(val):
                self._neg_ct -= 1

    def _calc(self) -> float:
        if self._nobs >= self.min_periods and self._nobs > 0:
            result = self._sum_x / float(self._nobs)
            if self._num_consecutive_same_value >= self._nobs:
                result = self._prev_value
            elif (self._neg_ct == 0 and result < 0) or (self._neg_ct == self._nobs and result > 0):
                result = 0.0
            return result
        return NAN

    def update(self, value: float) -> float:
        value = float(value)
        self._values.append(value)
        # the setup branch: the first bar, and every bar of a window of one,
        # for which the new window starts at or after the end of the old
        if len(self._values) == 1 or self.window == 1:
            self._values = deque(list(self._values)[-self.window :])
            self._nobs = self._neg_ct = 0
            self._sum_x = self._compensation_add = self._compensation_remove = 0.0
            self._prev_value = self._values[0]
            self._num_consecutive_same_value = 0
            for held in self._values:
                self._add(held)
            return self._calc()

        if len(self._values) > self.window:
            self._remove(self._values.popleft())
        self._add(value)
        return self._calc()


class RollingVar:
    """`Series.rolling(window, min_periods).var(ddof)`, one value at a time.

    Mirrors `roll_var`: Welford's online update with Kahan compensation, in
    the exact order the Cython writes it. `ssqdm_x` is accumulated as
    `(val - prev_mean) * (val - mean_x)` and not as any algebraically equal
    rearrangement, because an algebraically equal rearrangement is not a
    numerically equal one.
    """

    def __init__(self, window: int, min_periods: int | None = None, ddof: int = 1) -> None:
        self.window = int(window)
        self.min_periods = max(int(window if min_periods is None else min_periods), 1)
        self.ddof = int(ddof)
        self._values: deque[float] = deque()
        self._nobs = 0.0
        self._mean_x = 0.0
        self._ssqdm_x = 0.0
        self._compensation_add = 0.0
        self._compensation_remove = 0.0
        self._num_consecutive_same_value = 0
        self._prev_value = 0.0

    def _add(self, val: float) -> None:
        if _is_nan(val):
            return
        self._nobs += 1.0
        if val == self._prev_value:
            self._num_consecutive_same_value += 1
        else:
            self._num_consecutive_same_value = 1
        self._prev_value = val

        prev_mean = self._mean_x - self._compensation_add
        y = val - self._compensation_add
        t = y - self._mean_x
        self._compensation_add = t + self._mean_x - y
        delta = t
        if self._nobs:
            self._mean_x = self._mean_x + delta / self._nobs
        else:
            self._mean_x = 0.0
        self._ssqdm_x = self._ssqdm_x + (val - prev_mean) * (val - self._mean_x)

    def _remove(self, val: float) -> None:
        if _is_nan(val):
            return
        self._nobs -= 1.0
        if self._nobs:
            prev_mean = self._mean_x - self._compensation_remove
            y = val - self._compensation_remove
            t = y - self._mean_x
            self._compensation_remove = t + self._mean_x - y
            delta = t
            self._mean_x = self._mean_x - delta / self._nobs
            self._ssqdm_x = self._ssqdm_x - (val - prev_mean) * (val - self._mean_x)
        else:
            self._mean_x = 0.0
            self._ssqdm_x = 0.0

    def _calc(self) -> float:
        if self._nobs >= self.min_periods and self._nobs > self.ddof:
            if self._nobs == 1 or self._num_consecutive_same_value >= self._nobs:
                return 0.0
            return self._ssqdm_x / (self._nobs - float(self.ddof))
        return NAN

    def update(self, value: float) -> float:
        value = float(value)
        self._values.append(value)
        if len(self._values) == 1 or self.window == 1:
            self._values = deque(list(self._values)[-self.window :])
            self._nobs = 0.0
            self._mean_x = self._ssqdm_x = 0.0
            self._compensation_add = self._compensation_remove = 0.0
            self._prev_value = self._values[0]
            self._num_consecutive_same_value = 0
            for held in self._values:
                self._add(held)
            return self._calc()

        if len(self._values) > self.window:
            self._remove(self._values.popleft())
        self._add(value)
        return self._calc()


class RollingStd(RollingVar):
    """`.std()`, which pandas computes as the square root of `.var()`.

    A negative variance is impossible in exact arithmetic and reachable in
    floating point; pandas maps it to zero rather than to a NaN, and so does
    this.
    """

    def update(self, value: float) -> float:
        variance = super().update(value)
        if _is_nan(variance):
            return NAN
        return math.sqrt(variance) if variance > 0 else 0.0


class RollingExtreme:
    """`.max()` or `.min()` over a fixed window.

    No accumulation, so no arithmetic to reproduce: the answer is one of the
    inputs, and any correct implementation returns the same bits. NaN is
    skipped for the observation count exactly as pandas skips it.
    """

    def __init__(self, window: int, min_periods: int | None = None, largest: bool = True) -> None:
        self.window = int(window)
        self.min_periods = int(window if min_periods is None else min_periods)
        self.largest = largest
        self._values: deque[float] = deque()

    def update(self, value: float) -> float:
        self._values.append(float(value))
        if len(self._values) > self.window:
            self._values.popleft()
        present = [v for v in self._values if not _is_nan(v)]
        if len(present) < self.min_periods or not present:
            return NAN
        return max(present) if self.largest else min(present)


class Ewm:
    """`Series.ewm(..., adjust=False, ignore_na=False).mean()`, bar by bar.

    Transcribed from the `ewm` loop. Two details carry the exactness:

    - `alpha` is not the alpha the caller asked for. pandas converts the
      request to a centre of mass and back (`com = (1-alpha)/alpha`, then
      `alpha = 1/(1+com)`), and the round trip does not land on the same
      double. Passing the caller's alpha straight in produces a series that
      is right to twelve digits and wrong in the last two.
    - the update divides by `(old_wt + new_wt)`. With `adjust=False` that
      sum is `(1-alpha) + alpha`, which is not 1.0 for most alphas.
    """

    def __init__(self, com: float, min_periods: int = 1) -> None:
        self.com = float(com)
        self.min_periods = max(int(min_periods), 1)
        self._alpha = 1.0 / (1.0 + self.com)
        self._old_wt_factor = 1.0 - self._alpha
        self._new_wt = self._alpha
        self._weighted = NAN
        self._old_wt = 1.0
        self._nobs = 0
        self._started = False

    @classmethod
    def from_span(cls, span: float, min_periods: int = 1) -> Ewm:
        return cls((float(span) - 1.0) / 2.0, min_periods)

    @classmethod
    def from_alpha(cls, alpha: float, min_periods: int = 1) -> Ewm:
        return cls((1.0 - float(alpha)) / float(alpha), min_periods)

    def update(self, value: float) -> float:
        cur = float(value)
        is_observation = not _is_nan(cur)

        if not self._started:
            # the loop seeds `weighted` with the first value, whatever it is,
            # and only then enters the recursion
            self._started = True
            self._weighted = cur
            self._nobs = int(is_observation)
            self._old_wt = 1.0
            return self._weighted if self._nobs >= self.min_periods else NAN

        self._nobs += int(is_observation)
        if not _is_nan(self._weighted):
            # ignore_na is False, so a missing bar still decays the old weight
            self._old_wt *= self._old_wt_factor
            if is_observation:
                if self._weighted != cur:
                    self._weighted = self._old_wt * self._weighted + self._new_wt * cur
                    self._weighted /= self._old_wt + self._new_wt
                self._old_wt = 1.0
        elif is_observation:
            self._weighted = cur

        return self._weighted if self._nobs >= self.min_periods else NAN


class Lag:
    """The value `periods` bars ago, or NaN before there is one."""

    def __init__(self, periods: int = 1) -> None:
        self.periods = int(periods)
        self._buffer: deque[float] = deque(maxlen=self.periods + 1)

    def update(self, value: float) -> float:
        self._buffer.append(float(value))
        if len(self._buffer) <= self.periods:
            return NAN
        return self._buffer[0]


# -- sources ---------------------------------------------------------------


def resolve_source_value(bar: BarView, source: str) -> float:
    """Scalar twin of `registry.resolve_source`, same arithmetic and order."""
    if source == "hl2":
        return (bar["high"] + bar["low"]) / 2.0
    if source == "hlc3":
        return (bar["high"] + bar["low"] + bar["close"]) / 3.0
    if source == "ohlc4":
        return (bar["open"] + bar["high"] + bar["low"] + bar["close"]) / 4.0
    return float(bar[source])


# -- the indicators --------------------------------------------------------


class IndicatorState(ABC):
    """One indicator, advanced one closed bar at a time.

    `exact` declares whether this state reproduces the vectorized function
    bit for bit. A False here is not a licence to be approximately right: it
    routes the indicator back to a full recomputation, at the cost of the
    speed-up and nothing else.
    """

    exact: bool = True
    outputs: tuple[str, ...] = ()

    @abstractmethod
    def update(self, bar: BarView) -> float | dict[str, float]:
        """The indicator's value on this bar, given every bar before it."""


class SmaState(IndicatorState):
    def __init__(self, period: int = 14, source: str = "close") -> None:
        self.source = source
        self._mean = RollingMean(period, period)

    def update(self, bar: BarView) -> float:
        return self._mean.update(resolve_source_value(bar, self.source))


class EmaState(IndicatorState):
    def __init__(self, period: int = 14, source: str = "close") -> None:
        self.source = source
        self._ewm = Ewm.from_span(period, period)

    def update(self, bar: BarView) -> float:
        return self._ewm.update(resolve_source_value(bar, self.source))


class RsiState(IndicatorState):
    """Wilder's RSI, including its two conventions.

    A zero average loss is RSI 100 by definition rather than 0/0, and the
    output is undefined until both averages are - `_wilder` is fed a series
    whose first value is the NaN that `diff()` leaves behind, so the first
    defined value lands on bar `period`.
    """

    def __init__(self, period: int = 14, source: str = "close") -> None:
        self.source = source
        self._previous = NAN
        self._gain = Ewm.from_alpha(1.0 / period, period)
        self._loss = Ewm.from_alpha(1.0 / period, period)

    def update(self, bar: BarView) -> float:
        value = resolve_source_value(bar, self.source)
        delta = value - self._previous if not _is_nan(self._previous) else NAN
        self._previous = value

        if _is_nan(delta):
            gain_in = loss_in = NAN
        else:
            # `clip` returns the bound itself, so a positive delta produces a
            # loss of -0.0 and not 0.0. It reaches no comparison that can
            # tell them apart, and it is kept anyway: the point of this file
            # is that nothing is "obviously equivalent"
            gain_in = delta if delta > 0.0 else 0.0
            loss_in = -(delta if delta < 0.0 else 0.0)

        avg_gain = self._gain.update(gain_in)
        avg_loss = self._loss.update(loss_in)
        if _is_nan(avg_gain) or _is_nan(avg_loss):
            return NAN
        if avg_loss == 0.0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - 100.0 / (1.0 + rs)


class RocState(IndicatorState):
    def __init__(self, period: int = 12, source: str = "close") -> None:
        self.source = source
        self._lag = Lag(period)

    def update(self, bar: BarView) -> float:
        value = resolve_source_value(bar, self.source)
        previous = self._lag.update(value)
        if _is_nan(previous):
            return NAN
        return (value / previous - 1.0) * 100.0


class BollingerState(IndicatorState):
    outputs = ("middle", "upper", "lower", "width")

    def __init__(
        self, period: int = 20, deviations: float = 2.0, source: str = "close"
    ) -> None:
        self.source = source
        self.deviations = float(deviations)
        self._mean = RollingMean(period, period)
        # population deviation, as in Bollinger's definition
        self._std = RollingStd(period, period, ddof=0)

    def update(self, bar: BarView) -> dict[str, float]:
        value = resolve_source_value(bar, self.source)
        middle = self._mean.update(value)
        sigma = self._std.update(value)
        return {
            "middle": middle,
            "upper": middle + self.deviations * sigma,
            "lower": middle - self.deviations * sigma,
            "width": 2.0 * self.deviations * sigma,
        }


class MacdState(IndicatorState):
    outputs = ("macd", "signal", "histogram")

    def __init__(
        self,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
        source: str = "close",
    ) -> None:
        if fast >= slow:
            raise ValueError(f"macd: fast ({fast}) must be less than slow ({slow})")
        self.source = source
        self._fast = Ewm.from_span(fast, fast)
        self._slow = Ewm.from_span(slow, slow)
        self._signal = Ewm.from_span(signal, signal)

    def update(self, bar: BarView) -> dict[str, float]:
        value = resolve_source_value(bar, self.source)
        fast = self._fast.update(value)
        slow = self._slow.update(value)
        macd = fast - slow
        signal = self._signal.update(macd)
        # the signal line's warm-up starts after the macd's, and is masked
        # wherever the macd itself is undefined
        if _is_nan(macd):
            signal = NAN
        return {"macd": macd, "signal": signal, "histogram": macd - signal}


class AtrState(IndicatorState):
    def __init__(self, period: int = 14) -> None:
        self._previous_close = NAN
        self._wilder = Ewm.from_alpha(1.0 / period, period)

    def update(self, bar: BarView) -> float:
        high, low, close = bar["high"], bar["low"], bar["close"]
        previous = self._previous_close
        self._previous_close = close
        if _is_nan(previous):
            # no previous close means no true range: it stays undefined, and
            # the EWM is still advanced with the NaN so its weights decay the
            # way the vectorized version's do
            return self._wilder.update(NAN)
        true_range = max(high - low, abs(high - previous), abs(low - previous))
        return self._wilder.update(true_range)


class StochState(IndicatorState):
    outputs = ("k", "d")

    def __init__(self, period: int = 14, smooth_k: int = 3, smooth_d: int = 3) -> None:
        self._highest = RollingExtreme(period, period, largest=True)
        self._lowest = RollingExtreme(period, period, largest=False)
        self._k = RollingMean(smooth_k, smooth_k)
        self._d = RollingMean(smooth_d, smooth_d)

    def update(self, bar: BarView) -> dict[str, float]:
        highest = self._highest.update(bar["high"])
        lowest = self._lowest.update(bar["low"])
        span = highest - lowest
        if _is_nan(span):
            raw_k = NAN
        elif span == 0.0:
            # a flat range puts the price exactly mid-channel by convention
            raw_k = 50.0
        else:
            raw_k = 100.0 * (bar["close"] - lowest) / span
        k = self._k.update(raw_k)
        return {"k": k, "d": self._d.update(k)}


class DonchianState(IndicatorState):
    outputs = ("upper", "lower", "middle")

    def __init__(self, period: int = 20) -> None:
        # the channel is over the PREVIOUS `period` bars: including the
        # current one would make a breakout true by construction
        self._high_lag = Lag(1)
        self._low_lag = Lag(1)
        self._upper = RollingExtreme(period, period, largest=True)
        self._lower = RollingExtreme(period, period, largest=False)

    def update(self, bar: BarView) -> dict[str, float]:
        upper = self._upper.update(self._high_lag.update(bar["high"]))
        lower = self._lower.update(self._low_lag.update(bar["low"]))
        return {"upper": upper, "lower": lower, "middle": (upper + lower) / 2.0}


# -- the registry ----------------------------------------------------------

_STATES: dict[str, type[IndicatorState]] = {
    "sma": SmaState,
    "ema": EmaState,
    "rsi": RsiState,
    "roc": RocState,
    "bollinger": BollingerState,
    "macd": MacdState,
    "atr": AtrState,
    "stoch": StochState,
    "donchian": DonchianState,
}


def supported() -> list[str]:
    """Indicator types with an exact incremental state."""
    return sorted(_STATES)


def state_for(indicator_type: str, params: Mapping[str, Any]) -> IndicatorState | None:
    """The incremental state for one indicator, or None if there is none.

    None is the honest answer for an indicator nobody has transcribed yet:
    the caller falls back to recomputing it, which is slower and right,
    rather than to a state that is fast and nearly right.
    """
    from core.indicators import registry

    factory = _STATES.get(indicator_type)
    if factory is None:
        return None
    definition = registry.get(indicator_type)
    validated = definition.params_model(**dict(params)).model_dump()
    if definition.bar_inputs:
        # ATR, stochastic and Donchian read the bar itself and take no source
        validated.pop("source", None)
    return factory(**validated)


def bar_view(row: Mapping[str, Any]) -> dict[str, float]:
    """A frame row as the plain float mapping the states read.

    Going through numpy scalars would be harmless arithmetically and is
    avoided anyway: `float()` here means every number below this line has one
    type, and a `float32` sneaking in from a future cache format cannot
    quietly change a result.
    """
    return {
        key: float(value)
        for key, value in row.items()
        if isinstance(value, (int, float, np.floating, np.integer))
    }
