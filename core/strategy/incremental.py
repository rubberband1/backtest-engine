"""The condition tree evaluated on one bar, with no history to walk back over.

`core.strategy.evaluator` answers "which bars of this frame are signals?" and
answers it for the whole frame at once, which is the right shape for a
backtest. A live runner asks a different question - "is *this* bar a signal?"
- and was answering it by asking the first one again on every bar, over a
frame that grew by one row each time. Correct, quadratic, and the reason a
replay of a year of H1 cost more than the backtest it was being compared
against.

This module keeps the same answers and drops the history. Indicators advance
through `core.indicators.incremental`, which reproduces pandas' arithmetic
rather than approximating it; features come from the single bar, as they
always did; and the operators that need a previous value - the two crossings,
`rising` and `falling` - keep exactly as many past values as they ask for and
not one more.

The semantics are the vectorized module's, deliberately down to the corners:
a NaN operand makes a comparison False rather than NaN, `eq` is a relative
comparison and not a bit comparison, and a bar with both a long and a short
signal is left to the executor to discard. `tests/test_replay_equivalence.py`
is what holds the two implementations together - it replays real history
through the runner and demands the backtester's trades to the timestamp and
the price - and it is the reason this file exists as an optimization rather
than as a second opinion.
"""
from __future__ import annotations

import logging
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from core.indicators.incremental import IndicatorState, bar_view, state_for
from core.strategy import spec as sp
from core.strategy.features import FEATURE_NAMES

logger = logging.getLogger(__name__)

NAN = float("nan")

# The feed column is named tick_volume; the spec writes "volume".
_BAR_ALIASES: dict[str, str] = {"volume": "tick_volume"}


class UnsupportedSpec(RuntimeError):
    """This spec has no exact incremental evaluation.

    Raised so the caller falls back to recomputing rather than running
    something that is nearly the same. There is no third option: a live
    runner that disagrees with the backtester is the defect this whole
    engine is built around not having.
    """


@dataclass
class BarSignals:
    """What one bar produced, in the shape the executor's input needs."""

    long: bool = False
    short: bool = False
    exit_signal: bool = False
    indicators: dict[str, float] = field(default_factory=dict)
    features: dict[str, float] = field(default_factory=dict)
    # the indicator values of the PREVIOUS bar: an exit level sized in ATR is
    # read on the bar whose close decided the trade, not on the bar it fills on
    previous_indicators: dict[str, float] = field(default_factory=dict)


def bar_features(bar: Mapping[str, float], point: float) -> dict[str, float]:
    """`core.strategy.features.compute_features` for a single bar.

    A zero-range bar leaves the ratios undefined rather than zero: "no wick"
    is a claim, and there is nothing to base it on when high equals low.
    `range_points` is not masked, because a range of zero points is a fact
    about the bar and not a missing measurement - which is how the
    vectorized version has it too.
    """
    high, low = bar["high"], bar["low"]
    open_, close = bar["open"], bar["close"]
    span = high - low
    valid = span > 0
    body_top = max(open_, close)
    body_bottom = min(open_, close)
    return {
        "lower_wick_ratio": (body_bottom - low) / span if valid else NAN,
        "upper_wick_ratio": (high - body_top) / span if valid else NAN,
        "body_ratio": abs(close - open_) / span if valid else NAN,
        "range_points": span / point,
        "close_position_in_range": (close - low) / span if valid else NAN,
    }


class _History:
    """The last `depth` values of one operand, and nothing else."""

    def __init__(self, depth: int) -> None:
        self.depth = depth
        self._values: deque[float] = deque(maxlen=depth + 1)

    def push(self, value: float) -> None:
        self._values.append(float(value))

    def ago(self, periods: int) -> float:
        if len(self._values) <= periods:
            return NAN
        return self._values[-1 - periods]


class IncrementalEvaluator:
    """One spec, advanced one closed bar at a time."""

    def __init__(self, strategy: sp.StrategySpec, point: float) -> None:
        if point <= 0:
            raise ValueError(f"point must be positive, got {point}")
        self.strategy = strategy
        self.point = point

        self._states: list[tuple[sp.IndicatorSpec, IndicatorState]] = []
        for indicator in strategy.indicators:
            state = state_for(indicator.type, indicator.params)
            if state is None:
                raise UnsupportedSpec(
                    f"indicator {indicator.id!r} of type {indicator.type!r} has no "
                    f"exact incremental state: this spec must be recomputed"
                )
            self._states.append((indicator, state))

        # how far back each operand is asked about, so nothing keeps more
        self._depth: dict[str, int] = {}
        for condition in strategy.conditions():
            self._measure(condition)
        self._history: dict[str, _History] = {
            key: _History(depth) for key, depth in self._depth.items()
        }

        self.indicators: dict[str, float] = {}
        self.previous_indicators: dict[str, float] = {}
        self.features: dict[str, float] = dict.fromkeys(FEATURE_NAMES, NAN)
        self.bars_seen = 0

    # -- planning --------------------------------------------------------

    @staticmethod
    def _key(operand: sp.Operand) -> str:
        return operand.model_dump_json()

    def _want(self, operand: sp.Operand, depth: int) -> None:
        key = self._key(operand)
        self._depth[key] = max(self._depth.get(key, 0), depth)

    def _measure(self, condition: sp.Condition) -> None:
        """How many past values each operand of the tree is asked for."""
        if isinstance(condition, sp.AndOr):
            for child in condition.operands:
                self._measure(child)
        elif isinstance(condition, sp.Not):
            self._measure(condition.operand)
        elif isinstance(condition, sp.Compare):
            depth = 1 if condition.op in ("cross_above", "cross_below") else 0
            self._want(condition.left, depth)
            self._want(condition.right, depth)
        elif isinstance(condition, sp.Between):
            for operand in (condition.left, condition.low, condition.high):
                self._want(operand, 0)
        elif isinstance(condition, sp.Trend):
            self._want(condition.operand, condition.periods)
        else:  # pragma: no cover - the spec's union has no other member
            raise TypeError(f"unrecognized condition: {type(condition).__name__}")

    # -- the bar ---------------------------------------------------------

    def update(self, row: Mapping[str, Any]) -> BarSignals:
        """Advances every state by one bar and evaluates the tree on it."""
        bar = bar_view(row)
        self.previous_indicators = self.indicators
        self.indicators = self._advance_indicators(bar)
        self.features = bar_features(bar, self.point)
        self.bars_seen += 1
        self._bar = bar

        for key, history in self._history.items():
            history.push(self._resolve_key(key))

        long = self._evaluate(self.strategy.entry.long)
        short = self._evaluate(self.strategy.entry.short)
        exit_signal = self._evaluate(self.strategy.exit.signal_exit)
        if long and short:
            logger.warning(
                "bar with simultaneous long and short signals: it will be ignored"
            )
        return BarSignals(
            long=long,
            short=short,
            exit_signal=exit_signal,
            indicators=dict(self.indicators),
            features=dict(self.features),
            previous_indicators=dict(self.previous_indicators),
        )

    def _advance_indicators(self, bar: Mapping[str, float]) -> dict[str, float]:
        out: dict[str, float] = {}
        for indicator, state in self._states:
            value = state.update(bar)
            if isinstance(value, dict):
                for name, item in value.items():
                    out[f"{indicator.id}.{name}"] = item
            else:
                out[indicator.id] = value
        return out

    # -- operands --------------------------------------------------------

    def _resolve(self, operand: sp.Operand) -> float:
        if isinstance(operand, sp.ConstOperand):
            return float(operand.const)
        if isinstance(operand, sp.BarOperand):
            column = _BAR_ALIASES.get(operand.bar, operand.bar)
            if column not in self._bar:
                from core.indicators.registry import missing_column_reason

                raise KeyError(
                    missing_column_reason(
                        pd.DataFrame(columns=list(self._bar)), column
                    )
                )
            return float(self._bar[column])
        if isinstance(operand, sp.FeatureOperand):
            return self.features[operand.feature]
        if isinstance(operand, sp.RefOperand):
            try:
                return self.indicators[operand.ref]
            except KeyError:  # pragma: no cover - prevented by spec validation
                raise KeyError(f"indicator not computed: {operand.ref}") from None
        raise TypeError(f"unrecognized operand: {type(operand).__name__}")

    def _resolve_key(self, key: str) -> float:
        return self._resolve(_operand_from_key(key))

    def _ago(self, operand: sp.Operand, periods: int) -> float:
        return self._history[self._key(operand)].ago(periods)

    # -- the tree --------------------------------------------------------

    def _evaluate(self, condition: sp.Condition | None) -> bool:
        if condition is None:
            return False
        if isinstance(condition, sp.AndOr):
            results = [self._evaluate(child) for child in condition.operands]
            return all(results) if condition.op == "and" else any(results)
        if isinstance(condition, sp.Not):
            return not self._evaluate(condition.operand)
        if isinstance(condition, sp.Compare):
            return self._compare(condition)
        if isinstance(condition, sp.Between):
            value = self._resolve(condition.left)
            low = self._resolve(condition.low)
            high = self._resolve(condition.high)
            if _any_nan(value, low, high):
                return False
            return bool(value >= low and value <= high)
        if isinstance(condition, sp.Trend):
            value = self._resolve(condition.operand)
            previous = self._ago(condition.operand, condition.periods)
            if _any_nan(value, previous):
                return False
            return bool(value > previous) if condition.op == "rising" else bool(
                value < previous
            )
        raise TypeError(f"unrecognized condition: {type(condition).__name__}")

    def _compare(self, condition: sp.Compare) -> bool:
        op = condition.op
        left = self._resolve(condition.left)
        right = self._resolve(condition.right)

        if op in ("cross_above", "cross_below"):
            previous_left = self._ago(condition.left, 1)
            previous_right = self._ago(condition.right, 1)
            if _any_nan(left, right, previous_left, previous_right):
                return False
            if op == "cross_above":
                return bool(previous_left <= previous_right and left > right)
            return bool(previous_left >= previous_right and left < right)

        if _any_nan(left, right):
            return False
        if op == "gt":
            return bool(left > right)
        if op == "gte":
            return bool(left >= right)
        if op == "lt":
            return bool(left < right)
        if op == "lte":
            return bool(left <= right)
        if op == "eq":
            # exact float equality is almost always a bug: the vectorized
            # module compares up to the representation error, and so does this
            return bool(np.isclose(left, right, rtol=1e-9, atol=0.0))
        raise ValueError(f"unrecognized comparison operator: {op}")


def _any_nan(*values: float) -> bool:
    return any(value != value for value in values)


_OPERAND_CACHE: dict[str, sp.Operand] = {}


def _operand_from_key(key: str) -> sp.Operand:
    """The operand a history key stands for, parsed once and kept."""
    cached = _OPERAND_CACHE.get(key)
    if cached is None:
        import json

        payload = json.loads(key)
        for model in (sp.RefOperand, sp.ConstOperand, sp.BarOperand, sp.FeatureOperand):
            try:
                cached = model.model_validate(payload)
                break
            except Exception:
                continue
        if cached is None:  # pragma: no cover - keys come from operands
            raise ValueError(f"not an operand: {key}")
        _OPERAND_CACHE[key] = cached
    return cached


def supports(strategy: sp.StrategySpec) -> bool:
    """Whether this spec can be evaluated incrementally, without side effects."""
    try:
        IncrementalEvaluator(strategy, 1.0)
    except UnsupportedSpec:
        return False
    return True
