"""Vectorized evaluation of the condition tree.

No `eval`, no `exec`: the tree is a set of pydantic models and evaluation is a
recursive dispatch over known types. A spec is data, not code to execute.

Look-ahead: every operator uses only `shift(+n)` and `rolling`/`ewm` windows,
which look backwards. The value at bar t depends only on bars <= t, verified
by the causality test that recomputes signals over prefixes.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from core.indicators import functions as f
from core.strategy import spec as sp
from core.strategy.features import compute_features

logger = logging.getLogger(__name__)

# The feed column is named tick_volume; the spec writes "volume".
_BAR_ALIASES: dict[str, str] = {"volume": "tick_volume"}


@dataclass
class Signals:
    """Signals aligned to the bar index, plus what generated them."""

    long: pd.Series
    short: pd.Series
    indicators: dict[str, pd.Series] = field(default_factory=dict)
    features: pd.DataFrame = field(default_factory=pd.DataFrame)
    exit_signal: pd.Series | None = None

    @property
    def counts(self) -> dict[str, int]:
        return {"long": int(self.long.sum()), "short": int(self.short.sum())}


def compute_indicators(
    strategy: sp.StrategySpec, bars: pd.DataFrame
) -> dict[str, pd.Series]:
    """Computes every indicator of the spec, keyed by `ref`.

    Multi-output indicators land in the dictionary as `id.output`.
    """
    out: dict[str, pd.Series] = {}
    for indicator in strategy.indicators:
        values = indicator.definition.compute(bars, indicator.params)
        if isinstance(values, pd.DataFrame):
            for column in values.columns:
                out[f"{indicator.id}.{column}"] = values[column]
        else:
            out[indicator.id] = values
    return out


class _Context:
    """Already-computed series available to the operands."""

    def __init__(
        self,
        bars: pd.DataFrame,
        indicators: dict[str, pd.Series],
        features: pd.DataFrame,
    ) -> None:
        self.bars = bars
        self.indicators = indicators
        self.features = features
        self.index = bars.index

    def resolve(self, operand: sp.Operand) -> pd.Series:
        if isinstance(operand, sp.ConstOperand):
            return pd.Series(operand.const, index=self.index, dtype="float64")
        if isinstance(operand, sp.BarOperand):
            column = _BAR_ALIASES.get(operand.bar, operand.bar)
            return self.bars[column].astype("float64")
        if isinstance(operand, sp.FeatureOperand):
            return self.features[operand.feature]
        if isinstance(operand, sp.RefOperand):
            try:
                return self.indicators[operand.ref]
            except KeyError:  # pragma: no cover - prevented by spec validation
                raise KeyError(f"indicator not computed: {operand.ref}") from None
        raise TypeError(f"unrecognized operand: {type(operand).__name__}")


def _false(index: pd.Index) -> pd.Series:
    return pd.Series(False, index=index, dtype="bool")


def _evaluate(condition: sp.Condition, context: _Context) -> pd.Series:
    if isinstance(condition, sp.AndOr):
        parts = [_evaluate(child, context) for child in condition.operands]
        combined = parts[0]
        for part in parts[1:]:
            combined = combined & part if condition.op == "and" else combined | part
        return combined

    if isinstance(condition, sp.Not):
        return ~_evaluate(condition.operand, context)

    if isinstance(condition, sp.Compare):
        left = context.resolve(condition.left)
        right = context.resolve(condition.right)
        return _compare(condition.op, left, right)

    if isinstance(condition, sp.Between):
        value = context.resolve(condition.left)
        low = context.resolve(condition.low)
        high = context.resolve(condition.high)
        return _fill((value >= low) & (value <= high), value, low, high)

    if isinstance(condition, sp.Trend):
        series = context.resolve(condition.operand)
        fn = f.rising if condition.op == "rising" else f.falling
        return fn(series, condition.periods)

    raise TypeError(f"unrecognized condition: {type(condition).__name__}")


def _fill(result: pd.Series, *inputs: pd.Series) -> pd.Series:
    """A NaN input means 'condition not evaluable', therefore False."""
    valid = inputs[0].notna()
    for series in inputs[1:]:
        valid &= series.notna()
    return (result & valid).fillna(False).astype("bool")


def _compare(op: str, left: pd.Series, right: pd.Series) -> pd.Series:
    if op == "gt":
        return _fill(left > right, left, right)
    if op == "gte":
        return _fill(left >= right, left, right)
    if op == "lt":
        return _fill(left < right, left, right)
    if op == "lte":
        return _fill(left <= right, left, right)
    if op == "eq":
        # exact float equality is almost always a bug: compare up to the
        # representation error instead
        equal = pd.Series(
            np.isclose(left.to_numpy(), right.to_numpy(), rtol=1e-9, atol=0.0),
            index=left.index,
        )
        return _fill(equal, left, right)
    if op == "cross_above":
        return f.crossed_above(left, right)
    if op == "cross_below":
        return f.crossed_below(left, right)
    raise ValueError(f"unrecognized comparison operator: {op}")


def evaluate(strategy: sp.StrategySpec, bars: pd.DataFrame, point: float) -> Signals:
    """From spec + bars to boolean long/short signals aligned to the index.

    `point` comes from the instrument's SymbolSpec: it feeds the features
    expressed in points and must never be a constant in the code.
    """
    if bars.empty:
        empty = pd.Series(dtype="bool")
        return Signals(long=empty, short=empty)

    indicators = compute_indicators(strategy, bars)
    features = compute_features(bars, point)
    context = _Context(bars, indicators, features)

    long = (
        _evaluate(strategy.entry.long, context)
        if strategy.entry.long is not None
        else _false(bars.index)
    )
    short = (
        _evaluate(strategy.entry.short, context)
        if strategy.entry.short is not None
        else _false(bars.index)
    )
    exit_signal = (
        _evaluate(strategy.exit.signal_exit, context)
        if strategy.exit.signal_exit is not None
        else None
    )

    both = int((long & short).sum())
    if both:
        logger.warning(
            "%d bars with simultaneous long and short signals: they will be ignored", both
        )

    return Signals(
        long=long.rename("long"),
        short=short.rename("short"),
        indicators=indicators,
        features=features,
        exit_signal=exit_signal,
    )
