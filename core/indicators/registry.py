"""Name -> indicator registry.

The JSON spec references indicators by string: this is the single place where
that name becomes a function. Every entry also declares its parameter model,
so a spec with wrong parameters is rejected at load time and not mid-backtest.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from core.data.provider import MIN_SPREAD_M1_COLUMN
from core.indicators import functions as f

PriceSource = Literal[
    "open",
    "high",
    "low",
    "close",
    "tick_volume",
    "spread",
    "min_spread_m1",
    "hl2",
    "hlc3",
    "ohlc4",
]


def resolve_source(bars: pd.DataFrame, source: PriceSource) -> pd.Series:
    """Price series to run a single-series indicator on."""
    if source == "hl2":
        return (bars["high"] + bars["low"]) / 2.0
    if source == "hlc3":
        return (bars["high"] + bars["low"] + bars["close"]) / 3.0
    if source == "ohlc4":
        return (bars["open"] + bars["high"] + bars["low"] + bars["close"]) / 4.0
    if source not in bars.columns:
        raise KeyError(missing_column_reason(bars, source))
    return bars[source]


def missing_column_reason(bars: pd.DataFrame, column: str) -> str:
    """Why a bar field is not there, said in terms of what to do about it."""
    if column == "spread" and MIN_SPREAD_M1_COLUMN in bars.columns:
        return (
            f"these bars carry no {column!r}: above M1 the broker's field is "
            f"{MIN_SPREAD_M1_COLUMN!r}, the minimum of the M1 spreads inside "
            f"each bar. Reference it by that name if that is what you mean, or "
            f"run with spread_mode 'per_bar', which rebuilds a real spread from "
            f"the M1 sample of the same period"
        )
    return (
        f"these bars carry no {column!r} column. Available: "
        f"{', '.join(map(str, bars.columns))}"
    )


class _Params(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SourceParams(_Params):
    source: PriceSource = "close"


class PeriodParams(SourceParams):
    period: int = Field(default=14, ge=1)


class BollingerParams(PeriodParams):
    period: int = Field(default=20, ge=2)
    deviations: float = Field(default=2.0, gt=0)


class MacdParams(SourceParams):
    fast: int = Field(default=12, ge=1)
    slow: int = Field(default=26, ge=2)
    signal: int = Field(default=9, ge=1)


class AtrParams(_Params):
    period: int = Field(default=14, ge=1)


class StochParams(_Params):
    period: int = Field(default=14, ge=1)
    smooth_k: int = Field(default=3, ge=1)
    smooth_d: int = Field(default=3, ge=1)


class DonchianParams(_Params):
    period: int = Field(default=20, ge=1)


@dataclass(frozen=True)
class IndicatorDef:
    """How an indicator is computed and what it returns."""

    name: str
    fn: Callable[..., pd.Series | pd.DataFrame]
    params_model: type[_Params]
    # required OHLC columns; empty = single-series indicator over `source`
    bar_inputs: tuple[str, ...] = ()
    # names of the multiple outputs; empty = a single Series
    outputs: tuple[str, ...] = ()

    @property
    def is_multi_output(self) -> bool:
        return bool(self.outputs)

    def compute(self, bars: pd.DataFrame, params: Mapping[str, object]) -> pd.Series | pd.DataFrame:
        validated = self.params_model(**dict(params)).model_dump()
        if self.bar_inputs:
            args = [bars[column] for column in self.bar_inputs]
        else:
            source = validated.pop("source")
            args = [resolve_source(bars, source)]
        validated.pop("source", None)
        return self.fn(*args, **validated)


_REGISTRY: dict[str, IndicatorDef] = {}


def register(definition: IndicatorDef) -> None:
    if definition.name in _REGISTRY:
        raise ValueError(f"indicator already registered: {definition.name}")
    _REGISTRY[definition.name] = definition


def get(name: str) -> IndicatorDef:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"unknown indicator: {name!r}. Available: {', '.join(available())}"
        ) from None


def available() -> list[str]:
    return sorted(_REGISTRY)


for _definition in (
    IndicatorDef("sma", f.sma, PeriodParams),
    IndicatorDef("ema", f.ema, PeriodParams),
    IndicatorDef("rsi", f.rsi, PeriodParams),
    IndicatorDef("roc", f.roc, PeriodParams),
    IndicatorDef(
        "bollinger", f.bollinger, BollingerParams, outputs=("middle", "upper", "lower", "width")
    ),
    IndicatorDef("macd", f.macd, MacdParams, outputs=("macd", "signal", "histogram")),
    IndicatorDef("atr", f.atr, AtrParams, bar_inputs=("high", "low", "close")),
    IndicatorDef(
        "stoch", f.stoch, StochParams, bar_inputs=("high", "low", "close"), outputs=("k", "d")
    ),
    IndicatorDef(
        "donchian",
        f.donchian,
        DonchianParams,
        bar_inputs=("high", "low"),
        outputs=("upper", "lower", "middle"),
    ),
):
    register(_definition)
