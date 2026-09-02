"""Name -> indicator registry.

The JSON spec references indicators by string: this is the single place where
that name becomes a function. Every entry also declares its parameter model,
so a spec with wrong parameters is rejected at load time and not mid-backtest.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Mapping

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from core.indicators import functions as f

PriceSource = Literal[
    "open", "high", "low", "close", "tick_volume", "spread", "hl2", "hlc3", "ohlc4"
]


def resolve_source(bars: pd.DataFrame, source: PriceSource) -> pd.Series:
    """Price series to run a single-series indicator on."""
    if source == "hl2":
        return (bars["high"] + bars["low"]) / 2.0
    if source == "hlc3":
        return (bars["high"] + bars["low"] + bars["close"]) / 3.0
    if source == "ohlc4":
        return (bars["open"] + bars["high"] + bars["low"] + bars["close"]) / 4.0
    return bars[source]


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
            args = [resolve_source(bars, source)]  # type: ignore[arg-type]
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
