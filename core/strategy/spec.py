"""Pydantic models of the strategy spec.

A spec is a JSON file: no strategy parameter may exist in the code. Here that
JSON is checked for sanity **before** launching a backtest, because a wrong
indicator id discovered mid-run costs an hour and a results table to throw
away.
"""
from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from core.data.provider import Timeframe
from core.indicators import registry

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

# `spread` is a real spread on M1 and, above it, a column that exists only
# where it was rebuilt from the M1 sample; `min_spread_m1` is the broker's
# raw field above M1, which is the minimum of those spreads and not a cost.
BarField = Literal[
    "open", "high", "low", "close", "volume", "spread", "min_spread_m1"
]
FeatureName = Literal[
    "lower_wick_ratio",
    "upper_wick_ratio",
    "body_ratio",
    "range_points",
    "close_position_in_range",
]


class SpecError(ValueError):
    """Invalid spec, with the detail of where and why."""


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


# -- operands ------------------------------------------------------------


class RefOperand(_Model):
    """Reference to an indicator: `id` or `id.output`."""

    ref: str

    @property
    def indicator_id(self) -> str:
        return self.ref.split(".", 1)[0]

    @property
    def output(self) -> str | None:
        parts = self.ref.split(".", 1)
        return parts[1] if len(parts) == 2 else None


class ConstOperand(_Model):
    const: float


class BarOperand(_Model):
    bar: BarField


class FeatureOperand(_Model):
    feature: FeatureName


Operand = Annotated[
    RefOperand | ConstOperand | BarOperand | FeatureOperand,
    Field(union_mode="left_to_right"),
]


# -- conditions ----------------------------------------------------------


class AndOr(_Model):
    op: Literal["and", "or"]
    operands: list[Condition] = Field(min_length=1)


class Not(_Model):
    op: Literal["not"]
    operand: Condition


class Compare(_Model):
    op: Literal["gt", "gte", "lt", "lte", "eq", "cross_above", "cross_below"]
    left: Operand
    right: Operand


class Between(_Model):
    op: Literal["between"]
    left: Operand
    low: Operand
    high: Operand


class Trend(_Model):
    op: Literal["rising", "falling"]
    operand: Operand
    periods: int = Field(default=1, ge=1)


Condition = Annotated[
    AndOr | Not | Compare | Between | Trend,
    Field(discriminator="op"),
]


def walk_operands(condition: Condition) -> Iterator[Operand]:
    """All leaf operands of a condition tree."""
    if isinstance(condition, AndOr):
        for child in condition.operands:
            yield from walk_operands(child)
    elif isinstance(condition, Not):
        yield from walk_operands(condition.operand)
    elif isinstance(condition, Compare):
        yield condition.left
        yield condition.right
    elif isinstance(condition, Between):
        yield condition.left
        yield condition.low
        yield condition.high
    elif isinstance(condition, Trend):
        yield condition.operand


# -- spec blocks ---------------------------------------------------------


class Instrument(_Model):
    symbol: str = Field(min_length=1)
    timeframe: str

    @model_validator(mode="after")
    def _check_timeframe(self) -> Instrument:
        Timeframe.parse(self.timeframe)
        return self

    @property
    def tf(self) -> Timeframe:
        return Timeframe.parse(self.timeframe)


class IndicatorSpec(_Model):
    id: str = Field(min_length=1)
    type: str
    params: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_type_and_params(self) -> IndicatorSpec:
        try:
            definition = registry.get(self.type)
        except KeyError as exc:
            # pydantic only converts ValueError: without this the caller would
            # receive a raw KeyError instead of the spec summary
            raise ValueError(str(exc.args[0])) from None
        # the registry model rejects unknown or inconsistent parameters
        definition.params_model(**self.params)
        return self

    @property
    def definition(self) -> registry.IndicatorDef:
        return registry.get(self.type)


class Entry(_Model):
    long: Condition | None = None
    short: Condition | None = None

    @model_validator(mode="after")
    def _at_least_one_side(self) -> Entry:
        if self.long is None and self.short is None:
            raise ValueError("entry: at least one of 'long' and 'short' is required")
        return self


class PointsLevel(_Model):
    """Stop or target distance, in instrument points. Fixed across instruments."""

    type: Literal["points"]
    value: float = Field(gt=0)


class PercentLevel(_Model):
    """Stop or target distance as a percentage of the entry price.

    `value` is a percentage (0.15 means 0.15%), applied to the actual fill
    price at entry - unlike the ATR level below, the entry price needs no
    freezing: it exists exactly once, at the moment the trade opens.
    """

    type: Literal["percent"]
    value: float = Field(gt=0)


class AtrLevel(_Model):
    """Stop or target distance as a multiple of an ATR indicator's value.

    `indicator` must reference an `atr`-typed entry in the spec's
    `indicators` list. The value used is the one at the signal bar (the last
    closed bar when the trade was decided), frozen for the whole trade: this
    is a fixed stop/target sized by volatility at entry, not a trailing stop,
    which would require recomputing it bar by bar and is out of scope here.
    """

    type: Literal["atr"]
    indicator: str = Field(min_length=1)
    mult: float = Field(gt=0)


Level = Annotated[
    PointsLevel | PercentLevel | AtrLevel,
    Field(discriminator="type"),
]


class TimeStop(_Model):
    bars: int = Field(gt=0)


class Exit(_Model):
    stop_loss: Level | None = None
    take_profit: Level | None = None
    time_stop: TimeStop | None = None
    signal_exit: Condition | None = None

    @model_validator(mode="after")
    def _needs_a_way_out(self) -> Exit:
        if not any((self.stop_loss, self.take_profit, self.time_stop, self.signal_exit)):
            raise ValueError(
                "exit: without stop_loss, take_profit, time_stop or signal_exit "
                "a position never closes"
            )
        return self


class Sizing(_Model):
    type: Literal["equity_per_step"]
    equity_per_001_lot: float = Field(gt=0)
    min_lot: float = Field(gt=0)
    max_lot: float = Field(gt=0)

    @model_validator(mode="after")
    def _ordered(self) -> Sizing:
        if self.max_lot < self.min_lot:
            raise ValueError(f"sizing: max_lot ({self.max_lot}) < min_lot ({self.min_lot})")
        return self


class Session(_Model):
    """Time window in which opening is allowed, on the given clock."""

    start: str
    end: str
    timezone: Literal["server", "utc"] = "server"

    @model_validator(mode="after")
    def _parse_times(self) -> Session:
        for value in (self.start, self.end):
            hours, _, minutes = value.partition(":")
            if not (hours.isdigit() and minutes.isdigit()):
                raise ValueError(f"session: invalid time {value!r}, expected 'HH:MM'")
            if not (0 <= int(hours) <= 23 and 0 <= int(minutes) <= 59):
                raise ValueError(f"session: time out of range {value!r}")
        return self


class Risk(_Model):
    max_open_positions: int = Field(default=1, ge=1)
    cooldown_minutes: int = Field(default=0, ge=0)
    max_trades_per_day: int | None = Field(default=None, ge=1)
    max_spread_points: float | None = Field(default=None, gt=0)
    session: Session | None = None
    news_filter: None = None

    @model_validator(mode="after")
    def _news_not_implemented(self) -> Risk:
        if self.news_filter is not None:
            raise ValueError(
                "risk.news_filter: not implemented in this phase, must be null"
            )
        return self


class StrategySpec(_Model):
    schema_version: int
    id: str = Field(min_length=1)
    name: str
    description: str = ""
    instrument: Instrument
    indicators: list[IndicatorSpec] = Field(default_factory=list)
    entry: Entry
    exit: Exit
    sizing: Sizing
    risk: Risk = Field(default_factory=Risk)

    @model_validator(mode="after")
    def _cross_checks(self) -> StrategySpec:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"schema_version {self.schema_version} not supported "
                f"(expected {SCHEMA_VERSION})"
            )

        seen: set[str] = set()
        for indicator in self.indicators:
            if indicator.id in seen:
                raise ValueError(f"indicators: duplicate id {indicator.id!r}")
            seen.add(indicator.id)

        by_id = {i.id: i for i in self.indicators}
        referenced: set[str] = set()
        for condition in self.conditions():
            for operand in walk_operands(condition):
                if not isinstance(operand, RefOperand):
                    continue
                target = by_id.get(operand.indicator_id)
                if target is None:
                    known = ", ".join(sorted(by_id)) or "none"
                    raise ValueError(
                        f"ref {operand.ref!r}: no indicator with id "
                        f"{operand.indicator_id!r}. Defined: {known}"
                    )
                referenced.add(operand.indicator_id)
                outputs = target.definition.outputs
                if outputs and operand.output is None:
                    raise ValueError(
                        f"ref {operand.ref!r}: {target.type} has multiple outputs, "
                        f"name one of {', '.join(outputs)} "
                        f"(e.g. {operand.indicator_id}.{outputs[0]})"
                    )
                if operand.output is not None:
                    if not outputs:
                        raise ValueError(
                            f"ref {operand.ref!r}: {target.type} has a single output, "
                            f"use {operand.indicator_id!r} without a suffix"
                        )
                    if operand.output not in outputs:
                        raise ValueError(
                            f"ref {operand.ref!r}: output {operand.output!r} "
                            f"does not exist. Available: {', '.join(outputs)}"
                        )

        for level, where in (
            (self.exit.stop_loss, "exit.stop_loss"),
            (self.exit.take_profit, "exit.take_profit"),
        ):
            if not isinstance(level, AtrLevel):
                continue
            target = by_id.get(level.indicator)
            if target is None:
                known = ", ".join(sorted(by_id)) or "none"
                raise ValueError(
                    f"{where}: indicator {level.indicator!r} not found. "
                    f"Defined: {known}"
                )
            if target.type != "atr":
                raise ValueError(
                    f"{where}: indicator {level.indicator!r} is of type "
                    f"{target.type!r}, must be 'atr'"
                )
            referenced.add(level.indicator)

        for unused in sorted(seen - referenced):
            logger.warning(
                "indicator %r declared but never referenced: it is computed "
                "and ignored",
                unused,
            )
        return self

    def conditions(self) -> list[Condition]:
        found = [self.entry.long, self.entry.short, self.exit.signal_exit]
        return [c for c in found if c is not None]

    # -- serialization ---------------------------------------------------

    @classmethod
    def from_dict(cls, payload: dict[str, Any], origin: str = "<dict>") -> StrategySpec:
        try:
            return cls.model_validate(payload)
        except ValidationError as exc:
            raise SpecError(_format_errors(exc, origin)) from exc

    @classmethod
    def from_json(cls, source: str | Path) -> StrategySpec:
        """Loads from a file path or a JSON string."""
        path = Path(source) if not str(source).lstrip().startswith("{") else None
        if path is not None:
            text = path.read_text(encoding="utf-8")
            origin = str(path)
        else:
            text = str(source)
            origin = "<json string>"
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise SpecError(f"{origin}: invalid JSON at line {exc.lineno}: {exc.msg}") from exc
        return cls.from_dict(payload, origin)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(
            self.model_dump(mode="json", exclude_none=False), indent=indent, ensure_ascii=False
        )

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_json() + "\n", encoding="utf-8")
        return target


def _format_errors(exc: ValidationError, origin: str) -> str:
    lines = [f"invalid spec ({origin}): {exc.error_count()} problem(s)"]
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"]) or "<root>"
        lines.append(f"  - {location}: {error['msg']}")
    return "\n".join(lines)


AndOr.model_rebuild()
Not.model_rebuild()
