"""Broker-independent data interface.

No reference to MetaTrader here: the concrete implementations live in the
`*_provider.py` modules. The contract on returned DataFrames is part of the
interface and binding for every future provider.
"""
from __future__ import annotations

import abc
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from types import TracebackType
from typing import Sequence

import pandas as pd

BAR_COLUMNS: tuple[str, ...] = (
    "open",
    "high",
    "low",
    "close",
    "tick_volume",
    "spread",
    "real_volume",
)

TICK_COLUMNS: tuple[str, ...] = ("bid", "ask", "last", "volume")


class Timeframe(Enum):
    """Generic timeframe, measured in minutes."""

    M1 = 1
    M2 = 2
    M3 = 3
    M4 = 4
    M5 = 5
    M6 = 6
    M10 = 10
    M12 = 12
    M15 = 15
    M20 = 20
    M30 = 30
    H1 = 60
    H2 = 120
    H3 = 180
    H4 = 240
    H6 = 360
    H8 = 480
    H12 = 720
    D1 = 1440
    W1 = 10080

    @property
    def minutes(self) -> int:
        return self.value

    @property
    def pandas_freq(self) -> str:
        return f"{self.value}min"

    @classmethod
    def parse(cls, value: "str | Timeframe") -> "Timeframe":
        if isinstance(value, cls):
            return value
        try:
            return cls[str(value).upper()]
        except KeyError as exc:
            raise ValueError(f"unknown timeframe: {value!r}") from exc


@dataclass(frozen=True)
class SymbolSpec:
    """Contract specification of an instrument, normalized across providers."""

    name: str
    point: float
    digits: int
    contract_size: float
    tick_value: float
    tick_size: float
    volume_min: float
    volume_max: float
    volume_step: float
    swap_long: float
    swap_short: float
    currency_profit: str
    trade_mode: str


# Fields that change what a trade costs. A drift here changes results silently
# (the broker moves swap rates, tick_value tracks an FX pair) and must be part
# of a run's identity; digits/currency_profit/trade_mode/name are descriptive
# and left out on purpose.
SYMBOL_SPEC_COST_FIELDS: tuple[str, ...] = (
    "point",
    "digits",
    "contract_size",
    "tick_value",
    "tick_size",
    "swap_long",
    "swap_short",
    "volume_min",
    "volume_max",
    "volume_step",
)


@dataclass(frozen=True)
class SymbolSpecSnapshot:
    """A `SymbolSpec` together with when it was read from the broker.

    `tick_value` moves with FX rates and swap rates are changed by the broker
    without notice: two reads of "the same" instrument hours apart can differ
    in the numbers that decide the result. Persisting the spec without the
    read timestamp makes that drift invisible after the fact.
    """

    spec: SymbolSpec
    read_at: datetime

    def to_dict(self) -> dict[str, object]:
        return {"read_at": self.read_at.isoformat(), "spec": asdict(self.spec)}

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "SymbolSpecSnapshot":
        return cls(
            spec=SymbolSpec(**payload["spec"]),  # type: ignore[arg-type]
            read_at=datetime.fromisoformat(payload["read_at"]),  # type: ignore[arg-type]
        )


class DataProvider(abc.ABC):
    """Source of historical market data.

    Contract on returned values:

    - `get_bars` returns a DataFrame with a tz-aware **UTC** `DatetimeIndex`,
      index name `time`, sorted and without duplicates, with the
      `BAR_COLUMNS` columns. The `spread` column is in points and is the real
      transaction cost: it must not be discarded.
    - `get_ticks` returns a DataFrame with the same index type and the
      `TICK_COLUMNS` columns.
    - incoming `start`/`end` are interpreted as UTC when naive.
    """

    def connect(self) -> None:
        """Opens the connection. Stateless implementations do nothing."""

    def disconnect(self) -> None:
        """Closes the connection."""

    def __enter__(self) -> "DataProvider":
        self.connect()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.disconnect()

    @abc.abstractmethod
    def list_symbols(self) -> list[SymbolSpec]:
        ...

    @abc.abstractmethod
    def get_symbol_spec(self, symbol: str) -> SymbolSpec:
        ...

    @abc.abstractmethod
    def get_bars(
        self,
        symbol: str,
        timeframe: Timeframe | str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        ...

    @abc.abstractmethod
    def get_ticks(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        ...


def empty_bars() -> pd.DataFrame:
    """Empty bars DataFrame, still conforming to the contract."""
    index = pd.DatetimeIndex([], tz="UTC", name="time")
    return pd.DataFrame({c: pd.Series(dtype="float64") for c in BAR_COLUMNS}, index=index)


def empty_ticks() -> pd.DataFrame:
    index = pd.DatetimeIndex([], tz="UTC", name="time")
    return pd.DataFrame({c: pd.Series(dtype="float64") for c in TICK_COLUMNS}, index=index)


def normalize_bars(frame: pd.DataFrame, columns: Sequence[str] = BAR_COLUMNS) -> pd.DataFrame:
    """Sorts, deduplicates and enforces the common schema on a UTC frame."""
    if frame.empty:
        return empty_bars() if tuple(columns) == BAR_COLUMNS else empty_ticks()
    out = frame.sort_index()
    out = out[~out.index.duplicated(keep="last")]
    out.index.name = "time"
    missing = [c for c in columns if c not in out.columns]
    for name in missing:
        out[name] = 0.0
    return out[list(columns)]
