"""Broker-independent data interface.

No reference to MetaTrader here: the concrete implementations live in the
`*_provider.py` modules. The contract on returned DataFrames is part of the
interface and binding for every future provider.
"""
from __future__ import annotations

import abc
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from types import TracebackType
from typing import Any

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

# The name a bar's spread field is allowed to carry, per timeframe.
#
# On M1 the field is a spread. Above M1 the terminal reports the **minimum**
# of the spreads of the M1 bars inside the period - measured at 100% on three
# instruments over more than four thousand periods each (`core.data.spread`).
# Two different quantities must not share one name: a column called `spread`
# gets charged as a fill cost sooner or later, and above M1 that charges the
# best price of the bar. So above M1 the raw column is called what it is, and
# a `spread` column at those timeframes exists only when something honest put
# it there (`core.data.spread.reconstruct_from_m1`).
SPREAD_COLUMN = "spread"
MIN_SPREAD_M1_COLUMN = "min_spread_m1"


def spread_column_for(timeframe: Timeframe | str) -> str:
    """The name the feed's spread field carries at this timeframe."""
    return (
        SPREAD_COLUMN
        if Timeframe.parse(timeframe).minutes <= 1
        else MIN_SPREAD_M1_COLUMN
    )


def rename_aggregated_spread(
    frame: pd.DataFrame, timeframe: Timeframe | str
) -> pd.DataFrame:
    """Gives the raw spread field its honest name for `timeframe`.

    Applied wherever bars enter the engine. Idempotent, and a no-op on M1.
    """
    target = spread_column_for(timeframe)
    if target == SPREAD_COLUMN or SPREAD_COLUMN not in frame.columns:
        return frame
    if target in frame.columns:
        # both names present: the `spread` column was put there deliberately
        # (a reconstruction), and overwriting it with the raw minimum would be
        # exactly the substitution this rename exists to prevent
        return frame
    return frame.rename(columns={SPREAD_COLUMN: target})


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
    def parse(cls, value: str | Timeframe) -> Timeframe:
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
    def from_dict(cls, payload: dict[str, object]) -> SymbolSpecSnapshot:
        return cls(
            spec=SymbolSpec(**payload["spec"]),
            read_at=datetime.fromisoformat(payload["read_at"]),
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

    def connect(self) -> None:  # noqa: B027 - optional hook, not abstract
        """Opens the connection. Stateless implementations do nothing."""

    def disconnect(self) -> None:  # noqa: B027 - optional hook, not abstract
        """Closes the connection."""

    def __enter__(self) -> DataProvider:
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

    # Optional: how far back the source goes. Not every provider can answer
    # without downloading everything, so the base raises instead of guessing.
    def probe_depth(self, symbol: str, timeframe: Timeframe | str) -> Any:
        raise NotImplementedError(
            f"{type(self).__name__} cannot report its history depth"
        )


def bar_columns_for(timeframe: Timeframe | str) -> tuple[str, ...]:
    """`BAR_COLUMNS` with the spread field named honestly for this timeframe."""
    name = spread_column_for(timeframe)
    return tuple(name if c == SPREAD_COLUMN else c for c in BAR_COLUMNS)


def empty_frame(columns: Sequence[str]) -> pd.DataFrame:
    index = pd.DatetimeIndex([], tz="UTC", name="time")
    return pd.DataFrame({c: pd.Series(dtype="float64") for c in columns}, index=index)


def empty_bars(timeframe: Timeframe | str | None = None) -> pd.DataFrame:
    """Empty bars DataFrame, still conforming to the contract."""
    return empty_frame(BAR_COLUMNS if timeframe is None else bar_columns_for(timeframe))


def empty_ticks() -> pd.DataFrame:
    return empty_frame(TICK_COLUMNS)


def normalize_bars(frame: pd.DataFrame, columns: Sequence[str] = BAR_COLUMNS) -> pd.DataFrame:
    """Sorts, deduplicates and enforces the common schema on a UTC frame.

    A column the caller did not ask for is dropped, and one it asked for and
    the frame lacks is filled with zero. Both are dangerous around the spread
    field - a dropped `min_spread_m1` silently becomes a `spread` of zero,
    which is a free trade - so callers handling bars above M1 must pass
    `bar_columns_for(timeframe)` rather than the default.
    """
    if frame.empty:
        return empty_frame(columns)
    out = frame.sort_index()
    out = out[~out.index.duplicated(keep="last")]
    out.index.name = "time"
    missing = [c for c in columns if c not in out.columns]
    for name in missing:
        out[name] = 0.0
    return out[list(columns)]
