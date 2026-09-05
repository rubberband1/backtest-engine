"""A data source that needs no broker: deterministic synthetic bars.

Every other provider in this package talks to a terminal. This one talks to
nothing, and exists so that the repository is runnable by someone who has
never installed MetaTrader 5 - the tests, the demo, the API and the UI all
run against it. Without it the project is a description of a program rather
than a program.

**The data is invented, and nothing measured on it means anything.** It is
not anonymized market data, not a resampled real series, and not a
"realistic" reconstruction of any instrument: it is a random walk with
plausible microstructure bolted on. A backtest over it will report a Sharpe
ratio, a drawdown and a p-value, all of which describe a random number
generator. The provenance written into the cache says `synthetic_fixture` at
every level, so a run over this data cannot be mistaken, later or by someone
else, for a run over a market.

Why synthetic rather than a sample of real bars: redistributing a broker's
price history is a licensing question with no clear answer, and the honest
resolution of an unclear licence is not to redistribute. Inventing the data
removes the question, and for the purpose - proving the engine runs and that
its parts agree with each other - invented data is as good as real data,
because none of it is being used to make a claim about a market.

What is deliberately kept realistic, because the engine has logic that has to
be exercised by it:

- **The week has a shape.** The market opens Sunday 21:00 UTC and closes
  Friday 21:00 UTC, so `SessionCalendar` has something to infer, weekend gaps
  are real gaps, and a swap crosses a real midnight.
- **The spread moves.** It widens in the thin hours and narrows in the
  London/New York overlap, so a per-bar spread policy measured off M1 differs
  from a fixed one - which is the difference the cost model is built around.
- **Every timeframe comes from one M1 path.** H1, H4 and D1 are aggregations
  of the same minute series, not independent walks, so a bar's high really is
  the high of the minutes inside it and cross-timeframe checks mean something.
- **OHLC is internally consistent** on every bar: low <= min(open, close) and
  high >= max(open, close).

Determinism is part of the contract: the same symbol over the same period
produces the same bars on any machine and on any run, because the generator
is seeded from the symbol name and indexes into a fixed epoch. A fixture that
drifted would make the golden tests meaningless.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from core.data.provider import (
    DataProvider,
    SymbolSpec,
    Timeframe,
    bar_columns_for,
    empty_bars,
    empty_ticks,
    normalize_bars,
    rename_aggregated_spread,
)

if TYPE_CHECKING:
    from core.data.depth import DepthProbe

logger = logging.getLogger(__name__)

SOURCE_NAME = "synthetic_fixture"

# The clock the fixture pretends to run on. Europe/Athens is what the real
# broker used, and keeping it means the session and DST logic is exercised by
# the fixture exactly as it is in production.
SERVER_TIMEZONE = "Europe/Athens"

# The market week, in UTC: Sunday 21:00 -> Friday 21:00.
WEEK_OPEN_HOUR = 21
WEEK_CLOSE_HOUR = 21

# The fixture spans a fixed two years and stops. The span is fixed rather
# than open-ended because the whole minute series is generated in one piece
# and then sliced: a series whose length depended on what was asked for would
# consume its random stream differently on every request, and the same
# timestamp would come back with a different price depending on the width of
# the window around it. That bug is easy to write and hard to see, so the
# horizon is a constant and `get_bars` clamps to it.
EPOCH = datetime(2022, 1, 1, tzinfo=timezone.utc)
HORIZON = datetime(2024, 1, 1, tzinfo=timezone.utc)


@dataclass(frozen=True)
class FixtureInstrument:
    """One invented instrument: its contract, and how its price behaves."""

    spec: SymbolSpec
    start_price: float
    volatility: float          # annualized, as a fraction of price
    spread_floor: float        # points, at its narrowest
    spread_peak: float         # points, in the dead hours


def _spec(
    name: str,
    digits: int,
    contract_size: float,
    tick_value: float,
    swap_long: float,
    swap_short: float,
) -> SymbolSpec:
    point = 10.0 ** (-digits)
    return SymbolSpec(
        name=name,
        point=point,
        digits=digits,
        contract_size=contract_size,
        tick_value=tick_value,
        tick_size=point,
        volume_min=0.01,
        volume_max=20.0,
        volume_step=0.01,
        swap_long=swap_long,
        swap_short=swap_short,
        currency_profit="USD",
        trade_mode="full",
    )


# Two instruments, shaped differently on purpose: one with a large point
# value and a wide spread, one with a small point value and a tight spread. A
# cost model only ever exercised on one of those is not exercised.
INSTRUMENTS: dict[str, FixtureInstrument] = {
    "SYNTHGOLD": FixtureInstrument(
        spec=_spec("SYNTHGOLD", 2, 100.0, 1.0, -19.5, 8.4),
        start_price=1800.0,
        volatility=0.16,
        spread_floor=6.0,
        spread_peak=22.0,
    ),
    "SYNTHFX": FixtureInstrument(
        spec=_spec("SYNTHFX", 5, 100_000.0, 1.0, -7.2, 1.1),
        start_price=1.1000,
        volatility=0.08,
        spread_floor=4.0,
        spread_peak=26.0,
    ),
}


def market_is_open(index: pd.DatetimeIndex) -> np.ndarray:
    """The FX week: Sunday 21:00 UTC to Friday 21:00 UTC."""
    weekday = np.asarray(index.weekday)
    hour = np.asarray(index.hour)
    closed = (
        (weekday == 5)
        | ((weekday == 4) & (hour >= WEEK_CLOSE_HOUR))
        | ((weekday == 6) & (hour < WEEK_OPEN_HOUR))
    )
    return ~closed


def _seed(symbol: str) -> int:
    """A stable seed per symbol, so the series never moves between machines."""
    return int.from_bytes(hashlib.sha256(symbol.encode("utf-8")).digest()[:4], "big")


def _liquidity(index: pd.DatetimeIndex) -> np.ndarray:
    """How busy each minute is, 0.15 (dead) to 1 (London/New York overlap).

    Drives both the spread and the volume, which is why the two move together
    the way they do on a real feed.
    """
    hour = np.asarray(index.hour) + np.asarray(index.minute) / 60.0
    london = np.exp(-0.5 * ((hour - 9.0) / 3.2) ** 2)
    newyork = np.exp(-0.5 * ((hour - 14.5) / 3.0) ** 2)
    return np.clip(0.15 + 0.85 * np.maximum(london, newyork), 0.0, 1.0)


@lru_cache(maxsize=4)
def _full_minutes(symbol: str) -> pd.DataFrame:
    """The whole M1 series for `symbol`, EPOCH -> HORIZON. Generated once.

    Memoized because everything else in the module slices this: the H1, H4
    and D1 bars are aggregations of it, so they cost one generation between
    them rather than one each.
    """
    instrument = INSTRUMENTS[symbol]
    full = pd.date_range(
        pd.Timestamp(EPOCH), pd.Timestamp(HORIZON), freq="1min",
        tz="UTC", name="time", inclusive="left",
    )
    index = full[market_is_open(full)]
    if len(index) == 0:
        return empty_bars(Timeframe.M1)

    rng = np.random.default_rng(_seed(symbol))
    count = len(index)

    # per-minute volatility from the annual figure: ~252 trading days of 1440
    # minutes. The liquidity profile makes the busy hours move more than the
    # dead ones, which is what makes an ATR on this data mean anything.
    busy = _liquidity(index)
    per_minute = instrument.volatility / np.sqrt(252.0 * 1440.0)
    step = rng.standard_normal(count) * per_minute * (0.45 + 1.1 * busy)

    close = instrument.start_price * np.exp(np.cumsum(step))
    open_ = np.empty(count)
    open_[0] = instrument.start_price
    open_[1:] = close[:-1]

    # the wick: a fraction of the bar's own move, never inverted
    body = np.abs(close - open_)
    reach = (body + close * per_minute * 0.6) * rng.gamma(2.0, 0.5, count)
    high = np.maximum(open_, close) + reach * rng.random(count)
    low = np.minimum(open_, close) - reach * rng.random(count)

    spread = (
        instrument.spread_floor
        + (instrument.spread_peak - instrument.spread_floor) * (1.0 - busy) ** 2
    )
    # brokers quote whole points, and the occasional spike is part of the picture
    spike = rng.random(count) < 0.004
    spread = np.maximum(np.round(spread + spike * rng.gamma(3.0, 4.0, count)), 1.0)

    volume = np.round(1.0 + busy * 140.0 * rng.gamma(2.0, 0.5, count))

    digits = instrument.spec.digits
    frame = pd.DataFrame(
        {
            "open": np.round(open_, digits),
            "high": np.round(high, digits),
            "low": np.round(low, digits),
            "close": np.round(close, digits),
            "tick_volume": volume,
            "spread": spread,
            "real_volume": np.zeros(count),
        },
        index=index,
    )
    # rounding can push a high below the body it has to contain
    frame["high"] = frame[["open", "high", "close"]].max(axis=1)
    frame["low"] = frame[["open", "low", "close"]].min(axis=1)
    return frame


def minute_bars(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    """The base M1 series for `symbol` over [start, end).

    A slice of the whole generated series, never a fresh generation of the
    window: a sub-period returns exactly the bars that period had inside any
    wider request, which is what lets H4 be checked against the H1 inside it.
    """
    frame = _full_minutes(symbol)
    window = (frame.index >= pd.Timestamp(_utc(start))) & (
        frame.index < pd.Timestamp(_utc(end))
    )
    return frame[window]


def aggregate(minutes: pd.DataFrame, timeframe: Timeframe) -> pd.DataFrame:
    """M1 bars folded up into `timeframe`, the way the terminal does it.

    The spread of the aggregated bar is the **minimum** of the minute spreads
    inside it, because that is what MetaTrader reports above M1, and this
    fixture has no business being more honest than the thing it stands in
    for. The column is renamed to `min_spread_m1` on the way out, which is
    where the rest of the engine refuses to charge it as a fill cost.
    """
    if minutes.empty:
        return empty_bars(timeframe)
    frame = (
        minutes.resample(timeframe.pandas_freq, label="left", closed="left")
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "tick_volume": "sum",
                "spread": "min",
                "real_volume": "sum",
            }
        )
        .dropna(subset=["open"])
    )
    return rename_aggregated_spread(frame, timeframe)


class FixtureProvider(DataProvider):
    """`DataProvider` over invented data. No terminal, no network, no state.

    Interchangeable with `MT5Provider` at the interface, which is the point:
    the cache, the run orchestration and the API cannot tell which one filled
    them, and the provenance recorded in the cache is what tells them apart
    afterwards.
    """

    source_name = SOURCE_NAME

    def __init__(self, server_timezone: str = SERVER_TIMEZONE) -> None:
        from zoneinfo import ZoneInfo

        self.server_timezone = ZoneInfo(server_timezone)

    def list_symbols(self) -> list[SymbolSpec]:
        return [instrument.spec for instrument in INSTRUMENTS.values()]

    def get_symbol_spec(self, symbol: str) -> SymbolSpec:
        if symbol not in INSTRUMENTS:
            raise ValueError(
                f"{symbol!r} is not one of the fixture instruments "
                f"({', '.join(INSTRUMENTS)})"
            )
        return INSTRUMENTS[symbol].spec

    def get_bars(
        self,
        symbol: str,
        timeframe: Timeframe | str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        tf = Timeframe.parse(timeframe)
        self.get_symbol_spec(symbol)
        start, end = _utc(start), _utc(end)
        if end <= start:
            return empty_bars(tf)

        # aggregation needs whole periods: ask the minute series for the
        # bucket the start falls into, not for the start itself
        floor = pd.Timestamp(start).floor(tf.pandas_freq).to_pydatetime()
        minutes = minute_bars(symbol, floor, end)
        frame = minutes if tf is Timeframe.M1 else aggregate(minutes, tf)
        frame = frame[frame.index >= pd.Timestamp(start)]
        return normalize_bars(frame, bar_columns_for(tf))

    def get_ticks(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        """Not generated: a tick series would be invention on top of invention.

        The tick resolver exists to settle bars where the stop and the target
        were both inside the range, and settling those against made-up ticks
        would give a confident answer to a question the fixture cannot
        answer. Empty is the honest return, and the resolver then reports
        those bars as unresolved - which is the correct outcome here.
        """
        logger.info("the fixture provider has no ticks: ambiguous bars stay ambiguous")
        return empty_ticks()

    def probe_depth(self, symbol: str, timeframe: Timeframe | str) -> DepthProbe:
        """How far the fixture goes, which is a fixed and knowable answer.

        Never truncated: unlike a broker, this source is not holding anything
        back. The span is the whole fixture, and a caller asking for more
        gets nothing rather than a silently shortened series.
        """
        from core.data.depth import DepthProbe

        tf = Timeframe.parse(timeframe)
        self.get_symbol_spec(symbol)
        bars = self.get_bars(symbol, tf, EPOCH, HORIZON)
        return DepthProbe(
            symbol=symbol,
            timeframe=tf.name,
            source=self.source_name,
            bars=int(len(bars)),
            first_bar=bars.index[0].to_pydatetime() if len(bars) else None,
            last_bar=bars.index[-1].to_pydatetime() if len(bars) else None,
            truncated=False,
            probed_at=datetime.now(timezone.utc),
        )


FIXTURE_CACHE = Path(__file__).resolve().parents[2] / "fixtures" / "data_cache"


def resolve_cache_dir(explicit: str | None = None) -> tuple[Path, bool]:
    """Which cache to serve, and whether it is the synthetic one.

    The order is: what the caller asked for, then a real `data_cache/` if it
    has anything in it, then the fixture. The fixture is last so that a
    machine with a terminal and downloaded bars never silently gets invented
    data - but it is there, so that a machine without one is not left with an
    application that starts and shows nothing.
    """
    if explicit:
        target = Path(explicit)
        return target, target.resolve() == FIXTURE_CACHE.resolve()

    real = Path("data_cache")
    if real.exists() and any(real.glob("*/*/*.parquet")):
        return real, False
    if FIXTURE_CACHE.exists():
        return FIXTURE_CACHE, True
    return real, False


def _utc(moment: datetime) -> datetime:
    return (
        moment.replace(tzinfo=timezone.utc)
        if moment.tzinfo is None
        else moment.astimezone(timezone.utc)
    )
