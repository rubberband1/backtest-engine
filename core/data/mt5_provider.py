"""Concrete provider on a MetaTrader 5 terminal.

Read only: this module does not import, expose or call any order-sending
function.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone, tzinfo
from types import TracebackType
from typing import Any, Iterator

import numpy as np
import pandas as pd

from core.data.provider import (
    BAR_COLUMNS,
    TICK_COLUMNS,
    DataProvider,
    SymbolSpec,
    Timeframe,
    empty_bars,
    empty_ticks,
    normalize_bars,
)
from core.data.servertime import (
    measure_offset,
    resolve_timezone,
    server_epoch_ms_to_utc_index,
    server_epoch_to_utc_index,
    utc_to_broker_datetime,
)

try:  # the package is Windows-only: the import must not break tests elsewhere
    import MetaTrader5 as mt5
except ImportError:  # pragma: no cover
    mt5 = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# Beyond this offset the "fresh tick" assumption breaks down: it means the
# market is closed and the tick is stale, not that the broker sits at UTC-40.
MAX_PLAUSIBLE_SERVER_OFFSET = timedelta(hours=14)

_TRADE_MODES: dict[int, str] = {}


class MT5Error(RuntimeError):
    """Error reported by the terminal."""


class ServerTimeError(MT5Error):
    """The server timezone cannot be derived from the available data."""


def mt5_available() -> bool:
    return mt5 is not None


def _last_error() -> str:
    if mt5 is None:
        return "MetaTrader5 not installed"
    code, text = mt5.last_error()
    return f"({code}) {text}"


def _trade_mode_name(value: int) -> str:
    if not _TRADE_MODES and mt5 is not None:
        for attr in dir(mt5):
            if attr.startswith("SYMBOL_TRADE_MODE_"):
                label = attr[len("SYMBOL_TRADE_MODE_"):].lower()
                _TRADE_MODES[getattr(mt5, attr)] = label
    return _TRADE_MODES.get(value, str(value))


class MT5Provider(DataProvider):
    """Access to the local MT5 terminal's history.

    Notable parameters:
        clock_symbol: symbol used to measure the server time. If absent, the
            first symbol with a valid tick is chosen.
        server_timezone: forces the server timezone instead of detecting it.
            Needed when the market is closed and the last tick is too old to
            measure the offset.
        max_bars_per_request: long historical requests are split to stay
            below the terminal's bar limit.
    """

    def __init__(
        self,
        *,
        path: str | None = None,
        login: int | None = None,
        password: str | None = None,
        server: str | None = None,
        clock_symbol: str | None = None,
        server_timezone: tzinfo | None = None,
        max_retries: int = 5,
        backoff_base: float = 1.0,
        backoff_max: float = 30.0,
        max_bars_per_request: int = 200_000,
        tick_chunk: timedelta = timedelta(days=1),
    ) -> None:
        self._path = path
        self._login = login
        self._password = password
        self._server = server
        self._clock_symbol = clock_symbol
        self._server_tz: tzinfo | None = server_timezone
        self._max_retries = max_retries
        self._backoff_base = backoff_base
        self._backoff_max = backoff_max
        self._max_bars_per_request = max_bars_per_request
        self._tick_chunk = tick_chunk
        self._connected = False
        self._selected: set[str] = set()

    # -- connection ------------------------------------------------------

    def connect(self) -> None:
        if self._connected:
            return
        if mt5 is None:
            raise MT5Error("MetaTrader5 package not installed (requires Windows)")

        kwargs: dict[str, Any] = {}
        if self._path:
            kwargs["path"] = self._path
        if self._login is not None:
            kwargs["login"] = self._login
        if self._password is not None:
            kwargs["password"] = self._password
        if self._server is not None:
            kwargs["server"] = self._server

        delay = self._backoff_base
        for attempt in range(1, self._max_retries + 1):
            if mt5.initialize(**kwargs):
                self._connected = True
                info = mt5.terminal_info()
                logger.info(
                    "connected to MT5 build %s (%s)",
                    getattr(info, "build", "?"),
                    getattr(info, "company", "?"),
                )
                return
            logger.warning(
                "initialize failed (attempt %d/%d): %s",
                attempt,
                self._max_retries,
                _last_error(),
            )
            mt5.shutdown()
            if attempt < self._max_retries:
                time.sleep(delay)
                delay = min(delay * 2, self._backoff_max)
        raise MT5Error(f"cannot initialize MT5: {_last_error()}")

    def disconnect(self) -> None:
        if self._connected and mt5 is not None:
            mt5.shutdown()
            self._connected = False
            self._selected.clear()
            logger.info("MT5 connection closed")

    def __enter__(self) -> "MT5Provider":
        self.connect()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.disconnect()

    def _require_connection(self) -> None:
        if not self._connected:
            self.connect()

    # -- server timezone -------------------------------------------------

    @property
    def server_timezone(self) -> tzinfo:
        """Server timezone, detected from the feed and never hardcoded."""
        if self._server_tz is None:
            self._server_tz = self._detect_server_timezone()
        return self._server_tz

    def _clock_candidates(self) -> Iterator[str]:
        if self._clock_symbol:
            yield self._clock_symbol
            return
        assert mt5 is not None
        symbols = mt5.symbols_get() or ()
        for info in symbols:
            if info.visible:
                yield info.name
        for info in symbols[:50]:
            yield info.name

    def _detect_server_timezone(self) -> tzinfo:
        self._require_connection()
        assert mt5 is not None
        for symbol in self._clock_candidates():
            self._select(symbol, strict=False)
            tick = mt5.symbol_info_tick(symbol)
            if tick is None or not tick.time:
                continue
            reference = datetime.now(timezone.utc)
            # tick.time encodes the server clock, not a real UTC epoch
            server_wall = datetime(1970, 1, 1) + timedelta(seconds=int(tick.time))
            offset = measure_offset(server_wall, reference)
            if abs(offset) > MAX_PLAUSIBLE_SERVER_OFFSET:
                logger.debug(
                    "tick of %s too old to measure the timezone (offset %s)",
                    symbol,
                    offset,
                )
                continue
            zone = resolve_timezone(offset, at=reference)
            logger.info("server timezone detected on %s: offset %s -> %s", symbol, offset, zone)
            return zone
        raise ServerTimeError(
            "cannot detect the server timezone: no tick recent enough. With the "
            "market closed pass server_timezone explicitly; the value detected "
            "during the last session is saved in the cache metadata."
        )

    # -- symbols ---------------------------------------------------------

    def _select(self, symbol: str, strict: bool = True) -> None:
        if symbol in self._selected:
            return
        assert mt5 is not None
        if not mt5.symbol_select(symbol, True):
            if strict:
                raise MT5Error(f"symbol_select({symbol}) failed: {_last_error()}")
            return
        self._selected.add(symbol)

    @staticmethod
    def _to_spec(info: Any) -> SymbolSpec:
        return SymbolSpec(
            name=info.name,
            point=float(info.point),
            digits=int(info.digits),
            contract_size=float(info.trade_contract_size),
            tick_value=float(info.trade_tick_value),
            tick_size=float(info.trade_tick_size),
            volume_min=float(info.volume_min),
            volume_max=float(info.volume_max),
            volume_step=float(info.volume_step),
            swap_long=float(info.swap_long),
            swap_short=float(info.swap_short),
            currency_profit=str(info.currency_profit),
            trade_mode=_trade_mode_name(int(info.trade_mode)),
        )

    def list_symbols(self) -> list[SymbolSpec]:
        self._require_connection()
        assert mt5 is not None
        symbols = mt5.symbols_get()
        if symbols is None:
            raise MT5Error(f"symbols_get failed: {_last_error()}")
        return [self._to_spec(info) for info in symbols]

    def get_symbol_spec(self, symbol: str) -> SymbolSpec:
        self._require_connection()
        assert mt5 is not None
        self._select(symbol)
        info = mt5.symbol_info(symbol)
        if info is None:
            raise MT5Error(f"symbol {symbol} not found: {_last_error()}")
        return self._to_spec(info)

    # -- history ---------------------------------------------------------

    @staticmethod
    def _mt5_timeframe(timeframe: Timeframe) -> int:
        assert mt5 is not None
        constant = getattr(mt5, f"TIMEFRAME_{timeframe.name}", None)
        if constant is None:
            raise ValueError(f"timeframe {timeframe.name} not supported by MT5")
        return int(constant)

    def _warmup(self, symbol: str, timeframe: Timeframe) -> None:
        """Unlocks the deep history.

        Without a short preliminary request the terminal only returns the
        bars already in memory, and long copy_rates_range calls come back
        empty.
        """
        assert mt5 is not None
        self._select(symbol)
        rates = mt5.copy_rates_from_pos(symbol, self._mt5_timeframe(timeframe), 0, 10)
        if rates is None:
            logger.warning(
                "warm-up on %s %s returned no data: %s", symbol, timeframe.name, _last_error()
            )

    @staticmethod
    def _as_utc(moment: datetime) -> datetime:
        if moment.tzinfo is None:
            return moment.replace(tzinfo=timezone.utc)
        return moment.astimezone(timezone.utc)

    @staticmethod
    def _windows(
        start: datetime, end: datetime, span: timedelta
    ) -> Iterator[tuple[datetime, datetime]]:
        cursor = start
        while cursor < end:
            stop = min(cursor + span, end)
            yield cursor, stop
            cursor = stop

    def get_bars(
        self,
        symbol: str,
        timeframe: Timeframe | str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        self._require_connection()
        assert mt5 is not None
        tf = Timeframe.parse(timeframe)
        start_utc, end_utc = self._as_utc(start), self._as_utc(end)
        if end_utc <= start_utc:
            return empty_bars()

        self._warmup(symbol, tf)
        tz = self.server_timezone
        mt5_tf = self._mt5_timeframe(tf)
        span = timedelta(minutes=tf.minutes * self._max_bars_per_request)

        chunks: list[pd.DataFrame] = []
        for window_start, window_end in self._windows(start_utc, end_utc, span):
            rates = mt5.copy_rates_range(
                symbol,
                mt5_tf,
                utc_to_broker_datetime(window_start, tz),
                utc_to_broker_datetime(window_end, tz),
            )
            if rates is None:
                raise MT5Error(
                    f"copy_rates_range({symbol}, {tf.name}, {window_start}, {window_end}) "
                    f"failed: {_last_error()}"
                )
            if len(rates) == 0:
                continue
            frame = pd.DataFrame(rates)
            frame.index = server_epoch_to_utc_index(frame.pop("time").to_numpy(), tz)
            chunks.append(frame)

        if not chunks:
            logger.info("no bars %s %s between %s and %s", symbol, tf.name, start_utc, end_utc)
            return empty_bars()

        merged = normalize_bars(pd.concat(chunks), BAR_COLUMNS)
        # copy_rates_range includes the right endpoint: trimming it keeps the
        # cache coverage computation consistent with the request.
        merged = merged[(merged.index >= start_utc) & (merged.index < end_utc)]
        logger.info(
            "downloaded %d bars %s %s (%s -> %s UTC)",
            len(merged),
            symbol,
            tf.name,
            merged.index.min() if len(merged) else None,
            merged.index.max() if len(merged) else None,
        )
        return merged

    def get_ticks(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        self._require_connection()
        assert mt5 is not None
        start_utc, end_utc = self._as_utc(start), self._as_utc(end)
        if end_utc <= start_utc:
            return empty_ticks()

        self._select(symbol)
        tz = self.server_timezone
        chunks: list[pd.DataFrame] = []
        for window_start, window_end in self._windows(start_utc, end_utc, self._tick_chunk):
            ticks = mt5.copy_ticks_range(
                symbol,
                utc_to_broker_datetime(window_start, tz),
                utc_to_broker_datetime(window_end, tz),
                mt5.COPY_TICKS_ALL,
            )
            if ticks is None:
                raise MT5Error(
                    f"copy_ticks_range({symbol}, {window_start}, {window_end}) "
                    f"failed: {_last_error()}"
                )
            if len(ticks) == 0:
                continue
            frame = pd.DataFrame(ticks)
            if "time_msc" in frame.columns:
                index = server_epoch_ms_to_utc_index(frame.pop("time_msc").to_numpy(), tz)
                frame.pop("time")
            else:
                index = server_epoch_to_utc_index(frame.pop("time").to_numpy(), tz)
            frame.index = index
            chunks.append(frame)

        if not chunks:
            return empty_ticks()

        merged = pd.concat(chunks).sort_index()
        merged.index.name = "time"
        for column in TICK_COLUMNS:
            if column not in merged.columns:
                merged[column] = np.nan
        merged = merged[list(TICK_COLUMNS)]
        return merged[(merged.index >= start_utc) & (merged.index < end_utc)]
