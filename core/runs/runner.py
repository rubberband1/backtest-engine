"""Orchestration of a run: data -> backtest -> metrics -> store.

It lives in `core` and not in the API on purpose: the computation must be
runnable and testable without starting a server, and the API must remain a
thin layer that decides nothing.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone, tzinfo
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from core.data.cache import ParquetCache
from core.data.provider import SymbolSpec, Timeframe
from core.engine.backtester import BacktestConfig, run_backtest
from core.metrics.performance import buy_and_hold, compute_metrics
from core.research.edge import EdgeReport, edge_report
from core.runs.store import (
    RunConfig,
    RunMeta,
    RunStore,
    compute_run_id,
    data_fingerprint,
)
from core.strategy.spec import StrategySpec

logger = logging.getLogger(__name__)

SPEC_CACHE_FILE = "symbol_spec.json"
TZ_CACHE_FILE = "server_timezone.txt"


class DataUnavailable(RuntimeError):
    """The requested data is not in the cache."""


class EnvironmentUnavailable(RuntimeError):
    """The instrument spec or the server timezone is missing."""


class SymbolResolver:
    """SymbolSpec and server timezone, from the terminal or the on-disk copy.

    The MT5 terminal is the source, but it must not be a requirement for
    looking at past runs: the first time it answers, the spec is saved next
    to the data cache and from then on everything works offline.
    """

    def __init__(self, cache: ParquetCache) -> None:
        self.cache = cache
        self._memo: dict[str, SymbolSpec] = {}
        self._tz: tzinfo | None = None

    def _symbol_dir(self, symbol: str) -> Path:
        return self.cache.root / self.cache._slug(symbol)

    def _stored_spec(self, symbol: str) -> SymbolSpec | None:
        target = self._symbol_dir(symbol) / SPEC_CACHE_FILE
        if not target.exists():
            return None
        return SymbolSpec(**json.loads(target.read_text(encoding="utf-8")))

    def _store_spec(self, spec: SymbolSpec) -> None:
        folder = self._symbol_dir(spec.name)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / SPEC_CACHE_FILE).write_text(
            json.dumps(asdict(spec), indent=2), encoding="utf-8"
        )

    def _stored_timezone(self) -> tzinfo | None:
        target = self.cache.root / TZ_CACHE_FILE
        if target.exists():
            return ZoneInfo(target.read_text(encoding="utf-8").strip())
        for folder in self.cache.root.glob("*/*/*.json"):
            payload = json.loads(folder.read_text(encoding="utf-8"))
            name = payload.get("server_timezone")
            if name:
                return ZoneInfo(name)
        return None

    def _store_timezone(self, zone: tzinfo) -> None:
        self.cache.root.mkdir(parents=True, exist_ok=True)
        (self.cache.root / TZ_CACHE_FILE).write_text(str(zone), encoding="utf-8")

    def refresh(self, symbols: list[str] | None = None) -> list[SymbolSpec]:
        """Queries the terminal and updates the on-disk copies."""
        from core.data.mt5_provider import MT5Provider

        fallback = self._stored_timezone()
        with MT5Provider(server_timezone=fallback) as provider:
            specs = provider.list_symbols()
            zone = provider.server_timezone
        self._tz = zone
        self._store_timezone(zone)
        wanted = set(symbols) if symbols else None
        for spec in specs:
            if wanted is None or spec.name in wanted:
                self._memo[spec.name] = spec
                self._store_spec(spec)
        return specs

    def symbol_spec(self, symbol: str) -> SymbolSpec:
        if symbol in self._memo:
            return self._memo[symbol]
        stored = self._stored_spec(symbol)
        if stored is not None:
            self._memo[symbol] = stored
            return stored
        try:
            self.refresh([symbol])
        except Exception as exc:
            raise EnvironmentUnavailable(
                f"spec for {symbol} unavailable: the MT5 terminal is not "
                f"responding and there is no cached copy ({exc})"
            ) from exc
        if symbol not in self._memo:
            raise EnvironmentUnavailable(f"the broker does not list symbol {symbol}")
        return self._memo[symbol]

    def server_timezone(self) -> tzinfo:
        if self._tz is not None:
            return self._tz
        stored = self._stored_timezone()
        if stored is not None:
            self._tz = stored
            return stored
        try:
            self.refresh([])
        except Exception as exc:
            raise EnvironmentUnavailable(
                f"server timezone unavailable: the MT5 terminal is not "
                f"responding and the cache does not hold one ({exc})"
            ) from exc
        assert self._tz is not None
        return self._tz


def cached_years(cache: ParquetCache, symbol: str, timeframe: Timeframe) -> list[int]:
    folder = cache.paths(symbol, timeframe, 0)[0].parent
    if not folder.exists():
        return []
    return sorted(int(path.stem) for path in folder.glob("*.parquet"))


def load_bars(
    cache: ParquetCache,
    symbol: str,
    timeframe: Timeframe,
    start: datetime | None = None,
    end: datetime | None = None,
) -> pd.DataFrame:
    """Bars from the local cache. Downloads nothing: downloading is a separate step."""
    years = cached_years(cache, symbol, timeframe)
    if not years:
        raise DataUnavailable(
            f"no cached data for {symbol} {timeframe.name}. "
            f"Download it with examples.download_year."
        )
    if start is not None:
        years = [y for y in years if y >= start.year]
    if end is not None:
        years = [y for y in years if y <= end.year]
    frames = [cache.read_year(symbol, timeframe, year) for year in years]
    frames = [frame for frame in frames if len(frame)]
    if not frames:
        raise DataUnavailable(
            f"no bars for {symbol} {timeframe.name} in the requested period"
        )
    bars = pd.concat(frames).sort_index()
    if start is not None:
        bars = bars[bars.index >= pd.Timestamp(start).tz_convert("UTC")]
    if end is not None:
        bars = bars[bars.index < pd.Timestamp(end).tz_convert("UTC")]
    if bars.empty:
        raise DataUnavailable(
            f"no bars for {symbol} {timeframe.name} between {start} and {end}"
        )
    return bars


def plan_run(
    spec: StrategySpec, config: RunConfig, bars: pd.DataFrame
) -> tuple[str, str]:
    """run_id and data fingerprint, without executing anything."""
    fingerprint = data_fingerprint(bars)
    return compute_run_id(spec, config, fingerprint), fingerprint


def execute_run(
    store: RunStore,
    spec: StrategySpec,
    config: RunConfig,
    bars: pd.DataFrame,
    symbol_spec: SymbolSpec,
    server_tz: tzinfo,
    run_id: str | None = None,
) -> RunMeta:
    """Runs the backtest and persists the run. Raises if anything goes wrong."""
    fingerprint = data_fingerprint(bars)
    run_id = run_id or compute_run_id(spec, config, fingerprint)

    if not store.exists(run_id):
        store.begin_run(
            run_id, spec, config, fingerprint, len(bars),
            bars.index[0].to_pydatetime(), bars.index[-1].to_pydatetime(),
        )

    try:
        costs = config.cost_model()
        result = run_backtest(
            spec,
            bars,
            symbol_spec,
            server_tz,
            BacktestConfig(
                initial_equity=config.initial_equity,
                costs=costs,
                session_threshold=config.session_threshold,
            ),
        )
        strategy_report = compute_metrics(
            result.trades, result.equity, result.timeframe, config.initial_equity, spec.id
        )
        benchmark = buy_and_hold(
            bars, symbol_spec, spec.sizing, result.timeframe,
            config.initial_equity, costs, server_tz,
        )
        return store.finish_run(run_id, result, strategy_report, benchmark)
    except Exception as exc:
        store.fail_run(run_id, f"{type(exc).__name__}: {exc}")
        raise


def run_edge_gate(
    spec: StrategySpec,
    config: RunConfig,
    bars: pd.DataFrame,
    symbol_spec: SymbolSpec,
    horizons: list[int] | None = None,
    min_observations: int | None = None,
) -> EdgeReport:
    """Gate zero with the same spread policy as the backtest."""
    from core.metrics.breakeven import breakeven_prior
    from core.research.edge import DEFAULT_HORIZONS, DEFAULT_MIN_OBSERVATIONS

    costs = config.cost_model()
    report = edge_report(
        spec,
        bars,
        symbol_spec.point,
        horizons=horizons or list(DEFAULT_HORIZONS),
        spread=costs.spread,
        min_observations=min_observations or DEFAULT_MIN_OBSERVATIONS,
    )
    avg_spread = float(costs.spread.series(bars).mean()) if len(bars) else None
    report.breakeven_prior = breakeven_prior(
        spec, symbol_spec, costs.commission, avg_spread
    ).as_dict()
    return report


def utc(moment: datetime | None) -> datetime | None:
    if moment is None:
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)
