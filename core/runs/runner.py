"""Orchestration of a run: data -> backtest -> metrics -> store.

It lives in `core` and not in the API on purpose: the computation must be
runnable and testable without starting a server, and the API must remain a
thin layer that decides nothing.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone, tzinfo
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from core.data.cache import ParquetCache
from core.data.provider import SymbolSpec, SymbolSpecSnapshot, Timeframe
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
from core.strategy.binding import BoundSpec

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
        self._memo: dict[str, SymbolSpecSnapshot] = {}
        self._tz: tzinfo | None = None

    def _symbol_dir(self, symbol: str) -> Path:
        return self.cache.root / self.cache._slug(symbol)

    def _stored_snapshot(self, symbol: str) -> SymbolSpecSnapshot | None:
        target = self._symbol_dir(symbol) / SPEC_CACHE_FILE
        if not target.exists():
            return None
        try:
            payload = json.loads(target.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            # An unreadable copy is the same as no copy: the terminal is asked
            # again. Raising here would let one damaged file take down every
            # listing that walks the cache.
            logger.warning("%s unreadable (%s): treating it as absent", target, exc)
            return None
        if "spec" in payload and "read_at" in payload:
            return SymbolSpecSnapshot.from_dict(payload)
        # pre-A1 cache file: a bare SymbolSpec dict with no read timestamp.
        # The file's own mtime is the best available estimate of when it was
        # read; it will be replaced with a real one on the next refresh().
        read_at = datetime.fromtimestamp(target.stat().st_mtime, tz=timezone.utc)
        return SymbolSpecSnapshot(spec=SymbolSpec(**payload), read_at=read_at)

    def _store_spec(self, snapshot: SymbolSpecSnapshot) -> None:
        """Writes the spec atomically.

        `refresh()` rewrites several hundred of these while the dashboard is
        reading them; a plain write leaves a window in which a reader sees an
        empty file, and the request that hit that window returned a 500.
        """
        folder = self._symbol_dir(snapshot.spec.name)
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / SPEC_CACHE_FILE
        staging = target.with_suffix(".json.tmp")
        staging.write_text(json.dumps(snapshot.to_dict(), indent=2), encoding="utf-8")
        os.replace(staging, target)

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
        read_at = datetime.now(timezone.utc)
        self._tz = zone
        self._store_timezone(zone)
        wanted = set(symbols) if symbols else None
        for spec in specs:
            if wanted is None or spec.name in wanted:
                snapshot = SymbolSpecSnapshot(spec=spec, read_at=read_at)
                self._memo[spec.name] = snapshot
                self._store_spec(snapshot)
        return specs

    def symbol_spec_snapshot(self, symbol: str) -> SymbolSpecSnapshot:
        if symbol in self._memo:
            return self._memo[symbol]
        stored = self._stored_snapshot(symbol)
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

    def symbol_spec(self, symbol: str) -> SymbolSpec:
        return self.symbol_spec_snapshot(symbol).spec

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
        # set by `refresh`, which mypy cannot see through
        assert self._tz is not None
        return self._tz  # type: ignore[unreachable]


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
            f"{_where_to_get_it(cache, symbol)}"
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


def _where_to_get_it(cache: ParquetCache, symbol: str) -> str:
    """What to do about a missing instrument, which depends on the cache.

    On the synthetic fixture "download it" is the wrong advice: there is
    nothing to download, and the strategy specs in `strategies/` name real
    broker instruments the fixture does not have. Saying which symbols exist
    is more use than repeating a command that cannot work here.
    """
    from core.data.fixture_provider import FIXTURE_CACHE, INSTRUMENTS

    if cache.root.resolve() == FIXTURE_CACHE.resolve():
        available = ", ".join(INSTRUMENTS)
        return (
            f"This backend is serving the synthetic fixture, which only holds "
            f"{available} - {symbol} is a real broker instrument and is not in "
            f"it. Change the symbol to one of those, or point the backend at a "
            f"data_cache/ with real bars."
        )
    return "Download it with examples.download_year."


# Where `load_bars_for_run` leaves the spread coverage for `execute_run`.
SPREAD_COVERAGE_ATTR = "spread_coverage"


def load_bars_for_run(cache: ParquetCache, config: RunConfig) -> pd.DataFrame:
    """The bars a run will execute on, with an honest per-bar spread or none.

    `load_bars` returns what the cache holds: above M1 that is a
    `min_spread_m1` column, the minimum of the M1 spreads inside each bar,
    which the cost model refuses to charge. When the run asks for a per-bar
    spread anyway, this is where the real one is rebuilt, from the M1 sample
    of the same period. When that sample is missing the run does not start.

    Every path that executes a spec goes through here, which is the point:
    the refusal is not a warning somebody can decide to read. It is also why
    the spread coverage is measured here - one place, and no run can be
    stored without it.
    """
    from core.data.spread import attach, coverage

    bars = load_bars(cache, config.symbol, config.tf, config.start, config.end)
    if config.spread_mode in ("per_bar", "quantile"):
        bars = attach(
            cache, config.symbol, config.tf, bars, config.per_bar_spread_quantile
        )
    # ride along on the frame rather than through three call signatures: the
    # object handed to `execute_run` is this one, and `SPREAD_COVERAGE_ATTR`
    # is the only key anything reads out of `.attrs`
    bars.attrs[SPREAD_COVERAGE_ATTR] = coverage(
        cache, config.symbol, config.tf, pd.DatetimeIndex(bars.index)
    )
    return bars


def plan_run(
    bound: BoundSpec, config: RunConfig, bars: pd.DataFrame, symbol_spec: SymbolSpec
) -> tuple[str, str]:
    """run_id and data fingerprint, without executing anything."""
    bound.must_match(config.symbol, config.timeframe)
    fingerprint = data_fingerprint(bars)
    return compute_run_id(bound.spec, config, fingerprint, symbol_spec), fingerprint


def execute_run(
    store: RunStore,
    bound: BoundSpec,
    config: RunConfig,
    bars: pd.DataFrame,
    symbol_spec: SymbolSpecSnapshot,
    server_tz: tzinfo,
    run_id: str | None = None,
) -> RunMeta:
    """Runs the backtest and persists the run. Raises if anything goes wrong."""
    bound.must_match(config.symbol, config.timeframe)
    spec = bound.spec
    fingerprint = data_fingerprint(bars)
    run_id = run_id or compute_run_id(spec, config, fingerprint, symbol_spec.spec)

    if not store.exists(run_id):
        store.begin_run(
            run_id, spec, config, fingerprint, len(bars),
            bars.index[0].to_pydatetime(), bars.index[-1].to_pydatetime(),
            symbol_spec,
        )

    try:
        costs = config.cost_model()
        result = run_backtest(
            spec,
            bars,
            symbol_spec.spec,
            server_tz,
            BacktestConfig(
                initial_equity=config.initial_equity,
                costs=costs,
                session_threshold=config.session_threshold,
            ),
        )
        strategy_report = compute_metrics(
            result.trades, result.equity, result.timeframe, config.initial_equity,
            spec.id, server_tz=server_tz,
        )
        benchmark = buy_and_hold(
            bars, symbol_spec.spec, spec.sizing, result.timeframe,
            config.initial_equity, costs, server_tz,
        )
        return store.finish_run(
            run_id, result, strategy_report, benchmark, spec, symbol_spec.spec,
            spread_coverage=bars.attrs.get(SPREAD_COVERAGE_ATTR),
        )
    except Exception as exc:
        store.fail_run(run_id, f"{type(exc).__name__}: {exc}")
        raise


def run_edge_gate(
    bound: BoundSpec,
    config: RunConfig,
    bars: pd.DataFrame,
    symbol_spec: SymbolSpec,
    horizons: list[int] | None = None,
    min_observations: int | None = None,
) -> EdgeReport:
    """Gate zero with the same spread policy as the backtest."""
    from core.metrics.breakeven import breakeven_prior
    from core.research.edge import DEFAULT_HORIZONS, DEFAULT_MIN_OBSERVATIONS

    bound.must_match(config.symbol, config.timeframe)
    spec = bound.spec
    costs = config.cost_model()
    report = edge_report(
        spec,
        bars,
        symbol_spec.point,
        horizons=horizons or list(DEFAULT_HORIZONS),
        spread=costs.spread,
        min_observations=min_observations or DEFAULT_MIN_OBSERVATIONS,
    )
    avg_spread = (
        float(costs.spread.series(bars, config.tf).mean()) if len(bars) else None
    )
    # ATR and percent exits have no single distance: the ambiguity estimate
    # already measured the average the strategy would have placed, and the
    # break-even prior is stated over that same average
    measured = report.ambiguity_prior or {}
    report.breakeven_prior = breakeven_prior(
        spec,
        symbol_spec,
        costs.commission,
        avg_spread,
        stop_points=measured.get("stop_points"),
        target_points=measured.get("target_points"),
        stop_points_std=measured.get("stop_points_std"),
        target_points_std=measured.get("target_points_std"),
    ).as_dict()
    return report


def utc(moment: datetime | None) -> datetime | None:
    if moment is None:
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)
