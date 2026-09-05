"""Local API of the backtest engine.

Runs only on 127.0.0.1, with no authentication, no database, no external
services. It is the interface between the engine and the dashboard, and it
contains no computation logic: everything it returns was computed by `core`.

Errors leave with the right HTTP status and a readable sentence. A stack
trace in the browser helps nobody: it goes to the server log, where it
belongs.
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
import tempfile
import time
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import datetime, timedelta, timezone, tzinfo
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from scipy import stats
from starlette.requests import Request

from api import schemas as s
from api.downsample import downsample
from core.batch.runner import Period as BatchPeriod
from core.batch.runner import run_batch
from core.data.cache import ParquetCache
from core.data.fixture_provider import resolve_cache_dir
from core.data.provider import Timeframe
from core.data.quality import check_quality
from core.data.spread import SpreadUnavailable
from core.engine.backtester import BacktestConfig, run_backtest
from core.engine.costs import AggregatedSpreadRefused
from core.live.compare import compare as live_compare
from core.live.journal import Journal
from core.live.lock import RunLock, process_alive
from core.paths import project_relative
from core.research.screen import bind_cell, run_screen
from core.research.tradability import DEFAULT_MAX_SPREAD_ATR
from core.research.tradability import build_table as build_tradability
from core.runs.runner import (
    DataUnavailable,
    EnvironmentUnavailable,
    SymbolResolver,
    cached_years,
    execute_run,
    load_bars,
    load_bars_for_run,
    plan_run,
    run_edge_gate,
)
from core.runs.store import RunConfig, RunNotFound, RunStore, spec_hash
from core.serialization import json_safe
from core.strategy.spec import SpecError, StrategySpec
from core.validation.multiple_testing import (
    Trial,
    collect_trials,
    multiple_testing_report,
    sharpe_per_trade,
)
from core.validation.permutation import permutation_test
from core.validation.tick_resolve import resolve_ambiguous
from core.validation.walkforward import (
    WalkForwardConfig,
    WalkForwardError,
    apply_params,
    expand_grid,
    walk_forward,
)
from core.version import ENGINE_VERSION

logger = logging.getLogger(__name__)

# Beyond this the request stops waiting: the run_id is returned and the
# frontend polls.
SYNC_TIMEOUT_SECONDS = 2.0
# A run that stays "running" beyond this is declared failed on the first
# read. A thread cannot be killed in Python: it keeps going and its result
# is discarded.
HARD_TIMEOUT_SECONDS = 600.0
# Safety cap: a request cannot start an unbounded run.
MAX_BARS = 3_000_000
MAX_EQUITY_POINTS = 2000

# Which bars the backend serves. A machine with a terminal and a downloaded
# `data_cache/` gets that; a fresh clone gets the synthetic fixture, so the
# application starts with something in it instead of an empty instrument
# list. `IS_FIXTURE` is carried into /api/health and shown by the UI: a
# dashboard that presents invented data without saying so is the one failure
# mode this project cannot have.
CACHE_DIR, IS_FIXTURE = resolve_cache_dir(os.environ.get("BACKTEST_CACHE_DIR"))
RUNS_DIR = Path(os.environ.get("BACKTEST_RUNS_DIR", "runs"))
STRATEGIES_DIR = Path(os.environ.get("BACKTEST_STRATEGIES_DIR", "strategies"))
# The backend never starts a runner: scripts.run_live does, in its own
# process. This is only where its diaries are read from.
LIVE_DIR = Path(os.environ.get("BACKTEST_LIVE_DIR", "logs/live"))

cache = ParquetCache(CACHE_DIR)
store = RunStore(RUNS_DIR)
resolver = SymbolResolver(cache)
_executor: ThreadPoolExecutor | None = None
_jobs: dict[str, Future] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _executor
    _executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="backtest")
    logger.info(
        "cache=%s runs=%s strategies=%s",
        project_relative(CACHE_DIR),
        project_relative(RUNS_DIR),
        project_relative(STRATEGIES_DIR),
    )
    if IS_FIXTURE:
        logger.warning(
            "serving the SYNTHETIC FIXTURE from %s: the bars are invented and "
            "every number measured on them describes a random number "
            "generator, not a market", project_relative(CACHE_DIR),
        )
    yield
    _executor.shutdown(wait=False, cancel_futures=True)


app = FastAPI(
    title="backtest-engine",
    version=ENGINE_VERSION,
    description="Local API: cached data, strategies, backtests, saved runs.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    # local dev server only: no remote origin may talk to a service that
    # reads the filesystem
    allow_origin_regex=r"^http://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


# -- errors --------------------------------------------------------------


@app.exception_handler(RunNotFound)
def _run_not_found(request: Request, exc: RunNotFound) -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": f"run {exc.args[0]} does not exist"})


@app.exception_handler(SpecError)
def _spec_error(request: Request, exc: SpecError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.exception_handler(SpreadUnavailable)
def _spread_unavailable(request: Request, exc: SpreadUnavailable) -> JSONResponse:
    return JSONResponse(status_code=409, content={"detail": str(exc)})


@app.exception_handler(AggregatedSpreadRefused)
def _aggregated_spread(request: Request, exc: AggregatedSpreadRefused) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.exception_handler(DataUnavailable)
def _data_unavailable(request: Request, exc: DataUnavailable) -> JSONResponse:
    return JSONResponse(status_code=409, content={"detail": str(exc)})


@app.exception_handler(EnvironmentUnavailable)
def _environment_unavailable(request: Request, exc: EnvironmentUnavailable) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(Exception)
def _unexpected(request: Request, exc: Exception) -> JSONResponse:
    logger.error("unhandled error on %s", request.url.path, exc_info=exc)
    return JSONResponse(
        status_code=500,
        content={
            "detail": (
                f"internal error ({type(exc).__name__}). The detail is in the "
                f"server log."
            )
        },
    )


# -- instruments ---------------------------------------------------------


def _cached_symbols() -> list[str]:
    if not CACHE_DIR.exists():
        return []
    return sorted(
        folder.name
        for folder in CACHE_DIR.iterdir()
        if folder.is_dir() and any(folder.glob("*/*.parquet"))
    )


@app.get("/api/symbols", response_model=s.SymbolListOut, tags=["data"])
def get_symbols() -> s.SymbolListOut:
    """Broker instruments with their contract specification."""
    known = _cached_symbols()
    if IS_FIXTURE:
        # `refresh` writes a spec file per broker symbol into the cache root.
        # Against the fixture that would drop a few hundred real instruments
        # into a directory that is committed, so the terminal is not asked at
        # all: the fixture is a fixed artefact, not a cache to be filled.
        specs = []
        source = "fixture"
    else:
        try:
            specs = resolver.refresh()
            source = "terminal"
        except Exception as exc:
            logger.warning("terminal unavailable, using the cached specs: %s", exc)
            specs = []
            source = "cache"

    # symbols with cached data stay selectable even if the broker no longer
    # lists them: runs already made on them must stay reproducible
    listed = {spec.name for spec in specs}
    for name in known:
        if name in listed:
            continue
        try:
            specs.append(resolver.symbol_spec(name))
        except EnvironmentUnavailable:
            logger.warning("no spec for %s: excluded from the listing", name)

    if not specs:
        raise EnvironmentUnavailable(
            "the MT5 terminal is not responding and there are no saved specs: "
            "start MetaTrader 5 and retry"
        )

    try:
        zone: str | None = str(resolver.server_timezone())
    except EnvironmentUnavailable:
        zone = None

    # the ones you can actually work on come first
    ordered = sorted(specs, key=lambda spec: (spec.name not in known, spec.name))
    return s.SymbolListOut(
        symbols=[s.SymbolSpecOut(**asdict(spec)) for spec in ordered],
        source=source,
        server_timezone=zone,
        cached_symbols=known,
    )


@app.get("/api/symbols/{symbol}/coverage", response_model=s.CoverageOut, tags=["data"])
def get_coverage(
    symbol: str,
    timeframe: str = Query(default="M1"),
    quality: bool = Query(default=True, description="also compute the quality report"),
) -> s.CoverageOut:
    """What data is cached for this instrument, and how healthy it is."""
    tf = _timeframe(timeframe)
    years = cached_years(cache, symbol, tf)
    if not years:
        return s.CoverageOut(
            symbol=symbol, timeframe=tf.name, years=[], start=None, end=None, bars=0
        )

    bars = load_bars(cache, symbol, tf)
    report = None
    if quality:
        # the session grid has to be built on the broker clock: in UTC a DST
        # change moves every slot by an hour and a whole summer reads as a
        # gap. Without the terminal it falls back to UTC, and the report
        # carries which clock it used.
        try:
            session_tz: tzinfo | None = resolver.server_timezone()
        except EnvironmentUnavailable:
            session_tz = None
        computed = check_quality(bars, symbol, tf, server_tz=session_tz)
        worst = sorted(computed.gaps, key=lambda g: g.missing_bars, reverse=True)[:5]
        report = s.QualityOut(
            rows=computed.rows,
            first_bar=computed.first_bar,
            last_bar=computed.last_bar,
            expected_bars=computed.expected_bars,
            missing_bars=computed.missing_bars,
            completeness=computed.completeness,
            gaps=len(computed.gaps),
            duplicate_timestamps=computed.duplicate_timestamps,
            zero_spread=computed.zero_spread,
            session_confidence=computed.session_confidence,
            session_clock=computed.session_clock,
            text=computed.as_text(),
            worst_gaps=[
                s.GapOut(start=g.start, end=g.end, missing_bars=g.missing_bars) for g in worst
            ],
        )

    from core.data.spread import measure_from_cache

    reference = measure_from_cache(cache, symbol)
    return s.CoverageOut(
        symbol=symbol,
        timeframe=tf.name,
        years=years,
        start=bars.index[0].to_pydatetime(),
        end=bars.index[-1].to_pydatetime(),
        bars=len(bars),
        quality=report,
        spread_median_points=(
            float(reference.median_points) if reference is not None else None
        ),
    )


# -- strategies ----------------------------------------------------------


def _strategy_files() -> Iterator[Path]:
    if STRATEGIES_DIR.exists():
        yield from sorted(STRATEGIES_DIR.glob("*.json"))


def _to_strategy_out(spec: StrategySpec, path: Path) -> s.StrategyOut:
    return s.StrategyOut(
        id=spec.id,
        name=spec.name,
        description=spec.description,
        symbol=spec.instrument.symbol,
        timeframe=spec.instrument.timeframe,
        file=path.name,
        spec=spec.model_dump(mode="json"),
    )


@app.get("/api/strategies", response_model=list[s.StrategyOut], tags=["strategies"])
def list_strategies() -> list[s.StrategyOut]:
    out: list[s.StrategyOut] = []
    for path in _strategy_files():
        try:
            out.append(_to_strategy_out(StrategySpec.from_json(path), path))
        except SpecError as exc:
            # a broken file must not hide the others: it goes to the log
            logger.warning("strategy %s invalid: %s", path.name, exc)
    return out


@app.get("/api/strategies/{strategy_id}", response_model=s.StrategyOut, tags=["strategies"])
def get_strategy(strategy_id: str) -> s.StrategyOut:
    for path in _strategy_files():
        try:
            spec = StrategySpec.from_json(path)
        except SpecError:
            continue
        if spec.id == strategy_id:
            return _to_strategy_out(spec, path)
    raise HTTPException(
        status_code=404,
        detail=f"no strategy with id {strategy_id!r} in {STRATEGIES_DIR}/",
    )


@app.post("/api/strategies/validate", response_model=s.ValidateResponse, tags=["strategies"])
def validate_strategy(request: s.ValidateRequest) -> s.ValidateResponse:
    """Validates a spec without saving it. The errors are the human-readable ones."""
    try:
        spec = StrategySpec.from_dict(request.spec)
    except SpecError as exc:
        lines = str(exc).splitlines()
        return s.ValidateResponse(
            valid=False,
            message=lines[0] if lines else "invalid spec",
            errors=[line.strip(" -") for line in lines[1:]],
        )
    return s.ValidateResponse(
        valid=True,
        message=f"valid spec: {spec.name} on {spec.instrument.symbol} "
        f"{spec.instrument.timeframe}",
        normalized=spec.model_dump(mode="json"),
    )


# -- the vocabulary the editor builds from --------------------------------


def _params_out(model: Any) -> list[s.ParamOut]:
    """A pydantic params model as the flat parameter list the editor needs."""
    schema = model.model_json_schema()
    out: list[s.ParamOut] = []
    for name, field in schema.get("properties", {}).items():
        choices = field.get("enum")
        out.append(
            s.ParamOut(
                name=name,
                type="enum" if choices else str(field.get("type", "string")),
                default=field.get("default"),
                minimum=field.get("minimum"),
                maximum=field.get("maximum"),
                exclusive_minimum=field.get("exclusiveMinimum"),
                choices=[str(c) for c in choices] if choices else None,
            )
        )
    return out


@app.get("/api/vocabulary", response_model=s.VocabularyOut, tags=["strategies"])
def get_vocabulary() -> s.VocabularyOut:
    """What a spec may contain, straight from the registry and the models.

    The editor renders whatever is here and nothing else. Duplicating this
    list in the frontend is how a spec becomes valid on screen and invalid on
    the server.
    """
    from typing import get_args

    from core.indicators import registry
    from core.strategy.features import FEATURE_NAMES
    from core.strategy.spec import SCHEMA_VERSION, BarField, Compare, Sizing, Trend

    indicators = [
        s.IndicatorOut(
            name=name,
            params=_params_out(registry.get(name).params_model),
            outputs=list(registry.get(name).outputs),
            bar_inputs=list(registry.get(name).bar_inputs),
        )
        for name in registry.available()
    ]
    return s.VocabularyOut(
        schema_version=SCHEMA_VERSION,
        indicators=indicators,
        features=list(FEATURE_NAMES),
        bar_fields=list(get_args(BarField)),
        price_sources=list(get_args(registry.PriceSource)),
        comparison_operators=list(
            get_args(Compare.model_fields["op"].annotation)
        ),
        group_operators=["and", "or", "not"],
        trend_operators=list(get_args(Trend.model_fields["op"].annotation)),
        exit_level_types=["points", "percent", "atr"],
        sizing_types=list(get_args(Sizing.model_fields["type"].annotation)),
        timeframes=[tf.name for tf in Timeframe],
        spread_modes=["per_bar", "fixed", "quantile"],
        swap_modes=["points", "money", "none"],
    )


STRATEGY_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _unusable_as_filename(strategy_id: str) -> bool:
    """A spec id becomes a file name, so it has to be one.

    Not cosmetic: `..` and a separator in an id would let a save request
    write outside `strategies/`, and this endpoint takes its input from a
    browser.
    """
    return not STRATEGY_ID_PATTERN.match(strategy_id) or ".." in strategy_id


@app.post("/api/strategies", response_model=s.SaveStrategyResponse, tags=["strategies"])
def save_strategy(request: s.SaveStrategyRequest) -> s.SaveStrategyResponse:
    """Validates a spec and writes it to `strategies/`.

    Refuses to replace an existing file unless asked to: the most likely way
    to reach this endpoint is "duplicate one of the library strategies and
    change it", and silently overwriting the original would be the worst
    possible outcome of that flow.
    """
    spec = StrategySpec.from_dict(request.spec, "<request>")
    if _unusable_as_filename(spec.id):
        raise HTTPException(
            status_code=400,
            detail=f"strategy id {spec.id!r} is not usable as a file name: use "
            f"letters, digits, dashes and underscores",
        )

    STRATEGIES_DIR.mkdir(parents=True, exist_ok=True)
    target = STRATEGIES_DIR / f"{spec.id}.json"
    existing = target.exists()
    if existing and not request.overwrite:
        raise HTTPException(
            status_code=409,
            detail=f"{target.name} already exists. Change the strategy id, or "
            f"send overwrite=true to replace it",
        )
    spec.save(target)
    logger.info("strategy %s written to %s", spec.id, target)
    return s.SaveStrategyResponse(
        id=spec.id,
        file=target.name,
        created=not existing,
        message=(
            f"replaced {target.name}" if existing else f"saved as {target.name}"
        ),
    )


@app.post("/api/strategies/preview", response_model=s.PreviewResponse, tags=["strategies"])
def post_preview(request: s.PreviewRequest) -> s.PreviewResponse:
    """What this spec is about to cost, before any backtest is run."""
    from core.research.preview import preview as build_preview

    spec, run_config, bars, symbol_spec = _prepare(request, request.config)
    family = [
        store.load_run(summary.run_id)
        for summary in store.list_runs(symbol=run_config.symbol, status="done")
    ]
    # the same instrument over a period that does not overlap this one is a
    # different question and does not belong in the count
    records = [
        record
        for record in family
        if _periods_overlap(record.meta.data_start, record.meta.data_end,
                            run_config.start, run_config.end)
    ]
    report = build_preview(
        spec,
        bars,
        symbol_spec.spec,
        cache,
        commission=run_config.cost_model().commission,
        records=records,
        period_start=run_config.start,
        period_end=run_config.end,
        overall_trials=_whole_search(),
    )
    return s.PreviewResponse(**report.as_dict())


# The whole-search correction, cached. Building it loads every finished run
# from disk, and the editor asks for a preview on every edit; runs are
# immutable and only ever appended, so the cache is invalidated by the set of
# run ids changing rather than by a timer.
_TRIALS_CACHE: dict[str, Any] = {"key": None, "value": None}


def _whole_search() -> Any:
    """Every attempt this engine has registered, on any instrument."""
    from core.research.preview import trial_set

    ids = tuple(
        sorted(summary.run_id for summary in store.list_runs(status="done"))
    )
    key = hashlib.sha256("".join(ids).encode("utf-8")).hexdigest()
    if _TRIALS_CACHE["key"] != key:
        _TRIALS_CACHE["value"] = trial_set(
            [store.load_run(run_id) for run_id in ids]
        )
        _TRIALS_CACHE["key"] = key
    return _TRIALS_CACHE["value"]


def _periods_overlap(
    left_start: datetime | None,
    left_end: datetime | None,
    right_start: datetime | None,
    right_end: datetime | None,
) -> bool:
    """Whether two half-open periods share any time at all. None is unbounded."""
    # two guard clauses rather than one negated boolean: each line is one
    # way the periods can fail to overlap, and reads as such
    if left_end is not None and right_start is not None and left_end <= right_start:
        return False
    if right_end is not None and left_start is not None and right_end <= left_start:  # noqa: SIM103
        return False
    return True


# -- preparing a run -----------------------------------------------------


def _timeframe(value: str) -> Timeframe:
    try:
        return Timeframe.parse(value)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _resolve_spec(request: s.StrategyRef) -> StrategySpec:
    if request.spec is not None:
        return StrategySpec.from_dict(request.spec, "<request>")
    if request.strategy_id:
        return StrategySpec.from_dict(get_strategy(request.strategy_id).spec, "<file>")
    raise HTTPException(
        status_code=400, detail="the request needs either 'strategy_id' or 'spec'"
    )


def _run_config(config: s.RunConfigIn) -> RunConfig:
    if config.spread_mode == "fixed" and config.spread_value is None:
        raise HTTPException(
            status_code=400, detail="spread_mode 'fixed' requires spread_value in points"
        )
    if config.spread_mode == "quantile" and not (
        config.spread_value is not None and 0.0 <= config.spread_value <= 1.0
    ):
        raise HTTPException(
            status_code=400,
            detail="spread_mode 'quantile' requires spread_value between 0 and 1",
        )
    if not 0.5 <= config.per_bar_spread_quantile <= 1.0:
        raise HTTPException(
            status_code=400,
            detail="per_bar_spread_quantile must be between 0.5 and 1.0: below "
            "the median the reconstruction drifts back towards the minimum it "
            "exists to replace",
        )
    return RunConfig(
        symbol=config.symbol,
        timeframe=_timeframe(config.timeframe).name,
        start=config.start,
        end=config.end,
        initial_equity=config.initial_equity,
        spread_mode=config.spread_mode,
        spread_value=config.spread_value,
        commission_per_lot_per_side=config.commission_per_lot_per_side,
        swap_mode=config.swap_mode,
        session_threshold=config.session_threshold,
        per_bar_spread_quantile=config.per_bar_spread_quantile,
    )


def _prepare(request: s.StrategyRef, config: s.RunConfigIn):
    """The spec as it will actually be run, against the bars the config names.

    The instrument block of a spec is a default, not the request: the Run page
    exists to pick a symbol and a timeframe, and the campaign runner has always
    bound the spec to the cell it executes. This path did not, so a spec
    written on `XAUUSD.r H1` and run over `AUDUSD.r H4` bars kept reading H1 -
    and the engine takes the time stop, the spread realism check and the
    result's own instrument label from the spec, not from the config. The time
    stop was then out by the ratio of the two timeframes.
    """
    spec = bind_cell(
        _resolve_spec(request), config.symbol, _timeframe(config.timeframe).name
    )
    run_config = _run_config(config)
    bars = load_bars_for_run(cache, run_config)
    if len(bars) > MAX_BARS:
        raise HTTPException(
            status_code=413,
            detail=f"{len(bars)} bars exceed the cap of {MAX_BARS}: "
            f"narrow the period",
        )
    symbol_spec = resolver.symbol_spec_snapshot(run_config.symbol)
    return spec, run_config, bars, symbol_spec


# -- gate zero -----------------------------------------------------------


@app.post("/api/edge", response_model=s.EdgeResponse, tags=["research"])
def post_edge(request: s.EdgeRequest) -> s.EdgeResponse:
    """Gate zero: does the signal beat the spread, before even talking SL/TP?"""
    spec, run_config, bars, symbol_spec = _prepare(request, request.config)
    if request.horizons is not None and (
        not request.horizons or any(h < 1 for h in request.horizons)
    ):
        raise HTTPException(status_code=400, detail="horizons must be integers >= 1")

    started = time.perf_counter()
    report = run_edge_gate(
        spec, run_config, bars, symbol_spec.spec, request.horizons, request.min_observations
    )
    logger.info("gate zero in %.2fs", time.perf_counter() - started)
    return s.EdgeResponse(**report.as_dict())


# -- backtest ------------------------------------------------------------


@app.post("/api/backtest", response_model=s.BacktestResponse, tags=["backtest"])
def post_backtest(request: s.BacktestRequest) -> s.BacktestResponse:
    """Launches a backtest. If it takes over two seconds the run_id returns immediately."""
    spec, run_config, bars, symbol_spec = _prepare(request, request.config)
    server_tz = resolver.server_timezone()
    run_id, fingerprint = plan_run(spec, run_config, bars, symbol_spec.spec)

    if store.exists(run_id) and not request.force:
        meta = store.load_meta(run_id)
        if meta.status == "done":
            return s.BacktestResponse(
                run_id=run_id,
                status="done",
                reused=True,
                message="identical run already present: reused without recomputing",
                poll_url=f"/api/runs/{run_id}",
            )
        if meta.status == "running" and not _is_stale(meta.created_at):
            return s.BacktestResponse(
                run_id=run_id, status="running", reused=True,
                message="identical run already in progress",
                poll_url=f"/api/runs/{run_id}",
            )

    store.begin_run(
        run_id, spec, run_config, fingerprint, len(bars),
        bars.index[0].to_pydatetime(), bars.index[-1].to_pydatetime(),
        symbol_spec,
    )

    assert _executor is not None
    future = _executor.submit(
        execute_run, store, spec, run_config, bars, symbol_spec, server_tz, run_id
    )
    _jobs[run_id] = future

    try:
        meta = future.result(timeout=SYNC_TIMEOUT_SECONDS)
        return s.BacktestResponse(
            run_id=run_id, status=meta.status, reused=False,
            message=f"backtest completed in {meta.duration_seconds:.1f}s",
            poll_url=f"/api/runs/{run_id}",
        )
    except FutureTimeout:
        return s.BacktestResponse(
            run_id=run_id, status="running", reused=False,
            message="backtest in progress: poll poll_url until the status changes",
            poll_url=f"/api/runs/{run_id}",
        )
    except NotImplementedError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (DataUnavailable, EnvironmentUnavailable):
        raise
    except Exception as exc:
        logger.exception("backtest %s failed", run_id)
        raise HTTPException(
            status_code=500, detail=f"backtest failed: {type(exc).__name__}: {exc}"
        ) from exc


def _is_stale(created_at: datetime) -> bool:
    return (datetime.now(timezone.utc) - created_at).total_seconds() > HARD_TIMEOUT_SECONDS


# -- run -----------------------------------------------------------------


@app.get("/api/runs", response_model=list[s.RunSummaryOut], tags=["runs"])
def list_runs(
    symbol: str | None = None,
    strategy_id: str | None = None,
    status: s.RunStatus | None = None,
    limit: int | None = Query(default=None, ge=1, le=500),
) -> list[s.RunSummaryOut]:
    summaries = store.list_runs(symbol=symbol, strategy_id=strategy_id, status=status,
                               limit=limit)
    return [s.RunSummaryOut(**summary.to_dict()) for summary in summaries]


def _load_checked(run_id: str):
    record = store.load_run(run_id)
    if record.meta.status == "running" and _is_stale(record.meta.created_at):
        # the thread cannot be killed: the run is declared failed and whatever
        # it may still produce is discarded
        store.fail_run(
            run_id,
            f"timeout: the run exceeds the allowed {HARD_TIMEOUT_SECONDS:.0f}s",
        )
        record = store.load_run(run_id)
    return record


@app.get("/api/runs/{run_id}", response_model=s.RunDetailOut, tags=["runs"])
def get_run(run_id: str) -> s.RunDetailOut:
    record = _load_checked(run_id)
    metrics = record.metrics or {}
    return s.RunDetailOut(
        run_id=record.run_id,
        status=record.meta.status,
        created_at=record.meta.created_at,
        finished_at=record.meta.finished_at,
        duration_seconds=record.meta.duration_seconds,
        engine_version=record.meta.engine_version,
        spec_hash=record.meta.spec_hash,
        data_hash=record.meta.data_hash,
        symbol_spec_hash=record.meta.symbol_spec_hash,
        bars=record.meta.bars,
        data_start=record.meta.data_start,
        data_end=record.meta.data_end,
        error=record.meta.error,
        spec=record.spec.model_dump(mode="json"),
        config=record.config.to_dict(),
        strategy=s.MetricsOut(**metrics["strategy"]) if metrics.get("strategy") else None,
        benchmark=s.MetricsOut(**metrics["benchmark"]) if metrics.get("benchmark") else None,
        breakeven=s.BreakevenOut(**metrics["breakeven"]) if metrics.get("breakeven") else None,
        execution=s.ExecutionOut(**metrics["execution"]) if metrics.get("execution") else None,
        uncertainty=(
            s.UncertaintyOut(**metrics["uncertainty"]) if metrics.get("uncertainty") else None
        ),
        gates=s.GatesOut(**metrics["gates"]) if metrics.get("gates") else None,
        symbol_spec=(
            s.SymbolSpecOut(**asdict(record.symbol_spec.spec)) if record.symbol_spec else None
        ),
        symbol_spec_read_at=record.symbol_spec.read_at if record.symbol_spec else None,
        symbol_spec_registered=record.symbol_spec_registered,
        costs=s.SpreadRealismOut(**metrics["costs"]) if metrics.get("costs") else None,
        spread_coverage=(
            s.SpreadCoverageOut(**metrics["spread_coverage"])
            if metrics.get("spread_coverage")
            else None
        ),
        aggregated_spread_cost=bool(record.meta.aggregated_spread_cost),
    )


def _drawdown(equity: pd.Series) -> np.ndarray:
    peak = equity.cummax().to_numpy()
    values = equity.to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        relative = np.where(peak > 0, (values - peak) / peak, 0.0)
    return np.nan_to_num(relative, nan=0.0, posinf=0.0, neginf=0.0)


@app.get("/api/runs/{run_id}/equity", response_model=s.EquityOut, tags=["runs"])
def get_equity(
    run_id: str,
    points: int = Query(default=MAX_EQUITY_POINTS, ge=50, le=MAX_EQUITY_POINTS),
) -> s.EquityOut:
    """Equity curve and drawdown, reduced to ~`points` points.

    The drawdown is computed on the **whole** series and only then are the
    points reduced, otherwise the minimum would be measured on an already
    pruned curve.
    """
    record = _load_checked(run_id)
    equity = record.equity()
    if equity.empty:
        raise HTTPException(
            status_code=409,
            detail=f"run {run_id} has no equity curve (status: "
            f"{record.meta.status})",
        )

    drawdown = _drawdown(equity)
    x = equity.index.asi8.astype("float64")
    keep = downsample(x, equity.to_numpy(), points)
    # the vertices that tell the run's story must not vanish in the reduction
    keep = np.unique(np.concatenate([keep, [int(np.argmin(drawdown)),
                                            int(np.argmax(equity.to_numpy()))]]))

    index = equity.index[keep]
    values = equity.to_numpy()[keep]
    drops = drawdown[keep]
    return s.EquityOut(
        run_id=run_id,
        initial_equity=record.config.initial_equity,
        total_points=int(len(equity)),
        returned_points=int(len(keep)),
        downsampled=bool(len(keep) < len(equity)),
        points=[
            s.EquityPoint(t=t.to_pydatetime(), equity=float(v), drawdown=float(d))
            for t, v, d in zip(index, values, drops)
        ],
    )


@app.get("/api/runs/{run_id}/trades", response_model=s.TradesOut, tags=["runs"])
def get_trades(
    run_id: str,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=500),
    ambiguous_only: bool = False,
) -> s.TradesOut:
    record = _load_checked(run_id)
    trades = record.trades()
    if trades.empty:
        return s.TradesOut(
            run_id=run_id, total=0, offset=offset, limit=limit, ambiguous_total=0, items=[]
        )

    trades = trades.reset_index(drop=True)
    trades["index"] = trades.index
    ambiguous_total = int(trades["ambiguous"].sum())
    if ambiguous_only:
        trades = trades[trades["ambiguous"]]

    page = trades.iloc[offset : offset + limit]
    items = [s.TradeOut(**json_safe(row)) for row in page.to_dict(orient="records")]
    return s.TradesOut(
        run_id=run_id,
        total=int(len(trades)),
        offset=offset,
        limit=limit,
        ambiguous_total=ambiguous_total,
        items=items,
    )


@app.delete("/api/runs/{run_id}", response_model=s.DeleteResponse, tags=["runs"])
def delete_run(run_id: str) -> s.DeleteResponse:
    deleted = store.delete_run(run_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"run {run_id} does not exist")
    _jobs.pop(run_id, None)
    return s.DeleteResponse(run_id=run_id, deleted=True)


# -- comparison ----------------------------------------------------------

COMPARE_METRICS: tuple[tuple[str, str, str, bool | None], ...] = (
    ("total_return", "Total return", "pct", True),
    ("annual_return", "Annual return", "pct", True),
    ("final_equity", "Final equity", "money", True),
    ("sharpe", "Sharpe", "number", True),
    ("sortino", "Sortino", "number", True),
    ("max_drawdown_pct", "Max drawdown", "pct", False),
    ("max_drawdown_money", "Max drawdown", "money", False),
    ("calmar", "Calmar", "number", True),
    ("profit_factor", "Profit factor", "number", True),
    ("win_rate", "Win rate", "pct", True),
    ("expectancy", "Expectancy / trade", "money", True),
    ("trades", "Trades", "int", None),
    ("exposure", "Exposure", "pct", None),
    ("max_losing_streak", "Max losing streak", "int", False),
    ("t_stat", "t-stat", "number", True),
    # no "better" direction for the p-value: on a negative mean return a
    # lower p means "more reliably losing"
    ("p_value", "p-value", "number", None),
    ("ambiguous_trades", "Ambiguous trades", "int", False),
)


@app.post("/api/runs/compare", response_model=s.CompareResponse, tags=["runs"])
def compare_runs(request: s.CompareRequest) -> s.CompareResponse:
    """Curves aligned on the same axis and metrics side by side.

    The curves are normalized to 100 at the start of their own period: two
    runs with different initial equity, or over different periods, stay
    comparable.
    """
    records = []
    warnings: list[str] = []
    for run_id in request.run_ids:
        record = _load_checked(run_id)
        if record.meta.status != "done":
            raise HTTPException(
                status_code=409,
                detail=f"run {run_id} is in status {record.meta.status}: "
                f"not comparable",
            )
        records.append(record)

    symbols = {r.config.symbol for r in records}
    if len(symbols) > 1:
        warnings.append(
            f"runs on different instruments ({', '.join(sorted(symbols))}): the "
            f"curves do not measure the same thing"
        )
    periods = {(r.meta.data_start, r.meta.data_end) for r in records}
    if len(periods) > 1:
        warnings.append(
            "different periods: the non-overlapping part is not a comparison"
        )

    curves = [record.equity() for record in records]
    if any(curve.empty for curve in curves):
        raise HTTPException(status_code=409, detail="at least one run has no equity curve")

    axis_start = min(curve.index[0] for curve in curves)
    axis_end = max(curve.index[-1] for curve in curves)
    # rounded to the microsecond: Python datetimes have no nanoseconds
    axis = pd.date_range(axis_start, axis_end, periods=request.points, tz="UTC").round("us")

    aligned: list[np.ndarray] = []
    for curve in curves:
        # ffill: outside its own period the curve does not exist and stays
        # null, instead of being drawn flat as if the run were idle
        values = curve.reindex(curve.index.union(axis)).ffill().reindex(axis)
        series = values.to_numpy(dtype="float64")
        if request.normalize:
            first = curve.iloc[0]
            series = series / first * 100.0 if first else series
        aligned.append(series)

    series_points = [
        s.ComparePoint(
            t=moment.to_pydatetime(),
            values=[
                None if not np.isfinite(column[i]) else float(column[i])
                for column in aligned
            ],
        )
        for i, moment in enumerate(axis)
    ]

    metrics: list[s.CompareMetricRow] = []
    for key, label, fmt, higher in COMPARE_METRICS:
        values = [
            _metric_value(record.metrics.get("strategy") or {}, key) for record in records
        ]
        reference = values[0]
        delta = [
            None if (v is None or reference is None) else v - reference for v in values
        ]
        metrics.append(
            s.CompareMetricRow(
                key=key, label=label, format=fmt, higher_is_better=higher,
                values=values, delta=delta,
            )
        )

    # break-even win rate: only where the outcome distribution is binary
    # (metrics["breakeven"].valid); elsewhere the value stays null on purpose
    for key, label, higher in (
        ("breakeven_win_rate", "Break-even win rate", None),
        ("delta", "Win rate - break-even", True),
    ):
        values = [
            _metric_value(record.metrics.get("breakeven") or {}, key)
            if (record.metrics.get("breakeven") or {}).get("valid")
            else None
            for record in records
        ]
        reference = values[0]
        delta = [
            None if (v is None or reference is None) else v - reference for v in values
        ]
        metrics.append(
            s.CompareMetricRow(
                key=f"breakeven_{key}" if key == "delta" else key,
                label=label, format="pct", higher_is_better=higher,
                values=values, delta=delta,
            )
        )

    config_diff, configs_identical = _config_diff(records)
    symbol_spec_diff, symbol_specs_identical = _symbol_spec_diff(records)
    if any(not record.symbol_spec_registered for record in records):
        warnings.append(
            "at least one run predates SymbolSpec pinning (A1): its cost fields "
            "cannot be compared and must not be assumed identical to the others"
        )

    return s.CompareResponse(
        runs=[
            s.CompareRunOut(
                run_id=record.run_id,
                label=f"{record.spec.id} · {record.config.symbol} "
                f"{record.config.timeframe} · {record.run_id[:8]}",
                strategy_id=record.spec.id,
                symbol=record.config.symbol,
                timeframe=record.config.timeframe,
                start=record.meta.data_start,
                end=record.meta.data_end,
                initial_equity=record.config.initial_equity,
                symbol_spec_registered=record.symbol_spec_registered,
            )
            for record in records
        ],
        normalized=request.normalize,
        axis_start=axis_start.to_pydatetime(),
        axis_end=axis_end.to_pydatetime(),
        series=series_points,
        metrics=metrics,
        config_diff=config_diff,
        configs_identical=configs_identical,
        symbol_spec_diff=symbol_spec_diff,
        symbol_specs_identical=symbol_specs_identical,
        warnings=warnings,
    )


def _config_diff(records) -> tuple[list[s.CompareConfigRow], bool]:
    """Fields of config.json that differ between the compared runs.

    Generic over the whole config: any field added later shows up here without
    touching this code. The strategy id is included because two runs of
    different strategies differ in more than their config.
    """
    payloads = [
        {"strategy_id": record.spec.id, **record.config.to_dict()}
        for record in records
    ]
    keys: list[str] = []
    for payload in payloads:
        for key in payload:
            if key not in keys:
                keys.append(key)
    rows: list[s.CompareConfigRow] = []
    for key in keys:
        values = [payload.get(key) for payload in payloads]
        if any(value != values[0] for value in values[1:]):
            rows.append(
                s.CompareConfigRow(
                    key=key,
                    values=[None if v is None else str(v) for v in values],
                )
            )
    return rows, not rows


def _symbol_spec_diff(records) -> tuple[list[s.CompareConfigRow], bool]:
    """SymbolSpec fields that differ between the compared runs.

    An unregistered run (no symbol_spec.json, pre-A1) contributes `None` for
    every field: it never claims agreement with the others, since we have no
    pinned record of what it actually used.
    """
    payloads = [
        asdict(record.symbol_spec.spec) if record.symbol_spec else {}
        for record in records
    ]
    keys: list[str] = []
    for payload in payloads:
        for key in payload:
            if key not in keys:
                keys.append(key)
    rows: list[s.CompareConfigRow] = []
    for key in keys:
        values = [payload.get(key) for payload in payloads]
        if any(value != values[0] for value in values[1:]):
            rows.append(
                s.CompareConfigRow(
                    key=key,
                    values=[None if v is None else str(v) for v in values],
                )
            )
    identical = not rows and all(record.symbol_spec is not None for record in records)
    return rows, identical


def _metric_value(metrics: dict[str, Any], key: str) -> float | None:
    value = metrics.get(key)
    if value is None or isinstance(value, str):
        return None
    number = float(value)
    return number if np.isfinite(number) else None


# -- validation ----------------------------------------------------------


def _pinned_symbol_spec(record):
    """The SymbolSpec the run actually used, not whatever the broker says today.

    A run pins its SymbolSpec at execution time (A1): reusing the live
    resolver here for revalidation would reintroduce the exact drift the
    pinning exists to prevent (swap rates or tick_value moving between the
    original run and a later walk-forward/permutation/tick-resolve on it).
    Only a run older than the pinning mechanism (no symbol_spec.json) falls
    back to the resolver, and it does so best-effort.
    """
    if record.symbol_spec is not None:
        return record.symbol_spec.spec
    return resolver.symbol_spec(record.config.symbol)


def _run_context(run_id: str):
    """The pieces every validation endpoint needs: spec, config, bars, symbol."""
    record = _load_checked(run_id)
    if record.meta.status != "done":
        raise HTTPException(
            status_code=409,
            detail=f"run {run_id} is in status {record.meta.status}: "
            f"there is nothing to validate yet",
        )
    bars = load_bars(
        cache, record.config.symbol, record.config.tf, record.config.start, record.config.end
    )
    symbol_spec = _pinned_symbol_spec(record)
    return record, bars, symbol_spec


def _backtest_config(record) -> BacktestConfig:
    return BacktestConfig(
        initial_equity=record.config.initial_equity,
        costs=record.config.cost_model(),
        session_threshold=record.config.session_threshold,
    )


@app.post(
    "/api/validation/walkforward",
    response_model=s.WalkForwardResponse,
    tags=["validation"],
)
def post_walkforward(request: s.WalkForwardRequest) -> s.WalkForwardResponse:
    """Optimize in-sample, apply out-of-sample, one window at a time."""
    record, bars, symbol_spec = _run_context(request.run_id)
    server_tz = resolver.server_timezone()

    started = time.perf_counter()
    try:
        report = walk_forward(
            record.spec,
            bars,
            symbol_spec,
            server_tz,
            _backtest_config(record),
            WalkForwardConfig(
                mode=request.mode,
                train_days=request.train_days,
                test_days=request.test_days,
                min_train_trades=request.min_train_trades,
                objective=request.objective,
            ),
            grid=request.grid,
            reference_trades=record.trades(),
        )
    except WalkForwardError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    logger.info("walk-forward %s in %.1fs", request.run_id, time.perf_counter() - started)

    payload = report.as_dict()
    payload["run_id"] = request.run_id
    payload["oos_curve"] = _curve_points(report.oos_equity)
    return s.WalkForwardResponse(**payload)


def _curve_points(equity: pd.Series, points: int = MAX_EQUITY_POINTS) -> list[dict[str, Any]]:
    """The concatenated OOS curve, reduced but keeping its turning points."""
    if not len(equity):
        return []
    values = equity.to_numpy(dtype="float64")
    x = equity.index.asi8.astype("float64")
    keep = downsample(x, values, points)
    keep = np.unique(
        np.concatenate([keep, [int(np.argmin(values)), int(np.argmax(values))]])
    )
    return [
        {"t": equity.index[i].to_pydatetime(), "equity": float(values[i])} for i in keep
    ]


@app.post(
    "/api/validation/permutation",
    response_model=s.PermutationResponse,
    tags=["validation"],
)
def post_permutation(request: s.PermutationRequest) -> s.PermutationResponse:
    """The real result against random entries and against permuted price paths."""
    record, bars, symbol_spec = _run_context(request.run_id)
    server_tz = resolver.server_timezone()
    config = _backtest_config(record)

    tests: list[s.PermutationTestOut] = []
    for kind in request.tests:
        report = permutation_test(
            kind,
            record.spec,
            bars,
            symbol_spec,
            server_tz,
            config,
            iterations=request.iterations,
            block_bars=request.block_bars,
            seed=request.seed,
        )
        tests.append(s.PermutationTestOut(**report.as_dict()))

    return s.PermutationResponse(
        run_id=request.run_id, symbol=record.config.symbol, tests=tests
    )


@app.post(
    "/api/validation/multiple-testing",
    response_model=s.MultipleTestingResponse,
    tags=["validation"],
)
def post_multiple_testing(request: s.MultipleTestingRequest) -> s.MultipleTestingResponse:
    """What the observed Sharpe is worth given how many configurations were tried."""
    record = _load_checked(request.run_id)
    if record.meta.status != "done":
        raise HTTPException(
            status_code=409,
            detail=f"run {request.run_id} is in status {record.meta.status}",
        )

    # the family is every distinct spec tried on this symbol over this same
    # data: a different period or a different instrument is a different
    # question and does not belong in the correction
    siblings = [
        store.load_run(summary.run_id)
        for summary in store.list_runs(symbol=record.config.symbol, status="done")
    ]
    family = [
        sibling
        for sibling in siblings
        if sibling.meta.data_hash == record.meta.data_hash
    ]
    trials = collect_trials(family)

    trades = record.trades()
    pnl = trades["net_pnl"].astype("float64").to_numpy() if len(trades) else np.zeros(0)
    annualized = ((record.metrics or {}).get("strategy") or {}).get("sharpe")

    matrix = None
    if request.grid:
        try:
            matrix, grid_trials = _grid_trials(record, request.grid)
        except WalkForwardError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        # a grid candidate is an attempt on this data exactly like a stored
        # run is: leaving it out would deflate against a trial count the
        # search did not actually have
        known = {trial.spec_hash for trial in trials}
        trials = trials + [trial for trial in grid_trials if trial.spec_hash not in known]

    report = multiple_testing_report(
        run_id=request.run_id,
        symbol=record.config.symbol,
        period_start=record.meta.data_start,
        period_end=record.meta.data_end,
        observed_pnl=pnl,
        observed_annualized_sharpe=annualized,
        trials_detail=trials,
        alpha=request.alpha,
        performance_matrix=matrix,
    )
    return s.MultipleTestingResponse(**report.as_dict())


def _grid_trials(record, grid: dict[str, list[Any]]) -> tuple[np.ndarray, list[Trial]]:
    """Runs every grid candidate once: the CSCV matrix and the trial family.

    The matrix is (day x candidate) net PnL - days are the natural slice, the
    unit the strategy actually trades in. The same backtests also produce the
    per-trade Sharpe of each candidate, which is what the deflation needs to
    know how much of the winner's score the search bought for free.
    """
    bars = load_bars(
        cache, record.config.symbol, record.config.tf, record.config.start, record.config.end
    )
    symbol_spec = _pinned_symbol_spec(record)
    server_tz = resolver.server_timezone()
    config = _backtest_config(record)

    days = pd.date_range(bars.index[0].normalize(), bars.index[-1].normalize(), freq="D")
    columns: list[np.ndarray] = []
    trials: list[Trial] = []
    for params in expand_grid(grid):
        candidate = apply_params(record.spec, params)
        result = run_backtest(candidate, bars, symbol_spec, server_tz, config)
        series = pd.Series(0.0, index=days)
        pnl = np.zeros(0)
        if len(result.trades):
            pnl = result.trades["net_pnl"].astype("float64").to_numpy()
            by_day = (
                result.trades.assign(
                    day=pd.to_datetime(result.trades["exit_time"], utc=True).dt.normalize()
                )
                .groupby("day")["net_pnl"]
                .sum()
            )
            series = series.add(by_day.reindex(days).fillna(0.0), fill_value=0.0)
        columns.append(series.to_numpy(dtype="float64"))
        trials.append(
            Trial(
                run_id=f"grid:{spec_hash(candidate)[:8]}",
                strategy_id=candidate.id,
                spec_hash=spec_hash(candidate),
                trades=int(len(result.trades)),
                sharpe_per_trade=sharpe_per_trade(pnl),
                net_pnl=float(pnl.sum()) if len(pnl) else 0.0,
                p_value=_trade_p_value(pnl),
            )
        )
    return np.column_stack(columns), trials


def _trade_p_value(pnl: np.ndarray) -> float | None:
    """Two-sided t-test on the mean trade PnL, the same one the metrics use."""
    if len(pnl) < 2:
        return None
    deviation = float(pnl.std(ddof=1))
    if deviation <= 0:
        return None
    t_stat = float(pnl.mean() / (deviation / np.sqrt(len(pnl))))
    return float(2 * (1 - stats.t.cdf(abs(t_stat), df=len(pnl) - 1)))


@app.post(
    "/api/validation/tick-resolve",
    response_model=s.TickResolveResponse,
    tags=["validation"],
)
def post_tick_resolve(request: s.TickResolveRequest) -> s.TickResolveResponse:
    """Ambiguous trades settled with tick data, or a plain statement that they cannot be."""
    record = _load_checked(request.run_id)
    if record.meta.status != "done":
        raise HTTPException(
            status_code=409,
            detail=f"run {request.run_id} is in status {record.meta.status}",
        )
    symbol_spec = _pinned_symbol_spec(record)

    try:
        from core.data.mt5_provider import MT5Provider

        with MT5Provider(server_timezone=resolver.server_timezone()) as provider:
            report = resolve_ambiguous(
                run_id=request.run_id,
                spec=record.spec,
                trades=record.trades(),
                symbol_spec=symbol_spec,
                provider=provider,
                timeframe=record.config.tf,
                initial_equity=record.config.initial_equity,
            )
    except Exception as exc:
        # no terminal, no tick history, no connection: that is an answer, and
        # it is reported as one. Nothing is assumed in its place.
        logger.warning("tick resolve unavailable for %s: %s", request.run_id, exc)
        return s.TickResolveResponse(
            run_id=request.run_id,
            symbol=record.config.symbol,
            available=False,
            reason=(
                f"no tick source available ({type(exc).__name__}: {exc}). Start "
                f"MetaTrader 5 and leave it connected to retry. The conservative "
                f"stop assumption stands, untested"
            ),
            ambiguous_trades=int(
                ((record.metrics or {}).get("execution") or {}).get("ambiguous_trades") or 0
            ),
            resolved=0,
            unresolved=0,
            resolved_to_take_profit=0,
            resolved_to_stop_loss=0,
            original_net_pnl=0.0,
            resolved_net_pnl=0.0,
            delta_net_pnl=0.0,
            verdict=(
                "Tick data is not reachable, so the ambiguous trades stay as the "
                "engine assumed them: stops. That assumption is untested here, not "
                "confirmed."
            ),
        )
    return s.TickResolveResponse(**report.as_dict())


# -- batch ---------------------------------------------------------------


@app.post("/api/batch", response_model=s.BatchResponse, tags=["batch"])
def post_batch(request: s.BatchRequest) -> s.BatchResponse:
    """The same spec across many instruments, and whether they agree."""
    spec = _resolve_spec(request)
    base_config = _run_config(request.config)

    # the batch workers read the spec from disk, so an inline spec has to be
    # written down before they can see it
    spec_path = STRATEGIES_DIR / f"{spec.id}.json"
    temporary: Path | None = None
    if not spec_path.exists():
        temporary = Path(tempfile.gettempdir()) / f"batch-{spec.id}-{os.getpid()}.json"
        spec.save(temporary)
        spec_path = temporary

    try:
        report = run_batch(
            spec_path=spec_path,
            symbols=request.symbols,
            base_config=base_config,
            periods=[
                BatchPeriod(start=period.start, end=period.end)
                for period in request.periods
            ]
            or None,
            grid=request.grid,
            cache_dir=CACHE_DIR,
            runs_dir=RUNS_DIR,
            max_workers=request.max_workers,
            consistency_metric=request.consistency_metric,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

    return s.BatchResponse(**report.as_dict())


# -- screening campaign --------------------------------------------------

# A campaign is minutes of work, not seconds: it runs as a job and the client
# polls it. Kept in memory on purpose - the artefact worth keeping is the
# report the caller saves, and every run inside it is already in the store.
_screens: dict[str, dict[str, Any]] = {}
SCREEN_HISTORY = 20


def _run_screen_job(job_id: str, request: s.ScreenRequest) -> None:
    job = _screens[job_id]

    def progress(index: int, total: int, outcome: Any) -> None:
        job["completed_cells"] = index
        job["total_cells"] = total
        job["current"] = (
            f"{outcome.strategy_id} · {outcome.symbol} {outcome.timeframe}"
        )

    try:
        paths = []
        for strategy_id in request.strategy_ids:
            path = STRATEGIES_DIR / f"{strategy_id}.json"
            if not path.exists():
                raise FileNotFoundError(f"no strategy file for id {strategy_id!r}")
            paths.append(path)

        report = run_screen(
            strategies=paths,
            symbols=request.symbols,
            timeframes=request.timeframes,
            base_config=_run_config(request.config),
            cache_dir=CACHE_DIR,
            runs_dir=RUNS_DIR,
            min_trades=request.min_trades,
            permutation_iterations=request.permutation_iterations,
            on_cell=progress,
        )
        job["report"] = report.as_dict()
        job["status"] = "done"
    except Exception as exc:  # a failed campaign is a status, not a 500
        logger.exception("screening job %s failed", job_id)
        job["status"] = "error"
        job["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        job["finished_at"] = datetime.now(timezone.utc)


@app.post("/api/screen", response_model=s.ScreenJobOut, tags=["research"])
def post_screen(request: s.ScreenRequest) -> s.ScreenJobOut:
    """Starts a screening campaign and returns the job to poll."""
    total = len(request.strategy_ids) * len(request.symbols) * len(request.timeframes)
    job_id = f"screen-{int(time.time() * 1000):x}"
    _screens[job_id] = {
        "job_id": job_id,
        "status": "running",
        "started_at": datetime.now(timezone.utc),
        "finished_at": None,
        "completed_cells": 0,
        "total_cells": total,
        "current": None,
        "error": None,
        "report": None,
    }

    # oldest jobs are dropped: the reports live in the client, the runs in the store
    for stale in list(_screens)[:-SCREEN_HISTORY]:
        _screens.pop(stale, None)

    assert _executor is not None
    _executor.submit(_run_screen_job, job_id, request)
    return s.ScreenJobOut(**_screens[job_id])


@app.get("/api/screen/{job_id}", response_model=s.ScreenJobOut, tags=["research"])
def get_screen(job_id: str) -> s.ScreenJobOut:
    job = _screens.get(job_id)
    if job is None:
        raise HTTPException(
            status_code=404,
            detail=f"no screening job {job_id}: campaigns are kept in memory and "
            f"the last {SCREEN_HISTORY} are available until the backend restarts",
        )
    return s.ScreenJobOut(**job)


# -- tradability ---------------------------------------------------------


@app.get("/api/tradability", response_model=s.TradabilityOut, tags=["research"])
def get_tradability(
    symbols: str = Query(
        default="", description="comma-separated; default is everything cached"
    ),
    timeframes: str = Query(default="M5,M15,H1,H4,D1"),
    max_spread_atr: float = Query(default=DEFAULT_MAX_SPREAD_ATR, gt=0, le=1.0),
) -> s.TradabilityOut:
    """Stage zero: which instrument x timeframe pairs are worth testing at all.

    The spread comes from each instrument's M1 sample, never from the bars
    being judged: above M1 that column is the minimum spread inside the bar,
    which on this broker is zero on most FX hours.
    """
    wanted = [name.strip() for name in symbols.split(",") if name.strip()]
    wanted = wanted or _cached_symbols()
    frames = [name.strip() for name in timeframes.split(",") if name.strip()]
    points: dict[str, float] = {}
    for name in wanted:
        try:
            points[name] = resolver.symbol_spec(name).point
        except EnvironmentUnavailable:
            logger.warning("%s: no spec, left out of the tradability table", name)

    table = build_tradability(cache, wanted, frames, points, max_ratio=max_spread_atr)
    return s.TradabilityOut(**table.as_dict())


# -- the live runner -----------------------------------------------------


@app.get("/api/live", response_model=list[s.LiveSessionOut], tags=["live"])
def list_live_sessions() -> list[s.LiveSessionOut]:
    """Every diary under the live directory, most recent activity first.

    The backend does not run the runner: `scripts.run_live` does, in its own
    process, with its own lock. This reads what that process wrote, so the
    dashboard can be opened and closed without touching a running strategy.
    """
    if not LIVE_DIR.exists():
        return []
    epoch = datetime.min.replace(tzinfo=timezone.utc)
    return sorted(
        (_live_session(path) for path in sorted(LIVE_DIR.glob("*.jsonl"))),
        key=lambda session: session.last_event_at or epoch,
        reverse=True,
    )


@app.get("/api/live/{session_id}", response_model=s.LiveDetailOut, tags=["live"])
def get_live_session(
    session_id: str,
    events: int = Query(default=100, ge=1, le=1000),
) -> s.LiveDetailOut:
    """One runner: its state, its recent diary, and its closed trades."""
    path = _live_path(session_id)
    journal = Journal(path)
    trades = journal.trades()

    # A runner on H1 writes one `bar` line an hour and an order line once a
    # week. Returning the last N lines would push every decision out of the
    # window and leave a diary that shows only heartbeats, so the two are
    # tailed separately and merged back in order.
    everything = list(journal.events())
    bars = [event for event in everything if event.kind == "bar"][-events:]
    decisions = [event for event in everything if event.kind != "bar"][-events:]
    window = sorted(bars + decisions, key=lambda event: event.at)

    return s.LiveDetailOut(
        session=_live_session(path),
        events=[
            s.LiveEventOut(
                at=event.at,
                kind=event.kind,
                bar_time=event.bar_time,
                detail=json_safe(event.detail),
            )
            for event in reversed(window)
        ],
        trades=[s.LiveTradeOut(**json_safe(row)) for row in trades.to_dict("records")],
    )


@app.get(
    "/api/live/{session_id}/comparison",
    response_model=s.LiveComparisonOut,
    tags=["live"],
)
def get_live_comparison(session_id: str) -> s.LiveComparisonOut:
    """Expected versus realized: a backtest of exactly the diary's own period.

    The backtest is computed here rather than looked up, over the bars the
    runner actually saw, so the two records describe the same period by
    construction. Comparing against a backtest of a different window would
    measure the window.
    """
    path = _live_path(session_id)
    journal = Journal(path)
    session = _live_session(path)
    if not session.strategy_id or session.first_bar is None:
        raise HTTPException(
            status_code=409,
            detail=(
                f"the diary {session_id} holds no start event naming a strategy, "
                f"or no processed bar: there is nothing to reproduce a backtest "
                f"from"
            ),
        )

    spec = apply_params(
        _strategy_by_id(session.strategy_id),
        {
            "instrument.symbol": session.symbol,
            "instrument.timeframe": session.timeframe,
        },
    )
    tf = _timeframe(session.timeframe)
    end = (session.last_bar or session.first_bar) + timedelta(minutes=tf.minutes)
    bars = load_bars(cache, session.symbol, tf, session.first_bar, end)
    symbol_spec = resolver.symbol_spec(session.symbol)

    expected = run_backtest(
        spec,
        bars,
        symbol_spec,
        resolver.server_timezone(),
        BacktestConfig(initial_equity=session.initial_equity or 100.0),
    )
    report = live_compare(expected.trades, journal, symbol_spec, session.timeframe)
    return s.LiveComparisonOut(**report.as_dict())


def _live_path(session_id: str) -> Path:
    if "/" in session_id or "\\" in session_id or ".." in session_id:
        raise HTTPException(status_code=400, detail=f"invalid session id: {session_id}")
    path = LIVE_DIR / f"{session_id}.jsonl"
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                f"no live diary {session_id}. Diaries are written by "
                f"scripts.run_live into {LIVE_DIR}"
            ),
        )
    return path


def _strategy_by_id(strategy_id: str) -> StrategySpec:
    for path in sorted(STRATEGIES_DIR.glob("*.json")):
        try:
            spec = StrategySpec.from_json(path)
        except SpecError:
            continue
        if spec.id == strategy_id:
            return spec
    raise HTTPException(
        status_code=404,
        detail=(
            f"the diary names strategy {strategy_id}, which is not in "
            f"{STRATEGIES_DIR}: the spec it ran with is no longer available"
        ),
    )


def _live_session(path: Path) -> s.LiveSessionOut:
    """The header facts of one diary, read in a single pass."""
    journal = Journal(path)
    started: dict[str, Any] = {}
    symbol = timeframe = ""
    first_bar = last_bar = last_event = None
    bars = trades = errors = 0
    stopped = False

    for event in journal.events():
        symbol = event.symbol or symbol
        timeframe = event.timeframe or timeframe
        last_event = event.at
        if event.kind == "started":
            started = event.detail
        elif event.kind == "bar":
            bars += 1
            if first_bar is None:
                first_bar = event.bar_time
            last_bar = event.bar_time
        elif event.kind == "position_closed":
            trades += 1
        elif event.kind == "error":
            errors += 1
        elif event.kind == "stopped":
            stopped = True

    lock = LIVE_DIR / f"{path.stem}.lock"
    info = RunLock(lock).read()
    return s.LiveSessionOut(
        has_lock=lock.exists(),
        session_id=path.stem,
        symbol=symbol,
        timeframe=timeframe,
        strategy_id=started.get("strategy_id"),
        engine_version=started.get("engine_version"),
        dry_run=bool(started.get("dry_run", True)),
        account_guard=started.get("account_guard"),
        initial_equity=started.get("initial_equity"),
        bars_processed=bars,
        trades=trades,
        errors=errors,
        first_bar=first_bar,
        last_bar=last_bar,
        last_event_at=last_event,
        stopped=stopped,
        running=bool(info and process_alive(info.pid)),
        pid=info.pid if info else None,
    )


@app.get("/api/health", tags=["data"])
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "engine_version": ENGINE_VERSION,
        # the UI banners on this: invented data has to announce itself
        "synthetic_fixture": IS_FIXTURE,
        "cache_dir": project_relative(CACHE_DIR),
        "runs_dir": project_relative(RUNS_DIR),
        "strategies_dir": project_relative(STRATEGIES_DIR),
        "live_dir": project_relative(LIVE_DIR),
        "runs": len(store.list_runs(limit=500)),
    }
