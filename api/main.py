"""Local API of the backtest engine.

Runs only on 127.0.0.1, with no authentication, no database, no external
services. It is the interface between the engine and the dashboard, and it
contains no computation logic: everything it returns was computed by `core`.

Errors leave with the right HTTP status and a readable sentence. A stack
trace in the browser helps nobody: it goes to the server log, where it
belongs.
"""
from __future__ import annotations

import logging
import os
import tempfile
import time
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from scipy import stats
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.requests import Request

from api import schemas as s
from api.downsample import downsample
from core.batch.runner import Period as BatchPeriod
from core.batch.runner import run_batch
from core.data.cache import ParquetCache
from core.data.provider import Timeframe
from core.data.quality import check_quality
from core.engine.backtester import BacktestConfig, run_backtest
from core.runs.runner import (
    DataUnavailable,
    EnvironmentUnavailable,
    SymbolResolver,
    cached_years,
    execute_run,
    load_bars,
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

CACHE_DIR = Path(os.environ.get("BACKTEST_CACHE_DIR", "data_cache"))
RUNS_DIR = Path(os.environ.get("BACKTEST_RUNS_DIR", "runs"))
STRATEGIES_DIR = Path(os.environ.get("BACKTEST_STRATEGIES_DIR", "strategies"))

cache = ParquetCache(CACHE_DIR)
store = RunStore(RUNS_DIR)
resolver = SymbolResolver(cache)
_executor: ThreadPoolExecutor | None = None
_jobs: dict[str, Future] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _executor
    _executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="backtest")
    logger.info("cache=%s runs=%s strategies=%s", CACHE_DIR, RUNS_DIR, STRATEGIES_DIR)
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
        computed = check_quality(bars, symbol, tf)
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
            text=computed.as_text(),
            worst_gaps=[
                s.GapOut(start=g.start, end=g.end, missing_bars=g.missing_bars) for g in worst
            ],
        )

    return s.CoverageOut(
        symbol=symbol,
        timeframe=tf.name,
        years=years,
        start=bars.index[0].to_pydatetime(),
        end=bars.index[-1].to_pydatetime(),
        bars=len(bars),
        quality=report,
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
    )


def _prepare(request: s.StrategyRef, config: s.RunConfigIn):
    spec = _resolve_spec(request)
    run_config = _run_config(config)
    bars = load_bars(
        cache, run_config.symbol, run_config.tf, run_config.start, run_config.end
    )
    if len(bars) > MAX_BARS:
        raise HTTPException(
            status_code=413,
            detail=f"{len(bars)} bars exceed the cap of {MAX_BARS}: "
            f"narrow the period",
        )
    symbol_spec = resolver.symbol_spec(run_config.symbol)
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
        spec, run_config, bars, symbol_spec, request.horizons, request.min_observations
    )
    logger.info("gate zero in %.2fs", time.perf_counter() - started)
    return s.EdgeResponse(**report.as_dict())


# -- backtest ------------------------------------------------------------


@app.post("/api/backtest", response_model=s.BacktestResponse, tags=["backtest"])
def post_backtest(request: s.BacktestRequest) -> s.BacktestResponse:
    """Launches a backtest. If it takes over two seconds the run_id returns immediately."""
    spec, run_config, bars, symbol_spec = _prepare(request, request.config)
    server_tz = resolver.server_timezone()
    run_id, fingerprint = plan_run(spec, run_config, bars)

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


def _metric_value(metrics: dict[str, Any], key: str) -> float | None:
    value = metrics.get(key)
    if value is None or isinstance(value, str):
        return None
    number = float(value)
    return number if np.isfinite(number) else None


# -- validation ----------------------------------------------------------


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
    symbol_spec = resolver.symbol_spec(record.config.symbol)
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
    symbol_spec = resolver.symbol_spec(record.config.symbol)
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
    symbol_spec = resolver.symbol_spec(record.config.symbol)

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


@app.get("/api/health", tags=["data"])
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "cache_dir": str(CACHE_DIR.resolve()),
        "runs_dir": str(RUNS_DIR.resolve()),
        "strategies_dir": str(STRATEGIES_DIR.resolve()),
        "runs": len(store.list_runs(limit=500)),
    }
