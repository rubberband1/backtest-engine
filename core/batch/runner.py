"""Batch execution over symbols, periods and parameter combinations.

Every cell of the batch is a normal run: it goes through the same engine and
lands in the same store with the same deterministic id. Nothing here is a
shortcut around `core.runs.runner` - the batch is an orchestrator, not a
second implementation.

What the batch adds is the only thing a single run cannot give: a
**cross-sectional** view. Seventy-two long trades over nine months carry a
standard error wide enough to swallow any edge this size, so a single-symbol
result is not evidence in either direction. Ten symbols are ten more or less
independent draws of the same question, and the distribution of their answers
is a measurement where one of them was only an anecdote.

The consistency block is therefore the point of this module, not a summary
bolted onto it: how many instruments agree on the sign, how wide the spread
is, and whether a positive aggregate is carried by a single outlier.
"""
from __future__ import annotations

import logging
import os
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from core.data.cache import ParquetCache
from core.runs.runner import (
    SymbolResolver,
    execute_run,
    load_bars_for_run,
    plan_run,
)
from core.runs.store import RunConfig, RunStore
from core.serialization import json_safe
from core.strategy.binding import bind_cell
from core.strategy.spec import StrategySpec
from core.validation.walkforward import ParameterGrid, apply_params, expand_grid

logger = logging.getLogger(__name__)

# Below this many instruments the cross-sectional test has no power: the sign
# count is compatible with a coin toss whatever it says.
MIN_SYMBOLS_FOR_CONSISTENCY = 3
# A worker holds a full bar series in memory; past this the machine swaps and
# the batch gets slower, not faster.
MAX_WORKERS_CAP = 8


@dataclass(frozen=True)
class Period:
    start: datetime | None = None
    end: datetime | None = None

    def as_dict(self) -> dict[str, Any]:
        return json_safe(asdict(self))


@dataclass(frozen=True)
class BatchCell:
    """One unit of work: a spec, an instrument and a period."""

    symbol: str
    period: Period
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class CellResult:
    symbol: str
    params: dict[str, Any]
    period_start: datetime | None
    period_end: datetime | None
    run_id: str | None
    status: str
    error: str | None = None
    bars: int = 0
    trades: int = 0
    net_pnl: float | None = None
    expectancy: float | None = None
    expectancy_stderr: float | None = None
    mean_r: float | None = None
    mean_r_stderr: float | None = None
    sharpe: float | None = None
    profit_factor: float | None = None
    win_rate: float | None = None
    max_drawdown_pct: float | None = None
    final_equity: float | None = None
    p_value: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return json_safe(asdict(self))


@dataclass
class Consistency:
    """Whether the instruments agree, and with how much power."""

    metric: str
    observations: int
    positive: int
    negative: int
    zero_or_missing: int
    mean: float | None
    median: float | None
    std: float | None
    stderr: float | None
    minimum: float | None
    maximum: float | None
    sign_p_value: float | None
    carried_by_one_symbol: bool
    dominant_symbol: str | None
    verdict: str

    def as_dict(self) -> dict[str, Any]:
        return json_safe(asdict(self))


@dataclass
class BatchReport:
    strategy_id: str
    symbols: list[str]
    periods: list[Period]
    grid: ParameterGrid
    grid_size: int
    cells: list[CellResult]
    completed: int
    failed: int
    elapsed_seconds: float
    consistency: Consistency
    verdict: str
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "strategy_id": self.strategy_id,
            "symbols": self.symbols,
            "periods": [period.as_dict() for period in self.periods],
            "grid": self.grid,
            "grid_size": self.grid_size,
            "cells": [cell.as_dict() for cell in self.cells],
            "completed": self.completed,
            "failed": self.failed,
            "elapsed_seconds": self.elapsed_seconds,
            "consistency": self.consistency.as_dict(),
            "verdict": self.verdict,
            "warnings": self.warnings,
        }
        return json_safe(payload)


# -- worker --------------------------------------------------------------

_CONTEXT: dict[str, Any] = {}


def _init_worker(
    spec_json: str, config_payload: dict[str, Any], cache_dir: str, runs_dir: str
) -> None:
    _CONTEXT["spec"] = StrategySpec.from_json(spec_json)
    _CONTEXT["config"] = config_payload
    _CONTEXT["cache"] = ParquetCache(Path(cache_dir))
    _CONTEXT["store"] = RunStore(Path(runs_dir))
    _CONTEXT["resolver"] = SymbolResolver(_CONTEXT["cache"])


def _execute_cell(cell: BatchCell) -> CellResult:
    """Runs one cell. A failure here is data, not an exception to propagate.

    One missing instrument must not take the batch down with it: the cell is
    recorded as failed with its reason and the aggregate says how many of
    them there were.
    """
    result = CellResult(
        symbol=cell.symbol,
        params=cell.params,
        period_start=cell.period.start,
        period_end=cell.period.end,
        run_id=None,
        status="error",
    )
    try:
        spec = apply_params(_CONTEXT["spec"], cell.params)
        payload = dict(_CONTEXT["config"])
        payload["symbol"] = cell.symbol
        payload["start"] = cell.period.start
        payload["end"] = cell.period.end
        config = RunConfig(**payload)
        bound = bind_cell(spec, cell.symbol, config.timeframe)

        cache: ParquetCache = _CONTEXT["cache"]
        store: RunStore = _CONTEXT["store"]
        resolver: SymbolResolver = _CONTEXT["resolver"]

        bars = load_bars_for_run(cache, config)
        symbol_spec = resolver.symbol_spec_snapshot(cell.symbol)
        server_tz = resolver.server_timezone()
        run_id, _ = plan_run(bound, config, bars, symbol_spec.spec)
        result.run_id = run_id
        result.bars = int(len(bars))

        if store.exists(run_id) and store.load_meta(run_id).status == "done":
            record = store.load_run(run_id)
        else:
            execute_run(store, bound, config, bars, symbol_spec, server_tz, run_id)
            record = store.load_run(run_id)

        return _summarize(result, record)
    except Exception as exc:
        result.error = f"{type(exc).__name__}: {exc}"
        logger.warning("cell %s %s failed: %s", cell.symbol, cell.params, result.error)
        return result


def _summarize(result: CellResult, record: Any) -> CellResult:
    metrics = (record.metrics or {}).get("strategy") or {}
    trades = record.trades()
    result.status = "done"
    result.trades = int(len(trades))
    result.sharpe = metrics.get("sharpe")
    result.profit_factor = metrics.get("profit_factor")
    result.win_rate = metrics.get("win_rate")
    result.max_drawdown_pct = metrics.get("max_drawdown_pct")
    result.final_equity = metrics.get("final_equity")
    result.p_value = metrics.get("p_value")

    if len(trades):
        pnl = trades["net_pnl"].astype("float64").to_numpy()
        result.net_pnl = float(pnl.sum())
        result.expectancy = float(pnl.mean())
        result.expectancy_stderr = (
            float(pnl.std(ddof=1) / np.sqrt(len(pnl))) if len(pnl) > 1 else None
        )
        r_values = trades["r_multiple"].astype("float64").to_numpy()
        r_values = r_values[np.isfinite(r_values)]
        if len(r_values):
            result.mean_r = float(r_values.mean())
            result.mean_r_stderr = (
                float(r_values.std(ddof=1) / np.sqrt(len(r_values)))
                if len(r_values) > 1
                else None
            )
    else:
        result.net_pnl = 0.0
    return result


# -- consistency ---------------------------------------------------------


def consistency(cells: Sequence[CellResult], metric: str = "mean_r") -> Consistency:
    """Does the sign of the edge repeat across instruments, or not?

    The sign test is a binomial against a fair coin. It is deliberately crude:
    with ten symbols nothing finer is justified, and a crude test that is
    honest about its power beats a refined one that is not.
    """
    per_symbol: dict[str, float] = {}
    missing = 0
    for cell in cells:
        value = getattr(cell, metric, None)
        if cell.status != "done" or value is None or not np.isfinite(float(value)):
            missing += 1
            continue
        # several periods or grid cells on one symbol average into one vote:
        # a symbol is one draw, not one per configuration tried on it
        per_symbol.setdefault(cell.symbol, 0.0)
        per_symbol[cell.symbol] += float(value)

    counts: dict[str, int] = {}
    for cell in cells:
        value = getattr(cell, metric, None)
        if cell.status == "done" and value is not None and np.isfinite(float(value)):
            counts[cell.symbol] = counts.get(cell.symbol, 0) + 1
    values_by_symbol = {
        symbol: total / counts[symbol] for symbol, total in per_symbol.items()
    }

    values = np.array(list(values_by_symbol.values()), dtype="float64")
    n = int(len(values))
    if n == 0:
        return Consistency(
            metric=metric,
            observations=0,
            positive=0,
            negative=0,
            zero_or_missing=missing,
            mean=None,
            median=None,
            std=None,
            stderr=None,
            minimum=None,
            maximum=None,
            sign_p_value=None,
            carried_by_one_symbol=False,
            dominant_symbol=None,
            verdict=(
                "No instrument produced a usable result: there is nothing to be "
                "consistent about."
            ),
        )

    positive = int((values > 0).sum())
    negative = int((values < 0).sum())
    mean = float(values.mean())
    total = float(values.sum())

    # is the aggregate the work of a single instrument?
    dominant_symbol = None
    carried = False
    if n >= 2 and total != 0:
        ranked = sorted(values_by_symbol.items(), key=lambda item: abs(item[1]), reverse=True)
        top_symbol, top_value = ranked[0]
        rest = total - top_value
        if np.sign(total) != np.sign(rest) or rest == 0:
            carried = True
            dominant_symbol = top_symbol

    sign_p = None
    if n >= 1:
        winners = max(positive, negative)
        sign_p = float(stats.binomtest(winners, n, 0.5, alternative="greater").pvalue)

    return Consistency(
        metric=metric,
        observations=n,
        positive=positive,
        negative=negative,
        zero_or_missing=missing,
        mean=mean,
        median=float(np.median(values)),
        std=float(values.std(ddof=1)) if n > 1 else None,
        stderr=float(values.std(ddof=1) / np.sqrt(n)) if n > 1 else None,
        minimum=float(values.min()),
        maximum=float(values.max()),
        sign_p_value=sign_p,
        carried_by_one_symbol=carried,
        dominant_symbol=dominant_symbol,
        verdict=_consistency_verdict(metric, n, positive, negative, mean, sign_p, carried,
                                     dominant_symbol),
    )


def _consistency_verdict(
    metric: str,
    n: int,
    positive: int,
    negative: int,
    mean: float,
    sign_p: float | None,
    carried: bool,
    dominant: str | None,
) -> str:
    parts = [
        f"{metric} is positive on {positive} of {n} instrument(s) and negative on "
        f"{negative}, mean {mean:+.4f}"
    ]
    if n < MIN_SYMBOLS_FOR_CONSISTENCY:
        parts.append(
            f"with {n} instrument(s) the sign test has no power: this is a "
            f"description, not evidence"
        )
    elif sign_p is not None:
        parts.append(
            f"a split this lopsided or worse happens by chance with probability "
            f"{sign_p:.3f} under a fair coin"
        )
    if carried and dominant:
        parts.append(
            f"the aggregate sign is carried by {dominant} alone - remove it and it "
            f"flips, which is the signature of one instrument's history rather "
            f"than a strategy property"
        )
    elif positive == 1 and n >= MIN_SYMBOLS_FOR_CONSISTENCY:
        parts.append(
            "an edge present on exactly one instrument out of many is a suspect, "
            "not a discovery"
        )
    return ". ".join(parts) + "."


# -- the batch -----------------------------------------------------------


def run_batch(
    spec_path: str | Path,
    symbols: Sequence[str],
    base_config: RunConfig,
    periods: Sequence[Period] | None = None,
    grid: ParameterGrid | None = None,
    cache_dir: str | Path = "data_cache",
    runs_dir: str | Path = "runs",
    max_workers: int | None = None,
    consistency_metric: str = "mean_r",
) -> BatchReport:
    """Runs the cartesian product of symbols, periods and grid combinations."""
    started = time.perf_counter()
    spec = StrategySpec.from_json(spec_path)
    periods = list(periods) if periods else [Period()]
    grid = grid or {}
    combinations = expand_grid(grid)

    cells = [
        BatchCell(symbol=symbol, period=period, params=params)
        for symbol in symbols
        for period in periods
        for params in combinations
    ]
    if not cells:
        raise ValueError("the batch has no cell to run: give it at least one symbol")

    config_payload = {
        key: value
        for key, value in asdict(base_config).items()
        if key not in ("symbol", "start", "end")
    }
    workers = max_workers or min(len(cells), os.cpu_count() or 4, MAX_WORKERS_CAP)
    logger.info("batch: %d cells over %d workers", len(cells), workers)

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(str(spec_path), config_payload, str(cache_dir), str(runs_dir)),
    ) as pool:
        results = list(pool.map(_execute_cell, cells))

    completed = sum(1 for result in results if result.status == "done")
    failed = len(results) - completed
    measure = consistency(results, consistency_metric)

    warnings: list[str] = []
    if failed:
        warnings.append(
            f"{failed} of {len(results)} cells failed (most often: no cached data "
            f"for that instrument). They are excluded from the aggregate, which is "
            f"therefore computed on {completed} cells"
        )
    if measure.observations < MIN_SYMBOLS_FOR_CONSISTENCY:
        warnings.append(
            f"only {measure.observations} instrument(s) produced a result: below "
            f"{MIN_SYMBOLS_FOR_CONSISTENCY} the cross-sectional check cannot "
            f"distinguish agreement from coincidence"
        )
    thin = [result for result in results if result.status == "done" and result.trades < 30]
    if thin:
        warnings.append(
            f"{len(thin)} cell(s) hold fewer than 30 trades: their per-symbol "
            f"figures have a standard error of the same order as the estimate"
        )

    elapsed = time.perf_counter() - started
    return BatchReport(
        strategy_id=spec.id,
        symbols=list(symbols),
        periods=periods,
        grid=grid,
        grid_size=len(combinations),
        cells=results,
        completed=completed,
        failed=failed,
        elapsed_seconds=elapsed,
        consistency=measure,
        verdict=(
            f"{completed}/{len(results)} cells completed in {elapsed:.1f}s. "
            + measure.verdict
        ),
        warnings=warnings,
    )
