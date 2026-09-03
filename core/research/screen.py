"""Systematic screening: many strategies, many instruments, many timeframes.

A funnel, in three stages of increasing cost:

1. **gate zero** - fast, and the only stage every cell reaches. It measures
   whether the raw signal moves the price in the predicted direction by more
   than the spread. A signal that fails here cannot be rescued by any SL/TP
   combination, so there is nothing to backtest.
2. **backtest** - medium, run only on the cells the gate let through.
3. **permutation** - slow, run only on the cells that survived the backtest
   with a result worth attacking.

The funnel is not an optimization: it is the statement that a configuration
without directionality has nothing to test, and that spending a thousand
resampled backtests on it produces a p-value about noise.

**The trial count is the point of this module.** A machine that runs three
hundred backtests is a machine for producing false positives: at the 5% level
about fifteen of them come back "significant" with no edge anywhere in the
data. So the campaign counts its own attempts - every strategy x instrument x
timeframe cell that was *started*, not only those that finished - and the
report states, in one line, what Sharpe an observed result needs to reach
before that count stops explaining it.

Cells that stop at gate zero never produce a Sharpe, but they were attempts
all the same: the campaign counts them in N and estimates the spread of the
Sharpe on the cells that did produce one. That is an assumption about the
cells that never ran, and it is declared rather than hidden.
"""
from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, tzinfo
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from core.data.cache import ParquetCache
from core.data.provider import SymbolSpecSnapshot, Timeframe
from core.engine.backtester import BacktestConfig, run_backtest
from core.metrics.ambiguity import uncertainty_band
from core.metrics.gates import gate_accounting
from core.metrics.performance import compute_metrics
from core.runs.runner import (
    DataUnavailable,
    SymbolResolver,
    execute_run,
    load_bars,
    plan_run,
    run_edge_gate,
)
from core.runs.store import RunConfig, RunStore
from core.serialization import json_safe
from core.strategy.spec import StrategySpec
from core.validation.multiple_testing import (
    DEFAULT_ALPHA,
    expected_max_sharpe,
    required_sharpe_per_trade,
    sharpe_per_trade,
    thresholds,
)
from core.validation.permutation import permutation_test
from core.validation.walkforward import apply_params

logger = logging.getLogger(__name__)

Stage = str  # "gate" | "backtest" | "permutation"

# Below this many trades a backtest has no power whatever its equity curve
# says, and a permutation on it would only measure the small sample.
DEFAULT_MIN_TRADES = 30
# Screening runs a reduced permutation: the full 1000 iterations belong to the
# confirmation of a single survivor, not to a campaign of hundreds of cells.
DEFAULT_PERMUTATION_ITERATIONS = 200


@dataclass(frozen=True)
class ScreenCell:
    """One attempt: this strategy, on this instrument, at this timeframe."""

    strategy_id: str
    strategy_path: str
    symbol: str
    timeframe: str


@dataclass
class CellOutcome:
    """Everything the funnel learned about one cell, stage by stage."""

    strategy_id: str
    symbol: str
    timeframe: str
    stage_reached: Stage
    status: str = "ok"
    error: str | None = None

    # stage 1 - gate zero
    gate_passed: bool | None = None
    gate_signals: int | None = None
    gate_best_net_points: float | None = None
    gate_best_p_value: float | None = None
    gate_verdict: str | None = None
    expected_ambiguous_share: float | None = None
    ambiguity_flag: bool | None = None

    # stage 2 - backtest
    run_id: str | None = None
    bars: int | None = None
    trades: int | None = None
    net_pnl: float | None = None
    final_equity: float | None = None
    sharpe_per_trade: float | None = None
    sharpe_annualized: float | None = None
    mean_r: float | None = None
    win_rate: float | None = None
    max_drawdown_pct: float | None = None
    p_value: float | None = None
    ambiguous_share: float | None = None
    band_money: float | None = None
    top_gate: str | None = None
    top_gate_share: float | None = None
    gate_warnings: list[str] = field(default_factory=list)

    # stage 3 - permutation
    permutation_p_value: float | None = None
    permutation_kind: str | None = None
    permutation_iterations: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return json_safe(asdict(self))


@dataclass
class TrialPanel:
    """The correction owed for the size of the campaign."""

    attempts: int
    cells_backtested: int
    cells_permuted: int
    sharpes_observed: int
    variance_across_trials: float | None
    expected_max_sharpe: float | None
    required_sharpe_per_trade: float | None
    confidence: float
    alpha: float
    bonferroni_threshold: float | None
    best_strategy: str | None
    best_sharpe_per_trade: float | None
    best_clears_required: bool | None
    survivors_after_correction: int
    verdict: str
    assumptions: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return json_safe(asdict(self))


@dataclass
class ScreenReport:
    strategies: list[str]
    symbols: list[str]
    timeframes: list[str]
    period_start: datetime | None
    period_end: datetime | None
    initial_equity: float
    cells: list[CellOutcome]
    panel: TrialPanel
    thresholds: list[dict[str, Any]]
    elapsed_seconds: float
    engine_version: str
    verdict: str
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "strategies": self.strategies,
            "symbols": self.symbols,
            "timeframes": self.timeframes,
            "period_start": self.period_start,
            "period_end": self.period_end,
            "initial_equity": self.initial_equity,
            "cells": [cell.as_dict() for cell in self.cells],
            "panel": self.panel.as_dict(),
            "thresholds": self.thresholds,
            "elapsed_seconds": self.elapsed_seconds,
            "engine_version": self.engine_version,
            "verdict": self.verdict,
            "warnings": self.warnings,
        }
        return json_safe(payload)


def build_cells(
    strategies: Sequence[Path | str],
    symbols: Sequence[str],
    timeframes: Sequence[str],
) -> list[ScreenCell]:
    """Every combination, in a stable order. This list *is* the trial count."""
    cells: list[ScreenCell] = []
    for path in strategies:
        spec = StrategySpec.from_json(Path(path))
        for symbol in symbols:
            for timeframe in timeframes:
                cells.append(
                    ScreenCell(
                        strategy_id=spec.id,
                        strategy_path=str(path),
                        symbol=symbol,
                        timeframe=timeframe,
                    )
                )
    return cells


def bind_cell(spec: StrategySpec, symbol: str, timeframe: str) -> StrategySpec:
    """The spec as it is actually run on this cell.

    The instrument block is not decoration: the engine reads the timeframe
    from it to count session bars for the time stop, so a spec declaring M1
    executed over H1 bars would apply a time stop sixty times too short. The
    symbol is bound too, so the run folder says which instrument it was.
    """
    return apply_params(
        spec, {"instrument.symbol": symbol, "instrument.timeframe": timeframe}
    )


def _screen_cell(
    cell: ScreenCell,
    spec: StrategySpec,
    cache: ParquetCache,
    store: RunStore,
    resolver: SymbolResolver,
    base_config: RunConfig,
    min_trades: int,
    permutation_iterations: int,
    server_tz: tzinfo,
) -> CellOutcome:
    outcome = CellOutcome(
        strategy_id=cell.strategy_id,
        symbol=cell.symbol,
        timeframe=cell.timeframe,
        stage_reached="gate",
    )
    bound = bind_cell(spec, cell.symbol, cell.timeframe)
    config = RunConfig(
        **{
            **base_config.to_dict(),
            "symbol": cell.symbol,
            "timeframe": cell.timeframe,
            "start": base_config.start,
            "end": base_config.end,
        }
    )

    try:
        bars = load_bars(cache, cell.symbol, config.tf, config.start, config.end)
        snapshot: SymbolSpecSnapshot = resolver.symbol_spec_snapshot(cell.symbol)
    except (DataUnavailable, Exception) as exc:  # noqa: B014 - reported, not raised
        outcome.status = "error"
        outcome.error = f"{type(exc).__name__}: {exc}"
        return outcome

    outcome.bars = int(len(bars))

    # -- stage 1: gate zero ----------------------------------------------
    try:
        edge = run_edge_gate(bound, config, bars, snapshot.spec)
    except Exception as exc:
        outcome.status = "error"
        outcome.error = f"gate zero: {type(exc).__name__}: {exc}"
        return outcome

    usable = [s for s in edge.stats if np.isfinite(s.net_points)]
    best = max(usable, key=lambda s: s.net_points) if usable else None
    outcome.gate_passed = bool(edge.passed)
    outcome.gate_signals = edge.signals_long + edge.signals_short
    outcome.gate_best_net_points = float(best.net_points) if best else None
    outcome.gate_best_p_value = float(best.p_vs_cost) if best else None
    outcome.gate_verdict = edge.verdict
    prior = edge.ambiguity_prior or {}
    outcome.expected_ambiguous_share = prior.get("expected_ambiguous_share")
    outcome.ambiguity_flag = prior.get("exceeds_threshold")

    if not edge.passed:
        outcome.status = "stopped at gate zero"
        return outcome

    # -- stage 2: backtest -----------------------------------------------
    outcome.stage_reached = "backtest"
    try:
        run_id, _ = plan_run(bound, config, bars, snapshot.spec)
        outcome.run_id = run_id
        if not (store.exists(run_id) and store.load_meta(run_id).status == "done"):
            execute_run(store, bound, config, bars, snapshot, server_tz, run_id)
        record = store.load_run(run_id)
    except Exception as exc:
        outcome.status = "error"
        outcome.error = f"backtest: {type(exc).__name__}: {exc}"
        return outcome

    metrics = (record.metrics or {}).get("strategy") or {}
    trades = record.trades()
    pnl = (
        trades["net_pnl"].astype("float64").to_numpy() if len(trades) else np.zeros(0)
    )
    outcome.trades = int(len(trades))
    outcome.net_pnl = float(pnl.sum()) if len(pnl) else 0.0
    outcome.final_equity = metrics.get("final_equity")
    outcome.sharpe_per_trade = sharpe_per_trade(pnl) if len(pnl) > 1 else None
    outcome.sharpe_annualized = metrics.get("sharpe")
    outcome.win_rate = metrics.get("win_rate")
    outcome.max_drawdown_pct = metrics.get("max_drawdown_pct")
    outcome.p_value = metrics.get("p_value")
    if len(trades) and "r_multiple" in trades:
        finite = trades["r_multiple"].astype("float64").dropna()
        outcome.mean_r = float(finite.mean()) if len(finite) else None

    uncertainty = (record.metrics or {}).get("uncertainty") or {}
    outcome.ambiguous_share = uncertainty.get("ambiguous_share")
    outcome.band_money = uncertainty.get("band_money")

    gates = (record.metrics or {}).get("gates") or {}
    rows = gates.get("rows") or []
    if rows:
        outcome.top_gate = rows[0]["code"]
        outcome.top_gate_share = rows[0]["share"]
    outcome.gate_warnings = list(gates.get("warnings") or [])

    if outcome.trades < min_trades:
        outcome.status = f"too few trades ({outcome.trades} < {min_trades})"
        return outcome
    if outcome.net_pnl <= 0:
        outcome.status = "negative net PnL: nothing to attack with a null"
        return outcome

    # -- stage 3: permutation --------------------------------------------
    outcome.stage_reached = "permutation"
    try:
        report = permutation_test(
            "random_entries",
            bound,
            bars,
            snapshot.spec,
            server_tz,
            BacktestConfig(
                initial_equity=config.initial_equity,
                costs=config.cost_model(),
                session_threshold=config.session_threshold,
            ),
            iterations=permutation_iterations,
        )
    except Exception as exc:
        outcome.status = "error"
        outcome.error = f"permutation: {type(exc).__name__}: {exc}"
        return outcome

    if report.statistics:
        best_statistic = min(report.statistics, key=lambda s: s.p_value)
        outcome.permutation_p_value = float(best_statistic.p_value)
        outcome.permutation_kind = report.kind
        outcome.permutation_iterations = report.iterations
    outcome.status = "ok"
    return outcome


def _panel(
    cells: Sequence[CellOutcome], alpha: float, confidence: float
) -> TrialPanel:
    """The campaign-wide correction, over every attempt that was made."""
    attempts = len(cells)
    backtested = [c for c in cells if c.trades is not None]
    permuted = [c for c in cells if c.permutation_p_value is not None]
    sharpes = [
        c.sharpe_per_trade
        for c in backtested
        if c.sharpe_per_trade is not None and np.isfinite(c.sharpe_per_trade)
    ]

    variance = float(np.var(np.asarray(sharpes), ddof=1)) if len(sharpes) > 1 else None
    best = max(
        (c for c in backtested if c.sharpe_per_trade is not None),
        key=lambda c: c.sharpe_per_trade,
        default=None,
    )
    expected_max = expected_max_sharpe(attempts, variance) if variance else None
    required = (
        required_sharpe_per_trade(
            attempts, variance, best.trades or 0, confidence=confidence
        )
        if variance and best is not None
        else None
    )
    clears = (
        bool(best.sharpe_per_trade >= required)
        if required is not None and best is not None and best.sharpe_per_trade is not None
        else None
    )

    rows = thresholds(
        [
            (c.run_id or f"{c.strategy_id}:{c.symbol}:{c.timeframe}", c.strategy_id,
             float(c.p_value), bool((c.net_pnl or 0.0) > 0))
            for c in backtested
            if c.p_value is not None and np.isfinite(c.p_value)
        ],
        alpha=alpha,
    )
    survivors = sum(1 for row in rows if row.passes_bonferroni and row.mean_pnl_positive)

    assumptions = [
        f"{attempts} attempts counted: every strategy x instrument x timeframe cell "
        f"that was started, including the {attempts - len(backtested)} stopped at "
        f"gate zero",
        f"the spread of the Sharpe across trials is estimated on the "
        f"{len(sharpes)} cells that reached a backtest and assumed to hold for "
        f"the ones that did not",
        "the cells are not independent: the same instrument appears at three "
        "timeframes and the same strategy on ten instruments, so the effective "
        "number of independent trials is smaller than the count and this "
        "correction is, if anything, too generous",
    ]

    if best is None or best.sharpe_per_trade is None:
        verdict = (
            f"{attempts} attempts, none of which produced a usable Sharpe: there is "
            f"nothing to correct and nothing to report as a finding."
        )
    elif required is None:
        verdict = (
            f"{attempts} attempts. The Sharpe spread across trials could not be "
            f"estimated, so no corrected threshold can be stated."
        )
    else:
        verdict = (
            f"{attempts} attempts. The best cell is {best.strategy_id} on "
            f"{best.symbol} {best.timeframe} with a per-trade Sharpe of "
            f"{best.sharpe_per_trade:+.4f} over {best.trades} trades. After "
            f"{attempts} attempts, a search of this size is expected to reach "
            f"{expected_max:+.4f} by luck alone, and the observed Sharpe would have "
            f"to be at least {required:+.4f} to be credible at {confidence:.0%} "
            f"confidence. It is {'above' if clears else 'below'} that level."
        )

    return TrialPanel(
        attempts=attempts,
        cells_backtested=len(backtested),
        cells_permuted=len(permuted),
        sharpes_observed=len(sharpes),
        variance_across_trials=variance,
        expected_max_sharpe=expected_max,
        required_sharpe_per_trade=required,
        confidence=confidence,
        alpha=alpha,
        bonferroni_threshold=rows[0].bonferroni_threshold if rows else None,
        best_strategy=(
            f"{best.strategy_id} / {best.symbol} / {best.timeframe}" if best else None
        ),
        best_sharpe_per_trade=best.sharpe_per_trade if best else None,
        best_clears_required=clears,
        survivors_after_correction=survivors,
        verdict=verdict,
        assumptions=assumptions,
    )


def run_screen(
    strategies: Sequence[Path | str],
    symbols: Sequence[str],
    timeframes: Sequence[str],
    base_config: RunConfig,
    cache_dir: Path | str = "data_cache",
    runs_dir: Path | str = "runs",
    min_trades: int = DEFAULT_MIN_TRADES,
    permutation_iterations: int = DEFAULT_PERMUTATION_ITERATIONS,
    alpha: float = DEFAULT_ALPHA,
    confidence: float = 0.95,
    on_cell: "Callable[[int, int, CellOutcome], None] | None" = None,
) -> ScreenReport:
    """Runs the whole campaign and returns one reproducible table."""
    from core.version import ENGINE_VERSION

    started = time.perf_counter()
    cache = ParquetCache(Path(cache_dir))
    store = RunStore(Path(runs_dir))
    resolver = SymbolResolver(cache)
    server_tz = resolver.server_timezone()

    specs = {str(path): StrategySpec.from_json(Path(path)) for path in strategies}
    cells = build_cells(strategies, symbols, timeframes)
    logger.info(
        "screening %d cells: %d strategies x %d symbols x %d timeframes",
        len(cells), len(strategies), len(symbols), len(timeframes),
    )

    outcomes: list[CellOutcome] = []
    for index, cell in enumerate(cells, start=1):
        outcome = _screen_cell(
            cell,
            specs[cell.strategy_path],
            cache,
            store,
            resolver,
            base_config,
            min_trades,
            permutation_iterations,
            server_tz,
        )
        outcomes.append(outcome)
        logger.info(
            "[%3d/%3d] %-24s %-9s %-3s -> %-12s %s",
            index, len(cells), cell.strategy_id, cell.symbol, cell.timeframe,
            outcome.stage_reached, outcome.status,
        )
        if on_cell is not None:
            on_cell(index, len(cells), outcome)

    panel = _panel(outcomes, alpha, confidence)
    warnings: list[str] = []
    failed = [c for c in outcomes if c.status == "error"]
    if failed:
        warnings.append(
            f"{len(failed)} cells failed and are excluded from every aggregate: "
            + ", ".join(sorted({c.error.split(":")[0] for c in failed if c.error}))
        )
    flagged = [c for c in outcomes if c.ambiguity_flag]
    if flagged:
        warnings.append(
            f"{len(flagged)} cells are expected to produce more ambiguous trades "
            f"than the 5% threshold: on those, bar resolution cannot settle the "
            f"result whatever the equity curve shows"
        )

    survivors = [c for c in outcomes if c.stage_reached == "permutation"]
    verdict = (
        f"{len(cells)} cells screened, {panel.cells_backtested} reached a backtest, "
        f"{len(survivors)} were worth a permutation test. "
        + panel.verdict
    )

    return ScreenReport(
        strategies=[specs[str(p)].id for p in strategies],
        symbols=list(symbols),
        timeframes=list(timeframes),
        period_start=base_config.start,
        period_end=base_config.end,
        initial_equity=base_config.initial_equity,
        cells=outcomes,
        panel=panel,
        thresholds=[
            row.as_dict()
            for row in thresholds(
                [
                    (c.run_id or f"{c.strategy_id}:{c.symbol}:{c.timeframe}",
                     c.strategy_id, float(c.p_value), bool((c.net_pnl or 0.0) > 0))
                    for c in outcomes
                    if c.p_value is not None and np.isfinite(c.p_value)
                ],
                alpha=alpha,
            )
        ],
        elapsed_seconds=time.perf_counter() - started,
        engine_version=ENGINE_VERSION,
        verdict=verdict,
        warnings=warnings,
    )


def as_frame(report: ScreenReport) -> pd.DataFrame:
    """The campaign as one table, sorted by the metric that matters."""
    frame = pd.DataFrame([cell.as_dict() for cell in report.cells])
    if frame.empty:
        return frame
    return frame.sort_values(
        ["sharpe_per_trade", "net_pnl"], ascending=False, na_position="last"
    ).reset_index(drop=True)
