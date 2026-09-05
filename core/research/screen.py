"""Systematic screening: many strategies, many instruments, many timeframes.

A funnel, in four stages of increasing cost:

0. **tradability** - free, and it is a refusal rather than a test. If the
   broker's spread on an instrument is a large enough share of its typical
   stop distance, no realistic edge covers it and there is nothing to look
   for. Cells refused here **do not count as attempts**: nobody tested them,
   and inflating the multiple-testing N with experiments that never ran makes
   the correction look severe while measuring nothing.
1. **gate zero** - fast, and the first stage that is actually a test. It
   measures whether the raw signal moves the price in the predicted direction
   by more than the spread. A signal that fails here cannot be rescued by any
   SL/TP combination, so there is nothing to backtest.
2. **backtest** - medium, run only on the cells the gate let through.
3. **permutation** - slow, run only on the cells that survived the backtest
   with a result worth attacking.

The funnel is not an optimization: it is the statement that a configuration
without directionality has nothing to test, and that spending a thousand
resampled backtests on it produces a p-value about noise.

**The trial count is the point of this module.** A machine that runs three
hundred backtests is a machine for producing false positives: at the 5% level
about fifteen of them come back "significant" with no edge anywhere in the
data. So the campaign counts its own attempts - every admitted strategy x
instrument x timeframe cell that was *started*, not only those that finished -
and the report states, in one line, what Sharpe an observed result needs to
reach before that count stops explaining it.

Cells that stop at gate zero never produce a Sharpe, but they were attempts
all the same: the campaign counts them in N and estimates the spread of the
Sharpe on the cells that did produce one. That is an assumption about the
cells that never ran, and it is declared rather than hidden.

**The gates are part of the strategy, so they are reported with it.** On
bollinger-breakout at M5, 49% of the signals were thrown away because the
engine was already in a position: the system that produced that equity curve
is not the system written in the JSON. Every row therefore carries the share
of signals its gates rejected, and above a fifth the result is flagged as
materially altered. For those cells the same spec is re-run with the
discretionary gates relaxed - spread cap, cooldown, daily cap, session - and
the two results are shown side by side, so the size of the gates' effect is a
number rather than a suspicion. The one-position constraint cannot be relaxed
that way: it is structural to an engine that holds one position, and its share
is reported separately instead of being quietly folded in.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, tzinfo
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from core.data.cache import ParquetCache
from core.data.provider import SymbolSpecSnapshot
from core.engine.backtester import BacktestConfig
from core.metrics.gates import WARNING_SHARE
from core.research.tradability import (
    DEFAULT_MAX_SPREAD_ATR,
    TradabilityTable,
    build_table,
)
from core.runs.runner import (
    DataUnavailable,
    SymbolResolver,
    execute_run,
    load_bars_for_run,
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

if TYPE_CHECKING:
    from core.research.manifest import CampaignManifest

logger = logging.getLogger(__name__)

Stage = str  # "tradability" | "gate" | "backtest" | "permutation"


@dataclass(frozen=True)
class Candidate:
    """One cell that produced a Sharpe worth comparing, whichever campaign ran it.

    The correction counts every attempt of the search, so the maximum it is
    correcting has to be drawn from the same search. Reading it off this
    campaign's cells alone makes the best result depend on where the operator
    stopped for the night, and reports a smaller maximum than the search
    actually produced.
    """

    label: str
    sharpe_per_trade: float
    trades: int

# Below this many trades a backtest has no power whatever its equity curve
# says, and a permutation on it would only measure the small sample.
DEFAULT_MIN_TRADES = 30
# Screening runs a reduced permutation: the full 1000 iterations belong to the
# confirmation of a single survivor, not to a campaign of hundreds of cells.
DEFAULT_PERMUTATION_ITERATIONS = 200

# Gates that can be turned off without changing what the engine is. The
# one-position constraint is not among them: an engine that holds one position
# cannot be asked to hold two, and pretending otherwise would compare a result
# against a system that does not exist.
RELAXABLE_GATES: tuple[str, ...] = (
    "max_spread_points",
    "cooldown_minutes",
    "max_trades_per_day",
    "session",
)
STRUCTURAL_GATES: tuple[str, ...] = ("max_open_positions", "position_open")
# Not a gate at all: the account had grown too small to buy the broker's
# minimum lot. Folding this into "the risk block is selecting the trades"
# would describe a busted account as a design choice, which is the opposite
# of what happened.
EQUITY_GATES: tuple[str, ...] = ("insufficient_equity",)


def relax_gates(spec: StrategySpec) -> StrategySpec:
    """The same spec with every discretionary gate opened.

    Used to answer "how much of this result is the strategy and how much is
    the risk block". The structural one-position gate stays, because it is a
    property of the engine rather than a choice in the JSON.
    """
    return apply_params(
        spec,
        {
            "risk.max_spread_points": None,
            "risk.cooldown_minutes": 0,
            "risk.max_trades_per_day": None,
            "risk.session": None,
        },
    )


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

    # stage 0 - tradability. `counts_as_attempt` is False exactly when the
    # cell was refused here: it was not tested, so it does not belong in N
    counts_as_attempt: bool = True
    tradable: bool = True
    tradability_judged: bool = True
    spread_atr_ratio: float | None = None
    spread_stop_share: float | None = None
    tradability_reason: str | None = None

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

    # what the gates did to the strategy, on every row and not only in the
    # detail view of a single run
    signals: int | None = None
    gate_rejected: int | None = None
    gate_rejected_share: float | None = None
    structural_rejected_share: float | None = None
    discretionary_rejected_share: float | None = None
    equity_rejected_share: float | None = None
    gates_materially_altered: bool | None = None
    top_gate: str | None = None
    top_gate_share: float | None = None
    gate_warnings: list[str] = field(default_factory=list)

    # what the spread policy actually charged on this cell
    spread_charged_median: float | None = None
    spread_zero_share: float | None = None
    spread_trustworthy: bool | None = None
    # ...and how much of it was measured rather than assumed. A cell at 0%
    # was charged a constant taken from a different period on every bar; the
    # cumulative campaign has to be read knowing how many of its cells are
    # in that position
    spread_measured_share: float | None = None
    spread_assumed_bars: int | None = None

    # the same spec with the discretionary gates opened, run only when the
    # gates rejected enough signals to matter
    relaxed_trades: int | None = None
    relaxed_net_pnl: float | None = None
    relaxed_sharpe_per_trade: float | None = None
    relaxed_verdict: str | None = None

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
    # how much of the campaign was charged a measured spread rather than a
    # constant carried in from another period. A campaign whose cells are
    # mostly at zero is not wrong, but every number in it rests on the same
    # unverified cost, and the cumulative result has to be read that way
    spread_measured_share_median: float | None = None
    cells_with_no_measured_spread: int = 0

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
    tradability: dict[str, Any] | None
    elapsed_seconds: float
    engine_version: str
    verdict: str
    warnings: list[str] = field(default_factory=list)
    # the inputs this campaign was run against, frozen. Re-running from it
    # gives the same cells the same contracts, so a difference in the results
    # is the engine or the data and nothing else.
    manifest: dict[str, Any] | None = None

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
            "tradability": self.tradability,
            "elapsed_seconds": self.elapsed_seconds,
            "engine_version": self.engine_version,
            "verdict": self.verdict,
            "warnings": self.warnings,
            "manifest": self.manifest,
        }
        return json_safe(payload)


def build_cells(
    strategies: Sequence[Path | str],
    symbols: Sequence[str],
    timeframes: Sequence[str],
) -> list[ScreenCell]:
    """Every combination, in a stable order.

    This list is the grid, not the trial count: stage zero refuses some of
    these before anything is tested, and a refused cell is not an attempt.
    """
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


def _compare_relaxed(
    outcome: CellOutcome,
    bound: StrategySpec,
    config: RunConfig,
    bars: pd.DataFrame,
    snapshot: SymbolSpecSnapshot,
    server_tz: tzinfo,
    store: RunStore,
    threshold: float,
) -> None:
    """Runs the same spec with the discretionary gates opened, and compares.

    The point is not to prefer one result over the other: it is to put a
    number on how much of the reported curve was produced by the risk block
    rather than by the strategy. A relaxed run that trades three times as
    often for the same PnL says the gates were selecting the winners.
    """
    relaxed = relax_gates(bound)
    try:
        run_id, _ = plan_run(relaxed, config, bars, snapshot.spec)
        if not (store.exists(run_id) and store.load_meta(run_id).status == "done"):
            execute_run(store, relaxed, config, bars, snapshot, server_tz, run_id)
        record = store.load_run(run_id)
    except Exception as exc:
        outcome.relaxed_verdict = (
            f"the relaxed-gate comparison failed to run "
            f"({type(exc).__name__}: {exc}), so the size of the gates' effect "
            f"is unmeasured"
        )
        return

    trades = record.trades()
    pnl = trades["net_pnl"].astype("float64").to_numpy() if len(trades) else np.zeros(0)
    outcome.relaxed_trades = int(len(trades))
    outcome.relaxed_net_pnl = float(pnl.sum()) if len(pnl) else 0.0
    outcome.relaxed_sharpe_per_trade = sharpe_per_trade(pnl) if len(pnl) > 1 else None

    gated_trades = outcome.trades or 0
    factor = outcome.relaxed_trades / gated_trades if gated_trades else float("inf")
    outcome.relaxed_verdict = (
        f"the gates rejected {outcome.gate_rejected_share:.0%} of the signals, "
        f"{(outcome.discretionary_rejected_share or 0.0):.0%} of it by gates that "
        f"can be opened. With them open the same spec takes "
        f"{outcome.relaxed_trades} trades instead of {gated_trades} "
        f"({factor:.1f}x) for a net PnL of {outcome.relaxed_net_pnl:+.2f} against "
        f"{(outcome.net_pnl or 0.0):+.2f}. Above {threshold:.0%} rejection the "
        f"gated result describes the risk block as much as the strategy"
    )


def _why_altered(outcome: CellOutcome) -> str:
    """Names what actually threw the signals away, when nothing can be opened."""
    equity = outcome.equity_rejected_share or 0.0
    structural = outcome.structural_rejected_share or 0.0
    discretionary = outcome.discretionary_rejected_share or 0.0

    if equity >= max(structural, discretionary):
        return (
            f"{outcome.gate_rejected_share:.0%} of the signals never became "
            f"trades, and {equity:.0%} of them because the account no longer "
            f"held enough equity for the broker's minimum lot. That is not a "
            f"risk block choosing trades, it is a strategy that lost the money "
            f"it needed to keep trading: the equity curve after that point is "
            f"a flat line, not a result"
        )
    if structural >= discretionary:
        return (
            f"{outcome.gate_rejected_share:.0%} of the signals were rejected, "
            f"{structural:.0%} of them because a position was already open. That "
            f"is structural to a one-position engine and cannot be relaxed; the "
            f"discretionary gates rejected {discretionary:.0%}, so there is "
            f"nothing to open"
        )
    return (
        f"{outcome.gate_rejected_share:.0%} of the signals were rejected, "
        f"{discretionary:.0%} by gates that could be opened - below the "
        f"threshold for re-running the comparison"
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
    tradability: TradabilityTable | None = None,
    gate_alteration_threshold: float = WARNING_SHARE,
) -> CellOutcome:
    outcome = CellOutcome(
        strategy_id=cell.strategy_id,
        symbol=cell.symbol,
        timeframe=cell.timeframe,
        stage_reached="tradability",
    )

    # -- stage 0: tradability --------------------------------------------
    admission = tradability.get(cell.symbol, cell.timeframe) if tradability else None
    if admission is not None:
        outcome.tradable = admission.tradable
        outcome.tradability_judged = admission.judged
        outcome.spread_atr_ratio = admission.spread_atr_ratio
        outcome.spread_stop_share = admission.spread_stop_share
        outcome.tradability_reason = admission.reason
        if not admission.tradable:
            outcome.counts_as_attempt = False
            outcome.status = "refused before testing: spread too large for the stop"
            return outcome

    outcome.stage_reached = "gate"
    bound = bind_cell(spec, cell.symbol, cell.timeframe)
    # each instrument is charged its own measured spread, as a fixed number
    # the run_id records: one value for ten instruments whose spreads run from
    # 1 to 30 points would be a different experiment on every one of them, and
    # reading it off the bar column above M1 charges the best price of the bar
    overrides: dict[str, Any] = {}
    if admission is not None and admission.median_spread_points is not None:
        overrides = {
            "spread_mode": "fixed",
            "spread_value": float(admission.median_spread_points),
        }
    config = RunConfig(
        **{
            **base_config.to_dict(),
            **overrides,
            "symbol": cell.symbol,
            "timeframe": cell.timeframe,
            "start": base_config.start,
            "end": base_config.end,
        }
    )

    try:
        bars = load_bars_for_run(cache, config)
        snapshot: SymbolSpecSnapshot = resolver.symbol_spec_snapshot(cell.symbol)
    except (DataUnavailable, Exception) as exc:
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
    prior: dict[str, Any] = edge.ambiguity_prior or {}
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

    costs = (record.metrics or {}).get("costs") or {}
    outcome.spread_charged_median = costs.get("median_charged_points")
    outcome.spread_zero_share = costs.get("zero_charged_share")
    outcome.spread_trustworthy = costs.get("trustworthy")

    spread_coverage = (record.metrics or {}).get("spread_coverage") or {}
    outcome.spread_measured_share = spread_coverage.get("measured_share")
    outcome.spread_assumed_bars = spread_coverage.get("assumed_bars")

    gates = (record.metrics or {}).get("gates") or {}
    rows = gates.get("rows") or []
    if rows:
        outcome.top_gate = rows[0]["code"]
        outcome.top_gate_share = rows[0]["share"]
    outcome.gate_warnings = list(gates.get("warnings") or [])
    outcome.signals = gates.get("signals")
    outcome.gate_rejected = gates.get("rejected")
    denominator = max(gates.get("signals") or 0, 1)
    outcome.gate_rejected_share = (gates.get("rejected") or 0) / denominator
    outcome.structural_rejected_share = sum(
        row["share"] for row in rows if row["code"] in STRUCTURAL_GATES
    )
    outcome.discretionary_rejected_share = sum(
        row["share"] for row in rows if row["code"] in RELAXABLE_GATES
    )
    outcome.equity_rejected_share = sum(
        row["share"] for row in rows if row["code"] in EQUITY_GATES
    )
    outcome.gates_materially_altered = (
        outcome.gate_rejected_share > gate_alteration_threshold
    )

    # the same spec without the discretionary gates: only worth the extra
    # backtest when those gates actually rejected something material
    if (outcome.discretionary_rejected_share or 0.0) > gate_alteration_threshold:
        _compare_relaxed(
            outcome, bound, config, bars, snapshot, server_tz, store,
            gate_alteration_threshold,
        )
    elif outcome.gates_materially_altered:
        outcome.relaxed_verdict = _why_altered(outcome)

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
    cells: Sequence[CellOutcome],
    alpha: float,
    confidence: float,
    prior_attempts: int = 0,
    prior_results: Sequence[dict[str, Any]] = (),
    min_trades: int = DEFAULT_MIN_TRADES,
    period: str | None = None,
) -> TrialPanel:
    """The campaign-wide correction, over every attempt that was made.

    Only cells that were actually tested count. A cell refused at stage zero
    for being untradable was never an experiment, and adding it to N would
    make the correction look stricter while measuring nothing.

    `prior_attempts` and `prior_results` carry earlier campaigns on the same
    search. The research is one search whether or not it was run in one
    session, and a correction that resets every time the process restarts is
    not a correction. The carried cells are candidates for the best result as
    well as observations of its spread: a maximum taken over half of a search
    while the threshold is computed for all of it compares two different
    numbers.

    **The variance across trials is estimated only on cells with at least
    `min_trades` trades.** A per-trade Sharpe over two trades is not a small
    estimate of a Sharpe, it is a ratio of two numbers: on this campaign such
    cells reached -10.1, and including them took the variance from 0.011 to
    2.08 and the corrected threshold from a reachable +0.6 to an unreachable
    +4.8. A threshold nothing can clear is not a strict test, it is a broken
    one - it would have declared a real edge unproven for the same reason it
    declares noise unproven. The attempt *count* still includes those cells:
    they were attempts, they just cannot say how much Sharpe a search hands
    out for free.
    """
    tested = [c for c in cells if c.counts_as_attempt]
    attempts = len(tested) + int(prior_attempts)
    cells = tested
    backtested = [c for c in cells if c.trades is not None]
    permuted = [c for c in cells if c.permutation_p_value is not None]

    # the winner is drawn from the same population the variance was estimated
    # on. Crowning a two-trade cell as the campaign's best result and then
    # solving for the Sharpe it would need over two observations is not a
    # strict test of anything; it is arithmetic on noise
    # the period belongs in the label: pooled across two campaigns, the same
    # strategy, instrument and timeframe names two different cells, and a
    # headline result that cannot be traced back to one of them is a string,
    # not a finding
    suffix = f" / {period}" if period else ""
    mine = [
        Candidate(
            f"{c.strategy_id} / {c.symbol} / {c.timeframe}{suffix}",
            float(c.sharpe_per_trade),
            int(c.trades or 0),
        )
        for c in backtested
        if c.sharpe_per_trade is not None
        and np.isfinite(c.sharpe_per_trade)
        and (c.trades or 0) >= min_trades
    ]
    carried = [
        Candidate(
            str(r["cell"]), float(r["sharpe_per_trade"]), int(r["trades"])
        )
        for r in prior_results
        if r.get("sharpe_per_trade") is not None
        and np.isfinite(float(r["sharpe_per_trade"]))
    ]
    candidates = [*mine, *carried]

    # the spread of the Sharpe across trials is a property of the search, so
    # earlier campaigns on the same search contribute their observations too
    pooled = [c.sharpe_per_trade for c in candidates]
    variance = float(np.var(np.asarray(pooled), ddof=1)) if len(pooled) > 1 else None
    best = max(candidates, key=lambda c: c.sharpe_per_trade, default=None)
    expected_max = expected_max_sharpe(attempts, variance) if variance else None
    required = (
        required_sharpe_per_trade(
            attempts, variance, best.trades, confidence=confidence
        )
        if variance and best is not None
        else None
    )
    clears = (
        bool(best.sharpe_per_trade >= required)
        if required is not None and best is not None
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
        f"{attempts} attempts counted: {len(tested)} cells tested in this campaign"
        + (
            f" plus {prior_attempts} from earlier campaigns on the same search"
            if prior_attempts
            else ""
        )
        + f", including the {len(tested) - len(backtested)} stopped at gate zero. "
        f"Cells refused before testing are excluded: they were not experiments",
        f"the spread of the Sharpe across trials is estimated on "
        f"{len(pooled)} observed Sharpes ({len(mine)} from this campaign"
        + (f", {len(carried)} carried over" if carried else "")
        + f"), only from cells with at least {min_trades} trades - a per-trade "
        f"Sharpe over two trades is a ratio, not an estimate - and assumed to "
        f"hold for the cells that never produced one",
        f"the best result is taken from the same {len(candidates)} cells the "
        f"spread was estimated on"
        + (
            f", earlier campaigns on this search included: the maximum is over "
            f"the whole search the {attempts} attempts are counted from, not "
            f"over this campaign alone"
            if carried
            else ""
        )
        + f". A cell with fewer than {min_trades} trades cannot be a candidate "
        f"for a finding, whatever its Sharpe reads",
        "the cells are not independent: the same instrument appears at several "
        "timeframes and the same strategy on ten instruments, so the effective "
        "number of independent trials is smaller than the count and this "
        "correction is, if anything, too generous",
        _spread_coverage_assumption(backtested),
    ]

    if best is None or best.sharpe_per_trade is None:
        verdict = (
            f"{attempts} attempts, none of which produced a Sharpe over at least "
            f"{min_trades} trades: there is nothing to correct and nothing to "
            f"report as a finding."
        )
    elif required is None:
        verdict = (
            f"{attempts} attempts. The Sharpe spread across trials could not be "
            f"estimated, so no corrected threshold can be stated."
        )
    else:
        verdict = (
            f"{attempts} attempts. The best cell with at least {min_trades} trades "
            f"is {best.label}, with a "
            f"per-trade Sharpe of {best.sharpe_per_trade:+.4f} over "
            f"{best.trades} trades. After "
            f"{attempts} attempts, a search of this size is expected to reach "
            f"{expected_max:+.4f} by luck alone, and the observed Sharpe would have "
            f"to be at least {required:+.4f} to be credible at {confidence:.0%} "
            f"confidence. It is {'above' if clears else 'below'} that level."
        )

    return TrialPanel(
        attempts=attempts,
        cells_backtested=len(backtested),
        cells_permuted=len(permuted),
        sharpes_observed=len(pooled),
        variance_across_trials=variance,
        expected_max_sharpe=expected_max,
        required_sharpe_per_trade=required,
        confidence=confidence,
        alpha=alpha,
        bonferroni_threshold=rows[0].bonferroni_threshold if rows else None,
        best_strategy=best.label if best else None,
        best_sharpe_per_trade=best.sharpe_per_trade if best else None,
        best_clears_required=clears,
        survivors_after_correction=survivors,
        verdict=verdict,
        assumptions=assumptions,
        spread_measured_share_median=_median_measured_share(backtested),
        cells_with_no_measured_spread=sum(
            1 for cell in backtested if (cell.spread_measured_share or 0.0) <= 0.0
        ),
    )


def _period_label(start: datetime | None, end: datetime | None) -> str | None:
    if start is None or end is None:
        return None
    return f"{start.date().isoformat()}..{end.date().isoformat()}"


def _median_measured_share(cells: Sequence[CellOutcome]) -> float | None:
    shares = [
        cell.spread_measured_share
        for cell in cells
        if cell.spread_measured_share is not None
    ]
    return float(np.median(shares)) if shares else None


def _spread_coverage_assumption(cells: Sequence[CellOutcome]) -> str:
    """How much of this campaign's cost was measured, in one sentence.

    Belongs with the assumptions rather than the results because that is
    what it is. Where M1 exists the spread is measured; everywhere else a
    constant from a later period is charged to older bars, and the share of
    the campaign in that position is the size of the assumption the whole
    correction sits on.
    """
    share = _median_measured_share(cells)
    if share is None:
        return (
            "the share of bars with a measured spread was not recorded for "
            "these cells: they predate the measurement and their costs cannot "
            "be told apart from assumptions"
        )
    blind = sum(1 for cell in cells if (cell.spread_measured_share or 0.0) <= 0.0)
    return (
        f"the spread is measured only where M1 bars exist: across these cells "
        f"the median share of bars with one behind them is {share:.1%}, and "
        f"{blind} of {len(cells)} cells had none at all - on those, every bar "
        f"was charged a constant taken from a different period"
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
    on_cell: Callable[[int, int, CellOutcome], None] | None = None,
    check_tradability: bool = True,
    max_spread_atr: float = DEFAULT_MAX_SPREAD_ATR,
    gate_alteration_threshold: float = WARNING_SHARE,
    prior_attempts: int = 0,
    prior_results: Sequence[dict[str, Any]] = (),
    manifest: CampaignManifest | None = None,
) -> ScreenReport:
    """Runs the whole campaign and returns one reproducible table.

    `prior_attempts` and `prior_results` continue an earlier campaign's trial
    count instead of restarting it: the search is the same search, and the
    multiple-testing correction has to know how many times it has been run,
    and over which results.

    `manifest` re-runs a campaign against the instrument specs and settings
    frozen when it first ran, instead of reading them fresh. Without it the
    grid is the same but the contracts are not - `tick_value` moves with an
    FX rate - and two executions of the same campaign differ for reasons that
    have nothing to do with the engine. When no manifest is passed, one is
    frozen here and returned with the report, so the campaign that just ran
    can be re-run later.
    """
    from core.research.manifest import CampaignManifest
    from core.version import ENGINE_VERSION

    started = time.perf_counter()
    cache = ParquetCache(Path(cache_dir))
    store = RunStore(Path(runs_dir))
    resolver = SymbolResolver(cache)

    if manifest is not None:
        # every setting that changes a number comes from the manifest, not
        # from this call's arguments: a re-run that honoured half of each
        # would be reproducing neither
        manifest.apply(resolver)
        base_config = manifest.run_config()
        strategies = list(manifest.strategies)
        symbols = list(manifest.symbols)
        timeframes = list(manifest.timeframes)
        min_trades = manifest.min_trades
        permutation_iterations = manifest.permutation_iterations
        alpha = manifest.alpha
        confidence = manifest.confidence
        max_spread_atr = manifest.max_spread_atr
        gate_alteration_threshold = manifest.gate_alteration_threshold
        check_tradability = manifest.check_tradability
        prior_attempts = manifest.prior_attempts
        prior_results = list(manifest.prior_results)

    server_tz = resolver.server_timezone()

    if manifest is None:
        manifest = CampaignManifest.freeze(
            resolver,
            [str(path) for path in strategies],
            list(symbols),
            list(timeframes),
            base_config,
            min_trades=min_trades,
            permutation_iterations=permutation_iterations,
            alpha=alpha,
            confidence=confidence,
            max_spread_atr=max_spread_atr,
            gate_alteration_threshold=gate_alteration_threshold,
            check_tradability=check_tradability,
            prior_attempts=prior_attempts,
            prior_results=list(prior_results),
        )
        manifest.apply(resolver)

    # -- stage zero, computed once for the whole grid --------------------
    table: TradabilityTable | None = None
    if check_tradability:
        points: dict[str, float] = {}
        for symbol in symbols:
            try:
                points[symbol] = resolver.symbol_spec(symbol).point
            except Exception as exc:
                logger.warning("%s: no spec, cannot judge tradability (%s)", symbol, exc)
        table = build_table(
            cache, symbols, timeframes, points,
            base_config.start, base_config.end, max_ratio=max_spread_atr,
        )

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
            table,
            gate_alteration_threshold,
        )
        outcomes.append(outcome)
        logger.info(
            "[%3d/%3d] %-24s %-9s %-3s -> %-12s %s",
            index, len(cells), cell.strategy_id, cell.symbol, cell.timeframe,
            outcome.stage_reached, outcome.status,
        )
        if on_cell is not None:
            on_cell(index, len(cells), outcome)

    panel = _panel(
        outcomes,
        alpha,
        confidence,
        prior_attempts,
        prior_results,
        min_trades,
        period=_period_label(base_config.start, base_config.end),
    )
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

    refused = [c for c in outcomes if not c.counts_as_attempt]
    if refused:
        warnings.append(
            f"{len(refused)} of {len(cells)} cells were refused before testing "
            f"because the spread exceeds {max_spread_atr:.0%} of one ATR: they are "
            f"not counted as attempts, because nothing was attempted on them"
        )
    altered = [c for c in outcomes if c.gates_materially_altered]
    if altered:
        busted = [
            c
            for c in altered
            if (c.equity_rejected_share or 0.0)
            >= max(
                c.structural_rejected_share or 0.0,
                c.discretionary_rejected_share or 0.0,
            )
        ]
        warnings.append(
            f"{len(altered)} cells had more than "
            f"{gate_alteration_threshold:.0%} of their signals rejected before "
            f"becoming trades: on those the result describes the block as much "
            f"as the strategy written in the JSON"
        )
        if busted:
            warnings.append(
                f"on {len(busted)} of those, the largest single cause is that "
                f"the account fell below the broker's minimum lot. Those runs "
                f"did not decline trades, they ran out of money: read their "
                f"equity curve, not their trade count"
            )
    untrustworthy = [c for c in outcomes if c.spread_trustworthy is False]
    if untrustworthy:
        warnings.append(
            f"{len(untrustworthy)} cells charged a spread that cannot be a real "
            f"fill cost (an aggregated bar column, or zero on some bars): their "
            f"net PnL is optimistic by an unmeasured amount"
        )

    survivors = [c for c in outcomes if c.stage_reached == "permutation"]
    verdict = (
        f"{len(cells)} cells in the grid, {len(refused)} refused before testing, "
        f"{len(cells) - len(refused)} screened, {panel.cells_backtested} reached a "
        f"backtest, {len(survivors)} were worth a permutation test. "
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
        tradability=table.as_dict() if table else None,
        manifest=manifest.to_dict(),
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
