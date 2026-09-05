"""What a strategy is about to cost, before it is run.

A backtest answers "what would this have made". By the time it answers, the
attempt has been spent: it counts against every result the search will later
produce, and it counts whether or not anyone looks at it. Most of what makes
a configuration hopeless is knowable before that, from the spec and the
instrument alone, and knowing it beforehand is the difference between a
search of forty attempts and a search of four hundred.

Four questions, none of which needs a fill to be simulated:

- **How often does this fire?** The entry conditions are evaluated and the
  signals counted. Nothing else runs - no exits, no gates, no costs - which
  is what makes it fast enough to sit next to an editor. The count is an
  upper bound on the trades: a signal born while a position is open never
  becomes one.
- **What win rate would it need?** The break-even win rate the chosen exits
  imply, against the spread this instrument was actually measured at on M1.
  A stop of 150 against a target of 80 needs 65% before costs, which is a
  fact about the arithmetic and not about the market.
- **How much of the answer will the bar resolution decide?** The share of
  bars wide enough to touch both the stop and the target, estimated from the
  same distances. A configuration whose result is decided by the stop-first
  assumption does not have a result.
- **Is this pair worth testing at all?** The tradability verdict, which is
  stage zero of the funnel and refuses cells where the spread eats more than
  15% of one ATR.

And the one question that is not about this strategy at all: **how many
attempts are already on this instrument and period, and what would a Sharpe
have to be to survive that many.** An editor makes variants cheap to try,
which is the mechanism that produces overfitting; the count is shown whether
or not it is asked for.

Under thirty expected trades the preview says so first and plainly. A result
over twenty trades is not a weak result, it is not a result.
"""
from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from core.data.cache import ParquetCache
from core.data.provider import SymbolSpec
from core.data.spread import MEANINGFUL_TIMEFRAME, SpreadReference, measure_from_cache
from core.engine.costs import CommissionModel
from core.metrics.breakeven import BreakevenPrior, breakeven_prior
from core.research.edge import AmbiguityPrior, ambiguity_prior
from core.research.tradability import DEFAULT_MAX_SPREAD_ATR, TradabilityCell, assess
from core.serialization import json_safe
from core.strategy.evaluator import evaluate
from core.strategy.spec import StrategySpec
from core.validation.multiple_testing import (
    expected_max_sharpe,
    required_sharpe_per_trade,
    sharpe_per_trade,
)

logger = logging.getLogger(__name__)

# Below this many trades a result is not weak, it is absent. The same number
# the screening campaign uses, so the editor and the funnel agree on what
# counts as an observation.
MIN_JUDGEABLE_TRADES = 30

DEFAULT_CONFIDENCE = 0.95

# What each of the two attempt counts covers, in the words the UI shows.
LOCAL_SCOPE = "this instrument, over a period overlapping this one"
OVERALL_SCOPE = "every attempt registered by this engine, on any instrument"


@dataclass
class AttemptsPanel:
    """The multiple-testing correction owed, at two different scopes.

    There are two honest answers to "how many things have been tried", and
    they give different thresholds for the same observed Sharpe:

    - **local**: attempts on *this instrument over a period overlapping this
      one*. It answers "given what has already been tried here, how good
      would this have to be?"
    - **cumulative**: every attempt registered by the engine, across all
      instruments and periods - the whole search. It answers "given
      everything that has been tried anywhere, would a discovery here be
      credible?"

    Both are reported, and both are labelled, because a reader shown one
    number in the editor and a different one in the campaign report has no
    way to tell which is wrong - and neither is. The cumulative figure is the
    one that governs a claim of discovery; the local figure is diagnostic,
    and is always the lower bar of the two.
    """

    symbol: str
    period_start: datetime | None
    period_end: datetime | None
    attempts: int
    sharpes_observed: int
    variance_across_trials: float | None
    expected_max_sharpe: float | None
    required_sharpe_per_trade: float | None
    assumed_trades: int
    confidence: float
    verdict: str
    scope: str = LOCAL_SCOPE
    overall_scope: str = OVERALL_SCOPE
    overall_attempts: int = 0
    overall_sharpes_observed: int = 0
    overall_variance_across_trials: float | None = None
    overall_expected_max_sharpe: float | None = None
    overall_required_sharpe_per_trade: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return json_safe(asdict(self))


@dataclass
class StrategyPreview:
    """Everything knowable about a spec before a single fill is simulated."""

    strategy_id: str
    symbol: str
    timeframe: str
    period_start: datetime | None
    period_end: datetime | None
    bars: int

    signals_long: int
    signals_short: int
    signals_total: int
    signals_per_1000_bars: float
    trades_upper_bound: int
    min_judgeable_trades: int
    judgeable: bool

    median_spread_points: float | None
    spread_source: str

    breakeven: dict[str, Any] | None = None
    ambiguity: dict[str, Any] | None = None
    tradability: dict[str, Any] | None = None
    attempts: dict[str, Any] | None = None

    verdict: str = ""
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return json_safe(asdict(self))

    def as_text(self) -> str:
        lines = [
            f"Preview - {self.strategy_id} on {self.symbol} {self.timeframe}",
            f"  bars                : {self.bars}",
            f"  signals             : {self.signals_total} "
            f"({self.signals_long} long, {self.signals_short} short), "
            f"{self.signals_per_1000_bars:.2f} per 1000 bars",
            f"  trades at most      : {self.trades_upper_bound}"
            + ("" if self.judgeable else "  <- not judgeable"),
        ]
        if self.median_spread_points is not None:
            lines.append(
                f"  spread charged      : {self.median_spread_points:.1f} points "
                f"({self.spread_source})"
            )
        breakeven = self.breakeven or {}
        if breakeven.get("valid"):
            lines.append(
                f"  break-even win rate : "
                f"{float(breakeven['breakeven_win_rate']):.1%}"
            )
        ambiguity = self.ambiguity or {}
        if ambiguity.get("expected_ambiguous_share") is not None:
            lines.append(
                f"  ambiguous bars      : "
                f"{float(ambiguity['expected_ambiguous_share']):.1%} (a priori)"
            )
        tradability = self.tradability or {}
        if tradability:
            lines.append(
                f"  tradability         : "
                f"{'testable' if tradability.get('tradable') else 'EXCLUDED'} - "
                f"{tradability.get('reason', '')}"
            )
        attempts = self.attempts or {}
        if attempts:
            lines.append(f"  attempts registered : {attempts.get('attempts')}")
            required = attempts.get("required_sharpe_per_trade")
            if required is not None:
                lines.append(
                    f"  Sharpe needed       : {float(required):+.4f} per trade"
                )
        lines.append("")
        lines.append(f"  {self.verdict}")
        for warning in self.warnings:
            lines.append(f"  ! {warning}")
        return "\n".join(lines)


# -- signal frequency ------------------------------------------------------


def count_signals(
    strategy: StrategySpec, bars: pd.DataFrame, point: float
) -> tuple[np.ndarray, np.ndarray, dict[str, pd.Series]]:
    """Entry signals only, with the indicators that produced them.

    Simultaneous long and short is not a signal: the executor discards those
    bars, so counting them here would promise trades that never happen.
    """
    signals = evaluate(strategy, bars, point)
    long = signals.long.to_numpy()
    short = signals.short.to_numpy()
    both = long & short
    return long & ~both, short & ~both, signals.indicators


# -- the attempts already spent -------------------------------------------


@dataclass(frozen=True)
class TrialSet:
    """One scope's attempts, reduced to what the correction actually needs.

    Separated from the run records because building it means loading every
    run from disk and computing a Sharpe for each. The whole-search scope is
    hundreds of runs and does not change between two keystrokes in the
    editor, so the caller builds this once and reuses it; `attempts_panel`
    then costs arithmetic.
    """

    attempts: int
    sharpes: tuple[float, ...]


def trial_set(
    records: Sequence[Any], min_trades: int = MIN_JUDGEABLE_TRADES
) -> TrialSet:
    """Reduces run records to `TrialSet`. The expensive half, done once."""
    sharpes: list[float] = []
    for record in records:
        trades = record.trades()
        if len(trades) < min_trades:
            # a per-trade Sharpe over a handful of trades is a ratio, not an
            # estimate: including it inflates the variance and with it the
            # threshold, until nothing can clear it
            continue
        sharpes.append(
            sharpe_per_trade(trades["net_pnl"].astype("float64").to_numpy())
        )
    return TrialSet(attempts=len(records), sharpes=tuple(sharpes))


def _correction(
    trials: TrialSet,
    assumed_trades: int,
    confidence: float,
) -> tuple[int, int, float | None, float | None, float | None]:
    """attempts, usable Sharpes, variance, expected max, required - for one scope."""
    sharpes = list(trials.sharpes)
    attempts = trials.attempts
    variance = float(np.var(sharpes, ddof=1)) if len(sharpes) > 1 else None
    expected_max = (
        expected_max_sharpe(attempts, variance) if variance and attempts else None
    )
    required = (
        required_sharpe_per_trade(
            attempts, variance, assumed_trades, confidence=confidence
        )
        if variance and attempts and assumed_trades >= 3
        else None
    )
    return attempts, len(sharpes), variance, expected_max, required


def attempts_panel(
    records: Sequence[Any],
    symbol: str,
    period_start: datetime | None,
    period_end: datetime | None,
    assumed_trades: int,
    min_trades: int = MIN_JUDGEABLE_TRADES,
    confidence: float = DEFAULT_CONFIDENCE,
    overall_trials: TrialSet | None = None,
) -> AttemptsPanel:
    """The correction owed for what has already been tried, at both scopes.

    `records` are the local attempts (this instrument, overlapping period);
    `overall_trials` summarises the whole search, built once by the caller
    with `trial_set`. When it is not supplied the local count stands alone,
    and the panel says so rather than implying it is the whole story.

    `assumed_trades` is how many trades the candidate is expected to make.
    The required Sharpe depends on it - fewer observations make the same
    score less convincing - so a preview that promises 40 trades is held to a
    lower bar than one that promises 400, and the number it was held to is
    stated rather than implied.
    """
    local = trial_set(records, min_trades)
    attempts, observed, variance, expected_max, required = _correction(
        local, assumed_trades, confidence
    )
    (
        overall_attempts,
        overall_observed,
        overall_variance,
        overall_expected,
        overall_required,
    ) = (
        _correction(overall_trials, assumed_trades, confidence)
        if overall_trials is not None
        else (attempts, observed, variance, expected_max, required)
    )

    if attempts == 0:
        verdict = (
            f"no attempt has been registered on {symbol} over this period yet. "
            f"The first result carries no multiple-testing correction - and the "
            f"second one already does."
        )
    elif required is None:
        verdict = (
            f"{attempts} attempt(s) already registered on {symbol} over this "
            f"period, but fewer than two of them produced a Sharpe over at "
            f"least {min_trades} trades, so the spread across trials cannot be "
            f"estimated and no corrected threshold can be stated."
        )
    else:
        verdict = (
            f"{attempts} attempt(s) already registered on {symbol} over this "
            f"period. A search of that size reaches {expected_max:+.4f} per "
            f"trade by luck alone, and over {assumed_trades} trades this "
            f"candidate would have to show at least {required:+.4f} to be "
            f"credible at {confidence:.0%}. Running it makes the next candidate's "
            f"bar higher still."
        )

    # the two scopes give two thresholds for the same observed value, and a
    # reader who sees only one of them cannot tell which they are looking at
    if overall_trials is not None and overall_attempts > attempts:
        if overall_required is not None:
            verdict += (
                f" That is the bar for this instrument and period alone. Across "
                f"the whole search - {overall_attempts} attempts on every "
                f"instrument - the bar is {overall_required:+.4f}, and that is "
                f"the one a claim of discovery has to clear."
            )
        else:
            verdict += (
                f" That is the bar for this instrument and period alone; the "
                f"whole search stands at {overall_attempts} attempts, too few "
                f"of them judgeable to state a corrected threshold across it."
            )

    return AttemptsPanel(
        symbol=symbol,
        period_start=period_start,
        period_end=period_end,
        attempts=attempts,
        sharpes_observed=observed,
        variance_across_trials=variance,
        expected_max_sharpe=expected_max,
        required_sharpe_per_trade=required,
        assumed_trades=assumed_trades,
        confidence=confidence,
        verdict=verdict,
        scope=LOCAL_SCOPE,
        overall_scope=OVERALL_SCOPE,
        overall_attempts=overall_attempts,
        overall_sharpes_observed=overall_observed,
        overall_variance_across_trials=overall_variance,
        overall_expected_max_sharpe=overall_expected,
        overall_required_sharpe_per_trade=overall_required,
    )


# -- the whole preview -----------------------------------------------------


def preview(
    strategy: StrategySpec,
    bars: pd.DataFrame,
    symbol_spec: SymbolSpec,
    cache: ParquetCache | None = None,
    commission: CommissionModel | None = None,
    spread_reference: SpreadReference | None = None,
    records: Sequence[Any] = (),
    period_start: datetime | None = None,
    period_end: datetime | None = None,
    max_spread_atr: float = DEFAULT_MAX_SPREAD_ATR,
    overall_trials: TrialSet | None = None,
) -> StrategyPreview:
    """Signal frequency, break-even, ambiguity, tradability and attempts."""
    timeframe = strategy.instrument.tf
    symbol = strategy.instrument.symbol
    warnings: list[str] = []

    if spread_reference is None and cache is not None:
        spread_reference = measure_from_cache(cache, symbol)
    median_spread = (
        float(spread_reference.median_points)
        if spread_reference is not None and np.isfinite(spread_reference.median_points)
        else None
    )
    spread_source = (
        f"measured on {spread_reference.source_timeframe}, "
        f"{spread_reference.usable_bars} bars"
        if spread_reference is not None and median_spread is not None
        else "unknown"
    )
    if median_spread is None:
        warnings.append(
            f"no {MEANINGFUL_TIMEFRAME.name} spread has been measured for "
            f"{symbol}: the break-even estimate below has no cost in it, and "
            f"the tradability verdict cannot be reached"
        )

    if bars.empty:
        return StrategyPreview(
            strategy_id=strategy.id, symbol=symbol, timeframe=timeframe.name,
            period_start=period_start, period_end=period_end, bars=0,
            signals_long=0, signals_short=0, signals_total=0,
            signals_per_1000_bars=0.0, trades_upper_bound=0,
            min_judgeable_trades=MIN_JUDGEABLE_TRADES, judgeable=False,
            median_spread_points=median_spread, spread_source=spread_source,
            verdict="no bars in this period: there is nothing to preview",
            warnings=warnings,
        )

    long, short, indicators = count_signals(strategy, bars, symbol_spec.point)
    total = int(long.sum() + short.sum())
    per_1000 = total / len(bars) * 1000.0

    ambiguity = ambiguity_prior(
        strategy, bars, symbol_spec.point, indicators, long | short
    )
    measured = ambiguity.as_dict()
    prior = breakeven_prior(
        strategy,
        symbol_spec,
        commission or CommissionModel(),
        median_spread,
        stop_points=measured.get("stop_points"),
        target_points=measured.get("target_points"),
        stop_points_std=measured.get("stop_points_std"),
        target_points_std=measured.get("target_points_std"),
    )

    cell: TradabilityCell | None = None
    if spread_reference is not None:
        cell = assess(
            bars, symbol, timeframe, symbol_spec.point, spread_reference,
            max_ratio=max_spread_atr,
        )

    panel = attempts_panel(
        records, symbol, period_start, period_end, assumed_trades=total,
        overall_trials=overall_trials,
    )

    judgeable = total >= MIN_JUDGEABLE_TRADES
    verdict = _verdict(
        strategy, total, len(bars), judgeable, prior, ambiguity, cell, panel
    )
    if cell is not None and cell.judged and not cell.tradable:
        warnings.append(
            f"stage zero refuses this pair: {cell.reason}. A backtest here "
            f"produces a number, and the number is noise around a known "
            f"negative constant"
        )
    if ambiguity.applicable and ambiguity.exceeds_threshold:
        warnings.append(
            f"about {ambiguity.expected_ambiguous_share:.1%} of the bars are "
            f"wide enough to touch both levels: at that rate the stop-first "
            f"assumption, not the data, decides the sign of the result"
        )

    return StrategyPreview(
        strategy_id=strategy.id,
        symbol=symbol,
        timeframe=timeframe.name,
        period_start=period_start or bars.index[0].to_pydatetime(),
        period_end=period_end or bars.index[-1].to_pydatetime(),
        bars=int(len(bars)),
        signals_long=int(long.sum()),
        signals_short=int(short.sum()),
        signals_total=total,
        signals_per_1000_bars=per_1000,
        trades_upper_bound=total,
        min_judgeable_trades=MIN_JUDGEABLE_TRADES,
        judgeable=judgeable,
        median_spread_points=median_spread,
        spread_source=spread_source,
        breakeven=prior.as_dict(),
        ambiguity=measured,
        tradability=cell.as_dict() if cell is not None else None,
        attempts=panel.as_dict(),
        verdict=verdict,
        warnings=warnings,
    )


def _verdict(
    strategy: StrategySpec,
    signals: int,
    bars: int,
    judgeable: bool,
    breakeven: BreakevenPrior,
    ambiguity: AmbiguityPrior,
    cell: TradabilityCell | None,
    panel: AttemptsPanel,
) -> str:
    """The one sentence to read if nothing else is read."""
    if signals == 0:
        return (
            f"the entry conditions never fire over these {bars} bars. Whatever "
            f"else is configured, this spec has nothing to test"
        )
    if not judgeable:
        return (
            f"{signals} signal(s) over {bars} bars, and a trade cannot be born "
            f"from more than one of them at a time: this configuration will "
            f"produce at most {signals} trades, under the "
            f"{MIN_JUDGEABLE_TRADES} needed for any of its statistics to mean "
            f"anything. Widen the period, loosen the conditions, or accept that "
            f"the result will not be judgeable"
        )
    if cell is not None and cell.judged and not cell.tradable:
        return (
            f"{signals} signals, enough to measure - but the pair is refused "
            f"before any strategy runs: {cell.reason}"
        )
    parts = [f"{signals} signals over {bars} bars, at most {signals} trades"]
    if breakeven.valid and breakeven.breakeven_win_rate is not None:
        parts.append(
            f"needing {breakeven.breakeven_win_rate:.1%} of them to win before "
            f"the search correction is even applied"
        )
    if panel.required_sharpe_per_trade is not None:
        parts.append(
            f"and a per-trade Sharpe of at least "
            f"{panel.required_sharpe_per_trade:+.4f} after it"
        )
    return ", ".join(parts) + "."
