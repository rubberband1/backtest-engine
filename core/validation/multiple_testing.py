"""The correction owed for the number of things that were tried.

A Sharpe ratio computed once and a Sharpe ratio that won a search over forty
configurations are not the same number, even when they are numerically
identical. The second one had forty chances to be high. This module counts
how many chances were actually taken - from the run store, not from a figure
typed in by hand - and says what the observed Sharpe is worth after that.

Three instruments, reported together because they answer different questions:

- **Deflated Sharpe Ratio** (Bailey & Lopez de Prado, 2014): the probability
  that the true Sharpe is above zero, after subtracting the Sharpe that the
  best of N independent trials would be expected to reach by luck alone, and
  after correcting for the skew and the fat tails of the return series. Its
  input is the *per-trade* Sharpe, never the annualized one: annualizing
  multiplies the number by sqrt(periods) and would silently inflate the
  correction.
- **Bonferroni**: the blunt threshold. Controls the chance of even one false
  positive across the whole family, at the cost of missing real ones.
- **Benjamini-Hochberg**: controls the expected *share* of false positives
  among the discoveries. Less severe, and the more sensible default when
  several configurations are being screened rather than one being confirmed.

Both thresholds are reported, never one alone: they disagree exactly when the
answer is delicate, and hiding that disagreement is how a screening result
gets sold as a confirmation.

A warning that applies to every number here: the p-values come from a
two-sided t-test on mean trade PnL. On a *losing* strategy a small p-value
means "reliably losing", not "significant edge". The sign has to be read
before the p-value, every time.
"""
from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass, field
from itertools import combinations
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from core.serialization import json_safe

logger = logging.getLogger(__name__)

DEFAULT_ALPHA = 0.05
EULER_MASCHERONI = 0.5772156649015329
# CSCV splits the record into S disjoint slices and evaluates every way of
# halving them. 16 gives 12870 splits: enough resolution, still instant.
DEFAULT_CSCV_SLICES = 16


@dataclass(frozen=True)
class Trial:
    """One configuration that was tried on this symbol and period."""

    run_id: str
    strategy_id: str
    spec_hash: str
    trades: int
    sharpe_per_trade: float
    net_pnl: float
    p_value: float | None

    def as_dict(self) -> dict[str, Any]:
        return json_safe(asdict(self))


@dataclass
class DeflatedSharpe:
    valid: bool
    reason: str | None = None
    observed_sharpe_per_trade: float | None = None
    observed_sharpe_annualized: float | None = None
    expected_max_sharpe: float | None = None
    deflated_sharpe: float | None = None
    probabilistic_sharpe: float | None = None
    trials: int = 0
    observations: int = 0
    skewness: float | None = None
    kurtosis: float | None = None
    variance_across_trials: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return json_safe(asdict(self))


@dataclass
class ThresholdRow:
    run_id: str
    strategy_id: str
    p_value: float
    rank: int
    bonferroni_threshold: float
    benjamini_hochberg_threshold: float
    passes_bonferroni: bool
    passes_benjamini_hochberg: bool
    mean_pnl_positive: bool

    def as_dict(self) -> dict[str, Any]:
        return json_safe(asdict(self))


@dataclass
class PBOReport:
    valid: bool
    reason: str | None = None
    pbo: float | None = None
    splits: int = 0
    slices: int = 0
    candidates: int = 0
    logits: list[float] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return json_safe(asdict(self))


@dataclass
class MultipleTestingReport:
    run_id: str
    symbol: str
    period_start: Any
    period_end: Any
    trials: int
    trials_detail: list[Trial]
    alpha: float
    deflated_sharpe: DeflatedSharpe
    thresholds: list[ThresholdRow]
    pbo: PBOReport
    verdict: str
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "run_id": self.run_id,
            "symbol": self.symbol,
            "period_start": self.period_start,
            "period_end": self.period_end,
            "trials": self.trials,
            "trials_detail": [trial.as_dict() for trial in self.trials_detail],
            "alpha": self.alpha,
            "deflated_sharpe": self.deflated_sharpe.as_dict(),
            "thresholds": [row.as_dict() for row in self.thresholds],
            "pbo": self.pbo.as_dict(),
            "verdict": self.verdict,
            "warnings": self.warnings,
        }
        return json_safe(payload)


# -- per-trade Sharpe ----------------------------------------------------


def sharpe_per_trade(pnl: Sequence[float] | np.ndarray) -> float:
    """Mean over standard deviation of the trade PnL series, not annualized."""
    values = np.asarray(pnl, dtype="float64")
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return 0.0
    std = float(values.std(ddof=1))
    return float(values.mean() / std) if std > 0 else 0.0


# -- deflated Sharpe ratio -----------------------------------------------


def deflated_sharpe(
    pnl: Sequence[float] | np.ndarray,
    trial_sharpes: Sequence[float],
    annualized: float | None = None,
) -> DeflatedSharpe:
    """DSR of the observed series given the Sharpe spread across the trials.

    `trial_sharpes` must be per-trade Sharpes of every configuration tried on
    the same data: their variance is what says how much of the winner's score
    a search of that size buys for free.
    """
    values = np.asarray(pnl, dtype="float64")
    values = values[np.isfinite(values)]
    trials = len(trial_sharpes)
    observations = int(len(values))

    if observations < 3:
        return DeflatedSharpe(
            valid=False,
            reason=f"{observations} usable trades: too few for a Sharpe distribution",
            trials=trials,
            observations=observations,
        )
    if trials < 2:
        return DeflatedSharpe(
            valid=False,
            reason=(
                f"{trials} trial(s) on this symbol and period: the deflation needs "
                f"the variance of the Sharpe across at least two configurations. "
                f"Run the grid, or accept the undeflated Sharpe knowing it is "
                f"uncorrected"
            ),
            trials=trials,
            observations=observations,
            observed_sharpe_per_trade=sharpe_per_trade(values),
            observed_sharpe_annualized=annualized,
        )

    observed = sharpe_per_trade(values)
    variance = float(np.var(np.asarray(trial_sharpes, dtype="float64"), ddof=1))
    if variance <= 0:
        return DeflatedSharpe(
            valid=False,
            reason=(
                "every trial produced the same Sharpe: with zero variance across "
                "trials there is nothing to deflate"
            ),
            trials=trials,
            observations=observations,
            observed_sharpe_per_trade=observed,
            observed_sharpe_annualized=annualized,
            variance_across_trials=variance,
        )

    # Expected maximum Sharpe of N independent trials drawn from a normal with
    # this variance: the score the search hands out for free.
    n = float(trials)
    expected_max = math.sqrt(variance) * (
        (1.0 - EULER_MASCHERONI) * float(stats.norm.ppf(1.0 - 1.0 / n))
        + EULER_MASCHERONI * float(stats.norm.ppf(1.0 - 1.0 / (n * math.e)))
    )

    skew = float(stats.skew(values, bias=False))
    # non-excess kurtosis: a normal series scores 3, which is what the
    # Bailey & Lopez de Prado expression expects
    kurt = float(stats.kurtosis(values, fisher=False, bias=False))

    def _probability(benchmark: float) -> float | None:
        denominator = 1.0 - skew * observed + (kurt - 1.0) / 4.0 * observed**2
        if denominator <= 0:
            return None
        numerator = (observed - benchmark) * math.sqrt(observations - 1)
        return float(stats.norm.cdf(numerator / math.sqrt(denominator)))

    dsr = _probability(expected_max)
    psr = _probability(0.0)
    if dsr is None:
        return DeflatedSharpe(
            valid=False,
            reason=(
                "the skew/kurtosis correction term is not positive: the trade PnL "
                "distribution is too far from anything this estimator can handle"
            ),
            trials=trials,
            observations=observations,
            observed_sharpe_per_trade=observed,
            observed_sharpe_annualized=annualized,
            expected_max_sharpe=expected_max,
            skewness=skew,
            kurtosis=kurt,
            variance_across_trials=variance,
        )

    return DeflatedSharpe(
        valid=True,
        reason=None,
        observed_sharpe_per_trade=observed,
        observed_sharpe_annualized=annualized,
        expected_max_sharpe=expected_max,
        deflated_sharpe=dsr,
        probabilistic_sharpe=psr,
        trials=trials,
        observations=observations,
        skewness=skew,
        kurtosis=kurt,
        variance_across_trials=variance,
    )


# -- family-wise thresholds ----------------------------------------------


def thresholds(
    entries: Sequence[tuple[str, str, float, bool]], alpha: float = DEFAULT_ALPHA
) -> list[ThresholdRow]:
    """Bonferroni and Benjamini-Hochberg over the family of trial p-values.

    Entries are (run_id, strategy_id, p_value, mean_pnl_positive).
    """
    usable = [entry for entry in entries if entry[2] is not None and np.isfinite(entry[2])]
    total = len(usable)
    if total == 0:
        return []

    ordered = sorted(usable, key=lambda entry: entry[2])
    bonferroni = alpha / total

    # Benjamini-Hochberg: the largest rank k whose p is under k/m*alpha sets
    # the cut; everything ranked at or below it is a discovery.
    cutoff_rank = 0
    for rank, entry in enumerate(ordered, start=1):
        if entry[2] <= rank / total * alpha:
            cutoff_rank = rank

    rows: list[ThresholdRow] = []
    for rank, (run_id, strategy_id, p_value, positive) in enumerate(ordered, start=1):
        rows.append(
            ThresholdRow(
                run_id=run_id,
                strategy_id=strategy_id,
                p_value=float(p_value),
                rank=rank,
                bonferroni_threshold=bonferroni,
                benjamini_hochberg_threshold=rank / total * alpha,
                passes_bonferroni=bool(p_value <= bonferroni),
                passes_benjamini_hochberg=bool(rank <= cutoff_rank),
                mean_pnl_positive=bool(positive),
            )
        )
    return rows


# -- probability of backtest overfitting ---------------------------------


def pbo_cscv(matrix: np.ndarray, slices: int = DEFAULT_CSCV_SLICES) -> PBOReport:
    """Combinatorially symmetric cross-validation (Bailey et al., 2017).

    `matrix` is (observations x candidates) of per-period performance. The
    record is cut into `slices` disjoint blocks; every way of splitting those
    blocks into an in-sample half and an out-of-sample half is tried. Each
    time, the candidate that wins in-sample is looked up out-of-sample: PBO is
    the share of splits where that winner lands in the bottom half.

    A PBO near 0.5 means the in-sample winner is a coin flip out-of-sample -
    which is what selecting on noise looks like.
    """
    if matrix.ndim != 2:
        return PBOReport(valid=False, reason="the performance matrix must be 2-dimensional")
    rows, candidates = matrix.shape
    if candidates < 2:
        return PBOReport(
            valid=False,
            reason=(
                f"{candidates} candidate(s): CSCV compares the in-sample winner "
                f"against the others, so it needs at least two"
            ),
            candidates=candidates,
        )
    if slices % 2 != 0:
        slices -= 1
    if slices < 4 or rows < slices:
        return PBOReport(
            valid=False,
            reason=(
                f"{rows} observation(s) cannot be cut into {slices} usable slices"
            ),
            candidates=candidates,
            slices=slices,
        )

    blocks = np.array_split(np.arange(rows), slices)
    # sum per block: the performance of a candidate over that stretch
    block_scores = np.vstack([matrix[block].sum(axis=0) for block in blocks])

    half = slices // 2
    logits: list[float] = []
    for chosen in combinations(range(slices), half):
        in_sample = list(chosen)
        out_sample = [index for index in range(slices) if index not in chosen]
        is_score = block_scores[in_sample].sum(axis=0)
        oos_score = block_scores[out_sample].sum(axis=0)
        best = int(np.argmax(is_score))
        # relative rank of the in-sample winner within the OOS ordering
        rank = float(stats.rankdata(oos_score)[best]) / (candidates + 1)
        rank = min(max(rank, 1e-6), 1.0 - 1e-6)
        logits.append(math.log(rank / (1.0 - rank)))

    array = np.asarray(logits, dtype="float64")
    return PBOReport(
        valid=True,
        reason=None,
        pbo=float(np.mean(array <= 0.0)),
        splits=len(logits),
        slices=slices,
        candidates=candidates,
        logits=[float(value) for value in array],
    )


def _verdict(
    trials: int,
    dsr: DeflatedSharpe,
    rows: Sequence[ThresholdRow],
    pbo: PBOReport,
    negative: bool,
) -> str:
    parts: list[str] = [f"{trials} configuration(s) tried on this symbol and period"]

    if dsr.valid and dsr.observed_sharpe_per_trade is not None:
        parts.append(
            f"an observed per-trade Sharpe of {dsr.observed_sharpe_per_trade:+.4f} "
            f"corresponds to a deflated Sharpe of {dsr.deflated_sharpe:.4f} "
            f"(the probability the true Sharpe is above zero once the "
            f"{dsr.expected_max_sharpe:+.4f} that {trials} trials buy by luck is "
            f"subtracted)"
        )
    else:
        parts.append(f"deflated Sharpe not computable: {dsr.reason}")

    survivors = [row for row in rows if row.passes_benjamini_hochberg]
    strict = [row for row in rows if row.passes_bonferroni]
    if rows:
        parts.append(
            f"{len(strict)} of {len(rows)} trials clear Bonferroni and "
            f"{len(survivors)} clear Benjamini-Hochberg"
        )
    if pbo.valid:
        parts.append(
            f"PBO {pbo.pbo:.1%} over {pbo.splits} splits of {pbo.slices} slices"
        )
    else:
        parts.append(f"PBO not computed: {pbo.reason}")

    if negative:
        parts.append(
            "the mean trade PnL of this run is negative, so any small p-value here "
            "measures how reliably it loses, not an edge"
        )
    return ". ".join(parts) + "."


def multiple_testing_report(
    run_id: str,
    symbol: str,
    period_start: Any,
    period_end: Any,
    observed_pnl: Sequence[float] | np.ndarray,
    observed_annualized_sharpe: float | None,
    trials_detail: Sequence[Trial],
    alpha: float = DEFAULT_ALPHA,
    performance_matrix: np.ndarray | None = None,
) -> MultipleTestingReport:
    """Assembles the three corrections over the trials found in the store."""
    values = np.asarray(observed_pnl, dtype="float64")
    values = values[np.isfinite(values)]
    dsr = deflated_sharpe(
        values, [trial.sharpe_per_trade for trial in trials_detail], observed_annualized_sharpe
    )
    rows = thresholds(
        [
            (trial.run_id, trial.strategy_id, trial.p_value, trial.net_pnl > 0)
            for trial in trials_detail
            if trial.p_value is not None
        ],
        alpha,
    )
    pbo = (
        pbo_cscv(performance_matrix)
        if performance_matrix is not None
        else PBOReport(
            valid=False,
            reason=(
                "no parameter grid was supplied: CSCV needs the per-period "
                "performance of several candidates over the same data. Pass a grid "
                "to compute it"
            ),
        )
    )

    warnings: list[str] = []
    if len(trials_detail) < 2:
        warnings.append(
            "fewer than two trials were found in the run store: the deflation has "
            "nothing to work with, and an uncorrected Sharpe on a searched strategy "
            "is an overstatement of unknown size"
        )
    negative = bool(len(values) and values.mean() < 0)
    if negative:
        warnings.append(
            "mean trade PnL is negative: read the sign before any p-value on this page"
        )

    return MultipleTestingReport(
        run_id=run_id,
        symbol=symbol,
        period_start=period_start,
        period_end=period_end,
        trials=len(trials_detail),
        trials_detail=list(trials_detail),
        alpha=alpha,
        deflated_sharpe=dsr,
        thresholds=rows,
        pbo=pbo,
        verdict=_verdict(len(trials_detail), dsr, rows, pbo, negative),
        warnings=warnings,
    )


def collect_trials(
    records: Sequence[Any],
) -> list[Trial]:
    """Turns run records into the trial family, one entry per distinct spec.

    Two runs of the same spec on the same data are the same attempt, not two:
    counting them twice would inflate the correction and let a weak result
    hide behind a penalty it never earned.
    """
    seen: dict[str, Trial] = {}
    for record in records:
        trades: pd.DataFrame = record.trades()
        pnl = (
            trades["net_pnl"].astype("float64").to_numpy() if len(trades) else np.zeros(0)
        )
        strategy = (record.metrics or {}).get("strategy") or {}
        p_value = strategy.get("p_value")
        trial = Trial(
            run_id=record.run_id,
            strategy_id=record.spec.id,
            spec_hash=record.meta.spec_hash,
            trades=int(len(trades)),
            sharpe_per_trade=sharpe_per_trade(pnl),
            net_pnl=float(pnl.sum()) if len(pnl) else 0.0,
            p_value=(
                float(p_value)
                if isinstance(p_value, (int, float)) and np.isfinite(float(p_value))
                else None
            ),
        )
        seen.setdefault(record.meta.spec_hash, trial)
    return list(seen.values())
