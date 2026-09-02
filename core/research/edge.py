"""Gate zero: does the signal have an edge, before even talking about SL/TP?

This module answers a single question: after the signal, does the price move
in the predicted direction more than it would after entering at a random
moment, and by enough to cover the spread?

If the answer is no, **no stop-loss/take-profit combination can make it
profitable**. SL and TP redistribute returns across trades; they do not
create them. Searching for the right SL/TP grid on a signal without an edge
is the most efficient way to find noise and call it a discovery.

That is why the gate runs before the backtest: it measures the raw signal,
with no exits, no sizing, no risk gates.

Method, for each horizon N and each direction:

- forward return in points: entry at the **open of t+1** (like the engine,
  never on the bar that generated the signal), exit at the **close of t+N**;
- mean signed by direction, standard deviation, standard error, t-stat and
  p-value against zero;
- **drift baseline**: the same average return over *all* bars, weighted by
  the long/short imbalance of the sample. On an instrument that rose 40% in
  the period, a mostly-long signal profits with no merit of its own: the
  baseline is that merit, to be subtracted;
- **average spread** on the signal bars, which is the real threshold to beat;
- verdict: is the net (mean - baseline - spread) positive **with
  significance**, yes or no.
"""
from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats

from core.engine.costs import SpreadPolicy
from core.serialization import json_safe
from core.strategy.evaluator import evaluate
from core.strategy.spec import StrategySpec

logger = logging.getLogger(__name__)

DEFAULT_HORIZONS: tuple[int, ...] = (5, 15, 30, 60, 120)
DEFAULT_MIN_OBSERVATIONS = 30
DEFAULT_SIGNIFICANCE = 0.05
# inf is not valid JSON: degenerate t-stats are clamped to a large finite value
HUGE_T = 1e6

Direction = Literal["long", "short", "both"]


@dataclass(frozen=True)
class EdgeStat:
    """One report row: one horizon, one direction."""

    horizon: int
    direction: Direction
    observations: int
    mean_points: float
    std_points: float
    stderr_points: float
    t_stat: float
    p_value: float
    drift_baseline_points: float
    spread_cost_points: float
    net_points: float
    t_vs_cost: float
    p_vs_cost: float
    beats_cost: bool
    verdict: str

    def as_dict(self) -> dict[str, object]:
        return json_safe(asdict(self))


@dataclass
class EdgeReport:
    """Full gate outcome, with the verdict spelled out."""

    symbol: str
    timeframe: str
    bars: int
    start: datetime | None
    end: datetime | None
    horizons: list[int]
    signals_long: int
    signals_short: int
    min_observations: int
    significance: float
    stats: list[EdgeStat] = field(default_factory=list)
    passed: bool = False
    verdict: str = ""
    # a-priori break-even win rate: depends only on the spec and the average
    # spread, so it belongs here, before any backtest (see metrics.breakeven)
    breakeven_prior: "dict[str, object] | None" = None

    def as_dict(self) -> dict[str, object]:
        return json_safe({
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "bars": self.bars,
            "start": self.start.isoformat() if self.start else None,
            "end": self.end.isoformat() if self.end else None,
            "horizons": self.horizons,
            "signals_long": self.signals_long,
            "signals_short": self.signals_short,
            "min_observations": self.min_observations,
            "significance": self.significance,
            "stats": [s.as_dict() for s in self.stats],
            "passed": self.passed,
            "verdict": self.verdict,
            "breakeven_prior": self.breakeven_prior,
        })

    def as_text(self) -> str:
        head = (
            f"Gate zero - {self.symbol} {self.timeframe}  "
            f"({self.bars} bars, {self.start} -> {self.end})\n"
            f"  signals: {self.signals_long} long, {self.signals_short} short\n"
            f"  {'horiz':>6} {'dir':>6} {'obs':>6} {'mean':>9} {'std err':>8} "
            f"{'t':>7} {'p':>8} {'drift':>8} {'spread':>7} {'net':>9}"
        )
        rows = [
            f"  {s.horizon:>6} {s.direction:>6} {s.observations:>6} "
            f"{s.mean_points:>9.2f} {s.stderr_points:>8.2f} {s.t_stat:>7.2f} "
            f"{s.p_value:>8.4f} {s.drift_baseline_points:>8.2f} "
            f"{s.spread_cost_points:>7.1f} {s.net_points:>9.2f}"
            + ("  <<<" if s.beats_cost else "")
            for s in self.stats
        ]
        return "\n".join([head, *rows, "", f"  VERDICT: {self.verdict}"])


def forward_points(
    bars: pd.DataFrame, point: float, horizon: int
) -> np.ndarray:
    """Forward return in points for every bar, without look-ahead.

    Entry at the open of t+1, exit at the close of t+horizon. The last bars,
    which do not have enough future, stay NaN and are excluded.
    """
    if horizon < 1:
        raise ValueError(f"invalid horizon: {horizon}")
    open_ = bars["open"].to_numpy(dtype="float64")
    close = bars["close"].to_numpy(dtype="float64")
    n = len(bars)

    forward = np.full(n, np.nan)
    last = n - horizon  # beyond this index the exit does not exist yet
    if last <= 0:
        return forward
    entry = open_[1 : last + 1]
    exit_ = close[horizon:n]
    forward[:last] = (exit_ - entry) / point
    return forward


def _stat(
    horizon: int,
    direction: Direction,
    signed: np.ndarray,
    drift: float,
    spread: float,
    min_observations: int,
    significance: float,
) -> EdgeStat:
    count = int(len(signed))
    if count < 2:
        return EdgeStat(
            horizon=horizon,
            direction=direction,
            observations=count,
            mean_points=float("nan"),
            std_points=float("nan"),
            stderr_points=float("nan"),
            t_stat=float("nan"),
            p_value=float("nan"),
            drift_baseline_points=drift,
            spread_cost_points=spread,
            net_points=float("nan"),
            t_vs_cost=float("nan"),
            p_vs_cost=float("nan"),
            beats_cost=False,
            verdict="not enough observations",
        )

    mean = float(np.mean(signed))
    std = float(np.std(signed, ddof=1))
    stderr = std / np.sqrt(count) if std > 0 else 0.0
    net = mean - drift - spread

    if stderr > 0:
        t_stat = mean / stderr
        p_value = float(2 * (1 - stats.t.cdf(abs(t_stat), df=count - 1)))
        t_vs_cost = net / stderr
        # one-sided: only a net *above* zero is interesting
        p_vs_cost = float(1 - stats.t.cdf(t_vs_cost, df=count - 1))
    else:
        # zero variance: only happens on synthetic data, but a deterministic
        # edge must not be reported as "no edge". The t is clamped to a large
        # finite value because inf is not valid JSON.
        t_stat = math.copysign(HUGE_T, mean) if mean else 0.0
        p_value = 0.0 if mean else 1.0
        t_vs_cost = math.copysign(HUGE_T, net) if net else 0.0
        p_vs_cost = 0.0 if net > 0 else 1.0

    enough = count >= min_observations
    beats = bool(enough and net > 0 and p_vs_cost < significance)
    if not enough:
        verdict = f"only {count} observations, below the minimum of {min_observations}"
    elif net <= 0:
        verdict = f"net {net:+.2f} pt: does not cover spread and drift"
    elif p_vs_cost >= significance:
        verdict = f"net {net:+.2f} pt but p={p_vs_cost:.3f}: indistinguishable from noise"
    else:
        verdict = f"net {net:+.2f} pt, p={p_vs_cost:.4f}"

    return EdgeStat(
        horizon=horizon,
        direction=direction,
        observations=count,
        mean_points=mean,
        std_points=std,
        stderr_points=stderr,
        t_stat=float(t_stat),
        p_value=p_value,
        drift_baseline_points=drift,
        spread_cost_points=spread,
        net_points=net,
        t_vs_cost=float(t_vs_cost),
        p_vs_cost=p_vs_cost,
        beats_cost=beats,
        verdict=verdict,
    )


def edge_report(
    strategy: StrategySpec,
    bars: pd.DataFrame,
    point: float,
    horizons: "list[int] | tuple[int, ...]" = DEFAULT_HORIZONS,
    spread: SpreadPolicy | None = None,
    min_observations: int = DEFAULT_MIN_OBSERVATIONS,
    significance: float = DEFAULT_SIGNIFICANCE,
) -> EdgeReport:
    """Measures the raw edge of the spec's signal, with no exits or sizing."""
    timeframe = strategy.instrument.timeframe
    if bars.empty:
        return EdgeReport(
            symbol=strategy.instrument.symbol,
            timeframe=timeframe,
            bars=0,
            start=None,
            end=None,
            horizons=list(horizons),
            signals_long=0,
            signals_short=0,
            min_observations=min_observations,
            significance=significance,
            verdict="no bars to analyze",
        )

    signals = evaluate(strategy, bars, point)
    long = signals.long.to_numpy()
    short = signals.short.to_numpy()
    # a bar with both signals is not a signal: the engine discards it
    both = long & short
    long = long & ~both
    short = short & ~both

    spread_points = (spread or SpreadPolicy()).series(bars).to_numpy()

    report = EdgeReport(
        symbol=strategy.instrument.symbol,
        timeframe=timeframe,
        bars=int(len(bars)),
        start=bars.index[0].to_pydatetime(),
        end=bars.index[-1].to_pydatetime(),
        horizons=list(horizons),
        signals_long=int(long.sum()),
        signals_short=int(short.sum()),
        min_observations=min_observations,
        significance=significance,
    )

    for horizon in horizons:
        forward = forward_points(bars, point, horizon)
        usable = ~np.isnan(forward)
        drift_all = float(np.mean(forward[usable])) if usable.any() else 0.0

        for direction, mask, sign in (
            ("long", long & usable, 1.0),
            ("short", short & usable, -1.0),
        ):
            signed = forward[mask] * sign
            cost = float(np.mean(spread_points[mask])) if mask.any() else 0.0
            report.stats.append(
                _stat(
                    horizon, direction, signed, drift_all * sign, cost,
                    min_observations, significance,
                )
            )

        combined_mask = (long | short) & usable
        signed = np.where(long[combined_mask], 1.0, -1.0) * forward[combined_mask]
        # drift only matters through the imbalance between the two sides: a
        # balanced sample cancels it on its own
        n_long = int((long & usable).sum())
        n_short = int((short & usable).sum())
        total = n_long + n_short
        imbalance = (n_long - n_short) / total if total else 0.0
        cost = float(np.mean(spread_points[combined_mask])) if combined_mask.any() else 0.0
        report.stats.append(
            _stat(
                horizon, "both", signed, drift_all * imbalance, cost,
                min_observations, significance,
            )
        )

    winners = [s for s in report.stats if s.beats_cost]
    report.passed = bool(winners)
    if not report.stats:
        report.verdict = "no horizon evaluated"
    elif report.passed:
        best = max(winners, key=lambda s: s.t_vs_cost)
        report.verdict = (
            f"PASSES on {len(winners)} of {len(report.stats)} combinations. "
            f"Best: {best.direction} at {best.horizon} bars, "
            f"net {best.net_points:+.2f} pt (p={best.p_vs_cost:.4f}). "
            f"Caution: with {len(report.stats)} parallel tests some positives "
            f"at the 5% level are expected by chance; confirm out-of-sample."
        )
    else:
        best = max(report.stats, key=lambda s: (s.net_points if np.isfinite(s.net_points) else -1e9))
        report.verdict = (
            f"DOES NOT PASS: no horizon beats the spread with significance. "
            f"The best is {best.direction} at {best.horizon} bars with net "
            f"{best.net_points:+.2f} pt over {best.observations} observations, "
            f"but p={best.p_vs_cost:.3f} (standard error {best.stderr_points:.1f} pt): "
            f"indistinguishable from zero. No SL/TP combination can rescue a "
            f"signal without an edge."
        )

    logger.info("gate zero %s %s: %s", report.symbol, report.timeframe, report.verdict)
    return report
