"""Permutation tests: is the result distinguishable from luck?

Two different null hypotheses, because they fail in different ways and a
strategy can survive one while collapsing on the other.

**random_entries** - the exit logic, the costs and the risk gates stay
exactly as they are; only the *timing* of the entries is replaced by random
timestamps, keeping the trade count and the long/short mix. It answers: does
the entry rule know anything the calendar does not?

**permuted_returns** - the strategy stays exactly as it is; the price path is
rebuilt by resampling blocks of returns, which destroys the specific sequence
of events while keeping the autocorrelation and the volatility clustering
inside each block. It answers: does the edge depend on this history, or would
any history with these statistical properties do?

Both nulls go through the real `Backtester`, never a simplified re-implement-
ation of it. A null model that fills differently from the strategy it is
compared against measures the difference between the two simulators.

The reported p-value is (1 + #{null >= observed}) / (1 + N): the conservative
form, which cannot return 0 no matter how many iterations are run. A p-value
of 0.001 from a thousand iterations means "never seen in a thousand draws",
not "impossible".
"""
from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import tzinfo
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd

from core.data.provider import SymbolSpec
from core.engine.backtester import BacktestConfig, run_backtest
from core.serialization import json_safe
from core.strategy.evaluator import Signals, evaluate
from core.strategy.spec import StrategySpec

logger = logging.getLogger(__name__)

TestKind = Literal["random_entries", "permuted_returns"]

DEFAULT_ITERATIONS = 1000
# Long enough to contain the strategy's memory (indicator windows plus the
# longest holding) so a block is a self-contained piece of market, short
# enough to leave hundreds of independent blocks to resample from.
DEFAULT_BLOCK_BARS = 512
HISTOGRAM_BINS = 40
# Each worker holds its own copy of the bars and, during an iteration, the
# engine's per-bar datetime array on top. On a nine-month M1 series that is a
# few hundred megabytes per process, so more workers than this trades a
# machine's memory for very little wall clock.
MAX_WORKERS_CAP = 8

# Higher is better for all of these, so the p-value is the right tail.
STATISTICS: tuple[str, ...] = ("net_pnl", "expectancy", "sharpe")


@dataclass(frozen=True)
class Statistic:
    """One statistic of the real run against its null distribution."""

    key: str
    label: str
    observed: float
    null_mean: float
    null_std: float
    null_p05: float
    null_p50: float
    null_p95: float
    percentile: float
    p_value: float
    iterations: int
    better_than_null: bool

    def as_dict(self) -> dict[str, Any]:
        return json_safe(asdict(self))


@dataclass
class Histogram:
    edges: list[float]
    counts: list[int]
    observed: float

    def as_dict(self) -> dict[str, Any]:
        return json_safe(asdict(self))


@dataclass
class PermutationTestReport:
    kind: TestKind
    description: str
    iterations: int
    iterations_requested: int
    seed: int
    elapsed_seconds: float
    observed_trades: int
    null_trades_mean: float
    null_trades_std: float
    statistics: list[Statistic]
    histogram: Histogram | None
    verdict: str
    warnings: list[str] = field(default_factory=list)
    block_bars: int | None = None

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["statistics"] = [statistic.as_dict() for statistic in self.statistics]
        payload["histogram"] = self.histogram.as_dict() if self.histogram else None
        return json_safe(payload)


# -- statistics of a single simulated run --------------------------------


def _statistics(trades: pd.DataFrame, equity: pd.Series, initial_equity: float) -> dict[str, float]:
    """The few numbers a null iteration has to return. Cheap on purpose."""
    if not len(trades):
        return {"net_pnl": 0.0, "expectancy": 0.0, "sharpe": 0.0, "trades": 0.0}
    pnl = trades["net_pnl"].astype("float64").to_numpy()
    total = float(pnl.sum())
    expectancy = float(pnl.mean())
    # Sharpe on the per-trade series, not annualized: the null draws have
    # different trade counts, and a per-trade ratio stays comparable.
    std = float(pnl.std(ddof=1)) if len(pnl) > 1 else 0.0
    sharpe = expectancy / std if std > 0 else 0.0
    return {
        "net_pnl": total,
        "expectancy": expectancy,
        "sharpe": sharpe,
        "trades": float(len(trades)),
    }


# -- null model A: random entries ----------------------------------------


def random_entry_signals(
    index: pd.DatetimeIndex,
    n_long: int,
    n_short: int,
    rng: np.random.Generator,
    indicators: dict[str, pd.Series] | None = None,
) -> Signals:
    """`n_long` + `n_short` entry signals dropped at random bars.

    Placed without replacement so two entries never land on the same bar,
    which the real signal series also cannot do.

    `indicators` carries the strategy's own series through: the bars are the
    real ones here, only the entry timing is randomized, so an ATR-sized stop
    must be sized off exactly the same ATR the strategy would have read.
    """
    total = n_long + n_short
    long_flags = np.zeros(len(index), dtype=bool)
    short_flags = np.zeros(len(index), dtype=bool)
    if total and len(index):
        picked = rng.choice(len(index), size=min(total, len(index)), replace=False)
        long_flags[picked[:n_long]] = True
        short_flags[picked[n_long:]] = True
    return Signals(
        long=pd.Series(long_flags, index=index, name="long"),
        short=pd.Series(short_flags, index=index, name="short"),
        indicators=dict(indicators or {}),
    )


# -- null model B: block bootstrap of the price path ---------------------


def block_bootstrap_bars(
    bars: pd.DataFrame, block_bars: int, rng: np.random.Generator
) -> pd.DataFrame:
    """A synthetic price path resampled in blocks of consecutive returns.

    Each bar keeps its own shape: open, high and low travel with the return
    of the bar they came from, as log offsets from that bar's close. A bar
    resampled this way stays a valid bar - high above open and close, low
    below - which a naive shuffle of the four series does not guarantee.

    The index, the spread and the volumes are left untouched: shuffling the
    calendar as well would also destroy the session structure, and the test
    is about the price path.
    """
    if len(bars) < 3:
        return bars.copy()

    close = bars["close"].to_numpy(dtype="float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        returns = np.diff(np.log(close))
        open_offset = np.log(bars["open"].to_numpy(dtype="float64") / close)
        high_offset = np.log(bars["high"].to_numpy(dtype="float64") / close)
        low_offset = np.log(bars["low"].to_numpy(dtype="float64") / close)
    returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
    open_offset = np.nan_to_num(open_offset, nan=0.0, posinf=0.0, neginf=0.0)
    high_offset = np.nan_to_num(high_offset, nan=0.0, posinf=0.0, neginf=0.0)
    low_offset = np.nan_to_num(low_offset, nan=0.0, posinf=0.0, neginf=0.0)

    n_returns = len(returns)
    block = max(1, min(int(block_bars), n_returns))
    n_blocks = int(np.ceil(n_returns / block))
    starts = rng.integers(0, max(1, n_returns - block + 1), size=n_blocks)
    # source positions in the returns array; index i of returns is the move
    # from bar i to bar i+1, so bar i+1 keeps the shape of source bar i+1
    source = np.concatenate([np.arange(start, start + block) for start in starts])[:n_returns]
    source = np.clip(source, 0, n_returns - 1)

    new_close = np.empty(len(close), dtype="float64")
    new_close[0] = close[0]
    new_close[1:] = close[0] * np.exp(np.cumsum(returns[source]))

    shape = np.empty(len(close), dtype="int64")
    shape[0] = 0
    shape[1:] = source + 1

    out = bars.copy()
    out["close"] = new_close
    out["open"] = new_close * np.exp(open_offset[shape])
    out["high"] = new_close * np.exp(high_offset[shape])
    out["low"] = new_close * np.exp(low_offset[shape])
    return out


# -- worker --------------------------------------------------------------

_CONTEXT: dict[str, Any] = {}


def _init_worker(
    kind: TestKind,
    spec: StrategySpec,
    bars: pd.DataFrame,
    symbol_spec: SymbolSpec,
    server_tz: tzinfo,
    backtest_config: BacktestConfig,
    n_long: int,
    n_short: int,
    block_bars: int,
    indicators: dict[str, pd.Series],
) -> None:
    """Ships the bars to each worker once, not once per iteration."""
    _CONTEXT.update(
        kind=kind,
        spec=spec,
        bars=bars,
        symbol_spec=symbol_spec,
        server_tz=server_tz,
        backtest_config=backtest_config,
        n_long=n_long,
        n_short=n_short,
        block_bars=block_bars,
        indicators=indicators,
    )


def _iteration(seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    kind: TestKind = _CONTEXT["kind"]
    spec: StrategySpec = _CONTEXT["spec"]
    bars: pd.DataFrame = _CONTEXT["bars"]
    config: BacktestConfig = _CONTEXT["backtest_config"]

    if kind == "random_entries":
        signals = random_entry_signals(
            bars.index, _CONTEXT["n_long"], _CONTEXT["n_short"], rng,
            _CONTEXT.get("indicators"),
        )
        result = run_backtest(
            spec, bars, _CONTEXT["symbol_spec"], _CONTEXT["server_tz"], config, signals
        )
    else:
        synthetic = block_bootstrap_bars(bars, _CONTEXT["block_bars"], rng)
        result = run_backtest(
            spec, synthetic, _CONTEXT["symbol_spec"], _CONTEXT["server_tz"], config
        )
    return _statistics(result.trades, result.equity, config.initial_equity)


# -- the test ------------------------------------------------------------


def _histogram(null: np.ndarray, observed: float) -> Histogram:
    """Bins of the null distribution, robust to a degenerate one.

    Nothing forces the draws to spread out: a null where every iteration
    lands on the same value is unusual but legitimate, and it must come back
    as one bar rather than an exception from numpy.
    """
    distinct = np.unique(null[np.isfinite(null)])
    edges: np.ndarray | None = None
    if len(distinct) >= 2:
        low, high = float(distinct[0]), float(distinct[-1])
        # never ask for more bins than there are distinct values, and drop the
        # attempt entirely if the span is too narrow to give strictly
        # increasing edges in floating point
        candidate = np.linspace(low, high, min(HISTOGRAM_BINS, len(distinct)) + 1)
        if np.all(candidate[:-1] < candidate[1:]):
            edges = candidate

    if edges is None:
        centre = float(distinct[0]) if len(distinct) else 0.0
        width = max(abs(centre), 1.0) * 0.01
        edges = np.array([centre - width, centre + width], dtype="float64")
        counts = np.array([int(len(null))], dtype="int64")
    else:
        counts, edges = np.histogram(null, bins=edges)
    return Histogram(
        edges=[float(edge) for edge in edges],
        counts=[int(count) for count in counts],
        observed=observed,
    )


def _empirical(observed: float, null: np.ndarray) -> tuple[float, float]:
    """(percentile of the observed value, right-tail p-value)."""
    if not len(null):
        return float("nan"), float("nan")
    below = float(np.sum(null < observed))
    percentile = below / len(null) * 100.0
    p_value = (1.0 + float(np.sum(null >= observed))) / (1.0 + len(null))
    return percentile, p_value


def permutation_test(
    kind: TestKind,
    spec: StrategySpec,
    bars: pd.DataFrame,
    symbol_spec: SymbolSpec,
    server_tz: tzinfo,
    backtest_config: BacktestConfig,
    iterations: int = DEFAULT_ITERATIONS,
    block_bars: int = DEFAULT_BLOCK_BARS,
    seed: int = 12345,
    max_workers: int | None = None,
) -> PermutationTestReport:
    """Runs one of the two null models and scores the real result against it."""
    if iterations < 1:
        raise ValueError("iterations must be at least 1")

    started = time.perf_counter()
    real = run_backtest(spec, bars, symbol_spec, server_tz, backtest_config)
    observed = _statistics(real.trades, real.equity, backtest_config.initial_equity)
    real_signals = evaluate(spec, bars, symbol_spec.point)
    counts = real_signals.counts

    warnings: list[str] = []
    if observed["trades"] == 0:
        warnings.append("the real run produced no trade: there is nothing to test")

    seeds = [seed + step for step in range(iterations)]
    init_args = (
        kind,
        spec,
        bars,
        symbol_spec,
        server_tz,
        backtest_config,
        counts["long"],
        counts["short"],
        block_bars,
        # only the random-entries null runs on the real bars, so only it can
        # (and must) reuse the strategy's own indicator series
        real_signals.indicators if kind == "random_entries" else {},
    )
    workers = max_workers or min(os.cpu_count() or 4, MAX_WORKERS_CAP)
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_init_worker, initargs=init_args
    ) as pool:
        draws = list(pool.map(_iteration, seeds, chunksize=max(1, iterations // 64)))

    elapsed = time.perf_counter() - started
    null_trades = np.array([draw["trades"] for draw in draws], dtype="float64")

    statistics: list[Statistic] = []
    histogram: Histogram | None = None
    for key in STATISTICS:
        null = np.array([draw[key] for draw in draws], dtype="float64")
        null = null[np.isfinite(null)]
        if not len(null):
            continue
        percentile, p_value = _empirical(observed[key], null)
        statistics.append(
            Statistic(
                key=key,
                label=_LABELS[key],
                observed=float(observed[key]),
                null_mean=float(null.mean()),
                null_std=float(null.std(ddof=1)) if len(null) > 1 else 0.0,
                null_p05=float(np.percentile(null, 5)),
                null_p50=float(np.percentile(null, 50)),
                null_p95=float(np.percentile(null, 95)),
                percentile=percentile,
                p_value=p_value,
                iterations=int(len(null)),
                better_than_null=bool(observed[key] > np.median(null)),
            )
        )
        if key == "net_pnl":
            histogram = _histogram(null, float(observed[key]))

    if kind == "random_entries":
        mean_trades = float(null_trades.mean()) if len(null_trades) else 0.0
        if observed["trades"] and abs(mean_trades - observed["trades"]) / observed["trades"] > 0.2:
            warnings.append(
                f"the null draws average {mean_trades:.0f} trades against "
                f"{observed['trades']:.0f} in the real run: compare the per-trade "
                f"statistics (expectancy, Sharpe), not the total"
            )

    report = PermutationTestReport(
        kind=kind,
        description=_DESCRIPTIONS[kind],
        iterations=len(draws),
        iterations_requested=iterations,
        seed=seed,
        elapsed_seconds=elapsed,
        observed_trades=int(observed["trades"]),
        null_trades_mean=float(null_trades.mean()) if len(null_trades) else 0.0,
        null_trades_std=float(null_trades.std(ddof=1)) if len(null_trades) > 1 else 0.0,
        statistics=statistics,
        histogram=histogram,
        verdict=_verdict(kind, statistics, int(observed["trades"])),
        warnings=warnings,
        block_bars=block_bars if kind == "permuted_returns" else None,
    )
    logger.info(
        "%s: %d iterations in %.1fs, %s",
        kind,
        len(draws),
        elapsed,
        ", ".join(f"{s.key} p={s.p_value:.4f}" for s in statistics),
    )
    return report


_LABELS: dict[str, str] = {
    "net_pnl": "Net PnL (account currency)",
    "expectancy": "Expectancy per trade",
    "sharpe": "Sharpe per trade (not annualized)",
}

_DESCRIPTIONS: dict[TestKind, str] = {
    "random_entries": (
        "Same exit rules, same costs, same risk gates, same number of long and "
        "short entry signals - but placed at random timestamps. If the real "
        "result sits inside this distribution, the entry rule is not choosing "
        "its moments better than a coin would."
    ),
    "permuted_returns": (
        "Same strategy, run on price paths rebuilt by resampling blocks of "
        "consecutive returns. Autocorrelation and volatility clustering survive "
        "inside a block; the specific sequence of events does not. If the real "
        "result sits inside this distribution, the edge is a property of the "
        "return distribution, not of this history."
    ),
}


def _verdict(kind: TestKind, statistics: Sequence[Statistic], trades: int) -> str:
    if not statistics:
        return "No usable statistic: the null produced nothing to compare against."
    parts: list[str] = []
    for statistic in statistics:
        parts.append(
            f"{statistic.label}: observed {statistic.observed:+.4f} against a null "
            f"median of {statistic.null_p50:+.4f} "
            f"(percentile {statistic.percentile:.1f}, p={statistic.p_value:.4f})"
        )
    best = min(statistics, key=lambda item: item.p_value)
    if best.p_value > 0.10:
        head = (
            f"The real result is indistinguishable from the {kind.replace('_', ' ')} "
            f"null on every statistic (best p={best.p_value:.4f})."
        )
    elif best.p_value > 0.05:
        head = (
            f"Weak separation from the null (best p={best.p_value:.4f}): not enough "
            f"to claim anything, especially before correcting for the number of "
            f"configurations tried."
        )
    else:
        head = (
            f"The real result beats the null on {best.label} (p={best.p_value:.4f}). "
            f"This is one test on {trades} trades: it is not yet an edge, and it "
            f"still owes a multiple-testing correction."
        )
    return head + " " + "; ".join(parts) + "."
