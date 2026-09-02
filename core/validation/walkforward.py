"""Walk-forward analysis: optimize in-sample, measure out-of-sample.

The only series in this module that may be read as performance is the
concatenated OOS curve. Every in-sample number is the result of having picked
the best of N candidates on that very data, and is therefore an upper bound
by construction, not an expectation.

Three choices that decide whether the result means anything:

1. **Windows are sized in trades, not only in days.** Ninety days of M1 data
   with eleven trades in them optimize nothing: the best candidate is the one
   that got lucky eleven times. A window whose in-sample leg holds fewer than
   `min_train_trades` trades is discarded and reported as discarded, never
   silently averaged in.
2. **An embargo separates in-sample from out-of-sample**, as wide as the
   longest holding period the strategy can produce. Without it a trade opened
   before the boundary and closed after it belongs to both legs, and the OOS
   leg inherits information from the IS one.
3. **Equity is carried across windows.** Sizing depends on equity, so
   restarting every window at the initial capital would measure a strategy
   nobody could have traded. The concatenated curve compounds.
"""
from __future__ import annotations

import copy
import itertools
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, tzinfo
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd

from core.data.provider import SymbolSpec, Timeframe
from core.engine.backtester import BacktestConfig, run_backtest
from core.metrics.performance import PerformanceReport, compute_metrics
from core.serialization import json_safe
from core.strategy.spec import SpecError, StrategySpec

logger = logging.getLogger(__name__)

WindowMode = Literal["rolling", "anchored"]
Objective = Literal["net_pnl", "sharpe", "profit_factor", "expectancy"]

DEFAULT_TRAIN_DAYS = 90
DEFAULT_TEST_DAYS = 30
DEFAULT_MIN_TRAIN_TRADES = 30

# A parameter grid maps a dotted path into the spec to the values to try:
# {"exit.stop_loss.value": [100, 150, 200]}.
ParameterGrid = dict[str, list[Any]]


class WalkForwardError(ValueError):
    """The analysis cannot run as configured, with the reason why."""


@dataclass(frozen=True)
class WalkForwardConfig:
    mode: WindowMode = "rolling"
    train_days: int = DEFAULT_TRAIN_DAYS
    test_days: int = DEFAULT_TEST_DAYS
    min_train_trades: int = DEFAULT_MIN_TRAIN_TRADES
    objective: Objective = "net_pnl"

    def validate(self) -> None:
        if self.train_days <= 0 or self.test_days <= 0:
            raise WalkForwardError("train_days and test_days must be positive")
        if self.min_train_trades < 0:
            raise WalkForwardError("min_train_trades cannot be negative")


@dataclass
class WindowResult:
    """One walk-forward step, whether it produced a result or was discarded."""

    index: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    embargo_minutes: float
    skipped: bool = False
    skip_reason: str | None = None
    candidates_evaluated: int = 0
    chosen_params: dict[str, Any] = field(default_factory=dict)
    train_trades: int = 0
    test_trades: int = 0
    train_metrics: dict[str, Any] | None = None
    test_metrics: dict[str, Any] | None = None
    equity_start: float | None = None
    equity_end: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return json_safe(asdict(self))


@dataclass
class WalkForwardReport:
    symbol: str
    timeframe: str
    mode: WindowMode
    train_days: int
    test_days: int
    min_train_trades: int
    objective: Objective
    embargo_minutes: float
    grid: ParameterGrid
    grid_size: int
    optimized: bool
    windows: list[WindowResult]
    windows_evaluated: int
    windows_skipped: int
    oos_trades: int
    oos_metrics: dict[str, Any] | None
    oos_equity: pd.Series
    degradation: list[dict[str, Any]]
    parameter_stability: list[dict[str, Any]]
    verdict: str
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "mode": self.mode,
            "train_days": self.train_days,
            "test_days": self.test_days,
            "min_train_trades": self.min_train_trades,
            "objective": self.objective,
            "embargo_minutes": self.embargo_minutes,
            "grid": self.grid,
            "grid_size": self.grid_size,
            "optimized": self.optimized,
            "windows": [window.as_dict() for window in self.windows],
            "windows_evaluated": self.windows_evaluated,
            "windows_skipped": self.windows_skipped,
            "oos_trades": self.oos_trades,
            "oos_metrics": self.oos_metrics,
            "degradation": self.degradation,
            "parameter_stability": self.parameter_stability,
            "verdict": self.verdict,
            "warnings": self.warnings,
        }
        return json_safe(payload)


# -- spec parameter substitution -----------------------------------------


def set_in(payload: dict[str, Any], path: str, value: Any) -> None:
    """Sets `payload["a"]["b"] = value` for a dotted path "a.b"."""
    parts = path.split(".")
    node: Any = payload
    for part in parts[:-1]:
        if not isinstance(node, dict) or part not in node:
            raise WalkForwardError(
                f"grid: path {path!r} does not exist in the spec (stopped at {part!r})"
            )
        node = node[part]
    if not isinstance(node, dict) or parts[-1] not in node:
        raise WalkForwardError(
            f"grid: path {path!r} does not exist in the spec (stopped at {parts[-1]!r})"
        )
    node[parts[-1]] = value


def expand_grid(grid: ParameterGrid) -> list[dict[str, Any]]:
    """Every combination of the grid, in a stable order."""
    if not grid:
        return [{}]
    keys = sorted(grid)
    for key in keys:
        if not grid[key]:
            raise WalkForwardError(f"grid: parameter {key!r} has no value to try")
    return [
        dict(zip(keys, values)) for values in itertools.product(*(grid[key] for key in keys))
    ]


def apply_params(spec: StrategySpec, params: dict[str, Any]) -> StrategySpec:
    """A copy of the spec with the grid parameters substituted and revalidated."""
    if not params:
        return spec
    payload = copy.deepcopy(spec.model_dump(mode="json"))
    for path, value in params.items():
        set_in(payload, path, value)
    try:
        return StrategySpec.from_dict(payload, "<walk-forward candidate>")
    except SpecError as exc:
        raise WalkForwardError(f"invalid candidate {params}: {exc}") from exc


# -- embargo -------------------------------------------------------------


def embargo_minutes(spec: StrategySpec, trades: pd.DataFrame | None = None) -> float:
    """How long the IS and OOS legs must be kept apart.

    The longest holding a trade can reach: the spec's time stop when there is
    one, the longest holding actually observed when there are trades, and the
    larger of the two when both exist. A trade still open across the boundary
    would otherwise be scored on both sides of it.
    """
    candidates: list[float] = [0.0]
    if spec.exit.time_stop is not None:
        candidates.append(float(spec.exit.time_stop.bars * spec.instrument.tf.minutes))
    if trades is not None and len(trades):
        held = pd.to_datetime(trades["exit_time"]) - pd.to_datetime(trades["entry_time"])
        longest = held.max()
        if pd.notna(longest):
            candidates.append(float(longest.total_seconds() / 60.0))
    return max(candidates)


# -- windows -------------------------------------------------------------


def build_windows(
    start: datetime,
    end: datetime,
    config: WalkForwardConfig,
    embargo: timedelta,
) -> list[tuple[datetime, datetime, datetime, datetime]]:
    """(train_start, train_end, test_start, test_end) for every step.

    Only complete windows are produced: a truncated final test leg would be
    measured over a shorter period than the others and would not belong in the
    same table.
    """
    train = timedelta(days=config.train_days)
    test = timedelta(days=config.test_days)
    windows: list[tuple[datetime, datetime, datetime, datetime]] = []

    train_end = start + train
    while True:
        test_start = train_end + embargo
        test_end = test_start + test
        if test_end > end:
            break
        train_start = start if config.mode == "anchored" else train_end - train
        windows.append((train_start, train_end, test_start, test_end))
        train_end = train_end + test
    return windows


def _slice(bars: pd.DataFrame, start: datetime, end: datetime) -> pd.DataFrame:
    """Half-open [start, end): a bar belongs to exactly one leg."""
    left = pd.Timestamp(start).tz_convert("UTC")
    right = pd.Timestamp(end).tz_convert("UTC")
    return bars[(bars.index >= left) & (bars.index < right)]


# -- objective -----------------------------------------------------------


def objective_value(report: PerformanceReport, objective: Objective) -> float:
    """The number the in-sample choice is made on. NaN = unusable candidate."""
    if report.trades == 0:
        return float("-inf")
    if objective == "net_pnl":
        return float(report.final_equity - report.initial_equity)
    if objective == "sharpe":
        return float(report.sharpe)
    if objective == "profit_factor":
        value = float(report.profit_factor)
        # an infinite profit factor means "no losing trade yet": on a handful
        # of trades that is luck, and it must not win the comparison outright
        return value if np.isfinite(value) else float(report.trades)
    if objective == "expectancy":
        return float(report.expectancy)
    raise WalkForwardError(f"unknown objective: {objective}")


# -- the analysis --------------------------------------------------------


def _run_slice(
    spec: StrategySpec,
    bars: pd.DataFrame,
    symbol_spec: SymbolSpec,
    server_tz: tzinfo,
    backtest_config: BacktestConfig,
    label: str,
) -> tuple[PerformanceReport, pd.DataFrame, pd.Series]:
    result = run_backtest(spec, bars, symbol_spec, server_tz, backtest_config)
    report = compute_metrics(
        result.trades,
        result.equity,
        result.timeframe,
        backtest_config.initial_equity,
        label,
    )
    return report, result.trades, result.equity


def walk_forward(
    spec: StrategySpec,
    bars: pd.DataFrame,
    symbol_spec: SymbolSpec,
    server_tz: tzinfo,
    backtest_config: BacktestConfig,
    config: WalkForwardConfig | None = None,
    grid: ParameterGrid | None = None,
    reference_trades: pd.DataFrame | None = None,
) -> WalkForwardReport:
    """Optimize on each in-sample leg, apply the winner to the leg that follows."""
    config = config or WalkForwardConfig()
    config.validate()
    grid = grid or {}
    candidates = expand_grid(grid)
    optimized = len(candidates) > 1
    # fail on a bad path now, not five windows into the analysis
    specs = [apply_params(spec, params) for params in candidates]

    if bars.empty:
        raise WalkForwardError("no bars: nothing to validate")

    warnings: list[str] = []
    if not optimized:
        warnings.append(
            "no parameter grid: every window runs the same spec, so the IS->OOS "
            "comparison measures regime change over time, not overfitting from "
            "selection"
        )

    embargo = embargo_minutes(spec, reference_trades)
    windows_spans = build_windows(
        bars.index[0].to_pydatetime(),
        bars.index[-1].to_pydatetime(),
        config,
        timedelta(minutes=embargo),
    )
    if not windows_spans:
        span_days = (bars.index[-1] - bars.index[0]).total_seconds() / 86400.0
        raise WalkForwardError(
            f"the data covers {span_days:.0f} days: not enough for a "
            f"{config.train_days}-day train leg plus a {config.test_days}-day test "
            f"leg with a {embargo / 1440.0:.1f}-day embargo"
        )

    results: list[WindowResult] = []
    oos_equity_parts: list[pd.Series] = []
    oos_trades_parts: list[pd.DataFrame] = []
    equity = backtest_config.initial_equity

    for index, (train_start, train_end, test_start, test_end) in enumerate(windows_spans):
        window = WindowResult(
            index=index,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            embargo_minutes=embargo,
            candidates_evaluated=len(candidates),
        )
        train_bars = _slice(bars, train_start, train_end)
        test_bars = _slice(bars, test_start, test_end)

        if train_bars.empty or test_bars.empty:
            window.skipped = True
            window.skip_reason = "no bars in the train or test leg"
            results.append(window)
            continue

        train_config = BacktestConfig(
            initial_equity=backtest_config.initial_equity,
            costs=backtest_config.costs,
            session_threshold=backtest_config.session_threshold,
        )
        scored: list[tuple[float, int, PerformanceReport]] = []
        for candidate_index, candidate_spec in enumerate(specs):
            report, _, _ = _run_slice(
                candidate_spec, train_bars, symbol_spec, server_tz, train_config, "is"
            )
            if report.trades < config.min_train_trades:
                continue
            scored.append((objective_value(report, config.objective), candidate_index, report))

        if not scored:
            window.skipped = True
            window.skip_reason = (
                f"no candidate reaches {config.min_train_trades} trades in the "
                f"train leg: nothing to select on"
            )
            results.append(window)
            logger.info("window %d discarded: %s", index, window.skip_reason)
            continue

        # ties go to the first candidate in grid order, so the result does not
        # depend on the iteration order of a dict
        best_score, best_index, best_report = max(scored, key=lambda item: (item[0], -item[1]))
        window.chosen_params = candidates[best_index]
        window.train_trades = best_report.trades
        window.train_metrics = _metrics_subset(best_report)
        window.equity_start = equity

        test_config = BacktestConfig(
            initial_equity=equity,
            costs=backtest_config.costs,
            session_threshold=backtest_config.session_threshold,
        )
        test_report, test_trades, test_equity = _run_slice(
            specs[best_index], test_bars, symbol_spec, server_tz, test_config, "oos"
        )
        window.test_trades = test_report.trades
        window.test_metrics = _metrics_subset(test_report)
        window.equity_end = test_report.final_equity
        equity = test_report.final_equity

        if len(test_equity):
            oos_equity_parts.append(test_equity)
        if len(test_trades):
            oos_trades_parts.append(test_trades)
        results.append(window)
        logger.info(
            "window %d: %s | IS %d trades, %s %.4f -> OOS %d trades, equity %.2f",
            index,
            window.chosen_params or "fixed spec",
            best_report.trades,
            config.objective,
            best_score,
            test_report.trades,
            test_report.final_equity,
        )

    evaluated = [window for window in results if not window.skipped]
    oos_equity = (
        pd.concat(oos_equity_parts).sort_index() if oos_equity_parts else pd.Series(dtype="float64")
    )
    oos_equity = oos_equity[~oos_equity.index.duplicated(keep="last")]
    oos_trades = (
        pd.concat(oos_trades_parts, ignore_index=True)
        if oos_trades_parts
        else pd.DataFrame()
    )

    oos_metrics: dict[str, Any] | None = None
    if len(oos_equity):
        report = compute_metrics(
            oos_trades,
            oos_equity,
            spec.instrument.tf,
            backtest_config.initial_equity,
            "walk-forward OOS",
        )
        oos_metrics = _metrics_subset(report)

    return WalkForwardReport(
        symbol=spec.instrument.symbol,
        timeframe=spec.instrument.timeframe,
        mode=config.mode,
        train_days=config.train_days,
        test_days=config.test_days,
        min_train_trades=config.min_train_trades,
        objective=config.objective,
        embargo_minutes=embargo,
        grid=grid,
        grid_size=len(candidates),
        optimized=optimized,
        windows=results,
        windows_evaluated=len(evaluated),
        windows_skipped=len(results) - len(evaluated),
        oos_trades=int(len(oos_trades)),
        oos_metrics=oos_metrics,
        oos_equity=oos_equity,
        degradation=_degradation(evaluated),
        parameter_stability=_stability(evaluated, grid),
        verdict=_verdict(evaluated, results, oos_metrics, optimized),
        warnings=warnings,
    )


METRIC_KEYS: tuple[str, ...] = (
    "final_equity",
    "total_return",
    "annual_return",
    "sharpe",
    "sortino",
    "max_drawdown_pct",
    "profit_factor",
    "win_rate",
    "expectancy",
    "trades",
    "t_stat",
    "p_value",
)


def _metrics_subset(report: PerformanceReport) -> dict[str, Any]:
    payload = {key: getattr(report, key) for key in METRIC_KEYS}
    payload["net_pnl"] = report.final_equity - report.initial_equity
    payload["initial_equity"] = report.initial_equity
    payload["start"] = report.start
    payload["end"] = report.end
    return json_safe(payload)


DEGRADATION_KEYS: tuple[str, ...] = (
    "net_pnl",
    "total_return",
    "sharpe",
    "profit_factor",
    "win_rate",
    "expectancy",
)


def _degradation(windows: Sequence[WindowResult]) -> list[dict[str, Any]]:
    """Mean IS vs mean OOS per metric, with the observation count attached.

    The ratio is only reported where it means something: with an in-sample
    mean at or below zero, "OOS is 40% of IS" is arithmetic noise, so the
    field stays null and the two means speak for themselves.
    """
    rows: list[dict[str, Any]] = []
    usable = [
        window
        for window in windows
        if window.train_metrics is not None and window.test_metrics is not None
    ]
    for key in DEGRADATION_KEYS:
        train_values = [
            float(window.train_metrics[key])  # type: ignore[index]
            for window in usable
            if _finite(window.train_metrics, key)
        ]
        test_values = [
            float(window.test_metrics[key])  # type: ignore[index]
            for window in usable
            if _finite(window.test_metrics, key)
        ]
        if not train_values or not test_values:
            continue
        train_mean = float(np.mean(train_values))
        test_mean = float(np.mean(test_values))
        ratio = test_mean / train_mean if train_mean > 0 else None
        rows.append(
            json_safe(
                {
                    "metric": key,
                    "observations": len(usable),
                    "in_sample_mean": train_mean,
                    "out_of_sample_mean": test_mean,
                    "in_sample_stderr": _stderr(train_values),
                    "out_of_sample_stderr": _stderr(test_values),
                    "ratio": ratio,
                    "ratio_note": (
                        None
                        if ratio is not None
                        else "in-sample mean not positive: a ratio would not be readable"
                    ),
                }
            )
        )
    return rows


def _finite(metrics: dict[str, Any] | None, key: str) -> bool:
    if metrics is None:
        return False
    value = metrics.get(key)
    return isinstance(value, (int, float)) and np.isfinite(float(value))


def _stderr(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return None
    return float(np.std(values, ddof=1) / np.sqrt(len(values)))


def _stability(windows: Sequence[WindowResult], grid: ParameterGrid) -> list[dict[str, Any]]:
    """How often the optimizer picked the same value across windows.

    A parameter that changes at every window is not a parameter: it is the
    optimizer following noise, and whatever it selects will not survive into
    the future.
    """
    rows: list[dict[str, Any]] = []
    chosen = [window.chosen_params for window in windows if window.chosen_params]
    for key in sorted(grid):
        values = [params.get(key) for params in chosen if key in params]
        if not values:
            continue
        counts: dict[str, int] = {}
        for value in values:
            counts[str(value)] = counts.get(str(value), 0) + 1
        mode_value, mode_count = max(counts.items(), key=lambda item: item[1])
        rows.append(
            {
                "parameter": key,
                "observations": len(values),
                "distinct_values": len(counts),
                "chosen": [str(value) for value in values],
                "counts": counts,
                "mode": mode_value,
                "mode_share": mode_count / len(values),
            }
        )
    return rows


def _verdict(
    evaluated: Sequence[WindowResult],
    every: Sequence[WindowResult],
    oos_metrics: dict[str, Any] | None,
    optimized: bool,
) -> str:
    if not evaluated:
        return (
            f"No usable window: all {len(every)} were discarded for lack of trades "
            f"in the train leg. The walk-forward says nothing, in either direction."
        )
    parts: list[str] = []
    skipped = len(every) - len(evaluated)
    parts.append(
        f"{len(evaluated)} usable window(s) out of {len(every)}"
        + (f", {skipped} discarded for too few in-sample trades" if skipped else "")
    )
    if oos_metrics is not None:
        trades = int(oos_metrics.get("trades") or 0)
        net = float(oos_metrics.get("net_pnl") or 0.0)
        parts.append(
            f"concatenated out-of-sample: {trades} trades, net {net:+.2f} "
            f"in account currency"
        )
        if trades < 100:
            parts.append(
                f"{trades} out-of-sample trades carry a standard error too wide to "
                f"separate a small edge from zero: this is not a verdict, it is a "
                f"measurement with no power"
            )
    if not optimized:
        parts.append(
            "no grid was optimized, so nothing here speaks to overfitting from "
            "parameter selection"
        )
    return ". ".join(parts) + "."
