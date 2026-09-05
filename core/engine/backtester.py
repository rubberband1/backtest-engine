"""Event-driven backtest engine.

The execution rules live in `core.engine.execution` and are shared with the
live runner: this module only turns a DataFrame into the stream of closed
bars that state machine consumes, and collects what comes out. Nothing here
decides anything about a trade.

Keeping the decisions in one place is what makes
`tests/test_replay_equivalence.py` possible: the runner replaying history has
to produce identical trades, and it does so because it runs the same code,
not because two implementations were kept in step by hand.

Signal timing, fills on gaps, ambiguous bars, the session-bar time stop and
the risk gates are documented where they are implemented.
"""
from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from datetime import tzinfo

import numpy as np
import pandas as pd

from core.data.provider import SymbolSpec, Timeframe
from core.engine.costs import CostModel, SpreadRealism
from core.engine.execution import TRADE_COLUMNS, BarInput, Executor, ExitReason
from core.engine.session import SessionCalendar, session_ordinals
from core.strategy.evaluator import Signals, evaluate
from core.strategy.spec import AtrLevel, Level, StrategySpec

logger = logging.getLogger(__name__)

__all__ = [
    "TRADE_COLUMNS",
    "BacktestConfig",
    "BacktestResult",
    "Backtester",
    "ExitReason",
    "run_backtest",
]


@dataclass
class BacktestConfig:
    """Execution parameters, not strategy parameters (those live in the JSON)."""

    initial_equity: float = 100.0
    costs: CostModel = field(default_factory=CostModel)
    session_threshold: float = 0.5
    # Pinning the session calendar makes the time stop reproducible outside
    # this DataFrame. Left None it is inferred from the bars, as before; the
    # live runner pins the one it inferred from history, so that replaying the
    # same period counts the same session bars.
    session_calendar: SessionCalendar | None = None


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    equity: pd.Series
    blocked: Counter[str]
    signals: Signals
    initial_equity: float
    symbol: str
    timeframe: Timeframe
    # signals that reached the entry stage, i.e. the denominator of the gate
    # accounting: a signal born while a position is open is one of these too
    entry_attempts: int = 0
    # what the spread policy actually charged, and whether that can be a real
    # fill cost at this timeframe. Travels with the result so no number can be
    # read without it.
    spread_realism: SpreadRealism | None = None

    @property
    def ambiguous_trades(self) -> int:
        return int(self.trades["ambiguous"].sum()) if len(self.trades) else 0

    @property
    def gap_crossing_trades(self) -> int:
        return int(self.trades["crossed_gap"].sum()) if len(self.trades) else 0

    @property
    def final_equity(self) -> float:
        return float(self.equity.iloc[-1]) if len(self.equity) else self.initial_equity


def _empty_trades() -> pd.DataFrame:
    return pd.DataFrame({name: pd.Series(dtype="object") for name in TRADE_COLUMNS})


def trades_frame(trades: list[dict[str, object]]) -> pd.DataFrame:
    """The trade records as the typed frame every consumer expects."""
    if not trades:
        return _empty_trades()
    frame = pd.DataFrame(trades, columns=list(TRADE_COLUMNS))
    frame["entry_time"] = pd.to_datetime(frame["entry_time"], utc=True)
    frame["exit_time"] = pd.to_datetime(frame["exit_time"], utc=True)
    for column in TRADE_COLUMNS:
        if column not in ("entry_time", "exit_time", "exit_reason", "ambiguous",
                          "crossed_gap"):
            frame[column] = pd.to_numeric(frame[column])
    return frame


def exit_indicator_series(
    strategy: StrategySpec, signals: Signals, length: int
) -> tuple[np.ndarray, np.ndarray]:
    """Indicator values the exit levels reference, aligned to the signal bar.

    Position i holds the value that decides the exit distance of a trade
    executed at bar i - which is the value at bar i-1, the last closed bar
    when that trade was decided. Levels that need no indicator get NaN, and
    the executor never looks at them.
    """

    def shifted(level: Level | None) -> np.ndarray:
        out = np.full(length, np.nan, dtype="float64")
        if not isinstance(level, AtrLevel):
            return out
        series = signals.indicators[level.indicator].to_numpy(dtype="float64")
        out[1:] = series[: length - 1]
        return out

    return shifted(strategy.exit.stop_loss), shifted(strategy.exit.take_profit)


class Backtester:
    """Runs a spec over a DataFrame of bars."""

    def __init__(
        self,
        strategy: StrategySpec,
        symbol_spec: SymbolSpec,
        server_tz: tzinfo,
        config: BacktestConfig | None = None,
    ) -> None:
        self.strategy = strategy
        self.symbol = symbol_spec
        self.server_tz = server_tz
        self.config = config or BacktestConfig()
        # built here so an unsupported spec fails before any data work
        self._new_executor()

    def _new_executor(self) -> Executor:
        return Executor(
            self.strategy,
            self.symbol,
            self.server_tz,
            self.config.costs,
            self.config.initial_equity,
        )

    def run(self, bars: pd.DataFrame, signals: Signals | None = None) -> BacktestResult:
        """Executes the spec over `bars`.

        `signals` replaces the ones the spec would generate. It exists for the
        null models in `core.validation.permutation`: a synthetic entry series
        has to go through *these* execution rules - same fills, same costs,
        same gates - or the comparison measures the simulator, not the edge.
        """
        timeframe = self.strategy.instrument.tf
        if bars.empty:
            logger.warning("no bars: empty backtest")
            return BacktestResult(
                _empty_trades(),
                pd.Series(dtype="float64"),
                Counter(),
                Signals(pd.Series(dtype="bool"), pd.Series(dtype="bool")),
                self.config.initial_equity,
                self.strategy.instrument.symbol,
                timeframe,
            )

        if signals is None:
            signals = evaluate(self.strategy, bars, self.symbol.point)
        realism = self.config.costs.spread.realism(bars, timeframe)
        for warning in realism.warnings:
            logger.warning("spread policy: %s", warning)

        executor = self._new_executor()
        equity_curve = np.empty(len(bars), dtype="float64")
        for position, bar in enumerate(self.bar_stream(bars, signals, timeframe)):
            equity_curve[position] = executor.step(bar)

        # a position still open at the end of the data is closed and flagged
        if executor.position is not None:
            executor.finalize()
            equity_curve[-1] = executor.realized

        frame = trades_frame(executor.trades)
        equity = pd.Series(equity_curve, index=bars.index, name="equity")
        self._log_summary(frame, executor.blocked)
        return BacktestResult(
            frame, equity, executor.blocked, signals, self.config.initial_equity,
            self.strategy.instrument.symbol, timeframe, executor.entry_attempts,
            realism,
        )

    # -- the stream ------------------------------------------------------

    def bar_stream(
        self, bars: pd.DataFrame, signals: Signals, timeframe: Timeframe
    ) -> list[BarInput]:
        """The closed bars of `bars`, as the executor sees them."""
        spread_points = self.config.costs.spread.series(bars, timeframe).to_numpy()
        open_ = bars["open"].to_numpy(dtype="float64")
        high = bars["high"].to_numpy(dtype="float64")
        low = bars["low"].to_numpy(dtype="float64")
        close = bars["close"].to_numpy(dtype="float64")
        times = bars.index.to_pydatetime()
        ordinals = session_ordinals(
            bars.index,
            timeframe,
            self.config.session_threshold,
            self.config.session_calendar,
            self.server_tz,
        )

        long_signal = signals.long.to_numpy()
        short_signal = signals.short.to_numpy()
        exit_signal = (
            signals.exit_signal.to_numpy()
            if signals.exit_signal is not None
            else np.zeros(len(bars), dtype=bool)
        )
        stop_indicator, target_indicator = exit_indicator_series(
            self.strategy, signals, len(bars)
        )

        return [
            BarInput(
                time=times[i],
                open=float(open_[i]),
                high=float(high[i]),
                low=float(low[i]),
                close=float(close[i]),
                spread_points=float(spread_points[i]),
                ordinal=int(ordinals[i]),
                long=bool(long_signal[i]),
                short=bool(short_signal[i]),
                exit_signal=bool(exit_signal[i]),
                stop_indicator=float(stop_indicator[i]),
                target_indicator=float(target_indicator[i]),
            )
            for i in range(len(bars))
        ]

    # -- diagnostics -----------------------------------------------------

    def _log_summary(self, trades: pd.DataFrame, blocked: Counter[str]) -> None:
        if not len(trades):
            logger.warning("no trades generated")
            if blocked:
                logger.info("signals blocked by the gates: %s", dict(blocked))
            return
        logger.info(
            "%d trades, %d ambiguous, %d across a session gap",
            len(trades),
            int(trades["ambiguous"].sum()),
            int(trades["crossed_gap"].sum()),
        )
        if blocked:
            logger.info("signals blocked by the gates: %s", dict(blocked))


def run_backtest(
    strategy: StrategySpec,
    bars: pd.DataFrame,
    symbol_spec: SymbolSpec,
    server_tz: tzinfo,
    config: BacktestConfig | None = None,
    signals: Signals | None = None,
) -> BacktestResult:
    return Backtester(strategy, symbol_spec, server_tz, config).run(bars, signals)
