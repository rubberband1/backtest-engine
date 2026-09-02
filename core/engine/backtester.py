"""Event-driven backtest engine.

Iterates bar by bar. The execution rules below are what decide whether a
result is credible or a fantasy:

1. A signal is born on the **close** of bar t and executed at the **open** of
   t+1. Never on the bar that generated it.
2. MT5 feed bars are **bid** prices. A BUY enters at `open + spread` (ask) and
   exits on the bid; a SELL enters on the bid and exits on the ask. The same
   prices apply to stop and target touch tests: a stop on a short position is
   evaluated on the ask, not the bid, or it fires too late.
3. **Gaps**: if a bar opens beyond the stop, the fill is at the open price,
   not at the stop level. Same for the take profit. This is the difference
   between a real loss tail and one truncated by construction.
4. If stop and target are both touched within the **same bar**, the stop is
   assumed. The trade is flagged `ambiguous`: without tick data there is no
   way to know which came first, and the count must be watched.
5. The time stop counts **session** bars, not array rows.
6. The gates in `risk.py` are the same ones the future live runner will use.
"""
from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, tzinfo
from typing import Literal

import numpy as np
import pandas as pd

from core.data.provider import SymbolSpec, Timeframe
from core.engine.costs import CostModel, money_per_point, points_to_price
from core.engine.risk import RiskGate, RiskState
from core.engine.session import session_ordinals
from core.engine.sizing import lots_for
from core.strategy.evaluator import Signals, evaluate
from core.strategy.spec import StrategySpec

logger = logging.getLogger(__name__)

ExitReason = Literal[
    "stop_loss", "take_profit", "gap_stop_loss", "gap_take_profit", "time_stop",
    "signal_exit", "end_of_data",
]

TRADE_COLUMNS: tuple[str, ...] = (
    "direction",
    "entry_time",
    "entry_price",
    "exit_time",
    "exit_price",
    "exit_reason",
    "lots",
    "bars_held",
    "session_bars_held",
    "gross_pnl",
    "spread_points",
    "spread_cost",
    "commission",
    "swap",
    "net_pnl",
    "risk_money",
    "r_multiple",
    "ambiguous",
    "crossed_gap",
)


@dataclass
class BacktestConfig:
    """Execution parameters, not strategy parameters (those live in the JSON)."""

    initial_equity: float = 100.0
    costs: CostModel = field(default_factory=CostModel)
    session_threshold: float = 0.5


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    equity: pd.Series
    blocked: Counter[str]
    signals: Signals
    initial_equity: float
    symbol: str
    timeframe: Timeframe

    @property
    def ambiguous_trades(self) -> int:
        return int(self.trades["ambiguous"].sum()) if len(self.trades) else 0

    @property
    def gap_crossing_trades(self) -> int:
        return int(self.trades["crossed_gap"].sum()) if len(self.trades) else 0

    @property
    def final_equity(self) -> float:
        return float(self.equity.iloc[-1]) if len(self.equity) else self.initial_equity


@dataclass
class _Position:
    direction: int
    entry_index: int
    entry_time: datetime
    entry_price: float
    entry_raw: float
    lots: float
    value_per_point: float
    stop_level: float | None
    target_level: float | None
    entry_spread_points: float
    risk_money: float


def _empty_trades() -> pd.DataFrame:
    return pd.DataFrame({name: pd.Series(dtype="object") for name in TRADE_COLUMNS})


class Backtester:
    """Runs a spec over a DataFrame of bars."""

    def __init__(
        self,
        strategy: StrategySpec,
        symbol_spec: SymbolSpec,
        server_tz: tzinfo,
        config: BacktestConfig | None = None,
    ) -> None:
        if strategy.risk.max_open_positions > 1:
            raise NotImplementedError(
                "max_open_positions > 1 is not supported: the engine holds one "
                "position at a time"
            )
        self.strategy = strategy
        self.symbol = symbol_spec
        self.server_tz = server_tz
        self.config = config or BacktestConfig()
        self.gate = RiskGate(strategy.risk, server_tz)

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
        spread_points = self.config.costs.spread.series(bars).to_numpy()
        spread_price = spread_points * self.symbol.point

        open_ = bars["open"].to_numpy(dtype="float64")
        high = bars["high"].to_numpy(dtype="float64")
        low = bars["low"].to_numpy(dtype="float64")
        close = bars["close"].to_numpy(dtype="float64")
        times = bars.index.to_pydatetime()
        ordinals = session_ordinals(bars.index, timeframe, self.config.session_threshold)

        long_signal = signals.long.to_numpy()
        short_signal = signals.short.to_numpy()
        exit_signal = (
            signals.exit_signal.to_numpy()
            if signals.exit_signal is not None
            else np.zeros(len(bars), dtype=bool)
        )

        stop_points = self.strategy.exit.stop_loss.value if self.strategy.exit.stop_loss else None
        target_points = (
            self.strategy.exit.take_profit.value if self.strategy.exit.take_profit else None
        )
        time_stop = self.strategy.exit.time_stop.bars if self.strategy.exit.time_stop else None

        state = RiskState()
        blocked: Counter[str] = Counter()
        trades: list[dict[str, object]] = []
        equity_curve = np.empty(len(bars), dtype="float64")
        realized = self.config.initial_equity
        position: _Position | None = None
        pending_entry: int = 0
        pending_exit: ExitReason | None = None

        for i in range(len(bars)):
            # 1. exits decided on the previous close: filled at the open price
            if position is not None and pending_exit is not None:
                exit_price = (
                    open_[i] + spread_price[i] if position.direction < 0 else open_[i]
                )
                trades.append(
                    self._close(position, i, float(exit_price), spread_price[i],
                                pending_exit, times, ordinals, spread_points)
                )
                realized += float(trades[-1]["net_pnl"])
                state.register_exit(times[i])
                position = None
            pending_exit = None

            # 2. entries decided on the previous close
            if position is None and pending_entry:
                position = self._try_open(
                    pending_entry, i, open_, spread_points, spread_price, times, realized,
                    state, blocked, stop_points, target_points,
                )
            pending_entry = 0

            # 3. intrabar stop and target on the current bar
            if position is not None:
                closed = self._check_levels(position, i, open_, high, low, spread_price)
                if closed is not None:
                    price, reason, ambiguous = closed
                    trades.append(
                        self._close(position, i, price, spread_price[i], reason, times,
                                    ordinals, spread_points, ambiguous)
                    )
                    realized += float(trades[-1]["net_pnl"])
                    state.register_exit(times[i])
                    position = None

            # 4. time stop and signal exit: decided on close, executed at t+1
            if position is not None:
                elapsed = int(ordinals[i] - ordinals[position.entry_index])
                if time_stop is not None and elapsed >= time_stop:
                    pending_exit = "time_stop"
                elif exit_signal[i]:
                    pending_exit = "signal_exit"

            # 5. new signals from this bar's close
            if position is None and pending_exit is None:
                if long_signal[i] and not short_signal[i]:
                    pending_entry = 1
                elif short_signal[i] and not long_signal[i]:
                    pending_entry = -1

            equity_curve[i] = realized + self._floating(position, close[i], spread_price[i])

        # 6. position still open at the end of the data: closed and flagged
        if position is not None:
            last = len(bars) - 1
            final_price = (
                close[last] + spread_price[last] if position.direction < 0 else close[last]
            )
            trades.append(
                self._close(position, last, float(final_price), spread_price[last],
                            "end_of_data", times, ordinals, spread_points)
            )
            realized += float(trades[-1]["net_pnl"])
            equity_curve[last] = realized

        frame = pd.DataFrame(trades, columns=list(TRADE_COLUMNS)) if trades else _empty_trades()
        if len(frame):
            frame["entry_time"] = pd.to_datetime(frame["entry_time"], utc=True)
            frame["exit_time"] = pd.to_datetime(frame["exit_time"], utc=True)
            for column in TRADE_COLUMNS:
                if column not in ("entry_time", "exit_time", "exit_reason", "ambiguous",
                                  "crossed_gap"):
                    frame[column] = pd.to_numeric(frame[column])

        equity = pd.Series(equity_curve, index=bars.index, name="equity")
        self._log_summary(frame, blocked)
        return BacktestResult(
            frame, equity, blocked, signals, self.config.initial_equity,
            self.strategy.instrument.symbol, timeframe,
        )

    # -- entry -----------------------------------------------------------

    def _try_open(
        self,
        direction: int,
        i: int,
        open_: np.ndarray,
        spread_points: np.ndarray,
        spread_price: np.ndarray,
        times: np.ndarray,
        equity: float,
        state: RiskState,
        blocked: Counter[str],
        stop_points: float | None,
        target_points: float | None,
    ) -> _Position | None:
        decision = self.gate.check_entry(times[i], float(spread_points[i]), state)
        if not decision.allowed:
            blocked[str(decision.reason).split(":")[0]] += 1
            return None

        lots = lots_for(self.strategy.sizing, equity, self.symbol)
        if lots <= 0:
            blocked["insufficient equity"] += 1
            return None

        raw = float(open_[i])
        # a BUY pays the ask, a SELL collects the bid
        entry_price = raw + spread_price[i] if direction > 0 else raw
        value = money_per_point(self.symbol, lots)

        stop_level = target_level = None
        if stop_points is not None:
            distance = points_to_price(stop_points, self.symbol)
            stop_level = entry_price - distance if direction > 0 else entry_price + distance
        if target_points is not None:
            distance = points_to_price(target_points, self.symbol)
            target_level = entry_price + distance if direction > 0 else entry_price - distance

        risk_money = (stop_points or 0.0) * value
        state.register_entry(times[i], self.server_tz)
        return _Position(
            direction=direction,
            entry_index=i,
            entry_time=times[i],
            entry_price=entry_price,
            entry_raw=raw,
            lots=lots,
            value_per_point=value,
            stop_level=stop_level,
            target_level=target_level,
            entry_spread_points=float(spread_points[i]),
            risk_money=risk_money,
        )

    # -- exits -----------------------------------------------------------

    def _check_levels(
        self,
        position: _Position,
        i: int,
        open_: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        spread_price: np.ndarray,
    ) -> tuple[float, ExitReason, bool] | None:
        """Stop/target on bar i, priced on the correct side of the book."""
        if position.stop_level is None and position.target_level is None:
            return None

        if position.direction > 0:
            # a long exits on the bid: the raw feed prices are already right
            bar_open, bar_high, bar_low = open_[i], high[i], low[i]
            stop_hit_open = position.stop_level is not None and bar_open <= position.stop_level
            target_hit_open = position.target_level is not None and bar_open >= position.target_level
            stop_hit = position.stop_level is not None and bar_low <= position.stop_level
            target_hit = position.target_level is not None and bar_high >= position.target_level
        else:
            # a short exits on the ask: bid + this bar's spread
            offset = spread_price[i]
            bar_open, bar_high, bar_low = open_[i] + offset, high[i] + offset, low[i] + offset
            stop_hit_open = position.stop_level is not None and bar_open >= position.stop_level
            target_hit_open = position.target_level is not None and bar_open <= position.target_level
            stop_hit = position.stop_level is not None and bar_high >= position.stop_level
            target_hit = position.target_level is not None and bar_low <= position.target_level

        # gap at the open: the fill is at the price that exists, not the level
        if stop_hit_open:
            return float(bar_open), "gap_stop_loss", False
        if target_hit_open:
            return float(bar_open), "gap_take_profit", False
        if stop_hit and target_hit:
            # without tick data there is no way to know which came first: assume
            # the worse one and flag the trade
            return float(position.stop_level), "stop_loss", True
        if stop_hit:
            return float(position.stop_level), "stop_loss", False
        if target_hit:
            return float(position.target_level), "take_profit", False
        return None

    def _close(
        self,
        position: _Position,
        i: int,
        exit_price: float,
        exit_spread_price: float,
        reason: ExitReason,
        times: np.ndarray,
        ordinals: np.ndarray,
        spread_points: np.ndarray,
        ambiguous: bool = False,
    ) -> dict[str, object]:
        exit_time = times[i]
        # "raw" (bid) price, to isolate gross pnl from the spread cost
        exit_raw = exit_price if position.direction > 0 else exit_price - exit_spread_price
        gross = (
            position.direction
            * (exit_raw - position.entry_raw)
            / self.symbol.point
            * position.value_per_point
        )
        # a long pays the spread on entry, a short on exit
        spread_paid = (
            position.entry_spread_points
            if position.direction > 0
            else float(spread_points[i])
        )
        spread_cost = spread_paid * position.value_per_point
        commission = self.config.costs.commission.round_turn(position.lots)
        swap = self.config.costs.swap.charge(
            self.symbol, position.direction, position.lots,
            position.entry_time, exit_time, self.server_tz,
        )
        net = gross - spread_cost - commission + swap

        bars_held = i - position.entry_index
        session_bars = int(ordinals[i] - ordinals[position.entry_index])

        return {
            "direction": position.direction,
            "entry_time": position.entry_time,
            "entry_price": position.entry_price,
            "exit_time": exit_time,
            "exit_price": exit_price,
            "exit_reason": reason,
            "lots": position.lots,
            "bars_held": bars_held,
            "session_bars_held": session_bars,
            "gross_pnl": gross,
            "spread_points": spread_paid,
            "spread_cost": spread_cost,
            "commission": commission,
            "swap": swap,
            "net_pnl": net,
            "risk_money": position.risk_money,
            "r_multiple": net / position.risk_money if position.risk_money else np.nan,
            "ambiguous": ambiguous,
            "crossed_gap": session_bars > bars_held,
        }

    # -- equity ----------------------------------------------------------

    def _floating(
        self, position: _Position | None, close_price: float, spread_price: float
    ) -> float:
        if position is None:
            return 0.0
        # the feed close is a bid: it is already the raw price for both sides
        gross = (
            position.direction
            * (close_price - position.entry_raw)
            / self.symbol.point
            * position.value_per_point
        )
        spread_paid = (
            position.entry_spread_points
            if position.direction > 0
            else spread_price / self.symbol.point
        )
        commission = self.config.costs.commission.round_turn(position.lots)
        return gross - spread_paid * position.value_per_point - commission

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
