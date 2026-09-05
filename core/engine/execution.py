"""The execution rules, as one bar-at-a-time state machine.

This module exists so that there is exactly **one** implementation of what
happens when a signal meets a bar. The backtester drives it over a DataFrame
it already has; the live runner drives it with bars as they close. Neither
owns the rules.

That is not a tidiness argument. The previous project's central defect was a
backtest that simulated a different system from the one that traded: signals
on ticks instead of closed bars, different risk gates, a time stop present in
simulation and absent live. Any of those is enough to make every backtest
number meaningless, and none of them is visible in the results. Sharing the
state machine makes the two provably the same system, and
`tests/test_replay_equivalence.py` is the proof.

The rules themselves, unchanged from the vectorized engine:

1. A signal is born on the **close** of bar t and executed at the **open** of
   t+1. Never on the bar that generated it.
2. Feed bars are **bid** prices. A BUY enters at `open + spread` (ask) and
   exits on the bid; a SELL enters on the bid and exits on the ask. Stop and
   target touches are tested on the same side of the book.
3. **Gaps**: a bar opening beyond the stop fills at the open, not at the
   level.
4. Stop and target both touched inside one bar: the stop is assumed and the
   trade is flagged `ambiguous`.
5. The time stop counts **session** bars, not array rows.
6. Stop and target distances are fixed at entry, from the spec in points, in
   percent of the fill, or as a multiple of an indicator read on the
   **signal** bar. Warm-up skips the entry rather than inventing a distance.
"""
from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, tzinfo
from typing import Literal

import numpy as np

from core.data.provider import SymbolSpec
from core.engine.costs import CostModel, money_per_point
from core.engine.risk import RiskGate, RiskState
from core.engine.sizing import lots_for
from core.strategy.spec import Level, StrategySpec

logger = logging.getLogger(__name__)

ExitReason = Literal[
    "stop_loss", "take_profit", "gap_stop_loss", "gap_take_profit", "time_stop",
    "signal_exit", "end_of_data",
]

TRADE_COLUMNS: tuple[str, ...] = (
    "direction",
    "entry_time",
    "entry_price",
    # the levels actually placed, in price. With ATR or percent exits the
    # distance is a function of the entry bar and cannot be reconstructed from
    # the spec afterwards, so the trade record carries it.
    "stop_level",
    "target_level",
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


@dataclass(frozen=True)
class BarInput:
    """Everything the state machine needs to know about one closed bar.

    Deliberately free of indicators and DataFrames: whoever builds this has
    already decided what the signals are, and the executor cannot reach back
    into the future because it has no access to it.

    `stop_indicator` and `target_indicator` are the values of the indicators
    the exit levels reference, read on the bar **before** this one - the bar
    whose close decided a trade executed at this open. NaN means warm-up.
    """

    time: datetime
    open: float
    high: float
    low: float
    close: float
    spread_points: float
    ordinal: int
    long: bool = False
    short: bool = False
    exit_signal: bool = False
    stop_indicator: float = float("nan")
    target_indicator: float = float("nan")


@dataclass
class Position:
    direction: int
    entry_index: int
    entry_ordinal: int
    entry_time: datetime
    entry_price: float
    entry_raw: float
    lots: float
    value_per_point: float
    stop_level: float | None
    target_level: float | None
    entry_spread_points: float
    risk_money: float


class Executor:
    """One position at a time, driven bar by bar.

    Stateful on purpose: the state is what a live runner has to reconcile
    against the broker at start-up, and hiding it inside a vectorized loop is
    what makes a backtest impossible to compare with reality.
    """

    def __init__(
        self,
        strategy: StrategySpec,
        symbol_spec: SymbolSpec,
        server_tz: tzinfo,
        costs: CostModel,
        initial_equity: float,
    ) -> None:
        if strategy.risk.max_open_positions > 1:
            raise NotImplementedError(
                "max_open_positions > 1 is not supported: the engine holds one "
                "position at a time"
            )
        self.strategy = strategy
        self.symbol = symbol_spec
        self.server_tz = server_tz
        self.costs = costs
        self.initial_equity = initial_equity
        self.gate = RiskGate(strategy.risk, server_tz)

        self.time_stop = (
            strategy.exit.time_stop.bars if strategy.exit.time_stop else None
        )

        self.risk = RiskState()
        self.blocked: Counter[str] = Counter()
        self.trades: list[dict[str, object]] = []
        self.position: Position | None = None
        self.pending_entry: int = 0
        self.pending_exit: ExitReason | None = None
        self.realized: float = initial_equity
        self.entry_attempts: int = 0
        self.index: int = -1
        self.last_bar: BarInput | None = None

    # -- the loop body ---------------------------------------------------

    def step(self, bar: BarInput) -> float:
        """Processes one closed bar. Returns the equity after it.

        The order of the phases is the whole contract: an exit decided on the
        previous close is filled before an entry decided on the same close,
        and both happen before this bar's own range is examined.
        """
        self.index += 1
        self.last_bar = bar

        # 1. exits decided on the previous close: filled at the open price
        if self.position is not None and self.pending_exit is not None:
            exit_price = (
                bar.open + self._spread_price(bar)
                if self.position.direction < 0
                else bar.open
            )
            self._close(bar, float(exit_price), self.pending_exit)
        self.pending_exit = None

        # 2. entries decided on the previous close
        if self.position is None and self.pending_entry:
            self.entry_attempts += 1
            self._try_open(self.pending_entry, bar)
        self.pending_entry = 0

        # 3. intrabar stop and target on this bar
        if self.position is not None:
            closed = self._check_levels(bar)
            if closed is not None:
                price, reason, ambiguous = closed
                self._close(bar, price, reason, ambiguous)

        # 4. time stop and signal exit: decided on close, executed at t+1
        if self.position is not None:
            elapsed = int(bar.ordinal - self.position.entry_ordinal)
            if self.time_stop is not None and elapsed >= self.time_stop:
                self.pending_exit = "time_stop"
            elif bar.exit_signal:
                self.pending_exit = "signal_exit"

        # 5. new signals from this bar's close
        signal = 0
        if bar.long and not bar.short:
            signal = 1
        elif bar.short and not bar.long:
            signal = -1
        if signal:
            if self.position is None and self.pending_exit is None:
                self.pending_entry = signal
            else:
                # a signal born while the engine is already committed never
                # reaches the risk gates: counting it here is the only way it
                # does not silently vanish from the accounting
                self.blocked["position_open"] += 1

        return self.equity(bar)

    def finalize(self) -> None:
        """Closes a position still open when the data runs out."""
        if self.position is None or self.last_bar is None:
            return
        bar = self.last_bar
        final_price = (
            bar.close + self._spread_price(bar)
            if self.position.direction < 0
            else bar.close
        )
        self._close(bar, float(final_price), "end_of_data")

    def equity(self, bar: BarInput) -> float:
        return self.realized + self._floating(bar)

    # -- entry -----------------------------------------------------------

    def _spread_price(self, bar: BarInput) -> float:
        return bar.spread_points * self.symbol.point

    def _exit_distance(
        self, level: Level | None, entry_price: float, indicator_value: float
    ) -> tuple[float | None, bool]:
        """Distance in price of a stop/target level, and whether it is usable.

        `(None, True)` means the level is simply not configured; `(None,
        False)` means it is configured but its indicator is still in warm-up,
        which is a reason to skip the entry rather than to invent a distance.
        """
        if level is None:
            return None, True
        if level.type == "points":
            return level.value * self.symbol.point, True
        if level.type == "percent":
            return entry_price * level.value / 100.0, True
        if level.type == "atr":
            if not np.isfinite(indicator_value) or indicator_value <= 0:
                return None, False
            return indicator_value * level.mult, True
        raise ValueError(f"unrecognized exit level type: {level.type}")

    def _try_open(self, direction: int, bar: BarInput) -> None:
        decision = self.gate.check_entry(bar.time, bar.spread_points, self.risk)
        if not decision.allowed:
            # by code, never by the human reason: that one carries the numbers
            # of the single decision and would give one bucket per spread value
            self.blocked[decision.code or "risk_gate"] += 1
            return

        lots = lots_for(self.strategy.sizing, self.realized, self.symbol)
        if lots <= 0:
            self.blocked["insufficient_equity"] += 1
            return

        raw = float(bar.open)
        # a BUY pays the ask, a SELL collects the bid
        entry_price = raw + self._spread_price(bar) if direction > 0 else raw
        value = money_per_point(self.symbol, lots)

        stop_distance, stop_ready = self._exit_distance(
            self.strategy.exit.stop_loss, entry_price, bar.stop_indicator
        )
        target_distance, target_ready = self._exit_distance(
            self.strategy.exit.take_profit, entry_price, bar.target_indicator
        )
        if not (stop_ready and target_ready):
            self.blocked["exit_indicator_warmup"] += 1
            return

        stop_level = target_level = None
        if stop_distance is not None:
            stop_level = (
                entry_price - stop_distance if direction > 0 else entry_price + stop_distance
            )
        if target_distance is not None:
            target_level = (
                entry_price + target_distance if direction > 0 else entry_price - target_distance
            )

        risk_money = (stop_distance / self.symbol.point * value) if stop_distance else 0.0
        self.risk.register_entry(bar.time, self.server_tz)
        self.position = Position(
            direction=direction,
            entry_index=self.index,
            entry_ordinal=bar.ordinal,
            entry_time=bar.time,
            entry_price=entry_price,
            entry_raw=raw,
            lots=lots,
            value_per_point=value,
            stop_level=stop_level,
            target_level=target_level,
            entry_spread_points=bar.spread_points,
            risk_money=risk_money,
        )

    # -- exits -----------------------------------------------------------

    def _check_levels(self, bar: BarInput) -> tuple[float, ExitReason, bool] | None:
        """Stop/target on this bar, priced on the correct side of the book."""
        position = self.position
        assert position is not None
        if position.stop_level is None and position.target_level is None:
            return None

        if position.direction > 0:
            # a long exits on the bid: the raw feed prices are already right
            bar_open, bar_high, bar_low = bar.open, bar.high, bar.low
            stop_hit_open = position.stop_level is not None and bar_open <= position.stop_level
            target_hit_open = (
                position.target_level is not None and bar_open >= position.target_level
            )
            stop_hit = position.stop_level is not None and bar_low <= position.stop_level
            target_hit = position.target_level is not None and bar_high >= position.target_level
        else:
            # a short exits on the ask: bid + this bar's spread
            offset = self._spread_price(bar)
            bar_open, bar_high, bar_low = bar.open + offset, bar.high + offset, bar.low + offset
            stop_hit_open = position.stop_level is not None and bar_open >= position.stop_level
            target_hit_open = (
                position.target_level is not None and bar_open <= position.target_level
            )
            stop_hit = position.stop_level is not None and bar_high >= position.stop_level
            target_hit = position.target_level is not None and bar_low <= position.target_level

        # gap at the open: the fill is at the price that exists, not the level
        if stop_hit_open:
            return float(bar_open), "gap_stop_loss", False
        if target_hit_open:
            return float(bar_open), "gap_take_profit", False
        if stop_hit and target_hit:
            # without tick data there is no way to know which came first:
            # assume the worse one and flag the trade
            return float(position.stop_level), "stop_loss", True
        if stop_hit:
            return float(position.stop_level), "stop_loss", False
        if target_hit:
            return float(position.target_level), "take_profit", False
        return None

    def _close(
        self,
        bar: BarInput,
        exit_price: float,
        reason: ExitReason,
        ambiguous: bool = False,
    ) -> dict[str, object]:
        position = self.position
        assert position is not None
        exit_spread_price = self._spread_price(bar)
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
            else bar.spread_points
        )
        spread_cost = spread_paid * position.value_per_point
        commission = self.costs.commission.round_turn(position.lots)
        swap = self.costs.swap.charge(
            self.symbol, position.direction, position.lots,
            position.entry_time, bar.time, self.server_tz,
        )
        net = gross - spread_cost - commission + swap

        bars_held = self.index - position.entry_index
        session_bars = int(bar.ordinal - position.entry_ordinal)

        trade: dict[str, object] = {
            "direction": position.direction,
            "entry_time": position.entry_time,
            "entry_price": position.entry_price,
            "stop_level": position.stop_level if position.stop_level is not None else np.nan,
            "target_level": (
                position.target_level if position.target_level is not None else np.nan
            ),
            "exit_time": bar.time,
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
        self.trades.append(trade)
        self.realized += net
        self.risk.register_exit(bar.time)
        self.position = None
        return trade

    # -- equity ----------------------------------------------------------

    def _floating(self, bar: BarInput) -> float:
        position = self.position
        if position is None:
            return 0.0
        # the feed close is a bid: it is already the raw price for both sides
        gross = (
            position.direction
            * (bar.close - position.entry_raw)
            / self.symbol.point
            * position.value_per_point
        )
        spread_paid = (
            position.entry_spread_points
            if position.direction > 0
            else bar.spread_points
        )
        commission = self.costs.commission.round_turn(position.lots)
        return gross - spread_paid * position.value_per_point - commission
