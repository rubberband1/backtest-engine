"""The live runner: the same engine, fed one closed bar at a time.

It shares `core.engine.execution.Executor` with the backtester and
`core.engine.risk.RiskGate` with everything else. There is no second
implementation of when to enter, where the stop goes, or what the time stop
means - `tests/test_replay_equivalence.py` replays history through this class
and demands the same trades the backtester produced, to the timestamp and the
price.

What this class adds on top of the executor is the part a backtest never has
to do:

- **Only closed bars.** It keeps the timestamp of the last bar it processed
  and does nothing until a strictly newer one exists. The most recent bar the
  terminal returns is the one still forming, and it is discarded every time.
  No intra-bar evaluation, ever.
- **A pinned session calendar.** The time stop counts session bars, and the
  session is inferred from a sample. Inferring it again from a growing window
  would make the same trade time out on a different bar than the backtest
  said; it is inferred once from history and pinned.
- **Indicators advanced, not recomputed.** Every indicator carries a state
  that moves forward one bar (`core.strategy.incremental`), so the cost of a
  bar does not grow with the history behind it. The states reproduce pandas'
  arithmetic exactly rather than approximately - `tests/test_incremental.py`
  demands bit equality over five thousand bars on all nine indicators - which
  is what lets this be an optimization instead of a second opinion. Where a
  spec or a spread policy is outside what the states cover, the runner
  recomputes over the retained frame exactly as it used to, and says so in
  its diary.
- **Reconciliation at start-up.** The internal position is rebuilt from what
  the broker actually holds, matched by magic number. A position on the
  symbol that this engine did not open stops the runner: adopting it would
  mean managing a trade whose stop and target it never chose.
- **Idempotency.** A PID lock makes a second runner on the same spec refuse
  to start rather than double the position.
- **Continuity across restarts.** A forward test that runs for weeks will be
  restarted - a reboot, a terminal update, a power cut - and a runner that
  came back with a fresh equity and an empty trade list would size its next
  position off a number that never happened and hand the comparison a record
  with a hole in it. The closed trades and the realized equity are read back
  out of the diary at start-up, so the second process continues the first
  one's account rather than opening a new one.
- **A diary.** Every decision, gate outcome, order and fill is written to
  disk, so the live record can be diffed against a backtest of the same
  period (`core.live.compare`).

Dry run is the default, and the broker refuses a non-demo account even when
it is turned off.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone, tzinfo
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core.data.provider import SymbolSpec, Timeframe
from core.engine.backtester import exit_indicator_series
from core.engine.costs import AggregatedSpreadRefused, CostModel
from core.engine.execution import BarInput, Executor, Position
from core.engine.session import SessionCalendar
from core.live.broker import DEFAULT_MAGIC, LiveBroker, OpenPosition, OrderRequest
from core.live.journal import Journal, spec_payload
from core.live.lock import RunLock
from core.serialization import json_safe
from core.strategy.evaluator import evaluate
from core.strategy.incremental import (
    BarSignals,
    IncrementalEvaluator,
    UnsupportedSpec,
)
from core.strategy.spec import AtrLevel, Level, StrategySpec

# How many recent bars the runner keeps on the incremental path. Nothing
# reads them for a decision; they are there so a diary entry can be checked
# against the bars that produced it.
DIARY_BARS = 512

logger = logging.getLogger(__name__)


class ReconciliationError(RuntimeError):
    """The broker holds something this runner cannot account for."""


@dataclass
class LiveConfig:
    """Everything that is not the strategy but changes what the runner does."""

    initial_equity: float = 100.0
    costs: CostModel = field(default_factory=CostModel)
    session_threshold: float = 0.5
    magic: int = DEFAULT_MAGIC
    # bars of history loaded before the first live decision. The indicator
    # states are seeded with them one by one, so this is what decides whether
    # an EWM has converged before the first decision is taken
    warmup_bars: int = 2000
    # None keeps everything. Trimming is a memory bound and never an
    # optimization: on the recomputing path a trailing window changes
    # indicator values, which is why it is off by default. On the incremental
    # path the states carry the whole history in constant memory and the
    # retained frame is only kept for the diary, so this bounds nothing that
    # matters.
    retain_bars: int | None = None
    # Pinning the session calendar instead of inferring it from the warm-up
    # window is what makes the time stop agree with a backtest over the same
    # period. Live, it is inferred from history at start-up; in replay it is
    # the same object the backtester was given.
    session_calendar: SessionCalendar | None = None
    journal_path: Path | None = None
    lock_path: Path | None = None
    dry_run: bool = True


@dataclass
class RunnerState:
    """What the runner believes, in a form that can be compared with reality."""

    symbol: str
    timeframe: str
    last_bar_time: datetime | None
    bars_seen: int
    position: dict[str, Any] | None
    pending_entry: int
    pending_exit: str | None
    trades: int
    equity: float
    dry_run: bool
    started_at: datetime | None = None
    account_guard: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return json_safe(asdict(self))


class LiveRunner:
    """One strategy, one instrument, one position at a time."""

    def __init__(
        self,
        strategy: StrategySpec,
        symbol_spec: SymbolSpec,
        server_tz: tzinfo,
        config: LiveConfig | None = None,
        broker: LiveBroker | None = None,
    ) -> None:
        self.strategy = strategy
        self.symbol = symbol_spec
        self.server_tz = server_tz
        self.config = config or LiveConfig()
        self.timeframe: Timeframe = strategy.instrument.tf

        self.broker = broker or LiveBroker(
            dry_run=self.config.dry_run, magic=self.config.magic
        )
        self.journal = (
            Journal(self.config.journal_path) if self.config.journal_path else None
        )
        self.lock = (
            RunLock(
                self.config.lock_path,
                label=f"{strategy.id} {strategy.instrument.symbol} {self.timeframe.name}",
            )
            if self.config.lock_path
            else None
        )

        self.executor = Executor(
            strategy, symbol_spec, server_tz, self.config.costs,
            self.config.initial_equity,
        )
        self.calendar: SessionCalendar | None = None
        self.history: pd.DataFrame = pd.DataFrame()
        self.last_bar_time: datetime | None = None
        self.started_at: datetime | None = None
        self._equity: float = self.config.initial_equity

        self.evaluator: IncrementalEvaluator | None = None
        self.recompute_reason: str | None = _cannot_advance(strategy, self.config.costs)
        if self.recompute_reason is not None:
            logger.info(
                "%s: indicators will be recomputed over the retained frame on "
                "every bar (%s)",
                strategy.id,
                self.recompute_reason,
            )

    # -- lifecycle -------------------------------------------------------

    def start(self, history: pd.DataFrame) -> RunnerState:
        """Loads history, pins the session, reconciles, and takes the lock.

        `history` must end with a **closed** bar: the caller is responsible
        for dropping the forming one, because only the caller knows what time
        it is.
        """
        if self.lock is not None:
            self.lock.acquire()

        if history.empty and self.config.session_calendar is None:
            raise ValueError(
                "the runner needs history before it can decide anything: "
                "indicators have a warm-up and the session calendar is inferred "
                "from a sample. Pass history, or pin a session calendar if the "
                "caller already has one"
            )

        resumed = self._resume_from_journal()
        self.history = history.copy()
        if self.recompute_reason is None:
            # the warm-up is walked once, in order: the states end up holding
            # exactly what a recomputation over the same bars would have
            self.evaluator = IncrementalEvaluator(self.strategy, self.symbol.point)
            for row in self.history.to_dict("records"):
                self.evaluator.update(row)
        self.calendar = self.config.session_calendar or SessionCalendar.infer(
            self.history.index,
            self.timeframe,
            self.config.session_threshold,
            self.server_tz,
        )
        self.last_bar_time = (
            self.history.index[-1].to_pydatetime() if len(self.history) else None
        )
        self.started_at = datetime.now(timezone.utc)

        guard = self.broker.check_account()
        adopted = self.reconcile()

        if self.journal is not None:
            self.journal.append(
                "started",
                self.strategy.instrument.symbol,
                self.timeframe.name,
                bar_time=self.last_bar_time,
                strategy_id=self.strategy.id,
                engine_version=_engine_version(),
                symbol_spec=spec_payload(self.symbol),
                symbol_spec_hash=_symbol_spec_hash(self.symbol),
                dry_run=self.broker.dry_run,
                warmup_bars=int(len(self.history)),
                session_slots=len(self.calendar.active_slots),
                session_confidence=self.calendar.confidence,
                account_guard=guard.as_dict(),
                adopted_position=adopted.as_dict() if adopted else None,
                resumed_trades=resumed,
                resumed_equity=self.executor.realized,
            )
        return self.state()

    def _resume_from_journal(self) -> int:
        """Reads back what an earlier process of this run already did.

        Only the closed trades and the equity they left behind. The open
        position is *not* taken from here - it comes from the broker, in
        `reconcile`, because the diary records what this engine believed and
        the broker records what is actually held, and when those two differ
        the broker is right.
        """
        if self.journal is None:
            return 0
        trades = self.journal.trades()
        if not len(trades):
            return 0

        self.executor.trades = [
            {key: row[key] for key in trades.columns} for _, row in trades.iterrows()
        ]
        self.executor.realized = self.config.initial_equity + float(
            trades["net_pnl"].sum()
        )
        self._equity = self.executor.realized
        logger.info(
            "resumed from the diary: %d closed trade(s), equity %.2f",
            len(trades),
            self.executor.realized,
        )
        return int(len(trades))

    def stop(self) -> None:
        if self.journal is not None:
            self.journal.append(
                "stopped",
                self.strategy.instrument.symbol,
                self.timeframe.name,
                bar_time=self.last_bar_time,
                trades=len(self.executor.trades),
            )
        if self.lock is not None:
            self.lock.release()

    def __enter__(self) -> LiveRunner:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.stop()

    # -- reconciliation --------------------------------------------------

    def reconcile(self) -> OpenPosition | None:
        """Rebuilds the internal position from what the broker actually holds.

        Assuming the state instead of asking for it is how a restart ends up
        opening a second position, or managing one it thinks it closed. A
        position on this symbol with a foreign magic number is not adopted and
        not ignored: it stops the runner, because the engine would otherwise
        be sizing and stopping around exposure it does not control.
        """
        symbol = self.strategy.instrument.symbol
        foreign = self.broker.foreign_positions(symbol)
        if foreign:
            detail = ", ".join(
                f"ticket {p.ticket} {p.lots:g} lots" for p in foreign
            )
            message = (
                f"{len(foreign)} position(s) on {symbol} were not opened by this "
                f"engine (magic {self.config.magic}): {detail}. Refusing to start: "
                f"sizing and risk here assume this engine owns the exposure on "
                f"the symbol. Close them, or point the runner at a clean account"
            )
            if self.journal is not None:
                self.journal.append(
                    "error", symbol, self.timeframe.name,
                    reason="foreign_position", message=message,
                    positions=[p.as_dict() for p in foreign],
                )
            raise ReconciliationError(message)

        own = self.broker.own_positions(symbol)
        if len(own) > 1:
            raise ReconciliationError(
                f"the broker holds {len(own)} positions on {symbol} with this "
                f"engine's magic number, and the engine holds one at a time. "
                f"This cannot have been produced by a single runner: close them "
                f"and restart"
            )

        if not own:
            self.executor.position = None
            self.executor.risk.open_positions = 0
            return None

        adopted = own[0]
        self.executor.position = self._position_from_broker(adopted)
        self.executor.risk.open_positions = 1
        logger.info(
            "adopted the open position on %s: %s %.2f lots from %s",
            symbol,
            "long" if adopted.direction > 0 else "short",
            adopted.lots,
            adopted.opened_at,
        )
        if self.journal is not None:
            self.journal.append(
                "reconciled", symbol, self.timeframe.name,
                bar_time=self.last_bar_time, position=adopted.as_dict(),
            )
        return adopted

    def _position_from_broker(self, live: OpenPosition) -> Position:
        """A broker position expressed in the executor's terms.

        The entry ordinal is recomputed from the calendar so the time stop
        keeps counting from the real entry: a restart must not reset a trade's
        age to zero, which would leave it open indefinitely.
        """
        from core.engine.costs import money_per_point

        assert self.calendar is not None
        entry_ordinal = self.calendar.ordinal(live.opened_at)
        value = money_per_point(self.symbol, live.lots)
        stop_distance = (
            abs(live.entry_price - live.stop_level) if live.stop_level else 0.0
        )
        return Position(
            direction=live.direction,
            # the bar index is only used for `bars_held`, which after a restart
            # is a count of bars this process saw and is reported as such
            entry_index=self.executor.index,
            entry_ordinal=entry_ordinal,
            entry_time=live.opened_at,
            entry_price=live.entry_price,
            entry_raw=live.entry_price,
            lots=live.lots,
            value_per_point=value,
            stop_level=live.stop_level,
            target_level=live.target_level,
            entry_spread_points=0.0,
            risk_money=stop_distance / self.symbol.point * value,
        )

    # -- the bar loop ----------------------------------------------------

    def on_closed_bar(self, bar: pd.Series | pd.DataFrame) -> RunnerState:
        """Processes one newly closed bar. Ignores anything not strictly newer."""
        frame = bar.to_frame().T if isinstance(bar, pd.Series) else bar
        if len(frame) != 1:
            raise ValueError("on_closed_bar takes exactly one bar")

        moment = pd.DatetimeIndex(frame.index)[0].to_pydatetime()
        if self.last_bar_time is not None and moment <= self.last_bar_time:
            logger.debug("bar %s is not newer than %s: ignored", moment, self.last_bar_time)
            return self.state()

        if self.evaluator is not None:
            # the retained frame is a diary, not an input: the states already
            # hold everything the decision needs, so it is bounded here rather
            # than grown by one copy of itself per bar
            self.history = frame.copy() if self.history.empty else pd.concat(
                [self.history.iloc[-(DIARY_BARS - 1) :], frame]
            )
            row = frame.iloc[0]
            signals = self.evaluator.update(row.to_dict())
        else:
            row = None
            self.history = frame.copy() if self.history.empty else pd.concat(
                [self.history, frame]
            )
            if self.config.retain_bars is not None:
                self.history = self.history.iloc[-self.config.retain_bars :]
            signals = None
        self.last_bar_time = moment

        step_input = self._bar_input(signals, row, moment)
        position_before = self.executor.position
        self._equity = self.executor.step(step_input)
        self._journal_bar(step_input)
        self._act_on(position_before, step_input)
        return self.state()

    def _bar_input(
        self,
        signals: BarSignals | None = None,
        row: pd.Series | None = None,
        moment: datetime | None = None,
    ) -> BarInput:
        """The newest bar, as the executor sees it.

        With `signals` in hand the work is already done: the states were
        advanced by the bar that just closed and hold what the whole history
        implies. Without them the indicators are recomputed over the entire
        retained frame and not over a trailing window - an EWM over a window
        is not the same number as an EWM over the full series, and "not the
        same number" is exactly what the equivalence test exists to forbid.
        """
        assert self.calendar is not None
        if signals is not None:
            assert row is not None and moment is not None
            return self._incremental_input(signals, row, moment)
        bars = self.history
        # the parameter is one bar's signals; from here it is the whole
        # history's, which is a different type under the same name
        signals = evaluate(  # type: ignore[assignment]
            self.strategy, bars, self.symbol.point
        )
        spread = self.config.costs.spread.series(bars, self.timeframe)
        stop_indicator, target_indicator = exit_indicator_series(
            self.strategy, signals, len(bars)
        )
        last = len(bars) - 1
        row = bars.iloc[last]
        exit_signal = (
            bool(signals.exit_signal.iloc[last])
            if signals.exit_signal is not None
            else False
        )
        return BarInput(
            time=bars.index[last].to_pydatetime(),
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            spread_points=float(spread.iloc[last]),
            ordinal=int(self.calendar.ordinals(bars.index[last : last + 1])[0]),
            long=bool(signals.long.iloc[last]),
            short=bool(signals.short.iloc[last]),
            exit_signal=exit_signal,
            stop_indicator=float(stop_indicator[last]),
            target_indicator=float(target_indicator[last]),
        )

    def _incremental_input(
        self, signals: BarSignals, row: pd.Series, moment: datetime
    ) -> BarInput:
        """The same `BarInput`, built from the advanced states."""
        assert self.calendar is not None
        return BarInput(
            time=moment,
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            spread_points=self._spread_for(row),
            ordinal=self.calendar.ordinal(moment),
            long=signals.long,
            short=signals.short,
            exit_signal=signals.exit_signal,
            stop_indicator=_exit_indicator(
                self.strategy.exit.stop_loss, signals.previous_indicators
            ),
            target_indicator=_exit_indicator(
                self.strategy.exit.take_profit, signals.previous_indicators
            ),
        )

    def _spread_for(self, row: pd.Series) -> float:
        """The spread charged on one bar, without a frame to read it from.

        Only the two policies whose answer depends on this bar alone reach
        here; `_cannot_advance` keeps the rest on the recomputing path, where
        the distribution they need is available.
        """
        policy = self.config.costs.spread
        if policy.mode == "fixed":
            return float(policy.value or 0.0)
        if "spread" in row.index:
            return float(row["spread"])
        raise AggregatedSpreadRefused(
            f"spread_mode {policy.mode!r} on {self.timeframe.name}: this bar "
            f"carries no reconstructed spread, only the aggregated column"
        )

    # -- orders ----------------------------------------------------------

    def _act_on(self, position_before: Position | None, bar: BarInput) -> None:
        """Turns the executor's state change into orders on the broker.

        The executor has already decided; this only carries the decision out.
        Keeping the two apart is what lets the same decisions be replayed with
        no broker at all.
        """
        position_after = self.executor.position
        symbol = self.strategy.instrument.symbol

        if position_before is not None and position_after is None:
            trade = self.executor.trades[-1]
            live = self.broker.own_positions(symbol) if not self.broker.dry_run else []
            result = None
            if live:
                result = self.broker.close(live[0], self.symbol)
            if self.journal is not None:
                self.journal.append(
                    "position_closed", symbol, self.timeframe.name,
                    bar_time=bar.time, trade=json_safe(trade),
                    order_result=result.as_dict() if result else None,
                )

        if position_before is None and position_after is not None:
            request = OrderRequest(
                symbol=symbol,
                side="buy" if position_after.direction > 0 else "sell",
                lots=position_after.lots,
                stop_level=position_after.stop_level,
                target_level=position_after.target_level,
                comment=self.strategy.id[:31],
                magic=self.config.magic,
            )
            if self.journal is not None:
                self.journal.append(
                    "order_sent", symbol, self.timeframe.name,
                    bar_time=bar.time, request=request.as_dict(),
                    expected_price=position_after.entry_price,
                )
            result = self.broker.send(request, self.symbol)
            if self.journal is not None:
                self.journal.append(
                    "order_result", symbol, self.timeframe.name,
                    bar_time=bar.time, result=json_safe(result.as_dict()),
                    expected_price=position_after.entry_price,
                )
            if result.accepted and result.filled_lots < position_after.lots:
                # the engine sizes and stops on the lot it believes it holds:
                # believing the wrong one is worse than holding a smaller one
                logger.warning(
                    "partial fill on %s: engine state follows the broker (%.2f lots)",
                    symbol, result.filled_lots,
                )
                position_after.lots = result.filled_lots
            if not result.accepted and not result.dry_run:
                logger.error(
                    "entry rejected (%s): the engine believed it was in a "
                    "position and is not. Rolling back to flat",
                    result.retcode_name,
                )
                self.executor.position = None
                self.executor.risk.open_positions = 0

    def _journal_bar(self, bar: BarInput) -> None:
        if self.journal is None:
            return
        gates = dict(self.executor.blocked)
        self.journal.append(
            "bar",
            self.strategy.instrument.symbol,
            self.timeframe.name,
            bar_time=bar.time,
            bar={
                "open": bar.open, "high": bar.high, "low": bar.low,
                "close": bar.close, "spread_points": bar.spread_points,
                "ordinal": bar.ordinal,
            },
            signal={"long": bar.long, "short": bar.short, "exit": bar.exit_signal},
            indicators={
                "stop_level_input": _finite(bar.stop_indicator),
                "target_level_input": _finite(bar.target_indicator),
            },
            gates_cumulative=gates,
            pending_entry=self.executor.pending_entry,
            pending_exit=self.executor.pending_exit,
            in_position=self.executor.position is not None,
            equity=self._equity,
        )

    # -- polling ---------------------------------------------------------

    def closed_bars_after(
        self, bars: pd.DataFrame, now: datetime | None = None
    ) -> pd.DataFrame:
        """The bars in `bars` that are newer than the last processed and closed.

        A bar labelled t is closed once t + one timeframe has passed. The most
        recent bar the terminal returns is almost always the forming one, and
        acting on it is the exact defect this runner exists not to have.
        """
        if bars.empty:
            return bars
        moment = now or datetime.now(timezone.utc)
        span = timedelta(minutes=self.timeframe.minutes)
        closed = bars.loc[bars.index + span <= pd.Timestamp(moment)]
        if self.last_bar_time is not None:
            closed = closed.loc[closed.index > pd.Timestamp(self.last_bar_time)]
        # boolean-indexing a DataFrame returns a DataFrame; pandas-stubs types
        # `.loc[...]` as the union of everything it could return elsewhere
        return closed  # type: ignore[return-value]

    def poll(
        self,
        fetch: Callable[[datetime, datetime], pd.DataFrame],
        now: datetime | None = None,
    ) -> RunnerState:
        """Fetches, filters to closed bars, and processes them in order."""
        moment = now or datetime.now(timezone.utc)
        start = (self.last_bar_time or moment) - timedelta(
            minutes=self.timeframe.minutes * 3
        )
        try:
            fetched = fetch(start, moment)
        except Exception as exc:
            logger.warning("fetch failed, will retry on the next poll: %s", exc)
            if self.journal is not None:
                self.journal.append(
                    "error", self.strategy.instrument.symbol, self.timeframe.name,
                    reason="fetch_failed", message=f"{type(exc).__name__}: {exc}",
                )
            return self.state()

        for moment_index in self.closed_bars_after(fetched, moment).index:
            self.on_closed_bar(fetched.loc[[moment_index]])
        return self.state()

    # -- state -----------------------------------------------------------

    def state(self) -> RunnerState:
        position = self.executor.position
        return RunnerState(
            symbol=self.strategy.instrument.symbol,
            timeframe=self.timeframe.name,
            last_bar_time=self.last_bar_time,
            bars_seen=self.executor.index + 1,
            position=json_safe(asdict(position)) if position else None,
            pending_entry=self.executor.pending_entry,
            pending_exit=self.executor.pending_exit,
            trades=len(self.executor.trades),
            equity=self._equity,
            dry_run=self.broker.dry_run,
            started_at=self.started_at,
            account_guard=(
                self.broker.guard.as_dict() if self.broker.guard else None
            ),
        )


def _exit_indicator(level: Level | None, indicators: dict[str, float]) -> float:
    """The frozen indicator value an ATR exit level is sized from.

    It is read on the bar BEFORE this one - the last closed bar when the
    trade was decided - which is what `previous_indicators` holds. A level
    that needs no indicator gets NaN, and the executor never looks at it.
    """
    if not isinstance(level, AtrLevel):
        return float("nan")
    return indicators.get(level.indicator, float("nan"))


def _cannot_advance(strategy: StrategySpec, costs: CostModel) -> str | None:
    """Why this run has to recompute, or None when it can advance.

    Two things can stand in the way, and both are stated rather than worked
    around: an indicator with no exact incremental state, and a spread policy
    whose value depends on a distribution rather than on the bar. `quantile`
    is the second - it reads a level off the whole sample, which live means a
    level that moves every bar, and that is a question about the policy and
    not something this class should paper over.
    """
    if costs.spread.mode not in ("per_bar", "fixed"):
        return (
            f"spread_mode {costs.spread.mode!r} reads a level off the whole "
            f"sample, which one bar cannot supply"
        )
    try:
        IncrementalEvaluator(strategy, 1.0)
    except UnsupportedSpec as exc:
        return str(exc)
    return None


def _finite(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _engine_version() -> str:
    from core.version import ENGINE_VERSION

    return ENGINE_VERSION


def _symbol_spec_hash(spec: SymbolSpec) -> str:
    from core.runs.store import symbol_spec_cost_hash

    return symbol_spec_cost_hash(spec)


def uses_atr_exits(strategy: StrategySpec) -> bool:
    """Whether the runner has to carry indicator values into its exit levels."""
    return isinstance(strategy.exit.stop_loss, AtrLevel) or isinstance(
        strategy.exit.take_profit, AtrLevel
    )
