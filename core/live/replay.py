"""Feeds historical bars to the live runner as if they were arriving now.

The runner cannot tell the difference: it receives one closed bar at a time,
recomputes its indicators over the history it has accumulated, and decides.
No orders are sent - the broker is a stub that accepts nothing and records
everything - so a replay is safe to run in a test suite and safe to run
against a spec nobody has validated.

This exists for one reason: to make "the backtest and the live system are the
same system" a claim that can fail. Replaying a period through the runner and
diffing the trades against a backtest of the same period is the only check
that catches the class of bug that killed the previous project, where the
simulator and the trader quietly diverged and every reported number was about
a system that never ran.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import tzinfo
from typing import Any

import pandas as pd

from core.data.provider import SymbolSpec
from core.engine.backtester import BacktestConfig, BacktestResult, run_backtest
from core.engine.costs import CostModel
from core.engine.session import SessionCalendar
from core.live.broker import (
    AccountGuard,
    LiveBroker,
    OpenPosition,
    OrderRequest,
    OrderResult,
)
from core.live.runner import LiveConfig, LiveRunner
from core.strategy.binding import BoundSpec

logger = logging.getLogger(__name__)


class ReplayBroker(LiveBroker):
    """A broker that answers questions and refuses to do anything.

    Not a mock in the testing sense: it is the shape a broker has when the
    market being traded is in the past. It reports a flat account, accepts the
    account check, and returns dry-run results for every order.
    """

    def __init__(self, magic: int = 0) -> None:
        super().__init__(dry_run=True, magic=magic or 1)
        self.requests: list[dict[str, Any]] = []

    def check_account(self) -> AccountGuard:
        guard = AccountGuard(
            trade_mode=0,
            trade_mode_name="replay",
            algo_trading_enabled=False,
            dry_run=True,
            allowed_to_send=False,
            reason="replay: the market being traded is in the past, nothing is sent",
        )
        self._guard = guard
        return guard

    def open_positions(self, symbol: str) -> list[OpenPosition]:
        return []

    def send(self, request: OrderRequest, spec: SymbolSpec) -> OrderResult:
        self.requests.append(request.as_dict())
        return OrderResult(
            accepted=False,
            dry_run=True,
            retcode=None,
            retcode_name="replay",
            requested_lots=request.lots,
            filled_lots=0.0,
            requested_price=None,
            filled_price=None,
            order_ticket=None,
            position_ticket=None,
            comment="replay: nothing sent",
        )


@dataclass
class ReplayResult:
    """What the runner did over the replayed period."""

    trades: pd.DataFrame
    equity: pd.Series
    blocked: dict[str, int]
    entry_attempts: int
    bars: int
    calendar: SessionCalendar


def replay(
    bound: BoundSpec,
    bars: pd.DataFrame,
    symbol_spec: SymbolSpec,
    server_tz: tzinfo,
    costs: CostModel | None = None,
    initial_equity: float = 100.0,
    session_threshold: float = 0.5,
    calendar: SessionCalendar | None = None,
    journal_path: Any = None,
) -> ReplayResult:
    """Runs `bars` through the live runner, one closed bar at a time.

    The session calendar is pinned rather than inferred bar by bar. Inferring
    it from a growing window would give the runner a different calendar on
    every bar, and the time stop would count differently from a backtest for
    a reason that has nothing to do with the strategy. Live, the same pinning
    happens once from real history.
    """
    from core.engine.backtester import trades_frame

    bound.must_match(symbol_spec.name)
    pinned = calendar or SessionCalendar.infer(
        bars.index, bound.tf, session_threshold, server_tz
    )
    config = LiveConfig(
        initial_equity=initial_equity,
        costs=costs or CostModel(),
        session_threshold=session_threshold,
        session_calendar=pinned,
        journal_path=journal_path,
        dry_run=True,
    )
    runner = LiveRunner(
        bound, symbol_spec, server_tz, config, broker=ReplayBroker()
    )
    runner.start(bars.iloc[:0])

    equity: list[float] = []
    for moment in bars.index:
        runner.on_closed_bar(bars.loc[[moment]])
        equity.append(runner._equity)

    if runner.executor.position is not None:
        runner.executor.finalize()
        equity[-1] = runner.executor.realized

    return ReplayResult(
        trades=trades_frame(runner.executor.trades),
        equity=pd.Series(equity, index=bars.index, name="equity"),
        blocked=dict(runner.executor.blocked),
        entry_attempts=runner.executor.entry_attempts,
        bars=int(len(bars)),
        calendar=pinned,
    )


# -- the comparison ------------------------------------------------------

COMPARED_COLUMNS: tuple[str, ...] = (
    "direction",
    "entry_time",
    "entry_price",
    "stop_level",
    "target_level",
    "exit_time",
    "exit_price",
    "exit_reason",
    "lots",
    "session_bars_held",
    "net_pnl",
)


@dataclass
class Divergence:
    """One place where the replayed runner and the backtester disagree."""

    row: int
    column: str
    backtest: Any
    replay: Any

    def __str__(self) -> str:
        return (
            f"trade {self.row}, {self.column}: backtest {self.backtest!r} vs "
            f"replay {self.replay!r}"
        )


@dataclass
class EquivalenceReport:
    """Whether the simulated system and the live system are the same system."""

    strategy_id: str
    symbol: str
    timeframe: str
    bars: int
    backtest_trades: int
    replay_trades: int
    divergences: list[Divergence]
    verdict: str

    @property
    def equivalent(self) -> bool:
        return not self.divergences and self.backtest_trades == self.replay_trades


def compare_replay(
    bound: BoundSpec,
    bars: pd.DataFrame,
    symbol_spec: SymbolSpec,
    server_tz: tzinfo,
    costs: CostModel | None = None,
    initial_equity: float = 100.0,
    session_threshold: float = 0.5,
) -> tuple[EquivalenceReport, BacktestResult, ReplayResult]:
    """Runs both engines over the same bars and diffs the trades.

    The two are given the same pinned session calendar. That is not a
    concession: a calendar is an input, like the spread policy, and comparing
    two runs that were handed different inputs measures the inputs.
    """
    bound.must_match(symbol_spec.name)
    pinned = SessionCalendar.infer(
        bars.index, bound.tf, session_threshold, server_tz
    )
    model = costs or CostModel()
    expected = run_backtest(
        bound.spec,
        bars,
        symbol_spec,
        server_tz,
        BacktestConfig(
            initial_equity=initial_equity,
            costs=model,
            session_threshold=session_threshold,
            session_calendar=pinned,
        ),
    )
    actual = replay(
        bound, bars, symbol_spec, server_tz, model, initial_equity,
        session_threshold, pinned,
    )

    divergences = diff_trades(expected.trades, actual.trades)
    if divergences:
        verdict = (
            f"{len(divergences)} disagreement(s) between the backtester and the "
            f"runner replaying the same bars. Every one of them is a bug: the "
            f"two are meant to be the same code driven differently, so a "
            f"difference means one of them is not doing what the other reports"
        )
    elif len(expected.trades) != len(actual.trades):
        verdict = (
            f"the backtester produced {len(expected.trades)} trades and the "
            f"runner {len(actual.trades)} over the same bars"
        )
    else:
        verdict = (
            f"{len(expected.trades)} trades, identical on entry and exit times, "
            f"prices, levels, lots and exit reasons: over these {len(bars)} bars "
            f"the simulated system and the live system are the same system"
        )

    report = EquivalenceReport(
        strategy_id=bound.spec.id,
        symbol=bound.symbol,
        timeframe=bound.timeframe,
        bars=int(len(bars)),
        backtest_trades=int(len(expected.trades)),
        replay_trades=int(len(actual.trades)),
        divergences=divergences,
        verdict=verdict,
    )
    logger.info("replay equivalence: %s", verdict)
    return report, expected, actual


def diff_trades(
    expected: pd.DataFrame, actual: pd.DataFrame, columns: tuple[str, ...] = COMPARED_COLUMNS
) -> list[Divergence]:
    """Every cell where the two trade tables differ. No tolerance on prices.

    Timestamps and reasons are compared exactly. Numbers are compared exactly
    too: both sides run the same arithmetic in the same order, so anything
    other than bit equality means the order changed somewhere, which is the
    thing being looked for.
    """
    divergences: list[Divergence] = []
    for row in range(max(len(expected), len(actual))):
        if row >= len(expected):
            divergences.append(
                Divergence(row, "<missing>", None, _describe(actual.iloc[row]))
            )
            continue
        if row >= len(actual):
            divergences.append(
                Divergence(row, "<missing>", _describe(expected.iloc[row]), None)
            )
            continue
        left, right = expected.iloc[row], actual.iloc[row]
        for column in columns:
            if column not in expected.columns or column not in actual.columns:
                continue
            a, b = left[column], right[column]
            if pd.isna(a) and pd.isna(b):
                continue
            if a != b:
                divergences.append(Divergence(row, column, a, b))
    return divergences


def _describe(trade: pd.Series) -> str:
    return (
        f"{'long' if trade['direction'] > 0 else 'short'} "
        f"{trade['entry_time']} -> {trade['exit_time']} ({trade['exit_reason']})"
    )
