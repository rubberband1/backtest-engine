"""Deterministic risk gates.

This module is written to be used **identically** by the backtest and by the
future live runner: it takes an explicit state and a moment in time, and
answers yes/no with a reason. It holds no state of its own, knows nothing
about the broker, and does not know whether it is running in simulation. If
the backtest and live diverge here, the backtest results are worthless.
"""
from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, tzinfo

from core.strategy.spec import Risk, Session

logger = logging.getLogger(__name__)


@dataclass
class RiskState:
    """State the caller carries between decisions."""

    open_positions: int = 0
    last_exit_time: datetime | None = None
    trades_per_day: Counter[date] = field(default_factory=Counter)

    def register_entry(self, moment: datetime, server_tz: tzinfo) -> None:
        self.open_positions += 1
        self.trades_per_day[moment.astimezone(server_tz).date()] += 1

    def register_exit(self, moment: datetime) -> None:
        self.open_positions = max(0, self.open_positions - 1)
        self.last_exit_time = moment


@dataclass(frozen=True)
class RiskDecision:
    allowed: bool
    reason: str | None = None
    code: str | None = None

    def __bool__(self) -> bool:
        return self.allowed


ALLOWED = RiskDecision(True)

# Stable identifiers for the accounting. The `reason` carries the numbers of
# the single decision and is therefore unique per bar; counting on it produces
# one bucket per spread value instead of one per gate, which is how a gate
# rejecting half the signals stays invisible.
GATE_CODES: tuple[str, ...] = (
    "max_open_positions",
    "max_spread_points",
    "cooldown",
    "max_trades_per_day",
    "session",
)


def _in_session(moment: datetime, session: Session, server_tz: tzinfo) -> bool:
    local = moment.astimezone(server_tz) if session.timezone == "server" else moment
    minutes = local.hour * 60 + local.minute
    start_h, start_m = (int(p) for p in session.start.split(":"))
    end_h, end_m = (int(p) for p in session.end.split(":"))
    start = start_h * 60 + start_m
    end = end_h * 60 + end_m
    if start <= end:
        return start <= minutes < end
    # window straddling midnight (e.g. 22:00 -> 06:00)
    return minutes >= start or minutes < end


class RiskGate:
    """Applies the blocks of the spec's `risk` section."""

    def __init__(self, risk: Risk, server_tz: tzinfo) -> None:
        self.risk = risk
        self.server_tz = server_tz

    def check_entry(
        self, moment: datetime, spread_points: float, state: RiskState
    ) -> RiskDecision:
        if state.open_positions >= self.risk.max_open_positions:
            return RiskDecision(
                False,
                f"open positions {state.open_positions} >= {self.risk.max_open_positions}",
                "max_open_positions",
            )

        if self.risk.max_spread_points is not None and spread_points > self.risk.max_spread_points:
            return RiskDecision(
                False,
                f"spread {spread_points:.0f} > max {self.risk.max_spread_points:.0f} points",
                "max_spread_points",
            )

        if self.risk.cooldown_minutes and state.last_exit_time is not None:
            elapsed = moment - state.last_exit_time
            if elapsed < timedelta(minutes=self.risk.cooldown_minutes):
                return RiskDecision(
                    False,
                    f"cooldown: {elapsed.total_seconds() / 60:.0f} min of the "
                    f"{self.risk.cooldown_minutes} required",
                    "cooldown",
                )

        if self.risk.max_trades_per_day is not None:
            today = moment.astimezone(self.server_tz).date()
            if state.trades_per_day[today] >= self.risk.max_trades_per_day:
                return RiskDecision(
                    False,
                    f"reached {self.risk.max_trades_per_day} trades in "
                    f"server day {today}",
                    "max_trades_per_day",
                )

        if self.risk.session is not None and not _in_session(
            moment, self.risk.session, self.server_tz
        ):
            return RiskDecision(
                False,
                f"outside session {self.risk.session.start}-{self.risk.session.end} "
                f"({self.risk.session.timezone})",
                "session",
            )

        return ALLOWED
