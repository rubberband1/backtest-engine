"""Break-even win rate for binary-outcome strategies.

When every trade ends on either the stop or the target, the strategy is a
binary bet and the win rate needed to break even is arithmetic, not opinion:

    breakeven = avg_loss / (avg_loss + avg_win)

A note on spread, specific to this engine: SL and TP levels are anchored to
the spread-inclusive entry price, so the cash outcome of a stop is exactly
SL points and the cash outcome of a target exactly TP points (plus
commission). The spread does not appear in the per-trade amounts; it shifts
the trigger levels instead, which lowers the *achievable* win rate. The
break-even threshold is therefore (SL + c) / (SL + TP) with c the round-turn
commission in points, and the spread must be read as pressure on the realized
win rate, not on the threshold.

If the R-multiple distribution is not binary (trailing stops, signal exits,
frequent time stops, gap fills), the number is meaningless and must not be
shown. The validity checks below are explicit and reported.

Exits sized in ATR or in percent (A2) break the "two fixed amounts"
assumption on purpose: every trade has its own stop and target distance, so
wins and losses are clusters, not points. For those specs the threshold is
still computable - `avg_loss / (avg_loss + avg_win)` is a weighted average
over the realized outcomes - but it is a mean of a distribution rather than
an arithmetic constant, so the dispersion of both sides travels with it and
the dispersion check stops being a reason to refuse the number.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

from core.data.provider import SymbolSpec
from core.engine.costs import CommissionModel, money_per_point
from core.serialization import json_safe
from core.strategy.exits import has_variable_exits
from core.strategy.spec import StrategySpec

logger = logging.getLogger(__name__)

BINARY_EXIT_REASONS = frozenset({"stop_loss", "take_profit"})
# above this share of non-SL/TP exits the outcome distribution is not binary
MAX_NON_BINARY_SHARE = 0.05
# wins (and losses) must be tight clusters: relative std above this means the
# amounts are not two fixed outcomes
MAX_RELATIVE_STD = 0.05
MIN_WINS_AND_LOSSES = 2


@dataclass(frozen=True)
class BreakevenReport:
    """Break-even win rate computed from realized trades."""

    valid: bool
    reason: str | None
    observations: int
    breakeven_win_rate: float | None = None
    realized_win_rate: float | None = None
    delta: float | None = None
    avg_win: float | None = None
    avg_loss: float | None = None
    non_binary_trades: int = 0
    non_binary_reasons: dict[str, int] = field(default_factory=dict)
    # variable-distance exits (ATR, percent): the threshold is a weighted
    # average over trades that each had their own stop and target, and the
    # relative dispersion of both sides says how wide that average is
    variable_exits: bool = False
    win_relative_std: float | None = None
    loss_relative_std: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return json_safe(asdict(self))


@dataclass(frozen=True)
class BreakevenPrior:
    """A-priori break-even estimate from the spec alone, before any backtest."""

    valid: bool
    reason: str | None
    breakeven_win_rate: float | None = None
    loss_points: float | None = None
    win_points: float | None = None
    commission_points: float | None = None
    avg_spread_points: float | None = None
    caveats: list[str] = field(default_factory=list)
    # ATR/percent exits: the distances below are averages over the signal
    # bars, and their dispersion says how much the threshold moves per trade
    variable_exits: bool = False
    stop_points_std: float | None = None
    target_points_std: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return json_safe(asdict(self))


def breakeven_from_trades(
    trades: pd.DataFrame, variable_exits: bool = False
) -> BreakevenReport:
    """Break-even win rate from a realized trade list, or why it is invalid.

    `variable_exits` declares that the spec sizes its stop and target per
    trade (ATR or percent). Dispersed outcomes are then the expected shape of
    a correct result, not evidence that the bet is not binary, so the
    dispersion is reported instead of invalidating the number.
    """
    count = int(len(trades))
    if count == 0:
        return BreakevenReport(
            valid=False, reason="no trades", observations=0, variable_exits=variable_exits
        )

    reasons = trades["exit_reason"].value_counts().to_dict()
    non_binary = {
        str(reason): int(n)
        for reason, n in reasons.items()
        if reason not in BINARY_EXIT_REASONS
    }
    non_binary_count = sum(non_binary.values())
    if non_binary_count / count > MAX_NON_BINARY_SHARE:
        return BreakevenReport(
            valid=False,
            reason=(
                f"outcome distribution is not binary: {non_binary_count}/{count} "
                f"trades ({non_binary_count / count:.1%}) exited via "
                f"{', '.join(sorted(non_binary))} instead of stop/target"
            ),
            observations=count,
            non_binary_trades=non_binary_count,
            non_binary_reasons=non_binary,
            variable_exits=variable_exits,
        )

    pnl = trades["net_pnl"].astype("float64")
    wins = pnl[pnl > 0]
    losses = -pnl[pnl < 0]
    if len(wins) < MIN_WINS_AND_LOSSES or len(losses) < MIN_WINS_AND_LOSSES:
        return BreakevenReport(
            valid=False,
            reason=(
                f"not enough outcomes on both sides ({len(wins)} wins, "
                f"{len(losses)} losses)"
            ),
            observations=count,
            non_binary_trades=non_binary_count,
            non_binary_reasons=non_binary,
            variable_exits=variable_exits,
        )

    # binary-reason trades only: gap fills (tolerated above) execute at the
    # open price and would inflate the dispersion of a genuinely binary bet
    binary_mask = trades["exit_reason"].isin(BINARY_EXIT_REASONS)
    binary_pnl = trades.loc[binary_mask, "net_pnl"].astype("float64")
    binary_wins = binary_pnl[binary_pnl > 0]
    binary_losses = -binary_pnl[binary_pnl < 0]
    dispersion: dict[str, float | None] = {"win": None, "loss": None}
    for label, side in (("win", binary_wins), ("loss", binary_losses)):
        if len(side) >= 2 and side.mean() > 0:
            relative_std = float(side.std(ddof=1) / side.mean())
            dispersion[label] = relative_std
            if relative_std > MAX_RELATIVE_STD and not variable_exits:
                return BreakevenReport(
                    valid=False,
                    reason=(
                        f"{label} amounts are dispersed (relative std "
                        f"{relative_std:.1%} > {MAX_RELATIVE_STD:.0%}): the "
                        f"outcomes are not two fixed values"
                    ),
                    observations=count,
                    non_binary_trades=non_binary_count,
                    non_binary_reasons=non_binary,
                    win_relative_std=dispersion["win"],
                    loss_relative_std=dispersion["loss"],
                    variable_exits=variable_exits,
                )

    avg_win = float(wins.mean())
    avg_loss = float(losses.mean())
    breakeven = avg_loss / (avg_loss + avg_win)
    realized = float(len(wins) / count)
    return BreakevenReport(
        valid=True,
        reason=(
            "stop and target are sized per trade: this threshold is a weighted "
            "average over outcomes that each had their own distance, not a "
            "single arithmetic level"
            if variable_exits
            else None
        ),
        observations=count,
        breakeven_win_rate=breakeven,
        realized_win_rate=realized,
        delta=realized - breakeven,
        avg_win=avg_win,
        avg_loss=avg_loss,
        non_binary_trades=non_binary_count,
        non_binary_reasons=non_binary,
        win_relative_std=dispersion["win"],
        loss_relative_std=dispersion["loss"],
        variable_exits=variable_exits,
    )


def breakeven_prior(
    spec: StrategySpec,
    symbol_spec: SymbolSpec,
    commission: CommissionModel | None = None,
    avg_spread_points: float | None = None,
    stop_points: float | None = None,
    target_points: float | None = None,
    stop_points_std: float | None = None,
    target_points_std: float | None = None,
) -> BreakevenPrior:
    """Break-even estimate from the spec and cost assumptions alone.

    Needs a stop and a target. Fixed points levels are read off the spec;
    ATR or percent levels have no single distance, so the caller must measure
    the average distance on the signal bars and pass it in - the estimate is
    then declared as an average, with the dispersion attached. The spread is
    reported as context, not added to the threshold: with entry-anchored
    levels it degrades the achievable win rate instead (see module docstring).
    """
    exit_ = spec.exit
    if exit_.stop_loss is None or exit_.take_profit is None:
        return BreakevenPrior(
            valid=False,
            reason="the spec has no stop loss and take profit to compare",
            avg_spread_points=avg_spread_points,
        )

    variable = has_variable_exits(exit_)
    if variable:
        if stop_points is None or target_points is None:
            return BreakevenPrior(
                valid=False,
                reason=(
                    "exits are sized per trade (ATR or percent) and no average "
                    "distance was measured: there is no single threshold to state"
                ),
                avg_spread_points=avg_spread_points,
                variable_exits=True,
            )
        stop = float(stop_points)
        target = float(target_points)
    else:
        stop = float(stop_points if stop_points is not None else exit_.stop_loss.value)
        target = float(target_points if target_points is not None else exit_.take_profit.value)
    value_per_point = money_per_point(symbol_spec, 1.0)
    commission_points = (
        (commission or CommissionModel()).round_turn(1.0) / value_per_point
        if value_per_point > 0
        else 0.0
    )

    loss_points = stop + commission_points
    win_points = target - commission_points
    if win_points <= 0:
        return BreakevenPrior(
            valid=False,
            reason=(
                f"commission ({commission_points:.1f} points round-turn) eats "
                f"the whole target of {target:.0f} points"
            ),
            commission_points=commission_points,
            avg_spread_points=avg_spread_points,
        )

    caveats: list[str] = []
    if exit_.time_stop is not None:
        caveats.append(
            "the spec also has a time stop: if it fires often the realized "
            "distribution is not binary and this estimate does not apply"
        )
    if exit_.signal_exit is not None:
        caveats.append(
            "the spec also has a signal exit: if it fires often the realized "
            "distribution is not binary and this estimate does not apply"
        )
    if avg_spread_points:
        caveats.append(
            f"average spread {avg_spread_points:.1f} points: with "
            f"entry-anchored levels it does not raise this threshold but "
            f"pushes the realized win rate below it"
        )
    if variable:
        spread_note = (
            f" (stop ±{stop_points_std:.0f}, target ±{target_points_std:.0f} points)"
            if stop_points_std is not None and target_points_std is not None
            else ""
        )
        caveats.append(
            f"exits are sized per trade: {stop:.0f}/{target:.0f} points are the "
            f"average distances over the signal bars{spread_note}, so this "
            f"threshold is a mean and every trade has its own"
        )

    return BreakevenPrior(
        valid=True,
        reason=None,
        breakeven_win_rate=loss_points / (loss_points + win_points),
        loss_points=loss_points,
        win_points=win_points,
        commission_points=commission_points,
        avg_spread_points=avg_spread_points,
        caveats=caveats,
        variable_exits=variable,
        stop_points_std=stop_points_std,
        target_points_std=target_points_std,
    )
