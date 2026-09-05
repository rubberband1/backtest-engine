"""Resolving ambiguous trades with tick data.

When a bar touches both the stop and the target, the engine assumes the stop.
That assumption is conservative and it is the right default, but it is still
an assumption: on a run where ambiguous trades are a meaningful share of the
total, the reported result is a lower bound of unknown tightness.

Ticks settle it. For each ambiguous trade this module pulls the ticks of the
exit bar and asks which level was reached first, on the correct side of the
book - the bid for a long exit, the ask for a short one, exactly as the
backtester prices them.

What this module will never do is guess. Old periods have no tick history at
the broker, the terminal may be closed, a bar may come back empty: every one
of those cases is reported as unresolved and counted separately. The delta
against the conservative assumption is then stated over the trades that were
actually resolved, with that count attached.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from typing import Any, Literal

import numpy as np
import pandas as pd

from core.data.provider import DataProvider, SymbolSpec, Timeframe
from core.metrics.ambiguity import net_pnl_at
from core.serialization import json_safe
from core.strategy.spec import StrategySpec

logger = logging.getLogger(__name__)

Resolution = Literal["stop_loss", "take_profit", "unresolved"]


class TickDataUnavailable(RuntimeError):
    """No tick source at all: nothing can be resolved, and nothing is assumed."""


@dataclass
class ResolvedTrade:
    index: int
    direction: int
    entry_time: datetime
    exit_time: datetime
    original_reason: str
    resolution: Resolution
    reason: str | None
    ticks: int
    stop_level: float
    target_level: float
    original_net_pnl: float
    resolved_net_pnl: float
    delta: float

    def as_dict(self) -> dict[str, Any]:
        return json_safe(asdict(self))


@dataclass
class TickResolveReport:
    run_id: str
    symbol: str
    available: bool
    reason: str | None
    ambiguous_trades: int
    resolved: int
    unresolved: int
    resolved_to_take_profit: int
    resolved_to_stop_loss: int
    trades: list[ResolvedTrade]
    original_net_pnl: float
    resolved_net_pnl: float
    delta_net_pnl: float
    original_final_equity: float | None
    resolved_final_equity: float | None
    original_win_rate: float | None
    resolved_win_rate: float | None
    verdict: str
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["trades"] = [trade.as_dict() for trade in self.trades]
        return json_safe(payload)


def levels_for(
    trade: pd.Series, spec: StrategySpec, symbol_spec: SymbolSpec
) -> tuple[float | None, float | None]:
    """Stop and target of a trade, exactly as the engine placed them.

    Trades record the levels they were given, which is the only thing that
    works for ATR- or percent-sized exits: their distance depends on the entry
    bar and cannot be reconstructed from the spec afterwards. Records written
    before those levels existed fall back to recomputing a fixed points
    distance, the only kind that was possible then.
    """
    if "stop_level" in trade.index and "target_level" in trade.index:
        stop = trade["stop_level"]
        target = trade["target_level"]
        return (
            float(stop) if pd.notna(stop) else None,
            float(target) if pd.notna(target) else None,
        )

    if spec.exit.stop_loss is None or spec.exit.take_profit is None:
        return None, None
    if spec.exit.stop_loss.type != "points" or spec.exit.take_profit.type != "points":
        return None, None
    direction = int(trade["direction"])
    entry = float(trade["entry_price"])
    stop_distance = float(spec.exit.stop_loss.value) * symbol_spec.point
    target_distance = float(spec.exit.take_profit.value) * symbol_spec.point
    if direction > 0:
        return entry - stop_distance, entry + target_distance
    return entry + stop_distance, entry - target_distance


def first_touch(
    ticks: pd.DataFrame, direction: int, stop_level: float, target_level: float
) -> Resolution:
    """Which level the ticks reach first, priced on the side the exit uses.

    A long exits on the bid, a short on the ask. Comparing a short's stop
    against the bid would fire it late, which is the whole reason the
    backtester keeps the two sides apart.
    """
    if ticks.empty:
        return "unresolved"
    if direction > 0:
        prices = ticks["bid"].to_numpy(dtype="float64")
        stop_hits = prices <= stop_level
        target_hits = prices >= target_level
    else:
        prices = ticks["ask"].to_numpy(dtype="float64")
        stop_hits = prices >= stop_level
        target_hits = prices <= target_level

    stop_at = int(np.argmax(stop_hits)) if stop_hits.any() else None
    target_at = int(np.argmax(target_hits)) if target_hits.any() else None
    if stop_at is None and target_at is None:
        return "unresolved"
    if target_at is None:
        return "stop_loss"
    if stop_at is None:
        return "take_profit"
    # same tick touching both: the engine's conservative assumption stands,
    # because at this resolution there is genuinely no ordering to read
    return "take_profit" if target_at < stop_at else "stop_loss"


def _recompute_net_pnl(
    trade: pd.Series, exit_level: float, symbol_spec: SymbolSpec
) -> float:
    """Net PnL of the same trade exiting at `exit_level`.

    The same accounting the uncertainty band uses for its optimistic edge, so
    a tick-resolved result always lands inside the band the run reported.
    """
    return net_pnl_at(trade, exit_level, symbol_spec)


def resolve_ambiguous(
    run_id: str,
    spec: StrategySpec,
    trades: pd.DataFrame,
    symbol_spec: SymbolSpec,
    provider: DataProvider,
    timeframe: Timeframe,
    initial_equity: float,
    on_progress: Callable[[int, int], None] | None = None,
) -> TickResolveReport:
    """Replays every ambiguous trade against the ticks of its exit bar."""
    symbol = spec.instrument.symbol
    if not len(trades) or "ambiguous" not in trades.columns:
        return _empty_report(run_id, symbol, "the run has no trade to resolve")

    ambiguous = trades[trades["ambiguous"].astype(bool)]
    original_total = float(trades["net_pnl"].astype("float64").sum())
    if not len(ambiguous):
        return TickResolveReport(
            run_id=run_id,
            symbol=symbol,
            available=True,
            reason=None,
            ambiguous_trades=0,
            resolved=0,
            unresolved=0,
            resolved_to_take_profit=0,
            resolved_to_stop_loss=0,
            trades=[],
            original_net_pnl=original_total,
            resolved_net_pnl=original_total,
            delta_net_pnl=0.0,
            original_final_equity=initial_equity + original_total,
            resolved_final_equity=initial_equity + original_total,
            original_win_rate=_win_rate(trades["net_pnl"]),
            resolved_win_rate=_win_rate(trades["net_pnl"]),
            verdict=(
                "No ambiguous trade in this run: no bar touched stop and target "
                "together, so the conservative assumption never had to be used."
            ),
        )

    stop_level, target_level = levels_for(ambiguous.iloc[0], spec, symbol_spec)
    if stop_level is None:
        return _empty_report(
            run_id,
            symbol,
            "these trades carry no stop and target pair: there is no ordering of "
            "two levels for the ticks to resolve",
        )

    bar_length = timedelta(minutes=timeframe.minutes)
    resolved: list[ResolvedTrade] = []
    warnings: list[str] = []
    adjusted = trades["net_pnl"].astype("float64").copy()

    for position, (row_index, trade) in enumerate(ambiguous.iterrows()):
        if on_progress is not None:
            on_progress(position, len(ambiguous))
        stop_level, target_level = levels_for(trade, spec, symbol_spec)
        assert stop_level is not None and target_level is not None
        exit_time = pd.Timestamp(trade["exit_time"]).to_pydatetime()
        try:
            ticks = provider.get_ticks(symbol, exit_time, exit_time + bar_length)
        except Exception as exc:  # a bad bar must not lose the other resolutions
            logger.warning("ticks unavailable for %s at %s: %s", symbol, exit_time, exc)
            ticks = pd.DataFrame()

        resolution = first_touch(ticks, int(trade["direction"]), stop_level, target_level)
        original = float(trade["net_pnl"])
        if resolution == "take_profit":
            new_pnl = _recompute_net_pnl(trade, target_level, symbol_spec)
            adjusted.loc[row_index] = new_pnl
        elif resolution == "stop_loss":
            new_pnl = _recompute_net_pnl(trade, stop_level, symbol_spec)
            adjusted.loc[row_index] = new_pnl
        else:
            new_pnl = original

        resolved.append(
            ResolvedTrade(
                index=int(row_index),
                direction=int(trade["direction"]),
                entry_time=pd.Timestamp(trade["entry_time"]).to_pydatetime(),
                exit_time=exit_time,
                original_reason=str(trade["exit_reason"]),
                resolution=resolution,
                reason=(
                    "no tick covering this bar at the broker"
                    if resolution == "unresolved"
                    else None
                ),
                ticks=int(len(ticks)),
                stop_level=stop_level,
                target_level=target_level,
                original_net_pnl=original,
                resolved_net_pnl=new_pnl,
                delta=new_pnl - original,
            )
        )
        if position == 0 and len(ticks) == 0:
            warnings.append(
                "the first ambiguous bar came back with no tick: if the whole run "
                "predates the broker's tick history, nothing here can be resolved"
            )

    unresolved = sum(1 for trade in resolved if trade.resolution == "unresolved")
    to_target = sum(1 for trade in resolved if trade.resolution == "take_profit")
    to_stop = sum(1 for trade in resolved if trade.resolution == "stop_loss")
    resolved_total = float(adjusted.sum())

    if unresolved:
        warnings.append(
            f"{unresolved} of {len(resolved)} ambiguous trades could not be "
            f"resolved and keep the conservative stop assumption: the delta below "
            f"covers only the {len(resolved) - unresolved} that were"
        )

    return TickResolveReport(
        run_id=run_id,
        symbol=symbol,
        available=True,
        reason=None,
        ambiguous_trades=len(resolved),
        resolved=len(resolved) - unresolved,
        unresolved=unresolved,
        resolved_to_take_profit=to_target,
        resolved_to_stop_loss=to_stop,
        trades=resolved,
        original_net_pnl=original_total,
        resolved_net_pnl=resolved_total,
        delta_net_pnl=resolved_total - original_total,
        original_final_equity=initial_equity + original_total,
        resolved_final_equity=initial_equity + resolved_total,
        original_win_rate=_win_rate(trades["net_pnl"]),
        resolved_win_rate=_win_rate(adjusted),
        verdict=_verdict(len(resolved), unresolved, to_target, resolved_total - original_total),
        warnings=warnings,
    )


def _win_rate(pnl: pd.Series) -> float | None:
    values = pnl.astype("float64")
    return float((values > 0).sum() / len(values)) if len(values) else None


def _empty_report(run_id: str, symbol: str, reason: str) -> TickResolveReport:
    return TickResolveReport(
        run_id=run_id,
        symbol=symbol,
        available=False,
        reason=reason,
        ambiguous_trades=0,
        resolved=0,
        unresolved=0,
        resolved_to_take_profit=0,
        resolved_to_stop_loss=0,
        trades=[],
        original_net_pnl=0.0,
        resolved_net_pnl=0.0,
        delta_net_pnl=0.0,
        original_final_equity=None,
        resolved_final_equity=None,
        original_win_rate=None,
        resolved_win_rate=None,
        verdict=reason,
    )


def _verdict(total: int, unresolved: int, to_target: int, delta: float) -> str:
    if unresolved == total:
        return (
            f"None of the {total} ambiguous trades could be resolved: the broker "
            f"returned no tick for those bars. The conservative stop assumption "
            f"stands, unverified - it is not confirmed by this, only untested."
        )
    resolved = total - unresolved
    return (
        f"{resolved} of {total} ambiguous trades resolved from ticks: "
        f"{to_target} actually reached the target first. Net PnL moves by "
        f"{delta:+.2f} in account currency against the conservative assumption"
        + (
            f", with {unresolved} still unresolved and left as stops."
            if unresolved
            else "."
        )
    )
