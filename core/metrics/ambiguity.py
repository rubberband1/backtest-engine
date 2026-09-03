"""The uncertainty band of a bar-resolution run.

When a bar touches both the stop and the target, the engine assumes the stop.
That is the right default and it is still an assumption: on the baseline
strategy, eleven ambiguous trades out of 134 were worth about twelve points
of final equity - the difference between a run that loses 10% and one that
gains 1.5%. An assumption that decides the sign of the result is not a
footnote, so this module makes it a first-class number on every run.

The band is the distance between the two extreme readings of the same trade
list:

- **conservative** - every ambiguous trade exits on its stop. What the engine
  computed, and the lower bound.
- **optimistic** - every ambiguous trade exits on its target. The upper
  bound, and not a result anyone should quote: it is there to say how much
  room the assumption occupies.

Nothing in between is a guess: `core.validation.tick_resolve` settles the
individual trades with tick data, and the resolved value lands inside this
band. A band that is wide compared to the result itself means the run does
not have an answer at bar resolution, whatever its equity curve says.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from core.data.provider import SymbolSpec
from core.engine.costs import money_per_point
from core.serialization import json_safe

# above this share of ambiguous trades the result is decided by the
# assumption rather than by the data
DEFAULT_WARNING_SHARE = 0.05


@dataclass(frozen=True)
class UncertaintyBand:
    """Conservative and optimistic readings of the same run."""

    trades: int
    ambiguous_trades: int
    ambiguous_share: float
    resolvable: bool
    reason: str | None
    conservative_net_pnl: float
    optimistic_net_pnl: float
    band_money: float
    conservative_final_equity: float | None = None
    optimistic_final_equity: float | None = None
    band_equity_pct: float | None = None
    conservative_win_rate: float | None = None
    optimistic_win_rate: float | None = None
    exceeds_threshold: bool = False
    threshold: float = DEFAULT_WARNING_SHARE
    verdict: str = ""
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return json_safe(asdict(self))


def net_pnl_at(trade: pd.Series, exit_level: float, symbol_spec: SymbolSpec) -> float:
    """Net PnL of the same trade exiting at `exit_level`.

    Rebuilds the engine's accounting from the stored fields: the spread is
    already paid on the leg the engine charged it on, the commission is a
    round turn, and the swap does not move because the alternative exit is
    inside the same bar.
    """
    direction = int(trade["direction"])
    spread_price = float(trade["spread_points"]) * symbol_spec.point
    entry_raw = (
        float(trade["entry_price"]) - spread_price
        if direction > 0
        else float(trade["entry_price"])
    )
    exit_raw = exit_level if direction > 0 else exit_level - spread_price
    value_per_point = money_per_point(symbol_spec, float(trade["lots"]))
    gross = direction * (exit_raw - entry_raw) / symbol_spec.point * value_per_point
    return float(
        gross
        - float(trade["spread_cost"])
        - float(trade["commission"])
        + float(trade["swap"])
    )


def uncertainty_band(
    trades: pd.DataFrame,
    symbol_spec: SymbolSpec,
    initial_equity: float,
    threshold: float = DEFAULT_WARNING_SHARE,
) -> UncertaintyBand:
    """The band every run reports, ambiguous trades or not."""
    count = int(len(trades))
    if count == 0:
        return UncertaintyBand(
            trades=0,
            ambiguous_trades=0,
            ambiguous_share=0.0,
            resolvable=True,
            reason="no trades",
            conservative_net_pnl=0.0,
            optimistic_net_pnl=0.0,
            band_money=0.0,
            conservative_final_equity=initial_equity,
            optimistic_final_equity=initial_equity,
            band_equity_pct=0.0,
            threshold=threshold,
            verdict="No trade: nothing to be uncertain about.",
        )

    pnl = trades["net_pnl"].astype("float64")
    conservative_total = float(pnl.sum())
    ambiguous_mask = trades["ambiguous"].astype(bool)
    ambiguous_count = int(ambiguous_mask.sum())
    share = ambiguous_count / count

    optimistic = pnl.copy()
    warnings: list[str] = []
    missing_levels = 0
    if ambiguous_count and "target_level" in trades.columns:
        for row_index, trade in trades[ambiguous_mask].iterrows():
            target = trade["target_level"]
            if pd.isna(target):
                missing_levels += 1
                continue
            optimistic.loc[row_index] = net_pnl_at(trade, float(target), symbol_spec)
    elif ambiguous_count:
        missing_levels = ambiguous_count

    if missing_levels:
        warnings.append(
            f"{missing_levels} ambiguous trades carry no target level (run written "
            f"before the levels were recorded): they stay at their conservative "
            f"value and the band below is narrower than the real one"
        )

    optimistic_total = float(optimistic.sum())
    band = optimistic_total - conservative_total
    exceeds = share > threshold

    if ambiguous_count == 0:
        verdict = (
            "No ambiguous trade: no bar touched stop and target together, so "
            "the result does not depend on the stop-first assumption."
        )
    elif exceeds:
        verdict = (
            f"{ambiguous_count} of {count} trades ({share:.1%}) are ambiguous, above "
            f"the {threshold:.0%} threshold. Their outcome is worth {band:+.2f} in "
            f"account currency ({band / initial_equity:+.1%} of initial equity): at "
            f"bar resolution this run is not conclusive, and the assumption - not "
            f"the data - is choosing the answer. Resolve them against ticks."
        )
    else:
        verdict = (
            f"{ambiguous_count} of {count} trades ({share:.1%}) are ambiguous, worth "
            f"{band:+.2f} in account currency between the two extremes. Below the "
            f"{threshold:.0%} threshold: the stop-first assumption does not drive "
            f"the result."
        )

    return UncertaintyBand(
        trades=count,
        ambiguous_trades=ambiguous_count,
        ambiguous_share=share,
        resolvable=missing_levels == 0,
        reason=None,
        conservative_net_pnl=conservative_total,
        optimistic_net_pnl=optimistic_total,
        band_money=band,
        conservative_final_equity=initial_equity + conservative_total,
        optimistic_final_equity=initial_equity + optimistic_total,
        band_equity_pct=band / initial_equity if initial_equity else None,
        conservative_win_rate=_win_rate(pnl),
        optimistic_win_rate=_win_rate(optimistic),
        exceeds_threshold=exceeds,
        threshold=threshold,
        verdict=verdict,
        warnings=warnings,
    )


def _win_rate(pnl: pd.Series) -> float | None:
    values = pnl.astype("float64").to_numpy()
    return float(np.mean(values > 0)) if len(values) else None
