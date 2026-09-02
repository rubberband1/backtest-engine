"""Position sizing.

The parameters come from the JSON spec; only the broker's constraints
(minimum, maximum and step volume) are applied here, which are instrument
data.
"""
from __future__ import annotations

import logging
import math

from core.data.provider import SymbolSpec
from core.strategy.spec import Sizing

logger = logging.getLogger(__name__)

# The elementary step of `equity_per_step`: one hundredth of a lot.
LOT_STEP = 0.01


def _round_down_to_step(lots: float, step: float) -> float:
    if step <= 0:
        return lots
    # the outer round strips the binary error before the floor, otherwise
    # 0.03/0.01 = 2.9999... and the lot drops a step for no reason
    return math.floor(round(lots / step, 9)) * step


def lots_for(sizing: Sizing, equity: float, spec: SymbolSpec) -> float:
    """Lots to open with the current equity. 0.0 = position cannot be opened."""
    if equity <= 0:
        return 0.0

    # min_lot is a floor, not an entry threshold: below one equity step the
    # minimum lot is traded anyway, which is also the smallest order the
    # broker accepts. The per-trade percentage risk rises as the account
    # shrinks: a property of this sizing, not a bug.
    steps = math.floor(equity / sizing.equity_per_001_lot)
    lots = steps * LOT_STEP
    lots = min(max(lots, sizing.min_lot), sizing.max_lot)
    # the broker's cap can be respected by going down; its minimum cannot: if
    # it is above the strategy's max_lot the position would be bigger than
    # intended, so nothing is opened at all
    lots = min(lots, spec.volume_max)
    lots = _round_down_to_step(lots, spec.volume_step)
    lots = round(lots, 8)

    if lots < spec.volume_min:
        logger.debug(
            "equity %.2f: computed lot %.4f below the broker minimum %.4f",
            equity,
            lots,
            spec.volume_min,
        )
        return 0.0
    return lots
