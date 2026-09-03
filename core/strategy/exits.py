"""Exit distances, in points, for reports computed before or after a run.

The engine places its levels from the actual entry price (see
`core.engine.backtester`); this module answers a different question, which
the a-priori reports need: *how far away would the stop and the target be, in
points, for a signal born on each bar?*

For a points level the two answers coincide. For a percent level this one
uses the signal bar's close instead of the next bar's open, and for an ATR
level it reads the same frozen indicator value the engine would. Both are
estimates of the same quantity, one bar apart; anything that needs the exact
placed level reads it off the trade record, where the engine writes it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.strategy.spec import Exit, Level

# Levels whose distance changes from trade to trade. A break-even win rate
# over these is a weighted average, not an arithmetic constant.
VARIABLE_TYPES: frozenset[str] = frozenset({"atr", "percent"})


def is_variable(level: Level | None) -> bool:
    return level is not None and level.type in VARIABLE_TYPES


def has_variable_exits(exit_: Exit) -> bool:
    return is_variable(exit_.stop_loss) or is_variable(exit_.take_profit)


def distance_points(
    level: Level | None,
    bars: pd.DataFrame,
    indicators: dict[str, pd.Series],
    point: float,
) -> np.ndarray | None:
    """Per-bar distance of `level` in points, aligned to the bar index.

    NaN where the distance is not defined yet (indicator warm-up), never a
    filled-in number: a bar with no distance is a bar on which the trade
    could not have been placed.
    """
    if level is None:
        return None
    if level.type == "points":
        return np.full(len(bars), float(level.value), dtype="float64")
    if level.type == "percent":
        close = bars["close"].to_numpy(dtype="float64")
        return close * (level.value / 100.0) / point
    if level.type == "atr":
        series = indicators.get(level.indicator)
        if series is None:
            raise KeyError(f"indicator not computed: {level.indicator}")
        return series.to_numpy(dtype="float64") * level.mult / point
    raise ValueError(f"unrecognized exit level type: {level.type}")


def mean_distance_points(
    level: Level | None,
    bars: pd.DataFrame,
    indicators: dict[str, pd.Series],
    point: float,
    mask: np.ndarray | None = None,
) -> tuple[float | None, float | None]:
    """(mean, standard deviation) of a level's distance in points.

    `mask` restricts the average to the bars that matter - the signal bars,
    when the question is what this strategy would actually have placed.
    """
    values = distance_points(level, bars, indicators, point)
    if values is None:
        return None, None
    if mask is not None and mask.any():
        values = values[mask]
    values = values[np.isfinite(values)]
    if not len(values):
        return None, None
    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return float(values.mean()), std
