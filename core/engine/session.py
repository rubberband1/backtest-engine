"""Session bar counter.

A "60 bars" time stop must mean 60 bars of open market, not 60 array rows: if
two hours of quotes are missing in between, counting rows makes the trade
last much longer than the strategy intended. The session calendar comes from
phase 1, so it is derived from the data and not written by hand.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.data.provider import Timeframe
from core.data.quality import _slot_ids, infer_session_slots


def session_ordinals(
    index: pd.DatetimeIndex, timeframe: Timeframe, threshold: float = 0.5
) -> np.ndarray:
    """Position of every bar on the grid of expected in-session bars.

    The difference between two ordinals is the number of open-market bars
    between them, including the ones missing from the data.
    """
    if len(index) == 0:
        return np.zeros(0, dtype="int64")

    active, _ = infer_session_slots(index, timeframe, threshold)
    grid = pd.date_range(index[0], index[-1], freq=timeframe.pandas_freq, tz="UTC")
    expected = grid[np.isin(_slot_ids(grid, timeframe), active)]
    if len(expected) == 0:
        return np.arange(len(index), dtype="int64")

    # searchsorted instead of get_indexer: out-of-session bars (rare but they
    # exist) are not on the grid and must map to the previous slot
    return np.searchsorted(expected.to_numpy(), index.to_numpy(), side="right").astype("int64")
