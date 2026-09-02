"""Features derived from the single bar.

They are ratios over the candle geometry: they depend only on the current
bar, so they introduce no look-ahead by construction.

Zero-range bars (high == low): the ratios are 0/0, i.e. undefined. They stay
NaN and propagate as "no signal". Filling them with zero would declare
"no wick" about a bar we know nothing about.
"""
from __future__ import annotations

import pandas as pd

FEATURE_NAMES: tuple[str, ...] = (
    "lower_wick_ratio",
    "upper_wick_ratio",
    "body_ratio",
    "range_points",
    "close_position_in_range",
)


def compute_features(bars: pd.DataFrame, point: float) -> pd.DataFrame:
    """All bar features. `point` comes from the broker's SymbolSpec."""
    if point <= 0:
        raise ValueError(f"point must be positive, got {point}")

    high, low = bars["high"], bars["low"]
    open_, close = bars["open"], bars["close"]
    span = high - low
    valid = span > 0

    body_top = pd.concat([open_, close], axis=1).max(axis=1)
    body_bottom = pd.concat([open_, close], axis=1).min(axis=1)

    return pd.DataFrame(
        {
            "lower_wick_ratio": ((body_bottom - low) / span).where(valid),
            "upper_wick_ratio": ((high - body_top) / span).where(valid),
            "body_ratio": ((close - open_).abs() / span).where(valid),
            "range_points": span / point,
            "close_position_in_range": ((close - low) / span).where(valid),
        },
        index=bars.index,
    )
