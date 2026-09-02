"""Fixtures shared by the engine tests. Imported explicitly, not auto-used."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from core.data.provider import SymbolSpec
from core.strategy.spec import StrategySpec


def symbol_spec(
    name: str = "TEST",
    point: float = 0.01,
    contract_size: float = 100.0,
    tick_value: float = 1.0,
    tick_size: float = 0.01,
    swap_long: float = 0.0,
    swap_short: float = 0.0,
) -> SymbolSpec:
    return SymbolSpec(
        name=name,
        point=point,
        digits=2,
        contract_size=contract_size,
        tick_value=tick_value,
        tick_size=tick_size,
        volume_min=0.01,
        volume_max=100.0,
        volume_step=0.01,
        swap_long=swap_long,
        swap_short=swap_short,
        currency_profit="USD",
        trade_mode="full",
    )


def bars_from(rows: list[dict[str, float]], start: str = "2024-01-01 00:00") -> pd.DataFrame:
    """Builds bars from explicit OHLC: the tests must stay readable."""
    index = pd.date_range(start, periods=len(rows), freq="1min", tz="UTC", name="time")
    frame = pd.DataFrame(rows, index=index)
    frame["tick_volume"] = frame.get("tick_volume", pd.Series(100.0, index=index))
    frame["real_volume"] = 0.0
    return frame[["open", "high", "low", "close", "tick_volume", "spread", "real_volume"]]


def spec_from(
    entry_long: dict[str, Any] | None = None,
    entry_short: dict[str, Any] | None = None,
    exit_block: dict[str, Any] | None = None,
    risk: dict[str, Any] | None = None,
    sizing: dict[str, Any] | None = None,
    indicators: list[dict[str, Any]] | None = None,
) -> StrategySpec:
    """Minimal spec without indicators: no warm-up to wait for in tests."""
    return StrategySpec.from_dict(
        {
            "schema_version": 1,
            "id": "test",
            "name": "test",
            "instrument": {"symbol": "TEST", "timeframe": "M1"},
            "indicators": indicators or [],
            "entry": {
                "long": entry_long
                or {"op": "gt", "left": {"bar": "close"}, "right": {"bar": "open"}},
                "short": entry_short,
            },
            "exit": exit_block
            or {
                "stop_loss": {"type": "points", "value": 150},
                "take_profit": {"type": "points", "value": 80},
                "time_stop": None,
                "signal_exit": None,
            },
            "sizing": sizing
            or {
                "type": "equity_per_step",
                "equity_per_001_lot": 100,
                "min_lot": 0.01,
                "max_lot": 1.0,
            },
            "risk": risk or {"max_open_positions": 1},
        }
    )


def flat_bars(n: int, price: float = 2000.0, spread: float = 10.0) -> list[dict[str, float]]:
    """Flat bars: they generate no signals, they are filler."""
    return [
        {"open": price, "high": price, "low": price, "close": price, "spread": spread}
        for _ in range(n)
    ]


def random_walk(n: int, seed: int = 5, spread: float = 10.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC", name="time")
    close = 2000 + np.cumsum(rng.normal(0, 0.3, n))
    open_ = np.concatenate(([close[0]], close[:-1]))
    high = np.maximum(open_, close) + np.abs(rng.normal(0, 0.3, n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, 0.3, n))
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "tick_volume": 100.0,
            "spread": spread,
            "real_volume": 0.0,
        },
        index=index,
    )
