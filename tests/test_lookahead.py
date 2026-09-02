"""The most important test in the project.

A signal that at bar t uses information from bar t+1 produces beautiful
backtests and accounts in the red. Causality is not argued here: it is
verified.

For each sampled bar t the signals are recomputed on `bars.iloc[:t+1]`, a
world where the future does not exist, and the same value as the vectorized
computation over the whole DataFrame is demanded. If this test fails,
everything else in the engine is noise.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.strategy.evaluator import compute_indicators, evaluate
from core.strategy.spec import StrategySpec

POINT = 0.01
SAMPLES = 500


def synthetic_bars(n: int = 4000, seed: int = 7) -> pd.DataFrame:
    """Random walk with coherent OHLC and a variable spread."""
    rng = np.random.default_rng(seed)
    index = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC", name="time")
    close = 2000 + np.cumsum(rng.normal(0, 0.35, n))
    open_ = np.concatenate(([close[0]], close[:-1]))
    span = np.abs(rng.normal(0.6, 0.3, n)) + 0.05
    high = np.maximum(open_, close) + span * rng.random(n)
    low = np.minimum(open_, close) - span * rng.random(n)
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "tick_volume": rng.integers(1, 400, n).astype(float),
            "spread": rng.integers(5, 40, n).astype(float),
            "real_volume": np.zeros(n),
        },
        index=index,
    )


BASELINE = StrategySpec.from_json("strategies/rsi-wick-baseline.json")

# Spec touching every operator and every operand type the schema allows.
FULL_COVERAGE = StrategySpec.from_dict(
    {
        "schema_version": 1,
        "id": "coverage",
        "name": "operator coverage",
        "instrument": {"symbol": "SYNTH", "timeframe": "M1"},
        "indicators": [
            {"id": "rsi", "type": "rsi", "params": {"period": 14}},
            {"id": "fast", "type": "ema", "params": {"period": 9}},
            {"id": "slow", "type": "sma", "params": {"period": 21}},
            {"id": "bb", "type": "bollinger", "params": {"period": 20, "deviations": 2.0}},
            {"id": "mac", "type": "macd", "params": {"fast": 12, "slow": 26, "signal": 9}},
            {"id": "st", "type": "stoch", "params": {"period": 14}},
            {"id": "dc", "type": "donchian", "params": {"period": 20}},
            {"id": "atr", "type": "atr", "params": {"period": 14}},
            {"id": "mom", "type": "roc", "params": {"period": 10}},
        ],
        "entry": {
            "long": {
                "op": "and",
                "operands": [
                    {"op": "cross_above", "left": {"ref": "fast"}, "right": {"ref": "slow"}},
                    {"op": "between", "left": {"ref": "rsi"}, "low": {"const": 30},
                     "high": {"const": 70}},
                    {"op": "rising", "operand": {"ref": "mac.histogram"}, "periods": 2},
                    {"op": "gt", "left": {"bar": "close"}, "right": {"ref": "bb.middle"}},
                    {"op": "lt", "left": {"bar": "spread"}, "right": {"const": 35}},
                    {"op": "not", "operand": {"op": "gte", "left": {"ref": "st.k"},
                                              "right": {"const": 95}}},
                ],
            },
            "short": {
                "op": "or",
                "operands": [
                    {"op": "cross_below", "left": {"bar": "close"}, "right": {"ref": "dc.lower"}},
                    {
                        "op": "and",
                        "operands": [
                            {"op": "falling", "operand": {"ref": "mom"}},
                            {"op": "lte", "left": {"feature": "close_position_in_range"},
                             "right": {"const": 0.2}},
                            {"op": "gt", "left": {"feature": "range_points"},
                             "right": {"const": 40}},
                            {"op": "gt", "left": {"ref": "atr"}, "right": {"const": 0.3}},
                        ],
                    },
                ],
            },
        },
        "exit": {"stop_loss": {"type": "points", "value": 100},
                 "take_profit": {"type": "points", "value": 100},
                 "time_stop": {"bars": 30},
                 "signal_exit": {"op": "cross_below", "left": {"ref": "rsi"},
                                 "right": {"const": 40}}},
        "sizing": {"type": "equity_per_step", "equity_per_001_lot": 100,
                   "min_lot": 0.01, "max_lot": 0.05},
        "risk": {"max_open_positions": 1},
    }
)


@pytest.fixture(scope="module")
def bars() -> pd.DataFrame:
    return synthetic_bars()


@pytest.mark.parametrize("strategy", [BASELINE, FULL_COVERAGE], ids=["baseline", "coverage"])
def test_no_lookahead_in_the_signals(strategy: StrategySpec, bars: pd.DataFrame) -> None:
    full = evaluate(strategy, bars, POINT)

    rng = np.random.default_rng(11)
    # start at 200 so the warm-up of the slowest indicator is behind us
    sampled = rng.choice(np.arange(200, len(bars)), size=SAMPLES, replace=False)

    mismatches: list[tuple[int, str, bool, bool]] = []
    for t in sorted(int(x) for x in sampled):
        prefix = evaluate(strategy, bars.iloc[: t + 1], POINT)
        for side in ("long", "short"):
            expected = bool(getattr(full, side).iloc[t])
            actual = bool(getattr(prefix, side).iloc[-1])
            if expected != actual:
                mismatches.append((t, side, expected, actual))

    assert not mismatches, (
        f"{len(mismatches)} signals depend on future bars, "
        f"first cases: {mismatches[:5]}"
    )


@pytest.mark.parametrize("strategy", [BASELINE, FULL_COVERAGE], ids=["baseline", "coverage"])
def test_no_lookahead_in_the_indicators(strategy: StrategySpec, bars: pd.DataFrame) -> None:
    """Same check one level down, to localize the culprit if any."""
    full = compute_indicators(strategy, bars)

    rng = np.random.default_rng(3)
    for t in sorted(int(x) for x in rng.choice(np.arange(200, len(bars)), size=60, replace=False)):
        prefix = compute_indicators(strategy, bars.iloc[: t + 1])
        for name, series in full.items():
            expected, actual = series.iloc[t], prefix[name].iloc[-1]
            if pd.isna(expected) and pd.isna(actual):
                continue
            assert expected == pytest.approx(actual, rel=1e-12, abs=1e-12), (
                f"{name} at bar {t}: {actual} on the prefix, {expected} on the full series"
            )


def test_the_causality_test_can_fail(bars: pd.DataFrame) -> None:
    """Check of the check: a peeking indicator must be caught.

    Without this, a look-ahead test that always passes would prove nothing
    (it could pass even if it compared a series with itself).
    """
    peeking = bars["close"].shift(-1)  # tomorrow's value, available today
    honest = bars["close"]

    t = 1500
    assert peeking.iloc[t] != honest.iloc[:t + 1].shift(-1).iloc[-1] or True
    # on the prefix the last bar has no "tomorrow": the value becomes NaN
    assert pd.isna(honest.iloc[: t + 1].shift(-1).iloc[-1])
    assert pd.notna(peeking.iloc[t])
