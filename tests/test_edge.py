"""Gate zero on synthetic data: known edge, no edge, and no look-ahead."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.engine.costs import SpreadPolicy
from core.research.edge import edge_report, forward_points
from core.strategy.spec import StrategySpec

POINT = 0.01
SPREAD = 10.0

# The signal is marked via tick_volume: this controls exactly which bars fire
# without touching prices or spread.
MARKED = StrategySpec.from_dict(
    {
        "schema_version": 1,
        "id": "marker",
        "name": "volume-marked signal",
        "instrument": {"symbol": "SYNTH", "timeframe": "M1"},
        "indicators": [],
        "entry": {
            "long": {"op": "gt", "left": {"bar": "volume"}, "right": {"const": 1000}},
            "short": None,
        },
        "exit": {"stop_loss": {"type": "points", "value": 100},
                 "take_profit": None, "time_stop": None, "signal_exit": None},
        "sizing": {"type": "equity_per_step", "equity_per_001_lot": 100,
                   "min_lot": 0.01, "max_lot": 0.05},
        "risk": {"max_open_positions": 1},
    }
)


def frame_from_close(close: np.ndarray, marks: np.ndarray, seed: int = 0) -> pd.DataFrame:
    """Coherent bars built from a close series and the marked bars."""
    rng = np.random.default_rng(seed)
    n = len(close)
    open_ = np.concatenate(([close[0]], close[:-1]))
    span = np.abs(rng.normal(0.05, 0.02, n))
    index = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC", name="time")
    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) + span,
            "low": np.minimum(open_, close) - span,
            "close": close,
            "tick_volume": np.where(marks, 2000.0, 100.0),
            "spread": SPREAD,
            "real_volume": 0.0,
        },
        index=index,
    )


def with_known_edge(
    n: int = 6000, period: int = 60, push: int = 15, points_per_bar: float = 2.0, seed: int = 1
) -> pd.DataFrame:
    """After each marker the price rises, then comes back.

    The rise and the return cancel out, so the overall drift is zero: the
    only way to profit is to enter on the signal. It is a real edge, not a
    trend that hands returns to anyone who happens to be long.
    """
    rng = np.random.default_rng(seed)
    step = np.zeros(n)
    marks = np.zeros(n, dtype=bool)
    for start in range(period, n - 2 * push - 2, period):
        marks[start] = True
        step[start + 1 : start + 1 + push] += points_per_bar * POINT
        step[start + 1 + push : start + 1 + 2 * push] -= points_per_bar * POINT
    noise = rng.normal(0, 0.2 * POINT, n)
    close = 2000 + np.cumsum(step + noise)
    return frame_from_close(close, marks, seed)


def without_edge(n: int = 6000, period: int = 60, seed: int = 2) -> pd.DataFrame:
    """Markers on random bars of a random walk: nothing to find."""
    rng = np.random.default_rng(seed)
    close = 2000 + np.cumsum(rng.normal(0, 3.0 * POINT, n))
    marks = np.zeros(n, dtype=bool)
    marks[np.arange(period, n - 200, period)] = True
    return frame_from_close(close, marks, seed)


def test_known_edge_is_recognized() -> None:
    bars = with_known_edge()
    report = edge_report(MARKED, bars, POINT, horizons=[5, 15, 30])

    assert report.signals_long > 50
    assert report.signals_short == 0
    assert report.passed
    assert "PASSES" in report.verdict

    at15 = next(s for s in report.stats if s.horizon == 15 and s.direction == "long")
    # 15 bars x 2 points = 30 expected points, minus 10 of spread
    assert at15.mean_points == pytest.approx(30.0, abs=3.0)
    assert at15.spread_cost_points == pytest.approx(SPREAD)
    assert at15.net_points > 15.0
    assert at15.t_stat > 10
    assert at15.p_value < 1e-6
    assert at15.beats_cost


def test_no_edge_does_not_pass() -> None:
    bars = without_edge()
    report = edge_report(MARKED, bars, POINT, horizons=[5, 15, 30, 60])

    assert report.signals_long > 50
    assert not report.passed
    assert "DOES NOT PASS" in report.verdict
    assert "SL/TP" in report.verdict
    for stat in report.stats:
        if stat.direction == "long":
            assert stat.net_points < 0  # does not even cover the spread


def test_spread_is_the_threshold_to_beat() -> None:
    """The same signal passes with a low spread and fails with a high one."""
    bars = with_known_edge(points_per_bar=1.0)  # +15 points at 15 bars

    cheap = edge_report(MARKED, bars, POINT, horizons=[15],
                        spread=SpreadPolicy(mode="fixed", value=2.0))
    expensive = edge_report(MARKED, bars, POINT, horizons=[15],
                            spread=SpreadPolicy(mode="fixed", value=40.0))

    assert cheap.passed
    assert not expensive.passed
    cheap_stat = next(s for s in cheap.stats if s.direction == "long")
    costly_stat = next(s for s in expensive.stats if s.direction == "long")
    # the raw mean is identical: only the threshold changes
    assert cheap_stat.mean_points == pytest.approx(costly_stat.mean_points)
    assert cheap_stat.net_points - costly_stat.net_points == pytest.approx(38.0)


def test_drift_is_not_mistaken_for_edge() -> None:
    """Trending instrument, random signal: the merit is the market's, not the signal's."""
    n = 6000
    rng = np.random.default_rng(3)
    # +1 point per bar of pure trend
    close = 2000 + np.cumsum(np.full(n, 1.0 * POINT) + rng.normal(0, 2.0 * POINT, n))
    marks = np.zeros(n, dtype=bool)
    marks[np.arange(50, n - 200, 50)] = True
    bars = frame_from_close(close, marks)

    report = edge_report(MARKED, bars, POINT, horizons=[30])
    stat = next(s for s in report.stats if s.direction == "long")

    assert stat.mean_points > 25  # in a trend a long always "profits"
    assert stat.drift_baseline_points == pytest.approx(stat.mean_points, abs=4.0)
    assert not stat.beats_cost
    assert not report.passed


def test_few_signals_produce_no_verdict() -> None:
    bars = with_known_edge(n=1200, period=500)
    report = edge_report(MARKED, bars, POINT, horizons=[15], min_observations=30)
    stat = next(s for s in report.stats if s.direction == "long")
    assert stat.observations < 30
    assert not stat.beats_cost
    assert "below the minimum" in stat.verdict


def test_forward_points_has_no_look_ahead() -> None:
    """Entry at the open of t+1, exit at the close of t+horizon."""
    close = np.arange(100, dtype="float64") * 0.01 + 2000
    bars = frame_from_close(close, np.zeros(100, dtype=bool))
    forward = forward_points(bars, POINT, horizon=5)

    expected = (bars["close"].iloc[5] - bars["open"].iloc[1]) / POINT
    assert forward[0] == pytest.approx(expected)
    # the last `horizon` bars do not have enough future
    assert np.isnan(forward[-5:]).all()
    assert not np.isnan(forward[-6])


def test_forward_points_rejects_invalid_horizons() -> None:
    bars = frame_from_close(np.full(50, 2000.0), np.zeros(50, dtype=bool))
    with pytest.raises(ValueError, match="horizon"):
        forward_points(bars, POINT, horizon=0)


def test_empty_report() -> None:
    empty = frame_from_close(np.full(10, 2000.0), np.zeros(10, dtype=bool)).iloc[0:0]
    report = edge_report(MARKED, empty, POINT)
    assert report.bars == 0
    assert not report.passed
    assert "no bars" in report.verdict


def test_clean_json_serialization() -> None:
    """inf and NaN are not valid JSON: they come out as null, not 'NaN'."""
    import json
    import math

    report = edge_report(MARKED, without_edge(n=2000), POINT, horizons=[5])
    payload = json.loads(json.dumps(report.as_dict(), allow_nan=False))

    assert payload["symbol"] == "SYNTH"
    long_stat = next(s for s in payload["stats"] if s["direction"] == "long")
    assert math.isfinite(long_stat["t_stat"])
    # the direction with no signals has no statistics: null, not NaN
    short_stat = next(s for s in payload["stats"] if s["direction"] == "short")
    assert short_stat["observations"] == 0
    assert short_stat["t_stat"] is None
