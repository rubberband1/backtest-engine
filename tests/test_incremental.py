"""The incremental state must equal the recomputation exactly. Blocking.

"Exactly" is the whole point and it is meant literally: every value, on every
bar, identical to the last bit of the mantissa. Not `approx`, not `rtol`, not
"the difference is below what any trade could notice". The live runner and
the backtester share one execution state machine so that
`tests/test_replay_equivalence.py` can demand identical trades with zero
tolerance; an indicator that is right to twelve digits would put a floor
under that test and turn a proof into a habit.

Five thousand bars, all nine registered indicators, several parameter sets
each, on data with the awkward shapes real data has: flat stretches, gaps in
price, repeated values, and a warm-up.

If an indicator cannot meet this, it belongs in NOT_EXACT below with the
reason, and `core.strategy.incremental` recomputes it instead. That list is
currently empty, and this test is what would fill it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.indicators.incremental import bar_view, state_for, supported

BARS = 5000

# Indicators whose incremental state is not bit-exact. Each entry needs the
# reason; an empty dict is the claim that all nine are exact.
NOT_EXACT: dict[str, str] = {}


def frame(bars: int = BARS, seed: int = 3) -> pd.DataFrame:
    """Prices with the shapes that break naive incremental arithmetic.

    A pure random walk is the easy case. What catches a wrong compensation
    term is a long flat stretch (which trips the consecutive-same-value
    branch), a jump (which makes the running sum lose precision), and values
    that repeat exactly.
    """
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0, 0.35, bars)
    # a flat stretch, a jump, and a run of exactly repeated closes, placed as
    # fractions of the sample so a shorter frame still contains all three
    flat = slice(int(bars * 0.16), int(bars * 0.28))
    jump = int(bars * 0.40)
    repeat = slice(int(bars * 0.60), int(bars * 0.64))
    steps[flat] = 0.0
    steps[jump] = 45.0
    steps[repeat] = 0.0
    close = 2000.0 + np.cumsum(steps)

    index = pd.date_range("2024-01-01", periods=bars, freq="1min", tz="UTC", name="time")
    high = close + np.abs(rng.normal(0.0, 0.25, bars))
    low = close - np.abs(rng.normal(0.0, 0.25, bars))
    # flat bars: high == low == close, so the stochastic's zero-range branch
    # and the true range's degenerate case are both exercised
    high[repeat] = close[repeat]
    low[repeat] = close[repeat]
    open_ = np.concatenate(([close[0]], close[:-1]))
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "tick_volume": rng.integers(1, 500, bars).astype("float64"),
            "spread": rng.integers(1, 20, bars).astype("float64"),
            "real_volume": 0.0,
        },
        index=index,
    )


def vectorized(indicator_type: str, bars: pd.DataFrame, params: dict) -> pd.DataFrame:
    """The registry's own computation, as a frame of named outputs."""
    from core.indicators import registry

    definition = registry.get(indicator_type)
    values = definition.compute(bars, params)
    if isinstance(values, pd.Series):
        return values.to_frame(name="value")
    return values


def incrementally(indicator_type: str, bars: pd.DataFrame, params: dict) -> pd.DataFrame:
    """The same computation, fed one bar at a time and never looking back."""
    state = state_for(indicator_type, params)
    assert state is not None, f"no incremental state for {indicator_type}"

    rows: list[dict[str, float]] = []
    for row in bars.to_dict("records"):
        out = state.update(bar_view(row))
        rows.append({"value": out} if isinstance(out, float) else dict(out))
    return pd.DataFrame(rows, index=bars.index)


def assert_bit_identical(left: pd.DataFrame, right: pd.DataFrame, label: str) -> None:
    assert list(left.columns) == list(right.columns), (
        f"{label}: outputs differ - {list(left.columns)} vs {list(right.columns)}"
    )
    for column in left.columns:
        a = left[column].to_numpy(dtype="float64")
        b = right[column].to_numpy(dtype="float64")
        if np.array_equal(a, b, equal_nan=True):
            continue
        mismatched = ~(
            (a == b) | (np.isnan(a) & np.isnan(b))
        )
        first = int(np.flatnonzero(mismatched)[0])
        worst = float(np.nanmax(np.abs(a[mismatched] - b[mismatched])))
        pytest.fail(
            f"{label}.{column}: {int(mismatched.sum())} of {len(a)} values differ. "
            f"First at bar {first}: recomputed {a[first]!r} vs incremental "
            f"{b[first]!r} (largest absolute difference {worst!r}). "
            f"Exactness is the contract, not a tolerance: either the state "
            f"reproduces pandas' arithmetic or the indicator goes in NOT_EXACT "
            f"and is recomputed."
        )


CASES: list[tuple[str, dict]] = [
    ("sma", {"period": 20}),
    ("sma", {"period": 200, "source": "hlc3"}),
    ("ema", {"period": 12}),
    ("ema", {"period": 200, "source": "hl2"}),
    ("rsi", {"period": 9}),
    ("rsi", {"period": 14, "source": "ohlc4"}),
    ("roc", {"period": 12}),
    ("roc", {"period": 1}),
    ("bollinger", {"period": 20, "deviations": 2.0}),
    ("bollinger", {"period": 50, "deviations": 1.5, "source": "hl2"}),
    ("macd", {"fast": 12, "slow": 26, "signal": 9}),
    ("macd", {"fast": 5, "slow": 35, "signal": 5, "source": "hlc3"}),
    ("atr", {"period": 14}),
    ("atr", {"period": 50}),
    ("stoch", {"period": 14, "smooth_k": 3, "smooth_d": 3}),
    ("stoch", {"period": 21, "smooth_k": 5, "smooth_d": 5}),
    ("donchian", {"period": 20}),
    ("donchian", {"period": 55}),
]


@pytest.mark.parametrize(
    "indicator_type,params", CASES, ids=[f"{n}-{p}" for n, p in CASES]
)
def test_incremental_matches_the_recomputation_bit_for_bit(
    indicator_type: str, params: dict
) -> None:
    if indicator_type in NOT_EXACT:
        pytest.skip(f"{indicator_type} is declared inexact: {NOT_EXACT[indicator_type]}")

    bars = frame()
    assert len(bars) >= 5000

    expected = vectorized(indicator_type, bars, params)
    actual = incrementally(indicator_type, bars, params)
    assert_bit_identical(expected, actual, f"{indicator_type}{params}")


def test_every_registered_indicator_is_covered() -> None:
    """A tenth indicator must not slip in without an exactness verdict."""
    from core.indicators import registry

    registered = set(registry.available())
    assert registered == set(supported()) | set(NOT_EXACT), (
        "every registered indicator needs either an exact incremental state or "
        "an entry in NOT_EXACT saying why it has none"
    )
    assert registered <= {name for name, _ in CASES} | set(NOT_EXACT)


def test_the_warm_up_is_reproduced_and_not_filled_in() -> None:
    """A state that returns a number where the series is NaN fabricates signals."""
    bars = frame(300)
    for indicator_type, params in CASES:
        expected = vectorized(indicator_type, bars, params)
        actual = incrementally(indicator_type, bars, params)
        for column in expected.columns:
            assert expected[column].isna().equals(actual[column].isna()), (
                f"{indicator_type}{params}.{column}: the warm-up mask differs"
            )


def test_a_state_never_reaches_forward() -> None:
    """Feeding the same prefix twice must give the same answer both times.

    The property that makes an incremental state safe is that bar t's value
    depends on bars <= t and on nothing else. Running a prefix and running
    the whole series must agree on the prefix.
    """
    bars = frame(1200)
    for indicator_type, params in CASES:
        whole = incrementally(indicator_type, bars, params)
        prefix = incrementally(indicator_type, bars.iloc[:600], params)
        assert_bit_identical(
            whole.iloc[:600], prefix, f"prefix {indicator_type}{params}"
        )


def test_flat_and_repeated_values_are_handled_the_way_pandas_handles_them() -> None:
    """The consecutive-same-value branch exists in pandas; it must exist here.

    A window of identical values makes pandas return the value itself rather
    than the accumulated sum divided by the count, precisely to avoid the
    floating point artefact this test would otherwise show.
    """
    index = pd.date_range("2024-01-01", periods=400, freq="1min", tz="UTC", name="time")
    price = pd.Series(1234.5678, index=index)
    bars = pd.DataFrame(
        {
            "open": price, "high": price, "low": price, "close": price,
            "tick_volume": 1.0, "spread": 1.0, "real_volume": 0.0,
        },
        index=index,
    )
    for indicator_type, params in CASES:
        assert_bit_identical(
            vectorized(indicator_type, bars, params),
            incrementally(indicator_type, bars, params),
            f"flat {indicator_type}{params}",
        )


def test_an_unknown_indicator_has_no_state_rather_than_a_wrong_one() -> None:
    assert state_for("not-an-indicator", {}) is None


# -- the runner's two paths -----------------------------------------------


def test_the_runner_agrees_with_itself_on_both_paths() -> None:
    """Advancing the states and recomputing the frame must trade identically.

    This is the test that makes the optimization safe to keep. The live
    runner takes the incremental path whenever the spec and the spread policy
    allow it and falls back to recomputing otherwise, so the two have to be
    the same system - the same claim `tests/test_replay_equivalence.py` makes
    about the runner and the backtester, one level down.
    """
    from zoneinfo import ZoneInfo

    import core.live.runner as live_runner
    from core.engine.costs import CommissionModel, CostModel, SpreadPolicy, SwapModel
    from core.live.replay import replay
    from core.strategy.spec import StrategySpec
    from tests.conftest_engine import symbol_spec

    bars = frame(1500)
    costs = CostModel(
        spread=SpreadPolicy(mode="per_bar"),
        commission=CommissionModel(2.0),
        swap=SwapModel(mode="none"),
    )
    specs = [
        "rsi-wick-baseline",
        "rsi-mean-reversion",
        "macd-signal",
        "bollinger-mean-reversion",
        "stochastic-reversal",
        "donchian-breakout",
    ]
    for name in specs:
        spec = StrategySpec.from_json(f"strategies/{name}.json")
        spec = spec.model_copy(
            update={
                "instrument": spec.instrument.model_copy(
                    update={"symbol": "SYNTH", "timeframe": "M1"}
                )
            }
        )
        arms = {}
        original = live_runner._cannot_advance
        try:
            for label, forced in (("incremental", None), ("recomputing", "forced")):
                live_runner._cannot_advance = (
                    original if forced is None else (lambda s, c, f=forced: f)
                )
                arms[label] = replay(
                    spec, bars, symbol_spec(name="SYNTH"),
                    ZoneInfo("Europe/Athens"), costs,
                ).trades
        finally:
            live_runner._cannot_advance = original

        left, right = arms["incremental"], arms["recomputing"]
        assert len(left) == len(right), (
            f"{name}: {len(left)} trades advancing the states, {len(right)} "
            f"recomputing"
        )
        pd.testing.assert_frame_equal(left, right, check_exact=True, obj=name)


def test_a_spec_outside_the_states_falls_back_instead_of_guessing() -> None:
    """An unsupported piece must route to recomputation, never to an estimate."""
    import core.live.runner as live_runner
    from core.engine.costs import CostModel, SpreadPolicy
    from core.strategy.spec import StrategySpec

    spec = StrategySpec.from_json("strategies/rsi-wick-baseline.json")
    assert live_runner._cannot_advance(spec, CostModel()) is None

    # a quantile spread reads a level off the whole sample, which one bar
    # cannot supply: the runner says so rather than inventing a level
    reason = live_runner._cannot_advance(
        spec, CostModel(spread=SpreadPolicy(mode="quantile", value=0.9))
    )
    assert reason is not None and "quantile" in reason
