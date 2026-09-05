"""Two attempt counts, both labelled, and the relationship between them.

The panel corrects for what has been tried on *this instrument over this
period*; a campaign corrects for the *whole search*. Both are legitimate and
they give different thresholds for the same observed Sharpe, so a reader
shown one number in the editor and a different one in the campaign report has
no way to tell which is wrong - and neither is. These tests pin that both are
reported and that each says which scope it belongs to.
"""
from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass

import numpy as np
import pandas as pd
import pytest

from core.research.preview import (
    LOCAL_SCOPE,
    OVERALL_SCOPE,
    attempts_panel,
    trial_set,
)


@dataclass
class FakeRun:
    """A run record, reduced to the one thing the correction reads."""

    pnl: np.ndarray

    def trades(self) -> pd.DataFrame:
        return pd.DataFrame({"net_pnl": self.pnl})


def runs(count: int, trades: int = 60, seed: int = 0) -> list[FakeRun]:
    """`count` attempts, each with enough trades to be judgeable."""
    rng = np.random.default_rng(seed)
    return [
        FakeRun(rng.normal(loc=rng.normal(0, 0.4), scale=1.0, size=trades))
        for _ in range(count)
    ]


def test_both_scopes_are_reported_and_named() -> None:
    local = runs(6, seed=1)
    everything = local + runs(60, seed=2)

    panel = attempts_panel(
        local, "EURUSD", None, None, assumed_trades=200,
        overall_trials=trial_set(everything),
    )

    assert panel.attempts == 6
    assert panel.overall_attempts == 66
    assert panel.scope == LOCAL_SCOPE
    assert panel.overall_scope == OVERALL_SCOPE
    # both thresholds exist and are distinct
    assert panel.required_sharpe_per_trade is not None
    assert panel.overall_required_sharpe_per_trade is not None
    assert panel.overall_required_sharpe_per_trade != panel.required_sharpe_per_trade


def test_the_wider_search_asks_for_more() -> None:
    """More attempts is a higher bar. If it were not, the correction is wrong."""
    local = runs(4, seed=3)
    everything = local + runs(120, seed=4)

    panel = attempts_panel(
        local, "EURUSD", None, None, assumed_trades=200,
        overall_trials=trial_set(everything),
    )
    assert panel.overall_attempts > panel.attempts
    assert panel.overall_required_sharpe_per_trade > panel.required_sharpe_per_trade
    assert panel.overall_expected_max_sharpe > panel.expected_max_sharpe


def test_the_verdict_states_which_number_governs() -> None:
    local = runs(5, seed=5)
    everything = local + runs(80, seed=6)

    panel = attempts_panel(
        local, "EURUSD", None, None, assumed_trades=200,
        overall_trials=trial_set(everything),
    )
    verdict = panel.verdict
    assert "this instrument and period alone" in verdict
    assert "whole search" in verdict
    assert "claim of discovery has to clear" in verdict
    assert str(panel.overall_attempts) in verdict


def test_without_the_wider_search_the_panel_reports_one_scope_twice() -> None:
    """The old call shape still works, and does not invent a second number."""
    local = runs(5, seed=7)
    panel = attempts_panel(local, "EURUSD", None, None, assumed_trades=200)

    assert panel.overall_attempts == panel.attempts
    assert panel.overall_required_sharpe_per_trade == panel.required_sharpe_per_trade
    assert "whole search" not in panel.verdict


def test_a_two_trade_attempt_does_not_inflate_the_threshold() -> None:
    """The bug that once pushed the required Sharpe to an unreachable +4.77.

    A per-trade Sharpe over two trades reaches +-10 by arithmetic alone.
    Including those in the variance across trials makes the corrected
    threshold something nothing can clear, which looks like rigour and is a
    broken test.
    """
    real = runs(8, trades=60, seed=8)
    noise = [FakeRun(np.array([5.0, -0.01])) for _ in range(8)]

    clean = trial_set(real)
    polluted = trial_set(real + noise)

    # the two-trade runs are counted as attempts but not as Sharpes
    assert polluted.attempts == 16
    assert polluted.sharpes == clean.sharpes


def test_the_trial_set_is_the_expensive_half_and_is_reusable() -> None:
    """`trial_set` holds only numbers, so a caller can cache it."""
    summary = trial_set(runs(5, seed=9))
    assert summary.attempts == 5
    assert len(summary.sharpes) == 5
    assert all(isinstance(value, float) for value in summary.sharpes)
    # frozen: safe to hand around and keep between requests
    with pytest.raises(FrozenInstanceError):
        summary.attempts = 6  # type: ignore[misc]
