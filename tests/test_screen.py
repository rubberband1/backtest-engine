"""The screening funnel, and the trial count it owes to the whole campaign."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from core.research.screen import (
    CellOutcome,
    _panel,
    bind_cell,
    build_cells,
)
from core.strategy.spec import StrategySpec
from core.validation.multiple_testing import (
    expected_max_sharpe,
    required_sharpe_per_trade,
)

LIBRARY = Path("strategies")


def _cell(
    strategy: str = "s",
    symbol: str = "X",
    timeframe: str = "H1",
    sharpe: float | None = None,
    trades: int | None = None,
    p_value: float | None = None,
    net: float = 1.0,
    stage: str = "backtest",
) -> CellOutcome:
    return CellOutcome(
        strategy_id=strategy,
        symbol=symbol,
        timeframe=timeframe,
        stage_reached=stage,
        trades=trades,
        net_pnl=net,
        sharpe_per_trade=sharpe,
        p_value=p_value,
        run_id=f"{strategy}:{symbol}:{timeframe}",
    )


# -- the cell grid -------------------------------------------------------


def test_every_combination_is_one_attempt() -> None:
    cells = build_cells(
        [LIBRARY / "ma-crossover.json", LIBRARY / "macd-signal.json"],
        ["XAUUSD.r", "EURUSD.r", "XTIUSD"],
        ["M5", "M15", "H1"],
    )
    assert len(cells) == 2 * 3 * 3
    assert len({(c.strategy_id, c.symbol, c.timeframe) for c in cells}) == 18


def test_a_cell_binds_the_instrument_into_the_spec() -> None:
    """The engine reads the timeframe from the spec: leaving it stale is a bug."""
    spec = StrategySpec.from_json(LIBRARY / "ma-crossover.json")
    bound = bind_cell(spec, "XTIUSD", "M5")
    assert bound.symbol == "XTIUSD"
    assert bound.timeframe == "M5"
    assert bound.tf.minutes == 5
    assert bound.spec.instrument.symbol == "XTIUSD"
    assert bound.spec.instrument.timeframe == "M5"
    # everything else is untouched, including the id used for grouping
    assert bound.spec.id == spec.id
    assert bound.spec.exit.stop_loss == spec.exit.stop_loss


# -- the trial panel -----------------------------------------------------


def test_the_panel_counts_every_attempt_not_only_the_backtests() -> None:
    """The whole point: cells stopped at gate zero were still attempts."""
    cells = [
        _cell(sharpe=0.05, trades=100, p_value=0.4),
        _cell(sharpe=0.20, trades=120, p_value=0.02),
        *[_cell(stage="gate") for _ in range(48)],
    ]
    panel = _panel(cells, alpha=0.05, confidence=0.95)

    assert panel.attempts == 50
    assert panel.cells_backtested == 2
    assert panel.sharpes_observed == 2
    # the free Sharpe is computed at N=50, not at N=2
    variance = float(np.var([0.05, 0.20], ddof=1))
    assert panel.expected_max_sharpe == pytest.approx(expected_max_sharpe(50, variance))
    assert panel.expected_max_sharpe > expected_max_sharpe(2, variance)


def test_more_attempts_raise_the_bar() -> None:
    """Same observation, more tries: the Sharpe it must clear goes up."""
    small = _panel(
        [_cell(sharpe=0.05, trades=100), _cell(sharpe=0.20, trades=120)],
        alpha=0.05, confidence=0.95,
    )
    large = _panel(
        [_cell(sharpe=0.05, trades=100), _cell(sharpe=0.20, trades=120)]
        + [_cell(stage="gate") for _ in range(298)],
        alpha=0.05, confidence=0.95,
    )
    assert large.attempts == 300 and small.attempts == 2
    assert large.required_sharpe_per_trade > small.required_sharpe_per_trade
    assert large.expected_max_sharpe > small.expected_max_sharpe


def test_a_result_below_the_required_sharpe_is_not_credible() -> None:
    cells = [
        _cell(strategy="winner", sharpe=0.12, trades=150, p_value=0.03),
        _cell(strategy="loser", sharpe=-0.10, trades=140, p_value=0.30),
        *[_cell(stage="gate") for _ in range(98)],
    ]
    panel = _panel(cells, alpha=0.05, confidence=0.95)
    assert panel.best_sharpe_per_trade == pytest.approx(0.12)
    assert panel.best_clears_required is False
    assert "below that level" in panel.verdict
    assert "100 attempts" in panel.verdict


def test_the_panel_declares_its_assumptions() -> None:
    panel = _panel(
        [_cell(sharpe=0.1, trades=50), _cell(sharpe=0.2, trades=50),
         _cell(stage="gate")],
        alpha=0.05, confidence=0.95,
    )
    joined = " ".join(panel.assumptions)
    assert "stopped at gate zero" in joined
    assert "not independent" in joined


def test_bonferroni_is_computed_over_the_family_of_backtested_cells() -> None:
    cells = [
        _cell(strategy=f"s{i}", sharpe=0.01 * i, trades=100, p_value=0.01 * (i + 1))
        for i in range(10)
    ]
    panel = _panel(cells, alpha=0.05, confidence=0.95)
    assert panel.bonferroni_threshold == pytest.approx(0.05 / 10)


def test_a_campaign_without_a_single_sharpe_says_so() -> None:
    panel = _panel([_cell(stage="gate") for _ in range(20)], alpha=0.05, confidence=0.95)
    assert panel.attempts == 20
    assert panel.best_sharpe_per_trade is None
    assert "nothing to correct" in panel.verdict


# -- the statistics behind it --------------------------------------------


def test_required_sharpe_grows_with_the_number_of_trials() -> None:
    values = [
        required_sharpe_per_trade(n, variance_across_trials=0.01, observations=150)
        for n in (10, 100, 300, 1000)
    ]
    assert all(v is not None for v in values)
    assert values == sorted(values)


def test_required_sharpe_falls_with_more_observations() -> None:
    """More trades on the winner make the same score easier to believe."""
    few = required_sharpe_per_trade(300, 0.01, observations=40)
    many = required_sharpe_per_trade(300, 0.01, observations=400)
    assert few > many


# -- a search run in two sittings is still one search --------------------


def test_the_best_result_is_taken_over_the_whole_search() -> None:
    """A cell carried over from an earlier campaign can still win.

    The correction counts every attempt of the search, so the maximum it
    corrects has to come from the same search. Taking it from this campaign's
    cells alone made the reported best depend on where the operator stopped
    for the night, and understated it whenever the earlier half held the
    better cell.
    """
    carried = [{"cell": "earlier / X / D1", "sharpe_per_trade": 0.33, "trades": 32}]
    panel = _panel(
        [_cell(strategy="later", sharpe=0.22, trades=38)]
        + [_cell(stage="gate") for _ in range(99)],
        alpha=0.05,
        confidence=0.95,
        prior_attempts=230,
        prior_results=carried,
    )
    assert panel.attempts == 330
    assert panel.best_strategy == "earlier / X / D1"
    assert panel.best_sharpe_per_trade == pytest.approx(0.33)
    assert "the maximum is over the whole search" in " ".join(panel.assumptions)


def test_a_carried_cell_is_scored_against_its_own_trade_count() -> None:
    """The threshold follows the winner's sample, not this campaign's.

    `required_sharpe_per_trade` divides by the number of trades behind the
    observation. Carrying the Sharpe without its trade count would score the
    winner against a threshold computed for a different cell's sample.
    """
    cells = [_cell(strategy="later", sharpe=0.22, trades=38)]
    few = _panel(
        cells, alpha=0.05, confidence=0.95, prior_attempts=100,
        prior_results=[{"cell": "e / X / D1", "sharpe_per_trade": 0.33, "trades": 32}],
    )
    many = _panel(
        cells, alpha=0.05, confidence=0.95, prior_attempts=100,
        prior_results=[{"cell": "e / X / D1", "sharpe_per_trade": 0.33, "trades": 500}],
    )
    assert few.required_sharpe_per_trade > many.required_sharpe_per_trade


def test_a_pooled_candidate_is_labelled_with_its_period() -> None:
    """Two halves of one search can hold the same cell over different years.

    `ma-crossover / XTIUSD / H1` names one cell in a campaign over 2025 and a
    different one in a campaign over 2020-2026. Without the period the
    headline result cannot be traced back to the report that produced it.
    """
    panel = _panel(
        [_cell(strategy="ma-crossover", symbol="XTIUSD", sharpe=0.33, trades=32)],
        alpha=0.05,
        confidence=0.95,
        period="2025-01-01..2026-01-01",
    )
    assert panel.best_strategy == "ma-crossover / XTIUSD / H1 / 2025-01-01..2026-01-01"
