"""Cross-sectional consistency: what many instruments say together."""
from __future__ import annotations

import pytest

from core.batch.runner import CellResult, Period, consistency


def _cell(
    symbol: str,
    mean_r: float | None = 0.1,
    status: str = "done",
    trades: int = 50,
) -> CellResult:
    return CellResult(
        symbol=symbol,
        params={},
        period_start=None,
        period_end=None,
        run_id=f"run-{symbol}",
        status=status,
        bars=1000,
        trades=trades,
        mean_r=mean_r,
        expectancy=mean_r,
        net_pnl=(mean_r or 0.0) * trades,
    )


def test_the_sign_split_is_counted_per_instrument() -> None:
    cells = [_cell("A", 0.2), _cell("B", 0.1), _cell("C", -0.3)]
    measure = consistency(cells)
    assert measure.observations == 3
    assert measure.positive == 2
    assert measure.negative == 1


def test_several_cells_on_one_instrument_average_into_a_single_vote() -> None:
    """A symbol is one draw, not one per configuration tried on it."""
    cells = [_cell("A", 1.0), _cell("A", -1.0), _cell("B", 0.5)]
    measure = consistency(cells)
    assert measure.observations == 2
    assert measure.positive == 1
    # A averages to zero, so it counts as neither positive nor negative
    assert measure.negative == 0


def test_failed_cells_are_excluded_and_counted_separately() -> None:
    cells = [_cell("A", 0.2), _cell("B", None, status="error"), _cell("C", 0.1)]
    measure = consistency(cells)
    assert measure.observations == 2
    assert measure.zero_or_missing == 1


def test_no_usable_cell_gives_an_explicit_nothing() -> None:
    measure = consistency([_cell("A", None, status="error")])
    assert measure.observations == 0
    assert measure.mean is None
    assert "nothing to be consistent about" in measure.verdict


def test_an_aggregate_carried_by_one_instrument_is_flagged() -> None:
    """Remove the outlier and the sign flips: that is not a strategy property."""
    cells = [_cell("BIG", 5.0), _cell("A", -0.4), _cell("B", -0.3), _cell("C", -0.2)]
    measure = consistency(cells)
    assert measure.carried_by_one_symbol is True
    assert measure.dominant_symbol == "BIG"
    assert "carried by BIG alone" in measure.verdict


def test_a_broadly_positive_result_is_not_flagged_as_carried() -> None:
    cells = [_cell("A", 0.3), _cell("B", 0.25), _cell("C", 0.2), _cell("D", 0.28)]
    measure = consistency(cells)
    assert measure.carried_by_one_symbol is False
    assert measure.positive == 4


def test_too_few_instruments_disable_the_claim_rather_than_the_number() -> None:
    measure = consistency([_cell("A", 0.2), _cell("B", 0.3)])
    assert measure.observations == 2
    assert measure.mean == pytest.approx(0.25)
    assert "no power" in measure.verdict


def test_an_edge_on_exactly_one_instrument_is_called_a_suspect() -> None:
    cells = [_cell("A", 0.05), _cell("B", -0.2), _cell("C", -0.25), _cell("D", -0.3)]
    measure = consistency(cells)
    assert measure.positive == 1
    assert "suspect" in measure.verdict


def test_the_sign_test_gets_stronger_as_the_instruments_agree() -> None:
    unanimous = consistency([_cell(name, 0.2) for name in "ABCDEFGH"])
    split = consistency(
        [_cell(name, 0.2) for name in "ABCD"] + [_cell(name, -0.2) for name in "EFGH"]
    )
    assert unanimous.sign_p_value is not None and split.sign_p_value is not None
    assert unanimous.sign_p_value < split.sign_p_value


def test_the_spread_across_instruments_is_reported_with_its_standard_error() -> None:
    measure = consistency([_cell("A", 0.1), _cell("B", 0.3), _cell("C", 0.5)])
    assert measure.minimum == pytest.approx(0.1)
    assert measure.maximum == pytest.approx(0.5)
    assert measure.median == pytest.approx(0.3)
    assert measure.stderr is not None and measure.stderr > 0


def test_a_single_instrument_has_no_standard_error_to_report() -> None:
    measure = consistency([_cell("A", 0.2)])
    assert measure.std is None
    assert measure.stderr is None


def test_the_metric_can_be_switched() -> None:
    cells = [_cell("A", 0.1, trades=10), _cell("B", -0.2, trades=10)]
    by_r = consistency(cells, "mean_r")
    by_pnl = consistency(cells, "net_pnl")
    assert by_r.metric == "mean_r"
    assert by_pnl.metric == "net_pnl"
    assert by_pnl.mean == pytest.approx(((0.1 * 10) + (-0.2 * 10)) / 2)


def test_a_period_serializes_without_a_datetime_in_the_way() -> None:
    import json

    json.dumps(Period().as_dict())


def test_the_consistency_report_serializes_to_json_safe_primitives() -> None:
    import json

    json.dumps(consistency([_cell("A", 0.2), _cell("B", -0.1)]).as_dict())
