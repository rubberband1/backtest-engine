"""Deflated Sharpe, family-wise thresholds and PBO."""
from __future__ import annotations

import numpy as np
import pytest

from core.validation.multiple_testing import (
    DEFAULT_CSCV_SLICES,
    Trial,
    deflated_sharpe,
    multiple_testing_report,
    pbo_cscv,
    sharpe_per_trade,
    thresholds,
)


def _trial(run_id: str, sharpe: float, p_value: float | None = 0.5, net: float = 1.0) -> Trial:
    return Trial(
        run_id=run_id,
        strategy_id="s",
        spec_hash=run_id,
        trades=100,
        sharpe_per_trade=sharpe,
        net_pnl=net,
        p_value=p_value,
    )


# -- per-trade Sharpe ----------------------------------------------------


def test_sharpe_per_trade_is_mean_over_standard_deviation() -> None:
    values = np.array([1.0, -1.0, 2.0, -2.0, 3.0])
    assert sharpe_per_trade(values) == pytest.approx(values.mean() / values.std(ddof=1))


def test_a_constant_series_has_no_sharpe_rather_than_an_infinite_one() -> None:
    assert sharpe_per_trade([2.0, 2.0, 2.0]) == 0.0


def test_too_few_trades_give_zero_not_a_crash() -> None:
    assert sharpe_per_trade([1.0]) == 0.0
    assert sharpe_per_trade([]) == 0.0


# -- deflated Sharpe -----------------------------------------------------


def test_deflation_needs_at_least_two_trials_and_says_so() -> None:
    rng = np.random.default_rng(1)
    report = deflated_sharpe(rng.normal(0.1, 1.0, 200), [0.1])
    assert report.valid is False
    assert "at least two configurations" in (report.reason or "")


def test_deflation_refuses_a_series_too_short_to_describe() -> None:
    report = deflated_sharpe([1.0, 2.0], [0.1, 0.2])
    assert report.valid is False
    assert "too few" in (report.reason or "")


def test_identical_trials_leave_nothing_to_deflate() -> None:
    rng = np.random.default_rng(2)
    report = deflated_sharpe(rng.normal(0.1, 1.0, 200), [0.3, 0.3, 0.3])
    assert report.valid is False
    assert "nothing to deflate" in (report.reason or "")


def test_more_trials_raise_the_bar_the_winner_has_to_clear() -> None:
    """The whole point: the same Sharpe is worth less after more attempts."""
    rng = np.random.default_rng(3)
    pnl = rng.normal(0.15, 1.0, 500)
    spread = [0.0, 0.05, 0.1, 0.15, 0.2]
    few = deflated_sharpe(pnl, spread)
    many = deflated_sharpe(pnl, spread * 12)
    assert few.valid and many.valid
    assert many.expected_max_sharpe > few.expected_max_sharpe
    assert many.deflated_sharpe < few.deflated_sharpe


def test_the_deflated_sharpe_never_exceeds_the_uncorrected_one() -> None:
    rng = np.random.default_rng(4)
    report = deflated_sharpe(rng.normal(0.2, 1.0, 400), [0.0, 0.1, 0.2, 0.3])
    assert report.valid
    assert report.deflated_sharpe is not None
    assert report.probabilistic_sharpe is not None
    assert report.deflated_sharpe <= report.probabilistic_sharpe


def test_a_strong_series_deflates_to_a_high_probability() -> None:
    rng = np.random.default_rng(5)
    report = deflated_sharpe(rng.normal(0.5, 1.0, 1000), [0.0, 0.02, 0.04])
    assert report.valid
    assert report.deflated_sharpe is not None and report.deflated_sharpe > 0.9


def test_a_losing_series_deflates_to_almost_nothing() -> None:
    rng = np.random.default_rng(6)
    report = deflated_sharpe(rng.normal(-0.3, 1.0, 500), [0.0, 0.1, 0.2])
    assert report.valid
    assert report.deflated_sharpe is not None and report.deflated_sharpe < 0.05


def test_skew_and_kurtosis_are_reported_and_not_excess() -> None:
    rng = np.random.default_rng(7)
    report = deflated_sharpe(rng.normal(0.1, 1.0, 800), [0.0, 0.1, 0.2])
    assert report.valid
    # a normal series scores about 3 on non-excess kurtosis
    assert report.kurtosis == pytest.approx(3.0, abs=0.6)
    assert abs(report.skewness or 0.0) < 0.4


# -- thresholds ----------------------------------------------------------


def test_bonferroni_divides_alpha_by_the_family_size() -> None:
    rows = thresholds(
        [("a", "s", 0.01, True), ("b", "s", 0.02, True), ("c", "s", 0.5, False)], alpha=0.05
    )
    assert all(row.bonferroni_threshold == pytest.approx(0.05 / 3) for row in rows)
    assert rows[0].passes_bonferroni is True
    assert rows[1].passes_bonferroni is False


def test_benjamini_hochberg_is_less_severe_than_bonferroni() -> None:
    entries = [(f"r{i}", "s", p, True) for i, p in enumerate([0.001, 0.012, 0.03, 0.4, 0.9])]
    rows = thresholds(entries, alpha=0.05)
    bonferroni = sum(row.passes_bonferroni for row in rows)
    benjamini = sum(row.passes_benjamini_hochberg for row in rows)
    assert benjamini >= bonferroni


def test_rows_come_back_ranked_by_p_value() -> None:
    rows = thresholds([("a", "s", 0.4, True), ("b", "s", 0.01, True)], alpha=0.05)
    assert [row.run_id for row in rows] == ["b", "a"]
    assert [row.rank for row in rows] == [1, 2]


def test_the_sign_of_the_mean_pnl_travels_with_the_p_value() -> None:
    rows = thresholds([("a", "s", 0.001, False)], alpha=0.05)
    assert rows[0].passes_bonferroni is True
    # a tiny p-value on a losing strategy: the flag is what stops it being
    # read as an edge
    assert rows[0].mean_pnl_positive is False


def test_an_empty_family_produces_no_rows() -> None:
    assert thresholds([]) == []


# -- PBO -----------------------------------------------------------------


def test_pbo_needs_more_than_one_candidate() -> None:
    report = pbo_cscv(np.ones((100, 1)))
    assert report.valid is False
    assert "at least two" in (report.reason or "")


def test_pbo_needs_enough_observations_to_slice() -> None:
    report = pbo_cscv(np.ones((4, 3)), slices=DEFAULT_CSCV_SLICES)
    assert report.valid is False
    assert "slices" in (report.reason or "")


def test_pure_noise_candidates_give_a_pbo_near_a_coin_flip() -> None:
    """Selecting on noise: the in-sample winner is a coin flip out of sample."""
    rng = np.random.default_rng(11)
    report = pbo_cscv(rng.normal(0, 1, (400, 12)), slices=8)
    assert report.valid
    assert report.pbo is not None
    assert 0.3 < report.pbo < 0.7


def test_one_genuinely_better_candidate_gives_a_low_pbo() -> None:
    rng = np.random.default_rng(12)
    matrix = rng.normal(0, 1, (400, 8))
    matrix[:, 3] += 0.8  # a real, persistent edge on one candidate
    report = pbo_cscv(matrix, slices=8)
    assert report.valid
    assert report.pbo is not None and report.pbo < 0.1


def test_pbo_reports_how_many_splits_it_actually_evaluated() -> None:
    from math import comb

    rng = np.random.default_rng(13)
    report = pbo_cscv(rng.normal(0, 1, (200, 5)), slices=8)
    assert report.splits == comb(8, 4)
    assert report.slices == 8
    assert report.candidates == 5


def test_an_odd_slice_count_is_made_even() -> None:
    rng = np.random.default_rng(14)
    report = pbo_cscv(rng.normal(0, 1, (200, 4)), slices=9)
    assert report.slices == 8


# -- the assembled report ------------------------------------------------


def test_the_report_states_the_trial_count_in_words() -> None:
    rng = np.random.default_rng(15)
    report = multiple_testing_report(
        run_id="r1",
        symbol="TEST",
        period_start=None,
        period_end=None,
        observed_pnl=rng.normal(0.2, 1.0, 300),
        observed_annualized_sharpe=1.4,
        trials_detail=[_trial("a", 0.05), _trial("b", 0.1), _trial("c", 0.2)],
    )
    assert report.trials == 3
    assert "3 configuration(s) tried" in report.verdict
    assert "deflated Sharpe" in report.verdict


def test_a_negative_run_is_flagged_before_any_p_value_is_read() -> None:
    rng = np.random.default_rng(16)
    report = multiple_testing_report(
        run_id="r1",
        symbol="TEST",
        period_start=None,
        period_end=None,
        observed_pnl=rng.normal(-0.3, 1.0, 300),
        observed_annualized_sharpe=-1.0,
        trials_detail=[_trial("a", -0.1, 0.01, net=-5.0)],
    )
    assert any("negative" in warning for warning in report.warnings)
    assert "reliably it loses" in report.verdict


def test_without_a_grid_pbo_is_declared_not_computed() -> None:
    report = multiple_testing_report(
        run_id="r1",
        symbol="TEST",
        period_start=None,
        period_end=None,
        observed_pnl=np.random.default_rng(17).normal(0.1, 1.0, 200),
        observed_annualized_sharpe=0.5,
        trials_detail=[_trial("a", 0.1), _trial("b", 0.2)],
    )
    assert report.pbo.valid is False
    assert "no parameter grid" in (report.pbo.reason or "")
    assert "PBO not computed" in report.verdict


def test_a_single_trial_raises_a_warning_about_the_missing_correction() -> None:
    report = multiple_testing_report(
        run_id="r1",
        symbol="TEST",
        period_start=None,
        period_end=None,
        observed_pnl=np.random.default_rng(18).normal(0.1, 1.0, 200),
        observed_annualized_sharpe=0.5,
        trials_detail=[_trial("a", 0.1)],
    )
    assert any("fewer than two trials" in warning for warning in report.warnings)


def test_the_report_serializes_to_json_safe_primitives() -> None:
    import json

    report = multiple_testing_report(
        run_id="r1",
        symbol="TEST",
        period_start=None,
        period_end=None,
        observed_pnl=np.random.default_rng(19).normal(0.1, 1.0, 200),
        observed_annualized_sharpe=0.5,
        trials_detail=[_trial("a", 0.1), _trial("b", 0.3)],
        performance_matrix=np.random.default_rng(20).normal(0, 1, (200, 4)),
    )
    json.dumps(report.as_dict())
