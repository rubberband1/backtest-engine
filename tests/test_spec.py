"""Spec validation: a wrong spec must fail immediately and say so clearly."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from core.strategy.spec import SpecError, StrategySpec

BASELINE_PATH = Path("strategies/rsi-wick-baseline.json")


def baseline_payload() -> dict[str, Any]:
    return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))


def test_baseline_loads() -> None:
    spec = StrategySpec.from_json(BASELINE_PATH)
    assert spec.id == "rsi-wick-baseline"
    assert spec.instrument.symbol == "XAUUSD.r"
    assert spec.instrument.tf.minutes == 1
    assert {i.id for i in spec.indicators} == {"rsi", "atr"}
    assert spec.exit.stop_loss is not None and spec.exit.stop_loss.value == 150
    assert spec.risk.max_spread_points == 30


def test_json_roundtrip() -> None:
    spec = StrategySpec.from_json(BASELINE_PATH)
    again = StrategySpec.from_json(spec.to_json())
    assert again.model_dump() == spec.model_dump()


def test_save_and_reload(tmp_path: Path) -> None:
    spec = StrategySpec.from_json(BASELINE_PATH)
    saved = spec.save(tmp_path / "copy.json")
    assert StrategySpec.from_json(saved).id == spec.id


def test_duplicate_indicator_ids() -> None:
    payload = baseline_payload()
    payload["indicators"].append({"id": "rsi", "type": "rsi", "params": {"period": 21}})
    with pytest.raises(SpecError, match="duplicate id 'rsi'"):
        StrategySpec.from_dict(payload)


def test_ref_to_missing_indicator_lists_the_defined_ones() -> None:
    payload = baseline_payload()
    payload["entry"]["long"]["operands"][0]["left"] = {"ref": "fast_rsi"}
    with pytest.raises(SpecError) as error:
        StrategySpec.from_dict(payload)
    message = str(error.value)
    assert "fast_rsi" in message
    assert "Defined: atr, rsi" in message


def test_params_inconsistent_with_type() -> None:
    payload = baseline_payload()
    payload["indicators"][0]["params"] = {"period": 0}
    with pytest.raises(SpecError, match="greater than or equal to 1"):
        StrategySpec.from_dict(payload)

    payload = baseline_payload()
    payload["indicators"][0]["params"] = {"periodo": 9}
    with pytest.raises(SpecError, match="periodo"):
        StrategySpec.from_dict(payload)


def test_unknown_indicator() -> None:
    payload = baseline_payload()
    payload["indicators"][0]["type"] = "supertrend"
    with pytest.raises(SpecError, match="supertrend"):
        StrategySpec.from_dict(payload)


def test_macd_must_be_referenced_with_an_output() -> None:
    payload = baseline_payload()
    payload["indicators"] = [{"id": "m", "type": "macd", "params": {}}]
    payload["entry"]["long"] = {
        "op": "gt", "left": {"ref": "m"}, "right": {"const": 0}
    }
    payload["entry"]["short"] = None
    with pytest.raises(SpecError, match="name one of macd, signal, histogram"):
        StrategySpec.from_dict(payload)

    payload["entry"]["long"]["left"] = {"ref": "m.histo"}
    with pytest.raises(SpecError, match="output 'histo' does not exist"):
        StrategySpec.from_dict(payload)

    payload["entry"]["long"]["left"] = {"ref": "m.histogram"}
    assert StrategySpec.from_dict(payload).indicators[0].id == "m"


def test_suffix_on_single_output_indicator() -> None:
    payload = baseline_payload()
    payload["entry"]["long"]["operands"][0]["left"] = {"ref": "rsi.value"}
    with pytest.raises(SpecError, match="has a single output"):
        StrategySpec.from_dict(payload)


def test_unknown_operator() -> None:
    payload = baseline_payload()
    payload["entry"]["long"]["operands"][0]["op"] = "almost_equal"
    with pytest.raises(SpecError, match="almost_equal"):
        StrategySpec.from_dict(payload)


def test_malformed_operand() -> None:
    payload = baseline_payload()
    payload["entry"]["long"]["operands"][0]["right"] = {"value": 25}
    with pytest.raises(SpecError, match="right"):
        StrategySpec.from_dict(payload)


def test_no_exit_at_all() -> None:
    payload = baseline_payload()
    payload["exit"] = {
        "stop_loss": None, "take_profit": None, "time_stop": None, "signal_exit": None
    }
    with pytest.raises(SpecError, match="never closes"):
        StrategySpec.from_dict(payload)


def test_atr_and_percent_levels_are_accepted() -> None:
    payload = baseline_payload()
    payload["exit"]["stop_loss"] = {"type": "atr", "indicator": "atr", "mult": 2.0}
    payload["exit"]["take_profit"] = {"type": "percent", "value": 0.15}
    spec = StrategySpec.from_dict(payload)
    assert spec.exit.stop_loss.type == "atr"
    assert spec.exit.stop_loss.mult == 2.0
    assert spec.exit.take_profit.type == "percent"
    assert spec.exit.take_profit.value == 0.15


def test_atr_level_pointing_at_a_missing_indicator() -> None:
    payload = baseline_payload()
    payload["exit"]["stop_loss"] = {"type": "atr", "indicator": "atr20", "mult": 2.0}
    with pytest.raises(SpecError) as error:
        StrategySpec.from_dict(payload)
    message = str(error.value)
    assert "atr20" in message and "Defined: atr, rsi" in message


def test_atr_level_pointing_at_a_non_atr_indicator() -> None:
    payload = baseline_payload()
    payload["exit"]["stop_loss"] = {"type": "atr", "indicator": "rsi", "mult": 2.0}
    with pytest.raises(SpecError, match="must be 'atr'"):
        StrategySpec.from_dict(payload)


def test_an_indicator_used_only_by_an_exit_is_not_reported_unused(caplog) -> None:
    """The ATR of an ATR stop is referenced, even though no condition names it."""
    payload = baseline_payload()
    payload["exit"]["stop_loss"] = {"type": "atr", "indicator": "atr", "mult": 2.0}
    with caplog.at_level("WARNING"):
        StrategySpec.from_dict(payload)
    assert "never referenced" not in caplog.text


def test_a_level_needs_a_known_type() -> None:
    payload = baseline_payload()
    payload["exit"]["stop_loss"] = {"type": "atr_trailing", "indicator": "atr", "mult": 2.0}
    with pytest.raises(SpecError):
        StrategySpec.from_dict(payload)


def test_entry_without_sides() -> None:
    payload = baseline_payload()
    payload["entry"] = {"long": None, "short": None}
    with pytest.raises(SpecError, match="at least one of"):
        StrategySpec.from_dict(payload)


def test_inconsistent_sizing() -> None:
    payload = baseline_payload()
    payload["sizing"]["max_lot"] = 0.001
    with pytest.raises(SpecError, match="max_lot"):
        StrategySpec.from_dict(payload)


def test_wrong_schema_version() -> None:
    payload = baseline_payload()
    payload["schema_version"] = 2
    with pytest.raises(SpecError, match="schema_version 2"):
        StrategySpec.from_dict(payload)


def test_news_filter_not_implemented() -> None:
    payload = baseline_payload()
    payload["risk"]["news_filter"] = {"minutes_before": 30}
    with pytest.raises(SpecError, match="news_filter"):
        StrategySpec.from_dict(payload)


def test_unknown_field_rejected() -> None:
    payload = baseline_payload()
    payload["leverage"] = 500
    with pytest.raises(SpecError, match="leverage"):
        StrategySpec.from_dict(payload)


def test_unknown_timeframe() -> None:
    payload = baseline_payload()
    payload["instrument"]["timeframe"] = "M7"
    with pytest.raises(SpecError, match="M7"):
        StrategySpec.from_dict(payload)


def test_malformed_json_points_at_the_line(tmp_path: Path) -> None:
    broken = tmp_path / "broken.json"
    broken.write_text('{\n  "schema_version": 1,\n', encoding="utf-8")
    with pytest.raises(SpecError, match="invalid JSON at line"):
        StrategySpec.from_json(broken)


def test_message_lists_every_problem() -> None:
    payload = baseline_payload()
    payload["sizing"]["equity_per_001_lot"] = -5
    payload["risk"]["cooldown_minutes"] = -1
    with pytest.raises(SpecError) as error:
        StrategySpec.from_dict(payload)
    message = str(error.value)
    assert "2 problem(s)" in message
    assert "sizing.equity_per_001_lot" in message
    assert "risk.cooldown_minutes" in message
