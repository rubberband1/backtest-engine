"""Run persistence: deterministic id, reuse, filters, deletion."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from core.runs.runner import execute_run, plan_run
from core.runs.store import (
    ENGINE_VERSION,
    RunConfig,
    RunNotFound,
    RunStore,
    data_fingerprint,
)
from tests.conftest_engine import random_walk, spec_from, symbol_spec, symbol_spec_snapshot

ATHENS = ZoneInfo("Europe/Athens")


def config(**overrides) -> RunConfig:
    base = {
        "symbol": "TEST",
        "timeframe": "M1",
        "initial_equity": 1000.0,
        "spread_mode": "per_bar",
    }
    base.update(overrides)
    return RunConfig(**base)


def run_once(store: RunStore, bars, spec=None, cfg=None):
    spec = spec or spec_from()
    cfg = cfg or config()
    return execute_run(store, spec, cfg, bars, symbol_spec_snapshot(), ATHENS)


def test_deterministic_run_id(tmp_path: Path) -> None:
    bars = random_walk(400)
    spec = spec_from()
    first, fingerprint = plan_run(spec, config(), bars, symbol_spec())
    second, again = plan_run(spec, config(), bars, symbol_spec())
    assert first == second
    assert fingerprint == again
    assert len(first) == 16


def test_run_id_changes_with_spec_config_and_data() -> None:
    bars = random_walk(400)
    spec = spec_from()
    base, _ = plan_run(spec, config(), bars, symbol_spec())

    other_spec = spec_from(
        exit_block={"stop_loss": {"type": "points", "value": 200},
                    "take_profit": None, "time_stop": None, "signal_exit": None}
    )
    assert plan_run(other_spec, config(), bars, symbol_spec())[0] != base
    assert plan_run(spec, config(commission_per_lot_per_side=3.0), bars, symbol_spec())[0] != base

    touched = bars.copy()
    touched.iloc[100, touched.columns.get_loc("close")] += 0.01
    assert plan_run(spec, config(), touched, symbol_spec())[0] != base


def test_run_id_changes_with_symbol_spec_cost_fields() -> None:
    """A2 drift bug: the broker moving swap/tick_value must move the run_id."""
    bars = random_walk(400)
    spec = spec_from()
    base, _ = plan_run(spec, config(), bars, symbol_spec())
    assert plan_run(spec, config(), bars, symbol_spec(tick_value=2.0))[0] != base
    assert plan_run(spec, config(), bars, symbol_spec(swap_long=-5.0))[0] != base
    # a descriptive-only change (name) must NOT change the run_id
    assert plan_run(spec, config(), bars, symbol_spec(name="OTHER"))[0] == base


def test_data_fingerprint_reacts_to_a_single_candle() -> None:
    """The time range is not enough: if the broker rewrites one bar, everything changes."""
    bars = random_walk(300)
    other = bars.copy()
    other.iloc[42, other.columns.get_loc("high")] += 0.5
    assert data_fingerprint(bars) != data_fingerprint(other)
    assert data_fingerprint(bars) == data_fingerprint(bars.copy())


def test_saving_writes_every_file(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    bars = random_walk(600)
    meta = run_once(store, bars)

    path = store.path_for(meta.run_id)
    for name in ("spec.json", "config.json", "meta.json", "metrics.json",
                 "trades.parquet", "equity.parquet", "symbol_spec.json"):
        assert (path / name).exists(), name

    assert meta.status == "done"
    assert meta.engine_version == ENGINE_VERSION
    assert meta.bars == len(bars)
    assert meta.duration_seconds is not None and meta.duration_seconds >= 0
    assert meta.data_start == bars.index[0].to_pydatetime()
    assert meta.symbol_spec_hash is not None


def test_run_registers_its_symbol_spec_snapshot(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    snapshot = symbol_spec_snapshot(tick_value=0.77)
    meta = execute_run(store, spec_from(), config(), random_walk(300), snapshot, ATHENS)

    record = store.load_run(meta.run_id)
    assert record.symbol_spec_registered is True
    assert record.symbol_spec.spec.tick_value == pytest.approx(0.77)
    assert record.symbol_spec.read_at == snapshot.read_at


def test_run_without_symbol_spec_file_is_reported_unregistered(tmp_path: Path) -> None:
    """A run persisted before A1 has no symbol_spec.json: it must not be assumed current."""
    store = RunStore(tmp_path)
    meta = run_once(store, random_walk(300))
    (store.path_for(meta.run_id) / "symbol_spec.json").unlink()

    record = store.load_run(meta.run_id)
    assert record.symbol_spec is None
    assert record.symbol_spec_registered is False


def test_loaded_run_is_identical(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    bars = random_walk(600)
    meta = run_once(store, bars)

    record = store.load_run(meta.run_id)
    assert record.spec.id == "test"
    assert record.config.symbol == "TEST"
    assert record.config.initial_equity == 1000.0
    assert len(record.trades()) == record.metrics["execution"]["trades"]
    equity = record.equity()
    assert len(equity) == len(bars)
    assert str(equity.index.tz) == "UTC"
    assert record.metrics["strategy"]["final_equity"] == pytest.approx(equity.iloc[-1])


def test_metrics_serialized_without_nan(tmp_path: Path) -> None:
    """Infinite profit factor and timedelta durations must stay valid JSON."""
    store = RunStore(tmp_path)
    meta = run_once(store, random_walk(600))
    raw = (store.path_for(meta.run_id) / "metrics.json").read_text(encoding="utf-8")
    payload = json.loads(raw)  # fallirebbe su NaN/Infinity
    assert payload["strategy"]["label"] == "test"
    assert "text" in payload["strategy"]
    assert isinstance(payload["strategy"]["drawdown_unsustainable"], bool)


def test_identical_run_is_reused(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    bars = random_walk(600)
    first = run_once(store, bars)
    assert store.exists(first.run_id)

    run_id, _ = plan_run(spec_from(), config(), bars, symbol_spec())
    assert run_id == first.run_id
    # the caller decides whether to re-run: the store only says it is already there
    assert len(store.list_runs()) == 1


def test_listing_with_filters_and_summary_metrics(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    bars = random_walk(600)
    run_once(store, bars, cfg=config(symbol="TEST"))
    run_once(store, bars, cfg=config(symbol="OTHER"))

    everything = store.list_runs()
    assert len(everything) == 2
    assert everything[0].created_at >= everything[1].created_at

    only_test = store.list_runs(symbol="TEST")
    assert len(only_test) == 1
    summary = only_test[0]
    assert summary.symbol == "TEST"
    assert summary.strategy_id == "test"
    assert summary.trades is not None
    assert summary.final_equity is not None
    assert store.list_runs(strategy_id="missing") == []
    assert len(store.list_runs(status="done")) == 2
    assert len(store.list_runs(limit=1)) == 1


def test_failed_run_stays_tracked(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    bars = random_walk(200)
    spec = spec_from(risk={"max_open_positions": 5})  # not supported by the engine

    with pytest.raises(NotImplementedError):
        execute_run(store, spec, config(), bars, symbol_spec_snapshot(), ATHENS)

    runs = store.list_runs(status="error")
    assert len(runs) == 1
    assert "max_open_positions" in runs[0].error
    assert runs[0].final_equity is None


def test_deletion(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    meta = run_once(store, random_walk(300))
    assert store.delete_run(meta.run_id) is True
    assert store.delete_run(meta.run_id) is False
    assert store.list_runs() == []
    with pytest.raises(RunNotFound):
        store.load_run(meta.run_id)


def test_malformed_run_id_rejected(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    for bad in ("../escape", "a/b", "", "..\\other"):
        with pytest.raises(ValueError, match="run_id"):
            store.path_for(bad)


def test_corrupted_folder_does_not_block_the_listing(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    good = run_once(store, random_walk(300))
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "meta.json").write_text("{non json", encoding="utf-8")

    runs = store.list_runs()
    assert [r.run_id for r in runs] == [good.run_id]


def test_config_roundtrip() -> None:
    original = config(
        start=datetime(2025, 1, 1, tzinfo=timezone.utc),
        end=datetime(2025, 6, 1, tzinfo=timezone.utc),
        spread_mode="quantile",
        spread_value=0.9,
    )
    again = RunConfig.from_dict(original.to_dict())
    assert again == original
    costs = again.cost_model()
    assert costs.spread.mode == "quantile"
    assert costs.spread.value == 0.9


def test_an_int_and_a_float_configuration_are_the_same_run() -> None:
    """`100` and `100.0` are one configuration, so they must be one run_id."""
    from core.runs.store import RunConfig

    integers = RunConfig(symbol="X", timeframe="M1", initial_equity=100,
                         commission_per_lot_per_side=0, session_threshold=1)
    floats = RunConfig(symbol="X", timeframe="M1", initial_equity=100.0,
                       commission_per_lot_per_side=0.0, session_threshold=1.0)
    assert integers.to_dict() == floats.to_dict()
    assert integers == floats


def test_a_null_spread_value_stays_null() -> None:
    from core.runs.store import RunConfig

    assert RunConfig(symbol="X", timeframe="M1", spread_value=None).spread_value is None
    assert RunConfig(symbol="X", timeframe="M1", spread_value=2).spread_value == 2.0


def test_a_naive_bound_is_read_as_utc() -> None:
    """`start=2020-01-01` is the most ordinary input the API takes.

    The bars carry a tz-aware UTC index, so a naive bound could not be
    compared against it and came back from pandas as a TypeError - which the
    API could only report as an internal error. Every other entry point
    already read a naive value as UTC; the config does too, so the run_id is
    the same whichever way the bound arrived.
    """
    naive = RunConfig(
        symbol="X", timeframe="H1",
        start=datetime(2020, 1, 1), end=datetime(2021, 1, 1),
    )
    aware = RunConfig(
        symbol="X", timeframe="H1",
        start=datetime(2020, 1, 1, tzinfo=timezone.utc),
        end=datetime(2021, 1, 1, tzinfo=timezone.utc),
    )
    assert naive.start is not None and naive.start.tzinfo is timezone.utc
    assert naive.to_dict() == aware.to_dict()
