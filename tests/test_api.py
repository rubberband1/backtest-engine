"""API via TestClient: contract, readable errors, no stack trace leaking out."""
from __future__ import annotations

import importlib
import json
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from core.data.cache import ParquetCache
from core.data.provider import Timeframe
from tests.conftest_engine import random_walk, symbol_spec

SYMBOL = "SYNTH"
OTHER_SYMBOL = "SYNTH2"


def _bars(n: int = 4000, seed: int = 5) -> pd.DataFrame:
    """Synthetic bars with some guaranteed signals and a variable spread."""
    bars = random_walk(n, seed=seed)
    index = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC", name="time")
    bars.index = index
    rng = np.random.default_rng(seed)
    bars["spread"] = rng.integers(8, 20, n).astype(float)
    return bars


SPEC: dict[str, Any] = {
    "schema_version": 1,
    "id": "api-test",
    "name": "test strategy",
    "description": "used by the API tests",
    "instrument": {"symbol": SYMBOL, "timeframe": "M1"},
    "indicators": [{"id": "rsi", "type": "rsi", "params": {"period": 9}}],
    "entry": {
        "long": {"op": "cross_below", "left": {"ref": "rsi"}, "right": {"const": 30}},
        "short": {"op": "cross_above", "left": {"ref": "rsi"}, "right": {"const": 70}},
    },
    "exit": {
        "stop_loss": {"type": "points", "value": 150},
        "take_profit": {"type": "points", "value": 80},
        "time_stop": {"bars": 60},
        "signal_exit": None,
    },
    "sizing": {"type": "equity_per_step", "equity_per_001_lot": 100,
               "min_lot": 0.01, "max_lot": 0.05},
    "risk": {"max_open_positions": 1, "cooldown_minutes": 0, "max_trades_per_day": None,
             "max_spread_points": None, "session": None, "news_filter": None},
}


@pytest.fixture(scope="module")
def client(tmp_path_factory: pytest.TempPathFactory) -> Iterator[TestClient]:
    """Isolated app: cache, runs and strategies in a temporary folder."""
    root = tmp_path_factory.mktemp("api")
    cache_dir = root / "data_cache"
    runs_dir = root / "runs"
    strategies_dir = root / "strategies"
    strategies_dir.mkdir()
    (strategies_dir / "api-test.json").write_text(json.dumps(SPEC, indent=2), encoding="utf-8")
    (strategies_dir / "broken.json").write_text('{"schema_version": 99}', encoding="utf-8")

    cache = ParquetCache(cache_dir)
    bars = _bars()
    for symbol in (SYMBOL, OTHER_SYMBOL):
        cache.write_year(symbol, Timeframe.M1, 2024, bars,
                         [(bars.index[0].to_pydatetime(), bars.index[-1].to_pydatetime())])
        folder = cache_dir / symbol
        folder.mkdir(parents=True, exist_ok=True)
        spec = symbol_spec(name=symbol)
        (folder / "symbol_spec.json").write_text(
            json.dumps({**spec.__dict__}), encoding="utf-8"
        )
    (cache_dir / "server_timezone.txt").write_text("Europe/Athens", encoding="utf-8")

    import os

    os.environ["BACKTEST_CACHE_DIR"] = str(cache_dir)
    os.environ["BACKTEST_RUNS_DIR"] = str(runs_dir)
    os.environ["BACKTEST_STRATEGIES_DIR"] = str(strategies_dir)
    import api.main as main

    importlib.reload(main)
    with TestClient(main.app, raise_server_exceptions=False) as test_client:
        yield test_client


def config(**overrides: Any) -> dict[str, Any]:
    base = {"symbol": SYMBOL, "timeframe": "M1", "initial_equity": 1000.0,
            "spread_mode": "per_bar"}
    base.update(overrides)
    return base


def run_backtest(client: TestClient, **overrides: Any) -> dict[str, Any]:
    response = client.post(
        "/api/backtest",
        json={"strategy_id": "api-test", "config": config(**overrides)},
    )
    assert response.status_code == 200, response.text
    return response.json()


# -- health and instruments ----------------------------------------------


def test_health(client: TestClient) -> None:
    payload = client.get("/api/health").json()
    assert payload["status"] == "ok"


def test_symbols_always_include_those_with_data(client: TestClient) -> None:
    """Terminal or not, the symbols one can work on must be there.

    If MT5 does not answer the API uses the specs saved next to the cache
    instead of refusing the request.
    """
    payload = client.get("/api/symbols").json()
    assert payload["source"] in ("terminal", "cache")
    names = [item["name"] for item in payload["symbols"]]
    assert SYMBOL in names
    assert SYMBOL in payload["cached_symbols"]
    spec = next(item for item in payload["symbols"] if item["name"] == SYMBOL)
    assert spec["point"] == 0.01
    assert spec["volume_min"] == 0.01
    # the ones with data come first in the listing: it is a menu, not an archive
    assert names.index(SYMBOL) < len(payload["cached_symbols"])


def test_coverage_with_quality(client: TestClient) -> None:
    payload = client.get(f"/api/symbols/{SYMBOL}/coverage").json()
    assert payload["years"] == [2024]
    assert payload["bars"] == 4000
    assert payload["quality"]["rows"] == 4000
    assert "completeness" in payload["quality"]["text"]
    assert payload["start"].startswith("2024-01-01")


def test_coverage_of_symbol_without_data(client: TestClient) -> None:
    payload = client.get("/api/symbols/NEVER_DOWNLOADED/coverage").json()
    assert payload["bars"] == 0
    assert payload["years"] == []
    assert payload["quality"] is None


def test_coverage_invalid_timeframe(client: TestClient) -> None:
    response = client.get(f"/api/symbols/{SYMBOL}/coverage?timeframe=M7")
    assert response.status_code == 400
    assert "M7" in response.json()["detail"]


# -- strategies ----------------------------------------------------------


def test_strategy_listing_skips_broken_files(client: TestClient) -> None:
    items = client.get("/api/strategies").json()
    assert [item["id"] for item in items] == ["api-test"]
    assert items[0]["symbol"] == SYMBOL


def test_strategy_by_id(client: TestClient) -> None:
    payload = client.get("/api/strategies/api-test").json()
    assert payload["name"] == "test strategy"
    assert payload["spec"]["indicators"][0]["type"] == "rsi"


def test_missing_strategy(client: TestClient) -> None:
    response = client.get("/api/strategies/ghost")
    assert response.status_code == 404
    assert "ghost" in response.json()["detail"]


def test_validation_of_valid_spec(client: TestClient) -> None:
    payload = client.post("/api/strategies/validate", json={"spec": SPEC}).json()
    assert payload["valid"] is True
    assert payload["normalized"]["id"] == "api-test"


def test_validation_of_invalid_spec_lists_the_problems(client: TestClient) -> None:
    broken = json.loads(json.dumps(SPEC))
    broken["entry"]["long"]["left"] = {"ref": "missing"}
    broken["sizing"]["equity_per_001_lot"] = -1

    response = client.post("/api/strategies/validate", json={"spec": broken})
    assert response.status_code == 200  # validation answers, it does not blow up
    payload = response.json()
    assert payload["valid"] is False
    assert payload["errors"]
    assert any("missing" in line or "equity" in line for line in payload["errors"])
    assert "Traceback" not in json.dumps(payload)


# -- gate zero -----------------------------------------------------------


def test_edge_gate(client: TestClient) -> None:
    response = client.post(
        "/api/edge",
        json={"strategy_id": "api-test", "config": config(), "horizons": [5, 15, 30]},
    )
    assert response.status_code == 200, response.text
    payload = response.json()

    assert payload["horizons"] == [5, 15, 30]
    assert len(payload["stats"]) == 9  # 3 horizons x (long, short, both)
    assert payload["signals_long"] > 0
    assert isinstance(payload["passed"], bool)
    assert payload["verdict"]
    for stat in payload["stats"]:
        assert stat["direction"] in ("long", "short", "both")
        assert stat["spread_cost_points"] is None or stat["spread_cost_points"] > 0


def test_edge_gate_breakeven_prior(client: TestClient) -> None:
    payload = client.post(
        "/api/edge", json={"strategy_id": "api-test", "config": config(), "horizons": [5]}
    ).json()
    prior = payload["breakeven_prior"]
    assert prior["valid"] is True
    # SL 150, TP 80, no commission: 150 / 230
    assert prior["breakeven_win_rate"] == pytest.approx(150 / 230, abs=1e-9)
    assert prior["avg_spread_points"] is not None and prior["avg_spread_points"] > 0
    assert prior["caveats"]


def test_edge_gate_invalid_horizons(client: TestClient) -> None:
    response = client.post(
        "/api/edge", json={"strategy_id": "api-test", "config": config(), "horizons": [0]}
    )
    assert response.status_code == 400


def test_request_without_strategy(client: TestClient) -> None:
    response = client.post("/api/edge", json={"config": config()})
    assert response.status_code == 400
    assert "strategy_id" in response.json()["detail"]


def test_missing_data_yields_409(client: TestClient) -> None:
    response = client.post(
        "/api/edge", json={"strategy_id": "api-test", "config": config(symbol="MISSING")}
    )
    assert response.status_code == 409
    assert "cache" in response.json()["detail"]


def test_fixed_spread_without_value(client: TestClient) -> None:
    response = client.post(
        "/api/edge",
        json={"strategy_id": "api-test", "config": config(spread_mode="fixed")},
    )
    assert response.status_code == 400
    assert "spread_value" in response.json()["detail"]


def test_quantile_spread_out_of_range(client: TestClient) -> None:
    response = client.post(
        "/api/backtest",
        json={
            "strategy_id": "api-test",
            "config": config(spread_mode="quantile", spread_value=7.0),
        },
    )
    assert response.status_code == 400
    assert "quantile" in response.json()["detail"]


# -- backtest and runs ---------------------------------------------------


def test_backtest_and_reuse(client: TestClient) -> None:
    first = run_backtest(client)
    assert first["status"] in ("done", "running")
    assert first["reused"] is False
    assert first["poll_url"] == f"/api/runs/{first['run_id']}"

    second = run_backtest(client)
    assert second["run_id"] == first["run_id"]
    assert second["reused"] is True
    assert "reused" in second["message"]


def test_run_detail(client: TestClient) -> None:
    run_id = run_backtest(client)["run_id"]
    payload = client.get(f"/api/runs/{run_id}").json()

    assert payload["status"] == "done"
    assert payload["engine_version"]
    assert payload["spec"]["id"] == "api-test"
    assert payload["config"]["symbol"] == SYMBOL
    assert payload["strategy"]["label"] == "api-test"
    assert payload["benchmark"]["label"] == "buy & hold"
    assert payload["execution"]["trades"] >= 0
    assert isinstance(payload["strategy"]["drawdown_unsustainable"], bool)
    assert payload["breakeven"] is not None
    assert payload["breakeven"]["observations"] == payload["execution"]["trades"]
    # infinite or NaN profit factor comes out as null, not 'Infinity'
    assert "Infinity" not in response_text(client, f"/api/runs/{run_id}")


def response_text(client: TestClient, url: str) -> str:
    return client.get(url).text


def test_equity_and_downsampling(client: TestClient) -> None:
    run_id = run_backtest(client)["run_id"]
    payload = client.get(f"/api/runs/{run_id}/equity?points=200").json()

    assert payload["total_points"] == 4000
    assert payload["downsampled"] is True
    assert payload["returned_points"] <= 210
    assert len(payload["points"]) == payload["returned_points"]

    first, last = payload["points"][0], payload["points"][-1]
    assert first["t"] < last["t"]
    assert all(point["drawdown"] <= 0 for point in payload["points"])

    # the drawdown comes from the whole series: the minimum must not vanish in the cut
    full = client.get(f"/api/runs/{run_id}/equity?points=2000").json()
    worst_full = min(p["drawdown"] for p in full["points"])
    worst_small = min(p["drawdown"] for p in payload["points"])
    assert worst_small == pytest.approx(worst_full)


def test_paginated_trades(client: TestClient) -> None:
    run_id = run_backtest(client)["run_id"]
    page = client.get(f"/api/runs/{run_id}/trades?offset=0&limit=5").json()

    assert page["total"] > 5
    assert len(page["items"]) == 5
    assert page["items"][0]["index"] == 0
    assert set(page["items"][0]) >= {"entry_time", "exit_reason", "net_pnl", "ambiguous"}

    second = client.get(f"/api/runs/{run_id}/trades?offset=5&limit=5").json()
    assert second["items"][0]["index"] == 5

    only_ambiguous = client.get(f"/api/runs/{run_id}/trades?ambiguous_only=true").json()
    assert all(item["ambiguous"] for item in only_ambiguous["items"])
    assert only_ambiguous["ambiguous_total"] == page["ambiguous_total"]


def test_run_listing_with_filters(client: TestClient) -> None:
    run_backtest(client)
    run_backtest(client, symbol=OTHER_SYMBOL)

    everything = client.get("/api/runs").json()
    assert len(everything) >= 2
    assert everything[0]["created_at"] >= everything[-1]["created_at"]

    filtered = client.get(f"/api/runs?symbol={OTHER_SYMBOL}").json()
    assert filtered and all(item["symbol"] == OTHER_SYMBOL for item in filtered)
    assert client.get("/api/runs?strategy_id=missing").json() == []
    assert len(client.get("/api/runs?limit=1").json()) == 1


def test_missing_run(client: TestClient) -> None:
    for url in ("/api/runs/abc123", "/api/runs/abc123/equity", "/api/runs/abc123/trades"):
        response = client.get(url)
        assert response.status_code == 404, url
        assert "does not exist" in response.json()["detail"]


def test_malformed_run_id_cannot_escape_the_folder(client: TestClient) -> None:
    response = client.get("/api/runs/..%2F..%2Fetc/equity")
    assert response.status_code in (400, 404)
    assert "Traceback" not in response.text


# -- comparison ----------------------------------------------------------


def test_compare_aligns_and_normalizes(client: TestClient) -> None:
    first = run_backtest(client, initial_equity=1000.0)["run_id"]
    second = run_backtest(client, initial_equity=5000.0)["run_id"]

    response = client.post(
        "/api/runs/compare", json={"run_ids": [first, second], "points": 300}
    )
    assert response.status_code == 200, response.text
    payload = response.json()

    assert [run["run_id"] for run in payload["runs"]] == [first, second]
    assert payload["normalized"] is True
    assert len(payload["series"]) == 300
    assert all(len(point["values"]) == 2 for point in payload["series"])
    # normalized to 100: two different initial equities start from the same point
    assert payload["series"][0]["values"][0] == pytest.approx(100.0, abs=0.5)
    assert payload["series"][0]["values"][1] == pytest.approx(100.0, abs=0.5)

    keys = [row["key"] for row in payload["metrics"]]
    assert "total_return" in keys and "max_drawdown_pct" in keys
    assert "breakeven_win_rate" in keys
    for row in payload["metrics"]:
        assert len(row["values"]) == 2 and len(row["delta"]) == 2
        assert row["delta"][0] in (0.0, None)
        assert row["higher_is_better"] in (True, False, None)


def test_compare_warns_on_different_instruments(client: TestClient) -> None:
    first = run_backtest(client)["run_id"]
    second = run_backtest(client, symbol=OTHER_SYMBOL)["run_id"]
    payload = client.post(
        "/api/runs/compare", json={"run_ids": [first, second]}
    ).json()
    assert any("different instruments" in warning for warning in payload["warnings"])


def test_compare_config_diff(client: TestClient) -> None:
    first = run_backtest(client, initial_equity=1000.0)["run_id"]
    second = run_backtest(client, initial_equity=5000.0)["run_id"]
    payload = client.post(
        "/api/runs/compare", json={"run_ids": [first, second]}
    ).json()

    assert payload["configs_identical"] is False
    rows = {row["key"]: row["values"] for row in payload["config_diff"]}
    assert set(rows) == {"initial_equity"}
    assert rows["initial_equity"] == ["1000.0", "5000.0"]

    same = client.post("/api/runs/compare", json={"run_ids": [first, first]}).json()
    assert same["configs_identical"] is True
    assert same["config_diff"] == []


def test_compare_requires_at_least_two_runs(client: TestClient) -> None:
    run_id = run_backtest(client)["run_id"]
    response = client.post("/api/runs/compare", json={"run_ids": [run_id]})
    assert response.status_code == 422


def test_compare_with_missing_run(client: TestClient) -> None:
    run_id = run_backtest(client)["run_id"]
    response = client.post("/api/runs/compare", json={"run_ids": [run_id, "missing"]})
    assert response.status_code == 404


# -- deletion ------------------------------------------------------------


def test_deletion(client: TestClient) -> None:
    run_id = run_backtest(client, initial_equity=333.0)["run_id"]
    assert client.delete(f"/api/runs/{run_id}").json() == {"run_id": run_id, "deleted": True}
    assert client.get(f"/api/runs/{run_id}").status_code == 404
    assert client.delete(f"/api/runs/{run_id}").status_code == 404


# -- contract ------------------------------------------------------------


def test_openapi_has_explicit_models(client: TestClient) -> None:
    """Without response_model the generated TypeScript types would be useless."""
    schema = client.get("/openapi.json").json()
    for path, method in [
        ("/api/symbols", "get"),
        ("/api/runs", "get"),
        ("/api/runs/{run_id}", "get"),
        ("/api/runs/compare", "post"),
        ("/api/edge", "post"),
    ]:
        content = schema["paths"][path][method]["responses"]["200"]["content"]
        assert "$ref" in json.dumps(content["application/json"]["schema"]), path


def test_cors_only_from_localhost(client: TestClient) -> None:
    allowed = client.get("/api/health", headers={"Origin": "http://localhost:5173"})
    assert allowed.headers.get("access-control-allow-origin") == "http://localhost:5173"

    blocked = client.get("/api/health", headers={"Origin": "https://example.invalid"})
    assert "access-control-allow-origin" not in blocked.headers


# -- validation endpoints ------------------------------------------------


def test_walkforward_rejects_a_run_that_is_not_done(client: TestClient) -> None:
    response = client.post("/api/validation/walkforward", json={"run_id": "does-not-exist"})
    assert response.status_code == 404
    assert "does not exist" in response.json()["detail"]


def test_walkforward_returns_windows_and_a_concatenated_curve(client: TestClient) -> None:
    run = run_backtest(client)
    response = client.post(
        "/api/validation/walkforward",
        json={
            "run_id": run["run_id"],
            "train_days": 1,
            "test_days": 1,
            "min_train_trades": 0,
        },
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["run_id"] == run["run_id"]
    assert payload["windows"]
    assert payload["windows_evaluated"] + payload["windows_skipped"] == len(payload["windows"])
    assert payload["verdict"]
    # without a grid the report must say it optimized nothing
    assert payload["optimized"] is False
    assert payload["grid_size"] == 1
    for point in payload["oos_curve"]:
        assert "t" in point and "equity" in point


def test_walkforward_with_a_grid_reports_parameter_stability(client: TestClient) -> None:
    run = run_backtest(client)
    response = client.post(
        "/api/validation/walkforward",
        json={
            "run_id": run["run_id"],
            "train_days": 1,
            "test_days": 1,
            "min_train_trades": 0,
            "grid": {"exit.stop_loss.value": [100, 150, 200]},
        },
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["optimized"] is True
    assert payload["grid_size"] == 3
    if payload["windows_evaluated"]:
        assert payload["parameter_stability"]
        row = payload["parameter_stability"][0]
        assert row["parameter"] == "exit.stop_loss.value"
        assert 0.0 < row["mode_share"] <= 1.0


def test_walkforward_rejects_a_grid_path_that_does_not_exist(client: TestClient) -> None:
    run = run_backtest(client)
    response = client.post(
        "/api/validation/walkforward",
        json={
            "run_id": run["run_id"],
            "train_days": 1,
            "test_days": 1,
            "grid": {"exit.nonexistent.value": [1, 2]},
        },
    )
    assert response.status_code == 422
    assert "does not exist" in response.json()["detail"]


def test_walkforward_says_when_the_period_is_too_short(client: TestClient) -> None:
    run = run_backtest(client)
    response = client.post(
        "/api/validation/walkforward",
        json={"run_id": run["run_id"], "train_days": 3000, "test_days": 1000},
    )
    assert response.status_code == 422
    assert "not enough" in response.json()["detail"]


def test_permutation_returns_both_nulls_with_their_p_values(client: TestClient) -> None:
    run = run_backtest(client)
    response = client.post(
        "/api/validation/permutation",
        json={"run_id": run["run_id"], "iterations": 10, "block_bars": 128},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    kinds = {test["kind"] for test in payload["tests"]}
    assert kinds == {"random_entries", "permuted_returns"}
    for test in payload["tests"]:
        assert test["iterations"] == 10
        assert test["verdict"]
        for statistic in test["statistics"]:
            assert 0.0 < statistic["p_value"] <= 1.0
            assert 0.0 <= statistic["percentile"] <= 100.0


def test_permutation_can_run_a_single_null(client: TestClient) -> None:
    run = run_backtest(client)
    response = client.post(
        "/api/validation/permutation",
        json={"run_id": run["run_id"], "iterations": 10, "tests": ["random_entries"]},
    )
    assert response.status_code == 200, response.text
    assert [test["kind"] for test in response.json()["tests"]] == ["random_entries"]


def test_permutation_refuses_an_empty_test_list(client: TestClient) -> None:
    run = run_backtest(client)
    response = client.post(
        "/api/validation/permutation",
        json={"run_id": run["run_id"], "iterations": 10, "tests": []},
    )
    assert response.status_code == 422


def test_multiple_testing_counts_the_trials_from_the_store(client: TestClient) -> None:
    run = run_backtest(client)
    response = client.post("/api/validation/multiple-testing", json={"run_id": run["run_id"]})
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["run_id"] == run["run_id"]
    assert payload["trials"] >= 1
    assert payload["verdict"]
    assert "configuration(s) tried" in payload["verdict"]
    # without a grid, PBO must declare itself missing rather than invent one
    assert payload["pbo"]["valid"] is False
    assert "no parameter grid" in payload["pbo"]["reason"]


def test_multiple_testing_computes_pbo_when_a_grid_is_given(client: TestClient) -> None:
    run = run_backtest(client)
    response = client.post(
        "/api/validation/multiple-testing",
        json={
            "run_id": run["run_id"],
            "grid": {"exit.stop_loss.value": [100, 150, 200, 250]},
        },
    )
    assert response.status_code == 200, response.text
    pbo = response.json()["pbo"]
    if pbo["valid"]:
        assert 0.0 <= pbo["pbo"] <= 1.0
        assert pbo["candidates"] == 4
    else:
        assert pbo["reason"]


def test_multiple_testing_on_an_unknown_run_is_a_404(client: TestClient) -> None:
    response = client.post("/api/validation/multiple-testing", json={"run_id": "nope"})
    assert response.status_code == 404


def test_tick_resolve_degrades_gracefully_without_a_terminal(client: TestClient) -> None:
    """No tick source is an answer, not a crash - and never an assumption."""
    run = run_backtest(client)
    response = client.post("/api/validation/tick-resolve", json={"run_id": run["run_id"]})
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["run_id"] == run["run_id"]
    assert payload["verdict"]
    if not payload["available"]:
        assert payload["reason"]
        assert payload["resolved"] == 0


# -- batch ---------------------------------------------------------------


def test_batch_runs_every_symbol_and_reports_consistency(client: TestClient) -> None:
    response = client.post(
        "/api/batch",
        json={
            "strategy_id": "api-test",
            "symbols": [SYMBOL, OTHER_SYMBOL],
            "config": config(),
        },
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["strategy_id"] == "api-test"
    assert len(payload["cells"]) == 2
    assert payload["completed"] + payload["failed"] == 2
    assert payload["consistency"]["metric"] == "mean_r"
    assert payload["consistency"]["verdict"]
    # two instruments cannot support a cross-sectional claim, and it says so
    assert any("no power" in warning for warning in payload["warnings"]) or any(
        "cannot distinguish" in warning for warning in payload["warnings"]
    )


def test_batch_records_a_missing_instrument_as_a_failed_cell(client: TestClient) -> None:
    response = client.post(
        "/api/batch",
        json={
            "strategy_id": "api-test",
            "symbols": [SYMBOL, "NO-SUCH-SYMBOL"],
            "config": config(),
        },
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    failed = [cell for cell in payload["cells"] if cell["status"] != "done"]
    assert len(failed) == 1
    assert failed[0]["symbol"] == "NO-SUCH-SYMBOL"
    assert failed[0]["error"]
    # the batch survives it and says how many cells it lost
    assert payload["failed"] == 1
    assert any("failed" in warning for warning in payload["warnings"])


def test_batch_needs_at_least_one_symbol(client: TestClient) -> None:
    response = client.post(
        "/api/batch",
        json={"strategy_id": "api-test", "symbols": [], "config": config()},
    )
    assert response.status_code == 422


def test_batch_cells_land_in_the_run_store(client: TestClient) -> None:
    response = client.post(
        "/api/batch",
        json={
            "strategy_id": "api-test",
            "symbols": [SYMBOL],
            "config": config(initial_equity=777.0),
        },
    )
    assert response.status_code == 200, response.text
    cell = response.json()["cells"][0]
    assert cell["run_id"]
    detail = client.get(f"/api/runs/{cell['run_id']}")
    assert detail.status_code == 200
    assert detail.json()["config"]["initial_equity"] == 777.0


def test_a_screening_campaign_runs_as_a_polled_job(client: TestClient) -> None:
    """A campaign is minutes of work: it must not block the request."""
    started = client.post(
        "/api/screen",
        json={
            "strategy_ids": ["api-test"],
            "symbols": [SYMBOL, OTHER_SYMBOL],
            "timeframes": ["M1"],
            "config": config(),
            "min_trades": 5,
            "permutation_iterations": 10,
        },
    )
    assert started.status_code == 200, started.text
    job = started.json()
    assert job["status"] == "running"
    assert job["total_cells"] == 2

    for _ in range(120):
        job = client.get(f"/api/screen/{job['job_id']}").json()
        if job["status"] != "running":
            break
        time.sleep(0.5)

    assert job["status"] == "done", job.get("error")
    report = job["report"]
    # both cells count as attempts even if one never reaches a backtest
    assert report["panel"]["attempts"] == 2
    assert len(report["cells"]) == 2
    assert report["verdict"]


def test_an_unknown_screening_job_says_where_the_campaigns_live(
    client: TestClient,
) -> None:
    response = client.get("/api/screen/screen-nope")
    assert response.status_code == 404
    assert "kept in memory" in response.json()["detail"]


def test_a_grid_adds_its_candidates_to_the_trial_family(client: TestClient) -> None:
    """A grid candidate is an attempt too: leaving it out under-deflates."""
    run = run_backtest(client)
    without = client.post(
        "/api/validation/multiple-testing", json={"run_id": run["run_id"]}
    ).json()
    with_grid = client.post(
        "/api/validation/multiple-testing",
        json={
            "run_id": run["run_id"],
            "grid": {"exit.stop_loss.value": [100, 150, 200, 250, 300]},
        },
    ).json()
    assert with_grid["trials"] > without["trials"]
    assert "configuration(s) tried" in with_grid["verdict"]
    # with several trials the deflation finally has a variance to work with
    if with_grid["deflated_sharpe"]["valid"]:
        assert with_grid["deflated_sharpe"]["expected_max_sharpe"] is not None
        assert 0.0 <= with_grid["deflated_sharpe"]["deflated_sharpe"] <= 1.0
