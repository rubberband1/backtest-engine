"""Canonical, comparable snapshot of a backtest run.

The golden test needs a representation of a run that is stable across
serialization round-trips: timestamps as ISO strings, floats rounded to a
fixed precision, NaN mapped to null. Anything that changes here changes what
counts as "the same result", so keep it boring.
"""
from __future__ import annotations

import math
from dataclasses import asdict
from datetime import datetime, timedelta, timezone, tzinfo
from typing import Any

import pandas as pd

from core.data.cache import ParquetCache
from core.data.provider import SymbolSpec
from core.engine.backtester import (
    BacktestConfig,
    BacktestResult,
    TRADE_COLUMNS,
    run_backtest,
)
from core.metrics.performance import PerformanceReport, compute_metrics
from core.runs.store import RunConfig, data_fingerprint, spec_hash
from core.strategy.spec import StrategySpec

FLOAT_DECIMALS = 8


def _canonical_value(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return round(value, FLOAT_DECIMALS)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, timedelta):
        return round(value.total_seconds(), FLOAT_DECIMALS)
    if isinstance(value, dict):
        return {key: _canonical_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    return value


def trades_snapshot(trades: pd.DataFrame) -> list[dict[str, Any]]:
    """Full trade list in canonical form, one dict per trade."""
    rows: list[dict[str, Any]] = []
    for _, row in trades.iterrows():
        entry: dict[str, Any] = {}
        for column in TRADE_COLUMNS:
            value = row[column]
            if column in ("ambiguous", "crossed_gap"):
                entry[column] = bool(value)
            elif column in ("direction", "bars_held", "session_bars_held"):
                entry[column] = int(value)
            else:
                entry[column] = _canonical_value(
                    float(value) if isinstance(value, (int, float)) else value
                )
        rows.append(entry)
    return rows


def report_snapshot(report: PerformanceReport) -> dict[str, Any]:
    """PerformanceReport in canonical form. No rendered text: numbers only."""
    payload = asdict(report)
    return {key: _canonical_value(value) for key, value in payload.items()}


def result_snapshot(
    result: BacktestResult, strategy_report: PerformanceReport
) -> dict[str, Any]:
    return {
        "trades": trades_snapshot(result.trades),
        "metrics": report_snapshot(strategy_report),
        "execution": {
            "signals_long": int(result.signals.counts["long"]),
            "signals_short": int(result.signals.counts["short"]),
            "trades": int(len(result.trades)),
            "ambiguous_trades": result.ambiguous_trades,
            "gap_crossing_trades": result.gap_crossing_trades,
            "blocked": {key: int(count) for key, count in sorted(result.blocked.items())},
        },
    }


def diff_snapshots(expected: Any, actual: Any, path: str = "") -> list[str]:
    """Every leaf-level difference between two canonical snapshots."""
    diffs: list[str] = []
    if isinstance(expected, dict) and isinstance(actual, dict):
        for key in sorted(set(expected) | set(actual)):
            where = f"{path}.{key}" if path else str(key)
            if key not in expected:
                diffs.append(f"{where}: unexpected new field = {actual[key]!r}")
            elif key not in actual:
                diffs.append(f"{where}: missing (reference has {expected[key]!r})")
            else:
                diffs.extend(diff_snapshots(expected[key], actual[key], where))
        return diffs
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            diffs.append(f"{path}: length {len(expected)} != {len(actual)}")
        for index, (left, right) in enumerate(zip(expected, actual)):
            diffs.extend(diff_snapshots(left, right, f"{path}[{index}]"))
        return diffs
    if expected != actual:
        diffs.append(f"{path}: reference {expected!r} != actual {actual!r}")
    return diffs


def load_golden_bars(
    cache: ParquetCache, config: RunConfig
) -> pd.DataFrame:
    """Bars for a golden window, sliced exactly like core.runs.runner.load_bars."""
    from core.runs.runner import load_bars

    return load_bars(cache, config.symbol, config.tf, config.start, config.end)


def execute_golden(
    spec: StrategySpec,
    config: RunConfig,
    symbol_spec: SymbolSpec,
    server_tz: tzinfo,
    bars: pd.DataFrame,
) -> dict[str, Any]:
    """Run the golden window and return the full reference payload."""
    result = run_backtest(
        spec,
        bars,
        symbol_spec,
        server_tz,
        BacktestConfig(
            initial_equity=config.initial_equity,
            costs=config.cost_model(),
            session_threshold=config.session_threshold,
        ),
    )
    strategy_report = compute_metrics(
        result.trades, result.equity, result.timeframe, config.initial_equity, spec.id
    )
    return {
        "spec_hash": spec_hash(spec),
        "config": config.to_dict(),
        # The symbol spec and server timezone are inputs to the result
        # (tick_value moves with FX rates), so the reference pins them and the
        # golden test replays the pinned values instead of asking the terminal.
        "symbol_spec": asdict(symbol_spec),
        "server_timezone": str(server_tz),
        "data": {
            "fingerprint": data_fingerprint(bars),
            "bars": int(len(bars)),
            "start": bars.index[0].isoformat(),
            "end": bars.index[-1].isoformat(),
        },
        "snapshot": result_snapshot(result, strategy_report),
    }


def utc_datetime(text: str) -> datetime:
    moment = datetime.fromisoformat(text)
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)
