"""Persistence of backtest runs on disk.

A run is a folder, not a database row: `runs/<run_id>/` with the exact spec
used, the configuration, the trades, the equity curve, the metrics and the
metadata. It opens with a text editor and reads with pandas, without starting
anything.

`run_id` is **deterministic**: a hash of spec + configuration + data
fingerprint. Two identical runs share the same id, so relaunching the same
thing neither recomputes nor duplicates. If anything changes — a spec
parameter, the spread, one bar of data — the id changes, and the old run
stays there documenting how things were before.
"""
from __future__ import annotations

import hashlib
import json
import logging
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal

import numpy as np
import pandas as pd

from core.data.provider import SYMBOL_SPEC_COST_FIELDS, SymbolSpec, SymbolSpecSnapshot, Timeframe
from core.engine.backtester import BacktestResult
from core.engine.costs import CommissionModel, CostModel, SpreadPolicy, SwapModel
from core.metrics.breakeven import breakeven_from_trades
from core.metrics.performance import PerformanceReport
from core.serialization import json_safe
from core.strategy.exits import has_variable_exits
from core.strategy.spec import StrategySpec
from core.version import ENGINE_VERSION

logger = logging.getLogger(__name__)

RunStatus = Literal["running", "done", "error"]

SPEC_FILE = "spec.json"
CONFIG_FILE = "config.json"
META_FILE = "meta.json"
METRICS_FILE = "metrics.json"
TRADES_FILE = "trades.parquet"
EQUITY_FILE = "equity.parquet"
SYMBOL_SPEC_FILE = "symbol_spec.json"


class RunNotFound(KeyError):
    """No run with that id."""


@dataclass(frozen=True)
class RunConfig:
    """Everything that is not strategy but changes the result."""

    symbol: str
    timeframe: str
    start: datetime | None = None
    end: datetime | None = None
    initial_equity: float = 100.0
    spread_mode: str = "per_bar"
    spread_value: float | None = None
    commission_per_lot_per_side: float = 0.0
    swap_mode: str = "points"
    session_threshold: float = 0.5

    def __post_init__(self) -> None:
        # `100` and `100.0` are the same configuration but serialize
        # differently, and run_id is a hash of that serialization. Without
        # this, the same batch launched from the YAML runner and from the API
        # lands in two different folders and looks like two experiments.
        for name in (
            "initial_equity",
            "spread_value",
            "commission_per_lot_per_side",
            "session_threshold",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, float(value))

    def to_dict(self) -> dict[str, Any]:
        return json_safe(asdict(self))

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RunConfig":
        data = dict(payload)
        for field_name in ("start", "end"):
            value = data.get(field_name)
            data[field_name] = datetime.fromisoformat(value) if value else None
        return cls(**data)

    @property
    def tf(self) -> Timeframe:
        return Timeframe.parse(self.timeframe)

    def cost_model(self) -> CostModel:
        return CostModel(
            spread=SpreadPolicy(mode=self.spread_mode, value=self.spread_value),
            commission=CommissionModel(self.commission_per_lot_per_side),
            swap=SwapModel(mode=self.swap_mode),
        )


def _canonical(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def spec_hash(spec: StrategySpec) -> str:
    return hashlib.sha256(
        _canonical(spec.model_dump(mode="json")).encode("utf-8")
    ).hexdigest()


def symbol_spec_cost_hash(symbol_spec: SymbolSpec) -> str:
    """Hash of only the fields that change what a trade costs.

    Digits, currency_profit, trade_mode and name are descriptive and left
    out: including them would invalidate every run_id on a broker relabeling
    that changes nothing about the result.
    """
    payload = {field: getattr(symbol_spec, field) for field in SYMBOL_SPEC_COST_FIELDS}
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


def data_fingerprint(bars: pd.DataFrame) -> str:
    """Exact fingerprint of the bars used.

    The time range is not enough: if the broker rewrites a candle or the
    cache is re-downloaded, the result changes while the dates stay the same.
    Hashing the bytes makes different data a different run.
    """
    if bars.empty:
        return hashlib.sha256(b"empty").hexdigest()[:32]
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(bars.index.asi8).tobytes())
    for column in ("open", "high", "low", "close", "spread"):
        if column in bars.columns:
            digest.update(np.ascontiguousarray(bars[column].to_numpy("float64")).tobytes())
    return digest.hexdigest()[:32]


def compute_run_id(
    spec: StrategySpec, config: RunConfig, fingerprint: str, symbol_spec: SymbolSpec
) -> str:
    payload = {
        "engine": ENGINE_VERSION,
        "spec": spec.model_dump(mode="json"),
        "config": config.to_dict(),
        "data": fingerprint,
        # cost-relevant SymbolSpec fields: the broker can change swap rates or
        # tick_value between two otherwise identical runs, and without this the
        # same run_id would silently mean two different results.
        "symbol_spec": symbol_spec_cost_hash(symbol_spec),
    }
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()[:16]


@dataclass
class RunMeta:
    run_id: str
    status: RunStatus
    created_at: datetime
    engine_version: str
    spec_id: str
    spec_hash: str
    data_hash: str
    symbol_spec_hash: str | None = None
    bars: int = 0
    data_start: datetime | None = None
    data_end: datetime | None = None
    finished_at: datetime | None = None
    duration_seconds: float | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return json_safe(asdict(self))

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RunMeta":
        data = dict(payload)
        for name in ("created_at", "finished_at", "data_start", "data_end"):
            value = data.get(name)
            data[name] = datetime.fromisoformat(value) if value else None
        return cls(**data)


@dataclass
class RunSummary:
    """A listing row: just enough to pick a run, without loading it."""

    run_id: str
    status: RunStatus
    created_at: datetime
    strategy_id: str
    strategy_name: str
    symbol: str
    timeframe: str
    data_start: datetime | None
    data_end: datetime | None
    bars: int
    duration_seconds: float | None
    error: str | None
    trades: int | None = None
    final_equity: float | None = None
    total_return: float | None = None
    sharpe: float | None = None
    max_drawdown_pct: float | None = None
    profit_factor: float | None = None
    win_rate: float | None = None
    ambiguous_trades: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return json_safe(asdict(self))


@dataclass
class RunRecord:
    """A complete run, with the data loaded on demand."""

    run_id: str
    path: Path
    meta: RunMeta
    spec: StrategySpec
    config: RunConfig
    metrics: dict[str, Any] = field(default_factory=dict)
    symbol_spec: SymbolSpecSnapshot | None = None

    @property
    def symbol_spec_registered(self) -> bool:
        """False for a run persisted before the symbol_spec snapshot existed.

        Such a run must not be assumed to have used the SymbolSpec current on
        disk today: the broker may have changed swap rates or tick_value since
        then, and the run predates the mechanism that would have caught it.
        """
        return self.symbol_spec is not None

    def trades(self) -> pd.DataFrame:
        target = self.path / TRADES_FILE
        if not target.exists():
            return pd.DataFrame()
        return pd.read_parquet(target)

    def equity(self) -> pd.Series:
        target = self.path / EQUITY_FILE
        if not target.exists():
            return pd.Series(dtype="float64")
        frame = pd.read_parquet(target)
        series = frame["equity"]
        series.index = pd.DatetimeIndex(frame.index).tz_convert("UTC")
        return series


class RunStore:
    """Run archive on the filesystem."""

    def __init__(self, root: Path | str = Path("runs")) -> None:
        self.root = Path(root)

    # -- paths -----------------------------------------------------------

    def path_for(self, run_id: str) -> Path:
        if not run_id or "/" in run_id or "\\" in run_id or ".." in run_id:
            raise ValueError(f"invalid run_id: {run_id!r}")
        return self.root / run_id

    def exists(self, run_id: str) -> bool:
        return (self.path_for(run_id) / META_FILE).exists()

    # -- writing ---------------------------------------------------------

    def begin_run(
        self,
        run_id: str,
        spec: StrategySpec,
        config: RunConfig,
        fingerprint: str,
        bars: int,
        data_start: datetime | None,
        data_end: datetime | None,
        symbol_spec: SymbolSpecSnapshot,
    ) -> RunMeta:
        """Creates the folder and marks the run as running."""
        path = self.path_for(run_id)
        path.mkdir(parents=True, exist_ok=True)
        (path / SPEC_FILE).write_text(spec.to_json() + "\n", encoding="utf-8")
        (path / CONFIG_FILE).write_text(
            json.dumps(config.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (path / SYMBOL_SPEC_FILE).write_text(
            json.dumps(symbol_spec.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        meta = RunMeta(
            run_id=run_id,
            status="running",
            created_at=datetime.now(timezone.utc),
            engine_version=ENGINE_VERSION,
            spec_id=spec.id,
            spec_hash=spec_hash(spec),
            data_hash=fingerprint,
            symbol_spec_hash=symbol_spec_cost_hash(symbol_spec.spec),
            bars=bars,
            data_start=data_start,
            data_end=data_end,
        )
        self._write_meta(meta)
        return meta

    def finish_run(
        self,
        run_id: str,
        result: BacktestResult,
        strategy_report: PerformanceReport,
        benchmark_report: PerformanceReport | None = None,
        spec: StrategySpec | None = None,
    ) -> RunMeta:
        path = self.path_for(run_id)
        meta = self.load_meta(run_id)

        result.trades.to_parquet(path / TRADES_FILE, engine="pyarrow", compression="snappy")
        equity = result.equity.to_frame(name="equity")
        equity.to_parquet(path / EQUITY_FILE, engine="pyarrow", compression="snappy")

        metrics = {
            "strategy": report_to_dict(strategy_report),
            "benchmark": report_to_dict(benchmark_report) if benchmark_report else None,
            "breakeven": breakeven_from_trades(
                result.trades,
                variable_exits=has_variable_exits(spec.exit) if spec else False,
            ).as_dict(),
            "execution": {
                "signals_long": result.signals.counts["long"],
                "signals_short": result.signals.counts["short"],
                "trades": int(len(result.trades)),
                "ambiguous_trades": result.ambiguous_trades,
                "gap_crossing_trades": result.gap_crossing_trades,
                "blocked": dict(result.blocked),
                "exit_reasons": (
                    result.trades["exit_reason"].value_counts().to_dict()
                    if len(result.trades)
                    else {}
                ),
            },
        }
        (path / METRICS_FILE).write_text(
            json.dumps(json_safe(metrics), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        meta.status = "done"
        meta.finished_at = datetime.now(timezone.utc)
        meta.duration_seconds = (meta.finished_at - meta.created_at).total_seconds()
        self._write_meta(meta)
        logger.info("run %s saved to %s", run_id, path)
        return meta

    def fail_run(self, run_id: str, message: str) -> RunMeta:
        meta = self.load_meta(run_id)
        meta.status = "error"
        meta.error = message
        meta.finished_at = datetime.now(timezone.utc)
        meta.duration_seconds = (meta.finished_at - meta.created_at).total_seconds()
        self._write_meta(meta)
        logger.error("run %s failed: %s", run_id, message)
        return meta

    def _write_meta(self, meta: RunMeta) -> None:
        path = self.path_for(meta.run_id)
        path.mkdir(parents=True, exist_ok=True)
        (path / META_FILE).write_text(
            json.dumps(meta.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    # -- reading ---------------------------------------------------------

    def load_meta(self, run_id: str) -> RunMeta:
        target = self.path_for(run_id) / META_FILE
        if not target.exists():
            raise RunNotFound(run_id)
        return RunMeta.from_dict(json.loads(target.read_text(encoding="utf-8")))

    def load_run(self, run_id: str) -> RunRecord:
        path = self.path_for(run_id)
        meta = self.load_meta(run_id)
        spec = StrategySpec.from_json(path / SPEC_FILE)
        config = RunConfig.from_dict(json.loads((path / CONFIG_FILE).read_text(encoding="utf-8")))
        metrics_path = path / METRICS_FILE
        metrics = (
            json.loads(metrics_path.read_text(encoding="utf-8"))
            if metrics_path.exists()
            else {}
        )
        symbol_spec_path = path / SYMBOL_SPEC_FILE
        symbol_spec = (
            SymbolSpecSnapshot.from_dict(json.loads(symbol_spec_path.read_text(encoding="utf-8")))
            if symbol_spec_path.exists()
            else None
        )
        return RunRecord(run_id=run_id, path=path, meta=meta, spec=spec, config=config,
                         metrics=metrics, symbol_spec=symbol_spec)

    def list_runs(
        self,
        symbol: str | None = None,
        strategy_id: str | None = None,
        status: RunStatus | None = None,
        limit: int | None = None,
    ) -> list[RunSummary]:
        """Listing ordered newest first, with the summary metrics."""
        if not self.root.exists():
            return []
        summaries: list[RunSummary] = []
        for entry in self.root.iterdir():
            if not (entry / META_FILE).exists():
                continue
            try:
                summary = self._summarize(entry.name)
            except Exception as exc:  # a corrupted folder must not hide the others
                logger.warning("run %s unreadable: %s", entry.name, exc)
                continue
            if symbol and summary.symbol != symbol:
                continue
            if strategy_id and summary.strategy_id != strategy_id:
                continue
            if status and summary.status != status:
                continue
            summaries.append(summary)

        summaries.sort(key=lambda s: s.created_at, reverse=True)
        return summaries[:limit] if limit else summaries

    def _summarize(self, run_id: str) -> RunSummary:
        record = self.load_run(run_id)
        strategy = (record.metrics or {}).get("strategy") or {}
        execution = (record.metrics or {}).get("execution") or {}
        return RunSummary(
            run_id=run_id,
            status=record.meta.status,
            created_at=record.meta.created_at,
            strategy_id=record.spec.id,
            strategy_name=record.spec.name,
            symbol=record.config.symbol,
            timeframe=record.config.timeframe,
            data_start=record.meta.data_start,
            data_end=record.meta.data_end,
            bars=record.meta.bars,
            duration_seconds=record.meta.duration_seconds,
            error=record.meta.error,
            trades=execution.get("trades"),
            final_equity=strategy.get("final_equity"),
            total_return=strategy.get("total_return"),
            sharpe=strategy.get("sharpe"),
            max_drawdown_pct=strategy.get("max_drawdown_pct"),
            profit_factor=strategy.get("profit_factor"),
            win_rate=strategy.get("win_rate"),
            ambiguous_trades=execution.get("ambiguous_trades"),
        )

    # -- deletion --------------------------------------------------------

    def delete_run(self, run_id: str) -> bool:
        path = self.path_for(run_id)
        if not path.exists():
            return False
        shutil.rmtree(path)
        logger.info("run %s deleted", run_id)
        return True

    def delete_all(self, run_ids: Iterable[str]) -> int:
        return sum(1 for run_id in run_ids if self.delete_run(run_id))


def report_to_dict(report: PerformanceReport) -> dict[str, Any]:
    """PerformanceReport -> JSON dictionary, with the text already rendered."""
    payload = json_safe(asdict(report))
    payload["text"] = report.as_text()
    payload["drawdown_unsustainable"] = bool(
        report.max_drawdown_pct > 1.0 or report.final_equity <= 0
    )
    return payload
