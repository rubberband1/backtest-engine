"""Batch runner driven by a YAML file.

    python -m scripts.run_batch batch.yaml
    python -m scripts.run_batch batch.yaml --json report.json

The YAML holds what changes between batches and nothing else:

    strategy: strategies/rsi-wick-baseline.json
    symbols: [XAUUSD.r, XAGUSD.r, EURUSD.r]
    timeframe: M1
    initial_equity: 100
    spread_mode: per_bar
    commission_per_lot_per_side: 0
    consistency_metric: mean_r
    periods:
      - {start: 2025-04-02, end: 2025-08-01}
      - {start: 2025-08-01, end: 2025-12-31}
    grid:
      exit.stop_loss.value: [100, 150, 200]

`periods` and `grid` are optional; without them the batch runs one full-history
cell per symbol.
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from core.batch.runner import Period, run_batch
from core.runs.store import RunConfig

logger = logging.getLogger("batch")

FIELDS: tuple[str, ...] = (
    "timeframe",
    "initial_equity",
    "spread_mode",
    "spread_value",
    "commission_per_lot_per_side",
    "swap_mode",
    "session_threshold",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="YAML file describing the batch")
    parser.add_argument("--json", type=Path, default=None, help="write the report here")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--cache-dir", type=Path, default=Path("data_cache"))
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    return parser.parse_args()


def as_datetime(value: Any) -> datetime | None:
    """YAML gives back a date, a datetime or a string. All three arrive as UTC."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str):
        moment = datetime.fromisoformat(value)
        return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)
    # a bare `date`
    return datetime(value.year, value.month, value.day, tzinfo=timezone.utc)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(name)s | %(message)s")
    args = parse_args()
    payload = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}

    symbols = payload.get("symbols") or []
    if not symbols:
        logger.error("%s: 'symbols' is empty, there is nothing to run", args.config)
        return 1
    strategy = payload.get("strategy")
    if not strategy:
        logger.error("%s: 'strategy' is required (path to the spec JSON)", args.config)
        return 1

    config_fields = {key: payload[key] for key in FIELDS if key in payload}
    base_config = RunConfig(symbol=symbols[0], **config_fields)
    periods = [
        Period(start=as_datetime(item.get("start")), end=as_datetime(item.get("end")))
        for item in (payload.get("periods") or [])
    ]

    report = run_batch(
        spec_path=strategy,
        symbols=symbols,
        base_config=base_config,
        periods=periods or None,
        grid=payload.get("grid") or None,
        cache_dir=args.cache_dir,
        runs_dir=args.runs_dir,
        max_workers=args.workers,
        consistency_metric=payload.get("consistency_metric", "mean_r"),
    )

    print(render(report))
    if args.json:
        args.json.write_text(
            json.dumps(report.as_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        logger.info("report written to %s", args.json)
    return 0


def render(report: Any) -> str:
    """The symbol x metric table, plus the line that actually matters."""
    lines = [
        "",
        f"=== BATCH {report.strategy_id} ===",
        f"symbols   : {len(report.symbols)}  cells: {len(report.cells)}  "
        f"completed: {report.completed}  failed: {report.failed}",
        f"elapsed   : {report.elapsed_seconds:.1f}s",
        "",
        f"{'symbol':<14}{'trades':>8}{'net pnl':>11}{'expect.':>10}"
        f"{'mean R':>9}{'+/- se':>9}{'sharpe':>9}{'win':>8}{'maxDD':>9}",
        "-" * 87,
    ]
    for cell in sorted(report.cells, key=lambda item: item.symbol):
        if cell.status != "done":
            lines.append(f"{cell.symbol:<14}{'error':>8}  {cell.error or ''}"[:87])
            continue
        lines.append(
            f"{cell.symbol:<14}"
            f"{cell.trades:>8}"
            f"{_fmt(cell.net_pnl, 2):>11}"
            f"{_fmt(cell.expectancy, 4):>10}"
            f"{_fmt(cell.mean_r, 3):>9}"
            f"{_fmt(cell.mean_r_stderr, 3):>9}"
            f"{_fmt(cell.sharpe, 2):>9}"
            f"{_fmt(cell.win_rate, 3):>8}"
            f"{_fmt(cell.max_drawdown_pct, 3):>9}"
        )

    measure = report.consistency
    lines += [
        "",
        "--- CROSS-SECTIONAL CONSISTENCY ---",
        f"metric        : {measure.metric} averaged per instrument",
        f"instruments   : {measure.observations} "
        f"({measure.positive} positive, {measure.negative} negative)",
        f"mean          : {_fmt(measure.mean, 4)}  median {_fmt(measure.median, 4)}",
        f"spread        : min {_fmt(measure.minimum, 4)}  max {_fmt(measure.maximum, 4)}  "
        f"std {_fmt(measure.std, 4)}  stderr {_fmt(measure.stderr, 4)}",
        f"sign test p   : {_fmt(measure.sign_p_value, 4)}",
        "",
        measure.verdict,
    ]
    for warning in report.warnings:
        lines.append(f"WARNING: {warning}")
    return "\n".join(lines)


def _fmt(value: float | None, digits: int) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


if __name__ == "__main__":
    raise SystemExit(main())
