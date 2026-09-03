"""Screening campaign driven by a YAML file.

    python -m scripts.run_screen screen.yaml --json report.json

The YAML holds what changes between campaigns and nothing else:

    strategies: [strategies/ma-crossover.json, strategies/macd-signal.json]
    symbols: [XAUUSD.r, EURUSD.r]
    timeframes: [M5, M15, H1]
    start: 2025-01-01
    end: 2026-01-01
    initial_equity: 100
    spread_mode: per_bar
    min_trades: 30
    permutation_iterations: 200

`strategies` also accepts a directory or a glob, in which case every spec in
it is screened.
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from core.research.screen import ScreenReport, as_frame, run_screen
from core.runs.store import RunConfig

logger = logging.getLogger("screen")

CONFIG_FIELDS: tuple[str, ...] = (
    "initial_equity",
    "spread_mode",
    "spread_value",
    "commission_per_lot_per_side",
    "swap_mode",
    "session_threshold",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="YAML file describing the campaign")
    parser.add_argument("--json", type=Path, default=None, help="write the report here")
    parser.add_argument("--csv", type=Path, default=None, help="write the table here")
    parser.add_argument("--cache-dir", type=Path, default=Path("data_cache"))
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def resolve_strategies(entries: Any) -> list[Path]:
    """A list of files, a directory, or a glob - all end up as a sorted list."""
    if isinstance(entries, str):
        entries = [entries]
    paths: list[Path] = []
    for entry in entries:
        candidate = Path(entry)
        if candidate.is_dir():
            paths.extend(sorted(candidate.glob("*.json")))
        elif any(char in str(entry) for char in "*?["):
            paths.extend(sorted(Path().glob(str(entry))))
        else:
            paths.append(candidate)
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise SystemExit(f"strategy files not found: {', '.join(map(str, missing))}")
    return paths


def moment(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    return datetime.fromisoformat(str(value)).replace(tzinfo=timezone.utc)


def render(report: ScreenReport) -> str:
    """The table and the trial panel, in the terminal."""
    frame = as_frame(report)
    lines = [
        "",
        f"=== SCREENING CAMPAIGN ({report.engine_version}) ===",
        f"strategies : {len(report.strategies)}  symbols: {len(report.symbols)}  "
        f"timeframes: {len(report.timeframes)}",
        f"period     : {report.period_start} -> {report.period_end}",
        f"elapsed    : {report.elapsed_seconds:.1f}s",
        "",
    ]

    header = (
        f"{'strategy':<24} {'symbol':<9} {'tf':<4} {'stage':<12} {'signals':>7} "
        f"{'trades':>7} {'net':>9} {'sharpe/t':>9} {'meanR':>7} {'amb%':>6} {'perm p':>7}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for _, row in frame.iterrows():
        def num(value: Any, fmt: str) -> str:
            return format(value, fmt) if value is not None and value == value else "-"

        lines.append(
            f"{row['strategy_id']:<24} {row['symbol']:<9} {row['timeframe']:<4} "
            f"{row['stage_reached']:<12} {num(row['gate_signals'], '7.0f')} "
            f"{num(row['trades'], '7.0f')} {num(row['net_pnl'], '9.2f')} "
            f"{num(row['sharpe_per_trade'], '9.4f')} {num(row['mean_r'], '7.3f')} "
            f"{num(row['ambiguous_share'], '6.1%')} "
            f"{num(row['permutation_p_value'], '7.4f')}"
        )

    panel = report.panel
    lines += [
        "",
        "--- MULTIPLE TESTING OVER THE WHOLE CAMPAIGN ---",
        f"attempts             : {panel.attempts}",
        f"reached a backtest   : {panel.cells_backtested}",
        f"reached a permutation: {panel.cells_permuted}",
        f"Sharpe spread (var)  : "
        f"{panel.variance_across_trials:.6f}" if panel.variance_across_trials
        else "Sharpe spread (var)  : -",
        f"free Sharpe at N={panel.attempts:<4}: "
        f"{panel.expected_max_sharpe:+.4f}" if panel.expected_max_sharpe is not None
        else "free Sharpe          : -",
        f"required Sharpe/trade: "
        f"{panel.required_sharpe_per_trade:+.4f} (at {panel.confidence:.0%} confidence)"
        if panel.required_sharpe_per_trade is not None
        else "required Sharpe/trade: -",
        f"best observed        : {panel.best_strategy or '-'} "
        f"({panel.best_sharpe_per_trade:+.4f})"
        if panel.best_sharpe_per_trade is not None
        else "best observed        : -",
        f"Bonferroni threshold : "
        f"{panel.bonferroni_threshold:.5f}" if panel.bonferroni_threshold
        else "Bonferroni threshold : -",
        f"survivors            : {panel.survivors_after_correction}",
        "",
        panel.verdict,
        "",
    ]
    for assumption in panel.assumptions:
        lines.append(f"  - {assumption}")
    for warning in report.warnings:
        lines.append(f"  ! {warning}")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(levelname)-7s %(name)s | %(message)s",
    )
    payload = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    strategies = resolve_strategies(payload["strategies"])
    config_kwargs = {
        field: payload[field] for field in CONFIG_FIELDS if field in payload
    }
    base_config = RunConfig(
        symbol=payload["symbols"][0],
        timeframe=payload["timeframes"][0],
        start=moment(payload.get("start")),
        end=moment(payload.get("end")),
        **config_kwargs,
    )

    report = run_screen(
        strategies=strategies,
        symbols=payload["symbols"],
        timeframes=payload["timeframes"],
        base_config=base_config,
        cache_dir=args.cache_dir,
        runs_dir=args.runs_dir,
        min_trades=int(payload.get("min_trades", 30)),
        permutation_iterations=int(payload.get("permutation_iterations", 200)),
    )

    print(render(report))

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(report.as_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        logger.info("report written to %s", args.json)
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        as_frame(report).to_csv(args.csv, index=False)
        logger.info("table written to %s", args.csv)


if __name__ == "__main__":
    main()
