"""Re-run a campaign from its manifest and report what moved.

    python -m scripts.verify_campaign campaign.yaml --report old.json --manifest old.manifest.json

The claim under test is narrow and worth stating exactly: *given the same
instrument specs, the same settings and the same bars, this engine produces
the same campaign.* Everything a manifest freezes is handed back to the
engine, so a cell that differs differs because of the engine or because the
broker rewrote the bars underneath it - and the run's data fingerprint tells
those two apart.

Without this, "reproducible" was a claim nobody could check. Re-running a
campaign always produced small differences, and they were unattributable:
`tick_value` follows an FX rate, so every money column drifts between two
executions for reasons that have nothing to do with the code. That drift is
exactly what the manifest suppresses, which is why a difference that survives
it is worth looking at.

Exit code 0 when every compared cell matches, 1 when any does not.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import yaml  # noqa: E402

from core.research.manifest import CampaignManifest, drifted  # noqa: E402
from core.research.screen import ScreenReport, run_screen  # noqa: E402
from core.runs.store import RunConfig  # noqa: E402

logger = logging.getLogger("verify-campaign")

# What is compared, cell by cell. Metrics, not prose: a verdict string changes
# when the wording changes, which is not a regression.
COMPARED: tuple[str, ...] = (
    "stage_reached",
    "status",
    "counts_as_attempt",
    "tradable",
    "gate_passed",
    "gate_signals",
    "run_id",
    "bars",
    "trades",
    "net_pnl",
    "final_equity",
    "sharpe_per_trade",
    "mean_r",
    "win_rate",
    "max_drawdown_pct",
    "p_value",
    "ambiguous_share",
    "band_money",
    "signals",
    "gate_rejected",
    "spread_charged_median",
    "spread_measured_share",
    "permutation_p_value",
)

PANEL_COMPARED: tuple[str, ...] = (
    "attempts",
    "cells_backtested",
    "sharpes_observed",
    "variance_across_trials",
    "expected_max_sharpe",
    "required_sharpe_per_trade",
    "best_strategy",
    "best_sharpe_per_trade",
    "best_clears_required",
    "survivors_after_correction",
)

# Permutation p-values come from a seeded shuffle, but the seed is per cell
# and the iteration count is pinned by the manifest, so they are compared
# exactly like everything else. A tolerance exists only for floating-point
# reassociation, not for genuine variation.
TOLERANCE = 1e-9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="the campaign's YAML file")
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="the manifest the original campaign wrote",
    )
    parser.add_argument(
        "--report",
        type=Path,
        required=True,
        help="the original campaign's JSON report, to compare against",
    )
    parser.add_argument("--cache-dir", type=Path, default=REPO_ROOT / "data_cache")
    parser.add_argument("--runs-dir", type=Path, default=REPO_ROOT / "runs")
    parser.add_argument(
        "--json", type=Path, default=None, help="write the fresh report here"
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def cell_key(cell: dict) -> str:
    return f"{cell['strategy_id']}/{cell['symbol']}/{cell['timeframe']}"


def differs(left: object, right: object) -> bool:
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        if left is None or right is None:
            return left is not right
        return abs(float(left) - float(right)) > TOLERANCE
    return left != right


def compare_cells(before: list[dict], after: list[dict]) -> list[str]:
    left = {cell_key(c): c for c in before}
    right = {cell_key(c): c for c in after}

    problems: list[str] = []
    for key in sorted(set(left) - set(right)):
        problems.append(f"{key}: in the original report and not in the re-run")
    for key in sorted(set(right) - set(left)):
        problems.append(f"{key}: in the re-run and not in the original report")

    for key in sorted(set(left) & set(right)):
        for field in COMPARED:
            a, b = left[key].get(field), right[key].get(field)
            if differs(a, b):
                problems.append(f"{key}: {field} {a!r} -> {b!r}")
    return problems


def compare_panel(before: dict, after: dict) -> list[str]:
    return [
        f"panel: {field} {before.get(field)!r} -> {after.get(field)!r}"
        for field in PANEL_COMPARED
        if differs(before.get(field), after.get(field))
    ]


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=args.log_level.upper(), format="%(levelname)-7s %(name)s | %(message)s"
    )

    manifest = CampaignManifest.read(args.manifest)
    original = json.loads(args.report.read_text(encoding="utf-8"))
    payload = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    print(manifest.summary())
    print()

    from core.data.cache import ParquetCache
    from core.runs.runner import SymbolResolver

    moved = drifted(manifest, SymbolResolver(ParquetCache(args.cache_dir)))
    if moved:
        print("The broker's contracts have moved since this manifest was frozen:")
        for symbol, change in moved.items():
            print(f"  {symbol}: {change}")
        print(
            "  The re-run below uses the frozen values, so this drift is\n"
            "  suppressed rather than measured. That is the point of it."
        )
        print()

    from scripts.run_screen import resolve_strategies

    report: ScreenReport = run_screen(
        strategies=resolve_strategies(payload["strategies"]),
        symbols=manifest.symbols,
        timeframes=manifest.timeframes,
        base_config=RunConfig.from_dict(manifest.config),
        cache_dir=args.cache_dir,
        runs_dir=args.runs_dir,
        manifest=manifest,
    )
    fresh = report.as_dict()

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(fresh, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    problems = compare_cells(
        original.get("cells") or [], fresh.get("cells") or []
    ) + compare_panel(original.get("panel") or {}, fresh.get("panel") or {})

    print("=== VERIFICATION ===")
    print(f"  original : {args.report}")
    print(f"  cells    : {len(original.get('cells') or [])} then, "
          f"{len(fresh.get('cells') or [])} now")
    print(f"  engine   : {original.get('engine_version')} then, "
          f"{fresh.get('engine_version')} now")

    if not problems:
        print(
            "\n  Every compared field matches. Given the same frozen inputs "
            "this\n  engine reproduces this campaign."
        )
        return 0

    print(f"\n  {len(problems)} difference(s). With the inputs frozen, each one is")
    print("  the engine or the bars, not the broker:")
    for line in problems[:60]:
        print(f"    {line}")
    if len(problems) > 60:
        print(f"    ... and {len(problems) - 60} more")
    if original.get("engine_version") != fresh.get("engine_version"):
        print(
            f"\n  The engine changed from {original.get('engine_version')} to "
            f"{fresh.get('engine_version')}. Differences are expected; what "
            f"matters\n  is whether they are the ones the changelog declares."
        )
    return 1


if __name__ == "__main__":
    sys.exit(main())
