"""Regenerate the golden reference for the baseline strategy.

    python -m scripts.update_golden --update-golden --note "reason for the change"

Refuses to run without both flags: overwriting the golden reference is a
declaration that the engine's results changed on purpose, and the reason goes
into ENGINE_CHANGELOG.md where the next person can read it.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from core.data.cache import ParquetCache
from core.runs.runner import SymbolResolver
from core.runs.snapshot import execute_golden, load_golden_bars, utc_datetime
from core.runs.store import RunConfig
from core.strategy.spec import StrategySpec
from core.version import ENGINE_VERSION

REFERENCE_FILE = REPO_ROOT / "tests" / "golden" / "reference" / "rsi-wick-baseline.json"
CHANGELOG_FILE = REPO_ROOT / "ENGINE_CHANGELOG.md"

# The frozen window: 2025-04-02 through 2025-12-30 inclusive (end bound is
# exclusive, like everywhere else in the engine).
GOLDEN_CONFIG = RunConfig(
    symbol="XAUUSD.r",
    timeframe="M1",
    start=utc_datetime("2025-04-02T00:00:00+00:00"),
    end=utc_datetime("2025-12-31T00:00:00+00:00"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update-golden", action="store_true",
                        help="confirm the overwrite of the golden reference")
    parser.add_argument("--note", type=str, default=None,
                        help="mandatory changelog note explaining the change")
    args = parser.parse_args()

    if not args.update_golden:
        parser.error("pass --update-golden to confirm the overwrite")
    if not args.note or not args.note.strip():
        parser.error("--note is mandatory: say why the reference is changing")

    cache = ParquetCache(REPO_ROOT / "data_cache")
    resolver = SymbolResolver(cache)
    spec = StrategySpec.from_json(REPO_ROOT / "strategies" / "rsi-wick-baseline.json")
    symbol_spec = resolver.symbol_spec(GOLDEN_CONFIG.symbol)
    server_tz = resolver.server_timezone()
    bars = load_golden_bars(cache, GOLDEN_CONFIG)

    payload = execute_golden(spec, GOLDEN_CONFIG, symbol_spec, server_tz, bars)
    payload["engine_version"] = ENGINE_VERSION
    payload["generated_at"] = datetime.now(timezone.utc).isoformat()
    payload["note"] = args.note.strip()

    REFERENCE_FILE.parent.mkdir(parents=True, exist_ok=True)
    REFERENCE_FILE.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    entry = (
        f"\n### Golden reference update ({stamp}, engine {ENGINE_VERSION})\n\n"
        f"- trades: {payload['snapshot']['execution']['trades']}, "
        f"final equity: {payload['snapshot']['metrics']['final_equity']}\n"
        f"- reason: {args.note.strip()}\n"
    )
    with CHANGELOG_FILE.open("a", encoding="utf-8") as handle:
        handle.write(entry)

    print(f"golden reference written: {REFERENCE_FILE}")
    print(f"trades={payload['snapshot']['execution']['trades']} "
          f"final_equity={payload['snapshot']['metrics']['final_equity']}")
    print(f"changelog entry appended to {CHANGELOG_FILE}")


if __name__ == "__main__":
    main()
