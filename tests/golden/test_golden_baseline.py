"""Golden test: the baseline result is frozen.

The reference file pins the complete trade list and every metric produced by
the baseline strategy on a fixed data window. Any engine change that alters
any of those numbers makes this test fail. That is the point: results must
never drift silently again (between phase 2 and phase 3 a one-day difference
in the data window changed the trade count and nobody declared it).

To regenerate the reference after an intentional change:

    python -m scripts.update_golden --update-golden --note "why"

The note is mandatory and lands in ENGINE_CHANGELOG.md.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.data.cache import ParquetCache
from core.runs.snapshot import diff_snapshots, execute_golden, load_golden_bars
from core.runs.store import RunConfig, data_fingerprint
from core.strategy.spec import StrategySpec

REPO_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_FILE = Path(__file__).parent / "reference" / "rsi-wick-baseline.json"


def test_golden_baseline() -> None:
    if not REFERENCE_FILE.exists():
        pytest.fail(
            f"golden reference missing: {REFERENCE_FILE}. Generate it with "
            f"'python -m scripts.update_golden --update-golden --note ...'"
        )
    reference = json.loads(REFERENCE_FILE.read_text(encoding="utf-8"))
    config = RunConfig.from_dict(reference["config"])

    cache = ParquetCache(REPO_ROOT / "data_cache")
    if not cache.root.exists():
        pytest.skip("data cache not available on this machine")

    spec_path = REPO_ROOT / "strategies" / "rsi-wick-baseline.json"
    spec = StrategySpec.from_json(spec_path)

    # The symbol spec and timezone are replayed from the reference: tick_value
    # moves with FX rates, and the golden must not depend on a live terminal.
    from zoneinfo import ZoneInfo

    from core.data.provider import SymbolSpec
    from core.runs.runner import DataUnavailable

    symbol_spec = SymbolSpec(**reference["symbol_spec"])
    server_tz = ZoneInfo(reference["server_timezone"])
    try:
        bars = load_golden_bars(cache, config)
    except DataUnavailable as exc:
        pytest.skip(f"golden data not available: {exc}")

    fingerprint = data_fingerprint(bars)
    if fingerprint != reference["data"]["fingerprint"]:
        pytest.fail(
            "the cached data under the golden window changed "
            f"(fingerprint {fingerprint} != reference "
            f"{reference['data']['fingerprint']}). This is a DATA change, not "
            "an engine regression: investigate the cache before touching the "
            "reference."
        )

    actual = execute_golden(spec, config, symbol_spec, server_tz, bars)

    if actual["spec_hash"] != reference["spec_hash"]:
        pytest.fail(
            "the baseline spec changed: hash "
            f"{actual['spec_hash']} != reference {reference['spec_hash']}"
        )

    diffs = diff_snapshots(reference["snapshot"], actual["snapshot"])
    if diffs:
        shown = "\n".join(diffs[:50])
        more = f"\n... and {len(diffs) - 50} more differences" if len(diffs) > 50 else ""
        pytest.fail(
            f"golden mismatch ({len(diffs)} differences). If the change is "
            f"intentional, regenerate with 'python -m scripts.update_golden "
            f"--update-golden --note ...':\n{shown}{more}"
        )
