"""Golden test: the frozen results are frozen.

A reference file pins the complete trade list and every metric produced by one
strategy over one fixed data window. Any engine change that alters any of
those numbers makes this test fail. That is the point: results must never
drift silently again (between phase 2 and phase 3 a one-day difference in the
data window changed the trade count and nobody declared it).

There are two references, because the guard has to survive leaving this
machine:

- **baseline** - the real strategy over real XAUUSD.r M1 bars. The number
  that matters, and the one that can only run where `data_cache/` was
  downloaded. It skips elsewhere, which is honest but leaves a clone with no
  regression guard at all.
- **fixture** - the same machinery over the synthetic dataset committed in
  `fixtures/data_cache/`. It runs on any clone, with no terminal and no
  download. It says nothing about any market; it says the engine's arithmetic
  has not moved, which is the only thing a golden test ever said.

To regenerate after an intentional change:

    python -m scripts.update_golden --update-golden --note "why"

The note is mandatory and lands in ENGINE_CHANGELOG.md.
"""
from __future__ import annotations

import json
from zoneinfo import ZoneInfo

import pytest

from core.data.cache import ParquetCache
from core.data.provider import SymbolSpec
from core.runs.runner import DataUnavailable
from core.runs.snapshot import diff_snapshots, execute_golden, load_golden_bars
from core.runs.store import RunConfig, data_fingerprint
from core.strategy.spec import StrategySpec
from scripts.update_golden import REPO_ROOT, TARGETS, GoldenTarget


@pytest.mark.parametrize("target", TARGETS, ids=[t.name for t in TARGETS])
def test_golden(target: GoldenTarget) -> None:
    if not target.reference.exists():
        pytest.fail(
            f"golden reference missing: {target.reference}. Generate it with "
            f"'python -m scripts.update_golden --update-golden --target "
            f"{target.name} --note ...'"
        )
    reference = json.loads(target.reference.read_text(encoding="utf-8"))
    config = RunConfig.from_dict(reference["config"])

    cache = ParquetCache(target.cache_root)
    if not cache.root.exists():
        pytest.skip(f"no cache at {target.cache_root} on this machine")

    spec = StrategySpec.from_json(
        REPO_ROOT / "strategies" / f"{target.strategy}.json"
    )

    # The symbol spec and timezone are replayed from the reference: tick_value
    # moves with FX rates, and the golden must not depend on a live terminal.
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
            f"the {target.strategy} spec changed: hash "
            f"{actual['spec_hash']} != reference {reference['spec_hash']}"
        )

    diffs = diff_snapshots(reference["snapshot"], actual["snapshot"])
    if diffs:
        shown = "\n".join(diffs[:50])
        more = f"\n... and {len(diffs) - 50} more differences" if len(diffs) > 50 else ""
        pytest.fail(
            f"golden mismatch on {target.name} ({len(diffs)} differences). If "
            f"the change is intentional, regenerate with 'python -m "
            f"scripts.update_golden --update-golden --note ...':\n{shown}{more}"
        )


def test_the_fixture_golden_runs_without_a_broker() -> None:
    """The guard that the guard is portable.

    `baseline` skips on any machine that has not downloaded the real bars.
    Without a second reference that ships with its own data, a clone runs the
    whole suite green while the one test that pins the engine's output never
    executes - which is worse than not having it, because the suite claims
    coverage it does not have.
    """
    fixture = next(t for t in TARGETS if t.name == "fixture")
    assert fixture.cache_root.exists(), (
        "the committed fixture is missing: run 'python -m scripts.make_fixture'"
    )
    assert fixture.reference.exists()
