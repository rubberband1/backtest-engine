"""Regenerate the golden reference for the baseline strategy.

    python -m scripts.update_golden --update-golden --note "reason for the change"

Refuses to run without both flags: overwriting the golden reference is a
declaration that the engine's results changed on purpose, and the reason goes
into ENGINE_CHANGELOG.md where the next person can read it.

**The SymbolSpec and the server timezone are replayed from the existing
reference, not read from the terminal.** They are inputs to the result -
`tick_value` tracks an FX rate and moves between two reads of the same
instrument - and the golden test pins them for exactly that reason. Taking
today's values here would rebase the reference onto a different input while
the note in the changelog talked about an engine change: on the 4.0.0 update
that produced 548 differences, every one of them a `tick_value` that had
drifted from 0.8630 to 0.8612 since the reference was written, and none of
them anything the engine had done. `--refresh-symbol-spec` asks for the
rebase deliberately, and says so in the changelog.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone, tzinfo
from pathlib import Path
from zoneinfo import ZoneInfo

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from core.data.cache import ParquetCache  # noqa: E402
from core.data.provider import SymbolSpec  # noqa: E402
from core.runs.runner import SymbolResolver  # noqa: E402
from core.runs.snapshot import execute_golden, load_golden_bars, utc_datetime  # noqa: E402
from core.runs.store import RunConfig  # noqa: E402
from core.strategy.spec import StrategySpec  # noqa: E402
from core.version import ENGINE_VERSION  # noqa: E402

REFERENCE_DIR = REPO_ROOT / "tests" / "golden" / "reference"
REFERENCE_FILE = REFERENCE_DIR / "rsi-wick-baseline.json"
CHANGELOG_FILE = REPO_ROOT / "ENGINE_CHANGELOG.md"

# The frozen window: 2025-04-02 through 2025-12-30 inclusive (end bound is
# exclusive, like everywhere else in the engine).
GOLDEN_CONFIG = RunConfig(
    symbol="XAUUSD.r",
    timeframe="M1",
    start=utc_datetime("2025-04-02T00:00:00+00:00"),
    end=utc_datetime("2025-12-31T00:00:00+00:00"),
)


@dataclass(frozen=True)
class GoldenTarget:
    """One frozen result: which strategy, over which bars, from which cache."""

    name: str
    reference: Path
    cache_root: Path
    config: RunConfig
    strategy: str


# Two references, for two different jobs.
#
# `baseline` is the real one: the actual strategy over real XAUUSD bars, and
# the number that matters. It can only run where `data_cache/` exists, which
# is the machine that downloaded it.
#
# `fixture` runs anywhere. It pins the engine against the synthetic dataset
# that ships with the repository, so a stranger who clones this can run the
# guard that says "the engine still computes what it used to" without a
# terminal, a broker, or a download. It says nothing about any market - it is
# a regression test on arithmetic, which is all a golden test ever was.
TARGETS: tuple[GoldenTarget, ...] = (
    GoldenTarget(
        name="baseline",
        reference=REFERENCE_FILE,
        cache_root=REPO_ROOT / "data_cache",
        config=GOLDEN_CONFIG,
        strategy="rsi-wick-baseline",
    ),
    GoldenTarget(
        name="fixture",
        reference=REFERENCE_DIR / "rsi-mean-reversion-fixture.json",
        cache_root=REPO_ROOT / "fixtures" / "data_cache",
        config=RunConfig(
            symbol="SYNTHGOLD",
            timeframe="H1",
            start=utc_datetime("2022-01-01T00:00:00+00:00"),
            end=utc_datetime("2024-01-01T00:00:00+00:00"),
            spread_mode="fixed",
            spread_value=8.0,
        ),
        strategy="rsi-mean-reversion",
    ),
)


def target_by_name(name: str) -> GoldenTarget:
    for target in TARGETS:
        if target.name == name:
            return target
    raise SystemExit(f"unknown golden target {name!r}")


def pinned_environment(
    cache: ParquetCache,
    target: GoldenTarget,
    refresh: bool = False,
) -> tuple[SymbolSpec, tzinfo, str]:
    """The instrument spec and clock the reference is replayed with.

    The existing reference wins over the terminal, because the reference is
    what the golden test compares against and it replays these same values.
    Reading them fresh here would make every regeneration also a silent data
    change.
    """
    if not refresh and target.reference.exists():
        payload = json.loads(target.reference.read_text(encoding="utf-8"))
        if payload.get("symbol_spec") and payload.get("server_timezone"):
            return (
                SymbolSpec(**payload["symbol_spec"]),
                ZoneInfo(str(payload["server_timezone"])),
                "replayed from the existing reference",
            )

    resolver = SymbolResolver(cache)
    return (
        resolver.symbol_spec(target.config.symbol),
        resolver.server_timezone(),
        "read from the cache or the terminal (the reference is rebased onto them)",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update-golden", action="store_true",
                        help="confirm the overwrite of the golden reference")
    parser.add_argument("--note", type=str, default=None,
                        help="mandatory changelog note explaining the change")
    parser.add_argument(
        "--refresh-symbol-spec", action="store_true",
        help="re-read the instrument spec and server timezone from the "
             "terminal instead of replaying the pinned ones. This rebases the "
             "reference onto different inputs: use it when the broker's "
             "contract really changed, never to regenerate after an engine fix",
    )
    parser.add_argument(
        "--target",
        choices=[t.name for t in TARGETS] + ["all"],
        default="all",
        help="which reference to regenerate. 'baseline' needs data_cache/; "
        "'fixture' runs anywhere",
    )
    args = parser.parse_args()

    if not args.update_golden:
        parser.error("pass --update-golden to confirm the overwrite")
    if not args.note or not args.note.strip():
        parser.error("--note is mandatory: say why the reference is changing")

    targets = TARGETS if args.target == "all" else (target_by_name(args.target),)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    lines: list[str] = []

    for target in targets:
        cache = ParquetCache(target.cache_root)
        if not cache.root.exists():
            print(f"[{target.name}] skipped: no cache at {cache.root}")
            continue

        spec = StrategySpec.from_json(
            REPO_ROOT / "strategies" / f"{target.strategy}.json"
        )
        symbol_spec, server_tz, source = pinned_environment(
            cache, target, refresh=args.refresh_symbol_spec
        )
        print(f"[{target.name}] instrument spec and server clock: {source}")
        bars = load_golden_bars(cache, target.config)

        payload = execute_golden(spec, target.config, symbol_spec, server_tz, bars)
        payload["engine_version"] = ENGINE_VERSION
        payload["generated_at"] = datetime.now(timezone.utc).isoformat()
        payload["note"] = args.note.strip()
        payload["symbol_spec_source"] = source
        payload["target"] = target.name

        target.reference.parent.mkdir(parents=True, exist_ok=True)
        target.reference.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        trades = payload["snapshot"]["execution"]["trades"]
        equity = payload["snapshot"]["metrics"]["final_equity"]
        lines.append(
            f"- {target.name}: {trades} trades, final equity {equity} ({source})"
        )
        print(f"[{target.name}] written: {target.reference}")
        print(f"[{target.name}] trades={trades} final_equity={equity}")

    if not lines:
        raise SystemExit("no reference was regenerated: no cache was available")

    entry = (
        f"\n### Golden reference update ({stamp}, engine {ENGINE_VERSION})\n\n"
        + "\n".join(lines)
        + f"\n- reason: {args.note.strip()}\n"
    )
    with CHANGELOG_FILE.open("a", encoding="utf-8") as handle:
        handle.write(entry)
    print(f"changelog entry appended to {CHANGELOG_FILE}")


if __name__ == "__main__":
    main()
