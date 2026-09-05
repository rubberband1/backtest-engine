"""What the forward test has done, against what the backtest said it would.

    python -m scripts.compare_live logs/live/rsi-mean-reversion-AUDUSD.r-H4.jsonl

One command, runnable at any moment while the runner is still going. It reads
the diary, works out which period it covers, runs a backtest of the same spec
over exactly that period with the same costs, and diffs the two records trade
by trade.

The output is not a score. It is a decomposition, because the three ways
reality and simulation can differ have nothing to do with each other:

- **slippage** on the trades both took - the price the engine expected
  against the price the broker gave;
- **trades only one of them took**, each with the reason the diary gives: an
  order refused, a gate that fired live and not in simulation, or a bar the
  runner never saw because it was not running;
- **the rest**, which is the interesting number. An unexplained difference is
  a defect nobody has found yet, and it is printed on its own line rather
  than folded into a total.

In dry run the diary holds decisions and no fills, so slippage is not
measurable and the report says so instead of printing a zero. What it does
measure even then is the part that matters most for an infrastructure test:
whether the runner saw every bar it should have seen, and whether it decided
what the backtest decided on each of them.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from core.data.cache import ParquetCache  # noqa: E402
from core.data.mt5_provider import MT5Provider  # noqa: E402
from core.data.provider import Timeframe  # noqa: E402
from core.engine.backtester import BacktestConfig, run_backtest  # noqa: E402
from core.engine.costs import (  # noqa: E402
    CommissionModel,
    CostModel,
    SpreadPolicy,
    SwapModel,
)
from core.live.compare import compare  # noqa: E402
from core.live.journal import Journal, environment  # noqa: E402
from core.runs.runner import SymbolResolver  # noqa: E402
from core.strategy.spec import StrategySpec  # noqa: E402
from core.validation.walkforward import apply_params  # noqa: E402
from core.version import ENGINE_VERSION  # noqa: E402

logger = logging.getLogger("compare-live")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("journal", type=Path, help="the diary to compare")
    parser.add_argument(
        "--strategy",
        type=Path,
        default=None,
        help="spec file; by default it is found from the diary's strategy id",
    )
    parser.add_argument("--equity", type=float, default=100.0)
    parser.add_argument("--commission-per-lot-per-side", type=float, default=0.0)
    parser.add_argument(
        "--spread-points",
        type=float,
        default=None,
        help="fixed spread to charge in the backtest; default is the "
        "instrument's M1 median, which is what the runner charges",
    )
    parser.add_argument("--cache-dir", type=Path, default=REPO_ROOT / "data_cache")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="use the cached bars instead of asking the terminal for the "
        "period the diary covers",
    )
    parser.add_argument(
        "--allow-version-mismatch",
        action="store_true",
        help="compare even though the diary was written by another engine "
        "version; the difference will include the change in the engine and "
        "cannot be read as slippage",
    )
    parser.add_argument("--json", type=Path, default=None, help="write the report")
    parser.add_argument("--log-level", default="WARNING")
    return parser.parse_args()


def diary_facts(journal: Journal) -> dict[str, object]:
    """What the diary says about itself: the run's identity and its span."""
    started = next(
        (event for event in journal.events() if event.kind == "started"), None
    )
    if started is None:
        raise SystemExit(
            f"{journal.path} holds no 'started' event: it is not a runner diary, "
            f"or the runner never got past start-up"
        )
    bars = [
        event.bar_time
        for event in journal.events()
        if event.kind == "bar" and event.bar_time is not None
    ]
    return {
        "strategy_id": started.detail.get("strategy_id"),
        "symbol": started.symbol,
        "timeframe": started.timeframe,
        "dry_run": started.detail.get("dry_run", True),
        "engine_version": started.detail.get("engine_version"),
        "first_bar": min(bars) if bars else None,
        "last_bar": max(bars) if bars else None,
        "bars": len(bars),
        "restarts": sum(1 for e in journal.events() if e.kind == "started"),
        "errors": [
            e.detail.get("message") or e.detail.get("reason")
            for e in journal.events()
            if e.kind == "error"
        ],
    }


def find_spec(strategy_id: str | None, override: Path | None) -> StrategySpec:
    if override is not None:
        return StrategySpec.from_json(override)
    if not strategy_id:
        raise SystemExit("the diary does not name a strategy: pass --strategy")
    candidate = REPO_ROOT / "strategies" / f"{strategy_id}.json"
    if not candidate.exists():
        raise SystemExit(
            f"no spec for {strategy_id!r} in strategies/: pass --strategy with "
            f"the file the runner was started from"
        )
    return StrategySpec.from_json(candidate)


def bars_for_period(
    cache: ParquetCache,
    symbol: str,
    timeframe: Timeframe,
    start: datetime,
    end: datetime,
    server_tz,
    offline: bool,
):
    """The same bars the runner saw, from the terminal or from the cache.

    The terminal is preferred: the runner read its bars from there, and a
    cache that was filled at a different time can hold a candle the broker
    has since rewritten. `--offline` is for comparing a diary on a machine
    with no terminal, and the report says which source was used.
    """
    from core.runs.runner import load_bars

    if offline:
        return load_bars(cache, symbol, timeframe, start, end), "local cache"
    with MT5Provider(server_timezone=server_tz) as provider:
        bars = provider.get_bars(symbol, timeframe, start, end)
    return bars, "MT5 terminal"


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=args.log_level.upper(), format="%(levelname)-7s %(name)s | %(message)s"
    )

    if not args.journal.exists():
        raise SystemExit(f"{args.journal} does not exist")

    journal = Journal(args.journal)
    facts = diary_facts(journal)
    if facts["first_bar"] is None:
        raise SystemExit(
            f"{args.journal} holds no processed bar yet: the runner has started "
            f"but the market has not produced a closed bar since. Nothing to "
            f"compare, which is not a failure"
        )

    spec = find_spec(str(facts["strategy_id"] or ""), args.strategy)
    symbol = str(facts["symbol"])
    timeframe = Timeframe.parse(str(facts["timeframe"]))
    spec = apply_params(
        spec,
        {"instrument.symbol": symbol, "instrument.timeframe": timeframe.name},
    )

    env = environment(journal)
    if (
        env.engine_version != ENGINE_VERSION
        and not args.allow_version_mismatch
    ):
        written_by = env.engine_version or "an unrecorded version"
        raise SystemExit(
            f"the diary was written by engine {written_by} and this is "
            f"{ENGINE_VERSION}. Comparing them measures the change in the "
            f"engine as well as the market, and the difference would appear on "
            f"the slippage line where nobody could tell the two apart.\n\n"
            f"  Either replay the period with the current engine and compare "
            f"that, or pass --allow-version-mismatch and read the result "
            f"knowing what is in it."
        )

    cache = ParquetCache(args.cache_dir)
    resolver = SymbolResolver(cache)
    server_tz = resolver.server_timezone()

    # the spec the runner traded, not the one the broker quotes today:
    # tick_value follows an FX rate, and re-reading it rescales every money
    # column on one side of the diff only
    symbol_spec = env.symbol_spec or resolver.symbol_spec(symbol)

    # the backtest must cover exactly the bars the runner processed: a wider
    # window would give it trades the runner never had the chance to take,
    # and every one of those would land in "only expected" as if the runner
    # had missed it
    start = facts["first_bar"]
    end = facts["last_bar"] + timedelta(minutes=timeframe.minutes)
    bars, source = bars_for_period(
        cache, symbol, timeframe, start, end, server_tz, args.offline
    )
    bars = bars[(bars.index >= start) & (bars.index <= facts["last_bar"])]

    spread = (
        SpreadPolicy(mode="fixed", value=args.spread_points)
        if args.spread_points is not None
        else _measured_spread(cache, symbol)
    )
    costs = CostModel(
        spread=spread,
        commission=CommissionModel(args.commission_per_lot_per_side),
        swap=SwapModel(mode="points"),
    )
    expected = run_backtest(
        spec, bars, symbol_spec, server_tz,
        BacktestConfig(initial_equity=args.equity, costs=costs),
    )
    report = compare(expected.trades, journal, symbol_spec, timeframe.name)

    facts["spec_source"] = (
        "pinned by the diary"
        if env.symbol_spec is not None
        else "read from the cache - the diary pinned none"
    )
    print(_header(facts, source, bars, spread))
    print(report.as_text())
    if facts["dry_run"]:
        print(_dry_run_note(report))

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        payload = {"diary": _jsonable(facts), "bars_source": source, **report.as_dict()}
        args.json.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8",
        )
        print(f"\n  written: {args.json}")
    return 0


def _dry_run_note(report) -> str:
    """What the numbers above do and do not mean without a real fill.

    The "slippage" line is the bucket for trades both records hold whose PnL
    differs. Live, that difference is slippage. In a dry run no order was
    sent, both sides are the engine's own arithmetic, and the same line
    should read zero - so a number there is not a cost, it is the runner and
    the backtester disagreeing, which is the one thing this project is built
    not to allow.
    """
    lines = [
        "",
        "  Dry run: no order reached the market. The realized PnL above is the",
        "  engine's own arithmetic on live bars, so slippage is not measured -",
        "  what is measured is the infrastructure: whether the runner saw every",
        "  bar it should have, and decided what the backtest decided on each.",
    ]
    drift = report.pnl_from_slippage
    if abs(drift) > 1e-9:
        lines += [
            "",
            f"  ! The matched trades differ by {drift:+.4f} with nothing to",
            "    slip against. In a dry run that line should be zero: the two",
            "    records are the same engine over the same bars. A non-zero",
            "    value means an input differed - the spread charged, bars the",
            "    terminal has rewritten since, or the engine version between",
            "    the diary and this backtest - and which one is worth finding",
            "    out before reading anything else here.",
        ]
    return "\n".join(lines)


def _measured_spread(cache: ParquetCache, symbol: str) -> SpreadPolicy:
    from core.data.spread import measure_from_cache

    reference = measure_from_cache(cache, symbol)
    if reference is None:
        raise SystemExit(
            f"no M1 sample to measure {symbol}'s spread on, so the backtest "
            f"cannot be given the cost the runner charged. Pass --spread-points"
        )
    return SpreadPolicy(mode="fixed", value=reference.median_points)


def _header(facts: dict[str, object], source: str, bars, spread: SpreadPolicy) -> str:
    started = facts["first_bar"]
    ended = facts["last_bar"]
    days = (ended - started).total_seconds() / 86400.0 if started and ended else 0.0
    lines = [
        "",
        f"Forward test - {facts['strategy_id']} on {facts['symbol']} "
        f"{facts['timeframe']}",
        f"  mode              : {'DRY RUN' if facts['dry_run'] else 'SENDING ORDERS'}",
        f"  engine at start   : {facts['engine_version']}",
        f"  diary covers      : {started} -> {ended}  ({days:.1f} days)",
        f"  bars in the diary : {facts['bars']}",
        f"  bars replayed     : {len(bars)} (from the {source})",
        f"  process restarts  : {facts['restarts']}",
        f"  spread charged    : {spread.mode} {spread.value or ''}".rstrip(),
        f"  instrument spec   : {facts['spec_source']}",
        "",
    ]
    errors = [e for e in (facts["errors"] or []) if e]
    if errors:
        lines.append(f"  {len(errors)} error(s) recorded in the diary:")
        for message in errors[:5]:
            lines.append(f"    - {str(message)[:140]}")
        if len(errors) > 5:
            lines.append(f"    ... and {len(errors) - 5} more")
        lines.append("")
    return "\n".join(lines)


def _jsonable(facts: dict[str, object]) -> dict[str, object]:
    return {
        key: (value.isoformat() if isinstance(value, datetime) else value)
        for key, value in facts.items()
    }


if __name__ == "__main__":
    sys.exit(main())
