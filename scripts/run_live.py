"""Runs a strategy against the broker, on closed bars, from the command line.

    python -m scripts.run_live strategies/ma-crossover.json --symbol XAUUSD.r \
        --timeframe H1

Dry run is the default and has to be turned off explicitly:

    python -m scripts.run_live ... --send

Even then the broker refuses to start on anything but a demo account. That is
not a convenience: nothing in this repository has ever been validated against
a funded account, and the replay equivalence test says nothing about one.

Ctrl+C stops the loop, releases the lock and closes the diary. It does **not**
close an open position: the runner's job is to manage it on the next bar, and
a process that liquidates on exit would make a restart cost money.

**Built to be left alone for weeks.** A forward test that needs attention is
not a forward test, so the failures that actually happen over that span are
handled here rather than left to a supervisor nobody wrote:

- **The terminal goes away.** A dropped connection, an MT5 update, a machine
  asleep: the fetch fails, the failure goes in the diary, and the loop
  reconnects with a backoff that grows to a ceiling instead of hammering a
  terminal that is not there. Nothing is decided while disconnected, because
  no bar arrives to decide on.
- **The process dies and comes back.** The lock is released, the position is
  read back from the broker by magic number, and the closed trades and the
  equity are read back from the diary - so the second process continues the
  first one's account instead of starting a fresh one at the initial equity.
- **Weekends.** No closed bar arrives, so nothing happens; the session
  calendar counts the gap rather than the rows, so a Monday bar is one
  session bar after Friday's.
- **Clock changes.** The session calendar is pinned once, carries the server
  timezone, and reads each UTC instant's weekly slot on that clock, so a DST
  change moves neither the session nor the bar a time stop fires on.
"""
from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import FrameType

import pandas as pd

from core.data.cache import ParquetCache
from core.data.mt5_provider import MT5Provider
from core.data.provider import Timeframe
from core.data.spread import measure_from_cache
from core.engine.costs import CommissionModel, CostModel, SpreadPolicy, SwapModel
from core.live.broker import DEFAULT_MAGIC, LiveBroker
from core.live.runner import LiveConfig, LiveRunner, RunnerState
from core.runs.runner import SymbolResolver
from core.strategy.spec import StrategySpec
from core.validation.walkforward import apply_params

logger = logging.getLogger("live")

LIVE_DIR = Path("logs/live")
DEFAULT_POLL_SECONDS = 20.0
# A failed fetch is usually a blip and occasionally an outage. The wait
# doubles from the poll interval to this ceiling, so a terminal that is down
# for a weekend is asked about twice a minute rather than three times a
# second, and one that blinks is picked up on the next poll.
MAX_BACKOFF_SECONDS = 300.0

_stop = False


def _handle_signal(number: int, frame: FrameType | None) -> None:
    global _stop
    _stop = True
    logger.info("stop requested, finishing the current bar")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("strategy", type=Path)
    parser.add_argument("--symbol", default=None, help="overrides the spec")
    parser.add_argument("--timeframe", default=None, help="overrides the spec")
    parser.add_argument("--equity", type=float, default=100.0)
    parser.add_argument("--warmup-bars", type=int, default=2000)
    parser.add_argument("--commission-per-lot-per-side", type=float, default=0.0)
    parser.add_argument("--magic", type=int, default=DEFAULT_MAGIC)
    parser.add_argument("--poll-seconds", type=float, default=DEFAULT_POLL_SECONDS)
    parser.add_argument("--live-dir", type=Path, default=LIVE_DIR)
    parser.add_argument("--cache-dir", type=Path, default=Path("data_cache"))
    parser.add_argument(
        "--spread-points",
        type=float,
        default=None,
        help="fixed spread to charge; default is the instrument's M1 median",
    )
    parser.add_argument(
        "--send",
        action="store_true",
        help="actually send orders. Refused unless the account is a demo",
    )
    parser.add_argument("--once", action="store_true", help="one poll, then exit")
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="also write the log here. The process owning its own log is what "
        "lets it be started without a shell wrapper to redirect for it, and a "
        "wrapper is a second process that has to stay alive for weeks",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def spread_policy(
    cache: ParquetCache, symbol: str, override: float | None
) -> SpreadPolicy:
    """The cost to charge per fill, measured where the field means something.

    The `spread` column of a bar above M1 is the minimum spread inside it, so
    reading it live at H1 would charge the best price of the hour. The M1
    median is a number that was actually quoted.
    """
    if override is not None:
        return SpreadPolicy(mode="fixed", value=override)
    reference = measure_from_cache(cache, symbol)
    if reference is None:
        logger.warning(
            "%s: no M1 sample to measure the spread on, falling back to the bar "
            "column. Above M1 that understates the cost",
            symbol,
        )
        return SpreadPolicy(mode="per_bar")
    logger.info("charging %.0f points per fill (%s)", reference.median_points, reference.verdict)
    return SpreadPolicy(mode="fixed", value=reference.median_points)


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=args.log_level.upper(), format="%(levelname)-7s %(name)s | %(message)s"
    )
    if args.log_file is not None:
        args.log_file.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(args.log_file, encoding="utf-8")
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-7s %(name)s | %(message)s")
        )
        logging.getLogger().addHandler(handler)
    signal.signal(signal.SIGINT, _handle_signal)

    spec = StrategySpec.from_json(args.strategy)
    overrides = {}
    if args.symbol:
        overrides["instrument.symbol"] = args.symbol
    if args.timeframe:
        overrides["instrument.timeframe"] = args.timeframe
    if overrides:
        spec = apply_params(spec, overrides)

    symbol = spec.instrument.symbol
    timeframe: Timeframe = spec.instrument.tf
    cache = ParquetCache(args.cache_dir)
    resolver = SymbolResolver(cache)
    symbol_spec = resolver.symbol_spec(symbol)
    server_tz = resolver.server_timezone()

    stem = f"{spec.id}-{symbol}-{timeframe.name}".replace("/", "_")
    config = LiveConfig(
        initial_equity=args.equity,
        costs=CostModel(
            spread=spread_policy(cache, symbol, args.spread_points),
            commission=CommissionModel(args.commission_per_lot_per_side),
            swap=SwapModel(mode="points"),
        ),
        magic=args.magic,
        warmup_bars=args.warmup_bars,
        journal_path=args.live_dir / f"{stem}.jsonl",
        lock_path=args.live_dir / f"{stem}.lock",
        dry_run=not args.send,
    )

    with MT5Provider(server_timezone=server_tz) as provider:
        history = warmup_history(provider, symbol, timeframe, args.warmup_bars)
        if history.empty:
            raise SystemExit(
                f"no closed {timeframe.name} bars for {symbol}: is the market open "
                f"and the terminal connected?"
            )

        broker = LiveBroker(dry_run=not args.send, magic=args.magic)
        runner = LiveRunner(spec, symbol_spec, server_tz, config, broker)
        state = runner.start(history)
        logger.info(
            "%s on %s %s: %d bars of history, last %s, %s",
            spec.id, symbol, timeframe.name, len(history), state.last_bar_time,
            "DRY RUN" if state.dry_run else "SENDING ORDERS",
        )
        logger.info("diary: %s", config.journal_path)
        # The warm-up download is worth a line. The poll that follows it fetches
        # a handful of bars every twenty seconds for as long as the test runs,
        # and at INFO that is four thousand identical lines a day in a file
        # nobody will ever read to the end. The diary is the record; this is
        # only the process's own account of itself.
        logging.getLogger("core.data.mt5_provider").setLevel(logging.WARNING)
        if state.trades:
            logger.info(
                "continuing an earlier run: %d trade(s) already closed, equity %.2f",
                state.trades, state.equity,
            )

        try:
            state = poll_forever(runner, provider, symbol, timeframe, args)
        finally:
            runner.stop()

    logger.info(
        "stopped after %d bars, %d trades, equity %.2f",
        state.bars_seen, state.trades, state.equity,
    )
    return 0


def warmup_history(
    provider: MT5Provider, symbol: str, timeframe: Timeframe, bars: int
) -> pd.DataFrame:
    """Closed bars to seed the indicator states and the session calendar.

    The newest bar the terminal returns is the one still forming. Dropping it
    is the whole of "no intra-bar evaluation, ever", and it is done here
    because only the caller knows what time it is.
    """
    now = datetime.now(timezone.utc)
    span = timedelta(minutes=timeframe.minutes * (bars + 10))
    history = provider.get_bars(symbol, timeframe, now - span, now)
    return history[history.index + timedelta(minutes=timeframe.minutes) <= now]


def poll_forever(
    runner: LiveRunner,
    provider: MT5Provider,
    symbol: str,
    timeframe: Timeframe,
    args: argparse.Namespace,
) -> RunnerState:
    """The loop that has to survive everything a month can do to it.

    A fetch that raises is not fatal and is not silent: the runner writes it
    to the diary, the wait doubles, and the loop keeps going. A fetch that
    succeeds resets the wait. Nothing is decided while the feed is down,
    because deciding needs a closed bar and none arrives.
    """
    state = runner.state()
    wait = float(args.poll_seconds)
    failures = 0

    while not _stop:
        before = state.bars_seen
        try:
            state = runner.poll(
                lambda start, end: provider.get_bars(symbol, timeframe, start, end)
            )
        except Exception as exc:
            # runner.poll already swallows a failing fetch; reaching here means
            # something else broke, and a forward test that stops on the first
            # surprise measures nothing
            failures += 1
            logger.warning(
                "poll %d failed (%s: %s), retrying in %.0fs",
                failures, type(exc).__name__, exc, wait,
            )
            state = runner.state()
        else:
            if state.bars_seen > before:
                logger.info(
                    "%d new bar(s), %d trade(s), equity %.2f",
                    state.bars_seen - before, state.trades, state.equity,
                )
            failures = 0
            wait = float(args.poll_seconds)

        if args.once:
            break
        _sleep(wait)
        if failures:
            wait = min(wait * 2.0, MAX_BACKOFF_SECONDS)
    return state


def _sleep(seconds: float) -> None:
    """Sleeps in short slices so Ctrl+C is answered within a second."""
    deadline = time.monotonic() + seconds
    while not _stop and time.monotonic() < deadline:
        time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))


if __name__ == "__main__":
    sys.exit(main())
