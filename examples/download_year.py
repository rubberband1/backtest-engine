"""Downloads one year of history from MT5 and prints the quality report.

    python -m examples.download_year XAUUSD.r --timeframe M1 --year 2024

The symbol is mandatory: there is no default one in the code.
"""
from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path

from core.data.cache import ParquetCache
from core.data.mt5_provider import MT5Provider
from core.data.provider import Timeframe
from core.data.quality import check_quality

logger = logging.getLogger("download_year")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("symbol", help="symbol name as listed by the broker")
    parser.add_argument("--timeframe", default="M1")
    parser.add_argument("--year", type=int, default=datetime.now(timezone.utc).year - 1)
    parser.add_argument("--cache", type=Path, default=Path("data_cache"))
    parser.add_argument("--refresh", action="store_true", help="invalidate the year before downloading")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    )

    timeframe = Timeframe.parse(args.timeframe)
    start = datetime(args.year, 1, 1, tzinfo=timezone.utc)
    end = datetime(args.year + 1, 1, 1, tzinfo=timezone.utc)
    cache = ParquetCache(args.cache)

    if args.refresh:
        for path in cache.invalidate(args.symbol, timeframe, args.year):
            logger.info("removed %s", path)

    with MT5Provider() as provider:
        spec = provider.get_symbol_spec(args.symbol)
        logger.info(
            "%s: point=%s digits=%s contract=%s tick_value=%s (%s)",
            spec.name,
            spec.point,
            spec.digits,
            spec.contract_size,
            spec.tick_value,
            spec.currency_profit,
        )
        bars = cache.get_or_fetch(
            args.symbol,
            timeframe,
            start,
            end,
            provider.get_bars,
            server_timezone=provider.server_timezone,
        )

    report = check_quality(bars, args.symbol, timeframe)
    for line in report.as_text().splitlines():
        logger.info("%s", line)
    logger.info(
        "median spread: %.1f points (%.5f in price)",
        bars["spread"].median() if len(bars) else float("nan"),
        (bars["spread"].median() * spec.point) if len(bars) else float("nan"),
    )


if __name__ == "__main__":
    main()
