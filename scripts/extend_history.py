"""Extends the local cache to the deepest history the broker will give.

Five steps, in order, each one reported before the next runs:

1. probe the feed for every (symbol, timeframe) pair and print what exists;
2. download H1/H4/D1 from the first usable bar to now, recording the feed as
   the source of every interval it fills;
3. decode the terminal's own .hc cache for the same pairs and compare it to
   the feed on the overlap - if the two disagree, the seam is reported and
   nothing is silently merged;
4. fill from the .hc cache only what the feed did not provide, recorded under
   its own source name;
5. run the quality report over the whole extended series, which is where
   contract changes and renames in the distant past show up.

    python -m scripts.extend_history --timeframes H1 H4 D1
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from core.data.cache import ParquetCache
from core.data.continuity import ContinuityReport, compare_sources, summarize
from core.data.depth import DepthProbe, as_text, build_report, probe_all
from core.data.hc_reader import find_hc, read_hc_utc
from core.data.mt5_provider import MT5Provider
from core.data.provider import Timeframe
from core.data.quality import check_quality
from core.runs.runner import load_bars
from core.serialization import json_safe

logger = logging.getLogger("extend_history")

SYMBOLS: tuple[str, ...] = (
    "XAUUSD.r", "XAGUSD.r", "XAUEUR.r", "EURUSD.r", "GBPUSD.r",
    "USDJPY.r", "AUDUSD.r", "USDCAD.r", "XTIUSD", "XBRUSD",
)
PROBE_TIMEFRAMES: tuple[str, ...] = ("M5", "M15", "H1", "H4", "D1")
DOWNLOAD_TIMEFRAMES: tuple[str, ...] = ("H1", "H4", "D1")

HC_SOURCE = "mt5_hc_cache"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="*", default=list(SYMBOLS))
    parser.add_argument("--probe-timeframes", nargs="*", default=list(PROBE_TIMEFRAMES))
    parser.add_argument("--timeframes", nargs="*", default=list(DOWNLOAD_TIMEFRAMES))
    parser.add_argument("--cache-dir", default="data_cache")
    parser.add_argument("--out", default=".temp/history-depth.json")
    parser.add_argument(
        "--probe-only",
        action="store_true",
        help="report the available depth without downloading anything",
    )
    return parser.parse_args()


def download(
    cache: ParquetCache,
    provider: MT5Provider,
    symbol: str,
    timeframe: Timeframe,
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    return cache.get_or_fetch(
        symbol,
        timeframe,
        start,
        end,
        fetch=provider.get_bars,
        server_timezone=provider.server_timezone,
        source=provider.source_name,
    )


def hc_frame(
    provider: MT5Provider, symbol: str, timeframe: Timeframe
) -> pd.DataFrame | None:
    """The terminal .hc cache for a pair, or None when there is no file."""
    data_path = provider.data_path
    if data_path is None:
        return None
    path = find_hc(data_path, symbol, timeframe.name)
    if path is None:
        return None
    try:
        return read_hc_utc(path, provider.server_timezone)
    except Exception as exc:
        logger.warning("%s %s: .hc unreadable (%s)", symbol, timeframe.name, exc)
        return None


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)-7s %(name)s | %(message)s"
    )
    cache = ParquetCache(Path(args.cache_dir))
    now = datetime.now(timezone.utc)

    with MT5Provider() as provider:
        server_tz = provider.server_timezone
        probes: list[DepthProbe] = probe_all(
            provider, args.symbols, args.probe_timeframes
        )
        depth = build_report(probes)
        print("\n=== DEPTH AVAILABLE FROM THE FEED ===")
        print(as_text(depth))

        if args.probe_only:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(
                json.dumps(json_safe({"depth": depth.as_dict()}), indent=2),
                encoding="utf-8",
            )
            return 0

        seams: list[ContinuityReport] = []
        downloaded: list[dict[str, object]] = []
        for symbol in args.symbols:
            spec = provider.get_symbol_spec(symbol)
            for name in args.timeframes:
                tf = Timeframe.parse(name)
                start = depth.usable_from(symbol, tf.name)
                if start is None:
                    logger.warning("%s %s: nothing to download", symbol, tf.name)
                    continue

                bars = download(cache, provider, symbol, tf, start, now)
                source_counts = {provider.source_name: int(len(bars))}

                # the .hc cache is a second opinion on the same pair: compared
                # first, used only for what the feed did not have
                second = hc_frame(provider, symbol, tf)
                if second is not None and len(second):
                    seam = compare_sources(
                        bars,
                        second,
                        symbol,
                        tf.name,
                        spec.point,
                        left_source=provider.source_name,
                        right_source=HC_SOURCE,
                    )
                    seams.append(seam)
                    extra = second.index.difference(bars.index)
                    if len(extra):
                        missing = second.loc[extra]
                        for year in sorted({t.year for t in extra}):
                            piece = missing[missing.index.year == year]
                            if not len(piece):
                                continue
                            cache.write_year(
                                symbol, tf, year, piece,
                                [(piece.index[0].to_pydatetime(),
                                  piece.index[-1].to_pydatetime())],
                                str(provider.server_timezone),
                                HC_SOURCE,
                            )
                        source_counts[HC_SOURCE] = int(len(extra))
                        bars = download(cache, provider, symbol, tf, start, now)

                downloaded.append(
                    {
                        "symbol": symbol,
                        "timeframe": tf.name,
                        "bars": int(len(bars)),
                        "first_bar": bars.index[0] if len(bars) else None,
                        "last_bar": bars.index[-1] if len(bars) else None,
                        "sources": source_counts,
                    }
                )
                logger.info(
                    "%s %s: %d bars cached (%s)",
                    symbol, tf.name, len(bars),
                    ", ".join(f"{k}={v}" for k, v in source_counts.items()),
                )

    print("\n=== CONTINUITY BETWEEN SOURCES ===")
    print(summarize(seams))

    print("\n=== QUALITY OVER THE EXTENDED HISTORY ===")
    quality: list[dict[str, object]] = []
    for entry in downloaded:
        symbol, name = str(entry["symbol"]), str(entry["timeframe"])
        bars = load_bars(cache, symbol, Timeframe.parse(name))
        report = check_quality(bars, symbol, name, server_tz=server_tz)
        quality.append(
            json_safe({**asdict(report), "completeness": report.completeness})
        )
        print(report.as_text(max_gaps=3))
        print()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps(
            json_safe(
                {
                    "generated_at": now,
                    "depth": depth.as_dict(),
                    "downloaded": downloaded,
                    "continuity": [seam.as_dict() for seam in seams],
                    "quality": quality,
                }
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
