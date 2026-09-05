"""Writes the synthetic fixture dataset into a Parquet cache.

    python -m scripts.make_fixture

The output is what ships in the repository as `fixtures/data_cache/`, and it
is what `python run.py --fixture` serves. Running this again must reproduce
it byte for byte in content: the generator is seeded from the symbol name and
spans a fixed period, so a regenerated fixture that differs from the
committed one means the generator changed, and that is a change to every
golden number measured on it.

The M1 series is written for a short window only. All of it would be around
750k bars per instrument, and the repository does not need to carry that to
prove the point: the months that are there exist so the spread can be
measured where it is meaningful, which is the one thing higher timeframes
cannot supply.
"""
from __future__ import annotations

import argparse
import logging
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from core.data.cache import ParquetCache  # noqa: E402
from core.data.fixture_provider import (  # noqa: E402
    EPOCH,
    HORIZON,
    INSTRUMENTS,
    SERVER_TIMEZONE,
    SOURCE_NAME,
    FixtureProvider,
)
from core.data.provider import SymbolSpecSnapshot, Timeframe  # noqa: E402
from core.runs.runner import SPEC_CACHE_FILE, TZ_CACHE_FILE  # noqa: E402

logger = logging.getLogger("make-fixture")

DEFAULT_ROOT = REPO_ROOT / "fixtures" / "data_cache"

# The timeframes that ship in full, and the one that ships as a sample.
FULL_TIMEFRAMES = (Timeframe.H1, Timeframe.H4, Timeframe.D1)
SAMPLE_TIMEFRAME = Timeframe.M1
SAMPLE_START = datetime(2023, 12, 1, tzinfo=timezone.utc)

# A fixed read timestamp, so regenerating the fixture does not rewrite the
# spec file with a new "read_at" and show up as a spurious diff. The spec of
# an invented instrument was not read at any particular moment anyway.
PINNED_READ_AT = datetime(2024, 1, 1, tzinfo=timezone.utc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--clean", action="store_true", help="delete the target first"
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def build(root: Path, clean: bool = False) -> ParquetCache:
    """Fills `root` with the fixture. Returns the cache that now holds it."""
    if clean and root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    cache = ParquetCache(root)
    provider = FixtureProvider()
    (root / TZ_CACHE_FILE).write_text(SERVER_TIMEZONE, encoding="utf-8")

    for symbol in INSTRUMENTS:
        snapshot = SymbolSpecSnapshot(
            spec=provider.get_symbol_spec(symbol), read_at=PINNED_READ_AT
        )
        folder = root / cache._slug(symbol)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / SPEC_CACHE_FILE).write_text(
            _json(snapshot.to_dict()), encoding="utf-8"
        )

        for timeframe in (*FULL_TIMEFRAMES, SAMPLE_TIMEFRAME):
            start = SAMPLE_START if timeframe is SAMPLE_TIMEFRAME else EPOCH
            bars = provider.get_bars(symbol, timeframe, start, HORIZON)
            for year in sorted({moment.year for moment in bars.index}):
                slice_ = bars[bars.index.year == year]
                lower = max(start, datetime(year, 1, 1, tzinfo=timezone.utc))
                upper = min(HORIZON, datetime(year + 1, 1, 1, tzinfo=timezone.utc))
                cache.write_year(
                    symbol, timeframe, year, slice_, [(lower, upper)],
                    server_timezone=SERVER_TIMEZONE, source=SOURCE_NAME,
                )
            logger.info("%s %s: %d bars", symbol, timeframe.name, len(bars))
    return cache


def _json(payload: dict) -> str:
    import json

    return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=args.log_level.upper(), format="%(levelname)-7s %(name)s | %(message)s"
    )
    build(args.root, clean=args.clean)

    total = sum(f.stat().st_size for f in args.root.rglob("*") if f.is_file())
    logger.info("written: %s (%.1f MB)", args.root, total / 1e6)
    return 0


if __name__ == "__main__":
    sys.exit(main())
