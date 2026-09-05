"""One-shot: rename the raw spread column on disk, and mark the runs it fed.

Two migrations, both idempotent, both safe to re-run:

1. **The cache.** Above M1 the broker's `spread` field is the minimum of the
   spreads of the M1 bars inside the period - measured at 100% on three
   instruments over four thousand periods each. `ParquetCache.read_year`
   already renames it to `min_spread_m1` on the way in, so nothing reads it
   under the wrong name any more; this rewrites the files themselves so that
   nothing *stored* carries it either. A column that is not called `spread`
   cannot be charged as one by a future reader who did not read the comment.

2. **The run archive.** Every stored run above M1 whose spread policy was
   `per_bar` or `quantile` charged that column as a fill cost. Its costs are
   understated by an amount nobody measured, and its numbers are not
   comparable with anything produced since. Those runs get
   `aggregated_spread_cost: true` in their metadata, and the dashboard says
   so next to their equity.

    python -m scripts.migrate_spread_column --apply
    python -m scripts.migrate_spread_column            # dry run, changes nothing
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from core.data.provider import (  # noqa: E402
    MIN_SPREAD_M1_COLUMN,
    SPREAD_COLUMN,
    Timeframe,
)
from core.runs.store import META_FILE, RunStore  # noqa: E402

logger = logging.getLogger("migrate_spread_column")

AGGREGATED_MODES = ("per_bar", "quantile")


@dataclass
class Counts:
    seen: int = 0
    changed: int = 0
    already: int = 0
    skipped: int = 0


def migrate_cache(root: Path, apply: bool) -> Counts:
    """Rewrites every above-M1 parquet whose spread column still lies."""
    counts = Counts()
    for data_path in sorted(root.glob("*/*/*.parquet")):
        timeframe_name = data_path.parent.name
        try:
            timeframe = Timeframe.parse(timeframe_name)
        except ValueError:
            logger.warning("%s: unknown timeframe folder, left alone", data_path)
            counts.skipped += 1
            continue
        if timeframe.minutes <= 1:
            continue

        counts.seen += 1
        frame = pd.read_parquet(data_path)
        if SPREAD_COLUMN not in frame.columns:
            counts.already += 1
            continue
        if MIN_SPREAD_M1_COLUMN in frame.columns:
            # both names on disk: a reconstruction was written next to the raw
            # field. Renaming would collide, so it is reported, not guessed at.
            logger.warning("%s carries both spread columns: left alone", data_path)
            counts.skipped += 1
            continue

        counts.changed += 1
        logger.info(
            "%s %s %s: spread -> %s (%d bars)",
            data_path.parents[1].name,
            timeframe.name,
            data_path.stem,
            MIN_SPREAD_M1_COLUMN,
            len(frame),
        )
        if not apply:
            continue
        renamed = frame.rename(columns={SPREAD_COLUMN: MIN_SPREAD_M1_COLUMN})
        staging = data_path.with_suffix(".parquet.tmp")
        renamed.to_parquet(staging, engine="pyarrow", compression="snappy")
        staging.replace(data_path)
    return counts


def migrate_runs(store: RunStore, apply: bool) -> Counts:
    """Marks stored runs that charged the aggregated column as a fill cost."""
    counts = Counts()
    if not store.root.exists():
        return counts

    for entry in sorted(store.root.iterdir()):
        if not (entry / META_FILE).exists():
            continue
        counts.seen += 1
        meta_path = entry / META_FILE
        config_path = entry / "config.json"
        if not config_path.exists():
            counts.skipped += 1
            continue

        config = json.loads(config_path.read_text(encoding="utf-8"))
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        timeframe_name = str(config.get("timeframe") or "M1")
        try:
            above_m1 = Timeframe.parse(timeframe_name).minutes > 1
        except ValueError:
            counts.skipped += 1
            continue

        affected = above_m1 and str(config.get("spread_mode")) in AGGREGATED_MODES
        if meta.get("aggregated_spread_cost") == affected:
            counts.already += 1
            continue

        counts.changed += 1
        if affected:
            logger.info(
                "%s: %s %s spread_mode=%s -> charged the aggregated column",
                entry.name,
                config.get("symbol"),
                timeframe_name,
                config.get("spread_mode"),
            )
        if not apply:
            continue
        meta["aggregated_spread_cost"] = affected
        meta_path.write_text(
            json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="write the changes")
    parser.add_argument("--cache-dir", default=str(REPO_ROOT / "data_cache"))
    parser.add_argument("--runs-dir", default=str(REPO_ROOT / "runs"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    cache = migrate_cache(Path(args.cache_dir), args.apply)
    runs = migrate_runs(RunStore(Path(args.runs_dir)), args.apply)

    mode = "applied" if args.apply else "dry run, nothing written"
    print(f"\n-- {mode} --")
    print(
        f"cache : {cache.seen} above-M1 files, {cache.changed} renamed, "
        f"{cache.already} already honest, {cache.skipped} left alone"
    )
    print(
        f"runs  : {runs.seen} runs, {runs.changed} re-marked, "
        f"{runs.already} already correct, {runs.skipped} unreadable"
    )
    if not args.apply:
        print("\nre-run with --apply to write")


if __name__ == "__main__":
    main()
