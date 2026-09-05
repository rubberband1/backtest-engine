"""Local Parquet cache.

One file per (symbol, timeframe, year), plus a JSON metadata file next to it.
No pickle: the format must stay readable by other tools and stable across
pandas versions.

Coverage is stored as a list of UTC intervals, not a single range: two
requests far apart in time must not make the hole in between pass for
"already downloaded".

Alongside coverage, each interval records **where it came from**. Deep
history can arrive from the live feed or be decoded out of the terminal's own
.hc cache, and the two are not interchangeable: the .hc file holds whatever
was downloaded up to the last connection, which may be stale or partial.
Losing that distinction turns "my backtest used broker data" into an
unverifiable claim, so provenance travels with the bars.
"""
from __future__ import annotations

import json
import logging
import shutil
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone, tzinfo
from pathlib import Path

import pandas as pd

from core.data.provider import (
    Timeframe,
    bar_columns_for,
    empty_bars,
    normalize_bars,
    rename_aggregated_spread,
)

logger = logging.getLogger(__name__)

# 2 added per-interval provenance. Version 1 files are still read: their
# coverage is kept and reported as provenance "unrecorded", which is the
# truth about them rather than a guess dressed up as one.
SCHEMA_VERSION = 2
READABLE_SCHEMA_VERSIONS = (1, 2)

# What supplied a stretch of bars, when the meta file predates provenance.
UNRECORDED_SOURCE = "unrecorded"

Interval = tuple[datetime, datetime]
FetchFn = Callable[[str, Timeframe, datetime, datetime], pd.DataFrame]


def _utc(moment: datetime) -> datetime:
    if moment.tzinfo is None:
        return moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)


def merge_intervals(intervals: Iterable[Interval]) -> list[Interval]:
    """Merges overlapping or contiguous intervals, ordered by start."""
    ordered = sorted((_utc(a), _utc(b)) for a, b in intervals if b > a)
    merged: list[Interval] = []
    for start, end in ordered:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def subtract_intervals(target: Interval, covered: Sequence[Interval]) -> list[Interval]:
    """Parts of `target` not covered by `covered`."""
    start, end = _utc(target[0]), _utc(target[1])
    holes: list[Interval] = []
    cursor = start
    for c_start, c_end in merge_intervals(covered):
        if c_end <= cursor:
            continue
        if c_start >= end:
            break
        if c_start > cursor:
            holes.append((cursor, min(c_start, end)))
        cursor = max(cursor, c_end)
        if cursor >= end:
            break
    if cursor < end:
        holes.append((cursor, end))
    return holes


def year_bounds(year: int) -> Interval:
    return (
        datetime(year, 1, 1, tzinfo=timezone.utc),
        datetime(year + 1, 1, 1, tzinfo=timezone.utc),
    )


def split_by_year(interval: Interval) -> list[tuple[int, Interval]]:
    start, end = _utc(interval[0]), _utc(interval[1])
    out: list[tuple[int, Interval]] = []
    for year in range(start.year, end.year + 1):
        y_start, y_end = year_bounds(year)
        piece = (max(start, y_start), min(end, y_end))
        if piece[1] > piece[0]:
            out.append((year, piece))
    return out


@dataclass(frozen=True)
class SourceInterval:
    """Which source supplied one stretch of the coverage, and when."""

    source: str
    start: datetime
    end: datetime
    recorded_at: datetime

    def to_json(self) -> dict[str, object]:
        return {
            "source": self.source,
            "start_utc": self.start.isoformat(),
            "end_utc": self.end.isoformat(),
            "recorded_at_utc": self.recorded_at.isoformat(),
        }

    @classmethod
    def from_json(cls, payload: dict[str, object]) -> SourceInterval:
        return cls(
            source=str(payload["source"]),
            start=datetime.fromisoformat(str(payload["start_utc"])),
            end=datetime.fromisoformat(str(payload["end_utc"])),
            recorded_at=datetime.fromisoformat(str(payload["recorded_at_utc"])),
        )


@dataclass
class CacheMeta:
    """Metadata stored next to each Parquet file."""

    schema_version: int
    symbol: str
    timeframe: str
    year: int
    coverage: list[Interval] = field(default_factory=list)
    provenance: list[SourceInterval] = field(default_factory=list)
    rows: int = 0
    first_bar: datetime | None = None
    last_bar: datetime | None = None
    downloaded_at: datetime | None = None
    server_timezone: str | None = None

    @property
    def sources(self) -> list[str]:
        """Distinct sources behind this file, in first-recorded order."""
        seen: list[str] = []
        for entry in self.provenance:
            if entry.source not in seen:
                seen.append(entry.source)
        # coverage with no provenance entry behind it predates the field
        if self.coverage and not self.provenance:
            seen.append(UNRECORDED_SOURCE)
        return seen

    def to_json(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "year": self.year,
            "coverage_utc": [[a.isoformat(), b.isoformat()] for a, b in self.coverage],
            "provenance": [entry.to_json() for entry in self.provenance],
            "rows": self.rows,
            "first_bar_utc": self.first_bar.isoformat() if self.first_bar else None,
            "last_bar_utc": self.last_bar.isoformat() if self.last_bar else None,
            "downloaded_at_utc": (
                self.downloaded_at.isoformat() if self.downloaded_at else None
            ),
            "server_timezone": self.server_timezone,
        }

    @classmethod
    def from_json(cls, payload: dict[str, object]) -> CacheMeta:
        def parse(value: object) -> datetime | None:
            return datetime.fromisoformat(str(value)) if value else None

        coverage_raw = payload.get("coverage_utc") or []
        coverage = [
            (datetime.fromisoformat(a), datetime.fromisoformat(b))
            for a, b in coverage_raw
        ]
        provenance = [
            SourceInterval.from_json(entry)
            for entry in (payload.get("provenance") or [])
        ]
        return cls(
            schema_version=int(payload["schema_version"]),
            symbol=str(payload["symbol"]),
            timeframe=str(payload["timeframe"]),
            year=int(payload["year"]),
            coverage=coverage,
            provenance=provenance,
            rows=int(payload.get("rows", 0)),
            first_bar=parse(payload.get("first_bar_utc")),
            last_bar=parse(payload.get("last_bar_utc")),
            downloaded_at=parse(payload.get("downloaded_at_utc")),
            server_timezone=(
                str(payload["server_timezone"]) if payload.get("server_timezone") else None
            ),
        )


class ParquetCache:
    """On-disk archive of already-downloaded bars.

    Layout: `<root>/<symbol>/<timeframe>/<year>.parquet` + `<year>.json`.
    """

    def __init__(self, root: Path | str = Path("data_cache")) -> None:
        self.root = Path(root)

    # -- paths -----------------------------------------------------------

    @staticmethod
    def _slug(value: str) -> str:
        return value.replace("/", "_").replace("\\", "_")

    def _dir(self, symbol: str, timeframe: Timeframe) -> Path:
        return self.root / self._slug(symbol) / timeframe.name

    def paths(self, symbol: str, timeframe: Timeframe, year: int) -> tuple[Path, Path]:
        base = self._dir(symbol, timeframe)
        return base / f"{year}.parquet", base / f"{year}.json"

    # -- reading ---------------------------------------------------------

    def read_meta(self, symbol: str, timeframe: Timeframe, year: int) -> CacheMeta | None:
        _, meta_path = self.paths(symbol, timeframe, year)
        if not meta_path.exists():
            return None
        meta = CacheMeta.from_json(json.loads(meta_path.read_text(encoding="utf-8")))
        if meta.schema_version not in READABLE_SCHEMA_VERSIONS:
            logger.warning(
                "unreadable schema %d in %s (this build reads %s): treating the "
                "cache as empty",
                meta.schema_version,
                meta_path,
                READABLE_SCHEMA_VERSIONS,
            )
            return None
        return meta

    def read_year(self, symbol: str, timeframe: Timeframe, year: int) -> pd.DataFrame:
        """Bars for one year, with the spread field named for its timeframe.

        This is the door every backtest's data comes through, so it is where
        the rename happens: above M1 the file's `spread` column comes back as
        `min_spread_m1`, whatever the file on disk calls it. Nothing
        downstream can then charge it as a fill cost by writing `bars["spread"]`
        - that name raises instead.
        """
        data_path, _ = self.paths(symbol, timeframe, year)
        if not data_path.exists():
            return empty_bars(timeframe)
        frame = pd.read_parquet(data_path)
        frame.index = pd.DatetimeIndex(frame.index).tz_convert("UTC")
        frame.index.name = "time"
        frame = rename_aggregated_spread(frame, timeframe)
        return normalize_bars(frame, bar_columns_for(timeframe))

    def coverage(self, symbol: str, timeframe: Timeframe, year: int) -> list[Interval]:
        meta = self.read_meta(symbol, timeframe, year)
        return merge_intervals(meta.coverage) if meta else []

    def missing_ranges(
        self, symbol: str, timeframe: Timeframe, start: datetime, end: datetime
    ) -> list[Interval]:
        """Sub-intervals not yet cached, already split by year."""
        holes: list[Interval] = []
        for year, piece in split_by_year((start, end)):
            holes.extend(subtract_intervals(piece, self.coverage(symbol, timeframe, year)))
        return merge_intervals(holes)

    # -- writing ---------------------------------------------------------

    def write_year(
        self,
        symbol: str,
        timeframe: Timeframe,
        year: int,
        bars: pd.DataFrame,
        covered: Sequence[Interval],
        server_timezone: str | None = None,
        source: str = UNRECORDED_SOURCE,
    ) -> CacheMeta:
        data_path, meta_path = self.paths(symbol, timeframe, year)
        data_path.parent.mkdir(parents=True, exist_ok=True)

        existing = self.read_year(symbol, timeframe, year)
        columns = bar_columns_for(timeframe)
        incoming = rename_aggregated_spread(bars, timeframe)
        merged = (
            normalize_bars(pd.concat([existing, incoming]), columns)
            if len(bars)
            else existing
        )
        y_start, y_end = year_bounds(year)
        merged = merged[(merged.index >= y_start) & (merged.index < y_end)]
        merged.to_parquet(data_path, engine="pyarrow", compression="snappy")

        previous = self.read_meta(symbol, timeframe, year)
        coverage = merge_intervals(list(previous.coverage if previous else []) + list(covered))
        now = datetime.now(timezone.utc)
        # provenance intervals are appended, never merged: two stretches from
        # two sources must stay distinguishable even when they are adjacent
        provenance = list(previous.provenance if previous else []) + [
            SourceInterval(source=source, start=_utc(a), end=_utc(b), recorded_at=now)
            for a, b in covered
            if b > a
        ]
        meta = CacheMeta(
            schema_version=SCHEMA_VERSION,
            symbol=symbol,
            timeframe=timeframe.name,
            year=year,
            coverage=coverage,
            provenance=provenance,
            rows=int(len(merged)),
            first_bar=merged.index.min().to_pydatetime() if len(merged) else None,
            last_bar=merged.index.max().to_pydatetime() if len(merged) else None,
            downloaded_at=now,
            server_timezone=server_timezone
            or (previous.server_timezone if previous else None),
        )
        meta_path.write_text(
            json.dumps(meta.to_json(), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.debug("cache updated: %s (%d bars)", data_path, meta.rows)
        return meta

    # -- main API --------------------------------------------------------

    def get_or_fetch(
        self,
        symbol: str,
        timeframe: Timeframe | str,
        start: datetime,
        end: datetime,
        fetch: FetchFn,
        server_timezone: tzinfo | str | None = None,
        source: str | None = None,
    ) -> pd.DataFrame:
        """Returns the requested bars, downloading only the missing holes.

        `source` names what `fetch` reads from and is written into the
        metadata for every hole it fills. Left out, it is taken from the
        fetcher itself when it can say (a bound provider method), because a
        cache that cannot name its sources cannot be audited later.
        """
        tf = Timeframe.parse(timeframe)
        start_utc, end_utc = _utc(start), _utc(end)
        if end_utc <= start_utc:
            return empty_bars(tf)

        tz_label = str(server_timezone) if server_timezone is not None else None
        origin = source or getattr(
            getattr(fetch, "__self__", None), "source_name", UNRECORDED_SOURCE
        )
        holes = self.missing_ranges(symbol, tf, start_utc, end_utc)
        if not holes:
            logger.info("%s %s: request served entirely from the cache", symbol, tf.name)

        for hole_start, hole_end in holes:
            logger.info(
                "%s %s: downloading the hole %s -> %s", symbol, tf.name, hole_start, hole_end
            )
            downloaded = fetch(symbol, tf, hole_start, hole_end)
            for year, piece in split_by_year((hole_start, hole_end)):
                y_start, y_end = piece
                slice_ = (
                    downloaded[(downloaded.index >= y_start) & (downloaded.index < y_end)]
                    if len(downloaded)
                    else empty_bars(tf)
                )
                self.write_year(symbol, tf, year, slice_, [piece], tz_label, origin)

        parts = [
            self.read_year(symbol, tf, year) for year, _ in split_by_year((start_utc, end_utc))
        ]
        parts = [p for p in parts if len(p)]
        if not parts:
            return empty_bars(tf)
        out = normalize_bars(pd.concat(parts), bar_columns_for(tf))
        return out[(out.index >= start_utc) & (out.index < end_utc)]

    # -- invalidation ----------------------------------------------------

    def invalidate(
        self,
        symbol: str | None = None,
        timeframe: Timeframe | str | None = None,
        year: int | None = None,
    ) -> list[Path]:
        """Explicitly deletes parts of the cache. Returns what was removed."""
        removed: list[Path] = []

        if symbol is None:
            if self.root.exists():
                shutil.rmtree(self.root)
                removed.append(self.root)
            return removed

        if timeframe is None:
            target = self.root / self._slug(symbol)
            if target.exists():
                shutil.rmtree(target)
                removed.append(target)
            return removed

        tf = Timeframe.parse(timeframe)
        if year is None:
            target = self._dir(symbol, tf)
            if target.exists():
                shutil.rmtree(target)
                removed.append(target)
            return removed

        for path in self.paths(symbol, tf, year):
            if path.exists():
                path.unlink()
                removed.append(path)
        return removed
