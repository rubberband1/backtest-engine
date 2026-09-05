"""Do two sources agree where they overlap?

Deep history can arrive from the live feed or be decoded out of the
terminal's .hc cache. Stitching them into one series is only legitimate if
they say the same thing where they both speak. They usually do; when they do
not, the seam is exactly where a backtest would silently change its answer,
and it has to be reported rather than averaged away.

Nothing is corrected here. The module measures the disagreement, in points,
and says how many bars it affects.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd

from core.data.provider import MIN_SPREAD_M1_COLUMN
from core.serialization import json_safe

logger = logging.getLogger(__name__)

COMPARED_COLUMNS: tuple[str, ...] = (
    "open",
    "high",
    "low",
    "close",
    "spread",
    MIN_SPREAD_M1_COLUMN,
)

# Fields already expressed in points, so a drift in them is not divided by
# the instrument's point size.
POINT_COLUMNS: frozenset[str] = frozenset({"spread", MIN_SPREAD_M1_COLUMN})

# A seam is called clean when no compared field moves by more than this many
# points. One point is the smallest quotable increment: below it the two
# sources are the same number written twice.
DEFAULT_TOLERANCE_POINTS = 1.0


@dataclass(frozen=True)
class ColumnDrift:
    column: str
    max_points: float
    mean_points: float
    bars_beyond_tolerance: int


@dataclass
class ContinuityReport:
    """What two sources say about the same stretch of time."""

    symbol: str
    timeframe: str
    left_source: str
    right_source: str
    overlap_start: datetime | None
    overlap_end: datetime | None
    overlap_bars: int
    only_in_left: int
    only_in_right: int
    tolerance_points: float
    drift: list[ColumnDrift] = field(default_factory=list)
    samples: list[str] = field(default_factory=list)
    disagreeing_bars: int = 0
    only_last_bar_disagrees: bool = False
    verdict: str = ""

    @property
    def agrees(self) -> bool:
        """Whether the two sources say the same thing where they both speak.

        Coverage is deliberately not part of this. One source holding fewer
        bars than the other is a fact about what was downloaded, not a
        contradiction; a contradiction is two different numbers for the same
        bar. The last shared bar is excused when it is the only offender: a
        snapshot taken while that bar was still forming disagrees with one
        taken a minute later, and always will.
        """
        if self.overlap_bars == 0:
            return False
        if self.disagreeing_bars == 0:
            return True
        return self.only_last_bar_disagrees

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["agrees"] = self.agrees
        return json_safe(payload)

    def as_text(self) -> str:
        lines = [
            f"Continuity - {self.symbol} {self.timeframe}: "
            f"{self.left_source} vs {self.right_source}",
            f"  overlap        : {self.overlap_bars} bars "
            f"({self.overlap_start} -> {self.overlap_end})",
            f"  only in {self.left_source:<10}: {self.only_in_left}",
            f"  only in {self.right_source:<10}: {self.only_in_right}",
        ]
        for entry in self.drift:
            lines.append(
                f"  {entry.column:<8}: max {entry.max_points:.2f} pt, "
                f"mean {entry.mean_points:.4f} pt, "
                f"{entry.bars_beyond_tolerance} bars beyond "
                f"{self.tolerance_points:g} pt"
            )
        for sample in self.samples:
            lines.append(f"    {sample}")
        lines.append(f"  disagreeing    : {self.disagreeing_bars} bars")
        lines.append(f"  verdict        : {self.verdict}")
        return "\n".join(lines)


def compare_sources(
    left: pd.DataFrame,
    right: pd.DataFrame,
    symbol: str,
    timeframe: str,
    point: float,
    left_source: str = "left",
    right_source: str = "right",
    tolerance_points: float = DEFAULT_TOLERANCE_POINTS,
    max_samples: int = 5,
) -> ContinuityReport:
    """Compares two bar frames on the timestamps they share.

    Differences are expressed in instrument points, never in price: a
    0.01 disagreement means something entirely different on gold and on
    USDJPY, and a threshold in price would be a different threshold per
    instrument without saying so.
    """
    shared = left.index.intersection(right.index)
    only_left = int(len(left.index.difference(right.index)))
    only_right = int(len(right.index.difference(left.index)))

    if len(shared) == 0:
        report = ContinuityReport(
            symbol=symbol,
            timeframe=timeframe,
            left_source=left_source,
            right_source=right_source,
            overlap_start=None,
            overlap_end=None,
            overlap_bars=0,
            only_in_left=only_left,
            only_in_right=only_right,
            tolerance_points=tolerance_points,
            verdict=(
                "the two sources share no timestamp: they cannot be checked "
                "against each other, and stitching them is an untested claim"
            ),
        )
        logger.warning("%s %s: no overlap between sources", symbol, timeframe)
        return report

    a, b = left.loc[shared], right.loc[shared]
    drift: list[ColumnDrift] = []
    offenders: pd.DatetimeIndex = pd.DatetimeIndex([], tz="UTC")
    for column in COMPARED_COLUMNS:
        if column not in a.columns or column not in b.columns:
            continue
        # the spread fields are already in points; prices are not
        scale = 1.0 if column in POINT_COLUMNS else point
        delta = (
            a[column].astype("float64") - b[column].astype("float64")
        ).abs() / scale
        beyond = delta > tolerance_points
        drift.append(
            ColumnDrift(
                column=column,
                max_points=float(delta.max()),
                mean_points=float(delta.mean()),
                bars_beyond_tolerance=int(beyond.sum()),
            )
        )
        offenders = offenders.union(delta.index[beyond])

    samples = [
        f"{moment}: "
        + ", ".join(
            f"{c} {float(a.loc[moment, c]):.5f} vs {float(b.loc[moment, c]):.5f}"
            for c in COMPARED_COLUMNS
            if c in a.columns and c in b.columns
        )
        for moment in offenders[:max_samples]
    ]

    last_shared = shared.max()
    only_last = bool(len(offenders) == 1 and offenders[0] == last_shared)
    coverage = (
        f"coverage: {only_left} bars only in {left_source}, {only_right} only in "
        f"{right_source}"
        if (only_left or only_right)
        else "coverage: identical on both sides"
    )

    if len(offenders) == 0:
        verdict = (
            f"{len(shared)} shared bars agree within {tolerance_points:g} "
            f"point(s); {coverage}. The sources can be stitched at this seam"
        )
    elif only_last:
        verdict = (
            f"the only disagreement is on {last_shared}, the last shared bar: "
            f"one snapshot was taken while it was still forming, which is "
            f"expected and not a seam problem. The {len(shared) - 1} closed "
            f"bars before it agree within {tolerance_points:g} point(s); "
            f"{coverage}"
        )
    else:
        verdict = (
            f"{len(offenders)} of {len(shared)} shared bars differ by more than "
            f"{tolerance_points:g} point(s), and not only the last one: the seam "
            f"is not clean and the result depends on which source filled it. "
            f"{coverage}"
        )

    report = ContinuityReport(
        symbol=symbol,
        timeframe=timeframe,
        left_source=left_source,
        right_source=right_source,
        overlap_start=shared.min().to_pydatetime(),
        overlap_end=shared.max().to_pydatetime(),
        overlap_bars=int(len(shared)),
        only_in_left=only_left,
        only_in_right=only_right,
        tolerance_points=tolerance_points,
        drift=drift,
        samples=samples,
        disagreeing_bars=int(len(offenders)),
        only_last_bar_disagrees=only_last,
        verdict=verdict,
    )
    logger.info(
        "%s %s: %d shared bars, %d disagree%s",
        symbol,
        timeframe,
        len(shared),
        len(offenders),
        " (the open bar only)" if only_last else "",
    )
    return report


def stitch(
    preferred: pd.DataFrame, fallback: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Joins two frames, keeping `preferred` wherever both have a bar.

    Returns the joined frame and how many bars each side contributed, so the
    caller can record provenance instead of guessing at it afterwards.
    """
    if fallback.empty:
        return preferred, {"preferred": int(len(preferred)), "fallback": 0}
    if preferred.empty:
        return fallback, {"preferred": 0, "fallback": int(len(fallback))}

    extra_index = fallback.index.difference(preferred.index)
    joined = pd.concat([preferred, fallback.loc[extra_index]]).sort_index()
    joined.index.name = "time"
    return joined, {
        "preferred": int(len(preferred)),
        "fallback": int(len(extra_index)),
    }


def summarize(reports: list[ContinuityReport]) -> str:
    """One line per seam, plus the count that failed."""
    if not reports:
        return "no seams to check: every series came from a single source"
    failed = [r for r in reports if not r.agrees]
    lines = [r.as_text() for r in reports]
    lines.append(
        f"{len(reports) - len(failed)} of {len(reports)} seams agree within "
        f"tolerance"
        if not failed
        else f"{len(failed)} of {len(reports)} seams do NOT agree: "
        + ", ".join(f"{r.symbol} {r.timeframe}" for r in failed)
    )
    return "\n".join(lines)
