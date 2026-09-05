"""How much history the broker actually has, per instrument and timeframe.

The question this answers is not "what is in the cache" but "what could be in
the cache". A statistical test on H1 that only ever sees nine months has no
power no matter how it is written, and the first thing to establish is
whether that is a limit of the feed or a limit of what was downloaded.

Three claims are kept apart, because they are not the same claim:

- **depth**: the first bar the feed will return and how many bars that is. A
  probe asks for `max_bars` and, when exactly that many come back, the answer
  is a lower bound: the report says `truncated` rather than pretending the
  number is the depth.
- **inception**: whether that first bar can be true at all. This broker
  returns EURUSD H1 bars dated 1971, twenty-eight years before the euro
  existed. That is a fact about the world, and it is written down here.
- **cohort outlier**: whether a series claims history the rest of the feed
  does not have. Nine of ten instruments here start in 2020-2021; one
  claiming 1971 is an outlier against its own feed, and that is measured from
  the probes rather than asserted from a hardcoded calendar. It is a
  suspicion, not a verdict, and is reported as one.
"""
from __future__ import annotations

import logging
import statistics
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd

from core.data.provider import Timeframe
from core.serialization import json_safe

logger = logging.getLogger(__name__)

# Above this the terminal answers "invalid params" instead of clamping, so the
# probe would report zero depth for exactly the deepest series.
MAX_PROBE_BARS = 200_000

# Earliest date at which the underlying could have been quoted at all. Only
# instruments whose feed history predates their own existence need an entry.
#
#   EUR crosses: the euro was introduced 1999-01-01. Anything before that is
#   a synthetic ECU/DEM chain or pure backfill.
#   Crude benchmarks: WTI futures started 1983-03-30, Brent 1988-06-23.
INSTRUMENT_INCEPTION: dict[str, datetime] = {
    "EUR": datetime(1999, 1, 1, tzinfo=timezone.utc),
    "XTI": datetime(1983, 3, 30, tzinfo=timezone.utc),
    "WTI": datetime(1983, 3, 30, tzinfo=timezone.utc),
    "XBR": datetime(1988, 6, 23, tzinfo=timezone.utc),
    "BRENT": datetime(1988, 6, 23, tzinfo=timezone.utc),
}

# How far a series may start before the rest of the feed before it is called
# an outlier. A year absorbs the ordinary spread between listing dates.
COHORT_TOLERANCE = timedelta(days=365)


def inception(symbol: str) -> datetime | None:
    """Earliest date at which `symbol` could have been quoted, if known."""
    upper = symbol.upper()
    for prefix, moment in INSTRUMENT_INCEPTION.items():
        if upper.startswith(prefix):
            return moment
    return None


@dataclass(frozen=True)
class DepthProbe:
    """What one (symbol, timeframe) pair offers, from one source."""

    symbol: str
    timeframe: str
    source: str
    bars: int
    first_bar: datetime | None
    last_bar: datetime | None
    truncated: bool
    probed_at: datetime
    error: str | None = None

    @property
    def span_days(self) -> float | None:
        if self.first_bar is None or self.last_bar is None:
            return None
        return (self.last_bar - self.first_bar).total_seconds() / 86400.0

    @property
    def predates_inception(self) -> bool:
        """The series starts before the instrument could have been quoted."""
        floor = inception(self.symbol)
        return floor is not None and self.first_bar is not None and self.first_bar < floor

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(
            span_days=self.span_days,
            inception=inception(self.symbol),
            predates_inception=self.predates_inception,
        )
        return json_safe(payload)


@dataclass
class DepthReport:
    """Every probe, plus what the set of them says about any single one."""

    probes: list[DepthProbe]
    cohort_start: dict[str, datetime] = field(default_factory=dict)
    notes: dict[tuple[str, str], list[str]] = field(default_factory=dict)

    def note_lines(self, probe: DepthProbe) -> list[str]:
        return self.notes.get((probe.symbol, probe.timeframe), [])

    def usable_from(self, symbol: str, timeframe: str) -> datetime | None:
        """First bar to trust: the measured start, floored by inception.

        The cohort suspicion deliberately does not move this. Being older than
        the rest of the feed is a reason to look, not a reason to discard: the
        caller decides, with the note in front of it.
        """
        probe = next(
            (p for p in self.probes if p.symbol == symbol and p.timeframe == timeframe),
            None,
        )
        if probe is None or probe.first_bar is None:
            return None
        floor = inception(symbol)
        return max(probe.first_bar, floor) if floor else probe.first_bar

    def as_dict(self) -> dict[str, Any]:
        return json_safe(
            {
                "probes": [
                    {**probe.as_dict(), "notes": self.note_lines(probe)}
                    for probe in self.probes
                ],
                "cohort_start": self.cohort_start,
            }
        )


def probe_all(
    provider: Any,
    symbols: Sequence[str],
    timeframes: Sequence[str | Timeframe],
) -> list[DepthProbe]:
    """Probes every pair. A failure on one pair does not stop the others."""
    probes: list[DepthProbe] = []
    for symbol in symbols:
        for timeframe in timeframes:
            tf = Timeframe.parse(timeframe)
            try:
                probe = provider.probe_depth(symbol, tf)
            except Exception as exc:
                probe = DepthProbe(
                    symbol=symbol,
                    timeframe=tf.name,
                    source=getattr(provider, "source_name", "provider"),
                    bars=0,
                    first_bar=None,
                    last_bar=None,
                    truncated=False,
                    probed_at=datetime.now(timezone.utc),
                    error=f"{type(exc).__name__}: {exc}",
                )
            probes.append(probe)
            logger.info(
                "%s %s: %d bars from %s%s",
                probe.symbol,
                probe.timeframe,
                probe.bars,
                probe.first_bar,
                " (probe truncated)" if probe.truncated else "",
            )
    return probes


def build_report(
    probes: Sequence[DepthProbe], tolerance: timedelta = COHORT_TOLERANCE
) -> DepthReport:
    """Attaches to each probe what the rest of the feed says about it."""
    cohort: dict[str, datetime] = {}
    for timeframe in {p.timeframe for p in probes}:
        starts = sorted(
            p.first_bar
            for p in probes
            if p.timeframe == timeframe and p.first_bar is not None and not p.error
        )
        if len(starts) >= 3:
            cohort[timeframe] = statistics.median_low(starts)

    notes: dict[tuple[str, str], list[str]] = {}
    for probe in probes:
        lines: list[str] = []
        if probe.truncated:
            lines.append(
                f"probe capped at {probe.bars} bars: the depth is a lower bound, "
                f"not the depth"
            )
        floor = inception(probe.symbol)
        if probe.predates_inception and floor is not None:
            lines.append(
                f"starts {probe.first_bar:%Y-%m-%d}, before the instrument existed "
                f"({floor:%Y-%m-%d}): those bars are backfill, not history"
            )
        reference = cohort.get(probe.timeframe)
        if (
            reference is not None
            and probe.first_bar is not None
            and probe.first_bar < reference - tolerance
        ):
            lines.append(
                f"starts {(reference - probe.first_bar).days} days before the "
                f"median instrument on this feed ({reference:%Y-%m-%d}): older "
                f"than the feed itself, treat the excess as imported"
            )
        if lines:
            notes[(probe.symbol, probe.timeframe)] = lines
    return DepthReport(probes=list(probes), cohort_start=cohort, notes=notes)


def as_frame(report: DepthReport) -> pd.DataFrame:
    """The probes as one table, in the order they were taken."""
    if not report.probes:
        return pd.DataFrame()
    return pd.DataFrame(
        [{**p.as_dict(), "notes": "; ".join(report.note_lines(p))} for p in report.probes]
    )


def as_text(report: DepthReport) -> str:
    """Human-readable report. Every column is a measurement, not an estimate."""
    if not report.probes:
        return "no probes"
    lines = [
        f"{'symbol':<10} {'tf':<4} {'bars':>8}  {'first bar (UTC)':<17} "
        f"{'last bar (UTC)':<17} {'years':>6}",
        "-" * 72,
    ]
    for probe in report.probes:
        if probe.error:
            lines.append(f"{probe.symbol:<10} {probe.timeframe:<4} {'-':>8}  {probe.error}")
            continue
        years = (probe.span_days or 0.0) / 365.25
        lines.append(
            f"{probe.symbol:<10} {probe.timeframe:<4} {probe.bars:>8}  "
            f"{probe.first_bar:%Y-%m-%d %H:%M}  {probe.last_bar:%Y-%m-%d %H:%M}  "
            f"{years:>6.1f}"
        )
        for note in report.note_lines(probe):
            lines.append(f"{'':<15} ! {note}")
    return "\n".join(lines)
