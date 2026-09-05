"""Expected versus realized: what the simulation said, what the market did.

This is the deliverable the rest of the project exists to make possible. The
replay test proves the simulated system and the live system are the *same
system*; this module measures what that system costs in reality that it did
not cost in simulation.

Three questions, answered separately because they have different causes:

1. **Slippage.** For every trade both sides took, the difference between the
   price the engine expected at the fill and the price the broker gave, in
   points and in money. A backtest fills at the bar open plus the spread; a
   live order fills at whatever the book holds a few hundred milliseconds
   later.
2. **Missing and extra signals.** A signal present in one record and not the
   other, with the reason the diary gives: a rejected order, a gate that
   fired live and not in simulation, a bar the runner never saw because the
   process was down. A backtest never has downtime, and pretending otherwise
   is how a strategy's live results become inexplicable.
3. **The PnL difference, decomposed.** Not one number but a split: how much
   of the gap is slippage on trades both took, how much is trades only one of
   them took, and how much is left over. The leftover is the interesting part
   - it is the part nobody has an explanation for yet.

Nothing here reconciles the two records into an average. Where they differ,
the difference is the result.

**A comparison is only valid between the same engine and the same
instrument.** Both sides of every number here are money, and money is
`tick_value / tick_size` times a price difference. Re-reading the spec after
the diary was written rescales one side of the diff by a factor of its own,
which lands in `pnl_from_slippage` and reads exactly like a cost. So the
diary's pinned spec is used when it has one, and a version or spec mismatch
is reported as a mismatch rather than being quietly absorbed into a total.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from core.data.provider import SYMBOL_SPEC_COST_FIELDS, SymbolSpec
from core.engine.costs import money_per_point
from core.live.journal import Journal, environment
from core.serialization import json_safe
from core.version import ENGINE_VERSION

logger = logging.getLogger(__name__)

# Two records refer to the same trade when they entered on the same bar.
# Entry time is the engine's decision timestamp on both sides, so it matches
# exactly or the trades are genuinely different.
MATCH_ON = "entry_time"


@dataclass(frozen=True)
class TradeDeviation:
    """One trade both records hold, and where they disagree."""

    entry_time: datetime
    direction: int
    entry_slippage_points: float | None
    exit_slippage_points: float | None
    entry_slippage_money: float | None
    pnl_difference: float
    lots_expected: float
    lots_realized: float
    exit_reason_expected: str
    exit_reason_realized: str

    @property
    def exit_reason_differs(self) -> bool:
        return self.exit_reason_expected != self.exit_reason_realized

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["exit_reason_differs"] = self.exit_reason_differs
        return json_safe(payload)


@dataclass(frozen=True)
class UnmatchedTrade:
    """A trade only one of the two records has."""

    entry_time: datetime
    direction: int
    net_pnl: float
    exit_reason: str
    side: str
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return json_safe(asdict(self))


@dataclass
class ComparisonReport:
    """The whole expected-versus-realized picture for one period."""

    symbol: str
    timeframe: str
    period_start: datetime | None
    period_end: datetime | None
    expected_trades: int
    realized_trades: int
    matched: int
    deviations: list[TradeDeviation] = field(default_factory=list)
    only_expected: list[UnmatchedTrade] = field(default_factory=list)
    only_realized: list[UnmatchedTrade] = field(default_factory=list)
    expected_pnl: float = 0.0
    realized_pnl: float = 0.0
    pnl_from_slippage: float = 0.0
    pnl_from_unmatched: float = 0.0
    pnl_unexplained: float = 0.0
    median_entry_slippage_points: float | None = None
    p90_entry_slippage_points: float | None = None
    rejected_orders: int = 0
    partial_fills: int = 0
    bars_processed: int = 0
    verdict: str = ""
    warnings: list[str] = field(default_factory=list)
    diary_engine_version: str | None = None
    engine_version: str = ENGINE_VERSION
    spec_source: str = ""
    spec_matches_diary: bool | None = None
    comparable: bool = True

    def as_dict(self) -> dict[str, Any]:
        return json_safe(
            {
                **{
                    k: v
                    for k, v in asdict(self).items()
                    if k not in ("deviations", "only_expected", "only_realized")
                },
                "deviations": [d.as_dict() for d in self.deviations],
                "only_expected": [t.as_dict() for t in self.only_expected],
                "only_realized": [t.as_dict() for t in self.only_realized],
            }
        )

    def as_text(self, max_rows: int = 10) -> str:
        lines = [
            f"Expected vs realized - {self.symbol} {self.timeframe}",
            f"  period            : {self.period_start} -> {self.period_end}",
            f"  trades            : {self.expected_trades} expected, "
            f"{self.realized_trades} realized, {self.matched} matched",
            f"  PnL               : {self.expected_pnl:+.2f} expected, "
            f"{self.realized_pnl:+.2f} realized "
            f"({self.realized_pnl - self.expected_pnl:+.2f})",
            f"    of which slippage : {self.pnl_from_slippage:+.2f}",
            f"    unmatched trades  : {self.pnl_from_unmatched:+.2f}",
            f"    unexplained       : {self.pnl_unexplained:+.2f}",
            f"  entry slippage    : median "
            f"{_fmt(self.median_entry_slippage_points)} pt, "
            f"p90 {_fmt(self.p90_entry_slippage_points)} pt",
            f"  rejected orders   : {self.rejected_orders}",
            f"  partial fills     : {self.partial_fills}",
            f"  engine            : diary {self.diary_engine_version or 'unrecorded'}"
            f", comparing with {self.engine_version}",
            f"  instrument spec   : {self.spec_source}",
        ]
        for deviation in self.deviations[:max_rows]:
            if deviation.entry_slippage_points:
                lines.append(
                    f"    {deviation.entry_time}: entry slipped "
                    f"{deviation.entry_slippage_points:+.1f} pt, PnL "
                    f"{deviation.pnl_difference:+.2f}"
                )
        for trade in (self.only_expected + self.only_realized)[:max_rows]:
            lines.append(f"    {trade.entry_time}: {trade.reason}")
        lines.append(f"  verdict           : {self.verdict}")
        for warning in self.warnings:
            lines.append(f"  ! {warning}")
        return "\n".join(lines)


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.2f}"


def realized_fill_prices(journal: Journal) -> dict[datetime, dict[str, Any]]:
    """Broker fills from the diary, keyed by the bar that triggered them."""
    fills: dict[datetime, dict[str, Any]] = {}
    for event in journal.events():
        if event.kind != "order_result" or event.bar_time is None:
            continue
        result = event.detail.get("result") or {}
        fills[event.bar_time] = {
            "filled_price": result.get("filled_price"),
            "expected_price": event.detail.get("expected_price"),
            "filled_lots": result.get("filled_lots"),
            "accepted": result.get("accepted"),
            "retcode_name": result.get("retcode_name"),
            "partial": result.get("partial"),
        }
    return fills


def compare(
    expected: pd.DataFrame,
    journal: Journal,
    symbol_spec: SymbolSpec,
    timeframe: str,
) -> ComparisonReport:
    """Diffs a backtest's trades against what the live diary recorded.

    `expected` is the trade table of a backtest over the same period and the
    same spec. The diary supplies both the realized trades and the broker's
    answers, which is what makes slippage measurable at all: a trade table
    alone cannot say what price was asked for.

    `symbol_spec` is only used to turn points into money for the slippage
    columns. It should be the spec the diary pinned - the caller is expected
    to have run `expected` against that same spec - and when it is not, the
    report says so instead of letting the difference pass as a cost.
    """
    realized = journal.trades()
    fills = realized_fill_prices(journal)
    bars = sum(1 for event in journal.events() if event.kind == "bar")
    rejected = sum(
        1
        for event in journal.events()
        if event.kind == "order_result"
        and not (event.detail.get("result") or {}).get("accepted", True)
        and not (event.detail.get("result") or {}).get("dry_run", False)
    )
    partials = sum(
        1
        for event in journal.events()
        if event.kind == "order_result"
        and (event.detail.get("result") or {}).get("partial", False)
    )

    symbol = symbol_spec.name
    expected_pnl = float(expected["net_pnl"].sum()) if len(expected) else 0.0
    realized_pnl = float(realized["net_pnl"].sum()) if len(realized) else 0.0

    left = expected.set_index(MATCH_ON) if len(expected) else expected
    right = realized.set_index(MATCH_ON) if len(realized) else realized
    shared = (
        left.index.intersection(right.index)
        if len(left) and len(right)
        else pd.DatetimeIndex([], tz="UTC")
    )

    deviations: list[TradeDeviation] = []
    slippage_pnl = 0.0
    entry_slippage: list[float] = []
    for moment in shared:
        a, b = left.loc[moment], right.loc[moment]
        if isinstance(a, pd.DataFrame) or isinstance(b, pd.DataFrame):
            # two trades entered on the same bar cannot happen in a
            # one-position engine; if it does, the record is not comparable
            continue
        fill = fills.get(moment.to_pydatetime(), {})
        filled_price = fill.get("filled_price")
        entry_slip = (
            (float(filled_price) - float(a["entry_price"])) / symbol_spec.point
            if filled_price
            else None
        )
        if entry_slip is not None:
            entry_slippage.append(entry_slip * (1 if a["direction"] > 0 else -1))
        difference = float(b["net_pnl"]) - float(a["net_pnl"])
        slippage_pnl += difference
        deviations.append(
            TradeDeviation(
                entry_time=moment.to_pydatetime(),
                direction=int(a["direction"]),
                entry_slippage_points=entry_slip,
                exit_slippage_points=None,
                entry_slippage_money=(
                    entry_slip * money_per_point(symbol_spec, float(b["lots"]))
                    if entry_slip is not None
                    else None
                ),
                pnl_difference=difference,
                lots_expected=float(a["lots"]),
                lots_realized=float(b["lots"]),
                exit_reason_expected=str(a["exit_reason"]),
                exit_reason_realized=str(b["exit_reason"]),
            )
        )

    only_expected = [
        UnmatchedTrade(
            entry_time=moment.to_pydatetime(),
            direction=int(left.loc[moment, "direction"]),
            net_pnl=float(left.loc[moment, "net_pnl"]),
            exit_reason=str(left.loc[moment, "exit_reason"]),
            side="expected",
            reason=_why_missing(journal, moment.to_pydatetime()),
        )
        for moment in (left.index.difference(right.index) if len(left) else [])
    ]
    only_realized = [
        UnmatchedTrade(
            entry_time=moment.to_pydatetime(),
            direction=int(right.loc[moment, "direction"]),
            net_pnl=float(right.loc[moment, "net_pnl"]),
            exit_reason=str(right.loc[moment, "exit_reason"]),
            side="realized",
            reason="taken live and absent from the backtest of the same period",
        )
        for moment in (right.index.difference(left.index) if len(right) else [])
    ]

    unmatched_pnl = sum(t.net_pnl for t in only_realized) - sum(
        t.net_pnl for t in only_expected
    )
    total_difference = realized_pnl - expected_pnl
    unexplained = total_difference - slippage_pnl - unmatched_pnl

    slips = np.asarray(entry_slippage, dtype="float64")
    warnings: list[str] = []
    if len(realized) == 0:
        warnings.append(
            "the diary holds no closed trade: either the runner has not traded "
            "yet or it ran in dry run, and nothing here measures reality"
        )
    if rejected:
        warnings.append(
            f"{rejected} order(s) were rejected by the broker: the live system "
            f"took fewer trades than the strategy asked for, and the difference "
            f"is not slippage"
        )

    env = environment(journal)
    comparable = True
    if env.engine_version is None:
        comparable = False
        warnings.append(
            "the diary does not record which engine wrote it, so nothing here "
            "can be attributed: a difference may be reality or may be a change "
            "in the engine between then and now"
        )
    elif env.engine_version != ENGINE_VERSION:
        comparable = False
        warnings.append(
            f"the diary was written by engine {env.engine_version} and this is "
            f"{ENGINE_VERSION}: the two records were produced by different code, "
            f"so the difference below is not slippage and must not be read as it"
        )

    if env.symbol_spec is None:
        spec_source = (
            "read from the current environment - the diary pinned none, so a "
            "drift since it was written is invisible here"
        )
        spec_matches = None
        comparable = False
        warnings.append(
            "the diary did not pin the instrument spec it traded. Every money "
            "column scales with tick_value, which moves with an FX rate, so a "
            "non-zero difference here may be nothing but a re-read spec"
        )
    else:
        drifted = [
            f"{name} {getattr(env.symbol_spec, name)!r} -> {getattr(symbol_spec, name)!r}"
            for name in SYMBOL_SPEC_COST_FIELDS
            if getattr(env.symbol_spec, name) != getattr(symbol_spec, name)
        ]
        spec_matches = not drifted
        if drifted:
            comparable = False
            spec_source = "the diary's pinned spec disagrees with the one used here"
            warnings.append(
                "the backtest was run against a different instrument than the "
                "diary traded (" + "; ".join(drifted[:4]) + "): every money "
                "column is scaled by tick_value, so this difference is arithmetic, "
                "not slippage"
            )
        else:
            spec_source = "pinned by the diary, and the same one used here"

    verdict = _verdict(
        len(expected), len(realized), len(shared), total_difference,
        slippage_pnl, unmatched_pnl, unexplained, slips,
    )
    if not comparable:
        verdict = (
            "NOT COMPARABLE - the two records did not come from the same engine "
            "and instrument; see the warnings. " + verdict
        )

    return ComparisonReport(
        symbol=symbol,
        timeframe=timeframe,
        period_start=(
            min(
                [t for t in [_first(expected), _first(realized)] if t is not None],
                default=None,
            )
        ),
        period_end=(
            max(
                [t for t in [_last(expected), _last(realized)] if t is not None],
                default=None,
            )
        ),
        expected_trades=int(len(expected)),
        realized_trades=int(len(realized)),
        matched=int(len(shared)),
        deviations=deviations,
        only_expected=only_expected,
        only_realized=only_realized,
        expected_pnl=expected_pnl,
        realized_pnl=realized_pnl,
        pnl_from_slippage=slippage_pnl,
        pnl_from_unmatched=unmatched_pnl,
        pnl_unexplained=unexplained,
        median_entry_slippage_points=float(np.median(slips)) if len(slips) else None,
        p90_entry_slippage_points=(
            float(np.quantile(slips, 0.90)) if len(slips) else None
        ),
        rejected_orders=rejected,
        partial_fills=partials,
        bars_processed=bars,
        verdict=verdict,
        warnings=warnings,
        diary_engine_version=env.engine_version,
        engine_version=ENGINE_VERSION,
        spec_source=spec_source,
        spec_matches_diary=spec_matches,
        comparable=comparable,
    )


def _why_missing(journal: Journal, moment: datetime) -> str:
    """What the diary says about a trade the backtest took and the runner did not."""
    for event in journal.events():
        if event.bar_time != moment:
            continue
        if event.kind == "order_result":
            result = event.detail.get("result") or {}
            if not result.get("accepted"):
                return (
                    f"the order was sent and refused "
                    f"({result.get('retcode_name', 'unknown')})"
                )
        if event.kind == "gate_rejected":
            return f"rejected by the {event.detail.get('gate', 'unknown')} gate"
    if not any(event.bar_time == moment for event in journal.events()):
        return (
            "the runner never saw this bar: it was not running, or the bar "
            "arrived while it was disconnected"
        )
    return "the runner saw the bar and did not open a position on it"


def _first(frame: pd.DataFrame) -> datetime | None:
    return frame["entry_time"].min().to_pydatetime() if len(frame) else None


def _last(frame: pd.DataFrame) -> datetime | None:
    return frame["exit_time"].max().to_pydatetime() if len(frame) else None


def _verdict(
    expected: int,
    realized: int,
    matched: int,
    total: float,
    slippage: float,
    unmatched: float,
    unexplained: float,
    slips: np.ndarray,
) -> str:
    if realized == 0:
        return (
            f"nothing to compare: the backtest took {expected} trades and the "
            f"diary holds none"
        )
    if matched == 0:
        return (
            f"the two records share no trade: {expected} expected and {realized} "
            f"realized, none entered on the same bar. They are not describing "
            f"the same period, or the runner was not running when it mattered"
        )
    slip_text = (
        f"median entry slippage {np.median(slips):+.1f} points"
        if len(slips)
        else "no fill prices in the diary, so slippage is unmeasured"
    )
    return (
        f"{matched} of {expected} expected trades were also taken live. "
        f"Reality cost {total:+.2f} against the simulation, of which "
        f"{slippage:+.2f} on the trades both took, {unmatched:+.2f} on trades "
        f"only one of them took and {unexplained:+.2f} unaccounted for. "
        f"{slip_text.capitalize()}"
    )
