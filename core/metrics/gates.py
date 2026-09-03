"""What happened to the signals that never became trades.

A backtest reports its trades. It does not, by default, report the signals it
threw away - and on some configurations that is most of them. On XTIUSD the
median spread is 29-30 points against a `max_spread_points` of 30, so the
spread gate was silently rejecting about half the signals: the run looked
like a strategy with few opportunities when it was a strategy that was almost
never allowed to trade.

This module turns that into a table: how many signals reached the entry
stage, how many each gate rejected, how many were executed, and a warning
whenever a single gate throws away more than a fifth of them. A gate doing
that is not a safety margin, it is the strategy.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from core.serialization import json_safe

# a gate above this share is not filtering the tail, it is choosing the trades
WARNING_SHARE = 0.20

# human labels for the stable codes produced by the engine
GATE_LABELS: Mapping[str, str] = {
    "max_open_positions": "Position already open (risk gate)",
    "position_open": "Signal while already in a position",
    "max_spread_points": "Spread above max_spread_points",
    "cooldown": "Cooldown after the previous exit",
    "max_trades_per_day": "Daily trade cap reached",
    "session": "Outside the allowed session",
    "insufficient_equity": "Equity too small for the minimum lot",
    "exit_indicator_warmup": "Exit indicator still in warm-up",
    "risk_gate": "Risk gate (unlabelled)",
}


@dataclass(frozen=True)
class GateRow:
    code: str
    label: str
    rejected: int
    share: float
    exceeds_threshold: bool


@dataclass(frozen=True)
class GateAccounting:
    """Signals in, trades out, and every rejection in between."""

    signals: int
    entry_attempts: int
    executed: int
    rejected: int
    executed_share: float | None
    rows: list[GateRow] = field(default_factory=list)
    threshold: float = WARNING_SHARE
    warnings: list[str] = field(default_factory=list)
    verdict: str = ""

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["rows"] = [asdict(row) for row in self.rows]
        return json_safe(payload)


def gate_accounting(
    blocked: Counter[str] | Mapping[str, int],
    signals: int,
    executed: int,
    entry_attempts: int = 0,
    threshold: float = WARNING_SHARE,
) -> GateAccounting:
    """Rejections per gate as a share of the signals the strategy produced."""
    counts = dict(blocked)
    rejected = int(sum(counts.values()))
    # every signal either becomes a trade or is rejected somewhere; when the
    # two do not add up (a position still open at the end of the data) the
    # signal count stays the denominator, because that is what the strategy
    # actually asked for
    denominator = max(signals, executed + rejected, 1)

    rows = [
        GateRow(
            code=code,
            label=GATE_LABELS.get(code, code),
            rejected=int(count),
            share=int(count) / denominator,
            exceeds_threshold=int(count) / denominator > threshold,
        )
        for code, count in sorted(counts.items(), key=lambda item: -item[1])
        if count
    ]

    warnings = [
        f"{row.label.lower()} rejected {row.rejected} of {denominator} signals "
        f"({row.share:.0%}): above {threshold:.0%}, this gate is selecting the "
        f"trades rather than trimming the tail"
        for row in rows
        if row.exceeds_threshold
    ]

    executed_share = executed / denominator if denominator else None
    if not rejected:
        verdict = f"All {executed} signals reached execution: no gate rejected anything."
    else:
        top = rows[0]
        verdict = (
            f"{executed} of {denominator} signals executed ({executed_share:.0%}); "
            f"{rejected} rejected, most of them by {top.label.lower()} "
            f"({top.rejected}, {top.share:.0%})."
        )

    return GateAccounting(
        signals=signals,
        entry_attempts=entry_attempts,
        executed=executed,
        rejected=rejected,
        executed_share=executed_share,
        rows=rows,
        threshold=threshold,
        warnings=warnings,
        verdict=verdict,
    )
