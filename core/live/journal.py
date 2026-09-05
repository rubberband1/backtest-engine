"""The execution diary: every decision the runner made, on disk, in order.

One JSON object per line, append-only. The format is deliberately dull: it
has to be readable by a text editor six months from now, survive a crash
mid-write without corrupting what came before, and be loadable into pandas
without a schema migration.

What goes in is the whole decision, not its outcome: the bar, the indicator
values behind the signal, the signal, what each gate said, the order that was
sent, what the broker answered and the price it actually filled at. Recording
only the fills would make the diary agree with the backtest by construction -
the interesting rows are the ones where they differ, and those are the rows
where nothing was filled.

**No account identifiers.** Not the login, not the account number, not the
broker's name or server. The diary is meant to be sharable and diffable
against a backtest; identity adds nothing to that and is a liability. The
symbol and the server timezone are kept, because without them the rows cannot
be compared to anything.

What *is* kept is the environment the run was decided in: the engine version
and the full instrument spec, pinned in the `started` event. Both are needed
to diff the diary against a backtest later, and neither can be recovered
afterwards - `tick_value` tracks an FX rate and the broker moves swap rates
without notice, so re-reading the spec months later multiplies every money
column by a number nobody can see. A diary that does not carry its own spec
can only be compared to an instrument that no longer exists.
"""
from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from core.data.provider import SymbolSpec
from core.serialization import json_safe

logger = logging.getLogger(__name__)

EventKind = Literal[
    "started",
    "reconciled",
    "bar",
    "signal",
    "gate_rejected",
    "order_sent",
    "order_result",
    "position_opened",
    "position_closed",
    "error",
    "stopped",
]

# Field names that must never reach the diary, whatever a caller passes in.
FORBIDDEN_FIELDS: frozenset[str] = frozenset(
    {
        "login",
        "account",
        "account_number",
        "password",
        "server",
        "broker",
        "company",
        "name",
        "balance",
        "equity_total",
    }
)


class JournalError(RuntimeError):
    """The diary cannot be written, which is a reason to stop trading."""


@dataclass(frozen=True)
class JournalEvent:
    """One line of the diary."""

    at: datetime
    kind: EventKind
    symbol: str
    timeframe: str
    bar_time: datetime | None = None
    detail: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        payload = json_safe(
            {
                "at": self.at,
                "kind": self.kind,
                "symbol": self.symbol,
                "timeframe": self.timeframe,
                "bar_time": self.bar_time,
                "detail": scrub(self.detail),
            }
        )
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    @classmethod
    def from_json(cls, line: str) -> JournalEvent:
        payload = json.loads(line)
        return cls(
            at=datetime.fromisoformat(payload["at"]),
            kind=payload["kind"],
            symbol=payload["symbol"],
            timeframe=payload["timeframe"],
            bar_time=(
                datetime.fromisoformat(payload["bar_time"])
                if payload.get("bar_time")
                else None
            ),
            detail=payload.get("detail") or {},
        )


def scrub(detail: dict[str, Any]) -> dict[str, Any]:
    """Removes anything that identifies the account, at any nesting depth.

    A blocklist rather than an allowlist because the detail payload is open by
    design; the keys that must never appear are few and known, and a dropped
    field is visible in the diary as an absence rather than as a leak.
    """
    clean: dict[str, Any] = {}
    for key, value in detail.items():
        if key.lower() in FORBIDDEN_FIELDS:
            continue
        clean[key] = scrub(value) if isinstance(value, dict) else value
    return clean


class Journal:
    """Append-only diary for one runner."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(
        self,
        kind: EventKind,
        symbol: str,
        timeframe: str,
        bar_time: datetime | None = None,
        **detail: Any,
    ) -> JournalEvent:
        event = JournalEvent(
            at=datetime.now(timezone.utc),
            kind=kind,
            symbol=symbol,
            timeframe=timeframe,
            bar_time=bar_time,
            detail=detail,
        )
        self.write(event)
        return event

    def write(self, event: JournalEvent) -> None:
        """Writes one line and forces it to disk.

        The fsync is not paranoia: the diary is the only record of what the
        runner did, and a decision that reached the broker but not the disk is
        exactly the state nobody can reconstruct afterwards.
        """
        try:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(event.to_json() + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        except OSError as exc:
            raise JournalError(f"cannot write the diary at {self.path}: {exc}") from exc

    # -- reading ---------------------------------------------------------

    def events(self) -> Iterator[JournalEvent]:
        """Every readable line. A truncated last line is skipped, not fatal."""
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield JournalEvent.from_json(line)
                except (json.JSONDecodeError, KeyError) as exc:
                    logger.warning("diary line %d unreadable, skipped: %s", number, exc)

    def tail(self, limit: int = 50) -> list[JournalEvent]:
        return list(self.events())[-limit:]

    def last_bar_time(self) -> datetime | None:
        """The last bar the runner processed, from the diary itself.

        Restart state comes from here rather than from a separate file: two
        records of the same fact drift, and the diary is the one that has to
        be right.
        """
        latest: datetime | None = None
        for event in self.events():
            if (
                event.kind == "bar"
                and event.bar_time is not None
                and (latest is None or event.bar_time > latest)
            ):
                latest = event.bar_time
        return latest

    def as_frame(self) -> pd.DataFrame:
        """The diary as a table, one row per event, detail kept as a dict."""
        rows = [
            {
                "at": event.at,
                "kind": event.kind,
                "symbol": event.symbol,
                "timeframe": event.timeframe,
                "bar_time": event.bar_time,
                "detail": event.detail,
            }
            for event in self.events()
        ]
        if not rows:
            return pd.DataFrame(
                columns=["at", "kind", "symbol", "timeframe", "bar_time", "detail"]
            )
        frame = pd.DataFrame(rows)
        frame["at"] = pd.to_datetime(frame["at"], utc=True)
        frame["bar_time"] = pd.to_datetime(frame["bar_time"], utc=True)
        return frame

    def trades(self) -> pd.DataFrame:
        """The closed trades the diary recorded, in the engine's trade shape."""
        from core.engine.backtester import trades_frame

        closed = [
            dict(event.detail["trade"])
            for event in self.events()
            if event.kind == "position_closed" and "trade" in event.detail
        ]
        for trade in closed:
            for column in ("entry_time", "exit_time"):
                if isinstance(trade.get(column), str):
                    trade[column] = datetime.fromisoformat(trade[column])
        return trades_frame(closed)


# -- the pinned environment ---------------------------------------------
#
# `scrub` drops `name` at any depth, so the instrument spec goes into the
# diary without it and gets it back from the event's own `symbol` on the way
# out. Round-tripping through these two functions is what the test asserts.


def spec_payload(spec: SymbolSpec) -> dict[str, Any]:
    """The instrument spec as it is written into the `started` event."""
    payload = asdict(spec)
    payload.pop("name", None)
    return payload


def spec_from_payload(payload: dict[str, Any], symbol: str) -> SymbolSpec | None:
    """The instrument spec back out of a diary, or None if it was not pinned.

    A diary written before the spec was pinned is not an error - it is older,
    and the caller has to decide what to do about that. Returning None says
    "this diary cannot tell you what it traded" and leaves the decision where
    it belongs.
    """
    if not payload:
        return None
    fields = {**payload, "name": payload.get("name", symbol)}
    try:
        return SymbolSpec(**fields)
    except TypeError as exc:
        logger.warning("the diary's pinned spec is not readable, ignoring it: %s", exc)
        return None


@dataclass(frozen=True)
class DiaryEnvironment:
    """What the diary says about the engine and instrument that produced it."""

    engine_version: str | None
    symbol_spec: SymbolSpec | None
    symbol_spec_hash: str | None

    @property
    def pinned(self) -> bool:
        return self.symbol_spec is not None


def environment(journal: Journal) -> DiaryEnvironment:
    """Reads the pinned environment out of the diary's first `started` event."""
    started = next((e for e in journal.events() if e.kind == "started"), None)
    if started is None:
        return DiaryEnvironment(None, None, None)
    return DiaryEnvironment(
        engine_version=started.detail.get("engine_version"),
        symbol_spec=spec_from_payload(
            started.detail.get("symbol_spec") or {}, started.symbol
        ),
        symbol_spec_hash=started.detail.get("symbol_spec_hash"),
    )


def asdict_safe(value: Any) -> dict[str, Any]:
    """`asdict` for dataclasses, scrubbed. Convenience for callers."""
    return scrub(asdict(value))
