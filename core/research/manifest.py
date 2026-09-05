"""A campaign's frozen inputs, so it can be run twice and compared.

A single run pins the instrument spec it used, and that has been true since
4.0.0. A *campaign* did not. Its cells execute over minutes or hours, each
one reading the spec fresh, and the numbers in it move while it runs:
`tick_value` follows an FX rate and was observed at 0.8602, then 0.8605, then
0.8613 across three reads of the same instrument. Every money column in every
cell scales with it.

The consequence is that re-running a campaign did not reproduce it, and the
difference could not be attributed. It might be an engine regression; it
might be that the euro moved. Nothing in the output could tell those apart,
which makes "reproducible" a word rather than a property.

A manifest fixes that by freezing, at the moment the campaign starts:

- the instrument spec of every symbol in the grid, with the timestamp it was
  read at;
- the server timezone;
- the grid itself, the run configuration, and every knob that changes a
  number - the minimum trade count, the permutation iterations, the
  confidence level, the tradability threshold, and the trial count carried
  over from earlier campaigns.

Re-running from the manifest hands those same values back to the engine, so
the only remaining input is the bar data and the engine itself. A difference
in the results is then a real difference, and `scripts/verify_campaign.py`
says which cells moved and by how much.

What a manifest deliberately does **not** freeze is the bar data. The broker
rewrites candles occasionally, and pinning a copy of six years of bars inside
a JSON file solves the wrong problem: a run already stores a fingerprint of
the bytes it executed on, so changed data shows up as a changed run rather
than as a silent difference.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from core.data.provider import SymbolSpec, SymbolSpecSnapshot
from core.runs.runner import SymbolResolver
from core.runs.store import RunConfig, symbol_spec_cost_hash
from core.serialization import json_safe
from core.version import ENGINE_VERSION

MANIFEST_VERSION = 1


class ManifestError(RuntimeError):
    """The manifest cannot be used as it stands."""


@dataclass
class CampaignManifest:
    """Every input of a campaign except the bars and the engine."""

    created_at: datetime
    engine_version: str
    strategies: list[str]
    symbols: list[str]
    timeframes: list[str]
    config: dict[str, Any]
    server_timezone: str
    symbol_specs: dict[str, dict[str, Any]]
    min_trades: int
    permutation_iterations: int
    alpha: float
    confidence: float
    max_spread_atr: float
    gate_alteration_threshold: float
    check_tradability: bool = True
    prior_attempts: int = 0
    prior_sharpes: list[float] = field(default_factory=list)
    manifest_version: int = MANIFEST_VERSION
    note: str | None = None

    # -- construction ----------------------------------------------------

    @classmethod
    def freeze(
        cls,
        resolver: SymbolResolver,
        strategies: list[str],
        symbols: list[str],
        timeframes: list[str],
        config: RunConfig,
        *,
        min_trades: int,
        permutation_iterations: int,
        alpha: float,
        confidence: float,
        max_spread_atr: float,
        gate_alteration_threshold: float,
        check_tradability: bool = True,
        prior_attempts: int = 0,
        prior_sharpes: list[float] | None = None,
        note: str | None = None,
    ) -> CampaignManifest:
        """Reads every instrument spec now, and records when that was.

        A symbol whose spec cannot be read is left out rather than defaulted:
        the campaign will fail on it anyway, and a manifest that quietly
        invents a contract is worse than one that is short a symbol.
        """
        specs: dict[str, dict[str, Any]] = {}
        for symbol in symbols:
            try:
                snapshot = resolver.symbol_spec_snapshot(symbol)
            except Exception:
                continue
            specs[symbol] = snapshot.to_dict()

        return cls(
            created_at=datetime.now(timezone.utc),
            engine_version=ENGINE_VERSION,
            strategies=[str(path) for path in strategies],
            symbols=list(symbols),
            timeframes=list(timeframes),
            config=config.to_dict(),
            server_timezone=str(resolver.server_timezone()),
            symbol_specs=specs,
            min_trades=min_trades,
            permutation_iterations=permutation_iterations,
            alpha=alpha,
            confidence=confidence,
            max_spread_atr=max_spread_atr,
            gate_alteration_threshold=gate_alteration_threshold,
            check_tradability=check_tradability,
            prior_attempts=prior_attempts,
            prior_sharpes=list(prior_sharpes or []),
            note=note,
        )

    # -- persistence -----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return json_safe(
            {
                "manifest_version": self.manifest_version,
                "created_at": self.created_at,
                "engine_version": self.engine_version,
                "note": self.note,
                "strategies": self.strategies,
                "symbols": self.symbols,
                "timeframes": self.timeframes,
                "config": self.config,
                "server_timezone": self.server_timezone,
                "symbol_specs": self.symbol_specs,
                "min_trades": self.min_trades,
                "permutation_iterations": self.permutation_iterations,
                "alpha": self.alpha,
                "confidence": self.confidence,
                "max_spread_atr": self.max_spread_atr,
                "gate_alteration_threshold": self.gate_alteration_threshold,
                "check_tradability": self.check_tradability,
                "prior_attempts": self.prior_attempts,
                "prior_sharpes": self.prior_sharpes,
            }
        )

    def write(self, path: Path | str) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return target

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CampaignManifest:
        version = int(payload.get("manifest_version", 0))
        if version != MANIFEST_VERSION:
            raise ManifestError(
                f"manifest version {version} was written by a different engine "
                f"than this one, which writes version {MANIFEST_VERSION}. "
                f"Re-running from it would silently use a different set of "
                f"frozen inputs"
            )
        return cls(
            created_at=datetime.fromisoformat(str(payload["created_at"])),
            engine_version=str(payload["engine_version"]),
            note=payload.get("note"),
            strategies=list(payload["strategies"]),
            symbols=list(payload["symbols"]),
            timeframes=list(payload["timeframes"]),
            config=dict(payload["config"]),
            server_timezone=str(payload["server_timezone"]),
            symbol_specs=dict(payload["symbol_specs"]),
            min_trades=int(payload["min_trades"]),
            permutation_iterations=int(payload["permutation_iterations"]),
            alpha=float(payload["alpha"]),
            confidence=float(payload["confidence"]),
            max_spread_atr=float(payload["max_spread_atr"]),
            gate_alteration_threshold=float(payload["gate_alteration_threshold"]),
            check_tradability=bool(payload.get("check_tradability", True)),
            prior_attempts=int(payload.get("prior_attempts", 0)),
            prior_sharpes=list(payload.get("prior_sharpes") or []),
        )

    @classmethod
    def read(cls, path: Path | str) -> CampaignManifest:
        target = Path(path)
        if not target.exists():
            raise ManifestError(f"no manifest at {target}")
        return cls.from_dict(json.loads(target.read_text(encoding="utf-8")))

    # -- use -------------------------------------------------------------

    def run_config(self) -> RunConfig:
        return RunConfig.from_dict(self.config)

    def spec_for(self, symbol: str) -> SymbolSpecSnapshot | None:
        payload = self.symbol_specs.get(symbol)
        return SymbolSpecSnapshot.from_dict(payload) if payload else None

    def apply(self, resolver: SymbolResolver) -> SymbolResolver:
        """Loads the frozen specs and clock into `resolver`, in place.

        Everything downstream asks the resolver, and the resolver already
        memoises per symbol, so filling that memo is all it takes for every
        cell of the campaign to execute against the same contract the
        original one did.
        """
        for symbol, payload in self.symbol_specs.items():
            resolver._memo[symbol] = SymbolSpecSnapshot.from_dict(payload)
        resolver._tz = ZoneInfo(self.server_timezone)
        return resolver

    # -- reporting -------------------------------------------------------

    def cost_fields(self) -> dict[str, dict[str, Any]]:
        """The frozen values that actually change a result, per symbol."""
        from core.data.provider import SYMBOL_SPEC_COST_FIELDS

        out: dict[str, dict[str, Any]] = {}
        for symbol in sorted(self.symbol_specs):
            snapshot = self.spec_for(symbol)
            if snapshot is None:
                continue
            out[symbol] = {
                name: getattr(snapshot.spec, name) for name in SYMBOL_SPEC_COST_FIELDS
            }
            out[symbol]["read_at"] = snapshot.read_at.isoformat()
            out[symbol]["cost_hash"] = symbol_spec_cost_hash(snapshot.spec)
        return out

    def summary(self) -> str:
        lines = [
            f"Campaign manifest, engine {self.engine_version}",
            f"  frozen at   : {self.created_at.isoformat()}",
            f"  grid        : {len(self.strategies)} strategies x "
            f"{len(self.symbols)} symbols x {len(self.timeframes)} timeframes",
            f"  server clock: {self.server_timezone}",
            f"  specs frozen: {len(self.symbol_specs)} of {len(self.symbols)} symbols",
        ]
        if self.note:
            lines.append(f"  note        : {self.note}")
        for symbol, values in self.cost_fields().items():
            lines.append(
                f"    {symbol:<12} tick_value={values['tick_value']!r} "
                f"swap {values['swap_long']}/{values['swap_short']} "
                f"read {values['read_at'][:19]}"
            )
        return "\n".join(lines)


def drifted(manifest: CampaignManifest, resolver: SymbolResolver) -> dict[str, str]:
    """Which frozen specs no longer match what the broker says today.

    Not an error - it is the whole reason the manifest exists - but a
    campaign re-run from a manifest should be able to say how far the world
    has moved since, because that is the size of the difference the manifest
    is suppressing.
    """
    from core.data.provider import SYMBOL_SPEC_COST_FIELDS

    out: dict[str, str] = {}
    for symbol in sorted(manifest.symbol_specs):
        frozen = manifest.spec_for(symbol)
        if frozen is None:
            continue
        try:
            live: SymbolSpec = resolver.symbol_spec(symbol)
        except Exception:
            continue
        changes = [
            f"{name} {getattr(frozen.spec, name)!r} -> {getattr(live, name)!r}"
            for name in SYMBOL_SPEC_COST_FIELDS
            if getattr(frozen.spec, name) != getattr(live, name)
        ]
        if changes:
            out[symbol] = "; ".join(changes)
    return out
