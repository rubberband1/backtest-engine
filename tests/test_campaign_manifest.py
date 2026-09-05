"""A campaign's frozen inputs, and what they are for.

A run pins the instrument spec it used; a campaign did not, and its cells
execute over minutes or hours while `tick_value` follows an FX rate. Two
executions of the same campaign therefore differed, and the difference could
not be attributed - engine regression, or the euro moved? These tests pin
that the manifest makes that question answerable.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

import pytest

from core.data.provider import SymbolSpecSnapshot
from core.research.manifest import (
    MANIFEST_VERSION,
    CampaignManifest,
    ManifestError,
    drifted,
)
from core.runs.store import RunConfig
from tests.conftest_engine import symbol_spec

UTC = timezone.utc
READ_AT = datetime(2026, 1, 1, 12, tzinfo=UTC)


class FakeResolver:
    """A `SymbolResolver` reduced to what the manifest touches."""

    def __init__(self, specs: dict) -> None:
        self._memo = {
            name: SymbolSpecSnapshot(spec=spec, read_at=READ_AT)
            for name, spec in specs.items()
        }
        self._tz = None

    def symbol_spec_snapshot(self, symbol: str) -> SymbolSpecSnapshot:
        if symbol not in self._memo:
            raise KeyError(symbol)
        return self._memo[symbol]

    def symbol_spec(self, symbol: str):
        return self.symbol_spec_snapshot(symbol).spec

    def server_timezone(self):
        from zoneinfo import ZoneInfo

        return ZoneInfo("Europe/Athens")


def a_manifest(**overrides) -> CampaignManifest:
    resolver = FakeResolver(
        {
            "EURUSD.r": symbol_spec(name="EURUSD.r", tick_value=0.8605481691837701),
            "XAUUSD.r": symbol_spec(name="XAUUSD.r", tick_value=0.8605111436193099),
        }
    )
    return CampaignManifest.freeze(
        resolver,  # type: ignore[arg-type]
        ["strategies/rsi-mean-reversion.json"],
        ["EURUSD.r", "XAUUSD.r"],
        ["H1", "H4"],
        RunConfig(symbol="EURUSD.r", timeframe="H1", spread_mode="fixed", spread_value=3.0),
        min_trades=30,
        permutation_iterations=200,
        alpha=0.05,
        confidence=0.95,
        max_spread_atr=0.15,
        gate_alteration_threshold=0.3,
        **overrides,
    )


def test_the_manifest_freezes_every_cost_field() -> None:
    manifest = a_manifest()
    assert set(manifest.symbol_specs) == {"EURUSD.r", "XAUUSD.r"}

    fields = manifest.cost_fields()
    assert fields["XAUUSD.r"]["tick_value"] == 0.8605111436193099
    assert fields["XAUUSD.r"]["read_at"].startswith("2026-01-01")
    # the cost hash is what a run_id is built from, so it belongs here too
    assert len(fields["XAUUSD.r"]["cost_hash"]) == 64


def test_a_symbol_with_no_spec_is_left_out_rather_than_defaulted() -> None:
    """A manifest short a symbol is honest; one that invents a contract is not."""
    resolver = FakeResolver({"EURUSD.r": symbol_spec(name="EURUSD.r")})
    manifest = CampaignManifest.freeze(
        resolver,  # type: ignore[arg-type]
        ["s.json"], ["EURUSD.r", "GHOST"], ["H1"],
        RunConfig(symbol="EURUSD.r", timeframe="H1"),
        min_trades=30, permutation_iterations=10, alpha=0.05,
        confidence=0.95, max_spread_atr=0.15, gate_alteration_threshold=0.3,
    )
    assert set(manifest.symbol_specs) == {"EURUSD.r"}
    assert "GHOST" in manifest.symbols, "the grid still records what was asked for"


def test_a_manifest_round_trips_through_disk(tmp_path) -> None:
    manifest = a_manifest(note="phase 8 verification")
    path = manifest.write(tmp_path / "campaign.manifest.json")

    back = CampaignManifest.read(path)
    assert back.symbol_specs == manifest.symbol_specs
    assert back.config == manifest.config
    assert back.timeframes == manifest.timeframes
    assert back.min_trades == manifest.min_trades
    assert back.confidence == manifest.confidence
    assert back.note == "phase 8 verification"
    assert back.run_config() == manifest.run_config()


def test_a_manifest_from_another_version_is_refused(tmp_path) -> None:
    """Silently honouring half of a foreign manifest is worse than failing."""
    payload = a_manifest().to_dict()
    payload["manifest_version"] = MANIFEST_VERSION + 1
    with pytest.raises(ManifestError, match="different engine"):
        CampaignManifest.from_dict(payload)


def test_a_missing_manifest_says_so(tmp_path) -> None:
    with pytest.raises(ManifestError, match="no manifest"):
        CampaignManifest.read(tmp_path / "nothing.json")


def test_applying_a_manifest_overrides_what_the_broker_says_now() -> None:
    """The whole point: the campaign runs against the frozen contract."""
    manifest = a_manifest()

    moved = FakeResolver(
        {
            "EURUSD.r": symbol_spec(name="EURUSD.r", tick_value=0.9),
            "XAUUSD.r": symbol_spec(name="XAUUSD.r", tick_value=0.9),
        }
    )
    manifest.apply(moved)  # type: ignore[arg-type]

    assert moved.symbol_spec("XAUUSD.r").tick_value == 0.8605111436193099
    assert moved.symbol_spec("EURUSD.r").tick_value == 0.8605481691837701
    assert str(moved._tz) == "Europe/Athens"


def test_drift_is_reported_rather_than_hidden() -> None:
    """The manifest suppresses the drift; the report still has to name it."""
    manifest = a_manifest()
    moved = FakeResolver(
        {
            "EURUSD.r": symbol_spec(name="EURUSD.r", tick_value=0.8605481691837701),
            "XAUUSD.r": symbol_spec(name="XAUUSD.r", tick_value=0.8616298081060151),
        }
    )

    changes = drifted(manifest, moved)  # type: ignore[arg-type]
    assert set(changes) == {"XAUUSD.r"}, "only the one that moved"
    assert "tick_value" in changes["XAUUSD.r"]
    assert "0.8605111436193099" in changes["XAUUSD.r"]
    assert "0.8616298081060151" in changes["XAUUSD.r"]


def test_a_swap_change_counts_as_drift_too() -> None:
    """Brokers change swap rates without notice; one batch moved XTIUSD by 1.19."""
    manifest = a_manifest()
    base = symbol_spec(name="XAUUSD.r", tick_value=0.8605111436193099)
    moved = FakeResolver(
        {
            "EURUSD.r": symbol_spec(name="EURUSD.r", tick_value=0.8605481691837701),
            "XAUUSD.r": replace(base, swap_long=base.swap_long - 1.19),
        }
    )
    changes = drifted(manifest, moved)  # type: ignore[arg-type]
    assert "swap_long" in changes["XAUUSD.r"]


def test_the_summary_states_the_freeze_date_and_the_values() -> None:
    """The report has to be readable without remembering when it ran."""
    summary = a_manifest().summary()
    assert "2026" in summary
    assert "Europe/Athens" in summary
    assert "XAUUSD.r" in summary
    assert "0.8605111436193099" in summary
    assert "specs frozen: 2 of 2" in summary
