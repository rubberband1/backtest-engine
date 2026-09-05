"""One way to point a spec at an instrument, and no way around it.

The defect this file exists for: `POST /api/backtest` loaded the bars the
request named and then ran the spec exactly as written, instrument block
included, so a spec authored on `XAUUSD.r H1` and run over `AUDUSD.r H4` bars
executed as an H1 strategy on H4 data. The campaign runner had been binding
the spec to its cell since it was written; the API, the batch runner and the
live runner each did it by hand, or not at all.

The fix is not another careful call site. Every entry point that executes a
spec now takes a `BoundSpec`, which cannot hold a spec that disagrees with the
cell it names, and refuses to run against bars other than the ones it is bound
to. What follows checks both halves of that, on every entry point.
"""
from __future__ import annotations

import inspect
import re
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from core.data.provider import SymbolSpecSnapshot
from core.live.replay import compare_replay, replay
from core.live.runner import LiveRunner
from core.research.preview import preview
from core.runs.runner import execute_run, plan_run, run_edge_gate
from core.runs.store import RunConfig, RunStore
from core.strategy.binding import BoundSpec, CellMismatch, bind_cell
from core.strategy.spec import StrategySpec
from tests.conftest_engine import random_walk, spec_from, symbol_spec, symbol_spec_snapshot

REPO_ROOT = Path(__file__).resolve().parents[1]
SERVER_TZ = ZoneInfo("Europe/Athens")
HERE = "TEST"
ELSEWHERE = "OTHER"


def config(symbol: str = HERE, timeframe: str = "M1") -> RunConfig:
    return RunConfig(symbol=symbol, timeframe=timeframe, initial_equity=1000.0)


def snapshot(name: str) -> SymbolSpecSnapshot:
    return symbol_spec_snapshot(name=name)


# -- the type itself -------------------------------------------------------


def test_binding_rewrites_the_instrument_block() -> None:
    spec = StrategySpec.from_json("strategies/ma-crossover.json")
    assert spec.instrument.symbol != "XTIUSD"
    bound = bind_cell(spec, "XTIUSD", "H4")
    assert (bound.symbol, bound.timeframe) == ("XTIUSD", "H4")
    assert bound.spec.instrument.symbol == "XTIUSD"
    assert bound.spec.instrument.timeframe == "H4"


def test_a_bound_spec_cannot_disagree_with_itself() -> None:
    """Construction is the binding, so there is no unbound state to reach."""
    spec = StrategySpec.from_json("strategies/ma-crossover.json")
    direct = BoundSpec(spec, "XTIUSD", "H4")
    assert direct.spec.instrument.symbol == direct.symbol
    assert direct.spec.instrument.timeframe == direct.timeframe
    # and rebinding an already bound spec is idempotent, not cumulative
    again = bind_cell(direct.spec, "XTIUSD", "H4")
    assert again.spec.to_json() == direct.spec.to_json()


def test_the_timeframe_is_normalised_before_it_is_compared() -> None:
    bound = bind_cell(spec_from(), HERE, "h1")
    assert bound.timeframe == "H1"
    bound.must_match(HERE, "H1")


# -- every entry point demands one -----------------------------------------

SEAMS = [
    (plan_run, "bound"),
    (execute_run, "bound"),
    (run_edge_gate, "bound"),
    (preview, "bound"),
    (LiveRunner.__init__, "bound"),
    (replay, "bound"),
    (compare_replay, "bound"),
]


@pytest.mark.parametrize(
    "func,parameter", SEAMS, ids=[f.__qualname__ for f, _ in SEAMS]
)
def test_the_entry_point_takes_a_bound_spec(func, parameter: str) -> None:
    """Not a convention: the annotation is what stops an unbound path."""
    annotation = inspect.signature(func).parameters[parameter].annotation
    assert annotation is BoundSpec or annotation == "BoundSpec", (
        f"{func.__qualname__} takes {annotation!r} for {parameter!r}: an entry "
        f"point that accepts a bare StrategySpec is a way around the binder"
    )


# -- and refuses bars it is not bound to -----------------------------------


def test_plan_run_refuses_another_instrument() -> None:
    bound = bind_cell(spec_from(), ELSEWHERE, "M1")
    with pytest.raises(CellMismatch):
        plan_run(bound, config(), random_walk(200), symbol_spec(name=HERE))


def test_plan_run_refuses_another_timeframe() -> None:
    """The half that was silent: the time stop counts bars of the spec's tf."""
    bound = bind_cell(spec_from(), HERE, "H4")
    with pytest.raises(CellMismatch):
        plan_run(bound, config(timeframe="M1"), random_walk(200), symbol_spec(name=HERE))


def test_execute_run_refuses_another_cell(tmp_path: Path) -> None:
    bound = bind_cell(spec_from(), ELSEWHERE, "M1")
    with pytest.raises(CellMismatch):
        execute_run(
            RunStore(tmp_path / "runs"), bound, config(), random_walk(200),
            snapshot(HERE), SERVER_TZ,
        )


def test_the_edge_gate_refuses_another_cell() -> None:
    bound = bind_cell(spec_from(), HERE, "H1")
    with pytest.raises(CellMismatch):
        run_edge_gate(bound, config(timeframe="M1"), random_walk(200), symbol_spec(name=HERE))


def test_the_preview_refuses_another_instrument() -> None:
    bound = bind_cell(spec_from(), ELSEWHERE, "M1")
    with pytest.raises(CellMismatch):
        preview(bound, random_walk(200), symbol_spec(name=HERE))


def test_the_live_runner_refuses_another_instrument() -> None:
    bound = bind_cell(spec_from(), ELSEWHERE, "M1")
    with pytest.raises(CellMismatch):
        LiveRunner(bound, symbol_spec(name=HERE), SERVER_TZ)


def test_the_replay_refuses_another_instrument() -> None:
    bound = bind_cell(spec_from(), ELSEWHERE, "M1")
    with pytest.raises(CellMismatch):
        replay(bound, random_walk(200), symbol_spec(name=HERE), SERVER_TZ)


def test_the_equivalence_check_refuses_another_instrument() -> None:
    bound = bind_cell(spec_from(), ELSEWHERE, "M1")
    with pytest.raises(CellMismatch):
        compare_replay(bound, random_walk(200), symbol_spec(name=HERE), SERVER_TZ)


# -- and nothing else writes an instrument block ---------------------------

# `spec.instrument` written by hand, in any of the three shapes the codebase
# has used: a dotted parameter path, a payload key, a pydantic model_copy.
BY_HAND = re.compile(
    r'instrument\.symbol"|instrument\.timeframe"|instrument"\]\s*\[|'
    r'"instrument"\s*:\s*[a-zA-Z_]'
)


def test_the_binder_is_the_only_place_that_binds() -> None:
    """The screen path was right and the other three were not, because there
    were four. A second binder is how that comes back."""
    binder = REPO_ROOT / "core" / "strategy" / "binding.py"
    offenders = []
    for directory in ("core", "api", "scripts"):
        for path in sorted((REPO_ROOT / directory).rglob("*.py")):
            if path == binder:
                continue
            for number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                if BY_HAND.search(line):
                    offenders.append(f"{path.relative_to(REPO_ROOT)}:{number}: {line.strip()}")
    assert not offenders, (
        "these write a spec's instrument block outside core/strategy/binding.py; "
        "bind_cell is the one way to point a spec at a cell:\n  "
        + "\n  ".join(offenders)
    )
