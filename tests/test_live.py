"""The diary, the lock, reconciliation, and the expected-vs-realized report."""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from core.engine.session import SessionCalendar
from core.live.broker import (
    TRADE_MODE_DEMO,
    TRADE_MODE_REAL,
    LiveAccountRefused,
    LiveBroker,
    OpenPosition,
    OrderResult,
)
from core.live.compare import compare
from core.live.journal import Journal, scrub
from core.live.lock import LockHeld, RunLock
from core.live.replay import ReplayBroker, replay
from core.live.runner import LiveConfig, LiveRunner, ReconciliationError
from core.runs.store import symbol_spec_cost_hash
from core.strategy.binding import BoundSpec, bind_cell
from core.strategy.spec import StrategySpec
from core.version import ENGINE_VERSION
from tests.conftest_engine import random_walk, symbol_spec

SERVER_TZ = ZoneInfo("Europe/Athens")
SYMBOL = "SYNTH"


def bars(n: int = 400, seed: int = 4) -> pd.DataFrame:
    frame = random_walk(n, seed=seed)
    frame.index = pd.date_range(
        "2024-01-01", periods=n, freq="1min", tz="UTC", name="time"
    )
    frame["spread"] = 8.0
    return frame


def bind(name: str = "bollinger-breakout", timeframe: str = "M1") -> BoundSpec:
    return bind_cell(
        StrategySpec.from_json(f"strategies/{name}.json"), SYMBOL, timeframe
    )


# -- the diary -----------------------------------------------------------


def test_the_diary_is_one_json_object_per_line(tmp_path) -> None:
    journal = Journal(tmp_path / "diary.jsonl")
    journal.append("bar", SYMBOL, "M1", bar_time=datetime(2024, 1, 1, tzinfo=timezone.utc))
    journal.append("signal", SYMBOL, "M1", side="long")

    lines = (tmp_path / "diary.jsonl").read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2
    for line in lines:
        json.loads(line)
    assert [e.kind for e in journal.events()] == ["bar", "signal"]


def test_the_diary_never_records_the_account(tmp_path) -> None:
    """A diary meant to be shared must not carry identity."""
    journal = Journal(tmp_path / "diary.jsonl")
    journal.append(
        "order_result",
        SYMBOL,
        "M1",
        login=123456,
        account="real-9981",
        server="Broker-Live",
        result={"password": "hunter2", "filled_price": 2000.5},
    )
    text = (tmp_path / "diary.jsonl").read_text(encoding="utf-8")
    for secret in ("123456", "real-9981", "Broker-Live", "hunter2"):
        assert secret not in text
    assert "2000.5" in text


def test_scrubbing_reaches_nested_dictionaries() -> None:
    clean = scrub({"keep": 1, "login": 2, "inner": {"server": "x", "price": 3}})
    assert clean == {"keep": 1, "inner": {"price": 3}}


def test_a_truncated_last_line_does_not_lose_the_diary(tmp_path) -> None:
    path = tmp_path / "diary.jsonl"
    journal = Journal(path)
    journal.append("bar", SYMBOL, "M1")
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"at": "2024-01-0')  # killed mid-write
    assert len(list(journal.events())) == 1


def test_the_last_processed_bar_comes_from_the_diary(tmp_path) -> None:
    journal = Journal(tmp_path / "diary.jsonl")
    first = datetime(2024, 1, 1, tzinfo=timezone.utc)
    journal.append("bar", SYMBOL, "M1", bar_time=first)
    journal.append("bar", SYMBOL, "M1", bar_time=first + timedelta(minutes=1))
    journal.append("signal", SYMBOL, "M1")
    assert journal.last_bar_time() == first + timedelta(minutes=1)


# -- the lock ------------------------------------------------------------


def test_a_second_runner_on_the_same_spec_refuses_to_start(tmp_path) -> None:
    """Two runners would double the position, each believing it holds half."""
    first = RunLock(tmp_path / "run.lock", label="spec on SYNTH")
    first.acquire()
    with pytest.raises(LockHeld, match="already running"):
        RunLock(tmp_path / "run.lock").acquire()
    first.release()
    RunLock(tmp_path / "run.lock").acquire()


def test_a_lock_left_by_a_dead_process_is_taken_over(tmp_path) -> None:
    """A crash must not need manual cleanup before trading can resume."""
    path = tmp_path / "run.lock"
    path.write_text(
        json.dumps(
            {
                # a pid that cannot be running: 2^22 is above every OS maximum
                "pid": 4_194_304,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "label": "ghost",
            }
        ),
        encoding="utf-8",
    )
    lock = RunLock(path)
    info = lock.acquire()
    assert info.pid == os.getpid()


def test_releasing_a_lock_that_now_belongs_to_someone_else_is_a_no_op(tmp_path) -> None:
    path = tmp_path / "run.lock"
    lock = RunLock(path)
    lock.acquire()
    path.write_text(
        json.dumps(
            {
                "pid": os.getpid() + 1,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "label": "other",
            }
        ),
        encoding="utf-8",
    )
    lock.release()
    assert path.exists()


# -- the account guard ---------------------------------------------------


class FakeAccount:
    def __init__(self, trade_mode: int) -> None:
        self.trade_mode = trade_mode


class FakeTerminal:
    def __init__(self, trade_allowed: bool = True) -> None:
        self.trade_allowed = trade_allowed


class FakeMT5:
    def __init__(self, trade_mode: int, trade_allowed: bool = True) -> None:
        self._account = FakeAccount(trade_mode)
        self._terminal = FakeTerminal(trade_allowed)

    def account_info(self):
        return self._account

    def terminal_info(self):
        return self._terminal


def test_a_real_account_is_refused_when_sending_is_requested(monkeypatch) -> None:
    import core.live.broker as broker_module

    monkeypatch.setattr(broker_module, "mt5", FakeMT5(TRADE_MODE_REAL))
    broker = LiveBroker(dry_run=False)
    with pytest.raises(LiveAccountRefused, match="not demo"):
        broker.check_account()


def test_a_real_account_is_allowed_in_dry_run(monkeypatch) -> None:
    """Dry run sends nothing, so the account type cannot cost anything."""
    import core.live.broker as broker_module

    monkeypatch.setattr(broker_module, "mt5", FakeMT5(TRADE_MODE_REAL))
    guard = LiveBroker(dry_run=True).check_account()
    assert not guard.allowed_to_send
    assert guard.trade_mode_name == "real"


def test_a_demo_account_with_algo_trading_off_is_refused(monkeypatch) -> None:
    import core.live.broker as broker_module

    monkeypatch.setattr(
        broker_module, "mt5", FakeMT5(TRADE_MODE_DEMO, trade_allowed=False)
    )
    with pytest.raises(RuntimeError, match="Algo Trading is disabled"):
        LiveBroker(dry_run=False).check_account()


def test_a_demo_account_with_algo_trading_on_may_send(monkeypatch) -> None:
    import core.live.broker as broker_module

    monkeypatch.setattr(broker_module, "mt5", FakeMT5(TRADE_MODE_DEMO))
    guard = LiveBroker(dry_run=False).check_account()
    assert guard.allowed_to_send
    assert guard.trade_mode_name == "demo"


def test_dry_run_is_the_default() -> None:
    assert LiveBroker().dry_run is True


# -- reconciliation ------------------------------------------------------


def position(ticket: int = 1, magic: int = 1, direction: int = 1) -> OpenPosition:
    return OpenPosition(
        ticket=ticket,
        symbol=SYMBOL,
        direction=direction,
        lots=0.02,
        entry_price=2000.0,
        stop_level=1990.0,
        target_level=2010.0,
        magic=magic,
        opened_at=datetime(2024, 1, 1, 0, 30, tzinfo=timezone.utc),
    )


class StubBroker(ReplayBroker):
    def __init__(self, positions: list[OpenPosition], magic: int = 1) -> None:
        super().__init__(magic=magic)
        self._positions = positions

    def open_positions(self, symbol: str) -> list[OpenPosition]:
        return list(self._positions)


def runner_with(broker: ReplayBroker, history: pd.DataFrame) -> LiveRunner:
    spec = bind()
    calendar = SessionCalendar.infer(bars().index, spec.tf)
    return LiveRunner(
        spec,
        symbol_spec(name=SYMBOL),
        SERVER_TZ,
        LiveConfig(session_calendar=calendar, magic=1),
        broker=broker,
    )


def test_an_unrecognised_position_stops_the_runner() -> None:
    """Adopting it would mean managing a trade whose stop it never chose."""
    broker = StubBroker([position(magic=999)])
    runner = runner_with(broker, bars())
    with pytest.raises(ReconciliationError, match="not opened by this engine"):
        runner.start(bars().iloc[:10])


def test_an_own_position_is_adopted_with_its_levels() -> None:
    broker = StubBroker([position(magic=1)])
    runner = runner_with(broker, bars())
    runner.start(bars().iloc[:10])
    adopted = runner.executor.position
    assert adopted is not None
    assert adopted.direction == 1
    assert adopted.stop_level == 1990.0
    assert adopted.target_level == 2010.0
    # the time stop keeps counting from the real entry, not from the restart
    assert adopted.entry_time == datetime(2024, 1, 1, 0, 30, tzinfo=timezone.utc)


def test_two_own_positions_cannot_have_come_from_one_runner() -> None:
    broker = StubBroker([position(1, magic=1), position(2, magic=1)])
    runner = runner_with(broker, bars())
    with pytest.raises(ReconciliationError, match="holds 2 positions"):
        runner.start(bars().iloc[:10])


def test_a_flat_account_leaves_the_runner_flat() -> None:
    runner = runner_with(StubBroker([]), bars())
    state = runner.start(bars().iloc[:10])
    assert state.position is None
    assert runner.executor.risk.open_positions == 0


# -- expected versus realized -------------------------------------------


def test_the_comparison_reports_slippage_on_the_trades_both_took(tmp_path) -> None:
    spec = bind()
    frame = bars(600)
    diary = tmp_path / "diary.jsonl"
    result = replay(
        spec, frame, symbol_spec(name=SYMBOL), SERVER_TZ, journal_path=diary
    )
    assert len(result.trades) > 0

    journal = Journal(diary)
    report = compare(result.trades, journal, symbol_spec(name=SYMBOL), "M1")
    # replaying against itself: same trades, no slippage, nothing unexplained
    assert report.matched == report.expected_trades == report.realized_trades
    assert report.pnl_from_slippage == pytest.approx(0.0)
    assert report.pnl_unexplained == pytest.approx(0.0)


def test_a_trade_the_runner_never_took_is_reported_with_a_reason(tmp_path) -> None:
    spec = bind()
    frame = bars(600)
    diary = tmp_path / "diary.jsonl"
    result = replay(
        spec, frame, symbol_spec(name=SYMBOL), SERVER_TZ, journal_path=diary
    )
    assert len(result.trades) > 1

    report = compare(
        result.trades, Journal(diary), symbol_spec(name=SYMBOL), "M1"
    )
    assert not report.only_expected

    # now pretend the backtest also took a trade on a bar the runner never saw
    extra = result.trades.iloc[[0]].copy()
    extra["entry_time"] = extra["entry_time"] + pd.Timedelta(days=400)
    inflated = pd.concat([result.trades, extra], ignore_index=True)
    report = compare(inflated, Journal(diary), symbol_spec(name=SYMBOL), "M1")
    assert len(report.only_expected) == 1
    assert "never saw this bar" in report.only_expected[0].reason


def test_an_empty_diary_says_so_instead_of_reporting_zero_slippage(tmp_path) -> None:
    spec = bind()
    frame = bars(600)
    result = replay(spec, frame, symbol_spec(name=SYMBOL), SERVER_TZ)
    empty = Journal(tmp_path / "nothing.jsonl")
    report = compare(result.trades, empty, symbol_spec(name=SYMBOL), "M1")
    assert report.realized_trades == 0
    assert "nothing to compare" in report.verdict
    assert any("no closed trade" in w for w in report.warnings)


def test_the_pnl_difference_is_decomposed_and_adds_up(tmp_path) -> None:
    spec = bind()
    frame = bars(600)
    diary = tmp_path / "diary.jsonl"
    result = replay(
        spec, frame, symbol_spec(name=SYMBOL), SERVER_TZ, journal_path=diary
    )
    worse = result.trades.copy()
    worse["net_pnl"] = worse["net_pnl"] - 0.25

    report = compare(worse, Journal(diary), symbol_spec(name=SYMBOL), "M1")
    total = report.realized_pnl - report.expected_pnl
    parts = report.pnl_from_slippage + report.pnl_from_unmatched + report.pnl_unexplained
    assert total == pytest.approx(parts)
    assert report.pnl_from_slippage == pytest.approx(0.25 * len(worse))


def test_the_diary_records_a_bar_for_every_bar_processed(tmp_path) -> None:
    spec = bind()
    frame = bars(200)
    diary = tmp_path / "diary.jsonl"
    replay(spec, frame, symbol_spec(name=SYMBOL), SERVER_TZ, journal_path=diary)
    events = list(Journal(diary).events())
    assert sum(1 for e in events if e.kind == "bar") == len(frame)
    assert events[0].kind == "started"


def test_the_diary_holds_the_indicators_behind_each_decision(tmp_path) -> None:
    """Recording only fills would make the diary agree by construction."""
    spec = bind()
    diary = tmp_path / "diary.jsonl"
    replay(spec, bars(200), symbol_spec(name=SYMBOL), SERVER_TZ, journal_path=diary)
    bar_events = [e for e in Journal(diary).events() if e.kind == "bar"]
    assert bar_events
    sample = bar_events[-1]
    assert "signal" in sample.detail
    assert "bar" in sample.detail
    assert "gates_cumulative" in sample.detail
    assert set(sample.detail["signal"]) == {"long", "short", "exit"}


def test_an_order_result_carries_what_the_broker_answered() -> None:
    result = OrderResult(
        accepted=True,
        dry_run=False,
        retcode=10009,
        retcode_name="done",
        requested_lots=0.05,
        filled_lots=0.02,
        requested_price=2000.0,
        filled_price=2000.4,
        order_ticket=1,
        position_ticket=2,
        comment="",
    )
    assert result.partial
    assert result.slippage_price == pytest.approx(0.4)


# -- the diary's pinned environment --------------------------------------
#
# These exist because of a real false alarm. A replay diary written by engine
# 3.2.0 compared at +0.0517 against a backtest run months later, on a dry run
# where the difference can only be zero. It was read as an execution
# divergence and it was not one: every decision matched to the bit, and every
# money column was off by one identical factor, 1.000352809568884, because
# `tick_value` had moved from 0.86020765 to 0.86051114 between the two runs.
# The engine was innocent; the comparison was against a different instrument.


def _pinned_backtest(spec, frame, instrument, calendar):
    from core.engine.backtester import BacktestConfig, run_backtest

    return run_backtest(
        spec.spec, frame, instrument, SERVER_TZ,
        BacktestConfig(initial_equity=100.0, session_calendar=calendar),
    )


def test_a_dry_run_diary_compares_to_its_backtest_at_exactly_zero(tmp_path) -> None:
    """The central claim of the project, as an equality with no tolerance.

    Not `approx`: both sides are the same arithmetic over the same bars with
    the same spec, so the only acceptable difference is none at all. A
    tolerance here would have hidden the drift that prompted these tests.
    """
    spec = bind()
    frame = bars(600)
    instrument = symbol_spec(name=SYMBOL)
    calendar = SessionCalendar.infer(frame.index, spec.tf, 0.5, SERVER_TZ)

    diary = tmp_path / "diary.jsonl"
    replay(spec, frame, instrument, SERVER_TZ, initial_equity=100.0,
           calendar=calendar, journal_path=diary)
    expected = _pinned_backtest(spec, frame, instrument, calendar)
    assert len(expected.trades) > 0

    report = compare(expected.trades, Journal(diary), instrument, "M1")

    assert report.comparable
    assert report.spec_matches_diary is True
    assert report.matched == report.expected_trades == report.realized_trades
    assert report.realized_pnl == report.expected_pnl
    assert report.pnl_from_slippage == 0.0
    assert report.pnl_from_unmatched == 0.0
    assert report.pnl_unexplained == 0.0


def test_the_diary_pins_the_instrument_spec_it_traded(tmp_path) -> None:
    from core.live.journal import environment

    instrument = symbol_spec(name=SYMBOL)
    diary = tmp_path / "diary.jsonl"
    replay(bind(), bars(200), instrument, SERVER_TZ, journal_path=diary)

    env = environment(Journal(diary))
    assert env.engine_version == ENGINE_VERSION
    assert env.symbol_spec == instrument
    assert env.symbol_spec_hash == symbol_spec_cost_hash(instrument)


def test_a_diary_from_another_engine_is_not_comparable(tmp_path) -> None:
    spec = bind()
    frame = bars(600)
    instrument = symbol_spec(name=SYMBOL)
    calendar = SessionCalendar.infer(frame.index, spec.tf, 0.5, SERVER_TZ)

    diary = tmp_path / "diary.jsonl"
    replay(spec, frame, instrument, SERVER_TZ, initial_equity=100.0,
           calendar=calendar, journal_path=diary)
    _rewrite_started(diary, {"engine_version": "3.2.0"})

    expected = _pinned_backtest(spec, frame, instrument, calendar)
    report = compare(expected.trades, Journal(diary), instrument, "M1")

    assert not report.comparable
    assert report.diary_engine_version == "3.2.0"
    assert "NOT COMPARABLE" in report.verdict
    assert any("different code" in w for w in report.warnings)


def test_a_drifted_tick_value_is_named_and_not_charged_as_slippage(tmp_path) -> None:
    """The false alarm itself, reconstructed: same engine, moved spec."""
    from dataclasses import replace as replace_field

    spec = bind()
    frame = bars(600)
    instrument = symbol_spec(name=SYMBOL)
    calendar = SessionCalendar.infer(frame.index, spec.tf, 0.5, SERVER_TZ)

    diary = tmp_path / "diary.jsonl"
    replay(spec, frame, instrument, SERVER_TZ, initial_equity=100.0,
           calendar=calendar, journal_path=diary)

    drifted = replace_field(instrument, tick_value=instrument.tick_value * 1.0003528)
    expected = _pinned_backtest(spec, frame, drifted, calendar)
    report = compare(expected.trades, Journal(diary), drifted, "M1")

    assert not report.comparable
    assert report.spec_matches_diary is False
    assert any("tick_value" in w for w in report.warnings)
    assert any("not slippage" in w for w in report.warnings)


def test_a_diary_that_pinned_no_spec_says_so(tmp_path) -> None:
    spec = bind()
    frame = bars(600)
    instrument = symbol_spec(name=SYMBOL)
    calendar = SessionCalendar.infer(frame.index, spec.tf, 0.5, SERVER_TZ)

    diary = tmp_path / "diary.jsonl"
    replay(spec, frame, instrument, SERVER_TZ, initial_equity=100.0,
           calendar=calendar, journal_path=diary)
    _rewrite_started(diary, {"symbol_spec": None, "symbol_spec_hash": None})

    expected = _pinned_backtest(spec, frame, instrument, calendar)
    report = compare(expected.trades, Journal(diary), instrument, "M1")

    assert not report.comparable
    assert report.spec_matches_diary is None
    assert any("did not pin the instrument spec" in w for w in report.warnings)


def _rewrite_started(diary, changes: dict) -> None:
    """Edits the `started` event in place, to age a diary the tests just wrote."""
    lines = diary.read_text(encoding="utf-8").strip().split("\n")
    for index, line in enumerate(lines):
        payload = json.loads(line)
        if payload["kind"] != "started":
            continue
        for key, value in changes.items():
            if value is None:
                payload["detail"].pop(key, None)
            else:
                payload["detail"][key] = value
        lines[index] = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        break
    diary.write_text("\n".join(lines) + "\n", encoding="utf-8")
