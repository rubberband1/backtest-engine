"""The desktop shell, without opening a window.

What is worth testing here is not PyWebview - it is the four decisions around
it, each of which replaced a way the application used to fail without saying
anything: the port it picks, the page it shows when the backend does not come
up, the runners it refuses to abandon silently, and the promise that none of
it needs MetaTrader 5.

`webview` itself is never imported: it is an optional extra, and CI installs
neither it nor a display to put a window on.
"""
from __future__ import annotations

import socket
from pathlib import Path

import pytest

import desktop

# -- the port ------------------------------------------------------------


def test_free_port_returns_something_bindable():
    port = desktop.free_port()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", port))


def test_a_preferred_port_is_used_when_it_is_free():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as holder:
        holder.bind(("127.0.0.1", 0))
        free = int(holder.getsockname()[1])
    # the socket is closed again by the time this runs, so the number is free
    assert desktop.free_port(free) == free


def test_a_taken_port_is_stepped_over_rather_than_failed_on():
    """The whole reason this exists: 8000 being busy used to be fatal."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as holder:
        holder.bind(("127.0.0.1", 0))
        holder.listen(1)
        taken = int(holder.getsockname()[1])
        chosen = desktop.free_port(taken)
    assert chosen != taken
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", chosen))


# -- the window that opens when nothing else can -------------------------


def test_the_splash_says_what_is_being_waited_on():
    """The rule is not that the word "loading" is banned - it is that the word
    is never alone. A bare "Loading..." tells a reader nothing they did not
    already know from the fact that they are looking at it."""
    page = desktop.splash_page(51234)
    assert "Starting the engine" in page
    # the port it will serve on, and what is being read before it can
    assert "51234" in page
    assert "cache index" in page and "strategy library" in page


def test_neither_startup_page_loads_anything(monkeypatch):
    """Both are shown when the files this application ships with may be missing."""
    for page in (desktop.splash_page(8000), desktop.failure_page("a", "b")):
        assert "<link" not in page
        assert "src=" not in page
        assert "http://" not in page.replace("http://127.0.0.1", "")


def test_the_failure_page_does_not_execute_what_it_reports():
    """A traceback is text. It reaches the page as text."""
    page = desktop.failure_page("x", "<script>alert(1)</script>")
    assert "<script>alert(1)</script>" not in page
    assert "&lt;script&gt;" in page


def test_the_failure_page_carries_the_reason_not_a_summary_of_it():
    page = desktop.failure_page("could not start", "Traceback: ImportError foo")
    assert "Traceback: ImportError foo" in page
    assert "could not start" in page
    assert page.lstrip().startswith("<!doctype html>")


def test_the_failure_page_needs_no_files_from_the_build():
    """It is shown when `ui/dist` is the problem, so it cannot come from there."""
    page = desktop.failure_page("x", "y")
    assert "<link" not in page and "<script" not in page


# -- what closing the window must not do quietly -------------------------


def test_no_live_runners_when_the_directory_is_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("BACKTEST_LIVE_DIR", str(tmp_path))
    assert desktop.live_runners() == []


def test_no_live_runners_when_the_directory_does_not_exist(tmp_path, monkeypatch):
    monkeypatch.setenv("BACKTEST_LIVE_DIR", str(tmp_path / "nothing"))
    assert desktop.live_runners() == []


def test_a_lock_held_by_this_process_counts_as_a_live_runner(tmp_path, monkeypatch):
    """A lock naming a pid that is alive is the definition the API uses too."""
    from core.live.lock import RunLock

    monkeypatch.setenv("BACKTEST_LIVE_DIR", str(tmp_path))
    lock = RunLock(tmp_path / "session-xauusd.lock")
    lock.acquire()
    try:
        running = desktop.live_runners()
    finally:
        lock.release()
    assert len(running) == 1
    assert "session-xauusd" in running[0]


def test_a_stale_lock_is_not_reported_as_a_runner(tmp_path, monkeypatch):
    """A crashed process leaves its lock behind. That is not a reason to warn."""
    from core.live.lock import RunLock

    monkeypatch.setenv("BACKTEST_LIVE_DIR", str(tmp_path))
    path = tmp_path / "session-dead.lock"
    RunLock(path).acquire()
    # rewrite the pid to one nothing can be running under
    text = path.read_text(encoding="utf-8")
    import json
    import os

    payload = json.loads(text)
    payload["pid"] = 999_999_999
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert os.getpid() != 999_999_999
    assert desktop.live_runners() == []


# -- the promise that none of this needs a broker ------------------------


def test_the_bundled_dashboard_is_found_through_the_environment(monkeypatch, tmp_path):
    """One rule for both builds: a checkout and a bundle differ only in a path."""
    import importlib

    dist = tmp_path / "ui" / "dist"
    dist.mkdir(parents=True)
    (dist / "index.html").write_text("<!doctype html><title>x</title>", encoding="utf-8")
    monkeypatch.setenv("BACKTEST_UI_DIST", str(dist))

    import api.main

    reloaded = importlib.reload(api.main)
    try:
        assert reloaded.ui_is_built()
        assert dist == reloaded.UI_DIST
    finally:
        monkeypatch.delenv("BACKTEST_UI_DIST", raising=False)
        importlib.reload(api.main)


def test_an_index_naming_assets_that_are_gone_does_not_count_as_built(
    monkeypatch, tmp_path
):
    """The failure that actually happened, turned into a guard.

    A packaged build assembled while the frontend was being rebuilt shipped an
    index.html naming a bundle that had already been renamed. Every file that
    was checked for existed, so the application started, served the page, and
    opened a window on nothing at all.
    """
    import importlib

    dist = tmp_path / "ui" / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text(
        '<!doctype html><html><head>'
        '<link rel="stylesheet" href="/assets/index-OLD.css">'
        '</head><body><script src="/assets/index-OLD.js"></script></body></html>',
        encoding="utf-8",
    )
    (dist / "assets" / "index-NEW.js").write_text("//", encoding="utf-8")
    monkeypatch.setenv("BACKTEST_UI_DIST", str(dist))

    import api.main

    reloaded = importlib.reload(api.main)
    try:
        assert not reloaded.ui_is_built()
        # and the whole thing passes once the names line up again
        (dist / "assets" / "index-OLD.js").write_text("//", encoding="utf-8")
        (dist / "assets" / "index-OLD.css").write_text("/**/", encoding="utf-8")
        assert reloaded.ui_is_built()
    finally:
        monkeypatch.delenv("BACKTEST_UI_DIST", raising=False)
        importlib.reload(api.main)


def test_an_unbuilt_dashboard_says_how_to_build_it(monkeypatch, tmp_path):
    """A bare 404 at "/" told nobody that `npm run build` was the answer."""
    import importlib

    from fastapi.testclient import TestClient

    monkeypatch.setenv("BACKTEST_UI_DIST", str(tmp_path / "absent"))
    import api.main

    reloaded = importlib.reload(api.main)
    try:
        assert not reloaded.ui_is_built()
        with TestClient(reloaded.app) as client:
            response = client.get("/")
        assert response.status_code == 503
        assert "npm run build" in response.json()["detail"]
    finally:
        monkeypatch.delenv("BACKTEST_UI_DIST", raising=False)
        importlib.reload(api.main)


def test_the_application_serves_the_fixture_with_metatrader5_absent(monkeypatch):
    """The claim the packaged build rests on: no terminal, no package, still runs.

    `mt5 = None` is exactly the state on a machine where the Windows-only
    package was never installed, and it is the state the desktop build has to
    survive - somebody opening it on a laptop with no broker account should
    get an application with the synthetic dataset in it, not a stack trace.
    """
    import importlib

    from fastapi.testclient import TestClient

    from core.data import mt5_provider
    from core.data.fixture_provider import FIXTURE_CACHE

    monkeypatch.setattr(mt5_provider, "mt5", None)
    assert not mt5_provider.mt5_available()

    monkeypatch.setenv("BACKTEST_CACHE_DIR", str(FIXTURE_CACHE))
    import api.main

    reloaded = importlib.reload(api.main)
    try:
        with TestClient(reloaded.app) as client:
            health = client.get("/api/health")
            symbols = client.get("/api/symbols")
        assert health.status_code == 200
        assert health.json()["synthetic_fixture"] is True
        assert symbols.status_code == 200
        # invented bars, and the response says so rather than looking real
        assert symbols.json()["source"] == "fixture"
        assert symbols.json()["cached_symbols"]
    finally:
        monkeypatch.delenv("BACKTEST_CACHE_DIR", raising=False)
        importlib.reload(api.main)


def test_downloading_is_refused_against_the_fixture(monkeypatch):
    """The fixture is a fixed artefact, not a cache to fill from a broker."""
    import importlib

    from fastapi.testclient import TestClient

    from core.data.fixture_provider import FIXTURE_CACHE

    monkeypatch.setenv("BACKTEST_CACHE_DIR", str(FIXTURE_CACHE))
    import api.main

    reloaded = importlib.reload(api.main)
    try:
        with TestClient(reloaded.app) as client:
            response = client.post(
                "/api/data/download",
                json={
                    "symbol": "SYNTHFX",
                    "timeframe": "M1",
                    "start": "2024-01-01T00:00:00Z",
                    "end": "2024-02-01T00:00:00Z",
                },
            )
        assert response.status_code == 409
        assert "fixture" in response.json()["detail"]
    finally:
        monkeypatch.delenv("BACKTEST_CACHE_DIR", raising=False)
        importlib.reload(api.main)


# -- the icon the window and the executable share ------------------------


@pytest.mark.parametrize("name", ["falsify-mark.svg", "falsify-icon.svg", "falsify.ico"])
def test_the_brand_files_the_build_needs_are_present(name):
    assert (Path(__file__).resolve().parents[1] / "brand" / name).is_file()


def test_the_icon_holds_every_size_windows_asks_for():
    """A .ico with one size in it looks like a thumbnail in the taskbar."""
    import struct

    from scripts.make_icons import ICO_SIZES

    raw = (Path(__file__).resolve().parents[1] / "brand" / "falsify.ico").read_bytes()
    reserved, kind, count = struct.unpack_from("<HHH", raw, 0)
    assert (reserved, kind) == (0, 1)
    assert count == len(ICO_SIZES)
    sizes = []
    for index in range(count):
        width, _, _, _, _, bits, length, offset = struct.unpack_from(
            "<BBBBHHII", raw, 6 + 16 * index
        )
        sizes.append(width or 256)
        assert bits == 32
        assert raw[offset : offset + 8] == b"\x89PNG\r\n\x1a\n"
        assert offset + length <= len(raw)
    assert sizes == list(ICO_SIZES)


def test_the_mark_is_regenerated_identically():
    """The committed SVG is the generator's output, not a hand edit of it."""
    from scripts.make_icons import mark_svg

    committed = (Path(__file__).resolve().parents[1] / "brand" / "falsify-mark.svg")
    assert committed.read_text(encoding="utf-8") == mark_svg()
