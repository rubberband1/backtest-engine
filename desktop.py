"""Falsify as a window, with the backend inside it.

    python desktop.py

One process and one language: the API is Python already, so PyWebview hosts
the compiled frontend against a uvicorn started in a thread of the same
interpreter. There is no Node here and no dev server - `ui/dist` is served by
FastAPI, which is also what the packaged executable does.

`python run.py` still works and is still the development path: it starts Vite,
watches the sources and reloads. This file is the other one, and neither
replaces the other.

The three ways this used to fail silently, and what happens instead:

- port 8000 taken. It asks the operating system for a free port rather than
  refusing to start on a number nobody chose.
- backend does not come up. The window opens on a page that says so, with the
  traceback, instead of closing before anybody can read it.
- a live runner is trading. Closing the window asks first, and says what it
  would be walking away from.
"""
from __future__ import annotations

import argparse
import logging
import os
import socket
import sys
import threading
import traceback
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent

# PyInstaller unpacks a onefile build into a temporary directory and points
# `sys._MEIPASS` at it. The built dashboard and the icon live there, while the
# data, the runs and the strategies stay next to the executable, where the
# person running it can see them.
BUNDLE = Path(getattr(sys, "_MEIPASS", ROOT))
FROZEN = getattr(sys, "frozen", False)

STARTUP_TIMEOUT = 60.0
MIN_SIZE = (960, 640)
WINDOW_SIZE = (1440, 900)

logger = logging.getLogger("desktop")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--port", type=int, default=0, help="0 asks the OS for a free one"
    )
    parser.add_argument(
        "--fixture",
        action="store_true",
        help="serve the synthetic dataset instead of data_cache/. Implied on a "
        "machine with no data, which is what a fresh install is",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def free_port(preferred: int = 0) -> int:
    """A port nothing else holds.

    Binding to 0 and reading the number back is the only way to pick one
    without a race that a retry loop would still lose. `preferred` is honoured
    when it happens to be free, so `--port` stays meaningful.
    """
    if preferred:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if probe.connect_ex(("127.0.0.1", preferred)) != 0:
                return preferred
        logger.warning("port %d is taken: asking for another", preferred)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as chosen:
        chosen.bind(("127.0.0.1", 0))
        return int(chosen.getsockname()[1])


def wait_for(url: str, timeout: float, alive: threading.Event) -> bool:
    """Until the backend answers, it failed, or the deadline passes."""
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not alive.is_set():
            return False
        try:
            with urlopen(url, timeout=1.5) as response:
                if response.status < 500:
                    return True
        except (URLError, OSError):
            time.sleep(0.25)
    return False


class Backend:
    """uvicorn in a thread of this process, and what it did if it did not start."""

    def __init__(self, port: int) -> None:
        self.port = port
        self.error: str | None = None
        self.alive = threading.Event()
        self._server: object | None = None
        self._thread: threading.Thread | None = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> None:
        self.alive.set()
        self._thread = threading.Thread(target=self._serve, name="uvicorn", daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        try:
            import uvicorn

            from api.main import app, ui_is_built

            if not ui_is_built():
                raise RuntimeError(
                    "the dashboard is not built. Run `npm run build` in ui/ "
                    "before starting the desktop application"
                )
            config = uvicorn.Config(
                app, host="127.0.0.1", port=self.port, log_level="warning"
            )
            server = uvicorn.Server(config)
            # uvicorn installs signal handlers, which it may only do on the
            # main thread; this one is not it
            server.install_signal_handlers = lambda: None  # type: ignore[method-assign]
            self._server = server
            server.run()
        except BaseException:
            self.error = traceback.format_exc()
            logger.error("the backend stopped:\n%s", self.error)
        finally:
            self.alive.clear()

    def stop(self) -> None:
        server = self._server
        if server is not None:
            server.should_exit = True  # type: ignore[attr-defined]
        if self._thread is not None:
            self._thread.join(timeout=8)


def live_runners() -> list[str]:
    """Sessions with a live process behind them, by name.

    Read from the diaries and their locks, the same way the API does it: the
    desktop shell never talks to MT5, and closing a window must not be the
    thing that discovers a broker connection.
    """
    try:
        from core.live.lock import RunLock, process_alive

        live_dir = Path(os.environ.get("BACKTEST_LIVE_DIR", "logs/live"))
        if not live_dir.exists():
            return []
        running = []
        for lock_path in sorted(live_dir.glob("*.lock")):
            info = RunLock(lock_path).read()
            if info and process_alive(info.pid):
                running.append(f"{lock_path.stem} (pid {info.pid})")
        return running
    except Exception:
        # a shell that cannot answer this question must not block the close;
        # it says nothing rather than inventing a reassurance
        logger.exception("could not check for live runners")
        return []


def seed_workspace() -> None:
    """Puts a strategy library beside the executable on the first run.

    Only in a packaged build, and only when there is nothing there already.
    The bundle's copy is read-only and, for a onefile build, deleted on exit -
    a strategy saved into it would vanish. What the application writes belongs
    next to the executable, where the person running it can find it.
    """
    if not FROZEN:
        return
    import shutil

    local = Path("strategies")
    bundled = BUNDLE / "strategies"
    if local.exists() or not bundled.is_dir():
        return
    shutil.copytree(bundled, local)
    logger.info("first run: copied the strategy library to %s", local.resolve())


# The mark, as three rectangles on a 32 grid. Repeated here rather than read
# from brand/ because both pages below are shown when the files this
# application ships with may be exactly what is missing.
MARK = (
    '<svg width="34" height="34" viewBox="0 0 32 32" fill="#1b1f26">'
    '<rect x="10" y="3" width="4.5" height="26"/>'
    '<rect x="10" y="3" width="17" height="4.5"/>'
    '<rect x="2.5" y="16.75" width="22" height="4.5" '
    'transform="rotate(-30 13.5 19)"/></svg>'
)

_SHELL = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>{title}</title>
<style>
  :root {{ color-scheme: light; }}
  body {{ margin:0; padding:44px 40px; background:#fbfbfc; color:#1b1f26;
         font:15px/1.55 "Segoe UI", system-ui, sans-serif; }}
  header {{ display:flex; align-items:center; gap:11px; margin-bottom:26px; }}
  .word {{ font-weight:600; font-size:15px; letter-spacing:.17em; }}
  h1 {{ font-size:20px; margin:0 0 6px; letter-spacing:-.01em; }}
  p {{ margin:0 0 18px; color:#4b545e; max-width:64ch; }}
  pre {{ background:#f1f2f4; border:1px solid #dcdfe3; border-radius:4px;
         padding:14px 16px; overflow:auto; font:12px/1.5 Consolas, monospace;
         max-height:52vh; }}
  .waiting {{ display:flex; align-items:center; gap:14px; }}
  .spinner circle {{ fill:none; stroke-linecap:round; transform-origin:20px 20px;
                    animation:turn var(--turn) linear infinite; }}
  .outer {{ stroke:#c3c8ce; --turn:2.6s; }}
  .middle {{ stroke:#6b747d; --turn:1.9s; animation-direction:reverse; }}
  .inner {{ stroke:#1b1f26; --turn:1.3s; }}
  @keyframes turn {{ to {{ transform:rotate(360deg); }} }}
  @media (prefers-reduced-motion: reduce) {{
    .spinner circle {{ animation-duration:3s; }}
  }}
</style></head>
<body>
  <header>{mark}<span class="word">FALSIFY</span></header>
  {body}
</body></html>"""


def splash_page(port: int) -> str:
    """Shown from the moment the window opens until the backend answers.

    The window used to be created only after the backend was up, so starting
    the application meant several seconds of nothing on screen at all. This
    appears immediately and says what is being waited on - the same rule the
    dashboard follows: never "loading", always which work.
    """
    spinner = (
        '<svg class="spinner" width="30" height="30" viewBox="0 0 40 40">'
        '<circle class="outer" cx="20" cy="20" r="15" stroke-width="3.5" '
        'stroke-dasharray="24.5 94.25"/>'
        '<circle class="middle" cx="20" cy="20" r="10.5" stroke-width="3.5" '
        'stroke-dasharray="19.8 65.97"/>'
        '<circle class="inner" cx="20" cy="20" r="6" stroke-width="3.5" '
        'stroke-dasharray="12.8 37.7"/></svg>'
    )
    body = (
        f'<div class="waiting">{spinner}<div>'
        f"<h1>Starting the engine</h1>"
        f'<p style="margin:0">Loading the cache index and the strategy library, '
        f"then serving the dashboard on 127.0.0.1:{port}. Nothing leaves this "
        f"machine.</p></div></div>"
    )
    return _SHELL.format(title="Falsify", mark=MARK, body=body)


def failure_page(message: str, detail: str) -> str:
    """What the window shows when there is no backend to show.

    Written here rather than loaded from `ui/dist`, because the case this
    exists for is the one where that directory is the problem.
    """
    from html import escape

    body = (
        f"<h1>Falsify did not start</h1>"
        f"<p>{escape(message)}</p><pre>{escape(detail)}</pre>"
    )
    return _SHELL.format(title="Falsify - did not start", mark=MARK, body=body)


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=args.log_level.upper(), format="%(levelname)-7s %(name)s | %(message)s"
    )
    os.chdir(ROOT if not FROZEN else Path(sys.executable).parent)

    # The bundled build carries the dashboard inside it; a source checkout has
    # it in ui/dist. Both are found through the same environment variable the
    # API reads, so there is one rule rather than two code paths.
    os.environ.setdefault("BACKTEST_UI_DIST", str(BUNDLE / "ui" / "dist"))
    seed_workspace()
    if args.fixture:
        from core.data.fixture_provider import FIXTURE_CACHE

        os.environ["BACKTEST_CACHE_DIR"] = str(FIXTURE_CACHE)

    try:
        import webview
    except ImportError:
        print(
            "PyWebview is not installed. Install the desktop extra:\n"
            "    pip install -e \".[desktop]\"",
            file=sys.stderr,
        )
        return 2

    backend = Backend(free_port(args.port))
    backend.start()

    icon = BUNDLE / "brand" / "falsify.ico"
    closing = threading.Event()
    ready = threading.Event()

    # The window opens on the splash straight away and swaps to the dashboard
    # when the backend answers. Waiting first and creating the window
    # afterwards meant several seconds of nothing on screen, which is
    # indistinguishable from an application that failed to start.
    window = webview.create_window(
        "Falsify",
        html=splash_page(backend.port),
        width=WINDOW_SIZE[0],
        height=WINDOW_SIZE[1],
        min_size=MIN_SIZE,
        # no address bar, no browser furniture: it is an application window
        # that happens to be drawn by a web view
        text_select=True,
    )

    def on_started() -> None:
        """Runs once the GUI loop is up, on its own thread."""
        if wait_for(f"{backend.base_url}/api/health", STARTUP_TIMEOUT, backend.alive):
            ready.set()
            window.load_url(backend.base_url)
            return
        detail = backend.error or (
            f"No answer from {backend.base_url}/api/health after "
            f"{STARTUP_TIMEOUT:.0f} seconds."
        )
        window.load_html(
            failure_page(
                "The backend did not come up, so there is nothing to show. The "
                "detail below is the reason, not a summary of it.",
                detail,
            )
        )

    def on_closing() -> bool:
        """False keeps the window open. A trading runner gets a question first."""
        if closing.is_set():
            return True
        running = live_runners()
        if running:
            logger.warning("closing with %d live runner(s): %s", len(running), running)
        if running:
            answer = window.create_confirmation_dialog(
                "A live runner is still going",
                "These runner processes are alive and are not stopped by closing "
                "this window:\n\n"
                + "\n".join(f"  {name}" for name in running)
                + "\n\nThey keep trading in their own processes. Close Falsify "
                "anyway?",
            )
            if not answer:
                return False
        closing.set()
        backend.stop()
        return True

    window.events.closing += on_closing
    # `icon` is honoured by the GTK and Qt backends. On Windows the taskbar
    # icon comes from the executable, which is where PyInstaller puts the same
    # .ico; passing it here costs nothing and covers the other platforms.
    webview.start(on_started, icon=str(icon) if icon.is_file() else None)
    backend.stop()
    return 0 if ready.is_set() else 1


if __name__ == "__main__":
    raise SystemExit(main())
