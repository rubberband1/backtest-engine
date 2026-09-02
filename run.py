"""Starts backend and frontend and opens the browser. One command.

    python run.py

In order: exports the OpenAPI schema, regenerates the TypeScript types if the
schema changed, starts uvicorn and the Vite dev server, waits until they
actually respond (not "wait five seconds and hope"), opens the page. Ctrl+C
stops everything.

Everything local: no keys, no external services, no telemetry.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

ROOT = Path(__file__).parent.resolve()
UI_DIR = ROOT / "ui"
OPENAPI_FILE = UI_DIR / "src" / "api" / "openapi.json"
TYPES_FILE = UI_DIR / "src" / "api" / "schema.d.ts"

STARTUP_TIMEOUT = 90.0

# The Windows console is cp1252: without this, the first arrow printed by
# Vite kills the log-forwarding thread and the subprocess ends up blocked on
# a full pipe.
for stream_out in (sys.stdout, sys.stderr):
    if hasattr(stream_out, "reconfigure"):
        stream_out.reconfigure(errors="replace")

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(name)s | %(message)s")
logger = logging.getLogger("run")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-port", type=int, default=8000)
    parser.add_argument("--ui-port", type=int, default=5173)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--no-ui", action="store_true", help="backend only")
    parser.add_argument("--reload", action="store_true", help="uvicorn in auto-reload")
    return parser.parse_args()


def port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return probe.connect_ex(("127.0.0.1", port)) != 0


def export_schema() -> bool:
    """Writes openapi.json. True if it changed."""
    from api.main import app

    OPENAPI_FILE.parent.mkdir(parents=True, exist_ok=True)
    previous = OPENAPI_FILE.read_text(encoding="utf-8") if OPENAPI_FILE.exists() else ""
    payload = json.dumps(app.openapi(), indent=2, ensure_ascii=False) + "\n"
    if payload != previous:
        OPENAPI_FILE.write_text(payload, encoding="utf-8")
        return True
    return False


def npm() -> str | None:
    return shutil.which("npm") or shutil.which("npm.cmd")


def prepare_ui(schema_changed: bool) -> bool:
    """Dependencies installed and types aligned to the schema."""
    executable = npm()
    if executable is None:
        logger.error("npm not found in PATH: the frontend cannot start")
        return False

    if not (UI_DIR / "node_modules").exists():
        logger.info("installing frontend dependencies (one time only)...")
        result = subprocess.run([executable, "install", "--no-fund", "--no-audit"], cwd=UI_DIR)
        if result.returncode != 0:
            logger.error("npm install failed")
            return False

    if schema_changed or not TYPES_FILE.exists():
        logger.info("the OpenAPI schema changed: regenerating the TypeScript types")
        result = subprocess.run([executable, "run", "gen:api"], cwd=UI_DIR)
        if result.returncode != 0:
            logger.error("type generation failed: the UI would use stale types")
            return False
    return True


def wait_for(url: str, label: str, process: subprocess.Popen, timeout: float) -> bool:
    """Waits until the address actually responds, not for a fixed time."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            logger.error("%s exited immediately (code %s)", label, process.returncode)
            return False
        try:
            with urlopen(url, timeout=1.5) as response:
                if response.status < 500:
                    return True
        except (URLError, OSError, ConnectionError):
            time.sleep(0.4)
    logger.error("%s is not responding on %s after %.0fs", label, url, timeout)
    return False


def stream(process: subprocess.Popen, prefix: str) -> None:
    """Relays the subprocess output with a prefix, without mixing it."""
    assert process.stdout is not None
    for line in process.stdout:
        text = line.rstrip()
        if text:
            print(f"[{prefix}] {text}", flush=True)


def spawn(command: list[str], cwd: Path, prefix: str) -> subprocess.Popen:
    process = subprocess.Popen(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env={**os.environ, "PYTHONUNBUFFERED": "1", "FORCE_COLOR": "0"},
    )
    threading.Thread(target=stream, args=(process, prefix), daemon=True).start()
    return process


def terminate(process: subprocess.Popen | None, label: str) -> None:
    if process is None or process.poll() is not None:
        return
    logger.info("stopping %s", label)
    process.terminate()
    try:
        process.wait(timeout=8)
    except subprocess.TimeoutExpired:
        process.kill()


def main() -> int:
    args = parse_args()
    os.chdir(ROOT)

    for port, label in ((args.api_port, "API"), (args.ui_port, "UI")):
        if not args.no_ui or label == "API":
            if not port_is_free(port):
                logger.error(
                    "port %d (%s) is already taken: close the other process or pass "
                    "--%s-port",
                    port, label, "api" if label == "API" else "ui",
                )
                return 1

    schema_changed = export_schema()
    logger.info(
        "OpenAPI schema %s", "updated" if schema_changed else "already aligned"
    )

    ui_ready = True if args.no_ui else prepare_ui(schema_changed)
    if not ui_ready and not args.no_ui:
        logger.error("frontend not ready: starting the backend only")

    backend = frontend = None
    try:
        command = [
            sys.executable, "-m", "uvicorn", "api.main:app",
            "--host", "127.0.0.1", "--port", str(args.api_port),
        ]
        if args.reload:
            command.append("--reload")
        backend = spawn(command, ROOT, "api")
        if not wait_for(
            f"http://127.0.0.1:{args.api_port}/api/health", "the backend", backend,
            STARTUP_TIMEOUT,
        ):
            return 1
        logger.info("API ready on http://127.0.0.1:%d/docs", args.api_port)

        url = f"http://127.0.0.1:{args.ui_port}/"
        if ui_ready and not args.no_ui:
            executable = npm()
            assert executable is not None
            frontend = spawn(
                [executable, "run", "dev", "--", "--port", str(args.ui_port), "--strictPort"],
                UI_DIR, "ui",
            )
            if not wait_for(url, "the frontend", frontend, STARTUP_TIMEOUT):
                return 1
            logger.info("dashboard ready on %s", url)
        else:
            url = f"http://127.0.0.1:{args.api_port}/docs"

        if not args.no_browser:
            webbrowser.open(url)

        logger.info("running. Ctrl+C stops everything.")
        while True:
            if backend.poll() is not None:
                logger.error("the backend exited (code %s)", backend.returncode)
                return 1
            if frontend is not None and frontend.poll() is not None:
                logger.error("the frontend exited (code %s)", frontend.returncode)
                return 1
            time.sleep(0.5)
    except KeyboardInterrupt:
        print()
        logger.info("interrupted")
        return 0
    finally:
        terminate(frontend, "the frontend")
        terminate(backend, "the backend")


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal.default_int_handler)
    raise SystemExit(main())
