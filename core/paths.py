"""How a directory is named when it leaves this process.

The dashboard ends up in screenshots and screen recordings, and the terminal
banner ends up in the README. On a machine holding the project inside the
user's home directory, an absolute path puts the account name in both. Nothing
downstream needs the location - only whether the directory is the one the
project means - so what is shown is the path relative to the project root, and
"external" for anything outside it.
"""
from __future__ import annotations

import sys
from pathlib import Path


def _project_root() -> Path:
    """Where "the project" is, from the point of view of a path being named.

    In a checkout that is the repository. In a PyInstaller build it is the
    directory holding the executable, not the temporary one the bundle is
    unpacked into: everything a packaged run writes - the runs, the logs, the
    downloaded bars - lands beside the executable, and naming those "external"
    left the application unable to say where its own output had gone.
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[1]


PROJECT_ROOT = _project_root()


def project_relative(path: Path | str) -> str:
    try:
        return Path(path).resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return "external"
