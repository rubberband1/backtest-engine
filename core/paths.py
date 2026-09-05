"""How a directory is named when it leaves this process.

The dashboard ends up in screenshots and screen recordings, and the terminal
banner ends up in the README. On a machine holding the project inside the
user's home directory, an absolute path puts the account name in both. Nothing
downstream needs the location - only whether the directory is the one the
project means - so what is shown is the path relative to the project root, and
"external" for anything outside it.
"""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_relative(path: Path | str) -> str:
    try:
        return Path(path).resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return "external"
