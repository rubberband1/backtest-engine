"""One runner per strategy and instrument, enforced by a file with a PID in it.

Two runners started by accident on the same spec do not produce two
independent experiments: they produce one position of twice the intended
size, with each of them believing it holds half. That is the sort of mistake
that is obvious afterwards and invisible at the time, so it is made
impossible rather than documented.

The lock is a file holding the PID, the start time and what it is locking.
Acquisition is `O_CREAT | O_EXCL`, which is atomic on Windows and POSIX
alike. A file left behind by a process that no longer exists is stale and is
taken over, with the takeover logged - a crash must not require manual
cleanup before trading can resume. A file whose PID *is* alive is respected,
always.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import Any

logger = logging.getLogger(__name__)


class LockHeld(RuntimeError):
    """Another live process is running for this strategy and instrument."""


def process_alive(pid: int) -> bool:
    """Whether a process with this id exists.

    On Windows `os.kill(pid, 0)` raises for a missing process and succeeds
    for an existing one, including one this user cannot signal - which is the
    conservative answer: an unknown process is treated as alive.
    """
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError as exc:  # Windows raises OSError(errno=EINVAL) for dead pids
        return getattr(exc, "winerror", None) not in (87, 6)
    except SystemError:
        # Windows, asked about a process it will not open at all - pid 4, the
        # System process, is the one that does this. CPython cannot turn that
        # into a clean OSError and raises here instead, which used to escape
        # this function entirely: a diary whose lock named such a pid took the
        # live endpoint down with a 500 rather than answering the question.
        # The process exists; not being allowed to look at it is the same
        # answer PermissionError already gets.
        return True
    return True


@dataclass(frozen=True)
class LockInfo:
    pid: int
    started_at: datetime
    label: str

    def to_json(self) -> str:
        return json.dumps(
            {
                "pid": self.pid,
                "started_at": self.started_at.isoformat(),
                "label": self.label,
            },
            indent=2,
        )

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> LockInfo:
        return cls(
            pid=int(payload["pid"]),
            started_at=datetime.fromisoformat(str(payload["started_at"])),
            label=str(payload.get("label", "")),
        )


class RunLock:
    """Exclusive lock on one (strategy, instrument) pair."""

    def __init__(self, path: Path | str, label: str = "") -> None:
        self.path = Path(path)
        self.label = label
        self._acquired = False

    def read(self) -> LockInfo | None:
        if not self.path.exists():
            return None
        try:
            return LockInfo.from_json(json.loads(self.path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            logger.warning("lock file %s is unreadable (%s)", self.path, exc)
            return None

    def acquire(self) -> LockInfo:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        info = LockInfo(
            pid=os.getpid(), started_at=datetime.now(timezone.utc), label=self.label
        )
        try:
            handle = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            existing = self.read()
            if existing is not None and process_alive(existing.pid):
                raise LockHeld(
                    f"a live runner is already running for {self.label or self.path.stem} "
                    f"(pid {existing.pid}, started {existing.started_at:%Y-%m-%d %H:%M:%S} "
                    f"UTC). Two runners on the same spec would double the position: "
                    f"stop that one first, or delete {self.path} if you are certain it "
                    f"is gone"
                ) from None
            logger.warning(
                "taking over a stale lock at %s (pid %s is not running)",
                self.path,
                existing.pid if existing else "unknown",
            )
            self.path.unlink(missing_ok=True)
            handle = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)

        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(info.to_json())
        self._acquired = True
        logger.info("lock acquired at %s (pid %d)", self.path, info.pid)
        return info

    def release(self) -> None:
        """Removes the lock, but only if this process is the one holding it."""
        if not self._acquired:
            return
        existing = self.read()
        if existing is not None and existing.pid != os.getpid():
            logger.warning(
                "not releasing %s: it now belongs to pid %d", self.path, existing.pid
            )
            self._acquired = False
            return
        self.path.unlink(missing_ok=True)
        self._acquired = False
        logger.info("lock released at %s", self.path)

    def __enter__(self) -> RunLock:
        self.acquire()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.release()
