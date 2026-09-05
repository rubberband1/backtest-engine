"""Where long jobs say how far in they are.

The research endpoints are blocking calls on purpose: a permutation test
returns one report, and turning every one of them into a job with an
artefact to fetch afterwards would be a larger change to the API than the
problem deserves. What was missing is narrower - while such a call is in
flight, the client has nothing to show but a spinner, and "still working" is
not the same information as "iteration 340 of 1000".

So the caller invents a token, sends it with the request, and polls
`/api/progress/{token}` beside the request it is already waiting on. The
response of the call itself is untouched: nothing here can change a number.

FastAPI runs a non-async path operation in a worker thread, which is why the
polling request is answered while the blocking one is still running.
"""
from __future__ import annotations

import re
import threading
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

# The token reaches this module from a URL path and becomes a dictionary key.
# Nothing here touches the filesystem with it, but a shape is cheaper than
# trusting that to stay true.
TOKEN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

# Enough to cover every job a session can have in flight, plus the recently
# finished ones a client may still be polling. The report each job produces is
# returned by the call itself, so nothing of value is lost when one is dropped.
HISTORY = 32


class InvalidToken(ValueError):
    """The caller sent something that is not a progress token."""


@dataclass
class Job:
    """One unit of work a client is waiting on."""

    token: str
    label: str
    status: Literal["running", "done", "error"] = "running"
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: datetime | None = None
    # zero means the work has no countable steps: the client shows that it is
    # running rather than inventing a percentage
    completed: int = 0
    total: int = 0
    current: str | None = None
    error: str | None = None


_jobs: OrderedDict[str, Job] = OrderedDict()
_lock = threading.Lock()


def check(token: str) -> str:
    if not TOKEN.match(token):
        raise InvalidToken(
            "a progress token is up to 64 letters, digits, hyphens or underscores"
        )
    return token


def start(token: str, label: str, total: int = 0) -> Job:
    check(token)
    job = Job(token=token, label=label, total=max(0, total))
    with _lock:
        _jobs[token] = job
        _jobs.move_to_end(token)
        while len(_jobs) > HISTORY:
            _jobs.popitem(last=False)
    return job


def advance(
    token: str, completed: int, total: int | None = None, current: str | None = None
) -> None:
    with _lock:
        job = _jobs.get(token)
        if job is None or job.status != "running":
            return
        job.completed = completed
        if total is not None:
            job.total = total
        if current is not None:
            job.current = current


def finish(token: str, error: str | None = None) -> None:
    with _lock:
        job = _jobs.get(token)
        if job is None:
            return
        job.status = "error" if error else "done"
        job.error = error
        job.finished_at = datetime.now(timezone.utc)


def read(token: str) -> Job | None:
    check(token)
    with _lock:
        return _jobs.get(token)


def reporter(token: str | None) -> Callable[[int, int], None] | None:
    """The callback shape the core loops take, or None when nobody is watching.

    Returning None rather than a no-op is deliberate: a core function that
    receives None does not call anything at all, so a run started from a
    script or a test carries no reporting machinery whatsoever.
    """
    if token is None:
        return None
    check(token)

    def report(completed: int, total: int) -> None:
        advance(token, completed, total)

    return report


class watching:
    """Marks a job running for the length of a `with` block.

    An exception is recorded on the job and re-raised: the client polling it
    is told the work stopped, and the request it is waiting on still fails
    the way it always did.
    """

    def __init__(self, token: str | None, label: str, total: int = 0) -> None:
        self.token = token
        if token is not None:
            start(token, label, total)

    def __enter__(self) -> watching:
        return self

    def __exit__(self, kind: object, value: object, traceback: object) -> Literal[False]:
        if self.token is not None:
            finish(self.token, None if value is None else f"{type(value).__name__}: {value}")
        return False

    def step(self, completed: int, total: int | None = None, current: str | None = None) -> None:
        if self.token is not None:
            advance(self.token, completed, total, current)
