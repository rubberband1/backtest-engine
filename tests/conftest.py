from __future__ import annotations

import pytest


def _live_feed_reason() -> str | None:
    """Why the `mt5` tests cannot run, or None when they can.

    Reaching the terminal is not enough. Every one of these tests reads bars
    or ticks, and both are expressed on the server clock, which is measured
    by comparing a live quote against the local one. With the market closed
    the newest quote is as old as the session has been shut, and the provider
    refuses to name a timezone from it rather than rounding its age into a
    plausible-looking offset - so these tests have nothing to stand on, and
    are skipped with the reason instead of failing as if the code were
    broken.
    """
    try:
        from core.data.mt5_provider import MT5Provider, mt5_available
    except Exception as exc:  # pragma: no cover - the package is optional
        return f"MetaTrader5 package unavailable ({exc})"

    if not mt5_available():
        return "MetaTrader5 package unavailable"

    try:
        provider = MT5Provider()
        provider.connect()
    except Exception as exc:
        return f"MetaTrader 5 terminal not available ({exc})"

    try:
        provider.server_timezone  # noqa: B018 - the access is the probe
    except Exception as exc:
        return (
            f"the terminal is connected but the server clock cannot be "
            f"measured ({type(exc).__name__}): the market is closed and the "
            f"newest quote is stale"
        )
    finally:
        provider.disconnect()
    return None


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skips tests marked `mt5` when there is no live feed to test against."""
    reason = _live_feed_reason()
    if reason is None:
        return

    skip = pytest.mark.skip(reason=reason)
    for item in items:
        if "mt5" in item.keywords:
            item.add_marker(skip)
