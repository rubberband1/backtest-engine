from __future__ import annotations

import pytest


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skips tests marked `mt5` if the terminal is not reachable."""
    try:
        from core.data.mt5_provider import MT5Provider, mt5_available
    except Exception:  # pragma: no cover
        available = False
    else:
        available = mt5_available()
        if available:
            try:
                provider = MT5Provider()
                provider.connect()
                provider.disconnect()
            except Exception:
                available = False

    if available:
        return

    skip = pytest.mark.skip(reason="MetaTrader 5 terminal not available")
    for item in items:
        if "mt5" in item.keywords:
            item.add_marker(skip)
