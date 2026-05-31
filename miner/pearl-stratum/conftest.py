"""pytest config for pearl-stratum."""

import pytest


def pytest_collection_modifyitems(config, items):
    """Mark all async tests as asyncio. Works without pytest-asyncio config in pyproject."""
    for item in items:
        if item.get_closest_marker("asyncio") is None:
            try:
                import inspect
                if inspect.iscoroutinefunction(getattr(item, "function", None)):
                    item.add_marker(pytest.mark.asyncio)
            except Exception:
                pass
