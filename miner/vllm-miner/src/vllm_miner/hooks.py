"""One shared monkey-patch scaffold for the framework hook installs."""

import functools
from collections.abc import Callable


def wrap_callable_once(
    owner: object, name: str, make_wrapper: Callable, *, require: bool = True
) -> None:
    """Replace ``owner.<name>`` with ``make_wrapper(original)``; re-wrapping is a no-op.

    ``require=False`` skips a missing attribute (optional entry point) instead
    of raising. ``functools.wraps`` is applied here; factories must not re-apply it.
    """
    original = getattr(owner, name) if require else getattr(owner, name, None)
    if original is None or getattr(original, "_pearl_wrapped_once", False):
        return
    wrapper = functools.wraps(original)(make_wrapper(original))
    wrapper._pearl_wrapped_once = True
    setattr(owner, name, wrapper)
