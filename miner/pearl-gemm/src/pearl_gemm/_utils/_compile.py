"""Small shared helpers for tensor-free CuTe compilation and caching."""

import functools
import threading
from collections.abc import Callable

import cutlass.cute as cute

# One lock for every compile entry point: ``cute.compile`` mutates DSL-global
# state. First launches also initialize process-global CUDA library state.
_compile_lock = threading.Lock()


def single_flight_compile(fn):
    """Cache ``fn`` per argument tuple: single-flight and never evicting.

    A compiled CuTe callable owns its executor/module, whose teardown unloads
    the CUDA library while queued launches or captured graphs may still hold
    the kernel, so entries must live for the process (no LRU).
    Compilation and the first launch are serialized. After initialization the
    cache holds the original callable, so warm launches stay lock-free.
    """
    cache: dict = {}

    @functools.wraps(fn)
    def cached(*args):
        compiled = cache.get(args)
        if compiled is None:
            with _compile_lock:
                compiled = cache.get(args)
                if compiled is None:
                    target = fn(*args)

                    @functools.wraps(target)
                    def first_launch(*launch_args, **launch_kwargs):
                        # CuTe's lazy library initializer can strand its global
                        # lock when two threads initialize the same module.
                        with _compile_lock:
                            if cache[args] is first_launch:
                                result = target(*launch_args, **launch_kwargs)
                                cache[args] = target
                                return result
                        return target(*launch_args, **launch_kwargs)

                    compiled = cache[args] = first_launch
        return compiled

    cached.cache = cache
    return cached


def get_or_compile[CompiledT](
    cache: dict[tuple, CompiledT],
    key: tuple,
    factory: Callable[[], CompiledT],
) -> CompiledT:
    """Return a compiled variant, creating it once for a metadata key."""
    compiled = cache.get(key)
    if compiled is None:
        compiled = factory()
        cache[key] = compiled
    return compiled


def make_fake_tensor(dtype, shape, *, leading_dim=-1, divisibility=1):
    """Build a symbolic CuTe tensor matching a dynamic DLPack layout."""
    if dtype is None:
        return None
    if leading_dim is not None and leading_dim < 0:
        leading_dim += len(shape)
    stride = tuple(
        1 if index == leading_dim else cute.sym_int64(divisibility=divisibility)
        for index in range(len(shape))
    )
    return cute.runtime.make_fake_tensor(
        dtype,
        shape,
        stride=stride,
        assumed_align=divisibility * dtype.width // 8,
    )


def make_fake_stream():
    """Build a tensor-free stream placeholder for a compiled launch."""
    return cute.runtime.make_fake_stream()
