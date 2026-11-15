"""Build/load the CPU B preparation kernels (csrc/b_prepare_cpu.cpp) once per box.

The build never initializes CUDA, so the CUDA-free preparation process can load it.
"""

import fcntl
import hashlib
import os
from functools import cache
from types import ModuleType

_SOURCE = os.path.join(os.path.dirname(__file__), "csrc", "b_prepare_cpu.cpp")
# -ffp-contract=off: every fp32 multiply must round on its own.
# -fopenmp: at::parallel_for expands to OpenMP pragmas.
_CFLAGS = ["-O3", "-march=native", "-fopenmp", "-ffp-contract=off", "-DNDEBUG"]
_LDFLAGS = ["-fopenmp"]


def _cpu_model() -> bytes:
    with open("/proc/cpuinfo", "rb") as cpuinfo:
        return next((line for line in cpuinfo if line.startswith(b"model name")), b"")


@cache
def native() -> ModuleType:
    from torch.utils.cpp_extension import load

    sha = hashlib.sha1(repr((_CFLAGS, _LDFLAGS)).encode())
    # -march=native: a build serves only the CPU model it was compiled on.
    sha.update(_cpu_model())
    with open(_SOURCE, "rb") as source:
        sha.update(source.read())
    base = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
    build = os.path.join(base, "vllm_miner_ext", sha.hexdigest()[:10])
    os.makedirs(build, exist_ok=True)
    # Concurrent processes: one builds, the rest wait.
    with open(os.path.join(build, ".lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return load(
            name="b_prepare_cpu_ext",
            sources=[_SOURCE],
            build_directory=build,
            extra_cflags=_CFLAGS,
            extra_ldflags=_LDFLAGS,
            verbose=False,
        )
