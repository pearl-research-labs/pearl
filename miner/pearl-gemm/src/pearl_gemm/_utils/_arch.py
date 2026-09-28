"""Architecture families (one per compute-capability major) and the launch gate.

SM100: datacenter Blackwell (tcgen05, TMEM, clusters). SM120: workstation and
consumer Blackwell (warp-level ``mma.sync``, ~99 KB smem, no TMEM/clusters).
Each host declares the families it runs on through ``require_arch``.
"""

from enum import Enum

import cutlass.utils
import torch

from ..protocol_constants import SM100_CC_MAJOR, SM120_CC_MAJOR


class Arch(Enum):
    """One compute-capability major, named after its lead SM target."""

    SM100 = SM100_CC_MAJOR
    SM120 = SM120_CC_MAJOR

    @property
    def smem_capacity_bytes(self) -> int:
        """Opt-in dynamic shared memory per CTA of the family's lead target, ``sm_<major>0``."""
        return cutlass.utils.get_smem_capacity_in_bytes(f"sm_{self.value}0")


def arch_of(device: torch.device | int | tuple[int, int] | None = None) -> Arch:
    """The family of a CUDA device (the current one when omitted) or of a
    ``(major, minor)`` compute capability; unknown majors raise."""
    major, minor = device if isinstance(device, tuple) else torch.cuda.get_device_capability(device)
    try:
        return Arch(major)
    except ValueError:
        known = ", ".join(arch.name for arch in Arch)
        raise ValueError(
            f"unsupported GPU architecture sm{major}{minor}; kernels exist for {known}"
        ) from None


def require_arch(entry: str, device: torch.device | int | None, *supported: Arch) -> Arch:
    """Reject a launch of ``entry`` on a device outside ``supported``; return the family.

    Unknown majors fail with the same message shape as known-but-unsupported
    ones, so callers see one error surface.
    """
    major, minor = torch.cuda.get_device_capability(device)
    arch = Arch(major) if major in Arch._value2member_map_ else None
    if arch is None or arch not in supported:
        names = ", ".join(a.name for a in supported)
        want = names if len(supported) == 1 else f"one of {names}"
        raise ValueError(f"{entry} requires {want}, got sm{major}{minor}")
    return arch
