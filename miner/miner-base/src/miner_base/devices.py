"""Resolve CUDA devices to the committed mining ``Device``.

The kernels pin lottery-critical arithmetic to the local GPU's FP8 MMA, so
the mining configuration -- hashed into every job key -- must commit the
matching ``Device``. The verifier replays that arithmetic as its ``H100`` or
``B200`` device.
"""

import torch

from .params import Device

# Compute-capability majors with a committed mining Device.
_HOPPER_CC_MAJOR = 9
_BLACKWELL_CC_MAJOR = 10
# SM120's warp-level FP8 MMA reproduces the Blackwell atom arithmetic bit for bit.
_SM120_CC_MAJOR = 12


def device_for_capability(major: int, minor: int = 0) -> Device:
    """Map a CUDA compute capability to the committed mining ``Device``.

    - SM90 (Hopper: H100/H200) -> ``Device.HOPPER``
    - SM100 (Blackwell: B200/B300) -> ``Device.BLACKWELL``
    - SM120 (RTX PRO 6000, RTX 50) -> ``Device.BLACKWELL`` (same atom arithmetic)

    Every other capability raises ``ValueError`` (``minor`` only names the
    capability in that error).
    """
    if major == _HOPPER_CC_MAJOR:
        return Device.HOPPER
    if major in (_BLACKWELL_CC_MAJOR, _SM120_CC_MAJOR):
        return Device.BLACKWELL
    raise ValueError(f"no committed mining Device for sm{major}{minor}")


def local_device(device: torch.device | int | None = None) -> Device:
    """The committed ``Device`` matching ``device``'s FP8 MMA (the current CUDA
    device when omitted). Fails closed on anything without a committed device."""
    return device_for_capability(*torch.cuda.get_device_capability(device))


__all__ = ["device_for_capability", "local_device"]
