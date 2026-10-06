"""Resolve CUDA devices to the committed mining ``Device``.

The kernels pin lottery-critical arithmetic to the local GPU's FP8 MMA, so
the mining configuration -- hashed into every job key -- must commit the
matching ``Device``. The verifier replays that arithmetic as its ``H100`` or
``B200`` device.
"""

import torch

from .params import Device

# Compute-capability majors with a committed mining Device.
_AMPERE_CC_MAJOR = 8
# GA100 (A100) is sm_80 exactly; sm_86/87/89 are Ampere but not the committed model.
_GA100_CC_MINOR = 0
_HOPPER_CC_MAJOR = 9
_BLACKWELL_CC_MAJOR = 10
# SM120's warp-level FP8 MMA reproduces the Blackwell atom arithmetic bit for bit.
_SM120_CC_MAJOR = 12


def device_for_capability(major: int, minor: int = 0, *, allow_fp16: bool = False) -> Device:
    """Map a CUDA compute capability to the committed mining ``Device``.

    - SM90 (Hopper: H100/H200) -> ``Device.HOPPER``
    - SM100 (Blackwell: B200/B300) -> ``Device.BLACKWELL``
    - SM120 (RTX PRO 6000, RTX 50) -> ``Device.BLACKWELL`` (same atom arithmetic)
    - SM80 (Ampere: A100/GA100) -> ``Device.A100`` **only** when ``allow_fp16``.

    ``allow_fp16`` is the FP16/v5 opt-in: the FP8 (v4) datapath never passes it,
    so sm_80 stays rejected for FP8 exactly as before. Every other capability
    raises ``ValueError`` (``minor`` only names the capability in that error).
    """
    if major == _HOPPER_CC_MAJOR:
        return Device.HOPPER
    if major in (_BLACKWELL_CC_MAJOR, _SM120_CC_MAJOR):
        return Device.BLACKWELL
    # The FP16 accumulation model is reverse-engineered and validated bit-exact
    # ONLY on GA100 (sm_80). Other Ampere/Ada parts (sm_86/87/89) are a different
    # tensor-core design and are NOT validated against the committed model -- they
    # may or may not match it -- so admit sm_80 exactly (fail-closed): a miner that
    # does not reproduce A100 tiles would only waste work on rejected submissions.
    if allow_fp16 and (major, minor) == (_AMPERE_CC_MAJOR, _GA100_CC_MINOR):
        return Device.A100
    raise ValueError(f"no committed mining Device for sm{major}{minor}")


def local_device(device: torch.device | int | None = None, *, allow_fp16: bool = False) -> Device:
    """The committed ``Device`` matching ``device``'s MMA (the current CUDA
    device when omitted). Fails closed on anything without a committed device;
    ``allow_fp16`` additionally admits sm_80 as ``Device.A100`` for the FP16
    scheme."""
    return device_for_capability(
        *torch.cuda.get_device_capability(device), allow_fp16=allow_fp16
    )


def is_fp16_capable(device: torch.device | int | None = None) -> bool:
    """Whether ``device`` is an sm_80 (A100/GA100) that runs the FP16 scheme.

    sm_80 exactly: the model is validated bit-exact only on GA100, so other
    Ampere/Ada parts (sm_86/87/89) -- not validated against the committed model --
    must not run FP16 jobs."""
    return torch.cuda.get_device_capability(device) == (_AMPERE_CC_MAJOR, _GA100_CC_MINOR)


__all__ = ["device_for_capability", "is_fp16_capable", "local_device"]
