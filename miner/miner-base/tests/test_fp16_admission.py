"""FP16 (A100 / sm_80) committed-device admission.

``miner_base.devices`` resolves a CUDA compute capability to the committed
mining ``Device``. The FP16 scheme admits sm_80 as ``Device.A100`` -- but only
behind the explicit ``allow_fp16`` opt-in, so the FP8 (v4) datapath (which never
passes it) still rejects sm_80 exactly as before.

``devices`` imports ``torch`` at module top for ``local_device`` /
``is_fp16_capable``; only a minimal stub is needed to exercise the pure
capability->Device mapping, so a fake ``torch`` is injected for this module.
"""

from __future__ import annotations

import sys
import types

import pytest


@pytest.fixture()
def devices_module(monkeypatch):
    """Import ``miner_base.devices`` against a stub ``torch`` (capability only)."""
    fake_torch = types.ModuleType("torch")
    fake_torch.device = object  # only used in type hints
    fake_cuda = types.ModuleType("torch.cuda")
    fake_cuda._cap = (8, 0)

    def get_device_capability(device=None):
        return fake_cuda._cap

    fake_cuda.get_device_capability = get_device_capability
    fake_torch.cuda = fake_cuda
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torch.cuda", fake_cuda)
    monkeypatch.delitem(sys.modules, "miner_base.devices", raising=False)
    import miner_base.devices as devices

    yield devices
    sys.modules.pop("miner_base.devices", None)


def test_sm80_admitted_as_a100_only_for_fp16(devices_module):
    from miner_base.params import Device

    # FP8 datapath (no opt-in): sm_80 has no committed device -> rejected.
    with pytest.raises(ValueError):
        devices_module.device_for_capability(8, 0)
    # FP16 opt-in: sm_80 -> A100.
    assert devices_module.device_for_capability(8, 0, allow_fp16=True) is Device.A100


def test_non_ga100_ampere_rejected(devices_module):
    # sm_86/87/89 are Ampere but not the committed GA100 (sm_80) HMMA model, so
    # they must NOT be admitted as A100 even under the FP16 opt-in.
    for minor in (6, 7, 9):
        for allow in (False, True):
            with pytest.raises(ValueError):
                devices_module.device_for_capability(8, minor, allow_fp16=allow)


def test_fp8_families_unchanged_under_fp16_opt_in(devices_module):
    from miner_base.params import Device

    for major, expected in ((9, Device.HOPPER), (10, Device.BLACKWELL), (12, Device.BLACKWELL)):
        # The FP16 opt-in must not alter the FP8 families' mapping.
        assert devices_module.device_for_capability(major, 0) is expected
        assert devices_module.device_for_capability(major, 0, allow_fp16=True) is expected


def test_unknown_capability_rejected_both_ways(devices_module):
    for allow in (False, True):
        with pytest.raises(ValueError):
            devices_module.device_for_capability(7, 5, allow_fp16=allow)


def test_is_fp16_capable_detects_sm80(devices_module):
    import torch  # the stub

    torch.cuda._cap = (8, 0)
    assert devices_module.is_fp16_capable() is True
    torch.cuda._cap = (9, 0)
    assert devices_module.is_fp16_capable() is False
    # Non-GA100 Ampere (sm_86/87/89) is not FP16-capable.
    for minor in (6, 7, 9):
        torch.cuda._cap = (8, minor)
        assert devices_module.is_fp16_capable() is False


def test_device_enum_has_a100():
    from miner_base.params import Device

    assert int(Device.A100) == 2
    assert Device.HOPPER != Device.A100 and Device.BLACKWELL != Device.A100
