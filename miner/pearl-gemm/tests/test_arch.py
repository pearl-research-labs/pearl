"""The architecture gate: capability -> family mapping and the launch check."""

import pytest
import torch

from pearl_gemm._utils._arch import Arch, arch_of, require_arch
from pearl_gemm.protocol_constants import SM100_CC_MAJOR, SM120_CC_MAJOR


@pytest.mark.parametrize(
    ("capability", "arch"),
    [((10, 0), Arch.SM100), ((10, 3), Arch.SM100), ((12, 0), Arch.SM120), ((12, 1), Arch.SM120)],
)
def test_known_majors_map_to_their_family(capability, arch):
    assert arch_of(capability) is arch
    assert arch.value in (SM100_CC_MAJOR, SM120_CC_MAJOR)


def test_smem_capacity_is_the_lead_target_opt_in_maximum():
    assert Arch.SM100.smem_capacity_bytes == 232448
    assert Arch.SM120.smem_capacity_bytes == 101376


@pytest.mark.parametrize("capability", [(8, 9), (9, 0), (11, 0)])
def test_unknown_majors_fail_closed(capability):
    with pytest.raises(ValueError, match=f"sm{capability[0]}{capability[1]}"):
        arch_of(capability)


def test_require_arch_accepts_a_listed_family(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (12, 0))
    assert require_arch("op", None, Arch.SM100, Arch.SM120) is Arch.SM120
    assert require_arch("op", None, Arch.SM120) is Arch.SM120


@pytest.mark.parametrize(
    ("capability", "supported", "message"),
    [
        ((12, 0), (Arch.SM100,), "op requires SM100, got sm120"),
        ((10, 0), (Arch.SM120,), "op requires SM120, got sm100"),
        ((9, 0), (Arch.SM100, Arch.SM120), "op requires one of SM100, SM120, got sm90"),
    ],
)
def test_require_arch_rejects_other_devices(monkeypatch, capability, supported, message):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: capability)
    with pytest.raises(ValueError, match=message):
        require_arch("op", None, *supported)


def test_local_device_is_a_known_family():
    """The suite's own GPU resolves to a supported family."""
    assert arch_of() in Arch
