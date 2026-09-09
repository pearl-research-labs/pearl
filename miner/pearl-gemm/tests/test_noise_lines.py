"""``noise_lines`` vs the reference ``OperandNoiser``: bit-exact keyed line draws.

Every ``side | factor`` address and several counts, gated with byte equality:
the kernel shares the fused prep kernels' line generator, so this also pins
the E_A/E_B device draw in isolation.
"""

import pytest
import torch
from blake3 import blake3
from miner_base.commitment import Device
from miner_base.commitment_hash import noise_line_key
from miner_base.hardware import hardware_for
from miner_base.noise import Factor, OperandNoiser, Side

from pearl_gemm import LABEL_E1, LABEL_E2, LABEL_F1, LABEL_F2, noise_lines
from pearl_gemm.protocol_constants import R

# label -> (side, factor) of the reference draw it addresses.
_LABELS = {
    "e1": (LABEL_E1, Side.A, Factor.E),
    "e2": (LABEL_E2, Side.B, Factor.E),
    "f1": (LABEL_F1, Side.A, Factor.F),
    "f2": (LABEL_F2, Side.B, Factor.F),
}


def _seeds():
    return blake3(b"noise-lines-seed-a").digest(), blake3(b"noise-lines-seed-b").digest()


def _device_lines(key: bytes, label: bytes, count: int) -> torch.Tensor:
    key_dev = torch.frombuffer(bytearray(key), dtype=torch.uint8).cuda()
    out = torch.zeros(count, R, dtype=torch.float8_e4m3fn, device="cuda")
    noise_lines(key_dev, label, out)
    torch.cuda.synchronize()
    return out.cpu()


def test_labels_are_the_reference_addresses():
    """The device address prefixes are the wire ``side(1) | factor(1)`` bytes."""
    for label, side, factor in _LABELS.values():
        assert label == bytes((side, factor))


@pytest.mark.parametrize("label_name", sorted(_LABELS))
@pytest.mark.parametrize("count", [16, 128, 1000])
def test_lines_match_reference_noiser(label_name, count):
    """Raw draws equal ``OperandNoiser._lines`` under the side's noise-line key
    byte-for-byte.

    1000 exercises the tail CTA's bounds guard (not a multiple of the block).
    """
    seed_a, seed_b = _seeds()
    label, side, factor = _LABELS[label_name]
    seed = seed_a if side is Side.A else seed_b
    noiser = OperandNoiser(seed, side, R, count, hardware_for(Device.BLACKWELL).compute)
    assert noiser._key == noise_line_key(seed)
    ref = noiser._lines(factor, range(count))
    got = _device_lines(noiser._key, label, count)
    assert torch.equal(got.view(torch.uint8), ref.view(torch.uint8))


def test_f_basis_assembly_matches_reference():
    """Transposed draws equal ``OperandNoiser.F`` per side (the memoized basis)."""
    seed_a, seed_b = _seeds()
    k = 1536
    noise_a = OperandNoiser(seed_a, Side.A, R, k, hardware_for(Device.BLACKWELL).compute)
    noise_b = OperandNoiser(seed_b, Side.B, R, k, hardware_for(Device.BLACKWELL).compute)
    f_a = _device_lines(noise_line_key(seed_a), LABEL_F1, k).t().contiguous()
    f_b = _device_lines(noise_line_key(seed_b), LABEL_F2, k).t().contiguous()
    assert torch.equal(f_a.view(torch.uint8), noise_a.F().view(torch.uint8))
    assert torch.equal(f_b.view(torch.uint8), noise_b.F().view(torch.uint8))
    # The sides are keyed apart: the same seed under B's address is a different draw.
    assert not torch.equal(
        _device_lines(noise_line_key(seed_a), LABEL_F2, k).view(torch.uint8),
        f_a.t().contiguous().view(torch.uint8),
    )


def test_e_draws_match_reference_noiser():
    """E_A/E_B row draws equal the public ``OperandNoiser.E`` under their keys."""
    seed_a, seed_b = _seeds()
    n = 96
    noise_a = OperandNoiser(seed_a, Side.A, R, 512, hardware_for(Device.BLACKWELL).compute)
    noise_b = OperandNoiser(seed_b, Side.B, R, 512, hardware_for(Device.BLACKWELL).compute)
    e_a = _device_lines(noise_line_key(seed_a), LABEL_E1, n)
    e_b = _device_lines(noise_line_key(seed_b), LABEL_E2, n)
    assert torch.equal(e_a.view(torch.uint8), noise_a.E(list(range(n))).view(torch.uint8))
    assert torch.equal(e_b.view(torch.uint8), noise_b.E(list(range(n))).view(torch.uint8))


def test_rejects_malformed_requests():
    key = torch.zeros(32, dtype=torch.uint8, device="cuda")
    out = torch.zeros(16, R, dtype=torch.float8_e4m3fn, device="cuda")
    with pytest.raises(ValueError, match="address"):
        noise_lines(key, b"short", out)
    with pytest.raises(TypeError, match="label"):
        noise_lines(key, "pearl-v2/noise-f1", out)
    with pytest.raises(ValueError, match="out"):
        noise_lines(key, LABEL_F1, out.view(torch.uint8))
    # An empty request is rejected by the API, not by a compile-time assert.
    with pytest.raises(ValueError, match="count"):
        noise_lines(key, LABEL_F1, torch.zeros(0, R, dtype=torch.float8_e4m3fn, device="cuda"))
