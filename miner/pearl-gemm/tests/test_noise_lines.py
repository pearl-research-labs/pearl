"""``noise_lines``: bit-exact keyed line draws against pinned digests.

Every ``side | factor`` address and several counts, gated on the digest of the
protocol's draw (``helpers/digests.py``): the kernel shares the fused prep
kernels' line generator, so this also pins the E_A/E_B device draw in
isolation. Only ``E_A`` is keyed by seedA; the other three factors are keyed
by seedB.
"""

import pytest
import torch
from blake3 import blake3
from miner_base.commitment_hash import noise_line_key

from pearl_gemm import LABEL_E1, LABEL_E2, LABEL_F1, LABEL_F2, noise_lines
from pearl_gemm.protocol_constants import R
from tests.helpers.digests import digest, reference_digests

# label -> (device label, side, factor): the wire ``side(1) | factor(1)`` address.
_LABELS = {
    "e1": (LABEL_E1, 0, 0),
    "e2": (LABEL_E2, 1, 0),
    "f1": (LABEL_F1, 0, 1),
    "f2": (LABEL_F2, 1, 1),
}


def _seeds():
    return blake3(b"noise-lines-seed-a").digest(), blake3(b"noise-lines-seed-b").digest()


def _device_lines(key: bytes, label: bytes, count: int) -> torch.Tensor:
    key_dev = torch.frombuffer(bytearray(key), dtype=torch.uint8).cuda()
    out = torch.zeros(count, R, dtype=torch.float8_e4m3fn, device="cuda")
    noise_lines(key_dev, label, out)
    torch.cuda.synchronize()
    return out.cpu()


def _device_factors(seed_a: bytes, seed_b: bytes, rows: int, k: int) -> dict[str, torch.Tensor]:
    """The four factors as the kernel chain draws them: ``E_A`` under seedA's
    noise-line key, ``F_A``/``E_B``/``F_B`` under seedB's (each ``(lines x r)``)."""
    key_a, key_b = noise_line_key(seed_a), noise_line_key(seed_b)
    return {
        "E_A": _device_lines(key_a, LABEL_E1, rows),
        "F_A": _device_lines(key_b, LABEL_F1, k),
        "E_B": _device_lines(key_b, LABEL_E2, rows),
        "F_B": _device_lines(key_b, LABEL_F2, k),
    }


def _same_bits(lhs: torch.Tensor, rhs: torch.Tensor) -> bool:
    return torch.equal(lhs.view(torch.uint8), rhs.view(torch.uint8))


def test_labels_are_the_protocol_addresses():
    """The device address prefixes are the wire ``side(1) | factor(1)`` bytes."""
    for label, side, factor in _LABELS.values():
        assert label == bytes((side, factor))


@pytest.mark.parametrize("label_name", sorted(_LABELS))
@pytest.mark.parametrize("count", [16, 128, 1000])
def test_lines_match_pinned_digests(label_name, count):
    """Raw draws under the factor's noise-line key match the protocol's lines
    byte-for-byte (E lines row by row; F as the transposed ``k``-line basis).

    1000 exercises the tail CTA's bounds guard (not a multiple of the block).
    """
    seed_a, seed_b = _seeds()
    label = _LABELS[label_name][0]
    line_key = noise_line_key(seed_a if label_name == "e1" else seed_b)
    device_lines = _device_lines(line_key, label, count)
    expected = reference_digests()["noise_lines"]["lines"][str(count)][label_name]
    assert digest(device_lines) == expected


def test_only_e_a_moves_with_seed_a():
    """Draw all four factors for (A, B), then for (A', B): both match the
    pinned draws, F_A, E_B and F_B are unchanged, and only E_A moves."""
    seed_a, seed_b = _seeds()
    seed_a_prime = blake3(b"noise-lines-seed-a-prime").digest()
    rows, k = 96, 1536
    expected = reference_digests()["noise_lines"]["factors_96x1536"]
    device = _device_factors(seed_a, seed_b, rows, k)
    device_prime = _device_factors(seed_a_prime, seed_b, rows, k)
    assert digest(device["E_A"]) == expected["E_A"]
    assert digest(device_prime["E_A"]) == expected["E_A_prime"]
    for name in ("F_A", "E_B", "F_B"):
        assert digest(device[name]) == expected[name], name
        assert _same_bits(device[name], device_prime[name]), name
    assert not _same_bits(device["E_A"], device_prime["E_A"])
    # Same seedB, different Side addresses: F_A and F_B are distinct draws.
    assert not _same_bits(device["F_A"], device["F_B"])


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
