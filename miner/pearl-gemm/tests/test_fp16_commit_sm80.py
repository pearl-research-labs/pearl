"""On-GPU bit-exactness of the A100 (``sm_80``) FP16 operand commitment.

Runs the hand-written keyed-BLAKE3 Merkle kernel on the local GA100 and asserts
the 32-byte root matches, bit-for-bit, the pure-Python port of the verifier's
``commit_operand`` / ``pearl_blake3::MerkleTree::with_chunk_len``
(``tests/helpers/fp16_commit_reference.py``), itself cross-checked against the
Rust oracle dump (``api::fp16::commitment::dump_commit_oracle``). Gated to
``sm_80`` hardware.
"""

import numpy as np
import pytest
import torch

from pearl_gemm._utils._arch import Arch, arch_of

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or arch_of() is not Arch.SM80,
    reason="the A100 FP16 operand-commitment kernel targets sm_80 (GA100) hardware",
)


def _gen_rows(n, seed):
    # Arbitrary raw u16 bit patterns; commit_operand does not validate finiteness.
    return np.array([(i * 40503 + seed) & 0xFFFF for i in range(n)], dtype=np.uint16)


# (num_rows, k, chunk_len, key_byte, seed): single-leaf, multi-leaf, every
# allowed chunk_len, odd leaf counts, and row-spanning leaves.
_CASES = [
    (1, 1, 1024, 0x11, 0),
    (2, 64, 1024, 0x22, 7),
    (8, 256, 1024, 0x07, 123),
    (8, 256, 128, 0x5A, 55),
    (8, 256, 256, 0xA5, 99),
    (8, 256, 512, 0x33, 12),
    (5, 100, 512, 0x44, 321),
    (7, 333, 128, 0x99, 1000),
    (13, 777, 1024, 0xFE, 2024),
    (3, 1500, 256, 0x80, 65535),
    (1, 600, 1024, 0x01, 44),  # single leaf, multi-block chunk
    (16, 512, 256, 0xC3, 7777),
]


@pytest.mark.parametrize(("num_rows", "k", "chunk_len", "key_byte", "seed"), _CASES)
def test_commit_root_is_bit_exact_vs_verifier(num_rows, k, chunk_len, key_byte, seed):
    from pearl_gemm.fp16_commit import commit_operand

    from tests.helpers.fp16_commit_reference import commit_operand_root as ref_root

    rows = _gen_rows(num_rows * k, seed)
    key = bytes([key_byte]) * 32

    expected = ref_root(rows, num_rows, k, key, chunk_len)
    got = commit_operand(torch.from_numpy(rows.view(np.int16)), num_rows, k, key, chunk_len)

    assert got == expected, (
        f"root mismatch (nr={num_rows}, k={k}, chunk_len={chunk_len}): "
        f"{got.hex()} != {expected.hex()}"
    )


def test_commit_root_determinism():
    from pearl_gemm.fp16_commit import commit_operand

    rows = _gen_rows(8 * 256, 1)
    key = bytes(range(32))
    first = commit_operand(torch.from_numpy(rows.view(np.int16)), 8, 256, key, 1024)
    for _ in range(8):
        again = commit_operand(torch.from_numpy(rows.view(np.int16)), 8, 256, key, 1024)
        assert again == first, "commitment kernel is not deterministic"
