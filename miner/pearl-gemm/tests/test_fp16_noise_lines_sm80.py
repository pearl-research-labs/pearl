"""On-GPU bit-exactness of the A100 (``sm_80``) FP16 noise-line generation.

Runs the hand-written keyed-BLAKE3 + normalize kernel on the local GA100 and
asserts every ``u16`` of the ``E``/``F`` noise factors matches the Python port of
the verifier's ``sample_line`` / ``sample_noise``
(``tests/helpers/fp16_noise_lines_reference.py``, whose keyed-BLAKE3 XOF uses the
same ``blake3`` crate the reference hashes with). Gated to ``sm_80`` hardware.
"""

import numpy as np
import pytest
import torch

from pearl_gemm._utils._arch import Arch, arch_of

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or arch_of() is not Arch.SM80,
    reason="the A100 FP16 noise-line kernel targets sm_80 (GA100) hardware",
)

# (seed_a, seed_b, k, r, a_rows, b_cols). r = NOISE_RANK = 32 is the protocol
# rank; the extra ranks exercise XOF beyond one 32/64-byte output block.
_CASES = [
    (b"\x22" * 32, b"\x11" * 32, 128, 32, [0, 8, 64], [1, 2]),
    (b"\x01" * 32, b"\xfe" * 32, 64, 32, [0, 1, 2, 255], [7, 300, 1000]),
    (b"\xa5" * 32, b"\x5a" * 32, 256, 32, [0], [0]),
    (b"\x33" * 32, b"\x44" * 32, 40, 16, [3, 9], [4]),
    (b"\x7f" * 32, b"\x80" * 32, 48, 48, [0, 17], [1, 2, 65535]),
    (b"\x00" * 32, b"\xff" * 32, 24, 72, [5], [6]),
]


@pytest.mark.parametrize(("seed_a", "seed_b", "k", "r", "a_rows", "b_cols"), _CASES)
def test_noise_lines_are_bit_exact_vs_verifier(seed_a, seed_b, k, r, a_rows, b_cols):
    from pearl_gemm.fp16_noise_lines import sample_noise

    from tests.helpers.fp16_noise_lines_reference import sample_noise as ref_sample_noise

    ref_e_a, ref_f_a, ref_e_b, ref_f_b = ref_sample_noise(seed_a, seed_b, k, r, a_rows, b_cols)

    got = sample_noise(seed_a, seed_b, k, r, a_rows, b_cols)

    def u16(t):
        return t.cpu().numpy().view(np.uint16).reshape(-1)

    for name, g, ref in (
        ("e_a", got.e_a, ref_e_a),
        ("f_a", got.f_a, ref_f_a),
        ("e_b", got.e_b, ref_e_b),
        ("f_b", got.f_b, ref_f_b),
    ):
        g = u16(g)
        ref = ref.reshape(-1)
        mism = int((g != ref).sum())
        assert mism == 0, f"{name}: {mism}/{ref.size} u16 differ from the verifier model"


def test_line_key_matches_reference():
    from pearl_gemm.fp16_noise_lines import line_key

    from tests.helpers.fp16_noise_lines_reference import line_key as ref_line_key

    for seed in (b"\x22" * 32, b"\x01" * 32, bytes(range(32))):
        assert line_key(seed) == ref_line_key(seed)


def test_repeat_launch_determinism():
    from pearl_gemm.fp16_noise_lines import noise_lines

    seed = bytes(range(32))
    first = noise_lines(seed, 0, 0, list(range(64)), 32).cpu().numpy().copy()
    for _ in range(8):
        again = noise_lines(seed, 0, 0, list(range(64)), 32).cpu().numpy()
        assert np.array_equal(first, again), "noise-line kernel is not deterministic"
