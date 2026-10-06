"""On-GPU bit-exactness of the A100 (``sm_80``) FP16 accumulation policy census.

Runs the hand-written software-accumulation + fused policy-reduction kernel on
the local GA100 and asserts, against the Python port of the verifier's
``policy.rs`` (``tests/helpers/fp16_policy_reference.py``, built on the trusted
A100 accumulation oracle): every recomputed tile u32 bit matches, every per-cell
integer census quantity (``n_bp``, ``n_runs``, ``n_pt``) matches, and the folded
``f_bp`` / ``rho`` bit patterns and ``accept`` agree. Gated to ``sm_80``.
"""

import numpy as np
import pytest
import torch

from pearl_gemm._utils._arch import Arch, arch_of

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or arch_of() is not Arch.SM80,
    reason="the A100 FP16 policy census kernel targets sm_80 (GA100) hardware",
)

# (m, n, k): k includes non-multiples of 8 and k < 8; shapes mirror the Rust
# oracle dump. No tile-granularity constraint (software accumulation path).
_SHAPES = [
    (1, 1, 7),
    (2, 3, 8),
    (3, 2, 9),
    (4, 4, 16),
    (2, 2, 31),
    (5, 3, 33),
    (4, 6, 64),
    (3, 3, 100),
    (6, 5, 127),
    (2, 4, 256),
    (8, 8, 40),
    (7, 7, 48),
    (1, 16, 72),
    (16, 1, 72),
]


def _rand_fp16_bits(shape, rng, exp_lo, exp_hi):
    """Random finite FP16 bit patterns with a bounded exponent window (wide =>
    breakpoints/truncations, narrow => near-flat), ~12% exact zeros."""
    size = int(np.prod(shape))
    sign = (rng.integers(0, 2, size).astype(np.uint16)) << 15
    exp = rng.integers(exp_lo, exp_hi + 1, size).astype(np.uint16) << 10
    man = rng.integers(0, 0x400, size).astype(np.uint16)
    bits = (sign | exp | man).astype(np.uint16)
    zero = rng.random(size) < 0.12
    bits[zero] = (sign[zero])  # +/-0
    return bits.reshape(shape)


@pytest.mark.parametrize(("m", "n", "k"), _SHAPES)
@pytest.mark.parametrize(("exp_lo", "exp_hi"), [(1, 28), (10, 12), (1, 29)])
def test_policy_census_is_bit_exact_vs_verifier(m, n, k, exp_lo, exp_hi):
    from pearl_gemm.fp16_policy import evaluate, policy_census

    from tests.helpers.fp16_policy_reference import replay_and_evaluate as ref_replay

    rng = np.random.default_rng(99 + m * 131 + n * 17 + k * 7 + exp_lo * 3 + exp_hi)
    a_bits = _rand_fp16_bits((m, k), rng, exp_lo, exp_hi)
    b_bits = _rand_fp16_bits((n, k), rng, exp_lo, exp_hi)

    ref_tile, ref_percell, ref_report = ref_replay(a_bits, b_bits, m, n, k)

    a = torch.from_numpy(a_bits.view(np.float16)).cuda()
    b = torch.from_numpy(b_bits.view(np.float16)).cuda()
    tile, n_bp, n_runs, n_pt = policy_census(a, b)

    tile_bits = tile.cpu().numpy().view(np.uint32)
    assert int((tile_bits != ref_tile).sum()) == 0, "recomputed tile bits differ"

    got_percell = np.stack(
        [
            n_bp.cpu().numpy().reshape(-1),
            n_runs.cpu().numpy().reshape(-1),
            n_pt.cpu().numpy().reshape(-1),
        ],
        axis=1,
    ).astype(np.int64)
    assert int((got_percell != ref_percell).sum()) == 0, "per-cell census differs"

    report = evaluate(n_bp, n_runs, n_pt, k)
    assert report.breakpoints == ref_report.breakpoints
    assert report.numerator == ref_report.numerator
    # f_bp / rho are f64: compare exact bit patterns.
    assert np.float64(report.f_bp).view(np.uint64) == np.float64(ref_report.f_bp).view(np.uint64)
    assert np.float64(report.rho).view(np.uint64) == np.float64(ref_report.rho).view(np.uint64)
    assert report.accept == ref_report.accept


def test_tile_matches_fp16_gemm():
    """The census kernel's recomputed tile matches the committed fp16_gemm
    datapath exactly (k a multiple of 16 so the HMMA kernel applies)."""
    from pearl_gemm.fp16_gemm import fp16_gemm_a100
    from pearl_gemm.fp16_policy import policy_census

    rng = np.random.default_rng(5)
    a_bits = _rand_fp16_bits((16, 64), rng, 1, 28)
    b_bits = _rand_fp16_bits((8, 64), rng, 1, 28)
    a = torch.from_numpy(a_bits.view(np.float16)).cuda()
    b = torch.from_numpy(b_bits.view(np.float16)).cuda()

    gemm_tile = fp16_gemm_a100(a, b).cpu().numpy().view(np.uint32)
    policy_tile = policy_census(a, b)[0].cpu().numpy().view(np.uint32)
    assert int((gemm_tile != policy_tile).sum()) == 0, "policy tile != fp16_gemm tile"


def test_repeat_launch_determinism():
    from pearl_gemm.fp16_policy import policy_census

    rng = np.random.default_rng(11)
    a = torch.from_numpy(_rand_fp16_bits((8, 48), rng, 1, 28).view(np.float16)).cuda()
    b = torch.from_numpy(_rand_fp16_bits((8, 48), rng, 1, 28).view(np.float16)).cuda()
    first = [t.cpu().numpy().copy() for t in policy_census(a, b)]
    for _ in range(8):
        again = [t.cpu().numpy() for t in policy_census(a, b)]
        for x, y in zip(first, again):
            assert np.array_equal(x, y), "kernel is not deterministic across launches"


def test_shape_mismatch_rejected():
    from pearl_gemm.fp16_policy import policy_census

    a = torch.zeros((4, 16), dtype=torch.float16, device="cuda")
    b = torch.zeros((4, 8), dtype=torch.float16, device="cuda")  # k must match
    with pytest.raises(ValueError, match="b must have k="):
        policy_census(a, b)
