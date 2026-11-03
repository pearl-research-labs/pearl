"""On-GPU bit-exactness of the A100 (``sm_80``) end-to-end FP16 tile pipeline.

Chains the GA100 FP16 kernels on the local hardware exactly as
``zk-pow/src/api/fp16/verify.rs::verify_tile`` does -- rebuild the noised
operands, replay + score the tile, fold the jackpot ticket -- and asserts every
stage boundary is bit-exact to the independent Python port of the verifier
(``tests/helpers/fp16_pipeline_reference.py``): the rebuilt FP16 operands, the
``(h, w)`` f32 tile bits, the ``f_bp``/``rho``/``accept`` report, the 64-byte
XOR-fold message, the 32-byte jackpot ticket, and the difficulty verdict. Gated
to ``sm_80``. The reference reproduces Rust's two-step rounding (``mul_add`` ->
f32, then RNE -> f16), which the GA100 kernels (``fmaf`` + ``__float2half_rn``)
match bit-for-bit.
"""

import numpy as np
import pytest
import torch

from pearl_gemm._utils._arch import Arch, arch_of

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or arch_of() is not Arch.SM80,
    reason="the A100 FP16 pipeline kernels target sm_80 (GA100) hardware",
)

# DimType discriminants (crate::api::layout::DimType).
_BLAKE, _FOLD = 2, 1


class _Gen:
    """Deterministic xorshift FP16 stream with spread magnitudes, so the tile has
    dense breakpoints (the honest/realistic regime that clears the policy gate)."""

    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFFFFFFFFFF

    def _n(self):
        self.s ^= (self.s << 13) & 0xFFFFFFFFFFFFFFFF
        self.s ^= self.s >> 7
        self.s ^= (self.s << 17) & 0xFFFFFFFFFFFFFFFF
        return self.s

    def operand(self, n):
        out = np.empty(n, np.float16)
        for i in range(n):
            r = self._n()
            sign = 1.0 if (r & 1) == 0 else -1.0
            exp = int((r >> 1) % 9) - 3
            mant = 1.0 + ((r >> 8) % 1024) / 1024.0
            out[i] = np.float16(sign * mant * (2.0 ** exp))
        return out

    def noise(self, n, r_rank):
        scale = 256.0 / (r_rank ** 0.5)
        out = np.empty(n, np.float16)
        for i in range(n):
            out[i] = np.float16(scale if (self._n() & 1) == 0 else -scale)
        return out


def _cuda_f16(arr):
    return torch.from_numpy(np.ascontiguousarray(arr)).cuda()


def _assert_matches_reference(a, b, ea, fa, eb, fb, rp_dims, cp_dims, seed, r):
    from pearl_gemm.fp16_pipeline import AxisPattern, pipeline

    from tests.helpers.fp16_pipeline_reference import AxisPattern as RefAxis
    from tests.helpers.fp16_pipeline_reference import verify_tile as ref_verify_tile

    h, k = a.shape
    w = b.shape[0]
    ref = ref_verify_tile(a, b, ea, fa, eb, fb, RefAxis.new(rp_dims), RefAxis.new(cp_dims), seed, r)
    v = pipeline(
        _cuda_f16(a), _cuda_f16(b), _cuda_f16(ea), _cuda_f16(fa), _cuda_f16(eb), _cuda_f16(fb),
        AxisPattern.new(rp_dims), AxisPattern.new(cp_dims), seed, r,
    )

    ba = v.built_a.noised_part.cpu().numpy().view(np.uint16)
    bb = v.built_b.noised_part.cpu().numpy().view(np.uint16)
    assert int((ba != ref.built_a).sum()) == 0, "rebuilt A operand differs"
    assert int((bb != ref.built_b).sum()) == 0, "rebuilt B operand differs"
    tile_bits = v.tile.view(torch.int32).reshape(-1).cpu().numpy().view(np.uint32)
    assert int((tile_bits != ref.tile_bits.reshape(-1)).sum()) == 0, "tile f32 bits differ"
    assert np.float64(v.report.f_bp).view(np.uint64) == np.float64(ref.report.f_bp).view(np.uint64)
    assert np.float64(v.report.rho).view(np.uint64) == np.float64(ref.report.rho).view(np.uint64)
    assert v.report.accept == ref.report.accept
    assert v.message == ref.message, "XOR-fold message differs"
    assert v.ticket == ref.ticket, "jackpot ticket differs"
    return v, ref


# (h, w, k, r, rp_dims, cp_dims): h/w == pattern tile_size, 16 blake lanes.
_SHAPES = [
    (4, 64, 128, 32, [(4, _BLAKE)], [(4, _BLAKE), (16, _FOLD)]),
    (4, 64, 256, 32, [(4, _BLAKE)], [(4, _BLAKE), (16, _FOLD)]),
    (16, 64, 128, 32, [(4, _BLAKE), (4, _FOLD)], [(4, _BLAKE), (16, _FOLD)]),
]


@pytest.mark.parametrize(("h", "w", "k", "r", "rp", "cp"), _SHAPES)
def test_pipeline_is_bit_exact_vs_verifier(h, w, k, r, rp, cp):
    g = _Gen(0x1234_5678_9ABC_DEF1 ^ (h * 131 + w * 17 + k + r))
    a = g.operand(h * k).reshape(h, k)
    b = g.operand(w * k).reshape(w, k)
    ea = g.noise(h * r, r).reshape(h, r)
    fa = g.noise(k * r, r).reshape(k, r)
    eb = g.noise(w * r, r).reshape(w, r)
    fb = g.noise(k * r, r).reshape(k, r)
    v, ref = _assert_matches_reference(a, b, ea, fa, eb, fb, rp, cp, b"\x07" * 32, r)
    # Spread-magnitude operands clear the policy gate.
    assert v.report.accept and v.report.f_bp >= 0.30 and v.report.rho >= 1.2


def test_verify_tile_accepts_and_proof_checks_difficulty():
    from pearl_gemm.fp16_pipeline import AxisPattern, verify_tile, verify_tile_proof

    h, w, k, r = 4, 64, 256, 32
    rp, cp = [(4, _BLAKE)], [(4, _BLAKE), (16, _FOLD)]
    g = _Gen(0xFEED_FACE_CAFE_B0BA)
    a = _cuda_f16(g.operand(h * k).reshape(h, k))
    b = _cuda_f16(g.operand(w * k).reshape(w, k))
    ea = _cuda_f16(g.noise(h * r, r).reshape(h, r))
    fa = _cuda_f16(g.noise(k * r, r).reshape(k, r))
    eb = _cuda_f16(g.noise(w * r, r).reshape(w, r))
    fb = _cuda_f16(g.noise(k * r, r).reshape(k, r))
    seed = b"\x5a" * 32
    rpa, cpa = AxisPattern.new(rp), AxisPattern.new(cp)

    v = verify_tile(a, b, ea, fa, eb, fb, rpa, cpa, seed, r)
    assert v.report.accept
    # Easy target accepts; an impossible (zero) target raises.
    verify_tile_proof(a, b, ea, fa, eb, fb, rpa, cpa, seed, r, 0x207F_FFFF)
    with pytest.raises(ValueError, match="difficulty"):
        verify_tile_proof(a, b, ea, fa, eb, fb, rpa, cpa, seed, r, 0)


def test_verify_tile_rejects_flat_tile():
    from pearl_gemm.fp16_pipeline import AxisPattern, verify_tile

    h, w, k, r = 4, 64, 256, 32
    one = np.ones((h, k), np.float16)
    a = _cuda_f16(one)
    b = _cuda_f16(np.ones((w, k), np.float16))
    ea = _cuda_f16(np.zeros((h, r), np.float16))
    fa = _cuda_f16(np.zeros((k, r), np.float16))
    eb = _cuda_f16(np.zeros((w, r), np.float16))
    fb = _cuda_f16(np.zeros((k, r), np.float16))
    rp, cp = AxisPattern.new([(4, _BLAKE)]), AxisPattern.new([(4, _BLAKE), (16, _FOLD)])
    with pytest.raises(ValueError, match="not admissible"):
        verify_tile(a, b, ea, fa, eb, fb, rp, cp, b"\x00" * 32, r)


def test_pipeline_with_noise_sampled_from_seeds():
    """Close the loop with the deterministic noise module: draw E/F from seeds via
    ``fp16_noise_lines`` and feed the driver, asserting it still matches the
    reference built from the same sampled factors."""
    from pearl_gemm.fp16_noise_lines import sample_noise

    from tests.helpers.fp16_noise_lines_reference import sample_noise as ref_sample

    h, w, k, r = 4, 64, 128, 32
    rp, cp = [(4, _BLAKE)], [(4, _BLAKE), (16, _FOLD)]
    g = _Gen(0x0BAD_F00D_1337_D00D)
    a = g.operand(h * k).reshape(h, k)
    b = g.operand(w * k).reshape(w, k)
    seed_a, seed_b = b"\x11" * 32, b"\x22" * 32
    a_rows, b_cols = list(range(h)), list(range(w))

    # Device-sampled noise (int16 FP16 bits) and the reference's.
    nz = sample_noise(seed_a, seed_b, k, r, a_rows, b_cols)
    ea = nz.e_a.cpu().numpy().view(np.uint16).view(np.float16)
    fa = nz.f_a.cpu().numpy().view(np.uint16).view(np.float16)
    eb = nz.e_b.cpu().numpy().view(np.uint16).view(np.float16)
    fb = nz.f_b.cpu().numpy().view(np.uint16).view(np.float16)
    ref_ea, ref_fa, ref_eb, ref_fb = ref_sample(seed_a, seed_b, k, r, a_rows, b_cols)
    assert np.array_equal(ea.view(np.uint16), ref_ea), "sampled E_a differs from reference"
    assert np.array_equal(fb.view(np.uint16), ref_fb), "sampled F_b differs from reference"

    _assert_matches_reference(a, b, ea, fa, eb, fb, rp, cp, seed_a, r)
