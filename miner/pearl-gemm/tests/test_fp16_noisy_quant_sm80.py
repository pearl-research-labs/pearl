"""On-GPU bit-exactness of the A100 (``sm_80``) FP16 fused noisy quantize.

Runs the hand-written norms+scales and fused-elementwise kernels on the local
GA100 and asserts every output bit (per-row ``alpha``/``beta``/``l2`` BF16 scales
and the final FP16 codes) matches the Python port of the verifier's
``noisy_quantize`` (``tests/helpers/fp16_noisy_quant_reference.py``, whose
``E@F^T`` noise reuses the bit-exact A100 datapath oracle). Gated to ``sm_80``.
"""

import numpy as np
import pytest
import torch

from pearl_gemm._utils._arch import Arch, arch_of

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or arch_of() is not Arch.SM80,
    reason="the A100 FP16 noisy-quantize kernels target sm_80 (GA100) hardware",
)

# (num_rows, k, r): num_rows%16==0, k%8==0, r%16==0 (fp16_gemm_a100 tile grid).
_SHAPES = [
    (16, 8, 16),
    (16, 16, 16),
    (32, 8, 32),
    (16, 64, 16),
    (32, 24, 48),
    (48, 40, 16),
]


def _rand_fp16(shape, rng, scale):
    v = (rng.standard_normal(shape) * scale).astype(np.float16)
    v[rng.random(shape) < 0.1] = np.float16(0.0)
    return v


@pytest.mark.parametrize(("num_rows", "k", "r"), _SHAPES)
@pytest.mark.parametrize("scale", [0.25, 2.0, 16.0])
def test_noisy_quantize_is_bit_exact_vs_verifier(num_rows, k, r, scale):
    from pearl_gemm.fp16_noisy_quant import noisy_quantize

    from tests.helpers.fp16_noisy_quant_reference import noisy_quantize as ref_nq
    from tests.helpers.fp16_noisy_quant_reference import row_norms as ref_row_norms

    rng = np.random.default_rng(2024 + num_rows * 131 + k * 17 + r + int(scale * 7))
    rows_np = _rand_fp16((num_rows, k), rng, scale)
    e_np = _rand_fp16((num_rows, r), rng, scale)
    f_np = _rand_fp16((k, r), rng, scale)

    norms = [ref_row_norms(rows_np[i].view(np.uint16)) for i in range(num_rows)]
    ref = ref_nq(rows_np, e_np, f_np, norms, r)

    built = noisy_quantize(
        torch.from_numpy(rows_np).cuda(),
        torch.from_numpy(e_np).cuda(),
        torch.from_numpy(f_np).cuda(),
        r,
    )
    a_got = built.alpha.cpu().numpy().view(np.uint16)
    b_got = built.beta.cpu().numpy().view(np.uint16)
    l2_got = built.l2.cpu().numpy().view(np.uint16)
    n_got = built.noised_part.cpu().numpy().view(np.uint16)

    assert int((a_got != ref["alpha"]).sum()) == 0, "alpha scales differ"
    assert int((b_got != ref["beta"]).sum()) == 0, "beta scales differ"
    assert int((l2_got != ref["l2"]).sum()) == 0, "l2 differs"
    mism = int((n_got != ref["noised"]).sum())
    assert mism == 0, f"{mism}/{n_got.size} FP16 codes differ from the verifier"


def test_repeat_launch_determinism():
    from pearl_gemm.fp16_noisy_quant import noisy_quantize

    rng = np.random.default_rng(7)
    rows = torch.from_numpy(_rand_fp16((32, 64), rng, 2.0)).cuda()
    e = torch.from_numpy(_rand_fp16((32, 32), rng, 2.0)).cuda()
    f = torch.from_numpy(_rand_fp16((64, 32), rng, 2.0)).cuda()
    first = noisy_quantize(rows, e, f, 32).noised_part.cpu().numpy().view(np.uint16)
    for _ in range(8):
        again = noisy_quantize(rows, e, f, 32).noised_part.cpu().numpy().view(np.uint16)
        assert np.array_equal(first, again), "kernel is not deterministic across launches"


def test_shape_mismatch_rejected():
    from pearl_gemm.fp16_noisy_quant import noisy_quantize

    rows = torch.zeros((16, 16), dtype=torch.float16, device="cuda")
    e = torch.zeros((16, 16), dtype=torch.float16, device="cuda")
    f = torch.zeros((8, 16), dtype=torch.float16, device="cuda")  # k must be 16
    with pytest.raises(ValueError, match="f must be"):
        noisy_quantize(rows, e, f, 16)
