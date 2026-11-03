"""On-GPU bit-exactness of the A100 (``sm_80``) FP16 GEMM tile.

Runs the hand-written ``mma.sync.m16n8k16.f32.f16`` kernel on the local GA100 and
asserts every FP32 output bit matches the Python port of the verifier's A100
accumulation model (``tests/helpers/a100_fp16_reference.py``, itself validated
bit-exact against ``zk-pow/.../a100_dot_vectors.txt``). Also checks repeat-launch
determinism. Gated to ``sm_80`` hardware; a dead end on any other family.
"""

import numpy as np
import pytest
import torch

from pearl_gemm._utils._arch import Arch, arch_of

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or arch_of() is not Arch.SM80,
    reason="the A100 FP16 GEMM kernel targets sm_80 (GA100) hardware",
)

# Shapes: (m, n, k) with m%16==0, n%8==0, k%16==0, spanning one-tile and
# many-tile grids and a range of k depths (one group, several groups, long).
_SHAPES = [
    (16, 8, 16),
    (16, 8, 256),
    (32, 16, 32),
    (48, 24, 64),
    (64, 40, 128),
    (128, 64, 48),
]


def _rand_fp16(shape, rng, scale):
    """Random finite FP16 operands at a given magnitude scale, with some zeros."""
    v = (rng.standard_normal(shape) * scale).astype(np.float16)
    mask = rng.random(shape) < 0.1
    v[mask] = np.float16(0.0)
    return v


def _reference_bits(a_np, b_np, m, n, k, acc_bits=None):
    from tests.helpers.a100_fp16_reference import a100_matmul_bits

    return a100_matmul_bits(a_np.view(np.uint16), b_np.view(np.uint16), m, n, k, acc_bits)


@pytest.mark.parametrize(("m", "n", "k"), _SHAPES)
@pytest.mark.parametrize("scale", [0.25, 2.0, 32.0])
def test_tile_is_bit_exact_vs_verifier(m, n, k, scale):
    from pearl_gemm.fp16_gemm import fp16_gemm_a100

    rng = np.random.default_rng(1234 + m * 131 + n * 17 + k + int(scale * 7))
    a_np = _rand_fp16((m, k), rng, scale)
    b_np = _rand_fp16((n, k), rng, scale)

    ref = _reference_bits(a_np, b_np, m, n, k)
    d = fp16_gemm_a100(torch.from_numpy(a_np).cuda(), torch.from_numpy(b_np).cuda())
    got = d.cpu().numpy().view(np.uint32)

    mism = int((got != ref).sum())
    assert mism == 0, f"{mism}/{got.size} elements differ from the A100 verifier model"


@pytest.mark.parametrize(("m", "n", "k"), [(32, 16, 64), (48, 24, 32)])
def test_accumulator_carry_in_is_bit_exact(m, n, k):
    from pearl_gemm.fp16_gemm import fp16_gemm_a100

    rng = np.random.default_rng(99 + m + n + k)
    a_np = _rand_fp16((m, k), rng, 2.0)
    b_np = _rand_fp16((n, k), rng, 2.0)
    acc = (rng.standard_normal((m, n)) * 4.0).astype(np.float32)

    ref = _reference_bits(a_np, b_np, m, n, k, acc.view(np.uint32))
    d = fp16_gemm_a100(
        torch.from_numpy(a_np).cuda(),
        torch.from_numpy(b_np).cuda(),
        acc=torch.from_numpy(acc).cuda(),
    )
    got = d.cpu().numpy().view(np.uint32)
    assert int((got != ref).sum()) == 0


def test_repeat_launch_determinism():
    from pearl_gemm.fp16_gemm import fp16_gemm_a100

    rng = np.random.default_rng(7)
    a = torch.from_numpy(_rand_fp16((64, 128), rng, 2.0)).cuda()
    b = torch.from_numpy(_rand_fp16((40, 128), rng, 2.0)).cuda()
    first = fp16_gemm_a100(a, b).cpu().numpy().view(np.uint32)
    for _ in range(8):
        again = fp16_gemm_a100(a, b).cpu().numpy().view(np.uint32)
        assert np.array_equal(first, again), "kernel is not deterministic across launches"


def test_shape_constraints_rejected():
    from pearl_gemm.fp16_gemm import fp16_gemm_a100

    a = torch.zeros((16, 24), dtype=torch.float16, device="cuda")  # k=24 not %16
    b = torch.zeros((8, 24), dtype=torch.float16, device="cuda")
    with pytest.raises(RuntimeError, match="multiple of 16"):
        fp16_gemm_a100(a, b)
