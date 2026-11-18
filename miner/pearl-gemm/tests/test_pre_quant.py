"""``pre_quant`` vs the reference: the two block-scaled blobs.

Bit-exact gate: the ``(m, k)`` int8 codes and ``(m, k/8)`` BF16 scales equal
``PrequantMatrix.int_values`` / ``.scales`` (the vendored ``int8 blk8 bf16s``
input dtype). Shape coverage follows q8 ``test_quantization`` spirit: a grid
of legal (m, k) sizes, edge fill patterns, config invariance, and determinism.
The ``(sumsq, absmax)`` commit-stats partials are produced by
``tensor_hash_plus_stats`` and gated there; ``alpha``/``beta`` bit-exactness
over random data is covered end-to-end in ``test_noisy_quant``.
"""

import pytest
import torch
from miner_base.prequant import PrequantMatrix

from pearl_gemm import PreQuantConfig, pre_quant, pre_quant_output_shapes

# k % 512 == 0 (stats/commit chunks). Include production-ish widths and
# non-power-of-2 multiples of 512 (q8 quantization spirit).
_SHAPES = [
    (1, 512),
    (4, 512),
    (8, 1024),
    (32, 512),
    (32, 4096),
    (64, 1024),
    (64, 7168),
    (96, 1024),
    (128, 4096),
    (256, 2048),
    (512, 1536),
    (512, 8192),
    (1024, 4096),
    (2048, 5120),  # non-power-of-2 width
]


def _blob_buffers(m, k):
    codes_shape, scales_shape = pre_quant_output_shapes(m, k)
    return (
        torch.zeros(codes_shape, dtype=torch.int8, device="cuda"),
        torch.zeros(scales_shape, dtype=torch.bfloat16, device="cuda"),
    )


def _assert_blobs_match(codes, scales, a_cpu):
    ref = PrequantMatrix.encode(a_cpu)
    assert torch.equal(codes.cpu(), ref.int_values)
    assert torch.equal(scales.cpu().view(torch.uint8), ref.scales.view(torch.uint8))


def _shape_id(shape):
    return "x".join(map(str, shape))


@pytest.mark.parametrize("m,k", _SHAPES, ids=[_shape_id(s) for s in _SHAPES])
def test_blobs_bit_exact(m, k):
    torch.manual_seed(m * k)
    a = (torch.randn(m, k, dtype=torch.bfloat16, device="cuda") * 1.5).contiguous()
    a[:, :16] = 0  # exercise the all-zero-block scale floor path
    codes, scales = _blob_buffers(m, k)
    pre_quant(a, codes, scales)
    torch.cuda.synchronize()
    _assert_blobs_match(codes, scales, a.cpu())


@pytest.mark.parametrize(
    "fill",
    ["zeros", "ones", "max", "random"],
)
def test_edge_fill_patterns(fill):
    """Edge tensors mirroring q8 / inner-hash edge coverage."""
    m, k = 64, 1024
    if fill == "zeros":
        a = torch.zeros(m, k, dtype=torch.bfloat16, device="cuda")
    elif fill == "ones":
        a = torch.ones(m, k, dtype=torch.bfloat16, device="cuda")
    elif fill == "max":
        a = torch.full((m, k), 65504.0, dtype=torch.bfloat16, device="cuda")
    else:
        torch.manual_seed(42)
        a = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    codes, scales = _blob_buffers(m, k)
    pre_quant(a, codes, scales)
    torch.cuda.synchronize()
    _assert_blobs_match(codes, scales, a.cpu())


@pytest.mark.parametrize("threads_per_block", [128, 256, 512])
def test_configs_preserve_blobs(threads_per_block):
    m, k = 96, 1024
    torch.manual_seed(threads_per_block)
    a = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    codes, scales = _blob_buffers(m, k)
    pre_quant(a, codes, scales, config=PreQuantConfig(threads_per_block=threads_per_block))
    torch.cuda.synchronize()
    _assert_blobs_match(codes, scales, a.cpu())


def test_consistency():
    m, k = 128, 2048
    torch.manual_seed(7)
    a = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    codes1, scales1 = _blob_buffers(m, k)
    codes2, scales2 = _blob_buffers(m, k)
    pre_quant(a, codes1, scales1)
    pre_quant(a, codes2, scales2)
    torch.cuda.synchronize()
    assert torch.equal(codes1, codes2)
    assert torch.equal(scales1.view(torch.uint8), scales2.view(torch.uint8))


def test_opened_rows_are_a_quantization_fixpoint():
    """Re-quantizing the opened rows reproduces codes and scales bit-exactly."""
    torch.manual_seed(7)
    a = torch.randn(64, 1024, dtype=torch.bfloat16, device="cuda")
    ref = PrequantMatrix.encode(a.cpu())
    opened = ref.open().cuda().contiguous()
    codes, scales = _blob_buffers(*opened.shape)
    pre_quant(opened, codes, scales)
    torch.cuda.synchronize()
    _assert_blobs_match(codes, scales, ref.open())


def test_blobs_open_to_the_reference_rows():
    """The two blobs pair up in k order: dequantizing them matches ``open()``.

    The kernel packs two adjacent scale groups per u32 word, so a swapped
    pair would still match a per-row scale multiset while breaking this.
    """
    torch.manual_seed(1)
    m, k = 8, 1024
    a = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    codes, scales = _blob_buffers(m, k)
    pre_quant(a, codes, scales)
    torch.cuda.synchronize()
    assert torch.equal(
        PrequantMatrix(int_values=codes.cpu(), scales=scales.cpu()).open(),
        PrequantMatrix.encode(a.cpu()).open(),
    )


def test_rejects_malformed_buffers():
    m, k = 32, 512
    a = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    codes, scales = _blob_buffers(m, k)

    with pytest.raises(ValueError, match="512"):
        pre_quant(torch.randn(m, k - 8, dtype=torch.bfloat16, device="cuda"), codes, scales)
    with pytest.raises(ValueError, match="codes"):
        pre_quant(a, codes[:, :-8], scales)
    with pytest.raises(ValueError, match="scales"):
        pre_quant(a, codes, scales.to(torch.float32))
    with pytest.raises(ValueError, match="dtype"):
        pre_quant(a.to(torch.float32), codes, scales)
    with pytest.raises(ValueError, match="threads_per_block"):
        PreQuantConfig(threads_per_block=64)
