import math

import pytest
import torch
from pearl_gemm.quantization import DEFAULT_HADAMARD_BLOCK_SIZE, quantize

DEVICE = "cuda"


# ===== Helpers =====


def assert_quantization_close(xq, xq_ref, scales, scales_ref):
    """Assert that quantization results match reference within tolerance."""
    torch.testing.assert_close(scales, scales_ref, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(xq.float(), xq_ref.float(), atol=1, rtol=0)


# ===== Reference implementations =====


def build_hadamard_block(k: int = DEFAULT_HADAMARD_BLOCK_SIZE) -> torch.Tensor:
    """Build k x k normalised orthogonal Hadamard (no all-ones row)."""
    if k < 1 or (k & (k - 1)) != 0:
        raise ValueError(f"Block size must be a power of 2, got {k}")
    H = torch.ones(1, 1)
    while H.shape[0] < k:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    H[:, 0] *= -1
    return H / math.sqrt(k)


def apply_hadamard_blockwise(x, H_block):
    *leading, d = x.shape
    k = H_block.shape[0]
    return x.view(*leading, d // k, k).matmul(H_block).view(*leading, d)


def hadamard_quantize_ref(
    x: torch.Tensor,
    smooth_scale=None,
    max_val: int = 63,
    block_size: int = DEFAULT_HADAMARD_BLOCK_SIZE,
):
    """Pure PyTorch reference for Hadamard quantization."""
    x_f32 = x.to(torch.float32)
    H = build_hadamard_block(block_size).to(device=x.device, dtype=torch.float32)
    x_f32 = apply_hadamard_blockwise(x_f32, H)

    if smooth_scale is not None:
        if smooth_scale.dim() == 1:
            smooth_scale = smooth_scale.unsqueeze(0)
        x_f32 = x_f32 * smooth_scale.to(torch.float32)

    row_max = x_f32.abs().max(dim=-1, keepdim=True).values
    scales = row_max / float(max_val)

    safe_scales = scales.clone()
    safe_scales[safe_scales == 0] = 1.0
    xq = (x_f32 / safe_scales).round().clamp(-float(max_val), float(max_val)).to(torch.int8)

    return xq, scales


def quantize_ref(x: torch.Tensor, smooth_scale=None, max_val: int = 63):
    """Pure PyTorch reference for quantization without Hadamard."""
    x_f32 = x.to(torch.float32)

    if smooth_scale is not None:
        if smooth_scale.dim() == 1:
            smooth_scale = smooth_scale.unsqueeze(0)
        x_f32 = x_f32 * smooth_scale.to(torch.float32)

    row_max = x_f32.abs().max(dim=-1, keepdim=True).values
    scales = row_max / float(max_val)

    safe_scales = scales.clone()
    safe_scales[safe_scales == 0] = 1.0
    xq = (x_f32 / safe_scales).round().clamp(-float(max_val), float(max_val)).to(torch.int8)

    return xq, scales


# ===== Helpers =====


def make_smooth_scale(mode, M, N, low=0.5, span=1.5):
    """Build a smooth-scale tensor for the given broadcast mode.

    mode: None -> no smooth scale; "broadcast" -> (1, N); "per_token" -> (M, N).
    """
    if mode is None:
        return None
    rows = 1 if mode == "broadcast" else M
    return torch.rand(rows, N, dtype=torch.float32, device=DEVICE) * span + low


# Hadamard + smooth cases: (M, N, dtype, block_size, max_val, smooth_mode).
# Broadcast (1, N) and per-token (M, N) smooth scales share the same kernel
# path; the only difference is the smooth tensor's row count, so both modes are
# parametrized here instead of living in separate test classes.
_HADAMARD_SMOOTH_CASES = [
    # broadcast smooth, bf16, default block size
    *[
        (M, N, torch.bfloat16, 16, 63, "broadcast")
        for M, N in [(4, 64), (32, 128), (128, 4096), (1, 8192), (8192, 8192), (7, 4096)]
    ],
    # per-token smooth across dtypes and block sizes
    *[
        (M, N, dtype, block_size, 63, "per_token")
        for M, N in [
            (4, 64),
            (32, 128),
            (32, 1024),
            (128, 4096),
            (512, 4096),
            (512, 8192),
            (1024, 8192),
            (2048, 4096),
            (2048, 8192),
            (8192, 8192),
        ]
        for dtype in (torch.bfloat16, torch.float16)
        for block_size in (16, 32)
    ],
    # per-token smooth, larger max_val
    *[
        (M, N, torch.bfloat16, 16, max_val, "per_token")
        for M, N in [(32, 1024), (128, 4096), (1024, 8192), (2048, 8192)]
        for max_val in (63, 127)
    ],
    # Regression: when N is not a multiple of vecsize*threads_per_row (=2048) the
    # smooth-scale SMEM chunk stride diverged from X's, indexing shared memory out of
    # bounds (cudaErrorIllegalInstruction). N=18944 is one such hidden size.
    *[
        (16, N, dtype, block_size, 63, smooth_mode)
        for N in (5120, 11008, 16896, 18944)
        for dtype in (torch.bfloat16, torch.float16)
        for block_size in (16, 32)
        for smooth_mode in ("broadcast", "per_token")
    ],
]

# Quantization-only (block_size=0) + smooth cases: (num_tokens, hidden, dtype, max_val, smooth_mode).
_QUANT_SMOOTH_CASES = [
    # broadcast smooth
    *[
        (num_tokens, hidden_size, dtype, 63, "broadcast")
        for num_tokens in [1, 32, 128, 512]
        for hidden_size in [128, 4096, 8192]
        for dtype in (torch.float16, torch.bfloat16)
    ],
    # per-token smooth
    *[
        (num_tokens, hidden_size, dtype, 63, "per_token")
        for num_tokens in [1, 32, 128, 512, 1024, 2048]
        for hidden_size in [128, 1024, 4096, 8192]
        for dtype in (torch.float16, torch.bfloat16)
    ],
    # per-token smooth, larger max_val
    *[(128, 1024, torch.bfloat16, max_val, "per_token") for max_val in (63, 127)],
    # Regression: non-multiple-of-2048 hidden sizes with a smooth scale walked the
    # smooth-scale SMEM chunk index out of bounds (cudaErrorIllegalInstruction).
    # hidden=18944 is one such size.
    *[
        (16, hidden, dtype, 63, smooth_mode)
        for hidden in (5120, 11008, 16896, 18944)
        for dtype in (torch.float16, torch.bfloat16)
        for smooth_mode in ("broadcast", "per_token")
    ],
]


# ===== Fixtures =====


@pytest.fixture(autouse=True)
def seed():
    torch.random.manual_seed(42)


@pytest.fixture(autouse=True)
def clear_gpu_cache():
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ===== Hadamard + quantization tests =====


class TestHadamardQuantization:
    """Tests for fused Hadamard transform + quantization (block_size > 0)."""

    @pytest.mark.parametrize(
        "M,N",
        [
            (1, 32),
            (4, 64),
            (32, 128),
            (128, 4096),
            (512, 4096),
            (1, 8192),
            (8192, 8192),
            (32768, 8192),
            # Non-power-of-2 M values
            (3, 256),
            (7, 4096),
            (13, 8192),
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    @pytest.mark.parametrize("block_size", [16, 32])
    def test_basic(self, M, N, dtype, block_size):
        x = torch.randn(M, N, dtype=dtype, device=DEVICE)
        xq = torch.empty(M, N, dtype=torch.int8, device=DEVICE)
        scales = torch.empty(M, 1, dtype=torch.float32, device=DEVICE)
        quantize(x, xq, scales, max_val=63, block_size=block_size)

        assert xq.shape == (M, N)
        assert xq.dtype == torch.int8
        assert scales.shape == (M, 1)
        assert scales.dtype == torch.float32

        xq_ref, scales_ref = hadamard_quantize_ref(x, max_val=63, block_size=block_size)
        assert_quantization_close(xq, xq_ref, scales, scales_ref)

    @pytest.mark.parametrize("M,N,dtype,block_size,max_val,smooth_mode", _HADAMARD_SMOOTH_CASES)
    def test_with_smooth(self, M, N, dtype, block_size, max_val, smooth_mode):
        x = torch.randn(M, N, dtype=dtype, device=DEVICE)
        smooth_scale = make_smooth_scale(smooth_mode, M, N)

        xq = torch.empty(M, N, dtype=torch.int8, device=DEVICE)
        scales = torch.empty(M, 1, dtype=torch.float32, device=DEVICE)
        quantize(x, xq, scales, smooth_scale=smooth_scale, max_val=max_val, block_size=block_size)

        assert xq.shape == (M, N)
        assert xq.dtype == torch.int8
        assert scales.shape == (M, 1)
        assert xq.abs().max() <= max_val

        xq_ref, scales_ref = hadamard_quantize_ref(
            x, smooth_scale=smooth_scale, max_val=max_val, block_size=block_size
        )
        assert_quantization_close(xq, xq_ref, scales, scales_ref)

    @pytest.mark.parametrize("max_val", [63, 127])
    def test_max_val(self, max_val):
        M, N = 32, 128
        x = torch.randn(M, N, dtype=torch.bfloat16, device=DEVICE)

        xq = torch.empty(M, N, dtype=torch.int8, device=DEVICE)
        scales = torch.empty(M, 1, dtype=torch.float32, device=DEVICE)
        quantize(x, xq, scales, max_val=max_val)

        assert xq.abs().max() <= max_val
        xq_ref, scales_ref = hadamard_quantize_ref(x, max_val=max_val)
        assert_quantization_close(xq, xq_ref, scales, scales_ref)

    def test_zeros(self):
        M, N = 8, 64
        x = torch.zeros(M, N, dtype=torch.bfloat16, device=DEVICE)
        xq = torch.empty(M, N, dtype=torch.int8, device=DEVICE)
        scales = torch.empty(M, 1, dtype=torch.float32, device=DEVICE)
        quantize(x, xq, scales, max_val=63)

        assert (xq == 0).all()
        assert (scales == 0).all()

    def test_single_row(self):
        x = torch.randn(1, 4096, dtype=torch.bfloat16, device=DEVICE)
        xq = torch.empty(1, 4096, dtype=torch.int8, device=DEVICE)
        scales = torch.empty(1, 1, dtype=torch.float32, device=DEVICE)
        quantize(x, xq, scales, max_val=63)

        assert xq.shape == (1, 4096)
        assert scales.shape == (1, 1)

    def test_large(self):
        M, N = 2048, 8192
        x = torch.randn(M, N, dtype=torch.bfloat16, device=DEVICE)
        xq = torch.empty(M, N, dtype=torch.int8, device=DEVICE)
        scales = torch.empty(M, 1, dtype=torch.float32, device=DEVICE)
        quantize(x, xq, scales, max_val=63)

        assert xq.shape == (M, N)
        assert xq.dtype == torch.int8
        assert xq.abs().max() <= 63

    def test_dequant_error(self):
        """Check that dequantizing the output recovers a reasonable approximation."""
        M, N = 128, 4096
        x = torch.randn(M, N, dtype=torch.bfloat16, device=DEVICE)
        xq = torch.empty(M, N, dtype=torch.int8, device=DEVICE)
        scales = torch.empty(M, 1, dtype=torch.float32, device=DEVICE)
        quantize(x, xq, scales, max_val=63)

        dequant = xq.float() * scales

        xq_ref, scales_ref = hadamard_quantize_ref(x, max_val=63)
        dequant_ref = xq_ref.float() * scales_ref

        torch.testing.assert_close(dequant, dequant_ref, atol=scales.max().item() * 1.5, rtol=0)


# ===== Quantization-only tests (block_size=0) =====


class TestQuantizationCore:
    """Core functionality tests for dynamic per-token quantization (no Hadamard)."""

    @pytest.mark.parametrize("num_tokens", [1, 4, 7, 15, 16, 33, 64, 512])
    @pytest.mark.parametrize("hidden_size", [128, 256, 512, 1024, 4096])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("max_val", [63, 127])
    def test_quantize(self, num_tokens, hidden_size, dtype, max_val):
        x = torch.randn(num_tokens, hidden_size, dtype=dtype, device=DEVICE)
        xq = torch.empty(num_tokens, hidden_size, dtype=torch.int8, device=DEVICE)
        scales = torch.empty(num_tokens, 1, dtype=torch.float32, device=DEVICE)

        quantize(x, xq, scales, max_val=max_val, block_size=0)

        xq_ref, scales_ref = quantize_ref(x, max_val=max_val)

        assert xq.shape == x.shape
        assert scales.shape == (num_tokens, 1)
        assert torch.all(xq >= -max_val)
        assert torch.all(xq <= max_val)

        assert_quantization_close(xq, xq_ref, scales, scales_ref)

    @pytest.mark.parametrize(
        "num_tokens,hidden_size,dtype,max_val,smooth_mode", _QUANT_SMOOTH_CASES
    )
    def test_quantize_with_smooth(self, num_tokens, hidden_size, dtype, max_val, smooth_mode):
        x = torch.randn(num_tokens, hidden_size, dtype=dtype, device=DEVICE)
        smooth_scale = make_smooth_scale(smooth_mode, num_tokens, hidden_size)

        xq = torch.empty(num_tokens, hidden_size, dtype=torch.int8, device=DEVICE)
        scales = torch.empty(num_tokens, 1, dtype=torch.float32, device=DEVICE)

        quantize(x, xq, scales, smooth_scale=smooth_scale, max_val=max_val, block_size=0)

        assert xq.abs().max() <= max_val
        xq_ref, scales_ref = quantize_ref(x, smooth_scale=smooth_scale, max_val=max_val)
        assert_quantization_close(xq, xq_ref, scales, scales_ref)


class TestSmoothScale:
    """Tests specific to smooth scale functionality."""

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("max_val", [63, 127])
    def test_identity(self, dtype, max_val):
        """Smooth scale of 1.0 should match no smooth scale."""
        num_tokens, hidden_size = 32, 512
        x = torch.randn(num_tokens, hidden_size, dtype=dtype, device=DEVICE)
        smooth_scale = torch.ones(1, hidden_size, dtype=torch.float32, device=DEVICE)

        xq_with, scales_with = (
            torch.empty_like(x, dtype=torch.int8),
            torch.empty(num_tokens, 1, dtype=torch.float32, device=DEVICE),
        )
        quantize(x, xq_with, scales_with, max_val=max_val, smooth_scale=smooth_scale, block_size=0)

        xq_without, scales_without = (
            torch.empty_like(x, dtype=torch.int8),
            torch.empty(num_tokens, 1, dtype=torch.float32, device=DEVICE),
        )
        quantize(x, xq_without, scales_without, max_val=max_val, block_size=0)

        torch.testing.assert_close(xq_with, xq_without)
        torch.testing.assert_close(scales_with, scales_without)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("max_val", [63, 127])
    @pytest.mark.parametrize(
        "scale_type", ["uniform_0.5", "uniform_2.0", "uniform_10.0", "varying"]
    )
    def test_patterns(self, dtype, max_val, scale_type):
        """Test uniform and varying smooth scale patterns."""
        num_tokens, hidden_size = 32, 512
        x = torch.randn(num_tokens, hidden_size, dtype=dtype, device=DEVICE)

        if scale_type.startswith("uniform_"):
            scale_value = float(scale_type.split("_")[1])
            smooth_scale = torch.full(
                (1, hidden_size), scale_value, dtype=torch.float32, device=DEVICE
            )
        else:
            smooth_scale = torch.linspace(
                0.1, 10.0, hidden_size, dtype=torch.float32, device=DEVICE
            ).unsqueeze(0)

        xq = torch.empty(num_tokens, hidden_size, dtype=torch.int8, device=DEVICE)
        scales = torch.empty(num_tokens, 1, dtype=torch.float32, device=DEVICE)
        quantize(x, xq, scales, max_val=max_val, smooth_scale=smooth_scale, block_size=0)

        xq_ref, scales_ref = quantize_ref(x, max_val=max_val, smooth_scale=smooth_scale)
        assert_quantization_close(xq, xq_ref, scales, scales_ref)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("max_val", [63, 127])
    @pytest.mark.parametrize(
        "scale_value,input_scale",
        [
            (1000.0, 100.0),
            (0.001, 0.01),
        ],
    )
    def test_extreme_values(self, dtype, max_val, scale_value, input_scale):
        """Test extreme smooth scale values."""
        num_tokens, hidden_size = 8, 256
        x = torch.rand(num_tokens, hidden_size, dtype=dtype, device=DEVICE) * input_scale
        smooth_scale = torch.full((1, hidden_size), scale_value, dtype=torch.float32, device=DEVICE)

        xq = torch.empty(num_tokens, hidden_size, dtype=torch.int8, device=DEVICE)
        scales = torch.empty(num_tokens, 1, dtype=torch.float32, device=DEVICE)
        quantize(x, xq, scales, max_val=max_val, smooth_scale=smooth_scale, block_size=0)

        assert torch.all(xq >= -max_val)
        assert torch.all(xq <= max_val)

        xq_ref, scales_ref = quantize_ref(x, max_val=max_val, smooth_scale=smooth_scale)
        assert_quantization_close(xq, xq_ref, scales, scales_ref)


class TestQuantizationEdgeCases:
    """Edge case tests for quantization (block_size=0)."""

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("max_val", [63, 127])
    def test_zero_input(self, dtype, max_val):
        """Zero input rows should produce zero output and zero scale."""
        num_tokens, hidden_size = 4, 256
        x = torch.zeros(num_tokens, hidden_size, dtype=dtype, device=DEVICE)

        xq = torch.empty(num_tokens, hidden_size, dtype=torch.int8, device=DEVICE)
        scales = torch.empty(num_tokens, 1, dtype=torch.float32, device=DEVICE)
        quantize(x, xq, scales, max_val=max_val, block_size=0)

        assert torch.all(xq == 0)
        assert torch.all(scales == 0)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("max_val", [63, 127])
    def test_mixed_zero_and_nonzero_rows(self, dtype, max_val):
        """Mix of zero and non-zero rows."""
        num_tokens, hidden_size = 8, 256
        x = torch.randn(num_tokens, hidden_size, dtype=dtype, device=DEVICE)
        x[1, :] = 0
        x[3, :] = 0
        x[7, :] = 0

        xq = torch.empty(num_tokens, hidden_size, dtype=torch.int8, device=DEVICE)
        scales = torch.empty(num_tokens, 1, dtype=torch.float32, device=DEVICE)
        quantize(x, xq, scales, max_val=max_val, block_size=0)

        xq_ref, scales_ref = quantize_ref(x, max_val=max_val)

        assert torch.all(xq[1, :] == 0)
        assert torch.all(xq[3, :] == 0)
        assert torch.all(xq[7, :] == 0)
        assert scales[1, 0] == 0
        assert scales[3, 0] == 0
        assert scales[7, 0] == 0

        assert_quantization_close(xq, xq_ref, scales, scales_ref)


class TestQuantizationRangePreservation:
    """Tests for output range preservation."""

    @pytest.mark.parametrize("max_val", [63, 127])
    @pytest.mark.parametrize("use_smooth_scale", [False, True])
    @pytest.mark.parametrize("block_size", [0, 16])
    def test_output_range(self, max_val, use_smooth_scale, block_size):
        """Output values must be in [-max_val, max_val]."""
        num_tokens, hidden_size = 64, 1024
        x = torch.randn(num_tokens, hidden_size, dtype=torch.float16, device=DEVICE) * 1000

        smooth_scale = (
            torch.rand(1, hidden_size, dtype=torch.float32, device=DEVICE) * 9.9 + 0.1
            if use_smooth_scale
            else None
        )

        xq = torch.empty(num_tokens, hidden_size, dtype=torch.int8, device=DEVICE)
        scales = torch.empty(num_tokens, 1, dtype=torch.float32, device=DEVICE)
        quantize(x, xq, scales, max_val=max_val, smooth_scale=smooth_scale, block_size=block_size)

        assert torch.all(xq >= -max_val), f"Found values below -{max_val}"
        assert torch.all(xq <= max_val), f"Found values above {max_val}"
        assert torch.all(scales >= 0)


class TestQuantizationConsistency:
    """Consistency and determinism tests."""

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("max_val", [63, 127])
    @pytest.mark.parametrize("block_size", [0, 16])
    def test_deterministic(self, dtype, max_val, block_size):
        """Quantization should be deterministic."""
        num_tokens, hidden_size = 16, 512
        x = torch.randn(num_tokens, hidden_size, dtype=dtype, device=DEVICE)

        results = []
        for _ in range(3):
            xq = torch.empty(num_tokens, hidden_size, dtype=torch.int8, device=DEVICE)
            scales = torch.empty(num_tokens, 1, dtype=torch.float32, device=DEVICE)
            quantize(x, xq, scales, max_val=max_val, block_size=block_size)
            results.append((xq.clone(), scales.clone()))

        for i in range(1, len(results)):
            torch.testing.assert_close(results[0][0], results[i][0])
            torch.testing.assert_close(results[0][1], results[i][1])


# ===== Per-token smooth scale tests =====
#
# Numerical correctness of per-token (M, N) smooth scales is covered by the
# ``smooth_mode="per_token"`` parametrization of the smooth tests above. The
# tests below cover behaviour that is specific to the per-token path.


class TestPerTokenSmooth:
    """Behaviour specific to per-token (M, N) smooth scales."""

    def test_per_row_independence(self):
        """Different per-row smooth scales should yield independent results."""
        M, N = 32, 1024
        x = torch.randn(M, N, dtype=torch.bfloat16, device=DEVICE)

        # Row i gets smooth_scale = i+1 uniformly
        smooth_scale = torch.arange(1, M + 1, dtype=torch.float32, device=DEVICE)
        smooth_scale = smooth_scale.unsqueeze(1).expand(M, N).contiguous()

        xq = torch.empty(M, N, dtype=torch.int8, device=DEVICE)
        scales = torch.empty(M, 1, dtype=torch.float32, device=DEVICE)
        quantize(x, xq, scales, smooth_scale=smooth_scale, max_val=63, block_size=16)

        xq_ref, scales_ref = hadamard_quantize_ref(
            x, smooth_scale=smooth_scale, max_val=63, block_size=16
        )
        assert_quantization_close(xq, xq_ref, scales, scales_ref)

    def test_matches_broadcast_when_replicated(self):
        """Per-token smooth replicated from (1, N) should match broadcast (1, N)."""
        M, N = 128, 4096
        x = torch.randn(M, N, dtype=torch.bfloat16, device=DEVICE)
        smooth_1 = torch.rand(1, N, dtype=torch.float32, device=DEVICE) * 1.5 + 0.5
        smooth_m = smooth_1.expand(M, N).contiguous()

        xq_b = torch.empty(M, N, dtype=torch.int8, device=DEVICE)
        sc_b = torch.empty(M, 1, dtype=torch.float32, device=DEVICE)
        quantize(x, xq_b, sc_b, smooth_scale=smooth_1, max_val=63, block_size=16)

        xq_p = torch.empty(M, N, dtype=torch.int8, device=DEVICE)
        sc_p = torch.empty(M, 1, dtype=torch.float32, device=DEVICE)
        quantize(x, xq_p, sc_p, smooth_scale=smooth_m, max_val=63, block_size=16)

        torch.testing.assert_close(sc_b, sc_p, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(xq_b.float(), xq_p.float(), atol=1, rtol=0)

    def test_zeros_per_token(self):
        """Zero smooth for a specific row should produce zero output for that row."""
        M, N = 8, 256
        x = torch.randn(M, N, dtype=torch.bfloat16, device=DEVICE)
        smooth_scale = torch.ones(M, N, dtype=torch.float32, device=DEVICE)
        smooth_scale[3, :] = 0.0  # row 3 fully zeroed

        xq = torch.empty(M, N, dtype=torch.int8, device=DEVICE)
        scales = torch.empty(M, 1, dtype=torch.float32, device=DEVICE)
        quantize(x, xq, scales, smooth_scale=smooth_scale, max_val=63, block_size=0)

        assert torch.all(xq[3, :] == 0)
        assert scales[3, 0] == 0

    def test_broadcast_vs_per_token_same_content(self):
        """Same numerical smooth, different tensor shapes, should yield same output."""
        M, N = 64, 1024
        x = torch.randn(M, N, dtype=torch.bfloat16, device=DEVICE)
        smooth_1 = torch.rand(1, N, dtype=torch.float32, device=DEVICE) * 2 + 0.5
        smooth_m = smooth_1.expand(M, N).contiguous()

        for block_size in [0, 16]:
            xq_b = torch.empty(M, N, dtype=torch.int8, device=DEVICE)
            sc_b = torch.empty(M, 1, dtype=torch.float32, device=DEVICE)
            quantize(x, xq_b, sc_b, smooth_scale=smooth_1, max_val=63, block_size=block_size)

            xq_p = torch.empty(M, N, dtype=torch.int8, device=DEVICE)
            sc_p = torch.empty(M, 1, dtype=torch.float32, device=DEVICE)
            quantize(x, xq_p, sc_p, smooth_scale=smooth_m, max_val=63, block_size=block_size)

            torch.testing.assert_close(sc_b, sc_p, atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(xq_b.float(), xq_p.float(), atol=1, rtol=0)

    def test_mixed_block_sizes(self):
        """Per-token smooth should work across different block sizes."""
        M, N = 128, 4096
        x = torch.randn(M, N, dtype=torch.bfloat16, device=DEVICE)
        smooth_scale = torch.rand(M, N, dtype=torch.float32, device=DEVICE) + 0.5
        for block_size in [0, 16, 32]:
            xq = torch.empty(M, N, dtype=torch.int8, device=DEVICE)
            scales = torch.empty(M, 1, dtype=torch.float32, device=DEVICE)
            quantize(x, xq, scales, smooth_scale=smooth_scale, max_val=63, block_size=block_size)
            assert xq.abs().max() <= 63, f"block_size={block_size} out of range"
