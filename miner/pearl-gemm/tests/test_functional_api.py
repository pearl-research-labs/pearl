"""Regression tests for caller-owned launch buffers and compile-cache reuse."""

import pytest
import torch

from pearl_gemm import (
    LABEL_F1,
    HitSignal,
    HitSignalConfig,
    MixedGemmConfig,
    NoisyQuantBConfig,
    NoisyQuantConfig,
    mixed_gemm,
    noise_lines,
    noisy_quant,
    noisy_quant_b,
    pack_noise_factor,
    pre_quant,
    pre_quant_output_shapes,
)
from pearl_gemm.protocol_constants import R


def _pre_quant_buffers(m=32, k=512):
    codes_shape, scales_shape = pre_quant_output_shapes(m, k)
    return {
        "a": torch.randn(m, k, dtype=torch.bfloat16, device="cuda"),
        "codes": torch.zeros(codes_shape, dtype=torch.int8, device="cuda"),
        "scales": torch.zeros(scales_shape, dtype=torch.bfloat16, device="cuda"),
    }


def test_pre_quant_rebinds_input_buffers():
    buffers = _pre_quant_buffers()
    pre_quant(**buffers)
    torch.cuda.synchronize()
    first = buffers["codes"].clone()

    buffers["a"] = -buffers["a"]
    pre_quant(**buffers)
    torch.cuda.synchronize()
    assert not torch.equal(first, buffers["codes"])


def test_pre_quant_rejects_misaligned_assumed_buffer():
    buffers = _pre_quant_buffers()
    m, k = buffers["a"].shape
    storage = torch.empty(m * k + 1, dtype=torch.bfloat16, device="cuda")
    buffers["a"] = storage[1:].view(m, k)
    with pytest.raises(ValueError, match="aligned"):
        pre_quant(**buffers)


def _fake_noise_lines(k: int) -> torch.Tensor:
    """Noise-line-like e4m3 values (+-[0.5, ~200], the OperandNoiser's range)."""
    magnitudes = torch.rand(R, k, device="cuda") * 200 + 0.5
    signs = torch.randint(0, 2, (R, k), device="cuda") * 2 - 1
    return (magnitudes * signs).to(torch.float8_e4m3fn)


def _noisy_buffers(m=32, k=512):
    config = NoisyQuantConfig()
    codes_shape, scales_shape = pre_quant_output_shapes(m, k)
    codes = torch.zeros(codes_shape, dtype=torch.int8, device="cuda")
    scales = torch.zeros(scales_shape, dtype=torch.bfloat16, device="cuda")
    pre_quant(torch.randn(m, k, dtype=torch.bfloat16, device="cuda"), codes, scales)
    return {
        "codes": codes,
        "scales": scales,
        "c_a": torch.randint(0, 256, (32,), dtype=torch.uint8, device="cuda"),
        "commit_stats": torch.rand(2 * (m * k // 512), device="cuda") + 1,
        "f1_hl": pack_noise_factor(_fake_noise_lines(k)),
        "f2": torch.randn(R, k, device="cuda").to(torch.float8_e4m3fn),
        "alpha": torch.zeros(m, dtype=torch.bfloat16, device="cuda"),
        "beta": torch.zeros(m, dtype=torch.bfloat16, device="cuda"),
        "e1": torch.zeros(m, R, dtype=torch.float8_e4m3fn, device="cuda"),
        "a_prime": torch.zeros(m, k, dtype=torch.float8_e4m3fn, device="cuda"),
        "a_peel": torch.zeros(m, 2 * R, dtype=torch.bfloat16, device="cuda"),
        "config": config,
    }


def test_noisy_quant_rebinds_factor_buffers():
    buffers = _noisy_buffers()
    noisy_quant(**buffers)
    first = buffers["a_prime"].clone()

    buffers["f1_hl"] = torch.zeros_like(buffers["f1_hl"])
    noisy_quant(**buffers)
    torch.cuda.synchronize()
    assert not torch.equal(first.view(torch.uint8), buffers["a_prime"].view(torch.uint8))


def test_noisy_quant_rejects_malformed_stats():
    buffers = _noisy_buffers()
    buffers["commit_stats"] = buffers["commit_stats"].to(torch.float64)
    with pytest.raises(ValueError, match="commit_stats"):
        noisy_quant(**buffers)


def test_noisy_quant_rejects_misaligned_assumed_buffer():
    buffers = _noisy_buffers()
    m, k = buffers["codes"].shape
    storage = torch.empty(m * k + 1, dtype=torch.int8, device="cuda")
    buffers["codes"] = storage[1:].view(m, k)
    with pytest.raises(ValueError, match="aligned"):
        noisy_quant(**buffers)


def test_noisy_quant_rejects_mismatched_scales_blob():
    buffers = _noisy_buffers()
    m, groups = buffers["scales"].shape
    buffers["scales"] = buffers["scales"][:, : groups // 2].contiguous()
    with pytest.raises(ValueError, match="scales"):
        noisy_quant(**buffers)


def _noisy_b_buffers(n=32, k=512):
    config = NoisyQuantBConfig()
    codes_shape, scales_shape = pre_quant_output_shapes(n, k)
    codes = torch.zeros(codes_shape, dtype=torch.int8, device="cuda")
    scales = torch.zeros(scales_shape, dtype=torch.bfloat16, device="cuda")
    pre_quant(torch.randn(n, k, dtype=torch.bfloat16, device="cuda"), codes, scales)
    return {
        "codes": codes,
        "scales": scales,
        "c_b": torch.randint(0, 256, (32,), dtype=torch.uint8, device="cuda"),
        "commit_stats": torch.rand(2 * (n * k // 512), device="cuda") + 1,
        "f2_hl": pack_noise_factor(_fake_noise_lines(k)),
        "f1": torch.randn(R, k, device="cuda").to(torch.float8_e4m3fn),
        "alpha_b": torch.zeros(n, dtype=torch.bfloat16, device="cuda"),
        "beta_b": torch.zeros(n, dtype=torch.bfloat16, device="cuda"),
        "e2": torch.zeros(n, R, dtype=torch.float8_e4m3fn, device="cuda"),
        "b_prime": torch.zeros(n, k, dtype=torch.float8_e4m3fn, device="cuda"),
        "b_peel": torch.zeros(n, 2 * R, dtype=torch.bfloat16, device="cuda"),
        "gram": torch.zeros(R, R, dtype=torch.float32, device="cuda"),
        "config": config,
    }


def test_noisy_quant_b_rebinds_factor_buffers():
    buffers = _noisy_b_buffers()
    noisy_quant_b(**buffers)
    first = buffers["b_prime"].clone()

    buffers["f2_hl"] = torch.zeros_like(buffers["f2_hl"])
    noisy_quant_b(**buffers)
    torch.cuda.synchronize()
    assert not torch.equal(first.view(torch.uint8), buffers["b_prime"].view(torch.uint8))


def test_noisy_quant_b_rejects_b_named_buffers():
    buffers = _noisy_b_buffers()
    buffers["commit_stats"] = buffers["commit_stats"].to(torch.float64)
    with pytest.raises(ValueError, match="commit_stats"):
        noisy_quant_b(**buffers)

    buffers = _noisy_b_buffers()
    # Packed width is PACKED_NOISE_K, which equals R at R=32; drop a column
    # rather than slicing to R, which would be a no-op.
    buffers["f2_hl"] = buffers["f2_hl"][:, :-1].contiguous()
    with pytest.raises(ValueError, match="f2_hl"):
        noisy_quant_b(**buffers)

    buffers = _noisy_b_buffers()
    buffers["b_peel"] = buffers["b_peel"][:, :R].contiguous()
    with pytest.raises(ValueError, match="b_peel"):
        noisy_quant_b(**buffers)


def test_noise_lines_rebinds_key_buffer():
    key = torch.randint(0, 256, (32,), dtype=torch.uint8, device="cuda")
    out = torch.zeros(128, R, dtype=torch.float8_e4m3fn, device="cuda")
    noise_lines(key, LABEL_F1, out)
    torch.cuda.synchronize()
    first = out.clone()

    key.add_(1)
    noise_lines(key, LABEL_F1, out)
    torch.cuda.synchronize()
    assert not torch.equal(first.view(torch.uint8), out.view(torch.uint8))


def _mixed_buffers(m=256, n=128, k=512):
    return {
        "a_prime": torch.randn(m, k, device="cuda").to(torch.float8_e4m3fn),
        "b_prime": torch.randn(n, k, device="cuda").to(torch.float8_e4m3fn),
        "a_peel": torch.zeros(m, 2 * R, dtype=torch.bfloat16, device="cuda"),
        "b_peel": torch.zeros(n, 2 * R, dtype=torch.bfloat16, device="cuda"),
        "alpha_a": torch.ones(m, dtype=torch.bfloat16, device="cuda"),
        "inv_alpha_b": torch.ones(n, dtype=torch.float32, device="cuda"),
        "pow_key": torch.zeros(32, dtype=torch.uint8, device="cuda"),
        "threshold": torch.zeros(32, dtype=torch.uint8, device="cuda"),
        "out": torch.zeros(m, n, dtype=torch.bfloat16, device="cuda"),
        "hit_signal": HitSignal(HitSignalConfig(max_m=m, max_k=k)),
        "a_codes": torch.randint(-127, 128, (m, k), dtype=torch.int8, device="cuda"),
        "a_scales": torch.rand(m, k // 8, dtype=torch.bfloat16, device="cuda"),
        "commitment_hash_b": torch.zeros(32, dtype=torch.uint8, device="cuda"),
        "config": MixedGemmConfig(cluster_m=1, cluster_n=1),
    }


def test_mixed_gemm_rebinds_b_buffers():
    buffers = _mixed_buffers()
    mixed_gemm(**buffers)
    first = buffers["out"].clone()

    buffers["b_prime"] = (-buffers["b_prime"].float()).to(torch.float8_e4m3fn)
    mixed_gemm(**buffers)
    torch.cuda.synchronize()
    assert not torch.equal(first, buffers["out"])


def test_mixed_gemm_rejects_short_pow_key():
    buffers = _mixed_buffers()
    buffers["pow_key"] = buffers["pow_key"][:-1]
    with pytest.raises(ValueError, match="pow_key"):
        mixed_gemm(**buffers)


def test_mixed_gemm_requires_boolean_record_hits():
    buffers = _mixed_buffers()
    with pytest.raises(TypeError, match="record_hits must be bool"):
        mixed_gemm(**buffers, record_hits=1)


_OUTPUT_OPERANDS = ("out", "a_peel", "b_peel", "alpha_a", "inv_alpha_b")


@pytest.mark.parametrize("missing", _OUTPUT_OPERANDS)
def test_mixed_gemm_requires_every_output_operand(missing):
    """The output and its peel/unscale operands are all mandatory."""
    buffers = _mixed_buffers()
    buffers[missing] = None
    with pytest.raises(TypeError, match=f"{missing} must be a torch.Tensor"):
        mixed_gemm(**buffers)


def test_mixed_gemm_requires_hit_signal():
    buffers = _mixed_buffers()
    buffers["hit_signal"] = torch.zeros(4, dtype=torch.int32)
    with pytest.raises(TypeError, match="HitSignal"):
        mixed_gemm(**buffers)


def test_mixed_gemm_rejects_misaligned_assumed_buffer():
    buffers = _mixed_buffers()
    m, n = buffers["out"].shape
    storage = torch.empty(m * n + 1, dtype=torch.bfloat16, device="cuda")
    buffers["out"] = storage[1:].view(m, n)
    with pytest.raises(ValueError, match="aligned"):
        mixed_gemm(**buffers)


def test_mixed_gemm_signal_resets_between_hit_and_no_hit():
    buffers = _mixed_buffers()
    signal = buffers["hit_signal"]
    buffers["threshold"].fill_(255)
    mixed_gemm(**buffers)
    torch.cuda.synchronize()
    hit = signal.read_hit()
    assert hit is not None and hit.valid
    assert 0 <= hit.tile_row < buffers["a_prime"].shape[0] // 4
    assert 0 <= hit.tile_column < buffers["b_prime"].shape[0] // 128

    buffers["threshold"].zero_()
    signal.reset_hit()
    mixed_gemm(**buffers)
    torch.cuda.synchronize()
    assert not signal.doorbell()
    assert signal.read_hit() is None


def test_mixed_gemm_no_hit_path_is_cuda_graph_capturable():
    buffers = _mixed_buffers()
    signal = buffers["hit_signal"]
    mixed_gemm(**buffers)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        mixed_gemm(**buffers)
    graph.replay()
    torch.cuda.synchronize()

    assert signal.read_hit() is None
    assert torch.isfinite(buffers["out"]).all()
