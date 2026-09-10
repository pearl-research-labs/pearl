"""Correctness parity across noisy-quant tuning configurations."""

import pytest
import torch

from pearl_gemm import (
    NoiseLoadMode,
    NoisyQuantConfig,
    noisy_quant,
    pack_noise_factor,
    pre_quant,
    pre_quant_output_shapes,
)
from pearl_gemm.autotune import NOISY_QUANT_SPACE
from pearl_gemm.protocol_constants import R


def _open_blobs(codes, scales):
    """Dequantize the two committed blobs: bf16(f32(code) * f32(scale))."""
    m, k = codes.shape
    return (
        (codes.float().reshape(m, k // 8, 8) * scales.float().reshape(m, k // 8, 1))
        .to(torch.bfloat16)
        .reshape(m, k)
    )


def _fake_noise_lines(k: int) -> torch.Tensor:
    """Noise-line-like e4m3 values (+-[0.5, ~200], the Noiser's range)."""
    magnitudes = torch.rand(R, k, device="cuda") * 200 + 0.5
    signs = torch.randint(0, 2, (R, k), device="cuda") * 2 - 1
    return (magnitudes * signs).to(torch.float8_e4m3fn)


@pytest.fixture(scope="module")
def inputs():
    torch.manual_seed(11)
    m, k = 64, 512
    a = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    codes_shape, scales_shape = pre_quant_output_shapes(m, k)
    codes = torch.zeros(codes_shape, dtype=torch.int8, device="cuda")
    scales = torch.zeros(scales_shape, dtype=torch.bfloat16, device="cuda")
    pre_quant(a, codes, scales)
    torch.cuda.synchronize()
    # The hasher derives the commit stats from the opened rows, not from A.
    chunks = _open_blobs(codes, scales).float().reshape(-1, 512)
    stats = torch.stack(
        (chunks.square().sum(dim=1), chunks.abs().amax(dim=1)),
        dim=1,
    ).flatten()
    return {
        "codes": codes,
        "scales": scales,
        "c_a": torch.arange(32, dtype=torch.uint8, device="cuda"),
        "stats": stats,
        "f1_hl": pack_noise_factor(_fake_noise_lines(k)),
        "f2": torch.randn(R, k, device="cuda").to(torch.float8_e4m3fn),
    }


def _run(inputs, config):
    m, k = inputs["codes"].shape
    outputs = {
        "alpha": torch.empty(m, dtype=torch.bfloat16, device="cuda"),
        "beta": torch.empty(m, dtype=torch.bfloat16, device="cuda"),
        "e1": torch.empty(m, R, dtype=torch.float8_e4m3fn, device="cuda"),
        "a_prime": torch.empty(m, k, dtype=torch.float8_e4m3fn, device="cuda"),
        "a_peel": torch.empty(m, 2 * R, dtype=torch.bfloat16, device="cuda"),
    }
    noisy_quant(
        inputs["codes"],
        inputs["scales"],
        inputs["c_a"],
        inputs["stats"],
        inputs["f1_hl"],
        inputs["f2"],
        outputs["alpha"],
        outputs["beta"],
        outputs["e1"],
        outputs["a_prime"],
        outputs["a_peel"],
        config=config,
    )
    torch.cuda.synchronize()
    return outputs


def _assert_same(got, reference):
    for name in ("alpha", "beta", "e1", "a_prime"):
        assert torch.equal(got[name].view(torch.uint8), reference[name].view(torch.uint8)), name
    assert torch.equal(
        got["a_peel"][:, :R].view(torch.uint8),
        reference["a_peel"][:, :R].view(torch.uint8),
    )
    torch.testing.assert_close(
        got["a_peel"][:, R:].float(),
        reference["a_peel"][:, R:].float(),
        rtol=1e-4,
        atol=1e-3,
    )


@pytest.mark.parametrize("mode", list(NoiseLoadMode))
@pytest.mark.parametrize("bk", [128, 256])
def test_block_k_and_load_modes_match(inputs, bk, mode):
    """Fast smoke over every ``noise_bk`` x ``NoiseLoadMode`` compile key on
    one shape: load-mode coverage alone would leave the bk-dependent staging
    paths uncompiled on the PR gate."""
    reference = _run(inputs, NoisyQuantConfig())
    got = _run(inputs, NoisyQuantConfig(noise_bk=bk, noise_load_mode=mode))
    _assert_same(got, reference)


@pytest.mark.parametrize("mode", list(NoiseLoadMode))
@pytest.mark.parametrize("bk", [128, 256])
def test_rows64_path_matrix_matches(inputs, bk, mode):
    reference = _run(inputs, NoisyQuantConfig())
    got = _run(
        inputs,
        NoisyQuantConfig(
            noise_rows=64,
            noise_bk=bk,
            noise_stages=2,
            noise_load_mode=mode,
        ),
    )
    _assert_same(got, reference)


@pytest.mark.slow
@pytest.mark.parametrize("config_fields", NOISY_QUANT_SPACE)
def test_autotune_space_matches_default(inputs, config_fields):
    reference = _run(inputs, NoisyQuantConfig())
    got = _run(inputs, NoisyQuantConfig(**config_fields))
    _assert_same(got, reference)


@pytest.mark.slow
@pytest.mark.parametrize("rows", [16, 32, 64])
def test_row_block_sizes_match(inputs, rows):
    reference = _run(inputs, NoisyQuantConfig())
    got = _run(inputs, NoisyQuantConfig(noise_rows=rows))
    _assert_same(got, reference)


_CONSISTENCY_ITERS = 10_000


@pytest.mark.slow
@pytest.mark.parametrize("config_fields", NOISY_QUANT_SPACE)
def test_consistency_across_tuning_space(inputs, config_fields):
    """Relaunches under every tuned config are bit-identical (q8 noise-A
    consistency). Unlike cross-config parity, this holds for every output
    including the deviable ``A' @ F2.T`` columns."""
    config = NoisyQuantConfig(**config_fields)
    first = _run(inputs, config)
    for _ in range(_CONSISTENCY_ITERS):
        got = _run(inputs, config)
        for name, tensor in first.items():
            assert torch.equal(got[name].view(torch.uint8), tensor.view(torch.uint8)), name
