"""``noisy_quant`` vs the reference chain, over the two block-scaled A blobs.

A-side gates: E_A / alpha / beta / A' codes / peel_e bit-exact, A'@F_B.T
peel to tolerance -- with the reference chain driven through the new-scheme
API (``pre_quant`` -> commit over the codes and scales blobs -> the A keys,
exact norms off the committed int8 blocks + scales, ``noisy_quantize`` over
the opened rows).

The kernel reads the committed codes and scales blobs and opens them in
registers, while the reference is driven by ``PrequantMatrix.open()``: the
bit-exact gates therefore also pin the in-kernel decode.
"""

import pytest
import torch
from miner_base.commitment import Device
from miner_base.hardware import hardware_for
from miner_base.prequant import PrequantMatrix
from miner_base.quantization import Fp8QuantScheme
from miner_base.scheme import PearlScheme

from pearl_gemm import (
    NoisyQuantConfig,
    noisy_quant,
    pack_noise_factor,
    pre_quant,
    pre_quant_output_shapes,
    validate_noisy_quant_config,
)
from pearl_gemm.protocol_constants import R
from tests.helpers.chain import commit_a, noise_b


def _mism(got: torch.Tensor, ref: torch.Tensor) -> int:
    def bits(t):
        return t.view(torch.uint8) if t.dtype == torch.float8_e4m3fn else t

    return (bits(got.cpu()) != bits(ref)).sum().item()


def _rel(got: torch.Tensor, ref: torch.Tensor) -> float:
    got, ref = got.cpu().float(), ref.cpu().float()
    return ((got - ref).norm() / ref.norm().clamp_min(1e-30)).item()


# m % 32 == 0 (default noise_rows), k % 512 == 0 (commit chunks).
_SHAPES = [
    (32, 512),
    (64, 512),
    (96, 2560),
    (128, 1024),
    (128, 7680),
    (256, 2048),
    (256, 4096),
    (512, 1536),
    (512, 4096),
    (1024, 2048),
]


def _shape_id(shape):
    return "x".join(map(str, shape))


def _commit(a: torch.Tensor):
    """pre_quant -> commit on GPU; returns the blobs and the committed A."""
    m, k = a.shape
    codes_shape, scales_shape = pre_quant_output_shapes(m, k)
    codes = torch.zeros(codes_shape, dtype=torch.int8, device="cuda")
    scales = torch.zeros(scales_shape, dtype=torch.bfloat16, device="cuda")
    pre_quant(a, codes, scales)
    return codes, scales, commit_a(codes, scales)


def _assert_chain_matches_reference(a: torch.Tensor, config_fields=None) -> None:
    """Run pre_quant -> commit -> noisy_quant on ``a`` and gate every A-side
    output against the reference chain."""
    m, k = a.shape
    hw = hardware_for(Device.BLACKWELL)

    # Steps 1 + 2: block-scale quantization into the two blobs, commit both
    # blobs + fuse the stats, finalize the A keys (GPU).
    codes, scales, committed = _commit(a)

    # Downstream ops consume the opened (dequantized) block-scaled rows.
    aq_ref = PrequantMatrix.encode(a.cpu())
    opened = aq_ref.open()

    # Factors from the commitment chain (reference OperandNoiser per side).
    noise_a, noise_bb = committed.noise_a(k, hw.compute), noise_b(k=k, compute=hw.compute)
    e1, f1, f2 = noise_a.E(list(range(m))), noise_a.F(), noise_bb.F()

    # Step 3 on GPU.
    config = NoisyQuantConfig(**(config_fields or {}))
    alpha = torch.zeros(m, dtype=torch.bfloat16, device="cuda")
    beta = torch.zeros_like(alpha)
    e1_out = torch.zeros(m, R, dtype=torch.float8_e4m3fn, device="cuda")
    a_prime = torch.zeros(m, k, dtype=torch.float8_e4m3fn, device="cuda")
    a_peel = torch.zeros(m, 2 * R, dtype=torch.bfloat16, device="cuda")
    noisy_quant(
        codes,
        scales,
        committed.noise_key_a_dev,
        committed.commit_stats,
        pack_noise_factor(f1).cuda(),
        f2.cuda().contiguous(),
        alpha,
        beta,
        e1_out,
        a_prime,
        a_peel,
        config=config,
    )
    torch.cuda.synchronize()

    # Step 3 in the reference: rows are the opened blobs and the norms come
    # off the committed int8 blocks + scales (exact_norms), matching the
    # GPU's factored commit stats. beta is not carried on the built rows, so
    # it comes back from the quant step that derives it.
    row_norms = aq_ref.exact_norms()
    ref_stacked = PearlScheme(hw, Fp8QuantScheme(), k, R).build_a_rows(
        opened, noise_a, noise_bb, list(range(m)), row_norms
    )
    _, _, ref_beta, _ = Fp8QuantScheme().noisy_quantize(opened, e1, f1, hw, row_norms)

    assert _mism(e1_out, e1) == 0, "E_A codes"
    assert _mism(alpha, ref_stacked.alpha.flatten()) == 0, "alpha"
    assert _mism(beta, ref_beta.flatten()) == 0, "beta"
    assert _mism(a_prime, ref_stacked.quant_part) == 0, "A' codes"
    assert _mism(a_peel[:, :R], ref_stacked.peel_part[:, :R]) == 0, "peel_e"
    # The A'F_B half is the scheme's tolerance surface: both implementations
    # approximate the fp64-exact A'@F_B.T (single fixed-order fp32 chains
    # drift more at long k), so the gate is self-calibrating against the
    # reference's own error, like the B-side mid gate.
    af2_64 = ref_stacked.quant_part.float().double() @ f2.float().double().t()
    gpu_rel = _rel(a_peel[:, R:], af2_64)
    ref_rel = _rel(ref_stacked.peel_part[:, R:], af2_64)
    assert gpu_rel < max(2 * ref_rel, 1e-4), f"A'@F_B.T rel {gpu_rel} vs ref {ref_rel}"


@pytest.mark.parametrize("m,k", _SHAPES, ids=[_shape_id(s) for s in _SHAPES])
@pytest.mark.parametrize("seed", [0, 1])
def test_noisy_quant_matches_reference(m, k, seed):
    torch.manual_seed(seed)
    _assert_chain_matches_reference(torch.randn(m, k, dtype=torch.bfloat16, device="cuda") * 1.5)


# The pinned SM100 config families (consumer-E1 rows=64, both bk tiles)
# against the reference at a multi-row-block shape.
@pytest.mark.parametrize(
    "config_fields",
    [
        {
            "noise_rows": 64,
            "noise_bk": 256,
            "noise_stages": 2,
            "noise_out_stages": 3,
        },
        {
            "noise_rows": 64,
            "noise_bk": 128,
            "noise_stages": 2,
            "noise_out_stages": 3,
        },
    ],
)
def test_pinned_config_families_match_reference(config_fields):
    torch.manual_seed(0)
    a = torch.randn(512, 4096, dtype=torch.bfloat16, device="cuda") * 1.5
    _assert_chain_matches_reference(a, config_fields)


# 2^56 (not bf16-max): bf16-max overflows fp32 norm accum to NaN beta.
_EDGE_FILLS = {
    "zeros": 0.0,
    "ones": 1.0,
    "neg_ones": -1.0,
    "huge": 2.0**56,
    "neg_huge": -(2.0**56),
}


@pytest.mark.parametrize("fill", sorted(_EDGE_FILLS))
def test_edge_fill_activations_match_reference(fill):
    a = torch.full((64, 512), _EDGE_FILLS[fill], dtype=torch.bfloat16, device="cuda")
    _assert_chain_matches_reference(a)


def test_consistency():
    """Repeat launches over the same committed blobs are bit-identical."""
    m, k = 128, 1024
    torch.manual_seed(11)
    a = torch.randn(m, k, dtype=torch.bfloat16, device="cuda") * 1.5
    codes, scales, committed = _commit(a)
    f1, f2 = committed.noise_a(k).F(), noise_b(k=k).F()

    config = NoisyQuantConfig()

    def launch():
        alpha = torch.zeros(m, dtype=torch.bfloat16, device="cuda")
        beta = torch.zeros_like(alpha)
        e1_out = torch.zeros(m, R, dtype=torch.float8_e4m3fn, device="cuda")
        a_prime = torch.zeros(m, k, dtype=torch.float8_e4m3fn, device="cuda")
        a_peel = torch.zeros(m, 2 * R, dtype=torch.bfloat16, device="cuda")
        noisy_quant(
            codes,
            scales,
            committed.noise_key_a_dev,
            committed.commit_stats,
            pack_noise_factor(f1).cuda(),
            f2.cuda().contiguous(),
            alpha,
            beta,
            e1_out,
            a_prime,
            a_peel,
            config=config,
        )
        torch.cuda.synchronize()
        return alpha, beta, e1_out, a_prime, a_peel

    first = launch()
    second = launch()
    for left, right in zip(first, second, strict=True):
        assert torch.equal(left.view(torch.uint8), right.view(torch.uint8))


@pytest.mark.parametrize(
    "config,match",
    [
        ({"noise_rows": 48}, "noise_rows"),
        ({"noise_rows": 32.0}, "noise_rows"),
        ({"noise_bk": 64}, "noise_bk"),
        ({"noise_bk": 128.0}, "noise_bk"),
        ({"noise_bk": 96}, "noise_bk"),
        ({"noise_stages": 1}, "noise_stages"),
        ({"noise_stages": 2.0}, "noise_stages"),
        ({"noise_out_stages": 1}, "noise_out_stages"),
        ({"noise_out_stages": 2.0}, "noise_out_stages"),
    ],
)
def test_rejects_invalid_config_fields(config, match):
    with pytest.raises(ValueError, match=match):
        validate_noisy_quant_config(256, 2048, NoisyQuantConfig(**config))


def test_rejects_resident_slice_exceeding_shared_memory():
    """RESIDENT keeps a row block's whole k extent in smem, so a large k
    must fail validation instead of silently changing the layout."""
    with pytest.raises(ValueError, match="shared memory"):
        validate_noisy_quant_config(
            256,
            65536,
            NoisyQuantConfig(noise_load_mode="resident"),
        )
