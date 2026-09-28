"""``noisy_quant`` against pinned digests, over the two block-scaled A blobs.

A-side gates: E_A / alpha / beta / A' codes / peel_e bit-exact against the
protocol's operands for host-drawn inputs (``helpers/digests.py``), and the
A'@F_B.T peel half against its fp64-exact product -- with the chain driven
through the new-scheme API (``pre_quant`` -> commit over the codes and scales
blobs -> the A keys -> ``noisy_quant`` against the device F bases).

The kernel reads the committed codes and scales blobs and opens them in
registers, so the bit-exact gates also pin the in-kernel decode.
"""

import pytest
import torch

from pearl_gemm import (
    NoiseLoadMode,
    NoisyQuantConfig,
    noisy_quant,
    pack_noise_factor,
    pre_quant,
    pre_quant_output_shapes,
    validate_noisy_quant_config,
)
from pearl_gemm._utils._arch import Arch, arch_of
from pearl_gemm.protocol_constants import R
from tests.helpers.chain import commit_a, f_bases
from tests.helpers.digests import digest, fixture_input, reference_digests


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


def _case_id(m, k, seed):
    return f"{m}x{k}-seed{seed}"


def _commit(a: torch.Tensor):
    """pre_quant -> commit on GPU; returns the blobs and the committed A."""
    m, k = a.shape
    codes_shape, scales_shape = pre_quant_output_shapes(m, k)
    codes = torch.zeros(codes_shape, dtype=torch.int8, device="cuda")
    scales = torch.zeros(scales_shape, dtype=torch.bfloat16, device="cuda")
    pre_quant(a, codes, scales)
    return codes, scales, commit_a(codes, scales)


def _assert_chain_matches_pinned(a: torch.Tensor, case: str, config_fields=None) -> None:
    """Run pre_quant -> commit -> noisy_quant on ``a`` and gate every A-side
    output against the pinned ``case``."""
    m, k = a.shape

    # Steps 1 + 2: block-scale quantization into the two blobs, commit both
    # blobs + fuse the stats, finalize the A keys (GPU).
    codes, scales, committed = _commit(a)
    f1, f2 = f_bases(k)

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
        pack_noise_factor(f1),
        f2,
        alpha,
        beta,
        e1_out,
        a_prime,
        a_peel,
        config=config,
    )
    torch.cuda.synchronize()

    expected = reference_digests()["noisy_quant"][case]
    assert digest(e1_out) == expected["e1"], "E_A codes"
    assert digest(alpha) == expected["alpha"], "alpha"
    assert digest(beta) == expected["beta"], "beta"
    assert digest(a_prime) == expected["a_prime"], "A' codes"
    assert digest(a_peel[:, :R]) == expected["peel_e"], "peel_e"
    # The A'F_B half is the scheme's tolerance surface: it approximates the
    # fp64-exact A'@F_B.T of the (pinned) A' codes (single fixed-order fp32
    # chains drift more at long k), within twice the reference's own error.
    af2_64 = a_prime.cpu().float().double() @ f2.cpu().float().double().t()
    gpu_rel = _rel(a_peel[:, R:], af2_64)
    assert gpu_rel < expected["peel_tol"], f"A'@F_B.T rel {gpu_rel}"


@pytest.mark.parametrize("m,k", _SHAPES, ids=[_shape_id(s) for s in _SHAPES])
@pytest.mark.parametrize("seed", [0, 1])
def test_noisy_quant_matches_pinned(m, k, seed):
    _assert_chain_matches_pinned(fixture_input(m, k, seed).cuda(), _case_id(m, k, seed))


# The pinned config families (consumer-E1 rows=64, both bk tiles) at a
# multi-row-block shape: the config never changes the bits. SM120's
# shared memory holds the bk=256 family only behind the half-tile ring.
@pytest.mark.parametrize(
    "config_fields",
    [
        {
            "noise_rows": 64,
            "noise_bk": 256,
            "noise_stages": 2,
            "noise_out_stages": 3,
        }
        if arch_of() is Arch.SM100
        else {
            "noise_rows": 64,
            "noise_bk": 256,
            "noise_stages": 2,
            "noise_out_stages": 2,
            "noise_load_mode": NoiseLoadMode.RING,
        },
        {
            "noise_rows": 64,
            "noise_bk": 128,
            "noise_stages": 2,
            "noise_out_stages": 3,
        },
    ],
)
def test_pinned_config_families_match_pinned(config_fields):
    a = fixture_input(512, 4096, 0).cuda()
    _assert_chain_matches_pinned(a, _case_id(512, 4096, 0), config_fields)


# 2^56 (not bf16-max): bf16-max overflows fp32 norm accum to NaN beta.
_EDGE_FILLS = {
    "zeros": 0.0,
    "ones": 1.0,
    "neg_ones": -1.0,
    "huge": 2.0**56,
    "neg_huge": -(2.0**56),
}


@pytest.mark.parametrize("fill", sorted(_EDGE_FILLS))
def test_edge_fill_activations_match_pinned(fill):
    a = torch.full((64, 512), _EDGE_FILLS[fill], dtype=torch.bfloat16, device="cuda")
    _assert_chain_matches_pinned(a, fill)


def test_consistency():
    """Repeat launches over the same committed blobs are bit-identical."""
    m, k = 128, 1024
    torch.manual_seed(11)
    a = torch.randn(m, k, dtype=torch.bfloat16, device="cuda") * 1.5
    codes, scales, committed = _commit(a)
    f1, f2 = f_bases(k)

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
            pack_noise_factor(f1),
            f2,
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
