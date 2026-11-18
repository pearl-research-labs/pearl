"""``noisy_quant_b`` against pinned digests, over the two committed weight planes.

B-side gates: E_B / alpha_b / beta_b / B' codes / the ``-(beta (.) E_B)`` peel
half bit-exact against the protocol's operands for host-drawn inputs
(``helpers/digests.py``), the ``(beta (.) E_B@F_B - B') @ F_A^T`` mid half
against its fp64-exact product -- with the chain driven through the
new-scheme API (planes -> commit over the codes and scales planes with fused
stats -> ``noisy_quant_b`` against the device F bases). The kernel is keyed
by B's noise-line key (``Subkey("noise-line", seedB)``); F_A enters as the raw
peel factor and is keyed by the same seedB (Side.A address), so no seedA is
involved on the B side.

The mid tolerance is looser than the A side's ``A'@F2^T`` gate because the
kernel's reassociated epilogue sits ~3e-3 from the fp64-exact mid.
"""

import pytest
import torch
from blake3 import blake3
from miner_base.commitment_hash import noise_line_key

from pearl_gemm import (
    NoiseLoadMode,
    NoisyQuantBConfig,
    TensorHashConfig,
    noisy_quant_b,
    pack_noise_factor,
    pre_quant,
    pre_quant_output_shapes,
    tensor_hash_plus_stats_b,
    tensor_hash_workspace_bytes,
)
from pearl_gemm._utils._arch import Arch, arch_of
from pearl_gemm.protocol_constants import R
from tests.helpers.chain import KEY_A, SEED_B, device_bytes, f_bases
from tests.helpers.digests import blackwell_digests, digest, fixture_input


def _rel(got: torch.Tensor, ref: torch.Tensor) -> float:
    got, ref = got.cpu().float(), ref.cpu().float()
    return ((got - ref).norm() / ref.norm().clamp_min(1e-30)).item()


# n % 32 == 0 (default noise_rows), k % 512 == 0 (commit chunks).
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


def _case_id(n, k, seed):
    return f"{n}x{k}-seed{seed}"


def _committed_planes(b: torch.Tensor):
    """GPU planes + fused commit stats for ``b`` (the weights-path commit)."""
    n, k = b.shape
    codes_shape, scales_shape = pre_quant_output_shapes(n, k)
    codes = torch.zeros(codes_shape, dtype=torch.int8, device="cuda")
    scales = torch.zeros(scales_shape, dtype=torch.bfloat16, device="cuda")
    pre_quant(b, codes, scales)

    hash_config = TensorHashConfig()
    key = device_bytes(KEY_A)
    root_codes = torch.zeros(32, dtype=torch.uint8, device="cuda")
    root_scales = torch.zeros(32, dtype=torch.uint8, device="cuda")
    roots = torch.zeros(
        tensor_hash_workspace_bytes(n, k, hash_config),
        dtype=torch.uint8,
        device="cuda",
    )
    commit_stats = torch.zeros(2 * (n * k // 512), dtype=torch.float32, device="cuda")
    tensor_hash_plus_stats_b(
        codes,
        scales,
        key,
        root_codes,
        root_scales,
        roots,
        commit_stats,
        config=hash_config,
    )
    return codes, scales, commit_stats


def _launch_b(codes, scales, commit_stats, seed_b, f1, f2, config):
    """GPU ``noisy_quant_b`` (keyed by ``seed_b``'s noise-line key) over fresh
    caller-owned outputs; returns them."""
    n, k = codes.shape
    outputs = {
        "alpha_b": torch.zeros(n, dtype=torch.bfloat16, device="cuda"),
        "beta_b": torch.zeros(n, dtype=torch.bfloat16, device="cuda"),
        "e2": torch.zeros(n, R, dtype=torch.float8_e4m3fn, device="cuda"),
        "b_prime": torch.zeros(n, k, dtype=torch.float8_e4m3fn, device="cuda"),
        "b_peel": torch.zeros(n, 2 * R, dtype=torch.bfloat16, device="cuda"),
    }
    noisy_quant_b(
        codes,
        scales,
        device_bytes(noise_line_key(seed_b)),
        commit_stats,
        pack_noise_factor(f2),
        f1,
        *outputs.values(),
        torch.zeros(R, R, dtype=torch.float32, device="cuda"),
        config=config,
    )
    torch.cuda.synchronize()
    return outputs


def _assert_b_chain_matches_pinned(
    b: torch.Tensor, case: str, config_fields=None, seed_b: bytes = SEED_B
):
    """Run planes -> commit_b -> noisy_quant_b on ``b`` and gate every B-side
    output against the pinned ``case``; returns the outputs and F bases."""
    k = b.shape[1]
    codes, scales, commit_stats = _committed_planes(b)
    f1, f2 = f_bases(k, seed_b)
    config = NoisyQuantBConfig(**(config_fields or {}))
    got = _launch_b(codes, scales, commit_stats, seed_b, f1, f2, config)

    expected = blackwell_digests("noisy_quant_b")[case]
    assert digest(got["e2"]) == expected["e2"], "E_B codes"
    assert digest(got["alpha_b"]) == expected["alpha_b"], "alpha_b"
    assert digest(got["beta_b"]) == expected["beta_b"], "beta_b"
    assert digest(got["b_prime"]) == expected["b_prime"], "B' codes"
    assert digest(got["b_peel"][:, R:]) == expected["peel_beta_e"], "-(beta (.) E_B)"

    # The pinned noise dot is exact integer arithmetic, so the fp64 chain over
    # the (pinned) operands is the exact mid. Near-degenerate inputs (e.g. a
    # zero operand, where the mid is pure quantization residue after the noise
    # terms cancel) carry a wider pinned tolerance.
    e2f2_64 = got["e2"].cpu().double() @ f2.cpu().double()
    diff_64 = got["beta_b"].cpu().double()[:, None] * e2f2_64 - got["b_prime"].cpu().double()
    mid_64 = diff_64 @ f1.cpu().double().t()
    gpu_err = _rel(got["b_peel"][:, :R].double(), mid_64)
    assert gpu_err < expected["peel_tol"], f"mid err {gpu_err}"
    return got, f1, f2, (codes, scales, commit_stats)


@pytest.mark.parametrize("n,k", _SHAPES, ids=[_shape_id(s) for s in _SHAPES])
@pytest.mark.parametrize("seed", [0, 1])
def test_noisy_quant_b_matches_pinned(n, k, seed):
    _assert_b_chain_matches_pinned(fixture_input(n, k, seed).cuda(), _case_id(n, k, seed))


# The pinned config families (consumer-E rows=64, both bk tiles) at a
# multi-row-block shape, on the B operands: the config never changes the bits. SM120's
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
    b = fixture_input(512, 4096, 0).cuda()
    _assert_b_chain_matches_pinned(b, _case_id(512, 4096, 0), config_fields)


# 2^56 (not bf16-max): bf16-max overflows fp32 norm accum to NaN beta.
_EDGE_FILLS = {
    "zeros": 0.0,
    "ones": 1.0,
    "neg_ones": -1.0,
    "huge": 2.0**56,
    "neg_huge": -(2.0**56),
}


@pytest.mark.parametrize("fill", sorted(_EDGE_FILLS))
def test_edge_fill_weights_match_pinned(fill):
    b = torch.full((64, 512), _EDGE_FILLS[fill], dtype=torch.bfloat16, device="cuda")
    _assert_b_chain_matches_pinned(b, fill)


def test_other_seed_b_rekeys_the_draw():
    """Another job's ``seedB`` re-keys E_B (and F_B) and stays bit-exact."""
    b = fixture_input(64, 512, 2).cuda()
    seed_b = blake3(b"seed-b-other").digest()
    got, f1, f2, planes = _assert_b_chain_matches_pinned(b, "other_seed_b", seed_b=seed_b)

    # And the fixture seed over the same planes draws different noise.
    base = _launch_b(*planes, SEED_B, f1, f2, NoisyQuantBConfig())
    assert not torch.equal(base["e2"].view(torch.uint8), got["e2"].view(torch.uint8))


def test_consistency():
    """Repeat launches over the same committed planes are bit-identical."""
    n, k = 128, 1024
    torch.manual_seed(11)
    b = torch.randn(n, k, dtype=torch.bfloat16, device="cuda") * 1.5
    codes, scales, commit_stats = _committed_planes(b)
    f1, f2 = f_bases(k)
    config = NoisyQuantBConfig()

    first = _launch_b(codes, scales, commit_stats, SEED_B, f1, f2, config)
    second = _launch_b(codes, scales, commit_stats, SEED_B, f1, f2, config)
    for name in first:
        assert torch.equal(first[name].view(torch.uint8), second[name].view(torch.uint8)), name
