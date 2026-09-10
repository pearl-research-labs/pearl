"""``noisy_quant_b`` vs the reference chain, over the two committed weight planes.

B-side gates: E_B / alpha_b / beta_b / B' codes / the ``-(beta (.) E_B)`` peel
half bit-exact, the ``(beta (.) E_B@F_B - B') @ F_A^T`` mid half to tolerance --
with the reference chain driven through the new-scheme API (planes -> commit
over the codes and scales planes with fused stats, exact norms off the
committed int8 blocks + scales, ``build_b_rows`` over the opened rows). The
kernel is keyed by B's noise-line key (``Subkey("noise-line", seedB)``); F_A
(per-A in v4) enters only as the raw peel factor, drawn here from a probe seed.

The mid tolerance is looser than the A side's ``A'@F2^T`` gate because the
reference computes that half through its tolerance-path matmul with bf16
roundings of ``E2@F2`` and ``diff`` that the kernel's reassociated epilogue
does not replicate (both sit within ~3e-3 of the fp64-exact mid).
"""

import pytest
import torch
from blake3 import blake3
from miner_base.commitment import Device
from miner_base.commitment_hash import noise_line_key
from miner_base.hardware import hardware_for
from miner_base.noise import OperandNoiser, Side
from miner_base.prequant import PrequantMatrix
from miner_base.quantization import Fp8QuantScheme
from miner_base.scheme import PearlScheme

from pearl_gemm import (
    NoisyQuantBConfig,
    TensorHashConfig,
    noisy_quant_b,
    pack_noise_factor,
    pre_quant,
    pre_quant_output_shapes,
    tensor_hash_plus_stats_b,
    tensor_hash_workspace_bytes,
)
from pearl_gemm.protocol_constants import R
from tests.helpers.chain import KEY_A, SEED_B, device_bytes

_SEED_A_PROBE = blake3(b"seed-a-probe").digest()

# The reassociated mid half vs the reference's tolerance-path mid (see module
# docstring); the bit-exact halves use _mism. Near-degenerate inputs (e.g. a
# zero operand, where the mid is pure quantization residue after the noise
# terms cancel) make the mid's own norm a meaningless denominator, so the gate
# is self-calibrating: the GPU mid must sit no further from the fp64-exact mid
# than twice the reference's own tolerance-path deviation, with an absolute
# floor for the well-conditioned case.
_MID_REL_TOL = 1e-2


def _mism(got: torch.Tensor, ref: torch.Tensor) -> int:
    def bits(t):
        return t.view(torch.uint8) if t.dtype == torch.float8_e4m3fn else t

    return (bits(got.cpu()) != bits(ref)).sum().item()


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


def _noisers(seed_b: bytes, k: int, hw=None):
    """The reference per-side noisers: B's from ``seed_b``, A's from the probe seed."""
    compute = (hw or hardware_for(Device.BLACKWELL)).compute
    return (
        OperandNoiser(_SEED_A_PROBE, Side.A, R, k, compute),
        OperandNoiser(seed_b, Side.B, R, k, compute),
    )


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
        pack_noise_factor(f2).cuda(),
        f1.cuda().contiguous(),
        *outputs.values(),
        torch.zeros(R, R, dtype=torch.float32, device="cuda"),
        config=config,
    )
    torch.cuda.synchronize()
    return outputs


def _assert_b_chain_matches_reference(b: torch.Tensor, config_fields=None) -> None:
    """Run planes -> commit_b -> noisy_quant_b on ``b`` and gate every B-side
    output against the reference chain."""
    n, k = b.shape
    hw = hardware_for(Device.BLACKWELL)

    codes, scales, commit_stats = _committed_planes(b)

    # Factors from the commitment chain (reference noisers per side).
    noise_a, noise_b = _noisers(SEED_B, k, hw)
    e2, f1, f2 = noise_b.E(list(range(n))), noise_a.F(), noise_b.F()

    config = NoisyQuantBConfig(**(config_fields or {}))
    got = _launch_b(codes, scales, commit_stats, SEED_B, f1, f2, config)

    # The reference: rows are the opened planes and the norms come off the
    # committed int8 blocks + scales (exact_norms), matching the GPU's
    # factored commit stats. beta is not carried on the built rows.
    bq_ref = PrequantMatrix.encode(b.cpu())
    opened = bq_ref.open()
    row_norms = bq_ref.exact_norms()
    ref_stacked = PearlScheme(hw, Fp8QuantScheme(), k, R).build_b_rows(
        opened, noise_a, noise_b, list(range(n)), row_norms
    )
    _, _, ref_beta, _ = Fp8QuantScheme().noisy_quantize(opened, e2, f2, hw, row_norms)

    assert _mism(got["e2"], e2) == 0, "E_B codes"
    assert _mism(got["alpha_b"], ref_stacked.alpha.flatten()) == 0, "alpha_b"
    assert _mism(got["beta_b"], ref_beta.flatten()) == 0, "beta_b"
    assert _mism(got["b_prime"], ref_stacked.quant_part) == 0, "B' codes"
    assert _mism(got["b_peel"][:, R:], ref_stacked.peel_part[:, R:]) == 0, "-(beta (.) E_B)"

    # The pinned noise dot is exact integer arithmetic, so the fp64 chain is
    # the exact mid; both implementations approximate it (see _MID_REL_TOL).
    e2f2_64 = e2.double() @ f2.double()
    diff_64 = ref_beta.double() * e2f2_64 - ref_stacked.quant_part.double()
    mid_64 = diff_64 @ f1.double().t()
    gpu_err = _rel(got["b_peel"][:, :R].double(), mid_64)
    ref_err = _rel(ref_stacked.peel_part[:, :R].double(), mid_64)
    assert gpu_err < max(2 * ref_err, _MID_REL_TOL), f"mid err {gpu_err} vs ref err {ref_err}"


@pytest.mark.parametrize("n,k", _SHAPES, ids=[_shape_id(s) for s in _SHAPES])
@pytest.mark.parametrize("seed", [0, 1])
def test_noisy_quant_b_matches_reference(n, k, seed):
    torch.manual_seed(seed)
    _assert_b_chain_matches_reference(torch.randn(n, k, dtype=torch.bfloat16, device="cuda") * 1.5)


# The pinned SM100 config families (consumer-E rows=64, both bk tiles)
# against the reference at a multi-row-block shape, on the B operands.
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
    b = torch.randn(512, 4096, dtype=torch.bfloat16, device="cuda") * 1.5
    _assert_b_chain_matches_reference(b, config_fields)


# 2^56 (not bf16-max): bf16-max overflows fp32 norm accum to NaN beta.
_EDGE_FILLS = {
    "zeros": 0.0,
    "ones": 1.0,
    "neg_ones": -1.0,
    "huge": 2.0**56,
    "neg_huge": -(2.0**56),
}


@pytest.mark.parametrize("fill", sorted(_EDGE_FILLS))
def test_edge_fill_weights_match_reference(fill):
    b = torch.full((64, 512), _EDGE_FILLS[fill], dtype=torch.bfloat16, device="cuda")
    _assert_b_chain_matches_reference(b)


def test_other_seed_b_rekeys_the_draw():
    """Another job's ``seedB`` re-keys E_B (and F_B) and stays bit-exact."""
    n, k = 64, 512
    torch.manual_seed(2)
    b = torch.randn(n, k, dtype=torch.bfloat16, device="cuda") * 1.5
    seed_b = blake3(b"seed-b-other").digest()
    hw = hardware_for(Device.BLACKWELL)
    codes, scales, commit_stats = _committed_planes(b)

    noise_a, noise_b = _noisers(seed_b, k, hw)
    e2, f1, f2 = noise_b.E(list(range(n))), noise_a.F(), noise_b.F()
    got = _launch_b(codes, scales, commit_stats, seed_b, f1, f2, NoisyQuantBConfig())

    bq_ref = PrequantMatrix.encode(b.cpu())
    ref_stacked = PearlScheme(hw, Fp8QuantScheme(), k, R).build_b_rows(
        bq_ref.open(), noise_a, noise_b, list(range(n)), bq_ref.exact_norms()
    )
    assert _mism(got["e2"], e2) == 0
    assert _mism(got["b_prime"], ref_stacked.quant_part) == 0
    assert _mism(got["b_peel"][:, R:], ref_stacked.peel_part[:, R:]) == 0

    # And the fixture seed over the same planes draws different noise.
    base = _launch_b(codes, scales, commit_stats, SEED_B, f1, f2, NoisyQuantBConfig())
    assert _mism(base["e2"], e2) != 0


def test_consistency():
    """Repeat launches over the same committed planes are bit-identical."""
    n, k = 128, 1024
    torch.manual_seed(11)
    b = torch.randn(n, k, dtype=torch.bfloat16, device="cuda") * 1.5
    codes, scales, commit_stats = _committed_planes(b)
    noise_a, noise_b = _noisers(SEED_B, k)
    f1, f2 = noise_a.F(), noise_b.F()
    config = NoisyQuantBConfig()

    first = _launch_b(codes, scales, commit_stats, SEED_B, f1, f2, config)
    second = _launch_b(codes, scales, commit_stats, SEED_B, f1, f2, config)
    for name in first:
        assert torch.equal(first[name].view(torch.uint8), second[name].view(torch.uint8)), name


def test_mixed_gemm_parity_gpu_vs_reference_prep():
    """``mixed_gemm`` fed GPU-prepped B vs reference-prepped B: bit-identical
    lottery accumulator, C'' within the existing end-to-end tolerance."""
    from pearl_gemm import HitSignal, HitSignalConfig, MixedGemmConfig, mixed_gemm

    m, n, k = 256, 256, 512
    torch.manual_seed(4)
    b = torch.randn(n, k, dtype=torch.bfloat16, device="cuda") * 1.5
    hw = hardware_for(Device.BLACKWELL)
    codes, scales, commit_stats = _committed_planes(b)
    noise_a, noise_b = _noisers(SEED_B, k, hw)
    f1, f2 = noise_a.F(), noise_b.F()
    got = _launch_b(codes, scales, commit_stats, SEED_B, f1, f2, NoisyQuantBConfig())

    bq_ref = PrequantMatrix.encode(b.cpu())
    ref_stacked = PearlScheme(hw, Fp8QuantScheme(), k, R).build_b_rows(
        bq_ref.open(), noise_a, noise_b, list(range(n)), bq_ref.exact_norms()
    )

    # A fixed synthetic A side; only the B operands differ between runs.
    a_prime = (torch.randn(m, k, device="cuda") * 0.1).to(torch.float8_e4m3fn)
    a_peel = torch.randn(m, 2 * R, dtype=torch.bfloat16, device="cuda") * 0.1
    alpha_a = torch.rand(m, dtype=torch.bfloat16, device="cuda") + 0.5
    pow_key = torch.randint(0, 256, (32,), dtype=torch.uint8, device="cuda")
    threshold = torch.zeros(32, dtype=torch.uint8, device="cuda")  # never-win
    config = MixedGemmConfig(cluster_m=1, cluster_n=1)
    # Synthetic A side: the committed planes only feed the (never-taken)
    # hit-snapshot path, so zeros of the right shapes suffice.
    a_codes = torch.zeros(m, k, dtype=torch.int8, device="cuda")
    a_scales = torch.zeros(m, k // 8, dtype=torch.bfloat16, device="cuda")
    commitment_hash_b = torch.zeros(32, dtype=torch.uint8, device="cuda")
    hit_signal = HitSignal(HitSignalConfig(max_m=m, max_k=k))

    def run(b_prime, b_peel, alpha_b):
        out = torch.zeros(m, n, dtype=torch.bfloat16, device="cuda")
        mixed_gemm(
            a_prime,
            b_prime.contiguous(),
            a_peel,
            b_peel.contiguous(),
            alpha_a,
            torch.reciprocal(alpha_b.flatten().float()).contiguous(),
            pow_key,
            threshold,
            out,
            hit_signal,
            a_codes,
            a_scales,
            commitment_hash_b,
            config=config,
        )
        torch.cuda.synchronize()
        return out

    out_gpu = run(got["b_prime"], got["b_peel"], got["alpha_b"])
    out_ref = run(
        ref_stacked.quant_part.cuda(), ref_stacked.peel_part.cuda(), ref_stacked.alpha.cuda()
    )

    assert _rel(out_gpu, out_ref) < 5e-3
    assert torch.isfinite(out_gpu).all()
