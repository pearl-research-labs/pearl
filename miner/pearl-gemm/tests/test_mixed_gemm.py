"""``mixed_gemm`` kernel correctness -- q8 ``test_pearl_gemm`` spirit.

Isolated FP8 operands (no full miner chain): C'' peel algebra, hit-signal
publication, and repeat-launch determinism against the committed lottery
tiles (the default 4x128, the tall 16x32, and the 64-wide 4x64 on the
64-row kernel tile, parameterized via :data:`_VARIANTS`). The 16x32 tile
is what makes wide-``k`` layers such as GLM-5.2 ``o_proj`` (k=16384)
verifiable.
"""

import pytest
import torch
from miner_base.layout import lane_assignment

from pearl_gemm import HitSignal, HitSignalConfig, MixedGemmConfig, mixed_gemm
from pearl_gemm.protocol_constants import R
from tests.helpers.preprocess import tall_tile_config

_OUTPUT_RTOL = 5e-3

# Committed tile variants. The 16x32 variant exists for GLM-5.2 o_proj
# (256x6144x16384): its k exceeds the 4x128 tile's 4 MiB verifier limit.
_VARIANTS = {
    "4x128": {
        "config": MixedGemmConfig(),
        "lottery_shapes": [(256, 128, 512), (512, 256, 1024)],
    },
    "16x32": {
        "config": MixedGemmConfig(ltile_rows=16, ltile_cols=32),
        "lottery_shapes": [(256, 128, 512), (512, 256, 1024)],
    },
    # The 64-row CTA tile (decode m=64 shapes): 16dp TMEM loads split every
    # accumulator row between two threads and the 16 messages per CTA
    # exercise the sub-warp hashing gate. (64, 192, 512) is a decode-like
    # single-CTA-row problem.
    "4x64-tile64": {
        "config": MixedGemmConfig(tile_m=64, tile_n=64, cluster_m=1, cluster_n=1, ltile_cols=64),
        "lottery_shapes": [(256, 128, 512), (64, 192, 512)],
    },
}


def _cases(kind: str):
    """Expand ``(variant, m, n, k)`` params for one shape list across tiles."""
    return [
        pytest.param(vid, m, n, k, id=f"{vid}-{m}x{n}x{k}")
        for vid, variant in _VARIANTS.items()
        for (m, n, k) in variant[kind]
    ]


def _rel(got: torch.Tensor, ref: torch.Tensor) -> float:
    got, ref = got.cpu().float(), ref.cpu().float()
    return ((got - ref).norm() / ref.norm().clamp_min(1e-30)).item()


def _make_inputs(m: int, n: int, k: int, seed: int = 0):
    torch.manual_seed(seed)
    a_prime = (torch.randn(m, k, device="cuda") * 0.1).to(torch.float8_e4m3fn)
    b_prime = (torch.randn(n, k, device="cuda") * 0.1).to(torch.float8_e4m3fn)
    a_peel = torch.randn(m, 2 * R, dtype=torch.bfloat16, device="cuda") * 0.1
    b_peel = torch.randn(n, 2 * R, dtype=torch.bfloat16, device="cuda") * 0.1
    alpha_a = (torch.rand(m, device="cuda") + 0.5).to(torch.bfloat16)
    inv_alpha_b = torch.reciprocal(torch.rand(n, device="cuda") + 0.5)
    return {
        "a_prime": a_prime,
        "b_prime": b_prime,
        "a_peel": a_peel,
        "b_peel": b_peel,
        "alpha_a": alpha_a,
        "inv_alpha_b": inv_alpha_b,
        "pow_key": torch.arange(32, dtype=torch.uint8, device="cuda"),
        "threshold": torch.zeros(32, dtype=torch.uint8, device="cuda"),
        "hit_signal": HitSignal(HitSignalConfig(max_m=m, max_k=k)),
        "a_codes": torch.randint(-127, 128, (m, k), dtype=torch.int8, device="cuda"),
        "a_scales": torch.rand(m, k // 8, dtype=torch.bfloat16, device="cuda"),
        "commitment_hash_b": torch.zeros(32, dtype=torch.uint8, device="cuda"),
    }


def _run(
    inputs,
    *,
    threshold: int | None = None,
    config: MixedGemmConfig | None = None,
    record_hits: bool = True,
):
    """Launch mixed_gemm."""
    config = MixedGemmConfig() if config is None else config
    m, k = inputs["a_prime"].shape
    n = inputs["b_prime"].shape[0]
    out = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
    hit_signal = inputs["hit_signal"]
    thr = inputs["threshold"]
    if threshold is not None:
        thr = torch.frombuffer(
            bytearray(threshold.to_bytes(32, "little")),
            dtype=torch.uint8,
        ).cuda()
    if hit_signal.doorbell():
        hit_signal.reset_hit()
    mixed_gemm(
        inputs["a_prime"],
        inputs["b_prime"],
        inputs["a_peel"],
        inputs["b_peel"],
        inputs["alpha_a"],
        inputs["inv_alpha_b"],
        inputs["pow_key"],
        thr,
        out,
        hit_signal,
        inputs["a_codes"],
        inputs["a_scales"],
        inputs["commitment_hash_b"],
        config=config,
        record_hits=record_hits,
    )
    torch.cuda.synchronize()
    return out, hit_signal, config


def _output_reference(inputs) -> torch.Tensor:
    """Host FP8-upcast matmul + peel + unscale. Not bit-exact to SM100 MMA."""
    output = inputs["a_prime"].float() @ inputs["b_prime"].float().T
    output = output + inputs["a_peel"].float() @ inputs["b_peel"].float().T
    output *= torch.reciprocal(inputs["alpha_a"].float()).reshape(-1, 1)
    output *= inputs["inv_alpha_b"].reshape(1, -1)
    return output


def test_tall_tile_pattern_satisfies_verifier_bounds():
    """The 16x32 committed tile keeps the 4x128 area (difficulty-invariant) but
    shrinks rows+cols to 48, which keeps a k=16384 peel proof under the
    verifier's 4 MiB worker-input cap -- the shape the tall tile exists for."""
    protocol = tall_tile_config(16384)
    lanes = lane_assignment(protocol.rows_pattern, protocol.cols_pattern)
    assert len(lanes) == 16
    assert protocol.rows_pattern.tile_size == 16
    assert protocol.cols_pattern.tile_size == 32
    assert 16 * 32 == 4 * 128
    assert (16 + 32) * 16384 * 2 <= 1 << 22
    # subtile is 1x32 <= 256 elems; lane j is row j's 32 columns ascending.
    assert lanes == [[j * 32 + c for c in range(32)] for j in range(16)]


@pytest.mark.parametrize("variant,m,n,k", _cases("lottery_shapes"))
def test_consistency(variant, m, n, k):
    """Repeat launches over the same buffers are bit-identical (q8 consistency)."""
    config = _VARIANTS[variant]["config"]
    inputs = _make_inputs(m, n, k, seed=5)
    out1, _, _ = _run(inputs, threshold=(1 << 255), config=config)
    out2, _, _ = _run(inputs, threshold=(1 << 255), config=config)
    assert torch.equal(out1, out2)


@pytest.mark.slow
@pytest.mark.parametrize("variant", list(_VARIANTS), ids=list(_VARIANTS))
def test_consistency_many_iterations(variant):
    config = _VARIANTS[variant]["config"]
    m, n, k = _VARIANTS[variant]["lottery_shapes"][0]
    inputs = _make_inputs(m, n, k, seed=9)
    out0, _, _ = _run(inputs, threshold=(1 << 200), config=config)
    for _ in range(32):
        out, _, _ = _run(inputs, threshold=(1 << 200), config=config)
        assert torch.equal(out, out0)


@pytest.mark.parametrize("record_hits", [True, False], ids=["publish", "no_publish"])
def test_record_hits_gates_publication(record_hits):
    """``record_hits`` still hashes; only the flag gates publication."""
    m, n, k = 256, 128, 512
    inputs = _make_inputs(m, n, k, seed=29)
    _, signal, _ = _run(
        inputs,
        threshold=(1 << 256) - 1,
        record_hits=record_hits,
    )
    assert signal.doorbell() is record_hits


def test_is_cuda_graph_capturable():
    """Capture and replay reproduce the eager C'' exactly."""
    m, n, k = 512, 256, 1024
    inputs = _make_inputs(m, n, k, seed=37)
    out_eager, _, _ = _run(inputs, threshold=0)
    out = torch.zeros_like(out_eager)

    def launch():
        mixed_gemm(
            inputs["a_prime"],
            inputs["b_prime"],
            inputs["a_peel"],
            inputs["b_peel"],
            inputs["alpha_a"],
            inputs["inv_alpha_b"],
            inputs["pow_key"],
            inputs["threshold"],
            out,
            inputs["hit_signal"],
            inputs["a_codes"],
            inputs["a_scales"],
            inputs["commitment_hash_b"],
        )

    launch()  # warm the compile cache outside capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    for _ in range(3):
        out.zero_()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, out_eager)


def test_c_tracks_peel_algebra():
    """Peel + unscale reconstructs C'' from a host FP8-upcast matmul."""
    inputs = _make_inputs(256, 256, 1024, seed=2)
    out, _, _ = _run(inputs)
    assert _rel(out, _output_reference(inputs)) < _OUTPUT_RTOL


_EASY_THRESHOLD = (1 << 256) - 1  # all-0xff: every in-bounds tile wins
_GEMMA_K = 5376  # 64-aligned Gemma-3 31B width; not a multiple of 512


def test_k_not_multiple_of_512_snapshots_payload():
    """Gemma-3 31B k=5376: a win snapshots both committed planes."""
    m, n, k = 256, 128, _GEMMA_K
    inputs = _make_inputs(m, n, k, seed=_GEMMA_K)
    _, hit_signal, _ = _run(inputs, threshold=_EASY_THRESHOLD)
    assert hit_signal.doorbell()
    hit = hit_signal.read_hit()
    assert hit is not None and hit.valid
    assert (hit.m, hit.n, hit.k) == (m, n, k)
    assert hit.codes is not None and hit.scales is not None
    assert hit.codes.numel() == m * k
    assert hit.scales.numel() * hit.scales.element_size() == m * (k // 8) * 2
    assert torch.equal(hit.codes, inputs["a_codes"].cpu())
    assert torch.equal(hit.scales.view(torch.uint16), inputs["a_scales"].cpu().view(torch.uint16))
    hit_signal.reset_hit()
    assert not hit_signal.doorbell()
    assert int(hit_signal.lock.item()) == 0


def test_k_not_multiple_of_512_publishes_payloadless_when_signal_is_undersized():
    """Same k=5376 launch against a signal that cannot hold the planes."""
    m, n, k = 256, 128, _GEMMA_K
    inputs = _make_inputs(m, n, k, seed=1)
    inputs["hit_signal"] = HitSignal(HitSignalConfig(max_m=1, max_k=64))
    _, hit_signal, _ = _run(inputs, threshold=_EASY_THRESHOLD)
    hit = hit_signal.read_hit()
    assert hit is not None and hit.valid
    assert hit.codes is None and hit.scales is None
    assert (hit.m, hit.n, hit.k) == (m, n, k)
    hit_signal.reset_hit()
