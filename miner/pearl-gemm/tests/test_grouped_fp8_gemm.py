"""``grouped_fp8_gemm`` vs per-expert e4m3 dequant (layout / grouping).

The kernel is SM100-only (tcgen05 UMMA, TMEM, CTA pairs); on other families
the launch tests skip and only the host-side config / shape / geometry
checks run.
"""

import pytest
import torch

from pearl_gemm import (
    GroupedFp8GemmConfig,
    grouped_fp8_gemm,
    grouped_fp8_gemm_scale_shapes,
)
from pearl_gemm._utils._arch import Arch, arch_of
from pearl_gemm.grouped_fp8_gemm import MAX_GROUPED_ROWS, WIDE_TILE_MIN_TOKENS_PER_GROUP
from pearl_gemm.grouped_fp8_gemm._kernel import GroupedGemmSm100

_ON_SM100 = arch_of() is Arch.SM100
sm100_only = pytest.mark.skipif(
    not _ON_SM100, reason="grouped_fp8_gemm is the SM100 kernel (tcgen05 UMMA / TMEM)"
)

_SHAPES = [
    # E, tokens/expert, n, k  (tokens/expert multiple of 4)
    (4, 4, 256, 256),
    (8, 8, 512, 256),
    (32, 4, 4096, 6144),  # GLM-5.2 fused gate+up, m_total=128
]


def _randn_fp8(*shape: int) -> torch.Tensor:
    return (torch.randn(*shape, device="cuda") * 0.05).to(torch.bfloat16).to(torch.float8_e4m3fn)


def _buffers(num_groups: int, tokens: int, n: int, k: int):
    m = num_groups * tokens
    a = _randn_fp8(m, k)
    b = _randn_fp8(num_groups, n, k)
    a_shape, b_shape = grouped_fp8_gemm_scale_shapes(m, n, k, num_groups)
    a_scale = torch.ones(a_shape, device="cuda", dtype=torch.float32)
    b_scale = torch.ones(b_shape, device="cuda", dtype=torch.float32)
    m_indptr = torch.arange(num_groups + 1, device="cuda", dtype=torch.int32) * tokens
    out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
    return a, b, a_scale, b_scale, m_indptr, out


def _dequant_ref(a: torch.Tensor, b: torch.Tensor, tokens: int) -> torch.Tensor:
    chunks = []
    for expert in range(b.shape[0]):
        chunks.append(a[expert * tokens : (expert + 1) * tokens].float() @ b[expert].float().T)
    return torch.cat(chunks, dim=0)


def _shape_id(shape):
    e, t, n, k = shape
    return f"E{e}_t{t}_{n}x{k}"


@pytest.mark.parametrize("num_groups,tokens,n,k", _SHAPES, ids=[_shape_id(s) for s in _SHAPES])
@sm100_only
def test_matches_dequant(num_groups, tokens, n, k):
    torch.manual_seed(num_groups * n + k)
    a, b, a_scale, b_scale, m_indptr, out = _buffers(num_groups, tokens, n, k)
    grouped_fp8_gemm(a, b, a_scale, b_scale, m_indptr, out)
    torch.cuda.synchronize()
    ref = _dequant_ref(a, b, tokens)
    out_f = out.float().reshape(-1)
    ref_f = ref.reshape(-1)
    cosine = (out_f @ ref_f / (out_f.norm() * ref_f.norm())).item()
    assert cosine >= 0.999
    assert (out_f - ref_f).abs().max().item() / max(ref_f.std().item(), 1e-3) <= 0.05


def _expand_scales(a_scale, b_scale, k, mode):
    if mode == "MN":
        sa = a_scale.T.repeat_interleave(128, dim=1)[:, :k]
        sb = b_scale.permute(0, 2, 1)
    else:
        sa = a_scale.repeat_interleave(128, dim=1)[:, :k]
        sb = b_scale
    sb = sb.repeat_interleave(128, dim=1).repeat_interleave(128, dim=2)
    return sa, sb


def _scaled_ref(a, b, a_scale, b_scale, indptr, mode):
    k = a.shape[1]
    sa, sb = _expand_scales(a_scale, b_scale, k, mode)
    bounds = indptr.tolist()
    chunks = []
    for e in range(b.shape[0]):
        lo, hi = bounds[e], bounds[e + 1]
        chunks.append((a[lo:hi].float() * sa[lo:hi]) @ (b[e].float() * sb[e]).T)
    return torch.cat(chunks, dim=0)


def _assert_close(out, ref):
    out_f = out.float().reshape(-1)
    ref_f = ref.reshape(-1)
    cosine = (out_f @ ref_f / (out_f.norm() * ref_f.norm())).item()
    assert cosine >= 0.999
    assert (out_f - ref_f).abs().max().item() / max(ref_f.std().item(), 1e-3) <= 0.05


def _ragged_indptr(counts: list[int]) -> torch.Tensor:
    return torch.tensor(
        [0, *torch.tensor(counts).cumsum(0).tolist()], device="cuda", dtype=torch.int32
    )


@pytest.mark.parametrize("mode", ["MN", "K"])
@pytest.mark.parametrize("mma_sm", [1, 2])
@pytest.mark.parametrize("scheduler", ["static", "dynamic"])
@sm100_only
def test_ragged_groups_random_scales(mode, mma_sm, scheduler):
    """Uneven and empty groups, partial M tiles, non-unit scales."""
    torch.manual_seed(7)
    num_groups, n, k = 6, 512, 384
    counts = [20, 0, 300, 4, 0, 132]
    m = sum(counts)
    a = _randn_fp8(m, k)
    b = _randn_fp8(num_groups, n, k)
    a_shape, b_shape = grouped_fp8_gemm_scale_shapes(m, n, k, num_groups, scale_major_mode=mode)
    a_scale = torch.rand(a_shape, device="cuda") + 0.5
    b_scale = torch.rand(b_shape, device="cuda") + 0.5
    m_indptr = _ragged_indptr(counts)
    out = torch.zeros(m, n, device="cuda", dtype=torch.bfloat16)
    config = GroupedFp8GemmConfig(scale_major_mode=mode, mma_sm=mma_sm, scheduler=scheduler)
    grouped_fp8_gemm(a, b, a_scale, b_scale, m_indptr, out, config=config)
    torch.cuda.synchronize()
    _assert_close(out, _scaled_ref(a, b, a_scale, b_scale, m_indptr, mode))


@pytest.mark.parametrize("mode", ["MN", "K"])
@pytest.mark.parametrize("mma_sm", [1, 2])
@sm100_only
def test_full_k_acc_ragged_groups_k_constant_scales(mode, mma_sm):
    """Full-K chain with the scale applied once in the epilogue: a valid
    blockwise GEMM whenever the scales do not vary along K."""
    torch.manual_seed(11)
    num_groups, n, k = 6, 512, 384
    counts = [20, 0, 300, 4, 0, 132]
    m = sum(counts)
    a = _randn_fp8(m, k)
    b = _randn_fp8(num_groups, n, k)
    a_shape, b_shape = grouped_fp8_gemm_scale_shapes(m, n, k, num_groups, scale_major_mode=mode)
    row_scale = torch.rand(m, device="cuda") + 0.5
    col_scale = torch.rand(num_groups, n // 128, device="cuda") + 0.5
    if mode == "MN":
        a_scale = row_scale.reshape(1, m).expand(a_shape).contiguous()
        b_scale = col_scale.reshape(num_groups, 1, -1).expand(b_shape).contiguous()
    else:
        a_scale = row_scale.reshape(m, 1).expand(a_shape).contiguous()
        b_scale = col_scale.reshape(num_groups, -1, 1).expand(b_shape).contiguous()
    m_indptr = torch.tensor(
        [0, *torch.tensor(counts).cumsum(0).tolist()], device="cuda", dtype=torch.int32
    )
    out = torch.zeros(m, n, device="cuda", dtype=torch.bfloat16)
    config = GroupedFp8GemmConfig(scale_major_mode=mode, mma_sm=mma_sm, full_k_acc=True)
    grouped_fp8_gemm(a, b, a_scale, b_scale, m_indptr, out, config=config)
    torch.cuda.synchronize()
    _assert_close(out, _scaled_ref(a, b, a_scale, b_scale, m_indptr, mode))


_WIDE_CONFIGS = [
    # (mma_sm, tile_n, cluster_n)
    (2, 256, 2),
    (2, 256, 1),
    (1, 256, 1),
    (1, 128, 2),
    (2, 128, 2),
]


def _k_constant_scales(a_shape, b_shape, m, n, num_groups, mode):
    """Random 1x128 / 128x128 scales that are constant along K (full-K order)."""
    if mode == "MN":
        sa = (torch.rand(1, m, device="cuda") + 0.5).expand(a_shape)
        sb = (torch.rand(num_groups, 1, n // 128, device="cuda") + 0.5).expand(b_shape)
    else:
        sa = (torch.rand(m, 1, device="cuda") + 0.5).expand(a_shape)
        sb = (torch.rand(num_groups, n // 128, 1, device="cuda") + 0.5).expand(b_shape)
    return sa.contiguous(), sb.contiguous()


@pytest.mark.parametrize("mode", ["MN", "K"])
@pytest.mark.parametrize(
    "mma_sm,tile_n,cluster_n", _WIDE_CONFIGS, ids=[f"sm{s}_n{n}_c{c}" for s, n, c in _WIDE_CONFIGS]
)
@sm100_only
def test_wide_tile_ragged_groups(mode, mma_sm, tile_n, cluster_n):
    """256-wide tiles / N clusters: partial 256-row tiles, empty groups, odd n-tile counts.

    Per-column-block B scales differ inside one 256-wide tile, so a tile that
    applied a single block scale would fail the tolerance.
    """
    torch.manual_seed(11)
    num_groups, n, k = 7, 1280, 512
    counts = [20, 0, 300, 4, 0, 132, 260]
    m = sum(counts)
    a = _randn_fp8(m, k)
    b = _randn_fp8(num_groups, n, k)
    a_shape, b_shape = grouped_fp8_gemm_scale_shapes(m, n, k, num_groups, scale_major_mode=mode)
    a_scale, b_scale = _k_constant_scales(a_shape, b_shape, m, n, num_groups, mode)
    m_indptr = torch.tensor(
        [0, *torch.tensor(counts).cumsum(0).tolist()], device="cuda", dtype=torch.int32
    )
    out = torch.zeros(m, n, device="cuda", dtype=torch.bfloat16)
    config = GroupedFp8GemmConfig(
        scale_major_mode=mode,
        mma_sm=mma_sm,
        full_k_acc=True,
        tile_n=tile_n,
        cluster_n=cluster_n,
    )
    grouped_fp8_gemm(a, b, a_scale, b_scale, m_indptr, out, config=config)
    torch.cuda.synchronize()
    _assert_close(out, _scaled_ref(a, b, a_scale, b_scale, m_indptr, mode))


@sm100_only
def test_wide_tile_glm_shape():
    """GLM-5.2 gate+up at 256 tokens/expert on the 2-CTA 256x256 / (2,2) config."""
    torch.manual_seed(5)
    a, b, a_scale, b_scale, m_indptr, out = _buffers(32, 256, 4096, 6144)
    config = GroupedFp8GemmConfig(mma_sm=2, full_k_acc=True, tile_n=256, cluster_n=2)
    grouped_fp8_gemm(a, b, a_scale, b_scale, m_indptr, out, config=config)
    torch.cuda.synchronize()
    _assert_close(out, _dequant_ref(a, b, 256))


def test_config_rejects_wide_tile_without_full_k():
    with pytest.raises(ValueError, match="full_k_acc"):
        GroupedFp8GemmConfig(tile_n=256)


@pytest.mark.parametrize("mma_sm", [1, 2])
def test_config_rejects_dynamic_schedule_with_n_cluster(mma_sm):
    """The dynamic claim is shared across the M pair only."""
    with pytest.raises(ValueError, match="cluster_n=1"):
        GroupedFp8GemmConfig(scheduler="dynamic", cluster_n=2, mma_sm=mma_sm)
    with pytest.raises(ValueError, match="cluster_n=1"):
        GroupedGemmSm100(scale_major_k=False, mma_sm=mma_sm, scheduler="dynamic", cluster_n=2)


@pytest.mark.parametrize("knob", ["mma_sm", "tile_n", "cluster_n"])
@pytest.mark.parametrize("value", [True, 1.0, 128.0], ids=["bool", "float1", "float128"])
def test_config_rejects_non_int_modes(knob, value):
    """``True == 1`` and ``1.0 == 1``, but neither is a launch mode."""
    with pytest.raises(ValueError, match=knob):
        GroupedFp8GemmConfig(**{knob: value})


def test_auto_config_accepts_explicit_tile_knobs():
    config = GroupedFp8GemmConfig.auto(4096, 4, full_k_acc=True, mma_sm=1, tile_n=128)
    assert (config.mma_sm, config.tile_n) == (1, 128)


@pytest.mark.parametrize("bad", ["a", "b"])
def test_non_tensor_operands_raise_type_error(bad):
    a, b, a_scale, b_scale, m_indptr, out = _buffers(4, 4, 256, 256)
    operands = {"a": a, "b": b}
    operands[bad] = a.view(torch.uint8).cpu().numpy() if bad == "a" else [b]
    with pytest.raises(TypeError, match=f"{bad} must be a torch.Tensor"):
        grouped_fp8_gemm(operands["a"], operands["b"], a_scale, b_scale, m_indptr, out)


@pytest.mark.parametrize("mma_sm", [1, 2])
@pytest.mark.parametrize("group_order", ["index", "lpt"])
@sm100_only
def test_dynamic_schedule_matches_static_bitwise(mma_sm, group_order):
    """More tiles than clusters (the dynamic fetch path and the counter reset
    are exercised), skewed groups spanning several 31-group scan windows:
    every schedule and visiting order yields the same bits, launch after launch."""
    torch.manual_seed(11)
    counts = [4 * ((7 * g) % 23 + 1) for g in range(40)]
    counts[3], counts[17], counts[30], counts[38] = 1000, 0, 520, 0
    num_groups, n, k = len(counts), 2048, 512
    m = sum(counts)
    a = _randn_fp8(m, k)
    b = _randn_fp8(num_groups, n, k)
    a_shape, b_shape = grouped_fp8_gemm_scale_shapes(m, n, k, num_groups)
    a_scale = torch.ones(a_shape, device="cuda")
    b_scale = torch.ones(b_shape, device="cuda")
    m_indptr = _ragged_indptr(counts)
    order = None
    if group_order == "lpt":
        order = torch.tensor(
            sorted(range(num_groups), key=lambda g: -counts[g]), dtype=torch.int32, device="cuda"
        )
    out = torch.zeros(m, n, device="cuda", dtype=torch.bfloat16)
    static = GroupedFp8GemmConfig(mma_sm=mma_sm, full_k_acc=True)
    grouped_fp8_gemm(a, b, a_scale, b_scale, m_indptr, out, config=static)
    torch.cuda.synchronize()
    expected = out.clone()
    _assert_close(expected, _scaled_ref(a, b, a_scale, b_scale, m_indptr, "MN"))
    dynamic = GroupedFp8GemmConfig(mma_sm=mma_sm, full_k_acc=True, scheduler="dynamic")
    for config in (static, dynamic):
        for _ in range(4):
            out.zero_()
            grouped_fp8_gemm(
                a, b, a_scale, b_scale, m_indptr, out, config=config, group_order=order
            )
            torch.cuda.synchronize()
            assert torch.equal(out, expected)


@pytest.mark.parametrize("scheduler", ["static", "dynamic"])
@sm100_only
def test_cuda_graph_capture(scheduler):
    # Enough tiles (32 groups x 32 n-tiles) that the dynamic fetch runs.
    a, b, a_scale, b_scale, m_indptr, out = _buffers(32, 4, 4096, 256)
    config = GroupedFp8GemmConfig(scheduler=scheduler)
    grouped_fp8_gemm(a, b, a_scale, b_scale, m_indptr, out, config=config)
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
        grouped_fp8_gemm(a, b, a_scale, b_scale, m_indptr, out, config=config)
    torch.cuda.synchronize()
    a.copy_(_randn_fp8(*a.shape))
    for _ in range(3):
        out.zero_()
        graph.replay()
        torch.cuda.synchronize()
        _assert_close(out, _dequant_ref(a, b, 4))


def _expert_ref(a, b, m_indptr_ok, out):
    """Reference for a well-formed ``m_indptr`` (unit scales)."""
    ref = torch.zeros_like(out, dtype=torch.float32)
    bounds = m_indptr_ok.tolist()
    for e in range(b.shape[0]):
        lo, hi = bounds[e], bounds[e + 1]
        if hi > lo:
            ref[lo:hi] = a[lo:hi].float() @ b[e].float().T
    return ref


@pytest.mark.parametrize("scheduler", ["static", "dynamic"])
@sm100_only
def test_malformed_indptr_is_clamped_in_kernel(scheduler):
    """Malformed and decreasing group bounds clamp to the expected table;
    valid output rows match the reference."""
    a, b, a_scale, b_scale, _, out = _buffers(4, 4, 256, 256)
    bad = torch.tensor([0, 4, 4, 20, 16], dtype=torch.int32, device="cuda")
    good = torch.tensor([0, 4, 4, 16, 16], dtype=torch.int32, device="cuda")
    out.zero_()
    grouped_fp8_gemm(
        a, b, a_scale, b_scale, bad, out, config=GroupedFp8GemmConfig(scheduler=scheduler)
    )
    torch.cuda.synchronize()
    _assert_close(out, _expert_ref(a, b, good, out))


@sm100_only
def test_malformed_group_order_is_clamped_in_kernel():
    """Out-of-range and duplicate group-order entries clamp in-kernel;
    skipped groups remain untouched."""
    a, b, a_scale, b_scale, m_indptr, out = _buffers(4, 4, 256, 256)
    order = torch.tensor([0, 1, 1, 7], dtype=torch.int32, device="cuda")
    out.zero_()
    grouped_fp8_gemm(a, b, a_scale, b_scale, m_indptr, out, group_order=order)
    torch.cuda.synchronize()
    ref = _expert_ref(a, b, m_indptr, out)
    ref[8:12] = 0  # group 2 is never visited
    _assert_close(out[:8], ref[:8])
    assert torch.equal(out[8:12], torch.zeros_like(out[8:12]))
    _assert_close(out[12:], ref[12:])


@sm100_only
def test_rebinds_output_buffer():
    a, b, a_scale, b_scale, m_indptr, out = _buffers(4, 4, 256, 256)
    grouped_fp8_gemm(a, b, a_scale, b_scale, m_indptr, out)
    torch.cuda.synchronize()
    first = out.clone()
    a.copy_(_randn_fp8(*a.shape))
    grouped_fp8_gemm(a, b, a_scale, b_scale, m_indptr, out)
    torch.cuda.synchronize()
    assert not torch.equal(first, out)


def _ragged_inputs(counts, n, k, mode, *, sentinel_tail: int = 0):
    """Operands for arbitrary (unaligned) group boundaries.

    ``sentinel_tail`` extra rows are allocated behind ``a`` and ``out`` (the
    kernel sees ``[:m]`` views): they hold garbage / a sentinel that must be
    neither read into valid rows nor written.
    """
    num_groups = len(counts)
    m = sum(counts)
    a_big = _randn_fp8(m + sentinel_tail, k)
    a = a_big[:m]
    b = _randn_fp8(num_groups, n, k)
    a_shape, b_shape = grouped_fp8_gemm_scale_shapes(m, n, k, num_groups, scale_major_mode=mode)
    a_scale = torch.rand(a_shape, device="cuda") + 0.5
    b_scale = torch.rand(b_shape, device="cuda") + 0.5
    m_indptr = torch.tensor(
        [0, *torch.tensor(counts).cumsum(0).tolist()], device="cuda", dtype=torch.int32
    )
    out_big = torch.full((m + sentinel_tail, n), 12345.0, device="cuda", dtype=torch.bfloat16)
    return a, b, a_scale, b_scale, m_indptr, out_big


# Group boundaries that are not multiples of 4: rows 0..5 | 5..131 | 131..131 |
# 131..300, then a lattice of tiny groups so that every CTA tile overhangs
# into several later groups (a store that is not clipped to its group would
# race with theirs).
_UNALIGNED_COUNTS = [[5, 126, 0, 169], [3] * 64 + [1, 130, 2]]


@pytest.mark.parametrize("counts", _UNALIGNED_COUNTS, ids=["ragged-4", "tiny-67"])
@pytest.mark.parametrize("mode", ["MN", "K"])
@pytest.mark.parametrize("mma_sm", [1, 2])
@pytest.mark.parametrize("full_k_acc", [False, True], ids=["blockwise", "fullk"])
@sm100_only
def test_unaligned_group_boundaries(counts, mode, mma_sm, full_k_acc):
    """Any ``m_indptr`` is accepted; every group's rows are exact and rows
    behind the last group (the sentinel tail) are never written."""
    torch.manual_seed(len(counts) + mma_sm)
    n, k = 256, 384
    a, b, a_scale, b_scale, m_indptr, out_big = _ragged_inputs(
        counts, n, k, mode, sentinel_tail=128
    )
    m = sum(counts)
    if full_k_acc:
        # The full-K chain applies the K-block-0 scales to the whole product.
        k_dim_a, k_dim_b = (1, 2) if mode == "K" else (0, 1)
        a_scale.copy_(a_scale.narrow(k_dim_a, 0, 1).clone().expand_as(a_scale))
        b_scale.copy_(b_scale.narrow(k_dim_b, 0, 1).clone().expand_as(b_scale))
    config = GroupedFp8GemmConfig(scale_major_mode=mode, mma_sm=mma_sm, full_k_acc=full_k_acc)
    grouped_fp8_gemm(a, b, a_scale, b_scale, m_indptr, out_big[:m], config=config)
    torch.cuda.synchronize()
    _assert_close(out_big[:m], _scaled_ref(a, b, a_scale, b_scale, m_indptr, mode))
    assert torch.all(out_big[m:] == 12345.0), "rows behind the last group were written"


@sm100_only
def test_accepts_indptr_not_multiple_of_four():
    a, b, a_scale, b_scale, m_indptr, out = _buffers(4, 4, 256, 256)
    m_indptr[1] = 2
    grouped_fp8_gemm(a, b, a_scale, b_scale, m_indptr, out)
    torch.cuda.synchronize()
    _assert_close(out, _scaled_ref(a, b, a_scale, b_scale, m_indptr, "MN"))


@sm100_only
def test_rejects_wrong_scale_shape():
    a, b, a_scale, b_scale, m_indptr, out = _buffers(4, 4, 256, 256)
    with pytest.raises(ValueError, match="a_scale"):
        grouped_fp8_gemm(a, b, a_scale[:1], b_scale, m_indptr, out)


def _expanded(rows: int, *tail: int) -> torch.Tensor:
    """A zero-stride view with a huge leading extent: shape-only geometry
    checks see the advertised size without the allocation."""
    return torch.empty(1, *tail, dtype=torch.float8_e4m3fn, device="cuda").expand(rows, *tail)


def test_rejects_geometry_past_the_descriptor_and_int32_limits():
    """The ragged TMA views address ``MAX_GROUPED_ROWS`` rows; a group past
    them would load zero fill at negative coordinates. The bound, and the
    kernel's signed 32-bit shape arithmetic, are enforced on the host-known
    shapes before launch (the boundary itself is admitted)."""
    from pearl_gemm.grouped_fp8_gemm._host import _problem_dims

    k = 64
    b = _expanded(1, 128, k)
    with pytest.raises(ValueError, match="exceeds the .* rows the ragged TMA views address"):
        grouped_fp8_gemm(_expanded(MAX_GROUPED_ROWS + 4, k), b, b, b, b, b)
    assert _problem_dims(_expanded(MAX_GROUPED_ROWS, k), b) == (MAX_GROUPED_ROWS, 1, 128, k)
    with pytest.raises(ValueError, match="signed 32-bit"):
        grouped_fp8_gemm(_expanded(4, k), _expanded(2**15, 2**16, k), b, b, b, b)
    assert _problem_dims(_expanded(4, k), _expanded(2**15, 2**16 - 8, k))[1:] == (
        2**15,
        2**16 - 8,
        k,
    )
    # The plain GEMM keeps its empty-problem policy: cum_m = 0 is admitted.
    assert _problem_dims(_expanded(0, k), b)[0] == 0


@sm100_only
def test_rejects_tile_schedules_past_the_int32_scheduler():
    """Individually admissible extents whose derived tile schedule (rounded
    ``n``, tile-count bound, terminal-index headroom) would overflow the
    kernel's ``Int32`` scheduler are refused on the host-known shapes for the
    selected config."""
    k = 128

    def stacked_b(num_groups: int, n: int) -> torch.Tensor:
        # Zero-stride along groups and columns: the shape without the bytes.
        return torch.empty(1, 1, k, dtype=torch.float8_e4m3fn, device="cuda").expand(
            num_groups, n, k
        )

    placeholders = [_expanded(1, 1)] * 4
    # One group of 2**22 rows against 2**23 columns on the 128x128 tile:
    # 2**15 M tiles x 2**16 N tiles is a 2**31 tile product.
    with pytest.raises(ValueError, match="signed 32-bit scheduler"):
        grouped_fp8_gemm(
            _expanded(2**22, k),
            stacked_b(1, 2**23),
            *placeholders,
            config=GroupedFp8GemmConfig(mma_sm=1, tile_n=128),
        )
    # Rounding n up to the 256-column tile overflows although n itself fits.
    with pytest.raises(ValueError, match="rounded up to the 256-column cluster tile"):
        grouped_fp8_gemm(
            _expanded(4, k),
            stacked_b(1, 2**31 - 128),
            *placeholders,
            config=GroupedFp8GemmConfig(mma_sm=1, tile_n=256, full_k_acc=True),
        )


def test_scheduler_geometry_bounds():
    """The ``Int32`` tile-schedule bound on host-known shapes; the boundaries
    themselves are admitted."""
    from pearl_gemm.grouped_fp8_gemm._host import _INT32_MAX, _require_scheduler_geometry

    tile = GroupedFp8GemmConfig(mma_sm=1, tile_n=256, full_k_acc=True).cluster_tile_shape_mn
    assert tile == (128, 256)
    _require_scheduler_geometry(4, 1, _INT32_MAX - 255, tile)
    with pytest.raises(ValueError, match="rounded up"):
        _require_scheduler_geometry(4, 1, _INT32_MAX - 254, tile)
    # The tile-count bound counts every group's round-up: 2**14 M tiles
    # times 2**16 - 1 N tiles fits with one group, not with two.
    tile = GroupedFp8GemmConfig(mma_sm=1, tile_n=128).cluster_tile_shape_mn
    _require_scheduler_geometry(2**21 - 128, 1, 2**23 - 128, tile)
    with pytest.raises(ValueError, match="no headroom"):
        _require_scheduler_geometry(2**21 - 128, 2, 2**23 - 128, tile)
    # The 2-CTA 256-wide tile with an N cluster halves the tile count on both axes.
    wide = GroupedFp8GemmConfig(mma_sm=2, tile_n=256, cluster_n=2, full_k_acc=True)
    assert wide.cluster_tile_shape_mn == (256, 512)
    _require_scheduler_geometry(2**22, 1, 2**23, wide.cluster_tile_shape_mn)


def test_config_rejects_bad_mma_sm():
    with pytest.raises(ValueError, match="mma_sm"):
        GroupedFp8GemmConfig(mma_sm=3)


def test_config_rejects_bad_scheduler():
    with pytest.raises(ValueError, match="scheduler"):
        GroupedFp8GemmConfig(scheduler="clc")


def test_auto_config_selects_the_wide_tile_by_average_tokens_per_group():
    """``auto`` decides from ``cum_m / num_groups`` only, never picks the
    numerics-changing full-K order by itself, and keeps explicit knobs."""
    e = 32
    below, at = WIDE_TILE_MIN_TOKENS_PER_GROUP * e - 1, WIDE_TILE_MIN_TOKENS_PER_GROUP * e
    assert GroupedFp8GemmConfig.auto(at, e) == GroupedFp8GemmConfig()
    assert GroupedFp8GemmConfig.auto(below, e, full_k_acc=True) == GroupedFp8GemmConfig(
        full_k_acc=True
    )
    assert GroupedFp8GemmConfig.auto(at, e, full_k_acc=True) == GroupedFp8GemmConfig(
        full_k_acc=True, mma_sm=2, tile_n=256
    )
    assert GroupedFp8GemmConfig.auto(
        at, e, full_k_acc=True, tile_n=128, scheduler="dynamic"
    ) == GroupedFp8GemmConfig(full_k_acc=True, mma_sm=2, tile_n=128, scheduler="dynamic")


@sm100_only
def test_default_config_is_auto():
    """``config=None`` launches (the blockwise 128x128 tile at any size)."""
    a, b, a_scale, b_scale, m_indptr, out = _buffers(2, 256, 256, 256)
    grouped_fp8_gemm(a, b, a_scale, b_scale, m_indptr, out, config=None)
    torch.cuda.synchronize()
    _assert_close(out, _dequant_ref(a, b, 256))


def test_non_sm100_launch_is_rejected():
    if _ON_SM100:
        pytest.skip("SM100 runs the kernel")
    a, b, a_scale, b_scale, m_indptr, out = _buffers(4, 4, 256, 256)
    with pytest.raises(ValueError, match="grouped_fp8_gemm requires SM100"):
        grouped_fp8_gemm(a, b, a_scale, b_scale, m_indptr, out)


def test_scale_shapes_k_major():
    a_shape, b_shape = grouped_fp8_gemm_scale_shapes(16, 256, 256, 4, scale_major_mode="K")
    assert a_shape == (16, 2)
    assert b_shape == (4, 2, 2)
