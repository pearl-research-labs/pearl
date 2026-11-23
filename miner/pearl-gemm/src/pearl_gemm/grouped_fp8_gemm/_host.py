"""Caller-owned host launch for the SM100 FP8 grouped GEMM (CuTe DSL)."""

from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
import torch
from quack.cute_dsl_utils import get_max_active_clusters

from .._utils._arch import Arch, require_arch
from .._utils._compile import make_fake_stream, make_fake_tensor, single_flight_compile
from .._utils._stream import get_stream
from .._utils._validation import require_disjoint_writes, require_tensor
from ._kernel import (
    _RAGGED_ROWS,
    CTA_TILE_M,
    SCHED_COUNTER_WORDS,
    SCHEDULERS,
    GroupedGemmSm100,
)

SCALE_GRANULARITY_MNK = (1, 128, 128)
# The kernel's shape and coordinate arithmetic is signed 32-bit (``Int32``
# ``n``, ``k``, ``m_indptr`` values, the stacked B row ``group * n`` and the
# persistent scheduler's tile counts, see ``_require_scheduler_geometry``),
# and the ragged A / C TMA views map global rows ``[0, _RAGGED_ROWS)`` only: a
# group ending past that extent would read zero-filled rows at negative
# descriptor coordinates instead of its operand. Both public wrappers reject
# such geometry before launch; the bound is checked on the host-known shapes,
# never by reading ``m_indptr``.
MAX_GROUPED_ROWS = _RAGGED_ROWS
_INT32_MAX = 2**31 - 1
_BLOCK_N = SCALE_GRANULARITY_MNK[1]
_BLOCK_K = SCALE_GRANULARITY_MNK[2]
# Average tokens per group (``cum_m / num_groups``) from which the automatic
# tile selection prefers the 2-CTA 256-wide full-K tile over the 1-SM 128x128
# tile: the wide tile amortizes its per-tile cost only once a group spans a
# 256-row cluster tile and change.
WIDE_TILE_MIN_TOKENS_PER_GROUP = 160


@dataclass(frozen=True)
class GroupedFp8GemmConfig:
    """Launch knobs for the SM100 grouped FP8 path."""

    scale_major_mode: str = "MN"
    mma_sm: int = 1
    # Chain all of K into one TMEM accumulator (mining accumulation order)
    # instead of restarting per 128-wide K tile and rescaling the partials.
    # The kernel then applies only the K-block-0 scales, so the result is the
    # blockwise product only when every scale is constant along K.
    full_k_acc: bool = False
    # Persistent tile schedule: ``static`` (cluster ``c`` takes every
    # ``num_clusters``-th tile) or ``dynamic`` (tiles past the first are
    # claimed from a per-device global counter that the kernel resets, so
    # launches stay CUDA-graph replayable). The result does not depend on
    # which cluster computes a tile; concurrent launches on one device must
    # not share the counter, i.e. must be stream-ordered.
    scheduler: str = "static"
    # Per-CTA output tile width. 256 (two 256-column TMEM accumulator stages
    # read directly by the epilogue; requires ``full_k_acc``) trades small-m
    # efficiency for MMA throughput at large tokens/expert.
    tile_n: int = 128
    # Cluster extent along N: CTAs sharing an M tile multicast A.
    cluster_n: int = 1

    def __post_init__(self) -> None:
        if self.scale_major_mode not in ("MN", "K"):
            raise ValueError("scale_major_mode must be 'MN' or 'K'")
        # ``type is int``: ``True`` and ``1.0`` compare equal to 1 but are
        # not launch modes.
        if type(self.mma_sm) is not int or self.mma_sm not in (1, 2):
            raise ValueError("mma_sm must be 1 or 2")
        if self.scheduler not in SCHEDULERS:
            raise ValueError(f"scheduler must be one of {SCHEDULERS}")
        if type(self.tile_n) is not int or self.tile_n not in (128, 256):
            raise ValueError("tile_n must be 128 or 256")
        if type(self.cluster_n) is not int or self.cluster_n not in (1, 2):
            raise ValueError("cluster_n must be 1 or 2")
        # The dynamic scheduler shares one claim across the M pair only; the N
        # peers of a cluster would claim independently and diverge.
        if self.scheduler == "dynamic" and self.cluster_n != 1:
            raise ValueError("dynamic scheduling requires cluster_n=1")
        if self.tile_n != 128 and not self.full_k_acc:
            raise ValueError("tile_n=256 requires full_k_acc=True")

    @property
    def cluster_size(self) -> int:
        return self.mma_sm * self.cluster_n

    @property
    def cluster_tile_shape_mn(self) -> tuple[int, int]:
        """Output tile of one cluster, as ``GroupedGemmSm100`` derives it."""
        return CTA_TILE_M * self.mma_sm, self.tile_n * self.cluster_n

    @classmethod
    def auto(cls, cum_m: int, num_groups: int, **knobs) -> "GroupedFp8GemmConfig":
        """Tile shape from host-known quantities only (no device sync).

        With ``full_k_acc=True`` (the only order the wide tile supports) and
        an average of at least ``WIDE_TILE_MIN_TOKENS_PER_GROUP`` tokens per
        group, the 2-CTA 256-wide tile ``(mma_sm=2, tile_n=256)``; otherwise
        the 1-SM 128x128 tile. Explicit ``mma_sm`` / ``tile_n`` knobs win.
        """
        wide = (
            knobs.get("full_k_acc", False) and cum_m >= WIDE_TILE_MIN_TOKENS_PER_GROUP * num_groups
        )
        knobs.setdefault("mma_sm", 2 if wide else 1)
        knobs.setdefault("tile_n", 256 if wide else 128)
        return cls(**knobs)


def grouped_fp8_gemm_scale_shapes(
    m: int,
    n: int,
    k: int,
    num_groups: int,
    *,
    scale_major_mode: str = "MN",
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """``(a_scale, b_scale)`` shapes for unit 1×128 / 128×128 scales."""
    if type(m) is not int or type(n) is not int or type(k) is not int:
        raise ValueError("m, n, k must be ints")
    if type(num_groups) is not int or num_groups <= 0:
        raise ValueError("num_groups must be a positive int")
    if m < 0 or n <= 0 or k <= 0:
        raise ValueError("m must be >= 0 and n, k must be > 0")
    if n % _BLOCK_N or k % _BLOCK_K:
        raise ValueError(f"n must be a multiple of {_BLOCK_N} and k of {_BLOCK_K}")
    if scale_major_mode == "MN":
        return (k // _BLOCK_K, m), (num_groups, k // _BLOCK_K, n // _BLOCK_N)
    if scale_major_mode == "K":
        return (m, k // _BLOCK_K), (num_groups, n // _BLOCK_N, k // _BLOCK_K)
    raise ValueError("scale_major_mode must be 'MN' or 'K'")


# Dynamic-scheduler counters, zero between (stream-ordered) launches: the
# kernel's last cluster resets them, so graph replays need no host memset.
_sched_counters: dict[int | None, torch.Tensor] = {}


def _sched_counter(device: torch.device) -> torch.Tensor:
    counter = _sched_counters.get(device.index)
    if counter is None:
        counter = torch.zeros(SCHED_COUNTER_WORDS, dtype=torch.int32, device=device)
        _sched_counters[device.index] = counter
    return counter


def _sym(divisibility: int = 1):
    return cute.sym_int64(divisibility=divisibility)


def _validate_group_order(
    group_order: torch.Tensor | None, num_groups: int, device: torch.device
) -> None:
    """Shape / dtype / device only: the values are the kernel's to clamp
    (``_clamp_group``), never the host's to read -- see ``_validate_indptr``."""
    if group_order is None:
        return
    require_tensor(
        "group_order",
        group_order,
        dtype=torch.int32,
        shape=(num_groups,),
        device=device,
        alignment=4,
    )


@single_flight_compile
def _compile_variant(
    capability: tuple,
    scale_major_k: bool,
    mma_sm: int,
    full_k_acc: bool,
    scheduler: str,
    has_group_order: bool,
    tile_n: int,
    cluster_n: int,
    max_active_clusters: int,
):
    kernel = GroupedGemmSm100(
        scale_major_k=scale_major_k,
        mma_sm=mma_sm,
        full_k_acc=full_k_acc,
        scheduler=scheduler,
        has_group_order=has_group_order,
        tile_n=tile_n,
        cluster_n=cluster_n,
    )
    a_fake = make_fake_tensor(
        cutlass.Float8E4M3FN, (_sym(), _sym(16)), leading_dim=1, divisibility=16
    )
    b_fake = make_fake_tensor(
        cutlass.Float8E4M3FN, (_sym(), _sym(16)), leading_dim=1, divisibility=16
    )
    c_fake = make_fake_tensor(cutlass.BFloat16, (_sym(), _sym(8)), leading_dim=1, divisibility=8)
    sfa_fake = make_fake_tensor(cutlass.Float32, (_sym(), _sym()), leading_dim=1, divisibility=1)
    sfb_fake = make_fake_tensor(
        cutlass.Float32, (_sym(), _sym(), _sym()), leading_dim=2, divisibility=1
    )
    indptr_fake = make_fake_tensor(cutlass.Int32, (_sym(),), leading_dim=0, divisibility=1)
    sched_fake = (
        make_fake_tensor(cutlass.Int32, (SCHED_COUNTER_WORDS,), leading_dim=0, divisibility=1)
        if scheduler == "dynamic"
        else None
    )
    order_fake = (
        make_fake_tensor(cutlass.Int32, (_sym(),), leading_dim=0, divisibility=1)
        if has_group_order
        else None
    )
    return cute.compile(
        kernel,
        a_fake,
        b_fake,
        c_fake,
        sfa_fake,
        sfb_fake,
        indptr_fake,
        sched_fake,
        order_fake,
        max_active_clusters,
        make_fake_stream(),
        options="--enable-tvm-ffi",
    )


def _launch(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    m_indptr: torch.Tensor,
    out: torch.Tensor,
    num_groups: int,
    n: int,
    k: int,
    config: GroupedFp8GemmConfig,
    capability: tuple,
    group_order: torch.Tensor | None,
) -> None:
    device = a.device
    max_active_clusters = get_max_active_clusters(config.cluster_size)
    compiled = _compile_variant(
        capability,
        config.scale_major_mode == "K",
        config.mma_sm,
        config.full_k_acc,
        config.scheduler,
        group_order is not None,
        config.tile_n,
        config.cluster_n,
        max_active_clusters,
    )
    compiled(
        a,
        b.view(num_groups * n, k),
        out,
        a_scale,
        b_scale,
        m_indptr,
        _sched_counter(device) if config.scheduler == "dynamic" else None,
        group_order,
        get_stream(device.index or 0),
    )


def _problem_dims(a: torch.Tensor, b: torch.Tensor) -> tuple[int, int, int, int]:
    # The shapes below size every other operand's ``require_tensor`` check, so
    # the two operands they come from get the same stable TypeError first.
    for name, tensor in (("a", a), ("b", b)):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
    if a.ndim != 2:
        raise ValueError("a must be 2D (cum_m, k)")
    if b.ndim != 3:
        raise ValueError("b must be 3D (num_groups, n, k)")
    cum_m, k = a.shape
    num_groups, n, k_b = b.shape
    if k != k_b:
        raise ValueError(f"K mismatch: a has {k}, b has {k_b}")
    if n % 8:
        raise ValueError(f"n must be a multiple of 8, got {n}")
    if k % 16:
        raise ValueError(f"k must be a multiple of 16, got {k}")
    if cum_m > MAX_GROUPED_ROWS:
        raise ValueError(
            f"cum_m={cum_m} exceeds the {MAX_GROUPED_ROWS} rows the ragged TMA views address"
        )
    if k > _INT32_MAX or num_groups * n > _INT32_MAX:
        raise ValueError(
            f"k={k} and num_groups * n={num_groups * n} must fit the kernel's signed 32-bit "
            "shape arithmetic"
        )
    return cum_m, num_groups, n, k


def _require_scheduler_geometry(
    cum_m: int, num_groups: int, n: int, cluster_tile_shape_mn: tuple[int, int]
) -> None:
    """Reject geometry whose derived tile schedule overflows ``Int32``.

    The persistent scheduler counts tiles in signed 32-bit: ``n`` rounded up
    to the cluster tile, each group's ``m_tiles * n_tiles``, their prefix
    sums across groups, and a linear tile index that runs one cluster stride
    past the last tile (static advance or dynamic fetch) before the terminal
    ``linear_idx + 1``. Host-known shapes bound all of it without reading
    ``m_indptr``: a well-formed table has at most ``tiles_bound`` tiles (each
    group rounds up at most once) and ``num_clusters <= tiles_bound``, so
    ``2 * tiles_bound`` covers every intermediate. The wider launch-side
    tensor extents do not widen this arithmetic; past the bound the kernel
    would schedule the wrong tiles (rows stay clamped, never out of bounds).
    """
    cluster_tile_m, cluster_tile_n = cluster_tile_shape_mn
    n_rounded = n + cluster_tile_n - 1
    if n_rounded > _INT32_MAX:
        raise ValueError(
            f"n={n} rounded up to the {cluster_tile_n}-column cluster tile must fit the kernel's "
            "signed 32-bit scheduler arithmetic"
        )
    n_tiles = n_rounded // cluster_tile_n
    m_tiles_bound = (cum_m + cluster_tile_m - 1) // cluster_tile_m + num_groups
    tiles_bound = m_tiles_bound * n_tiles
    if 2 * tiles_bound > _INT32_MAX:
        raise ValueError(
            f"tile count bound {tiles_bound} (cum_m={cum_m}, num_groups={num_groups}, n={n}, "
            f"cluster tile {cluster_tile_m}x{cluster_tile_n}) leaves the kernel's signed 32-bit "
            "scheduler no headroom for its terminal tile index"
        )


def _validate_indptr(m_indptr: torch.Tensor, num_groups: int, device: torch.device) -> None:
    """Check shape, dtype and device only.

    The values are not read on the host: ``m_indptr`` comes from the caller's
    routing kernels moments before this launch, so a ``.cpu()`` here would
    stall the stream once per call. Memory safety is the kernel's
    (``GroupedGemmSm100._clamp_rows`` clamps every group into ``[0, cum_m)``);
    a malformed table yields unspecified rows, never an out-of-bounds access.
    """
    require_tensor(
        "m_indptr",
        m_indptr,
        dtype=torch.int32,
        shape=(num_groups + 1,),
        device=device,
        alignment=4,
    )


def grouped_fp8_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    m_indptr: torch.Tensor,
    out: torch.Tensor,
    *,
    config: GroupedFp8GemmConfig | None = None,
    group_order: torch.Tensor | None = None,
) -> None:
    """``out[start:end] = a[start:end] @ b[e].T`` with 1x128 / 128x128 FP32 scales.

    Contract. Callers own every buffer. ``m_indptr`` holds exclusive prefix
    sums of tokens per group (a *group* is one expert's rows); no alignment
    is required, and only rows inside a group are written. ``out`` must not
    share bytes with any other operand. Every operand, ``m_indptr`` included,
    is read asynchronously on the current stream: do not modify it on any
    stream until the launch completes.

    No host sync. The host never reads ``m_indptr``'s values. The kernel
    clamps each group into ``[0, cum_m)``, so a malformed table yields
    unspecified rows, never an out-of-bounds access. ``cum_m`` is bounded by
    ``MAX_GROUPED_ROWS`` (the row extent of the ragged TMA views) and
    ``num_groups * n``, ``k`` and the selected config's tile schedule
    (``_require_scheduler_geometry``) by the kernel's signed 32-bit
    arithmetic; larger geometry is rejected here.

    ``group_order`` (optional ``(num_groups,)`` int32 permutation) sets the
    order the persistent schedule visits groups, for example largest first.

    ``config=None`` means ``GroupedFp8GemmConfig.auto(cum_m, num_groups)``:
    the blockwise order on the 1-SM 128x128 tile. The wide tile needs
    ``full_k_acc``, which changes numerics, so it is never chosen implicitly.

    ``full_k_acc=True`` is a caller contract, not a checked property: the
    kernel applies only the K-block-0 scales, so the result equals the
    blockwise product only when every scale is constant along K (the
    mining and per-tensor/per-row schemes). The scale tensors keep the
    blockwise shape and are not inspected -- a value check would be a
    device sync on every launch -- so K-varying scales under ``full_k_acc``
    are silently wrong by construction. Callers with blockwise scales use
    the default order.
    """
    cum_m, num_groups, n, k = _problem_dims(a, b)
    device = a.device
    require_arch("grouped_fp8_gemm", device, Arch.SM100)
    capability = torch.cuda.get_device_capability(device)
    if config is None:
        config = GroupedFp8GemmConfig.auto(cum_m, num_groups)
    _require_scheduler_geometry(cum_m, num_groups, n, config.cluster_tile_shape_mn)
    a_scale_shape, b_scale_shape = grouped_fp8_gemm_scale_shapes(
        cum_m, n, k, num_groups, scale_major_mode=config.scale_major_mode
    )
    require_tensor("a", a, dtype=torch.float8_e4m3fn, shape=(cum_m, k), device=device, alignment=16)
    require_tensor(
        "b",
        b,
        dtype=torch.float8_e4m3fn,
        shape=(num_groups, n, k),
        device=device,
        alignment=16,
    )
    require_tensor(
        "a_scale",
        a_scale,
        dtype=torch.float32,
        shape=a_scale_shape,
        device=device,
        alignment=16,
    )
    require_tensor(
        "b_scale",
        b_scale,
        dtype=torch.float32,
        shape=b_scale_shape,
        device=device,
        alignment=16,
    )
    require_tensor(
        "out",
        out,
        dtype=torch.bfloat16,
        shape=(cum_m, n),
        device=device,
        alignment=16,
    )
    _validate_indptr(m_indptr, num_groups, device)
    _validate_group_order(group_order, num_groups, device)
    require_disjoint_writes(
        [("out", out)],
        [
            ("a", a),
            ("b", b),
            ("a_scale", a_scale),
            ("b_scale", b_scale),
            ("m_indptr", m_indptr),
            ("group_order", group_order),
        ],
    )
    _launch(
        a, b, a_scale, b_scale, m_indptr, out, num_groups, n, k, config, capability, group_order
    )
