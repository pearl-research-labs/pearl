"""Caller-owned host launch for the MoE mining grouped GEMM (CuTe DSL).

The grouped counterpart of ``mixed_gemm``: one launch runs every expert's
``A'[rows_e] @ B'[e].T`` with the mining fold, keyed BLAKE3 lottery, first-hit
publish, the BF16 peel and the per-row / per-column unscale. Operands are the
protocol's MoE inputs with A rows permuted so each expert's tokens are
contiguous and in ascending token order, so a lottery tile's rows are the
expert's routed tokens ``4r .. 4r + 3``. SM100 runs ``GroupedGemmSm100``,
SM90 / SM120 the grouped mode of their fused kernel.
"""

from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32
from quack.cute_dsl_utils import get_max_active_clusters

from .._utils._arch import Arch, arch_of, require_arch
from .._utils._compile import make_fake_stream, make_fake_tensor, single_flight_compile
from .._utils._stream import get_stream
from .._utils._validation import require_disjoint_writes, require_tensor
from ..grouped_fp8_gemm._host import (
    WIDE_TILE_MIN_TOKENS_PER_GROUP,
    _problem_dims,
    _require_scheduler_geometry,
    _sched_counter,
    _sym,
    _validate_group_order,
    _validate_indptr,
)
from ..grouped_fp8_gemm._kernel import (
    CTA_TILE_M,
    SCHED_COUNTER_WORDS,
    SCHEDULERS,
    GroupedGemmSm100,
)
from ..mixed_gemm._kernel_register_acc import LTILE_ROWS
from ..mixed_gemm._lottery import (
    DEFAULT_LTILE_COLS,
    DEFAULT_LTILE_ROWS,
    R2,
    SUPPORTED_LTILE_COLS,
    SUPPORTED_LTILE_ROWS,
)
from ..pow import HIT_PAYLOAD_K_ALIGN, HitSignal
from ..protocol_constants import BLOCK_SCALE_GROUP
from ._kernel import _GroupedFusedGemmSm90, _GroupedFusedGemmSm120

_SUPPORTED_ARCHS = (Arch.SM90, Arch.SM100, Arch.SM120)


def supports_grouped_mixed_gemm(
    device: torch.device | int | tuple[int, int] | None = None,
) -> bool:
    """Whether ``device`` (the current one when omitted; a ``(major, minor)``
    capability is also accepted) has the MoE mining grouped GEMM: SM100
    (``GroupedGemmSm100``) and SM90 / SM120 (their grouped fused kernel).
    Unknown majors return false, so expert layers stay on their unquantized
    path."""
    try:
        return arch_of(device) in _SUPPORTED_ARCHS
    except ValueError:
        return False


# The lottery column lattice: ``n`` is a multiple of it, so a lottery tile
# never straddles experts whatever the CTA tile width.
_LOTTERY_N = 128


@dataclass(frozen=True)
class GroupedMixedGemmConfig:
    """Compile-time knobs of the MoE mining kernel.

    SM100: ``mma_sm`` / ``tile_n`` select the UMMA shape and the per-CTA tile
    width exactly as ``GroupedFp8GemmConfig`` does (128 x ``tile_n`` per CTA,
    the 2-CTA pair covering 256 rows); SM120 has no CTA pairs (``mma_sm=1``)
    and runs the static schedule only. SM90 runs the static schedule with
    ``tile_n=128``; ``mma_sm=2`` selects the 256-row tile, one CTA with two
    consumer warpgroups.
    The lottery observables are the same for every tile shape: the lottery
    lattice is expert-local and in ``ltile_rows x ltile_cols`` units, and hit
    records report lottery-tile coordinates.
    """

    ltile_rows: int = DEFAULT_LTILE_ROWS
    ltile_cols: int = DEFAULT_LTILE_COLS
    # Persistent tile schedule, as ``GroupedFp8GemmConfig.scheduler``. The
    # lottery observables (accumulators, messages, the single first winner
    # at a one-winner threshold) do not depend on which CTA runs a tile.
    scheduler: str = "static"
    mma_sm: int = 1
    tile_n: int = 128

    def __post_init__(self) -> None:
        if self.scheduler not in SCHEDULERS:
            raise ValueError(f"scheduler must be one of {SCHEDULERS}")
        if type(self.mma_sm) is not int or self.mma_sm not in (1, 2):
            raise ValueError("mma_sm must be 1 or 2")
        if type(self.tile_n) is not int or self.tile_n not in (128, 256):
            raise ValueError("tile_n must be 128 or 256")
        for name, value in (("ltile_rows", self.ltile_rows), ("ltile_cols", self.ltile_cols)):
            if type(value) is not int:
                raise ValueError(f"{name} must be an int, got {value!r}")
        if self.ltile_rows not in SUPPORTED_LTILE_ROWS:
            raise ValueError(f"ltile_rows must be one of {SUPPORTED_LTILE_ROWS}")
        if self.ltile_cols not in SUPPORTED_LTILE_COLS[self.ltile_rows]:
            raise ValueError(
                f"ltile_cols must be one of {SUPPORTED_LTILE_COLS[self.ltile_rows]} "
                f"for ltile_rows={self.ltile_rows}"
            )
        if self.tile_n % self.ltile_cols:
            raise ValueError(f"ltile_cols must divide the {self.tile_n}-wide CTA tile")

    @property
    def cluster_tile_shape_mn(self) -> tuple[int, int]:
        """Output tile of one cluster (the mining kernel has no N cluster)."""
        return CTA_TILE_M * self.mma_sm, self.tile_n

    @classmethod
    def auto(
        cls,
        cum_m: int,
        num_groups: int,
        *,
        n: int | None = None,
        arch: Arch | None = None,
        device: torch.device | int | None = None,
        **knobs,
    ) -> "GroupedMixedGemmConfig":
        """Tile shape for ``arch`` (default: ``device``'s) from the host-known
        average tokens per expert.

        SM100: the rule of ``GroupedFp8GemmConfig.auto`` (the mining kernel
        is always full-K): the 2-CTA 256-wide tile from
        ``WIDE_TILE_MIN_TOKENS_PER_GROUP`` tokens per expert on average, the
        1-SM 128x128 tile below. SM90: the same rule picks the 256- or
        128-row tile, always 128 wide. SM120: the 128x128 tile. Explicit tile
        knobs win.

        ``n`` (the expert width) clamps the auto-selected ``tile_n`` so the
        CTA tile never exceeds the expert: pass the per-expert ``n_e``, not
        the stacked ``num_groups * n_e``.
        """
        if arch is None:
            arch = arch_of(device)
        if arch is Arch.SM120:
            return cls(**knobs)
        wide = cum_m >= WIDE_TILE_MIN_TOKENS_PER_GROUP * num_groups
        knobs.setdefault("mma_sm", 2 if wide else 1)
        default_tile_n = 256 if wide and arch is Arch.SM100 else 128
        if n is not None and default_tile_n > n:
            default_tile_n = 128
        knobs.setdefault("tile_n", default_tile_n)
        return cls(**knobs)


def _u32_fake():
    return make_fake_tensor(cutlass.Uint32, (_sym(),), leading_dim=0, divisibility=4)


@single_flight_compile
def _compile_variant(
    capability: tuple,
    snapshot_payload: bool,
    ltile_rows: int,
    ltile_cols: int,
    scheduler: str,
    has_group_order: bool,
    mma_sm: int,
    tile_n: int,
    max_active_clusters: int,
):
    variant_flags = {
        "snapshot_payload": snapshot_payload,
        "ltile_rows": ltile_rows,
        "ltile_cols": ltile_cols,
    }
    arch = arch_of(capability)
    if arch is Arch.SM120:
        kernel = _GroupedFusedGemmSm120(CTA_TILE_M, tile_n, **variant_flags)
    elif arch is Arch.SM90:
        kernel = _GroupedFusedGemmSm90(CTA_TILE_M * mma_sm, tile_n, **variant_flags)
    else:
        kernel = GroupedGemmSm100(
            scale_major_k=False,
            mma_sm=mma_sm,
            full_k_acc=True,
            mining=True,
            scheduler=scheduler,
            has_group_order=has_group_order,
            tile_n=tile_n,
            **variant_flags,
        )
    fp8 = cutlass.Float8E4M3FN
    bf16 = cutlass.BFloat16
    a_fake = make_fake_tensor(fp8, (_sym(), _sym(16)), leading_dim=1, divisibility=16)
    b_fake = make_fake_tensor(fp8, (_sym(), _sym(16)), leading_dim=1, divisibility=16)
    indptr_fake = make_fake_tensor(cutlass.Int32, (_sym(),), leading_dim=0, divisibility=1)
    valid_fake = make_fake_tensor(cutlass.Int32, (_sym(),), leading_dim=0, divisibility=1)
    c_fake = make_fake_tensor(bf16, (_sym(), _sym(8)), leading_dim=1, divisibility=8)
    pa_fake = make_fake_tensor(bf16, (_sym(), R2), leading_dim=1, divisibility=8)
    pb_fake = make_fake_tensor(bf16, (_sym(), R2), leading_dim=1, divisibility=8)
    ala_fake = make_fake_tensor(bf16, (_sym(),), leading_dim=0, divisibility=8)
    alb_fake = make_fake_tensor(cutlass.Float32, (_sym(),), leading_dim=0, divisibility=4)
    key_fake, thr_fake, hash_b_fake = _u32_fake(), _u32_fake(), _u32_fake()
    record_fake = _u32_fake()
    lock_fake = make_fake_tensor(cutlass.Int32, (_sym(),), leading_dim=0, divisibility=4)
    payload_fakes = tuple(_u32_fake() if snapshot_payload else None for _ in range(4))
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
        kernel.mine,
        a_fake,
        b_fake,
        c_fake,
        indptr_fake,
        valid_fake,
        sched_fake,
        order_fake,
        pa_fake,
        pb_fake,
        ala_fake,
        alb_fake,
        key_fake,
        thr_fake,
        record_fake,
        lock_fake,
        hash_b_fake,
        *payload_fakes,
        Int32(0),  # n
        Int32(0),  # layer_id
        Int32(0),  # record_hits
        max_active_clusters,
        make_fake_stream(),
        options="--enable-tvm-ffi",
    )


def _validate_u32_tag(name: str, value: int) -> None:
    if type(value) is not int or not 0 <= value < 2**32:
        raise ValueError(
            f"{name} must be an int fitting an unsigned 32-bit record word, got "
            f"{value!r} ({type(value).__name__})"
        )


def _validate_output_operands(
    out: torch.Tensor,
    a_peel: torch.Tensor,
    b_peel: torch.Tensor,
    alpha_a: torch.Tensor,
    inv_alpha_b: torch.Tensor,
    cum_m: int,
    num_groups: int,
    n: int,
    device: torch.device,
) -> None:
    """Validate the output and its peel/unscale operands."""
    for name, tensor, dtype, shape in (
        ("a_peel", a_peel, torch.bfloat16, (cum_m, R2)),
        ("b_peel", b_peel, torch.bfloat16, (num_groups, n, R2)),
        ("alpha_a", alpha_a, torch.bfloat16, (cum_m,)),
        ("inv_alpha_b", inv_alpha_b, torch.float32, (num_groups, n)),
        ("out", out, torch.bfloat16, (cum_m, n)),
    ):
        require_tensor(name, tensor, dtype=dtype, shape=shape, device=device, alignment=16)


def _resolve_payload_planes(
    a_codes: torch.Tensor | None,
    a_scales: torch.Tensor | None,
    cum_m: int,
    k: int,
    device: torch.device,
) -> bool:
    """Select and validate the payload (both planes) or payload-less (both
    ``None``) variant; a mixture is a caller bug and raises."""
    if (a_codes is None) != (a_scales is None):
        raise ValueError("a_codes and a_scales must be both present (payload) or both None")
    if a_codes is None:
        return False
    require_tensor(
        "a_codes", a_codes, dtype=torch.int8, shape=(cum_m, k), device=device, alignment=16
    )
    require_tensor(
        "a_scales",
        a_scales,
        dtype=torch.bfloat16,
        shape=(cum_m, k // BLOCK_SCALE_GROUP),
        device=device,
        alignment=16,
    )
    return True


def _validate_hit_signal(hit_signal: HitSignal, device: torch.device) -> None:
    if not isinstance(hit_signal, HitSignal):
        raise TypeError("hit_signal must be a pearl_gemm.pow.HitSignal")
    if hit_signal.device != device:
        raise ValueError(f"hit_signal must live on {device}, got {hit_signal.device}")
    hit_signal.require_usable()


def _require_fused_kernel_config(
    arch: Arch, config: GroupedMixedGemmConfig, group_order: torch.Tensor | None
) -> None:
    """SM90 / SM120 run the static schedule in expert order with one CTA per
    tile and the register accumulator's 4-row lottery tiles; SM90 tiles are
    128 wide."""
    if arch is Arch.SM100:
        return
    if (
        config.scheduler != "static"
        or group_order is not None
        or config.ltile_rows != LTILE_ROWS
        or (arch is Arch.SM120 and config.mma_sm != 1)
        or (arch is Arch.SM90 and config.tile_n != 128)
    ):
        raise ValueError(
            f"{arch.name} runs the static schedule in expert order: scheduler must be "
            f"'static', group_order None and ltile_rows {LTILE_ROWS}"
            + (", mma_sm 1 (no CTA pairs)" if arch is Arch.SM120 else ", tile_n 128")
        )


def grouped_mixed_gemm(
    a_prime: torch.Tensor,
    b_prime: torch.Tensor,
    a_peel: torch.Tensor,
    b_peel: torch.Tensor,
    alpha_a: torch.Tensor,
    inv_alpha_b: torch.Tensor,
    pow_key: torch.Tensor,
    threshold: torch.Tensor,
    out: torch.Tensor,
    m_indptr: torch.Tensor,
    hit_signal: HitSignal,
    a_codes: torch.Tensor | None,
    a_scales: torch.Tensor | None,
    commitment_hash_b: torch.Tensor,
    m_valid: torch.Tensor,
    *,
    config: GroupedMixedGemmConfig | None = None,
    layer_id: int = 0,
    record_hits: bool = True,
    group_order: torch.Tensor | None = None,
) -> None:
    """Launch the MoE mining grouped GEMM into caller-owned outputs.

    Operands. A-side tensors (``a_prime`` ``(cum_m, k)`` e4m3, ``a_peel``
    ``(cum_m, 2R)`` bf16, ``alpha_a`` ``(cum_m,)`` bf16, ``a_codes``,
    ``a_scales``) are permuted so expert ``e`` owns rows ``[m_indptr[e],
    m_indptr[e + 1])`` in ascending token order. B-side tensors (``b_prime``
    ``(num_groups, n, k)`` e4m3, ``b_peel`` ``(num_groups, n, 2R)`` bf16,
    ``inv_alpha_b`` ``(num_groups, n)`` f32) are stacked per expert; ``out``
    is ``(cum_m, n)`` bf16. ``m_indptr`` (``(num_groups + 1,)`` int32
    exclusive prefix sums) needs no alignment: the lottery lattice restarts at
    each expert's first row, and an expert's trailing partial tile is hashed
    but never published. ``m_valid`` (``(num_groups,)`` int32, caller-owned)
    is the number of lottery-eligible rows per expert: tiles that reach past
    it are hashed but never published. It is at most ``m_indptr[1:] -
    m_indptr[:-1]``. ``n`` must be a multiple of 128 and ``k`` of
    ``HIT_PAYLOAD_K_ALIGN``. Written buffers (``out``, the hit signal's
    record, latch and payload regions) must not share bytes with any other
    operand.

    No host sync. The host never reads ``m_indptr``, ``m_valid`` or
    ``group_order`` values: the kernel clamps every expert's rows into
    ``[0, cum_m)``, ``m_valid`` into ``[0, rows]`` and ``group_order``
    entries into ``[0, num_groups)``, so a malformed table yields unspecified
    rows, never an out-of-bounds access. Every operand is read asynchronously
    on the current stream.

    Hits. ``pow_key`` / ``threshold`` / ``commitment_hash_b`` / ``layer_id``
    / ``record_hits`` are as in ``mixed_gemm`` (one ``pow_key`` per launch,
    shared by every expert). A hit publishes into ``hit_signal`` with ``M`` =
    the expert's block row count (``m_indptr[e + 1] - m_indptr[e]``),
    ``TILE_ROW`` expert-local and ``GROUP_ID`` = the expert
    (``Hit.group_id``). With ``a_codes`` / ``a_scales`` (the permuted
    committed A planes, ``(cum_m, k)`` int8 and ``(cum_m, k / 8)`` bf16) the
    record also carries the expert's rows as payload when they fit the
    signal; pass both as ``None`` to publish payload-less records (the
    consumer then opens the planes it retained itself).

    ``group_order`` (SM100 only) is the persistent schedule's group visiting
    order (see ``grouped_fp8_gemm``). ``config=None`` picks the tile shape
    from the average tokens per expert (``GroupedMixedGemmConfig.auto``).
    SM120 has no CTA pairs and runs the static schedule in expert order:
    ``mma_sm`` must be 1, ``scheduler`` ``"static"`` and ``group_order``
    ``None`` there. SM90 runs the same static schedule with ``tile_n=128``,
    and ``mma_sm=2`` selects its 256-row single-CTA tile. Both take only the
    4-row lottery family.
    """
    cum_m, num_groups, n, k = _problem_dims(a_prime, b_prime)
    device = a_prime.device
    arch = require_arch("grouped_mixed_gemm", device, *_SUPPORTED_ARCHS)
    capability = torch.cuda.get_device_capability(device)
    if config is None:
        config = GroupedMixedGemmConfig.auto(cum_m, num_groups, n=n, arch=arch)
    _require_fused_kernel_config(arch, config, group_order)
    # ``_problem_dims`` only checks divisibility; the mining launch has no
    # empty-problem policy, so every extent must be positive.
    if num_groups <= 0 or n <= 0 or k <= 0:
        raise ValueError("num_groups, n, and k must be positive")
    if cum_m <= 0:
        raise ValueError("cum_m must be positive")
    if n % _LOTTERY_N:
        raise ValueError(f"n must be a multiple of {_LOTTERY_N}, got {n}")
    if k % HIT_PAYLOAD_K_ALIGN:
        raise ValueError(f"k must be divisible by {HIT_PAYLOAD_K_ALIGN}")
    _require_scheduler_geometry(cum_m, num_groups, n, config.cluster_tile_shape_mn)
    _validate_output_operands(
        out, a_peel, b_peel, alpha_a, inv_alpha_b, cum_m, num_groups, n, device
    )
    _validate_u32_tag("layer_id", layer_id)
    if type(record_hits) is not bool:
        raise TypeError(f"record_hits must be bool, got {type(record_hits).__name__}")
    _validate_hit_signal(hit_signal, device)
    require_tensor(
        "a_prime",
        a_prime,
        dtype=torch.float8_e4m3fn,
        shape=(cum_m, k),
        device=device,
        alignment=16,
    )
    require_tensor(
        "b_prime",
        b_prime,
        dtype=torch.float8_e4m3fn,
        shape=(num_groups, n, k),
        device=device,
        alignment=16,
    )
    for name, tensor in (
        ("pow_key", pow_key),
        ("threshold", threshold),
        ("commitment_hash_b", commitment_hash_b),
    ):
        require_tensor(name, tensor, dtype=torch.uint8, shape=(32,), device=device, alignment=16)
    _validate_indptr(m_indptr, num_groups, device)
    snapshot_payload = _resolve_payload_planes(a_codes, a_scales, cum_m, k, device)
    # Values are the kernel's to clamp into the group's block (see
    # ``_validate_indptr``); no host read.
    require_tensor(
        "m_valid", m_valid, dtype=torch.int32, shape=(num_groups,), device=device, alignment=4
    )
    _validate_group_order(group_order, num_groups, device)
    # Every buffer the kernel writes, the signal's record (published fields;
    # its CUDA alias shares the pinned record's address) and claim latch
    # included: an input carved from either would be overwritten mid-launch.
    require_disjoint_writes(
        [
            ("out", out),
            ("hit_signal.record", hit_signal.record_device_view),
            ("hit_signal.lock", hit_signal.lock),
            ("hit_signal.codes_payload", hit_signal.codes_payload),
            ("hit_signal.scales_payload", hit_signal.scales_payload),
        ],
        [
            ("a_prime", a_prime),
            ("b_prime", b_prime),
            ("a_peel", a_peel),
            ("b_peel", b_peel),
            ("alpha_a", alpha_a),
            ("inv_alpha_b", inv_alpha_b),
            ("pow_key", pow_key),
            ("threshold", threshold),
            ("m_indptr", m_indptr),
            ("m_valid", m_valid),
            ("a_codes", a_codes),
            ("a_scales", a_scales),
            ("commitment_hash_b", commitment_hash_b),
            ("group_order", group_order),
        ],
    )

    # With the planes present the snapshot is decided per hit at run time
    # (the winning group's rows must fit the signal); without them the
    # variant carries no copy and records publish payload-less.
    payload_views = (
        (
            a_codes.view(torch.uint32).reshape(-1),
            a_scales.view(torch.uint32).reshape(-1),
            hit_signal.codes_payload.view(torch.uint32),
            hit_signal.scales_payload.view(torch.uint32),
        )
        if snapshot_payload
        else (None, None, None, None)
    )
    # An SM90 256-row tile is one CTA, not a CTA pair.
    max_active_clusters = get_max_active_clusters(1 if arch is Arch.SM90 else config.mma_sm)
    compiled = _compile_variant(
        capability,
        snapshot_payload,
        config.ltile_rows,
        config.ltile_cols,
        config.scheduler,
        group_order is not None,
        config.mma_sm,
        config.tile_n,
        max_active_clusters,
    )
    compiled(
        a_prime,
        b_prime.view(num_groups * n, k),
        out,
        m_indptr,
        m_valid,
        _sched_counter(device) if config.scheduler == "dynamic" else None,
        group_order,
        a_peel,
        b_peel.view(num_groups * n, R2),
        alpha_a,
        inv_alpha_b.view(num_groups * n),
        pow_key.view(torch.uint32),
        threshold.view(torch.uint32),
        hit_signal.record_device_view,
        hit_signal.lock,
        commitment_hash_b.view(torch.uint32),
        *payload_views,
        Int32(n),
        Int32(layer_id),
        Int32(record_hits),
        get_stream(device.index or 0),
    )
