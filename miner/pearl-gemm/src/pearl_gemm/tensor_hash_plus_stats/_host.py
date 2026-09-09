"""Functional keyed tensor hash over the block-scaled activation blobs."""

from dataclasses import dataclass
from functools import lru_cache

import cutlass
import cutlass.cute as cute
import torch

from .._utils._compile import make_fake_stream, make_fake_tensor, single_flight_compile
from .._utils._stream import get_stream
from .._utils._validation import require_buffer, require_tensor
from ..pre_quant import pre_quant_output_shapes
from . import _blake3
from ._finalize_kernel import A_KEYS_BYTES, _finalize_launch, p_a_words
from ._merkle_host import (
    _validate_hash_tunables,
    get_required_scratchpad_bytes,
    tensor_hash_launch,
    tensor_hash_pair_launch,
    tensor_hash_smem_fits,
)

_CONFIG_CACHE_SIZE = 128


@dataclass(frozen=True)
class TensorHashConfig:
    """Compile-time Merkle pipeline configuration (the chunk_size /
    sync_loads / chunks_per_thread / mad_rot caveats live in the docs
    README)."""

    threads_per_block: int = 128
    num_stages: int = 2
    leaves_per_mt_block: int = 256
    thread_load_size: int = 128
    chunk_size: int = _blake3.CHUNK_SIZE
    sync_loads: bool = False
    # Last, so positional construction of the older fields keeps its meaning.
    chunks_per_thread: int = 1
    mad_rot: bool = False

    def __post_init__(self) -> None:
        # Exact type: these feed Constexpr[bool] compile parameters, so a
        # truthy string or int in a persisted record must not slip through.
        for name in ("sync_loads", "mad_rot"):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be a bool")
        _validate_hash_tunables(
            self.threads_per_block,
            self.num_stages,
            self.leaves_per_mt_block,
            self.thread_load_size,
            self.chunk_size,
            self.chunks_per_thread,
        )


# Shared default for the raw ``tensor_hash`` entry point and for callers
# that pass an explicit ``TensorHashConfig()``. The plus-stats wrappers
# resolve ``config=None`` from the device autotune records instead.
_DEFAULT_CONFIG = TensorHashConfig()

# When True, commitment wrappers reject leaves other than BLAKE3's native
# 1024. Production miners overlay that protocol leaf on autotune knobs;
# tests keep this False to run other leaves.
_REQUIRE_PROTOCOL_LEAF = False


def tensor_hash_plus_stats_record_is_legal(kwargs: dict) -> bool:
    """True when ``kwargs`` construct a config a commitment record may carry.

    Any stats-legal ``chunk_size`` is allowed: miners overlay the protocol
    1024-byte leaf on autotune knobs (cert-v4 can only open that leaf).
    Constructibility still enforces the kernel's leaf / load / stats gates.

    It also requires the estimated CTA shared memory to fit
    (``tensor_hash_smem_fits`` -- the same estimator that generates the tune
    space), so a hand-edited or stale record cannot carry a config the
    device cannot launch.
    """
    try:
        config = TensorHashConfig(**kwargs)
    except (TypeError, ValueError):
        return False
    return tensor_hash_smem_fits(
        config.threads_per_block,
        config.num_stages,
        config.thread_load_size,
        config.chunk_size,
        sync_loads=config.sync_loads,
    )


def get_tensor_hash_plus_stats_config(
    m: int,
    k: int,
    records: dict | None = None,
    device: torch.device | int | None = None,
) -> TensorHashConfig:
    """Return the autotuned ``tensor_hash_plus_stats`` config for ``(m, k)``.

    Exact JSON record if one exists, else the nearest legal neighbour in
    log2 ``(m, k)`` space (``autotune.get_tuned``). ``records=None`` reads
    the file of ``device`` (current CUDA device when omitted) and LRU-caches
    the result; pass a dict to bypass both. The raw ``tensor_hash`` keeps
    its own library default and never reads these records.
    """
    if records is not None:
        return _config_from_records(m, k, records)
    # Resolve unindexed handles (None, plain "cuda") to a concrete index
    # before caching, so the entry cannot go stale when the current device
    # changes between calls.
    index = None if device is None else torch.device(device).index
    if index is None:
        index = torch.cuda.current_device()
    return _device_config(m, k, index)


@lru_cache(maxsize=_CONFIG_CACHE_SIZE)
def _device_config(m: int, k: int, device_index: int) -> TensorHashConfig:
    # ``autotune.autotune`` clears this cache after publishing new records.
    from ..autotune import load_config

    # ``or {}``: a device without a config file fails closed to the library
    # defaults instead of falling through to the current device's file
    # inside ``get_tuned`` (wrong records on a mixed-GPU host).
    return _config_from_records(m, k, load_config(device=device_index) or {})


def _config_from_records(m: int, k: int, records: dict | None) -> TensorHashConfig:
    from ..autotune import get_tuned

    return TensorHashConfig(
        **get_tuned(
            "tensor_hash_plus_stats",
            records,
            legal=tensor_hash_plus_stats_record_is_legal,
            m=m,
            k=k,
        )
    )


def _resolve_config(
    config: TensorHashConfig | None,
    m: int,
    k: int,
    device: torch.device | int | None = None,
) -> TensorHashConfig:
    return get_tensor_hash_plus_stats_config(m, k, device=device) if config is None else config


def tensor_hash_workspace_split(
    m: int,
    k: int,
    config: TensorHashConfig | None = None,
    device: torch.device | int | None = None,
) -> tuple[int, int]:
    """Return the per-blob roots workspace sizes ``(codes, scales)`` in bytes.

    A sizing/capacity formula only: the runtime split point between the two
    blobs' slices is internal to the launch (it shrinks when thread
    coarsening takes effect) and the digests land in ``root_codes`` /
    ``root_scales`` regardless. ``config=None`` uses the same autotune
    record the plus-stats launch would pick for ``(m, k)`` on ``device``
    (current CUDA device when omitted).
    """
    config = _resolve_config(config, m, k, device)
    codes_shape, scales_shape = pre_quant_output_shapes(m, k)
    codes_bytes = codes_shape[0] * codes_shape[1]
    scales_bytes = scales_shape[0] * scales_shape[1] * 2
    return (
        tensor_hash_scratchpad_bytes(codes_bytes, config),
        tensor_hash_scratchpad_bytes(scales_bytes, config),
    )


def tensor_hash_workspace_bytes(
    m: int,
    k: int,
    config: TensorHashConfig | None = None,
    device: torch.device | int | None = None,
) -> int:
    """Return the required caller-owned roots workspace size.

    ``(m, k)`` is the logical activation shape; the two hashed payloads are
    the ``(m, k)`` int8 codes blob and the ``(m, k/8)`` BF16 scales blob,
    which share one workspace as two back-to-back slices. ``config=None``
    uses the autotune record for ``(m, k)`` on ``device`` (current CUDA
    device when omitted).
    """
    return sum(tensor_hash_workspace_split(m, k, config, device))


def tensor_hash_scratchpad_bytes(
    num_bytes: int,
    config: TensorHashConfig = _DEFAULT_CONFIG,
) -> int:
    """Return roots workspace bytes for an arbitrary contiguous byte payload."""
    if num_bytes <= 0:
        raise ValueError("num_bytes must be positive")
    return get_required_scratchpad_bytes(num_bytes, config.threads_per_block, config.chunk_size)


def tensor_hash(
    data: torch.Tensor,
    key: torch.Tensor,
    root: torch.Tensor,
    roots: torch.Tensor,
    *,
    config: TensorHashConfig = _DEFAULT_CONFIG,
) -> None:
    """Hash a contiguous CUDA tensor's raw bytes into caller-owned buffers."""
    tensor_hash_launch(
        data,
        key,
        root,
        roots,
        threads_per_block=config.threads_per_block,
        num_stages=config.num_stages,
        leaves_per_mt_block=config.leaves_per_mt_block,
        thread_load_size=config.thread_load_size,
        chunk_size=config.chunk_size,
        chunks_per_thread=config.chunks_per_thread,
        mad_rot=config.mad_rot,
        sync_loads=config.sync_loads,
    )


def _require_disjoint_storage(*named_tensors: tuple[str, torch.Tensor | None]) -> None:
    """Reject aliased operands: the launches write the roots workspace, the
    digests and the stats while reading the planes, so any storage overlap
    silently corrupts the commitment."""
    spans = []
    for name, tensor in named_tensors:
        if tensor is None:
            continue
        start = tensor.data_ptr()
        spans.append((name, start, start + tensor.numel() * tensor.element_size()))
    for i, (name_a, start_a, end_a) in enumerate(spans):
        for name_b, start_b, end_b in spans[i + 1 :]:
            if start_a < end_b and start_b < end_a:
                raise ValueError(f"{name_a} and {name_b} must not overlap in memory")


def _validate_committed_planes(
    codes: torch.Tensor,
    scales: torch.Tensor,
    key: torch.Tensor,
    root_codes: torch.Tensor,
    root_scales: torch.Tensor,
    roots: torch.Tensor,
    stats: torch.Tensor | None,
    config: TensorHashConfig,
    extra_disjoint: tuple = (),
) -> None:
    """Shared plane/digest/workspace validation for both commitment wrappers.

    ``extra_disjoint`` extends the alias check with already shape-validated
    operands (the activation path's ``seed_b``/``a_keys``), so each
    wrapper runs one complete disjoint-storage validation.
    """
    if _REQUIRE_PROTOCOL_LEAF and config.chunk_size != _blake3.CHUNK_SIZE:
        raise ValueError(
            f"commitments require the fixed {_blake3.CHUNK_SIZE}-byte proof leaf, "
            f"got chunk_size={config.chunk_size} (the proof chain cannot open "
            "other leaves; they stay an offline protocol-leaf experiment)"
        )
    m, k = codes.shape
    codes_shape, scales_shape = pre_quant_output_shapes(m, k)
    device = codes.device
    require_tensor("codes", codes, dtype=torch.int8, shape=codes_shape, alignment=16)
    require_tensor(
        "scales",
        scales,
        dtype=torch.bfloat16,
        shape=scales_shape,
        device=device,
        alignment=16,
    )
    require_tensor(
        "key",
        key,
        dtype=torch.uint8,
        shape=(32,),
        device=device,
        alignment=4,
    )
    for name, tensor in (("root_codes", root_codes), ("root_scales", root_scales)):
        require_tensor(
            name,
            tensor,
            dtype=torch.uint8,
            shape=(32,),
            device=device,
            alignment=4,
        )
    require_buffer(
        "roots",
        roots,
        dtype=torch.uint8,
        min_elements=tensor_hash_workspace_bytes(m, k, config),
        device=device,
        alignment=4,
    )
    if stats is not None:
        require_tensor(
            "stats",
            stats,
            dtype=torch.float32,
            shape=(2 * (m * k // 512),),
            device=device,
            # The prep kernels this feeds fetch a row's pairs with LDG.128,
            # so the handoff contract is the consumers' 16, not fp32x2.
            alignment=16,
        )
    _require_disjoint_storage(
        ("codes", codes),
        ("scales", scales),
        ("key", key),
        ("root_codes", root_codes),
        ("root_scales", root_scales),
        ("roots", roots),
        ("stats", stats),
        *extra_disjoint,
    )


def _launch_pair(
    codes: torch.Tensor,
    scales: torch.Tensor,
    key: torch.Tensor,
    root_codes: torch.Tensor,
    root_scales: torch.Tensor,
    roots: torch.Tensor,
    stats: torch.Tensor | None,
    config: TensorHashConfig,
) -> None:
    """Forward ``TensorHashConfig`` into the fused pair-grid launch."""
    tensor_hash_pair_launch(
        codes,
        scales,
        key,
        root_codes,
        root_scales,
        roots,
        threads_per_block=config.threads_per_block,
        num_stages=config.num_stages,
        leaves_per_mt_block=config.leaves_per_mt_block,
        thread_load_size=config.thread_load_size,
        chunk_size=config.chunk_size,
        chunks_per_thread=config.chunks_per_thread,
        mad_rot=config.mad_rot,
        sync_loads=config.sync_loads,
        stats=stats,
    )


def tensor_hash_plus_stats_b(
    codes: torch.Tensor,
    scales: torch.Tensor,
    key: torch.Tensor,
    root_codes: torch.Tensor,
    root_scales: torch.Tensor,
    roots: torch.Tensor,
    stats: torch.Tensor | None,
    *,
    config: TensorHashConfig | None = None,
) -> None:
    """Hash the committed weight planes in one grid and reduce their commit stats.

    The weights-path sibling of ``tensor_hash_plus_stats``: the same fused
    pair launch over the ``(n, k)`` int8 codes plane and the ``(n, k/8)`` BF16
    scales plane, without the device finalize. Each plane's keyed Merkle root
    is bit-identical to a standalone ``tensor_hash`` over its bytes (``key``
    is ``keyB``); ``HB = blake3(root_codes || root_scales, key=keyB)`` and
    ``noise seedB`` are two host-side combines that stay on the caller's CPU
    path (``miner_base.commitment_hash``; they run once per job, off the hot
    path).

    ``stats`` receives the fp32 ``(sumsq, absmax)`` pair per 512-element
    block -- the partials ``noisy_quant_b`` combines into ``alpha_b`` /
    ``beta_b`` (oracle: ``PrequantMatrix.exact_norms``). Pass ``stats=None``
    to produce only the two roots. ``config=None`` resolves the same way
    as ``tensor_hash_plus_stats``.
    """
    if codes.ndim != 2:
        raise ValueError("codes must be 2D")
    config = _resolve_config(config, *codes.shape, codes.device)
    _validate_committed_planes(codes, scales, key, root_codes, root_scales, roots, stats, config)
    _launch_pair(codes, scales, key, root_codes, root_scales, roots, stats, config)


def tensor_hash_plus_stats(
    codes: torch.Tensor,
    scales: torch.Tensor,
    key: torch.Tensor,
    seed_b: torch.Tensor,
    root_codes: torch.Tensor,
    root_scales: torch.Tensor,
    roots: torch.Tensor,
    a_keys: torch.Tensor,
    stats: torch.Tensor | None,
    *,
    p_a: bytes,
    config: TensorHashConfig | None = None,
) -> None:
    """Commit the block-scaled activation blobs and reduce its commit stats.

    ``codes`` / ``scales`` are ``pre_quant``'s outputs: ``(m, k)`` int8 and
    ``(m, k/8)`` BF16. Each gets its own Merkle chain keyed by ``key`` (the
    header's ``keyA``) over its raw bytes -- the two chains share every
    launch, as one grid partitioned by ``blockIdx`` -- and the finalize runs
    the v4 A-side chain down from the two roots:

        HA          = blake3(root_codes || root_scales, key=keyA)
        seedA       = H_"seed-A"(HA || seedB || keyA || pA)
        a_keys      = seedA || Subkey("noise-line", seedA) || Subkey("jackpot", seedA)

    ``seed_b`` is B's 32-byte noise seed and ``p_a`` the 11-byte dense ``pA``
    encoding (``MiningConfiguration.p_a(m)``; a compile-time constant, so a
    different lottery layout or leaf recompiles the finalize). ``a_keys`` is
    the 96-byte output (``miner_base.commitment_hash.AKeys``).

    ``stats`` receives the fp32 ``(sumsq, absmax)`` pair per 512-element
    block, reduced over the *dequantized* codes and scales -- the partials
    ``noisy_quant`` combines into ``alpha``/``beta``, so the row scales
    commit to the block-scaled tensor. Pass ``stats=None`` to skip the fused
    sweep and produce only the commitment, as a verifier recomputing the
    chain would; the digest is identical either way.

    ``config=None`` (the default) picks the ``codes.device`` autotune record
    for ``(m, k)``, or the nearest legal neighbour; pass an explicit
    ``TensorHashConfig()`` to force the library defaults. Size the
    ``roots`` workspace with the same ``config`` (or omit it there too).
    """
    if codes.ndim != 2:
        raise ValueError("codes must be 2D")
    m, k = codes.shape
    config = _resolve_config(config, m, k, codes.device)
    device = codes.device
    require_tensor("seed_b", seed_b, dtype=torch.uint8, shape=(32,), device=device, alignment=4)
    require_tensor(
        "a_keys", a_keys, dtype=torch.uint8, shape=(A_KEYS_BYTES,), device=device, alignment=4
    )
    words = p_a_words(p_a)
    # One complete alias check across all nine operands: the finalize writes
    # a_keys, so aliasing even a read-only input (key, a plane) would corrupt
    # the caller's material for later commitments.
    _validate_committed_planes(
        codes,
        scales,
        key,
        root_codes,
        root_scales,
        roots,
        stats,
        config,
        extra_disjoint=(("seed_b", seed_b), ("a_keys", a_keys)),
    )

    _launch_pair(codes, scales, key, root_codes, root_scales, roots, stats, config)
    finalize = _compile_finalize(torch.cuda.get_device_capability(device), words)
    # Explicit, like every other launch in the package. Resolved from the FFI
    # environment instead, this one stops tracking the caller's stream once
    # the GPU is contended, and publishes keys read from unwritten roots.
    finalize(
        root_codes.view(torch.uint32),
        root_scales.view(torch.uint32),
        key.view(torch.uint32),
        seed_b.view(torch.uint32),
        a_keys.view(torch.uint32),
        get_stream(device.index or 0),
    )


@single_flight_compile
def _compile_finalize(capability: tuple, p_a: tuple):
    digests = [
        make_fake_tensor(cutlass.Uint32, (8,), leading_dim=0, divisibility=1) for _ in range(4)
    ]
    out = make_fake_tensor(cutlass.Uint32, (A_KEYS_BYTES // 4,), leading_dim=0, divisibility=1)
    return cute.compile(
        _finalize_launch,
        *digests,
        out,
        make_fake_stream(),
        p_a=p_a,
        options="--enable-tvm-ffi",
    )
