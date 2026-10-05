"""MoE routing for the grouped mining launch: the canonical routing table
(``Rflat``, ``O``) built and committed on the device.

The activation is committed, seeded and noised once at its token addresses,
exactly like a dense layer; only then are the noised rows gathered into
expert order for ``grouped_mixed_gemm``. The routing table itself enters the
A-side seed through its two keyed roots ``HR`` / ``HO``
(``miner_base.transcript.noise_seeds``), so it is fixed before any A-side
noise is learned. Nothing here synchronizes with the host.
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class MoeRouting:
    """One launch's canonical routing, on the layer's device.

    ``tokens`` is ``Rflat``: for each expert in order, the tokens it received
    in ascending token order (``(cum_m,)`` int32), which is also the row
    permutation of the grouped GEMM. ``m_indptr`` (``(experts + 1,)`` int32)
    is the exclusive prefix of the per-expert counts -- ``m_indptr[1:]`` is
    the cumulative end offsets ``O`` -- and ``m_valid`` (``(experts,)`` int32)
    the counts themselves, the grouped GEMM's caller-owned row-count
    operand. ``slots`` maps every permuted row back to
    its ``token * top_k + slot`` position, for the router weights.
    ``commitments`` is ``HR || HO`` under ``keyA``.
    """

    experts: int
    top_k: int
    tokens: torch.Tensor
    m_indptr: torch.Tensor
    m_valid: torch.Tensor
    slots: torch.Tensor
    commitments: torch.Tensor

    @property
    def cum_m(self) -> int:
        return self.tokens.shape[0]


def _commit_u32(values: torch.Tensor, key_a_dev: torch.Tensor, root: torch.Tensor) -> None:
    """Keyed chunk-tree root of little-endian ``u32`` values zero-padded to whole
    1024-byte leaves (``miner_base.commitment.commit_routing`` / ``hash_offsets``
    at the committed activation leaf)."""
    from pearl_gemm import TensorHashConfig, tensor_hash, tensor_hash_scratchpad_bytes

    config = TensorHashConfig()
    raw = values.numel() * 4
    padded = -(-raw // config.chunk_size) * config.chunk_size
    payload = torch.nn.functional.pad(values.contiguous().view(torch.uint8), (0, padded - raw))
    roots = torch.empty(
        tensor_hash_scratchpad_bytes(padded, config), dtype=torch.uint8, device=values.device
    )
    tensor_hash(payload, key_a_dev, root, roots, config=config)


def validate_moe_dims(experts: int, top_k: int) -> None:
    """``experts`` and ``top_k`` are positive exact ints (no bools, no floats)
    with ``top_k <= experts``: both become tensor extents downstream."""
    for name, value in (("experts", experts), ("top_k", top_k)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive int, got {value!r}")
    if top_k > experts:
        raise ValueError(f"top_k={top_k} exceeds experts={experts}")


def topk_ids_shape_error(topk_ids: torch.Tensor, m_tokens: int, top_k: int) -> str | None:
    """Why the router's table cannot route ``m_tokens`` rows with ``top_k``
    experts each (``None`` when it can): checked on the host only."""
    if not isinstance(topk_ids, torch.Tensor):
        return f"topk_ids must be a tensor, got {type(topk_ids).__name__}"
    if topk_ids.dim() != 2 or tuple(topk_ids.shape) != (m_tokens, top_k):
        return f"topk_ids must be ({m_tokens}, {top_k}), got {tuple(topk_ids.shape)}"
    if topk_ids.dtype not in (torch.int32, torch.int64):
        return f"topk_ids must be int32/int64, got {topk_ids.dtype}"
    return None


def route_tokens(topk_ids: torch.Tensor, experts: int, key_a_dev: torch.Tensor) -> MoeRouting:
    """Canonicalize the router's ``(m_tokens, top_k)`` expert ids and commit them.

    A stable sort of the token-major flat ids groups rows by expert and keeps
    each expert's tokens ascending: the canonical routing-table order, and
    the row order ``grouped_mixed_gemm`` publishes lottery tiles in.

    This function never reads the device: the ids were produced on this
    stream moments ago, and a host read would stall the forward. It checks
    only host-known facts (rank, ``top_k``, dtype, device). The device clamps
    ids into ``[0, experts)``, so a malformed table misroutes those rows but
    cannot fault.
    """
    if topk_ids.dim() != 2:
        raise ValueError(f"topk_ids must be (m_tokens, top_k), got {tuple(topk_ids.shape)}")
    validate_moe_dims(experts, topk_ids.shape[1])
    if topk_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"topk_ids must be int32/int64, got {topk_ids.dtype}")
    if topk_ids.device != key_a_dev.device:
        raise ValueError(f"topk_ids on {topk_ids.device} but key_a on {key_a_dev.device}")
    device = topk_ids.device
    _, top_k = topk_ids.shape
    flat_ids = topk_ids.reshape(-1)
    expert_of_slot, slots = torch.sort(flat_ids.clamp(0, experts - 1), stable=True)
    tokens = torch.div(slots, top_k, rounding_mode="floor").to(torch.int32)
    # ``expert_of_slot`` is sorted, so each expert's end offset is where its
    # successor would be inserted; the output size is host-known (no sync).
    m_indptr = torch.zeros(experts + 1, dtype=torch.int32, device=device)
    m_indptr[1:] = torch.searchsorted(
        expert_of_slot,
        torch.arange(experts, device=device, dtype=expert_of_slot.dtype),
        side="right",
    ).to(torch.int32)
    m_valid = m_indptr.diff()
    commitments = torch.empty(64, dtype=torch.uint8, device=device)
    _commit_u32(tokens, key_a_dev, commitments[:32])
    _commit_u32(m_indptr[1:], key_a_dev, commitments[32:])
    return MoeRouting(experts, top_k, tokens, m_indptr, m_valid, slots, commitments)


def combine_routed_rows(
    rows: torch.Tensor, topk_weights: torch.Tensor, routing: MoeRouting
) -> torch.Tensor:
    """Router-weighted sum of each token's ``top_k`` expert outputs.

    ``rows`` is ``(cum_m, n)`` in the routing's expert-permuted order (the
    grouped GEMM output). The weighted rows are scattered back to router
    ``(token, slot)`` order with ``index_copy_`` and reduced over the slot axis
    with a plain ``sum``: two passes, no atomics, so the result is
    deterministic. Every ``(token, slot)`` position is written because the
    routing covers all ``m_tokens * top_k`` slots (no dropped rows).
    """
    weights = topk_weights.reshape(-1)[routing.slots].unsqueeze(1).to(rows.dtype)
    by_slot = torch.empty_like(rows).index_copy_(0, routing.slots, rows * weights)
    return by_slot.view(-1, routing.top_k, rows.shape[1]).sum(1)


def round_robin_topk_ids(
    m_tokens: int, experts: int, top_k: int, device: torch.device
) -> torch.Tensor:
    """A synthetic balanced routing (warmup / tests): token ``t`` takes experts
    ``t, t + 1, ..`` modulo ``experts``."""
    base = torch.arange(m_tokens, dtype=torch.int64, device=device).unsqueeze(1)
    return (base + torch.arange(top_k, device=device)) % experts


def round_robin_routing(
    m_tokens: int, experts: int, top_k: int, key_a_dev: torch.Tensor
) -> MoeRouting:
    """The committed round-robin routing of a synthetic (warmup) launch."""
    ids = round_robin_topk_ids(m_tokens, experts, top_k, key_a_dev.device)
    return route_tokens(ids, experts, key_a_dev)


@dataclass(frozen=True)
class MoeLaunch:
    """What a winner check retains beyond the payload-less MoE hit record:
    the routing table and the *unpermuted* committed planes. The runtime
    launches the grouped kernel without its optional expert-row snapshot;
    the proof opens the full activation tree at the tile's global token
    indices."""

    routing: MoeRouting
    codes: torch.Tensor
    scales: torch.Tensor
