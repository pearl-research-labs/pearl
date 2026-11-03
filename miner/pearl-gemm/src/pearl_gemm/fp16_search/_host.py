"""Host launch for the bit-exact A100 (``sm_80``) full-matrix FP16 lottery search.

The FP16 analogue of the lottery fused inside the FP8 ``mixed_gemm``: scan every
committed tile of a noised matmul ``A' @ B'^T`` on-GPU and latch the FIRST tile
(lowest flat tile index) whose jackpot ticket clears the difficulty threshold.

Per tile ``(tr, tc)`` over the ``(m/h) x (n/w)`` grid the kernel recomputes the
``h x w`` tile with the SAME bit-exact A100 accumulation as
:func:`pearl_gemm.fp16_policy.replay_and_evaluate` /
:func:`pearl_gemm.fp16_gemm.fp16_gemm_a100` (``a100_dot``), folds the tile's f32
bit patterns into the 64-byte XOR-fold message over the committed 16-lane layout
(:func:`pearl_gemm.fp16_pipeline.lane_assignment` /
:func:`~pearl_gemm.fp16_pipeline.xor_fold_extract`), computes
``ticket = keyed-BLAKE3(message, key=pow_key)`` (the host-precomputed jackpot
subkey, exactly as fp8 passes ``pow_key`` into ``mixed_gemm``), and tests
``le(ticket) <= bound`` where ``bound`` is the host-precomputed 256-bit target of
:func:`pearl_gemm.fp16_pipeline.check_jackpot_difficulty`.

**Policy is NOT evaluated here.** Latching is purely on the jackpot difficulty
threshold, mirroring fp8 (which filters policy-inadmissible winners host-side at
submission). The host driver runs :func:`pearl_gemm.fp16_policy` on the latched
tile afterwards.

The kernel is a small hand-written CUDA extension compiled with
``nvcc -arch=sm_80`` (loaded through ``torch.utils.cpp_extension``), exactly like
``fp16_gemm`` / ``fp16_policy`` / ``fp16_commit``; the ``a100_dot`` accumulation
and the keyed-BLAKE3 compression are reused verbatim from those GA100-validated
siblings.
"""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass

import torch

from .._utils._arch import Arch, require_arch
from ..fp16_pipeline._layout import (
    AxisPattern,
    JACKPOT_ENTRIES,
    lane_assignment,
    nbits_to_difficulty,
)

_SUPPORTED_ARCHS = (Arch.SM80,)

_U256_MAX = (1 << 256) - 1


@dataclass(frozen=True)
class SearchHit:
    """The first-winner latch (or ``found=False`` when no tile cleared ``nbits``).

    ``tile_row`` / ``tile_col`` are the winning tile's grid coordinates (lowest
    flat index ``tile_row * (n // w) + tile_col`` wins ties). ``ticket`` is the
    32-byte keyed-BLAKE3 jackpot digest of that tile. The host driver reconstructs
    the absolute operand rows/cols from ``(tile_row, tile_col)`` and the tile
    geometry ``(h, w)`` -- rows ``[tile_row*h, tile_row*h + h)`` of ``A'`` and
    rows ``[tile_col*w, tile_col*w + w)`` of ``B'`` -- then runs ``fp16_policy``
    on it before submission.
    """

    found: bool
    tile_row: int
    tile_col: int
    ticket: bytes


def difficulty_bound(nbits: int, h: int, w: int, k: int) -> int:
    """The 256-bit win bound ``min(U256::MAX, difficulty(nbits) * sat_u32(h*w*k))``.

    Bit-for-bit with ``crate::api::proof_utils::check_jackpot_difficulty`` /
    :func:`pearl_gemm.fp16_pipeline.check_jackpot_difficulty`: a ticket wins iff
    ``le(ticket) <= bound``.
    """
    target = nbits_to_difficulty(nbits)
    adjustment = min(h * w * k, 0xFFFFFFFF)  # u32 checked_mul saturates to u32::MAX
    if adjustment != 0 and target > _U256_MAX // adjustment:
        return _U256_MAX
    return target * adjustment


@functools.lru_cache(maxsize=1)
def _extension():
    """Compile (once) and return the loaded ``sm_80`` CUDA extension."""
    from torch.utils.cpp_extension import load

    src = os.path.join(os.path.dirname(__file__), "_kernel_sm80.cu")
    return load(
        name="pearl_fp16_search_sm80",
        sources=[src],
        extra_cuda_cflags=["-arch=sm_80", "-O3"],
        verbose=False,
    )


def _u32_words_tensor(value: int, device: torch.device) -> torch.Tensor:
    """``value`` as 8 little-endian ``u32`` words (word 0 least significant), as a
    signed int32 CUDA tensor on ``device``."""
    words = [(value >> (32 * i)) & 0xFFFFFFFF for i in range(8)]
    signed = [w - (1 << 32) if w >= (1 << 31) else w for w in words]
    return torch.tensor(signed, dtype=torch.int32, device=device)


def _lanes_tensor(
    rows_pattern: AxisPattern, cols_pattern: AxisPattern, h: int, w: int, device: torch.device
) -> torch.Tensor:
    """The precomputed ``16 x (h*w/16)`` lane-index table, flattened to int32 on
    ``device``. Computed host-side so the kernel is pattern-agnostic."""
    lanes = lane_assignment(rows_pattern, cols_pattern)
    if len(lanes) != JACKPOT_ENTRIES:
        raise ValueError(f"expected {JACKPOT_ENTRIES} lanes, got {len(lanes)}")
    lane_len = len(lanes[0])
    if any(len(lane) != lane_len for lane in lanes):
        raise ValueError("all lanes must have the same fold size")
    if lane_len * JACKPOT_ENTRIES != h * w:
        raise ValueError(f"lane table covers {lane_len * JACKPOT_ENTRIES} cells, tile is {h * w}")
    flat = [idx for lane in lanes for idx in lane]
    return torch.tensor(flat, dtype=torch.int32, device=device)


def search(
    a_prime: torch.Tensor,
    b_prime: torch.Tensor,
    pow_key: bytes,
    nbits: int,
    rows_pattern: AxisPattern,
    cols_pattern: AxisPattern,
    collect_tickets: bool = False,
):
    """Full-matrix FP16 lottery search over ``a_prime @ b_prime.T`` on ``sm_80``.

    ``a_prime`` is ``(m, k)`` float16 (the noised A') and ``b_prime`` is
    ``(n, k)`` float16 (the noised, transposed B': row ``j`` is logical column
    ``j``). ``pow_key`` is the 32-byte jackpot subkey (the host precomputes
    ``BLAKE3("pearl/v4/FP8/jackpot", key=seed_a)``). ``nbits`` is the compact
    difficulty. The committed tile is ``h = rows_pattern.tile_size()`` by
    ``w = cols_pattern.tile_size()``; the 16-lane fold layout is precomputed from
    the patterns host-side and passed to the kernel, so the kernel is
    pattern-agnostic.

    Returns a :class:`SearchHit` for the first tile (lowest flat tile index
    ``tr * (n // w) + tc``) whose ticket clears ``nbits`` scaled by ``h*w*k``, or
    ``found=False`` when none does. When ``collect_tickets`` is set, returns
    ``(SearchHit, tickets)`` where ``tickets`` is an ``(num_tiles, 8)`` uint32
    CPU tensor of every tile's ticket words (for validation/debugging).
    """
    if a_prime.ndim != 2 or b_prime.ndim != 2:
        raise ValueError("a_prime and b_prime must be 2D")
    if a_prime.dtype != torch.float16 or b_prime.dtype != torch.float16:
        raise ValueError("a_prime and b_prime must be float16")
    if len(pow_key) != 32:
        raise ValueError(f"pow_key must be 32 bytes, got {len(pow_key)}")
    m, k = a_prime.shape
    n, kb = b_prime.shape
    if kb != k:
        raise ValueError(f"a_prime and b_prime must share k ({k} != {kb})")
    h = rows_pattern.tile_size()
    w = cols_pattern.tile_size()
    if m % h != 0:
        raise ValueError(f"m={m} is not a multiple of h={h}")
    if n % w != 0:
        raise ValueError(f"n={n} is not a multiple of w={w}")
    require_arch("fp16_search", a_prime.device, *_SUPPORTED_ARCHS)

    dev = a_prime.device
    lanes = _lanes_tensor(rows_pattern, cols_pattern, h, w, dev)
    key_words = [int.from_bytes(pow_key[4 * i : 4 * i + 4], "little") for i in range(8)]
    key_signed = [kw - (1 << 32) if kw >= (1 << 31) else kw for kw in key_words]
    key_t = torch.tensor(key_signed, dtype=torch.int32, device=dev)
    bound_t = _u32_words_tensor(difficulty_bound(nbits, int(h), int(w), int(k)), dev)

    found, min_idx, latch, tickets = _extension().fp16_search(
        a_prime.contiguous(), b_prime.contiguous(), lanes, key_t, bound_t,
        int(h), int(w), 1 if collect_tickets else 0,
    )

    found_h = bool(found.cpu().item())
    if found_h:
        latch_w = latch.cpu().numpy()
        tile_row = int(latch_w[0])
        tile_col = int(latch_w[1])
        ticket_words = latch_w[2:10].view("uint32")
        ticket = b"".join(int(tw).to_bytes(4, "little") for tw in ticket_words)
        hit = SearchHit(found=True, tile_row=tile_row, tile_col=tile_col, ticket=ticket)
    else:
        hit = SearchHit(found=False, tile_row=-1, tile_col=-1, ticket=b"")

    if collect_tickets:
        ntc = n // w
        num_tiles = (m // h) * ntc
        tickets_u32 = tickets.cpu().numpy().view("uint32").reshape(num_tiles, 8)
        return hit, torch.from_numpy(tickets_u32.copy())
    return hit
