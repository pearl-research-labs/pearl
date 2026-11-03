"""Host-side FP16 (A100 / ``sm_80``) lottery-search driver.

The standalone analogue of the FP8 ``_launch_stages``: from a job header and the
full plaintext FP16 operands, derive the seed chain, chain the GA100-validated
FP16 sm_80 kernels, run the full-matrix lottery search, and decode the latched
hit into a verifiable winning tile.

Data flow (mirrors ``zk-pow/src/api/fp16/plain_proof.rs::parse_proof`` run
*forward* -- the miner holds the plaintext operands, so it commits and derives
rather than parsing a proof):

1. **Keys.** ``keyA = H_"key-A"(proposed_header)``, ``keyB =
   H_"key-B"(ancestor_header)`` (:mod:`._seed_chain`). The miner proposes at depth
   0, so ``ancestor_header == proposed_header`` unless one is supplied.
2. **Commit.** :func:`pearl_gemm.fp16_commit.commit_operand` commits A under
   ``keyA`` and B under ``keyB`` -> Merkle ``root_A`` / ``root_B``.
3. **Seeds.** ``pA``/``pB`` are encoded (:func:`._seed_chain.encode_p_a` /
   ``encode_p_b``) and ``(seedA, seedB) = noise_seeds(keys, roots, p)`` (B-then-A).
   ``pow_key = subkey("…/jackpot", seedA)``.
4. **Noise.** :func:`pearl_gemm.fp16_noise_lines.sample_noise` draws ``E_A``
   (all ``m`` rows), ``E_B`` (all ``n`` cols) and the shared ``F_A``/``F_B`` bases.
5. **Noisy-quantize.** :func:`pearl_gemm.fp16_noisy_quant.noisy_quantize` (via
   the ``fp16_pipeline`` padding helper) rebuilds ``A'`` and ``B'``.
6. **Search.** :func:`pearl_gemm.fp16_search.search` scans every ``h x w`` tile of
   ``A' @ B'^T`` on-GPU and latches the first (lowest flat index) tile whose
   jackpot ticket clears ``nbits``.
7. **Decode.** The search latches on difficulty only (mirroring fp8), but
   ``verify_tile`` also requires policy admissibility, so pick the lowest-index
   tile clearing BOTH (re-scanning the per-tile tickets only if the difficulty
   latch is policy-inadmissible), then slice the opened A rows
   ``[tr*h,(tr+1)*h)`` / B rows ``[tc*w,(tc+1)*w)`` and the matching ``E`` noise
   factors and return the winning tile.

The opened rows + sliced noise + ``seedA`` are exactly the witness
``zk-pow/src/api/fp16/verify.rs::verify_tile`` consumes: noisy-quantize is
row-independent, so the tile rebuilt from the slice is bit-identical to the
searched ``A'``/``B'`` tile. Everything is gated to ``sm_80`` via ``require_arch``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch

from .._utils._arch import Arch, require_arch
from ..fp16_commit import DEFAULT_CHUNK_LEN, commit_operand
from ..fp16_noise_lines import sample_noise
from ..fp16_noisy_quant import noisy_quantize
from ..fp16_pipeline import AxisPattern
from ..fp16_pipeline._host import _rebuild_operand
from ..fp16_policy import PolicyReport, replay_and_evaluate
from ..fp16_search import difficulty_bound, search
from . import _seed_chain

_SUPPORTED_ARCHS = (Arch.SM80,)

# The FP16 scheme's fixed device (A100) and noise rank.
_A100_DEVICE_TAG = 0
NOISE_RANK = 32


@dataclass(frozen=True)
class Fp16OperandParams:
    """One operand's committed-tree public parameters (``Fp16OperandParams``)."""

    num_rows: int
    pattern: AxisPattern
    chunk_len: int = DEFAULT_CHUNK_LEN

    @property
    def hash_id(self) -> int:
        try:
            return _seed_chain.HASH_ID_FROM_CHUNK_LEN[self.chunk_len]
        except KeyError:
            raise ValueError(
                f"chunk_len must be one of {sorted(_seed_chain.HASH_ID_FROM_CHUNK_LEN)}, "
                f"got {self.chunk_len}"
            ) from None


@dataclass(frozen=True)
class Fp16JobParams:
    """The public statement for one FP16 search job (mirrors ``Fp16JobParams``).

    ``a`` is the A-side (rows, ``m``) operand, ``b`` the B-side (cols, ``n``).
    ``nbits`` is the compact difficulty target the search latches on.
    ``ancestor_header`` defaults to the proposed header (depth-0 coincidence:
    the miner proposes at depth 0, so ``σ_Δ == σ̂``); supply it only for a
    non-trivial ancestor. ``device_tag`` and ``r`` are fixed for A100.
    """

    k: int
    a: Fp16OperandParams
    b: Fp16OperandParams
    nbits: int
    r: int = NOISE_RANK
    device_tag: int = _A100_DEVICE_TAG
    ancestor_header: Optional[bytes] = None


@dataclass
class SeedChain:
    """The derived host seed chain (all 32-byte, torch-free)."""

    key_a: bytes
    key_b: bytes
    root_a: bytes
    root_b: bytes
    p_a: bytes
    p_b: bytes
    seed_a: bytes
    seed_b: bytes
    pow_key: bytes


@dataclass
class WinningTile:
    """The decoded, verifiable winner (or ``found=False`` -- no tile cleared
    ``nbits``).

    ``opened_a_rows`` is the ``(h, k)`` and ``opened_b_rows`` the ``(w, k)`` FP16
    (float16) operand strip the tile opened; ``e_a``/``e_b`` are the matching
    ``(h, r)`` / ``(w, r)`` sliced ``E`` factors and ``f_a``/``f_b`` the shared
    ``(k, r)`` ``F`` bases (all float16 views of the FP16 bit patterns) -- exactly
    the witness ``verify.rs::verify_tile`` consumes. ``report`` is the host-side
    ``fp16_policy`` admissibility verdict on the latched tile (``None`` when no
    winner). ``seed_a`` keys the jackpot ticket.
    """

    found: bool
    tile_row: int
    tile_col: int
    opened_a_rows: Optional[torch.Tensor]
    opened_b_rows: Optional[torch.Tensor]
    e_a: Optional[torch.Tensor]
    f_a: Optional[torch.Tensor]
    e_b: Optional[torch.Tensor]
    f_b: Optional[torch.Tensor]
    seed_a: bytes
    ticket: bytes
    report: Optional[PolicyReport]
    seeds: SeedChain = field(repr=False, default=None)


def _as_f16(t: torch.Tensor) -> torch.Tensor:
    """View an ``int16``/``uint16`` FP16-bit-pattern tensor as ``float16``."""
    if t.dtype == torch.float16:
        return t
    if t.dtype in (torch.int16, torch.uint16):
        return t.view(torch.float16)
    raise ValueError(f"expected float16 / int16 FP16 bit patterns, got {t.dtype}")


def _to_f16_rows(rows, num_rows: int, k: int, device: torch.device) -> torch.Tensor:
    """Normalize a ``num_rows x k`` FP16 operand (any ``u16`` bit-pattern array or
    torch tensor) to a contiguous ``float16`` CUDA tensor on ``device``."""
    if isinstance(rows, torch.Tensor):
        t = rows
    else:
        import numpy as np

        arr = np.ascontiguousarray(rows)
        if arr.dtype == np.float16:
            t = torch.from_numpy(arr)
        else:
            t = torch.from_numpy(arr.view(np.uint16).astype(np.int16))
    t = t.reshape(num_rows, k).to(device)
    return _as_f16(t).contiguous()


def derive_seed_chain(
    header_bytes: bytes,
    a_rows,
    b_rows,
    params: Fp16JobParams,
    device: torch.device | int | None = None,
) -> SeedChain:
    """Commit both operands and derive the full seed chain (steps 1-3).

    ``header_bytes`` is the 76-byte serialized proposed header. ``a_rows`` /
    ``b_rows`` are the ``m x k`` / ``n x k`` FP16 operands (``u16`` bit patterns).
    """
    dev = torch.device("cuda", device) if isinstance(device, int) else (device or torch.device("cuda"))
    require_arch("fp16_miner", dev, *_SUPPORTED_ARCHS)

    ancestor = params.ancestor_header if params.ancestor_header is not None else bytes(header_bytes)
    ka, kb = _seed_chain.commitment_keys(bytes(header_bytes), ancestor)

    m, n, k = params.a.num_rows, params.b.num_rows, params.k
    a16 = _to_f16_rows(a_rows, m, k, dev)
    b16 = _to_f16_rows(b_rows, n, k, dev)

    root_a = commit_operand(a16.view(torch.int16), m, k, ka, params.a.chunk_len, dev)
    root_b = commit_operand(b16.view(torch.int16), n, k, kb, params.b.chunk_len, dev)

    p_a = _seed_chain.encode_p_a(k, params.r, params.device_tag, m, params.a.hash_id, params.a.pattern)
    p_b = _seed_chain.encode_p_b(
        ancestor, k, params.r, params.device_tag, n, params.b.hash_id, params.b.pattern
    )
    seed_a, seed_b = _seed_chain.noise_seeds(ka, kb, root_a, root_b, p_a, p_b)
    pow_key = _seed_chain.jackpot_pow_key(seed_a)
    return SeedChain(
        key_a=ka, key_b=kb, root_a=root_a, root_b=root_b, p_a=p_a, p_b=p_b,
        seed_a=seed_a, seed_b=seed_b, pow_key=pow_key,
    )


def search_block(
    header_bytes: bytes,
    a_rows,
    b_rows,
    params: Fp16JobParams,
    device: torch.device | int | None = None,
) -> WinningTile:
    """Run the full FP16 lottery search for one job on ``sm_80``.

    ``header_bytes`` is the 76-byte serialized proposed header; ``a_rows`` is the
    ``m x k`` and ``b_rows`` the ``n x k`` FP16 operand (``u16`` bit patterns or a
    float16 tensor). Derives the seed chain, rebuilds ``A'``/``B'``, scans every
    ``h x w`` tile, and on the first winner returns a :class:`WinningTile` with the
    opened rows, sliced noise, ``seed_a``, ticket and the host-side
    :class:`~pearl_gemm.fp16_policy.PolicyReport` admissibility verdict. When no
    tile clears ``nbits``, returns ``found=False``.
    """
    dev = torch.device("cuda", device) if isinstance(device, int) else (device or torch.device("cuda"))
    require_arch("fp16_miner", dev, *_SUPPORTED_ARCHS)

    m, n, k, r = params.a.num_rows, params.b.num_rows, params.k, params.r
    rows_pattern, cols_pattern = params.a.pattern, params.b.pattern
    h, w = rows_pattern.tile_size(), cols_pattern.tile_size()
    if m % h != 0:
        raise ValueError(f"m={m} is not a multiple of h={h}")
    if n % w != 0:
        raise ValueError(f"n={n} is not a multiple of w={w}")

    a16 = _to_f16_rows(a_rows, m, k, dev)
    b16 = _to_f16_rows(b_rows, n, k, dev)

    chain = derive_seed_chain(header_bytes, a16, b16, params, dev)

    # 4. Deterministic FP16 noise for ALL rows/cols (E keyed off the GLOBAL
    #    row/col index, so the slice for any tile is exactly that tile's noise).
    noise = sample_noise(chain.seed_a, chain.seed_b, k, r, range(m), range(n), dev)
    e_a, f_a = _as_f16(noise.e_a), _as_f16(noise.f_a)
    e_b, f_b = _as_f16(noise.e_b), _as_f16(noise.f_b)

    # 5. Rebuild A' and B' (noisy-quantize; row-independent, so the per-tile slice
    #    below is bit-identical to the Rust verify_tile rebuild from the opening).
    built_a = _rebuild_operand(a16, e_a, f_a, r)
    built_b = _rebuild_operand(b16, e_b, f_b, r)

    def _no_winner() -> WinningTile:
        return WinningTile(
            found=False, tile_row=-1, tile_col=-1,
            opened_a_rows=None, opened_b_rows=None,
            e_a=None, f_a=None, e_b=None, f_b=None,
            seed_a=chain.seed_a, ticket=b"", report=None, seeds=chain,
        )

    def _evaluate(tr: int, tc: int) -> PolicyReport:
        """Host-side fp16_policy admissibility of tile ``(tr, tc)`` on the A'/B' slices."""
        a_prime_tile = built_a.noised_part[tr * h : (tr + 1) * h].contiguous()
        b_prime_tile = built_b.noised_part[tc * w : (tc + 1) * w].contiguous()
        _tile, rep = replay_and_evaluate(a_prime_tile, b_prime_tile)
        return rep

    # 6. Full-matrix lottery search. The kernel latches the lowest flat-index tile
    #    clearing ``nbits`` on DIFFICULTY ALONE (mirroring fp8). But ``verify_tile``
    #    also requires POLICY admissibility (f_bp >= 0.30, rho >= 1.2), and consensus
    #    only ever checks the OPENED tile's own difficulty + policy -- it does not pin
    #    a particular tile index -- so the submittable block is the lowest-index tile
    #    clearing BOTH. Take the difficulty latch first (the common case: honest
    #    operands are almost always admissible); only if it fails policy do we
    #    re-scan the per-tile tickets for the lowest-index tile clearing difficulty
    #    AND policy. Returning the raw latch when it is inadmissible would hand the
    #    submitter a tile the verifier rejects, silently missing a valid block.
    hit = search(built_a.noised_part, built_b.noised_part, chain.pow_key, params.nbits,
                 rows_pattern, cols_pattern)
    if not hit.found:
        return _no_winner()

    tr, tc, ticket = hit.tile_row, hit.tile_col, hit.ticket
    report = _evaluate(tr, tc)
    if not report.accept:
        import numpy as np

        _, tickets = search(
            built_a.noised_part, built_b.noised_part, chain.pow_key, params.nbits,
            rows_pattern, cols_pattern, collect_tickets=True,
        )
        bound = difficulty_bound(params.nbits, int(h), int(w), int(k))
        ntc = n // w
        words = tickets.numpy().reshape(-1, 8).astype(np.uint64)
        chosen = None
        for flat in range(words.shape[0]):
            le = sum(int(words[flat, i]) << (32 * i) for i in range(8))
            if le > bound:  # does not clear the jackpot difficulty
                continue
            cand_tr, cand_tc = divmod(flat, ntc)
            cand_report = _evaluate(cand_tr, cand_tc)
            if cand_report.accept:
                ticket = b"".join((int(words[flat, i]) & 0xFFFFFFFF).to_bytes(4, "little") for i in range(8))
                chosen = (cand_tr, cand_tc, cand_report)
                break
        if chosen is None:
            # Difficulty winners exist but none are policy-admissible: no valid block.
            return _no_winner()
        tr, tc, report = chosen

    # 7. Decode the winning tile into the opening + sliced noise.
    #
    # Tiling is CONTIGUOUS: the committed A100 patterns have dense tile offsets
    # (``P.tile_offsets() == range(P.tile_size())`` with ``total() == tile_size()``),
    # so tile ``(tr, tc)`` occupies exactly the global rows ``[tr*h, (tr+1)*h)`` /
    # ``[tc*w, (tc+1)*w)``. Those global indices ARE the certificate opening's
    # ``row_indices`` -- the cert's ``base + P.tile_offsets()`` with ``base = tr*h``
    # (``tr=tc=0`` is the origin tile). The CPU assembler
    # ``miner_base.fp16_block_submission.create_fp16_proof`` opens them from the
    # full committed trees, and the verifier's ``parse_proof`` accepts them and
    # keys the E noise on them (matching the full-matrix search here). No tile
    # base/stride translation is needed, so the GA100-validated kernel math is
    # untouched.
    opened_a = a16[tr * h : (tr + 1) * h].contiguous()
    opened_b = b16[tc * w : (tc + 1) * w].contiguous()
    e_a_tile = e_a[tr * h : (tr + 1) * h].contiguous()
    e_b_tile = e_b[tc * w : (tc + 1) * w].contiguous()

    return WinningTile(
        found=True, tile_row=tr, tile_col=tc,
        opened_a_rows=opened_a, opened_b_rows=opened_b,
        e_a=e_a_tile, f_a=f_a, e_b=e_b_tile, f_b=f_b,
        seed_a=chain.seed_a, ticket=ticket, report=report, seeds=chain,
    )


__all__ = [
    "Fp16JobParams",
    "Fp16OperandParams",
    "SeedChain",
    "WinningTile",
    "derive_seed_chain",
    "difficulty_bound",
    "search_block",
]
