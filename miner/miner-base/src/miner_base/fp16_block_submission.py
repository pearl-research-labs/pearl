"""FP16 (A100) plaintext certificate assembly: winner -> ``Fp16PlainProof``.

The FP16 analogue of :func:`miner_base.block_submission.create_proof`. Given the
FULL committed FP16 operands, the public job parameters, and which lottery tile
won (``tile_row`` / ``tile_col``, or an explicit tile base), it commits both
operand trees (bit-exact to the miner's GPU ``fp16_commit`` so the roots -- and
therefore the whole seed chain and the jackpot ticket -- match the search),
opens each at the winning tile's GLOBAL rows, and assembles the wire
``pearl_mining.Fp16PlainProof``.

Why the FULL operands (not just the opened strip): the noise-seed chain binds the
full ``m x k`` / ``n x k`` Merkle roots, so the certificate tree must be the same
one the search committed. The verifier re-derives ``seedA`` from those roots and
replays the opened tile; a per-tile re-commitment would change the root -> the
seed -> the noise -> the tile, and the winning ticket would not reproduce.

Tiling (the #7 reconciliation): the committed A100 patterns are *contiguous*
(``P.tile_offsets() == range(P.tile_size())`` with ``P.total() == P.tile_size()``),
so tile ``(tr, tc)`` opens the contiguous global rows ``range(tr*h, tr*h+h)`` /
``range(tc*w, tc*w+w)`` -- exactly how ``pearl_gemm.fp16_miner.search_block`` /
``fp16_search`` slice a hit. Those global indices are passed straight through as
the opening's ``row_indices``; the Rust ``parse_proof`` accepts them as
``base + tile_offsets()`` (``base = tr*h``; ``base == 0`` is the origin tile) and
keys the ``E`` noise lines on them, matching the full-matrix search.

Needs the ``pearl_mining`` extension (py3.13 here); the opener in
:mod:`miner_base.fp16_commitment` is torch-free and interpreter-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Sequence

import pearl_mining

from .fp16_commitment import (
    DEFAULT_HASH_ID,
    Fp16MatrixMerkleProof,
    commit_fp16_operand,
    key_a as fp16_key_a,
    key_b as fp16_key_b,
    rows_to_bytes,
)
from .layout import AxisPattern
from .params import HashId

if TYPE_CHECKING:
    from pearl_gateway.comm.dataclasses import MiningJob

    from .block_submission import PlainProofClient

__all__ = [
    "NOISE_RANK",
    "Fp16OpenedBlock",
    "create_fp16_proof",
    "submit_fp16_block",
]

# The FP16 scheme's fixed noise rank (``zk-pow/src/api/fp16/params.rs``).
NOISE_RANK = 32

_NATIVE_HASH_ID = {
    HashId.BLAKE3_CHUNK_128: pearl_mining.HashId.Blake3Chunk128,
    HashId.BLAKE3_CHUNK_256: pearl_mining.HashId.Blake3Chunk256,
    HashId.BLAKE3_CHUNK_512: pearl_mining.HashId.Blake3Chunk512,
    HashId.BLAKE3_CHUNK_1024: pearl_mining.HashId.Blake3Chunk1024,
}


def _header_bytes(header: object) -> bytes:
    """The 76-byte serialized header from raw bytes or any ``.to_bytes()`` form."""
    if isinstance(header, (bytes, bytearray, memoryview)):
        raw = bytes(header)
    else:
        raw = bytes(header.to_bytes())
    if len(raw) != 76:
        raise ValueError(f"block header must be exactly 76 bytes, got {len(raw)}")
    return raw


def _native_merkle_proof(opening: Fp16MatrixMerkleProof) -> pearl_mining.MerkleProof:
    """Rebuild the opener's (``pearl_blake3``) multi-leaf proof as the verifier's
    own ``pearl_mining.MerkleProof`` type (the two extensions are distinct shared
    objects, so the Rust type cannot be shared across them -- mirrors the FP8
    ``_native_merkle_proof``)."""
    p = opening.proof
    return pearl_mining.MerkleProof(
        [bytes(leaf) for leaf in p.leaf_data],
        list(p.leaf_indices),
        bytes(p.root),
        [bytes(sibling) for sibling in p.siblings],
        p.total_leaves,
    )


def _native_matrix_proof(opening: Fp16MatrixMerkleProof) -> pearl_mining.Fp16MatrixProof:
    return pearl_mining.Fp16MatrixProof(_native_merkle_proof(opening), list(opening.row_indices))


def _native_pattern(pattern: AxisPattern) -> pearl_mining.AxisPattern:
    """A :class:`miner_base.layout.AxisPattern` as the verifier's own type, via
    its canonical committed wire form (mirrors the FP8 ``_native_operand``)."""
    return pearl_mining.AxisPattern.from_bytes(bytes(pattern.to_bytes()))


def _tile_indices(pattern: AxisPattern, tile_index: int, num_rows: int, name: str) -> list[int]:
    """The GLOBAL opened rows for tile number ``tile_index`` along this axis:
    ``base + P.tile_offsets()`` with ``base = tile_index * P.tile_size()`` (the
    contiguous-layout tile base). Validates the base is a legal periodic offset
    and the whole tile fits in ``num_rows``."""
    offsets = pattern.tile_offsets
    base = int(tile_index) * pattern.tile_size
    if base < 0 or not pattern.offset_is_valid(base):
        raise ValueError(f"{name}: tile {tile_index} base {base} is not a valid periodic tile offset")
    indices = [base + off for off in offsets]
    if indices[-1] >= num_rows:
        raise ValueError(
            f"{name}: tile {tile_index} selects row {indices[-1]} outside [0, {num_rows})"
        )
    return indices


def create_fp16_proof(
    proposed_header: object,
    a_full: object,
    b_full: object,
    *,
    k: int,
    m: int,
    n: int,
    rows_pattern: AxisPattern,
    cols_pattern: AxisPattern,
    tile_row: int = 0,
    tile_col: int = 0,
    ancestor_header: object | None = None,
    hash_id: HashId = DEFAULT_HASH_ID,
    r: int = NOISE_RANK,
    device: object | None = None,
    a_row_indices: Sequence[int] | None = None,
    b_column_indices: Sequence[int] | None = None,
) -> pearl_mining.Fp16PlainProof:
    """Assemble the FP16 plaintext certificate for one winning tile.

    ``proposed_header`` (sigma-hat) keys the A tree and the A branch of the seed
    chain; ``ancestor_header`` (sigma-Delta, defaulting to the proposed header for
    the depth-0 miner) keys the B tree and the B branch. ``a_full`` / ``b_full``
    are the FULL ``m x k`` / ``n x k`` committed FP16 operands (any form
    :func:`miner_base.fp16_commitment.rows_to_bytes` accepts).

    ``tile_row`` / ``tile_col`` select the winning tile (``WinningTile.tile_row``
    / ``tile_col`` from ``pearl_gemm.fp16_miner.search_block``); the opened rows
    are ``tile_row*h + P_A.tile_offsets()`` / ``tile_col*w + P_B.tile_offsets()``.
    Pass ``a_row_indices`` / ``b_column_indices`` instead to open an explicit
    GLOBAL row set (they must still be a legal ``base + tile_offsets()`` tile).

    Returns a ``pearl_mining.Fp16PlainProof`` ready for
    ``pearl_mining.verify_fp16_plain_proof`` / submission.
    """
    proposed = _header_bytes(proposed_header)
    ancestor = proposed if ancestor_header is None else _header_bytes(ancestor_header)

    a_idx = (
        [int(i) for i in a_row_indices]
        if a_row_indices is not None
        else _tile_indices(rows_pattern, tile_row, m, "P_A")
    )
    b_idx = (
        [int(i) for i in b_column_indices]
        if b_column_indices is not None
        else _tile_indices(cols_pattern, tile_col, n, "P_B")
    )

    comm_a = commit_fp16_operand(a_full, m, k, fp16_key_a(proposed), hash_id)
    comm_b = commit_fp16_operand(b_full, n, k, fp16_key_b(ancestor), hash_id)
    open_a = comm_a.open(a_idx)
    open_b = comm_b.open(b_idx)

    return pearl_mining.Fp16PlainProof(
        pearl_mining.IncompleteBlockHeader.from_bytes(ancestor),
        device if device is not None else pearl_mining.Fp16Device.A100,
        int(k),
        int(r),
        pearl_mining.Fp16OperandParams(int(m), _NATIVE_HASH_ID[hash_id], _native_pattern(rows_pattern)),
        pearl_mining.Fp16OperandParams(int(n), _NATIVE_HASH_ID[hash_id], _native_pattern(cols_pattern)),
        _native_matrix_proof(open_a),
        _native_matrix_proof(open_b),
    )


@dataclass(frozen=True)
class Fp16OpenedBlock:
    """One winning FP16 tile's full committed operands + opening parameters.

    The FP16 analogue of :class:`miner_base.block_submission.OpenedBlockInfo`:
    the self-contained witness the async submission path hands to a worker
    thread. Unlike FP8 -- which carries int8 code/scale planes and lazy per-side
    commitments -- FP16 carries the FULL ``m x k`` / ``n x k`` raw FP16 operands
    (the noise-seed chain binds their complete Merkle roots, so the certificate
    tree must be the same one the search committed) and the winning tile's grid
    coordinates. :func:`create_fp16_proof` re-commits and opens from these.

    ``a`` / ``b`` are any form :func:`miner_base.fp16_commitment.rows_to_bytes`
    accepts; :meth:`owned_copy` snapshots them to the immutable ``u16`` byte
    image so a queued submission cannot observe the operand changing underneath
    it (the serving thread may reuse the activation buffer on its next forward).
    """

    a: object
    b: object
    k: int
    m: int
    n: int
    rows_pattern: AxisPattern
    cols_pattern: AxisPattern
    tile_row: int = 0
    tile_col: int = 0
    hash_id: HashId = DEFAULT_HASH_ID
    r: int = NOISE_RANK
    ancestor_header: bytes | None = None
    # Explicit GLOBAL row/col overrides; default to the contiguous tile the
    # grid coordinates name. Mirrors create_fp16_proof's two addressing modes.
    a_row_indices: tuple[int, ...] | None = None
    b_column_indices: tuple[int, ...] | None = None

    def owned_copy(self) -> "Fp16OpenedBlock":
        """Validate and snapshot the operands for an asynchronous handoff.

        Normalizes both operands to their committed ``u16`` LE byte image (bytes
        are immutable and torch/numpy-free), so the queued proof builds from a
        stable copy no later forward can mutate."""
        a_bytes = rows_to_bytes(self.a, self.m, self.k)
        b_bytes = rows_to_bytes(self.b, self.n, self.k)
        return replace(
            self,
            a=a_bytes,
            b=b_bytes,
            ancestor_header=None if self.ancestor_header is None else bytes(self.ancestor_header),
            a_row_indices=None if self.a_row_indices is None else tuple(self.a_row_indices),
            b_column_indices=(
                None if self.b_column_indices is None else tuple(self.b_column_indices)
            ),
        )

    def build_proof(self, proposed_header: bytes) -> pearl_mining.Fp16PlainProof:
        """Assemble the ``Fp16PlainProof`` for this opening under ``proposed_header``."""
        return create_fp16_proof(
            proposed_header,
            self.a,
            self.b,
            k=self.k,
            m=self.m,
            n=self.n,
            rows_pattern=self.rows_pattern,
            cols_pattern=self.cols_pattern,
            tile_row=self.tile_row,
            tile_col=self.tile_col,
            ancestor_header=self.ancestor_header,
            hash_id=self.hash_id,
            r=self.r,
            a_row_indices=self.a_row_indices,
            b_column_indices=self.b_column_indices,
        )


def submit_fp16_block(
    opened_block: Fp16OpenedBlock,
    mining_job: "MiningJob",
    client: "PlainProofClient",
) -> pearl_mining.Fp16PlainProof | None:
    """Verify and hand one certificate-v5 FP16 proof to the gateway.

    The FP16 analogue of :func:`miner_base.block_submission.submit_opened_block`:
    assembles the ``Fp16PlainProof`` from the winning tile's full operands, runs
    the standalone FP16 verifier against the proposed header, and submits the
    admissible proof. ``None`` means the proof did not verify against the job's
    difficulty and was filtered (the search latches on difficulty only; host
    policy and the verifier are the independent admissibility checks) -- the
    same filtered-winner contract the FP8 path returns ``None`` for.
    """
    # Checked directly against the pearl_mining constant (not via
    # block_submission.is_plain_fp16_job) so this whole FP16 submission module
    # stays torch-free and importable under the proof interpreter alone.
    if int(mining_job.cert_version) != pearl_mining.CERT_VERSION_PLAIN_FP16:
        raise ValueError(
            "FP16 proof submission requires certificate version "
            f"{pearl_mining.CERT_VERSION_PLAIN_FP16}, got {int(mining_job.cert_version)}"
        )

    proposed = bytes(mining_job.incomplete_header_bytes)
    header = pearl_mining.IncompleteBlockHeader.from_bytes(proposed)
    plain_proof = opened_block.build_proof(proposed)
    is_valid, message = pearl_mining.verify_fp16_plain_proof(header, plain_proof, None)
    if not is_valid:
        return None if _is_fp16_filtered(message) else _raise_fp16_verify_failure(message)

    client.submit_plain_proof(plain_proof, mining_job)
    return plain_proof


# The verifier currently reports only ``(bool, str)``. A difficulty miss (the
# tile's ticket did not clear the job target) is a normally-filtered winner, not
# a protocol error; anything else is a genuine construction/commitment bug. Keep
# this match narrow until the binding exposes an error code.
_FP16_DIFFICULTY_MISS = "jackpot"


def _is_fp16_filtered(message: str) -> bool:
    return _FP16_DIFFICULTY_MISS in message.lower()


def _raise_fp16_verify_failure(message: str):
    raise RuntimeError(f"FP16 plain proof verification failed: {message}")
