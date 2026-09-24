"""Mining-job adapter: per-layer-shape committed configuration, thresholds, and
credited-work formulas for the SM100 schema lottery."""

from dataclasses import dataclass
from functools import cache

from miner_base.commitment import Device, MiningConfiguration, commitment_keys
from miner_base.layout import AxisPattern
from miner_base.mining_config import (
    COMMITMENT_CHUNK_SIZE,
    default_mining_config,
    tall_tile_mining_config,
)
from miner_base.policy import effective_work
from miner_base.prequant import DEFAULT_BLOCK_SIZE
from pearl_gateway.comm.dataclasses import MiningJob

# Default committed lottery tile: 4 contiguous rows x 128 contiguous cols
# (mixed_gemm's merged tile).
TILE_ROWS = 4
TILE_COLS = 128
# pearl_gemm.protocol_constants.R / PACKED_NOISE_K, duplicated so this module imports
# without the CuTe DSL.
RANK = 32
PACKED_NOISE_K = 32
# pearl_gemm.protocol_constants.BLOCK_SCALE_GROUP is this value; take it from the protocol
# rather than restating it, so the FP10 scale grouping has one definition.
SCALE_BLOCK = DEFAULT_BLOCK_SIZE

_MAX_256 = (1 << 256) - 1

# Verifier bounds for a peel proof (pinned py-pearl-mining, zk-pow
# api/fp8/public_params.rs). A winning tile exposes TILE_ROWS + TILE_COLS rows
# of k BF16 elements to one verifier worker, capped at 4 MiB; cert-v4 rejects
# k below 2048 (and above 2^16) outright. Work outside this domain can never
# become an accepted proof, so it must never be credited.
_VERIFIER_MIN_K = 2048
_VERIFIER_MAX_K = 1 << 16
_VERIFIER_MAX_WORKER_INPUT_BYTES = 1 << 22
_PEEL_ELEM_BYTES = 2
# k must also stay a multiple of 512 for the four kernels.
_K_ALIGNMENT = 512


# Public m/n dimension cap (zk-pow api/sanity_checks.rs), exclusive. A shape at
# the boundary would allocate its (n, k) buffers and only then fail
# preparation, so it must never be admitted.
_VERIFIER_MAX_DIM = 1 << 24


@dataclass(frozen=True)
class LotteryTileSpec:
    """A committed merged lottery tile (``rows`` x ``cols``)."""

    rows: int
    cols: int

    def __post_init__(self) -> None:
        # rows/cols feed ``bytes_per_k`` and the chained ``//`` in
        # ``max_verifiable_k`` and the credited-message count; a float or bool
        # (an int subclass) would silently corrupt every derived integer, so
        # reject non-ints at construction rather than downstream.
        for name, value in (("rows", self.rows), ("cols", self.cols)):
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"lottery tile {name} must be an int, got {value!r}")
            if value <= 0:
                raise ValueError(f"lottery tile {name} must be positive, got {value}")


# Committed lottery tiles. Difficulty is area-normalized, not tile-invariant:
# each message's win chance scales with ``effective_work`` (== rows*cols*(k - k%r)), so
# the 4x64 tile (256-element area) halves the per-message threshold while
# doubling the message count -- total win rate and credited work (m*n*k) are
# identical across tiles. The tiles differ in:
# - kernel reach: only the 4-row family can run mixed_gemm's 64-row kernel
#   tile (the decode small-m fast path; 16-row words cannot be folded by a
#   single thread there);
# - peel-proof size (``(rows+cols)*k*2`` bytes), capping verifiable k under
#   the verifier's 4 MiB worker input: 30720 (4x64), 43520 (16x32),
#   15872 (4x128);
# - n-alignment: ``n % cols`` must be 0.
# 4x64 is preferred wherever it fits (it unlocks the measured 64-row decode
# tile); the tall 16x32 covers what it cannot (n % 32 shapes that are not
# 64-aligned, and k in (30720, 43520]). 4x128 is kept as the shape-omitted
# (``n``-less) commitment for back-compat and as a coarse fallback.
DEFAULT_TILE = LotteryTileSpec(TILE_ROWS, TILE_COLS)
TALL_TILE = LotteryTileSpec(16, 32)
SMALL_TILE = LotteryTileSpec(4, 64)
# Only consumed by ``max_mineable_k`` (a max, so order-independent).
_TILES: tuple[LotteryTileSpec, ...] = (SMALL_TILE, TALL_TILE, DEFAULT_TILE)


def max_verifiable_k(tile_cols: int = TILE_COLS, tile_rows: int = TILE_ROWS) -> int:
    """Largest ``k`` whose peel proof fits the verifier's worker input for a
    committed ``tile_rows x tile_cols`` lottery tile."""
    bytes_per_k = (tile_rows + tile_cols) * _PEEL_ELEM_BYTES
    return (_VERIFIER_MAX_WORKER_INPUT_BYTES // bytes_per_k) // _K_ALIGNMENT * _K_ALIGNMENT


def _tile_max_k(tile: LotteryTileSpec) -> int:
    return max_verifiable_k(tile_cols=tile.cols, tile_rows=tile.rows)


def max_mineable_k() -> int:
    """Largest ``k`` any committed tile can prove -- the complete legal domain
    across all tiles. Sizes the persistent hit signal so no mineable layer's
    proof planes overflow it."""
    return max(_tile_max_k(tile) for tile in _TILES)


def select_tile(n: int, k: int) -> LotteryTileSpec | None:
    """The committed lottery tile for one layer's ``(n, k)`` shape, or ``None``
    if no committed tile can mine it.

    Prefers the 4x64 tile -- the narrowest 4-row commitment, giving the
    64-row kernel tile the runtime routes small-m (decode) launches to its
    smallest tile_n -- then the tall 16x32 for shapes 4x64 cannot take, and
    4x128 last (see the tile-commitment comment above ``_TILES``).

    Anything counting messages or thresholds must use the tile this returns
    rather than assuming a geometry (see :func:`lottery_hashes_per_matmul`
    and :func:`lottery_threshold`).

    TODO(vaktibabat/o-k-d): this preference is a heuristic. Once the kernel
    autotunes tile geometry, select the committed tile from the autotune result
    rather than this fixed order."""
    # Exact integers only: a float/bool ``n`` or ``k`` would pass the arithmetic
    # gates below and be stored into ``mining_configuration`` (whose
    # ``or DEFAULT_TILE`` fallback means a ``None`` return would not stop it), so
    # reject here -- matching the credited-work path's contract.
    for name, value in (("n", n), ("k", k)):
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"lottery dimension {name} must be an int, got {value!r}")
    if not (0 < n < _VERIFIER_MAX_DIM):
        return None
    if k % _K_ALIGNMENT or not (_VERIFIER_MIN_K <= k <= _VERIFIER_MAX_K):
        return None
    for tile in (SMALL_TILE, TALL_TILE, DEFAULT_TILE):
        if n % tile.cols == 0 and k <= _tile_max_k(tile):
            return tile
    return None


def _tall_tile_configuration(k: int) -> MiningConfiguration:
    """The committed 16x32 layout: a 16x1 grid of 1x32 subtiles -- one lane per
    tile row, each lane folding its row's 32 contiguous cols ascending (the
    kernel's thread-local single-row fold, ltile_rows=16 / ltile_cols=32)."""
    return tall_tile_mining_config(
        k,
        RANK,
        Device.BLACKWELL,
        chunk_size=COMMITMENT_CHUNK_SIZE,
        a_chunk_size=COMMITMENT_CHUNK_SIZE,
    )


def mining_configuration(k: int, n: int | None = None) -> MiningConfiguration:
    """The committed mining configuration for one layer's ``(n, k)`` shape.

    ``n`` selects the committed tile via :func:`select_tile` (the 4x64 tile is
    preferred; see there). It is optional: omitting it commits the 4x128 tile
    for back-compat, but the runtime commit path (``job_prep``) always passes
    ``n`` so every layer commits the tile it actually mines.

    Both Merkle trees commit BLAKE3's native 1024-byte leaf
    (``HashId.BLAKE3_CHUNK_1024``). Autotune records may carry a faster hasher
    leaf, but the leaf is committed through each operand's ``HashId``, so GPU
    launches overlay this protocol leaf (see ``with_committed_leaf``).
    """
    return _cached_mining_configuration(k, n)


@cache
def _cached_mining_configuration(k: int, n: int | None) -> MiningConfiguration:
    tile = (select_tile(n, k) if n is not None else DEFAULT_TILE) or DEFAULT_TILE
    if tile == TALL_TILE:
        return _tall_tile_configuration(k)
    # The 4x128 layout has exactly one definition -- the miner-base builder the
    # CPU plain-peel miner commits too. Duplicating it here would let the two
    # drift, silently changing GPU job keys.
    return default_mining_config(
        k,
        RANK,
        ltile_cols=tile.cols,
        ltile_rows=tile.rows,
        chunk_size=COMMITMENT_CHUNK_SIZE,
        a_chunk_size=COMMITMENT_CHUNK_SIZE,
    )


def commitment_keys_for(job: MiningJob) -> tuple[bytes, bytes]:
    """``(keyA, keyB)``: the header's per-side Merkle opening keys.

    v4 keys the trees by the proposed header alone (``H_"key-A"(header)``,
    ``H_"key-B"(header)`` at ancestor depth 0), so every layer of a job shares
    one pair; the layer shape and committed tile enter the chain later, through
    ``pA``/``pB`` in the noise seeds (``miner_base.commitment_hash``)."""
    return commitment_keys(bytes(job.incomplete_header_bytes))


def lottery_threshold(target: int, k: int, n: int | None = None) -> int:
    """``min(target * effective_work(tile), 2^256-1)`` for the committed tile.

    Area-aware: each message's win chance scales with the committed tile's
    ``rows*cols*k``, so the 256-element 4x64 tile halves the per-message
    threshold that the 512-element tiles get (its launches hash twice the
    messages; total win rate is identical). ``n`` selects the committed tile
    exactly like :func:`mining_configuration` -- pass the launch's actual
    matrix dimension everywhere, or a 4x64/16x32-committed layer gets a
    threshold its verifier rejects. Omitting ``n`` matches only the
    back-compat 4x128 commitment.
    """
    tile = (select_tile(n, k) if n is not None else DEFAULT_TILE) or DEFAULT_TILE
    return min(target * effective_work(tile.rows, tile.cols, k, RANK), _MAX_256)


def lottery_hashes_per_matmul(m: int, n: int, tile: LotteryTileSpec = DEFAULT_TILE) -> int:
    """Protocol-credited, in-bounds lottery messages for one padded launch.

    ``tile`` must be the tile the launch actually committed (see
    :func:`select_tile`); the tall tile divides ``n`` by 32, not 128, so
    defaulting it for a tall-tile shape would wrongly reject the shape.
    """
    # Exact integers only: the count is forwarded verbatim as the gateway's
    # credited-hash ``amount`` (bool is an int subclass, so exclude it), and a
    # float would be recorded as a non-integer credit.
    for name, value in (("m", m), ("n", n), ("tile.rows", tile.rows), ("tile.cols", tile.cols)):
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"lottery dimension {name} must be an int, got {value!r}")
    if m <= 0 or n <= 0:
        raise ValueError(f"lottery dimensions must be positive, got m={m}, n={n}")
    if m % tile.rows or n % tile.cols:
        raise ValueError(
            f"lottery dimensions must tile exactly, got m={m}, n={n} vs {tile.rows}x{tile.cols}"
        )
    return (m // tile.rows) * (n // tile.cols)


def effective_work_per_matmul(m: int, n: int, k: int) -> int:
    """Difficulty-normalized credited work for one launch.

    Raw ``lottery_hashes_per_matmul`` counts messages ((m/rows)*(n/cols)); each
    message's win chance is ``target * effective_work(k) / 2^256`` with
    ``effective_work = rows*cols*(k - k % r)`` (v4 floors ``k`` to the rank
    multiple; ``k % 512 == 0`` here so it is ``rows*cols*k``). Weighting
    messages by that work collapses to ``m*n*k`` (tiling-invariant) and is
    what actually predicts wins (``E[wins] = target * m*n*k / 2^256``). Use this for a difficulty-normalized
    hashrate; the raw message count is not comparable across shapes/k.

    The tile's area cancels in that collapse (``(m/rows)*(n/cols) *
    rows*cols*k == m*n*k``), so the result is identical whichever committed
    tile ``(n, k)`` selects -- even though the 4x64 tile has half the area of
    the 512-element tiles -- and the tile only decides which divisions are
    exact.

    Raises ``ValueError`` if ``(n, k)`` is not mineable by any committed tile:
    such a shape can never become an accepted proof, so crediting work for it
    (which a ``DEFAULT_TILE`` fallback would silently do) is never correct.
    """
    if not isinstance(k, int) or isinstance(k, bool):
        raise TypeError(f"lottery common dimension k must be an int, got {k!r}")
    if k <= 0:
        raise ValueError(f"lottery common dimension must be positive, got k={k}")
    tile = select_tile(n, k)
    if tile is None:
        raise ValueError(
            f"({n}, {k}) is not mineable by any committed tile; refusing to credit work"
        )
    return lottery_hashes_per_matmul(m, n, tile) * effective_work(tile.rows, tile.cols, k, RANK)


def threshold_bytes_for(job: MiningJob, k: int, n: int | None = None) -> bytes:
    """32-byte little-endian lottery threshold consumed by ``mixed_gemm``.

    ``n`` selects the committed tile (see :func:`lottery_threshold`); pass the
    layer's actual dimension everywhere, like :func:`mining_configuration`."""
    return lottery_threshold(job.target, k, n).to_bytes(32, "little")


def tile_indices(pattern: AxisPattern, tile_index: int) -> list[int]:
    """Absolute row/col indices of the ``tile_index``-th committed tile,
    in ``mixed_gemm`` hit-record order."""
    base = tile_index * pattern.total
    return [base + off for off in pattern.tile_offsets]


def is_mineable_shape(n: int, k: int) -> bool:
    """Admissibility for both the kernels and the verifier: some committed tile
    accepts ``(n, k)`` -- ``k % 512`` inside that tile's legal ``k`` domain, and
    whole lottery tiles in ``n``. High-``k`` layers up to 30720 (e.g. o_proj,
    k=16384) are admitted via the preferred 4x64 tile; the tall 16x32 extends
    the domain to k=43520 and to 32-but-not-64-aligned ``n``."""
    return select_tile(n, k) is not None
