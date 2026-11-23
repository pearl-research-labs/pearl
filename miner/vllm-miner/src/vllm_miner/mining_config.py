"""Per-layer committed lottery tiles, thresholds, and credited-work formulas."""

from dataclasses import dataclass, replace
from functools import cache

from miner_base.prequant import DEFAULT_BLOCK_SIZE
from miner_base.protocol import (
    COMMITMENT_CHUNK_SIZE,
    Device,
    MiningConfiguration,
    commitment_keys,
    default_mining_config,
    tall_tile_mining_config,
)
from pearl_gateway.comm.dataclasses import MiningJob

TILE_ROWS = 4
TILE_COLS = 128
RANK = 32
PACKED_NOISE_K = 32
SCALE_BLOCK = DEFAULT_BLOCK_SIZE
MOE_LOTTERY_N = 128

_MAX_256 = (1 << 256) - 1
_VERIFIER_MIN_K = 1024
_VERIFIER_MAX_K = 1 << 16
# Largest 512-aligned k per tile whose Blake3 witness fits the verifier's 2^22 byte bound.
_VERIFIER_TILE_MAX_K = {(4, 64): 42496, (16, 32): 58368, (4, 128): 22016}
_K_ALIGNMENT = 512
_VERIFIER_MAX_DIM = 1 << 24


def _require_int(value: object, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an int, got {value!r}")
    return value


@dataclass(frozen=True)
class LotteryTileSpec:
    rows: int
    cols: int

    def __post_init__(self) -> None:
        for name, value in (("rows", self.rows), ("cols", self.cols)):
            _require_int(value, f"lottery tile {name}")
            if value <= 0:
                raise ValueError(f"lottery tile {name} must be positive, got {value}")


DEFAULT_TILE = LotteryTileSpec(TILE_ROWS, TILE_COLS)
TALL_TILE = LotteryTileSpec(16, 32)
SMALL_TILE = LotteryTileSpec(4, 64)
_TILES: tuple[LotteryTileSpec, ...] = (SMALL_TILE, TALL_TILE, DEFAULT_TILE)


def _committed_tiles(device: Device) -> tuple[LotteryTileSpec, ...]:
    # Hopper WGMMA only implements the 4-row tile family; 16x32 is Blackwell-only.
    if device is Device.HOPPER:
        return (SMALL_TILE, DEFAULT_TILE)
    if device is Device.BLACKWELL:
        return (SMALL_TILE, TALL_TILE, DEFAULT_TILE)
    raise ValueError(f"no committed lottery tiles for {device!r}")


def max_verifiable_k(tile_cols: int = TILE_COLS, tile_rows: int = TILE_ROWS) -> int:
    return _VERIFIER_TILE_MAX_K[(tile_rows, tile_cols)]


def max_mineable_k() -> int:
    return max(_VERIFIER_TILE_MAX_K[(t.rows, t.cols)] for t in _TILES)


def select_tile(n: int, k: int, *, device: Device) -> LotteryTileSpec | None:
    """Preferred committed lottery tile for (n, k) on ``device``, or None if unmineable."""
    _require_int(n, "lottery dimension n")
    _require_int(k, "lottery dimension k")
    if not (0 < n < _VERIFIER_MAX_DIM):
        return None
    if k % _K_ALIGNMENT or not (_VERIFIER_MIN_K <= k <= _VERIFIER_MAX_K):
        return None
    for tile in _committed_tiles(device):
        if n % tile.cols == 0 and k <= _VERIFIER_TILE_MAX_K[(tile.rows, tile.cols)]:
            return tile
    return None


def expert_n(n: int, experts: int) -> int:
    if experts and n % experts:
        raise ValueError(f"stacked n={n} is not a multiple of {experts} experts")
    return n // experts if experts else n


@cache
def mining_configuration(
    k: int, n: int | None = None, experts: int = 0, *, device: Device
) -> MiningConfiguration:
    n_e = None if n is None else expert_n(n, experts)
    tile = (select_tile(n_e, k, device=device) if n_e is not None else DEFAULT_TILE) or DEFAULT_TILE
    if tile == TALL_TILE:
        cfg = tall_tile_mining_config(
            k, RANK, device, chunk_size=COMMITMENT_CHUNK_SIZE, a_chunk_size=COMMITMENT_CHUNK_SIZE
        )
    else:
        cfg = default_mining_config(
            k,
            RANK,
            ltile_cols=tile.cols,
            ltile_rows=tile.rows,
            device=device,
            chunk_size=COMMITMENT_CHUNK_SIZE,
            a_chunk_size=COMMITMENT_CHUNK_SIZE,
        )
    return replace(cfg, experts=experts) if experts else cfg


def commitment_keys_for(
    job: MiningJob, ancestor_header: bytes | None = None
) -> tuple[bytes, bytes]:
    if ancestor_header is None:
        ancestor_header = job.parent_header
    return commitment_keys(bytes(job.incomplete_header_bytes), bytes(ancestor_header))


def lottery_threshold(target: int, k: int, n: int | None = None, *, device: Device) -> int:
    tile = (select_tile(n, k, device=device) if n is not None else DEFAULT_TILE) or DEFAULT_TILE
    return min(target * tile.rows * tile.cols * k, _MAX_256)


def lottery_hashes_per_matmul(m: int, n: int, tile: LotteryTileSpec = DEFAULT_TILE) -> int:
    for name, value in (("m", m), ("n", n), ("tile.rows", tile.rows), ("tile.cols", tile.cols)):
        _require_int(value, f"lottery dimension {name}")
    if m <= 0 or n <= 0:
        raise ValueError(f"lottery dimensions must be positive, got m={m}, n={n}")
    if m % tile.rows or n % tile.cols:
        raise ValueError(
            f"lottery dimensions must tile exactly, got m={m}, n={n} vs {tile.rows}x{tile.cols}"
        )
    return (m // tile.rows) * (n // tile.cols)


def effective_work_per_matmul(m: int, n: int, k: int, *, device: Device) -> int:
    """Difficulty-normalized credited work (m * n * k) for one mineable launch."""
    _require_int(k, "lottery common dimension k")
    if k <= 0:
        raise ValueError(f"lottery common dimension must be positive, got k={k}")
    tile = select_tile(n, k, device=device)
    if tile is None:
        raise ValueError(
            f"({n}, {k}) is not mineable by any committed tile; refusing to credit work"
        )
    return lottery_hashes_per_matmul(m, n, tile) * tile.rows * tile.cols * k


def threshold_bytes_for(job: MiningJob, k: int, n: int | None = None, *, device: Device) -> bytes:
    return lottery_threshold(job.target, k, n, device=device).to_bytes(32, "little")


def tile_indices(size: int, tile_index: int) -> list[int]:
    base = tile_index * size
    return list(range(base, base + size))


def is_mineable_shape(n: int, k: int, *, device: Device) -> bool:
    return select_tile(n, k, device=device) is not None
