"""The committed lottery layout for the FP8 miners.

:func:`default_mining_config` builds the lottery layout the GPU miner uses
(the merged tile must match the ``mixed_gemm`` kernel; see
``miner/pearl-gemm``).
"""

from .commitment import Device, HashId, MiningConfiguration
from .layout import AxisPattern, DimType

# Rows per merged lottery tile. This is the ``mixed_gemm`` kernel's
# ``DEFAULT_LTILE_ROWS`` constant (pearl_gemm.mixed_gemm), duplicated here so this
# module stays free of the CUDA-kernel dependency; the GPU miner passes the
# kernel constant explicitly.
DEFAULT_LTILE_ROWS = 4

# Column widths the ``mixed_gemm`` kernel supports for a merged lottery tile
# (its ``ltile_cols`` config values).
SUPPORTED_LTILE_COLS = (64, 128, 192, 256)

# Tall 16x32 lottery tile (mixed_gemm's ltile_rows=16, ltile_cols=32 variant).
TALL_TILE_ROWS = 16
TALL_TILE_COLS = 32

# Blake lanes on the 4x128 family: four column subtiles.
_COLS_BLAKE = 4
# Inner fold digit of the 4x128 family (interleaved pairs inside each 8-col group).
_COLS_INNER_FOLD = 2

# Merkle leaf every miner on this pin commits by default: BLAKE3's native
# 1024-byte chunk (``HashId.BLAKE3_CHUNK_1024``). The verifier also opens
# 128/256/512-byte leaves; each tree's leaf is committed through its
# operand's ``HashId``, so the A and B trees may differ.
COMMITMENT_CHUNK_SIZE = HashId.BLAKE3_CHUNK_1024.chunk_len


def activation_leaf(config: MiningConfiguration) -> int:
    """The Merkle leaf the A (activation) tree commits under ``config``."""
    return config.a_chunk_size or config.chunk_size


def default_mining_config(
    k: int,
    rank: int,
    ltile_cols: int = 128,
    device: Device = Device.BLACKWELL,
    ltile_rows: int = DEFAULT_LTILE_ROWS,
    chunk_size: int = COMMITMENT_CHUNK_SIZE,
    a_chunk_size: int | None = None,
) -> MiningConfiguration:
    """The committed lottery layout implemented by the ``mixed_gemm`` kernel.

    One merged tile is ``ltile_rows`` contiguous rows x ``ltile_cols``
    contiguous cols. ``ltile_cols`` is a per-job choice committed into the
    operands' ``pA``/``pB`` and must match the kernel config's ``ltile_cols``
    when the tiles are drawn by the kernel. ``chunk_size`` is the weight (B)
    tree's Merkle leaf; ``a_chunk_size`` (defaulting to it) is the activation
    (A) tree's.
    """
    assert ltile_cols in SUPPORTED_LTILE_COLS
    assert ltile_cols % 8 == 0
    rows = AxisPattern(((ltile_rows, DimType.BLAKE),))
    cols = AxisPattern(
        (
            (_COLS_INNER_FOLD, DimType.FOLD),
            (_COLS_BLAKE, DimType.BLAKE),
            (ltile_cols // 8, DimType.FOLD),
        )
    )
    return MiningConfiguration(
        device=device,
        common_dim=k,
        rank=rank,
        rows_pattern=rows,
        cols_pattern=cols,
        chunk_size=chunk_size,
        a_chunk_size=a_chunk_size,
    )


def tall_tile_mining_config(
    k: int,
    rank: int,
    device: Device = Device.BLACKWELL,
    chunk_size: int = COMMITMENT_CHUNK_SIZE,
    a_chunk_size: int | None = None,
) -> MiningConfiguration:
    """The committed 16x32 layout: one Blake lane per tile row, each folding
    32 contiguous columns. Same 512-element area as 4x128."""
    return MiningConfiguration(
        device=device,
        common_dim=k,
        rank=rank,
        rows_pattern=AxisPattern(((TALL_TILE_ROWS, DimType.BLAKE),)),
        cols_pattern=AxisPattern(((TALL_TILE_COLS, DimType.FOLD),)),
        chunk_size=chunk_size,
        a_chunk_size=a_chunk_size,
    )
