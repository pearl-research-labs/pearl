"""Committed extractor layout: typed mixed-radix axis patterns.

Twin of the Rust verifier's ``zk-pow/src/api/layout.rs``.
An :class:`AxisPattern` is an ordered list of dims ``(length, DimType)`` with
implicit stride = product of the preceding lengths, covering one period
``[0..total)``. ``Fold`` dims span one subtile (folded by a single extractor
lane), ``Blake`` dims enumerate the subtiles (lanes), ``Null`` dims are the
free placement digits of the tile base offset within one period.

Placement is periodic: valid tile bases along an axis are
``lattice_point + q * total`` for any ``q >= 0``, where ``lattice_point``
ranges over the Null-digit lattice of one period. The quotient above the top
explicit dim is an implicit free digit, so a trailing Null dim is redundant
(rejected by construction). Matrix bounds are the caller's responsibility
(the verifier's sanity check ``t_rows + tile_max < m``).

The lottery tile per axis is the direct sum of the Fold and Blake digit
offsets; illegal layouts (overlapping subtile/grid) are unrepresentable by
construction.
"""

from __future__ import annotations

import enum
import math
from collections.abc import Callable
from dataclasses import dataclass

# The extracted lottery message is LANES x u32 = 64 bytes: one BLAKE3 block.
LANES = 16

# Per-subtile element bound. Mirrors MAX_SUBTILE_ELEMS in
# zk-pow/src/api/layout.rs.
MAX_SUBTILE_ELEMS = 256

# Lower bound on the per-subtile element count (fold product across both
# axes): each lane must fold at least this much committed data.
MIN_SUBTILE_ELEMS = 16

# Minimum rows tile size (tile_size = fold_size * blake_size): the number of
# strips opened from the activations matrix.
MIN_TILE_ROWS = 4

# Minimum cols tile size: the number of strips opened from the weight matrix.
MIN_TILE_COLS = 16

# Upper bound on the whole lottery tile (rows.tile_size * cols.tile_size).
MAX_TILE_ELEMS = 2048

# Upper bound on AxisPattern.total (one period of the placement lattice;
# keeps `offset mod total` arithmetic in u32).
MAX_PATTERN_TOTAL = 1 << 24

# Largest dim length a single serialized dim byte can carry.
_MAX_DIM_LEN = 64


class DimType(enum.IntEnum):
    """The role of one mixed-radix dim in the committed layout."""

    # Free placement digit: the tile may be based at any value of this digit.
    NULL = 0
    # Subtile digit: folded (summed) by a single extractor lane.
    FOLD = 1
    # Grid digit: enumerates the extractor lanes (subtiles).
    BLAKE = 2
    # Wire padding for unused trailing dim slots in the fixed-size encoding;
    # never carries a length > 1, never appears in a canonical dims list.
    NONE = 3


def _encode(dims: list[tuple[int, DimType]]) -> bytes:
    """The committed byte form of a canonical dims list: exactly
    ``AxisPattern.NUM_DIMS`` bytes, one per dim, ``(length - 1) << 2 | type``,
    unused trailing slots padded with the ``NONE`` byte. Dims longer than 64
    are split greedily (largest divisor <= 64 of the remaining run first).
    A dims list is encodable iff every dim length has no prime factor > 64
    and the split yields at most ``AxisPattern.NUM_DIMS`` dim bytes.
    """
    dim_bytes = bytearray()
    for length, dim_type in dims:
        rest = length
        while rest > 1:
            part = next((d for d in range(min(_MAX_DIM_LEN, rest), 1, -1) if rest % d == 0), None)
            if part is None:
                raise ValueError(
                    f"dim length {length} has a prime factor > {_MAX_DIM_LEN} "
                    "and cannot be serialized"
                )
            dim_bytes.append((part - 1) << 2 | int(dim_type))
            rest //= part
    if len(dim_bytes) > AxisPattern.NUM_DIMS:
        raise ValueError(f"pattern needs more than {AxisPattern.NUM_DIMS} dim bytes to serialize")
    return bytes(dim_bytes) + bytes([int(DimType.NONE)]) * (AxisPattern.NUM_DIMS - len(dim_bytes))


@dataclass(frozen=True)
class AxisPattern:
    """A typed mixed-radix pattern for one axis (see module docstring).

    Canonical form (normalization in ``__post_init__``): every dim length
    ``>= 2`` and no two adjacent dims of the same type (length-1 dims are
    dropped, adjacent same-type dims merged). ``total <= 2^24``,
    serializability, and a non-trailing ``Null`` dim are enforced at
    construction (a trailing Null is redundant). ``NONE`` dims are wire
    padding only and must have length 1.

    The canonical dims are the single source of truth; every derived
    quantity (total, digit offsets, encoding) is computed on demand.
    """

    # Number of dim slots in the fixed-size wire encoding (one byte each).
    NUM_DIMS = 6

    dims: tuple[tuple[int, DimType], ...]

    def __post_init__(self) -> None:
        canonical: list[tuple[int, DimType]] = []
        total = 1
        for length, dim_type in self.dims:
            length, dim_type = int(length), DimType(dim_type)
            if length < 1:
                raise ValueError("dim length must be >= 1")
            if dim_type is DimType.NONE and length != 1:
                raise ValueError("None dims are wire padding only and must have length 1")
            total *= length
            if total > MAX_PATTERN_TOTAL:
                raise ValueError(f"pattern total {total} exceeds 2^24")
            if length == 1:
                continue
            if canonical and canonical[-1][1] == dim_type:
                canonical[-1] = (canonical[-1][0] * length, dim_type)
            else:
                canonical.append((length, dim_type))

        if canonical and canonical[-1][1] is DimType.NULL:
            raise ValueError("trailing Null dim is redundant; omit it")

        # Serializability is a construction-time invariant: reject patterns
        # whose encoding cannot be produced, so ``to_bytes`` cannot fail.
        _encode(canonical)

        object.__setattr__(self, "dims", tuple(canonical))

    @property
    def total(self) -> int:
        """Product of all dim lengths: one period of the placement lattice."""
        return math.prod(length for length, _ in self.dims)

    def _offsets_where(self, include: Callable[[DimType], bool]) -> list[int]:
        """Sorted offsets generated by the dims selected by ``include``: sums
        of ``{digit * stride}`` over those dims, stride the running product of
        lengths. Mixed radix makes each new dim's stride exceed every prior
        offset, so the per-digit blocks are ascending and disjoint — appended
        in place, no sorting needed."""
        offsets = [0]
        stride = 1
        for length, t in self.dims:
            if include(t):
                base = len(offsets)
                offsets += [offsets[i] + d * stride for d in range(1, length) for i in range(base)]
            stride *= length
        return offsets

    @property
    def fold_offsets(self) -> list[int]:
        """Sorted offsets generated by the Fold dims (within-subtile offsets)."""
        return self._offsets_where(lambda t: t is DimType.FOLD)

    @property
    def blake_offsets(self) -> list[int]:
        """Sorted offsets generated by the Blake dims (subtile base offsets)."""
        return self._offsets_where(lambda t: t is DimType.BLAKE)

    @property
    def tile_offsets(self) -> list[int]:
        """Sorted offsets of the lottery tile: the Fold and Blake digit offsets."""
        return self._offsets_where(lambda t: t is not DimType.NULL)

    def _dim_product(self, dim_type: DimType) -> int:
        """Product of the dim lengths of one type."""
        return math.prod(length for length, t in self.dims if t is dim_type)

    @property
    def fold_size(self) -> int:
        """Product of the Fold dim lengths (elements per subtile on this axis)."""
        return self._dim_product(DimType.FOLD)

    @property
    def blake_size(self) -> int:
        """Product of the Blake dim lengths (subtiles on this axis)."""
        return self._dim_product(DimType.BLAKE)

    @property
    def tile_size(self) -> int:
        """Number of rows/cols the tile selects (``fold_size * blake_size``)."""
        return self.fold_size * self.blake_size

    @property
    def tile_max(self) -> int:
        """Largest offset the tile selects."""
        return self.tile_offsets[-1]

    def offset_is_valid(self, offset: int) -> bool:
        """Whether ``offset`` is a valid tile base: every Fold/Blake digit of
        ``offset mod total`` is zero (Null digits are free, including the
        implicit quotient above the top dim), and one full period from the
        base fits in ``u32`` (``offset + total <= 2**32``). Matrix-fit is the
        caller's responsibility."""
        if offset + self.total > 1 << 32:
            return False
        if offset < 0:
            return False
        rest = offset
        for length, dim_type in self.dims:
            digit = rest % length
            rest //= length
            if digit != 0 and dim_type is not DimType.NULL:
                return False
        return True

    def valid_offsets(self) -> list[int]:
        """All valid tile base offsets within one period ``[0..total)``, sorted:
        the Null-digit lattice. The full placement set is
        ``lattice_point + q * total``."""
        return self._offsets_where(lambda t: t is DimType.NULL)

    def to_bytes(self) -> bytes:
        """The committed byte form (see ``_encode``): exactly ``NUM_DIMS``
        bytes. Cannot fail: serializability is enforced at construction."""
        return _encode(self.dims)

    @classmethod
    def from_bytes(cls, data: bytes) -> AxisPattern:
        """Parse a pattern from its fixed-size committed byte form. Enforces
        canonicality by round-trip: the bytes must equal the re-serialization
        (which also rejects non-canonical padding: a ``NONE`` dim with length
        > 1, a ``NONE`` dim before a real dim, or any other pad byte value).
        """
        if len(data) != cls.NUM_DIMS:
            raise ValueError(f"Expected {cls.NUM_DIMS} bytes, got {len(data)}")
        dims = [((byte >> 2) + 1, DimType(byte & 3)) for byte in data]
        pattern = cls(tuple(dims))
        if pattern.to_bytes() != data:
            raise ValueError(f"non-canonical pattern encoding: {data.hex()}")
        return pattern


def lane_assignment(rows: AxisPattern, cols: AxisPattern) -> list[list[int]]:
    """Map each lane to its subtile's flat tile indices, in the pinned fold order:
    lanes row-major over sorted ``(blake_row, blake_col)`` offsets, each folding
    its subtile row-major over sorted ``(fold_row, fold_col)`` offsets, tile
    indexed by the sorted tile offsets."""
    rows_tile, cols_tile = rows.tile_offsets, cols.tile_offsets
    row_pos = {off: i for i, off in enumerate(rows_tile)}
    col_pos = {off: j for j, off in enumerate(cols_tile)}
    n_cols = len(cols_tile)
    rows_fold, cols_fold = rows.fold_offsets, cols.fold_offsets
    rows_blake, cols_blake = rows.blake_offsets, cols.blake_offsets

    return [
        [row_pos[a_r + b_r] * n_cols + col_pos[a_c + b_c] for a_r in rows_fold for a_c in cols_fold]
        for b_r in rows_blake
        for b_c in cols_blake
    ]
