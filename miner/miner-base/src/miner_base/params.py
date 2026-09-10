"""Protocol parameters committed by the FP8/v4 transcript.

These records mirror the verifier's ``CommonParams``, ``OperandParams``, and
``MoeParams``. They are deliberately separate: v4 has no serialized
``MiningConfiguration`` aggregate.
"""

from __future__ import annotations

import enum
import struct
from dataclasses import dataclass

from .hardware import Blackwell, Hardware
from .layout import (
    LANES,
    MAX_SUBTILE_ELEMS,
    MAX_TILE_ELEMS,
    MIN_SUBTILE_ELEMS,
    MIN_TILE_COLS,
    MIN_TILE_ROWS,
    AxisPattern,
)


class HashId(enum.IntEnum):
    """Whitelisted keyed-BLAKE3 Merkle leaf size and wire discriminant."""

    BLAKE3_CHUNK_128 = 0
    BLAKE3_CHUNK_256 = 1
    BLAKE3_CHUNK_512 = 2
    BLAKE3_CHUNK_1024 = 3

    @property
    def chunk_len(self) -> int:
        return {
            HashId.BLAKE3_CHUNK_128: 128,
            HashId.BLAKE3_CHUNK_256: 256,
            HashId.BLAKE3_CHUNK_512: 512,
            HashId.BLAKE3_CHUNK_1024: 1024,
        }[self]

    def pad(self, data: bytes) -> bytes:
        padded_len = ((len(data) + self.chunk_len - 1) // self.chunk_len) * self.chunk_len
        return data.ljust(padded_len, b"\0")


class Quant(enum.IntEnum):
    """Committed input quantization scheme."""

    FP8_E4M3_PREQUANT = 0


class Device(enum.IntEnum):
    """Committed mining device and concrete arithmetic implementation."""

    BLACKWELL = 1

    def hardware(self) -> Hardware:
        return Blackwell()


@dataclass(frozen=True)
class BlockHeader:
    """The 76-byte incomplete block header that keys the v4 transcript."""

    version: int
    prev_block: bytes
    merkle_root: bytes
    timestamp: int
    nbits: int

    def __post_init__(self) -> None:
        _check_u32(self.version, "version")
        _check_u32(self.timestamp, "timestamp")
        _check_u32(self.nbits, "nbits")
        if len(self.prev_block) != 32:
            raise ValueError("prev_block must be exactly 32 bytes")
        if len(self.merkle_root) != 32:
            raise ValueError("merkle_root must be exactly 32 bytes")

    def to_bytes(self) -> bytes:
        return (
            struct.pack("<I", self.version)
            + self.prev_block[::-1]
            + self.merkle_root[::-1]
            + struct.pack("<II", self.timestamp, self.nbits)
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> BlockHeader:
        if len(data) != 76:
            raise ValueError(f"block header must be exactly 76 bytes, got {len(data)}")
        version = struct.unpack_from("<I", data, 0)[0]
        timestamp, nbits = struct.unpack_from("<II", data, 68)
        return cls(version, data[4:36][::-1], data[36:68][::-1], timestamp, nbits)


@dataclass(frozen=True)
class CommonParams:
    """Fields shared by both operands in ``pB``."""

    k: int
    r: int
    quant: Quant
    device: Device

    def __post_init__(self) -> None:
        _check_u32(self.k, "k")
        _check_u16(self.r, "r")
        if not isinstance(self.quant, Quant):
            raise TypeError("quant must be a Quant")
        if not isinstance(self.device, Device):
            raise TypeError("device must be a Device")


@dataclass(frozen=True)
class OperandParams:
    """Committed row count, Merkle layout, and lottery pattern for one operand."""

    num_rows: int
    hash_id: HashId
    pattern: AxisPattern

    def __post_init__(self) -> None:
        _check_u32(self.num_rows, "num_rows")
        if self.num_rows == 0:
            raise ValueError("num_rows must be positive")
        if not isinstance(self.hash_id, HashId):
            raise TypeError("hash_id must be a HashId")


@dataclass(frozen=True)
class MoeParams:
    """MoE expert count and independent routing/offset hash layouts."""

    experts: int
    hash_id_r: HashId
    hash_id_o: HashId

    def __post_init__(self) -> None:
        if not 1 <= self.experts <= 1024:
            raise ValueError(f"MoE expert count {self.experts} not in 1..=1024")
        if not isinstance(self.hash_id_r, HashId):
            raise TypeError("hash_id_r must be a HashId")
        if not isinstance(self.hash_id_o, HashId):
            raise TypeError("hash_id_o must be a HashId")


def encode_p_a(a: OperandParams, moe: MoeParams | None = None) -> bytes:
    """Encode ``pA = (m, hash_idA, Prow[, hash_idR, hash_idO])``."""
    out = struct.pack("<IB", a.num_rows, a.hash_id) + a.pattern.to_bytes()
    if moe is not None:
        out += bytes((moe.hash_id_r, moe.hash_id_o))
    return out


def encode_p_b(common: CommonParams, b: OperandParams, experts: int = 0) -> bytes:
    """Encode ``pB = (n, k, r, Quant, Device, hash_idB, Pcol, e)``."""
    _check_u16(experts, "experts")
    return (
        struct.pack(
            "<IIHBBB",
            b.num_rows,
            common.k,
            common.r,
            common.quant,
            common.device,
            b.hash_id,
        )
        + b.pattern.to_bytes()
        + struct.pack("<H", experts)
    )


def validate_lottery_layout(a: OperandParams, b: OperandParams) -> None:
    """Validate the cross-axis lottery geometry shared by dense and MoE mining."""
    n_lanes = a.pattern.blake_size * b.pattern.blake_size
    if n_lanes != LANES:
        raise ValueError(f"Blake dims must select exactly {LANES} subtiles, got {n_lanes}")
    subtile_elems = a.pattern.fold_size * b.pattern.fold_size
    if not MIN_SUBTILE_ELEMS <= subtile_elems <= MAX_SUBTILE_ELEMS:
        raise ValueError(
            f"subtile has {subtile_elems} elements, "
            f"must be {MIN_SUBTILE_ELEMS}..{MAX_SUBTILE_ELEMS}"
        )
    if a.pattern.tile_size < MIN_TILE_ROWS:
        raise ValueError(f"rows tile selects {a.pattern.tile_size} strips, min is {MIN_TILE_ROWS}")
    if b.pattern.tile_size < MIN_TILE_COLS:
        raise ValueError(f"cols tile selects {b.pattern.tile_size} strips, min is {MIN_TILE_COLS}")
    tile_elems = a.pattern.tile_size * b.pattern.tile_size
    if tile_elems > MAX_TILE_ELEMS:
        raise ValueError(f"lottery tile has {tile_elems} elements, max is {MAX_TILE_ELEMS}")


def _check_u16(value: int, name: str) -> None:
    if not 0 <= value < 1 << 16:
        raise ValueError(f"{name} must fit u16, got {value}")


def _check_u32(value: int, name: str) -> None:
    if not 0 <= value < 1 << 32:
        raise ValueError(f"{name} must fit u32, got {value}")
