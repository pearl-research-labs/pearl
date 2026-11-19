"""FP8/v4 mining protocol parameters, transcript hashing, device mapping, and layouts.

Mirrors ``zk-pow/src/v4/api/{layout,public_params,transcript}.rs``.
"""

from __future__ import annotations

import enum
import struct
from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple

import torch
from blake3 import blake3

LANES = 16
MIN_SUBTILE_ELEMS = 16
MAX_SUBTILE_ELEMS = 256
MIN_TILE_ROWS = 4
MIN_TILE_COLS = 16
MAX_TILE_ELEMS = 2048

PROTOCOL_RANK = 32

LABEL_PREFIX = b"pearl/v4/FP8/"
LABEL_KEY_A = b"pearl/v4/FP8/key-A"
LABEL_KEY_B = b"pearl/v4/FP8/key-B"
LABEL_SEED_A = b"pearl/v4/FP8/seed-A"
LABEL_SEED_B = b"pearl/v4/FP8/seed-B"
LABEL_NOISE_LINE = b"pearl/v4/FP8/noise-line"
LABEL_JACKPOT = b"pearl/v4/FP8/jackpot"

_HOPPER_CC_MAJOR = 9
_BLACKWELL_CC_MAJOR = 10
_SM120_CC_MAJOR = 12


class HashId(enum.IntEnum):
    """Whitelisted keyed-BLAKE3 Merkle leaf size and wire discriminant."""

    BLAKE3_CHUNK_128 = 0
    BLAKE3_CHUNK_256 = 1
    BLAKE3_CHUNK_512 = 2
    BLAKE3_CHUNK_1024 = 3

    @property
    def chunk_len(self) -> int:
        return 128 << int(self)

    def pad(self, data: bytes) -> bytes:
        chunk = self.chunk_len
        padded_len = ((len(data) + chunk - 1) // chunk) * chunk
        return data.ljust(padded_len, b"\0")


COMMITMENT_CHUNK_SIZE = HashId.BLAKE3_CHUNK_1024.chunk_len
_HASH_ID_FOR_LEAF = {h.chunk_len: h for h in HashId}


def hash_id_for_leaf(chunk_size: int) -> HashId:
    try:
        return _HASH_ID_FOR_LEAF[chunk_size]
    except KeyError:
        raise ValueError(
            f"no committed Merkle leaf of {chunk_size} bytes; "
            f"cert-v4 opens {sorted(_HASH_ID_FOR_LEAF)}"
        ) from None


class Quant(enum.IntEnum):
    FP8_E4M3_PREQUANT = 0


class Device(enum.IntEnum):
    HOPPER = 0
    BLACKWELL = 1


def device_for_capability(major: int, minor: int = 0) -> Device:
    """Map CUDA compute capability to the committed Device (SM90->Hopper, SM100/120->Blackwell)."""
    if major == _HOPPER_CC_MAJOR:
        return Device.HOPPER
    if major in (_BLACKWELL_CC_MAJOR, _SM120_CC_MAJOR):
        return Device.BLACKWELL
    raise ValueError(f"no committed mining Device for sm{major}{minor}")


def local_device(device: torch.device | int | None = None) -> Device:
    return device_for_capability(*torch.cuda.get_device_capability(device))


def _check_u16(value: int, name: str) -> None:
    if not 0 <= value < 1 << 16:
        raise ValueError(f"{name} must fit u16, got {value}")


def _check_u32(value: int, name: str) -> None:
    if not 0 <= value < 1 << 32:
        raise ValueError(f"{name} must fit u32, got {value}")


@dataclass(frozen=True)
class CommonParams:
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
    num_rows: int
    hash_id: HashId
    tile_size: int

    def __post_init__(self) -> None:
        _check_u32(self.num_rows, "num_rows")
        if self.num_rows == 0:
            raise ValueError("num_rows must be positive")
        _check_u32(self.tile_size, "tile_size")
        if self.tile_size == 0:
            raise ValueError("tile_size must be positive")
        if not isinstance(self.hash_id, HashId):
            raise TypeError("hash_id must be a HashId")


@dataclass(frozen=True)
class MoeParams:
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
    """Encode ``pA = (m, hash_idA, h[, hash_idR, hash_idO])``."""
    out = struct.pack("<IBI", a.num_rows, a.hash_id, a.tile_size)
    if moe is not None:
        out += bytes((moe.hash_id_r, moe.hash_id_o))
    return out


def encode_p_b(common: CommonParams, b: OperandParams, experts: int = 0) -> bytes:
    """Encode ``pB = (k, r, Quant, Device, n, hash_idB, w, e)``."""
    _check_u16(experts, "experts")
    return struct.pack(
        "<IHBBIBIH",
        common.k,
        common.r,
        common.quant,
        common.device,
        b.num_rows,
        b.hash_id,
        b.tile_size,
        experts,
    )


def validate_lottery_layout(a: OperandParams, b: OperandParams) -> None:
    tile_elems = a.tile_size * b.tile_size
    if tile_elems % LANES:
        raise ValueError(f"lottery tile element count must be divisible by {LANES}")
    subtile_elems = tile_elems // LANES
    if not MIN_SUBTILE_ELEMS <= subtile_elems <= MAX_SUBTILE_ELEMS:
        raise ValueError(
            f"subtile has {subtile_elems} elements, "
            f"must be {MIN_SUBTILE_ELEMS}..{MAX_SUBTILE_ELEMS}"
        )
    if a.tile_size < MIN_TILE_ROWS:
        raise ValueError(f"rows tile selects {a.tile_size} strips, min is {MIN_TILE_ROWS}")
    if b.tile_size < MIN_TILE_COLS:
        raise ValueError(f"cols tile selects {b.tile_size} strips, min is {MIN_TILE_COLS}")
    if tile_elems > MAX_TILE_ELEMS:
        raise ValueError(f"lottery tile has {tile_elems} elements, max is {MAX_TILE_ELEMS}")


def lane_assignment(h: int, w: int) -> list[list[int]]:
    """Flat cell indices for the sixteen consecutive row-major fold chunks."""
    if (h * w) % LANES:
        raise ValueError(f"tile element count must be divisible by {LANES}")
    subtile_elems = h * w // LANES
    return [list(range(lane * subtile_elems, (lane + 1) * subtile_elems)) for lane in range(LANES)]


def bits_to_target(nbits: int) -> int:
    """Decode compact ``nBits`` to the verifier's unsigned 256-bit target."""
    exponent = (nbits >> 24) & 0xFF
    mantissa = nbits & 0xFFFFFF
    if exponent == 0 or mantissa == 0 or mantissa & 0x800000:
        return 0
    if exponent <= 3:
        return mantissa >> (8 * (3 - exponent))
    return (mantissa << (8 * (exponent - 3))) & ((1 << 256) - 1)


def encode_u32_le(x: int) -> bytes:
    return struct.pack("<I", x)


def subkey(label: bytes, key: bytes | None = None) -> bytes:
    assert label.startswith(LABEL_PREFIX), "v4 labels must start with pearl/v4/FP8/"
    return blake3(label, key=key).digest() if key is not None else blake3(label).digest()


def hash_labelled(message: bytes, label: bytes, key: bytes | None = None) -> bytes:
    return blake3(message, key=subkey(label, key)).digest()


def commitment_keys(sigma_hat: bytes, sigma_d: bytes) -> tuple[bytes, bytes]:
    """``(keyA, keyB)`` from the 76-byte proposed header and 108-byte ancestor header."""
    if len(sigma_d) != 108:
        raise ValueError(f"sigma_d must be a complete 108-byte header, got {len(sigma_d)} bytes")
    return hash_labelled(sigma_hat, LABEL_KEY_A), hash_labelled(sigma_d, LABEL_KEY_B)


def operand_digest(roots: Sequence[bytes], key: bytes) -> bytes:
    return blake3(b"".join(roots), key=key).digest()


def noise_seed_b(hash_b: bytes, key_b: bytes, p_b: bytes) -> bytes:
    return hash_labelled(hash_b + key_b + p_b, LABEL_SEED_B)


def noise_seed_a(
    hash_a: bytes, seed_b: bytes, key_a: bytes, p_a: bytes, routing_commitments: bytes = b""
) -> bytes:
    if len(routing_commitments) not in (0, 64):
        raise ValueError("routing_commitments must be empty or the 64-byte HR || HO")
    return hash_labelled(hash_a + routing_commitments + seed_b + key_a + p_a, LABEL_SEED_A)


def noise_line_key(seed: bytes) -> bytes:
    return subkey(LABEL_NOISE_LINE, seed)


def jackpot_key(seed_a: bytes) -> bytes:
    return subkey(LABEL_JACKPOT, seed_a)


class AKeys(NamedTuple):
    """Everything downstream of ``HA``: matches the finalize kernel's 96-byte output."""

    seed_a: bytes
    noise_line_key: bytes
    jackpot_key: bytes

    @classmethod
    def from_bytes(cls, raw: bytes) -> AKeys:
        if len(raw) != 96:
            raise ValueError(f"a_keys is 96 bytes, got {len(raw)}")
        return cls(bytes(raw[0:32]), bytes(raw[32:64]), bytes(raw[64:96]))

    def to_bytes(self) -> bytes:
        return self.seed_a + self.noise_line_key + self.jackpot_key


def a_keys(
    hash_a: bytes, seed_b: bytes, key_a: bytes, p_a: bytes, routing_commitments: bytes = b""
) -> AKeys:
    seed_a = noise_seed_a(hash_a, seed_b, key_a, p_a, routing_commitments)
    return AKeys(seed_a, noise_line_key(seed_a), jackpot_key(seed_a))


@dataclass
class MiningConfiguration:
    """Parameters committed per layer's launches and proofs."""

    device: Device
    common_dim: int
    rank: int
    tile_rows: int
    tile_cols: int
    chunk_size: int = COMMITMENT_CHUNK_SIZE
    a_chunk_size: int | None = None
    experts: int = 0

    def __post_init__(self) -> None:
        if self.a_chunk_size is None:
            self.a_chunk_size = self.chunk_size
        if self.rank != PROTOCOL_RANK:
            raise ValueError(f"cert-v4 mines at rank {PROTOCOL_RANK} only, got rank={self.rank}")
        validate_lottery_layout(self.a_params(1), self.b_params(1))

    @property
    def a_hash_id(self) -> HashId:
        assert self.a_chunk_size is not None
        return hash_id_for_leaf(self.a_chunk_size)

    @property
    def b_hash_id(self) -> HashId:
        return hash_id_for_leaf(self.chunk_size)

    def common_params(self) -> CommonParams:
        return CommonParams(self.common_dim, self.rank, Quant.FP8_E4M3_PREQUANT, self.device)

    def a_params(self, m: int) -> OperandParams:
        return OperandParams(m, self.a_hash_id, self.tile_rows)

    def b_params(self, n: int) -> OperandParams:
        return OperandParams(n, self.b_hash_id, self.tile_cols)

    def moe_params(self) -> MoeParams | None:
        if not self.experts:
            return None
        return MoeParams(self.experts, self.a_hash_id, self.a_hash_id)

    def p_a(self, m: int) -> bytes:
        return encode_p_a(self.a_params(m), self.moe_params())

    def p_b(self, n: int) -> bytes:
        return encode_p_b(self.common_params(), self.b_params(n), experts=self.experts)


def default_mining_config(
    k: int,
    rank: int,
    ltile_cols: int = 128,
    device: Device = Device.BLACKWELL,
    ltile_rows: int = 4,
    chunk_size: int = COMMITMENT_CHUNK_SIZE,
    a_chunk_size: int | None = None,
) -> MiningConfiguration:
    assert ltile_cols in (64, 128, 192, 256)
    return MiningConfiguration(
        device=device,
        common_dim=k,
        rank=rank,
        tile_rows=ltile_rows,
        tile_cols=ltile_cols,
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
    return MiningConfiguration(
        device=device,
        common_dim=k,
        rank=rank,
        tile_rows=16,
        tile_cols=32,
        chunk_size=chunk_size,
        a_chunk_size=a_chunk_size,
    )
