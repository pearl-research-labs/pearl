#!/usr/bin/env python3
"""Akoya pool wire protocol helpers.

This module is intentionally pure Python and does not import the Pearl Rust
extension. It is the local gate for P1K-131: decode captured accepted-share
fixtures, validate the inferred PlainProofShare schema, and build structurally
correct Akoya frames before any GPU run.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import socket
import struct
from typing import Any
import uuid

import msgpack


TYPE_REGISTER = 0
TYPE_REGISTER_ACK = 1
TYPE_JOB_ASSIGNMENT = 2
TYPE_PLAIN_PROOF_SHARE = 3
TYPE_SHARE_RESULT = 4
TYPE_TELEMETRY = 5
TYPE_POOL_STATUS = 6
TYPE_DIFFICULTY_ADJUST = 7

FRAME_PREFIX_SIZE = 4
HEADER_SIZE = 76
HASH_SIZE = 32
MINING_CONFIG_SIZE = 52
BLAKE3_CHUNK_SIZE = 1024


class AkoyaProtocolError(ValueError):
    """Raised when an Akoya frame is structurally invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AkoyaProtocolError(message)


def pack_payload(type_code: int, fields: list[Any]) -> bytes:
    """Serialize a top-level Akoya payload: [type_code, fields].

    Captured Akoya frames encode the top-level type code as MessagePack int32
    (`0xd2` + four bytes) even for small values like 0 and 3.
    """

    _require(isinstance(type_code, int), "type_code must be int")
    _require(-(1 << 31) <= type_code < (1 << 31), "type_code must fit signed int32")
    _require(isinstance(fields, list), "fields must be list")
    return b"\x92\xd2" + struct.pack(">i", type_code) + msgpack.packb(fields, use_bin_type=True)


def pack_frame(type_code: int, fields: list[Any]) -> bytes:
    """Serialize an Akoya frame with a 4-byte big-endian length prefix."""

    payload = pack_payload(type_code, fields)
    return len(payload).to_bytes(FRAME_PREFIX_SIZE, "big") + payload


def unpack_payload(payload: bytes) -> tuple[int, list[Any]]:
    """Decode a MessagePack payload and validate the top-level shape."""

    obj = msgpack.unpackb(payload, raw=False)
    _require(isinstance(obj, list) and len(obj) == 2, "payload must be [type_code, fields]")
    type_code, fields = obj
    _require(isinstance(type_code, int), "type_code must be int")
    _require(isinstance(fields, list), "fields must be list")
    return type_code, fields


def unpack_frame(frame: bytes) -> tuple[int, list[Any]]:
    """Decode a full length-prefixed Akoya frame."""

    _require(len(frame) >= FRAME_PREFIX_SIZE, "frame is shorter than length prefix")
    payload_len = int.from_bytes(frame[:FRAME_PREFIX_SIZE], "big")
    payload = frame[FRAME_PREFIX_SIZE:]
    _require(payload_len == len(payload), f"frame length prefix {payload_len} != payload {len(payload)}")
    return unpack_payload(payload)


def read_exact(sock: socket.socket, nbytes: int) -> bytes:
    """Read exactly nbytes from a socket or raise EOFError."""

    chunks: list[bytes] = []
    remaining = nbytes
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise EOFError(f"socket closed while reading {nbytes} bytes")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def read_frame(sock: socket.socket) -> tuple[int, list[Any]]:
    """Read and decode one length-prefixed Akoya frame from a socket."""

    prefix = read_exact(sock, FRAME_PREFIX_SIZE)
    payload_len = int.from_bytes(prefix, "big")
    return unpack_payload(read_exact(sock, payload_len))


def _bytes(value: Any, size: int | None, name: str) -> bytes:
    _require(isinstance(value, (bytes, bytearray)), f"{name} must be bytes")
    value = bytes(value)
    if size is not None:
        _require(len(value) == size, f"{name} must be {size} bytes, got {len(value)}")
    return value


def _int(value: Any, name: str) -> int:
    _require(isinstance(value, int), f"{name} must be int")
    return value


def _str(value: Any, name: str) -> str:
    _require(isinstance(value, str), f"{name} must be str")
    return value


def _uuid(value: Any, name: str) -> str:
    text = _str(value, name)
    uuid.UUID(text)
    return text


@dataclass(frozen=True)
class AkoyaMessage:
    type_code: int
    fields: list[Any]

    @classmethod
    def from_payload(cls, payload: bytes) -> "AkoyaMessage":
        type_code, fields = unpack_payload(payload)
        return cls(type_code, fields)

    @classmethod
    def from_frame(cls, frame: bytes) -> "AkoyaMessage":
        type_code, fields = unpack_frame(frame)
        return cls(type_code, fields)

    def payload(self) -> bytes:
        return pack_payload(self.type_code, self.fields)

    def frame(self) -> bytes:
        return pack_frame(self.type_code, self.fields)


@dataclass(frozen=True)
class RegisterMiner:
    session_uuid: str
    wallet_address: str
    worker_name: str
    gpu_name: str
    common_dim: int
    miner_version: str
    git_sha: str

    @classmethod
    def from_fields(cls, fields: list[Any]) -> "RegisterMiner":
        _require(len(fields) == 7, f"register fields length must be 7, got {len(fields)}")
        return cls(
            session_uuid=_uuid(fields[0], "session_uuid"),
            wallet_address=_str(fields[1], "wallet_address"),
            worker_name=_str(fields[2], "worker_name"),
            gpu_name=_str(fields[3], "gpu_name"),
            common_dim=_int(fields[4], "common_dim"),
            miner_version=_str(fields[5], "miner_version"),
            git_sha=_str(fields[6], "git_sha"),
        )

    def to_fields(self) -> list[Any]:
        return [
            self.session_uuid,
            self.wallet_address,
            self.worker_name,
            self.gpu_name,
            self.common_dim,
            self.miner_version,
            self.git_sha,
        ]


@dataclass(frozen=True)
class RegisterAck:
    accepted: bool
    pool_uuid: str
    share_difficulty: int
    miner_id: str

    @classmethod
    def from_fields(cls, fields: list[Any]) -> "RegisterAck":
        _require(len(fields) == 4, f"register ack fields length must be 4, got {len(fields)}")
        _require(isinstance(fields[0], bool), "register accepted field must be bool")
        return cls(
            accepted=fields[0],
            pool_uuid=_uuid(fields[1], "pool_uuid"),
            share_difficulty=_int(fields[2], "share_difficulty"),
            miner_id=_uuid(fields[3], "miner_id"),
        )


@dataclass(frozen=True)
class JobAssignment:
    job_uuid: str
    header_bytes: bytes
    share_difficulty: int
    height: int
    seed_hash: bytes
    reserved: bytes
    network_nbits: int

    @classmethod
    def from_fields(cls, fields: list[Any]) -> "JobAssignment":
        _require(len(fields) == 7, f"job assignment fields length must be 7, got {len(fields)}")
        return cls(
            job_uuid=_uuid(fields[0], "job_uuid"),
            header_bytes=_bytes(fields[1], HEADER_SIZE, "header_bytes"),
            share_difficulty=_int(fields[2], "share_difficulty"),
            height=_int(fields[3], "height"),
            seed_hash=_bytes(fields[4], HASH_SIZE, "seed_hash"),
            reserved=_bytes(fields[5], None, "reserved"),
            network_nbits=_int(fields[6], "network_nbits"),
        )


@dataclass(frozen=True)
class ShareResult:
    share_id: str
    outcome_code: int
    message: str

    @classmethod
    def from_fields(cls, fields: list[Any]) -> "ShareResult":
        _require(len(fields) == 3, f"share result fields length must be 3, got {len(fields)}")
        return cls(
            share_id=_uuid(fields[0], "share_id"),
            outcome_code=_int(fields[1], "outcome_code"),
            message=_str(fields[2], "message"),
        )

    @property
    def accepted(self) -> bool:
        return self.outcome_code == 0


@dataclass(frozen=True)
class PeriodicPattern:
    """Python mirror of zk-pow::api::proof_utils::PeriodicPattern."""

    shape: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]

    @classmethod
    def from_bytes(cls, data: bytes) -> "PeriodicPattern":
        _require(len(data) == 6, f"periodic pattern must be 6 bytes, got {len(data)}")
        shape: list[tuple[int, int]] = []
        min_stride = 1
        is_done = False
        for index in range(0, 6, 2):
            factor = 1 + data[index]
            length = 1 + data[index + 1]
            if length == 1 or is_done:
                _require(factor == 1 and length == 1, "non-canonical periodic pattern")
                is_done = True
            elif factor <= 1 and min_stride != 1:
                raise AkoyaProtocolError("single stride must not be broken")
            stride = factor * min_stride
            shape.append((stride, length))
            min_stride = stride * length
        return cls(tuple(shape))  # type: ignore[arg-type]

    def to_bytes(self) -> bytes:
        out = bytearray()
        min_stride = 1
        for stride, length in self.shape:
            factor = stride // min_stride
            out.append(factor - 1)
            out.append(length - 1)
            min_stride = stride * length
        return bytes(out)

    def to_list(self) -> list[int]:
        result = [0]
        for stride, length in self.shape:
            next_result: list[int] = []
            for i in range(length):
                for value in result:
                    next_result.append(value + i * stride)
            result = next_result
        return result

    def indices_with_offset(self, offset: int) -> list[int]:
        return [offset + value for value in self.to_list()]

    def size(self) -> int:
        out = 1
        for _, length in self.shape:
            out *= length
        return out


@dataclass(frozen=True)
class MiningConfiguration:
    common_dim: int
    rank: int
    mma_type: int
    rows_pattern: PeriodicPattern
    cols_pattern: PeriodicPattern
    reserved: bytes

    @classmethod
    def from_bytes(cls, data: bytes) -> "MiningConfiguration":
        _require(len(data) == MINING_CONFIG_SIZE, f"mining config must be 52 bytes, got {len(data)}")
        common_dim, rank, mma_type = struct.unpack_from("<IHH", data, 0)
        reserved = data[20:52]
        _require(reserved == bytes(32), "mining config reserved bytes must be zero")
        return cls(
            common_dim=common_dim,
            rank=rank,
            mma_type=mma_type,
            rows_pattern=PeriodicPattern.from_bytes(data[8:14]),
            cols_pattern=PeriodicPattern.from_bytes(data[14:20]),
            reserved=reserved,
        )

    def to_bytes(self) -> bytes:
        return b"".join(
            [
                struct.pack("<IHH", self.common_dim, self.rank, self.mma_type),
                self.rows_pattern.to_bytes(),
                self.cols_pattern.to_bytes(),
                self.reserved,
            ]
        )


@dataclass(frozen=True)
class MatrixProofWire:
    leaf_data: tuple[bytes, ...]
    leaf_indices: tuple[int, ...]
    total_leaves: int
    siblings: tuple[bytes, ...]

    @classmethod
    def from_wire(cls, value: Any, name: str) -> "MatrixProofWire":
        _require(isinstance(value, list) and len(value) == 4, f"{name} must be 4-element list")
        leaf_data_raw, leaf_indices_raw, total_leaves, siblings_raw = value
        _require(isinstance(leaf_data_raw, list), f"{name}.leaf_data must be list")
        _require(isinstance(leaf_indices_raw, list), f"{name}.leaf_indices must be list")
        _require(isinstance(siblings_raw, list), f"{name}.siblings must be list")
        leaf_data = tuple(_bytes(chunk, BLAKE3_CHUNK_SIZE, f"{name}.leaf_data[]") for chunk in leaf_data_raw)
        leaf_indices = tuple(_int(index, f"{name}.leaf_indices[]") for index in leaf_indices_raw)
        siblings = tuple(_bytes(sibling, HASH_SIZE, f"{name}.siblings[]") for sibling in siblings_raw)
        _require(len(leaf_data) == len(leaf_indices), f"{name} leaf_data and leaf_indices length mismatch")
        return cls(
            leaf_data=leaf_data,
            leaf_indices=leaf_indices,
            total_leaves=_int(total_leaves, f"{name}.total_leaves"),
            siblings=siblings,
        )

    def to_wire(self) -> list[Any]:
        return [
            list(self.leaf_data),
            list(self.leaf_indices),
            self.total_leaves,
            list(self.siblings),
        ]

    def opened_bytes(self) -> bytes:
        return b"".join(self.leaf_data)

    def row_indices(self, common_dim: int) -> list[int]:
        leaves_per_row = (common_dim + BLAKE3_CHUNK_SIZE - 1) // BLAKE3_CHUNK_SIZE
        _require(leaves_per_row > 0, "common_dim must be positive")
        return sorted({index // leaves_per_row for index in self.leaf_indices})


@dataclass(frozen=True)
class PlainProofShare:
    share_id: str
    header_bytes: bytes
    opened_a: bytes
    seed_hash: bytes
    t_rows: int
    t_cols: int
    compact_result: bytes
    claimed_hash: bytes
    share_difficulty: int
    hash_a: bytes
    hash_b: bytes
    opened_b: bytes
    mining_config_bytes: bytes
    a_proof: MatrixProofWire
    b_proof: MatrixProofWire

    @classmethod
    def from_fields(cls, fields: list[Any]) -> "PlainProofShare":
        _require(len(fields) == 15, f"PlainProofShare fields length must be 15, got {len(fields)}")
        return cls(
            share_id=_uuid(fields[0], "share_id"),
            header_bytes=_bytes(fields[1], HEADER_SIZE, "header_bytes"),
            opened_a=_bytes(fields[2], None, "opened_a"),
            seed_hash=_bytes(fields[3], HASH_SIZE, "seed_hash"),
            t_rows=_int(fields[4], "t_rows"),
            t_cols=_int(fields[5], "t_cols"),
            compact_result=_bytes(fields[6], 64, "compact_result"),
            claimed_hash=_bytes(fields[7], HASH_SIZE, "claimed_hash"),
            share_difficulty=_int(fields[8], "share_difficulty"),
            hash_a=_bytes(fields[9], HASH_SIZE, "hash_a"),
            hash_b=_bytes(fields[10], HASH_SIZE, "hash_b"),
            opened_b=_bytes(fields[11], None, "opened_b"),
            mining_config_bytes=_bytes(fields[12], MINING_CONFIG_SIZE, "mining_config"),
            a_proof=MatrixProofWire.from_wire(fields[13], "a_proof"),
            b_proof=MatrixProofWire.from_wire(fields[14], "b_proof"),
        )

    @property
    def mining_config(self) -> MiningConfiguration:
        return MiningConfiguration.from_bytes(self.mining_config_bytes)

    def to_fields(self) -> list[Any]:
        return [
            self.share_id,
            self.header_bytes,
            self.opened_a,
            self.seed_hash,
            self.t_rows,
            self.t_cols,
            self.compact_result,
            self.claimed_hash,
            self.share_difficulty,
            self.hash_a,
            self.hash_b,
            self.opened_b,
            self.mining_config_bytes,
            self.a_proof.to_wire(),
            self.b_proof.to_wire(),
        ]

    def frame(self) -> bytes:
        return pack_frame(TYPE_PLAIN_PROOF_SHARE, self.to_fields())

    def validate(self) -> dict[str, Any]:
        config = self.mining_config
        _require(self.opened_a == self.a_proof.opened_bytes(), "opened_a does not match a_proof leaf_data")
        _require(self.opened_b == self.b_proof.opened_bytes(), "opened_b does not match b_proof leaf_data")
        a_rows = self.a_proof.row_indices(config.common_dim)
        b_cols = self.b_proof.row_indices(config.common_dim)
        expected_a_rows = config.rows_pattern.indices_with_offset(self.t_rows)
        expected_b_cols = config.cols_pattern.indices_with_offset(self.t_cols)
        _require(a_rows == expected_a_rows, f"A row indices {a_rows} != pattern rows {expected_a_rows}")
        _require(b_cols == expected_b_cols, f"B col indices {b_cols} != pattern cols {expected_b_cols}")
        return {
            "share_id": self.share_id,
            "common_dim": config.common_dim,
            "rank": config.rank,
            "rows": a_rows,
            "cols_head": b_cols[:8],
            "cols_count": len(b_cols),
            "a_leaf_count": len(self.a_proof.leaf_data),
            "b_leaf_count": len(self.b_proof.leaf_data),
            "a_sibling_count": len(self.a_proof.siblings),
            "b_sibling_count": len(self.b_proof.siblings),
        }


def parse_message(type_code: int, fields: list[Any]) -> Any:
    if type_code == TYPE_REGISTER:
        return RegisterMiner.from_fields(fields)
    if type_code == TYPE_REGISTER_ACK:
        return RegisterAck.from_fields(fields)
    if type_code == TYPE_JOB_ASSIGNMENT:
        return JobAssignment.from_fields(fields)
    if type_code == TYPE_PLAIN_PROOF_SHARE:
        return PlainProofShare.from_fields(fields)
    if type_code == TYPE_SHARE_RESULT:
        return ShareResult.from_fields(fields)
    return AkoyaMessage(type_code, fields)


def parse_payload(payload: bytes) -> Any:
    return parse_message(*unpack_payload(payload))


def parse_frame(frame: bytes) -> Any:
    return parse_message(*unpack_frame(frame))


def load_msgpack(path: str | Path) -> Any:
    return parse_payload(Path(path).read_bytes())


def load_frame(path: str | Path) -> Any:
    return parse_frame(Path(path).read_bytes())
