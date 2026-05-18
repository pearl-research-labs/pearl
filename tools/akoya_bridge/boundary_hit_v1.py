#!/usr/bin/env python3
"""BoundaryHitV1 transport helpers for the Hardy split proof scaffold."""

from __future__ import annotations

from dataclasses import dataclass
import struct
from typing import Any, Mapping
import uuid

from akoya_protocol import FRAME_PREFIX_SIZE, HEADER_SIZE, HASH_SIZE, MINING_CONFIG_SIZE, MiningConfiguration


SCHEMA_VERSION = 1
PAYLOAD_SIZE = 220
FRAME_SIZE = FRAME_PREFIX_SIZE + PAYLOAD_SIZE

A_MODE_FIXED = 0
A_MODE_NONCE_PREFIX = 1
A_MODE_FULL_RANDOM = 2

A_MODE_NAMES = {
    A_MODE_FIXED: "fixed",
    A_MODE_NONCE_PREFIX: "nonce-prefix",
    A_MODE_FULL_RANDOM: "full-random",
}
A_MODE_CODES = {name: code for code, name in A_MODE_NAMES.items()}

_PAYLOAD_STRUCT = struct.Struct(
    f">H16s{HEADER_SIZE}s{HASH_SIZE}sIIII{MINING_CONFIG_SIZE}sIIBBIIQ"
)


class BoundaryHitV1Error(ValueError):
    """Raised when a BoundaryHitV1 frame is malformed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise BoundaryHitV1Error(message)


def _bytes(value: Any, size: int, name: str) -> bytes:
    _require(isinstance(value, (bytes, bytearray)), f"{name} must be bytes")
    out = bytes(value)
    _require(len(out) == size, f"{name} must be {size} bytes, got {len(out)}")
    return out


def _int(value: Any, name: str) -> int:
    _require(isinstance(value, int), f"{name} must be int")
    return value


def _u32(value: Any, name: str) -> int:
    out = _int(value, name)
    _require(0 <= out <= 0xFFFFFFFF, f"{name} must fit in u32")
    return out


def _u8(value: Any, name: str) -> int:
    out = _int(value, name)
    _require(0 <= out <= 0xFF, f"{name} must fit in u8")
    return out


def _u64(value: Any, name: str) -> int:
    out = _int(value, name)
    _require(0 <= out <= 0xFFFFFFFFFFFFFFFF, f"{name} must fit in u64")
    return out


def _to_uuid_bytes(value: str) -> bytes:
    return uuid.UUID(value).bytes


def _from_uuid_bytes(value: bytes) -> str:
    return str(uuid.UUID(bytes=value))


def _as_bool(value: Any, name: str) -> bool:
    _require(isinstance(value, bool), f"{name} must be bool")
    return value


def _extract_proof_artifact(data: Mapping[str, Any]) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    if "attempts" in data:
        attempts = data.get("attempts")
        _require(isinstance(attempts, list), "summary attempts must be list")
        for attempt in attempts:
            if isinstance(attempt, Mapping) and attempt.get("proof_artifact"):
                artifact = attempt["proof_artifact"]
                _require(isinstance(artifact, Mapping), "proof_artifact must be object")
                return attempt, artifact
        raise BoundaryHitV1Error("no proof_artifact found in summary attempts")
    if "proof_artifact" in data:
        artifact = data.get("proof_artifact")
        _require(isinstance(artifact, Mapping), "proof_artifact must be object")
        return data, artifact
    return data, data


@dataclass(frozen=True)
class BoundaryHitV1:
    schema_version: int
    job_uuid: str
    incomplete_header_bytes: bytes
    seed_hash: bytes
    share_verify_nbits: int
    network_nbits: int
    m: int
    n: int
    mining_config_bytes: bytes
    t_rows: int
    t_cols: int
    a_mode: int
    a_initial_random: bool
    a_nonce_row: int
    a_nonce_bytes: int
    a_nonce_counter: int

    def validate(self) -> "BoundaryHitV1":
        _require(self.schema_version == SCHEMA_VERSION, f"schema_version must be {SCHEMA_VERSION}")
        uuid.UUID(self.job_uuid)
        _bytes(self.incomplete_header_bytes, HEADER_SIZE, "incomplete_header_bytes")
        _bytes(self.seed_hash, HASH_SIZE, "seed_hash")
        _u32(self.share_verify_nbits, "share_verify_nbits")
        _u32(self.network_nbits, "network_nbits")
        _u32(self.m, "m")
        _u32(self.n, "n")
        _require(self.m > 0 and self.n > 0, "m and n must be positive")
        _bytes(self.mining_config_bytes, MINING_CONFIG_SIZE, "mining_config_bytes")
        _u32(self.t_rows, "t_rows")
        _u32(self.t_cols, "t_cols")
        _u8(self.a_mode, "a_mode")
        _require(self.a_mode in A_MODE_NAMES, f"unknown a_mode code: {self.a_mode}")
        _require(isinstance(self.a_initial_random, bool), "a_initial_random must be bool")
        _u32(self.a_nonce_row, "a_nonce_row")
        _u32(self.a_nonce_bytes, "a_nonce_bytes")
        _u64(self.a_nonce_counter, "a_nonce_counter")

        config = self.mining_config
        _require(config.common_dim > 0, "common_dim must be positive")
        _require(config.rank > 0, "rank must be positive")

        if self.a_mode == A_MODE_NONCE_PREFIX:
            _require(self.a_nonce_row < self.m, "a_nonce_row must be within [0, m)")
            _require(0 < self.a_nonce_bytes <= config.common_dim, "a_nonce_bytes must be within [1, k]")

        return self

    @property
    def mining_config(self) -> MiningConfiguration:
        return MiningConfiguration.from_bytes(self.mining_config_bytes)

    @property
    def k(self) -> int:
        return self.mining_config.common_dim

    @property
    def rank(self) -> int:
        return self.mining_config.rank

    @property
    def a_mode_name(self) -> str:
        return A_MODE_NAMES[self.a_mode]

    @property
    def a_row_indices(self) -> list[int]:
        return self.mining_config.rows_pattern.indices_with_offset(self.t_rows)

    @property
    def b_column_indices(self) -> list[int]:
        return self.mining_config.cols_pattern.indices_with_offset(self.t_cols)

    @property
    def can_reconstruct_a_locally(self) -> bool:
        return not self.a_initial_random and self.a_mode in {A_MODE_FIXED, A_MODE_NONCE_PREFIX}

    def reconstruction_blockers(self) -> list[str]:
        blockers: list[str] = []
        if self.a_initial_random:
            blockers.append("a_initial_random=true requires shipping or saving the initialized A matrix")
        if self.a_mode == A_MODE_FULL_RANDOM:
            blockers.append("a_mode=full-random requires shipping or saving the full A matrix")
        return blockers

    def to_payload(self) -> bytes:
        self.validate()
        payload = _PAYLOAD_STRUCT.pack(
            self.schema_version,
            _to_uuid_bytes(self.job_uuid),
            self.incomplete_header_bytes,
            self.seed_hash,
            self.share_verify_nbits,
            self.network_nbits,
            self.m,
            self.n,
            self.mining_config_bytes,
            self.t_rows,
            self.t_cols,
            self.a_mode,
            int(self.a_initial_random),
            self.a_nonce_row,
            self.a_nonce_bytes,
            self.a_nonce_counter,
        )
        _require(len(payload) == PAYLOAD_SIZE, f"payload must be {PAYLOAD_SIZE} bytes, got {len(payload)}")
        return payload

    def to_frame(self) -> bytes:
        payload = self.to_payload()
        return len(payload).to_bytes(FRAME_PREFIX_SIZE, "big") + payload

    def to_dict(self) -> dict[str, Any]:
        config = self.mining_config
        return {
            "schema_version": self.schema_version,
            "job_uuid": self.job_uuid,
            "incomplete_header_bytes_hex": self.incomplete_header_bytes.hex(),
            "seed_hash_hex": self.seed_hash.hex(),
            "share_verify_nbits": self.share_verify_nbits,
            "network_nbits": self.network_nbits,
            "m": self.m,
            "n": self.n,
            "k": config.common_dim,
            "rank": config.rank,
            "mining_config_bytes_hex": self.mining_config_bytes.hex(),
            "t_rows": self.t_rows,
            "t_cols": self.t_cols,
            "a_mode": self.a_mode_name,
            "a_initial_random": self.a_initial_random,
            "a_nonce_row": self.a_nonce_row,
            "a_nonce_bytes": self.a_nonce_bytes,
            "a_nonce_counter": self.a_nonce_counter,
            "a_row_indices": self.a_row_indices,
            "b_column_indices_head": self.b_column_indices[:8],
            "b_column_indices_count": len(self.b_column_indices),
            "payload_bytes": PAYLOAD_SIZE,
            "frame_bytes": FRAME_SIZE,
        }

    @classmethod
    def from_payload(cls, payload: bytes) -> "BoundaryHitV1":
        _require(len(payload) == PAYLOAD_SIZE, f"payload must be {PAYLOAD_SIZE} bytes, got {len(payload)}")
        unpacked = _PAYLOAD_STRUCT.unpack(payload)
        boundary = cls(
            schema_version=unpacked[0],
            job_uuid=_from_uuid_bytes(unpacked[1]),
            incomplete_header_bytes=unpacked[2],
            seed_hash=unpacked[3],
            share_verify_nbits=unpacked[4],
            network_nbits=unpacked[5],
            m=unpacked[6],
            n=unpacked[7],
            mining_config_bytes=unpacked[8],
            t_rows=unpacked[9],
            t_cols=unpacked[10],
            a_mode=unpacked[11],
            a_initial_random=bool(unpacked[12]),
            a_nonce_row=unpacked[13],
            a_nonce_bytes=unpacked[14],
            a_nonce_counter=unpacked[15],
        )
        return boundary.validate()

    @classmethod
    def from_frame(cls, frame: bytes) -> "BoundaryHitV1":
        _require(len(frame) == FRAME_SIZE, f"frame must be {FRAME_SIZE} bytes, got {len(frame)}")
        payload_len = int.from_bytes(frame[:FRAME_PREFIX_SIZE], "big")
        _require(payload_len == PAYLOAD_SIZE, f"frame length prefix {payload_len} != payload {PAYLOAD_SIZE}")
        return cls.from_payload(frame[FRAME_PREFIX_SIZE:])

    @classmethod
    def from_proof_artifact(cls, artifact: Mapping[str, Any]) -> "BoundaryHitV1":
        context, proof = _extract_proof_artifact(artifact)
        b_generation = proof.get("b_generation")
        a_generation = proof.get("a_generation")
        indices = proof.get("indices")
        _require(isinstance(b_generation, Mapping), "proof_artifact.b_generation must be object")
        _require(isinstance(a_generation, Mapping), "proof_artifact.a_generation must be object")
        _require(isinstance(indices, Mapping), "proof_artifact.indices must be object")

        job_uuid = context.get("akoya_job_uuid") or proof.get("job_uuid") or context.get("job_uuid")
        network_nbits = context.get("akoya_network_nbits") or proof.get("network_nbits") or context.get("network_nbits")
        share_verify_nbits = proof.get("share_verify_nbits") or context.get("akoya_share_difficulty")
        _require(job_uuid is not None, "proof artifact context is missing job_uuid")
        _require(network_nbits is not None, "proof artifact context is missing network_nbits")
        _require(share_verify_nbits is not None, "proof artifact is missing share_verify_nbits")

        a_rows = indices.get("A_row_indices")
        b_cols = indices.get("B_column_indices")
        _require(isinstance(a_rows, list) and a_rows, "A_row_indices must be non-empty list")
        _require(isinstance(b_cols, list) and b_cols, "B_column_indices must be non-empty list")

        mode_name = a_generation.get("mode")
        _require(isinstance(mode_name, str), "a_generation.mode must be str")
        _require(mode_name in A_MODE_CODES, f"unsupported a_generation.mode: {mode_name}")

        boundary = cls(
            schema_version=SCHEMA_VERSION,
            job_uuid=str(job_uuid),
            incomplete_header_bytes=bytes.fromhex(str(proof["incomplete_header_bytes_hex"])),
            seed_hash=bytes.fromhex(str(b_generation["seed_hash"])),
            share_verify_nbits=_u32(int(share_verify_nbits), "share_verify_nbits"),
            network_nbits=_u32(int(network_nbits), "network_nbits"),
            m=_u32(int(proof["m"]), "m"),
            n=_u32(int(proof["n"]), "n"),
            mining_config_bytes=bytes.fromhex(str(proof["mining_config_bytes_hex"])),
            t_rows=min(int(value) for value in a_rows),
            t_cols=min(int(value) for value in b_cols),
            a_mode=A_MODE_CODES[mode_name],
            a_initial_random=_as_bool(a_generation.get("initial_random", False), "a_generation.initial_random"),
            a_nonce_row=_u32(int(a_generation.get("nonce_row", 0) or 0), "a_nonce_row"),
            a_nonce_bytes=_u32(int(a_generation.get("nonce_bytes", 0) or 0), "a_nonce_bytes"),
            a_nonce_counter=_u64(int(a_generation.get("nonce_counter", 0) or 0), "a_nonce_counter"),
        ).validate()

        actual_rows = [int(value) for value in a_rows]
        actual_cols = [int(value) for value in b_cols]
        _require(
            actual_rows == boundary.a_row_indices,
            f"proof artifact A rows {actual_rows} do not match mining_config + t_rows {boundary.a_row_indices}",
        )
        _require(
            actual_cols == boundary.b_column_indices,
            f"proof artifact B cols {actual_cols} do not match mining_config + t_cols {boundary.b_column_indices}",
        )
        return boundary
