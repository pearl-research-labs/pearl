#!/usr/bin/env python3
"""Small Akoya pool client used by the P1K-131 bridge."""

from __future__ import annotations

from dataclasses import dataclass
import base64
import socket
from types import TracebackType
from typing import Any

from akoya_protocol import (
    TYPE_DIFFICULTY_ADJUST,
    TYPE_JOB_ASSIGNMENT,
    TYPE_PLAIN_PROOF_SHARE,
    TYPE_REGISTER,
    TYPE_REGISTER_ACK,
    TYPE_SHARE_RESULT,
    AkoyaMessage,
    JobAssignment,
    PlainProofShare,
    RegisterAck,
    RegisterMiner,
    ShareResult,
    pack_frame,
    parse_message,
    read_frame,
)


def nbits_to_target(nbits: int) -> int:
    """Mirror zk_pow::api::proof_utils::nbits_to_difficulty."""

    exponent = (nbits >> 24) & 0xFF
    mantissa = nbits & 0x00FFFFFF
    if mantissa == 0 or exponent == 0 or (mantissa & 0x00800000):
        return 0
    if exponent <= 3:
        return mantissa >> (8 * (3 - exponent))
    return mantissa << (8 * (exponent - 3))


def mining_job_dict_for_akoya(job: JobAssignment) -> dict[str, Any]:
    """Gateway-compatible MiningJob dict for an Akoya share-difficulty job."""

    return {
        "incomplete_header_bytes": base64.b64encode(job.header_bytes).decode("ascii"),
        "target": nbits_to_target(job.share_difficulty),
    }


@dataclass(frozen=True)
class AkoyaJobContext:
    ack: RegisterAck
    job: JobAssignment

    @property
    def mining_job_dict(self) -> dict[str, Any]:
        return mining_job_dict_for_akoya(self.job)


class AkoyaPoolSession:
    """Persistent Akoya pool connection.

    A submitted type-3 share should be sent on the same connection that received
    its job assignment.
    """

    def __init__(self, host: str = "pool.akoyapool.com", port: int = 3333, timeout: float = 20.0) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout
        self.sock: socket.socket | None = None

    def connect(self) -> None:
        self.sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
        self.sock.settimeout(self.timeout)

    def close(self) -> None:
        if self.sock is not None:
            self.sock.close()
            self.sock = None

    def __enter__(self) -> "AkoyaPoolSession":
        self.connect()
        return self

    def __exit__(
        self,
        type_: type[BaseException] | None,
        value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        self.close()
        return None

    def _sock(self) -> socket.socket:
        if self.sock is None:
            raise RuntimeError("AkoyaPoolSession is not connected")
        return self.sock

    def send_register(self, register: RegisterMiner) -> None:
        self._sock().sendall(pack_frame(TYPE_REGISTER, register.to_fields()))

    def read_message(self) -> Any:
        return parse_message(*read_frame(self._sock()))

    def register_and_wait_job(self, register: RegisterMiner) -> AkoyaJobContext:
        self.send_register(register)
        ack: RegisterAck | None = None
        while True:
            message = self.read_message()
            if isinstance(message, RegisterAck):
                if not message.accepted:
                    raise RuntimeError(f"Akoya registration rejected: {message}")
                ack = message
            elif isinstance(message, JobAssignment):
                if ack is None:
                    raise RuntimeError("Akoya sent job before register ack")
                return AkoyaJobContext(ack=ack, job=message)

    def submit_share(self, share: PlainProofShare) -> ShareResult:
        """Submit a type-3 share and wait for its matching type-4 result."""

        self._sock().sendall(pack_frame(TYPE_PLAIN_PROOF_SHARE, share.to_fields()))
        while True:
            message = self.read_message()
            if isinstance(message, ShareResult) and message.share_id == share.share_id:
                return message
            if isinstance(message, AkoyaMessage) and message.type_code in {
                TYPE_REGISTER_ACK,
                TYPE_JOB_ASSIGNMENT,
                TYPE_DIFFICULTY_ADJUST,
                TYPE_SHARE_RESULT,
            }:
                continue

