"""Server-side `pearl.challenge` v1.5 DDoS-pacing handshake.

Spec (reverse-engineered, captured in pearl-investigation/wave16-domination/
58_pearl_challenge_protocol.md):

  - On TCP accept, server pushes UNSOLICITED:
      {"id": null, "method": "pearl.challenge",
       "params": {"seed": "<64 hex>", "difficulty": 32}}

  - Client must respond (before any other method) with:
      {"id": <client int>, "method": "pearl.challenge_response",
       "params": {"seed": "<same 64 hex>", "nonce": "<u64 LE hex>"}}
    such that the FIRST 32 BITS of `blake3(seed_bytes || nonce_le_bytes)`
    are zero. (Difficulty=32 = 4-byte all-zero prefix.)

  - Server validates the nonce, responds:
      {"jsonrpc":"2.0","id":<same>,"result":true,"error":null}
    Then stratum proceeds normally.

  - Mid-session re-challenges allowed; client solves while continuing to mine.

This server uses `blake3` from PyPI (already in our deps via pearl-stratum).

Use:
    challenge = Challenge.issue(difficulty=32)
    # send challenge.to_notification_params() over the wire
    # ...receive client params...
    ok = challenge.verify(client_seed_hex, client_nonce_hex)
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass

import blake3 as blake3_module


@dataclass
class Challenge:
    seed_hex: str
    difficulty: int  # number of leading zero BITS required in blake3 output

    @classmethod
    def issue(cls, difficulty: int = 32) -> "Challenge":
        """Generate a fresh 32-byte random seed."""
        if difficulty < 0 or difficulty > 64:
            raise ValueError("difficulty must be 0..64 bits")
        return cls(seed_hex=secrets.token_hex(32), difficulty=difficulty)

    def to_notification_params(self) -> dict:
        """Payload for the unsolicited `pearl.challenge` push.

        Pearl/v1.5's `params` is a JSON OBJECT (not the usual positional list),
        per the captured wire format.
        """
        return {"seed": self.seed_hex, "difficulty": self.difficulty}

    def verify(self, response_seed_hex: str, response_nonce_hex: str) -> bool:
        """True iff `blake3(seed || nonce_le)` has `difficulty` leading zero bits.

        Also requires the response seed to match the issued seed (a client
        echoing back the wrong seed indicates either a buggy miner or a
        wire-tampering attempt).
        """
        if response_seed_hex.lower() != self.seed_hex.lower():
            return False
        try:
            seed = bytes.fromhex(self.seed_hex)
            nonce_int = int(response_nonce_hex, 16)
        except ValueError:
            return False
        if nonce_int < 0 or nonce_int > 2**64 - 1:
            return False
        nonce_le = nonce_int.to_bytes(8, "little")
        digest = blake3_module.blake3(seed + nonce_le).digest()
        return _has_leading_zero_bits(digest, self.difficulty)


def _has_leading_zero_bits(data: bytes, n: int) -> bool:
    """Check that `data`'s first `n` bits are zero (big-endian bit order)."""
    full_zero_bytes, leftover_bits = divmod(n, 8)
    # Need full_zero_bytes complete zero bytes AND, if leftover>0, one more
    # byte to check the partial prefix in.
    needed = full_zero_bytes + (1 if leftover_bits else 0)
    if needed > len(data):
        return False
    if any(b != 0 for b in data[:full_zero_bytes]):
        return False
    if leftover_bits == 0:
        return True
    mask = 0xFF << (8 - leftover_bits) & 0xFF
    return (data[full_zero_bytes] & mask) == 0
