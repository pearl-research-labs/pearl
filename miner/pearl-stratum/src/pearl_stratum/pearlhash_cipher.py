"""Pearlhash XOR stream cipher.

Per-frame XOR against a 48-byte periodic keystream indexed by frame number.
See `51_pearlhash_cipher_re.md` for the cipher RE; this module is a faithful
implementation of §2.4 of that memo.

Wire format (just the cipher body, NOT the outer hex+newline framing):

    inner = htonl(rand_N) || ciphertext         # 4 bytes counter + N-4 body
    plaintext = XOR(ciphertext, KEY_MSG[N] * (len // 48 + 1))[:len(ciphertext)]

The counter is NOT XORed — it's a glibc rand output and serves as the
on-wire frame index. The cipher key lookup uses the frame index N derived
from the counter (see `pearlhash_keys.COUNTER_TO_FRAME_INDEX`).

Frames beyond the highest known index raise `KeyNotKnownError`; callers
should drop those frames and continue. Submit frames (index >=16) are not
yet keyed — see PEARLHASH_README.md.
"""

from __future__ import annotations

from .pearlhash_keys import KEY_MSG


class KeyNotKnownError(KeyError):
    """The frame index has no recovered keystream — recapture needed."""

    def __init__(self, frame_index: int):
        super().__init__(frame_index)
        self.frame_index = frame_index

    def __str__(self) -> str:
        return (
            f"no recovered keystream for frame index {self.frame_index}; "
            f"see PEARLHASH_README.md for the recapture procedure"
        )


def _xor_with_key(data: bytes, key: bytes) -> bytes:
    """XOR `data` against a periodic `key` (key is repeated to cover data)."""
    klen = len(key)
    return bytes(b ^ key[i % klen] for i, b in enumerate(data))


def encrypt(plaintext: bytes, frame_index: int) -> bytes:
    """Encrypt one frame body. Returns ciphertext, same length as plaintext.

    `frame_index` is the 0-based outgoing-frame counter (0 for login, 1..15 for
    keepalives). Raises `KeyNotKnownError` if the key for that index is unknown.
    """
    key = KEY_MSG.get(frame_index)
    if key is None:
        raise KeyNotKnownError(frame_index)
    return _xor_with_key(plaintext, key)


def decrypt(ciphertext: bytes, frame_index: int) -> bytes:
    """Decrypt one frame body. Symmetric with `encrypt` (XOR is its own inverse)."""
    key = KEY_MSG.get(frame_index)
    if key is None:
        raise KeyNotKnownError(frame_index)
    return _xor_with_key(ciphertext, key)


def recover_keystream(known_plaintext: bytes, ciphertext: bytes) -> bytes:
    """Recover a per-frame keystream from a known plaintext + ciphertext pair.

    Returns up to 48 bytes (the cipher period). If the inputs are >= 48 bytes,
    the function also verifies periodicity over the full overlap and raises
    `ValueError` if the implied keystream is not 48-byte periodic — which would
    indicate the cipher model is wrong for this frame.

    Usage flow for a new frame index N (once a capture is available):
        keystream = recover_keystream(known_pt, observed_ct)
        # then write keystream.hex() into pearlhash_keys.KEY_MSG_HEX[N].
    """
    if len(known_plaintext) != len(ciphertext):
        raise ValueError(
            f"plaintext/ciphertext length mismatch: "
            f"{len(known_plaintext)} vs {len(ciphertext)}"
        )
    if not known_plaintext:
        raise ValueError("empty plaintext — nothing to recover")

    implied = bytes(p ^ c for p, c in zip(known_plaintext, ciphertext))
    period = 48
    base = implied[:period]

    # Verify periodicity over the rest of the overlap.
    for i in range(period, len(implied)):
        if implied[i] != base[i % period]:
            raise ValueError(
                f"recovered keystream is not 48-byte periodic at offset {i}: "
                f"expected {base[i % period]:#04x}, got {implied[i]:#04x}. "
                "Cipher model may be wrong for this frame index."
            )
    return base
