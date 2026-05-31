"""Tests for the Pearlhash XOR cipher against the four reference captures.

Test vectors are taken from `51_pearlhash_cipher_re.md` §3 (decoded captures
table). Each captured C->S frame gives us (frame_index, plaintext, ciphertext)
— we verify symmetric encrypt/decrypt against the recovered keystreams and
that `recover_keystream` reproduces the published table.
"""

from __future__ import annotations

import pytest

from pearl_stratum.pearlhash_cipher import (
    KeyNotKnownError,
    decrypt,
    encrypt,
    recover_keystream,
)
from pearl_stratum.pearlhash_keys import KEY_MSG, MAX_KNOWN_FRAME_INDEX


# --------------------------------------------------------------------------
# Reference frames from `cap_long.pcap` and the inline hex in memo 51 §2.4.
# All four are C->S frames; counter is stripped (we test cipher only, not the
# 4-byte rand counter prefix).
# --------------------------------------------------------------------------

# Frame 4 (cap_long): login, frame_index=0, counter=0x6b8b4567, 108B body.
# Inner hex from memo 51 §2.4 reference decryption.
LOGIN_INNER_HEX = (
    "6b8b45677e7da4ddd4f95c8813b87c2204a5bdf78385a196cdc8af88bf83fc61"
    "97610aea130e969b8977f958eef30601338fa0236b3ca3d2c3a500d343ad203d"
    "5bbcacedd89ea692d091aadbf493e063863310ff0351a68c9563a71ae6e1145e"
    "66d4ac7f2773ef9bdae15c8a04f7442b"
)
LOGIN_PLAINTEXT = (
    b'{"id":0,"method":"login","params":'
    b'["prl1pja266mlncnk5flwrx9k7vu8a9kkz0kqg2lcc3wf2ek5lf2sxxsmcma0","","0.5"]}'
)

# Frame 12 (cap_long): report_info id=1, hashrate 19703248663347.199219.
ID1_PLAINTEXT = (
    b'{"id":1,"method":"report_info","params":'
    b'[{"name":"NVIDIA GeForce RTX 4070 Ti SUPER",'
    b'"hashrate":19703248663347.199219}]}'
)

# Frame 15 (cap_long): report_info id=2, hashrate 22517998472396.800781.
ID2_PLAINTEXT = (
    b'{"id":2,"method":"report_info","params":'
    b'[{"name":"NVIDIA GeForce RTX 4070 Ti SUPER",'
    b'"hashrate":22517998472396.800781}]}'
)

# cap_verify frame 4: decoy-wallet login.
DECOY_LOGIN_PLAINTEXT = (
    b'{"id":0,"method":"login","params":'
    b'["prl1pgk8j7vj0xkxppzux5vqgqur9t03k9zvmm5qkam5hzaavzs69vjkqzz28wg",'
    b'"","0.5"]}'
)


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------


def test_key_table_shape() -> None:
    """All 16 recovered keystreams are exactly 48 bytes."""
    assert MAX_KNOWN_FRAME_INDEX == 15
    assert set(KEY_MSG.keys()) == set(range(16))
    for idx, k in KEY_MSG.items():
        assert len(k) == 48, f"KEY_MSG[{idx}] has wrong length {len(k)}"


def test_decrypt_login_frame_cap_long() -> None:
    """Cap_long.pcap frame 4 decrypts to the published login plaintext."""
    inner = bytes.fromhex(LOGIN_INNER_HEX)
    counter = int.from_bytes(inner[:4], "big")
    body = inner[4:]
    assert counter == 0x6B8B4567
    assert len(body) == 108
    pt = decrypt(body, frame_index=0)
    assert pt == LOGIN_PLAINTEXT


def test_decrypt_decoy_wallet_login() -> None:
    """The SAME KEY_MSG[0] decrypts a decoy wallet's login (keys are wallet-independent)."""
    # Re-encrypt the decoy login with KEY_MSG[0]; round-trip must equal decoy pt.
    ct = encrypt(DECOY_LOGIN_PLAINTEXT, frame_index=0)
    pt = decrypt(ct, frame_index=0)
    assert pt == DECOY_LOGIN_PLAINTEXT


def test_encrypt_decrypt_roundtrip_id1() -> None:
    """Encrypt then decrypt id=1 keepalive roundtrips bit-exact."""
    ct = encrypt(ID1_PLAINTEXT, frame_index=1)
    assert ct != ID1_PLAINTEXT, "XOR with nonzero key should change bytes"
    pt = decrypt(ct, frame_index=1)
    assert pt == ID1_PLAINTEXT


def test_encrypt_decrypt_roundtrip_id2() -> None:
    """Encrypt then decrypt id=2 keepalive roundtrips bit-exact."""
    ct = encrypt(ID2_PLAINTEXT, frame_index=2)
    pt = decrypt(ct, frame_index=2)
    assert pt == ID2_PLAINTEXT


def test_xor_periodicity_login() -> None:
    """The keystream truly cycles every 48 bytes across the 108-byte login body."""
    inner = bytes.fromhex(LOGIN_INNER_HEX)
    body = inner[4:]
    pt = LOGIN_PLAINTEXT
    implied_key = bytes(p ^ c for p, c in zip(pt, body))
    # implied_key has length 108 = 48*2 + 12; verify periodicity.
    period = 48
    base = implied_key[:period]
    for i in range(period, len(implied_key)):
        assert implied_key[i] == base[i % period], (
            f"non-periodic at byte {i}: {implied_key[i]:02x} vs {base[i % period]:02x}"
        )
    assert base == KEY_MSG[0]


def test_recover_keystream_from_login() -> None:
    """`recover_keystream` against captured login pair reproduces KEY_MSG[0]."""
    inner = bytes.fromhex(LOGIN_INNER_HEX)
    body = inner[4:]
    recovered = recover_keystream(LOGIN_PLAINTEXT, body)
    assert recovered == KEY_MSG[0]


def test_recover_keystream_rejects_aperiodic_input() -> None:
    """If plaintext/ciphertext imply a non-48-periodic stream, raise ValueError."""
    pt = b"A" * 50
    # Hand-craft a ciphertext where bytes 0 and 48 imply DIFFERENT key bytes.
    ct = bytearray(pt)
    ct[0] = pt[0] ^ 0x11
    ct[48] = pt[48] ^ 0x22  # disagrees with the implied byte at offset 0
    with pytest.raises(ValueError, match="not 48-byte periodic"):
        recover_keystream(pt, bytes(ct))


def test_unknown_frame_index_raises() -> None:
    """Frame index 16+ has no recovered key; encrypt/decrypt should raise."""
    with pytest.raises(KeyNotKnownError) as excinfo:
        encrypt(b"some plaintext", frame_index=16)
    assert excinfo.value.frame_index == 16
    with pytest.raises(KeyNotKnownError):
        decrypt(b"some ciphertext", frame_index=42)


def test_recover_keystream_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        recover_keystream(b"abc", b"abcd")


def test_recover_keystream_empty_raises() -> None:
    with pytest.raises(ValueError, match="empty plaintext"):
        recover_keystream(b"", b"")


def test_short_body_under_48_bytes() -> None:
    """Bodies shorter than 48 bytes still encrypt/decrypt symmetrically."""
    short = b'{"id":0,"method":"x"}'  # 21 bytes
    assert len(short) < 48
    ct = encrypt(short, frame_index=0)
    assert len(ct) == len(short)
    assert decrypt(ct, frame_index=0) == short
