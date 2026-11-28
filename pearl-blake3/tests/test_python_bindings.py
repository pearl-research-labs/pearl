"""The Python compressions rebuild MerkleTree's nodes and root.

Run after ``maturin develop`` in this directory: ``pytest tests``."""

import pytest
from pearl_blake3 import ALLOWED_CHUNK_LENS, MerkleTree, chunk_cvs, parent_cvs, root_cv

KEY = bytes(range(32))


def _data(size: int) -> bytes:
    return bytes((i * 2654435761 + 7) & 0xFF for i in range(size))


def _digests(blob: bytes) -> list[bytes]:
    return [blob[i : i + 32] for i in range(0, len(blob), 32)]


def _root(leaves: list[bytes]) -> bytes:
    """MerkleTree's fold: pairs combine, an odd last node is promoted, and
    the top two combine with the ROOT flag."""
    layer = leaves
    while len(layer) > 2:
        pairs = len(layer) // 2
        parents = _digests(
            parent_cvs(b"".join(layer[0 : 2 * pairs : 2]), b"".join(layer[1 : 2 * pairs : 2]), KEY)
        )
        layer = parents + layer[2 * pairs :]
    return root_cv(layer[0], layer[1], KEY)


@pytest.mark.parametrize("chunk_len", ALLOWED_CHUNK_LENS)
@pytest.mark.parametrize("leaves", [2, 3, 5, 8, 37])
def test_the_compressions_rebuild_the_tree(chunk_len, leaves):
    data = _data(leaves * chunk_len)
    tree = MerkleTree(data, KEY, chunk_len)
    cvs = _digests(chunk_cvs(data, KEY, 0, chunk_len))
    assert cvs == list(tree.leaf_hashes)
    assert _root(cvs) == tree.root


def test_leaves_hash_at_their_own_position():
    data = _data(16 * 256)
    whole = chunk_cvs(data, KEY, 0, 256)
    assert chunk_cvs(data[5 * 256 :], KEY, 5, 256) == whole[5 * 32 :]
    assert chunk_cvs(data[5 * 256 :], KEY, 0, 256) != whole[5 * 32 :]


def test_bad_arguments_raise():
    data = _data(4 * 256)
    with pytest.raises(ValueError, match="key must be exactly 32 bytes"):
        chunk_cvs(data, KEY[:31], 0, 256)
    with pytest.raises(ValueError, match="not in"):
        chunk_cvs(data, KEY, 0, 100)
    with pytest.raises(ValueError, match="whole 256-byte leaves"):
        chunk_cvs(data[:300], KEY, 0, 256)
    with pytest.raises(ValueError, match="addressable"):
        chunk_cvs(data, KEY, 2**64 - 3, 256)
    with pytest.raises(ValueError, match="whole 32-byte digests"):
        parent_cvs(bytes(33), bytes(33), KEY)
    with pytest.raises(ValueError, match="left holds 2 digests, right 1"):
        parent_cvs(bytes(64), bytes(32), KEY)
    with pytest.raises(ValueError, match="exactly 32 bytes"):
        root_cv(bytes(31), bytes(32), KEY)
