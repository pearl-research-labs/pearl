"""FP8 Python ZK verification against the node's existing proof fixtures."""

import struct
from pathlib import Path

import pearl_mining as pm
import pytest

_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def verifiers():
    cache = _ROOT / "zk-pow/src/api/fp8/fp8_cache.bin"
    if not cache.exists():
        pytest.skip("requires the generated FP8 verifier cache (task build:zk-cache)")
    data = cache.read_bytes()
    count = struct.unpack_from("<Q", data)[0]
    offset = 8
    result = {}
    for _ in range(count):
        device, length = struct.unpack_from("<BQ", data, offset)
        offset += 9
        result[device] = pm.Fp8Verifier.from_bytes(data[offset : offset + length])
        offset += length
    assert offset == len(data)
    return result


@pytest.fixture(scope="module", params=[(0, "h100"), (1, "b200")], ids=["h100", "b200"])
def zk_case(request, verifiers):
    device, name = request.param
    raw = (_ROOT / f"node/zkpow/testdata/fp8_zk_proof_{name}.bin").read_bytes()
    public_len = struct.unpack_from("<I", raw, 76)[0]
    chain_at = 80 + public_len
    depth = raw[chain_at]
    assert depth == 1, "fixture must carry the parent of its depth-2 ancestor"
    proof_at = chain_at + 1 + 108 * depth
    chain = [pm.BlockHeader.from_bytes(raw[chain_at + 1 : proof_at])]
    return (
        verifiers[device],
        pm.IncompleteBlockHeader.from_bytes(raw[:76]),
        chain,
        raw[80:chain_at],
        raw[proof_at:],
    )


@pytest.mark.parametrize("entry", ["block", "share"])
@pytest.mark.parametrize(
    "mutation",
    [
        "valid",
        "missing",
        "too_deep",
        "proposed_as_ancestor",
        "intermediate_commitment",
        "ancestor_commitment",
    ],
)
def test_fp8_zk_ancestry(zk_case, entry, mutation):
    verifier, header, original_chain, original_public, proof = zk_case
    chain = list(original_chain)
    public = bytearray(original_public)
    if mutation == "missing":
        chain = []
    elif mutation == "too_deep":
        chain *= 4
    elif mutation == "proposed_as_ancestor":
        public[:108] = bytes(header.to_bytes()) + bytes(32)
        chain = []
    elif mutation == "intermediate_commitment":
        wire = bytearray(chain[0].to_bytes())
        wire[76] ^= 1
        chain[0] = pm.BlockHeader.from_bytes(wire)
    elif mutation == "ancestor_commitment":
        public[76] ^= 1
    verify = getattr(verifier, f"verify_{entry}")
    args = (header, chain, bytes(public), proof)
    if entry == "share":
        args += (header.nbits,)
    if mutation == "valid":
        verify(*args)
    else:
        with pytest.raises(RuntimeError, match="does not connect|state window"):
            verify(*args)
