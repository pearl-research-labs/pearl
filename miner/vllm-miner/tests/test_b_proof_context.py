"""The B proof context's leaf-aware CPU commitment rebuild."""

import pytest
import torch
from miner_base.block_submission import PROTOCOL_COMMITMENT_LEAF
from miner_base.commitment import commit_planes, hash_id_for_leaf
from miner_base.commitment_hash import noise_seed_b
from vllm_miner.state import BProofContext

_KEY_B = bytes(range(32))
_ANCESTOR = bytes(range(108))
_P_B = bytes(range(64, 96))


@pytest.fixture(autouse=True)
def async_manager():
    """CPU commitment tests do not need the serving runtime."""
    yield


def _planes() -> list[torch.Tensor]:
    return [
        torch.zeros(4, 2048, dtype=torch.int8),
        torch.zeros(4, 256, dtype=torch.bfloat16),
    ]


def test_prebuilt_commitment_rebuilds_at_the_protocol_leaf():
    codes, scales = _planes()
    digest = commit_planes(
        [codes, scales], _KEY_B, hash_id_for_leaf(PROTOCOL_COMMITMENT_LEAF)
    ).digest
    seed_b = noise_seed_b(digest, _KEY_B, _P_B)
    ctx = BProofContext(
        key_b=_KEY_B, ancestor_header=_ANCESTOR, seed_b=seed_b, p_b=_P_B, codes=codes, scales=scales
    )
    prebuilt = ctx.prebuilt_commitment()
    assert prebuilt.key == _KEY_B
    assert prebuilt.commitment.digest == digest


def test_prebuilt_commitment_rejects_a_seed_the_cpu_tree_does_not_reproduce():
    """A GPU-derived ``seedB`` the CPU rebuild cannot reach means the two
    commitments disagree; the proof must not be built on it."""
    codes, scales = _planes()
    ctx = BProofContext(
        key_b=_KEY_B,
        ancestor_header=_ANCESTOR,
        seed_b=bytes(32),
        p_b=_P_B,
        codes=codes,
        scales=scales,
    )
    with pytest.raises(RuntimeError, match="disagrees"):
        ctx.prebuilt_commitment()
