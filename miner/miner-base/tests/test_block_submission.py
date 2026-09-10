from dataclasses import replace
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch
from miner_base.async_loop_manager import AsyncLoopManager
from miner_base.block_submission import (
    PROTOCOL_COMMITMENT_LEAF,
    OpenedBlockInfo,
    PrebuiltCommitment,
    commit_planes_for_leaf,
    create_proof,
    submit_opened_block,
)
from miner_base.commitment import (
    Device,
    MiningConfiguration,
    commit_planes,
    commitment_keys,
    hash_id_for_leaf,
)
from miner_base.layout import AxisPattern
from miner_base.mining_config import default_mining_config, tall_tile_mining_config
from miner_base.prequant import PrequantMatrix
from miner_base.settings import MinerSettings
from pearl_gateway.blockchain_utils.zk_certificate import CertificateVersion
from pearl_gateway.comm.dataclasses import MiningJob
from pearl_gateway.config import MinerRpcConfig
from pearl_mining import (
    CERT_VERSION_PLAIN_FP8,
    IncompleteBlockHeader,
    verify_plain_proof_for_cert_version,
)

_M = 256
_N = 256
_K = 2048  # cert-v4 verifier floor
# Hard default; the maximum compact target makes every lottery tile win.
DEFAULT_NBITS = 0x173FFFFF
ALWAYS_WIN_NBITS = 0x207FFFFF


def make_plain_peel_header(
    nbits: int = DEFAULT_NBITS, timestamp: int = 1_700_000_000
) -> IncompleteBlockHeader:
    """A static block-template header (no chain to extend)."""
    return IncompleteBlockHeader(
        version=1,
        prev_block=b"\x11" * 32,
        merkle_root=b"\x22" * 32,
        timestamp=timestamp,
        nbits=nbits,
    )


def _tile_indices(pattern: AxisPattern, tile_index: int) -> list[int]:
    base = tile_index * pattern.total
    return [base + offset for offset in pattern.tile_offsets]


def _tall_config(k: int, rank: int = 32) -> MiningConfiguration:
    """The committed 16x32 tall tile used for high-``k`` (GLM o_proj) proofs."""
    return tall_tile_mining_config(k, rank, Device.BLACKWELL)


_TILE_CONFIGS = {
    "4x128": lambda: default_mining_config(k=_K, rank=32, ltile_cols=128),
    "16x32": lambda: _tall_config(k=16384),
}


def _opening(
    *, config: MiningConfiguration | None = None, policy_inadmissible: bool = False
) -> OpenedBlockInfo:
    if config is None:
        config = default_mining_config(k=_K, rank=32, ltile_cols=128)
    k = config.common_dim
    a_values = torch.zeros(_M, k, dtype=torch.bfloat16)
    if policy_inadmissible:
        # 96 unit spikes per row (4.7% of k) exceed the cert-v4 policy's
        # eps_idle = 1/64 entry-liveness budget while remaining a structurally
        # valid, canonically committed opening. Each spike stays dead: the row
        # norm l2 = sqrt(96/2048) ~ 0.217 keeps the dead bound
        # tau_idle * DELTA * l2 ~ 0.87 below the unit spike. (128 spikes would
        # sit exactly on the bound; more would lift l2 above it.)
        a_values[:, :96] = 1
    a = PrequantMatrix.encode(a_values)
    b = PrequantMatrix.encode(torch.zeros(_N, k, dtype=torch.bfloat16))
    return OpenedBlockInfo(
        a_row_indices=tuple(_tile_indices(config.rows_pattern, 0)),
        b_column_indices=tuple(_tile_indices(config.cols_pattern, 0)),
        a_codes=a.int_values,
        a_scales=a.scales,
        b_codes=b.int_values,
        b_scales=b.scales,
        mining_config=config,
    )


def test_commit_planes_for_leaf_forwards_the_leaf():
    """The resolver is ``commit_planes`` at the protocol 1024-byte leaf."""
    planes = [
        torch.zeros(4, _K, dtype=torch.int8),
        torch.zeros(4, _K // 8, dtype=torch.bfloat16),
    ]
    key = bytes(32)
    resolved = commit_planes_for_leaf(planes, key, PROTOCOL_COMMITMENT_LEAF)
    expected = commit_planes(planes, key, hash_id_for_leaf(PROTOCOL_COMMITMENT_LEAF))
    assert resolved.digest == expected.digest
    assert commit_planes_for_leaf(planes, key).digest == resolved.digest


@pytest.mark.parametrize("tile", list(_TILE_CONFIGS))
def test_plain_peel_proof_verifies_and_binds_header(tile):
    header = make_plain_peel_header(nbits=ALWAYS_WIN_NBITS)
    config = _TILE_CONFIGS[tile]()
    proof = create_proof(_opening(config=config), header)

    assert proof.min_cert_version == CERT_VERSION_PLAIN_FP8
    assert proof.a.hash_id == config.a_hash_id.value
    assert proof.b.hash_id == config.b_hash_id.value
    accepted, message = verify_plain_proof_for_cert_version(CERT_VERSION_PLAIN_FP8, header, proof)
    assert accepted, message

    other_header = make_plain_peel_header(nbits=ALWAYS_WIN_NBITS, timestamp=header.timestamp + 1)
    accepted, _ = verify_plain_proof_for_cert_version(CERT_VERSION_PLAIN_FP8, other_header, proof)
    assert not accepted


def test_tall_tile_keeps_high_k_proof_under_the_verifier_cap():
    """At k=16384 (GLM o_proj) the area-equivalent 16x32 tile's peel proof fits
    the 4 MiB worker-input cap the 4x128 tile would blow -- why the tall tile
    exists."""
    k = 16384
    config = _tall_config(k)
    assert config.rows_pattern.tile_size == 16
    assert config.cols_pattern.tile_size == 32
    assert 16 * 32 == 4 * 128  # same area -> difficulty-invariant
    cap = 1 << 22
    assert (16 + 32) * k * 2 <= cap
    assert (4 + 128) * k * 2 > cap


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("a_codes", lambda value: value.to("meta"), "CPU"),
        ("a_codes", lambda value: value.to(torch.int16), "int8"),
        ("a_scales", lambda value: value[:, :-1].contiguous(), "a_scales must be"),
        ("a_row_indices", lambda value: (value[0], value[0], *value[2:]), "lottery tile"),
        ("a_row_indices", lambda value: value[:-1], "exactly one"),
        ("a_row_indices", lambda value: (*value, value[-1] + 1), "exactly one"),
        ("a_row_indices", lambda value: tuple(index + 1 for index in value), "lottery tile"),
    ],
)
def test_opening_validation_rejects_malformed_planes(field, replacement, message):
    opening = _opening()
    malformed = replace(opening, **{field: replacement(getattr(opening, field))})

    with pytest.raises(ValueError, match=message):
        create_proof(malformed, make_plain_peel_header())


def test_opening_validation_rejects_non_integer_indices():
    opening = _opening()
    malformed = replace(opening, a_row_indices=("0", *opening.a_row_indices[1:]))

    with pytest.raises(TypeError, match="must contain integers"):
        create_proof(malformed, make_plain_peel_header())


def test_owned_copy_detaches_mutable_plane_storage():
    opening = _opening()
    owned = opening.owned_copy()

    opening.a_codes.fill_(7)
    opening.a_scales.fill_(3)
    opening.b_codes.fill_(5)
    opening.b_scales.fill_(2)

    assert owned.a_row_indices is opening.a_row_indices
    assert owned.b_column_indices is opening.b_column_indices
    assert owned.mining_config is not opening.mining_config
    assert owned.mining_config.p_a(_M) == opening.mining_config.p_a(_M)
    assert owned.mining_config.p_b(_N) == opening.mining_config.p_b(_N)
    assert not torch.equal(owned.a_codes, opening.a_codes)
    assert not torch.equal(owned.a_scales, opening.a_scales)
    assert not torch.equal(owned.b_codes, opening.b_codes)
    assert not torch.equal(owned.b_scales, opening.b_scales)


def test_owned_copy_reuses_b_planes_when_the_opening_tree_is_prebuilt():
    opening = _opening()
    header = make_plain_peel_header()
    _, key_b = commitment_keys(bytes(header.to_bytes()))
    commitment = commit_planes(
        [opening.b_codes, opening.b_scales], key_b, opening.mining_config.b_hash_id
    )
    opening = replace(
        opening,
        b_commitment=PrebuiltCommitment(commitment=commitment, key=key_b),
    )

    owned = opening.owned_copy()

    assert owned.a_codes is not opening.a_codes
    assert owned.a_scales is not opening.a_scales
    assert owned.b_codes is opening.b_codes
    assert owned.b_scales is opening.b_scales
    assert owned.b_commitment is opening.b_commitment
    assert create_proof(owned, header).to_base64() == create_proof(opening, header).to_base64()


def test_prebuilt_a_commitment_is_reused_without_rehashing_or_recloning():
    opening = _opening()
    header = make_plain_peel_header()
    key_a, key_b = commitment_keys(bytes(header.to_bytes()))
    comm_a = commit_planes(
        [opening.a_codes, opening.a_scales], key_a, opening.mining_config.a_hash_id
    )
    comm_b = commit_planes(
        [opening.b_codes, opening.b_scales], key_b, opening.mining_config.b_hash_id
    )
    opening = replace(
        opening,
        a_commitment=PrebuiltCommitment(comm_a, key_a),
        b_commitment=PrebuiltCommitment(comm_b, key_b),
    )

    owned = opening.owned_copy()

    assert owned.a_codes is opening.a_codes
    assert owned.a_scales is opening.a_scales
    assert owned.a_commitment is opening.a_commitment
    with patch(
        "miner_base.block_submission.commit_planes",
        side_effect=AssertionError("prebuilt operand was committed twice"),
    ):
        proof = create_proof(owned, header)
    assert proof is not None


def test_prebuilt_a_commitment_rejects_wrong_job_and_shape():
    opening = _opening()
    header = make_plain_peel_header()
    key_a, _ = commitment_keys(bytes(header.to_bytes()))
    commitment = commit_planes(
        [opening.a_codes, opening.a_scales], key_a, opening.mining_config.a_hash_id
    )

    wrong_job = replace(
        opening,
        a_commitment=PrebuiltCommitment(commitment, b"wrong"),
    )
    with pytest.raises(ValueError, match="another job"):
        create_proof(wrong_job, header)

    wrong_shape = replace(
        opening,
        a_codes=torch.zeros(_M + 1, _K, dtype=torch.int8),
        a_scales=torch.zeros(_M + 1, _K // 8, dtype=torch.bfloat16),
        a_commitment=PrebuiltCommitment(commitment, key_a),
    )
    with pytest.raises(ValueError, match="covers"):
        create_proof(wrong_shape, header)


def _plain_job() -> MiningJob:
    return MiningJob(
        incomplete_header_bytes=bytes(make_plain_peel_header().to_bytes()),
        target=1,
        cert_version=CertificateVersion.PLAIN_FP8,
    )


def test_submission_rejects_non_fp8_job_before_gateway_handoff():
    job = MiningJob(
        incomplete_header_bytes=bytes(make_plain_peel_header().to_bytes()),
        target=1,
        cert_version=CertificateVersion.ZK_DENSE,
    )
    client = Mock()

    with pytest.raises(ValueError, match="requires certificate version 4"):
        submit_opened_block(_opening(), job, client)

    client.submit_plain_proof.assert_not_called()


def test_real_policy_inadmissibility_sentinel_is_filtered():
    header = make_plain_peel_header(nbits=ALWAYS_WIN_NBITS)
    job = MiningJob(
        incomplete_header_bytes=bytes(header.to_bytes()),
        target=(1 << 256) - 1,
        cert_version=CertificateVersion.PLAIN_FP8,
    )
    opening = _opening(policy_inadmissible=True)
    proof = create_proof(opening, header)

    accepted, message = verify_plain_proof_for_cert_version(CERT_VERSION_PLAIN_FP8, header, proof)
    assert (accepted, message) == (False, "The jackpot is not admissible")

    client = Mock()
    assert submit_opened_block(opening, job, client) is None
    client.submit_plain_proof.assert_not_called()


@patch(
    "miner_base.block_submission.verify_plain_proof_for_cert_version",
    return_value=(False, "commitment mismatch"),
)
def test_unexpected_verifier_failure_remains_fatal(mock_verify):
    client = Mock()

    with pytest.raises(RuntimeError, match="commitment mismatch"):
        submit_opened_block(_opening(), _plain_job(), client)

    mock_verify.assert_called_once()
    client.submit_plain_proof.assert_not_called()


@patch(
    "miner_base.block_submission.verify_plain_proof_for_cert_version",
    return_value=(False, "commitment mismatch"),
)
def test_failed_canonical_proof_is_rejected(mock_verify):
    client = MagicMock()
    client.__enter__.return_value = client
    config = MinerRpcConfig(transport="uds", socket_path="/tmp/unused-gateway.sock")
    manager = AsyncLoopManager(config, MinerSettings(no_gateway=False))

    with (
        patch("miner_base.async_loop_manager._make_client", return_value=client),
        pytest.raises(RuntimeError, match="commitment mismatch"),
    ):
        manager._submit_block(_opening(), _plain_job())

    mock_verify.assert_called_once()
    client.submit_plain_proof.assert_not_called()
