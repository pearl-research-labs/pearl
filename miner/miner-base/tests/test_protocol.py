"""Tests for cert-v4 protocol parameters, commitments, and device mapping."""

import pytest
import torch
from miner_base.block_submission import PrebuiltCommitment, _checked_commitment, _validate_planes
from miner_base.commitment import commit_matrix, commit_planes
from miner_base.prequant import DEFAULT_BLOCK_SIZE, PrequantMatrix
from miner_base.protocol import (
    COMMITMENT_CHUNK_SIZE,
    Device,
    HashId,
    MiningConfiguration,
    Quant,
    commitment_keys,
    default_mining_config,
    device_for_capability,
    local_device,
    noise_seed_a,
    noise_seed_b,
    operand_digest,
)

# cert-v4 pB wire: k(4) | r(2) | quant(1) | device(1) | n(4) | hash_idB(1) | w(4) | e(2)
_P_B_DEVICE_OFFSET = 7
_P_B_HASH_ID_OFFSET = 12
# pA wire: m(4) | hash_idA(1) | h(4)
_P_A_HASH_ID_OFFSET = 4


def _config(k: int = 2048) -> MiningConfiguration:
    return default_mining_config(k, rank=32, device=Device.BLACKWELL)


def test_configuration_binds_device_rank_and_leaves_into_p_b_and_p_a():
    config = _config()
    assert config.chunk_size == COMMITMENT_CHUNK_SIZE
    assert config.b_hash_id is HashId.BLAKE3_CHUNK_1024
    common = config.common_params()
    assert (common.k, common.r, common.quant, common.device) == (
        2048,
        32,
        Quant.FP8_E4M3_PREQUANT,
        Device.BLACKWELL,
    )
    p_b = config.p_b(256)
    assert int.from_bytes(p_b[:4], "little") == 2048
    assert int.from_bytes(p_b[4:6], "little") == 32
    assert int.from_bytes(p_b[8:12], "little") == 256
    assert p_b[_P_B_DEVICE_OFFSET] == Device.BLACKWELL.value
    assert p_b[_P_B_HASH_ID_OFFSET] == HashId.BLAKE3_CHUNK_1024.value
    p_a = config.p_a(64)
    assert int.from_bytes(p_a[:4], "little") == 64
    assert p_a[_P_A_HASH_ID_OFFSET] == HashId.BLAKE3_CHUNK_1024.value


def test_split_leaf_changes_only_the_activation_side():
    single = _config()
    equal = default_mining_config(
        2048, rank=32, device=Device.BLACKWELL, a_chunk_size=COMMITMENT_CHUNK_SIZE
    )
    assert equal.p_a(64) == single.p_a(64) and equal.p_b(256) == single.p_b(256)
    assert equal.a_chunk_size == single.a_chunk_size == COMMITMENT_CHUNK_SIZE

    split = default_mining_config(2048, rank=32, device=Device.BLACKWELL, a_chunk_size=128)
    assert split.p_b(256) == single.p_b(256)
    assert split.p_a(64) != single.p_a(64)
    assert split.p_a(64)[_P_A_HASH_ID_OFFSET] == HashId.BLAKE3_CHUNK_128.value
    assert split.a_hash_id is HashId.BLAKE3_CHUNK_128
    assert split.a_chunk_size == 128
    assert split.chunk_size == COMMITMENT_CHUNK_SIZE

    with pytest.raises(ValueError, match="no committed Merkle leaf"):
        default_mining_config(2048, rank=32, device=Device.BLACKWELL, a_chunk_size=96)


@pytest.mark.parametrize(
    ("capability", "device"),
    [((9, 0), Device.HOPPER), ((10, 3), Device.BLACKWELL), ((12, 0), Device.BLACKWELL)],
)
def test_capability_resolves_to_the_committed_device(capability, device):
    assert device_for_capability(*capability) is device


@pytest.mark.parametrize("capability", [(8, 9), (11, 0), (13, 0)])
def test_capability_without_a_committed_device_fails_closed(capability):
    with pytest.raises(ValueError, match=f"sm{capability[0]}{capability[1]}"):
        device_for_capability(*capability)


def test_local_device_reads_the_cuda_capability(monkeypatch):
    queried = []

    def capability(device=None):
        queried.append(device)
        return (9, 0)

    monkeypatch.setattr(torch.cuda, "get_device_capability", capability)
    assert local_device() is Device.HOPPER
    assert local_device(1) is Device.HOPPER
    assert queried == [None, 1]


def test_hopper_commits_its_device_into_p_b():
    hopper = default_mining_config(2048, rank=32, device=Device.HOPPER)
    assert hopper.p_b(256)[_P_B_DEVICE_OFFSET] == Device.HOPPER.value
    assert hopper.p_b(256) != _config(2048).p_b(256)


def test_matrix_commitment_is_keyed_and_records_its_geometry():
    key = b"k" * 32
    flat = torch.arange(8, dtype=torch.bfloat16)
    first = commit_matrix(flat.reshape(2, 4), key, HashId.BLAKE3_CHUNK_1024)
    second = commit_matrix(flat.reshape(4, 2), key, HashId.BLAKE3_CHUNK_1024)

    assert first.digest == second.digest
    assert (first.rows, first.row_nbytes) == (2, 8)
    assert (second.rows, second.row_nbytes) == (4, 4)
    assert (
        commit_matrix(flat.reshape(2, 4), b"j" * 32, HashId.BLAKE3_CHUNK_1024).digest
        != first.digest
    )


def test_planar_commitment_combines_the_plane_digests():
    key = b"k" * 32
    operand = PrequantMatrix.encode(torch.arange(32, dtype=torch.bfloat16).reshape(2, 16))
    planar = commit_planes(operand.planes(), key, HashId.BLAKE3_CHUNK_1024)

    assert [part.digest for part in planar.parts] == [
        commit_matrix(operand.int_values, key, HashId.BLAKE3_CHUNK_1024).digest,
        commit_matrix(operand.scales, key, HashId.BLAKE3_CHUNK_1024).digest,
    ]
    assert planar.digest == operand_digest([part.digest for part in planar.parts], key)


def test_key_b_hashes_the_complete_ancestor_header():
    sigma_hat = b"\x11" * 76
    with pytest.raises(ValueError, match="108-byte"):
        commitment_keys(sigma_hat, sigma_hat)
    _, key_b = commitment_keys(sigma_hat, b"\x22" * 108)
    assert commitment_keys(sigma_hat, b"\x22" * 76 + b"\x33" * 32)[1] != key_b


def test_gpu_side_digest_combine_matches_the_cpu_tree():
    config = _config(1024)
    key_a, key_b = commitment_keys(b"\x11" * 76, b"\x22" * 108)
    operand = PrequantMatrix.encode(torch.arange(64 * 1024, dtype=torch.bfloat16).reshape(64, 1024))
    planar_b = commit_planes(operand.planes(), key_b, config.b_hash_id)
    planar_a = commit_planes(operand.planes(), key_a, config.a_hash_id)
    roots_b = [part.digest for part in planar_b.parts]
    roots_a = [part.digest for part in planar_a.parts]

    assert operand_digest(roots_b, key_b) == planar_b.digest
    seed_b = noise_seed_b(planar_b.digest, key_b, config.p_b(64))
    assert noise_seed_b(operand_digest(roots_b, key_b), key_b, config.p_b(64)) == seed_b
    assert noise_seed_a(operand_digest(roots_a, key_a), seed_b, key_a, config.p_a(64)) == (
        noise_seed_a(planar_a.digest, seed_b, key_a, config.p_a(64))
    )


def test_moe_seed_a_binds_the_routing_commitments():
    hash_a, hash_b, key_a, key_b = (bytes([i]) * 32 for i in range(1, 5))
    p_a, p_b = b"pA", b"pB"
    hr, ho = b"\x05" * 32, b"\x06" * 32
    seed_b = noise_seed_b(hash_b, key_b, p_b)
    assert noise_seed_a(hash_a, seed_b, key_a, p_a, hr + ho) != noise_seed_a(
        hash_a, seed_b, key_a, p_a
    )
    with pytest.raises(ValueError, match="64-byte"):
        noise_seed_a(hash_a, seed_b, key_a, p_a, hr)


def test_prebuilt_commitment_is_checked_against_the_job_and_planes():
    key = b"k" * 32
    leaf = HashId.BLAKE3_CHUNK_1024
    operand = PrequantMatrix.encode(torch.arange(64, dtype=torch.bfloat16).reshape(4, 16))
    planes = operand.planes()
    prebuilt = PrebuiltCommitment(commit_planes(planes, key, leaf), key)

    assert _checked_commitment(prebuilt, key, tuple(planes), leaf) is prebuilt.commitment

    with pytest.raises(ValueError, match="belongs to another job"):
        _checked_commitment(prebuilt, b"j" * 32, tuple(planes), leaf)

    with pytest.raises(ValueError, match="uses leaf 1024"):
        _checked_commitment(prebuilt, key, tuple(planes), HashId.BLAKE3_CHUNK_128)

    bigger = PrequantMatrix.encode(torch.arange(128, dtype=torch.bfloat16).reshape(8, 16))
    with pytest.raises(ValueError, match="rows"):
        _checked_commitment(prebuilt, key, tuple(bigger.planes()), leaf)

    with pytest.raises(ValueError, match="expected 1 planes|has 2 planes"):
        _checked_commitment(prebuilt, key, tuple(planes[:1]), leaf)

    tampered = PrebuiltCommitment(
        commit_planes(planes, key, leaf).__class__(prebuilt.commitment.parts, b"\x00" * 32), key
    )
    with pytest.raises(ValueError, match="does not match its own planes"):
        _checked_commitment(tampered, key, tuple(planes), leaf)


def test_validate_planes_enforces_operand_contract():
    rows, cols = 4, 8 * DEFAULT_BLOCK_SIZE
    codes = torch.zeros(rows, cols, dtype=torch.int8)
    scales = torch.zeros(rows, cols // DEFAULT_BLOCK_SIZE, dtype=torch.bfloat16)
    assert _validate_planes("B", codes, scales) == (rows, cols)

    with pytest.raises(ValueError, match="2D"):
        _validate_planes("B", torch.zeros(8, dtype=torch.int8), scales)
    with pytest.raises(ValueError, match="int8"):
        _validate_planes("B", torch.zeros(rows, cols, dtype=torch.bfloat16), scales)
    with pytest.raises(ValueError, match="bfloat16"):
        _validate_planes(
            "B", codes, torch.zeros(rows, cols // DEFAULT_BLOCK_SIZE, dtype=torch.int8)
        )
    with pytest.raises(ValueError, match="Tensor"):
        _validate_planes("B", [[1, 2], [3, 4]], scales)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="b_scales must be"):
        _validate_planes("B", codes, torch.zeros(rows, cols, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="b_scales must be"):
        _validate_planes(
            "B", codes, torch.zeros(rows + 1, cols // DEFAULT_BLOCK_SIZE, dtype=torch.bfloat16)
        )
    with pytest.raises(ValueError, match="divisible by the block size"):
        _validate_planes(
            "B",
            torch.zeros(rows, cols + 1, dtype=torch.int8),
            torch.zeros(rows, (cols + 1) // DEFAULT_BLOCK_SIZE, dtype=torch.bfloat16),
        )
