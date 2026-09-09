"""Production ownership tests for plain-peel commitments and configuration."""

import pytest
import torch
from miner_base.commitment import (
    Device,
    HashId,
    MiningConfiguration,
    Quant,
    commit_matrix,
    commit_planes,
    commitment_keys,
    noise_seeds,
)
from miner_base.commitment_hash import noise_seed_a, noise_seed_b, operand_digest
from miner_base.mining_config import COMMITMENT_CHUNK_SIZE, activation_leaf, default_mining_config
from miner_base.prequant import DEFAULT_BLOCK_SIZE, PrequantMatrix

# cert-v4 ``pB`` wire: n(4) | k(4) | r(2) | quant(1) | device(1) | hash_idB(1) | Pcol | e(2)
_P_B_DEVICE_OFFSET = 11
_P_B_HASH_ID_OFFSET = 12
# ``pA`` wire: m(4) | hash_idA(1) | Prow
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
    assert int.from_bytes(p_b[:4], "little") == 256
    assert int.from_bytes(p_b[4:8], "little") == 2048
    assert int.from_bytes(p_b[8:10], "little") == 32
    assert p_b[_P_B_DEVICE_OFFSET] == Device.BLACKWELL.value
    assert p_b[_P_B_HASH_ID_OFFSET] == HashId.BLAKE3_CHUNK_1024.value
    p_a = config.p_a(64)
    assert int.from_bytes(p_a[:4], "little") == 64
    assert p_a[_P_A_HASH_ID_OFFSET] == HashId.BLAKE3_CHUNK_1024.value


def test_split_leaf_changes_only_the_activation_side():
    """Equal leaves are one ``HashId`` on both trees; a split config commits
    the A tree at its own leaf, which only ``pA`` (hence ``seedA``) sees."""
    single = _config()
    equal = default_mining_config(
        2048, rank=32, device=Device.BLACKWELL, a_chunk_size=COMMITMENT_CHUNK_SIZE
    )
    assert equal.p_a(64) == single.p_a(64) and equal.p_b(256) == single.p_b(256)
    assert activation_leaf(equal) == activation_leaf(single) == COMMITMENT_CHUNK_SIZE

    split = default_mining_config(2048, rank=32, device=Device.BLACKWELL, a_chunk_size=128)
    assert split.p_b(256) == single.p_b(256)  # weight side untouched
    assert split.p_a(64) != single.p_a(64)
    assert split.p_a(64)[_P_A_HASH_ID_OFFSET] == HashId.BLAKE3_CHUNK_128.value
    assert split.a_hash_id is HashId.BLAKE3_CHUNK_128
    assert activation_leaf(split) == 128
    assert split.chunk_size == COMMITMENT_CHUNK_SIZE

    with pytest.raises(ValueError, match="no committed Merkle leaf"):
        default_mining_config(2048, rank=32, device=Device.BLACKWELL, a_chunk_size=96)


def test_matrix_commitment_is_keyed_and_records_its_geometry():
    """cert-v4 binds the operand shape through ``pA``/``pB`` (``m``/``n`` and the
    leaf), not the tree root: same bytes reshaped share a root, another side's
    key does not."""
    key = b"k" * 32
    flat = torch.arange(8, dtype=torch.bfloat16)
    first = commit_matrix(flat.reshape(2, 4), key, HashId.BLAKE3_CHUNK_1024)
    second = commit_matrix(flat.reshape(4, 2), key, HashId.BLAKE3_CHUNK_1024)

    assert first.digest == second.digest
    assert (first.rows, first.row_nbytes) == (2, 8)
    assert (second.rows, second.row_nbytes) == (4, 4)
    assert commit_matrix(flat.reshape(2, 4), b"j" * 32, HashId.BLAKE3_CHUNK_1024).digest != (
        first.digest
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


def test_gpu_side_digest_combine_matches_the_cpu_tree():
    """The GPU B path derives ``HB`` (and ``seedB``) from the device plane roots
    alone; that combine must equal the CPU ``commit_planes`` digest."""
    config = _config(1024)
    key_a, key_b = commitment_keys(b"\x11" * 76)
    operand = PrequantMatrix.encode(torch.arange(64 * 1024, dtype=torch.bfloat16).reshape(64, 1024))
    planar_b = commit_planes(operand.planes(), key_b, config.b_hash_id)
    planar_a = commit_planes(operand.planes(), key_a, config.a_hash_id)
    roots_b = [part.digest for part in planar_b.parts]
    roots_a = [part.digest for part in planar_a.parts]

    assert operand_digest(roots_b, key_b) == planar_b.digest
    seed_a, seed_b = noise_seeds(
        planar_a.digest, planar_b.digest, key_a, key_b, config.p_a(64), config.p_b(64)
    )
    assert noise_seed_b(operand_digest(roots_b, key_b), key_b, config.p_b(64)) == seed_b
    assert noise_seed_a(operand_digest(roots_a, key_a), seed_b, key_a, config.p_a(64)) == seed_a


def test_prebuilt_commitment_is_checked_against_the_job_and_planes():
    """The handoff skips ``commit_planes``, so its cheap checks are all that
    stands between a mismatched commitment and an invalid proof (or an
    out-of-range opening)."""
    from miner_base.block_submission import PrebuiltCommitment, _checked_commitment

    key = b"k" * 32
    leaf = HashId.BLAKE3_CHUNK_1024
    operand = PrequantMatrix.encode(torch.arange(64, dtype=torch.bfloat16).reshape(4, 16))
    planes = operand.planes()
    prebuilt = PrebuiltCommitment(commit_planes(planes, key, leaf), key)

    assert _checked_commitment(prebuilt, key, planes, leaf) is prebuilt.commitment

    with pytest.raises(ValueError, match="belongs to another job"):
        _checked_commitment(prebuilt, b"j" * 32, planes, leaf)

    with pytest.raises(ValueError, match="uses leaf 1024"):
        _checked_commitment(prebuilt, key, planes, HashId.BLAKE3_CHUNK_128)

    # Differently shaped planes: also what would open out of range.
    bigger = PrequantMatrix.encode(torch.arange(128, dtype=torch.bfloat16).reshape(8, 16))
    with pytest.raises(ValueError, match="rows"):
        _checked_commitment(prebuilt, key, bigger.planes(), leaf)

    with pytest.raises(ValueError, match="expected 1 planes|has 2 planes"):
        _checked_commitment(prebuilt, key, planes[:1], leaf)

    # A commitment whose own digest does not follow from its trees.
    tampered = PrebuiltCommitment(
        commit_planes(planes, key, leaf).__class__(prebuilt.commitment.parts, b"\x00" * 32), key
    )
    with pytest.raises(ValueError, match="does not match its own planes"):
        _checked_commitment(tampered, key, planes, leaf)


def test_create_proof_rejects_planes_that_break_the_operand_contract():
    """The proof boundary feeds these planes into ``commit_planes`` / Merkle
    serialization, so a rank-1, wrong-dtype, or non-tensor operand must fail
    loudly here rather than as an opaque error (or a wrong proof) downstream."""
    from miner_base.block_submission import _validate_plane

    _validate_plane(torch.zeros(4, 8, dtype=torch.int8), "b_codes", torch.int8)  # accepted

    with pytest.raises(ValueError, match="2D"):
        _validate_plane(torch.zeros(8, dtype=torch.int8), "b_codes", torch.int8)
    with pytest.raises(ValueError, match="int8"):
        _validate_plane(torch.zeros(4, 8, dtype=torch.bfloat16), "b_codes", torch.int8)
    with pytest.raises(ValueError, match="bfloat16"):
        _validate_plane(torch.zeros(4, 8, dtype=torch.int8), "b_scales", torch.bfloat16)
    with pytest.raises(ValueError, match="Tensor"):
        _validate_plane([[1, 2], [3, 4]], "b_codes", torch.int8)


def test_create_proof_rejects_codes_and_scales_that_do_not_correspond():
    """Codes and their block scales are opened in lockstep, so a scale plane
    that isn't ``(rows, cols / block)`` for its codes -- or codes whose columns
    aren't a block multiple -- must be rejected before committing/opening."""
    from miner_base.block_submission import _validate_operand

    rows, cols = 4, 8 * DEFAULT_BLOCK_SIZE
    codes = torch.zeros(rows, cols, dtype=torch.int8)
    scales = torch.zeros(rows, cols // DEFAULT_BLOCK_SIZE, dtype=torch.bfloat16)
    _validate_operand(codes, scales, "b_codes", "b_scales")  # accepted

    with pytest.raises(ValueError, match="b_scales must be"):
        _validate_operand(
            codes, torch.zeros(rows, cols, dtype=torch.bfloat16), "b_codes", "b_scales"
        )
    with pytest.raises(ValueError, match="b_scales must be"):
        _validate_operand(
            codes,
            torch.zeros(rows + 1, cols // DEFAULT_BLOCK_SIZE, dtype=torch.bfloat16),
            "b_codes",
            "b_scales",
        )
    with pytest.raises(ValueError, match="divisible by the block size"):
        _validate_operand(
            torch.zeros(rows, cols + 1, dtype=torch.int8),
            torch.zeros(rows, (cols + 1) // DEFAULT_BLOCK_SIZE, dtype=torch.bfloat16),
            "b_codes",
            "b_scales",
        )
