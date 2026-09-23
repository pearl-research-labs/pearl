"""The v4 commitment chain (``miner_base.commitment_hash``) against the reference."""

import secrets

import pytest
import torch
from blake3 import blake3
from miner_base.commitment import (
    LABEL_JACKPOT,
    commit_planes,
    commitment_keys,
    jackpot_digest,
    noise_seeds,
    subkey,
)
from miner_base.commitment_hash import (
    AKeys,
    a_keys,
    jackpot_key,
    noise_line_key,
    noise_seed_a,
    noise_seed_b,
    operand_digest,
)
from miner_base.hardware import hardware_for
from miner_base.mining_config import default_mining_config
from miner_base.noise import Noiser


@pytest.fixture
def chain():
    torch.manual_seed(7)
    config = default_mining_config(1024, 32, ltile_cols=64)
    m, n, k = 64, 256, 1024
    key_a, key_b = commitment_keys(secrets.token_bytes(76))
    a_planes = [
        torch.randint(-127, 127, (m, k), dtype=torch.int8),
        torch.rand(m, k // 8).bfloat16(),
    ]
    b_planes = [
        torch.randint(-127, 127, (n, k), dtype=torch.int8),
        torch.rand(n, k // 8).bfloat16(),
    ]
    comm_a = commit_planes(a_planes, key_a, config.a_hash_id)
    comm_b = commit_planes(b_planes, key_b, config.b_hash_id)
    return config, m, n, key_a, key_b, comm_a, comm_b


def test_operand_digest_is_the_planar_commitment_digest(chain):
    config, m, n, key_a, key_b, comm_a, comm_b = chain
    assert operand_digest([part.digest for part in comm_a.parts], key_a) == comm_a.digest
    assert operand_digest([part.digest for part in comm_b.parts], key_b) == comm_b.digest
    # Keyed: another side's key gives another digest of the same roots.
    assert operand_digest([part.digest for part in comm_a.parts], key_b) != comm_a.digest


def test_seeds_match_the_reference_transcript(chain):
    config, m, n, key_a, key_b, comm_a, comm_b = chain
    p_a, p_b = config.p_a(m), config.p_b(n)
    seed_a, seed_b = noise_seeds(comm_a.digest, comm_b.digest, key_a, key_b, p_a, p_b)
    assert noise_seed_b(comm_b.digest, key_b, p_b) == seed_b
    assert noise_seed_a(comm_a.digest, seed_b, key_a, p_a) == seed_a
    keys = a_keys(comm_a.digest, seed_b, key_a, p_a)
    assert keys == AKeys(seed_a, noise_line_key(seed_a), jackpot_key(seed_a))
    assert AKeys.from_bytes(keys.to_bytes()) == keys and len(keys.to_bytes()) == 96


def test_subkeys_are_the_reference_noise_and_jackpot_keys(chain):
    config, m, n, key_a, key_b, comm_a, comm_b = chain
    seed_a, seed_b = noise_seeds(
        comm_a.digest, comm_b.digest, key_a, key_b, config.p_a(m), config.p_b(n)
    )
    compute = hardware_for(config.device).compute
    noise = Noiser(seed_b, 32, 1024, compute, seed_a=seed_a)
    # E_A under seedA's noise-line key; F_A, E_B and F_B under seedB's.
    assert noise._a._key == noise_line_key(seed_a)
    assert noise._a._f_key == noise._b._key == noise._b._f_key == noise_line_key(seed_b)
    # Same seedB, different Side addresses: F_A and F_B are distinct draws.
    assert not _same_bits(noise.F_A(), noise.F_B())
    extracted = secrets.token_bytes(64)
    assert jackpot_key(seed_a) == subkey(LABEL_JACKPOT, seed_a)
    assert blake3(extracted, key=jackpot_key(seed_a)).digest() == jackpot_digest(extracted, seed_a)


def _same_bits(lhs: torch.Tensor, rhs: torch.Tensor) -> bool:
    """Bitwise FP8 equality (``torch.equal`` has no CPU FP8 kernel)."""
    return torch.equal(lhs.contiguous().view(torch.uint8), rhs.contiguous().view(torch.uint8))


def test_only_e_a_moves_with_seed_a(chain):
    """Draw the factors for (A, B), then for (A', B): F_A, E_B and F_B are
    unchanged, only E_A moves. Conversely a new seedB under the same seedA
    moves those three and leaves E_A alone."""
    config, m, n, key_a, key_b, comm_a, comm_b = chain
    seed_a, seed_b = noise_seeds(
        comm_a.digest, comm_b.digest, key_a, key_b, config.p_a(m), config.p_b(n)
    )
    compute = hardware_for(config.device).compute
    rows, cols = list(range(m)), list(range(n))
    job = Noiser(seed_b, 32, 1024, compute)  # B side: exists before any A
    with pytest.raises(ValueError, match="E_A"):
        job.E_A(rows)
    with_a = job.with_seed_a(seed_a)
    with_a_prime = job.with_seed_a(secrets.token_bytes(32))
    for lhs, rhs in ((job, with_a), (with_a, with_a_prime)):
        assert _same_bits(lhs.F_A(), rhs.F_A())
        assert _same_bits(lhs.E_B(cols), rhs.E_B(cols))
        assert _same_bits(lhs.F_B(), rhs.F_B())
    assert not _same_bits(with_a.E_A(rows), with_a_prime.E_A(rows))
    other_b = Noiser(secrets.token_bytes(32), 32, 1024, compute, seed_a=seed_a)
    assert _same_bits(with_a.E_A(rows), other_b.E_A(rows))
    assert not _same_bits(with_a.F_A(), other_b.F_A())
    assert not _same_bits(with_a.E_B(cols), other_b.E_B(cols))
    assert not _same_bits(with_a.F_B(), other_b.F_B())


def test_p_a_binds_shape_leaf_and_pattern(chain):
    config, m, n, key_a, key_b, comm_a, comm_b = chain
    seed_b = noise_seed_b(comm_b.digest, key_b, config.p_b(n))
    base = a_keys(comm_a.digest, seed_b, key_a, config.p_a(m))
    assert a_keys(comm_a.digest, seed_b, key_a, config.p_a(m + 4)) != base
    other_leaf = default_mining_config(1024, 32, ltile_cols=64, a_chunk_size=256)
    assert a_keys(comm_a.digest, seed_b, key_a, other_leaf.p_a(m)) != base
    # The column tile lives in pB: it moves seedB, and through it every A key.
    other_tile = default_mining_config(1024, 32, ltile_cols=128)
    other_seed_b = noise_seed_b(comm_b.digest, key_b, other_tile.p_b(n))
    assert other_seed_b != seed_b
    assert a_keys(comm_a.digest, other_seed_b, key_a, config.p_a(m)) != base


def test_a_keys_rejects_wrong_length():
    with pytest.raises(ValueError, match="96 bytes"):
        AKeys.from_bytes(b"\0" * 64)
