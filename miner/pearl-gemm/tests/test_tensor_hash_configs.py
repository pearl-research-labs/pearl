"""Configuration and validation coverage for tensor hashing."""

from dataclasses import replace

import pytest
import torch
from blake3 import blake3
from miner_base.block_submission import PrebuiltCommitment, _checked_commitment
from miner_base.commitment import commit_planes, hash_id_for_leaf
from miner_base.commitment_hash import a_keys as ref_a_keys
from miner_base.prequant import PrequantMatrix
from pearl_mining import MERKLE_LEAF_SIZE

from pearl_gemm import (
    TensorHashConfig,
    get_tensor_hash_plus_stats_config,
    tensor_hash_plus_stats,
    tensor_hash_plus_stats_b,
    tensor_hash_workspace_bytes,
)
from pearl_gemm.autotune import tensor_hash_launch_space
from pearl_gemm.tensor_hash_plus_stats._merkle_host import SUPPORTED_THREAD_LOAD_SIZES
from tests.helpers.chain import p_a_for

_DEFAULT_CONFIG = TensorHashConfig()
_REPRESENTATIVE_CONFIG = TensorHashConfig(threads_per_block=256, num_stages=3, thread_load_size=256)


def _protocol_leaf(config: TensorHashConfig) -> TensorHashConfig:
    """Overlay the cert-v4 1024-byte leaf on an autotune config's launch knobs."""
    load = config.thread_load_size
    if MERKLE_LEAF_SIZE % load:
        load = max(size for size in SUPPORTED_THREAD_LOAD_SIZES if MERKLE_LEAF_SIZE % size == 0)
    return replace(config, chunk_size=MERKLE_LEAF_SIZE, thread_load_size=load)


def _run(config):
    m, k = 32, 512
    values = (torch.arange(m * k, device="cuda", dtype=torch.int32) % 17) - 8
    a = values.reshape(m, k).to(torch.bfloat16)
    ref = PrequantMatrix.encode(a.cpu())
    codes = ref.int_values.cuda().contiguous()
    scales = ref.scales.cuda().contiguous()
    key_bytes = blake3(b"config-key").digest()
    seed_b_bytes = blake3(b"config-seed-b").digest()
    key = torch.frombuffer(bytearray(key_bytes), dtype=torch.uint8).cuda()
    seed_b = torch.frombuffer(bytearray(seed_b_bytes), dtype=torch.uint8).cuda()
    root_codes = torch.empty(32, dtype=torch.uint8, device="cuda")
    root_scales = torch.empty(32, dtype=torch.uint8, device="cuda")
    roots = torch.empty(
        tensor_hash_workspace_bytes(m, k, config),
        dtype=torch.uint8,
        device="cuda",
    )
    a_keys = torch.empty(96, dtype=torch.uint8, device="cuda")
    stats = torch.zeros(2 * (m * k // 512), dtype=torch.float32, device="cuda")
    p_a = p_a_for(m, k)
    tensor_hash_plus_stats(
        codes,
        scales,
        key,
        seed_b,
        root_codes,
        root_scales,
        roots,
        a_keys,
        stats,
        p_a=p_a,
        config=config,
    )
    torch.cuda.synchronize()

    # Protocol-oracle parity at the cert-v4 leaf. Custom-leaf digests are the
    # hasher matrix in test_tensor_hash_plus_stats; commit_planes cannot open
    # them on this pin.
    resolved = config if config is not None else get_tensor_hash_plus_stats_config(m, k)
    if resolved.chunk_size != MERKLE_LEAF_SIZE:
        return
    commitment = commit_planes(
        [codes.cpu(), scales.cpu()], key_bytes, hash_id_for_leaf(resolved.chunk_size)
    )
    assert bytes(root_codes.cpu().numpy()) == commitment.parts[0].tree.root
    assert bytes(root_scales.cpu().numpy()) == commitment.parts[1].tree.root
    expected = ref_a_keys(commitment.digest, seed_b_bytes, key_bytes, p_a).to_bytes()
    assert bytes(a_keys.cpu().numpy()) == expected


def test_default_config_preserves_digest():
    _run(_DEFAULT_CONFIG)


def test_omitted_config_preserves_digest():
    """``config=None`` shares the autotune record with workspace sizing."""
    _run(None)


def test_representative_config_preserves_digest():
    """One positive (non-default) tuning point on the PR gate."""
    _run(_REPRESENTATIVE_CONFIG)


@pytest.mark.slow
@pytest.mark.parametrize(
    "config_fields",
    [
        *tensor_hash_launch_space(),
        {
            "threads_per_block": 128,
            "num_stages": 2,
            "leaves_per_mt_block": 512,
            "thread_load_size": 128,
        },
        {
            "threads_per_block": 128,
            "num_stages": 2,
            "leaves_per_mt_block": 1024,
            "thread_load_size": 128,
        },
    ],
)
def test_all_tuning_configs_preserve_digest(config_fields):
    _run(TensorHashConfig(**config_fields))


@pytest.mark.parametrize(
    "field,value",
    [
        ("threads_per_block", 64),
        ("num_stages", 1),
        ("leaves_per_mt_block", 128),
        ("thread_load_size", 32),
    ],
)
def test_invalid_config_values_are_rejected(field, value):
    with pytest.raises(ValueError, match=field):
        TensorHashConfig(**{field: value})


def test_resolved_b_config_round_trips_the_proof_chain():
    """The checked-in resolved config for the (n=128, k=2048) weight planes
    -- the winner handoff's B path -- must produce plane roots ``commit_planes``
    rebuilds at the protocol leaf, accepts as a prebuilt commitment, and opens."""
    n, k = 128, 2048
    config = _protocol_leaf(get_tensor_hash_plus_stats_config(n, k))
    values = (torch.arange(n * k, device="cuda", dtype=torch.int32) % 23) - 11
    ref = PrequantMatrix.encode(values.reshape(n, k).to(torch.bfloat16).cpu())
    codes = ref.int_values.cuda().contiguous()
    scales = ref.scales.cuda().contiguous()
    key_bytes = blake3(b"b-proof-key").digest()
    key = torch.frombuffer(bytearray(key_bytes), dtype=torch.uint8).cuda()
    root_codes = torch.empty(32, dtype=torch.uint8, device="cuda")
    root_scales = torch.empty(32, dtype=torch.uint8, device="cuda")
    roots = torch.empty(tensor_hash_workspace_bytes(n, k, config), dtype=torch.uint8, device="cuda")
    stats = torch.zeros(2 * (n * k // 512), dtype=torch.float32, device="cuda")
    tensor_hash_plus_stats_b(
        codes, scales, key, root_codes, root_scales, roots, stats, config=config
    )
    torch.cuda.synchronize()

    # Root recomputation: the CPU tree over the same planes must agree.
    hash_id = hash_id_for_leaf(config.chunk_size)
    commitment = commit_planes([codes.cpu(), scales.cpu()], key_bytes, hash_id)
    assert bytes(root_codes.cpu().numpy()) == commitment.parts[0].tree.root
    assert bytes(root_scales.cpu().numpy()) == commitment.parts[1].tree.root

    # Proof-side acceptance and opening: the same validation and opening the
    # winner handoff runs (``create_proof`` with a prebuilt B commitment).
    prebuilt = PrebuiltCommitment(commitment, key_bytes)
    checked = _checked_commitment(prebuilt, key_bytes, (codes.cpu(), scales.cpu()), hash_id)
    columns = [0, 37, n - 1]
    opened_values, opened_scales = checked.open(columns)
    assert opened_values.row_indices == columns
    assert opened_scales.row_indices == columns
    # The multileaf proofs open against the GPU-committed plane roots.
    assert opened_values.root == bytes(root_codes.cpu().numpy())
    assert opened_scales.root == bytes(root_scales.cpu().numpy())
