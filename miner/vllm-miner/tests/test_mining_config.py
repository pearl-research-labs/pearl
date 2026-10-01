"""Protocol-adapter contracts: committed configuration, job keys, thresholds,
credited-hash formulas, and the layer-selection settings."""

import pytest
from miner_base.commitment import BlockHeader, Device, commitment_keys
from miner_base.mining_config import COMMITMENT_CHUNK_SIZE
from miner_base.prequant import DEFAULT_BLOCK_SIZE
from pearl_gateway.blockchain_utils.zk_certificate import CertificateVersion
from pearl_gateway.comm.dataclasses import MiningJob
from vllm_miner.mining_config import (
    DEFAULT_TILE,
    PACKED_NOISE_K,
    RANK,
    SCALE_BLOCK,
    SMALL_TILE,
    TALL_TILE,
    TILE_COLS,
    TILE_ROWS,
    LotteryTileSpec,
    commitment_keys_for,
    effective_work_per_matmul,
    is_mineable_shape,
    lottery_hashes_per_matmul,
    lottery_threshold,
    max_mineable_k,
    max_verifiable_k,
    mining_configuration,
    select_tile,
    threshold_bytes_for,
    tile_indices,
)
from vllm_miner.settings import RuntimeSettings, is_layer_ignored

_K = 2048
_MAX_256 = (1 << 256) - 1

# Host-only contracts: every callee takes the committed device, so both run
# on any machine.
_DEVICES = pytest.mark.parametrize("device", list(Device), ids=lambda d: d.name.lower())


def _tall_committed(device: Device) -> bool:
    """The tall 16x32 tile is committed only on Blackwell: SM90's WGMMA
    fragments cannot fold 16-row words, so tall-tile-only shapes are
    unmineable on Hopper."""
    return device is Device.BLACKWELL


def _job(target: int = 1 << 200) -> MiningJob:
    header = BlockHeader(
        version=1,
        prev_block=b"\x11" * 32,
        merkle_root=b"\x22" * 32,
        timestamp=1_700_000_000,
        nbits=0x1D3FFFFF,
    )
    return MiningJob(
        incomplete_header_bytes=bytes(header.to_bytes()),
        target=target,
        cert_version=CertificateVersion.PLAIN_FP8,
        ancestor_headers=[b"\x33" * 108],
    )


@_DEVICES
def test_mining_configuration_commits_the_kernel_tile(device):
    config = mining_configuration(_K, device=device)
    assert config.common_dim == _K
    assert config.rank == RANK
    assert config.rows_pattern.tile_size == TILE_ROWS
    assert config.cols_pattern.tile_size == TILE_COLS
    # The committed ``pB``/``pA`` (noise-seed inputs) must be well-formed and cached.
    assert len(config.p_b(6144)) == 21  # common params (k, r, device, hash) + B's operand params
    assert len(config.p_a(256)) == 11  # A's operand params alone (dense: no routing)
    assert config.chunk_size == 1024
    assert config.device is device
    assert mining_configuration(_K, device=device) is config


@_DEVICES
def test_mining_configuration_commits_the_protocol_leaf(device):
    """With ``n`` present the committed tile is selected, but both Merkle
    trees stay at the cert-v4 1024-byte leaf. Autotune may pick a faster
    hasher leaf; GPU launches overlay 1024 so pearld can open the proof."""
    n, k = 6144, 16384
    config = mining_configuration(k, n, device=device)
    assert config.chunk_size == COMMITMENT_CHUNK_SIZE
    assert config.a_chunk_size == COMMITMENT_CHUNK_SIZE
    omitted = mining_configuration(k, device=device)
    assert omitted.chunk_size == COMMITMENT_CHUNK_SIZE
    assert omitted.a_chunk_size == COMMITMENT_CHUNK_SIZE


def test_scale_block_matches_reference_and_kernels():
    from pearl_gemm import protocol_constants as kernel_constants

    assert SCALE_BLOCK == DEFAULT_BLOCK_SIZE == kernel_constants.BLOCK_SCALE_GROUP
    assert RANK == kernel_constants.R
    assert PACKED_NOISE_K == kernel_constants.PACKED_NOISE_K


def test_commitment_keys_match_reference_derivation():
    """v4 keys A by the header and B by its parent's complete header; the layer
    shape and tile enter later through ``pA``/``pB`` (see
    ``test_commitment_binds_n_and_k``)."""
    job = _job()
    key_a, key_b = commitment_keys_for(job)
    assert (key_a, key_b) == commitment_keys(bytes(job.incomplete_header_bytes), b"\x33" * 108)
    assert key_a != key_b
    assert commitment_keys_for(_job(target=5)) == (key_a, key_b)  # target is not in the header


@_DEVICES
def test_tile_indices_are_the_committed_offsets(device):
    config = mining_configuration(_K, device=device)
    rows = config.rows_pattern
    cols = config.cols_pattern
    assert tile_indices(rows, 0) == list(rows.tile_offsets)
    assert tile_indices(rows, 1) == [rows.total + off for off in rows.tile_offsets]
    assert tile_indices(cols, 0) == list(cols.tile_offsets)
    assert tile_indices(cols, 3) == [3 * cols.total + off for off in cols.tile_offsets]


@_DEVICES
def test_lottery_threshold_is_area_aware_and_clamped(device):
    """``target * rows*cols*k`` for the committed tile, clamped to ``2^256-1``.
    The 4x64 tile folds 256 elements -- half the 512-element tiles'
    per-message work -- so its threshold halves (its launches hash twice the
    messages)."""
    # n omitted: the back-compat 4x128 commitment.
    assert lottery_threshold(3, _K, device=device) == 3 * 4 * 128 * _K
    assert lottery_threshold(_MAX_256, _K, device=device) == _MAX_256  # clamped
    threshold = threshold_bytes_for(_job(target=3), _K, device=device)
    assert int.from_bytes(threshold, "little") == 3 * 4 * 128 * _K
    assert select_tile(6144, 2048, device=device) == SMALL_TILE
    assert lottery_threshold(3, 2048, 6144, device=device) == 3 * 4 * 64 * 2048
    # n=96 commits the 512-element tall tile on Blackwell; Hopper commits no
    # tile for it, so its threshold is the n-omitted 4x128 one (never launched).
    assert 2 * lottery_threshold(3, 2048, 6144, device=device) == lottery_threshold(
        3, 2048, 96, device=device
    )


@_DEVICES
def test_credited_work_is_mnk_and_rejects_unmineable(device):
    """Difficulty-normalized credit is exactly m*n*k (tiling-invariant, in the
    same unit as the threshold it credits) and is refused -- not silently fitted
    to the default tile -- for shapes no committed tile accepts."""
    m, n, k = 2048, 6144, 16384  # o_proj: past the 4x128 cap; the 4x64 tile commits
    assert select_tile(n, k, device=device) == SMALL_TILE
    assert effective_work_per_matmul(m, n, k, device=device) == m * n * k
    # Independently: it scales linearly in each of m and k. k*2=32768 needs
    # the tall tile (past the 4x64 proof cap), so on Hopper it is refused.
    assert effective_work_per_matmul(m // 2, n, k, device=device) == (m // 2) * n * k
    if _tall_committed(device):
        assert effective_work_per_matmul(m, n, k * 2, device=device) == m * n * (k * 2)
    else:
        with pytest.raises(ValueError, match="not mineable by any committed tile"):
            effective_work_per_matmul(m, n, k * 2, device=device)
    # Message count * per-message work, with the tiling cancelling out --
    # for the committed tile and for any other tile that divides the shape.
    for tile in (SMALL_TILE, TALL_TILE):
        assert effective_work_per_matmul(m, n, k, device=device) == lottery_hashes_per_matmul(
            m, n, tile
        ) * (tile.rows * tile.cols * k)
    # Same unit as the credited threshold: messages * threshold == target * m*n*k.
    target = 1 << 100  # below the clamp
    assert lottery_hashes_per_matmul(m, n, SMALL_TILE) * lottery_threshold(
        target, k, n, device=device
    ) == (target * effective_work_per_matmul(m, n, k, device=device))
    # n is 32-aligned but not 128-aligned: only the tall tile tiles it, and
    # crediting it under the wrong tile raises rather than miscounting.
    assert lottery_hashes_per_matmul(m, 96, TALL_TILE) == (m // 16) * (96 // 32)
    with pytest.raises(ValueError, match="tile exactly"):
        lottery_hashes_per_matmul(m, 96, DEFAULT_TILE)
    # A shape no committed tile accepts (k below the verifier's floor) is refused.
    assert select_tile(n=256, k=512, device=device) is None
    with pytest.raises(ValueError, match="not mineable by any committed tile"):
        effective_work_per_matmul(256, 256, 512, device=device)
    # A nonpositive k is refused before any tile is consulted.
    for bad_k in (0, -512):
        with pytest.raises(ValueError, match="must be positive"):
            effective_work_per_matmul(m, n, bad_k, device=device)
    # The count is the gateway's credited amount, so floats/bools are rejected
    # (bool is an int subclass, so it must be excluded explicitly) at every entry
    # that feeds a stored config: the tile spec, tile selection, and the credit.
    for bad in (256.0, True):
        with pytest.raises(TypeError, match="must be an int"):
            lottery_hashes_per_matmul(bad, 256)
        with pytest.raises(TypeError, match="must be an int"):
            effective_work_per_matmul(256, 512, bad, device=device)
        with pytest.raises(TypeError, match="must be an int"):
            select_tile(bad, 1024, device=device)
        with pytest.raises(TypeError, match="must be an int"):
            select_tile(256, bad, device=device)
        with pytest.raises(TypeError, match="must be an int"):
            LotteryTileSpec(bad, 128)


@_DEVICES
def test_mineable_domain_and_tile_selection(device):
    """The mineable (n, k) domain and the tile it commits: the 4x64 tile is
    preferred wherever it fits (n % 64, 1024 <= k <= 61440) -- the narrowest
    4-row tile mixed_gemm's 64-row decode kernel tile can run -- the tall
    16x32 covers what it cannot (n % 32 not 64-aligned, or k in
    (61440, 65536]), the default 4x128 caps at k <= 31744, all stay within
    the verifier's 2^22 element limit, and n is bounded exclusively below 2^24.
    Out-of-domain shapes select no tile, so ``select_tile`` and
    ``is_mineable_shape`` agree."""
    assert max_verifiable_k() == 31744  # 4x128 cap
    assert max_verifiable_k(tile_cols=32, tile_rows=16) == 65536  # tall cap (MAX_K)
    assert max_verifiable_k(tile_cols=64, tile_rows=4) == 61440  # 4x64 cap
    assert max_mineable_k() == 65536  # full domain across tiles (MAX_K)
    # 4x64 preferred wherever it fits: it unlocks the 64-row decode tile.
    for n, k in ((256, 1024), (256, 31744), (6144, 16384), (6144, 61440)):
        assert select_tile(n, k, device=device) == SMALL_TILE
        assert is_mineable_shape(n=n, k=k, device=device)
    # The tall tile covers what 4x64 cannot: 32-aligned n that is not
    # 64-aligned, and k past the 4x64 proof cap. On Hopper the tall tile is
    # not committed, so these shapes are refused instead of mis-tiled.
    for n, k in ((96, 4096), (6144, 65536), (256, 61440 + 512)):
        if _tall_committed(device):
            assert select_tile(n, k, device=device) == TALL_TILE
            assert is_mineable_shape(n=n, k=k, device=device)
        else:
            assert select_tile(n, k, device=device) is None
            assert not is_mineable_shape(n=n, k=k, device=device)
    # Out of domain -> no tile: under one 32-col tile, below the verifier's
    # k floor (1024), k not 512-aligned, or past the tall cap (MAX_K).
    for n, k in ((16, 4096), (256, 512), (256, 768), (256, 4104), (256, 65536 + 512)):
        assert select_tile(n, k, device=device) is None
        assert not is_mineable_shape(n=n, k=k, device=device)
    # Peel proof stays within the verifier's 2^22 element limit for every tile.
    assert (TILE_ROWS + TILE_COLS) * max_verifiable_k() <= 1 << 22
    assert (16 + 32) * max_verifiable_k(tile_cols=32, tile_rows=16) <= 1 << 22
    assert (4 + 64) * max_verifiable_k(tile_cols=64, tile_rows=4) <= 1 << 22
    # The public dimension cap is exclusive: n == 2^24 is rejected after the
    # (n, k) buffers allocate; the largest aligned dim below it stays mineable.
    # (2^24 - 32 is 32- but not 64-aligned: tall-tile-only, so Blackwell-only.)
    assert not is_mineable_shape(n=1 << 24, k=16384, device=device)
    assert is_mineable_shape(n=(1 << 24) - 32, k=16384, device=device) == _tall_committed(device)
    assert is_mineable_shape(n=(1 << 24) - 64, k=16384, device=device)


def test_select_tile_respects_passed_device():
    """Hopper has no 16x32 family; Blackwell does. Callers must not rely on
    the implicit current CUDA device when the target GPU is known."""
    assert select_tile(96, 2048, device=Device.BLACKWELL) == TALL_TILE
    assert select_tile(96, 2048, device=Device.HOPPER) is None
    assert is_mineable_shape(96, 2048, device=Device.BLACKWELL)
    assert not is_mineable_shape(96, 2048, device=Device.HOPPER)
    # Both families commit the 4-row tiles, each under its own device.
    for device in Device:
        assert select_tile(6144, 16384, device=device) == SMALL_TILE
        assert mining_configuration(16384, 6144, device=device).device is device
    assert mining_configuration(16384, 6144, device=Device.HOPPER).p_b(
        6144
    ) != mining_configuration(16384, 6144, device=Device.BLACKWELL).p_b(6144)


@_DEVICES
def test_commitment_binds_n_and_k(device):
    """The runtime always passes n, so every layer commits the tile it mines:
    passing a 64-aligned n commits the 4x64 tile (differing from the n-omitted
    4x128 back-compat commitment), and both n and k are bound into the
    committed ``pA``/``pB`` (hence the noise seeds) -- a caller that
    omits/mismatches n publishes a commitment that matches no launch
    (``job_prep.prepare_layer`` target-only fast-path regression guard)."""
    for k in (1024, 31744, 16384):  # 16384 is within the 4x64 cap
        with_n = mining_configuration(k, 6144, device=device)
        without_n = mining_configuration(k, device=device)
        assert (with_n.rows_pattern.tile_size, with_n.cols_pattern.tile_size) == (4, 64)
        assert without_n.rows_pattern.tile_size == TILE_ROWS  # 4x128 back-compat
        assert without_n.cols_pattern.tile_size == TILE_COLS
        assert with_n.p_b(6144) != without_n.p_b(6144)  # tile bound (pB carries the cols pattern)
    # k past the 4x64 proof cap falls back to the tall tile even at 64-aligned
    # n. On Hopper no committed tile takes it: the builder emits the 4x128
    # back-compat commitment, but ``select_tile`` refuses the shape so the
    # runtime (which gates on ``can_mine_layer``) never launches it.
    if _tall_committed(device):
        assert mining_configuration(k=65536, n=6144, device=device).rows_pattern.tile_size == 16
    else:
        assert select_tile(6144, 65536, device=device) is None
        assert (
            mining_configuration(k=65536, n=6144, device=device).rows_pattern.tile_size == TILE_ROWS
        )
    assert mining_configuration(16384, 6144, device=device).p_b(6144) != mining_configuration(
        12288, 6144, device=device
    ).p_b(6144)  # k bound
    assert mining_configuration(16384, 6144, device=device).p_b(6144) != mining_configuration(
        16384, 6144, device=device
    ).p_b(6208)  # n bound


@_DEVICES
def test_tile_layout_matches_reference_and_kernel(device):
    """Committed layouts must match what the verifier re-folds: the 4-row
    family comes from the one miner-base builder (a local copy could drift the
    job key), each tile exposes exactly the kernel's 16 fold lanes over its
    rows*cols elements, and the tall ``tile_indices`` enumerate with 16-row /
    32-col strides."""
    from miner_base.layout import lane_assignment
    from miner_base.mining_config import default_mining_config, tall_tile_mining_config

    # The 4-row family is single-sourced from the miner-base builder (which
    # must be given the runtime's committed device to compare bytes).
    shared = default_mining_config(
        31744, RANK, ltile_cols=TILE_COLS, ltile_rows=TILE_ROWS, device=device
    )
    assert mining_configuration(31744, device=device).p_b(6144) == shared.p_b(6144)
    small_shared = default_mining_config(
        16384, RANK, ltile_cols=SMALL_TILE.cols, ltile_rows=SMALL_TILE.rows, device=device
    )
    assert mining_configuration(16384, 6144, device=device).p_b(6144) == small_shared.p_b(6144)
    assert mining_configuration(16384, 6144, device=device).p_b(6144) != shared.p_b(6144)

    # The 16x32 layout checks are device-independent; only Blackwell actually
    # commits it through ``mining_configuration``.
    tall = (
        mining_configuration(65536, 6144, device=device)
        if _tall_committed(device)
        else tall_tile_mining_config(65536, RANK, Device.BLACKWELL)
    )
    tall_lanes = lane_assignment(tall.rows_pattern, tall.cols_pattern)
    # 16x32: lane j is row j's 32 contiguous columns, ascending.
    assert tall_lanes == [
        [j * TALL_TILE.cols + c for c in range(TALL_TILE.cols)] for j in range(TALL_TILE.rows)
    ]
    default = mining_configuration(4096, device=device)
    default_lanes = lane_assignment(default.rows_pattern, default.cols_pattern)
    small = mining_configuration(4096, 6144, device=device)
    small_lanes = lane_assignment(small.rows_pattern, small.cols_pattern)
    # Every tile folds its rows*cols elements over the same 16 kernel lanes
    # (the 4x64 area is 256, half the others': its threshold halves to match).
    for lanes, tile in (
        (default_lanes, DEFAULT_TILE),
        (tall_lanes, TALL_TILE),
        (small_lanes, SMALL_TILE),
    ):
        assert len(lanes) == 16  # mixed_gemm's LANES
        assert sum(len(lane) for lane in lanes) == tile.rows * tile.cols
    # tile_indices enumerates whole tall tiles: 16-row and 32-col strides.
    assert tile_indices(tall.rows_pattern, 1) == list(range(16, 32))
    assert tile_indices(tall.cols_pattern, 1) == list(range(32, 64))


def test_m_buckets_parse_and_validate(monkeypatch):
    monkeypatch.setenv("PEARL_M_BUCKETS", "8192,2048")
    assert RuntimeSettings().m_buckets == (2048, 8192)
    # 64-multiples below 256 are legal: decode batches use them and mine under
    # the 64-row kernel tile instead of a 256-row bucket.
    monkeypatch.setenv("PEARL_M_BUCKETS", "64,2048")
    assert RuntimeSettings().m_buckets == (64, 2048)
    for invalid in ("1000", "0", "-256", "32"):
        monkeypatch.setenv("PEARL_M_BUCKETS", invalid)
        with pytest.raises(ValueError, match="positive multiples of 64"):
            RuntimeSettings()
    # A bucket large enough that bucket * max_mineable_k overflows the hit
    # signal's uint32 A-codes plane is rejected early, not at signal allocation.
    overflow = (((1 << 32) - 1) // max_mineable_k() // 256 + 1) * 256
    monkeypatch.setenv("PEARL_M_BUCKETS", str(overflow))
    with pytest.raises(ValueError, match="overflows the hit signal"):
        RuntimeSettings()


@pytest.mark.parametrize("value", [0, float("inf"), float("nan")])
def test_oom_cooldown_is_positive_and_finite(value):
    with pytest.raises(ValueError):
        RuntimeSettings(oom_cooldown_s=value)
    assert RuntimeSettings().oom_cooldown_s == 30.0


def test_ignored_layers_use_vllm_ignore_semantics(monkeypatch):
    monkeypatch.setenv("PEARL_IGNORED_LAYERS", "model.layers.0.mlp.down_proj, re:.*\\.qkv_proj")
    ignored = RuntimeSettings().ignored_layers
    assert is_layer_ignored("model.layers.0.mlp.down_proj", ignored)  # exact
    assert is_layer_ignored("model.layers.7.self_attn.qkv_proj", ignored)  # regex
    assert not is_layer_ignored("model.layers.1.mlp.down_proj", ignored)  # no substring match
    assert not is_layer_ignored("model.layers.0.mlp.up_proj", ignored)

    monkeypatch.setenv("PEARL_IGNORED_LAYERS", "re:[unclosed")
    with pytest.raises(ValueError, match="invalid ignored-layers regex"):
        RuntimeSettings()


def test_default_settings_mine_everything():
    settings = RuntimeSettings()
    assert settings.ignored_layers == ()
    assert not is_layer_ignored("model.layers.0.mlp.gate_up_proj", settings.ignored_layers)
