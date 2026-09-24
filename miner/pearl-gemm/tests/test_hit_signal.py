"""Unit tests: the mining hit signal (capture-safe, in-process consumer).

Validate the state/protocol invariants of the signal both without the
producer kernel (records forged with plain host stores into the pinned
record, payloads poked over the device regions) and against the real
producer: easy-target ``mixed_gemm`` launches that publish snapshot-carrying
hits, the first-wins latch, payload-less publishes, and hits fired from
CUDA-graph replays with in-place threshold updates (no recapture).
"""

import threading
import time

import pytest
import torch

from pearl_gemm import (
    Hit,
    HitRecordLayout,
    HitSignal,
    HitSignalConfig,
    HitSignalPoisonedError,
    MixedGemmConfig,
    mixed_gemm,
)
from pearl_gemm.pow import HIT_PAYLOAD_K_ALIGN, HIT_RECORD_MAGIC_WORDS
from pearl_gemm.protocol_constants import BLOCK_SCALE_GROUP, R

STATUS_IDLE = 0
STATUS_PUBLISHED = 1
DEFAULT_MAX_M = 8
DEFAULT_MAX_K = 512
FORGE_M = 4
FORGE_N = 128
FORGE_K = 64
FORGE_LAYER_ID = 7
KERNEL_LAYER_ID = 42
OVERSIZED_DIM = 1 << 20


def _make_signal(max_m=DEFAULT_MAX_M, max_k=DEFAULT_MAX_K) -> HitSignal:
    return HitSignal(HitSignalConfig(max_m=max_m, max_k=max_k))


def _forge_hit(
    signal: HitSignal,
    m=FORGE_M,
    n=FORGE_N,
    k=FORGE_K,
    with_payload=True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Simulate a kernel publish with plain host stores: payloads, record
    fields, magic, then the status doorbell LAST."""
    codes = torch.randint(-127, 128, (m, k), dtype=torch.int8)
    scales = torch.rand(m, k // 8, dtype=torch.bfloat16)
    if with_payload:
        signal.codes_payload.narrow(0, 0, m * k).copy_(codes.reshape(-1))
        signal.scales_payload.narrow(0, 0, m * (k // 8)).copy_(scales.reshape(-1))
        torch.cuda.synchronize()
    record = signal.record
    record[HitRecordLayout.M] = m
    record[HitRecordLayout.N] = n
    record[HitRecordLayout.K] = k
    record[HitRecordLayout.TILE_ROW] = 0
    record[HitRecordLayout.TILE_COLUMN] = 0
    record[HitRecordLayout.LTILE_ROWS] = 4
    record[HitRecordLayout.LTILE_COLS] = 128
    record[HitRecordLayout.CODES_PAYLOAD_BYTES] = m * k if with_payload else 0
    record[HitRecordLayout.SCALES_PAYLOAD_BYTES] = m * (k // 8) * 2 if with_payload else 0
    record[HitRecordLayout.LAYER_ID] = FORGE_LAYER_ID
    for word in range(8):
        record[HitRecordLayout.TARGET + word] = word + 1
        record[HitRecordLayout.HASH_A + word] = word + 100
        record[HitRecordLayout.HASH_B + word] = word + 200
    record[HitRecordLayout.MAGIC] = HIT_RECORD_MAGIC_WORDS[0]
    record[HitRecordLayout.MAGIC + 1] = HIT_RECORD_MAGIC_WORDS[1]
    record[HitRecordLayout.STATUS] = STATUS_PUBLISHED
    return codes, scales


def test_hit_preserves_legacy_positional_constructor_order():
    codes = torch.ones(1, 8, dtype=torch.int8)
    scales = torch.ones(1, 1, dtype=torch.bfloat16)
    hit = Hit(
        True,
        1,
        2,
        8,
        3,
        4,
        5,
        6,
        7,
        b"target",
        b"hash-a",
        b"hash-b",
        codes,
        scales,
    )

    assert hit.layer_id == 7
    assert hit.target == b"target"
    assert hit.commitment_hash_A == b"hash-a"
    assert hit.commitment_hash_B == b"hash-b"
    assert hit.codes is codes
    assert hit.scales is scales


def test_offsets_are_sane():
    layout = HitRecordLayout
    assert layout.STATUS == 0
    assert layout.M < layout.N < layout.K < layout.TILE_ROW
    assert layout.CODES_PAYLOAD_BYTES < layout.SCALES_PAYLOAD_BYTES < layout.LAYER_ID
    assert layout.LAYER_ID < layout.TARGET
    assert layout.TARGET + 8 <= layout.HASH_A
    assert layout.HASH_A + 8 <= layout.HASH_B
    assert layout.HASH_B + 8 <= layout.MAGIC
    assert layout.MAGIC + 2 <= layout.RECORD_WORDS


def test_allocation_idle_at_init():
    signal = _make_signal()
    assert signal.codes_payload_capacity_bytes == DEFAULT_MAX_M * DEFAULT_MAX_K
    assert signal.scales_payload_capacity_bytes == DEFAULT_MAX_M * (DEFAULT_MAX_K // 8) * 2
    assert not signal.doorbell()
    assert signal.read_hit() is None
    assert int(signal.lock.item()) == 0


def test_config_validated_before_allocating():
    """Bad configs must raise cleanly instead of attempting an absurd
    (possibly OOM-ing) allocation first."""
    with pytest.raises(ValueError, match="positive"):
        HitSignalConfig(max_m=0, max_k=DEFAULT_MAX_K)
    with pytest.raises(ValueError, match="divisible by 64"):
        HitSignalConfig(max_m=DEFAULT_MAX_M, max_k=DEFAULT_MAX_K + 1)
    with pytest.raises(ValueError, match="overflows uint32"):
        HitSignalConfig(max_m=1 << 20, max_k=1 << 20)
    with pytest.raises(TypeError, match="must be ints"):
        HitSignalConfig(max_m=True, max_k=DEFAULT_MAX_K)
    with pytest.raises(TypeError, match="must be ints"):
        HitSignalConfig(max_m=1.0, max_k=float(DEFAULT_MAX_K))
    # Gemma-3 31B k=5376: 64-aligned, previously rejected as not a multiple of 512.
    HitSignalConfig(max_m=DEFAULT_MAX_M, max_k=5376)


def test_hit_signal_rejects_non_cuda_device():
    with pytest.raises(ValueError, match="CUDA device"):
        HitSignal(HitSignalConfig(max_m=DEFAULT_MAX_M, max_k=DEFAULT_MAX_K), device="cpu")


def test_doorbell_gates_read():
    """A forged record without the status flip is invisible -- idle polling
    is one native acquire load and never parses the record."""
    signal = _make_signal()
    _forge_hit(signal)
    signal.record[HitRecordLayout.STATUS] = STATUS_IDLE  # undo the ring
    assert not signal.doorbell()
    assert signal.read_hit() is None
    signal.record[HitRecordLayout.STATUS] = STATUS_PUBLISHED
    assert signal.doorbell()
    hit = signal.read_hit()
    assert hit is not None and hit.valid
    signal.reset_hit()


def test_read_hit_and_reset_protocol():
    signal = _make_signal()
    codes, scales = _forge_hit(signal)

    assert signal.doorbell()
    hit = signal.read_hit()
    assert hit is not None and hit.valid
    assert hit.m == FORGE_M and hit.n == FORGE_N and hit.k == FORGE_K
    assert (hit.tile_row, hit.tile_column) == (0, 0)
    assert (hit.ltile_rows, hit.ltile_cols) == (4, 128)
    assert hit.layer_id == FORGE_LAYER_ID
    assert hit.target == b"".join((w + 1).to_bytes(4, "little") for w in range(8))
    assert hit.commitment_hash_A == b"".join((w + 100).to_bytes(4, "little") for w in range(8))
    assert hit.commitment_hash_B == b"".join((w + 200).to_bytes(4, "little") for w in range(8))
    assert torch.equal(hit.codes, codes)
    assert torch.equal(hit.scales.view(torch.uint16), scales.view(torch.uint16))

    # read does NOT consume: still readable until reset
    assert signal.read_hit() is not None

    signal.reset_hit()
    assert not signal.doorbell()
    assert signal.read_hit() is None
    # reset zeroed the magic: re-ringing the doorbell alone parses invalid
    signal.record[HitRecordLayout.STATUS] = STATUS_PUBLISHED
    hit = signal.read_hit()
    assert hit is not None and not hit.valid
    signal.reset_hit()


def test_owned_hit_payload_survives_rearm_and_next_read():
    signal = _make_signal()
    first_codes, first_scales = _forge_hit(signal)
    hit = signal.read_hit()
    assert hit is not None and hit.valid
    owned = hit.owned_copy()

    signal.reset_hit()
    second_codes, _ = _forge_hit(signal)
    assert not torch.equal(first_codes, second_codes)
    signal.read_hit()

    assert torch.equal(owned.codes, first_codes)
    assert torch.equal(
        owned.scales.view(torch.uint16),
        first_scales.view(torch.uint16),
    )
    signal.reset_hit()


def test_two_consumers_cannot_take_the_same_record():
    signal = _make_signal()
    _forge_hit(signal)
    barrier = threading.Barrier(3)
    results = []

    def consume():
        barrier.wait()
        results.append(signal.take_owned_hit())

    threads = [threading.Thread(target=consume) for _ in range(2)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(timeout=30)
        assert not thread.is_alive()

    assert sum(hit is not None for hit in results) == 1
    assert not signal.doorbell()


def test_reset_does_not_reopen_an_unpublished_producer_claim():
    signal = _make_signal()
    signal.lock.fill_(1)
    signal.record[HitRecordLayout.STATUS] = STATUS_IDLE

    # Public hardening: no published doorbell means this consumer owns nothing.
    signal.reset_hit()
    assert int(signal.lock.item()) == 1

    # Device-side defense in depth applies even to the private reset launch.
    signal._reset_hit_locked()
    assert int(signal.lock.item()) == 1

    _forge_hit(signal)
    signal._reset_hit_locked()
    assert int(signal.lock.item()) == 0
    assert not signal.doorbell()


def test_take_owned_hit_rearms_after_snapshot_failure(monkeypatch):
    signal = _make_signal()
    _forge_hit(signal)

    def fail_read():
        raise RuntimeError("synthetic D2H failure")

    monkeypatch.setattr(signal, "_read_published_hit_locked", fail_read)
    with pytest.raises(RuntimeError, match="D2H failure"):
        signal.take_owned_hit()

    # The real reset ran before the original snapshot error escaped.
    assert not signal.doorbell()
    signal.require_usable()


def test_reset_failure_irreversibly_poisons_signal(monkeypatch):
    signal = _make_signal()
    _forge_hit(signal)
    monkeypatch.setattr(
        signal,
        "_reset_hit_locked",
        lambda: (_ for _ in ()).throw(RuntimeError("synthetic reset failure")),
    )

    with pytest.raises(HitSignalPoisonedError, match="mining must stop"):
        signal.take_owned_hit()
    with pytest.raises(HitSignalPoisonedError, match="poisoned"):
        signal.require_usable()
    with pytest.raises(HitSignalPoisonedError, match="poisoned"):
        signal.reset_hit()


def test_snapshot_and_reset_failure_preserve_both_causes(monkeypatch):
    signal = _make_signal()
    _forge_hit(signal)
    monkeypatch.setattr(
        signal,
        "_read_published_hit_locked",
        lambda: (_ for _ in ()).throw(ValueError("snapshot failed")),
    )
    monkeypatch.setattr(
        signal,
        "_reset_hit_locked",
        lambda: (_ for _ in ()).throw(RuntimeError("reset failed")),
    )

    with pytest.raises(HitSignalPoisonedError) as captured:
        signal.take_owned_hit()

    group = captured.value.__cause__
    assert isinstance(group, BaseExceptionGroup)
    assert [str(error) for error in group.exceptions] == [
        "snapshot failed",
        "reset failed",
    ]


def test_payloadless_record_is_valid_without_planes():
    """Payload bytes == 0 (hit bigger than the regions): the record is valid
    and complete, but carries no snapshot."""
    signal = _make_signal()
    _forge_hit(signal, with_payload=False)
    hit = signal.read_hit()
    assert hit is not None and hit.valid
    assert hit.codes is None and hit.scales is None
    assert hit.m == FORGE_M and hit.k == FORGE_K
    signal.reset_hit()


def test_malformed_records_rejected():
    signal = _make_signal()

    # codes_payload_bytes inconsistent with m*k
    _forge_hit(signal)
    signal.record[HitRecordLayout.CODES_PAYLOAD_BYTES] = FORGE_M * FORGE_K - 1
    hit = signal.read_hit()
    assert hit is not None and not hit.valid
    signal.reset_hit()

    # payload bytes claim more than the region capacity
    _forge_hit(signal)
    signal.record[HitRecordLayout.M] = OVERSIZED_DIM
    signal.record[HitRecordLayout.CODES_PAYLOAD_BYTES] = OVERSIZED_DIM * FORGE_K
    hit = signal.read_hit()
    assert hit is not None and not hit.valid
    signal.reset_hit()

    # winning tile outside the recorded problem
    _forge_hit(signal)
    signal.record[HitRecordLayout.TILE_ROW] = FORGE_M  # tile_row * ltile_rows >= m
    hit = signal.read_hit()
    assert hit is not None and not hit.valid
    signal.reset_hit()

    # zeroed dims never parse
    _forge_hit(signal)
    signal.record[HitRecordLayout.K] = 0
    hit = signal.read_hit()
    assert hit is not None and not hit.valid
    signal.reset_hit()

    # k divisible by the scale group but not the 16-byte payload copy
    for bad_k in (BLOCK_SCALE_GROUP, HIT_PAYLOAD_K_ALIGN + BLOCK_SCALE_GROUP):
        _forge_hit(signal, k=bad_k)
        hit = signal.read_hit()
        assert hit is not None and not hit.valid
        signal.reset_hit()


def test_record_words_parse_unsigned():
    """The record is uint32 end to end: a top-bit word (the config admits
    payload bytes and dims through UINT32_MAX) must not parse negative and
    flunk the consumer's equality checks."""
    signal = _make_signal()
    _forge_hit(signal)
    signal.record[HitRecordLayout.N] = 1 << 31
    hit = signal.read_hit()
    assert hit is not None and hit.valid
    assert hit.n == 1 << 31
    signal.reset_hit()


# --------------------------------------------------------------------------
# Real producer: easy-target mixed_gemm publishes into the signal.
# --------------------------------------------------------------------------


def _gemm_buffers(
    m=256,
    n=128,
    k=512,
    signal: HitSignal | None = None,
    config: MixedGemmConfig | None = None,
):
    return {
        "a_prime": torch.randn(m, k, device="cuda").to(torch.float8_e4m3fn),
        "b_prime": torch.randn(n, k, device="cuda").to(torch.float8_e4m3fn),
        "a_peel": torch.zeros(m, 2 * R, dtype=torch.bfloat16, device="cuda"),
        "b_peel": torch.zeros(n, 2 * R, dtype=torch.bfloat16, device="cuda"),
        "alpha_a": torch.ones(m, dtype=torch.bfloat16, device="cuda"),
        "inv_alpha_b": torch.ones(n, dtype=torch.float32, device="cuda"),
        "pow_key": torch.randint(0, 256, (32,), dtype=torch.uint8, device="cuda"),
        "threshold": torch.full((32,), 255, dtype=torch.uint8, device="cuda"),
        "out": torch.zeros(m, n, dtype=torch.bfloat16, device="cuda"),
        "hit_signal": signal if signal is not None else _make_signal(max_m=m, max_k=k),
        "a_codes": torch.randint(-127, 128, (m, k), dtype=torch.int8, device="cuda"),
        "a_scales": torch.rand(m, k // 8, dtype=torch.bfloat16, device="cuda"),
        "commitment_hash_b": torch.randint(0, 256, (32,), dtype=torch.uint8, device="cuda"),
        "config": config if config is not None else MixedGemmConfig(cluster_m=1, cluster_n=1),
        "layer_id": KERNEL_LAYER_ID,
    }


def test_record_hits_false_executes_without_claiming_the_signal():
    buffers = _gemm_buffers()
    signal = buffers["hit_signal"]

    mixed_gemm(**buffers, record_hits=False)
    torch.cuda.synchronize()

    assert not signal.doorbell()
    assert int(signal.lock.item()) == 0


def test_record_hits_toggle_reuses_one_compiled_variant():
    from pearl_gemm.mixed_gemm import _host

    buffers = _gemm_buffers(m=384)
    mixed_gemm(**buffers, record_hits=False)
    torch.cuda.synchronize()
    variants_after_disabled = len(_host._compile_cache)

    mixed_gemm(**buffers, record_hits=True)
    torch.cuda.synchronize()

    assert len(_host._compile_cache) == variants_after_disabled
    assert buffers["hit_signal"].doorbell()
    buffers["hit_signal"].reset_hit()


def test_kernel_publish_snapshot_matches_source():
    buffers = _gemm_buffers()
    signal = buffers["hit_signal"]
    mixed_gemm(**buffers)
    torch.cuda.synchronize()

    assert signal.doorbell()
    hit = signal.read_hit()
    assert hit is not None and hit.valid
    m, k = buffers["a_prime"].shape
    n = buffers["b_prime"].shape[0]
    assert (hit.m, hit.n, hit.k) == (m, n, k)
    assert (hit.ltile_rows, hit.ltile_cols) == (4, 128)
    assert 0 <= hit.tile_row < m // 4
    assert 0 <= hit.tile_column < n // 128
    assert hit.layer_id == KERNEL_LAYER_ID
    assert hit.target == b"\xff" * 32
    assert hit.commitment_hash_A == bytes(buffers["pow_key"].cpu().tolist())
    assert hit.commitment_hash_B == bytes(buffers["commitment_hash_b"].cpu().tolist())
    assert torch.equal(hit.codes, buffers["a_codes"].cpu())
    assert torch.equal(hit.scales.view(torch.uint16), buffers["a_scales"].cpu().view(torch.uint16))
    signal.reset_hit()


def test_hit_visible_without_cuda_synchronize():
    """Publication is a release/acquire pair, not a CUDA API handoff: the
    kernel's release store of the status word must become visible to the
    doorbell's acquire load -- and the record must parse complete -- with NO
    ``torch.cuda.synchronize()`` between launch and poll (the API sync every
    other producer test issues would mask a missing acquire edge)."""
    buffers = _gemm_buffers()
    signal = buffers["hit_signal"]
    mixed_gemm(**buffers)
    deadline = time.monotonic() + 60.0
    while not signal.doorbell():
        if time.monotonic() > deadline:
            pytest.fail("published hit never became visible to the doorbell")
    hit = signal.read_hit()
    assert hit is not None and hit.valid
    assert torch.equal(hit.codes, buffers["a_codes"].cpu())
    signal.reset_hit()
    torch.cuda.synchronize()  # drain the launch before the buffers die


def test_first_hit_wins_until_reset():
    buffers = _gemm_buffers()
    signal = buffers["hit_signal"]
    mixed_gemm(**buffers)
    torch.cuda.synchronize()
    first = signal.read_hit()
    assert first is not None and first.valid
    first_hash_a = first.commitment_hash_A

    # A second hit while one is pending is dropped: the latch is never
    # released by the kernel, and the record is immutable until reset.
    buffers["pow_key"] = torch.randint(0, 256, (32,), dtype=torch.uint8, device="cuda")
    mixed_gemm(**buffers)
    torch.cuda.synchronize()
    pending = signal.read_hit()
    assert pending is not None and pending.valid
    assert pending.commitment_hash_A == first_hash_a
    assert int(signal.lock.item()) == 1

    # reset re-arms: the next launch publishes the new key
    signal.reset_hit()
    assert int(signal.lock.item()) == 0
    mixed_gemm(**buffers)
    torch.cuda.synchronize()
    rearmed = signal.read_hit()
    assert rearmed is not None and rearmed.valid
    assert rearmed.commitment_hash_A == bytes(buffers["pow_key"].cpu().tolist())
    signal.reset_hit()


def test_atomic_rearm_is_safe_while_another_stream_is_still_running():
    """Reset may overlap loser/prolonged producers, but records stay whole."""
    signal = _make_signal(max_m=2048, max_k=512)
    slow = _gemm_buffers(m=2048, signal=signal)
    fast = _gemm_buffers(m=128, signal=signal)
    slow_stream = torch.cuda.Stream()
    fast_stream = torch.cuda.Stream()

    for _ in range(8):
        with torch.cuda.stream(slow_stream):
            mixed_gemm(**slow)
        with torch.cuda.stream(fast_stream):
            mixed_gemm(**fast)
            fast_done = torch.cuda.Event()
            fast_done.record()
        fast_done.synchronize()

        deadline = time.monotonic() + 60.0
        while not signal.doorbell():
            if time.monotonic() > deadline:
                pytest.fail("concurrent producer never published a hit")
        hit = signal.read_hit()
        assert hit is not None and hit.valid
        signal.reset_hit()

        slow_stream.synchronize()
        fast_stream.synchronize()
        # A producer that was still running at re-arm may legitimately claim
        # once more. It too must publish a complete record.
        if signal.doorbell():
            later = signal.read_hit()
            assert later is not None and later.valid
            signal.reset_hit()


def test_kernel_publishes_payloadless_when_planes_exceed_capacity():
    signal = _make_signal(max_m=1, max_k=512)  # too small for the (256, 512) planes
    buffers = _gemm_buffers(signal=signal)
    mixed_gemm(**buffers)
    torch.cuda.synchronize()
    hit = signal.read_hit()
    assert hit is not None and hit.valid
    assert hit.codes is None and hit.scales is None
    signal.reset_hit()


def test_partial_n_publishes_in_bounds_tile():
    """An edge-N problem still publishes an in-bounds lottery tile.

    With ``n=64, tile_n=256, ltile_cols=64`` the first epilogue warp maps
    lanes 1-3 out of bounds. The published winner must still be a real
    tile inside the problem, not an OOB lane that won the ballot.
    """
    m, n, k = 8, 64, 512
    config = MixedGemmConfig(tile_n=256, ltile_cols=64, cluster_m=1, cluster_n=1)
    signal = _make_signal(max_m=m, max_k=k)
    buffers = _gemm_buffers(m=m, n=n, k=k, signal=signal, config=config)
    mixed_gemm(**buffers)
    torch.cuda.synchronize()
    hit = signal.read_hit()
    assert hit is not None and hit.valid
    assert 0 <= hit.tile_row < m // config.ltile_rows
    assert 0 <= hit.tile_column < n // config.ltile_cols
    signal.reset_hit()


def test_hit_under_cuda_graph_replay():
    """The persistent signal survives CUDA-graph capture: replays with an
    in-place threshold refresh (no recapture) publish complete records with
    payload snapshots taken at publish time."""
    buffers = _gemm_buffers()
    signal = buffers["hit_signal"]
    buffers["threshold"].zero_()  # capture the never-win path
    mixed_gemm(**buffers)  # warm the compile cache outside capture
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        mixed_gemm(**buffers)

    graph.replay()
    torch.cuda.synchronize()
    assert signal.read_hit() is None  # silent no-hit replay

    # In-place operand refresh, then replay: the same captured kernel must
    # publish a hit with the refreshed operands' snapshot.
    buffers["threshold"].fill_(255)
    buffers["a_codes"].random_(-127, 128)
    graph.replay()
    torch.cuda.synchronize()
    hit = signal.read_hit()
    assert hit is not None and hit.valid
    assert torch.equal(hit.codes, buffers["a_codes"].cpu())

    # reset re-arms across replays of the same graph
    signal.reset_hit()
    graph.replay()
    torch.cuda.synchronize()
    rearmed = signal.read_hit()
    assert rearmed is not None and rearmed.valid
    signal.reset_hit()
