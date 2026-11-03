"""On-GPU exercise of the PoW hit-signal record + reset path on the A100 (``sm_80``).

The signal's device touchpoint is the re-arm kernel (``pow/_reset.py``): a tiny
all-atomics kernel (acquire load, system fence, GPU-scope ``atomic_cas``, all
sm_70+) that is arch-neutral, so one source serves every family -- including
SM80 (A100), the FP16 scheme's native target. This test forges published
records with plain host stores into the pinned record (the same oracle
``tests/test_hit_signal.py`` uses, which cannot allocate a signal on an A100
until ``_reset.py`` admits ``Arch.SM80``) and asserts the record/reset/latch
protocol holds on real GA100 silicon: the expected record layout parses, the
first-wins latch stays closed until reset, and the device re-arm kernel
clears the magic + doorbell and atomically reopens the latch. Gated to
``sm_80`` hardware; a dead end on any other family.
"""

import pytest
import torch

from pearl_gemm import (
    HitRecordLayout,
    HitSignal,
    HitSignalConfig,
)
from pearl_gemm._utils._arch import Arch, arch_of
from pearl_gemm.pow import HIT_RECORD_MAGIC_WORDS

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or arch_of() is not Arch.SM80,
    reason="the sm_80 PoW hit-signal path targets A100 (GA100) hardware",
)

STATUS_IDLE = 0
STATUS_PUBLISHED = 1
DEFAULT_MAX_M = 8
DEFAULT_MAX_K = 512
FORGE_M = 4
FORGE_N = 128
FORGE_K = 64
FORGE_LAYER_ID = 7


def _make_signal(max_m=DEFAULT_MAX_M, max_k=DEFAULT_MAX_K) -> HitSignal:
    return HitSignal(HitSignalConfig(max_m=max_m, max_k=max_k))


def _forge_hit(
    signal: HitSignal,
    m=FORGE_M,
    n=FORGE_N,
    k=FORGE_K,
    with_payload=True,
    group_id=0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Simulate a kernel publish with plain host stores: payloads, record
    fields, magic, then the status doorbell LAST (same oracle as
    ``tests/test_hit_signal.py``)."""
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
    record[HitRecordLayout.GROUP_ID] = group_id
    for word in range(8):
        record[HitRecordLayout.TARGET + word] = word + 1
        record[HitRecordLayout.HASH_A + word] = word + 100
        record[HitRecordLayout.HASH_B + word] = word + 200
    record[HitRecordLayout.MAGIC] = HIT_RECORD_MAGIC_WORDS[0]
    record[HitRecordLayout.MAGIC + 1] = HIT_RECORD_MAGIC_WORDS[1]
    record[HitRecordLayout.STATUS] = STATUS_PUBLISHED
    return codes, scales


def test_signal_allocates_on_sm80():
    """``HitSignal.__init__`` compiles the re-arm kernel via ``prepare_hit_reset``
    (the sole arch gate for the whole pow stage): it must now admit the A100."""
    signal = _make_signal()
    assert signal.record_device_view.data_ptr() == signal.record.data_ptr()
    assert int(signal.lock.item()) == 0
    assert not signal.doorbell()


def test_record_parses_with_expected_layout_on_sm80():
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
    signal.reset_hit()


def test_device_reset_rearms_latch_on_sm80():
    """The device re-arm kernel (``cute.arch.atomic_cas``) runs on the A100:
    it clears the magic + doorbell, fences, and atomically reopens the latch."""
    signal = _make_signal()
    _forge_hit(signal)
    signal.lock.fill_(1)  # a producer claimed the first-wins latch

    assert signal.doorbell()
    signal.reset_hit()

    # The device kernel cleared the host-visible markers and the latch.
    assert not signal.doorbell()
    assert signal.read_hit() is None
    assert int(signal.lock.item()) == 0

    # Re-ringing the doorbell alone parses invalid: reset zeroed the magic.
    signal.record[HitRecordLayout.STATUS] = STATUS_PUBLISHED
    hit = signal.read_hit()
    assert hit is not None and not hit.valid
    signal.reset_hit()


def test_reset_does_not_reopen_an_unpublished_producer_claim_on_sm80():
    """Device defense in depth: lock=1/status=0 is an in-flight producer
    claim, never consumer ownership, so the re-arm CAS must not fire."""
    signal = _make_signal()
    signal.lock.fill_(1)
    signal.record[HitRecordLayout.STATUS] = STATUS_IDLE

    signal.reset_hit()
    assert int(signal.lock.item()) == 1

    # The private launch carries the same device-side guard.
    signal._reset_hit_locked()
    assert int(signal.lock.item()) == 1

    _forge_hit(signal)
    signal._reset_hit_locked()
    assert int(signal.lock.item()) == 0
    assert not signal.doorbell()


def test_first_wins_latch_holds_until_device_reset_on_sm80():
    """A second forged publish while one is pending is dropped; only the
    device re-arm reopens the latch for the next record."""
    signal = _make_signal()
    first_codes, _ = _forge_hit(signal)
    signal.lock.fill_(1)

    first = signal.read_hit()
    assert first is not None and first.valid
    first_hash_a = first.commitment_hash_A
    assert torch.equal(first.codes, first_codes)

    signal.reset_hit()
    assert int(signal.lock.item()) == 0

    second_codes, _ = _forge_hit(signal, m=FORGE_M, k=FORGE_K)
    rearmed = signal.read_hit()
    assert rearmed is not None and rearmed.valid
    assert rearmed.commitment_hash_A == first_hash_a  # same forged header fields
    assert torch.equal(rearmed.codes, second_codes)
    signal.reset_hit()


def test_repeat_reset_determinism_on_sm80():
    """The all-atomics re-arm kernel is idempotent across many launches."""
    signal = _make_signal()
    for _ in range(8):
        _forge_hit(signal)
        signal.lock.fill_(1)
        assert signal.doorbell()
        signal.reset_hit()
        assert not signal.doorbell()
        assert int(signal.lock.item()) == 0
