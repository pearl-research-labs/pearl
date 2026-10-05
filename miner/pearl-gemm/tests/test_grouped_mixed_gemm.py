"""``grouped_mixed_gemm`` kernel correctness -- the MoE twin of ``test_mixed_gemm``.

Isolated FP8 operands laid out the way an MoE miner hands them over: A rows
permuted so expert ``w`` owns the contiguous, ascending-token block
``[m_indptr[w], m_indptr[w + 1])``, B stacked per expert. The FP8 operands
hold small integers, so every accumulator is an exact FP32 integer whatever
the MMA's summation order: the host reproduces it bit-exactly, and with it
the folded lottery messages and their keyed BLAKE3 digests. Gates: ``C''``
against a host FP8-upcast peel/unscale reference to tolerance, the first
winner (expert, expert-local tile) against a host enumeration of the
expert-local lottery (tiles restart at each expert's row 0, one jackpot key
per launch), the expert tag and expert-rows payload of a published hit, and
tiles past ``m_valid`` or partial tiles never publishing.
"""

import dataclasses

import numpy as np
import pytest
import torch
from blake3 import blake3
from miner_base.layout import lane_assignment

from pearl_gemm import GroupedMixedGemmConfig, HitSignal, HitSignalConfig, grouped_mixed_gemm
from pearl_gemm._utils._arch import Arch, arch_of
from pearl_gemm.grouped_fp8_gemm import WIDE_TILE_MIN_TOKENS_PER_GROUP
from pearl_gemm.protocol_constants import R
from tests.helpers.preprocess import default_config

_OUTPUT_RTOL = 5e-3
_OUTPUT_ATOL = 0.25
_LTILE_ROWS = 4
_EASY_THRESHOLD = (1 << 256) - 1  # every publishable tile wins
_FOLD_MUL = 0x9E3779B1

# CTA tile shapes every test runs under, per family. SM100: the 1-SM 128x128
# tile and the 2-CTA 128x256-per-CTA tile (256x256 per pair). SM90: the
# 128x128 and 256x128 tiles. SM120: the 128x128 and 128x256 tiles. Every
# lottery observable must be identical across them.
_FAMILY_TILE_SHAPES = {
    Arch.SM90: {
        "m128": {"mma_sm": 1, "tile_n": 128},
        "m256": {"mma_sm": 2, "tile_n": 128},
    },
    Arch.SM100: {
        "sm1_n128": {"mma_sm": 1, "tile_n": 128},
        "sm2_n256": {"mma_sm": 2, "tile_n": 256},
    },
    Arch.SM120: {
        "n128": {"tile_n": 128},
        "n256": {"tile_n": 256},
    },
}
_TILE_SHAPES = _FAMILY_TILE_SHAPES[arch_of()]
# The fused SM90 / SM120 kernels run only the static schedule, in expert order.
_FUSED = arch_of() in (Arch.SM90, Arch.SM120)
_FUSED_STATIC_ONLY = (
    "the fused SM90 / SM120 kernels run the static schedule in expert order "
    "(no dynamic / group_order)"
)
_SCHEDULERS = ["static"] if _FUSED else ["static", "dynamic"]


@pytest.fixture(autouse=True, params=list(_TILE_SHAPES), ids=list(_TILE_SHAPES))
def _tile_shape(request, monkeypatch):
    monkeypatch.setitem(globals(), "_TILE_KW", _TILE_SHAPES[request.param])


_TILE_KW: dict = next(iter(_TILE_SHAPES.values()))


def _config(config: GroupedMixedGemmConfig | None = None) -> GroupedMixedGemmConfig:
    """The test's config with the parametrized tile shape applied."""
    config = GroupedMixedGemmConfig() if config is None else config
    return dataclasses.replace(config, **_TILE_KW)


# (expert token counts, n, k). Counts mix whole CTA tiles, multi-CTA experts,
# a single lottery tile, an empty expert, and ragged counts.
_CASES = {
    "4x128-even": ([128, 128, 128, 128], 128, 512),
    "4x128-ragged": ([128, 260, 4, 120], 256, 512),
    "4x128-empty-expert": ([256, 0, 64, 12], 128, 1024),
    "4x128-wide": ([64, 320, 128], 384, 2048),
}
_PARTIAL_VALID_CASES = {
    # m_valid below the block row count: the trailing lottery tile of
    # experts 0 and 2 reaches past m_valid.
    "valid-tail": ([128, 128, 64, 8], [126, 128, 61, 8], 128, 512),
    # Every expert's second tile reaches past m_valid: half the kernel's tiles.
    "valid-half": ([8, 8, 8, 8], [5, 6, 7, 4], 128, 512),
}
# Unaligned expert blocks: boundaries that are not multiples of the lottery
# tile (rows 0..5 | 5..131 | 131..131 | 131..300), and a lattice of tiny
# experts whose CTA tiles all overhang into later experts' rows.
_UNALIGNED_CASES = {
    "unaligned-4": ([5, 126, 0, 169], 128, 512),
    "unaligned-tiny": ([3] * 40 + [1, 130, 2], 256, 512),
}


def _indptr(counts: list[int], aligned: bool = True) -> torch.Tensor:
    if aligned:
        counts = [(c + _LTILE_ROWS - 1) // _LTILE_ROWS * _LTILE_ROWS for c in counts]
    return torch.tensor([0, *torch.cumsum(torch.tensor(counts), 0).tolist()], dtype=torch.int32)


def _int_fp8(*shape: int) -> torch.Tensor:
    """Integer-valued e4m3 operands: every partial sum is an exact FP32 integer."""
    return torch.randint(-2, 3, shape, device="cuda").to(torch.float8_e4m3fn)


def _make_inputs(
    counts: list[int],
    n: int,
    k: int,
    seed: int = 0,
    valid: list[int] | None = None,
    *,
    aligned: bool = True,
    out_tail: int = 0,
):
    torch.manual_seed(seed)
    indptr = _indptr(counts, aligned)
    cum_m = int(indptr[-1])
    num_groups = len(counts)
    # Caller-owned eligible row counts: the block's rows unless ``valid`` is given.
    m_valid = (
        (indptr[1:] - indptr[:-1]).cuda()
        if valid is None
        else torch.tensor(valid, dtype=torch.int32, device="cuda")
    )
    return {
        "a_prime": _int_fp8(cum_m, k),
        "b_prime": _int_fp8(num_groups, n, k),
        "a_peel": torch.randn(cum_m, 2 * R, dtype=torch.bfloat16, device="cuda") * 0.1,
        "b_peel": torch.randn(num_groups, n, 2 * R, dtype=torch.bfloat16, device="cuda") * 0.1,
        "alpha_a": (torch.rand(cum_m, device="cuda") + 0.5).to(torch.bfloat16),
        "inv_alpha_b": torch.reciprocal(torch.rand(num_groups, n, device="cuda") + 0.5),
        "pow_key": torch.arange(32, dtype=torch.uint8, device="cuda"),
        "threshold": torch.zeros(32, dtype=torch.uint8, device="cuda"),
        "m_indptr": indptr.cuda(),
        "m_valid": m_valid,
        "hit_signal": HitSignal(HitSignalConfig(max_m=max(1, max(counts)), max_k=k)),
        "a_codes": torch.randint(-127, 128, (cum_m, k), dtype=torch.int8, device="cuda"),
        "a_scales": torch.rand(cum_m, k // 8, dtype=torch.bfloat16, device="cuda"),
        "commitment_hash_b": torch.zeros(32, dtype=torch.uint8, device="cuda"),
        # ``out`` is carved from a buffer with ``out_tail`` sentinel rows behind
        # the last expert: rows outside every expert must never be written.
        "out_tail": out_tail,
    }


_SENTINEL = 12345.0


def _threshold_tensor(threshold: int) -> torch.Tensor:
    return torch.frombuffer(bytearray(threshold.to_bytes(32, "little")), dtype=torch.uint8).cuda()


def _run(
    inputs,
    *,
    threshold: int | None = None,
    config: GroupedMixedGemmConfig | None = None,
    record_hits: bool = True,
    layer_id: int = 0,
    group_order: torch.Tensor | None = None,
    payload: bool = True,
):
    """Launch grouped_mixed_gemm; ``payload=False`` withholds the committed
    planes (payload-less records)."""
    config = _config(config)
    cum_m, _ = inputs["a_prime"].shape
    n = inputs["b_prime"].shape[1]
    out_big = torch.full(
        (cum_m + inputs["out_tail"], n), _SENTINEL, dtype=torch.bfloat16, device="cuda"
    )
    out = out_big[:cum_m]
    inputs["out_big"] = out_big
    hit_signal = inputs["hit_signal"]
    thr = inputs["threshold"] if threshold is None else _threshold_tensor(threshold)
    if hit_signal.doorbell():
        hit_signal.reset_hit()
    grouped_mixed_gemm(
        inputs["a_prime"],
        inputs["b_prime"],
        inputs["a_peel"],
        inputs["b_peel"],
        inputs["alpha_a"],
        inputs["inv_alpha_b"],
        inputs["pow_key"],
        thr,
        out,
        inputs["m_indptr"],
        hit_signal,
        inputs["a_codes"] if payload else None,
        inputs["a_scales"] if payload else None,
        inputs["commitment_hash_b"],
        inputs["m_valid"],
        config=config,
        layer_id=layer_id,
        record_hits=record_hits,
        group_order=group_order,
    )
    torch.cuda.synchronize()
    return out, hit_signal


def _groups(inputs):
    """Yield ``(g, start, end)`` of every expert's row block."""
    indptr = inputs["m_indptr"].cpu().tolist()
    for g in range(len(indptr) - 1):
        yield g, indptr[g], indptr[g + 1]


def _valid_counts(inputs) -> list[int]:
    return inputs["m_valid"].cpu().tolist()


def _reference_acc(inputs) -> torch.Tensor:
    """Every expert's ``A'[rows] @ B'[e].T``: exact integers, so bit-identical
    to the kernel's FP32 accumulator whatever its summation order."""
    a = inputs["a_prime"].double().cpu()
    b = inputs["b_prime"].double().cpu()
    acc = torch.zeros(a.shape[0], b.shape[1], dtype=torch.float64)
    for g, s, e in _groups(inputs):
        if e > s:
            acc[s:e] = a[s:e] @ b[g].T
    return acc.float()


def _output_reference(inputs, acc: torch.Tensor) -> torch.Tensor:
    """Host FP8-upcast accumulator + peel + unscale. Not bit-exact to the MMA peel."""
    out = acc.cuda().clone()
    for g, s, e in _groups(inputs):
        peel = inputs["a_peel"][s:e].float() @ inputs["b_peel"][g].float().T
        out[s:e] = (out[s:e] + peel) * inputs["inv_alpha_b"][g].reshape(1, -1)
    out *= torch.reciprocal(inputs["alpha_a"].float()).reshape(-1, 1)
    return out


def _fold_messages(tiles: torch.Tensor, protocol) -> list[bytes]:
    """The committed lottery fold of ``(T, ltile_rows, ltile_cols)`` FP32
    tiles: lane ``j`` folds its subtile's words in ``lane_assignment`` order
    as ``h = rotl13(h * 0x9E3779B1 + word)``; the 16 lanes are the 64-byte
    message."""
    lanes = np.asarray(lane_assignment(protocol.rows_pattern, protocol.cols_pattern))
    words = tiles.reshape(tiles.shape[0], -1).contiguous().view(torch.int32).numpy()
    words = words.astype(np.uint32).astype(np.uint64)[:, lanes]
    h = np.zeros(words.shape[:2], dtype=np.uint64)
    for step in range(words.shape[2]):
        h = (h * _FOLD_MUL + words[:, :, step]) & 0xFFFFFFFF
        h = ((h << 13) | (h >> 19)) & 0xFFFFFFFF
    return [row.astype("<u4").tobytes() for row in h]


def _digests(tiles: torch.Tensor, pow_key: bytes, protocol) -> list[int]:
    return [
        int.from_bytes(blake3(msg, key=pow_key).digest(), "little")
        for msg in _fold_messages(tiles, protocol)
    ]


def _lottery_hashes(inputs, acc, protocol, *, rows=None) -> dict[tuple[int, int, int], int]:
    """Mirror the MoE lottery enumeration: every lottery tile of every expert,
    the lattice restarting at the expert's first row and spanning only its
    real (``m_valid``) rows unless ``rows(g, s, e)`` overrides the count."""
    pow_key = bytes(inputs["pow_key"].cpu().numpy())
    ltile_cols = protocol.cols_pattern.tile_size
    col_tiles = acc.shape[1] // ltile_cols
    coords, tiles = [], []
    for (g, s, e), valid in zip(_groups(inputs), _valid_counts(inputs), strict=True):
        count = valid if rows is None else rows(g, s, e)
        for tile_row in range(count // _LTILE_ROWS):
            r0 = s + tile_row * _LTILE_ROWS
            for tile_col in range(col_tiles):
                coords.append((g, tile_row, tile_col))
                tiles.append(acc[r0 : r0 + _LTILE_ROWS, tile_col * ltile_cols :][:, :ltile_cols])
    if not tiles:
        return {}
    return dict(zip(coords, _digests(torch.stack(tiles), pow_key, protocol), strict=True))


def _winner(hit_signal: HitSignal) -> tuple[int, int, int] | None:
    hit = hit_signal.read_hit()
    if hit is None or not hit.valid:
        return None
    return hit.group_id, hit.tile_row, hit.tile_column


def _assert_output_close(got, ref):
    torch.testing.assert_close(got.float(), ref, rtol=_OUTPUT_RTOL, atol=_OUTPUT_ATOL)


@pytest.mark.parametrize("case", list(_CASES), ids=list(_CASES))
def test_output_matches_peel_unscale_reference(case):
    """``C''`` of every expert tracks the host peel/unscale algebra."""
    counts, n, k = _CASES[case]
    inputs = _make_inputs(counts, n, k, seed=sum(counts) + n + k, out_tail=128)
    out, _ = _run(inputs)
    _assert_output_close(out, _output_reference(inputs, _reference_acc(inputs)))
    assert torch.all(inputs["out_big"][out.shape[0] :] == _SENTINEL)


@pytest.mark.parametrize("case", ["4x128-ragged", "4x128-empty-expert", "4x128-wide"])
@pytest.mark.parametrize("scheduler", _SCHEDULERS)
def test_first_winner_matches_expert_local_lottery(case, scheduler):
    """A single-winner threshold pins the exact (expert, tile) of the host
    enumeration; one below it publishes nothing.

    With several admitted tiles the published one is whichever CTA claims the
    lock first, so only membership is asserted -- for either schedule."""
    counts, n, k = _CASES[case]
    inputs = _make_inputs(counts, n, k, seed=3)
    config = GroupedMixedGemmConfig(scheduler=scheduler)
    hashes = _lottery_hashes(inputs, _reference_acc(inputs), default_config(k))
    assert len(hashes) == sum(c // _LTILE_ROWS for c in counts) * (n // 128)
    ordered = sorted(hashes.items(), key=lambda item: item[1])

    median_hash = ordered[len(ordered) // 2][1]
    admitted = {coord for coord, value in hashes.items() if value <= median_hash}
    _, hit_signal = _run(inputs, threshold=median_hash, config=config)
    assert _winner(hit_signal) in admitted

    minimum_coordinate, minimum_hash = ordered[0]
    _, hit_signal = _run(inputs, threshold=minimum_hash, config=config)
    assert _winner(hit_signal) == minimum_coordinate
    _, hit_signal = _run(inputs, threshold=minimum_hash - 1, config=config)
    assert _winner(hit_signal) is None


@pytest.mark.skipif(_FUSED, reason=_FUSED_STATIC_ONLY)
def test_dynamic_schedule_matches_static_on_every_lottery_observable():
    """More tiles than SMs with skewed experts: the dynamic schedule (and the
    LPT visiting order) reproduce the static output and the single winner bit
    for bit, across repeated launches."""
    counts = [4 * ((5 * g) % 17 + 1) for g in range(40)]
    counts[2], counts[9], counts[21], counts[33] = 600, 0, 260, 0
    n, k = 2048, 512
    inputs = _make_inputs(counts, n, k, seed=53)
    out0, _ = _run(inputs, threshold=0)
    out0 = out0.clone()
    only_winner = min(_lottery_hashes(inputs, _reference_acc(inputs), default_config(k)).values())
    _, signal = _run(inputs, threshold=only_winner)
    expected_hit = _hit_fields(signal)
    order = torch.tensor(
        sorted(range(len(counts)), key=lambda g: -counts[g]), dtype=torch.int32, device="cuda"
    )
    dynamic = GroupedMixedGemmConfig(scheduler="dynamic")
    for config, group_order in ((dynamic, None), (dynamic, order), (None, order)):
        for _ in range(3):
            out, _ = _run(inputs, threshold=0, config=config, group_order=group_order)
            assert torch.equal(out, out0)
        _, signal = _run(inputs, threshold=only_winner, config=config, group_order=group_order)
        _assert_same_hit(expected_hit, _hit_fields(signal))


@pytest.mark.parametrize("case", list(_PARTIAL_VALID_CASES), ids=list(_PARTIAL_VALID_CASES))
def test_tiles_past_m_valid_never_publish(case):
    """Tiles reaching past ``m_valid`` are hashed for uniformity but are not
    protocol tiles: a threshold that admits only such a tile publishes nothing,
    and the first real winner is unaffected by them."""
    counts, valid, n, k = _PARTIAL_VALID_CASES[case]
    protocol = default_config(k)
    # Seed until the global minimum digest lands on an ineligible tile so the
    # exclusion is observable (the valid-half case hits within a few seeds).
    for seed in range(64):
        inputs = _make_inputs(counts, n, k, seed=41 + seed, valid=valid)
        acc = _reference_acc(inputs)
        real = _lottery_hashes(inputs, acc, protocol)
        everything = _lottery_hashes(inputs, acc, protocol, rows=lambda g, s, e: e - s)
        ineligible = {coord: h for coord, h in everything.items() if coord not in real}
        assert ineligible, "the case must contain tiles past m_valid"
        assert all(everything[c] == h for c, h in real.items())
        real_min = min(real.values())
        ineligible_min = min(ineligible.values())
        if ineligible_min < real_min or case == "valid-tail":
            break
    else:
        pytest.fail("no seed produced an ineligible-tile global minimum")

    if ineligible_min < real_min:
        _, hit_signal = _run(inputs, threshold=ineligible_min)
        assert _winner(hit_signal) is None
    _, hit_signal = _run(inputs, threshold=real_min)
    assert _winner(hit_signal) == min(real.items(), key=lambda item: item[1])[0]


def test_malformed_m_valid_is_clamped_in_kernel():
    """Overlarge valid-row counts clamp to the block size; negative counts
    publish no lottery hits."""
    counts, n, k = _CASES["4x128-ragged"]
    inputs = _make_inputs(counts, n, k, seed=53)
    valid_tiles = _lottery_hashes(inputs, _reference_acc(inputs), default_config(k))
    out, hit_signal = _run(inputs, threshold=_EASY_THRESHOLD)
    out = out.clone()
    assert _winner(hit_signal) in valid_tiles
    over = dict(inputs, m_valid=inputs["m_valid"] + 1000)
    out2, hit_signal2 = _run(over, threshold=_EASY_THRESHOLD)
    assert torch.equal(out2, out)
    # Publication elects the first successful claimant, not the minimum
    # digest, so two launches may name different winners: both must be real.
    assert _winner(hit_signal2) in valid_tiles
    # No expert has real rows: nothing may publish, whatever the threshold.
    none = dict(inputs, m_valid=torch.full_like(inputs["m_valid"], -1))
    _, hit_signal3 = _run(none, threshold=_EASY_THRESHOLD)
    assert _winner(hit_signal3) is None


def test_hit_record_names_the_expert_and_carries_its_rows():
    """A win tags the expert, reports its block row count, and snapshots the
    expert's rows of the permuted planes."""
    counts, n, k = _CASES["4x128-ragged"]
    inputs = _make_inputs(counts, n, k, seed=19)
    pow_key = bytes(inputs["pow_key"].cpu().numpy())
    hashes = _lottery_hashes(inputs, _reference_acc(inputs), default_config(k))
    (group, tile_row, tile_col), only_winner = min(hashes.items(), key=lambda item: item[1])

    _, signal = _run(inputs, threshold=only_winner, layer_id=7)
    hit = signal.read_hit()
    assert hit is not None and hit.valid
    s, e = inputs["m_indptr"][group].item(), inputs["m_indptr"][group + 1].item()
    assert (hit.group_id, hit.tile_row, hit.tile_column) == (group, tile_row, tile_col)
    assert (hit.m, hit.n, hit.k) == (e - s, n, k)
    assert (hit.ltile_rows, hit.ltile_cols) == (_LTILE_ROWS, 128)
    assert hit.layer_id == 7
    assert hit.codes is not None and hit.scales is not None
    assert torch.equal(hit.codes.reshape(e - s, k), inputs["a_codes"][s:e].cpu())
    assert torch.equal(
        hit.scales.view(torch.uint16).reshape(e - s, k // 8),
        inputs["a_scales"][s:e].cpu().view(torch.uint16),
    )
    assert hit.commitment_hash_A == pow_key
    assert hit.target == only_winner.to_bytes(32, "little")


def test_undersized_signal_publishes_payloadless():
    counts, n, k = _CASES["4x128-even"]
    inputs = _make_inputs(counts, n, k, seed=23)
    inputs["hit_signal"] = HitSignal(HitSignalConfig(max_m=1, max_k=64))
    _, signal = _run(inputs, threshold=_EASY_THRESHOLD)
    hit = signal.read_hit()
    assert hit is not None and hit.valid
    assert hit.codes is None and hit.scales is None
    assert hit.m == 128 and hit.k == k


def test_payload_of_an_expert_past_two_gib_is_snapshotted():
    """An expert whose rows start past 2^31 bytes of the plane snapshots the
    right rows (the payload offset arithmetic is 64-bit)."""
    k, n = 16384, 128
    counts = [131072, 4]  # expert 1's codes start at byte 2^31
    if torch.cuda.mem_get_info()[0] < 6 * 2**30:
        pytest.skip("needs ~5 GiB of free device memory for the 2 GiB planes")
    cum_m = sum(counts)
    inputs = _make_inputs([4, 4], n, k, seed=31)
    torch.manual_seed(31)
    a_prime = torch.zeros(cum_m, k, dtype=torch.float8_e4m3fn, device="cuda")
    a_codes = torch.empty(cum_m, k, dtype=torch.int8, device="cuda")
    a_scales = torch.empty(cum_m, k // 8, dtype=torch.bfloat16, device="cuda")
    a_prime[-4:] = _int_fp8(4, k)
    a_codes[-4:] = torch.randint(-127, 128, (4, k), dtype=torch.int8, device="cuda")
    a_scales[-4:] = torch.rand(4, k // 8, dtype=torch.bfloat16, device="cuda")
    inputs.update(
        a_prime=a_prime,
        a_codes=a_codes,
        a_scales=a_scales,
        a_peel=torch.zeros(cum_m, 2 * R, dtype=torch.bfloat16, device="cuda"),
        alpha_a=torch.ones(cum_m, dtype=torch.bfloat16, device="cuda"),
        m_indptr=_indptr(counts).cuda(),
        m_valid=torch.tensor([0, 4], dtype=torch.int32, device="cuda"),
        hit_signal=HitSignal(HitSignalConfig(max_m=4, max_k=k)),
    )
    _, signal = _run(inputs, threshold=_EASY_THRESHOLD)
    hit = signal.read_hit()
    assert hit is not None and hit.valid
    assert (hit.group_id, hit.m, hit.k) == (1, 4, k)
    assert hit.codes is not None and hit.scales is not None
    assert torch.equal(hit.codes.reshape(4, k), a_codes[-4:].cpu())
    assert torch.equal(
        hit.scales.view(torch.uint16).reshape(4, k // 8), a_scales[-4:].cpu().view(torch.uint16)
    )


def test_payload_of_exactly_two_gib_is_snapshotted_whole():
    """An expert whose code payload is 2^31 bytes -- inside the signal's
    ``uint32`` byte budget -- must be copied whole. A signed 32-bit byte count
    would be negative, copy nothing, and still publish a record declaring the
    full snapshot, so the host would read stale staging as the expert's rows."""
    k, n = 16384, 128
    m = 131072  # one expert: m * k == 2^31 code bytes, m * k / 4 scale bytes
    if torch.cuda.mem_get_info()[0] < 12 * 2**30:
        pytest.skip("needs ~8 GiB of free device memory for the 2 GiB planes and signal")
    if _TILE_KW["tile_n"] != 128:
        pytest.skip("one tile shape suffices for the host/kernel size arithmetic")
    inputs = _make_inputs([4], n, k, seed=37)
    # Row- and column-dependent byte patterns (no 2 GiB int32 temporaries).
    rows = torch.arange(m, device="cuda", dtype=torch.int32)
    a_codes = torch.empty(m, k, dtype=torch.int8, device="cuda")
    torch.add(
        (rows % 251).to(torch.int8).view(m, 1),
        (torch.arange(k, device="cuda", dtype=torch.int32) % 241).to(torch.int8).view(1, k),
        out=a_codes,
    )
    a_scales = torch.empty(m, k // 8, dtype=torch.bfloat16, device="cuda")
    torch.add(
        (rows % 251).to(torch.int16).view(m, 1),
        (torch.arange(k // 8, device="cuda", dtype=torch.int32) % 239).to(torch.int16).view(1, -1),
        out=a_scales.view(torch.int16),
    )
    inputs.update(
        a_prime=torch.zeros(m, k, dtype=torch.float8_e4m3fn, device="cuda"),
        a_codes=a_codes,
        a_scales=a_scales,
        a_peel=torch.zeros(m, 2 * R, dtype=torch.bfloat16, device="cuda"),
        alpha_a=torch.ones(m, dtype=torch.bfloat16, device="cuda"),
        m_indptr=_indptr([m]).cuda(),
        m_valid=torch.tensor([4], dtype=torch.int32, device="cuda"),
        hit_signal=HitSignal(HitSignalConfig(max_m=m, max_k=k)),
    )
    _, signal = _run(inputs, threshold=_EASY_THRESHOLD)
    hit = signal.read_hit()
    assert hit is not None and hit.valid
    assert (hit.group_id, hit.m, hit.k) == (0, m, k)
    assert hit.codes is not None and hit.scales is not None
    assert hit.codes.shape == (m, k) and hit.scales.shape == (m, k // 8)
    # Spot-check rows across the whole plane, the last one included: a
    # truncated (or skipped) copy leaves them as stale staging.
    for row in (0, 1, m // 2, m - 2, m - 1):
        assert torch.equal(hit.codes[row], a_codes[row].cpu()), f"codes row {row}"
        assert torch.equal(
            hit.scales[row].view(torch.int16), a_scales[row].cpu().view(torch.int16)
        ), f"scales row {row}"


@pytest.mark.parametrize("n", [128, 384])
def test_partial_column_tiles_never_publish(n):
    """A 256-wide lottery tile overhanging the expert's columns is not a
    protocol tile: ``n=128`` publishes nothing, ``n=384`` only column tile 0."""
    if _TILE_KW["tile_n"] != 256:
        pytest.skip("the 4x256 lottery needs the 256-wide CTA tile")
    inputs = _make_inputs([128, 8, 64, 260], n, 512, seed=29)
    config = GroupedMixedGemmConfig(ltile_cols=256, **_TILE_KW)
    _, signal = _run(inputs, threshold=_EASY_THRESHOLD, config=config)
    winner = _winner(signal)
    if n == 128:
        assert winner is None
    else:
        assert winner is not None and winner[2] == 0


def test_single_tile_launches_are_stable_under_repetition():
    """Repeated single-tile launches preserve the output and the hit
    coordinates/dimensions."""
    inputs = _make_inputs([4], 128, 512, seed=31)
    reference = None
    for _ in range(200):
        out, signal = _run(inputs, threshold=_EASY_THRESHOLD)
        hit = signal.read_hit()
        assert hit is not None and hit.valid and hit.group_id == 0
        snapshot = (out.clone(), (hit.tile_row, hit.tile_column, hit.m, hit.n))
        if reference is None:
            reference = snapshot
            continue
        assert torch.equal(snapshot[0], reference[0])
        assert snapshot[1] == reference[1]


def _hit_fields(hit_signal: HitSignal) -> dict:
    hit = hit_signal.read_hit()
    assert hit is not None and hit.valid
    return {
        name: value.clone() if isinstance(value, torch.Tensor) else value
        for name, value in vars(hit).items()
    }


def _assert_same_hit(expected: dict, got: dict) -> None:
    assert set(expected) == set(got)
    for name, value in expected.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, got[name]), f"hit field {name} differs"
        else:
            assert value == got[name], f"hit field {name} differs: {value!r} != {got[name]!r}"


def test_payload_less_variant_publishes_the_same_record_without_planes():
    """Withholding ``a_codes`` / ``a_scales`` selects the variant that carries
    no copy: same winner, same expert tag and coordinates, zero payload bytes."""
    counts, n, k = _CASES["4x128-ragged"]
    inputs = _make_inputs(counts, n, k, seed=17)
    only_winner = min(_lottery_hashes(inputs, _reference_acc(inputs), default_config(k)).values())
    _, signal = _run(inputs, threshold=only_winner)
    with_payload = _hit_fields(signal)
    assert with_payload["codes"] is not None
    _, signal = _run(inputs, threshold=only_winner, payload=False)
    without = _hit_fields(signal)
    assert without["codes"] is None and without["scales"] is None
    for name in ("group_id", "tile_row", "tile_column", "m", "n", "k", "commitment_hash_A"):
        assert with_payload[name] == without[name], name
    with pytest.raises(ValueError, match="both present"):
        grouped_mixed_gemm(
            inputs["a_prime"],
            inputs["b_prime"],
            inputs["a_peel"],
            inputs["b_peel"],
            inputs["alpha_a"],
            inputs["inv_alpha_b"],
            inputs["pow_key"],
            inputs["threshold"],
            torch.empty(inputs["a_prime"].shape[0], n, dtype=torch.bfloat16, device="cuda"),
            inputs["m_indptr"],
            inputs["hit_signal"],
            inputs["a_codes"],
            None,
            inputs["commitment_hash_b"],
            inputs["m_valid"],
            config=_config(),
        )


@pytest.mark.parametrize("record_hits", [True, False], ids=["publish", "no_publish"])
def test_record_hits_gates_publication(record_hits):
    """``record_hits`` still hashes; only the flag gates publication."""
    counts, n, k = _CASES["4x128-even"]
    inputs = _make_inputs(counts, n, k, seed=29)
    _, signal = _run(inputs, threshold=_EASY_THRESHOLD, record_hits=record_hits)
    assert signal.doorbell() is record_hits


@pytest.mark.parametrize("scheduler", _SCHEDULERS)
def test_consistency(scheduler):
    """Repeat launches over the same buffers are bit-identical."""
    counts, n, k = _CASES["4x128-ragged"]
    inputs = _make_inputs(counts, n, k, seed=5)
    config = GroupedMixedGemmConfig(scheduler=scheduler)
    out0, _ = _run(inputs, threshold=1 << 255, config=config)
    out0 = out0.clone()
    for _ in range(4):
        out, _ = _run(inputs, threshold=1 << 255, config=config)
        assert torch.equal(out, out0)


@pytest.mark.parametrize("scheduler", _SCHEDULERS)
def test_is_cuda_graph_capturable(scheduler):
    """Capture and replay reproduce the eager ``C''`` exactly and publish."""
    # 40 experts x 128 rows x 16 n-tiles: more tiles than SMs, so the dynamic
    # fetch and the in-kernel counter reset run on every replay.
    counts, n, k = [128] * 40, 2048, 512
    inputs = _make_inputs(counts, n, k, seed=37)
    config = _config(GroupedMixedGemmConfig(scheduler=scheduler))
    out_eager, _ = _run(inputs, threshold=0, config=config)
    out_eager = out_eager.clone()
    out = torch.zeros_like(out_eager)
    threshold = torch.zeros(32, dtype=torch.uint8, device="cuda")
    signal = inputs["hit_signal"]

    def launch():
        grouped_mixed_gemm(
            inputs["a_prime"],
            inputs["b_prime"],
            inputs["a_peel"],
            inputs["b_peel"],
            inputs["alpha_a"],
            inputs["inv_alpha_b"],
            inputs["pow_key"],
            threshold,
            out,
            inputs["m_indptr"],
            signal,
            inputs["a_codes"],
            inputs["a_scales"],
            inputs["commitment_hash_b"],
            inputs["m_valid"],
            config=config,
        )

    launch()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    for _ in range(3):
        out.zero_()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, out_eager)
    assert not signal.doorbell()
    threshold.fill_(0xFF)  # in-place: the captured launch now always wins
    graph.replay()
    torch.cuda.synchronize()
    hit = signal.read_hit()
    assert hit is not None and hit.valid and 0 <= hit.group_id < len(counts)
    signal.reset_hit()


def test_auto_config_selects_the_tile_by_average_tokens_per_group():
    """SM100 widens to the 2-CTA 256-wide tile from ``WIDE_TILE_MIN_TOKENS_PER_GROUP``
    tokens per expert; SM90 widens to the 256x128 tile by the same rule; SM120
    always takes the 128x128 tile."""
    e = 32
    auto = GroupedMixedGemmConfig.auto
    below, at = WIDE_TILE_MIN_TOKENS_PER_GROUP * e - 1, WIDE_TILE_MIN_TOKENS_PER_GROUP * e
    assert auto(below, e, arch=Arch.SM100) == GroupedMixedGemmConfig()
    assert auto(at, e, arch=Arch.SM100) == GroupedMixedGemmConfig(mma_sm=2, tile_n=256)
    assert auto(at, e, arch=Arch.SM100, scheduler="dynamic") == GroupedMixedGemmConfig(
        mma_sm=2, tile_n=256, scheduler="dynamic"
    )
    # Explicit tile knobs override the rule instead of clashing with it.
    assert auto(at, e, arch=Arch.SM100, mma_sm=1, tile_n=128) == GroupedMixedGemmConfig()
    assert auto(below, e, arch=Arch.SM90) == GroupedMixedGemmConfig()
    assert auto(at, e, arch=Arch.SM90) == GroupedMixedGemmConfig(mma_sm=2)
    assert auto(at, e, arch=Arch.SM120) == GroupedMixedGemmConfig()
    with pytest.raises(ValueError, match="tile_n"):
        GroupedMixedGemmConfig(tile_n=192)
    with pytest.raises(ValueError, match="mma_sm"):
        GroupedMixedGemmConfig(mma_sm=True)
    with pytest.raises(ValueError, match="tile_n"):
        GroupedMixedGemmConfig(tile_n=128.0)
    with pytest.raises(ValueError, match="ltile_cols must divide"):
        GroupedMixedGemmConfig(tile_n=128, ltile_cols=256)


def test_default_config_is_auto_and_matches_the_explicit_tile():
    """``config=None`` resolves inside the host (``GroupedMixedGemmConfig.auto``)
    and matches the explicit launch of the tile it picks."""
    counts, n, k = [256, 256, 256, 256], 256, 512
    inputs = _make_inputs(counts, n, k, seed=59)
    cum_m = inputs["a_prime"].shape[0]
    # Bypass the tile-shape fixture: ``None`` must resolve inside the host.
    outs = []
    for config in (None, GroupedMixedGemmConfig.auto(cum_m, len(counts))):
        out = torch.empty(cum_m, n, dtype=torch.bfloat16, device="cuda")
        grouped_mixed_gemm(
            inputs["a_prime"],
            inputs["b_prime"],
            inputs["a_peel"],
            inputs["b_peel"],
            inputs["alpha_a"],
            inputs["inv_alpha_b"],
            inputs["pow_key"],
            inputs["threshold"],
            out,
            inputs["m_indptr"],
            inputs["hit_signal"],
            inputs["a_codes"],
            inputs["a_scales"],
            inputs["commitment_hash_b"],
            inputs["m_valid"],
            config=config,
        )
        outs.append(out)
    torch.cuda.synchronize()
    assert torch.equal(outs[0], outs[1])
    _assert_output_close(outs[0], _output_reference(inputs, _reference_acc(inputs)))


def test_missing_output_operand_raises():
    """Output, peel and unscale operands are always required."""
    counts, n, k = _CASES["4x128-even"]
    inputs = _make_inputs(counts, n, k, seed=43)
    with pytest.raises(TypeError, match="b_peel must be a torch.Tensor"):
        grouped_mixed_gemm(
            inputs["a_prime"],
            inputs["b_prime"],
            inputs["a_peel"],
            None,
            inputs["alpha_a"],
            inputs["inv_alpha_b"],
            inputs["pow_key"],
            inputs["threshold"],
            torch.empty(inputs["a_prime"].shape[0], n, dtype=torch.bfloat16, device="cuda"),
            inputs["m_indptr"],
            inputs["hit_signal"],
            inputs["a_codes"],
            inputs["a_scales"],
            inputs["commitment_hash_b"],
            inputs["m_valid"],
            config=_config(),
        )


def test_fused_kernels_reject_unsupported_tiles_and_schedules():
    if not _FUSED:
        pytest.skip("SM100 runs the 2-CTA pair and every schedule")
    inputs = _make_inputs([4, 4], 128, 512, seed=67)
    order = torch.tensor([1, 0], dtype=torch.int32, device="cuda")
    # A knob the tile-shape fixture does not override.
    unsupported_tile = (
        GroupedMixedGemmConfig(mma_sm=2)
        if arch_of() is Arch.SM120
        else GroupedMixedGemmConfig(ltile_rows=16, ltile_cols=32)
    )
    for config, group_order in (
        (unsupported_tile, None),
        (GroupedMixedGemmConfig(scheduler="dynamic"), None),
        (None, order),
    ):
        with pytest.raises(ValueError, match="static schedule in expert order"):
            _run(inputs, config=config, group_order=group_order)


def test_malformed_indptr_and_group_order_stay_in_bounds():
    """Out-of-range and decreasing routing tables launch and write only ``out``'s rows."""
    counts, n, k = _CASES["4x128-ragged"]
    inputs = _make_inputs(counts, n, k, seed=71, out_tail=128)
    cum_m = inputs["a_prime"].shape[0]
    inputs["m_indptr"] = torch.tensor(
        [0, cum_m + 999, -5, cum_m, 3], dtype=torch.int32, device="cuda"
    )
    order = torch.tensor([7, -1, 2, 2], dtype=torch.int32, device="cuda")
    launches = [(None, None)]
    if not _FUSED:
        launches += [(None, order), (GroupedMixedGemmConfig(scheduler="dynamic"), order)]
    for config, group_order in launches:
        _run(inputs, config=config, group_order=group_order)
        assert not torch.all(inputs["out_big"][:cum_m] == _SENTINEL)
        assert torch.all(inputs["out_big"][cum_m:] == _SENTINEL)


@pytest.mark.parametrize("extent", ["groups", "n", "k", "cum_m"])
def test_zero_extents_are_rejected_before_launch(extent):
    """The mining launch has no empty-problem policy: zero experts, width,
    depth or rows are refused on the host (the plain grouped GEMM alone
    admits ``cum_m = 0``, checked in its own suite)."""
    inputs = _make_inputs([4], 128, 64, seed=61)
    zero = {
        "groups": {"b_prime": inputs["b_prime"][:0]},
        "n": {"b_prime": inputs["b_prime"][:, :0]},
        "k": {"a_prime": inputs["a_prime"][:, :0], "b_prime": inputs["b_prime"][:, :, :0]},
        "cum_m": {"a_prime": inputs["a_prime"][:0]},
    }[extent]
    with pytest.raises(ValueError, match="positive"):
        grouped_mixed_gemm(
            zero.get("a_prime", inputs["a_prime"]),
            zero.get("b_prime", inputs["b_prime"]),
            *([None] * 13),
        )


def test_rejects_groups_past_the_ragged_descriptor_rows():
    """Same bound as ``grouped_fp8_gemm``: a launch whose rows exceed the
    ragged TMA views' extent would hash zero-filled rows (and could publish
    them); it is refused on the host-known shape, boundary admitted."""
    from pearl_gemm.grouped_fp8_gemm import MAX_GROUPED_ROWS
    from pearl_gemm.grouped_fp8_gemm._host import _problem_dims

    inputs = _make_inputs([4], 128, 64, seed=59)
    huge = torch.empty(1, 64, dtype=torch.float8_e4m3fn, device="cuda")
    a_prime = huge.expand(MAX_GROUPED_ROWS + 4, 64)  # the shape without the 64 GiB
    with pytest.raises(ValueError, match="exceeds the .* rows the ragged TMA views address"):
        grouped_mixed_gemm(a_prime, inputs["b_prime"], *([None] * 13))
    assert _problem_dims(huge.expand(MAX_GROUPED_ROWS, 64), inputs["b_prime"])[0] == (
        MAX_GROUPED_ROWS
    )


def test_rejects_tile_schedules_past_the_int32_scheduler():
    """A launch whose one group of 2**22 rows against 2**23 columns gives the
    128x128 tile a 2**31 tile product is refused on the host-known shapes for
    the selected config, before any operand or device read."""
    from pearl_gemm.grouped_fp8_gemm._host import _require_scheduler_geometry

    m, n = 2**22, 2**23
    one = torch.empty(1, 1, 64, dtype=torch.float8_e4m3fn, device="cuda")
    with pytest.raises(ValueError, match="signed 32-bit scheduler"):
        grouped_mixed_gemm(
            one[0].expand(m, 64),
            one.expand(1, n, 64),
            *([None] * 13),
            config=GroupedMixedGemmConfig(mma_sm=1, tile_n=128),
        )
    # The mining config derives the same cluster tile as the plain GEMM's.
    tile = GroupedMixedGemmConfig(mma_sm=2, tile_n=256).cluster_tile_shape_mn
    assert tile == (256, 256)
    _require_scheduler_geometry(m, 1, n, tile)
    with pytest.raises(ValueError, match="no headroom"):
        _require_scheduler_geometry(m, 1, 2 * n, tile)


def test_written_signal_buffers_cannot_alias_an_input():
    """The kernel writes the signal's record and claim latch as well as its
    payload regions: an input carved from any of them (whatever its dtype or
    offset) is rejected before launch, while read/read sharing stays legal."""
    inputs = _make_inputs([4], 128, 512, seed=53)
    signal = inputs["hit_signal"]
    record_as_bytes = signal.record_device_view.view(torch.uint8)
    aliases = [
        ("hit_signal.record", "commitment_hash_b", record_as_bytes[:32]),  # other dtype
        ("hit_signal.record", "pow_key", record_as_bytes[16:48]),  # offset, partial
        ("hit_signal.lock", "m_valid", signal.lock),  # one group: the latch's exact shape
        ("hit_signal.codes_payload", "threshold", signal.codes_payload.view(torch.uint8)[:32]),
    ]
    for written, name, alias in aliases:
        with pytest.raises(ValueError, match=f"{written} is written and must not overlap {name}"):
            _run({**inputs, name: alias}, threshold=None if name == "threshold" else 0)
    # Two reads of one buffer are harmless.
    _run({**inputs, "commitment_hash_b": inputs["pow_key"]}, threshold=_EASY_THRESHOLD)


@pytest.mark.parametrize("case", list(_UNALIGNED_CASES), ids=list(_UNALIGNED_CASES))
def test_unaligned_experts_output_and_untouched_rows(case):
    """Unaligned expert blocks: the output matches on every row, and the
    sentinel rows behind the last expert (rows outside every expert) are
    never written."""
    counts, n, k = _UNALIGNED_CASES[case]
    inputs = _make_inputs(counts, n, k, seed=sum(counts) + n, aligned=False, out_tail=128)
    out, _ = _run(inputs)
    _assert_output_close(out, _output_reference(inputs, _reference_acc(inputs)))
    cum_m = inputs["a_prime"].shape[0]
    assert torch.all(inputs["out_big"][cum_m:] == _SENTINEL), (
        "rows behind the last expert were written"
    )


def _straddling_tile_hashes(inputs, acc, protocol) -> dict:
    """Hashes of every expert's trailing partial lottery tile.

    The kernel folds such a tile from the expert's last ``count % ltile_rows``
    accumulator rows followed by rows that TMA zero-filled (they lie past the
    expert's end); these tiles are not protocol tiles and must never publish.
    """
    pow_key = bytes(inputs["pow_key"].cpu().numpy())
    ltile_cols = protocol.cols_pattern.tile_size
    coords, tiles = [], []
    for g, s, e in _groups(inputs):
        count = e - s
        if count % _LTILE_ROWS == 0:
            continue
        tile_row = count // _LTILE_ROWS
        rows = torch.zeros(_LTILE_ROWS, acc.shape[1], dtype=acc.dtype)
        rows[: count % _LTILE_ROWS] = acc[s + tile_row * _LTILE_ROWS : e]
        for tile_col in range(acc.shape[1] // ltile_cols):
            coords.append((g, tile_row, tile_col))
            tiles.append(rows[:, tile_col * ltile_cols : (tile_col + 1) * ltile_cols])
    return dict(zip(coords, _digests(torch.stack(tiles), pow_key, protocol), strict=True))


def test_unaligned_experts_partial_tiles_never_publish():
    """With unaligned blocks the trailing partial tile of an expert reaches into
    the next expert's rows; it is folded (over zero-filled rows) but is not a
    protocol tile: a threshold admitting only such tiles publishes nothing and
    the first real winner is the host enumeration's."""
    counts, n, k = _UNALIGNED_CASES["unaligned-4"]
    protocol = default_config(k)
    for seed in range(64):
        inputs = _make_inputs(counts, n, k, seed=53 + seed, aligned=False)
        acc = _reference_acc(inputs)
        real = _lottery_hashes(inputs, acc, protocol)
        straddling = _straddling_tile_hashes(inputs, acc, protocol)
        assert straddling and not set(straddling) & set(real)
        real_min = min(real.values())
        straddle_min = min(straddling.values())
        if straddle_min < real_min:
            break
    else:
        pytest.fail("no seed produced a partial-tile global minimum")

    _, hit_signal = _run(inputs, threshold=straddle_min)
    assert _winner(hit_signal) is None
    _, hit_signal = _run(inputs, threshold=real_min)
    (group, tile_row, tile_col), _ = min(real.items(), key=lambda item: item[1])
    assert _winner(hit_signal) == (group, tile_row, tile_col)
    hit = hit_signal.read_hit()
    assert hit.m == counts[group]
    s, e = inputs["m_indptr"][group].item(), inputs["m_indptr"][group + 1].item()
    assert torch.equal(hit.codes.reshape(e - s, k), inputs["a_codes"][s:e].cpu())
