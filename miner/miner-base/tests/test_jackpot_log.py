"""The debug replay names the same check the cert-v4 verifier stops on."""

import struct
from pathlib import Path

import torch
from miner_base.jackpot_log import (
    _bf16_fma,
    _delta_r,
    _gauss_kernel,
    _gauss_row_constants,
    _lambda,
    _row_alpha,
    format_report,
    rejection_reason,
)
from miner_base.params import Device
from miner_base.prequant import PrequantMatrix


def _decode(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", (bits & 0xFFFF) << 16))[0]


def _code(value: float) -> int:
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    assert bits & 0xFFFF == 0, hex(bits)
    return bits >> 16


def _encoded(rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    operand = PrequantMatrix.encode(rows)
    return operand.int_values, operand.scales


def test_bf16_fma_matches_the_verifier():
    # (a, b, c, result) bit patterns from the verifier's bf16_fma reference vectors.
    cases = (
        (0x3F80, 0x4000, 0x3F80, 0x4040),  # 1*2 + 1 = 3
        (0x4049, 0x3FCC, 0xBF80, 0x4080),  # 3.14*1.59 - 1 ~= 4
        (0x3F00, 0x3F00, 0x3F00, 0x3F40),  # 0.5*0.5 + 0.5 = 0.75
        (0x4228, 0x3D23, 0x40B0, 0x40E5),
        (0xBF80, 0x4000, 0x4080, 0x4000),  # -1*2 + 4 = 2
        (0x4315, 0x3F80, 0x3B95, 0x4315),  # alpha*1 + tiny ~= alpha
    )
    for a, b, c, want in cases:
        assert _code(_bf16_fma(_decode(a), _decode(b), _decode(c))) == want


def test_scale_derivation_matches_the_verifier():
    # H100 at rank 32, and the B200 delta_r table.
    assert _code(_delta_r(1.0, 32)) == 0x40B5
    assert _code(_delta_r(0.5, 16)) == 0x4000
    assert _code(_delta_r(0.5, 128)) == 0x40B5
    # B200 derive_row_scales reference vectors: (l2, linf, rank) -> alpha.
    for l2, linf, rank, alpha in (
        (0x3F80, 0x3F80, 16, 0x4315),
        (0x3F80, 0x4080, 16, 0x4295),
        (0x2F80, 0x2F80, 16, 0x5315),
        (0x3F00, 0x4040, 64, 0x42B3),
    ):
        got = _row_alpha(_decode(l2), _decode(linf), 0.5, rank)
        assert _code(got) == alpha


def test_spike_rows_fail_liveness_on_the_spiked_side():
    # 64 unit spikes in k=2048 is twice the 1/64 dead-entry budget. On Blackwell
    # each spike sits above tau*DELTA*l2, so the side is rejected before the
    # noise floor.
    k = 2048
    flat = torch.ones(2, k, dtype=torch.bfloat16)
    spiked = torch.zeros(2, k, dtype=torch.bfloat16)
    spiked[:, :64] = 1
    assert rejection_reason(*_encoded(spiked), *_encoded(flat), Device.BLACKWELL, 32) == (
        "liveness (A)"
    )
    assert rejection_reason(*_encoded(flat), *_encoded(spiked), Device.BLACKWELL, 32) == (
        "liveness (B)"
    )


def test_liveness_is_counted_over_the_whole_tile():
    # 40 spikes in one row of 2048 is over 1/64 of that row and under 1/64 of a
    # two-row tile. The other row is live. The tile passes liveness.
    k = 2048
    rows = torch.ones(2, k, dtype=torch.bfloat16)
    rows[0].zero_()
    rows[0, :40] = 1
    flat = torch.ones(2, k, dtype=torch.bfloat16)
    assert rejection_reason(*_encoded(rows), *_encoded(flat), Device.BLACKWELL, 32) == (
        "unpredictable summands"
    )


def test_zero_and_flat_tiles_reach_the_census():
    for rows in (
        torch.zeros(2, 256, dtype=torch.bfloat16),
        torch.ones(2, 256, dtype=torch.bfloat16),
    ):
        for device in (Device.HOPPER, Device.BLACKWELL):
            assert rejection_reason(*_encoded(rows), *_encoded(rows), device, 32) == (
                "unpredictable summands"
            )


def test_one_spike_on_a_long_row_fails_the_noise_floor():
    # A single unit spike on k=65536 stays inside the liveness budget (one
    # entry) and drops sigma under the signal-RMS floor (delta * 4 on A,
    # delta * 8 on B) on both devices.
    k = 65536
    spike = torch.zeros(1, k, dtype=torch.bfloat16)
    spike[0, 0] = 1
    flat = torch.ones(1, k, dtype=torch.bfloat16)
    assert rejection_reason(*_encoded(spike), *_encoded(flat), Device.BLACKWELL, 32) == (
        "noise floor (A)"
    )
    assert rejection_reason(*_encoded(flat), *_encoded(spike), Device.BLACKWELL, 32) == (
        "noise floor (B)"
    )
    assert rejection_reason(*_encoded(spike), *_encoded(flat), Device.HOPPER, 32) == (
        "noise floor (A)"
    )


def test_blackwell_strip_constants_match_the_verifier():
    # gauss_row_constants and gauss_kernel vectors from zk-pow jackpot_policy.rs.
    assert _gauss_row_constants(16.0, Device.BLACKWELL) == (6720, 65536)
    assert _gauss_row_constants(4.0, Device.BLACKWELL) == (8192, 23488)
    assert _lambda(Device.BLACKWELL, 0x0380, 0, 0x2F80) == 12928
    assert _gauss_kernel(0) == 65536
    assert _gauss_kernel(1) == 65180
    assert _gauss_kernel(100) == 24674
    assert _gauss_kernel(150) == 8537
    assert _gauss_kernel(290) == 1
    assert sum(_gauss_kernel(d) for d in range(291)) == 5_617_275
    assert _gauss_kernel(291) == 0


def test_report_records_every_check_and_its_quantities():
    k = 2048
    flat = torch.ones(2, k, dtype=torch.bfloat16)
    spiked = torch.zeros(2, k, dtype=torch.bfloat16)
    spiked[:, :64] = 1
    report = format_report(*_encoded(spiked), *_encoded(flat), device=Device.BLACKWELL, rank=32)
    for name in (
        "liveness (A): fail",
        "liveness (B): pass",
        "noise floor (A):",
        "noise floor (B):",
        "unpredictable summands: unevaluated",
        "noise coverage (A):",
        "noise coverage (B):",
    ):
        assert name in report
    assert "dead=" in report and "limit=" in report
    assert "min_sigma=" in report and "floor=" in report
    assert "skippable=unknown" in report and "budget=" in report
    assert "min_product=" in report or "signal_rms=" in report
    example = Path(__file__).parent / "fixtures" / "jackpot_rejection_example.txt"
    assert report == example.read_text().rstrip("\n")
