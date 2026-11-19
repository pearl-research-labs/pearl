"""CUDA-free smoke and determinism tests for CPU B preparation via pearl_mining kernels."""

import pytest
import torch
from miner_base.protocol import Device
from vllm_miner import b_prepare_cpu as bp

_KEY = bytes(range(32))


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.uint8)


def _layer(n: int, k: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    codes = torch.randint(-127, 128, (n, k), generator=gen, dtype=torch.int8)
    scales = (torch.rand(n, k // 8, generator=gen) * 2.0**-7).to(torch.bfloat16)
    # Small-rms rows take the other strip gain; a zero row exercises the norm floor.
    scales[: n // 4] *= 2.0**-12
    codes[n // 4] = 0
    return codes, scales


@pytest.mark.parametrize("label", [bp.LABEL_F1, bp.LABEL_F2, bp.LABEL_E2])
def test_native_lines_shape_and_determinism(label):
    count = 300
    prev_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        single = bp._lines(_KEY, label, count)
        torch.set_num_threads(4)
        multi = bp._lines(_KEY, label, count)
    finally:
        torch.set_num_threads(prev_threads)

    assert single.shape == (count, bp.RANK)
    assert single.dtype == torch.float8_e4m3fn
    assert torch.equal(_bits(single), _bits(multi))
    assert not ((_bits(single) & 0x7F) == 0x7F).any()


@pytest.mark.parametrize("device", [Device.BLACKWELL, Device.HOPPER])
@pytest.mark.parametrize(("n", "k"), [(37, 512), (64, 768)])
def test_native_rows_shape_peel_and_determinism(device, n, k):
    arithmetic = bp.ARITHMETIC[device]
    codes, scales = _layer(n, k, seed=n * k)
    f1 = bp._lines(_KEY, bp.LABEL_F1, k).t().contiguous()
    f2 = bp._lines(_KEY, bp.LABEL_F2, k).t().contiguous()
    e2 = bp._lines(_KEY, bp.LABEL_E2, n)
    row_scales = bp._layer_row_scales(codes, scales, arithmetic.delta, None)
    assert row_scales is not None

    prev_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        b_prime, peel = bp._native_rows(codes, scales, row_scales, e2, f1, f2, arithmetic)
        torch.set_num_threads(4)
        b_prime_mt, peel_mt = bp._native_rows(codes, scales, row_scales, e2, f1, f2, arithmetic)
    finally:
        torch.set_num_threads(prev_threads)

    assert b_prime.shape == (n, k)
    assert b_prime.dtype == torch.float8_e4m3fn
    assert peel.shape == (n, 2 * bp.RANK)
    assert peel.dtype == torch.bfloat16
    assert torch.equal(_bits(b_prime), _bits(b_prime_mt))
    assert torch.equal(_bits(peel), _bits(peel_mt))
    expected_right = -(row_scales.beta_b.unsqueeze(1) * e2.to(torch.bfloat16))
    assert torch.equal(peel[:, bp.RANK :], expected_right)
