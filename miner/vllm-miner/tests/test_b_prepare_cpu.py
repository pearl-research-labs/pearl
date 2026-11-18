"""CPU B preparation's native kernels against its torch reference chain (no CUDA)."""

import pytest
import torch
from miner_base.params import Device
from vllm_miner import b_prepare_cpu as bp

_KEY = bytes(range(32))


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.uint8)


def _layer(n: int, k: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    codes = torch.randint(-127, 128, (n, k), generator=gen, dtype=torch.int8)
    scales = (torch.rand(n, k // 8, generator=gen) * 2.0**-7).to(torch.bfloat16)
    # Small-rms rows take the other strip gain; a zero row the norm floor.
    scales[: n // 4] *= 2.0**-12
    codes[n // 4] = 0
    return codes, scales


@pytest.mark.parametrize("label", [bp.LABEL_F1, bp.LABEL_F2, bp.LABEL_E2])
def test_native_lines_match_reference(label):
    count = 300
    native = bp._lines(_KEY, label, count, reference=False)
    assert torch.equal(_bits(native), _bits(bp._lines(_KEY, label, count, reference=True)))


@pytest.mark.parametrize("device", [Device.BLACKWELL, Device.HOPPER])
@pytest.mark.parametrize(("n", "k"), [(37, 512), (64, 768)])
def test_native_rows_match_reference(device, n, k):
    arithmetic = bp.ARITHMETIC[device]
    codes, scales = _layer(n, k, seed=n * k)
    f1 = bp._lines(_KEY, bp.LABEL_F1, k, reference=True).t().contiguous()
    f2 = bp._lines(_KEY, bp.LABEL_F2, k, reference=True).t().contiguous()
    e2 = bp._lines(_KEY, bp.LABEL_E2, n, reference=True)
    row_scales = bp._layer_row_scales(codes, scales, arithmetic.delta, None)

    b_prime, peel = bp._native_rows(codes, scales, row_scales, e2, f1, f2, arithmetic)
    ref_b_prime, ref_peel = bp._reference_rows(
        codes, scales, row_scales, e2, f1, f2, arithmetic, None
    )

    assert torch.equal(_bits(b_prime), _bits(ref_b_prime))
    assert torch.equal(peel[:, bp.RANK :], ref_peel[:, bp.RANK :])
    mid, ref_mid = peel[:, : bp.RANK].double(), ref_peel[:, : bp.RANK].double()
    assert (mid - ref_mid).norm() <= (ref_mid.norm() / 128)
