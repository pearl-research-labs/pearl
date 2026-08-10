import pytest
import torch
from vllm import _custom_ops as vllm_ops
from vllm_miner.gemm_operators import pearl_gemm_vanilla


def _torch_scaled_mm_reference(a, b, scale_a, scale_b, out_dtype, bias):
    acc = torch._int_mm(a, b.T.contiguous()).to(torch.float32)
    acc *= scale_a.view(-1, 1)
    acc *= scale_b.view(1, -1)
    if bias is not None:
        acc += bias.view(1, -1)
    return acc.to(out_dtype)


def _scaled_mm_reference(a, b, scale_a, scale_b, out_dtype, bias):
    try:
        return vllm_ops.cutlass_scaled_mm(
            a, b.T, scale_a=scale_a, scale_b=scale_b, out_dtype=out_dtype, bias=bias
        )
    except RuntimeError as exc:
        if torch.cuda.get_device_capability(a.device)[0] < 10:
            raise
        if "Int8 not supported" not in str(exc):
            raise
        return _torch_scaled_mm_reference(a, b, scale_a, scale_b, out_dtype, bias)


@pytest.mark.parametrize("m, n, k", [(1024, 1024, 1024), (8192, 8192, 8192)])
def test_pearl_gemm_vanilla_correctness(make_random_test_matrices, m, n, k):
    """
    Tests that pearl_gemm_vanilla runs without crashing and produces output
    of the expected shape.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    a, b, scale_a, scale_b, out_dtype, bias = make_random_test_matrices(m, n, k)

    output = pearl_gemm_vanilla(a, b, scale_a.squeeze(), scale_b.squeeze(), out_dtype)

    ref_output = _scaled_mm_reference(a, b, scale_a, scale_b, out_dtype, bias)
    torch.cuda.synchronize()

    assert output.shape == (m, n)
    assert output.dtype == out_dtype
    assert ref_output.shape == (m, n)
    assert ref_output.dtype == out_dtype
    assert torch.allclose(output, ref_output, atol=1e-2, rtol=1e-2)
