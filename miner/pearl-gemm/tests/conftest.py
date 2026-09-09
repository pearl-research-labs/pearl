import pytest
import torch


@pytest.fixture(scope="session", autouse=True)
def require_sm100():
    """Fail the B200-only suite instead of silently skipping hardware coverage."""
    assert torch.cuda.is_available(), "pearl_gemm tests require CUDA"
    major, minor = torch.cuda.get_device_capability()
    assert major == 10, f"pearl_gemm tests require SM100, got sm{major}{minor}"


@pytest.fixture(autouse=True)
def clear_gpu_cache():
    """Clear GPU memory cache before and after each test."""
    torch.cuda.empty_cache()
    yield
    torch.cuda.empty_cache()
