import pytest
import torch

from pearl_gemm._utils._arch import arch_of


@pytest.fixture(scope="session", autouse=True)
def require_supported_arch():
    """Fail the GPU suite on unknown hardware instead of silently skipping coverage.

    Any ``Arch`` family runs the suite; an op a family does not implement yet
    fails its own tests there through the host's ``require_arch`` gate.
    """
    assert torch.cuda.is_available(), "pearl_gemm tests require CUDA"
    arch_of()


@pytest.fixture(autouse=True)
def clear_gpu_cache():
    """Clear GPU memory cache before and after each test."""
    torch.cuda.empty_cache()
    yield
    torch.cuda.empty_cache()
