"""Validation harness for sm_89 noiseless GEMM.

JIT-builds pearl_gemm_sm89 (a single-config extension exposing
gemm_sm89(A, B, A_scales, B_scales, C)) and compares against torch._int_mm.

Skips/xfails gracefully if the current device isn't sm_89 — by design the
binary won't run on sm_120 (5090) or older archs.
"""

import os
import sys
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

GEMM_DIR = Path(__file__).resolve().parent
CUTLASS_INCLUDE = GEMM_DIR.parent.parent / "third_party" / "cutlass" / "include"
CSRC = GEMM_DIR.parent  # /pearl-gemm/csrc

assert (GEMM_DIR / "pearl_gemm_sm89_inst.cu").exists(), "missing inst.cu"
assert (GEMM_DIR / "pearl_gemm_sm89_pybind.cpp").exists(), "missing pybind.cpp"
assert (CUTLASS_INCLUDE / "cute" / "tensor.hpp").exists(), "cutlass submodule not populated"

ext = load(
    name="pearl_gemm_sm89_jit",
    sources=[
        str(GEMM_DIR / "pearl_gemm_sm89_inst.cu"),
        str(GEMM_DIR / "pearl_gemm_sm89_pybind.cpp"),
    ],
    extra_include_paths=[
        str(GEMM_DIR),
        str(CSRC),
        str(CUTLASS_INCLUDE),
        str(CUTLASS_INCLUDE.parent / "tools" / "util" / "include"),
        str(CUTLASS_INCLUDE.parent / "examples" / "common"),
    ],
    extra_cflags=["-O3", "-std=c++20"],
    extra_cuda_cflags=[
        "-O3", "-std=c++20",
        "-gencode", "arch=compute_89,code=sm_89",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "-U__CUDA_NO_BFLOAT16_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-DNDEBUG",
    ],
    verbose=True,
)


def ref_gemm(A: torch.Tensor, B: torch.Tensor,
             A_scales: torch.Tensor, B_scales: torch.Tensor) -> torch.Tensor:
    """CPU/PyTorch reference: C = (A @ B.T).to(fp32) * (a_scales × b_scales)."""
    assert A.dtype == torch.int8 and B.dtype == torch.int8
    # int32 matmul of (M,K) x (K,N) where B is (N,K) so we compute A @ B.T
    C_i32 = torch._int_mm(A.contiguous(), B.contiguous().t())
    C_f32 = C_i32.to(torch.float32)
    C_f32.mul_(A_scales.view(-1, 1))
    C_f32.mul_(B_scales.view(1, -1))
    return C_f32.to(torch.bfloat16)


def run_case(M: int, N: int, K: int, seed: int = 0):
    torch.manual_seed(seed)
    A = torch.randint(-127, 127, (M, K), dtype=torch.int8, device="cuda")
    B = torch.randint(-127, 127, (N, K), dtype=torch.int8, device="cuda")
    A_scales = torch.rand(M, dtype=torch.float32, device="cuda") * 0.02 + 0.005
    B_scales = torch.rand(N, dtype=torch.float32, device="cuda") * 0.02 + 0.005

    C_ref = ref_gemm(A, B, A_scales, B_scales)
    C_out = torch.zeros((M, N), dtype=torch.bfloat16, device="cuda")
    ext.gemm_sm89(A, B, A_scales, B_scales, C_out)

    abs_diff = (C_out.to(torch.float32) - C_ref.to(torch.float32)).abs()
    max_diff = abs_diff.max().item()
    rel_diff = abs_diff.div(C_ref.to(torch.float32).abs().clamp(min=1e-6)).max().item()
    print(f"M={M} N={N} K={K} seed={seed}: max|err|={max_diff:.4e} max_rel={rel_diff:.4e}")
    # Tolerance matches test_pearl_gemm.py: atol=1e-1 rtol=1e-2
    torch.testing.assert_close(C_out, C_ref, atol=1e-1, rtol=1e-2)
    print(f"  PASS")


def main():
    if not torch.cuda.is_available():
        sys.exit("no CUDA device")
    props = torch.cuda.get_device_properties(0)
    cc = props.major * 10 + props.minor
    print(f"device: {props.name} sm_{cc}")
    if cc != 89:
        sys.exit(f"this binary targets sm_89; current device is sm_{cc} — expected to fail launch")

    # Sweep: M, N, K all multiples of (bM=bN=bK=128), Is_Even paths only.
    for (M, N, K) in [(128, 128, 128), (256, 256, 128), (512, 512, 512), (1024, 1024, 1024)]:
        run_case(M, N, K)
    print("\nALL PASS")


if __name__ == "__main__":
    main()
