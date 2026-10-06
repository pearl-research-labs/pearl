// Bit-exact A100 (sm_80) FP16 -> FP32 GEMM tile.
//
// Reproduces the FP16 scheme verifier's accumulation model
// (zk-pow/src/api/fp16/accumulate.rs) natively: the native HMMA.16816.F32
// datapath *is* the model (the model was measured from this silicon). Each warp
// computes one 16x8 output subtile D = A(16xK) . B(8xK)^T with FP32 accumulation
// chained in ascending k order (groups of 16 per mma.sync, no split-k, no
// atomics), carrying c0..c3 forward across the whole k axis. This fixed
// reduction order is what makes the tile match the verifier bit-for-bit.
//
//   A: (M, K) row-major FP16  (logical left operand)
//   B: (N, K) row-major FP16  (the transposed logical right operand: row j is
//      logical column j, exactly a100_matmul's `b`)
//   C: (M, N) row-major FP32 carry-in, or null for +0
//   D: (M, N) row-major FP32 output
//   Requires K % 16 == 0, M % 16 == 0, N % 8 == 0.

#include <cuda_fp16.h>
#include <torch/extension.h>

__device__ __forceinline__ unsigned pack2(const __half* p, int i0, int i1) {
    __half2 h = __halves2half2(p[i0], p[i1]);
    return *reinterpret_cast<unsigned*>(&h);
}

__global__ void fp16_gemm_a100_kernel(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    const float* __restrict__ C,
    float* __restrict__ D,
    int M, int N, int K) {
    // One warp (one block) owns one 16x8 output subtile.
    // grid.x indexes the N tiles (8 cols each), grid.y the M tiles (16 rows each).
    int m0 = blockIdx.y * 16;
    int n0 = blockIdx.x * 8;
    if (m0 >= M || n0 >= N) return;

    int lane = threadIdx.x & 31;
    int gid = lane >> 2;   // 0..7
    int t4 = lane & 3;     // 0..3

    const __half* Am = A + (long)m0 * K;        // rows m0..m0+15
    const __half* Bn = B + (long)n0 * K;        // rows n0..n0+7 (logical cols)

    float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
    if (C != nullptr) {
        const float* Ct = C + (long)m0 * N + n0;
        c0 = Ct[(gid) * N + t4 * 2];
        c1 = Ct[(gid) * N + t4 * 2 + 1];
        c2 = Ct[(gid + 8) * N + t4 * 2];
        c3 = Ct[(gid + 8) * N + t4 * 2 + 1];
    }

    for (int k0 = 0; k0 < K; k0 += 16) {
        unsigned a0 = pack2(Am, (gid) * K + k0 + t4 * 2,     (gid) * K + k0 + t4 * 2 + 1);
        unsigned a1 = pack2(Am, (gid + 8) * K + k0 + t4 * 2, (gid + 8) * K + k0 + t4 * 2 + 1);
        unsigned a2 = pack2(Am, (gid) * K + k0 + t4 * 2 + 8, (gid) * K + k0 + t4 * 2 + 9);
        unsigned a3 = pack2(Am, (gid + 8) * K + k0 + t4 * 2 + 8, (gid + 8) * K + k0 + t4 * 2 + 9);
        unsigned b0 = pack2(Bn, (gid) * K + k0 + t4 * 2,     (gid) * K + k0 + t4 * 2 + 1);
        unsigned b1 = pack2(Bn, (gid) * K + k0 + t4 * 2 + 8, (gid) * K + k0 + t4 * 2 + 9);
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    }

    float* Dt = D + (long)m0 * N + n0;
    Dt[(gid) * N + t4 * 2]       = c0;
    Dt[(gid) * N + t4 * 2 + 1]   = c1;
    Dt[(gid + 8) * N + t4 * 2]   = c2;
    Dt[(gid + 8) * N + t4 * 2 + 1] = c3;
}

torch::Tensor fp16_gemm_a100(torch::Tensor A, torch::Tensor B,
                             c10::optional<torch::Tensor> C) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "A and B must be CUDA tensors");
    TORCH_CHECK(A.scalar_type() == torch::kFloat16, "A must be float16");
    TORCH_CHECK(B.scalar_type() == torch::kFloat16, "B must be float16");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), "A and B must be contiguous");
    int M = A.size(0), K = A.size(1);
    int N = B.size(0), Kb = B.size(1);
    TORCH_CHECK(K == Kb, "A and B must share K");
    TORCH_CHECK(K % 16 == 0, "K must be a multiple of 16");
    TORCH_CHECK(M % 16 == 0, "M must be a multiple of 16");
    TORCH_CHECK(N % 8 == 0, "N must be a multiple of 8");

    const float* Cptr = nullptr;
    torch::Tensor Ct;
    if (C.has_value()) {
        Ct = C.value();
        TORCH_CHECK(Ct.is_cuda() && Ct.scalar_type() == torch::kFloat32, "C must be float32 CUDA");
        TORCH_CHECK(Ct.is_contiguous() && Ct.size(0) == M && Ct.size(1) == N, "C must be (M,N) contiguous");
        Cptr = (const float*)Ct.data_ptr();
    }

    auto D = torch::empty({M, N}, torch::dtype(torch::kFloat32).device(A.device()));
    dim3 grid(N / 8, M / 16);
    dim3 block(32, 1);
    fp16_gemm_a100_kernel<<<grid, block>>>(
        (const __half*)A.data_ptr(), (const __half*)B.data_ptr(),
        Cptr, (float*)D.data_ptr(), M, N, K);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "fp16_gemm_a100 launch failed");
    return D;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fp16_gemm_a100", &fp16_gemm_a100, "A100 sm_80 bit-exact FP16->FP32 GEMM",
          pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("C") = c10::nullopt);
}
