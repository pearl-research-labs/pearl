// SPDX-License-Identifier: see LICENSE
//
// Standalone correctness test for the kPersistB hook in CollectiveMainloopSm89.
//
// Runs the sm_89 mainloop twice on the same CTA. First call uses defaults
// (first_nonce_in_cohort=true) → full A+B cp.async pipeline. Second call uses
// (first_nonce_in_cohort=false, kPersistB=true) → B cp.async issuance is
// skipped. A device-side uint32 counter is incremented by thread 0 on every
// would-be B issuance; the test verifies:
//
//   first_call_b_issues  == K_PIPE_MAX - 1 + num_steady_b_issues  (>0)
//   second_call_b_issues == 0                                      (skipped)
//
// Build (on WSL CUDA 12.1 / 12.8 with system nvcc):
//   See _build_persist_b.sh in this directory for the exact command line.
//   The script invokes nvcc with -gencode arch=compute_89,code=sm_89, -std=c++20
//   -O3, --expt-relaxed-constexpr, --expt-extended-lambda, and the cutlass
//   include paths under ../../third_party/cutlass/.
//
// Run on a 4070 Ti SUPER (cap 8,9):
//   /tmp/test_persist_b
//
// ============================================================================
// SMEM ACCOUNTING (bM=128, bN=128, bK=64, kStages=3, R=64, SkipDenoising=true)
// ============================================================================
//
// SmemLayoutA cosize = bM * bK * kStages = 128 * 64 * 3 = 24576 int8 = 24 KB
// SmemLayoutB cosize = bN * bK * kStages = 128 * 64 * 3 = 24576 int8 = 24 KB
// SmemLayoutC cosize = bM * bN           = 128 * 128    = 16384 bf16 = 32 KB
//   (overlaps A+B in union; A+B = 48 KB ≥ 32 KB → A+B dominates the union)
// smem_scale_a       = bM * sizeof(float) = 128 * 4 = 512 B
// smem_scale_b       = bN * sizeof(float) = 128 * 4 = 512 B
// pipeline storage   = ~ 64 B per pipeline × 3 pipelines = ~192 B
// alignment + struct padding ≈ ~512 B
// ------------------------------------------------------
// Total                                  ≈ 48 KB + 1 KB + 0.5 KB ≈ ~49.6 KB
//
// Sm_89 dynamic smem cap (opt-in) = 99 KB → 50 KB headroom.
//
// === After kPersistB structural hook lands (this patch) ===
// Identical. The hook is a compile-time flag + runtime predicate; no smem
// reshaping. The actual B-skip relies on B-stages being valid in smem from the
// prior call's last-written K-tiles, which only holds when num_k_tiles ≤
// kStages. At K=192 (3 K-tiles × bK=64): all 3 B-stages in smem are valid for
// the next call's first 3 K-tile reads (in the same order). At K>192 this
// invariant breaks — the launcher must guard.
//
// === Hazards ===
// H1. Cross-call smem aliasing (Denoise variant): in SharedStorageDenoise the
//     A+B arms union with smem_AxEBL/EBR/EAL/EARxBpEB used by the denoise
//     epilogue. If the host invokes denoise() between two persisted-B mainloop
//     calls, smem_B is clobbered. Persistent-B path is only safe with
//     SkipDenoising=true (the production noiseless GEMM tile).
// H2. Cross-call smem aliasing (Epilogue C): smem_B unions with smem_C. Any
//     epilogue.scale()/store() between calls clobbers smem_B. Persistent-B
//     path requires the inter-call orchestration to skip the epilogue, e.g.
//     accumulate hashes/results into a per-nonce gmem buffer and run the
//     epilogue once per cohort.
// H3. Ring-buffer end state: at the end of a K-loop the steady-state has
//     issued (k_tile_count) more B cp.async into stages following a cyclic
//     pattern (smem_pipe_write incremented per tile). For num_k_tiles ==
//     kStages, the final smem state holds B[k_tile_count - kStages..
//     k_tile_count - 1] in stages [0..kStages-1] in cyclic order — which IS
//     what the next call's prologue + first kStages-1 steady reads expect.
//     For num_k_tiles ≠ kStages this aliasing breaks; see H4.
// H4. Loop-direction asymmetry: today's K-loop traverses forward only. To
//     enable persisted-B at num_k_tiles > kStages, the next call would need
//     to reverse-traverse K so the previous call's smem-resident B at K=last
//     is read first. Not implemented here; flagged for the multi-nonce
//     scheduler agent.
// H5. b_issue_counter is incremented inside divergent branches via atomicAdd
//     on a single thread (thread_idx == 0). Safe but produces serial atomic
//     contention — only enable in test builds, not production.
//
// === Touch points for the next session ===
// 1. MultiNonceTileScheduler (sibling agent): supply
//    `WorkTileInfo::first_nonce_in_cohort` + `cohort_size` and plumb to the
//    mainloop call site in pearl_gemm_kernel_sm89.h.
// 2. pearl_gemm_kernel_sm89.h: add nonce loop AROUND the mainloop call. On
//    nonce 0 of cohort, `first_nonce_in_cohort=true`; subsequent iters
//    `false`. Accumulators reset between nonces. Epilogue runs once per
//    cohort (or per nonce if cohort_size==1).
// 3. pearl_gemm_sm89_host.h: read env `PEARL_SM89_PERSISTENT_NONCE`. If set,
//    use the kPersistB=true Mainloop instantiation + a launcher that pre-
//    computes cohort_size from num_k_tiles (≤ kStages) and configures the
//    MultiNonceTileScheduler.
// 4. kernel_traits_sm89.hpp: add a static_assert that ties kStages to the
//    chosen tile so the launcher's runtime guard has a compile-time
//    counterpart for the per-cohort case.
// 5. Tier 2c (alpha-miner parity): extend to N-axis nonce amortization where
//    a CTA computes 256 output tiles sharing the same B-row but different
//    M-rows — that's the alpha-miner cubin pattern (`R185 0xFF`).

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cutlass/numeric_types.h>

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "cute/tensor.hpp"
#include "kernel_traits_sm89.hpp"
#include "collective_mainloop_sm89.hpp"

namespace pearl {

// Use the production noiseless tile: bM=bN=128, bK=64, R=64, kStages=3.
// SkipReduction=true (no PoW hash accumulator) keeps the test focused on the
// load/cp.async pipeline. SkipDenoising=true puts us on the SharedStorageNoDenoise
// path (smem_A + smem_B union).
using TestTraits = KernelTraitsSm89<
    /*ElementIn=*/      int8_t,
    /*ElementOut=*/     cutlass::bfloat16_t,
    /*ElementDenoise=*/ cutlass::half_t,
    /*ElementScale=*/   float,
    /*TileShape_MNKR=*/ cute::Shape<cute::Int<128>, cute::Int<128>,
                                    cute::Int<64>,  cute::Int<64>>,
    /*Is_Even_M=*/      true,
    /*Is_Even_N=*/      true,
    /*cM=*/             1,
    /*cN=*/             1,
    /*SkipReduction=*/  true,
    /*SkipDenoising=*/  true,
    /*kStages=*/        3,
    /*EnableDebug=*/    false>;

using MainloopPersist = CollectiveMainloopSm89<TestTraits, /*kPersistB=*/true>;

// Kernel that calls the mainloop twice. First call: full A+B fetch (counter
// records issuances). Second call: A-only fetch under the persist-B flag.
__global__ void __launch_bounds__(TestTraits::kNumThreads, 1)
test_persist_b_kernel(typename MainloopPersist::Params params_nonce0,
                      typename MainloopPersist::Params params_nonce1,
                      int                              k_tile_count,
                      unsigned int*                    counter_first,
                      unsigned int*                    counter_second) {
    extern __shared__ char smem_buf[];
    using SS = typename TestTraits::SharedStorage;
    auto& smem = *reinterpret_cast<SS*>(smem_buf);

    typename TestTraits::TiledMma tiled_mma;
    auto tCrC = cute::partition_fragment_C(
        tiled_mma, cute::select<0, 1>(typename TestTraits::TileShape_MNK{}));

    auto transcript = cute::make_tensor<uint32_t>(cute::Int<16>{});
    cute::clear(transcript);

    MainloopPersist collective;
    bool found = false;
    int  found_k = 0;

    collective.mma_init();

    // ---------- FIRST CALL: full A+B fetch ----------
    cute::clear(tCrC);
    collective.mainloop(
        params_nonce0, smem,
        cute::make_tuple(0, 0, 0), k_tile_count,
        tCrC, transcript, found, found_k, threadIdx.x,
        /*first_nonce_in_cohort=*/true,
        /*b_issue_counter=*/counter_first);

    // Block-wide barrier between the two calls so smem state from the first
    // call's `cp_async_wait<0> + __syncthreads()` epilogue is visible.
    __syncthreads();

    // ---------- SECOND CALL: A-only fetch (B persisted in smem) ----------
    cute::clear(tCrC);
    collective.mainloop(
        params_nonce1, smem,
        cute::make_tuple(0, 0, 0), k_tile_count,
        tCrC, transcript, found, found_k, threadIdx.x,
        /*first_nonce_in_cohort=*/false,
        /*b_issue_counter=*/counter_second);

    (void)found;
    (void)found_k;
}

}  // namespace pearl

#define CUCHK(x) do { auto _e = (x); if (_e != cudaSuccess) { \
    std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                 cudaGetErrorString(_e)); std::exit(1); } } while (0)

int main(int argc, char** argv) {
    int dev = 0;
    if (argc > 1) dev = std::atoi(argv[1]);
    CUCHK(cudaSetDevice(dev));
    cudaDeviceProp prop{};
    CUCHK(cudaGetDeviceProperties(&prop, dev));
    std::printf("Device %d: %s  cap %d.%d  smemPerBlockOptin %zu KB\n",
                dev, prop.name, prop.major, prop.minor,
                prop.sharedMemPerBlockOptin / 1024);
    if (prop.major < 8 || (prop.major == 8 && prop.minor < 9)) {
        std::fprintf(stderr, "Test requires sm_89+ (Ada Lovelace).\n");
        return 1;
    }

    // Problem dims: K small enough that the persisted-B invariant holds. With
    // kStages=3 and bK=64, we choose K=192 so num_k_tiles = K / bK = 3 = kStages.
    // This means at end of first call, smem_B holds B-tiles 0,1,2 in stages
    // [0,1,2] (cyclic, last-written = stage 2 since smem_pipe_write cycled
    // through). The second call's prologue (kStages-1 = 2 prefetches) reads
    // stages [0,1] for K-tiles 0,1 — which are still the correct B-tile data
    // (since both nonces use the same B). The steady-state will overwrite
    // stage 2 with A only; B-stage-2 retains its prior data, which is still
    // B-tile 2 — correct for k_tile=2.
    constexpr int bM = 128, bN = 128, bK = 64;
    int const M = bM, N = bN;
    int const K = 192;  // = kStages * bK
    int const k_tile_count = K / bK;  // = 3

    std::printf("Problem: M=%d N=%d K=%d  bM=%d bN=%d bK=%d  k_tile_count=%d\n",
                M, N, K, bM, bN, bK, k_tile_count);

    // Allocate A0, A1 (different content per nonce), B (shared).
    std::vector<int8_t> hA0(size_t(M) * K), hA1(size_t(M) * K), hB(size_t(N) * K);
    std::srand(0x12345);
    for (auto& v : hA0) v = int8_t((std::rand() % 255) - 127);
    for (auto& v : hA1) v = int8_t((std::rand() % 255) - 127);
    for (auto& v : hB)  v = int8_t((std::rand() % 255) - 127);

    int8_t *dA0, *dA1, *dB;
    CUCHK(cudaMalloc(&dA0, hA0.size()));
    CUCHK(cudaMalloc(&dA1, hA1.size()));
    CUCHK(cudaMalloc(&dB,  hB.size()));
    CUCHK(cudaMemcpy(dA0, hA0.data(), hA0.size(), cudaMemcpyHostToDevice));
    CUCHK(cudaMemcpy(dA1, hA1.data(), hA1.size(), cudaMemcpyHostToDevice));
    CUCHK(cudaMemcpy(dB,  hB.data(),  hB.size(),  cudaMemcpyHostToDevice));

    unsigned int *d_counter_first, *d_counter_second;
    CUCHK(cudaMalloc(&d_counter_first,  sizeof(unsigned int)));
    CUCHK(cudaMalloc(&d_counter_second, sizeof(unsigned int)));
    CUCHK(cudaMemset(d_counter_first,  0, sizeof(unsigned int)));
    CUCHK(cudaMemset(d_counter_second, 0, sizeof(unsigned int)));

    using Mainloop = pearl::MainloopPersist;
    typename Mainloop::Params p0{}, p1{};
    p0.ptr_A    = dA0;
    p0.layout_A = cute::make_layout(cute::make_shape(M, K),
                                    cute::make_stride(int64_t(K), cute::_1{}));
    p0.ptr_B    = dB;
    p0.layout_B = cute::make_layout(cute::make_shape(N, K),
                                    cute::make_stride(int64_t(K), cute::_1{}));
    p0.problem_shape = cute::make_tuple(M, N, K, /*R=*/64);

    p1 = p0;
    p1.ptr_A = dA1;  // only A varies between nonces

    size_t smem_size = sizeof(typename pearl::TestTraits::SharedStorage);
    CUCHK(cudaFuncSetAttribute(
        (void const*)&pearl::test_persist_b_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem_size)));

    std::printf("Launching probe kernel: 1 CTA × %d threads, %zu B smem\n",
                pearl::TestTraits::kNumThreads, smem_size);

    pearl::test_persist_b_kernel<<<1, pearl::TestTraits::kNumThreads,
                                   smem_size>>>(
        p0, p1, k_tile_count, d_counter_first, d_counter_second);
    CUCHK(cudaGetLastError());
    CUCHK(cudaDeviceSynchronize());

    unsigned int counter_first = 0, counter_second = 0;
    CUCHK(cudaMemcpy(&counter_first,  d_counter_first,
                     sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CUCHK(cudaMemcpy(&counter_second, d_counter_second,
                     sizeof(unsigned int), cudaMemcpyDeviceToHost));

    // EXPECTED: with k_tile_count=3 and kStages=3, the first call issues:
    //   prologue:    K_PIPE_MAX - 1 = 2 B-tiles
    //   steady-state: issues B once per outer iter as long as k_tile_count>0.
    //                 Loop runs k_tile_total = k_tile_count + (K_PIPE_MAX-1) = 5
    //                 outer iters; at each k_block==0 inside the iter,
    //                 k_tile_count is decremented; B is issued only when
    //                 k_tile_count > 0 BEFORE the decrement.
    //                 At entry to steady-state, k_tile_count was decremented
    //                 to 1 by the prologue (3 → 2 → 1, since the prologue
    //                 stops decrementing once k_tile_count reaches 0).
    //                 More precisely: prologue runs K_PIPE_MAX-1=2 iters and
    //                 decrements k_tile_count from 3 → 2 → 1.
    //                 Steady iter 0: k_tile_count=1, fetch issued, then dec
    //                                to 0.
    //                 Steady iter 1: k_tile_count=0, no fetch.
    //                 Steady iter ≥ 2: same.
    //                 → Steady fetches B once.
    // So first-call counter = 2 (prologue) + 1 (steady) = 3.
    int const expected_first = 3;
    int const expected_second = 0;

    std::printf("\n=== Result ===\n");
    std::printf("  first  call B cp.async issues = %u  (expected %d)\n",
                counter_first,  expected_first);
    std::printf("  second call B cp.async issues = %u  (expected %d)\n",
                counter_second, expected_second);

    int ok = 1;
    if (counter_first != (unsigned)expected_first) {
        std::printf("FAIL: first-call B issuance mismatch.\n");
        ok = 0;
    }
    if (counter_second != (unsigned)expected_second) {
        std::printf("FAIL: second-call B issuance not skipped — "
                    "kPersistB hook not effective.\n");
        ok = 0;
    }
    if (ok) {
        std::printf("PASS: kPersistB hook correctly skips B cp.async on "
                    "subsequent nonce.\n");
    }

    cudaFree(d_counter_first);
    cudaFree(d_counter_second);
    cudaFree(dA0);
    cudaFree(dA1);
    cudaFree(dB);
    return ok ? 0 : 1;
}
