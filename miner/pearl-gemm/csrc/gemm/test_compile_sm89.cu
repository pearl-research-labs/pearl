// Stand-alone compile test for sm_89 substrate.
// Compiles kernel_traits_sm89.hpp AND the mainloop collective with a probe
// instantiation. Bug-finds in isolation before tackling the full codegen.
//
// Build:
//   nvcc -gencode arch=compute_89,code=sm_89 -std=c++20 -O0
//        -I . -I ../../third_party/cutlass/include
//        -I ../../third_party/cutlass/tools/util/include
//        -I ../../third_party/cutlass/examples/common
//        --expt-relaxed-constexpr --expt-extended-lambda
//        -DNDEBUG -c test_compile_sm89.cu -o /tmp/test_compile_sm89.o

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "cute/tensor.hpp"
#include "kernel_traits_sm89.hpp"
#include "collective_mainloop_sm89.hpp"
#include "collective_epilogue_sm89.hpp"

namespace pearl {

using TestTraits = KernelTraitsSm89<
    /*ElementIn=*/      int8_t,
    /*ElementOut=*/     cutlass::bfloat16_t,
    /*ElementDenoise=*/ cutlass::half_t,
    /*ElementScale=*/   float,
    /*TileShape_MNKR=*/ cute::Shape<cute::Int<128>, cute::Int<128>,
                                    cute::Int<128>, cute::Int<64>>,
    /*Is_Even_M=*/      true,
    /*Is_Even_N=*/      true,
    /*cM=*/             1,
    /*cN=*/             1,
    /*SkipReduction=*/  false,
    /*SkipDenoising=*/  true,   // exercise no-denoise path first
    /*kStages=*/        3,
    /*EnableDebug=*/    false>;

static_assert(TestTraits::kNumThreads == 256, "expect 8 warps");

using Mainloop = CollectiveMainloopSm89<TestTraits>;
using Epilogue = CollectiveEpilogueSm89<TestTraits>;

__global__ void probe_kernel() {
    extern __shared__ char smem_buf[];
    using SS = typename TestTraits::SharedStorage;
    auto& smem = *reinterpret_cast<SS*>(smem_buf);

    typename TestTraits::TiledMma tiled_mma;
    auto tCrC = cute::partition_fragment_C(
        tiled_mma, cute::select<0, 1>(typename TestTraits::TileShape_MNK{}));
    cute::clear(tCrC);

    auto transcript = cute::make_tensor<uint32_t>(
        cute::Int<blake3::MSG_BLOCK_SIZE_U32>{});
    cute::clear(transcript);

    typename Mainloop::Params params{};
    Mainloop mainloop;
    bool found = false;
    int  found_k = 0;
    mainloop.mma_init();
    mainloop.mainloop(params, smem,
                      cute::make_tuple(0, 0, 0), 1,
                      tCrC, transcript, found, found_k, threadIdx.x);

    // Epilogue probe: fp32 cast + scale + store
    auto tCrD_fp32 = cute::make_tensor_like<float>(tCrC);
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < cute::size(tCrD_fp32); ++i) {
      tCrD_fp32(i) = static_cast<float>(tCrC(i));
    }

    typename Epilogue::Params ep_params{};
    Epilogue epilogue;
    epilogue.scale(ep_params, tCrD_fp32, smem, tiled_mma, threadIdx.x,
                   cute::make_tuple(0, 0, 0));
    epilogue.store(ep_params, smem, threadIdx.x, cute::make_tuple(0, 0, 0));
}

}  // namespace pearl

int main() { return 0; }
