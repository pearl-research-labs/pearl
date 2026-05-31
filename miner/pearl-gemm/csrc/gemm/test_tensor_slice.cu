// Reproducer for the partition_fragment_A error — now using KernelTraitsSm89.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "cute/tensor.hpp"
#include "kernel_traits_sm89.hpp"

using namespace cute;
using namespace pearl;

using T = KernelTraitsSm89<
    int8_t, cutlass::bfloat16_t, cutlass::half_t, float,
    Shape<Int<128>, Int<128>, Int<128>, Int<64>>,
    true, true, 1, 1, false, true, 3, false>;

__global__ void test() {
    static_assert(rank_v<typename T::SmemLayoutA> == 3,
                  "SmemLayoutA must be 3D — if this fires the layout is degenerate");

    __shared__ int8_t buf[cosize_v<typename T::SmemLayoutA>];
    auto sA = make_tensor(make_smem_ptr(buf), typename T::SmemLayoutA{});

    // Type probe
    static_assert(is_tensor<decltype(sA)>::value, "sA must be a Tensor");

    auto sA_2d = sA(_, _, Int<0>{});
    static_assert(is_tensor<decltype(sA_2d)>::value, "sA_2d must be a Tensor");
    static_assert(rank_v<decltype(sA_2d)> == 2, "slice must be 2D");

    typename T::TiledMma mma;
    auto thr_mma = mma.get_thread_slice(threadIdx.x);
    auto tCrA = thr_mma.partition_fragment_A(sA_2d);
    (void)tCrA;
}

int main() { return 0; }
