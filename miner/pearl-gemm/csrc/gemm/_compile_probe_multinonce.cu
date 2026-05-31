// SPDX-License-Identifier: see LICENSE
//
// Stand-alone compile-probe for the MultiNonceTileScheduler integration.
// Pulls in:
//   * The new pearl_gemm_sm89_multinonce_scheduler.hpp
//   * The patched pearl_gemm_kernel_sm89.h (with HasNonceContextsField concept
//     and the per-iteration NonceContext override block)
//   * The patched pearl_gemm_sm89_host.h (with PEARL_SM89_PERSISTENT_NONCE env
//     gating and the second cudaFuncSetAttribute path)
//   * The NoiselessTraits128x128x64_R64 KTraits — same instantiation as
//     pearl_gemm_sm89_inst.cu (production R=64 tile).
//
// Goal: confirm the multi-nonce kernel symbol pearl::ada_gemm<KTraits,
//       MultiNonceTileScheduler<256>> instantiates and links cleanly alongside
//       the existing pearl::ada_gemm<KTraits, PersistentSwizzledTileScheduler>.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "cute/tensor.hpp"

#include "kernel_traits_sm89.hpp"
#include "collective_mainloop_sm89.hpp"
#include "collective_epilogue_sm89.hpp"
#include "pearl_gemm_kernel_sm89.h"
#include "pearl_gemm_sm89_host.h"
#include "pearl_gemm_sm89_multinonce_scheduler.hpp"

namespace pearl {

// Mirror the production noiseless R=64 trait.
using ProbeTraits = KernelTraitsSm89<
    /*ElementIn=*/      int8_t,
    /*ElementOut=*/     cutlass::bfloat16_t,
    /*ElementDenoise=*/ cutlass::half_t,
    /*ElementScale=*/   float,
    /*TileShape_MNKR=*/ cute::Shape<cute::Int<128>, cute::Int<128>,
                                    cute::Int<64>, cute::Int<64>>,
    /*Is_Even_M=*/      true,
    /*Is_Even_N=*/      true,
    /*cM=*/             1,
    /*cN=*/             1,
    /*SkipReduction=*/  true,
    /*SkipDenoising=*/  true,
    /*kStages=*/        3,
    /*EnableDebug=*/    false>;

// Indirect instantiation by taking the kernel address (explicit-instantiation
// syntax trips on CTK 12.8's __grid_constant__ annotation propagation; address-
// take instantiates equivalently and works without extra annotations).
using KernelPtrT = void (*)(
    typename CollectiveMainloopSm89<ProbeTraits>::Params,
    typename CollectiveEpilogueSm89<ProbeTraits>::Params,
    typename sm89::MultiNonceTileScheduler<256>::Params);
KernelPtrT g_probe_kernel_ptr =
    &ada_gemm<ProbeTraits, sm89::MultiNonceTileScheduler<256>>;

// Also exercise the host launcher with the same traits — this drags in the
// PEARL_SM89_PERSISTENT_NONCE env-gated branch + cudaFuncSetAttribute calls.
void probe_host_launch(
    typename CollectiveMainloopSm89<ProbeTraits>::Arguments const& m_args,
    typename CollectiveEpilogueSm89<ProbeTraits>::Arguments const& e_args) {
  sm89::pearl_gemm_sm89_run<ProbeTraits>(m_args, e_args,
                                         128, 128, 64, /*stream=*/0);
}

}  // namespace pearl

int main() { return 0; }
