#pragma once

#include <cutlass/arch/arch.h>
#include <cutlass/arch/barrier.h>
#include <cutlass/cutlass.h>
#include "cute/tensor.hpp"

#include "cutlass/gemm/collective/collective_builder.hpp"

#include "convert_util.h"
#include "named_barrier.hpp"
#include "pearl_gemm_constants.hpp"
#include "utils.h"

namespace pearl {

using namespace cute;

template <typename KTraits>
struct CollectiveEpilogue {

  // dtypes
  using ElementIn = typename KTraits::ElementIn;
  using ElementOutput = typename KTraits::ElementOut;
  using ElementDenoise = typename KTraits::ElementDenoise;
  using ElementAccum = typename KTraits::ElementAccum;
  using ElementScale = typename KTraits::ElementScale;

  // Warp specialization
  static constexpr int kNumWarps = KTraits::kNumWarps;
  static constexpr int kNThreads = kNumWarps * cutlass::NumThreadsPerWarp;
  static constexpr int kNumCopyThreads = cutlass::NumThreadsPerWarpGroup;
  static constexpr int kNumMmaThreads = KTraits::kNumMmaThreads;
  static constexpr int srcLane = KTraits::srcLane;

  // Shapes and tile dimensions
  using ProblemShape = typename KTraits::ProblemShape;
  using TileShape_MNK = typename KTraits::TileShape_MNK;
  using TileShape_MNR = typename KTraits::TileShape_MNR;
  static constexpr int bM = get<0>(TileShape_MNR{});
  static constexpr int bN = get<1>(TileShape_MNR{});
  static constexpr int R = KTraits::R;

  // Cluster layout
  using ClusterShape_MNK = typename KTraits::ClusterShape_MNK;
  static constexpr int kClusterSizeM = KTraits::kClusterSizeM;
  static constexpr int kClusterSizeN = KTraits::kClusterSizeN;

  // GMEM layouts
  // A_scales, B_scales: (M) or (N)
  using Layout1DT = Layout<Shape<int32_t>, Stride<_1>>;

  // C: (M, N) N-major
  using Shape2DT = Shape<int32_t, int32_t>;
  using Stride2DT = Stride<int32_t, _1>;
  using Layout2DT = Layout<Shape2DT, Stride2DT>;

  // AxEBL, EARxBpEB, EAL, EBR: (M, R) or (N, R), R-major
  using ShapeDenoiseT = Shape<int, Int<R>>;
  using StrideDenoiseT = Stride<Int<R>, _1>;
  using LayoutDenoiseT = Layout<ShapeDenoiseT, StrideDenoiseT>;

  // SMEM layouts
  using SharedStorage = typename KTraits::SharedStorage;
  using SmemLayoutScaleA = typename KTraits::SmemLayoutScaleA;
  using SmemLayoutScaleB = typename KTraits::SmemLayoutScaleB;
  using SmemLayoutC = typename KTraits::SmemLayoutC;
  using SmemCopyAtomC = typename KTraits::SmemCopyAtomC;
  using SmemLayoutAxEBL = typename KTraits::SmemLayoutAxEBL;
  using SmemLayoutEAL = typename KTraits::SmemLayoutEAL;
  using SmemLayoutEARxBpEB = typename KTraits::SmemLayoutEARxBpEB;
  using SmemLayoutEBR = typename KTraits::SmemLayoutEBR;
  // TMA copy atoms
  // Load denoise factors (M, R) and (N, R), multicast along M or N
  using TMAOpA = KTraits::TMAOpA;
  using TMAOpB = KTraits::TMAOpB;

  using TMA_AxEBL = decltype(make_tma_copy(
      TMAOpA{},
      make_tensor(make_gmem_ptr(recast_ptr<ElementDenoise>(nullptr)),
                  ShapeDenoiseT{}, StrideDenoiseT{}),
      SmemLayoutAxEBL{}, select<0, 2>(TileShape_MNR{}), kClusterSizeN));
  using TMA_EBR = decltype(make_tma_copy(
      TMAOpB{},
      make_tensor(make_gmem_ptr(recast_ptr<ElementDenoise>(nullptr)),
                  ShapeDenoiseT{}, StrideDenoiseT{}),
      SmemLayoutEBR{}, select<1, 2>(TileShape_MNR{}), kClusterSizeM));
  using TMA_EAL = decltype(make_tma_copy(
      TMAOpA{},
      make_tensor(make_gmem_ptr(recast_ptr<ElementDenoise>(nullptr)),
                  ShapeDenoiseT{}, StrideDenoiseT{}),
      SmemLayoutEAL{}, select<0, 2>(TileShape_MNR{}), kClusterSizeN));
  using TMA_EARxBpEB = decltype(make_tma_copy(
      TMAOpB{},
      make_tensor(make_gmem_ptr(recast_ptr<ElementDenoise>(nullptr)),
                  ShapeDenoiseT{}, StrideDenoiseT{}),
      SmemLayoutEARxBpEB{}, select<1, 2>(TileShape_MNR{}), kClusterSizeM));

  // Store C, no multicast
  using TMA_C = decltype(make_tma_copy(
      SM90_TMA_STORE{},
      make_tensor(make_gmem_ptr(static_cast<ElementOutput*>(nullptr)),
                  Shape2DT{}, Stride2DT{}),
      SmemLayoutC{}, select<0, 1>(TileShape_MNK{}), _1{}));

  static constexpr uint32_t TmaTransactionBytesAxEBL = static_cast<uint32_t>(
      size(SmemLayoutAxEBL{}) * cutlass::sizeof_bits_v<ElementDenoise> / 8);
  static constexpr uint32_t TmaTransactionBytesEBR = static_cast<uint32_t>(
      size(SmemLayoutEBR{}) * cutlass::sizeof_bits_v<ElementDenoise> / 8);
  static constexpr uint32_t TmaTransactionBytesEAL = static_cast<uint32_t>(
      size(SmemLayoutEAL{}) * cutlass::sizeof_bits_v<ElementDenoise> / 8);
  static constexpr uint32_t TmaTransactionBytesEARxBpEB = static_cast<uint32_t>(
      size(SmemLayoutEARxBpEB{}) * cutlass::sizeof_bits_v<ElementDenoise> / 8);
  static constexpr uint32_t TmaTransactionBytesAxEB =
      TmaTransactionBytesAxEBL + TmaTransactionBytesEBR;
  static constexpr uint32_t TmaTransactionBytesEAxBpEB =
      TmaTransactionBytesEAL + TmaTransactionBytesEARxBpEB;

  // G2S load scales (cp.async)
  using G2SScales_copy_atom = typename KTraits::G2SScales_copy_atom;
  using G2SScalesCopyA = typename KTraits::G2SScalesCopyA;
  using G2SScalesCopyB = typename KTraits::G2SScalesCopyB;

  // Pipelines
  using MainloopPipeline = typename KTraits::MainloopPipeline;
  using PipelineState = typename MainloopPipeline::PipelineState;
  using DenoisePipeline = typename KTraits::DenoisePipeline;
  using DenoisePipelineState = typename DenoisePipeline::PipelineState;
  using DenoisePipelineBarrierType =
      typename DenoisePipeline::ProducerBarrierType;

  // Host side kernel arguments
  struct Arguments {
    ElementOutput* ptr_C;
    ElementScale const* ptr_A_scales;
    ElementScale const* ptr_B_scales;
    ElementDenoise const* ptr_EAL;
    ElementDenoise const* ptr_EARxBpEB;
    ElementDenoise const* ptr_AxEBL;
    ElementDenoise const* ptr_EBR;
    ProblemShape problem_shape;
  };

  // Device side kernel params
  struct Params {
    ElementOutput* ptr_C;
    ElementDenoise const* EAL;
    ElementDenoise const* EARxBpEB;
    ElementDenoise const* AxEBL;
    ElementDenoise const* EBR;
    ElementScale const* ptr_A_scales;
    ElementScale const* ptr_B_scales;

    TMA_C tma_store;
    TMA_EAL tma_load_EAL;
    TMA_EARxBpEB tma_load_EARxBpEB;
    TMA_AxEBL tma_load_AxEBL;
    TMA_EBR tma_load_EBR;

    Layout2DT const layout_C;
    Layout1DT const layout_A_scales;
    Layout1DT const layout_B_scales;
    LayoutDenoiseT const layout_EAL;
    LayoutDenoiseT const layout_EARxBpEB;
    LayoutDenoiseT const layout_AxEBL;
    LayoutDenoiseT const layout_EBR;

    ProblemShape problem_shape;
  };

  static Params to_underlying_arguments(Arguments const& args) {
    auto [M, N, K, R_] = args.problem_shape;

    Layout1DT layout_A_scales = make_layout(make_shape(M));
    Layout1DT layout_B_scales = make_layout(make_shape(N));
    LayoutDenoiseT layout_AxEBL =
        make_layout(make_shape(M, Int<R>{}), Stride<Int<R>, _1>{});
    LayoutDenoiseT layout_EAL =
        make_layout(make_shape(M, Int<R>{}), Stride<Int<R>, _1>{});
    LayoutDenoiseT layout_EBR =
        make_layout(make_shape(N, Int<R>{}), Stride<Int<R>, _1>{});
    LayoutDenoiseT layout_EARxBpEB =
        make_layout(make_shape(N, Int<R>{}), Stride<Int<R>, _1>{});
    Layout2DT layout_C = make_layout(make_shape(M, N), make_stride(N, _1{}));

    Tensor mAxEBL = make_tensor(make_gmem_ptr(args.ptr_AxEBL), layout_AxEBL);
    Tensor mEAL = make_tensor(make_gmem_ptr(args.ptr_EAL), layout_EAL);
    Tensor mEBR = make_tensor(make_gmem_ptr(args.ptr_EBR), layout_EBR);
    Tensor mEARxBpEB =
        make_tensor(make_gmem_ptr(args.ptr_EARxBpEB), layout_EARxBpEB);
    Tensor mC = make_tensor(make_gmem_ptr(args.ptr_C), layout_C);

    TMA_EAL tma_load_EAL =
        make_tma_copy(TMAOpA{}, mEAL, SmemLayoutEAL{},
                      select<0, 2>(TileShape_MNR{}), kClusterSizeN);
    TMA_EARxBpEB tma_load_EARxBpEB =
        make_tma_copy(TMAOpB{}, mEARxBpEB, SmemLayoutEARxBpEB{},
                      select<1, 2>(TileShape_MNR{}), kClusterSizeM);
    TMA_AxEBL tma_load_AxEBL =
        make_tma_copy(TMAOpA{}, mAxEBL, SmemLayoutAxEBL{},
                      select<0, 2>(TileShape_MNR{}), kClusterSizeN);
    TMA_EBR tma_load_EBR =
        make_tma_copy(TMAOpB{}, mEBR, SmemLayoutEBR{},
                      select<1, 2>(TileShape_MNR{}), kClusterSizeM);
    TMA_C tma_store =
        make_tma_copy(cute::SM90_TMA_STORE{}, mC, SmemLayoutC{},
                      select<0, 1>(TileShape_MNK{}), _1{});  // no mcast for C

    return {args.ptr_C,        args.ptr_EAL,      args.ptr_EARxBpEB,
            args.ptr_AxEBL,    args.ptr_EBR,      args.ptr_A_scales,
            args.ptr_B_scales, tma_store,         tma_load_EAL,
            tma_load_EARxBpEB, tma_load_AxEBL,    tma_load_EBR,
            layout_C,          layout_A_scales,   layout_B_scales,
            layout_EAL,        layout_EARxBpEB,   layout_AxEBL,
            layout_EBR,        args.problem_shape};
  }

  /// Issue Tma Descriptor Prefetch -- ideally from a single thread for best performance
  CUTLASS_DEVICE
  static void prefetch_tma_descriptors(Params const& epilogue_params) {
    cute::prefetch_tma_descriptor(
        epilogue_params.tma_load_AxEBL.get_tma_descriptor());
    cute::prefetch_tma_descriptor(
        epilogue_params.tma_load_EBR.get_tma_descriptor());
    cute::prefetch_tma_descriptor(
        epilogue_params.tma_load_EAL.get_tma_descriptor());
    cute::prefetch_tma_descriptor(
        epilogue_params.tma_load_EARxBpEB.get_tma_descriptor());
    cute::prefetch_tma_descriptor(
        epilogue_params.tma_store.get_tma_descriptor());
  }

  CUTLASS_DEVICE void load_denoise(
      MainloopPipeline mainloop_pipeline,
      PipelineState& mainloop_smem_pipe_write, Params const& epilogue_params,
      DenoisePipeline AxEB_pipeline, DenoisePipeline EAxBpEB_pipeline,
      DenoisePipelineState& AxEB_pipe_write,
      DenoisePipelineState& EAxBpEB_pipe_write, SharedStorage& shared_storage,
      cute::tuple<int32_t, int32_t, int32_t> block_coord,
      uint16_t const tma_mcast_mask_a, uint16_t const tma_mcast_mask_b) {

    auto [m_block, n_block, bidb] = block_coord;
    // SMEM tiles: (bM or bN, R, Stages) where Stages is typically 1
    Tensor sAxEBL = make_tensor(make_smem_ptr(shared_storage.smem_AxEBL.data()),
                                SmemLayoutAxEBL{});
    Tensor sEBR = make_tensor(make_smem_ptr(shared_storage.smem_EBR.data()),
                              SmemLayoutEBR{});
    Tensor sEAL = make_tensor(make_smem_ptr(shared_storage.smem_EAL.data()),
                              SmemLayoutEAL{});
    Tensor sEARxBpEB =
        make_tensor(make_smem_ptr(shared_storage.smem_EARxBpEB.data()),
                    SmemLayoutEARxBpEB{});

    Tensor mAxEBL = epilogue_params.tma_load_AxEBL.get_tma_tensor(
        epilogue_params.layout_AxEBL.shape());
    Tensor mEBR = epilogue_params.tma_load_EBR.get_tma_tensor(
        epilogue_params.layout_EBR.shape());
    Tensor mEAL = epilogue_params.tma_load_EAL.get_tma_tensor(
        epilogue_params.layout_EAL.shape());
    Tensor mEARxBpEB = epilogue_params.tma_load_EARxBpEB.get_tma_tensor(
        epilogue_params.layout_EARxBpEB.shape());

    // GMEM tiles: (bMN, R, _1)
    Tensor gAxEBL =
        local_tile(mAxEBL, Shape<Int<bM>, Int<R>>{}, make_coord(m_block, _));
    Tensor gEBR =
        local_tile(mEBR, Shape<Int<bN>, Int<R>>{}, make_coord(n_block, _));
    Tensor gEAL =
        local_tile(mEAL, Shape<Int<bM>, Int<R>>{}, make_coord(m_block, _));
    Tensor gEARxBpEB =
        local_tile(mEARxBpEB, Shape<Int<bN>, Int<R>>{}, make_coord(n_block, _));

    // TMA partitions:
    // tXgX: (TMA, _1) where TMA = bMN * R
    // tXsX: (TMA, Stages)
    auto [tAxEBLgAxEBL, tAxEBLsAxEBL] =
        tma_partition(epilogue_params.tma_load_AxEBL,
                      get<1>(block_id_in_cluster()), make_layout(kClusterSizeN),
                      group_modes<0, 2>(sAxEBL), group_modes<0, 2>(gAxEBL));
    auto [tEBRgEBR, tEBRsEBR] =
        tma_partition(epilogue_params.tma_load_EBR,
                      get<0>(block_id_in_cluster()), make_layout(kClusterSizeM),
                      group_modes<0, 2>(sEBR), group_modes<0, 2>(gEBR));
    auto [tEALgEAL, tEALsEAL] =
        tma_partition(epilogue_params.tma_load_EAL,
                      get<1>(block_id_in_cluster()), make_layout(kClusterSizeN),
                      group_modes<0, 2>(sEAL), group_modes<0, 2>(gEAL));
    auto [tEARxBpEBgEARxBpEB, tEARxBpEBsEARxBpEB] = tma_partition(
        epilogue_params.tma_load_EARxBpEB, get<0>(block_id_in_cluster()),
        make_layout(kClusterSizeM), group_modes<0, 2>(sEARxBpEB),
        group_modes<0, 2>(gEARxBpEB));

    int lane_predicate = cute::elect_one_sync();
    // Just a single load for each tensor per worktile

    if (lane_predicate) {
      // wait for mainloop mmas to finish before loading denoise factors
      mainloop_pipeline.producer_tail(mainloop_smem_pipe_write);
      EAxBpEB_pipeline.producer_acquire(EAxBpEB_pipe_write);
      DenoisePipelineBarrierType* EAxBpEB_tmaBar =
          EAxBpEB_pipeline.producer_get_barrier(EAxBpEB_pipe_write);
      auto EAxBpEB_stage = EAxBpEB_pipe_write.index();
      copy(epilogue_params.tma_load_EAL.with(*EAxBpEB_tmaBar, tma_mcast_mask_a),
           tEALgEAL(_, _0{}), tEALsEAL(_, EAxBpEB_stage));
      copy(epilogue_params.tma_load_EARxBpEB.with(*EAxBpEB_tmaBar,
                                                  tma_mcast_mask_b),
           tEARxBpEBgEARxBpEB(_, _0{}), tEARxBpEBsEARxBpEB(_, EAxBpEB_stage));
      // no-op commit
      EAxBpEB_pipeline.producer_commit(EAxBpEB_pipe_write,
                                       TmaTransactionBytesEAxBpEB);
      ++EAxBpEB_pipe_write;
    }

    // AxEB pipeline

    // Path 3: wait for consumer to release EAxBpEB SMEM before reusing the
    // aliased region for AxEB TMA loads. All producer-warp threads participate.
    cutlass::arch::NamedBarrier::sync(
        kNumMmaThreads + cutlass::NumThreadsPerWarp,
        static_cast<cutlass::arch::ReservedNamedBarriers>(
            pearl::NamedBarriers::DenoisePhase1Consumed));

    if (lane_predicate) {

      AxEB_pipeline.producer_acquire(AxEB_pipe_write);
      DenoisePipelineBarrierType* AxEB_tmaBar =
          AxEB_pipeline.producer_get_barrier(AxEB_pipe_write);
      auto AxEB_stage = AxEB_pipe_write.index();
      copy(epilogue_params.tma_load_AxEBL.with(*AxEB_tmaBar, tma_mcast_mask_a),
           tAxEBLgAxEBL(_, _0{}), tAxEBLsAxEBL(_, AxEB_stage));
      copy(epilogue_params.tma_load_EBR.with(*AxEB_tmaBar, tma_mcast_mask_b),
           tEBRgEBR(_, _0{}), tEBRsEBR(_, AxEB_stage));
      // no-op commit
      AxEB_pipeline.producer_commit(AxEB_pipe_write, TmaTransactionBytesAxEB);
      ++AxEB_pipe_write;
    }

    // Notify producer that it can load A and B for next worktile
    cutlass::arch::fence_view_async_shared();
    cutlass::arch::NamedBarrier::arrive(
        kNumMmaThreads + cutlass::NumThreadsPerWarp,
        static_cast<cutlass::arch::ReservedNamedBarriers>(
            pearl::NamedBarriers::DenoiseComplete));
  }

  CUTLASS_DEVICE void load_denoise_tail(
      DenoisePipeline AxEB_pipeline, DenoisePipeline EAxBpEB_pipeline,
      DenoisePipelineState& AxEB_pipe_write,
      DenoisePipelineState& EAxBpEB_pipe_write) {
    int lane_predicate = cute::elect_one_sync();
    int warp_idx_in_warpgroup =
        __shfl_sync(0xffffffff,
                    (threadIdx.x / cutlass::NumThreadsPerWarp) %
                        cutlass::NumWarpsPerWarpGroup,
                    srcLane);
    // Issue the epilogue waits
    if (warp_idx_in_warpgroup == 0 && lane_predicate) {
      /* This helps avoid early exit of blocks in Cluster
          * Waits for all stages to either be released (all Consumer UNLOCKs), or if the stage was never used
          * then would just be acquired since the phase was still inverted from make_producer_start_state
          */
      AxEB_pipeline.producer_tail(AxEB_pipe_write);
      EAxBpEB_pipeline.producer_tail(EAxBpEB_pipe_write);
    }
  }

  template <typename FrgTensor>
  CUTLASS_DEVICE void denoise(FrgTensor& tCrD,  // fp32
                              SharedStorage& shared_storage,
                              DenoisePipeline AxEB_pipeline,
                              DenoisePipeline EAxBpEB_pipeline,
                              DenoisePipelineState& AxEB_pipe_read,
                              DenoisePipelineState& EAxBpEB_pipe_read,
                              int const thread_idx) {

    // Divide int tensors by a constant factor of 1 << 12 for fp16 tensor core MMA for denoising.
    // fp16 denoise factors AxEBL, EARxBpEB were already scaled in noising kernels.
    // int32 denoise factors AxEBL, EARxBpEB were already converted and scaled by denoise conversion kernel.
    // fp16 denoise factors EBR, EAL were produced and scaled in noisegen kernel

    // Apply constant scales before accum
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(tCrD); ++i) {
      tCrD(i) /= static_cast<float>(pearl::kIntToFp16ScaleFactor);
    }

    // SMEM tensors for WGMMA (all in FP16)
    // (bM/bN, R)
    Tensor sAxEBL_mma = make_tensor(
        make_smem_ptr(shared_storage.smem_AxEBL.data()), SmemLayoutAxEBL{});
    Tensor sEBR_mma = make_tensor(make_smem_ptr(shared_storage.smem_EBR.data()),
                                  SmemLayoutEBR{});
    Tensor sEAL_mma = make_tensor(make_smem_ptr(shared_storage.smem_EAL.data()),
                                  SmemLayoutEAL{});
    Tensor sEARxBpEB_mma =
        make_tensor(make_smem_ptr(shared_storage.smem_EARxBpEB.data()),
                    SmemLayoutEARxBpEB{});

    Tensor sAxEBL_mma_no_pi =
        as_position_independent_swizzle_tensor(sAxEBL_mma);
    Tensor sEBR_mma_no_pi = as_position_independent_swizzle_tensor(sEBR_mma);
    Tensor sEAL_mma_no_pi = as_position_independent_swizzle_tensor(sEAL_mma);
    Tensor sEARxBpEB_mma_no_pi =
        as_position_independent_swizzle_tensor(sEARxBpEB_mma);

    typename KTraits::TiledMmaDenoise tiled_mma_denoise;
    auto thr_mma_denoise = tiled_mma_denoise.get_slice(thread_idx);

    Tensor tXsAxEBL = thr_mma_denoise.partition_A(
        sAxEBL_mma(_, _, _0{}));  // (MMA, MMA_M, MMA_R)
    Tensor tXsEBR = thr_mma_denoise.partition_B(
        sEBR_mma(_, _, _0{}));  // (MMA, MMA_N, MMA_R)
    Tensor tYsEAL = thr_mma_denoise.partition_A(
        sEAL_mma(_, _, _0{}));  // (MMA, MMA_M, MMA_R)
    Tensor tYsEARxBpEB = thr_mma_denoise.partition_B(
        sEARxBpEB_mma(_, _, _0{}));  // (MMA, MMA_N, MMA_R)

    // Allocate "fragments" -- these are WGMMA matrix descriptors
    Tensor tXrAxEBL =
        thr_mma_denoise.make_fragment_A(tXsAxEBL);  // (MMA, MMA_M, MMA_R)
    Tensor tXrEBR =
        thr_mma_denoise.make_fragment_B(tXsEBR);  // (MMA, MMA_N, MMA_R)
    Tensor tYrEAL =
        thr_mma_denoise.make_fragment_A(tYsEAL);  // (MMA, MMA_M, MMA_R)

    Tensor tYrEARxBpEB =
        thr_mma_denoise.make_fragment_B(tYsEARxBpEB);  // (MMA, MMA_N, MMA_R)

    // Y = -EAL * EARxBpEB
    // Wait for TMA load of EAL, EARxBpEB
    EAxBpEB_pipeline.consumer_wait(EAxBpEB_pipe_read);
    // do GEMM
    // Hopper-only WGMMA sync; no-op on Blackwell consumer (sm_120/121)
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
    warpgroup_fence_operand(tCrD);
    warpgroup_arrive();
#endif
    gemm(tiled_mma_denoise, tYrEAL, tYrEARxBpEB, tCrD);
    // Hopper-only WGMMA sync; no-op on Blackwell consumer (sm_120/121)
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
    warpgroup_commit_batch();
#endif
    // wait for WGMMA to finish before releasing pipeline
    // Hopper-only WGMMA sync; no-op on Blackwell consumer (sm_120/121)
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
    warpgroup_wait<0>();
    warpgroup_fence_operand(tCrD);
#endif
    cutlass::arch::fence_view_async_shared();
    EAxBpEB_pipeline.consumer_release(EAxBpEB_pipe_read);
    ++EAxBpEB_pipe_read;

    // Path 3: signal producer that phase-1 SMEM is consumed; aliased SMEM
    // may now be reused for phase-2 (AxEBL + EBR) TMA loads.
    cutlass::arch::NamedBarrier::arrive(
        kNumMmaThreads + cutlass::NumThreadsPerWarp,
        static_cast<cutlass::arch::ReservedNamedBarriers>(
            pearl::NamedBarriers::DenoisePhase1Consumed));

    // X = -AxEBL * EBR
    // Wait for TMA load of AxEBL, EBR
    AxEB_pipeline.consumer_wait(AxEB_pipe_read);
    // do GEMM
    // Hopper-only WGMMA sync; no-op on Blackwell consumer (sm_120/121)
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
    warpgroup_fence_operand(tCrD);
    warpgroup_arrive();
#endif
    gemm(tiled_mma_denoise, tXrAxEBL, tXrEBR, tCrD);
    // Hopper-only WGMMA sync; no-op on Blackwell consumer (sm_120/121)
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
    warpgroup_commit_batch();
#endif
    // wait for WGMMA to finish before releasing pipeline
    // Hopper-only WGMMA sync; no-op on Blackwell consumer (sm_120/121)
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
    warpgroup_wait<0>();
    warpgroup_fence_operand(tCrD);
#endif
    cutlass::arch::fence_view_async_shared();
    AxEB_pipeline.consumer_release(AxEB_pipe_read);
    ++AxEB_pipe_read;

    // Finally, reverse 2^-12 scaling
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(tCrD); ++i) {
      tCrD(i) *= pearl::kIntToFp16ScaleFactor;
    }
  }

  template <typename FrgTensor, typename TiledMma>
  CUTLASS_DEVICE void scale(
      Params const& epilogue_params,
      FrgTensor& tCrD,  // fp32
      SharedStorage& shared_storage, TiledMma tiled_mma, int thread_idx,
      cute::tuple<int32_t, int32_t, int32_t> const& block_coord) {
    // RMEM -> SMEM COPY
    auto [m_block, n_block, bidb] = block_coord;
    auto [M, N, K, R_] = epilogue_params.problem_shape;
    int const residual_M = M - m_block * bM;
    int const residual_N = N - n_block * bN;

    // Path 3: sC removed; epilogue no longer uses SMEM C buffer.
    Tensor AScales = make_tensor(make_gmem_ptr(epilogue_params.ptr_A_scales),
                                 epilogue_params.layout_A_scales);
    Tensor BScales = make_tensor(make_gmem_ptr(epilogue_params.ptr_B_scales),
                                 epilogue_params.layout_B_scales);

    Tensor gAscales =
        local_tile(AScales, select<0>(TileShape_MNK{}), make_coord(m_block));
    Tensor gBscales =
        local_tile(BScales, select<1>(TileShape_MNK{}), make_coord(n_block));
    auto sAscales = make_tensor(
        make_smem_ptr(shared_storage.smem_scale_a.data()), SmemLayoutScaleA{});
    auto sBscales = make_tensor(
        make_smem_ptr(shared_storage.smem_scale_b.data()), SmemLayoutScaleB{});

    G2SScalesCopyA g2s_scale_copy_a;
    G2SScalesCopyB g2s_scale_copy_b;

    auto g2s_scale_thr_copy_a = g2s_scale_copy_a.get_slice(thread_idx);
    auto g2s_scale_thr_copy_b = g2s_scale_copy_b.get_slice(thread_idx);

    auto tAscalegAscale = g2s_scale_thr_copy_a.partition_S(gAscales);
    auto tAscalesAscale = g2s_scale_thr_copy_a.partition_D(sAscales);

    auto tBscalegBscale = g2s_scale_thr_copy_b.partition_S(gBscales);
    auto tBscalesBscale = g2s_scale_thr_copy_b.partition_D(sBscales);

    if (thread_idx < bM) {
      if constexpr (KTraits::Is_Even_M) {
        cute::copy(g2s_scale_copy_a, tAscalegAscale, tAscalesAscale);
      } else {
        if (thread_idx < residual_M) {
          cute::copy(g2s_scale_copy_a, tAscalegAscale, tAscalesAscale);
        }
      }
    }
    if (thread_idx < bN) {
      if constexpr (KTraits::Is_Even_N) {
        cute::copy(g2s_scale_copy_b, tBscalegBscale, tBscalesBscale);
      } else {
        if (thread_idx < residual_N) {
          cute::copy(g2s_scale_copy_b, tBscalegBscale, tBscalesBscale);
        }
      }
    }
    cp_async_fence();  // always issue fence so that cp_async_wait waits properly
    cp_async_wait<0>();
    cutlass::arch::NamedBarrier::sync(
        kNumMmaThreads, static_cast<cutlass::arch::ReservedNamedBarriers>(
                            pearl::NamedBarriers::LoadScales));

    // Scale
    auto thr_mma = tiled_mma.get_slice(thread_idx);
    Tensor cD = make_identity_tensor(select<0, 1>(TileShape_MNK{}));
    Tensor tCcD = thr_mma.partition_C(cD);
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size<1>(tCrD); ++i) {
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < size<2>(tCrD); ++j) {
        CUTLASS_PRAGMA_UNROLL
        for (int v = 0; v < size<0>(tCrD); ++v) {
          int m_idx = get<0>(tCcD(v, i, j));
          int n_idx = get<1>(tCcD(v, i, j));
          tCrD(v, i, j) *= (sAscales(m_idx) * sBscales(n_idx));
        }
      }
    }

    // cast from float to output type
    Tensor tCrC_out = convert_type<ElementOutput>(tCrD);

    // Path 3 (Blackwell consumer SMEM-fit): bypass the SMEM C buffer and
    // store directly from registers to gmem. Saves 64 KB of SMEM for the
    // canonical 128x256 bf16 tile. Each thread writes its mma.sync C
    // fragment slots predicated by M/N residue. Coalescing relies on the
    // compiler issuing STG.E.{32,64,128} based on the layout of adjacent
    // fragment elements; the SM80 16x8x16 atom packs 2 N-adjacent bf16 per
    // thread so we get at minimum STG.E.32 (2x16-bit) per write.
    Tensor mC_direct = make_tensor(
        make_gmem_ptr(epilogue_params.ptr_C), epilogue_params.layout_C);
    Tensor gC_direct = local_tile(mC_direct, select<0, 1>(TileShape_MNK{}),
                                  make_coord(m_block, n_block));
    Tensor tCgC_direct = thr_mma.partition_C(gC_direct);

    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < size<2>(tCrC_out); ++j) {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size<1>(tCrC_out); ++i) {
        CUTLASS_PRAGMA_UNROLL
        for (int v = 0; v < size<0>(tCrC_out); ++v) {
          int m_idx = get<0>(tCcD(v, i, j));
          int n_idx = get<1>(tCcD(v, i, j));
          if (m_idx < residual_M && n_idx < residual_N) {
            tCgC_direct(v, i, j) = tCrC_out(v, i, j);
          }
        }
      }
    }

  }

  CUTLASS_DEVICE void store(
      Params const& epilogue_params, SharedStorage& shared_storage,
      int thread_idx,
      cute::tuple<int32_t, int32_t, int32_t> const& block_coord) {
    // Path 3 (Blackwell consumer SMEM-fit): direct register->gmem stores
    // happen in scale() now (no SMEM C buffer). This function is reduced
    // to its barrier-arrival + early-return; the SM90_TMA_STORE block
    // below is conditionally compiled out.
    cutlass::arch::NamedBarrier::arrive(
        kNumMmaThreads + cutlass::NumThreadsPerWarp,
        static_cast<cutlass::arch::ReservedNamedBarriers>(
            pearl::NamedBarriers::Epilogue));
    return;
  }

  CUTLASS_DEVICE void store_tail() {
    // Path 3: no TMA store was issued, so no wait needed.
  }
};

}  // namespace pearl
