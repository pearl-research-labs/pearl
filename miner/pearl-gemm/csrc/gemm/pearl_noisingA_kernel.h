#pragma once

#include <variant>

#include "cute/tensor.hpp"

#include <cutlass/arch/arch.h>
#include <cutlass/arch/memory.h>
#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/fast_math.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

#include "cutlass/epilogue/collective/builders/sm90_common.inl"
#include "cutlass/gemm/collective/builders/sm90_common.inl"

#include "pearl_gemm_constants.hpp"
#include "utils.h"

namespace pearl {

using namespace cute;

template <class TileShape_MRK_, int kNumThreads, class Element,
          class ElementDenoise, int kStages, bool IsEvenK, bool NoReduction>
class NoisingKernelA {

 public:
  using ElementScale = float;
  using ElementIndex = uint8_t;
  using ElementAccum = int32_t;
  static_assert(cute::is_same_v<Element, int8_t>);
  static constexpr int denoise_dtype_bits = cute::sizeof_bits_v<ElementDenoise>;
  static_assert(denoise_dtype_bits == 16 || denoise_dtype_bits == 32,
                "Denoise dtype size must be 16 or 32 bits");
  static_assert(denoise_dtype_bits == 32 || NoReduction,
                "Don't support reduction with fp16");

  // Type Aliases
  // Contains the dimensions of the tile that a single CTA works with
  using TileShape_MRK = TileShape_MRK_;
  using ArchTag = cutlass::arch::Sm90;
  static constexpr int kBlockM = get<0>(TileShape_MRK{});  // bM
  static constexpr int R = get<1>(TileShape_MRK{});        // R
  static constexpr int kBlockK = get<2>(TileShape_MRK{});  // bK
  using TileShape_MKR = Shape<Int<kBlockM>, Int<kBlockK>, Int<R>>;

  static constexpr uint32_t kNumThreadsPerWarpGroup = 128;
  static constexpr uint32_t kNumWarpGroups = 3;  // 1 producer, 2 consumers
  static constexpr uint32_t MaxThreadsPerBlock =
      kNumWarpGroups * kNumThreadsPerWarpGroup;
  static constexpr uint32_t MinBlocksPerMultiprocessor = R == 64 ? 2 : 1;

  static constexpr uint32_t kNumAxEBLThreads = kNumThreadsPerWarpGroup;  // 1 WG
  static constexpr uint32_t kNumApEAThreads = kNumThreadsPerWarpGroup;   // 1 WG
  static constexpr uint32_t kNumApEAStoreThreads = 32;  // 1 Warp

  /* Register calculations:
     65535 regs total per SM. We want 2 CTAs/SM here, so <32768 regs/CTA.
     (kNumLoadRegisters + kNumAxEBLRegisters + kNumApEARegisters) * 128 < 32768
     kNumLoadRegisters + kNumAxEBLRegisters + kNumApEARegisters < 256
     Runtime register counts must be:
     - multiples of 8
     - at least 24
     - set across a whole warpgroup.
     All this also has to be true at launch time (before register reallocation),
     at which point register counts are equal for all warps. So,
     3 * kInitialRegisters < 256
     which forces kInitialRegisters <= 80 since it's a multiple of 8, so in fact
     kNumLoadRegisters + kNumAxEBLRegisters + kNumApEARegisters <= 240.
     We found that 24 registers for the load warpgroup caused spilling (probably
     because of the extra index calculations required for the stores), so we used
     the next possible value of 32. The values for the compute warps are then as
     high as possible.
   */
  static constexpr uint32_t kNumLoadRegisters = 32;
  static constexpr uint32_t kNumAxEBLRegisters = 104;
  static constexpr uint32_t kNumApEARegisters = 104;

  static_assert(kBlockM == 64);
  static_assert(R == 64 || R == 128);
  static_assert(kBlockK == 64);  // assumed below

  static constexpr int kClusterM = 1;
  static constexpr int kClusterN = 1;
  using ClusterShape = Shape<Int<kClusterM>, Int<kClusterN>, _1>;

  // Number of smem pipline stages used as a buffer.
  // kStages denoted P for "pipe"
  static constexpr int kStagesOut = kStages;  // denote OP (for "pipe")

  static constexpr int kNumMmaWarpgroups = 1;
  static constexpr int kNumMmaThreads =
      kNumMmaWarpgroups * kNumThreadsPerWarpGroup;
  using AtomLayoutMma = Layout<Shape<Int<kNumMmaWarpgroups>, _1, _1>>;

  /*
  Tiled WGMMA for A * EBL, int8 * int8 -> int32, size (bM, R, bK) for one k_block.
  The op selector will select the appropriate GMEM Atom; it will try to fill 0-mode
  and 1-mode, and tile along 2-mode (which must be 32). So we will get bMxRx32 atom.
  The only supported atom size in the m-mode is 64, so bM%64==0. The warpspecialized
  kernel assigns warpgroup 1 to handle A*EBL, and because we only use one WG the
  atom layout is trivial. Each WGMMA consumes one A tile (bM,bK) and one EBL tile (R,bK).
  The operands are both sourced from SMEM and must have a specific layout; this
  layout is handled by swizzles in the SMEM layouts.
  */
  using TiledMmaMRK = decltype(cute::make_tiled_mma(
      cute::GMMA::ss_op_selector<Element, Element, ElementAccum,
                                 TileShape_MRK>(),
      AtomLayoutMma{}));

  /*
  Tiled WGMMA for A + EAL * EAR, size (bM, bK, R) for one k_block iteration.
  The selected atom will be bMxbKx32. The warpspecialized kernel assigns
  warpgroup 2 to handle A*EBL. Each WGMMA consumes one EAL tile (bM,R) and
  one EAR tile (bK, R). We also set the accumulator to A first to do the A + EA.
  The operands are both sourced from SMEM, but accumulator is in RMEM.
  */
  using TiledMmaMKR = decltype(cute::make_tiled_mma(
      cute::GMMA::ss_op_selector<Element, Element, ElementAccum,
                                 TileShape_MKR>(),
      AtomLayoutMma{}));

  /*
  Smem layouts for the smem tensors, used primarily for TMA and WGMMA. Sizes are
   determined by the tile_shape, all except EAL are pipelined to kStages. ss_smem_selector
   is used to automatically add the neccessary swizzles for the WGMMA/TMA.
  */
  // A: bMxbK (K-major).
  using SmemLayoutAtomA =
      decltype(cutlass::gemm::collective::detail::ss_smem_selector<
               GMMA::Major::K, Element, Int<kBlockM>, Int<kBlockK>>());
  using SmemLayoutA = decltype(tile_to_shape(
      SmemLayoutAtomA{}, Shape<Int<kBlockM>, Int<kBlockK>, Int<kStages>>{}));
  // EBL: bRxbK (K-major)
  using SmemLayoutAtomEBL =
      decltype(cutlass::gemm::collective::detail::ss_smem_selector<
               GMMA::Major::K, Element, Int<R>, Int<kBlockK>>());
  using SmemLayoutEBL = decltype(tile_to_shape(
      SmemLayoutAtomEBL{}, Shape<Int<R>, Int<kBlockK>, Int<kStages>>{}));
  // EAL: bMxbR (R-major, no pipeline)
  using SmemLayoutAtomEAL =
      decltype(cutlass::gemm::collective::detail::ss_smem_selector<
               GMMA::Major::K, Element, Int<kBlockM>, Int<R>>());
  using SmemLayoutEAL = decltype(tile_to_shape(SmemLayoutAtomEAL{},
                                               Shape<Int<kBlockM>, Int<R>>{}));
  // EAR: bKxbR (R-major)
  using SmemLayoutAtomEAR =
      decltype(cutlass::gemm::collective::detail::ss_smem_selector<
               GMMA::Major::K, Element, Int<kBlockK>, Int<R>>());
  using SmemLayoutEAR = decltype(tile_to_shape(
      SmemLayoutAtomEAR{}, Shape<Int<kBlockK>, Int<R>, Int<kStages>>{}));

  // AxEBL: bMxbR (R-major, no pipeline)
  using SmemLayoutAtomAxEBL =
      decltype(cutlass::gemm::collective::detail::ss_smem_selector<
               GMMA::Major::K, ElementDenoise, Int<kBlockM>, Int<R>>());
  using SmemLayoutAxEBL = decltype(tile_to_shape(
      SmemLayoutAtomAxEBL{}, Shape<Int<kBlockM>, Int<R>>{}));
  // ApEA: bMxbK (K-major)
  using SmemLayoutAtomApEA =
      decltype(cutlass::gemm::collective::detail::ss_smem_selector<
               GMMA::Major::K, Element, Int<kBlockM>, Int<kBlockK>>());
  using SmemLayoutApEA = decltype(tile_to_shape(
      SmemLayoutAtomApEA{},
      Shape<Int<kBlockM>, Int<kBlockK>, Int<kStagesOut>>{}));

  // Place holder types for TMA type definitions, correspoding to
  // the GMEM Tensors. Note that the stride assumes majorness along
  // "K-mode" (mode 1)
  using ShapeT = cute::Shape<int32_t, int32_t>;
  using StrideT = cute::Shape<int32_t, _1>;
  using LayoutT = cute::Layout<ShapeT, StrideT>;

  // Creating TMA types for loads
  using TMA_A = decltype(make_tma_copy(
      cute::SM90_TMA_LOAD{},
      make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), ShapeT{},
                  StrideT{}),         // placeholder src GMEM Tensor
      take<0, 2>(SmemLayoutA{}),      // dst shape (stripped stage)
      select<0, 2>(TileShape_MRK{}),  // tiler
      _1{}));                         // no multicast
  using TMA_EBL = decltype(make_tma_copy(
      cute::SM90_TMA_LOAD{},
      make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), ShapeT{},
                  StrideT{}),
      take<0, 2>(SmemLayoutEBL{}), select<1, 2>(TileShape_MRK{}), _1{}));
  using TMA_EAL = decltype(make_tma_copy(
      cute::SM90_TMA_LOAD{},
      make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), ShapeT{},
                  StrideT{}),
      take<0, 2>(SmemLayoutEAL{}), select<0, 1>(TileShape_MRK{}), _1{}));
  using TMA_EAR = decltype(make_tma_copy(
      cute::SM90_TMA_LOAD{},
      make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), ShapeT{},
                  StrideT{}),
      take<0, 2>(SmemLayoutEAR{}), select<2, 1>(TileShape_MRK{}), _1{}));

  // Creating TMA types for stores
  using GmemTiledCopyAxEBL =
      cute::conditional_t<NoReduction, cute::SM90_TMA_STORE,
                          cute::SM90_TMA_REDUCE_ADD>;

  using TMA_AxEBL = decltype(make_tma_copy(
      GmemTiledCopyAxEBL{},
      make_tensor(make_gmem_ptr(static_cast<ElementDenoise const*>(nullptr)),
                  ShapeT{}, StrideT{}),
      SmemLayoutAxEBL{}, select<0, 1>(TileShape_MRK{}), _1{}));
  using TMA_ApEA = decltype(make_tma_copy(
      cute::SM90_TMA_STORE{},
      make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), ShapeT{},
                  StrideT{}),
      take<0, 2>(SmemLayoutApEA{}), select<0, 2>(TileShape_MRK{}), _1{}));

  // Computing the tx_counts for a single TMA load/store, used for mbarrier
  static constexpr uint32_t TmaTransactionBytesA = static_cast<uint32_t>(
      size(take<0, 2>(SmemLayoutA{})) * cutlass::sizeof_bits_v<Element> / 8);
  static constexpr uint32_t TmaTransactionBytesEBL = static_cast<uint32_t>(
      size(take<0, 2>(SmemLayoutEBL{})) * cutlass::sizeof_bits_v<Element> / 8);
  static constexpr uint32_t TmaTransactionBytesEAL = static_cast<uint32_t>(
      size(take<0, 2>(SmemLayoutEAL{})) * cutlass::sizeof_bits_v<Element> / 8);
  static constexpr uint32_t TmaTransactionBytesEAR = static_cast<uint32_t>(
      size(take<0, 2>(SmemLayoutEAR{})) * cutlass::sizeof_bits_v<Element> / 8);
  static constexpr uint32_t TmaTransactionBytesAxEBL =
      static_cast<uint32_t>(size(take<0, 2>(SmemLayoutAxEBL{})) *
                            cutlass::sizeof_bits_v<ElementDenoise> / 8);

  // RMEM->SMEM for AxEBL, StrideT assumes R-major
  using TileShape_MR = decltype(select<0, 2>(TileShape_MKR{}));
  using CopyOpR2S = AutoVectorizingCopyWithAssumedAlignment<128>;
  using SmemCopyAtomAxEBL = Copy_Atom<CopyOpR2S, ElementDenoise>;

  // RMEM<->SMEM for A, used as accumulator in ApEA computation
  // To avoid a lot of hardcoded layouts, we will use a trick
  // where we create a dummy MMA with half the N-mode (K in this case)
  // And load it as uint16 using LDSM. We can get away with this because
  // we then reshuffle the values in the registers to match unit32
  // accumulator.
  using TileShape_MKR_half = Shape<Int<kBlockM>, Int<kBlockK / 2>, Int<R>>;
  using TiledMmaMKR_half = decltype(cute::make_tiled_mma(
      cute::GMMA::ss_op_selector<Element, Element, ElementAccum,
                                 TileShape_MKR_half>(),
      AtomLayoutMma{}));
  using S2RCopyAtomA = Copy_Atom<SM75_U32x4_LDSM_N, uint16_t>;
  using R2SCopyAtomA = Copy_Atom<SM90_U32x4_STSM_N, uint16_t>;

  // Defining load pipeline types (initialized in kernel). Used by A, EBL and EAR
  using MainloopLoadPipeline = typename cutlass::PipelineTmaAsync<kStages>;
  using LoadPipelineParams = typename MainloopLoadPipeline::Params;
  using LoadPipelineState = typename MainloopLoadPipeline::PipelineState;
  using LoadBarrierType = typename MainloopLoadPipeline::ProducerBarrierType;

  // Defining store pipeline, used by ApEA
  using MainloopStorePipeline = typename cutlass::PipelineAsync<kStages>;
  using StorePipelineParams = typename MainloopStorePipeline::Params;
  using StorePipelineState = typename MainloopStorePipeline::PipelineState;
  using StoreBarrierType = typename MainloopStorePipeline::ProducerBarrierType;

  // arrier type to be used by the mbar for EAL
  using ProducerBarType = cutlass::arch::ClusterTransactionBarrier;

  // TMA requires 128B alignment
  static constexpr size_t Alignment = 128;

  struct SharedStorage : cute::aligned_struct<Alignment> {
    // mainloop
    union {
      struct {
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutA>, Alignment>
            smem_A;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutEBL>, Alignment>
            smem_EBL;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutEAL>, Alignment>
            smem_EAL;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutEAR>, Alignment>
            smem_EAR;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutApEA>, Alignment>
            smem_ApEA;
      };

      // epilogue
      cute::array_aligned<ElementDenoise, cute::cosize_v<SmemLayoutAxEBL>,
                          Alignment>
          smem_AxEBL;
    };

    struct {
      typename MainloopLoadPipeline::SharedStorage pipeline_a;
      typename MainloopLoadPipeline::SharedStorage pipeline_ebl;
      typename MainloopLoadPipeline::SharedStorage pipeline_ear;
      typename MainloopStorePipeline::SharedStorage pipeline_apea;
      uint64_t eal_mbar;
    };
  };

  static constexpr int SharedStorageSize = sizeof(SharedStorage);

  // Device side arguments
  struct Arguments {
    // device pointers
    Element const* const ptr_A;
    Element const* const ptr_EAL;
    Element const* const ptr_EAR;
    Element const* const ptr_EBL;
    Element* const ptr_A_out;
    ElementDenoise* const ptr_AxEBL;
    // dimensions
    int m;
    int k;
    int num_k_blocks;  // k blocks per split
    int total_k_blocks;
  };

  // Kernel entry point API
  struct Params {
    // device pointers
    Element const* const ptr_A;
    Element const* const ptr_EAL;
    Element const* const ptr_EAR;
    Element const* const ptr_EBL;
    Element* const ptr_A_out;
    ElementDenoise* const ptr_AxEBL;
    // dimensions
    int m;
    int k;
    int num_k_blocks;  // k blocks per split
    int total_k_blocks;
    // Layouts for GMEM Tensors (needed for TMA partition)
    LayoutT layout_A;
    LayoutT layout_EBL;
    LayoutT layout_EAL;
    LayoutT layout_EAR;
    LayoutT layout_AxEBL;
    LayoutT layout_ApEA;
    // TMAs
    TMA_A tma_load_A;
    TMA_EBL tma_load_EBL;
    TMA_EAL tma_load_EAL;
    TMA_EAR tma_load_EAR;
    TMA_AxEBL tma_store_AxEBL;
    TMA_ApEA tma_store_ApEA;
  };

  enum struct NamedBarriers {
    S2RCopyADone,
    R2SCopyAxEBLDone,
    AxEBLSMEMReady,
  };

  // Convert to underlying arguments
  static Params to_underlying_arguments(Arguments const& args) {

    // Create the GMEM layout and the instantiated TMA objects
    LayoutT layout_A =
        make_layout(make_shape(args.m, args.k), make_stride(args.k, _1{}));
    Tensor mA = make_tensor(make_gmem_ptr(args.ptr_A), layout_A);
    TMA_A tma_load_A =
        make_tma_copy(cute::SM90_TMA_LOAD{}, mA, take<0, 2>(SmemLayoutA{}),
                      select<0, 2>(TileShape_MRK{}), _1{});  // no mcast

    LayoutT layout_EBL =
        make_layout(make_shape(R, args.k), make_stride(args.k, _1{}));
    Tensor mEBL = make_tensor(make_gmem_ptr(args.ptr_EBL), layout_EBL);
    TMA_EBL tma_load_EBL =
        make_tma_copy(cute::SM90_TMA_LOAD{}, mEBL, take<0, 2>(SmemLayoutEBL{}),
                      select<1, 2>(TileShape_MRK{}), _1{});  // no mcast

    LayoutT layout_EAL =
        make_layout(make_shape(args.m, R), make_stride(R, _1{}));
    Tensor mEAL = make_tensor(make_gmem_ptr(args.ptr_EAL), layout_EAL);
    TMA_EAL tma_load_EAL =
        make_tma_copy(cute::SM90_TMA_LOAD{}, mEAL, take<0, 2>(SmemLayoutEAL{}),
                      select<0, 1>(TileShape_MRK{}), _1{});  // no mcast

    LayoutT layout_EAR =
        make_layout(make_shape(args.k, R), make_stride(R, _1{}));
    Tensor mEAR = make_tensor(make_gmem_ptr(args.ptr_EAR), layout_EAR);
    TMA_EAR tma_load_EAR =
        make_tma_copy(cute::SM90_TMA_LOAD{}, mEAR, take<0, 2>(SmemLayoutEAR{}),
                      select<2, 1>(TileShape_MRK{}), _1{});  // no mcast

    LayoutT layout_AxEBL =
        make_layout(make_shape(args.m, R), make_stride(R, _1{}));
    Tensor mAxEBL = make_tensor(make_gmem_ptr(args.ptr_AxEBL), layout_AxEBL);
    TMA_AxEBL tma_store_AxEBL =
        make_tma_copy(GmemTiledCopyAxEBL{}, mAxEBL, SmemLayoutAxEBL{},
                      select<0, 1>(TileShape_MRK{}), _1{});  // no mcast

    LayoutT layout_ApEA =
        make_layout(make_shape(args.m, args.k), make_stride(args.k, _1{}));
    Tensor mApEA = make_tensor(make_gmem_ptr(args.ptr_A_out), layout_ApEA);
    TMA_ApEA tma_store_ApEA = make_tma_copy(
        cute::SM90_TMA_STORE{}, mApEA, take<0, 2>(SmemLayoutApEA{}),
        select<0, 2>(TileShape_MRK{}), _1{});  // no mcast
    return {.ptr_A = args.ptr_A,
            .ptr_EAL = args.ptr_EAL,
            .ptr_EAR = args.ptr_EAR,
            .ptr_EBL = args.ptr_EBL,
            .ptr_A_out = args.ptr_A_out,
            .ptr_AxEBL = args.ptr_AxEBL,
            .m = args.m,
            .k = args.k,
            .num_k_blocks = args.num_k_blocks,
            .total_k_blocks = args.total_k_blocks,
            .layout_A = layout_A,
            .layout_EBL = layout_EBL,
            .layout_EAL = layout_EAL,
            .layout_EAR = layout_EAR,
            .layout_AxEBL = layout_AxEBL,
            .layout_ApEA = layout_ApEA,
            .tma_load_A = tma_load_A,
            .tma_load_EBL = tma_load_EBL,
            .tma_load_EAL = tma_load_EAL,
            .tma_load_EAR = tma_load_EAR,
            .tma_store_AxEBL = tma_store_AxEBL,
            .tma_store_ApEA = tma_store_ApEA};
  }

  // Computes the kernel launch grid shape based on runtime parameters
  static dim3 get_grid_shape(Params const& params) {
    if constexpr (NoReduction) {
      return dim3(ceil_div(params.m, kBlockM), 1, 1);
    } else {
      return dim3(ceil_div(params.m, kBlockM),
                  ceil_div(params.k, params.num_k_blocks * kBlockK), 1);
    }
  }

  static dim3 get_block_shape() { return dim3(MaxThreadsPerBlock, 1, 1); }

  CUTLASS_DEVICE void load(Params const& params,
                           MainloopLoadPipeline pipeline_a,
                           MainloopLoadPipeline pipeline_ebl,
                           MainloopLoadPipeline pipeline_ear,
                           LoadPipelineState& smem_pipe_write_a,
                           LoadPipelineState& smem_pipe_write_ebl,
                           LoadPipelineState& smem_pipe_write_ear,
                           SharedStorage& shared_storage, const int m_block,
                           const int k_block_min, const int k_block_max) {
    // smem tensors
    Tensor sA = make_tensor(make_smem_ptr(shared_storage.smem_A.data()),
                            SmemLayoutA{});  // (bM, bK, P)
    Tensor sEBL = make_tensor(make_smem_ptr(shared_storage.smem_EBL.data()),
                              SmemLayoutEBL{});  // (R, bK, P)
    Tensor sEAL = make_tensor(make_smem_ptr(shared_storage.smem_EAL.data()),
                              SmemLayoutEAL{});  // (bM, R)
    Tensor sEAR = make_tensor(make_smem_ptr(shared_storage.smem_EAR.data()),
                              SmemLayoutEAR{});  // (bK, R, P)

    // GMEM tensors (conains pointer to data), retrieved from tma objects
    Tensor mA = params.tma_load_A.get_tma_tensor(params.layout_A.shape());
    Tensor mEBL = params.tma_load_EBL.get_tma_tensor(params.layout_EBL.shape());
    Tensor mEAR = params.tma_load_EAR.get_tma_tensor(params.layout_EAR.shape());
    Tensor mEAL = params.tma_load_EAL.get_tma_tensor(params.layout_EAL.shape());

    // CTA local partition of GMEM tensors
    Tensor gA =
        local_tile(mA, select<0, 2>(TileShape_MRK{}), make_coord(m_block, _));
    Tensor gEBL =
        local_tile(mEBL, select<1, 2>(TileShape_MRK{}), make_coord(_0{}, _));
    Tensor gEAR =
        local_tile(mEAR, select<2, 1>(TileShape_MRK{}), make_coord(_, _0{}));
    Tensor gEAL = local_tile(mEAL, select<0, 1>(TileShape_MRK{}),
                             make_coord(m_block, _0{}));

    // Create the TMA partitioned src/dst tensors
    auto [tAgA, tAsA] =
        tma_partition(params.tma_load_A, Int<0>{}, Layout<_1>{},
                      group_modes<0, 2>(sA), group_modes<0, 2>(gA));
    auto [tEBLgEBL, tEBLsEBL] =
        tma_partition(params.tma_load_EBL, Int<0>{}, Layout<_1>{},
                      group_modes<0, 2>(sEBL), group_modes<0, 2>(gEBL));
    auto [tEALgEAL, tEALsEAL] =
        tma_partition(params.tma_load_EAL, Int<0>{}, Layout<_1>{},
                      group_modes<0, 2>(sEAL), group_modes<0, 2>(gEAL));
    auto [tEARgEAR, tEARsEAR] =
        tma_partition(params.tma_load_EAR, Int<0>{}, Layout<_1>{},
                      group_modes<0, 2>(sEAR), group_modes<0, 2>(gEAR));

    if (cute::elect_one_sync()) {
      uint64_t* eal_mbar = &shared_storage.eal_mbar;
      ProducerBarType::arrive_and_expect_tx(eal_mbar, TmaTransactionBytesEAL);
      copy(params.tma_load_EAL.with(*eal_mbar), tEALgEAL, tEALsEAL);

      CUTLASS_PRAGMA_NO_UNROLL
      for (int k_block = k_block_min; k_block < k_block_max; ++k_block) {
        // Load A, EBL, EAR. They are on separate pipelines to maximize overlap
        pipeline_ear.producer_acquire(smem_pipe_write_ear);
        LoadBarrierType* tmaBarEAR =
            pipeline_ear.producer_get_barrier(smem_pipe_write_ear);
        copy(params.tma_load_EAR.with(*tmaBarEAR, 0), tEARgEAR(_, k_block),
             tEARsEAR(_, smem_pipe_write_ear.index()));
        pipeline_ear.producer_commit(smem_pipe_write_ear,
                                     TmaTransactionBytesEAR);
        ++smem_pipe_write_ear;
        pipeline_a.producer_acquire(smem_pipe_write_a);
        LoadBarrierType* tmaBarA =
            pipeline_a.producer_get_barrier(smem_pipe_write_a);
        copy(params.tma_load_A.with(*tmaBarA, 0), tAgA(_, k_block),
             tAsA(_, smem_pipe_write_a.index()));
        pipeline_a.producer_commit(smem_pipe_write_a, TmaTransactionBytesA);
        ++smem_pipe_write_a;

        pipeline_ebl.producer_acquire(smem_pipe_write_ebl);
        LoadBarrierType* tmaBarEBL =
            pipeline_ebl.producer_get_barrier(smem_pipe_write_ebl);
        copy(params.tma_load_EBL.with(*tmaBarEBL, 0), tEBLgEBL(_, k_block),
             tEBLsEBL(_, smem_pipe_write_ebl.index()));
        pipeline_ebl.producer_commit(smem_pipe_write_ebl,
                                     TmaTransactionBytesEBL);
        ++smem_pipe_write_ebl;
      }
    }
  }

  template <typename FrgTensorC>
  CUTLASS_DEVICE void compute_AxEBL(Params const& params,
                                    MainloopLoadPipeline pipeline_a,
                                    MainloopLoadPipeline pipeline_ebl,
                                    SharedStorage& shared_storage,
                                    FrgTensorC& tCrAxEBL, const int m_block,
                                    const int k_block_min,
                                    const int k_block_max, const int tid) {
    // smem tensors
    Tensor sA = make_tensor(make_smem_ptr(shared_storage.smem_A.data()),
                            SmemLayoutA{});  // (bM, bK, P)
    Tensor sEBL = make_tensor(make_smem_ptr(shared_storage.smem_EBL.data()),
                              SmemLayoutEBL{});  // (R, bK, P)

    // 64xRx32 MMA atom
    TiledMmaMRK tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tid);

    Tensor tCsA = thr_mma.partition_A(sA);      // (MMA,MMA_M,MMA_K,P)
    Tensor tCsEBL = thr_mma.partition_B(sEBL);  // (MMA,MMA_N,MMA_K,P)

    // Allocate "fragments" -- these are WGMMA matrix descriptors
    Tensor tCrA = thr_mma.make_fragment_A(tCsA);      // (MMA,MMA_M,MMA_K,P)
    Tensor tCrEBL = thr_mma.make_fragment_B(tCsEBL);  // (MMA,MMA_N,MMA_K,P)

    LoadPipelineState smem_pipe_read_a;
    LoadPipelineState smem_pipe_read_ebl;

    CUTLASS_PRAGMA_NO_UNROLL
    for (int k_block = k_block_min; k_block < k_block_max; ++k_block) {
      pipeline_a.consumer_wait(smem_pipe_read_a);
      pipeline_ebl.consumer_wait(smem_pipe_read_ebl);

      warpgroup_fence_operand(tCrAxEBL);
      warpgroup_arrive();
      // WGMMA with dispatch mode (V,M,K) x (V,N,K) => (V,M,N)
      gemm(tiled_mma, tCrA(_, _, _, smem_pipe_read_a.index()),
           tCrEBL(_, _, _, smem_pipe_read_ebl.index()), tCrAxEBL);
      warpgroup_commit_batch();
      // Wait for all MMAs across the warp group to complete
      warpgroup_wait<0>();

      warpgroup_fence_operand(tCrAxEBL);

      cutlass::arch::fence_view_async_shared();
      pipeline_a.consumer_release(smem_pipe_read_a);
      pipeline_ebl.consumer_release(smem_pipe_read_ebl);
      ++smem_pipe_read_a;
      ++smem_pipe_read_ebl;
    }
  }

  CUTLASS_DEVICE void compute_ApEA(Params const& params,
                                   MainloopLoadPipeline pipeline_a,
                                   MainloopLoadPipeline pipeline_ear,
                                   MainloopStorePipeline pipeline_apea,
                                   SharedStorage& shared_storage,
                                   const int m_block, const int k_block_min,
                                   const int k_block_max, const int tid) {
    // (M, K)
    Tensor mA_out =
        make_tensor(make_gmem_ptr(params.ptr_A_out), params.layout_A);
    // (bM, bK, k_tiles)
    Tensor gA_out = local_tile(mA_out, select<0, 1>(TileShape_MKR{}),
                               make_coord(m_block, _));
    // (bM, bK, P)
    Tensor sA =
        make_tensor(make_smem_ptr(shared_storage.smem_A.data()), SmemLayoutA{});
    Tensor sA_pi = as_position_independent_swizzle_tensor(sA);
    // (bM, R)
    Tensor sEAL = make_tensor(make_smem_ptr(shared_storage.smem_EAL.data()),
                              SmemLayoutEAL{});
    // (bK, R, P)
    Tensor sEAR = make_tensor(make_smem_ptr(shared_storage.smem_EAR.data()),
                              SmemLayoutEAR{});
    // (bM, bK, OP)
    Tensor sApEA = make_tensor(make_smem_ptr(shared_storage.smem_ApEA.data()),
                               SmemLayoutApEA{});

    // 64xbKx32 MMA atom
    TiledMmaMKR tiled_mma;
    auto thr_mma = tiled_mma.get_slice(tid);

    // (ATOM, REST_M, REST_K)
    Tensor tCrApEA =
        partition_fragment_C(tiled_mma, select<0, 1>(TileShape_MKR{}));
    Tensor tCrApEA_int8 = make_fragment_like<Element>(tCrApEA);

    // (ATOM, REST_M, REST_R)
    Tensor tCsEAL = thr_mma.partition_A(sEAL);
    // (1, REST_M, REST_R) -- tensor of WGMMA descriptors, 1 per atom
    Tensor tCrEAL = thr_mma.make_fragment_A(tCsEAL);

    // (ATOM, REST_K, REST_R)
    Tensor tCsEAR = thr_mma.partition_B(sEAR);
    // (1, REST_K, REST_R) -- tensor of WGMMA descriptors, 1 per atom
    Tensor tCrEAR = thr_mma.make_fragment_B(tCsEAR);

    // Wait for EAL load. It does not change with K, so only once
    uint64_t* eal_mbar = &shared_storage.eal_mbar;
    ProducerBarType::wait(eal_mbar, 0);

    // S2R copy for A to add to the accumulator
    TiledMmaMKR_half tiled_mma_half;
    auto s2r_tiled_copy_A = make_tiled_copy_C(S2RCopyAtomA{}, tiled_mma_half);
    auto s2r_thr_copy_A = s2r_tiled_copy_A.get_slice(tid);
    auto sA_u16 = recast<uint16_t>(sA_pi);
    auto taccCsA = s2r_thr_copy_A.partition_S(sA_u16);

    auto tCrA = partition_fragment_C(tiled_mma_half,
                                     select<0, 1>(TileShape_MKR_half{}));
    auto tCrA_u16 = make_tensor_like<uint16_t>(tCrA);  // (MMA_V, MMA_M, MM_N)
    // ((ATOM_V, COPY_V), COPY_M, COPY_N)
    auto taccCrA = s2r_thr_copy_A.retile_D(tCrA_u16);
    auto taccCrA_int8 = recast<Element>(taccCrA);

    // R2S copy for A, because stsm and ldsm has the same shape, we don't
    // need to retile the register
    auto r2s_tiled_copy_A = make_tiled_copy_C(R2SCopyAtomA{}, tiled_mma_half);
    auto r2s_thr_copy_A = r2s_tiled_copy_A.get_slice(tid);
    auto sApEA_u16 = recast<uint16_t>(sApEA);
    auto taccCsApEA = s2r_thr_copy_A.partition_S(sApEA_u16);

    LoadPipelineState smem_pipe_read_a;
    LoadPipelineState smem_pipe_read_ear;
    StorePipelineState smem_pipe_write_apea =
        cutlass::make_producer_start_state<MainloopStorePipeline>();

    CUTLASS_PRAGMA_NO_UNROLL
    for (int k_block = k_block_min; k_block < k_block_max; ++k_block) {
      // WGMMA: A + EAL * EAR
      clear(tCrApEA);  // We store every iter so we need to clear
      pipeline_ear.consumer_wait(smem_pipe_read_ear);
      warpgroup_fence_operand(tCrApEA);
      warpgroup_arrive();
      gemm(tiled_mma, tCrEAL, tCrEAR(_, _, _, smem_pipe_read_ear.index()),
           tCrApEA);
      warpgroup_commit_batch();
      warpgroup_wait<0>();
      warpgroup_fence_operand(tCrApEA);
      pipeline_ear.consumer_release(smem_pipe_read_ear);
      ++smem_pipe_read_ear;
      // Convert down from int32 accumulator to int8
      pearl::convert_type_out(tCrApEA, tCrApEA_int8);

      // Shuffle to allow for bank conflict free store and load
      permute_Aregs_fp8(tCrApEA_int8);

      // Load A to registers
      pipeline_a.consumer_wait(smem_pipe_read_a);
      cute::copy(s2r_tiled_copy_A, taccCsA(_, _, _, smem_pipe_read_a.index()),
                 taccCrA);
      cutlass::arch::NamedBarrier::sync(
          kNumApEAThreads, static_cast<uint32_t>(NamedBarriers::S2RCopyADone));
      cutlass::arch::fence_view_async_shared();
      pipeline_a.consumer_release(smem_pipe_read_a);
      ++smem_pipe_read_a;

      // Add A to EA
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(taccCrA_int8); ++i) {
        taccCrA_int8[i] += tCrApEA_int8[i];
      }

      // RMEM->SMEM copy for ApEA to "stage" it for TMA_STORE
      pipeline_apea.producer_acquire(smem_pipe_write_apea);
      cute::copy(r2s_tiled_copy_A, taccCrA,
                 taccCsApEA(_, _, _, smem_pipe_write_apea.index()));
      cutlass::arch::fence_view_async_shared();
      pipeline_apea.producer_commit(smem_pipe_write_apea);
      ++smem_pipe_write_apea;
    }
  }

  template <typename FrgTensorC>
  CUTLASS_DEVICE void store_AxEBL(Params const& params, FrgTensorC& tCrAxEBL,
                                  SharedStorage& shared_storage,
                                  const int m_block, const int tid) {
    // SMEM Tensor
    Tensor sAxEBL = make_tensor(make_smem_ptr(shared_storage.smem_AxEBL.data()),
                                SmemLayoutAxEBL{});  // (bM, R, P)
    Tensor sAxEBL_pi = as_position_independent_swizzle_tensor(sAxEBL);

    // RMEM -> SMEM copy
    TiledMmaMRK tiled_mma;
    auto r2s_tiled_copy_O = make_tiled_copy_C(SmemCopyAtomAxEBL{}, tiled_mma);
    auto r2s_thr_copy_O = r2s_tiled_copy_O.get_thread_slice(tid);
    Tensor tR2SsAxEBL = r2s_thr_copy_O.partition_D(sAxEBL_pi);

    // TMA out
    Tensor mAxEBL =
        params.tma_store_AxEBL.get_tma_tensor(params.layout_AxEBL.shape());
    Tensor gAxEBL = local_tile(mAxEBL, select<0, 1>(TileShape_MRK{}),
                               make_coord(m_block, _0{}));  // (M, R)
    auto tma_thr_AxEBL = params.tma_store_AxEBL.get_slice(_0{});
    Tensor tOgAxEBL = tma_thr_AxEBL.partition_D(gAxEBL);  // (TMA, TMA_M, TMA_R)
    Tensor tOsAxEBL = tma_thr_AxEBL.partition_S(sAxEBL);  // (TMA, TMA_M, TMA_R)

    // RMEM -> SMEM tiled copy, scale if fp16
    if constexpr (!cute::is_same_v<ElementDenoise, int>) {
      Tensor tCrAxEBL_out = make_fragment_like<ElementDenoise>(tCrAxEBL);
      // Convert to output type
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(tCrAxEBL); ++i) {
        tCrAxEBL_out(i) = static_cast<ElementDenoise>(
            static_cast<ElementScale>(tCrAxEBL(i)) /
            static_cast<ElementScale>(pearl::kAxEBLScaleFactor));
      }
      Tensor tR2SrAxEBL = r2s_thr_copy_O.retile_S(
          tCrAxEBL_out);  // ((Atom,AtomNum), MMA_M, MMA_R)
      cute::copy(r2s_tiled_copy_O, tR2SrAxEBL, tR2SsAxEBL);
    } else {
      Tensor tR2SrAxEBL =
          r2s_thr_copy_O.retile_S(tCrAxEBL);  // ((Atom,AtomNum), MMA_M, MMA_R)
      cute::copy(r2s_tiled_copy_O, tR2SrAxEBL, tR2SsAxEBL);
    }

    // ensure smem writes are completed and visible to TMA
    cutlass::arch::NamedBarrier::sync(
        kNumAxEBLThreads,
        static_cast<uint32_t>(NamedBarriers::R2SCopyAxEBLDone));
    cutlass::arch::fence_view_async_shared();
    // SMEM -> GMEM TMA copy
    int warp_idx_in_warpgroup = pearl::warp_idx_in_warpgroup_sync();
    if (warp_idx_in_warpgroup == 0 && cute::elect_one_sync()) {  // Load warp
      cute::copy(params.tma_store_AxEBL, tOsAxEBL, tOgAxEBL);
      tma_store_arrive();
      tma_store_wait<0>();
    }
  }

  CUTLASS_DEVICE void store_ApEA(Params const& params,
                                 MainloopStorePipeline pipeline_apea,
                                 SharedStorage& shared_storage,
                                 const int m_block, const int k_block_min,
                                 const int k_block_max) {

    // (bM, bK, OP)
    Tensor sApEA = make_tensor(make_smem_ptr(shared_storage.smem_ApEA.data()),
                               SmemLayoutApEA{});

    // TMA out
    Tensor mApEA = params.tma_store_ApEA.get_tma_tensor(
        params.layout_ApEA.shape());  // (M, K)
    // (bM, bK, k_tiles)
    Tensor gApEA = local_tile(mApEA, select<0, 2>(TileShape_MRK{}),
                              make_coord(m_block, _));
    auto tma_thr_ApEA = params.tma_store_ApEA.get_slice(_0{});
    // (TMA, k_tiles)
    Tensor tOgApEA = group_modes<0, 3>(tma_thr_ApEA.partition_D(gApEA));
    // (TMA, OP)
    Tensor tOsApEA = group_modes<0, 3>(tma_thr_ApEA.partition_S(sApEA));

    StorePipelineState smem_pipe_read_apea;
    StorePipelineState smem_pipe_release_apea;

    if (cute::elect_one_sync()) {
      constexpr int kPrologueCount = kStagesOut - 1;
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kPrologueCount; ++i) {
        int k_block = k_block_min + i;
        if (k_block < k_block_max) {
          pipeline_apea.consumer_wait(smem_pipe_read_apea);
          cutlass::arch::fence_view_async_shared();
          copy(params.tma_store_ApEA, tOsApEA(_, smem_pipe_read_apea.index()),
               tOgApEA(_, k_block));
          tma_store_arrive();
          ++smem_pipe_read_apea;
        }
      }

      CUTLASS_PRAGMA_NO_UNROLL
      for (int k_block = k_block_min + kPrologueCount; k_block < k_block_max;
           ++k_block) {
        pipeline_apea.consumer_wait(smem_pipe_read_apea);
        cutlass::arch::fence_view_async_shared();
        copy(params.tma_store_ApEA, tOsApEA(_, smem_pipe_read_apea.index()),
             tOgApEA(_, k_block));
        tma_store_arrive();
        ++smem_pipe_read_apea;

        // wait on prior completion
        tma_store_wait<kPrologueCount>();
        cutlass::arch::fence_view_async_shared();
        pipeline_apea.consumer_release(smem_pipe_release_apea);
        ++smem_pipe_release_apea;
      }
      tma_store_wait<0>();
    }
  }

  CUTLASS_DEVICE
  void operator()(Params const& params, char* smem_buf) {
    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);

    int const tid = threadIdx.x;
    int const warp_idx = cutlass::canonical_warp_idx_sync();
    int const warp_group_idx = cutlass::canonical_warp_group_idx();
    int const warp_group_thread_idx = tid % cutlass::NumThreadsPerWarpGroup;
    int const lane_predicate = cute::elect_one_sync();

    // Issue Tma Descriptor Prefetch from a single thread
    if (warp_idx == 0 && lane_predicate) {
      cute::prefetch_tma_descriptor(params.tma_load_A.get_tma_descriptor());
      cute::prefetch_tma_descriptor(params.tma_load_EBL.get_tma_descriptor());
      cute::prefetch_tma_descriptor(params.tma_load_EAL.get_tma_descriptor());
      cute::prefetch_tma_descriptor(params.tma_load_EAR.get_tma_descriptor());
      cute::prefetch_tma_descriptor(
          params.tma_store_AxEBL.get_tma_descriptor());
      cute::prefetch_tma_descriptor(params.tma_store_ApEA.get_tma_descriptor());
    }
    // The matrix A is logically partitioned into m/kblockM tiles in the m direction.
    //  m_block is the index of the tile that this CTA will work with
    int const m_block = blockIdx.x;
    // Each CTA will work with several tiles in the k direction, one in each mainloop step.
    // This is the smallest index of these blocks. If NoReduction,
    //  always equals 0 (each CTA will work on all blocks in the k direction).
    int const k_block_min = blockIdx.y * params.num_k_blocks;

    // In case num_k_blocks doens't divide total_k_blocks
    int const num_k_blocks_cta =
        cute::min(params.num_k_blocks, params.total_k_blocks - k_block_min);
    // Largest k_block index this CTA will work with
    int const k_block_max = k_block_min + num_k_blocks_cta;

    // There's 1 load+wait of EAL per CTA, so we just use an mbarrier, not a pipeline
    uint64_t* eal_mbar = &shared_storage.eal_mbar;
    if (lane_predicate) {
      ProducerBarType::init(eal_mbar, 1);
    }
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    // mbarrier-init fence is Hopper sm_90+ only. On Ada/sm_89 the fat-binary
    // dispatch never reaches this kernel (template branch in
    // pearl_gemm_launch_template.h routes to NoisingKernelASm89), but guarding
    // here keeps the kernel safe if it is ever compiled into a multi-arch
    // binary and accidentally launched on a pre-Hopper target.
    cutlass::arch::fence_barrier_init();
#endif

    // TMA load pipeline for A: WG0 is producer, and both WG1 and WG2 are consumers
    LoadPipelineParams pipeline_params_a;
    pipeline_params_a.transaction_bytes = TmaTransactionBytesA;
    pipeline_params_a.role =
        warp_group_idx == 0 ? MainloopLoadPipeline::ThreadCategory::Producer
                            : MainloopLoadPipeline::ThreadCategory::Consumer;
    pipeline_params_a.is_leader = warp_group_thread_idx == 0;
    pipeline_params_a.num_consumers = kNumMmaThreads * 2;  // Both WGs release A
    // We're counting on pipeline to call cutlass::arch::fence_barrier_init();
    MainloopLoadPipeline pipeline_a(shared_storage.pipeline_a,
                                    pipeline_params_a, ClusterShape{});

    // TMA load pipeline for EBL: WG0 is producer, WG1 is consumer
    LoadPipelineParams pipeline_params_ebl;
    pipeline_params_ebl.transaction_bytes = TmaTransactionBytesEBL;
    pipeline_params_ebl.role =
        warp_group_idx == 0 ? MainloopLoadPipeline::ThreadCategory::Producer
        : warp_group_idx == 1
            ? MainloopLoadPipeline::ThreadCategory::Consumer
            : MainloopLoadPipeline::ThreadCategory::NonParticipant;
    pipeline_params_ebl.is_leader = warp_group_thread_idx == 0;
    pipeline_params_ebl.num_consumers = kNumMmaThreads;
    MainloopLoadPipeline pipeline_ebl(shared_storage.pipeline_ebl,
                                      pipeline_params_ebl, ClusterShape{});

    // TMA load pipeline for EAR: WG0 is producer, WG2 is consumer
    LoadPipelineParams pipeline_params_ear;
    pipeline_params_ear.transaction_bytes = TmaTransactionBytesEAR;
    pipeline_params_ear.role =
        warp_group_idx == 0 ? MainloopLoadPipeline::ThreadCategory::Producer
        : warp_group_idx == 2
            ? MainloopLoadPipeline::ThreadCategory::Consumer
            : MainloopLoadPipeline::ThreadCategory::NonParticipant;
    pipeline_params_ear.is_leader = warp_group_thread_idx == 0;
    pipeline_params_ear.num_consumers = kNumMmaThreads;
    MainloopLoadPipeline pipeline_ear(shared_storage.pipeline_ear,
                                      pipeline_params_ear, ClusterShape{});

    StorePipelineParams pipeline_params_apea;
    // Producer is compute threads, consumer is TMA store warp
    pipeline_params_apea.role =
        warp_group_idx == 2 ? MainloopStorePipeline::ThreadCategory::Producer
        : warp_idx == 1     ? MainloopStorePipeline::ThreadCategory::Consumer
                        : MainloopStorePipeline::ThreadCategory::NonParticipant;
    pipeline_params_apea.producer_arv_count = kNumMmaThreads;  // one WG
    pipeline_params_apea.consumer_arv_count = 1;  // one thread for TMA store
    MainloopStorePipeline pipeline_apea(shared_storage.pipeline_apea,
                                        pipeline_params_apea);

    // Sync to ensure pipeline barrier init is visible across threads
    __syncthreads();

    if (warp_group_idx == 0) {  // Producer
      cutlass::arch::warpgroup_reg_dealloc<kNumLoadRegisters>();

      int warp_idx_in_warpgroup = pearl::warp_idx_in_warpgroup_sync();
      if (warp_idx_in_warpgroup == 0) {  // Load warp
        LoadPipelineState smem_pipe_write_a =
            cutlass::make_producer_start_state<MainloopLoadPipeline>();
        LoadPipelineState smem_pipe_write_ebl =
            cutlass::make_producer_start_state<MainloopLoadPipeline>();
        LoadPipelineState smem_pipe_write_ear =
            cutlass::make_producer_start_state<MainloopLoadPipeline>();
        load(params, pipeline_a, pipeline_ebl, pipeline_ear, smem_pipe_write_a,
             smem_pipe_write_ebl, smem_pipe_write_ear, shared_storage, m_block,
             k_block_min, k_block_max);
      } else if (warp_idx_in_warpgroup == 1) {  // Store ApEA warp
        store_ApEA(params, pipeline_apea, shared_storage, m_block, k_block_min,
                   k_block_max);
        cutlass::arch::NamedBarrier::arrive(
            kNumAxEBLThreads + kNumApEAStoreThreads,
            static_cast<uint32_t>(NamedBarriers::AxEBLSMEMReady));
      }
    } else if (warp_group_idx == 1) {  // A * EBL
      cutlass::arch::warpgroup_reg_alloc<kNumAxEBLRegisters>();
      TiledMmaMRK tiled_mma;
      // (ATOM, REST_M, REST_R)
      Tensor tCrAxEBL =
          partition_fragment_C(tiled_mma, select<0, 1>(TileShape_MRK{}));
      clear(tCrAxEBL);

      constexpr int ThreadOffset = kNumThreadsPerWarpGroup;
      compute_AxEBL(params, pipeline_a, pipeline_ebl, shared_storage, tCrAxEBL,
                    m_block, k_block_min, k_block_max, tid - ThreadOffset);

      // AxEBL smem is overlapped via union, so we need to make sure
      //  it is free. We check this by waiting for the last ApEA out.
      cutlass::arch::NamedBarrier::sync(
          kNumAxEBLThreads + kNumApEAStoreThreads,
          static_cast<uint32_t>(NamedBarriers::AxEBLSMEMReady));
      // "Epilogue" to store AxEBL
      store_AxEBL(params, tCrAxEBL, shared_storage, m_block,
                  tid - ThreadOffset);
    } else if (warp_group_idx == 2) {  // A + EAL * EAR
      cutlass::arch::warpgroup_reg_alloc<kNumApEARegisters>();

      constexpr int ThreadOffset = 2 * kNumThreadsPerWarpGroup;
      compute_ApEA(params, pipeline_a, pipeline_ear, pipeline_apea,
                   shared_storage, m_block, k_block_min, k_block_max,
                   tid - ThreadOffset);
    }
  }
};

}  // namespace pearl
