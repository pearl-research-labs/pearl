#pragma once

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

#include "named_barrier.hpp"
#include "pearl_gemm_constants.hpp"
#include "utils.h"

namespace pearl {

using namespace cute;

template <class TileShape_NRK_, int kNumThreads, class Element,
          class ElementDenoise, int kStages, bool IsEvenK, bool NoReduction>
class NoisingKernelB {

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
  using TileShape_NRK = TileShape_NRK_;
  using ArchTag = cutlass::arch::Sm90;
  static constexpr int kBlockN = get<0>(TileShape_NRK{});  // bN
  static constexpr int R = get<1>(TileShape_NRK{});        // R
  static constexpr int kBlockK = get<2>(TileShape_NRK{});  // bK
  using TileShape_NKR = Shape<Int<kBlockN>, Int<kBlockK>, Int<R>>;

  static constexpr uint32_t kNumThreadsPerWarpGroup = 128;

  // 1 producer WG + 1 EARxBpEB consumer WG + 1 BpEB consumer WG
  static constexpr uint32_t kActiveWarpGroups = 3;
  static constexpr uint32_t MaxThreadsPerBlock =
      kActiveWarpGroups * kNumThreadsPerWarpGroup;
  static constexpr uint32_t MinBlocksPerMultiprocessor = R == 64 ? 2 : 1;

  static constexpr uint32_t kNumEARxBpEBThreads =
      kNumThreadsPerWarpGroup;                                          // 1 WG
  static constexpr uint32_t kNumBpEBThreads = kNumThreadsPerWarpGroup;  // 1 WG
  static constexpr uint32_t kNumBpEBStoreThreads = 32;  // 1 Warp

  /* Register calculations:
     65536 regs total per SM. We want 2 CTAs/SM here, so <32768 regs/CTA.
     (kNumLoadRegisters + kNumEARxBpEBRegisters + kNumBpEBRegisters) * 128 < 32768
     kNumLoadRegisters + kNumEARxBpEBRegisters + kNumBpEBRegisters < 256
     Runtime register counts must be:
     - multiples of 8
     - at least 24
     - set across a whole warpgroup.
     All this also has to be true at launch time (before register reallocation),
     at which point register counts are equal for all warps. So,
     3 * kInitialRegisters < 256
     which forces kInitialRegisters <= 80 since it's a multiple of 8, so in fact
     kNumLoadRegisters + kNumEARxBpEBRegisters + kNumBpEBRegisters <= 240.
     We found that 24 registers for the load warpgroup caused spilling (probably
     because of the extra index calculations required for the stores), so we used
     the next possible value of 32. The values for the compute warps are then as
     high as possible.
   */
  static constexpr uint32_t kNumLoadRegisters = 32;
  static constexpr uint32_t kNumEARxBpEBRegisters = 104;
  static constexpr uint32_t kNumBpEBRegisters = 104;

  static_assert(kBlockN == 64);
  static_assert(R == 64 || R == 128);
  static_assert(kBlockK == 64);  // assumed below

  static constexpr int kClusterM = 1;
  static constexpr int kClusterN = 1;
  using ClusterShape = Shape<Int<kClusterM>, Int<kClusterN>, _1>;

  // Number of smem pipeline stages used as a buffer.
  // pipeline stages denoted P (for "pipe")
  static constexpr int kStagesOut = kStages;  // denote OP (for "pipe")

  static constexpr int kNumMmaWarpgroups = 1;
  static constexpr int kNumMmaThreads =
      kNumMmaWarpgroups * kNumThreadsPerWarpGroup;
  // Blackwell port: SM80 mma.sync 16x8x32 int8 atom tiled as 4 warps per
  // warpgroup (matching Hopper's CLayout_64xN per-thread fragment ownership
  // exactly — same row groups per warp, same value packing). See
  // kernel_traits.hpp for the bit-identicality argument.
  static constexpr int kWarpsPerWarpGroupMma = 4;
  using AtomLayoutMma = Layout<Shape<Int<kWarpsPerWarpGroupMma * kNumMmaWarpgroups>, _1, _1>>;

  /*
  Tiled WGMMA for EAR * BpEB, int8 * int8 -> int32, size (bN, R, bK) for one k_block.
  The op selector will select the appropriate GMEM Atom; it will try to fill 0-mode
  and 1-mode, and tile along 2-mode (which must be 32). So we will get bNxRx32 atom.
  The only supported atom size in the n-mode is 64, so bN%64==0. The warpspecialized
  kernel assigns warpgroup 1 to handle EAR * BpEB, and because we only use one WG the
  atom layout is trivial. Each WGMMA consumes one BpEB tile (bN,bK) and one EAR tile (R,bK).
  The operands are both sourced from SMEM and must have a specific layout; this
  layout is handled by swizzles in the SMEM layouts.
  */
  using TiledMmaNRK = decltype(cute::make_tiled_mma(
      cute::MMA_Atom<cute::SM80_16x8x32_S32S8S8S32_TN>{},
      AtomLayoutMma{},
      cute::Tile<cute::Underscore, cute::Int<R>, cute::Underscore>{}));

  /*
  Tiled WGMMA for B + EBR * EBL, size (bN, bK, R) for one k_block iteration.
  The selected atom will be bNxbKx32. The warpspecialized kernel assigns
  warpgroup 2 to handle B + EBR * EBL. Each WGMMA consumes one EBR tile (bN,R) and
  one EBL tile (bK, R). We also set the accumulator to B first to do the B + EB.
  The operands are both sourced from SMEM, but accumulator is in RMEM.
  */
  using TiledMmaNKR = decltype(cute::make_tiled_mma(
      cute::MMA_Atom<cute::SM80_16x8x32_S32S8S8S32_TN>{},
      AtomLayoutMma{},
      cute::Tile<cute::Underscore, cute::Int<kBlockK>, cute::Underscore>{}));

  /*
  Smem layouts for the smem tensors, used primarily for TMA and WGMMA. Sizes are
   determined by the tile_shape, all except EBR are pipelined to kStages. ss_smem_selector
   is used to automatically add the necessary swizzles for the WGMMA/TMA.
  */
  // B: bNxbK (K-major).
  using SmemLayoutAtomB =
      decltype(cutlass::gemm::collective::detail::ss_smem_selector<
               GMMA::Major::K, Element, Int<kBlockN>, Int<kBlockK>>());
  using SmemLayoutB = decltype(tile_to_shape(
      SmemLayoutAtomB{}, Shape<Int<kBlockN>, Int<kBlockK>, Int<kStages>>{}));

  // EAR: bRxbK (K-major)
  using SmemLayoutAtomEAR =
      decltype(cutlass::gemm::collective::detail::ss_smem_selector<
               GMMA::Major::K, Element, Int<R>, Int<kBlockK>>());
  using SmemLayoutEAR = decltype(tile_to_shape(
      SmemLayoutAtomEAR{}, Shape<Int<R>, Int<kBlockK>, Int<kStages>>{}));

  // EBR: bNxbR (R-major, no pipeline)
  using SmemLayoutAtomEBR =
      decltype(cutlass::gemm::collective::detail::ss_smem_selector<
               GMMA::Major::K, Element, Int<kBlockN>, Int<R>>());
  using SmemLayoutEBR = decltype(tile_to_shape(SmemLayoutAtomEBR{},
                                               Shape<Int<kBlockN>, Int<R>>{}));

  // EBL: bKxbR (R-major)
  using SmemLayoutAtomEBL =
      decltype(cutlass::gemm::collective::detail::ss_smem_selector<
               GMMA::Major::K, Element, Int<kBlockK>, Int<R>>());
  using SmemLayoutEBL = decltype(tile_to_shape(
      SmemLayoutAtomEBL{}, Shape<Int<kBlockK>, Int<R>, Int<kStages>>{}));

  // EARxBpEB: bNxbR (R-major, no pipeline)
  using SmemLayoutAtomEARxBpEB =
      decltype(cutlass::gemm::collective::detail::ss_smem_selector<
               GMMA::Major::K, ElementDenoise, Int<kBlockN>, Int<R>>());
  using SmemLayoutEARxBpEB = decltype(tile_to_shape(
      SmemLayoutAtomEARxBpEB{}, Shape<Int<kBlockN>, Int<R>>{}));

  // BpEB: bNxbK (K-major)
  // No swizzle needed here: threads do 16b store to SMEM, then BpEB is stored
  // with TMA.
  using SmemLayoutAtomBpEB =
      decltype(cutlass::gemm::collective::detail::ss_smem_selector<
               GMMA::Major::K, Element, Int<kBlockN>, Int<kBlockK>>());
  using SmemLayoutBpEB = decltype(tile_to_shape(
      SmemLayoutAtomBpEB{},
      Shape<Int<kBlockN>, Int<kBlockK>, Int<kStagesOut>>{}));

  // Place holder types for TMA type definitions, correspoding to
  // the GMEM Tensors. Note that the stride assumes majorness along
  // "K-mode" (mode 1)
  using ShapeT = cute::Shape<int32_t, int32_t>;
  using StrideT = cute::Shape<int32_t, _1>;
  using LayoutT = cute::Layout<ShapeT, StrideT>;

  // Creating TMA types for loads
  using TMA_B = decltype(make_tma_copy(
      cute::SM90_TMA_LOAD{},
      make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), ShapeT{},
                  StrideT{}),         // placeholder src GMEM Tensor
      take<0, 2>(SmemLayoutB{}),      // dst shape (stripped stage)
      select<0, 2>(TileShape_NRK{}),  // tiler
      _1{}));                         // no multicast

  using TMA_EAR = decltype(make_tma_copy(
      cute::SM90_TMA_LOAD{},
      make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), ShapeT{},
                  StrideT{}),
      take<0, 2>(SmemLayoutEAR{}), select<1, 2>(TileShape_NRK{}), _1{}));

  using TMA_EBR = decltype(make_tma_copy(
      cute::SM90_TMA_LOAD{},
      make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), ShapeT{},
                  StrideT{}),
      take<0, 2>(SmemLayoutEBR{}), select<0, 1>(TileShape_NRK{}), _1{}));

  using TMA_EBL = decltype(make_tma_copy(
      cute::SM90_TMA_LOAD{},
      make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), ShapeT{},
                  StrideT{}),
      take<0, 2>(SmemLayoutEBL{}), select<2, 1>(TileShape_NRK{}), _1{}));

  // Creating TMA types for stores
  using GmemTiledCopyEARxBpEB =
      cute::conditional_t<NoReduction, cute::SM90_TMA_STORE,
                          cute::SM90_TMA_REDUCE_ADD>;

  using TMA_EARxBpEB = decltype(make_tma_copy(
      GmemTiledCopyEARxBpEB{},
      make_tensor(make_gmem_ptr(static_cast<ElementDenoise const*>(nullptr)),
                  ShapeT{}, StrideT{}),
      SmemLayoutEARxBpEB{}, select<0, 1>(TileShape_NRK{}), _1{}));

  using TMA_BpEB_Store = decltype(make_tma_copy(
      cute::SM90_TMA_STORE{},
      make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), ShapeT{},
                  StrideT{}),
      take<0, 2>(SmemLayoutBpEB{}), select<0, 2>(TileShape_NRK{}), _1{}));

  // Computing the tx_counts for a single TMA load/store, used for mbarrier
  static constexpr uint32_t TmaTransactionBytesB = static_cast<uint32_t>(
      size(take<0, 2>(SmemLayoutB{})) * cutlass::sizeof_bits_v<Element> / 8);
  static constexpr uint32_t TmaTransactionBytesEAR = static_cast<uint32_t>(
      size(take<0, 2>(SmemLayoutEAR{})) * cutlass::sizeof_bits_v<Element> / 8);
  static constexpr uint32_t TmaTransactionBytesEBR = static_cast<uint32_t>(
      size(take<0, 2>(SmemLayoutEBR{})) * cutlass::sizeof_bits_v<Element> / 8);
  static constexpr uint32_t TmaTransactionBytesEBL = static_cast<uint32_t>(
      size(take<0, 2>(SmemLayoutEBL{})) * cutlass::sizeof_bits_v<Element> / 8);
  static constexpr uint32_t TmaTransactionBytesEARxBpEB =
      static_cast<uint32_t>(size(take<0, 2>(SmemLayoutEARxBpEB{})) *
                            cutlass::sizeof_bits_v<ElementDenoise> / 8);

  // RMEM->SMEM for EARxBpEB, StrideT assumes R-major
  using TileShape_NR = decltype(select<0, 2>(TileShape_NKR{}));
  using CopyOpR2S = AutoVectorizingCopyWithAssumedAlignment<128>;
  using SmemCopyAtomEARxBpEB = Copy_Atom<CopyOpR2S, ElementDenoise>;

  // RMEM<->SMEM for B, used as accumulator in BpEB computation
  // To avoid a lot of hardcoded layouts, we will use a trick
  // where we create a dummy MMA with half the N-mode (R in this case)
  // And load it as uint16 using LDSM. We can get away with this because
  // we then reshuffle the values in the registers to match unit32
  // accumulator.
  using TileShape_NKR_half = Shape<Int<kBlockN>, Int<kBlockK / 2>, Int<R>>;
  using TiledMmaNKR_half = decltype(cute::make_tiled_mma(
      cute::MMA_Atom<cute::SM80_16x8x32_S32S8S8S32_TN>{},
      AtomLayoutMma{},
      cute::Tile<cute::Underscore, cute::Int<kBlockK / 2>, cute::Underscore>{}));
  using S2RCopyAtomB = Copy_Atom<SM75_U32x4_LDSM_N, uint16_t>;
  using R2SCopyAtomB = Copy_Atom<SM90_U32x4_STSM_N, uint16_t>;

  // Defining load pipeline types (initialized in kernel). Used by B, EAR and EBL
  using MainloopLoadPipeline = typename cutlass::PipelineTmaAsync<kStages>;
  using LoadPipelineParams = typename MainloopLoadPipeline::Params;
  using LoadPipelineState = typename MainloopLoadPipeline::PipelineState;
  using LoadBarrierType = typename MainloopLoadPipeline::ProducerBarrierType;

  // Defining store pipeline, used by BpEB
  using MainloopStorePipeline = typename cutlass::PipelineAsync<kStages>;
  using StorePipelineParams = typename MainloopStorePipeline::Params;
  using StorePipelineState = typename MainloopStorePipeline::PipelineState;
  using StoreBarrierType = typename MainloopStorePipeline::ProducerBarrierType;

  // arrier type to be used by the mbar for EBR
  using ProducerBarType = cutlass::arch::ClusterTransactionBarrier;

  // TMA requires 128B alignment
  static constexpr size_t Alignment = 128;

  struct SharedStorage : cute::aligned_struct<Alignment> {

    union {
      // mainloop
      struct {
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutB>, Alignment>
            smem_B;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutEAR>, Alignment>
            smem_EAR;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutEBR>, Alignment>
            smem_EBR;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutEBL>, Alignment>
            smem_EBL;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutBpEB>, Alignment>
            smem_BpEB;
      };

      // epilogue
      cute::array_aligned<ElementDenoise, cute::cosize_v<SmemLayoutEARxBpEB>,
                          Alignment>
          smem_EARxBpEB;
    };

    struct {
      typename MainloopLoadPipeline::SharedStorage pipeline_B;
      typename MainloopLoadPipeline::SharedStorage pipeline_EAR;
      typename MainloopLoadPipeline::SharedStorage pipeline_EBL;
      typename MainloopStorePipeline::SharedStorage pipeline_BpEB_store;
      uint64_t EBR_mbar;
    };
  };

  static constexpr int SharedStorageSize = sizeof(SharedStorage);

  // Device side arguments
  struct Arguments {
    // device pointers
    Element const* const ptr_B;
    Element const* const ptr_EBR;
    Element const* const ptr_EAR;
    Element const* const ptr_EBL;
    Element* const ptr_BpEB;
    ElementDenoise* const ptr_EARxBpEB;
    // dimensions
    int n;
    int k;
    int num_k_blocks;  // k blocks per split
    int total_k_blocks;
  };

  // Kernel entry point API
  struct Params {
    // device pointers
    Element const* const ptr_B;
    Element const* const ptr_EBR;
    Element const* const ptr_EBL;
    Element const* const ptr_EAR;
    Element* const ptr_BpEB;
    ElementDenoise* const ptr_EARxBpEB;
    // dimensions
    int n;
    int k;
    int num_k_blocks;  // k blocks per split
    int total_k_blocks;
    // Layouts for GMEM Tensors (needed for TMA partition)
    LayoutT layout_B;
    LayoutT layout_EAR;
    LayoutT layout_EBR;
    LayoutT layout_EBL;
    LayoutT layout_EARxBpEB;
    LayoutT layout_BpEB;
    // TMAs
    TMA_B tma_load_B;
    TMA_EAR tma_load_EAR;
    TMA_EBR tma_load_EBR;
    TMA_EBL tma_load_EBL;
    TMA_BpEB_Store tma_store_BpEB;
    TMA_EARxBpEB tma_store_EARxBpEB;
  };

  enum struct NamedBarriers {
    S2RCopyBDone,
    R2SCopyEARxBpEBDone,
    EARxBpEBSMEMReady,
  };

  // Convert to underlying arguments
  static Params to_underlying_arguments(Arguments const& args) {

    // Create the GMEM layout and the instantiated TMA objects
    LayoutT layout_B =
        make_layout(make_shape(args.n, args.k), make_stride(args.k, _1{}));
    LayoutT layout_EAR =
        make_layout(make_shape(R, args.k), make_stride(args.k, _1{}));
    LayoutT layout_EBR =
        make_layout(make_shape(args.n, R), make_stride(R, _1{}));
    LayoutT layout_EBL =
        make_layout(make_shape(args.k, R), make_stride(R, _1{}));
    LayoutT layout_BpEB =
        make_layout(make_shape(args.n, args.k), make_stride(args.k, _1{}));
    LayoutT layout_EARxBpEB =
        make_layout(make_shape(args.n, R), make_stride(R, _1{}));

    Tensor mB = make_tensor(make_gmem_ptr(args.ptr_B), layout_B);
    TMA_B tma_load_B =
        make_tma_copy(cute::SM90_TMA_LOAD{}, mB, take<0, 2>(SmemLayoutB{}),
                      select<0, 2>(TileShape_NRK{}), _1{});

    Tensor mEAR = make_tensor(make_gmem_ptr(args.ptr_EAR), layout_EAR);
    TMA_EAR tma_load_EAR =
        make_tma_copy(cute::SM90_TMA_LOAD{}, mEAR, take<0, 2>(SmemLayoutEAR{}),
                      select<1, 2>(TileShape_NRK{}), _1{});

    Tensor mEBR = make_tensor(make_gmem_ptr(args.ptr_EBR), layout_EBR);
    TMA_EBR tma_load_EBR =
        make_tma_copy(cute::SM90_TMA_LOAD{}, mEBR, take<0, 2>(SmemLayoutEBR{}),
                      select<0, 1>(TileShape_NRK{}), _1{});

    Tensor mEBL = make_tensor(make_gmem_ptr(args.ptr_EBL), layout_EBL);
    TMA_EBL tma_load_EBL =
        make_tma_copy(cute::SM90_TMA_LOAD{}, mEBL, take<0, 2>(SmemLayoutEBL{}),
                      select<2, 1>(TileShape_NRK{}), _1{});

    Tensor mBpEB = make_tensor(make_gmem_ptr(args.ptr_BpEB), layout_BpEB);

    TMA_BpEB_Store tma_store_BpEB = make_tma_copy(
        cute::SM90_TMA_STORE{}, mBpEB, take<0, 2>(SmemLayoutBpEB{}),
        select<0, 2>(TileShape_NRK{}), _1{});

    Tensor mEARxBpEB =
        make_tensor(make_gmem_ptr(args.ptr_EARxBpEB), layout_EARxBpEB);
    TMA_EARxBpEB tma_store_EARxBpEB =
        make_tma_copy(GmemTiledCopyEARxBpEB{}, mEARxBpEB, SmemLayoutEARxBpEB{},
                      select<0, 1>(TileShape_NRK{}), _1{});

    return {.ptr_B = args.ptr_B,
            .ptr_EBR = args.ptr_EBR,
            .ptr_EBL = args.ptr_EBL,
            .ptr_EAR = args.ptr_EAR,
            .ptr_BpEB = args.ptr_BpEB,
            .ptr_EARxBpEB = args.ptr_EARxBpEB,
            .n = args.n,
            .k = args.k,
            .num_k_blocks = args.num_k_blocks,
            .total_k_blocks = args.total_k_blocks,
            .layout_B = layout_B,
            .layout_EAR = layout_EAR,
            .layout_EBR = layout_EBR,
            .layout_EBL = layout_EBL,
            .layout_EARxBpEB = layout_EARxBpEB,
            .layout_BpEB = layout_BpEB,
            .tma_load_B = tma_load_B,
            .tma_load_EAR = tma_load_EAR,
            .tma_load_EBR = tma_load_EBR,
            .tma_load_EBL = tma_load_EBL,
            .tma_store_BpEB = tma_store_BpEB,
            .tma_store_EARxBpEB = tma_store_EARxBpEB};
  }

  // Computes the kernel launch grid shape based on runtime parameters
  static dim3 get_grid_shape(Params const& params) {
    if constexpr (NoReduction) {
      return dim3(ceil_div(params.n, kBlockN), 1, 1);
    } else {
      return dim3(ceil_div(params.n, kBlockN),
                  ceil_div(params.k, params.num_k_blocks * kBlockK), 1);
    }
  }

  static dim3 get_block_shape() { return dim3(MaxThreadsPerBlock, 1, 1); }

 private:
  // Load all input tensors via TMA (WG0 load warp)
  CUTLASS_DEVICE void load_tensors(
      Params const& params, SharedStorage& shared_storage, const int n_block,
      const int k_block_min, const int k_block_max,
      MainloopLoadPipeline& pipeline_B, MainloopLoadPipeline& pipeline_EAR,
      MainloopLoadPipeline& pipeline_EBL, LoadPipelineState& smem_pipe_write_B,
      LoadPipelineState& smem_pipe_write_EAR,
      LoadPipelineState& smem_pipe_write_EBL) {

    // SMEM tensor setup
    Tensor sB = make_tensor(make_smem_ptr(shared_storage.smem_B.data()),
                            SmemLayoutB{});  // (bN, bK, P)
    Tensor sEAR = make_tensor(make_smem_ptr(shared_storage.smem_EAR.data()),
                              SmemLayoutEAR{});  // (R, bK, P)
    Tensor sEBR = make_tensor(make_smem_ptr(shared_storage.smem_EBR.data()),
                              SmemLayoutEBR{});  // (bN, R)
    Tensor sEBL = make_tensor(make_smem_ptr(shared_storage.smem_EBL.data()),
                              SmemLayoutEBL{});  // (bK, R, P)
    // GMEM tensor setup
    Tensor mB = params.tma_load_B.get_tma_tensor(params.layout_B.shape());
    Tensor mEAR = params.tma_load_EAR.get_tma_tensor(params.layout_EAR.shape());
    Tensor mEBL = params.tma_load_EBL.get_tma_tensor(params.layout_EBL.shape());
    Tensor mEBR = params.tma_load_EBR.get_tma_tensor(params.layout_EBR.shape());

    // CTA local partitions
    Tensor gB =
        local_tile(mB, select<0, 2>(TileShape_NRK{}), make_coord(n_block, _));
    Tensor gEAR =
        local_tile(mEAR, select<1, 2>(TileShape_NRK{}), make_coord(_0{}, _));
    Tensor gEBL =
        local_tile(mEBL, select<2, 1>(TileShape_NRK{}), make_coord(_, _0{}));
    Tensor gEBR = local_tile(mEBR, select<0, 1>(TileShape_NRK{}),
                             make_coord(n_block, _0{}));

    if (cute::elect_one_sync()) {
      // Load EBR once (doesn't change with K)
      uint64_t* EBR_mbar = &shared_storage.EBR_mbar;
      ProducerBarType::arrive_and_expect_tx(EBR_mbar, TmaTransactionBytesEBR);
      auto [tEBRgEBR, tEBRsEBR] =
          tma_partition(params.tma_load_EBR, Int<0>{}, Layout<_1>{},
                        group_modes<0, 2>(sEBR), group_modes<0, 2>(gEBR));
      copy(params.tma_load_EBR.with(*EBR_mbar), tEBRgEBR, tEBRsEBR);

      // Load pipelined tensors for each k_block
      CUTLASS_PRAGMA_NO_UNROLL
      for (int k_block = k_block_min; k_block < k_block_max; ++k_block) {

        auto [tEBLgEBL, tEBLsEBL] =
            tma_partition(params.tma_load_EBL, Int<0>{}, Layout<_1>{},
                          group_modes<0, 2>(sEBL), group_modes<0, 2>(gEBL));
        pipeline_EBL.producer_acquire(smem_pipe_write_EBL);
        LoadBarrierType* tmaBarEBL =
            pipeline_EBL.producer_get_barrier(smem_pipe_write_EBL);
        copy(params.tma_load_EBL.with(*tmaBarEBL, 0), tEBLgEBL(_, k_block),
             tEBLsEBL(_, smem_pipe_write_EBL.index()));
        pipeline_EBL.producer_commit(smem_pipe_write_EBL,
                                     TmaTransactionBytesEBL);
        ++(smem_pipe_write_EBL);

        auto [tBgB, tBsB] =
            tma_partition(params.tma_load_B, Int<0>{}, Layout<_1>{},
                          group_modes<0, 2>(sB), group_modes<0, 2>(gB));
        pipeline_B.producer_acquire(smem_pipe_write_B);
        LoadBarrierType* tmaBarB =
            pipeline_B.producer_get_barrier(smem_pipe_write_B);
        copy(params.tma_load_B.with(*tmaBarB, 0), tBgB(_, k_block),
             tBsB(_, smem_pipe_write_B.index()));
        pipeline_B.producer_commit(smem_pipe_write_B, TmaTransactionBytesB);
        ++(smem_pipe_write_B);

        auto [tEARgEAR, tEARsEAR] =
            tma_partition(params.tma_load_EAR, Int<0>{}, Layout<_1>{},
                          group_modes<0, 2>(sEAR), group_modes<0, 2>(gEAR));
        pipeline_EAR.producer_acquire(smem_pipe_write_EAR);
        LoadBarrierType* tmaBarEAR =
            pipeline_EAR.producer_get_barrier(smem_pipe_write_EAR);
        copy(params.tma_load_EAR.with(*tmaBarEAR, 0), tEARgEAR(_, k_block),
             tEARsEAR(_, smem_pipe_write_EAR.index()));
        pipeline_EAR.producer_commit(smem_pipe_write_EAR,
                                     TmaTransactionBytesEAR);
        ++(smem_pipe_write_EAR);
      }
    }
  }

 public:
  // Compute EARxBpEB: WG1 consumes BpEB from store pipeline and EAR from load pipeline
  template <typename FrgTensorC>
  CUTLASS_DEVICE void compute_EARxBpEB(Params const& params,
                                       MainloopStorePipeline& pipeline_BpEB,
                                       MainloopLoadPipeline& pipeline_EAR,
                                       SharedStorage& shared_storage,
                                       FrgTensorC& tCrEARxBpEB,
                                       const int n_block, const int k_block_min,
                                       const int k_block_max, const int tid) {
    // smem tensors
    Tensor sBpEB = make_tensor(make_smem_ptr(shared_storage.smem_BpEB.data()),
                               SmemLayoutBpEB{});  // (bN, bK, P)
    Tensor sEAR = make_tensor(make_smem_ptr(shared_storage.smem_EAR.data()),
                              SmemLayoutEAR{});  // (R, bK, P)

    // 64xRx32 MMA atom
    TiledMmaNRK tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tid);

    Tensor tCsBpEB = thr_mma.partition_A(sBpEB);  // (MMA,MMA_N,MMA_K,P)
    Tensor tCsEAR = thr_mma.partition_B(sEAR);    // (MMA,MMA_N,MMA_K,P)

    // Allocate "fragments" -- these are WGMMA matrix descriptors
    Tensor tCrBpEB = thr_mma.make_fragment_A(tCsBpEB);  // (MMA,MMA_N,MMA_K,P)
    Tensor tCrEAR = thr_mma.make_fragment_B(tCsEAR);    // (MMA,MMA_N,MMA_K,P)

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    using S2RCopyAtom_NB1 = Copy_Atom<SM75_U32x4_LDSM_N, Element>;
    auto s2r_tiled_copy_BpEB1 = make_tiled_copy_A(S2RCopyAtom_NB1{}, tiled_mma);
    auto s2r_tiled_copy_EAR1 = make_tiled_copy_B(S2RCopyAtom_NB1{}, tiled_mma);
    auto s2r_thr_copy_BpEB1 = s2r_tiled_copy_BpEB1.get_thread_slice(tid);
    auto s2r_thr_copy_EAR1 = s2r_tiled_copy_EAR1.get_thread_slice(tid);
    Tensor tCsBpEB_s2r1 = s2r_thr_copy_BpEB1.partition_S(sBpEB);
    Tensor tCsEAR_s2r1 = s2r_thr_copy_EAR1.partition_S(sEAR);
    Tensor tCrBpEB_view1 = s2r_thr_copy_BpEB1.retile_D(tCrBpEB);
    Tensor tCrEAR_view1 = s2r_thr_copy_EAR1.retile_D(tCrEAR);
#endif

    StorePipelineState smem_pipe_read_BpEB;
    LoadPipelineState smem_pipe_read_EAR;

    CUTLASS_PRAGMA_NO_UNROLL
    for (int k_block = k_block_min; k_block < k_block_max; ++k_block) {
      pipeline_BpEB.consumer_wait(smem_pipe_read_BpEB);
      pipeline_EAR.consumer_wait(smem_pipe_read_EAR);
      cutlass::arch::fence_view_async_shared();
      // Hopper-only WGMMA sync; no-op on Blackwell consumer (sm_120/121)
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
      warpgroup_fence_operand(tCrEARxBpEB);
      warpgroup_arrive();
#endif
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
      cute::copy(s2r_tiled_copy_BpEB1,
                 tCsBpEB_s2r1(_, _, _, smem_pipe_read_BpEB.index()),
                 tCrBpEB_view1(_, _, _, smem_pipe_read_BpEB.index()));
      cute::copy(s2r_tiled_copy_EAR1,
                 tCsEAR_s2r1(_, _, _, smem_pipe_read_EAR.index()),
                 tCrEAR_view1(_, _, _, smem_pipe_read_EAR.index()));
#endif
      // WGMMA with dispatch mode (V,M,K) x (V,N,K) => (V,M,N)
      gemm(tiled_mma, tCrBpEB(_, _, _, smem_pipe_read_BpEB.index()),
           tCrEAR(_, _, _, smem_pipe_read_EAR.index()), tCrEARxBpEB);
      // Hopper-only WGMMA sync; no-op on Blackwell consumer (sm_120/121)
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
      warpgroup_commit_batch();
#endif
      // Wait for all MMAs across warp group to complete
      // Hopper-only WGMMA sync; no-op on Blackwell consumer (sm_120/121)
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
      warpgroup_wait<0>();
#endif

      // Hopper-only WGMMA sync; no-op on Blackwell consumer (sm_120/121)
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
      warpgroup_fence_operand(tCrEARxBpEB);
#endif

      cutlass::arch::fence_view_async_shared();
      pipeline_BpEB.consumer_release(smem_pipe_read_BpEB);
      pipeline_EAR.consumer_release(smem_pipe_read_EAR);
      ++smem_pipe_read_BpEB;
      ++smem_pipe_read_EAR;
    }
  }

  CUTLASS_DEVICE void compute_BpEB(Params const& params,
                                   MainloopLoadPipeline pipeline_B,
                                   MainloopLoadPipeline pipeline_EBL,
                                   MainloopStorePipeline pipeline_BpEB,
                                   SharedStorage& shared_storage,
                                   const int n_block, const int k_block_min,
                                   const int k_block_max, const int tid) {
    // (N, K)
    Tensor mB_out =
        make_tensor(make_gmem_ptr(params.ptr_BpEB), params.layout_BpEB);
    // (bN, bK, k_tiles)
    Tensor gB_out = local_tile(mB_out, select<0, 1>(TileShape_NKR{}),
                               make_coord(n_block, _));
    // (bN, bK, P)
    Tensor sB =
        make_tensor(make_smem_ptr(shared_storage.smem_B.data()), SmemLayoutB{});
    Tensor sB_pi = as_position_independent_swizzle_tensor(sB);
    // (bN, R)
    Tensor sEBR = make_tensor(make_smem_ptr(shared_storage.smem_EBR.data()),
                              SmemLayoutEBR{});
    // (bK, R, P)
    Tensor sEBL = make_tensor(make_smem_ptr(shared_storage.smem_EBL.data()),
                              SmemLayoutEBL{});
    // (bN, bK, OP)
    Tensor sBpEB = make_tensor(make_smem_ptr(shared_storage.smem_BpEB.data()),
                               SmemLayoutBpEB{});

    // 64xbKx32 MMA atom
    TiledMmaNKR tiled_mma;
    auto thr_mma = tiled_mma.get_slice(tid);

    // (ATOM, REST_N, REST_K)
    Tensor tCrBpEB =
        partition_fragment_C(tiled_mma, select<0, 1>(TileShape_NKR{}));
    Tensor tCrBpEB_int8 = make_fragment_like<Element>(tCrBpEB);

    // (ATOM, REST_N, REST_R)
    Tensor tCsEBR = thr_mma.partition_A(sEBR);
    // (1, REST_N, REST_R) -- tensor of WGMMA descriptors, 1 per atom
    Tensor tCrEBR = thr_mma.make_fragment_A(tCsEBR);

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    using S2RCopyAtom_NB2 = Copy_Atom<SM75_U32x4_LDSM_N, Element>;
    auto s2r_tiled_copy_EBR2 = make_tiled_copy_A(S2RCopyAtom_NB2{}, tiled_mma);
    auto s2r_thr_copy_EBR2 = s2r_tiled_copy_EBR2.get_thread_slice(tid);
    Tensor tCsEBR_s2r2 = s2r_thr_copy_EBR2.partition_S(sEBR);
    Tensor tCrEBR_view2 = s2r_thr_copy_EBR2.retile_D(tCrEBR);
#endif

    // (ATOM, REST_K, REST_R)
    Tensor tCsEBL = thr_mma.partition_B(sEBL);
    // (1, REST_K, REST_R) -- tensor of WGMMA descriptors, 1 per atom
    Tensor tCrEBL = thr_mma.make_fragment_B(tCsEBL);

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    auto s2r_tiled_copy_EBL2 = make_tiled_copy_B(S2RCopyAtom_NB2{}, tiled_mma);
    auto s2r_thr_copy_EBL2 = s2r_tiled_copy_EBL2.get_thread_slice(tid);
    Tensor tCsEBL_s2r2 = s2r_thr_copy_EBL2.partition_S(sEBL);
    Tensor tCrEBL_view2 = s2r_thr_copy_EBL2.retile_D(tCrEBL);
#endif

    // Wait for EBR load. It does not change with K, so only once
    uint64_t* EBR_mbar = &shared_storage.EBR_mbar;
    ProducerBarType::wait(EBR_mbar, 0);
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    cute::copy(s2r_tiled_copy_EBR2, tCsEBR_s2r2, tCrEBR_view2);
#endif

    // S2R copy for B to add to the accumulator
    TiledMmaNKR_half tiled_mma_half;
    auto s2r_tiled_copy_B = make_tiled_copy_C(S2RCopyAtomB{}, tiled_mma_half);
    auto s2r_thr_copy_B = s2r_tiled_copy_B.get_slice(tid);
    auto sB_u16 = recast<uint16_t>(sB_pi);
    auto taccCsB = s2r_thr_copy_B.partition_S(sB_u16);

    auto tCrB = partition_fragment_C(tiled_mma_half,
                                     select<0, 1>(TileShape_NKR_half{}));
    auto tCrB_u16 = make_tensor_like<uint16_t>(tCrB);  // (MMA_V, MMA_M, MM_N)
    // ((ATOM_V, COPY_V), COPY_M, COPY_N)
    auto taccCrB = s2r_thr_copy_B.retile_D(tCrB_u16);
    auto taccCrB_int8 = recast<Element>(taccCrB);

    // R2S copy for B, because stsm and ldsm has the same shape, we don't
    // need to retile the register
    auto r2s_tiled_copy_B = make_tiled_copy_C(R2SCopyAtomB{}, tiled_mma_half);
    auto r2s_thr_copy_B = r2s_tiled_copy_B.get_slice(tid);
    auto sBpEB_u16 = recast<uint16_t>(sBpEB);
    auto taccCsBpEB = s2r_thr_copy_B.partition_S(sBpEB_u16);

    LoadPipelineState smem_pipe_read_B;
    LoadPipelineState smem_pipe_read_EBL;
    StorePipelineState smem_pipe_write_BpEB =
        cutlass::make_producer_start_state<MainloopStorePipeline>();

    CUTLASS_PRAGMA_NO_UNROLL
    for (int k_block = k_block_min; k_block < k_block_max; ++k_block) {

      clear(tCrBpEB);  // We store every iter so we need to clear
      // WGMMA: B + EBR * EBL
      pipeline_EBL.consumer_wait(smem_pipe_read_EBL);
      // Hopper-only WGMMA sync; no-op on Blackwell consumer (sm_120/121)
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
      warpgroup_fence_operand(tCrBpEB);
      warpgroup_arrive();
#endif
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
      cute::copy(s2r_tiled_copy_EBL2,
                 tCsEBL_s2r2(_, _, _, smem_pipe_read_EBL.index()),
                 tCrEBL_view2(_, _, _, smem_pipe_read_EBL.index()));
#endif
      gemm(tiled_mma, tCrEBR, tCrEBL(_, _, _, smem_pipe_read_EBL.index()),
           tCrBpEB);
      // Hopper-only WGMMA sync; no-op on Blackwell consumer (sm_120/121)
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
      warpgroup_commit_batch();
      warpgroup_wait<0>();
      warpgroup_fence_operand(tCrBpEB);
#endif

      pipeline_EBL.consumer_release(smem_pipe_read_EBL);
      ++smem_pipe_read_EBL;

      // Load B to registers
      pipeline_B.consumer_wait(smem_pipe_read_B);
      cute::copy(s2r_tiled_copy_B, taccCsB(_, _, _, smem_pipe_read_B.index()),
                 taccCrB);
      cutlass::arch::NamedBarrier::sync(
          kNumBpEBThreads, static_cast<uint32_t>(NamedBarriers::S2RCopyBDone));
      cutlass::arch::fence_view_async_shared();
      pipeline_B.consumer_release(smem_pipe_read_B);
      ++smem_pipe_read_B;

      // Convert down from int32 accumulator to int8
      pearl::convert_type_out(tCrBpEB, tCrBpEB_int8);

      // Shuffle to allow for bank conflict free store and load
      permute_Aregs_fp8(tCrBpEB_int8);

      // Add B to EB
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(taccCrB_int8); ++i) {
        taccCrB_int8[i] += tCrBpEB_int8[i];
      }

      // RMEM->SMEM copy for BpEB to "stage" it for TMA_STORE
      pipeline_BpEB.producer_acquire(smem_pipe_write_BpEB);
      cute::copy(r2s_tiled_copy_B, taccCrB,
                 taccCsBpEB(_, _, _, smem_pipe_write_BpEB.index()));
      pipeline_BpEB.producer_commit(smem_pipe_write_BpEB);
      ++smem_pipe_write_BpEB;
    }
  }

  template <typename FrgTensorC>
  CUTLASS_DEVICE void store_EARxBpEB(Params const& params,
                                     FrgTensorC& tCrEARxBpEB,
                                     SharedStorage& shared_storage,
                                     const int n_block, const int tid) {
    // SMEM Tensor
    Tensor sEARxBpEB =
        make_tensor(make_smem_ptr(shared_storage.smem_EARxBpEB.data()),
                    SmemLayoutEARxBpEB{});  // (bN, bR, P)
    Tensor sEARxBpEB_pi = as_position_independent_swizzle_tensor(sEARxBpEB);

    // RMEM -> SMEM copy
    TiledMmaNRK tiled_mma;
    auto r2s_tiled_copy_O =
        make_tiled_copy_C(SmemCopyAtomEARxBpEB{}, tiled_mma);
    auto r2s_thr_copy_O = r2s_tiled_copy_O.get_thread_slice(tid);
    Tensor tR2SsEARxBpEB = r2s_thr_copy_O.partition_D(sEARxBpEB_pi);

    // TMA out
    Tensor mEARxBpEB = params.tma_store_EARxBpEB.get_tma_tensor(
        params.layout_EARxBpEB.shape());
    Tensor gEARxBpEB = local_tile(mEARxBpEB, select<0, 1>(TileShape_NRK{}),
                                  make_coord(n_block, _0{}));  // (N, R)
    auto tma_thr_EARxBpEB = params.tma_store_EARxBpEB.get_slice(_0{});
    Tensor tOgEARxBpEB =
        tma_thr_EARxBpEB.partition_D(gEARxBpEB);  // (TMA, TMA_N, TMA_R)
    Tensor tOsEARxBpEB =
        tma_thr_EARxBpEB.partition_S(sEARxBpEB);  // (TMA, TMA_N, TMA_R)

    // RMEM -> SMEM tiled copy, scale if fp16
    if constexpr (!cute::is_same_v<ElementDenoise, int>) {
      Tensor tCrEARxBpEB_out = make_fragment_like<ElementDenoise>(tCrEARxBpEB);
      // Convert to output type
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(tCrEARxBpEB); ++i) {
        tCrEARxBpEB_out(i) = static_cast<ElementDenoise>(
            static_cast<ElementScale>(tCrEARxBpEB(i)) /
            static_cast<ElementScale>(pearl::kEARxBpEBScaleFactor));
      }
      Tensor tR2SrEARxBpEB = r2s_thr_copy_O.retile_S(
          tCrEARxBpEB_out);  // ((Atom,AtomNum), MMA_N, MMA_R)
      cute::copy(r2s_tiled_copy_O, tR2SrEARxBpEB, tR2SsEARxBpEB);
    } else {
      Tensor tR2SrEARxBpEB = r2s_thr_copy_O.retile_S(
          tCrEARxBpEB);  // ((Atom,AtomNum), MMA_N, MMA_R)
      cute::copy(r2s_tiled_copy_O, tR2SrEARxBpEB, tR2SsEARxBpEB);
    }

    // ensure smem writes are completed and visible to TMA
    cutlass::arch::NamedBarrier::sync(
        kNumEARxBpEBThreads,
        static_cast<uint32_t>(NamedBarriers::R2SCopyEARxBpEBDone));
    cutlass::arch::fence_view_async_shared();
    // SMEM -> GMEM TMA copy
    int warp_idx_in_warpgroup = pearl::warp_idx_in_warpgroup_sync();
    if (warp_idx_in_warpgroup == 0 && cute::elect_one_sync()) {  // Load warp
      cute::copy(params.tma_store_EARxBpEB, tOsEARxBpEB, tOgEARxBpEB);
      tma_store_arrive();
      tma_store_wait<0>();
    }
  }

  CUTLASS_DEVICE void store_BpEB(Params const& params,
                                 MainloopStorePipeline pipeline_BpEB,
                                 SharedStorage& shared_storage,
                                 const int n_block, const int k_block_min,
                                 const int k_block_max) {
    // (bN, bK, OP)
    Tensor sBpEB = make_tensor(make_smem_ptr(shared_storage.smem_BpEB.data()),
                               SmemLayoutBpEB{});

    // TMA out
    Tensor mBpEB = params.tma_store_BpEB.get_tma_tensor(
        params.layout_BpEB.shape());  // (N, K)
    // (bN, bK, k_tiles)
    Tensor gBpEB = local_tile(mBpEB, select<0, 2>(TileShape_NRK{}),
                              make_coord(n_block, _));
    auto tma_thr_BpEB = params.tma_store_BpEB.get_slice(_0{});
    // (TMA, k_tiles)
    Tensor tOgBpEB = group_modes<0, 3>(tma_thr_BpEB.partition_D(gBpEB));
    // (TMA, OP)
    Tensor tOsBpEB = group_modes<0, 3>(tma_thr_BpEB.partition_S(sBpEB));

    StorePipelineState smem_pipe_read_BpEB;
    StorePipelineState smem_pipe_release_BpEB;

    if (cute::elect_one_sync()) {
      constexpr int kPrologueCount = kStagesOut - 1;
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kPrologueCount; ++i) {
        int k_block = k_block_min + i;
        if (k_block < k_block_max) {
          pipeline_BpEB.consumer_wait(smem_pipe_read_BpEB);
          cutlass::arch::fence_view_async_shared();
          copy(params.tma_store_BpEB, tOsBpEB(_, smem_pipe_read_BpEB.index()),
               tOgBpEB(_, k_block));
          tma_store_arrive();
          ++smem_pipe_read_BpEB;
        }
      }

      CUTLASS_PRAGMA_NO_UNROLL
      for (int k_block = k_block_min + kPrologueCount; k_block < k_block_max;
           ++k_block) {
        pipeline_BpEB.consumer_wait(smem_pipe_read_BpEB);
        cutlass::arch::fence_view_async_shared();
        copy(params.tma_store_BpEB, tOsBpEB(_, smem_pipe_read_BpEB.index()),
             tOgBpEB(_, k_block));
        tma_store_arrive();
        ++smem_pipe_read_BpEB;

        // wait on prior completion
        tma_store_wait<kStagesOut - 1>();
        cutlass::arch::fence_view_async_shared();
        pipeline_BpEB.consumer_release(smem_pipe_release_BpEB);
        ++smem_pipe_release_BpEB;
      }
      tma_store_wait<0>();
    }
  }

  // WG1: compute EARxBpEB, wait for BpEB store warp to finish, then store result
  CUTLASS_DEVICE void run_EARxBpEB_consumer(
      Params const& params, MainloopStorePipeline& pipeline_BpEB,
      MainloopLoadPipeline& pipeline_EAR, SharedStorage& shared_storage,
      const int n_block, const int k_block_min, const int k_block_max,
      const int tid) {
    TiledMmaNRK tiled_mma;
    Tensor tCrEARxBpEB =
        partition_fragment_C(tiled_mma, select<0, 1>(TileShape_NRK{}));
    clear(tCrEARxBpEB);

    compute_EARxBpEB(params, pipeline_BpEB, pipeline_EAR, shared_storage,
                     tCrEARxBpEB, n_block, k_block_min, k_block_max, tid);

    // Wait for BpEB store warp to finish before reusing SMEM for EARxBpEB store
    cutlass::arch::NamedBarrier::sync(
        kNumEARxBpEBThreads + kNumBpEBStoreThreads,
        static_cast<uint32_t>(NamedBarriers::EARxBpEBSMEMReady));

    store_EARxBpEB(params, tCrEARxBpEB, shared_storage, n_block, tid);
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
      cute::prefetch_tma_descriptor(params.tma_load_B.get_tma_descriptor());
      cute::prefetch_tma_descriptor(params.tma_load_EBR.get_tma_descriptor());
      cute::prefetch_tma_descriptor(params.tma_load_EBL.get_tma_descriptor());
      cute::prefetch_tma_descriptor(params.tma_store_BpEB.get_tma_descriptor());
      cute::prefetch_tma_descriptor(params.tma_load_EAR.get_tma_descriptor());
      cute::prefetch_tma_descriptor(
          params.tma_store_EARxBpEB.get_tma_descriptor());
    }
    // The matrix B is logically partitioned into n/kblockN tiles in the n direction.
    //  n_block is the index of the tile that this CTA will work with
    int const n_block = blockIdx.x;
    // Each CTA will work with several tiles in the k direction, one in each mainloop step.
    // This is the smallest index of these blocks. If NoReduction,
    //  always equals 0 (each CTA will work on all blocks in the k direction).
    int const k_block_min = blockIdx.y * params.num_k_blocks;

    // In case num_k_blocks doens't divide total_k_blocks
    int const num_k_blocks_cta =
        cute::min(params.num_k_blocks, params.total_k_blocks - k_block_min);
    // Largest k_block index this CTA will work with
    int const k_block_max = k_block_min + num_k_blocks_cta;

    // There's 1 load+wait of EBR per CTA, so we just use an mbarrier, not a pipeline
    uint64_t* EBR_mbar = &shared_storage.EBR_mbar;
    if (lane_predicate) {
      ProducerBarType::init(EBR_mbar, 1);
    }
    cutlass::arch::fence_barrier_init();

    // TMA load pipeline for B: WG0 is producer, WG2 is consumer
    LoadPipelineParams pipeline_params_B;
    pipeline_params_B.transaction_bytes = TmaTransactionBytesB;
    pipeline_params_B.role =
        warp_group_idx == 0 ? MainloopLoadPipeline::ThreadCategory::Producer
        : warp_group_idx == 2
            ? MainloopLoadPipeline::ThreadCategory::Consumer
            : MainloopLoadPipeline::ThreadCategory::NonParticipant;
    pipeline_params_B.is_leader = warp_group_thread_idx == 0;
    pipeline_params_B.num_consumers = kNumMmaThreads;
    MainloopLoadPipeline pipeline_B(shared_storage.pipeline_B,
                                    pipeline_params_B, ClusterShape{});

    // TMA load pipeline for EAR: WG0 is producer, WG1 is consumer
    LoadPipelineParams pipeline_params_EAR;
    pipeline_params_EAR.transaction_bytes = TmaTransactionBytesEAR;
    pipeline_params_EAR.role =
        warp_group_idx == 0 ? MainloopLoadPipeline::ThreadCategory::Producer
        : warp_group_idx == 1
            ? MainloopLoadPipeline::ThreadCategory::Consumer
            : MainloopLoadPipeline::ThreadCategory::NonParticipant;
    pipeline_params_EAR.is_leader = warp_group_thread_idx == 0;
    pipeline_params_EAR.num_consumers = kNumMmaThreads;
    MainloopLoadPipeline pipeline_EAR(shared_storage.pipeline_EAR,
                                      pipeline_params_EAR, ClusterShape{});

    // TMA load pipeline for EBL: WG0 is producer, WG2 is consumer
    LoadPipelineParams pipeline_params_EBL;
    pipeline_params_EBL.transaction_bytes = TmaTransactionBytesEBL;
    pipeline_params_EBL.role =
        warp_group_idx == 0 ? MainloopLoadPipeline::ThreadCategory::Producer
        : warp_group_idx == 2
            ? MainloopLoadPipeline::ThreadCategory::Consumer
            : MainloopLoadPipeline::ThreadCategory::NonParticipant;
    pipeline_params_EBL.is_leader = warp_group_thread_idx == 0;
    pipeline_params_EBL.num_consumers = kNumMmaThreads;
    MainloopLoadPipeline pipeline_EBL(shared_storage.pipeline_EBL,
                                      pipeline_params_EBL, ClusterShape{});

    // Store pipeline for BpEB
    // Producer: WG2 (compute BpEB), Consumers: TMA store warp + WG1 (EARxBpEB)
    StorePipelineParams pipeline_params_BpEB_store;
    pipeline_params_BpEB_store.role =
        warp_group_idx == 2 ? MainloopStorePipeline::ThreadCategory::Producer
        : (warp_idx == 1 || warp_group_idx == 1)
            ? MainloopStorePipeline::ThreadCategory::Consumer
            : MainloopStorePipeline::ThreadCategory::NonParticipant;
    pipeline_params_BpEB_store.producer_arv_count = kNumMmaThreads;  // one WG
    pipeline_params_BpEB_store.consumer_arv_count =
        1 + kNumMmaThreads;  // one thread for TMA store + WG1
    MainloopStorePipeline pipeline_BpEB_store(
        shared_storage.pipeline_BpEB_store, pipeline_params_BpEB_store);

    __syncthreads();

    if (warp_group_idx == 0) {  // Producer
      cutlass::arch::warpgroup_reg_dealloc<kNumLoadRegisters>();

      int warp_idx_in_warpgroup = pearl::warp_idx_in_warpgroup_sync();
      if (warp_idx_in_warpgroup == 0) {  // Load warp
        LoadPipelineState smem_pipe_write_B =
            cutlass::make_producer_start_state<MainloopLoadPipeline>();
        LoadPipelineState smem_pipe_write_EBL =
            cutlass::make_producer_start_state<MainloopLoadPipeline>();
        LoadPipelineState smem_pipe_write_EAR =
            cutlass::make_producer_start_state<MainloopLoadPipeline>();

        load_tensors(params, shared_storage, n_block, k_block_min, k_block_max,
                     pipeline_B, pipeline_EAR, pipeline_EBL, smem_pipe_write_B,
                     smem_pipe_write_EAR, smem_pipe_write_EBL);

      } else if (warp_idx_in_warpgroup == 1) {  // Store BpEB warp
        store_BpEB(params, pipeline_BpEB_store, shared_storage, n_block,
                   k_block_min, k_block_max);
        cutlass::arch::NamedBarrier::arrive(
            kNumEARxBpEBThreads + kNumBpEBStoreThreads,
            static_cast<uint32_t>(NamedBarriers::EARxBpEBSMEMReady));
      }
    } else if (warp_group_idx == 1) {  // EARxBpEB consumer
      constexpr int ThreadOffset = kNumThreadsPerWarpGroup;
      cutlass::arch::warpgroup_reg_alloc<kNumEARxBpEBRegisters>();
      run_EARxBpEB_consumer(params, pipeline_BpEB_store, pipeline_EAR,
                            shared_storage, n_block, k_block_min, k_block_max,
                            tid - ThreadOffset);
    } else if (warp_group_idx == 2) {  // BpEB consumer
      constexpr int ThreadOffset = 2 * kNumThreadsPerWarpGroup;
      cutlass::arch::warpgroup_reg_alloc<kNumBpEBRegisters>();
      compute_BpEB(params, pipeline_B, pipeline_EBL, pipeline_BpEB_store,
                   shared_storage, n_block, k_block_min, k_block_max,
                   tid - ThreadOffset);
    }
  }
};

}  // namespace pearl
