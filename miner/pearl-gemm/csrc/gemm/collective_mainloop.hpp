#pragma once

#include <cutlass/arch/arch.h>
#include <cutlass/arch/barrier.h>
#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>
#include "cutlass/pipeline/pipeline.hpp"

#include "cute/tensor.hpp"

#include "cutlass/gemm/collective/collective_builder.hpp"

#include "named_barrier.hpp"

#include "blake3/blake3_constants.hpp"
#include "host_signal_header.hpp"
#include "pow_utils.hpp"
#include "utils.h"

namespace pearl {

using namespace cute;

#if defined(PEARL_P1K175_NON_CONSUMER_PUBLICATION)
#if !defined(PEARL_P1K148_SCALAR16_DEFERRED_TRANSCRIPT)
#error "PEARL_P1K175_NON_CONSUMER_PUBLICATION requires PEARL_P1K148_SCALAR16_DEFERRED_TRANSCRIPT"
#endif
#if !defined(PEARL_P1K165_TWO_PHASE_POW_CHECK)
#error "PEARL_P1K175_NON_CONSUMER_PUBLICATION requires PEARL_P1K165_TWO_PHASE_POW_CHECK"
#endif
#if !defined(PEARL_P1K165_SINGLE_RECORD_SIDEBAND_STORE)
#error "PEARL_P1K175_NON_CONSUMER_PUBLICATION requires PEARL_P1K165_SINGLE_RECORD_SIDEBAND_STORE"
#endif
#if !defined(PEARL_P1K175_WORD_INDEX)
#define PEARL_P1K175_WORD_INDEX 0
#endif
#if !defined(PEARL_P1K175_STAGE_SLOT)
#define PEARL_P1K175_STAGE_SLOT 0
#endif
#if !defined(PEARL_P1K175_PUBLISHER_CONSUMER)
#define PEARL_P1K175_PUBLISHER_CONSUMER 0
#endif
#if ((defined(PEARL_P1K175_ROW_ACTUAL_SHARED_NONCONSUMER_GLOBAL) ? 1 : 0) + \
     (defined(PEARL_P1K175_ROW_CONST_SHARED_NONCONSUMER_GLOBAL) ? 1 : 0) + \
     (defined(PEARL_P1K175_ROW_NONCONSUMER_CONST_GLOBAL_NO_STAGE) ? 1 : 0)) != 1
#error "PEARL_P1K175_NON_CONSUMER_PUBLICATION requires exactly one P1K175 row macro"
#endif
#if PEARL_P1K175_WORD_INDEX < 0 || PEARL_P1K175_WORD_INDEX > 15
#error "PEARL_P1K175_WORD_INDEX must be in [0,15]"
#endif
#if PEARL_P1K175_STAGE_SLOT < 0
#error "PEARL_P1K175_STAGE_SLOT must be non-negative"
#endif
#if PEARL_P1K175_PUBLISHER_CONSUMER < 0
#error "PEARL_P1K175_PUBLISHER_CONSUMER must be non-negative"
#endif
#if PEARL_P1K175_PUBLISHER_CONSUMER == PEARL_P1K165_SINGLE_RECORD_CONSUMER
#error "PEARL_P1K175_PUBLISHER_CONSUMER must differ from PEARL_P1K165_SINGLE_RECORD_CONSUMER"
#endif
#endif

#if defined(PEARL_P1K176_TOKEN_ONLY_DELAYED_PROOF)
#if !defined(PEARL_P1K148_SCALAR16_DEFERRED_TRANSCRIPT)
#error "PEARL_P1K176_TOKEN_ONLY_DELAYED_PROOF requires PEARL_P1K148_SCALAR16_DEFERRED_TRANSCRIPT"
#endif
#if !defined(PEARL_P1K165_TWO_PHASE_POW_CHECK)
#error "PEARL_P1K176_TOKEN_ONLY_DELAYED_PROOF requires PEARL_P1K165_TWO_PHASE_POW_CHECK"
#endif
#if !defined(PEARL_P1K165_SINGLE_RECORD_SIDEBAND_STORE)
#error "PEARL_P1K176_TOKEN_ONLY_DELAYED_PROOF requires PEARL_P1K165_SINGLE_RECORD_SIDEBAND_STORE"
#endif
#if defined(PEARL_P1K174_STAGED_DECOUPLED_STORE) || \
    defined(PEARL_P1K175_NON_CONSUMER_PUBLICATION) || \
    defined(PEARL_P1K172_ACTUAL_STORE_WORDS) || \
    defined(PEARL_P1K171_DUMMY_SIDEBAND_VALUES) || \
    defined(PEARL_P1K171_SIDEBAND_BISECT)
#error "PEARL_P1K176_TOKEN_ONLY_DELAYED_PROOF must run without publication-bisect branch macros"
#endif
#if !defined(PEARL_P1K176_TOKEN_SCHEMA_VERSION)
#define PEARL_P1K176_TOKEN_SCHEMA_VERSION 1
#endif
#endif

#if defined(PEARL_P1K179_BLIND_SMEM_HARVEST)
#if !defined(PEARL_P1K148_SCALAR16_DEFERRED_TRANSCRIPT)
#error "PEARL_P1K179_BLIND_SMEM_HARVEST requires PEARL_P1K148_SCALAR16_DEFERRED_TRANSCRIPT"
#endif
#if !defined(PEARL_P1K165_TWO_PHASE_POW_CHECK)
#error "PEARL_P1K179_BLIND_SMEM_HARVEST requires PEARL_P1K165_TWO_PHASE_POW_CHECK"
#endif
#if !defined(PEARL_P1K165_SINGLE_RECORD_SIDEBAND_STORE)
#error "PEARL_P1K179_BLIND_SMEM_HARVEST requires PEARL_P1K165_SINGLE_RECORD_SIDEBAND_STORE"
#endif
#if defined(PEARL_P1K174_STAGED_DECOUPLED_STORE) || \
    defined(PEARL_P1K175_NON_CONSUMER_PUBLICATION) || \
    defined(PEARL_P1K176_TOKEN_ONLY_DELAYED_PROOF) || \
    defined(PEARL_P1K172_ACTUAL_STORE_WORDS) || \
    defined(PEARL_P1K171_DUMMY_SIDEBAND_VALUES) || \
    defined(PEARL_P1K171_SIDEBAND_BISECT)
#error "PEARL_P1K179_BLIND_SMEM_HARVEST must run without prior publication-bisect branch macros"
#endif
#if defined(PEARL_P1K179B_STMATRIX_BITCAST)
#error "PEARL_P1K179B_STMATRIX_BITCAST is intentionally not implemented; see the P1K179 runbook for the stmatrix b16/recast risks"
#endif
#if !defined(PEARL_P1K179_ACCUM_INDEX)
#define PEARL_P1K179_ACCUM_INDEX 0
#endif
#if !defined(PEARL_P1K179_WORD_COUNT)
#define PEARL_P1K179_WORD_COUNT 1
#endif
#if !defined(PEARL_P1K179_STAGE_SLOT)
#define PEARL_P1K179_STAGE_SLOT 0
#endif
#if ((defined(PEARL_P1K179A_VOLATILE_SHARED_STORE) ? 1 : 0) + \
     (defined(PEARL_P1K179C_CONST_GLOBAL_AFTER_SHARED_HARVEST) ? 1 : 0)) != 1
#error "PEARL_P1K179_BLIND_SMEM_HARVEST requires exactly one P1K179 row macro"
#endif
#if PEARL_P1K179_ACCUM_INDEX < 0 || PEARL_P1K179_ACCUM_INDEX > 15
#error "PEARL_P1K179_ACCUM_INDEX must be in [0,15]"
#endif
#if PEARL_P1K179_WORD_COUNT < 1 || PEARL_P1K179_WORD_COUNT > 4
#error "PEARL_P1K179_WORD_COUNT must be in [1,4]"
#endif
#if PEARL_P1K179_ACCUM_INDEX + PEARL_P1K179_WORD_COUNT > 16
#error "PEARL_P1K179_ACCUM_INDEX + PEARL_P1K179_WORD_COUNT must stay within 16 accumulator words"
#endif
#if PEARL_P1K179_STAGE_SLOT < 0 || PEARL_P1K179_STAGE_SLOT + PEARL_P1K179_WORD_COUNT > 15
#error "PEARL_P1K179_STAGE_SLOT leaves no room for harvested word(s) plus marker"
#endif
#endif

template <typename KTraits>
struct CollectiveMainloop {

  using ElementIn = typename KTraits::ElementIn;
  using TileShape_MNK = typename KTraits::TileShape_MNK;
  using TileShape_MNR = typename KTraits::TileShape_MNR;

  using ProblemShape = typename KTraits::ProblemShape;
  using ClusterShape_MNK = typename KTraits::ClusterShape_MNK;

  static constexpr int kStages = KTraits::kStages;
  static constexpr int SkipReduction = KTraits::SkipReduction;
  static constexpr int kClusterSizeM = KTraits::kClusterSizeM;
  static constexpr int kClusterSizeN = KTraits::kClusterSizeN;
  static constexpr int srcLane = KTraits::srcLane;

  using MMAAtom_K = typename KTraits::MMAAtom_K;

  using SmemLayoutA = typename KTraits::SmemLayoutA;
  using SmemLayoutB = typename KTraits::SmemLayoutB;

  using ShapeT = cute::Shape<int32_t, int32_t>;
  using StrideT = cute::Shape<int32_t, _1>;
  using LayoutT = cute::Layout<ShapeT, StrideT>;
  using TMAOpA = KTraits::TMAOpA;
  using TMAOpB = KTraits::TMAOpB;

  // mcast in n direction of cluster
  using TMA_A = decltype(make_tma_copy(
      TMAOpA{},
      make_tensor(make_gmem_ptr(static_cast<ElementIn const*>(nullptr)),
                  ShapeT{}, StrideT{}),
      take<0, 2>(SmemLayoutA{}), select<0, 2>(TileShape_MNK{}), kClusterSizeN));

  // mcast in m direction of cluster
  using TMA_B = decltype(make_tma_copy(
      TMAOpB{},
      make_tensor(make_gmem_ptr(static_cast<ElementIn const*>(nullptr)),
                  ShapeT{}, StrideT{}),
      take<0, 2>(SmemLayoutB{}), select<1, 2>(TileShape_MNK{}), kClusterSizeM));

  static constexpr int kNumMmaThreads = KTraits::kNumMmaThreads;
  using MainloopPipeline = typename KTraits::MainloopPipeline;
  using PipelineParams = typename MainloopPipeline::Params;
  using PipelineState = typename MainloopPipeline::PipelineState;
  using BarrierType = typename MainloopPipeline::ProducerBarrierType;

  // Set the bytes transferred in this TMA transaction (may involve multiple issues)
  static constexpr uint32_t TmaTransactionBytesA = static_cast<uint32_t>(
      size(take<0, 2>(SmemLayoutA{})) * cutlass::sizeof_bits_v<ElementIn> / 8);
  static constexpr uint32_t TmaTransactionBytesB = static_cast<uint32_t>(
      size(take<0, 2>(SmemLayoutB{})) * cutlass::sizeof_bits_v<ElementIn> / 8);
  static constexpr uint32_t TmaTransactionBytes =
      TmaTransactionBytesA + TmaTransactionBytesB;

  struct Arguments {
    ElementIn const* ptr_A;
    ElementIn const* ptr_B;
    void* host_signal_header_pinned;
    void* host_signal_sync;
    ProblemShape const problem_shape;
    uint64_t* inner_hash_counter;
    bool coalesce_receipts;
	    bool enable_dummy_reduction;
    uint32_t* ptr_global_sideband_journal;
    int global_sideband_tiles;
    int global_sideband_consumers;
    int global_sideband_boundaries;
	    bool enable_panel_partial_transcript;
    PanelPartialTranscriptRecordV2* ptr_panel_partial_transcripts;
    int panel_partial_transcript_capacity;
    int panel_partial_panel_count;
    uint32_t const* ptr_pow_target;
    uint32_t const* ptr_pow_key;
  };

  struct Params {
    ElementIn const* ptr_A;  // needed for host signal
    ElementIn const* ptr_B;  // needed for host signal
    LayoutT layout_A;
    LayoutT layout_B;
    TMA_A tma_load_A;
    TMA_B tma_load_B;
    HostSignalHeader* host_signal_header_pinned;
    HostSignalSync* host_signal_sync;
    ProblemShape const problem_shape;
    uint64_t* inner_hash_counter;
    bool coalesce_receipts;
	    bool enable_dummy_reduction;
    uint32_t* ptr_global_sideband_journal;
    int global_sideband_tiles;
    int global_sideband_consumers;
    int global_sideband_boundaries;
	    bool enable_panel_partial_transcript;
    PanelPartialTranscriptRecordV2* ptr_panel_partial_transcripts;
    int panel_partial_transcript_capacity;
    int panel_partial_panel_count;
    uint32_t const* ptr_pow_target;
    uint32_t const* ptr_pow_key;
  };

  static Params to_underlying_arguments(Arguments const& args) {
    auto [M, N, K, R] = args.problem_shape;
    LayoutT layout_A = make_layout(make_shape(M, K), make_stride(K, _1{}));
    LayoutT layout_B = make_layout(make_shape(N, K), make_stride(K, _1{}));
    Tensor mA = make_tensor(make_gmem_ptr(args.ptr_A), layout_A);
    Tensor mB = make_tensor(make_gmem_ptr(args.ptr_B), layout_B);
    // tile is divided into kClusterSizeN or kClusterSizeM many pieces to be multicasted
    // mcast in n direction of cluster
    TMA_A tma_load_A =
        make_tma_copy(TMAOpA{}, mA, SmemLayoutA{}(_, _, _0{}),
                      select<0, 2>(TileShape_MNK{}), kClusterSizeN);
    // mcast in m direction of cluster
    TMA_B tma_load_B =
        make_tma_copy(TMAOpB{}, mB, SmemLayoutB{}(_, _, _0{}),
                      select<1, 2>(TileShape_MNK{}), kClusterSizeM);

    return {.ptr_A = args.ptr_A,
            .ptr_B = args.ptr_B,
            .layout_A = layout_A,
            .layout_B = layout_B,
            .tma_load_A = tma_load_A,
            .tma_load_B = tma_load_B,
            .host_signal_header_pinned = reinterpret_cast<HostSignalHeader*>(
                args.host_signal_header_pinned),
            .host_signal_sync =
                reinterpret_cast<HostSignalSync*>(args.host_signal_sync),
            .problem_shape = args.problem_shape,
            .inner_hash_counter = args.inner_hash_counter,
	            .coalesce_receipts = args.coalesce_receipts,
	            .enable_dummy_reduction = args.enable_dummy_reduction,
            .ptr_global_sideband_journal = args.ptr_global_sideband_journal,
            .global_sideband_tiles = args.global_sideband_tiles,
            .global_sideband_consumers = args.global_sideband_consumers,
            .global_sideband_boundaries = args.global_sideband_boundaries,
	            .enable_panel_partial_transcript =
	                args.enable_panel_partial_transcript,
            .ptr_panel_partial_transcripts =
                args.ptr_panel_partial_transcripts,
            .panel_partial_transcript_capacity =
                args.panel_partial_transcript_capacity,
            .panel_partial_panel_count = args.panel_partial_panel_count,
            .ptr_pow_target = args.ptr_pow_target,
            .ptr_pow_key = args.ptr_pow_key};
  }

  /// Issue Tma Descriptor Prefetch -- ideally from a single thread for best performance
  CUTLASS_DEVICE
  static void prefetch_tma_descriptors(Params const& mainloop_params) {
    cute::prefetch_tma_descriptor(
        mainloop_params.tma_load_A.get_tma_descriptor());
    cute::prefetch_tma_descriptor(
        mainloop_params.tma_load_B.get_tma_descriptor());
  }

  template <typename SharedStorage>
  CUTLASS_DEVICE void load(Params const& mainloop_params,
                           MainloopPipeline pipeline,
                           PipelineState& smem_pipe_write,
                           SharedStorage& shared_storage,
                           cute::tuple<int32_t, int32_t, int32_t> block_coord,
                           int k_tile_count, uint16_t const tma_mcast_mask_a,
                           uint16_t const tma_mcast_mask_b) {

    // Fetch logical block coordinates
    auto [m_block, n_block, bidb] = block_coord;

    // Define SMEM tensors
    Tensor sA = make_tensor(make_smem_ptr(shared_storage.smem_A.data()),
                            SmemLayoutA{});  // (BLK_M,BLK_K,PIPE)
    Tensor sB = make_tensor(make_smem_ptr(shared_storage.smem_B.data()),
                            SmemLayoutB{});  // (BLK_N,BLK_K,PIPE)

    // Define GMEM tensors as TMA tensors
    Tensor mA = mainloop_params.tma_load_A.get_tma_tensor(
        mainloop_params.layout_A.shape());
    Tensor mB = mainloop_params.tma_load_B.get_tma_tensor(
        mainloop_params.layout_B.shape());

    // Get CTA views of GMEM
    Tensor gA = local_tile(mA, select<0, 2>(TileShape_MNK{}),
                           make_coord(m_block, _));  // (BLK_M,BLK_K,k)
    Tensor gB = local_tile(mB, select<1, 2>(TileShape_MNK{}),
                           make_coord(n_block, _));  // (BLK_N,BLK_K,k)

    // Partition the copying of A and B tiles, including which part of the tile this
    //  CTA is responsible for when participating in multicast
    auto [tAgA, tAsA] =
        tma_partition(mainloop_params.tma_load_A, get<1>(block_id_in_cluster()),
                      make_layout(kClusterSizeN), group_modes<0, 2>(sA),
                      group_modes<0, 2>(gA));  // (TMA,k) and (TMA,PIPE)
    auto [tBgB, tBsB] =
        tma_partition(mainloop_params.tma_load_B, get<0>(block_id_in_cluster()),
                      make_layout(kClusterSizeM), group_modes<0, 2>(sB),
                      group_modes<0, 2>(gB));  // (TMA,k) and (TMA,PIPE)
    // DO TMA LOAD from a single thread
    int lane_predicate = cute::elect_one_sync();

    if constexpr (!KTraits::SkipDenoising) {
      // Wait for EAxBpEB matmul to finish on previous tile before loading current tile A, B
      cutlass::arch::NamedBarrier::sync(
          kNumMmaThreads + cutlass::NumThreadsPerWarp,
          static_cast<cutlass::arch::ReservedNamedBarriers>(
              pearl::NamedBarriers::DenoiseComplete));
    }

    if (lane_predicate) {
      // MAINLOOP LOADS
      CUTLASS_PRAGMA_NO_UNROLL
      for (int k_tile = 0; k_tile < k_tile_count; ++k_tile) {
        pipeline.producer_acquire(smem_pipe_write);
        BarrierType* tmaBar = pipeline.producer_get_barrier(smem_pipe_write);
        auto stage = smem_pipe_write.index();
        copy(mainloop_params.tma_load_A.with(*tmaBar, tma_mcast_mask_a),
             tAgA(_, k_tile), tAsA(_, stage));
        copy(mainloop_params.tma_load_B.with(*tmaBar, tma_mcast_mask_b),
             tBgB(_, k_tile), tBsB(_, stage));
        pipeline.producer_commit(smem_pipe_write, TmaTransactionBytes);
        ++smem_pipe_write;
      }
    }
  }

  /// Perform a Producer Epilogue to prevent early exit of blocks in a Cluster
  CUTLASS_DEVICE void load_tail(MainloopPipeline pipeline,
                                PipelineState& smem_pipe_write) {
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
      pipeline.producer_tail(smem_pipe_write);
    }
  }

  CUTLASS_DEVICE void mma_init() {
    if constexpr (!KTraits::SkipDenoising) {
      // Allow producer warp to issue initial loads of A and B
      cutlass::arch::NamedBarrier::arrive(
          kNumMmaThreads + cutlass::NumThreadsPerWarp,
          static_cast<cutlass::arch::ReservedNamedBarriers>(
              pearl::NamedBarriers::DenoiseComplete));
    }
  }

  template <typename SharedStorage, typename FrgTensorC,
            typename TranscriptTensor, typename BlockCoord>
  CUTLASS_DEVICE void mma(Params const& mainloop_params,
                          MainloopPipeline pipeline,
                          PipelineState& smem_pipe_read, FrgTensorC& tCrC,
                          TranscriptTensor& transcript_extraction_tensor,
                          bool& block_found, int& block_found_k_tile,
                          int thread_idx, SharedStorage& shared_storage,
                          BlockCoord const& block_coord,
                          int k_tile_count) {

    Tensor sA =
        make_tensor(make_smem_ptr(shared_storage.smem_A.data()), SmemLayoutA{});
    Tensor sB =
        make_tensor(make_smem_ptr(shared_storage.smem_B.data()), SmemLayoutB{});

    typename KTraits::TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(thread_idx);

    Tensor tCsA = thr_mma.partition_A(sA);  // (MMA,MMA_M,MMA_K,PIPE)
    Tensor tCsB = thr_mma.partition_B(sB);  // (MMA,MMA_N,MMA_K,PIPE)

    // Allocate "fragments" -- these are WGMMA matrix descriptors
    Tensor tCrA = thr_mma.make_fragment_A(tCsA);  // (MMA,MMA_M,MMA_K,PIPE)
    Tensor tCrB = thr_mma.make_fragment_B(tCsB);  // (MMA,MMA_N,MMA_K,PIPE)
    constexpr int k_blocks_per_tile = size<2>(tCrA);
#if defined(PEARL_P1K111_NATIVE_SIDEBAND_ORACLE) || \
    defined(PEARL_P1K112_NATIVE_SIDEBAND_FILL_ONLY) || \
    defined(PEARL_P1K113_NATIVE_SIDEBAND_CONSUMER_SYNTH_CHECK) || \
    defined(PEARL_P1K114_NATIVE_SIDEBAND_CONSUMER_FILL_ONLY) || \
    defined(PEARL_P1K115_NATIVE_SIDEBAND_FILL_SWEEP) || \
    defined(PEARL_P1K116_NATIVE_SIDEBAND_CHECK_SWEEP)
    constexpr bool kNativeSidebandOracle = KTraits::EnableNative2x64Ring;
#else
    constexpr bool kNativeSidebandOracle = false;
#endif
#if defined(PEARL_P1K150_SCALAR16_FINAL_GLOBAL_STORE) || \
    defined(PEARL_P1K154_SCALAR16_FINAL_SHARED_STORE) || \
    defined(PEARL_P1K165_TWO_PHASE_POW_CHECK)
    constexpr bool kNativeGlobalJournalFillProducerPath = false;
#else
    constexpr bool kNativeGlobalJournalFillProducerPath =
        KTraits::EnableNativeGlobalJournalFill;
#endif
    constexpr bool kElideConsumerTranscript =
        native_global_journal_elides_consumer_transcript<KTraits>();

    if constexpr (KTraits::EnableRawRing2x64 ||
		                  KTraits::EnableRawXorOnly2x64 ||
		                  KTraits::EnableRawStoreOnly2x64 ||
		                  KTraits::EnableRawDeferredRing2x64 ||
		                  KTraits::EnableRawGlobalSink2x64 ||
	                  KTraits::EnableNative2x64Ring ||
	                  kNativeGlobalJournalFillProducerPath) {
      uint32_t deferred_xq = 0;
      CUTLASS_PRAGMA_NO_UNROLL
      for (int k_tile = 0; k_tile < k_tile_count; ++k_tile) {
        pipeline.consumer_wait(smem_pipe_read);
        auto stage = smem_pipe_read.index();

        CUTLASS_PRAGMA_UNROLL
        for (int k_block = 0; k_block < k_blocks_per_tile; ++k_block) {
          warpgroup_fence_operand(tCrC);
          warpgroup_arrive();
          gemm(tiled_mma, tCrA(_, _, k_block, stage),
               tCrB(_, _, k_block, stage), tCrC);
          warpgroup_commit_batch();
        }

        // R128/bK128 default 2x64 proof boundaries land at tile end. These
        // diagnostics bisect the cost of accumulator XOR versus the shared-ring
        // handoff shape without constructing TileHashAccumulator state.
        warpgroup_wait<0>();
        if constexpr (KTraits::EnableRawGlobalSink2x64) {
          warpgroup_fence_operand(tCrC);
          uint32_t const cell = tCrC[0];
          if (thread_idx == 0 && k_tile == k_tile_count - 1 &&
              mainloop_params.inner_hash_counter != nullptr) {
            uint64_t const global_sink_addr =
                reinterpret_cast<uint64_t>(mainloop_params.inner_hash_counter);
            asm volatile("st.global.u32 [%0], %1;"
                         :
                         : "l"(global_sink_addr), "r"(cell));
          }
        } else if constexpr (KTraits::EnableRawXorOnly2x64 ||
                             KTraits::EnableRawRing2x64 ||
                             KTraits::EnableRawDeferredRing2x64 ||
                             (KTraits::EnableNative2x64Ring &&
                              !kNativeSidebandOracle)) {
          warpgroup_fence_operand(tCrC);
          uint32_t const xq = xor_reduction(tCrC);
		          if constexpr (KTraits::EnableRawRing2x64 ||
		                        KTraits::EnableNative2x64Ring) {
		            if (thread_idx == 0 && k_tile == k_tile_count - 1) {
		              auto* raw_ring =
		                  reinterpret_cast<uint32_t*>(shared_storage.xq_journal.data());
		              int const offset = 0;
#if defined(PEARL_P1K123_RAW_RING_STORE_ORDINARY)
		              uint32_t const ordinary_value =
		                  (uint32_t(thread_idx) << 8) ^ uint32_t(k_tile);
		              raw_ring[offset] = ordinary_value;
		              asm volatile("" : : "r"(xq), "r"(ordinary_value), "r"(offset));
#else
		              raw_ring[offset] = xq;
		              asm volatile("" : : "r"(xq), "r"(offset));
#endif
		            }
			          } else if constexpr (KTraits::EnableRawDeferredRing2x64) {
			            deferred_xq ^= xq + 0x9e3779b9u + uint32_t(k_tile);
			          } else {
	            asm volatile("" : : "r"(xq) : "memory");
	          }
        } else if constexpr (KTraits::EnableRawStoreOnly2x64) {
          if (k_tile < KTraits::kXqJournalMaxBoundaries) {
            volatile uint32_t* raw_ring =
                reinterpret_cast<volatile uint32_t*>(
                    shared_storage.xq_journal.data());
            int const offset =
                thread_idx * KTraits::kXqJournalMaxBoundaries + k_tile;
            uint32_t const ordinary_value =
                (uint32_t(thread_idx) << 8) ^ uint32_t(k_tile);
            raw_ring[offset] = ordinary_value;
            asm volatile("" : : "r"(ordinary_value), "r"(offset) : "memory");
          }
        }

        pipeline.consumer_release(smem_pipe_read);
        ++smem_pipe_read;
      }

		      if constexpr (KTraits::EnableRawDeferredRing2x64) {
#if defined(PEARL_P1K123_DEFERRED_REG_ONLY)
		        asm volatile("" : : "r"(deferred_xq) : "memory");
#else
		        volatile uint32_t* raw_ring = reinterpret_cast<volatile uint32_t*>(
		            shared_storage.xq_journal.data());
		        int const offset = thread_idx * KTraits::kXqJournalMaxBoundaries;
		        raw_ring[offset] = deferred_xq;
		        asm volatile("" : : "r"(deferred_xq), "r"(offset) : "memory");
#endif
		      }
	    } else if constexpr (KTraits::EnableDummyReduction) {
      CUTLASS_PRAGMA_NO_UNROLL
      for (int k_tile = 0; k_tile < k_tile_count; ++k_tile) {
        pipeline.consumer_wait(smem_pipe_read);
        auto stage = smem_pipe_read.index();

        CUTLASS_PRAGMA_UNROLL
        for (int k_block = 0; k_block < k_blocks_per_tile; ++k_block) {
          warpgroup_fence_operand(tCrC);
          warpgroup_arrive();
          gemm(tiled_mma, tCrA(_, _, k_block, stage),
               tCrB(_, _, k_block, stage), tCrC);
          warpgroup_commit_batch();
        }

        warpgroup_wait<0>();
        pipeline.consumer_release(smem_pipe_read);
        ++smem_pipe_read;
      }

    } else {
#if !defined(PEARL_P1K131_NO_PANEL_PARTIAL)
    Tensor cD = make_identity_tensor(select<0, 1>(TileShape_MNK{}));
    Tensor tCcD = thr_mma.partition_C(cD);
#endif

    const uint32_t last_full_k_block =
        shape<1>(mainloop_params.layout_A) / MMAAtom_K{};

    // Compile-time constants for tile hash accumulation
    // R/32
    constexpr int reduce_every_k = get<2>(TileShape_MNR{}) / MMAAtom_K{};

#if !defined(PEARL_P1K128_SINGLE_REGISTER_REDUCTION) && \
    !defined(PEARL_P1K130_CELL_ONLY_REDUCTION)
    using HashAccumulator =
        cute::conditional_t<kElideConsumerTranscript, NullTranscriptAccumulator,
                            TileHashAccumulator<k_blocks_per_tile,
                                                reduce_every_k,
                                                KTraits::EnableDebug,
                                                KTraits::EnableDummyReduction>>;
    HashAccumulator hash_accumulator(last_full_k_block,
                                     mainloop_params.inner_hash_counter);
#endif
	    using XqJournalAccumulator =
	        TileXqJournalAccumulator<k_blocks_per_tile, reduce_every_k,
	                                 KTraits::kXqJournalMaxBoundaries,
	                                 KTraits::EnableDebug>;
	    XqJournalAccumulator xq_journal_accumulator(
	        last_full_k_block, mainloop_params.inner_hash_counter);
    using CanonicalSlotAccumulator =
        CanonicalSlotStreamAccumulator<k_blocks_per_tile, reduce_every_k,
                                       KTraits::EnableDebug>;
    CanonicalSlotAccumulator canonical_slot_accumulator(
        last_full_k_block, mainloop_params.inner_hash_counter);
#if !defined(PEARL_P1K131_NO_PANEL_PARTIAL)
    using PanelPartialAccumulator =
        cute::conditional_t<kElideConsumerTranscript, NullTranscriptAccumulator,
                            TileSelectedTranscriptAccumulator<
                                k_blocks_per_tile, reduce_every_k,
                                KTraits::EnableDebug>>;
    PanelPartialAccumulator panel_partial_accumulator(
        last_full_k_block, mainloop_params.inner_hash_counter);
#endif

    if constexpr (KTraits::EnableCanonicalTranscript) {
      clear_canonical_transcript_slots(shared_storage.canonical_transcript.data(),
                                       thread_idx);
    }

    constexpr bool kTileEndHashSnapshot =
        !SkipReduction && !KTraits::EnableXqJournal &&
        !KTraits::EnableCanonicalTranscript && !KTraits::EnableDummyReduction &&
        !kElideConsumerTranscript &&
        (reduce_every_k == k_blocks_per_tile);
	    constexpr bool kTileEndCanonicalTranscriptSnapshot =
	        !SkipReduction && !KTraits::EnableXqJournal &&
	        KTraits::EnableCanonicalTranscript &&
	        (reduce_every_k == k_blocks_per_tile);

#if defined(PEARL_P1K128_SINGLE_REGISTER_REDUCTION) || \
    defined(PEARL_P1K130_CELL_ONLY_REDUCTION)
		    uint32_t p1k_single_register_state = 0;
#elif defined(PEARL_P1K134_STRUCT_FREE_R128_HASH) || \
    defined(PEARL_P1K135_PRELOAD_WRITEBACK_ONLY) || \
    defined(PEARL_P1K136_HASH_INDEPENDENT_WRITEBACK) || \
    defined(PEARL_P1K137_ASM_DECOUPLED_HASH_WRITEBACK)
		    uint32_t p1k134_tile_hash_state = 0;
		    uint32_t p1k134_reduction_count = 0;
#elif defined(PEARL_P1K148_SCALAR16_DEFERRED_TRANSCRIPT)
    uint32_t p1k148_t0 = 0;
    uint32_t p1k148_t1 = 0;
    uint32_t p1k148_t2 = 0;
    uint32_t p1k148_t3 = 0;
    uint32_t p1k148_t4 = 0;
    uint32_t p1k148_t5 = 0;
    uint32_t p1k148_t6 = 0;
    uint32_t p1k148_t7 = 0;
    uint32_t p1k148_t8 = 0;
    uint32_t p1k148_t9 = 0;
    uint32_t p1k148_t10 = 0;
    uint32_t p1k148_t11 = 0;
    uint32_t p1k148_t12 = 0;
    uint32_t p1k148_t13 = 0;
    uint32_t p1k148_t14 = 0;
    uint32_t p1k148_t15 = 0;
    uint32_t p1k148_reduction_count = 0;
#endif

		    CUTLASS_PRAGMA_NO_UNROLL
		    for (int k_tile = 0; k_tile < k_tile_count; ++k_tile) {
	      if constexpr (!SkipReduction && !KTraits::EnableXqJournal &&
	                    !KTraits::EnableCanonicalTranscript &&
	                    !KTraits::EnableDummyReduction &&
	                    !kElideConsumerTranscript) {
#if defined(PEARL_P1K128_SINGLE_REGISTER_REDUCTION) || \
    defined(PEARL_P1K130_CELL_ONLY_REDUCTION)
		        // Keep the real branch shape, but do not make TileHashAccumulator
		        // state live in the single-register diagnostic.
#elif defined(PEARL_P1K134_STRUCT_FREE_R128_HASH) || \
    defined(PEARL_P1K135_PRELOAD_WRITEBACK_ONLY) || \
    defined(PEARL_P1K136_HASH_INDEPENDENT_WRITEBACK) || \
    defined(PEARL_P1K137_ASM_DECOUPLED_HASH_WRITEBACK)
		        if constexpr (kTileEndHashSnapshot) {
		          p1k134_tile_hash_state =
		              transcript_extraction_tensor(p1k134_reduction_count);
		        } else {
		          hash_accumulator.preload(transcript_extraction_tensor);
		        }
#elif defined(PEARL_P1K148_SCALAR16_DEFERRED_TRANSCRIPT)
            // P1K-148 keeps transcript words in scalar registers for the
            // whole output tile. Avoid publishing accumulator-derived values
            // into the transcript tensor between WGMMA tiles.
#elif defined(PEARL_P1K127_NO_PRELOAD_REGISTER_ONLY)
		        hash_accumulator.init_zero_state();
#else
	        hash_accumulator.preload(transcript_extraction_tensor);
#endif
	      }

      // Wait for TMA to load this stage of the pipeline
      pipeline.consumer_wait(smem_pipe_read);
      auto stage = smem_pipe_read.index();

      CUTLASS_PRAGMA_UNROLL
      for (int k_block = 0; k_block < k_blocks_per_tile; ++k_block) {
        warpgroup_fence_operand(tCrC);
        warpgroup_arrive();
        // WGMMA with dispatch mode (V,M,K) x (V,N,K) => (V,M,N)
        gemm(tiled_mma, tCrA(_, _, k_block, stage), tCrB(_, _, k_block, stage),
             tCrC);
        warpgroup_commit_batch();

		        if constexpr (!SkipReduction) {
#if !defined(PEARL_P1K131_NO_PANEL_PARTIAL)
		          if (mainloop_params.enable_panel_partial_transcript) {
		            panel_partial_accumulator.accumulate(
		                tCrC, tCcD, /*row_start=*/0, /*row_count=*/1,
		                /*col_start=*/0, /*col_count=*/128);
		          }
#endif
		          if constexpr (KTraits::EnableXqJournal) {
	            xq_journal_accumulator.accumulate(
	                tCrC, thread_idx, shared_storage.xq_journal.data());
          } else if constexpr (KTraits::EnableCanonicalTranscript) {
            if constexpr (!kTileEndCanonicalTranscriptSnapshot) {
              canonical_slot_accumulator.accumulate(
                  tCrC, thread_idx, shared_storage.canonical_transcript.data());
            }
          } else {
		            if constexpr (kTileEndHashSnapshot) {
		              // The default R128/bK128 proof boundary is exactly tile-end.
		              // Defer the accumulator read until after the loop so it shares
		              // the tile-end wait/fence instead of draining the WGMMA group
		              // from inside the k_block body.
#if !defined(PEARL_P1K131_NO_PANEL_PARTIAL)
		              if (mainloop_params.enable_panel_partial_transcript) {
		                hash_accumulator.accumulate(tCrC, k_block);
		              }
#endif
		            } else {
	              hash_accumulator.accumulate(tCrC, k_block);
	            }
	          }
	        }
	      }

	      bool waited_for_wgmma = false;
      if constexpr (kTileEndHashSnapshot) {
#if defined(PEARL_P1K131_NO_PANEL_PARTIAL)
        {
#else
        if (!mainloop_params.enable_panel_partial_transcript) {
#endif
#if defined(PEARL_P1K179_BLIND_SMEM_HARVEST)
          if constexpr (KTraits::ProofOnly) {
            if (k_tile == k_tile_count - 1) {
              auto [p1k179_M, p1k179_N, p1k179_K, p1k179_R] =
                  mainloop_params.problem_shape;
              (void)p1k179_M;
              (void)p1k179_K;
              (void)p1k179_R;
              int const p1k179_num_blocks_n =
                  (p1k179_N + KTraits::bN - 1) / KTraits::bN;
              int const p1k179_tile_id =
                  int(get<0>(block_coord)) * p1k179_num_blocks_n +
                  int(get<1>(block_coord));
              if (p1k179_tile_id == PEARL_P1K165_SINGLE_RECORD_TILE_ID &&
                  thread_idx == PEARL_P1K165_SINGLE_RECORD_CONSUMER) {
                volatile uint32_t* p1k179_stage =
                    reinterpret_cast<volatile uint32_t*>(
                        shared_storage.p1k179_stage.data());
                uint32_t const p1k179_w0 =
                    uint32_t(tCrC[PEARL_P1K179_ACCUM_INDEX + 0]);
                p1k179_stage[PEARL_P1K179_STAGE_SLOT + 0] = p1k179_w0;
#if PEARL_P1K179_WORD_COUNT > 1
                uint32_t const p1k179_w1 =
                    uint32_t(tCrC[PEARL_P1K179_ACCUM_INDEX + 1]);
                p1k179_stage[PEARL_P1K179_STAGE_SLOT + 1] = p1k179_w1;
#endif
#if PEARL_P1K179_WORD_COUNT > 2
                uint32_t const p1k179_w2 =
                    uint32_t(tCrC[PEARL_P1K179_ACCUM_INDEX + 2]);
                p1k179_stage[PEARL_P1K179_STAGE_SLOT + 2] = p1k179_w2;
#endif
#if PEARL_P1K179_WORD_COUNT > 3
                uint32_t const p1k179_w3 =
                    uint32_t(tCrC[PEARL_P1K179_ACCUM_INDEX + 3]);
                p1k179_stage[PEARL_P1K179_STAGE_SLOT + 3] = p1k179_w3;
#endif
                p1k179_stage[PEARL_P1K179_STAGE_SLOT +
                              PEARL_P1K179_WORD_COUNT] =
                    0x179a0000u ^ uint32_t(p1k179_tile_id) ^
                    (uint32_t(thread_idx) << 8) ^
                    uint32_t(PEARL_P1K179_WORD_COUNT);
                asm volatile("" : : "r"(p1k179_w0), "r"(thread_idx)
                             : "memory");
              }
            }
          }
#endif
		          warpgroup_wait<0>();
		          warpgroup_fence_operand(tCrC);
		          waited_for_wgmma = true;
#if defined(PEARL_P1K130_CELL_ONLY_REDUCTION)
	          uint32_t const p1k_cell_hash = uint32_t(tCrC[0]);
	          p1k_single_register_state =
	              rotl_xor<HASH_ACCUMULATE_ROTATION>(
	                  p1k_single_register_state,
	                  p1k_cell_hash ^ uint32_t(k_tile));
#elif defined(PEARL_P1K128_SINGLE_REGISTER_REDUCTION)
		          uint32_t const p1k128_hash = xor_reduction(tCrC);
		          p1k_single_register_state =
		              rotl_xor<HASH_ACCUMULATE_ROTATION>(
		                  p1k_single_register_state,
		                  p1k128_hash ^ uint32_t(k_tile));
#elif defined(PEARL_P1K134_STRUCT_FREE_R128_HASH)
		          uint32_t const p1k134_hash = xor_reduction(tCrC);
		          p1k134_tile_hash_state =
		              rotl_xor<HASH_ACCUMULATE_ROTATION>(
		                  p1k134_tile_hash_state, p1k134_hash);
#elif defined(PEARL_P1K137_ASM_DECOUPLED_HASH_WRITEBACK)
		          uint32_t const p1k137_hash = xor_reduction(tCrC);
		          p1k134_tile_hash_state =
		              rotl_xor<HASH_ACCUMULATE_ROTATION>(
		                  p1k134_tile_hash_state, p1k137_hash);
#elif defined(PEARL_P1K135_PRELOAD_WRITEBACK_ONLY)
		          asm volatile("" : : "r"(p1k134_tile_hash_state) : "memory");
#elif defined(PEARL_P1K136_HASH_INDEPENDENT_WRITEBACK)
		          uint32_t const p1k136_hash = xor_reduction(tCrC);
		          asm volatile("" : : "r"(p1k136_hash) : "memory");
#elif defined(PEARL_P1K148_SCALAR16_DEFERRED_TRANSCRIPT)
              uint32_t const p1k148_hash = xor_reduction(tCrC);
              uint32_t const p1k148_slot =
                  p1k148_reduction_count &
                  (blake3::MSG_BLOCK_SIZE_U32 - 1);
              switch (p1k148_slot) {
                case 0:
                  p1k148_t0 = rotl_xor<HASH_ACCUMULATE_ROTATION>(
                      p1k148_t0, p1k148_hash);
                  break;
                case 1:
                  p1k148_t1 = rotl_xor<HASH_ACCUMULATE_ROTATION>(
                      p1k148_t1, p1k148_hash);
                  break;
                case 2:
                  p1k148_t2 = rotl_xor<HASH_ACCUMULATE_ROTATION>(
                      p1k148_t2, p1k148_hash);
                  break;
                case 3:
                  p1k148_t3 = rotl_xor<HASH_ACCUMULATE_ROTATION>(
                      p1k148_t3, p1k148_hash);
                  break;
                case 4:
                  p1k148_t4 = rotl_xor<HASH_ACCUMULATE_ROTATION>(
                      p1k148_t4, p1k148_hash);
                  break;
                case 5:
                  p1k148_t5 = rotl_xor<HASH_ACCUMULATE_ROTATION>(
                      p1k148_t5, p1k148_hash);
                  break;
                case 6:
                  p1k148_t6 = rotl_xor<HASH_ACCUMULATE_ROTATION>(
                      p1k148_t6, p1k148_hash);
                  break;
                case 7:
                  p1k148_t7 = rotl_xor<HASH_ACCUMULATE_ROTATION>(
                      p1k148_t7, p1k148_hash);
                  break;
                case 8:
                  p1k148_t8 = rotl_xor<HASH_ACCUMULATE_ROTATION>(
                      p1k148_t8, p1k148_hash);
                  break;
                case 9:
                  p1k148_t9 = rotl_xor<HASH_ACCUMULATE_ROTATION>(
                      p1k148_t9, p1k148_hash);
                  break;
                case 10:
                  p1k148_t10 = rotl_xor<HASH_ACCUMULATE_ROTATION>(
                      p1k148_t10, p1k148_hash);
                  break;
                case 11:
                  p1k148_t11 = rotl_xor<HASH_ACCUMULATE_ROTATION>(
                      p1k148_t11, p1k148_hash);
                  break;
                case 12:
                  p1k148_t12 = rotl_xor<HASH_ACCUMULATE_ROTATION>(
                      p1k148_t12, p1k148_hash);
                  break;
                case 13:
                  p1k148_t13 = rotl_xor<HASH_ACCUMULATE_ROTATION>(
                      p1k148_t13, p1k148_hash);
                  break;
                case 14:
                  p1k148_t14 = rotl_xor<HASH_ACCUMULATE_ROTATION>(
                      p1k148_t14, p1k148_hash);
                  break;
                default:
                  p1k148_t15 = rotl_xor<HASH_ACCUMULATE_ROTATION>(
                      p1k148_t15, p1k148_hash);
                  break;
              }
              ++p1k148_reduction_count;
#elif defined(PEARL_P1K129_PRELOAD_ONLY_REGISTER_SINK)
		          asm volatile("" : : "r"(k_tile) : "memory");
#else
	          hash_accumulator.accumulate_after_wait(
	              tCrC, k_blocks_per_tile - 1, k_blocks_per_tile);
#endif
	        }
	      } else if constexpr (kTileEndCanonicalTranscriptSnapshot) {
        warpgroup_wait<0>();
        warpgroup_fence_operand(tCrC);
        waited_for_wgmma = true;
        canonical_slot_accumulator.accumulate_after_wait(
            tCrC, thread_idx, shared_storage.canonical_transcript.data(),
            k_blocks_per_tile);
      }

      // Write back transcript elements after tile completes
	      if constexpr (!SkipReduction && !KTraits::EnableXqJournal &&
	                      !KTraits::EnableCanonicalTranscript &&
	                      !KTraits::EnableDummyReduction &&
	                      !kElideConsumerTranscript) {
#if defined(PEARL_P1K128_SINGLE_REGISTER_REDUCTION) || \
    defined(PEARL_P1K130_CELL_ONLY_REDUCTION)
		        asm volatile("" : : "r"(p1k_single_register_state) : "memory");
#elif defined(PEARL_P1K134_STRUCT_FREE_R128_HASH) || \
    defined(PEARL_P1K135_PRELOAD_WRITEBACK_ONLY) || \
    defined(PEARL_P1K136_HASH_INDEPENDENT_WRITEBACK) || \
    defined(PEARL_P1K137_ASM_DECOUPLED_HASH_WRITEBACK)
		        if constexpr (kTileEndHashSnapshot) {
		          uint32_t p1k134_store_value = p1k134_tile_hash_state;
#if defined(PEARL_P1K137_ASM_DECOUPLED_HASH_WRITEBACK)
		          asm volatile("mov.u32 %0, %1;"
		                       : "=r"(p1k134_store_value)
		                       : "r"(p1k134_tile_hash_state)
		                       : "memory");
#endif
		          transcript_extraction_tensor(p1k134_reduction_count) =
		              p1k134_store_value;
		          p1k134_reduction_count =
		              (p1k134_reduction_count + 1) % blake3::MSG_BLOCK_SIZE_U32;
		        } else {
		          hash_accumulator.writeback(transcript_extraction_tensor);
		        }
#elif defined(PEARL_P1K148_SCALAR16_DEFERRED_TRANSCRIPT)
            // Deliberately no per-k-tile transcript writeback. The final
            // materialization happens once after all WGMMA tiles complete.
            asm volatile("" : : "r"(p1k148_reduction_count) : "memory");
#if defined(PEARL_P1K175_NON_CONSUMER_PUBLICATION) && \
    !defined(PEARL_P1K175_ROW_NONCONSUMER_CONST_GLOBAL_NO_STAGE)
            if constexpr (KTraits::ProofOnly && kTileEndHashSnapshot) {
              if (k_tile == k_tile_count - 1) {
                auto [p1k175_M, p1k175_N, p1k175_K, p1k175_R] =
                    mainloop_params.problem_shape;
                (void)p1k175_M;
                (void)p1k175_K;
                (void)p1k175_R;
                int const p1k175_num_blocks_n =
                    (p1k175_N + KTraits::bN - 1) / KTraits::bN;
                int const p1k175_tile_id =
                    int(get<0>(block_coord)) * p1k175_num_blocks_n +
                    int(get<1>(block_coord));
                if (p1k175_tile_id == PEARL_P1K165_SINGLE_RECORD_TILE_ID &&
                    thread_idx == PEARL_P1K165_SINGLE_RECORD_CONSUMER) {
                  uint32_t p1k175_actual =
#if PEARL_P1K175_WORD_INDEX == 0
                      p1k148_t0;
#elif PEARL_P1K175_WORD_INDEX == 1
                      p1k148_t1;
#elif PEARL_P1K175_WORD_INDEX == 2
                      p1k148_t2;
#elif PEARL_P1K175_WORD_INDEX == 3
                      p1k148_t3;
#elif PEARL_P1K175_WORD_INDEX == 4
                      p1k148_t4;
#elif PEARL_P1K175_WORD_INDEX == 5
                      p1k148_t5;
#elif PEARL_P1K175_WORD_INDEX == 6
                      p1k148_t6;
#elif PEARL_P1K175_WORD_INDEX == 7
                      p1k148_t7;
#elif PEARL_P1K175_WORD_INDEX == 8
                      p1k148_t8;
#elif PEARL_P1K175_WORD_INDEX == 9
                      p1k148_t9;
#elif PEARL_P1K175_WORD_INDEX == 10
                      p1k148_t10;
#elif PEARL_P1K175_WORD_INDEX == 11
                      p1k148_t11;
#elif PEARL_P1K175_WORD_INDEX == 12
                      p1k148_t12;
#elif PEARL_P1K175_WORD_INDEX == 13
                      p1k148_t13;
#elif PEARL_P1K175_WORD_INDEX == 14
                      p1k148_t14;
#else
                      p1k148_t15;
#endif
                  uint32_t p1k175_word = p1k175_actual;
#if defined(PEARL_P1K175_ROW_CONST_SHARED_NONCONSUMER_GLOBAL)
                  p1k175_word =
                      0x175c0000u ^ uint32_t(p1k175_tile_id) ^
                      (uint32_t(thread_idx) << 8) ^
                      uint32_t(PEARL_P1K175_WORD_INDEX);
#endif
                  asm volatile("" : "+r"(p1k175_actual), "+r"(p1k175_word)
                               :
                               : "memory");
	                  volatile uint32_t* p1k175_stage =
	                      reinterpret_cast<volatile uint32_t*>(
	                          shared_storage.p1k175_stage.data());
                  p1k175_stage[PEARL_P1K175_STAGE_SLOT] = p1k175_word;
                  p1k175_stage[PEARL_P1K175_STAGE_SLOT + 1] =
                      0x1755a000u ^ uint32_t(p1k175_tile_id) ^
                      (uint32_t(thread_idx) << 8) ^
                      uint32_t(PEARL_P1K175_WORD_INDEX);
                  asm volatile("" : : "r"(p1k175_actual), "r"(p1k175_word)
                               : "memory");
                }
              }
            }
#endif
#elif defined(PEARL_P1K126_REGISTER_ONLY_REDUCTION) || \
    defined(PEARL_P1K127_NO_PRELOAD_REGISTER_ONLY) || \
    defined(PEARL_P1K129_PRELOAD_ONLY_REGISTER_SINK)
	        hash_accumulator.sink_register_state();
#else
	        hash_accumulator.writeback(transcript_extraction_tensor);
#endif
	      }

      if (!waited_for_wgmma) {
        warpgroup_wait<0>();
      }
	      // Release the stage of the pipeline for TMA
	      pipeline.consumer_release(smem_pipe_read);
	      ++smem_pipe_read;
	    }

#if defined(PEARL_P1K175_NON_CONSUMER_PUBLICATION)
    if constexpr (KTraits::ProofOnly && kTileEndHashSnapshot) {
      cutlass::arch::fence_view_async_shared();
      cutlass::arch::NamedBarrier::sync(
          kNumMmaThreads,
          static_cast<cutlass::arch::ReservedNamedBarriers>(11));
      asm volatile("" : : : "memory");
    }
#endif

#if defined(PEARL_P1K148_SCALAR16_DEFERRED_TRANSCRIPT)
    if constexpr (!SkipReduction && !KTraits::EnableXqJournal &&
                  !KTraits::EnableCanonicalTranscript &&
                  !KTraits::EnableDummyReduction) {
#if defined(PEARL_P1K150_SCALAR16_FINAL_GLOBAL_STORE) || \
    defined(PEARL_P1K154_SCALAR16_FINAL_SHARED_STORE) || \
    defined(PEARL_P1K165_TWO_PHASE_POW_CHECK)
      constexpr bool p1k150_writes_sideband =
          KTraits::EnableNativeGlobalJournalFill;
#else
      constexpr bool p1k150_writes_sideband = false;
#endif
      if constexpr (!p1k150_writes_sideband) {
      transcript_extraction_tensor(0) = p1k148_t0;
      transcript_extraction_tensor(1) = p1k148_t1;
      transcript_extraction_tensor(2) = p1k148_t2;
      transcript_extraction_tensor(3) = p1k148_t3;
      transcript_extraction_tensor(4) = p1k148_t4;
      transcript_extraction_tensor(5) = p1k148_t5;
      transcript_extraction_tensor(6) = p1k148_t6;
      transcript_extraction_tensor(7) = p1k148_t7;
      transcript_extraction_tensor(8) = p1k148_t8;
      transcript_extraction_tensor(9) = p1k148_t9;
      transcript_extraction_tensor(10) = p1k148_t10;
      transcript_extraction_tensor(11) = p1k148_t11;
      transcript_extraction_tensor(12) = p1k148_t12;
      transcript_extraction_tensor(13) = p1k148_t13;
      transcript_extraction_tensor(14) = p1k148_t14;
      transcript_extraction_tensor(15) = p1k148_t15;
      }
#if defined(PEARL_P1K150_SCALAR16_FINAL_GLOBAL_STORE) || \
    defined(PEARL_P1K165_TWO_PHASE_POW_CHECK)
      if constexpr (KTraits::EnableNativeGlobalJournalFill) {
      auto* p1k150_journal = mainloop_params.ptr_global_sideband_journal;
      int const p1k150_boundaries = mainloop_params.global_sideband_boundaries;
      int const p1k150_consumers = mainloop_params.global_sideband_consumers;
      int const p1k150_tiles = mainloop_params.global_sideband_tiles;
#if defined(PEARL_P1K152_SINGLE_CONSUMER_GLOBAL_STORE)
      bool const p1k150_consumer_in_range = (thread_idx == 0);
      int const p1k150_consumer_index = 0;
#else
      bool const p1k150_consumer_in_range =
          (thread_idx >= 0 && thread_idx < p1k150_consumers);
      int const p1k150_consumer_index = thread_idx;
#endif
      if (p1k150_journal != nullptr &&
          p1k150_boundaries >= blake3::MSG_BLOCK_SIZE_U32 &&
          p1k150_consumers > 0 && p1k150_tiles > 0 &&
          p1k150_consumer_in_range) {
        auto [M, N, K, R] = mainloop_params.problem_shape;
        (void)M;
        (void)K;
        (void)R;
        int const p1k150_num_blocks_n = (N + KTraits::bN - 1) / KTraits::bN;
        int const p1k150_tile_id = int(get<0>(block_coord)) * p1k150_num_blocks_n +
                                   int(get<1>(block_coord));
	        bool const p1k165_sideband_selected =
#if defined(PEARL_P1K175_NON_CONSUMER_PUBLICATION)
            p1k150_tile_id == PEARL_P1K165_SINGLE_RECORD_TILE_ID &&
            p1k150_consumers > PEARL_P1K165_SINGLE_RECORD_CONSUMER &&
            p1k150_boundaries > PEARL_P1K175_WORD_INDEX &&
            p1k150_consumer_index == PEARL_P1K175_PUBLISHER_CONSUMER;
	#elif defined(PEARL_P1K165_SINGLE_RECORD_SIDEBAND_STORE)
	            p1k150_tile_id == PEARL_P1K165_SINGLE_RECORD_TILE_ID &&
	            p1k150_consumer_index == PEARL_P1K165_SINGLE_RECORD_CONSUMER;
#elif defined(PEARL_P1K165_SPARSE_TILE_SIDEBAND_STORE)
#if !defined(PEARL_P1K165_SINGLE_CONSUMER) || \
    !defined(PEARL_P1K165_SPARSE_TILE_MASK) || \
    !defined(PEARL_P1K165_SPARSE_TILE_VALUE)
#error "PEARL_P1K165_SPARSE_TILE_SIDEBAND_STORE requires PEARL_P1K165_SINGLE_CONSUMER, PEARL_P1K165_SPARSE_TILE_MASK, and PEARL_P1K165_SPARSE_TILE_VALUE"
#endif
            p1k150_consumer_index == PEARL_P1K165_SINGLE_CONSUMER &&
            ((p1k150_tile_id & PEARL_P1K165_SPARSE_TILE_MASK) ==
             PEARL_P1K165_SPARSE_TILE_VALUE);
#elif defined(PEARL_P1K165_SINGLE_CONSUMER_SIDEBAND_STORE)
            p1k150_consumer_index == PEARL_P1K165_SINGLE_CONSUMER;
#else
            true;
#endif
        if (p1k150_tile_id >= 0 && p1k150_tile_id < p1k150_tiles &&
            p1k165_sideband_selected) {
#if defined(PEARL_P1K165_TWO_PHASE_POW_CHECK)
	          int const p1k150_base =
	              p1k150_tile_id * p1k150_consumers * p1k150_boundaries +
	#if defined(PEARL_P1K175_NON_CONSUMER_PUBLICATION)
	              PEARL_P1K165_SINGLE_RECORD_CONSUMER;
	#else
	              p1k150_consumer_index;
	#endif
#if defined(PEARL_P1K171_DUMMY_SIDEBAND_VALUES) && \
    !defined(PEARL_P1K171_CONST_REGSINK_NO_STORE) && \
    !defined(PEARL_P1K171_ACTUAL_REGSINK_NO_STORE) && \
    !defined(PEARL_P1K171_CONST_GLOBAL_STORE)
#define PEARL_P1K171_CONST_GLOBAL_STORE
#endif
#if defined(PEARL_P1K176_TOKEN_ONLY_DELAYED_PROOF)
          p1k150_journal[p1k150_base + 0 * p1k150_consumers] =
              0x17610000u ^ uint32_t(PEARL_P1K176_TOKEN_SCHEMA_VERSION);
          p1k150_journal[p1k150_base + 1 * p1k150_consumers] =
              uint32_t(p1k150_tile_id);
          p1k150_journal[p1k150_base + 2 * p1k150_consumers] =
              uint32_t(p1k150_consumer_index);
          p1k150_journal[p1k150_base + 3 * p1k150_consumers] =
              (uint32_t(KTraits::bM) << 24) ^
              (uint32_t(KTraits::bN) << 12) ^ uint32_t(KTraits::bK);
          p1k150_journal[p1k150_base + 4 * p1k150_consumers] =
              uint32_t(KTraits::R);
          p1k150_journal[p1k150_base + 5 * p1k150_consumers] =
              uint32_t(M);
          p1k150_journal[p1k150_base + 6 * p1k150_consumers] =
              uint32_t(N);
          p1k150_journal[p1k150_base + 7 * p1k150_consumers] =
              uint32_t(K);
          p1k150_journal[p1k150_base + 8 * p1k150_consumers] =
              uint32_t(R);
          p1k150_journal[p1k150_base + 9 * p1k150_consumers] =
              uint32_t(p1k150_boundaries);
          p1k150_journal[p1k150_base + 10 * p1k150_consumers] =
              uint32_t(p1k150_consumers);
          p1k150_journal[p1k150_base + 11 * p1k150_consumers] =
              uint32_t(PEARL_P1K165_SINGLE_RECORD_TILE_ID);
          p1k150_journal[p1k150_base + 12 * p1k150_consumers] =
              uint32_t(PEARL_P1K165_SINGLE_RECORD_CONSUMER);
          p1k150_journal[p1k150_base + 13 * p1k150_consumers] =
              0x1761000du;
          p1k150_journal[p1k150_base + 14 * p1k150_consumers] =
              0x1761000eu;
          p1k150_journal[p1k150_base + 15 * p1k150_consumers] =
              0x1761000fu;
#elif defined(PEARL_P1K171_SIDEBAND_BISECT)
#if ((defined(PEARL_P1K171_CONST_REGSINK_NO_STORE) ? 1 : 0) + \
     (defined(PEARL_P1K171_ACTUAL_REGSINK_NO_STORE) ? 1 : 0) + \
     (defined(PEARL_P1K171_CONST_GLOBAL_STORE) ? 1 : 0)) != 1
#error "PEARL_P1K171_SIDEBAND_BISECT requires exactly one P1K171 row macro"
#endif
#endif
#if defined(PEARL_P1K176_TOKEN_ONLY_DELAYED_PROOF)
          asm volatile("" : : "r"(p1k150_base) : "memory");
#elif defined(PEARL_P1K179_BLIND_SMEM_HARVEST)
          if constexpr (KTraits::ProofOnly) {
#if defined(PEARL_P1K179C_CONST_GLOBAL_AFTER_SHARED_HARVEST)
            volatile uint32_t* p1k179_stage =
                reinterpret_cast<volatile uint32_t*>(
                    shared_storage.p1k179_stage.data());
            cutlass::arch::fence_view_async_shared();
            uint32_t const p1k179_marker =
                p1k179_stage[PEARL_P1K179_STAGE_SLOT +
                              PEARL_P1K179_WORD_COUNT];
            p1k150_journal[p1k150_base +
                           PEARL_P1K179_ACCUM_INDEX * p1k150_consumers] =
                p1k179_marker;
            asm volatile("" : : "r"(p1k179_marker), "r"(p1k150_base)
                         : "memory");
#else
            asm volatile("" : : "r"(p1k150_base) : "memory");
#endif
          }
#elif defined(PEARL_P1K175_NON_CONSUMER_PUBLICATION)
          if constexpr (KTraits::ProofOnly) {
            uint32_t p1k175_word =
#if defined(PEARL_P1K175_ROW_NONCONSUMER_CONST_GLOBAL_NO_STAGE)
                0x175f0000u ^ uint32_t(p1k150_tile_id) ^
                (uint32_t(PEARL_P1K165_SINGLE_RECORD_CONSUMER) << 8) ^
                uint32_t(PEARL_P1K175_WORD_INDEX);
#else
	                reinterpret_cast<volatile uint32_t*>(
	                    shared_storage.p1k175_stage.data())[PEARL_P1K175_STAGE_SLOT];
#endif
            p1k150_journal[p1k150_base +
                           PEARL_P1K175_WORD_INDEX * p1k150_consumers] =
                p1k175_word;
            asm volatile("" : : "r"(p1k175_word), "r"(p1k150_base)
                         : "memory");
          }
#elif defined(PEARL_P1K174_STAGED_DECOUPLED_STORE)
#if ((defined(PEARL_P1K174_ROW_A_ACTUAL_STAGE_NO_GLOBAL) ? 1 : 0) + \
     (defined(PEARL_P1K174_ROW_B_ACTUAL_BARRIER_BLOCK_GLOBAL) ? 1 : 0) + \
     (defined(PEARL_P1K174_ROW_C_ACTUAL_TILE_END_BULK_FLUSH) ? 1 : 0) + \
     (defined(PEARL_P1K174_ROW_D_CONST_LIVE_GLOBAL_STORE) ? 1 : 0)) != 1
#error "PEARL_P1K174_STAGED_DECOUPLED_STORE requires exactly one P1K174 row macro"
#endif
#if !defined(PEARL_P1K174_WORD_INDEX) || !defined(PEARL_P1K174_WORD_COUNT)
#error "PEARL_P1K174_STAGED_DECOUPLED_STORE requires PEARL_P1K174_WORD_INDEX and PEARL_P1K174_WORD_COUNT"
#endif
#if PEARL_P1K174_WORD_COUNT != 1
#error "P1K174 initial diagnostic supports PEARL_P1K174_WORD_COUNT=1 only"
#endif
#if PEARL_P1K174_WORD_INDEX < 0 || PEARL_P1K174_WORD_INDEX > 15
#error "PEARL_P1K174_WORD_INDEX must be in [0,15]"
#endif
#if !defined(PEARL_P1K174_STAGE_RING_KIND)
#define PEARL_P1K174_STAGE_RING_KIND 0
#endif
          uint32_t p1k174_stage =
#if PEARL_P1K174_WORD_INDEX == 0
              p1k148_t0;
#elif PEARL_P1K174_WORD_INDEX == 1
              p1k148_t1;
#elif PEARL_P1K174_WORD_INDEX == 2
              p1k148_t2;
#elif PEARL_P1K174_WORD_INDEX == 3
              p1k148_t3;
#elif PEARL_P1K174_WORD_INDEX == 4
              p1k148_t4;
#elif PEARL_P1K174_WORD_INDEX == 5
              p1k148_t5;
#elif PEARL_P1K174_WORD_INDEX == 6
              p1k148_t6;
#elif PEARL_P1K174_WORD_INDEX == 7
              p1k148_t7;
#elif PEARL_P1K174_WORD_INDEX == 8
              p1k148_t8;
#elif PEARL_P1K174_WORD_INDEX == 9
              p1k148_t9;
#elif PEARL_P1K174_WORD_INDEX == 10
              p1k148_t10;
#elif PEARL_P1K174_WORD_INDEX == 11
              p1k148_t11;
#elif PEARL_P1K174_WORD_INDEX == 12
              p1k148_t12;
#elif PEARL_P1K174_WORD_INDEX == 13
              p1k148_t13;
#elif PEARL_P1K174_WORD_INDEX == 14
              p1k148_t14;
#else
              p1k148_t15;
#endif
          asm volatile("" : "+r"(p1k174_stage) : : "memory");
#if defined(PEARL_P1K174_ROW_A_ACTUAL_STAGE_NO_GLOBAL)
#if PEARL_P1K174_STAGE_RING_KIND == 0
          asm volatile("" : "+r"(p1k174_stage) : : "memory");
#elif PEARL_P1K174_STAGE_RING_KIND == 1
          volatile uint32_t* p1k174_shared =
              reinterpret_cast<volatile uint32_t*>(
                  shared_storage.xq_journal.data());
          p1k174_shared[thread_idx] = p1k174_stage;
          asm volatile("" : : "r"(p1k174_stage), "r"(thread_idx) : "memory");
#else
#error "PEARL_P1K174_STAGE_RING_KIND must be 0 or 1"
#endif
#elif defined(PEARL_P1K174_ROW_B_ACTUAL_BARRIER_BLOCK_GLOBAL)
          {
            uint32_t p1k174_staged_copy = p1k174_stage;
            asm volatile("" : "+r"(p1k174_staged_copy) : : "memory");
            p1k150_journal[p1k150_base +
                           PEARL_P1K174_WORD_INDEX * p1k150_consumers] =
                p1k174_staged_copy;
          }
#elif defined(PEARL_P1K174_ROW_C_ACTUAL_TILE_END_BULK_FLUSH)
          {
            uint32_t p1k174_flush_word = p1k174_stage;
            uint32_t* p1k174_dst =
                p1k150_journal + p1k150_base +
                PEARL_P1K174_WORD_INDEX * p1k150_consumers;
            uint64_t p1k174_dst_global = 0;
            asm volatile("cvta.to.global.u64 %0, %1;"
                         : "=l"(p1k174_dst_global)
                         : "l"(p1k174_dst));
            asm volatile("st.global.u32 [%0], %1;"
                         :
                         : "l"(p1k174_dst_global), "r"(p1k174_flush_word)
                         : "memory");
          }
#elif defined(PEARL_P1K174_ROW_D_CONST_LIVE_GLOBAL_STORE)
          {
            uint32_t p1k174_live_actual = p1k174_stage;
            uint32_t const p1k174_const =
                0x17400000u ^ uint32_t(p1k150_tile_id) ^
                (uint32_t(p1k150_consumer_index) << 8) ^
                uint32_t(PEARL_P1K174_WORD_INDEX);
            asm volatile("" : "+r"(p1k174_live_actual) : : "memory");
            p1k150_journal[p1k150_base +
                           PEARL_P1K174_WORD_INDEX * p1k150_consumers] =
                p1k174_const;
            asm volatile("" : : "r"(p1k174_live_actual), "r"(p1k174_const)
                         : "memory");
          }
#endif
#elif defined(PEARL_P1K171_SIDEBAND_BISECT) && \
    defined(PEARL_P1K171_CONST_REGSINK_NO_STORE)
          uint32_t const p1k171_c0 =
              0x17100000u ^ uint32_t(p1k150_tile_id);
          uint32_t const p1k171_c1 =
              0x17100001u ^ uint32_t(p1k150_consumer_index);
          uint32_t const p1k171_c2 = 0x17100002u;
          uint32_t const p1k171_c3 = 0x17100003u;
          uint32_t const p1k171_c4 = 0x17100004u;
          uint32_t const p1k171_c5 = 0x17100005u;
          uint32_t const p1k171_c6 = 0x17100006u;
          uint32_t const p1k171_c7 = 0x17100007u;
          uint32_t const p1k171_c8 = 0x17100008u;
          uint32_t const p1k171_c9 = 0x17100009u;
          uint32_t const p1k171_c10 = 0x1710000au;
          uint32_t const p1k171_c11 = 0x1710000bu;
          uint32_t const p1k171_c12 = 0x1710000cu;
          uint32_t const p1k171_c13 = 0x1710000du;
          uint32_t const p1k171_c14 = 0x1710000eu;
          uint32_t const p1k171_c15 = 0x1710000fu;
          asm volatile("" : : "r"(p1k171_c0), "r"(p1k171_c1),
                       "r"(p1k171_c2), "r"(p1k171_c3) : "memory");
          asm volatile("" : : "r"(p1k171_c4), "r"(p1k171_c5),
                       "r"(p1k171_c6), "r"(p1k171_c7) : "memory");
          asm volatile("" : : "r"(p1k171_c8), "r"(p1k171_c9),
                       "r"(p1k171_c10), "r"(p1k171_c11) : "memory");
          asm volatile("" : : "r"(p1k171_c12), "r"(p1k171_c13),
                       "r"(p1k171_c14), "r"(p1k171_c15) : "memory");
#elif defined(PEARL_P1K171_SIDEBAND_BISECT) && \
    defined(PEARL_P1K171_ACTUAL_REGSINK_NO_STORE)
          asm volatile("" : : "r"(p1k148_t0), "r"(p1k148_t1),
                       "r"(p1k148_t2), "r"(p1k148_t3) : "memory");
          asm volatile("" : : "r"(p1k148_t4), "r"(p1k148_t5),
                       "r"(p1k148_t6), "r"(p1k148_t7) : "memory");
          asm volatile("" : : "r"(p1k148_t8), "r"(p1k148_t9),
                       "r"(p1k148_t10), "r"(p1k148_t11) : "memory");
          asm volatile("" : : "r"(p1k148_t12), "r"(p1k148_t13),
                       "r"(p1k148_t14), "r"(p1k148_t15) : "memory");
#elif defined(PEARL_P1K172_ACTUAL_STORE_WORDS)
#if (PEARL_P1K172_ACTUAL_STORE_WORDS > 0)
          p1k150_journal[p1k150_base + 0 * p1k150_consumers] = p1k148_t0;
#else
          p1k150_journal[p1k150_base + 0 * p1k150_consumers] =
              0x17200000u ^ uint32_t(p1k150_tile_id);
#endif
#if (PEARL_P1K172_ACTUAL_STORE_WORDS > 1)
          p1k150_journal[p1k150_base + 1 * p1k150_consumers] = p1k148_t1;
#else
          p1k150_journal[p1k150_base + 1 * p1k150_consumers] =
              0x17200001u ^ uint32_t(p1k150_consumer_index);
#endif
#if (PEARL_P1K172_ACTUAL_STORE_WORDS > 2)
          p1k150_journal[p1k150_base + 2 * p1k150_consumers] = p1k148_t2;
#else
          p1k150_journal[p1k150_base + 2 * p1k150_consumers] =
              0x17200002u;
#endif
#if (PEARL_P1K172_ACTUAL_STORE_WORDS > 3)
          p1k150_journal[p1k150_base + 3 * p1k150_consumers] = p1k148_t3;
#else
          p1k150_journal[p1k150_base + 3 * p1k150_consumers] =
              0x17200003u;
#endif
#if (PEARL_P1K172_ACTUAL_STORE_WORDS > 4)
          p1k150_journal[p1k150_base + 4 * p1k150_consumers] = p1k148_t4;
#else
          p1k150_journal[p1k150_base + 4 * p1k150_consumers] =
              0x17200004u;
#endif
#if (PEARL_P1K172_ACTUAL_STORE_WORDS > 5)
          p1k150_journal[p1k150_base + 5 * p1k150_consumers] = p1k148_t5;
#else
          p1k150_journal[p1k150_base + 5 * p1k150_consumers] =
              0x17200005u;
#endif
#if (PEARL_P1K172_ACTUAL_STORE_WORDS > 6)
          p1k150_journal[p1k150_base + 6 * p1k150_consumers] = p1k148_t6;
#else
          p1k150_journal[p1k150_base + 6 * p1k150_consumers] =
              0x17200006u;
#endif
#if (PEARL_P1K172_ACTUAL_STORE_WORDS > 7)
          p1k150_journal[p1k150_base + 7 * p1k150_consumers] = p1k148_t7;
#else
          p1k150_journal[p1k150_base + 7 * p1k150_consumers] =
              0x17200007u;
#endif
#if (PEARL_P1K172_ACTUAL_STORE_WORDS > 8)
          p1k150_journal[p1k150_base + 8 * p1k150_consumers] = p1k148_t8;
#else
          p1k150_journal[p1k150_base + 8 * p1k150_consumers] =
              0x17200008u;
#endif
#if (PEARL_P1K172_ACTUAL_STORE_WORDS > 9)
          p1k150_journal[p1k150_base + 9 * p1k150_consumers] = p1k148_t9;
#else
          p1k150_journal[p1k150_base + 9 * p1k150_consumers] =
              0x17200009u;
#endif
#if (PEARL_P1K172_ACTUAL_STORE_WORDS > 10)
          p1k150_journal[p1k150_base + 10 * p1k150_consumers] = p1k148_t10;
#else
          p1k150_journal[p1k150_base + 10 * p1k150_consumers] =
              0x1720000au;
#endif
#if (PEARL_P1K172_ACTUAL_STORE_WORDS > 11)
          p1k150_journal[p1k150_base + 11 * p1k150_consumers] = p1k148_t11;
#else
          p1k150_journal[p1k150_base + 11 * p1k150_consumers] =
              0x1720000bu;
#endif
#if (PEARL_P1K172_ACTUAL_STORE_WORDS > 12)
          p1k150_journal[p1k150_base + 12 * p1k150_consumers] = p1k148_t12;
#else
          p1k150_journal[p1k150_base + 12 * p1k150_consumers] =
              0x1720000cu;
#endif
#if (PEARL_P1K172_ACTUAL_STORE_WORDS > 13)
          p1k150_journal[p1k150_base + 13 * p1k150_consumers] = p1k148_t13;
#else
          p1k150_journal[p1k150_base + 13 * p1k150_consumers] =
              0x1720000du;
#endif
#if (PEARL_P1K172_ACTUAL_STORE_WORDS > 14)
          p1k150_journal[p1k150_base + 14 * p1k150_consumers] = p1k148_t14;
#else
          p1k150_journal[p1k150_base + 14 * p1k150_consumers] =
              0x1720000eu;
#endif
#if (PEARL_P1K172_ACTUAL_STORE_WORDS > 15)
          p1k150_journal[p1k150_base + 15 * p1k150_consumers] = p1k148_t15;
#else
          p1k150_journal[p1k150_base + 15 * p1k150_consumers] =
              0x1720000fu;
#endif
#elif defined(PEARL_P1K171_DUMMY_SIDEBAND_VALUES) || \
    (defined(PEARL_P1K171_SIDEBAND_BISECT) && \
     defined(PEARL_P1K171_CONST_GLOBAL_STORE))
          p1k150_journal[p1k150_base + 0 * p1k150_consumers] =
              0x17100000u ^ uint32_t(p1k150_tile_id);
          p1k150_journal[p1k150_base + 1 * p1k150_consumers] =
              0x17100001u ^ uint32_t(p1k150_consumer_index);
          p1k150_journal[p1k150_base + 2 * p1k150_consumers] =
              0x17100002u;
          p1k150_journal[p1k150_base + 3 * p1k150_consumers] =
              0x17100003u;
          p1k150_journal[p1k150_base + 4 * p1k150_consumers] =
              0x17100004u;
          p1k150_journal[p1k150_base + 5 * p1k150_consumers] =
              0x17100005u;
          p1k150_journal[p1k150_base + 6 * p1k150_consumers] =
              0x17100006u;
          p1k150_journal[p1k150_base + 7 * p1k150_consumers] =
              0x17100007u;
          p1k150_journal[p1k150_base + 8 * p1k150_consumers] =
              0x17100008u;
          p1k150_journal[p1k150_base + 9 * p1k150_consumers] =
              0x17100009u;
          p1k150_journal[p1k150_base + 10 * p1k150_consumers] =
              0x1710000au;
          p1k150_journal[p1k150_base + 11 * p1k150_consumers] =
              0x1710000bu;
          p1k150_journal[p1k150_base + 12 * p1k150_consumers] =
              0x1710000cu;
          p1k150_journal[p1k150_base + 13 * p1k150_consumers] =
              0x1710000du;
          p1k150_journal[p1k150_base + 14 * p1k150_consumers] =
              0x1710000eu;
          p1k150_journal[p1k150_base + 15 * p1k150_consumers] =
              0x1710000fu;
#else
          p1k150_journal[p1k150_base + 0 * p1k150_consumers] = p1k148_t0;
          p1k150_journal[p1k150_base + 1 * p1k150_consumers] = p1k148_t1;
          p1k150_journal[p1k150_base + 2 * p1k150_consumers] = p1k148_t2;
          p1k150_journal[p1k150_base + 3 * p1k150_consumers] = p1k148_t3;
          p1k150_journal[p1k150_base + 4 * p1k150_consumers] = p1k148_t4;
          p1k150_journal[p1k150_base + 5 * p1k150_consumers] = p1k148_t5;
          p1k150_journal[p1k150_base + 6 * p1k150_consumers] = p1k148_t6;
          p1k150_journal[p1k150_base + 7 * p1k150_consumers] = p1k148_t7;
          p1k150_journal[p1k150_base + 8 * p1k150_consumers] = p1k148_t8;
          p1k150_journal[p1k150_base + 9 * p1k150_consumers] = p1k148_t9;
          p1k150_journal[p1k150_base + 10 * p1k150_consumers] = p1k148_t10;
          p1k150_journal[p1k150_base + 11 * p1k150_consumers] = p1k148_t11;
          p1k150_journal[p1k150_base + 12 * p1k150_consumers] = p1k148_t12;
          p1k150_journal[p1k150_base + 13 * p1k150_consumers] = p1k148_t13;
          p1k150_journal[p1k150_base + 14 * p1k150_consumers] = p1k148_t14;
          p1k150_journal[p1k150_base + 15 * p1k150_consumers] = p1k148_t15;
#endif
#else
          int const p1k150_base =
              (p1k150_tile_id * p1k150_consumers + p1k150_consumer_index) *
              p1k150_boundaries;
          p1k150_journal[p1k150_base + 0] = p1k148_t0;
          p1k150_journal[p1k150_base + 1] = p1k148_t1;
          p1k150_journal[p1k150_base + 2] = p1k148_t2;
          p1k150_journal[p1k150_base + 3] = p1k148_t3;
          p1k150_journal[p1k150_base + 4] = p1k148_t4;
          p1k150_journal[p1k150_base + 5] = p1k148_t5;
          p1k150_journal[p1k150_base + 6] = p1k148_t6;
          p1k150_journal[p1k150_base + 7] = p1k148_t7;
          p1k150_journal[p1k150_base + 8] = p1k148_t8;
          p1k150_journal[p1k150_base + 9] = p1k148_t9;
          p1k150_journal[p1k150_base + 10] = p1k148_t10;
          p1k150_journal[p1k150_base + 11] = p1k148_t11;
          p1k150_journal[p1k150_base + 12] = p1k148_t12;
          p1k150_journal[p1k150_base + 13] = p1k148_t13;
          p1k150_journal[p1k150_base + 14] = p1k148_t14;
          p1k150_journal[p1k150_base + 15] = p1k148_t15;
#endif
        }
      }
      }
#endif
#if defined(PEARL_P1K154_SCALAR16_FINAL_SHARED_STORE)
      if constexpr (KTraits::EnableNativeGlobalJournalFill) {
        uint32_t* p1k154_journal =
            reinterpret_cast<uint32_t*>(shared_storage.xq_journal.data());
        int const p1k154_base =
            thread_idx * KTraits::kXqJournalMaxBoundaries;
        p1k154_journal[p1k154_base + 0] = p1k148_t0;
        p1k154_journal[p1k154_base + 1] = p1k148_t1;
        p1k154_journal[p1k154_base + 2] = p1k148_t2;
        p1k154_journal[p1k154_base + 3] = p1k148_t3;
        p1k154_journal[p1k154_base + 4] = p1k148_t4;
        p1k154_journal[p1k154_base + 5] = p1k148_t5;
        p1k154_journal[p1k154_base + 6] = p1k148_t6;
        p1k154_journal[p1k154_base + 7] = p1k148_t7;
        p1k154_journal[p1k154_base + 8] = p1k148_t8;
        p1k154_journal[p1k154_base + 9] = p1k148_t9;
        p1k154_journal[p1k154_base + 10] = p1k148_t10;
        p1k154_journal[p1k154_base + 11] = p1k148_t11;
        p1k154_journal[p1k154_base + 12] = p1k148_t12;
        p1k154_journal[p1k154_base + 13] = p1k148_t13;
        p1k154_journal[p1k154_base + 14] = p1k148_t14;
        p1k154_journal[p1k154_base + 15] = p1k148_t15;
      }
#endif
    }
#endif

    if constexpr (!SkipReduction && KTraits::EnableCanonicalTranscript) {
      load_canonical_transcript_slots(transcript_extraction_tensor,
                                      shared_storage.canonical_transcript.data(),
                                      thread_idx);
    }

#if !defined(PEARL_P1K131_NO_PANEL_PARTIAL)
	    if constexpr (!SkipReduction && !kElideConsumerTranscript) {
	      if (mainloop_params.enable_panel_partial_transcript) {
        auto [m_block, n_block, bidb] = block_coord;
        auto [ctaid_in_cluster_x, ctaid_in_cluster_y, ctaid_in_cluster_z] =
            block_id_in_cluster();
        (void)ctaid_in_cluster_y;
        (void)ctaid_in_cluster_z;
        bool const is_first_logical_panel_cluster =
            (n_block == 0) &&
            (m_block < mainloop_params.panel_partial_panel_count);
        if (is_first_logical_panel_cluster &&
            mainloop_params.panel_partial_panel_count == 2) {
          uint32_t transcript_words[blake3::MSG_BLOCK_SIZE_U32];
          panel_partial_accumulator.write_words(transcript_words);
          uint16_t const panel_slot =
              static_cast<uint16_t>(ctaid_in_cluster_x);
          uint32_t const logical_row_start =
              static_cast<uint32_t>(panel_slot) *
              static_cast<uint32_t>(KTraits::bM);
          uint3 producer_block =
              make_uint3(blockIdx.x, blockIdx.y, blockIdx.z);
          uint3 producer_tile = make_uint3(
              static_cast<uint32_t>(m_block), static_cast<uint32_t>(n_block),
              static_cast<uint32_t>(bidb));
	          write_panel_partial_transcript_record_v2(
	              mainloop_params.ptr_panel_partial_transcripts,
	              mainloop_params.panel_partial_transcript_capacity,
	              static_cast<int>(panel_slot),
              /*logical_receipt_id=*/0ULL, panel_slot,
              static_cast<uint16_t>(mainloop_params.panel_partial_panel_count),
              logical_row_start, /*row_count=*/1, /*col_start=*/0,
              /*col_count=*/128, producer_block, producer_tile,
	              /*producer_thread=*/0xffffffffU, transcript_words);
		      }
		    }
		  }
#endif
	    }

	    // Notify producer that main gemm is complete
    cutlass::arch::NamedBarrier::arrive(
        kNumMmaThreads + cutlass::NumThreadsPerWarp,
        static_cast<cutlass::arch::ReservedNamedBarriers>(
            pearl::NamedBarriers::MmaComplete));
  }
};

}  // namespace pearl
