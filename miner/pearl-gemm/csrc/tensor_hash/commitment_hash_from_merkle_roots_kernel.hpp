#include "blake3/blake3.cuh"

namespace pearl {

// V3 domain-separation salts, encoded as little-endian u32 words. These are
// blake3("pearl/cert-v3/noise-seed/{A,B}"), hardcoded so the derivation doesn't
// depend on runtime string hashing.
__device__ __constant__ uint32_t
    SEED_SALT_A_U32[blake3::CHAINING_VALUE_SIZE_U32] = {
        0x6C404982, 0x1615EDA0, 0x92F61696, 0xF876F0FC,
        0x2ADBDB92, 0x52B82370, 0x1977D4F0, 0x7B0190C3};
__device__ __constant__ uint32_t
    SEED_SALT_B_U32[blake3::CHAINING_VALUE_SIZE_U32] = {
        0x32063011, 0xCA0163EC, 0x71AFE22B, 0x4F4D3F8B,
        0x39C6E91A, 0x04CCE888, 0x1D304448, 0xA99AB871};

class CommitmentHashFromMerkleRootsKernel {
 public:
  using Element = uint8_t;
  static constexpr uint32_t MaxThreadsPerBlock = 1;
  static constexpr uint32_t MinBlocksPerMultiprocessor = 1;

  static constexpr int SharedStorageSize = 0;

  using RmemChainingValueLayout =
      Layout<Shape<Int<blake3::CHAINING_VALUE_SIZE_U32>>>;
  using RmemBlockLayout = Layout<Shape<Int<blake3::MSG_BLOCK_SIZE_U32>>>;

  // Device side arguments
  struct Arguments {
    Element const* const ptr_A_merkle_root;
    Element const* const ptr_B_merkle_root;
    Element const* const ptr_key;
    Element* const ptr_A_commitment_hash;
    Element* const ptr_B_commitment_hash;
    Element const* const ptr_routing_root;
    Element const* const ptr_offsets_hash;
    bool apply_salt;
    uint32_t salted_dim_a;
    uint32_t salted_dim_b;
  };

  // Kernel entry point API
  struct Params {
    Element const* const ptr_A_merkle_root;
    Element const* const ptr_B_merkle_root;
    Element const* const ptr_key;
    Element* const ptr_A_commitment_hash;
    Element* const ptr_B_commitment_hash;
    Element const* const ptr_routing_root;
    Element const* const ptr_offsets_hash;
    bool apply_salt;
    uint32_t salted_dim_a;
    uint32_t salted_dim_b;
  };

  static Params to_underlying_arguments(Arguments const& args) {
    return {args.ptr_A_merkle_root,
            args.ptr_B_merkle_root,
            args.ptr_key,
            args.ptr_A_commitment_hash,
            args.ptr_B_commitment_hash,
            args.ptr_routing_root,
            args.ptr_offsets_hash,
            args.apply_salt,
            args.salted_dim_a,
            args.salted_dim_b};
  }

  static dim3 get_grid_shape(Params const& params) { return dim3(1); }

  static dim3 get_block_shape() { return dim3(1); }

  CUTLASS_DEVICE
  void operator()(Params const& params, char* smem_buf) {
    // V3 binds each raw Merkle root to its matrix dimension.
    Tensor rARoot = make_tensor<uint32_t>(RmemChainingValueLayout{});
    Tensor rBRoot = make_tensor<uint32_t>(RmemChainingValueLayout{});
    uint32_t const* A_merkle_root_u32 =
        (uint32_t const*)params.ptr_A_merkle_root;
    uint32_t const* B_merkle_root_u32 =
        (uint32_t const*)params.ptr_B_merkle_root;

    if (params.apply_salt) {
      bind_root(A_merkle_root_u32, params.salted_dim_a, SEED_SALT_A_U32,
                rARoot);
      bind_root(B_merkle_root_u32, params.salted_dim_b, SEED_SALT_B_U32,
                rBRoot);
    } else {
      load_chaining_value(A_merkle_root_u32, rARoot);
      load_chaining_value(B_merkle_root_u32, rBRoot);
    }

    // In the MoE case the routing commitment is folded into A's root:
    //   hash_routing     = BLAKE3(routing_root || offsets_hash)
    //   hash_activations = BLAKE3(A_root || hash_routing)
    // When the optional routing inputs are absent (nullptr) A's root is used
    // as is (dense case).
    if (params.ptr_routing_root != nullptr) {
      Tensor rRoutingRoot = make_tensor<uint32_t>(RmemChainingValueLayout{});
      Tensor rOffsetsHash = make_tensor<uint32_t>(RmemChainingValueLayout{});
      load_chaining_value((uint32_t const*)params.ptr_routing_root,
                          rRoutingRoot);
      load_chaining_value((uint32_t const*)params.ptr_offsets_hash,
                          rOffsetsHash);

      Tensor rHashRouting = make_tensor<uint32_t>(RmemChainingValueLayout{});
      hash_pair(rRoutingRoot, rOffsetsHash, rHashRouting);

      Tensor rHashActivations =
          make_tensor<uint32_t>(RmemChainingValueLayout{});
      hash_pair(rARoot, rHashRouting, rHashActivations);
      copy(rHashActivations, rARoot);
    }

    // The seed chain: BLAKE3(key || B_root), then BLAKE3(that || A_root).
    Tensor rKey = make_tensor<uint32_t>(RmemChainingValueLayout{});
    load_chaining_value((uint32_t const*)params.ptr_key, rKey);

    Tensor rChainingValueB = make_tensor<uint32_t>(RmemChainingValueLayout{});
    hash_pair(rKey, rBRoot, rChainingValueB);

    // A's seed is derived from B's commitment.
    Tensor rChainingValueA = make_tensor<uint32_t>(RmemChainingValueLayout{});
    hash_pair(rChainingValueB, rARoot, rChainingValueA);

    // Copy the result back to global memory
    uint32_t* A_commitment_hash_u32 = (uint32_t*)params.ptr_A_commitment_hash;
    uint32_t* B_commitment_hash_u32 = (uint32_t*)params.ptr_B_commitment_hash;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < blake3::CHAINING_VALUE_SIZE_U32; ++i) {
      A_commitment_hash_u32[i] = rChainingValueA(i);
      B_commitment_hash_u32[i] = rChainingValueB(i);
    }
  }

 private:
  template <class TensorOut>
  CUTLASS_DEVICE static void load_chaining_value(uint32_t const* src,
                                                 TensorOut& out) {
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < blake3::CHAINING_VALUE_SIZE_U32; ++i) {
      out(i) = src[i];
    }
  }

  // Unkeyed BLAKE3(left || right) over two 32-byte chaining values, which
  // together are exactly one message block.
  template <class TensorLeft, class TensorRight, class TensorOut>
  CUTLASS_DEVICE static void hash_pair(TensorLeft const& left,
                                       TensorRight const& right,
                                       TensorOut& out) {
    Tensor rBlock = make_tensor<uint32_t>(RmemBlockLayout{});
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < blake3::CHAINING_VALUE_SIZE_U32; ++i) {
      rBlock(i) = left(i);
      rBlock(i + blake3::CHAINING_VALUE_SIZE_U32) = right(i);
      out(i) = blake3::IV[i];
    }
    blake3::compress_msg_block_u32(rBlock, out,
                                   blake3::COMPRESS_PARAMS_SINGLE_BLOCK);
  }

  // V3 salting of one Merkle root: keyed BLAKE3 of a single block holding
  // `root || dim as u32 LE || zero padding`, with `salt` as the BLAKE3 key.
  template <class TensorOut>
  CUTLASS_DEVICE static void bind_root(uint32_t const* root, uint32_t dim,
                                       uint32_t const* salt, TensorOut& out) {
    Tensor rBlock = make_tensor<uint32_t>(RmemBlockLayout{});
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < blake3::MSG_BLOCK_SIZE_U32; ++i) {
      rBlock(i) = 0;
    }
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < blake3::CHAINING_VALUE_SIZE_U32; ++i) {
      rBlock(i) = root[i];
      out(i) = salt[i];
    }
    rBlock(blake3::CHAINING_VALUE_SIZE_U32) = dim;
    blake3::compress_msg_block_u32(rBlock, out,
                                   blake3::COMPRESS_PARAMS_SINGLE_BLOCK_KEYED);
  }
};  // class CommitmentHashFromMerkleRootsKernel
}  // namespace pearl