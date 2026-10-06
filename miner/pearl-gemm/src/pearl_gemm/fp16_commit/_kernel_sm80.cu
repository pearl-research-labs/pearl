// Bit-exact A100 (sm_80) keyed-BLAKE3 Merkle commitment over FP16 operand rows.
//
// Reproduces zk-pow/src/api/fp16/commitment.rs::commit_operand -- i.e.
// pearl_blake3::MerkleTree::with_chunk_len(hash_id.pad(rows_to_bytes(rows)),
// key, hash_id.chunk_len()) -- producing the 32-byte Merkle root bit-for-bit.
//
// Tree discipline (see pearl-blake3/src/merkle.rs + hasher.rs):
//   * Committed bytes: FP16 rows as little-endian u16 row-major, zero-padded up
//     to a multiple of chunk_len (done on the host).
//   * Leaf i hash = keyed BLAKE3 *non-root* chunk CV of that chunk_len-byte leaf
//     with the BLAKE3 chunk counter = i. Every allowed chunk_len (128/256/512/
//     1024) is <= one native BLAKE3 chunk (1024), so a leaf is one chunk of
//     chunk_len/64 full 64-byte blocks: first block flags |= CHUNK_START, last
//     block flags |= CHUNK_END, base flag KEYED_HASH, counter = i throughout.
//   * Internal nodes: keyed non-root parent compression of (left||right); a lone
//     odd node is promoted unchanged.
//   * When a layer has exactly two nodes, the root is their keyed *root*-
//     finalized parent compression.
//   * Single-leaf tree (padded image <= chunk_len): root is the ROOT-finalized
//     keyed hash of the one chunk, not a non-root chunk CV.
//
// The BLAKE3 compression is the one agent B validated bit-identical to the
// `blake3` crate on this GA100 (fp16_noise_lines/_kernel_sm80.cu), reused
// verbatim; only the leaf/parent/root wrappers and the tree reduction are new.

#include <torch/extension.h>

// ---- BLAKE3 keyed compression (from fp16_noise_lines; crate-identical) ----

__constant__ unsigned int BLAKE3_IV[8] = {
    0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au,
    0x510E527Fu, 0x9B05688Cu, 0x1F83D9ABu, 0x5BE0CD19u};

// Domain-separation flags (pearl_blake3::hasher).
#define B3F_CHUNK_START 1u
#define B3F_CHUNK_END 2u
#define B3F_PARENT 4u
#define B3F_ROOT 8u
#define B3F_KEYED_HASH 16u

__device__ __forceinline__ unsigned int rotr32(unsigned int x, unsigned int n) {
    return (x >> n) | (x << (32u - n));
}

// One BLAKE3 compression of a single 64-byte message block `m` under chaining
// value `cv`, at `counter` (lo/hi), block length 64, with `flags`. Writes the
// full 16-word output: out[0..8] = s[0..8]^s[8..16] (the next chaining value /
// 32-byte hash output); out[8..16] = s[8..16]^cv[0..8].
__device__ void blake3_compress(const unsigned int cv[8], const unsigned int m[16],
                                unsigned int counter_lo, unsigned int counter_hi,
                                unsigned int flags, unsigned int out[16]) {
    unsigned int s[16];
#pragma unroll
    for (int i = 0; i < 8; ++i) s[i] = cv[i];
    s[8] = BLAKE3_IV[0];
    s[9] = BLAKE3_IV[1];
    s[10] = BLAKE3_IV[2];
    s[11] = BLAKE3_IV[3];
    s[12] = counter_lo;
    s[13] = counter_hi;
    s[14] = 64u;  // block_len (always a full 64-byte block here)
    s[15] = flags;

    unsigned int v[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) v[i] = m[i];

    const int PERM[16] = {2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8};

#define G(a, b, c, d, x, y)         \
    s[a] = s[a] + s[b] + (x);       \
    s[d] = rotr32(s[d] ^ s[a], 16); \
    s[c] = s[c] + s[d];             \
    s[b] = rotr32(s[b] ^ s[c], 12); \
    s[a] = s[a] + s[b] + (y);       \
    s[d] = rotr32(s[d] ^ s[a], 8);  \
    s[c] = s[c] + s[d];             \
    s[b] = rotr32(s[b] ^ s[c], 7);

    for (int round = 0; round < 7; ++round) {
        G(0, 4, 8, 12, v[0], v[1]);
        G(1, 5, 9, 13, v[2], v[3]);
        G(2, 6, 10, 14, v[4], v[5]);
        G(3, 7, 11, 15, v[6], v[7]);
        G(0, 5, 10, 15, v[8], v[9]);
        G(1, 6, 11, 12, v[10], v[11]);
        G(2, 7, 8, 13, v[12], v[13]);
        G(3, 4, 9, 14, v[14], v[15]);
        if (round < 6) {
            unsigned int t[16];
#pragma unroll
            for (int i = 0; i < 16; ++i) t[i] = v[PERM[i]];
#pragma unroll
            for (int i = 0; i < 16; ++i) v[i] = t[i];
        }
    }
#undef G

#pragma unroll
    for (int i = 0; i < 8; ++i) {
        out[i] = s[i] ^ s[i + 8];
        out[i + 8] = s[i + 8] ^ cv[i];
    }
}

// ---- Merkle wrappers ----

// Keyed BLAKE3 chunk chaining value of one leaf: `words_per_leaf` u32 words
// (= chunk_len/4, a multiple of 16) starting at `leaf`, counter = leaf index.
// `root` toggles ROOT finalization on the last block.
__device__ void leaf_cv(const unsigned int* __restrict__ data, long leaf,
                        int words_per_leaf, const unsigned int key[8], bool root,
                        unsigned int out_cv[8]) {
    unsigned int cv[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) cv[i] = key[i];

    int num_blocks = words_per_leaf / 16;  // 64-byte blocks in this leaf
    const unsigned int* base = data + leaf * (long)words_per_leaf;
    unsigned int counter_lo = (unsigned int)leaf;
    unsigned int counter_hi = (unsigned int)((unsigned long long)leaf >> 32);

    for (int b = 0; b < num_blocks; ++b) {
        unsigned int m[16];
#pragma unroll
        for (int w = 0; w < 16; ++w) m[w] = base[b * 16 + w];
        unsigned int flags = B3F_KEYED_HASH;
        if (b == 0) flags |= B3F_CHUNK_START;
        if (b == num_blocks - 1) {
            flags |= B3F_CHUNK_END;
            if (root) flags |= B3F_ROOT;
        }
        unsigned int o[16];
        blake3_compress(cv, m, counter_lo, counter_hi, flags, o);
#pragma unroll
        for (int i = 0; i < 8; ++i) cv[i] = o[i];
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) out_cv[i] = cv[i];
}

// Keyed parent compression of (left||right); `root` toggles ROOT finalization.
__device__ void parent_cv(const unsigned int left[8], const unsigned int right[8],
                          const unsigned int key[8], bool root, unsigned int out_cv[8]) {
    unsigned int m[16];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        m[i] = left[i];
        m[i + 8] = right[i];
    }
    unsigned int flags = B3F_KEYED_HASH | B3F_PARENT | (root ? B3F_ROOT : 0u);
    unsigned int o[16];
    blake3_compress(key, m, 0u, 0u, flags, o);
#pragma unroll
    for (int i = 0; i < 8; ++i) out_cv[i] = o[i];
}

// ---- Kernels ----

// One thread per leaf: compute the non-root leaf CVs into `cvs` (8 u32 each).
__global__ void leaves_kernel(const unsigned int* __restrict__ data, long num_leaves,
                              int words_per_leaf, const unsigned int* __restrict__ key,
                              unsigned int* __restrict__ cvs) {
    long tid = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_leaves) return;
    unsigned int k[8], cv[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) k[i] = key[i];
    leaf_cv(data, tid, words_per_leaf, k, /*root=*/false, cv);
#pragma unroll
    for (int i = 0; i < 8; ++i) cvs[tid * 8 + i] = cv[i];
}

// Single-leaf tree: ROOT-finalized keyed hash of the one chunk.
__global__ void single_leaf_root_kernel(const unsigned int* __restrict__ data,
                                        int words_per_leaf,
                                        const unsigned int* __restrict__ key,
                                        unsigned int* __restrict__ out) {
    if (blockIdx.x || threadIdx.x) return;
    unsigned int k[8], cv[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) k[i] = key[i];
    leaf_cv(data, 0, words_per_leaf, k, /*root=*/true, cv);
#pragma unroll
    for (int i = 0; i < 8; ++i) out[i] = cv[i];
}

// One layer of the reduction: out[j] = parent_cv(in[2j], in[2j+1]) for a full
// pair, else carry in[2j] (lone odd node). Never applies ROOT (handled apart).
__global__ void combine_kernel(const unsigned int* __restrict__ in, long in_len,
                               const unsigned int* __restrict__ key,
                               unsigned int* __restrict__ out) {
    long j = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long out_len = (in_len + 1) / 2;
    if (j >= out_len) return;
    unsigned int k[8], res[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) k[i] = key[i];
    long l = 2 * j;
    if (l + 1 < in_len) {
        unsigned int left[8], right[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            left[i] = in[l * 8 + i];
            right[i] = in[(l + 1) * 8 + i];
        }
        parent_cv(left, right, k, /*root=*/false, res);
    } else {
#pragma unroll
        for (int i = 0; i < 8; ++i) res[i] = in[l * 8 + i];
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) out[j * 8 + i] = res[i];
}

// Final ROOT-finalized combine of exactly two nodes.
__global__ void root_kernel(const unsigned int* __restrict__ in,
                            const unsigned int* __restrict__ key,
                            unsigned int* __restrict__ out) {
    if (blockIdx.x || threadIdx.x) return;
    unsigned int k[8], left[8], right[8], res[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        k[i] = key[i];
        left[i] = in[i];
        right[i] = in[8 + i];
    }
    parent_cv(left, right, k, /*root=*/true, res);
#pragma unroll
    for (int i = 0; i < 8; ++i) out[i] = res[i];
}

// ---- Host launcher ----

// `data32`: padded operand image as u32 (num_leaves * chunk_len/4 words).
// `key`: 8 u32 BLAKE3 key words. Returns the 32-byte root as 8 u32 words
// (little-endian byte order within each word gives the root bytes).
torch::Tensor fp16_commit_root(torch::Tensor data32, torch::Tensor key,
                               long num_leaves, long chunk_len) {
    TORCH_CHECK(data32.is_cuda() && data32.scalar_type() == torch::kInt32,
                "data32 must be an int32 CUDA tensor");
    TORCH_CHECK(key.is_cuda() && key.scalar_type() == torch::kInt32 && key.numel() == 8,
                "key must be an int32 CUDA tensor of 8 words");
    TORCH_CHECK(chunk_len == 128 || chunk_len == 256 || chunk_len == 512 || chunk_len == 1024,
                "chunk_len must be one of 128/256/512/1024");
    TORCH_CHECK(num_leaves >= 1, "num_leaves must be >= 1");
    int words_per_leaf = (int)(chunk_len / 4);
    TORCH_CHECK(data32.numel() == num_leaves * words_per_leaf,
                "data32 length must equal num_leaves * chunk_len/4");
    data32 = data32.contiguous();
    key = key.contiguous();

    auto u32 = torch::dtype(torch::kInt32).device(data32.device());
    auto out = torch::empty({8}, u32);
    const unsigned int* data_p = (const unsigned int*)data32.data_ptr();
    const unsigned int* key_p = (const unsigned int*)key.data_ptr();

    if (num_leaves == 1) {
        single_leaf_root_kernel<<<1, 1>>>(data_p, words_per_leaf, key_p,
                                          (unsigned int*)out.data_ptr());
        TORCH_CHECK(cudaGetLastError() == cudaSuccess, "single_leaf_root launch failed");
        return out;
    }

    // Leaf layer.
    auto cur = torch::empty({num_leaves * 8}, u32);
    {
        int threads = 128;
        long blocks = (num_leaves + threads - 1) / threads;
        leaves_kernel<<<blocks, threads>>>(data_p, num_leaves, words_per_leaf, key_p,
                                           (unsigned int*)cur.data_ptr());
        TORCH_CHECK(cudaGetLastError() == cudaSuccess, "leaves launch failed");
    }

    // Reduce pairwise until exactly two nodes remain.
    long len = num_leaves;
    while (len > 2) {
        long out_len = (len + 1) / 2;
        auto nxt = torch::empty({out_len * 8}, u32);
        int threads = 128;
        long blocks = (out_len + threads - 1) / threads;
        combine_kernel<<<blocks, threads>>>((const unsigned int*)cur.data_ptr(), len, key_p,
                                            (unsigned int*)nxt.data_ptr());
        TORCH_CHECK(cudaGetLastError() == cudaSuccess, "combine launch failed");
        cur = nxt;
        len = out_len;
    }

    // Final ROOT combine of the two surviving nodes.
    root_kernel<<<1, 1>>>((const unsigned int*)cur.data_ptr(), key_p,
                          (unsigned int*)out.data_ptr());
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "root launch failed");
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fp16_commit_root", &fp16_commit_root,
          "A100 sm_80 keyed-BLAKE3 Merkle root over FP16 operand rows",
          pybind11::arg("data32"), pybind11::arg("key"), pybind11::arg("num_leaves"),
          pybind11::arg("chunk_len"));
}
