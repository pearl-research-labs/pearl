#pragma once
// Bit-exact keyed BLAKE3 full-tree hash (arbitrary length), host C++, no deps.
//
// This is the matrix-root primitive for real-commitment Pearl mining:
//   A_root = blake3_keyed_tree(job_key, pad1024(A_bytes))
// which equals pearl_mining.MerkleTree(pad1024(A), key=job_key).root  (== the
// verifier's commitment root; pearl_mining.MerkleTree IS standard keyed BLAKE3).
// VALIDATED bit-exact vs pearl_mining across sizes {1024, 2048, 3072, 64K,
// 1048577, 5 MiB, 1 MiB-multi-level} (see _test_blake3_tree.cpp).
//
// Built on the validated single-block `compress` from blake3_single.c plus the
// BLAKE3 reference tree (chunk state + CV stack + parent/root). Matrix bytes are
// always a multiple of 1024 (K multiple of 1024), so no padding is needed at the
// mining shape; the tree still handles arbitrary length correctly.
#include <cstdint>
#include <cstring>
#include <vector>
#include <set>

namespace pearl_miner {
namespace b3tree {

inline const uint32_t IV[8] = {0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au,
                               0x510E527Fu, 0x9B05688Cu, 0x1F83D9ABu, 0x5BE0CD19u};
inline const uint8_t MSG_SCHEDULE[7][16] = {
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    {2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8},
    {3, 4, 10, 12, 13, 2, 7, 14, 6, 5, 9, 0, 11, 15, 8, 1},
    {10, 7, 12, 9, 14, 3, 13, 15, 4, 0, 11, 2, 5, 8, 1, 6},
    {12, 13, 9, 11, 15, 10, 14, 8, 7, 2, 5, 3, 0, 1, 6, 4},
    {9, 14, 11, 5, 8, 12, 15, 1, 13, 3, 0, 10, 2, 6, 4, 7},
    {11, 15, 5, 0, 1, 9, 8, 6, 14, 10, 2, 12, 3, 4, 7, 13}};
enum { CHUNK_START = 1, CHUNK_END = 2, PARENT = 4, ROOT = 8, KEYED_HASH = 16 };
inline constexpr uint32_t BLOCK_LEN = 64, CHUNK_LEN = 1024;

#define PEARL_B3_ROTR32(x, n) (((x) >> (n)) | ((x) << (32 - (n))))
inline uint32_t ld32(const uint8_t* p) {
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
inline void st32(uint8_t* p, uint32_t v) {
  p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); p[2] = (uint8_t)(v >> 16); p[3] = (uint8_t)(v >> 24);
}
inline void gmix(uint32_t* s, int a, int b, int c, int d, uint32_t x, uint32_t y) {
  s[a] = s[a] + s[b] + x; s[d] = PEARL_B3_ROTR32(s[d] ^ s[a], 16);
  s[c] = s[c] + s[d];     s[b] = PEARL_B3_ROTR32(s[b] ^ s[c], 12);
  s[a] = s[a] + s[b] + y; s[d] = PEARL_B3_ROTR32(s[d] ^ s[a], 8);
  s[c] = s[c] + s[d];     s[b] = PEARL_B3_ROTR32(s[b] ^ s[c], 7);
}
inline void round_fn(uint32_t s[16], const uint32_t m[16], const uint8_t* z) {
  gmix(s, 0, 4, 8, 12, m[z[0]], m[z[1]]);  gmix(s, 1, 5, 9, 13, m[z[2]], m[z[3]]);
  gmix(s, 2, 6, 10, 14, m[z[4]], m[z[5]]); gmix(s, 3, 7, 11, 15, m[z[6]], m[z[7]]);
  gmix(s, 0, 5, 10, 15, m[z[8]], m[z[9]]); gmix(s, 1, 6, 11, 12, m[z[10]], m[z[11]]);
  gmix(s, 2, 7, 8, 13, m[z[12]], m[z[13]]); gmix(s, 3, 4, 9, 14, m[z[14]], m[z[15]]);
}
inline void compress(const uint32_t cv[8], const uint32_t block[16], uint64_t counter,
                     uint32_t block_len, uint32_t flags, uint32_t out[16]) {
  uint32_t s[16] = {cv[0], cv[1], cv[2], cv[3], cv[4], cv[5], cv[6], cv[7],
                    IV[0], IV[1], IV[2], IV[3],
                    (uint32_t)counter, (uint32_t)(counter >> 32), block_len, flags};
  uint32_t m[16]; memcpy(m, block, sizeof(m));
  for (int r = 0; r < 7; ++r) round_fn(s, m, MSG_SCHEDULE[r]);
  for (int i = 0; i < 8; ++i) { out[i] = s[i] ^ s[i + 8]; out[i + 8] = s[i + 8] ^ cv[i]; }
}

struct ChunkState { uint32_t cv[8]; uint64_t counter; uint8_t block[64]; uint8_t block_len, blocks_compressed; uint32_t flags; };
inline void words_from_block(const uint8_t b[64], uint32_t w[16]) { for (int i = 0; i < 16; ++i) w[i] = ld32(b + 4 * i); }
inline uint32_t start_flag(const ChunkState* cs) { return cs->blocks_compressed == 0 ? CHUNK_START : 0; }
inline void chunk_init(ChunkState* cs, const uint32_t key[8], uint64_t counter, uint32_t flags) {
  for (int i = 0; i < 8; ++i) cs->cv[i] = key[i];
  cs->counter = counter; memset(cs->block, 0, 64); cs->block_len = 0; cs->blocks_compressed = 0; cs->flags = flags;
}
inline uint32_t chunk_len(const ChunkState* cs) { return (uint32_t)cs->blocks_compressed * BLOCK_LEN + cs->block_len; }
inline void chunk_update(ChunkState* cs, const uint8_t* in, size_t len) {
  while (len > 0) {
    if (cs->block_len == BLOCK_LEN) {
      uint32_t bw[16], out[16]; words_from_block(cs->block, bw);
      compress(cs->cv, bw, cs->counter, BLOCK_LEN, cs->flags | start_flag(cs), out);
      for (int i = 0; i < 8; ++i) cs->cv[i] = out[i];
      cs->blocks_compressed++; memset(cs->block, 0, 64); cs->block_len = 0;
    }
    size_t want = BLOCK_LEN - cs->block_len, take = want < len ? want : len;
    memcpy(cs->block + cs->block_len, in, take); cs->block_len += (uint8_t)take; in += take; len -= take;
  }
}
struct Output { uint32_t cv[8], block[16]; uint64_t counter; uint32_t block_len, flags; };
inline Output chunk_output(const ChunkState* cs) {
  Output o; for (int i = 0; i < 8; ++i) o.cv[i] = cs->cv[i];
  words_from_block(cs->block, o.block); o.counter = cs->counter; o.block_len = cs->block_len;
  o.flags = cs->flags | start_flag(cs) | CHUNK_END; return o;
}
inline void output_cv(const Output* o, uint32_t cv[8]) {
  uint32_t out[16]; compress(o->cv, o->block, o->counter, o->block_len, o->flags, out);
  for (int i = 0; i < 8; ++i) cv[i] = out[i];
}
inline void output_root(const Output* o, uint8_t out32[32]) {
  uint32_t out[16]; compress(o->cv, o->block, 0, o->block_len, o->flags | ROOT, out);
  for (int i = 0; i < 8; ++i) st32(out32 + 4 * i, out[i]);
}
inline Output parent_output(const uint32_t l[8], const uint32_t r[8], const uint32_t key[8], uint32_t flags) {
  Output o; for (int i = 0; i < 8; ++i) o.cv[i] = key[i];
  for (int i = 0; i < 8; ++i) { o.block[i] = l[i]; o.block[i + 8] = r[i]; }
  o.counter = 0; o.block_len = BLOCK_LEN; o.flags = flags | PARENT; return o;
}

// Reduce precomputed per-chunk output-CVs (cvs[c*8..]) to the keyed-BLAKE3 root.
// Requires n_chunks to be a power of two >= 2 (the mining shape: M*K/1024 = 2^19,
// N*K/1024 = 2^19). For a perfect binary tree this pairwise reduce is bit-identical
// to blake3_keyed_tree (verified): internal nodes use output_cv(parent), the single
// top node uses output_root(parent). Lets the expensive chunk hashing run on-GPU
// (only the 16 MB CV array is copied back, not the 512 MB matrix).
inline void blake3_root_from_chunk_cvs(const uint8_t key32[32], const uint32_t* cvs,
                                       size_t n_chunks, uint8_t out32[32]) {
  uint32_t key[8]; for (int i = 0; i < 8; ++i) key[i] = ld32(key32 + 4 * i);
  // PING-PONG between two buffers so each level folds in PARALLEL (OpenMP). An
  // in-place fold races (slot i is written while another thread reads it as the
  // left child of slot i/2). Bit-exact with the serial reduce: same pairwise
  // (2i,2i+1) order, same output_cv on internal nodes, output_root only on the
  // single top (n==2) node. This fold is ~1M compresses/attempt (2x 2^19-leaf
  // trees), historically the single-threaded ~200ms/attempt commit cost.
  std::vector<uint32_t> bufA(cvs, cvs + n_chunks * 8);
  std::vector<uint32_t> bufB(n_chunks * 4);          // holds the first (largest) folded level: n_chunks/2 nodes
  uint32_t* in = bufA.data();
  uint32_t* out = bufB.data();
  size_t n = n_chunks;
  while (n > 2) {
    size_t half = n / 2;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (long i = 0; i < (long)half; ++i) {
      Output p = parent_output(&in[(size_t)(2 * i) * 8], &in[(size_t)(2 * i + 1) * 8], key, KEYED_HASH);
      uint32_t cv[8]; output_cv(&p, cv);
      for (int j = 0; j < 8; ++j) out[(size_t)i * 8 + j] = cv[j];
    }
    uint32_t* tmp = in; in = out; out = tmp;       // swap: this level's output is next level's input
    n = half;
  }
  // n == 2: the single top node carries the ROOT flag.
  Output top = parent_output(&in[0], &in[8], key, KEYED_HASH);
  output_root(&top, out32);
}

// Build the full keyed-BLAKE3 tree from precomputed chunk CVs and return the root
// plus the multileaf-proof siblings for `leaf_indices` (sorted, unique) — bit-exact
// with pearl_blake3 MerkleTree::get_multileaf_proof (same level-walk + sibling order).
// `out_siblings` is appended with 32 bytes per sibling. n_chunks must be a power of 2.
inline void multileaf_proof_from_chunk_cvs(
    const uint8_t key32[32], const uint32_t* cvs, size_t n_chunks,
    const std::vector<size_t>& leaf_indices,
    uint8_t out_root32[32], std::vector<uint8_t>& out_siblings) {
  uint32_t key[8]; for (int i = 0; i < 8; ++i) key[i] = ld32(key32 + 4 * i);
  // layers[level] = flat array of 32-byte digests (level 0 = chunk CVs).
  std::vector<std::vector<uint8_t>> layers;
  { std::vector<uint8_t> l0(n_chunks * 32);
    for (size_t c = 0; c < n_chunks; ++c)
      for (int j = 0; j < 8; ++j) st32(&l0[c * 32 + 4 * j], cvs[c * 8 + j]);
    layers.push_back(std::move(l0)); }
  while (layers.back().size() / 32 > 1) {
    const std::vector<uint8_t>& prev = layers.back();
    size_t n = prev.size() / 32, half = n / 2;
    std::vector<uint8_t> next(half * 32);
    for (size_t i = 0; i < half; ++i) {
      uint32_t l[8], r[8];
      for (int j = 0; j < 8; ++j) { l[j] = ld32(&prev[(2*i)*32 + 4*j]); r[j] = ld32(&prev[(2*i+1)*32 + 4*j]); }
      Output p = parent_output(l, r, key, KEYED_HASH);
      if (n == 2) { output_root(&p, &next[0]); }
      else { uint32_t cv[8]; output_cv(&p, cv); for (int j = 0; j < 8; ++j) st32(&next[i*32 + 4*j], cv[j]); }
    }
    layers.push_back(std::move(next));
  }
  memcpy(out_root32, &layers.back()[0], 32);
  // Level-walk collecting missing siblings (matches get_multileaf_proof).
  std::set<size_t> cur(leaf_indices.begin(), leaf_indices.end());
  size_t level_len = n_chunks, level = 0;
  while (level_len > 1 && !cur.empty()) {
    const std::vector<uint8_t>& nodes = layers[level];
    for (size_t i : cur) {
      if (i % 2 == 1) {
        if (!cur.count(i - 1)) out_siblings.insert(out_siblings.end(), &nodes[(i-1)*32], &nodes[(i-1)*32 + 32]);
      } else if (!cur.count(i + 1) && (i + 1) < level_len) {
        out_siblings.insert(out_siblings.end(), &nodes[(i+1)*32], &nodes[(i+1)*32 + 32]);
      }
    }
    std::set<size_t> nxt; for (size_t i : cur) nxt.insert(i / 2);
    cur = std::move(nxt);
    level_len = (level_len + 1) / 2; level++;
  }
}

// Chaining value of one chunk (counter = chunk index). NO root flag.
inline void chunk_cv(const uint32_t key[8], const uint8_t* data, size_t len,
                     uint64_t counter, uint32_t cv_out[8]) {
  ChunkState cs; chunk_init(&cs, key, counter, KEYED_HASH);
  chunk_update(&cs, data, len);
  Output o = chunk_output(&cs);
  output_cv(&o, cv_out);
}

// Keyed BLAKE3 of `data[0..len)` -> 32-byte digest. == pearl_mining.MerkleTree.root
// when `data` is the (chunk-padded) matrix bytes and key is the job_key.
//
// The chunk chaining values are independent, so they are computed in parallel
// (OpenMP) — that is ~94% of the work for a 512 MB matrix. The tree reduce that
// folds them is the SAME left-balanced merge as the streaming reference (replayed
// over the precomputed CVs), so the result is bit-identical. The final chunk keeps
// its full Output for the root fold (the ROOT flag must land on the top node).
inline void blake3_keyed_tree(const uint8_t key32[32], const uint8_t* data, size_t len, uint8_t out32[32]) {
  uint32_t key[8]; for (int i = 0; i < 8; ++i) key[i] = ld32(key32 + 4 * i);
  // number of chunks (each up to CHUNK_LEN; last may be short). >=1 always.
  size_t n_chunks = len == 0 ? 1 : (len + CHUNK_LEN - 1) / CHUNK_LEN;
  size_t last_len = len == 0 ? 0 : (len - (n_chunks - 1) * CHUNK_LEN);

  if (n_chunks == 1) {
    ChunkState cs; chunk_init(&cs, key, 0, KEYED_HASH);
    chunk_update(&cs, data, len);
    Output o = chunk_output(&cs);
    output_root(&o, out32);
    return;
  }

  // Parallel: CV of every chunk EXCEPT the last (the last is folded with ROOT).
  std::vector<uint32_t> cvs((n_chunks - 1) * 8);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (long c = 0; c < (long)(n_chunks - 1); ++c) {
    chunk_cv(key, data + (size_t)c * CHUNK_LEN, CHUNK_LEN, (uint64_t)c, &cvs[(size_t)c * 8]);
  }

  // Sequential left-balanced merge over the precomputed chunk CVs (identical to
  // the streaming reference's add_chunk_chaining_value).
  uint32_t cv_stack[54][8]; int stack_len = 0;
  for (size_t c = 0; c < n_chunks - 1; ++c) {
    uint32_t cv[8]; for (int i = 0; i < 8; ++i) cv[i] = cvs[c * 8 + i];
    uint64_t total = c + 1;
    while ((total & 1) == 0) {
      Output p = parent_output(cv_stack[stack_len - 1], cv, key, KEYED_HASH);
      output_cv(&p, cv); stack_len--; total >>= 1;
    }
    for (int i = 0; i < 8; ++i) cv_stack[stack_len][i] = cv[i];
    stack_len++;
  }
  // Fold the final chunk's Output through the stack, applying ROOT at the top.
  ChunkState last; chunk_init(&last, key, n_chunks - 1, KEYED_HASH);
  chunk_update(&last, data + (n_chunks - 1) * CHUNK_LEN, last_len);
  Output o = chunk_output(&last);
  for (int i = stack_len - 1; i >= 0; --i) {
    uint32_t right[8]; output_cv(&o, right);
    o = parent_output(cv_stack[i], right, key, KEYED_HASH);
  }
  output_root(&o, out32);
}

}  // namespace b3tree
}  // namespace pearl_miner
