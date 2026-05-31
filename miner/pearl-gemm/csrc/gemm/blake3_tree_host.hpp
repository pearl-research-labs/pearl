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

// Keyed BLAKE3 of `data[0..len)` -> 32-byte digest. == pearl_mining.MerkleTree.root
// when `data` is the (chunk-padded) matrix bytes and key is the job_key.
inline void blake3_keyed_tree(const uint8_t key32[32], const uint8_t* data, size_t len, uint8_t out32[32]) {
  uint32_t key[8]; for (int i = 0; i < 8; ++i) key[i] = ld32(key32 + 4 * i);
  uint32_t cv_stack[54][8]; int stack_len = 0;
  ChunkState cs; chunk_init(&cs, key, 0, KEYED_HASH);
  size_t pos = 0;
  while (pos < len) {
    if (chunk_len(&cs) == CHUNK_LEN) {
      Output o = chunk_output(&cs); uint32_t cv[8]; output_cv(&o, cv);
      uint64_t total_chunks = cs.counter + 1;
      while ((total_chunks & 1) == 0) {
        Output p = parent_output(cv_stack[stack_len - 1], cv, key, KEYED_HASH);
        output_cv(&p, cv); stack_len--; total_chunks >>= 1;
      }
      for (int i = 0; i < 8; ++i) cv_stack[stack_len][i] = cv[i];
      stack_len++; chunk_init(&cs, key, cs.counter + 1, KEYED_HASH);
    }
    size_t want = CHUNK_LEN - chunk_len(&cs), rem = len - pos, take = want < rem ? want : rem;
    chunk_update(&cs, data + pos, take); pos += take;
  }
  Output o = chunk_output(&cs);
  for (int i = stack_len - 1; i >= 0; --i) {
    uint32_t right[8]; output_cv(&o, right);
    o = parent_output(cv_stack[i], right, key, KEYED_HASH);
  }
  output_root(&o, out32);
}

}  // namespace b3tree
}  // namespace pearl_miner
