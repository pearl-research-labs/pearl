// SPDX-License-Identifier: see LICENSE
//
// Host-side, dependency-free reference for the Pearl PoW *derivation chain* used
// by the standalone sm_89 miner `pearl_miner_sm89`. This is the arch-independent
// part of the pipeline — it reproduces, byte-for-byte, the captured-oracle values
// (job_key / b_noise_seed / a_noise_seed / gpu_hash) without any GPU, torch, numpy
// or the `blake3` PyPI package. It is the cheap, decisive correctness gate.
//
// Authorities reproduced here:
//   * job_key      = blake3(header[76B] || mining_config[52B])         (unkeyed)
//       miner-base/commitment_hash.py::CommitmentHasher.get_key
//   * b_noise_seed = blake3(job_key || B_root)                          (unkeyed)
//   * a_noise_seed = blake3(b_noise_seed || A_root)                     (unkeyed)
//       commitment_hash.py::commitment_hash_from_merkle_roots
//       (A_root/B_root are the captured `hash_a`/`hash_b` merkle roots)
//   * gpu_hash     = blake3_keyed(transcript[16xu32 LE], key=a_noise_seed)
//       pow_utils.hpp::check_pow_target
//   * noise        = dense [-32,31] (EAL,EBR) / sparse {-1,0,+1} (EAR,EBL)
//       miner-base/noise_generation.py
//   * transcript   = per-(rc%16) rotl13-XOR of XOR-reduced int32 hash-tiles
//       inner_hash.py + noisy_gemm.py (matches merged_kernel/gemm_patterned.c)
//
// BLAKE3 here is the portable reference (same core as merged_kernel/blake3_single.c)
// but generalized to multi-block single-chunk hashing so the 128-byte job_key
// (header||config) is handled correctly. All Pearl messages here are <= 1024 bytes
// (one BLAKE3 chunk), so no parent-node tree logic is needed.

#pragma once

#include <cstdint>
#include <cstring>
#include <cstddef>
#include <vector>

namespace pearl_miner {

// ===========================================================================
// BLAKE3 (single chunk, up to 1024 bytes; keyed + unkeyed) — portable reference
// ===========================================================================
namespace blake3 {

static const uint32_t IV[8] = {
    0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au,
    0x510E527Fu, 0x9B05688Cu, 0x1F83D9ABu, 0x5BE0CD19u};

static const uint8_t MSG_SCHEDULE[7][16] = {
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    {2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8},
    {3, 4, 10, 12, 13, 2, 7, 14, 6, 5, 9, 0, 11, 15, 8, 1},
    {10, 7, 12, 9, 14, 3, 13, 15, 4, 0, 11, 2, 5, 8, 1, 6},
    {12, 13, 9, 11, 15, 10, 14, 8, 7, 2, 5, 3, 0, 1, 6, 4},
    {9, 14, 11, 5, 8, 12, 15, 1, 13, 3, 0, 10, 2, 6, 4, 7},
    {11, 15, 5, 0, 1, 9, 8, 6, 14, 10, 2, 12, 3, 4, 7, 13},
};

// BLAKE3 domain-separation flags.
static constexpr uint32_t CHUNK_START = 1u << 0;
static constexpr uint32_t CHUNK_END   = 1u << 1;
static constexpr uint32_t ROOT        = 1u << 3;
static constexpr uint32_t KEYED_HASH  = 1u << 4;

inline uint32_t rotr32(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

inline uint32_t load_le32(const uint8_t* p) {
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) |
         ((uint32_t)p[3] << 24);
}

inline void g(uint32_t* s, int a, int b, int c, int d, uint32_t x, uint32_t y) {
  s[a] = s[a] + s[b] + x; s[d] = rotr32(s[d] ^ s[a], 16);
  s[c] = s[c] + s[d];     s[b] = rotr32(s[b] ^ s[c], 12);
  s[a] = s[a] + s[b] + y; s[d] = rotr32(s[d] ^ s[a], 8);
  s[c] = s[c] + s[d];     s[b] = rotr32(s[b] ^ s[c], 7);
}

inline void round_fn(uint32_t s[16], const uint32_t m[16], const uint8_t* z) {
  g(s, 0, 4, 8, 12, m[z[0]], m[z[1]]);
  g(s, 1, 5, 9, 13, m[z[2]], m[z[3]]);
  g(s, 2, 6, 10, 14, m[z[4]], m[z[5]]);
  g(s, 3, 7, 11, 15, m[z[6]], m[z[7]]);
  g(s, 0, 5, 10, 15, m[z[8]], m[z[9]]);
  g(s, 1, 6, 11, 12, m[z[10]], m[z[11]]);
  g(s, 2, 7, 8, 13, m[z[12]], m[z[13]]);
  g(s, 3, 4, 9, 14, m[z[14]], m[z[15]]);
}

// Compress one 64-byte block. `out` holds the full 16-word state (we keep the
// low 8 words as the chaining value / root digest).
inline void compress(const uint32_t cv[8], const uint32_t block[16],
                     uint64_t counter, uint32_t block_len, uint32_t flags,
                     uint32_t out[16]) {
  uint32_t s[16] = {
      cv[0], cv[1], cv[2], cv[3], cv[4], cv[5], cv[6], cv[7],
      IV[0], IV[1], IV[2], IV[3],
      (uint32_t)counter, (uint32_t)(counter >> 32), block_len, flags};
  uint32_t m[16];
  std::memcpy(m, block, sizeof(m));
  for (int r = 0; r < 7; ++r) round_fn(s, m, MSG_SCHEDULE[r]);
  for (int i = 0; i < 8; ++i) {
    out[i]     = s[i] ^ s[i + 8];
    out[i + 8] = s[i + 8] ^ cv[i];
  }
}

// Hash a message of length `len` (<= 1024 bytes => exactly one chunk) and return
// the 32-byte root digest. If `key` is non-null it is a 32-byte keyed hash
// (flags |= KEYED_HASH and cv = key words); else cv = IV.
inline void hash(const uint8_t* msg, size_t len, const uint8_t* key,
                 uint8_t out[32]) {
  uint32_t cv[8];
  uint32_t base_flags = 0;
  if (key) {
    for (int i = 0; i < 8; ++i) cv[i] = load_le32(key + 4 * i);
    base_flags = KEYED_HASH;
  } else {
    for (int i = 0; i < 8; ++i) cv[i] = IV[i];
  }

  // Single chunk: up to 16 blocks of 64 bytes. CHUNK_START on first block,
  // CHUNK_END | ROOT on the last block; chaining value flows between blocks.
  size_t nblocks = (len + 63) / 64;
  if (nblocks == 0) nblocks = 1;  // empty msg -> one zero block
  uint32_t state16[16];
  for (size_t b = 0; b < nblocks; ++b) {
    uint8_t blk[64] = {0};
    size_t off = b * 64;
    size_t blen = (len > off) ? (len - off) : 0;
    if (blen > 64) blen = 64;
    if (blen) std::memcpy(blk, msg + off, blen);
    uint32_t m[16];
    for (int i = 0; i < 16; ++i) m[i] = load_le32(blk + 4 * i);

    uint32_t flags = base_flags;
    if (b == 0) flags |= CHUNK_START;
    bool last = (b == nblocks - 1);
    if (last) flags |= CHUNK_END | ROOT;

    // block_len is the actual data bytes in this block (0..64); for the final
    // partial block it is the remainder, full blocks are 64.
    uint32_t block_len = (uint32_t)((last && blen) ? blen : (last && !blen ? 0 : 64));
    if (!last) block_len = 64;
    else if (blen == 0 && len != 0) block_len = 64;  // exact multiple of 64

    compress(cv, m, /*counter=*/0, block_len, flags, state16);
    for (int i = 0; i < 8; ++i) cv[i] = state16[i];
  }
  for (int i = 0; i < 8; ++i) {
    uint32_t v = state16[i];
    out[4 * i]     = (uint8_t)v;
    out[4 * i + 1] = (uint8_t)(v >> 8);
    out[4 * i + 2] = (uint8_t)(v >> 16);
    out[4 * i + 3] = (uint8_t)(v >> 24);
  }
}

// blake3(a || b) convenience for the 64-byte commitment-chain steps.
inline void hash_concat(const uint8_t* a, size_t la, const uint8_t* b, size_t lb,
                        const uint8_t* key, uint8_t out[32]) {
  std::vector<uint8_t> buf(la + lb);
  std::memcpy(buf.data(), a, la);
  std::memcpy(buf.data() + la, b, lb);
  hash(buf.data(), buf.size(), key, out);
}

}  // namespace blake3

// ===========================================================================
// Pearl seed-derivation chain (host, arch-independent)
// ===========================================================================

struct Seeds {
  uint8_t job_key[32];       // blake3(header || mining_config)
  uint8_t b_noise_seed[32];  // commitment_B = blake3(job_key || B_root)
  uint8_t a_noise_seed[32];  // commitment_A = blake3(b_noise_seed || A_root) == pow_key
};

// Derive job_key + commitment chain from the job (header+config) and the matrix
// merkle roots (A_root/B_root). For a *captured* share, A_root/B_root are the
// `hash_a`/`hash_b` from meta.txt; for a freshly *mined* share, they are
// blake3(pad1024(A), key=job_key) / blake3(pad1024(B^T), key=job_key).
inline Seeds derive_seeds(const uint8_t* header, size_t hlen,
                          const uint8_t* config, size_t clen,
                          const uint8_t A_root[32], const uint8_t B_root[32]) {
  Seeds s;
  blake3::hash_concat(header, hlen, config, clen, nullptr, s.job_key);
  blake3::hash_concat(s.job_key, 32, B_root, 32, nullptr, s.b_noise_seed);
  blake3::hash_concat(s.b_noise_seed, 32, A_root, 32, nullptr, s.a_noise_seed);
  return s;
}

// keyed-blake3 of the 16-word transcript (LE) under a_noise_seed -> gpu_hash.
inline void transcript_hash(const uint32_t transcript[16], const uint8_t key[32],
                            uint8_t out[32]) {
  uint8_t tb[64];
  for (int w = 0; w < 16; ++w) {
    uint32_t v = transcript[w];
    tb[4 * w]     = (uint8_t)v;
    tb[4 * w + 1] = (uint8_t)(v >> 8);
    tb[4 * w + 2] = (uint8_t)(v >> 16);
    tb[4 * w + 3] = (uint8_t)(v >> 24);
  }
  blake3::hash(tb, 64, key, out);
}

// ===========================================================================
// Noise generation (host reference; matches miner-base/noise_generation.py)
// ===========================================================================

// Dense random noise in [-32,31], one row of length R, for absolute matrix
// row `row` (so the BLAKE3 message counter matches the full-matrix layout).
inline void noise_dense_row(int row, int R, const uint8_t seed[32],
                            const uint8_t key[32], int8_t* out_row /*len R*/) {
  const int NOISE_ABS_MAX = 128;
  const int NOISE_RANGE = 64;  // 128 / 2
  for (int c0 = 0; c0 < R; c0 += 32) {
    int i = (row * R + c0) / 32;  // message index in the full (rows,R) stream
    int32_t prep[8] = {0};
    prep[0] = 1 + i;  // dense -> slot 0
    uint8_t msg[64];
    std::memcpy(msg, prep, 32);
    std::memcpy(msg + 32, seed, 32);
    uint8_t h[32];
    blake3::hash(msg, 64, key, h);
    for (int j = 0; j < 32; ++j) {
      int8_t hv = (int8_t)h[j];
      int8_t val = (int8_t)(((int32_t)hv + NOISE_ABS_MAX) % NOISE_RANGE - NOISE_RANGE / 2);
      out_row[c0 + j] = val;
    }
  }
}

// Full sparse permutation matrix EAR/EBL: (k, R) int8, one +1 and one -1 per row.
inline void noise_sparse(int k, int R, const uint8_t seed[32], const uint8_t key[32],
                         std::vector<int8_t>& E /*k*R*/) {
  E.assign((size_t)k * R, 0);
  const int per = 8;  // 32 bytes / 4 bytes
  int nmsg = (k + per - 1) / per;
  uint32_t rank_mask = (uint32_t)(R - 1);
  for (int i = 0; i < nmsg; ++i) {
    int ko = i * per;
    int32_t prep[8] = {0};
    prep[1] = 1 + i;  // sparse -> slot 1
    uint8_t msg[64];
    std::memcpy(msg, prep, 32);
    std::memcpy(msg + 32, seed, 32);
    uint8_t h[32];
    blake3::hash(msg, 64, key, h);
    const uint32_t* hu = reinterpret_cast<const uint32_t*>(h);
    for (int j = 0; j < per; ++j) {
      if (ko + j >= k) break;
      uint32_t u = hu[j];
      uint32_t k0 = u & rank_mask;
      uint32_t mulhi = (uint32_t)(((uint64_t)rank_mask * (uint64_t)u) >> 32);
      uint32_t k1 = (k0 ^ (1u + mulhi)) % (uint32_t)R;
      E[(size_t)(ko + j) * R + (k0 % (uint32_t)R)] = 1;
      E[(size_t)(ko + j) * R + k1] = -1;
    }
  }
}

// Full dense matrix EAL/EBR: (rows, R) int8 in [-32,31].
inline void noise_dense(int rows, int R, const uint8_t seed[32],
                        const uint8_t key[32], std::vector<int8_t>& E /*rows*R*/) {
  E.assign((size_t)rows * R, 0);
  for (int row = 0; row < rows; ++row)
    noise_dense_row(row, R, seed, key, &E[(size_t)row * R]);
}

// K-major transpose of a (k,R) matrix into (R,k).
inline void transpose_kR_to_Rk(const std::vector<int8_t>& src /*k*R*/, int k, int R,
                               std::vector<int8_t>& dst /*R*k*/) {
  dst.assign((size_t)R * k, 0);
  for (int i = 0; i < k; ++i)
    for (int j = 0; j < R; ++j)
      dst[(size_t)j * k + i] = src[(size_t)i * R + j];
}

inline int8_t i8wrap(int32_t x) { return (int8_t)(uint8_t)(x & 0xFF); }

// ===========================================================================
// Host transcript over disclosed strips (the arch-independent PoW core)
// ===========================================================================
//
// Given the opened A-rows (rows of A, each length K, in [-64,63]) and B-cols
// (columns of B = rows of B^T, each length K), the captured commitments, and the
// opened tile geometry, recompute the 16-word transcript exactly as the GPU /
// Rust verifier do. This is what `--verify-mode` prints for the user to diff
// against the captured `gpu_transcript`.
//
//   A_n[r] = i8wrap(A[r]   + EAL[r]   @ EAR^T)
//   B_n[c] = i8wrap(B^T[c] + EBR[c]   @ EBL^T)
//   per R-chunk rc: tile(8x16) = A_n[:,p:p+R] @ B_n[:,p:p+R]^T  (int32)
//                   x = XOR-reduce(tile);  T[rc%16] = rotl13(T[rc%16]) ^ x
//
// a_rows are the 8 opened rows; b_cols the 16 opened cols.
inline void transcript_from_strips(
    const std::vector<int8_t>& A_strip /*8*K*/,
    const std::vector<int8_t>& Bt_strip /*16*K*/,
    const int* a_rows, const int* b_cols, int n_arows, int n_bcols, int K, int R,
    const uint8_t commitment_A[32], const uint8_t commitment_B[32],
    uint32_t transcript[16]) {
  const uint8_t SEED_A[32] = {'A','_','t','e','n','s','o','r',0,0,0,0,0,0,0,0,
                              0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
  const uint8_t SEED_B[32] = {'B','_','t','e','n','s','o','r',0,0,0,0,0,0,0,0,
                              0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

  // Sparse EAR/EBL (k x R) — full, keyed by commitment_A / commitment_B.
  std::vector<int8_t> EAR, EBL;
  noise_sparse(K, R, SEED_A, commitment_A, EAR);
  noise_sparse(K, R, SEED_B, commitment_B, EBL);

  // Noised A rows.
  std::vector<int8_t> A_n((size_t)n_arows * K);
  for (int ri = 0; ri < n_arows; ++ri) {
    int8_t eal[256];  // R<=256
    noise_dense_row(a_rows[ri], R, SEED_A, commitment_A, eal);
    // E_A row = i8wrap(EAL[row] @ EAR^T)  -> length K
    for (int kk = 0; kk < K; ++kk) {
      int32_t acc = 0;
      const int8_t* ear_row = &EAR[(size_t)kk * R];
      for (int r = 0; r < R; ++r) acc += (int32_t)eal[r] * (int32_t)ear_row[r];
      A_n[(size_t)ri * K + kk] = i8wrap((int32_t)A_strip[(size_t)ri * K + kk] + acc);
    }
  }
  // Noised B columns (rows of B^T).
  std::vector<int8_t> B_n((size_t)n_bcols * K);
  for (int ci = 0; ci < n_bcols; ++ci) {
    int8_t ebr[256];
    noise_dense_row(b_cols[ci], R, SEED_B, commitment_B, ebr);
    for (int kk = 0; kk < K; ++kk) {
      int32_t acc = 0;
      const int8_t* ebl_row = &EBL[(size_t)kk * R];
      for (int r = 0; r < R; ++r) acc += (int32_t)ebr[r] * (int32_t)ebl_row[r];
      B_n[(size_t)ci * K + kk] = i8wrap((int32_t)Bt_strip[(size_t)ci * K + kk] + acc);
    }
  }

  for (int i = 0; i < 16; ++i) transcript[i] = 0;
  int nK = K / R;
  for (int rc = 0; rc < nK; ++rc) {
    int p = rc * R;
    uint32_t x = 0;
    for (int i = 0; i < n_arows; ++i) {
      for (int j = 0; j < n_bcols; ++j) {
        int32_t acc = 0;
        const int8_t* ar = &A_n[(size_t)i * K + p];
        const int8_t* br = &B_n[(size_t)j * K + p];
        for (int r = 0; r < R; ++r) acc += (int32_t)ar[r] * (int32_t)br[r];
        x ^= (uint32_t)acc;
      }
    }
    int idx = rc % 16;
    transcript[idx] = ((transcript[idx] << 13) | (transcript[idx] >> 19)) ^ x;
  }
}

}  // namespace pearl_miner
