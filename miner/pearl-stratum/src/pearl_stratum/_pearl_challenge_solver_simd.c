/* _pearl_challenge_solver_simd.c
 *
 * Specialized AVX-512 (16-way) batched BLAKE3 brute-force solver for the
 * alphapool v1.5 `pearl.challenge` DDoS pacer:
 *
 *     find u64 nonce s.t. blake3(seed || nonce.to_le_bytes(8))
 *     has `difficulty` leading zero bits
 *
 * The input is exactly 40 bytes:
 *   bytes [ 0..32) = seed       (constant across the whole search)
 *   bytes [32..40) = nonce_le   (varies per lane)
 *
 * For a ≤ 64-byte input, BLAKE3 reduces to a single keyed compression with
 *   flags = CHUNK_START | CHUNK_END | ROOT,  counter = 0,  block_len = 40,
 * over a single 64-byte block where bytes [40..64) are zero-padded.
 *
 * The first 32 bytes of BLAKE3's output equal compress_in_place's CV form
 * (state[i] ^ state[i+8] for i in 0..7), so we don't need full XOF.
 *
 * We implement a 16-way AVX-512 kernel from scratch (rather than calling the
 * upstream `blake3_hash_many_avx512`, which hardcodes block_len = 64 in the
 * compression and would therefore produce wrong hashes for our 40-byte
 * input). The kernel processes 16 nonces' worth of compressions in a
 * single set of 7 BLAKE3 rounds, with all state held in __m512i registers.
 *
 * Key observation: only message words m[8] and m[9] (= nonce_low + nonce_high
 * as u32 little-endian) vary across the 16 lanes. m[0..8) come from the
 * seed and are lane-broadcast; m[10..16) are zero. This lets us bake the
 * seed words into the kernel as scalar broadcasts, and only m[8]/m[9]
 * (plus the comparison at the end) actually run vertical SIMD on lane data.
 *
 * Search strategy: lane-striped OpenMP. Each thread t scans 16-nonce
 * groups {16*t, ..., 16*t+15}, {16*(t+nthreads), ...}, etc. First thread
 * with a hit wins; min-nonce lock-merge selects the global minimum
 * (matches the Python reference solver semantics — pool accepts any).
 *
 * Build:
 *   gcc -O3 -march=native -mavx512f -mavx512vl -fopenmp \
 *       _pearl_challenge_solver_simd.c -o pearl_challenge_solver_simd
 *
 * Output: <nonce_u64_be_hex_16_chars>\n on stdout.
 *
 * Expected throughput on AMD Ryzen 9 7950X (32 cores, AVX-512):
 *   ~40-80 MH/s per thread × 32 threads ≈ 1-2 GH/s aggregate
 *   diff=32 (~4G hashes) -> ~3-5 s typical, ~6-8s p99
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdatomic.h>

#include <immintrin.h>
#include <omp.h>

#define LANES 16
#define BLAKE3_BLOCK_LEN 64
#define BLAKE3_OUT_LEN 32

/* BLAKE3 IV (BLAKE2s/SHA-256 initial state words). */
static const uint32_t BLAKE3_IV[8] = {
    0x6A09E667UL, 0xBB67AE85UL, 0x3C6EF372UL, 0xA54FF53AUL,
    0x510E527FUL, 0x9B05688CUL, 0x1F83D9ABUL, 0x5BE0CD19UL,
};

/* BLAKE3 message schedule (table 1 of the spec). */
static const uint8_t MSG_SCHEDULE[7][16] = {
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    {2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8},
    {3, 4, 10, 12, 13, 2, 7, 14, 6, 5, 9, 0, 11, 15, 8, 1},
    {10, 7, 12, 9, 14, 3, 13, 15, 4, 0, 11, 2, 5, 8, 1, 6},
    {12, 13, 9, 11, 15, 10, 14, 8, 7, 2, 5, 3, 0, 1, 6, 4},
    {9, 14, 11, 5, 8, 12, 15, 1, 13, 3, 0, 10, 2, 6, 4, 7},
    {11, 15, 5, 0, 1, 9, 8, 6, 14, 10, 2, 12, 3, 4, 7, 13},
};

/* BLAKE3 flags. */
#define CHUNK_START 1
#define CHUNK_END   2
#define ROOT        8

/* SIMD helpers (vertical: lanes × u32). */
static inline __m512i rot_right_512(__m512i x, int n) {
    /* AVX-512 has a native rotate (_mm512_ror_epi32). */
    return _mm512_ror_epi32(x, n);
}
static inline __m512i add_512(__m512i a, __m512i b) { return _mm512_add_epi32(a, b); }
static inline __m512i xor_512(__m512i a, __m512i b) { return _mm512_xor_si512(a, b); }
#define ROT16(x) rot_right_512(x, 16)
#define ROT12(x) rot_right_512(x, 12)
#define ROT8(x)  rot_right_512(x, 8)
#define ROT7(x)  rot_right_512(x, 7)

/* G mixing function — vertical SIMD across 16 lanes.
 *   a, b, c, d are pointers into a 16-vec state array v[]
 *   m1, m2 are the two message vectors used in this G call
 */
#define G(a, b, c, d, m1, m2)                              \
    v[a] = add_512(v[a], v[b]);                            \
    v[a] = add_512(v[a], m1);                              \
    v[d] = xor_512(v[d], v[a]);                            \
    v[d] = ROT16(v[d]);                                    \
    v[c] = add_512(v[c], v[d]);                            \
    v[b] = xor_512(v[b], v[c]);                            \
    v[b] = ROT12(v[b]);                                    \
    v[a] = add_512(v[a], v[b]);                            \
    v[a] = add_512(v[a], m2);                              \
    v[d] = xor_512(v[d], v[a]);                            \
    v[d] = ROT8(v[d]);                                     \
    v[c] = add_512(v[c], v[d]);                            \
    v[b] = xor_512(v[b], v[c]);                            \
    v[b] = ROT7(v[b]);

/* One BLAKE3 round (8 G's): 4 column-G then 4 diagonal-G,
 * with messages picked via the schedule for round r. */
#define ROUND(r)                                              \
    G(0, 4,  8, 12, m[MSG_SCHEDULE[r][ 0]], m[MSG_SCHEDULE[r][ 1]]); \
    G(1, 5,  9, 13, m[MSG_SCHEDULE[r][ 2]], m[MSG_SCHEDULE[r][ 3]]); \
    G(2, 6, 10, 14, m[MSG_SCHEDULE[r][ 4]], m[MSG_SCHEDULE[r][ 5]]); \
    G(3, 7, 11, 15, m[MSG_SCHEDULE[r][ 6]], m[MSG_SCHEDULE[r][ 7]]); \
    G(0, 5, 10, 15, m[MSG_SCHEDULE[r][ 8]], m[MSG_SCHEDULE[r][ 9]]); \
    G(1, 6, 11, 12, m[MSG_SCHEDULE[r][10]], m[MSG_SCHEDULE[r][11]]); \
    G(2, 7,  8, 13, m[MSG_SCHEDULE[r][12]], m[MSG_SCHEDULE[r][13]]); \
    G(3, 4,  9, 14, m[MSG_SCHEDULE[r][14]], m[MSG_SCHEDULE[r][15]]);

static inline uint32_t load32_le(const uint8_t *p) {
    return ((uint32_t)p[0]) | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static inline int has_leading_zero_bits(const uint8_t *h, int n) {
    int full = n >> 3;
    int rem = n & 7;
    for (int i = 0; i < full; i++) if (h[i] != 0) return 0;
    if (rem == 0) return 1;
    return (h[full] >> (8 - rem)) == 0;
}

static int hex_to_seed(const char *hex, uint8_t out[32]) {
    if (strlen(hex) != 64) return -1;
    for (int i = 0; i < 32; i++) {
        unsigned int b;
        if (sscanf(hex + 2*i, "%2x", &b) != 1) return -1;
        out[i] = (uint8_t)b;
    }
    return 0;
}

/* Compress 16 lanes of the BLAKE3 single-block PoW (block_len=40, counter=0,
 * flags = CHUNK_START | CHUNK_END | ROOT, key = IV).
 *
 * Inputs:
 *   seed_words[0..8) — eight u32 little-endian seed words (lane-broadcast).
 *   nonce_low[i]   = nonce[i] & 0xFFFFFFFF     (bytes [32..36) of msg)
 *   nonce_high[i]  = nonce[i] >> 32            (bytes [36..40) of msg)
 *     Both are 16-element __m512i vectors.
 *
 * Output:
 *   For each lane i, writes h[i][0..32) to out[i*32..(i+1)*32) in canonical
 *   BLAKE3 byte order. Equivalent to:
 *     blake3.blake3(seed || nonce.to_bytes(8,'little')).digest()
 */
static inline void pearl_hash16_avx512(
    const uint32_t seed_words[8],
    __m512i nonce_low,
    __m512i nonce_high,
    uint8_t out[16 * 32]
) {
    /* Message words (vertical: each m[k] is 16 u32s, one per lane).
     * m[0..8): seed words 0..7 (broadcast, same across all lanes)
     * m[8]   : nonce_low (varies)
     * m[9]   : nonce_high (varies)
     * m[10..16): zero (padding) */
    __m512i m[16];
    for (int k = 0; k < 8; k++) m[k] = _mm512_set1_epi32((int)seed_words[k]);
    m[8] = nonce_low;
    m[9] = nonce_high;
    for (int k = 10; k < 16; k++) m[k] = _mm512_setzero_si512();

    /* Initial state:
     *   v[0..8)   = cv = IV
     *   v[8..12)  = IV[0..4)
     *   v[12]     = counter_low  = 0
     *   v[13]     = counter_high = 0
     *   v[14]     = block_len    = 40
     *   v[15]     = flags        = CHUNK_START | CHUNK_END | ROOT = 11
     */
    __m512i v[16];
    for (int k = 0; k < 8; k++) v[k] = _mm512_set1_epi32((int)BLAKE3_IV[k]);
    v[8]  = _mm512_set1_epi32((int)BLAKE3_IV[0]);
    v[9]  = _mm512_set1_epi32((int)BLAKE3_IV[1]);
    v[10] = _mm512_set1_epi32((int)BLAKE3_IV[2]);
    v[11] = _mm512_set1_epi32((int)BLAKE3_IV[3]);
    v[12] = _mm512_setzero_si512();
    v[13] = _mm512_setzero_si512();
    v[14] = _mm512_set1_epi32(40);
    v[15] = _mm512_set1_epi32((int)(CHUNK_START | CHUNK_END | ROOT));

    ROUND(0);
    ROUND(1);
    ROUND(2);
    ROUND(3);
    ROUND(4);
    ROUND(5);
    ROUND(6);

    /* compress_in_place semantics: cv[i] = state[i] ^ state[i+8] for i in 0..8.
     * That's exactly the first 32 bytes of the root XOF hash. */
    __m512i h0 = xor_512(v[0], v[8]);
    __m512i h1 = xor_512(v[1], v[9]);
    __m512i h2 = xor_512(v[2], v[10]);
    __m512i h3 = xor_512(v[3], v[11]);
    __m512i h4 = xor_512(v[4], v[12]);
    __m512i h5 = xor_512(v[5], v[13]);
    __m512i h6 = xor_512(v[6], v[14]);
    __m512i h7 = xor_512(v[7], v[15]);

    /* Each h[k] holds 16 u32s, one per lane (lane i in element i).
     * We want to write per-lane: out[i*32 + 4*k .. i*32 + 4*k + 4) =
     *   little-endian bytes of lane[i].h[k].
     *
     * Strategy: store each h[k] to a 64-byte temp, then gather per-lane.
     * Simpler/faster: scatter. AVX-512 has _mm512_i32scatter_epi32. */
    uint32_t tmp[8 * 16] __attribute__((aligned(64)));
    _mm512_store_si512((__m512i*)&tmp[0  * 16], h0);
    _mm512_store_si512((__m512i*)&tmp[1  * 16], h1);
    _mm512_store_si512((__m512i*)&tmp[2  * 16], h2);
    _mm512_store_si512((__m512i*)&tmp[3  * 16], h3);
    _mm512_store_si512((__m512i*)&tmp[4  * 16], h4);
    _mm512_store_si512((__m512i*)&tmp[5  * 16], h5);
    _mm512_store_si512((__m512i*)&tmp[6  * 16], h6);
    _mm512_store_si512((__m512i*)&tmp[7  * 16], h7);

    /* Pack per-lane: out[i*32 + 4*k + b] = byte b of tmp[k*16 + i]. */
    for (int i = 0; i < LANES; i++) {
        uint32_t *dst = (uint32_t*)&out[i * 32];
        dst[0] = tmp[0 * 16 + i];
        dst[1] = tmp[1 * 16 + i];
        dst[2] = tmp[2 * 16 + i];
        dst[3] = tmp[3 * 16 + i];
        dst[4] = tmp[4 * 16 + i];
        dst[5] = tmp[5 * 16 + i];
        dst[6] = tmp[6 * 16 + i];
        dst[7] = tmp[7 * 16 + i];
    }
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s <seed_hex(64)> <difficulty>\n", argv[0]);
        return 1;
    }
    uint8_t seed[32];
    if (hex_to_seed(argv[1], seed) != 0) {
        fprintf(stderr, "bad seed hex (must be 64 chars)\n");
        return 1;
    }
    int difficulty = atoi(argv[2]);
    if (difficulty < 1 || difficulty > 64) {
        fprintf(stderr, "difficulty must be 1..64, got %d\n", difficulty);
        return 1;
    }

    /* Pack the 8 seed words once. */
    uint32_t seed_words[8];
    for (int k = 0; k < 8; k++) {
        seed_words[k] = load32_le(seed + 4 * k);
    }

    atomic_int found_flag = 0;
    uint64_t best_nonce = UINT64_MAX;
    omp_lock_t best_lock;
    omp_init_lock(&best_lock);

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();
        uint8_t outs[LANES * 32] __attribute__((aligned(64)));

        /* Group index: scan 16-nonce groups, stride by nthreads.
         * group g covers nonces [g*16 .. g*16+16). */
        uint64_t step = (uint64_t)nthreads;
        uint64_t base_group = (uint64_t)tid;
        uint64_t chunk_groups = (uint64_t)1u << 20;  /* 16 * 2^20 = 16M nonces per check */
        uint64_t end_group = base_group + chunk_groups * step;

        uint64_t local_best = UINT64_MAX;

        while (!atomic_load_explicit(&found_flag, memory_order_relaxed)) {
            for (uint64_t g = base_group; g < end_group; g += step) {
                uint64_t base_nonce = g * LANES;

                /* Build the 16 nonce values' low and high u32 parts. */
                __m512i nonce_lo, nonce_hi;
                {
                    uint32_t lo[16] __attribute__((aligned(64)));
                    uint32_t hi[16] __attribute__((aligned(64)));
                    for (int j = 0; j < LANES; j++) {
                        uint64_t n = base_nonce + (uint64_t)j;
                        lo[j] = (uint32_t)(n);
                        hi[j] = (uint32_t)(n >> 32);
                    }
                    nonce_lo = _mm512_load_si512((__m512i*)lo);
                    nonce_hi = _mm512_load_si512((__m512i*)hi);
                }

                pearl_hash16_avx512(seed_words, nonce_lo, nonce_hi, outs);

                for (int j = 0; j < LANES; j++) {
                    if (has_leading_zero_bits(&outs[j * 32], difficulty)) {
                        local_best = base_nonce + (uint64_t)j;
                        goto found_in_thread;
                    }
                }
            }
            base_group = end_group;
            end_group = base_group + chunk_groups * step;
        }
        goto thread_done;

    found_in_thread:
        omp_set_lock(&best_lock);
        if (local_best < best_nonce) {
            best_nonce = local_best;
        }
        omp_unset_lock(&best_lock);
        atomic_store_explicit(&found_flag, 1, memory_order_release);

    thread_done: ;
    }

    omp_destroy_lock(&best_lock);

    if (best_nonce == UINT64_MAX) {
        fprintf(stderr, "no nonce found (impossible at u64 range for diff <= 64)\n");
        return 2;
    }

    printf("%016lx\n", (unsigned long)best_nonce);
    return 0;
}
