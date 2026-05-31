/* pearl_challenge_solver.c
 *
 * Brute-force solver for the alphapool v1.5 `pearl.challenge` DDoS pacer:
 *   find u64 nonce s.t. blake3(seed||nonce.to_le_bytes(8)) has `difficulty`
 *   leading zero bits.
 *
 * Build (statically linked, no libblake3 on system needed):
 *   gcc -O3 -march=native -fopenmp -DBLAKE3_NO_AVX512 \
 *       -I<blake3-c-src> \
 *       _pearl_challenge_solver.c \
 *       <blake3-c-src>/blake3.c <blake3-c-src>/blake3_dispatch.c \
 *       <blake3-c-src>/blake3_portable.c \
 *       <blake3-c-src>/blake3_sse2.c <blake3-c-src>/blake3_sse41.c \
 *       <blake3-c-src>/blake3_avx2.c \
 *       <blake3-c-src>/blake3_sse2_x86-64_unix.S \
 *       <blake3-c-src>/blake3_sse41_x86-64_unix.S \
 *       <blake3-c-src>/blake3_avx2_x86-64_unix.S \
 *       -o pearl_challenge_solver
 *
 * Usage:  ./pearl_challenge_solver <seed_hex(64 chars)> <difficulty>
 * Output: <nonce_u64_be_hex_16_chars>\n   on stdout
 *
 * Search strategy: parallel forward scan from nonce=0. Each OpenMP thread
 * grabs a stride-spaced lane. First thread to find a solution stores the
 * minimum nonce found and signals others via a shared 'found' flag. We bias
 * for the SMALLEST nonce (matches the Python reference solver semantics)
 * but the pool accepts ANY qualifying nonce.
 *
 * Performance budget on AMD Ryzen 9 7950X (32 logical cores, AVX-512):
 *   single-thread blake3 40-byte msg ~ 200-400 ns -> 2.5-5 MH/s
 *   32 threads, AVX2 (--no-avx512): ~80-160 MH/s expected
 *   2^32 hashes at 100 MH/s = 43 sec; at 160 MH/s = 27 sec
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdatomic.h>

#include <omp.h>

#include "blake3.h"

static inline int has_leading_zero_bits(const uint8_t *h, int n) {
    int full = n >> 3;
    int rem = n & 7;
    for (int i = 0; i < full; i++) {
        if (h[i] != 0) return 0;
    }
    if (rem == 0) return 1;
    /* check top `rem` bits of h[full] are zero */
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

    /* Shared state across threads. We want the MINIMUM nonce satisfying the
     * predicate. Use atomic for the flag; locked update of the best nonce. */
    atomic_int found_flag = 0;
    uint64_t best_nonce = UINT64_MAX;
    omp_lock_t best_lock;
    omp_init_lock(&best_lock);

    int max_threads = omp_get_max_threads();

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();
        blake3_hasher hasher;
        uint8_t hash[BLAKE3_OUT_LEN];
        uint8_t msg[40];
        memcpy(msg, seed, 32);

        /* Lane-striping: thread t scans nonce = t, t+N, t+2N, ... */
        uint64_t local_best = UINT64_MAX;
        uint64_t chunk = (uint64_t)1u << 22;   /* 4M nonces per check-in */
        uint64_t nonce = (uint64_t)tid;
        uint64_t step = (uint64_t)nthreads;
        uint64_t end_of_chunk = nonce + chunk * step;

        while (!atomic_load_explicit(&found_flag, memory_order_relaxed)) {
            for (; nonce < end_of_chunk; nonce += step) {
                /* Write nonce as u64 little-endian into msg[32..40] */
                msg[32] = (uint8_t)(nonce);
                msg[33] = (uint8_t)(nonce >> 8);
                msg[34] = (uint8_t)(nonce >> 16);
                msg[35] = (uint8_t)(nonce >> 24);
                msg[36] = (uint8_t)(nonce >> 32);
                msg[37] = (uint8_t)(nonce >> 40);
                msg[38] = (uint8_t)(nonce >> 48);
                msg[39] = (uint8_t)(nonce >> 56);
                blake3_hasher_init(&hasher);
                blake3_hasher_update(&hasher, msg, 40);
                blake3_hasher_finalize(&hasher, hash, BLAKE3_OUT_LEN);
                if (has_leading_zero_bits(hash, difficulty)) {
                    local_best = nonce;
                    goto found_in_thread;
                }
            }
            end_of_chunk = nonce + chunk * step;
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
        (void)max_threads;
    }

    omp_destroy_lock(&best_lock);

    if (best_nonce == UINT64_MAX) {
        fprintf(stderr, "no nonce found (impossible at u64 range for diff <= 64)\n");
        return 2;
    }

    /* Print as 16-hex-char string, matching Python f"{nonce:016x}" */
    printf("%016lx\n", (unsigned long)best_nonce);
    return 0;
}
