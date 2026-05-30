// Copyright (c) icemining contributors
// SPDX-License-Identifier: MIT
//
// C1 — Goldilocks Poseidon permutation (width 12), bit-exact to plonky2's
// `<GoldilocksField as Poseidon>::poseidon`. We implement the *naive* schedule
// (poseidon.rs `poseidon_naive`): full(4) -> partial(22) -> full(4), which
// plonky2 ships as the verified bit-identical oracle of the production fast
// permutation. Needs only ALL_ROUND_CONSTANTS + MDS_MATRIX_CIRC/DIAG.

#pragma once
#include "goldilocks.cuh"
#include "poseidon_constants.h"

__constant__ uint64_t d_ALL_RC[360];
__constant__ uint64_t d_MDS_CIRC[12];
__constant__ uint64_t d_MDS_DIAG[12];

inline void poseidon_upload_constants() {
    cudaMemcpyToSymbol(d_ALL_RC, ALL_ROUND_CONSTANTS, sizeof(ALL_ROUND_CONSTANTS));
    cudaMemcpyToSymbol(d_MDS_CIRC, MDS_MATRIX_CIRC, sizeof(MDS_MATRIX_CIRC));
    cudaMemcpyToSymbol(d_MDS_DIAG, MDS_MATRIX_DIAG, sizeof(MDS_MATRIX_DIAG));
}

// x |--> x^7  (poseidon.rs sbox_monomial). Any correct x^7 is rep-independent.
__device__ __forceinline__ uint64_t gl_sbox(uint64_t x) {
    uint64_t x2 = gl_mul(x, x);
    uint64_t x4 = gl_mul(x2, x2);
    uint64_t x6 = gl_mul(x4, x2);
    return gl_mul(x6, x);
}

// mds_layer via mds_row_shf (poseidon.rs:180/271): result[r] = reduce96(
//   sum_i v[(i+r)%12]*CIRC[i] + v[r]*DIAG[r] ), u128 accumulation (fits < 2^73).
__device__ __forceinline__ void poseidon_mds_layer(uint64_t* s) {
    uint64_t res[12];
#pragma unroll
    for (int r = 0; r < 12; r++) {
        unsigned __int128 sum = 0;
#pragma unroll
        for (int i = 0; i < 12; i++)
            sum += (unsigned __int128)s[(i + r) % 12] * (unsigned __int128)d_MDS_CIRC[i];
        sum += (unsigned __int128)s[r] * (unsigned __int128)d_MDS_DIAG[r];
        res[r] = gl_reduce96((uint64_t)sum, (uint32_t)(sum >> 64));
    }
#pragma unroll
    for (int i = 0; i < 12; i++) s[i] = res[i];
}

// constant_layer (poseidon.rs:632): add_canonical_u64 of the round constants.
__device__ __forceinline__ void poseidon_constant_layer(uint64_t* s, int round_ctr) {
#pragma unroll
    for (int i = 0; i < 12; i++)
        s[i] = gl_add_no_canon(s[i], d_ALL_RC[i + 12 * round_ctr]);
}

// poseidon_naive (poseidon.rs:791): full(4) -> partial-naive(22) -> full(4).
__device__ void poseidon12(uint64_t* s) {
    int rc = 0;
    for (int k = 0; k < 4; k++) {
        poseidon_constant_layer(s, rc);
        for (int i = 0; i < 12; i++) s[i] = gl_sbox(s[i]);
        poseidon_mds_layer(s);
        rc++;
    }
    for (int k = 0; k < 22; k++) {
        poseidon_constant_layer(s, rc);
        s[0] = gl_sbox(s[0]);          // partial round: s-box on lane 0 only
        poseidon_mds_layer(s);
        rc++;
    }
    for (int k = 0; k < 4; k++) {
        poseidon_constant_layer(s, rc);
        for (int i = 0; i < 12; i++) s[i] = gl_sbox(s[i]);
        poseidon_mds_layer(s);
        rc++;
    }
}

// --- Hash wrappers (hashing.rs / merkle_tree.rs), RATE=8, NUM_HASH_OUT_ELTS=4 ---

// two_to_one / compress (hashing.rs:103): state=[l(4),r(4),0,0,0,0]; permute; out=state[0..4].
__device__ __forceinline__ void gl_two_to_one(const uint64_t* l, const uint64_t* r, uint64_t* out) {
    uint64_t s[12];
#pragma unroll
    for (int i = 0; i < 4; i++) s[i] = l[i];
#pragma unroll
    for (int i = 0; i < 4; i++) s[4 + i] = r[i];
#pragma unroll
    for (int i = 8; i < 12; i++) s[i] = 0;
    poseidon12(s);
#pragma unroll
    for (int i = 0; i < 4; i++) out[i] = s[i];
}

// hash_n_to_hash_no_pad (hashing.rs:124): overwrite-mode sponge, RATE=8, squeeze 4.
__device__ __forceinline__ void gl_hash_no_pad(const uint64_t* in, int len, uint64_t* out) {
    uint64_t s[12];
#pragma unroll
    for (int i = 0; i < 12; i++) s[i] = 0;
    int off = 0;
    while (off < len) {
        int chunk = (len - off) < 8 ? (len - off) : 8;
        for (int i = 0; i < chunk; i++) s[i] = in[off + i];   // overwrite first `chunk` of rate
        poseidon12(s);
        off += chunk;
    }
#pragma unroll
    for (int i = 0; i < 4; i++) out[i] = s[i];
}

// hash_or_noop (hashing.rs:14 / Hasher default): <=4 elems pad with zero; else sponge.
__device__ __forceinline__ void gl_hash_or_noop(const uint64_t* in, int len, uint64_t* out) {
    if (len <= 4) {
#pragma unroll
        for (int i = 0; i < 4; i++) out[i] = (i < len) ? in[i] : 0ULL;
    } else {
        gl_hash_no_pad(in, len, out);
    }
}
