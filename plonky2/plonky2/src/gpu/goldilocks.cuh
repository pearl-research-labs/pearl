// Copyright (c) icemining contributors
// SPDX-License-Identifier: MIT
//
// C1 — Goldilocks field device arithmetic, bit-exact to plonky2's
// `field/src/goldilocks_field.rs`. p = 2^64 - 2^32 + 1.
//
// plonky2 stores *noncanonical* u64s (value mod p, possibly >= ORDER) and every
// op preserves the correct value mod p. We replicate the same ops, so the final
// canonicalized output is byte-identical to plonky2 regardless of intermediate
// representation. Validated against `poseidon_fixtures` (CPU oracle).

#pragma once
#include <cstdint>

#define GL_EPSILON 0xFFFFFFFFULL              // 2^32 - 1
#define GL_ORDER   0xFFFFFFFF00000001ULL      // 2^64 - 2^32 + 1

// add_no_canonicalize_trashing_input / add_canonical_u64:
//   (res, carry) = x + y; res + EPSILON*carry   (no second-order overflow when
//   one operand is canonical — see plonky2 goldilocks_field.rs:438).
__device__ __forceinline__ uint64_t gl_add_no_canon(uint64_t x, uint64_t y) {
    uint64_t res = x + y;
    uint64_t carry = (res < x) ? 1ULL : 0ULL;
    return res + GL_EPSILON * carry;
}

// GoldilocksField::add (goldilocks_field.rs:304): two-step canonicalizing add.
__device__ __forceinline__ uint64_t gl_add(uint64_t a, uint64_t b) {
    uint64_t sum = a + b;
    uint64_t over = (sum < a) ? 1ULL : 0ULL;
    uint64_t sum2 = sum + over * GL_EPSILON;
    uint64_t over2 = (sum2 < sum) ? 1ULL : 0ULL;
    if (over2) sum2 += GL_EPSILON;           // exceedingly rare double-overflow
    return sum2;
}

// reduce128 (goldilocks_field.rs:456): 128-bit -> noncanonical 64-bit.
__device__ __forceinline__ uint64_t gl_reduce128(unsigned __int128 x) {
    uint64_t x_lo = (uint64_t)x;
    uint64_t x_hi = (uint64_t)(x >> 64);
    uint64_t x_hi_hi = x_hi >> 32;
    uint64_t x_hi_lo = x_hi & GL_EPSILON;

    uint64_t t0 = x_lo - x_hi_hi;
    if (x_lo < x_hi_hi) t0 -= GL_EPSILON;    // borrow (rare)
    uint64_t t1 = x_hi_lo * GL_EPSILON;
    return gl_add_no_canon(t0, t1);
}

// reduce96 (goldilocks_field.rs:447): (lo:u64, hi:u32) -> noncanonical 64-bit.
__device__ __forceinline__ uint64_t gl_reduce96(uint64_t lo, uint32_t hi) {
    uint64_t t1 = (uint64_t)hi * GL_EPSILON;
    return gl_add_no_canon(lo, t1);
}

__device__ __forceinline__ uint64_t gl_mul(uint64_t a, uint64_t b) {
    return gl_reduce128((unsigned __int128)a * (unsigned __int128)b);
}

// to_canonical_u64 (goldilocks_field.rs:264): one conditional subtraction.
__device__ __forceinline__ uint64_t gl_to_canonical(uint64_t x) {
    return (x >= GL_ORDER) ? (x - GL_ORDER) : x;
}

// a - b mod p. Canonicalize both (correct value mod p, rep-independent).
__device__ __forceinline__ uint64_t gl_sub(uint64_t a, uint64_t b) {
    uint64_t ca = gl_to_canonical(a), cb = gl_to_canonical(b);
    return (ca >= cb) ? (ca - cb) : (ca + (GL_ORDER - cb));
}

// base^e mod p (square-and-multiply). 1 is canonical; gl_mul(1,b)=b.
__device__ __forceinline__ uint64_t gl_pow(uint64_t base, uint64_t e) {
    uint64_t r = 1ULL, b = base;
    while (e) { if (e & 1ULL) r = gl_mul(r, b); b = gl_mul(b, b); e >>= 1; }
    return r;
}
