// Copyright (c) icemining contributors
// SPDX-License-Identifier: MIT
// Self-contained kernels for the GPU from_values FFI (field + poseidon + tree).
#pragma once
#include "goldilocks.cuh"
#include "poseidon.cuh"

__global__ void k_bitrev(const uint64_t* in, uint64_t* out, int n, int bits) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    out[i] = in[__brev((uint32_t)i) >> (32 - bits)];
}
__global__ void k_stage(uint64_t* a, const uint64_t* rs, int half_m, int n) {
    int t = blockIdx.x * blockDim.x + threadIdx.x; if (t >= n / 2) return;
    int j = t & (half_m - 1), b = t / half_m, k0 = b * (half_m << 1), i1 = k0 + j, i2 = i1 + half_m;
    uint64_t u = a[i1], tw = gl_mul(rs[j], a[i2]); a[i1] = gl_add(u, tw); a[i2] = gl_sub(u, tw);
}
__global__ void k_intt_post(uint64_t* a, uint64_t ninv, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n / 2) return;
    if (i == 0) { a[0] = gl_mul(a[0], ninv); a[n / 2] = gl_mul(a[n / 2], ninv); }
    else { uint64_t ai = a[i], aj = a[n - i]; a[i] = gl_mul(aj, ninv); a[n - i] = gl_mul(ai, ninv); }
}
__global__ void k_coset(const uint64_t* c, uint64_t* o, uint64_t shift, int n, int total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= total) return;
    o[i] = (i < n) ? gl_mul(c[i], gl_pow(shift, (uint64_t)i)) : 0ULL;
}
__global__ void k_transpose(const uint64_t* lde, uint64_t* lv, int nc, int total) {
    size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x; if (t >= (size_t)nc * total) return;
    lv[(size_t)(t % total) * nc + (t / total)] = lde[t];
}
// out[i row] = in[bitrev(i) row]   (each row nc wide)
__global__ void k_leaves_bitrev(const uint64_t* in, uint64_t* out, int nc, int total, int bits) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= total) return;
    uint32_t src = __brev((uint32_t)i) >> (32 - bits);
    for (int e = 0; e < nc; e++) out[(size_t)i * nc + e] = in[(size_t)src * nc + e];
}
__global__ void k_hash_rows(const uint64_t* lv, int nc, uint64_t* out, int n_leaves) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; if (idx >= n_leaves) return;
    gl_hash_or_noop(lv + (size_t)idx * nc, nc, out + (size_t)idx * 4);
}
__global__ void k_reduce(const uint64_t* cur, uint64_t* nx, int m) {
    int j = blockIdx.x * blockDim.x + threadIdx.x; if (j >= m) return;
    gl_two_to_one(cur + (size_t)(2 * j) * 4, cur + (size_t)(2 * j + 1) * 4, nx + (size_t)j * 4);
}
