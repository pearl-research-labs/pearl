// Copyright (c) icemining contributors
// SPDX-License-Identifier: MIT
//
// FFI: GPU `PolynomialBatch::from_values` for GoldilocksField / PoseidonGoldilocks
// (blinding=false). Produces coeffs + bit-reversed transposed LDE leaves + Merkle
// digests (plonky2 layout) + cap. Validated byte-exact-by-value by pearl_gpu_commit.
// Compiled by plonky2 build.rs under feature `gpu_commit`.

#include <cstdint>
#include <vector>
#include "pearl_commit_kernels.cuh"   // poseidon/goldilocks + the from_values kernels

// out_coeffs:  num_polys * n
// out_leaves:  total * num_polys  (bit-reversed leaf order, as MerkleTree.leaves)
// out_digests: dig_len * 4        (plonky2 interleaved layout)
// out_cap:     (1<<cap_height) * 4
extern "C" int pearl_gpu_from_values_f64(
    const uint64_t* values, int num_polys, int degree_log, int rate_bits, int cap_height,
    const uint64_t* rtn, const int* offn, const uint64_t* rtl, const int* offl,
    uint64_t coset_shift, uint64_t n_inv,
    uint64_t* out_coeffs, uint64_t* out_leaves, uint64_t* out_digests, uint64_t* out_cap) {
    const int n = 1 << degree_log, lg_t = degree_log + rate_bits, total = 1 << lg_t;
    const int len_cap = 1 << cap_height, S = lg_t - cap_height, tpb = 256;

    poseidon_upload_constants();
    uint64_t *d_vals, *d_lde, *d_leaves, *d_tmp, *d_rtn, *d_rtl, *d_lvl, *d_coeffs;
    size_t rtn_len = (size_t)offn[degree_log - 1] + (1 << (degree_log - 1) > 2 ? (1 << (degree_log - 1)) : 2);
    // Sizes of root tables are passed implicitly via offsets; copy enough: recompute total len.
    // rtn has lg_n rows (sizes max(2,2^i)); rtl has lg_t rows. Caller packs them contiguously and
    // offn[i]/offl[i] are start offsets; total length = last_off + last_row_len. We compute below.
    size_t rtn_total = (size_t)offn[degree_log - 1] + (size_t)((1 << (degree_log - 1)) < 2 ? 2 : (1 << (degree_log - 1)));
    size_t rtl_total = (size_t)offl[lg_t - 1] + (size_t)((1 << (lg_t - 1)) < 2 ? 2 : (1 << (lg_t - 1)));

    cudaMalloc(&d_vals, (size_t)num_polys * n * 8);
    cudaMalloc(&d_coeffs, (size_t)num_polys * n * 8);
    cudaMalloc(&d_lde, (size_t)num_polys * total * 8);
    cudaMalloc(&d_leaves, (size_t)total * num_polys * 8);
    cudaMalloc(&d_tmp, (size_t)total * 8);
    cudaMalloc(&d_rtn, rtn_total * 8);
    cudaMalloc(&d_rtl, rtl_total * 8);
    std::vector<size_t> lvoff(S + 2); size_t acc = 0;
    for (int L = 0; L <= S; L++) { lvoff[L] = acc; acc += (size_t)(total >> L) * 4; } lvoff[S + 1] = acc;
    cudaMalloc(&d_lvl, acc * 8);
    if (cudaGetLastError() != cudaSuccess) return 10;

    cudaMemcpy(d_vals, values, (size_t)num_polys * n * 8, cudaMemcpyHostToDevice);
    cudaMemcpy(d_rtn, rtn, rtn_total * 8, cudaMemcpyHostToDevice);
    cudaMemcpy(d_rtl, rtl, rtl_total * 8, cudaMemcpyHostToDevice);

    for (int c = 0; c < num_polys; c++) {
        // iNTT(values[c]) -> coeffs[c]
        k_bitrev<<<(n + tpb - 1) / tpb, tpb>>>(d_vals + (size_t)c * n, d_tmp, n, degree_log);
        for (int s = 0; s < degree_log; s++) k_stage<<<((n / 2) + tpb - 1) / tpb, tpb>>>(d_tmp, d_rtn + offn[s], 1 << s, n);
        k_intt_post<<<((n / 2) + tpb - 1) / tpb, tpb>>>(d_tmp, n_inv, n);
        cudaMemcpy(d_coeffs + (size_t)c * n, d_tmp, (size_t)n * 8, cudaMemcpyDeviceToDevice);
        // coset-LDE -> forward NTT -> lde[c]
        uint64_t* ldc = d_lde + (size_t)c * total;
        k_coset<<<(total + tpb - 1) / tpb, tpb>>>(d_tmp, ldc, coset_shift, n, total);
        k_bitrev<<<(total + tpb - 1) / tpb, tpb>>>(ldc, d_tmp, total, lg_t);
        cudaMemcpy(ldc, d_tmp, (size_t)total * 8, cudaMemcpyDeviceToDevice);
        for (int s = 0; s < lg_t; s++) k_stage<<<((total / 2) + tpb - 1) / tpb, tpb>>>(ldc, d_rtl + offl[s], 1 << s, total);
    }
    // transpose (natural row order), then leaves in bit-reversed order + Merkle levels.
    k_transpose<<<(int)(((size_t)num_polys * total + tpb - 1) / tpb), tpb>>>(d_lde, d_leaves, num_polys, total);
    k_leaves_bitrev<<<(int)(((size_t)total + tpb - 1) / tpb), tpb>>>(d_leaves, (uint64_t*)d_lde /*scratch*/, num_polys, total, lg_t);
    // d_lde now holds bit-reversed leaves (reused as scratch); copy out.
    cudaMemcpy(out_leaves, d_lde, (size_t)total * num_polys * 8, cudaMemcpyDeviceToHost);
    k_hash_rows<<<(total + tpb - 1) / tpb, tpb>>>(d_lde, num_polys, d_lvl + lvoff[0], total);
    for (int L = 0; L < S; L++) { int m = (total >> L) / 2; k_reduce<<<(m + tpb - 1) / tpb, tpb>>>(d_lvl + lvoff[L], d_lvl + lvoff[L + 1], m); }
    if (cudaDeviceSynchronize() != cudaSuccess) return 11;

    cudaMemcpy(out_coeffs, d_coeffs, (size_t)num_polys * n * 8, cudaMemcpyDeviceToHost);
    std::vector<uint64_t> lvl(acc); cudaMemcpy(lvl.data(), d_lvl, acc * 8, cudaMemcpyDeviceToHost);

    // Gather level-order digests into plonky2 layout (raw noncanonical reps; the
    // Rust side wraps them and plonky2 compares by canonical value). out_leaves /
    // out_coeffs are left as the GPU produced them — no host canonicalize pass.
    const int n_sub = total >> cap_height, block_len = 2 * (n_sub - 1);
    const int dig_len = 2 * (total - len_cap);
    for (int c = 0; c < len_cap; c++) {
        for (int L = 0; L < S; L++) {
            int cnt = 1 << (S - L);
            for (int j = 0; j < cnt; j++) {
                size_t src = lvoff[L] + (size_t)(c * cnt + j) * 4;
                int pos = 2 * (((j >> 1) << (L + 1)) + (1 << L) - 1) + (j & 1);
                size_t dst = (size_t)(c * block_len + pos) * 4;
                for (int e = 0; e < 4; e++) out_digests[dst + e] = lvl[src + e];
            }
        }
        size_t cs = lvoff[S] + (size_t)c * 4;
        for (int e = 0; e < 4; e++) out_cap[(size_t)c * 4 + e] = lvl[cs + e];
    }

    cudaFree(d_vals); cudaFree(d_coeffs); cudaFree(d_lde); cudaFree(d_leaves); cudaFree(d_tmp);
    cudaFree(d_rtn); cudaFree(d_rtl); cudaFree(d_lvl);
    (void)rtn_len;
    return 0;
}
