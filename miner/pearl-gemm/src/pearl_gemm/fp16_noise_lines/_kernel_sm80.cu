// Bit-exact A100 (sm_80) FP16 deterministic noise-line generation.
//
// Reproduces the FP16 scheme verifier's keyed-BLAKE3 noise lines
// (zk-pow/src/api/fp16/noise.rs) element-for-element. One line is:
//
//   key      = subkey(LABEL_NOISE_LINE, seed)   (host-derived keyed BLAKE3)
//   material = [side:u8, factor:u8, line:u32 LE] zero-padded to 64 bytes
//   bytes    = keyed-BLAKE3-XOF(key, material)[0 .. rank]
//   x_i      = sign * magnitude, sign = 1-2*(b>>7), magnitude = (b&0x7F)+1
//   norm     = floor(||x||_2 * 32) via EXACT integer isqrt of sumsq*32^2
//   scale    = bf16(256*32) / bf16(norm)                  (one BF16 division)
//   entry_i  = fp16( bf16( bf16(x_i) * scale ) )          (RNE to FP16, u16)
//
// The ONLY difference from the FP8 recipe is the final cast target: FP16 (u16)
// rather than e4m3. Every BF16 op decodes exactly to f32, runs in f32 and
// rounds back RNE; the final cast is RNE-to-FP16 (__float2half_rn). The line
// key is derived on the host (blake3 keyed hash of the 24-byte label under the
// seed) and passed in as 8 little-endian u32 words, exactly as the FP8
// noise_lines kernel takes its side noise-line key.
//
// Thread mapping: one thread per line (count lines, each `rank` entries). The
// line index is indices[tid] (arbitrary global E rows/cols, or 0..k for F).

#include <cuda_fp16.h>
#include <torch/extension.h>

// ---- BF16 helpers (bit-exact to crate::api::fp8::{compute,dtype}) ----

__device__ __forceinline__ float bf16_to_f32(unsigned short bits) {
    return __uint_as_float((unsigned int)bits << 16);
}

// f32 -> BF16 round-to-nearest-ties-to-even (dtype::f32_to_bf16). Works on the
// raw bit pattern, so the sign rides through unchanged (matches the reference).
__device__ __forceinline__ unsigned short f32_to_bf16(float x) {
    unsigned int bits = __float_as_uint(x);
    unsigned int round_bit = (bits >> 16) & 1u;
    return (unsigned short)((bits + 0x7FFFu + round_bit) >> 16);
}

__device__ __forceinline__ unsigned short bf16_mul(unsigned short a, unsigned short b) {
    return f32_to_bf16(bf16_to_f32(a) * bf16_to_f32(b));
}

__device__ __forceinline__ unsigned short bf16_div(unsigned short a, unsigned short b) {
    return f32_to_bf16(bf16_to_f32(a) / bf16_to_f32(b));
}

// ---- BLAKE3 keyed single-block XOF (standard BLAKE3, bit-identical to the
// blake3 crate the reference hashes with) ----

__constant__ unsigned int BLAKE3_IV[8] = {
    0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au,
    0x510E527Fu, 0x9B05688Cu, 0x1F83D9ABu, 0x5BE0CD19u};

// Flags of a one-shot (single 64-byte block) keyed BLAKE3 hash:
// CHUNK_START | CHUNK_END | ROOT | KEYED_HASH = 1 | 2 | 8 | 16 = 27.
#define BLAKE3_SINGLE_KEYED_FLAGS 27u

__device__ __forceinline__ unsigned int rotr32(unsigned int x, unsigned int n) {
    return (x >> n) | (x << (32u - n));
}

// One BLAKE3 compression of a single 64-byte message block `m` under chaining
// value `cv`, for output-block counter `counter_lo` (hi word always 0 for our
// message sizes). Produces the full 16-word output (XOF): out[0..8] =
// state[0..8] ^ state[8..16]; out[8..16] = state[8..16] ^ cv[0..8].
__device__ void blake3_compress_xof(const unsigned int cv[8], const unsigned int m[16],
                                    unsigned int counter_lo, unsigned int flags,
                                    unsigned int out[16]) {
    unsigned int s[16];
#pragma unroll
    for (int i = 0; i < 8; ++i) s[i] = cv[i];
    s[8] = BLAKE3_IV[0];
    s[9] = BLAKE3_IV[1];
    s[10] = BLAKE3_IV[2];
    s[11] = BLAKE3_IV[3];
    s[12] = counter_lo;
    s[13] = 0u;
    s[14] = 64u;  // block_len (always a full 64-byte block here)
    s[15] = flags;

    unsigned int v[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) v[i] = m[i];

    // The fixed BLAKE3 message permutation.
    const int PERM[16] = {2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8};

#define G(a, b, c, d, x, y)                     \
    s[a] = s[a] + s[b] + (x);                   \
    s[d] = rotr32(s[d] ^ s[a], 16);             \
    s[c] = s[c] + s[d];                         \
    s[b] = rotr32(s[b] ^ s[c], 12);             \
    s[a] = s[a] + s[b] + (y);                   \
    s[d] = rotr32(s[d] ^ s[a], 8);              \
    s[c] = s[c] + s[d];                         \
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

// Exact floor integer sqrt of a u64.
__device__ __forceinline__ unsigned long long isqrt_u64(unsigned long long v) {
    if (v == 0ull) return 0ull;
    unsigned long long c = (unsigned long long)sqrt((double)v);
    // Clamp either side of the double rounding to the exact floor.
    while (c > 0ull && c * c > v) --c;
    while ((c + 1ull) * (c + 1ull) <= v) ++c;
    return c;
}

// ---- Kernel: one keyed-BLAKE3 noise line per thread ----

__global__ void fp16_noise_lines_kernel(const unsigned int* __restrict__ line_key,
                                        unsigned char side, unsigned char factor,
                                        const int* __restrict__ indices, int count, int rank,
                                        unsigned short* __restrict__ out) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= count) return;

    unsigned int cv[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) cv[i] = line_key[i];

    unsigned int line = (unsigned int)indices[tid];
    unsigned int m[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) m[i] = 0u;
    // material = side | factor<<8 | line(u32 LE), zero-padded to one block.
    m[0] = (unsigned int)side | ((unsigned int)factor << 8) | ((line & 0xFFFFu) << 16);
    m[1] = line >> 16;

    // Keyed-BLAKE3-XOF `rank` bytes (64 per output block).
    // rank is small (protocol r = 32); support arbitrary rank via the counter.
    unsigned long long sumsq = 0ull;
    unsigned char bytes[256];  // rank well below this in every use
    for (int off = 0; off < rank; off += 64) {
        unsigned int w[16];
        blake3_compress_xof(cv, m, (unsigned int)(off / 64), BLAKE3_SINGLE_KEYED_FLAGS, w);
#pragma unroll
        for (int b = 0; b < 64; ++b) {
            int idx = off + b;
            if (idx < rank) bytes[idx] = (unsigned char)((w[b >> 2] >> (8 * (b & 3))) & 0xFFu);
        }
    }
    for (int i = 0; i < rank; ++i) {
        unsigned int mag = (unsigned int)(bytes[i] & 0x7F) + 1u;
        sumsq += (unsigned long long)mag * (unsigned long long)mag;
    }

    unsigned long long norm_scaled = isqrt_u64(sumsq * 1024ull);  // INT_SQRT_PREC^2 = 1024
    unsigned short numer = f32_to_bf16(8192.0f);                  // NOISE_TARGET_NORM * 32
    unsigned short denom = f32_to_bf16((float)norm_scaled);
    unsigned short scale_bf = bf16_div(numer, denom);
    float scale_f = bf16_to_f32(scale_bf);

    unsigned short* line_out = out + (long)tid * rank;
    for (int i = 0; i < rank; ++i) {
        unsigned char b = bytes[i];
        int sign = 1 - 2 * (int)(b >> 7);
        int mag = (int)(b & 0x7F) + 1;
        float xi = (float)(sign * mag);  // |xi| <= 128, exact in bf16 and f32
        unsigned short prod_bf = f32_to_bf16(xi * scale_f);
        line_out[i] = __half_as_ushort(__float2half_rn(bf16_to_f32(prod_bf)));
    }
}

// ---- Host launcher ----

torch::Tensor fp16_noise_lines(torch::Tensor line_key, long side, long factor,
                               torch::Tensor indices, long rank) {
    TORCH_CHECK(line_key.is_cuda() && line_key.scalar_type() == torch::kInt32,
                "line_key must be an int32 CUDA tensor of 8 words");
    TORCH_CHECK(line_key.numel() == 8, "line_key must have 8 u32 words (32 bytes)");
    TORCH_CHECK(indices.is_cuda() && indices.scalar_type() == torch::kInt32,
                "indices must be an int32 CUDA tensor");
    TORCH_CHECK(rank > 0 && rank <= 256, "rank must be in 1..=256");
    line_key = line_key.contiguous();
    indices = indices.contiguous();
    int count = (int)indices.numel();

    auto opts = torch::dtype(torch::kInt16).device(line_key.device());
    auto out = torch::empty({count, (long)rank}, opts);
    if (count == 0) return out;

    int threads = 128;
    int blocks = (count + threads - 1) / threads;
    fp16_noise_lines_kernel<<<blocks, threads>>>(
        (const unsigned int*)line_key.data_ptr(), (unsigned char)side, (unsigned char)factor,
        (const int*)indices.data_ptr(), count, (int)rank, (unsigned short*)out.data_ptr());
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "fp16_noise_lines launch failed");
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fp16_noise_lines", &fp16_noise_lines,
          "A100 sm_80 keyed-BLAKE3 FP16 noise-line generation (u16 E/F factors)",
          pybind11::arg("line_key"), pybind11::arg("side"), pybind11::arg("factor"),
          pybind11::arg("indices"), pybind11::arg("rank"));
}
