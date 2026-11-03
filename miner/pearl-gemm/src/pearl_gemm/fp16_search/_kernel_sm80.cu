// Bit-exact A100 (sm_80) full-matrix FP16 lottery search + first-winner latch.
//
// The FP16 analogue of the lottery fused inside the FP8 `mixed_gemm`: scan every
// committed tile of a noised matmul A'@B'^T on-GPU and latch the FIRST tile (by
// flat tile index) whose jackpot ticket clears the difficulty threshold.
//
// For each tile (tr, tc) over the (m/h) x (n/w) grid:
//   * compute the h x w tile A'[tr-block] @ B'[tc-block]^T with the SAME
//     bit-exact A100 accumulation as fp16_policy / fp16_gemm
//     (zk-pow/src/api/fp16/accumulate.rs :: a100_dot) -- reused verbatim here,
//     minus the policy census the search does not need, so each cell is the
//     identical f32 bit pattern;
//   * fold the tile's f32 BIT patterns into a 64-byte message over the committed
//     16-lane layout (crate::api::fp8::utils::xor_fold_extract): per lane a
//     rolling acc = (acc*0x9E3779B1 + f32_bits).rotate_left(13) over the lane's
//     cells, 16 u32 little-endian;
//   * ticket = keyed-BLAKE3(message, key = pow_key) -- pow_key is the jackpot
//     subkey the host precomputes (BLAKE3("pearl/v4/FP8/jackpot", key=seed_a)),
//     exactly like fp8 passes pow_key into mixed_gemm. A 64-byte message is one
//     full BLAKE3 block / one chunk, so the ticket is a single ROOT-finalized
//     keyed compression (reused verbatim from fp16_commit / fp16_noise_lines,
//     the compression agent B validated bit-identical to the `blake3` crate);
//   * win iff le(ticket) <= bound, where bound = min(U256::MAX, difficulty(nbits)
//     * saturating_u32(h*w*k)) is precomputed host-side (exact bigint) and passed
//     in as 8 little-endian u32 words -- a 256-bit little-endian compare here is
//     bit-for-bit crate::api::proof_utils::check_jackpot_difficulty.
//
// First-winner discipline (deterministic, lowest flat tile index wins ties):
// the scan kernel (one block per tile) does atomicMin over the winning tiles'
// flat indices, so the latched index is independent of block completion order.
// A second single-block kernel then recomputes that one tile to fill the ticket,
// avoiding both a per-tile ticket scratch and any racy "store my ticket under a
// lock" across the grid.
//
//   A: (m, k) row-major FP16  (noised left operand A')
//   B: (n, k) row-major FP16  (noised, transposed right operand B': row j is
//      logical column j, exactly a100_matmul's `b`)
//   k is arbitrary (>= 1); no tile-granularity constraint (software path).

#include <cuda_fp16.h>
#include <torch/extension.h>

#define SEARCH_W 24
#define SEARCH_GROUP 8
// i32::MIN / 2, the Rust "no exponent" sentinel.
#define SEARCH_NEG (-1073741824)
#define FP32_MIN_EXP (-149)

// ---- FP16 operand decomposition (dtype::decompose_fp16) ----
__device__ __forceinline__ void decompose_fp16(unsigned short bits, int* sign,
                                                long long* sig, int* eps) {
    unsigned exp = (bits >> 10) & 0x1F;
    long long man = (long long)(bits & 0x03FF);
    *sign = (bits & 0x8000) ? -1 : 1;
    if (exp == 0 && man == 0) {
        *sign = 1;
        *sig = 0;
        *eps = 0;
    } else if (exp == 0) {
        *sig = man;           // subnormal
        *eps = -14;
    } else {
        *sig = 0x400 | man;   // normal: implicit 1
        *eps = (int)exp - 15;
    }
}

// ---- FP32 accumulator decomposition (accumulate::acc_parts) ----
__device__ __forceinline__ void acc_parts(float c, long long* sign,
                                           unsigned long long* sig, int* el, int* ulp) {
    unsigned int bits = __float_as_uint(c);
    if ((bits & 0x7FFFFFFFu) == 0u) {  // +/-0
        *sign = 1;
        *sig = 0ull;
        *el = SEARCH_NEG;
        *ulp = 0;
        return;
    }
    *sign = (bits >> 31) ? -1 : 1;
    int exp_field = (int)((bits >> 23) & 0xFF);
    unsigned long long man = (unsigned long long)(bits & 0x7FFFFFu);
    if (exp_field > 0) {
        int el_v = exp_field - 127;
        *sig = 0x800000ull | man;  // 24-bit significand (implicit 1)
        *el = el_v;
        *ulp = el_v - 23;
    } else {
        *sig = man;                // subnormal
        *el = -126;
        *ulp = FP32_MIN_EXP;
    }
}

// `x >> (-s)` truncating toward zero for s<0, `x << s` for s>=0 (x non-negative).
__device__ __forceinline__ long long shift_nonneg(long long x, int s) {
    if (s >= 0) return x << s;
    int rs = -s;
    if (rs >= 64) return 0;
    return x >> rs;
}

// Rounds the integer s * 2^unit toward zero to FP32 (accumulate::rz_to_f32).
__device__ __forceinline__ float rz_to_f32(long long s, int unit) {
    if (s == 0) return 0.0f;
    double sign = (s < 0) ? -1.0 : 1.0;
    unsigned long long a = (unsigned long long)(s < 0 ? -s : s);
    int nb = 63 - __clzll(a);
    int keep = nb + unit - 23;
    if (keep < FP32_MIN_EXP) keep = FP32_MIN_EXP;
    int drop = keep - unit;
    if (drop < 0) drop = 0;
    if (drop > 127) drop = 127;
    unsigned long long truncated = (drop >= 64) ? 0ull : ((a >> drop) << drop);
    double val = sign * (double)truncated * ldexp(1.0, unit);
    return (float)val;
}

// One output cell's a100_dot (fp16_policy's accumulation, census dropped).
__device__ float a100_dot(const __half* __restrict__ a, const __half* __restrict__ b, int k) {
    float cur = 0.0f;
    for (int g0 = 0; g0 < k; g0 += SEARCH_GROUP) {
        int g1 = min(g0 + SEARCH_GROUP, k);
        long long csgn;
        unsigned long long cm;
        int cel, culp;
        acc_parts(cur, &csgn, &cm, &cel, &culp);

        int eta = cel;
        for (int u = g0; u < g1; ++u) {
            int sa, sb;
            long long ma, mb;
            int ea, eb;
            decompose_fp16(__half_as_ushort(a[u]), &sa, &ma, &ea);
            decompose_fp16(__half_as_ushort(b[u]), &sb, &mb, &eb);
            if (ma != 0 && mb != 0) {
                int e = ea + eb;
                if (e > eta) eta = e;
            }
        }
        if (eta == SEARCH_NEG) continue;  // empty no-op group leaves cur unchanged
        int unit = eta - SEARCH_W;

        long long sum = 0;
        for (int u = g0; u < g1; ++u) {
            int sa, sb;
            long long ma, mb;
            int ea, eb;
            decompose_fp16(__half_as_ushort(a[u]), &sa, &ma, &ea);
            decompose_fp16(__half_as_ushort(b[u]), &sb, &mb, &eb);
            if (ma == 0 || mb == 0) continue;
            long long prod = ma * mb;
            int sh = (ea + eb) - 20 - unit;
            long long aligned = shift_nonneg(prod, sh);
            sum += (long long)(sa * sb) * aligned;
        }
        int csh = culp - unit;
        long long acc_aligned = shift_nonneg((long long)cm, csh);
        sum += csgn * acc_aligned;

        cur = rz_to_f32(sum, unit);
    }
    return cur;
}

// ---- BLAKE3 keyed compression (from fp16_commit / fp16_noise_lines; crate-identical) ----

__constant__ unsigned int BLAKE3_IV[8] = {
    0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au,
    0x510E527Fu, 0x9B05688Cu, 0x1F83D9ABu, 0x5BE0CD19u};

#define B3F_CHUNK_START 1u
#define B3F_CHUNK_END 2u
#define B3F_ROOT 8u
#define B3F_KEYED_HASH 16u

__device__ __forceinline__ unsigned int rotr32(unsigned int x, unsigned int n) {
    return (x >> n) | (x << (32u - n));
}

__device__ void blake3_compress(const unsigned int cv[8], const unsigned int m[16],
                                unsigned int counter_lo, unsigned int counter_hi,
                                unsigned int flags, unsigned int out[16]) {
    unsigned int s[16];
#pragma unroll
    for (int i = 0; i < 8; ++i) s[i] = cv[i];
    s[8] = BLAKE3_IV[0];
    s[9] = BLAKE3_IV[1];
    s[10] = BLAKE3_IV[2];
    s[11] = BLAKE3_IV[3];
    s[12] = counter_lo;
    s[13] = counter_hi;
    s[14] = 64u;  // block_len (always a full 64-byte block here)
    s[15] = flags;

    unsigned int v[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) v[i] = m[i];

    const int PERM[16] = {2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8};

#define G(a, b, c, d, x, y)         \
    s[a] = s[a] + s[b] + (x);       \
    s[d] = rotr32(s[d] ^ s[a], 16); \
    s[c] = s[c] + s[d];             \
    s[b] = rotr32(s[b] ^ s[c], 12); \
    s[a] = s[a] + s[b] + (y);       \
    s[d] = rotr32(s[d] ^ s[a], 8);  \
    s[c] = s[c] + s[d];             \
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

// ---- Lottery fold + ticket + difficulty ----

__device__ __forceinline__ unsigned int rotl32(unsigned int x, unsigned int n) {
    return (x << n) | (x >> (32u - n));
}

// XOR-fold the h x w tile's f32 bit patterns into the 16 u32 message words over
// the committed lane layout, then one ROOT-finalized keyed-BLAKE3 compression of
// that 64-byte message under `key` -> the 32-byte ticket (8 LE u32 words).
__device__ void fold_and_hash(const unsigned int* __restrict__ tile_bits,
                              const int* __restrict__ lanes, int lane_len,
                              const unsigned int* __restrict__ key,
                              unsigned int ticket[8]) {
    unsigned int msg[16];
    for (int lane = 0; lane < 16; ++lane) {
        const int* idx = lanes + (long)lane * lane_len;
        unsigned int acc = 0u;
        for (int t = 0; t < lane_len; ++t) {
            acc = acc * 0x9E3779B1u;
            acc = acc + tile_bits[idx[t]];
            acc = rotl32(acc, 13u);
        }
        msg[lane] = acc;
    }
    unsigned int cv[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) cv[i] = key[i];
    unsigned int o[16];
    blake3_compress(cv, msg, 0u, 0u,
                    B3F_KEYED_HASH | B3F_CHUNK_START | B3F_CHUNK_END | B3F_ROOT, o);
#pragma unroll
    for (int i = 0; i < 8; ++i) ticket[i] = o[i];
}

// le(ticket) <= le(bound) as 256-bit little-endian unsigned (word 0 least sig).
__device__ __forceinline__ bool le_u256(const unsigned int x[8], const unsigned int b[8]) {
#pragma unroll
    for (int i = 7; i >= 0; --i) {
        if (x[i] < b[i]) return true;
        if (x[i] > b[i]) return false;
    }
    return true;  // equal
}

// Fill s_tile[0..h*w) with the tile's f32 bit patterns (block-strided).
__device__ void fill_tile(const __half* __restrict__ A, const __half* __restrict__ B,
                          int k, int h, int w, int tr, int tc,
                          unsigned int* __restrict__ s_tile) {
    int hw = h * w;
    for (int c = threadIdx.x; c < hw; c += blockDim.x) {
        int r = c / w;
        int col = c % w;
        const __half* a = A + (long)(tr * h + r) * k;
        const __half* b = B + (long)(tc * w + col) * k;
        s_tile[c] = __float_as_uint(a100_dot(a, b, k));
    }
}

// ---- Kernels ----

// Scan: one block per tile. Winners atomicMin their flat index into *min_idx
// (sentinel = num_tiles) and raise *found. Optionally dumps every tile's ticket.
__global__ void search_scan_kernel(const __half* __restrict__ A, const __half* __restrict__ B,
                                    int k, int h, int w, int ntc,
                                    const int* __restrict__ lanes, int lane_len,
                                    const unsigned int* __restrict__ key,
                                    const unsigned int* __restrict__ bound,
                                    int* __restrict__ found, int* __restrict__ min_idx,
                                    unsigned int* __restrict__ all_tickets, int collect) {
    extern __shared__ unsigned int s_tile[];
    int tile = blockIdx.x;
    int tr = tile / ntc;
    int tc = tile % ntc;
    fill_tile(A, B, k, h, w, tr, tc, s_tile);
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned int ticket[8];
        fold_and_hash(s_tile, lanes, lane_len, key, ticket);
        if (collect) {
#pragma unroll
            for (int i = 0; i < 8; ++i) all_tickets[(long)tile * 8 + i] = ticket[i];
        }
        if (le_u256(ticket, bound)) {
            atomicMin(min_idx, tile);
            atomicMax(found, 1);
        }
    }
}

// Latch: recompute the single winning tile and write (tr, tc, ticket[8]) out.
__global__ void search_latch_kernel(const __half* __restrict__ A, const __half* __restrict__ B,
                                     int k, int h, int w, int ntc, int tile,
                                     const int* __restrict__ lanes, int lane_len,
                                     const unsigned int* __restrict__ key,
                                     int* __restrict__ out) {
    extern __shared__ unsigned int s_tile[];
    int tr = tile / ntc;
    int tc = tile % ntc;
    fill_tile(A, B, k, h, w, tr, tc, s_tile);
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned int ticket[8];
        fold_and_hash(s_tile, lanes, lane_len, key, ticket);
        out[0] = tr;
        out[1] = tc;
#pragma unroll
        for (int i = 0; i < 8; ++i) out[2 + i] = (int)ticket[i];
    }
}

// ---- Host launcher ----
// Returns {found (1,) i32, min_idx (1,) i32, latch (10,) i32 = tr,tc,ticket[8],
//          tickets (num_tiles*8 or 0,) i32}.
std::vector<torch::Tensor> fp16_search(torch::Tensor A, torch::Tensor B,
                                       torch::Tensor lanes, torch::Tensor key,
                                       torch::Tensor bound, long h, long w, int collect) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "A and B must be CUDA tensors");
    TORCH_CHECK(A.scalar_type() == torch::kFloat16, "A must be float16");
    TORCH_CHECK(B.scalar_type() == torch::kFloat16, "B must be float16");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), "A and B must be contiguous");
    TORCH_CHECK(lanes.is_cuda() && lanes.scalar_type() == torch::kInt32 && lanes.is_contiguous(),
                "lanes must be a contiguous int32 CUDA tensor");
    TORCH_CHECK(key.is_cuda() && key.scalar_type() == torch::kInt32 && key.numel() == 8,
                "key must be an int32 CUDA tensor of 8 words");
    TORCH_CHECK(bound.is_cuda() && bound.scalar_type() == torch::kInt32 && bound.numel() == 8,
                "bound must be an int32 CUDA tensor of 8 words");
    long m = A.size(0), k = A.size(1);
    long n = B.size(0), kb = B.size(1);
    TORCH_CHECK(k == kb, "A and B must share k");
    TORCH_CHECK(k >= 1, "k must be >= 1");
    TORCH_CHECK(h >= 1 && w >= 1, "h and w must be >= 1");
    TORCH_CHECK(m % h == 0, "m must be a multiple of h");
    TORCH_CHECK(n % w == 0, "n must be a multiple of w");
    long hw = h * w;
    TORCH_CHECK(hw % 16 == 0, "h*w must be a multiple of 16 (16 lanes)");
    long lane_len = hw / 16;
    TORCH_CHECK(lanes.numel() == 16 * lane_len, "lanes must be 16 x (h*w/16)");
    long smem = hw * (long)sizeof(unsigned int);
    TORCH_CHECK(smem <= 48 * 1024, "tile too large for 48 KB shared memory (h*w <= 12288)");

    long ntr = m / h, ntc = n / w;
    long num_tiles = ntr * ntc;

    auto i32 = torch::dtype(torch::kInt32).device(A.device());
    auto found = torch::zeros({1}, i32);
    auto min_idx = torch::full({1}, (int)num_tiles, i32);
    auto latch = torch::zeros({10}, i32);
    auto tickets = torch::zeros({collect ? num_tiles * 8 : 0}, i32);
    if (num_tiles == 0) return {found, min_idx, latch, tickets};

    int threads = (int)min(hw, (long)256);
    const __half* Ap = (const __half*)A.data_ptr();
    const __half* Bp = (const __half*)B.data_ptr();
    const int* lanes_p = (const int*)lanes.data_ptr();
    const unsigned int* key_p = (const unsigned int*)key.data_ptr();
    const unsigned int* bound_p = (const unsigned int*)bound.data_ptr();

    search_scan_kernel<<<num_tiles, threads, smem>>>(
        Ap, Bp, (int)k, (int)h, (int)w, (int)ntc, lanes_p, (int)lane_len, key_p, bound_p,
        (int*)found.data_ptr(), (int*)min_idx.data_ptr(),
        (unsigned int*)tickets.data_ptr(), collect);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "fp16_search scan launch failed");

    int found_h = found.cpu().item<int>();
    if (found_h) {
        int winner = min_idx.cpu().item<int>();
        search_latch_kernel<<<1, threads, smem>>>(
            Ap, Bp, (int)k, (int)h, (int)w, (int)ntc, winner, lanes_p, (int)lane_len, key_p,
            (int*)latch.data_ptr());
        TORCH_CHECK(cudaGetLastError() == cudaSuccess, "fp16_search latch launch failed");
    }
    return {found, min_idx, latch, tickets};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, mod) {
    mod.def("fp16_search", &fp16_search,
            "A100 sm_80 full-matrix FP16 lottery search + first-winner latch",
            pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("lanes"),
            pybind11::arg("key"), pybind11::arg("bound"), pybind11::arg("h"),
            pybind11::arg("w"), pybind11::arg("collect"));
}
