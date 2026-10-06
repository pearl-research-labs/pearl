// Bit-exact A100 (sm_80) FP16 "unpredictable accumulation steps" policy census.
//
// Replays the FP16 scheme verifier's per-cell accumulation
// (zk-pow/src/api/fp16/accumulate.rs :: a100_dot with census = Some) and fuses
// the per-cell policy reduction (zk-pow/src/api/fp16/policy.rs :: evaluate)
// in-kernel. One thread owns one output cell (i, j) of the m x n tile and:
//
//   * reproduces the integer accumulation model natively in software (NOT the
//     native HMMA datapath): the k axis is split into groups of GROUP = 8
//     products, each group aligns onto the 2^(eta - W) grid (W = 24) by
//     truncation toward zero, sums exactly, then rounds the sum toward zero back
//     to FP32. This software path is what exposes the census the HMMA
//     instruction hides;
//   * emits, per group, the two census quantities a100_dot records --
//     breakpoint (the accumulator alignment OR the final FP32 round-toward-zero
//     discarded a nonzero bit) and products_truncated (how many of the group's
//     products lost nonzero bits in their right-shift alignment) -- and folds
//     them into the per-cell integers evaluate() consumes:
//       n_bp   = # breakpoint steps,
//       n_runs = # maximal runs of non-breakpoint steps (empty no-op groups are
//                non-breakpoint steps with 0 truncations, so they extend the
//                surrounding run, exactly as PolicyStep::default() does),
//       n_pt   = sum of products_truncated over non-breakpoint steps.
//
// The kernel writes the recomputed FP32 tile (bit-exact to the verifier's
// replay / fp16_gemm) and the per-cell (n_bp, n_runs, n_pt). The host folds the
// per-cell integers into the tile totals (breakpoints, numerator) and the f64
// f_bp / rho / accept, matching evaluate() bit-for-bit.
//
// All intermediate magnitudes are bounded: a group sum is < 2^30 (8 products
// each < 2^26 plus an accumulator term < 2^25), so a 64-bit signed accumulator
// is exact and no i128 is needed (unlike the Rust reference, whose i128 is only
// for headroom). Every shift operand is non-negative (signs are applied
// separately), so a plain logical >> is the Rust truncating shift_i128.
//
//   A: (m, k) row-major FP16  (logical left operand)
//   B: (n, k) row-major FP16  (transposed logical right operand: row j is
//      logical column j, exactly a100_matmul's `b`)
//   k is arbitrary (>= 1); no tile-granularity constraint (software path).

#include <cuda_fp16.h>
#include <torch/extension.h>

#define POLICY_W 24
#define POLICY_GROUP 8
// i32::MIN / 2, the Rust "no exponent" sentinel.
#define POLICY_NEG (-1073741824)
#define FP32_MIN_EXP (-149)

// ---- FP16 operand decomposition (dtype::decompose_fp16) ----
// value = sign * sig * 2^(eps - 10); eps clamped to -14 for subnormals.
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
// value = sign * sig * 2^ulp; el is the stored exponent for the alignment max
// (clamped to -126 for subnormals), POLICY_NEG for a zero accumulator.
__device__ __forceinline__ void acc_parts(float c, long long* sign,
                                           unsigned long long* sig, int* el, int* ulp) {
    unsigned int bits = __float_as_uint(c);
    if ((bits & 0x7FFFFFFFu) == 0u) {  // +/-0
        *sign = 1;
        *sig = 0ull;
        *el = POLICY_NEG;
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

// `x >> (-s)` truncating toward zero for s<0, `x << s` for s>=0. x is always
// non-negative here (prod, cm), so a logical shift is the Rust shift_i128.
// Guards shifts >= 64 (x < 2^24, so the result is 0, as i128 >> gives).
__device__ __forceinline__ long long shift_nonneg(long long x, int s) {
    if (s >= 0) return x << s;
    int rs = -s;
    if (rs >= 64) return 0;
    return x >> rs;
}

// Rounds the integer s * 2^unit toward zero to FP32 (accumulate::rz_to_f32).
// |s| < 2^30 so a u64 magnitude is exact. Returns the FP32 value.
__device__ __forceinline__ float rz_to_f32(long long s, int unit) {
    if (s == 0) return 0.0f;
    double sign = (s < 0) ? -1.0 : 1.0;
    unsigned long long a = (unsigned long long)(s < 0 ? -s : s);
    int nb = 63 - __clzll(a);                 // floor(log2|s|)
    int keep = nb + unit - 23;
    if (keep < FP32_MIN_EXP) keep = FP32_MIN_EXP;
    int drop = keep - unit;
    if (drop < 0) drop = 0;
    if (drop > 127) drop = 127;
    unsigned long long truncated = (drop >= 64) ? 0ull : ((a >> drop) << drop);
    double val = sign * (double)truncated * ldexp(1.0, unit);
    return (float)val;
}

// One output cell's a100_dot + fused policy census reduction.
__global__ void fp16_policy_kernel(const __half* __restrict__ A,
                                   const __half* __restrict__ B,
                                   int m, int n, int k,
                                   float* __restrict__ D,
                                   long long* __restrict__ N_bp,
                                   long long* __restrict__ N_runs,
                                   long long* __restrict__ N_pt) {
    int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= m * n) return;
    int i = cell / n;
    int j = cell % n;
    const __half* a = A + (long)i * k;
    const __half* b = B + (long)j * k;

    float cur = 0.0f;
    long long n_bp = 0, n_runs = 0, n_pt = 0;
    bool in_run = false;

    for (int g0 = 0; g0 < k; g0 += POLICY_GROUP) {
        int g1 = min(g0 + POLICY_GROUP, k);
        long long csgn;
        unsigned long long cm;
        int cel, culp;
        acc_parts(cur, &csgn, &cm, &cel, &culp);

        // Alignment exponent over nonzero products and the accumulator.
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
        if (eta == POLICY_NEG) {
            // Empty no-op group: a non-breakpoint step with 0 truncations. It
            // extends the surrounding run and contributes nothing else.
            if (!in_run) { n_runs += 1; in_run = true; }
            continue;
        }
        int unit = eta - POLICY_W;

        long long sum = 0;
        unsigned int products_truncated = 0;
        for (int u = g0; u < g1; ++u) {
            int sa, sb;
            long long ma, mb;
            int ea, eb;
            decompose_fp16(__half_as_ushort(a[u]), &sa, &ma, &ea);
            decompose_fp16(__half_as_ushort(b[u]), &sb, &mb, &eb);
            if (ma == 0 || mb == 0) continue;
            long long prod = ma * mb;               // < 2^22, exact
            int sh = (ea + eb) - 20 - unit;          // product LSB is 2^(ea+eb-20)
            long long aligned = shift_nonneg(prod, sh);
            if (sh < 0) {
                int nbits = (-sh < 62) ? -sh : 62;   // prod < 2^22: 62 covers it
                if ((prod & (((long long)1 << nbits) - 1)) != 0) products_truncated += 1;
            }
            sum += (long long)(sa * sb) * aligned;
        }
        // Accumulator term.
        int csh = culp - unit;
        long long acc_aligned = shift_nonneg((long long)cm, csh);
        bool acc_truncated = false;
        if (csh < 0 && cm != 0ull) {
            int nbits = (-csh < 63) ? -csh : 63;     // cm < 2^24: 63 covers it
            if ((cm & (((unsigned long long)1 << nbits) - 1)) != 0ull) acc_truncated = true;
        }
        sum += csgn * acc_aligned;

        float nw = rz_to_f32(sum, unit);
        // rz dropped a nonzero bit iff the result differs from the exact sum.
        bool rz_dropped = ((double)nw) != ((double)sum) * ldexp(1.0, unit);
        bool breakpoint = acc_truncated || rz_dropped;

        if (breakpoint) {
            n_bp += 1;
            in_run = false;
        } else {
            n_pt += (long long)products_truncated;
            if (!in_run) { n_runs += 1; in_run = true; }
        }
        cur = nw;
    }

    D[cell] = cur;
    N_bp[cell] = n_bp;
    N_runs[cell] = n_runs;
    N_pt[cell] = n_pt;
}

// ---- Host launcher ----
// Returns {D (m,n) f32, n_bp (m,n) i64, n_runs (m,n) i64, n_pt (m,n) i64}.
std::vector<torch::Tensor> fp16_policy_census(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "A and B must be CUDA tensors");
    TORCH_CHECK(A.scalar_type() == torch::kFloat16, "A must be float16");
    TORCH_CHECK(B.scalar_type() == torch::kFloat16, "B must be float16");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), "A and B must be contiguous");
    int m = A.size(0), k = A.size(1);
    int n = B.size(0), kb = B.size(1);
    TORCH_CHECK(k == kb, "A and B must share k");
    TORCH_CHECK(k >= 1, "k must be >= 1");

    auto f32 = torch::dtype(torch::kFloat32).device(A.device());
    auto i64 = torch::dtype(torch::kInt64).device(A.device());
    auto D = torch::empty({m, n}, f32);
    auto N_bp = torch::empty({m, n}, i64);
    auto N_runs = torch::empty({m, n}, i64);
    auto N_pt = torch::empty({m, n}, i64);
    int cells = m * n;
    if (cells == 0) return {D, N_bp, N_runs, N_pt};

    int threads = 128;
    int blocks = (cells + threads - 1) / threads;
    fp16_policy_kernel<<<blocks, threads>>>(
        (const __half*)A.data_ptr(), (const __half*)B.data_ptr(), m, n, k,
        (float*)D.data_ptr(), (long long*)N_bp.data_ptr(),
        (long long*)N_runs.data_ptr(), (long long*)N_pt.data_ptr());
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "fp16_policy_census launch failed");
    return {D, N_bp, N_runs, N_pt};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fp16_policy_census", &fp16_policy_census,
          "A100 sm_80 bit-exact FP16 accumulation policy census (tile + per-cell n_bp/n_runs/n_pt)",
          pybind11::arg("A"), pybind11::arg("B"));
}
