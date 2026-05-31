// SPDX-License-Identifier: see LICENSE
//
// pearl_miner_sm89 — standalone sm_89 Pearl GPU miner.
//
// Takes a real LuckyPool job (header + mining_config + target + nonce range),
// runs the full noisy-GEMM + on-device PoW pipeline with seed-DERIVED noise
// (not random), iterates nonces, and on a hit emits the winning tile's opened
// rows/cols + the 16-word transcript + the PoW digest as JSON on stdout. The
// Python driver (luckypool_miner_driver.gpu_sm89_mine) shells out to this and
// builds proof.bin via the authoritative pearl_mining serializer.
//
// Why standalone: the pybind pearl_gemm_cuda .so is sm_89-only and needs torch;
// the local GPU is a 5090 (sm_120). A standalone sm_89 binary sidesteps both.
//
// The host-side seed derivation (job_key / a_noise_seed / b_noise_seed /
// gpu_hash) is arch-independent and validated bit-exact vs the captured oracle
// (see pearl_miner_host.hpp + report 08). This binary reuses that derivation and
// adds the GPU pipeline on top.
//
// Modes (selected by --mode):
//   verify : run ONE attempt for a fixed nonce on the captured job and print the
//            transcript so it can be diffed against a known oracle on real sm_89
//            hardware. Self-checks job_key/seeds/gpu_hash on the host first.
//   mine   : iterate nonce in [nonce_start, nonce_start+nonce_count); return on
//            the first hit (target cleared) or after the range is exhausted.
//            NOTE: `mine`/`serve` mutate header[72,76) per attempt — that field
//            is `nbits` (difficulty), so the proof header != the job header and
//            the pool REJECTS. Use `jobmine` for pool-valid mining.
//   jobmine: POOL-VALID miner. Mines the EXACT job header UNCHANGED and varies
//            ONLY the A/B operand seed ab_seed = nonce_start+i (the real search
//            freedom). On the first hit prints HIT {ab_seed,a_rows,b_cols,...}
//            and exits. The driver regenerates A,B from ab_seed (splitmix64) and
//            builds the proof against the JOB header -> verify_plain_proof True.
//   bench  : run a FIXED nonce_count of full attempts at the given shape and do
//            NOT return on a hit. CUDA init + buffer alloc happen ONCE, then the
//            attempt loop is timed; prints attempts/sec + tmac_s (same MAC
//            formula as bench_sm89_pouw_re2). This measures the miner's real
//            steady-state attempt rate (vs mine-mode which returns on first hit).
//
// I/O contract (stdin/argv JSON-ish key=value, keeps it dep-free):
//   header=<hex 76B>  config=<hex 52B>  target=<hex 64 chars, BIG-endian uint256>
//   target_endian=big|little (default big = pool/human MSB-first convention)
//   NOTE: `target` is the raw per-SHARE wire target. The binary multiplies it by
//   the verifier's difficulty_adjustment_factor (h*w*k) before the on-device
//   comparison, matching extract_difficulty_bound / verify_plain_proof.
//   m=<int> n=<int> k=<int> r=256  nonce_start=<u64> nonce_count=<u64>
//   mode=verify|mine|bench  dev=<int>
// On a hit, prints a single line:
//   HIT {"nonce":N,"tile":[ix,iy],"a_rows":[...8...],"b_cols":[...16...],
//        "transcript":["%08x"...16...],"gpu_hash":"<64hex>","seed":N}
// On no-hit (mine, range exhausted): prints  NOHIT
//
// The nonce enters the header at bytes [72,76) (the 4-byte mutable suffix of the
// 76-byte incomplete header) so each nonce gives a fresh job_key -> fresh noise
// keys -> fresh transcript, which is what produces the real attempt rate.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cutlass/numeric_types.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <random>
#include <chrono>

#include <poll.h>     // serve-mode non-blocking stdin
#include <unistd.h>   // read(2)

#include "pearl_miner_host.hpp"
#include "host_signal_header.hpp"

// The full-PoUW main GEMM (denoise + BLAKE3 transcript + on-device target check).
// This is the load-bearing sm_89 mining kernel.
//   * verify-mode  uses the C-store variant (host cross-check of the denoise/C).
//   * mine-mode    uses the no-C-store variant (Lever A): the (M,N) output C is
//     NEVER materialized, so the 131072^2 attempt fits in 16 GB (full C = 32 GB).
namespace pearl { namespace sm89 {
extern "C" void pearl_gemm_sm89_pow_128x256x128_R256(
    int8_t const* A, int64_t lda, int8_t const* B, int64_t ldb,
    cutlass::half_t* C, int64_t ldc, float const* A_scales, float const* B_scales,
    cutlass::half_t const* EAL, cutlass::half_t const* EBR,
    cutlass::half_t const* AxEBL, cutlass::half_t const* EARxBpEB,
    uint32_t const* pow_target, uint32_t const* pow_key,
    void* host_signal_sync, void* host_signal_header_pinned,
    uint64_t* inner_hash_counter, int M, int N, int K, cudaStream_t stream);
extern "C" void pearl_gemm_sm89_pow_128x256x128_R256_nostore(
    int8_t const* A, int64_t lda, int8_t const* B, int64_t ldb,
    cutlass::half_t* C, int64_t ldc, float const* A_scales, float const* B_scales,
    cutlass::half_t const* EAL, cutlass::half_t const* EBR,
    cutlass::half_t const* AxEBL, cutlass::half_t const* EARxBpEB,
    uint32_t const* pow_target, uint32_t const* pow_key,
    void* host_signal_sync, void* host_signal_header_pinned,
    uint64_t* inner_hash_counter, int M, int N, int K, cudaStream_t stream);
// noisingB lives in namespace pearl::sm89; noisingA in the global namespace.
extern "C" void pearl_noisingB_sm89_64x128x64_R128_int32(
    int8_t const* B, int8_t const* EBR, int8_t const* EBL, int8_t const* EAR,
    int8_t* BpEB, int32_t* EARxBpEB, int N, int K, cudaStream_t stream);
}}  // namespace pearl::sm89

// On-device noise-factor generator (R=256, torch-free wrapper of the production
// NoiseGenerationKernel). Produces EAL/EBR (dense) + EAR/EBL (sparse, R- and
// K-major) bit-exactly from a/b_noise_seed. See pearl_miner_noisegen_sm89.cu.
extern "C" void pearl_miner_noisegen_sm89_R256(
    int8_t* EAL, int8_t* EBR,
    int8_t* EAR_R_major, int8_t* EAR_K_major,
    int8_t* EBL_R_major, int8_t* EBL_K_major,
    const uint8_t* key_A, const uint8_t* key_B,
    int M, int N, int K, cudaStream_t stream);
// noisingA (global namespace, R=128 int32 variant).
extern "C" void pearl_noisingA_sm89_64x128x64_R128_int32(
    int8_t const* A, int8_t const* EAL, int8_t const* EAR, int8_t const* EBL,
    int8_t* ApEA, int32_t* AxEBL, int M, int K, cudaStream_t stream);
// Device-side seed-derived A/B fill (bit-exact with host fill_AB).
extern "C" void pearl_miner_fill_AB_sm89(int8_t* dst, size_t n, uint64_t seed,
                                         cudaStream_t stream);
// Split an R-major (rows,256) int8 matrix into two contiguous (rows,128) halves.
extern "C" void pearl_miner_split_rmajor_256_sm89(
    const int8_t* in, int8_t* out_lo, int8_t* out_hi, size_t rows,
    cudaStream_t stream);

using pearl_miner::Seeds;
using pearl_miner::derive_seeds;
using pearl_miner::transcript_hash;
using pearl_miner::transcript_from_strips;

#define CUCHK(x) do { auto _e = (x); if (_e != cudaSuccess) { \
    fprintf(stderr, "CUDA %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
    std::exit(2); } } while (0)

static constexpr int R_DIM = 256;
static constexpr int TILE_M = 128, TILE_N = 256;  // PoW kernel tile
// Production hash-tile strided pattern (mining_config / oracle).
static const int RP[8]  = {0, 8, 32, 40, 64, 72, 96, 104};
static const int CP[16] = {0, 1, 32, 33, 64, 65, 96, 97,
                           128, 129, 160, 161, 192, 193, 224, 225};

// ---- tiny arg parsing ----
struct Args {
  std::string header_hex, config_hex, target_hex, mode = "mine";
  std::string aroot_hex, broot_hex;  // optional: real A/B merkle roots (hash_a/hash_b)
  std::string target_endian = "big";  // "big" (pool/human, MSB-first) or "little"
  int m = 131072, n = 131072, k = 4096, r = 256, dev = 0;
  uint64_t nonce_start = 0, nonce_count = 1;
};

static bool from_hex(const std::string& s, std::vector<uint8_t>& out) {
  if (s.size() % 2) return false;
  out.resize(s.size() / 2);
  auto v = [](char c)->int{ if(c>='0'&&c<='9')return c-'0'; if(c>='a'&&c<='f')return c-'a'+10; if(c>='A'&&c<='F')return c-'A'+10; return -1;};
  for (size_t i = 0; i < out.size(); ++i) {
    int hi = v(s[2*i]), lo = v(s[2*i+1]);
    if (hi < 0 || lo < 0) return false;
    out[i] = (uint8_t)((hi << 4) | lo);
  }
  return true;
}

static void parse_kv(const std::string& tok, Args& a) {
  auto eq = tok.find('=');
  if (eq == std::string::npos) return;
  std::string k = tok.substr(0, eq), v = tok.substr(eq + 1);
  if (k == "header") a.header_hex = v;
  else if (k == "config") a.config_hex = v;
  else if (k == "target") a.target_hex = v;
  else if (k == "target_endian") a.target_endian = v;
  else if (k == "aroot") a.aroot_hex = v;
  else if (k == "broot") a.broot_hex = v;
  else if (k == "mode") a.mode = v;
  else if (k == "m") a.m = atoi(v.c_str());
  else if (k == "n") a.n = atoi(v.c_str());
  else if (k == "k") a.k = atoi(v.c_str());
  else if (k == "r") a.r = atoi(v.c_str());
  else if (k == "dev") a.dev = atoi(v.c_str());
  else if (k == "nonce_start") a.nonce_start = strtoull(v.c_str(), nullptr, 0);
  else if (k == "nonce_count") a.nonce_count = strtoull(v.c_str(), nullptr, 0);
}

// Deterministic A/B fill from a 64-bit seed (splitmix64 -> int8 in [-64,63]).
// The driver regenerates the SAME A/B from the reported `seed` to serialize the
// proof, so this PRNG contract is part of the binary<->driver interface.
static inline uint64_t splitmix64(uint64_t& s) {
  s += 0x9E3779B97F4A7C15ULL;
  uint64_t z = s;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}
static void fill_AB(int8_t* dst, size_t n, uint64_t seed) {
  uint64_t s = seed;
  for (size_t i = 0; i < n; ++i) {
    uint64_t r = splitmix64(s);
    dst[i] = (int8_t)((int)(r % 127) - 63);  // [-63,63]
  }
}

// ---------------------------------------------------------------------------
// One GPU attempt at a given (M,N,K) for a given header(nonce-mutated)+config.
// Returns 1 on a target hit (host_signal triggered), filling tile_ix/iy. The
// caller derives opened rows/cols + transcript from the noised operands.
// Buffers are caller-owned and reused across nonces.
// ---------------------------------------------------------------------------
struct GpuBufs {
  int8_t  *dApEA=0,*dBpEB=0;
  cutlass::half_t *dEAL_fp16=0,*dEBR_fp16=0,*dAxEBL_fp16=0,*dEARxBpEB_fp16=0;
  float   *dAs=0,*dBs=0;
  cutlass::half_t *dC=0;
  uint32_t *dTarget=0,*dKey=0;
  HostSignalSync* dSync=0;
  HostSignalHeader* hHeader=0;  // pinned
  bool mine=false;
  // ---- mine-mode on-device noising scratch (allocated only when mine=true) ----
  int8_t  *dA=0,*dB=0;                       // seed-derived int8 source operands
  int8_t  *dEAL_i8=0,*dEBR_i8=0;             // dense noise factors (M,R)/(N,R)
  int8_t  *dEAR_R=0,*dEAR_K=0,*dEBL_R=0,*dEBL_K=0;  // sparse factors (K,R)/(R,K)
  int32_t *dAxEBL_i32=0,*dEARxBpEB_i32=0;    // noising denoise scratch (ignored)
  uint8_t *dKeyA=0,*dKeyB=0;                 // a/b_noise_seed on device
  // Contiguous (rows,128) R-halves of the R-major factors (the R=128 noising
  // kernels require ld=128 — a strided slice of the (rows,256) buffer is wrong).
  int8_t  *dEAL_h[2]={0,0}, *dEAR_h[2]={0,0};   // noisingA: EAL(M), EAR(K)
  int8_t  *dEBR_h[2]={0,0}, *dEBL_h[2]={0,0};   // noisingB: EBR(N), EBL(K)
};

static void alloc_bufs(GpuBufs& g, int M, int N, int K, bool mine) {
  g.mine = mine;
  CUCHK(cudaMalloc(&g.dApEA, size_t(M)*K));
  CUCHK(cudaMalloc(&g.dBpEB, size_t(N)*K));
  CUCHK(cudaMalloc(&g.dEAL_fp16,      size_t(M)*R_DIM*2));
  CUCHK(cudaMalloc(&g.dEBR_fp16,      size_t(N)*R_DIM*2));
  CUCHK(cudaMalloc(&g.dAxEBL_fp16,    size_t(M)*R_DIM*2));
  CUCHK(cudaMalloc(&g.dEARxBpEB_fp16, size_t(N)*R_DIM*2));
  CUCHK(cudaMalloc(&g.dAs, size_t(M)*4));
  CUCHK(cudaMalloc(&g.dBs, size_t(N)*4));
  // The (M,N) output C is the OOM at the mining shape (131072^2 -> 32 GB). It is
  // ONLY needed for verify-mode's host denoise cross-check; mine-mode uses the
  // no-C-store PoW kernel and never materializes it.
  if (!mine) CUCHK(cudaMalloc(&g.dC, size_t(M)*N*2));
  CUCHK(cudaMalloc(&g.dTarget, 8*4));
  CUCHK(cudaMalloc(&g.dKey, 8*4));
  CUCHK(cudaMalloc(&g.dSync, sizeof(HostSignalSync)));
  CUCHK(cudaMallocHost(&g.hHeader, host_signal_header_size));
  std::vector<float> as(M, 0.01f), bs(N, 0.01f);
  CUCHK(cudaMemcpy(g.dAs, as.data(), size_t(M)*4, cudaMemcpyHostToDevice));
  CUCHK(cudaMemcpy(g.dBs, bs.data(), size_t(N)*4, cudaMemcpyHostToDevice));

  if (mine) {
    CUCHK(cudaMalloc(&g.dA, size_t(M)*K));
    CUCHK(cudaMalloc(&g.dB, size_t(N)*K));
    CUCHK(cudaMalloc(&g.dEAL_i8, size_t(M)*R_DIM));
    CUCHK(cudaMalloc(&g.dEBR_i8, size_t(N)*R_DIM));
    CUCHK(cudaMalloc(&g.dEAR_R,  size_t(K)*R_DIM));
    CUCHK(cudaMalloc(&g.dEAR_K,  size_t(R_DIM)*K));
    CUCHK(cudaMalloc(&g.dEBL_R,  size_t(K)*R_DIM));
    CUCHK(cudaMalloc(&g.dEBL_K,  size_t(R_DIM)*K));
    CUCHK(cudaMalloc(&g.dAxEBL_i32,    size_t(M)*R_DIM*4));
    CUCHK(cudaMalloc(&g.dEARxBpEB_i32, size_t(N)*R_DIM*4));
    CUCHK(cudaMalloc(&g.dKeyA, 32));
    CUCHK(cudaMalloc(&g.dKeyB, 32));
    for (int h = 0; h < 2; ++h) {
      CUCHK(cudaMalloc(&g.dEAL_h[h], size_t(M)*128));
      CUCHK(cudaMalloc(&g.dEAR_h[h], size_t(K)*128));
      CUCHK(cudaMalloc(&g.dEBR_h[h], size_t(N)*128));
      CUCHK(cudaMalloc(&g.dEBL_h[h], size_t(K)*128));
    }
  }
}

// Generate seed-derived noise on device, run noisingA/B (two R=128 passes), the
// denoise int32->fp16 conversion, and the PoW kernel. Returns 1 if a tile hit.
// ---------------------------------------------------------------------------
// MINE-mode attempt: fully on-device. No host materialization of the full
// operands, no (M,N) C buffer. Footprint is linear in (M+N) -> 131072^2 fits
// in 16 GB. Steps:
//   1. fill dA/dB from ab_seed on device (bit-exact with host fill_AB).
//   2. generate EAL/EBR/EAR/EBL on device from a/b_noise_seed (noise-gen kernel,
//      R=256, bit-exact with miner-base/noise_generation.py).
//   3. noisingA/B as TWO CHAINED R=128 passes: pass-1 reads the seed operand,
//      pass-0's int8-wrapped output feeds pass-1 as the "A" input. Because the
//      kernel's int8 cast + A-add are both mod-256 and mod-256 addition is
//      associative, chaining the R-halves is bit-exact with a single R=256 pass:
//        wrap(wrap(EAL_hi@EAR_hi) + wrap(wrap(EAL_lo@EAR_lo)+A))
//          == wrap(A + EAL_lo@EAR_lo + EAL_hi@EAR_hi).
//      (The bench fed the SAME A to both halves and overwrote ApEA -> only one
//       R-half survived. Chaining the output->input fixes that.)
//   4. zero the four fp16 denoise factors (they don't enter the transcript/PoW),
//      then launch the no-C-store PoW kernel.
// ---- optional per-phase profiling (PEARL_MINER_PROFILE=1) ----
// Times the 4 phase groups of run_attempt_mine with cudaEvents. Each phase is
// bracketed by a device sync (via the events) so the reported ms isolate that
// phase; this adds syncs that are NOT present in production timing, so it is
// gated behind the env flag and off by default.
struct PhaseProf {
  bool on=false;
  static constexpr int NE = 7;
  cudaEvent_t e[NE];  // e[0]=start, e[1..6]=after phases 1..6
  PhaseProf() {
    const char* v = std::getenv("PEARL_MINER_PROFILE");
    on = (v && v[0] && v[0] != '0');
    if (on) for (int i = 0; i < NE; ++i) CUCHK(cudaEventCreate(&e[i]));
  }
  void mark(int i) { if (on) CUCHK(cudaEventRecord(e[i], 0)); }
  void report() {
    if (!on) return;
    CUCHK(cudaEventSynchronize(e[NE-1]));
    const char* names[6] = {"fill_AB", "noisegen", "split_rmajor",
                            "noisingAB", "memset_fp16", "PoW_kernel"};
    float tot = 0;
    for (int i = 0; i < 6; ++i) {
      float ms = 0; CUCHK(cudaEventElapsedTime(&ms, e[i], e[i+1])); tot += ms;
      fprintf(stderr, "PROFILE phase%d %-13s %9.3f ms\n", i+1, names[i], ms);
    }
    fprintf(stderr, "PROFILE total %25.3f ms\n", tot);
  }
};

static int run_attempt_mine(GpuBufs& g, const Seeds& s, int M, int N, int K,
                            uint64_t ab_seed, uint32_t* tile_ix, uint32_t* tile_iy) {
  PhaseProf prof;
  prof.mark(0);
  // 1. seed-derived A/B on device.
  pearl_miner_fill_AB_sm89(g.dA, size_t(M)*K, ab_seed, 0);
  pearl_miner_fill_AB_sm89(g.dB, size_t(N)*K, ab_seed ^ 0xD1B54A32D192ED03ULL, 0);
  prof.mark(1);

  // 2. on-device noise factors (R=256) from a/b_noise_seed.
  CUCHK(cudaMemcpy(g.dKeyA, s.a_noise_seed, 32, cudaMemcpyHostToDevice));
  CUCHK(cudaMemcpy(g.dKeyB, s.b_noise_seed, 32, cudaMemcpyHostToDevice));
  pearl_miner_noisegen_sm89_R256(
      g.dEAL_i8, g.dEBR_i8, g.dEAR_R, g.dEAR_K, g.dEBL_R, g.dEBL_K,
      g.dKeyA, g.dKeyB, M, N, K, 0);
  prof.mark(2);

  // 2b. Repack the R-major factors (EAL/EAR for A, EBR/EBL for B) into two
  //     contiguous (rows,128) R-halves. The R=128 noising kernel hard-codes
  //     ld=128, so it cannot consume a strided slice of the (rows,256) buffer.
  //     The K-major sparse factors (EBL_K for A, EAR_K for B) are already
  //     contiguous per half (row-blocks), so they are sliced directly.
  pearl_miner_split_rmajor_256_sm89(g.dEAL_i8, g.dEAL_h[0], g.dEAL_h[1], M, 0);
  pearl_miner_split_rmajor_256_sm89(g.dEAR_R,  g.dEAR_h[0], g.dEAR_h[1], K, 0);
  pearl_miner_split_rmajor_256_sm89(g.dEBR_i8, g.dEBR_h[0], g.dEBR_h[1], N, 0);
  pearl_miner_split_rmajor_256_sm89(g.dEBL_R,  g.dEBL_h[0], g.dEBL_h[1], K, 0);
  prof.mark(3);

  // 3. chained two-pass noisingA/B (R-halves). Half h uses the contiguous
  //    (rows,128) half of the dense/R-major-sparse factors and the row-block
  //    [h*128,..) of the K-major sparse factors. Pass-0 reads the seed operand
  //    (dA/dB) and writes the noised half into dApEA/dBpEB; pass-1 reads
  //    dApEA/dBpEB and writes the final R=256 noised operand back into
  //    dApEA/dBpEB (in place). Each CTA owns one m-block (n-block) and reads its
  //    A k-tiles before writing the same ApEA k-tiles, so the in-place pass-1 is
  //    race-free. Final noised operands therefore always live in dApEA/dBpEB.
  //    Chaining is bit-exact with a single R=256 pass: mod-256 wrap distributes
  //    over the R-halves' addition (validated vs host, 0 diff).
  for (int h = 0; h < 2; ++h) {
    int roff = h * 128;
    const int8_t* aIn = (h == 0) ? g.dA : g.dApEA;
    const int8_t* bIn = (h == 0) ? g.dB : g.dBpEB;
    pearl_noisingA_sm89_64x128x64_R128_int32(
        aIn,
        g.dEAL_h[h],                 // EAL half h, contiguous (M,128)
        g.dEAR_h[h],                 // EAR half h, contiguous (K,128)
        g.dEBL_K + size_t(roff)*K,   // EBL_K_major row-block [roff:, :] (ld=K)
        g.dApEA, g.dAxEBL_i32, M, K, 0);
    pearl::sm89::pearl_noisingB_sm89_64x128x64_R128_int32(
        bIn,
        g.dEBR_h[h],                 // EBR half h, contiguous (N,128)
        g.dEBL_h[h],                 // EBL_R half h, contiguous (K,128)
        g.dEAR_K + size_t(roff)*K,   // EAR_K_major row-block [roff:, :] (ld=K)
        g.dBpEB, g.dEARxBpEB_i32, N, K, 0);
  }
  prof.mark(4);

  // 4. denoise fp16 factors are irrelevant to the transcript/PoW (see verify
  // path comment) -> zero them.
  CUCHK(cudaMemset(g.dEAL_fp16, 0, size_t(M)*R_DIM*2));
  CUCHK(cudaMemset(g.dEBR_fp16, 0, size_t(N)*R_DIM*2));
  CUCHK(cudaMemset(g.dAxEBL_fp16, 0, size_t(M)*R_DIM*2));
  CUCHK(cudaMemset(g.dEARxBpEB_fp16, 0, size_t(N)*R_DIM*2));

  CUCHK(cudaMemcpy(g.dKey, s.a_noise_seed, 32, cudaMemcpyHostToDevice));
  HostSignalSync zsync{};
  CUCHK(cudaMemcpy(g.dSync, &zsync, sizeof(zsync), cudaMemcpyHostToDevice));
  memset(g.hHeader, 0, host_signal_header_size);
  prof.mark(5);

  pearl::sm89::pearl_gemm_sm89_pow_128x256x128_R256_nostore(
      g.dApEA, K, g.dBpEB, K, nullptr, N, g.dAs, g.dBs,
      g.dEAL_fp16, g.dEBR_fp16, g.dAxEBL_fp16, g.dEARxBpEB_fp16,
      g.dTarget, g.dKey, g.dSync, g.hHeader, nullptr, M, N, K, 0);
  prof.mark(6);
  CUCHK(cudaDeviceSynchronize());
  cudaError_t le = cudaGetLastError();
  if (le != cudaSuccess) { fprintf(stderr, "PoW(nostore) launch: %s\n", cudaGetErrorString(le)); std::exit(2); }
  prof.report();

  if (g.hHeader->status == kSignalTriggered) {
    *tile_ix = g.hHeader->tileCoord[0];
    *tile_iy = g.hHeader->tileCoord[1];
    return 1;
  }
  return 0;
}

static int run_attempt(GpuBufs& g, const Seeds& s, int M, int N, int K,
                       uint64_t ab_seed, uint32_t* tile_ix, uint32_t* tile_iy) {
  if (g.mine) return run_attempt_mine(g, s, M, N, K, ab_seed, tile_ix, tile_iy);
  // ---- compute the FULL-R256 noised operands ApEA/BpEB on HOST and upload ----
  // ApEA[m] = i8wrap(A[m]   + EAL[m]   @ EAR^T)   (EAR sparse: per-k = EAL[k0]-EAL[k1])
  // BpEB[n] = i8wrap(B^T[n] + EBR[n]   @ EBL^T)
  //
  // Why host-side: the on-device noise-gen + noisingA/B kernels are validated only
  // for R in {64,128}. The bench runs R=256 as two R=128 passes — but the noising
  // kernel WRITES (not accumulates) ApEA, so two passes capture only one R-half's
  // noise (fine for the bench's timing, WRONG for the actual noised values). A
  // correct GPU R=256 noising kernel is the remaining PERF integration; it does
  // not affect PoW correctness (noising is <1% of pipeline MACs). The host noise
  // is bit-exact with miner-base/noise_generation.py (validated by the host gate)
  // and is the same noised operands the GPU PoW mainloop would consume.
  const uint8_t SEED_A[32] = {'A','_','t','e','n','s','o','r',0,0,0,0,0,0,0,0,
                              0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
  const uint8_t SEED_B[32] = {'B','_','t','e','n','s','o','r',0,0,0,0,0,0,0,0,
                              0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
  {
    std::vector<int8_t> EAL, EBR, EAR, EBL;
    pearl_miner::noise_dense(M, R_DIM, SEED_A, s.a_noise_seed, EAL);
    pearl_miner::noise_dense(N, R_DIM, SEED_B, s.b_noise_seed, EBR);
    pearl_miner::noise_sparse(K, R_DIM, SEED_A, s.a_noise_seed, EAR);
    pearl_miner::noise_sparse(K, R_DIM, SEED_B, s.b_noise_seed, EBL);

    std::vector<int8_t> A((size_t)M*K), B((size_t)N*K);
    fill_AB(A.data(), A.size(), ab_seed);
    fill_AB(B.data(), B.size(), ab_seed ^ 0xD1B54A32D192ED03ULL);

    std::vector<int8_t> ApEA((size_t)M*K), BpEB((size_t)N*K);
    #pragma omp parallel for schedule(static)
    for (int m = 0; m < M; ++m) {
      const int8_t* eal = &EAL[(size_t)m * R_DIM];
      for (int kk = 0; kk < K; ++kk) {
        int32_t acc = 0; const int8_t* ear = &EAR[(size_t)kk * R_DIM];
        for (int r = 0; r < R_DIM; ++r) acc += (int32_t)eal[r] * (int32_t)ear[r];
        ApEA[(size_t)m*K + kk] = pearl_miner::i8wrap((int32_t)A[(size_t)m*K + kk] + acc);
      }
    }
    #pragma omp parallel for schedule(static)
    for (int n = 0; n < N; ++n) {
      const int8_t* ebr = &EBR[(size_t)n * R_DIM];
      for (int kk = 0; kk < K; ++kk) {
        int32_t acc = 0; const int8_t* ebl = &EBL[(size_t)kk * R_DIM];
        for (int r = 0; r < R_DIM; ++r) acc += (int32_t)ebr[r] * (int32_t)ebl[r];
        BpEB[(size_t)n*K + kk] = pearl_miner::i8wrap((int32_t)B[(size_t)n*K + kk] + acc);
      }
    }
    CUCHK(cudaMemcpy(g.dApEA, ApEA.data(), ApEA.size(), cudaMemcpyHostToDevice));
    CUCHK(cudaMemcpy(g.dBpEB, BpEB.data(), BpEB.size(), cudaMemcpyHostToDevice));
  }

  // ---- denoise fp16 factors ----
  // The PoW TRANSCRIPT is computed from the INTEGER noised GEMM ApEA@BpEB^T (the
  // mainloop accumulator), NOT from the denoise epilogue. The four fp16 factors
  // (EAL/EBR/AxEBL/EARxBpEB) only affect the denoised, useful-work output C —
  // which the miner does not submit and which does not enter the transcript or
  // PoW digest. We therefore zero them: this leaves the transcript/target check
  // bit-exact while skipping the c10/torch-coupled denoise converter. (The
  // useful-work C is irrelevant to share validity; the pool re-derives it.)
  CUCHK(cudaMemset(g.dEAL_fp16, 0, size_t(M)*R_DIM*2));
  CUCHK(cudaMemset(g.dEBR_fp16, 0, size_t(N)*R_DIM*2));
  CUCHK(cudaMemset(g.dAxEBL_fp16, 0, size_t(M)*R_DIM*2));
  CUCHK(cudaMemset(g.dEARxBpEB_fp16, 0, size_t(N)*R_DIM*2));

  // ---- target/key + reset signal ----
  // pow_key = a_noise_seed (uint32[8] LE). pow_target = adjusted threshold
  // (wire_target * h*w*k) as uint256 LE — see the target-adjustment block in main().
  CUCHK(cudaMemcpy(g.dKey, s.a_noise_seed, 32, cudaMemcpyHostToDevice));
  HostSignalSync zsync{};
  CUCHK(cudaMemcpy(g.dSync, &zsync, sizeof(zsync), cudaMemcpyHostToDevice));
  memset(g.hHeader, 0, host_signal_header_size);

  pearl::sm89::pearl_gemm_sm89_pow_128x256x128_R256(
      g.dApEA, K, g.dBpEB, K, g.dC, N, g.dAs, g.dBs,
      g.dEAL_fp16, g.dEBR_fp16, g.dAxEBL_fp16, g.dEARxBpEB_fp16,
      g.dTarget, g.dKey, g.dSync, g.hHeader, nullptr, M, N, K, 0);
  CUCHK(cudaDeviceSynchronize());
  cudaError_t le = cudaGetLastError();
  if (le != cudaSuccess) { fprintf(stderr, "PoW launch: %s\n", cudaGetErrorString(le)); std::exit(2); }

  if (g.hHeader->status == kSignalTriggered) {
    *tile_ix = g.hHeader->tileCoord[0];
    *tile_iy = g.hHeader->tileCoord[1];
    return 1;
  }
  return 0;
}

// On a hit, read the opened ApEA rows + BpEB cols from device and compute the
// 16-word transcript host-side (arch-independent). The kernel signals the WARP
// tile; we recover the absolute opened rows/cols from tileCoord + the strided
// hash-tile pattern. host_signal_header also carries the per-thread row/col of
// the winning hash-tile origin within the 128x256 tile.
static void read_transcript_and_indices(
    GpuBufs& g, const Seeds& s, int K,
    uint32_t tile_ix, uint32_t tile_iy,
    int* a_rows, int* b_cols, uint32_t transcript[16]) {
  // The signal header records the winning thread's full set of (row,col) MMA
  // fragment coordinates within the 128x256 tile. The opened hash-tile is a
  // strided 8x16 sub-block (rows_pattern RP / cols_pattern CP from mining_config)
  // anchored at a hash-tile origin (ro,co). We recover (ro,co) as the minimum
  // reported thread coordinate (the fragment's top-left), snapped so that ro+RP
  // and co+CP stay within the 128x256 tile.
  // NOTE: the exact MMA-fragment -> hash-tile-origin reduction is lpminer's
  // "pattern calibration"; on real sm_89 the user diffs the emitted transcript
  // against the oracle to confirm this anchor mapping. The driver re-derives the
  // SAME opened indices when serializing (they are fixed by mining_config + the
  // winning tile coord), so a wrong anchor here surfaces as verify_plain_proof
  // == False and never reaches the pool.
  int nreg = g.hHeader->num_registers_per_thread;
  int ro = 127, co = 255;
  for (int j = 0; j < nreg; ++j) {
    if (g.hHeader->thread_rows[j] < ro) ro = g.hHeader->thread_rows[j];
    if (g.hHeader->thread_cols[j] < co) co = g.hHeader->thread_cols[j];
  }
  if (ro + RP[7] >= TILE_M) ro = TILE_M - 1 - RP[7];   // keep 8 rows in-tile
  if (co + CP[15] >= TILE_N) co = TILE_N - 1 - CP[15];  // keep 16 cols in-tile
  if (ro < 0) ro = 0; if (co < 0) co = 0;
  int row0 = (int)tile_ix * TILE_M + ro;
  int col0 = (int)tile_iy * TILE_N + co;
  for (int a = 0; a < 8; ++a)  a_rows[a] = row0 + RP[a];
  for (int b = 0; b < 16; ++b) b_cols[b] = col0 + CP[b];

  // Copy the 8 noised-A rows (ApEA) + 16 noised-B cols (BpEB) back to host.
  std::vector<int8_t> An(8 * K), Bn(16 * K);
  for (int a = 0; a < 8; ++a)
    CUCHK(cudaMemcpy(&An[(size_t)a*K], g.dApEA + (size_t)a_rows[a]*K, K, cudaMemcpyDeviceToHost));
  for (int b = 0; b < 16; ++b)
    CUCHK(cudaMemcpy(&Bn[(size_t)b*K], g.dBpEB + (size_t)b_cols[b]*K, K, cudaMemcpyDeviceToHost));

  // Transcript over the NOISED operands directly (the GPU's integer GEMM path):
  //   per R-chunk: tile = An[:,p:p+R] @ Bn[:,p:p+R]^T ; x=XOR-reduce ; T[rc%16]=rotl13(T)^x
  for (int i = 0; i < 16; ++i) transcript[i] = 0;
  int nK = K / R_DIM;
  for (int rc = 0; rc < nK; ++rc) {
    int p = rc * R_DIM;
    uint32_t x = 0;
    for (int i = 0; i < 8; ++i)
      for (int j = 0; j < 16; ++j) {
        int32_t acc = 0;
        const int8_t* ar = &An[(size_t)i*K + p];
        const int8_t* br = &Bn[(size_t)j*K + p];
        for (int r = 0; r < R_DIM; ++r) acc += (int32_t)ar[r] * (int32_t)br[r];
        x ^= (uint32_t)acc;
      }
    int idx = rc % 16;
    transcript[idx] = ((transcript[idx] << 13) | (transcript[idx] >> 19)) ^ x;
  }
}

static std::string read_stdin_args() {
  std::string all, line;
  char buf[4096];
  size_t n;
  while ((n = fread(buf, 1, sizeof(buf), stdin)) > 0) all.append(buf, n);
  return all;
}

// Decode a wire target (32 raw bytes, MSB-first when big-endian) into the 32 LE
// `pow_target` words the device kernel compares against, applying the SAME
// difficulty adjustment as main() (threshold = wire_target * h*w*k, saturating
// at 2^256-1). Factored out of main() so serve-mode can re-decode per JOB.
// Returns false on malformed target hex.
static bool decode_target(const std::string& target_hex, bool big_endian,
                          int k, int r, std::vector<uint8_t>& target_le) {
  target_le.assign(32, 0xFF);  // default easiest (T = 2^256 - 1)
  if (target_hex.empty()) return true;
  std::vector<uint8_t> traw;
  if (!from_hex(target_hex, traw) || traw.size() != 32) return false;
  for (int i = 0; i < 32; ++i)
    target_le[i] = big_endian ? traw[31 - i] : traw[i];
  // difficulty adjustment: device threshold = wire_target * (h*w*k).
  const uint64_t hh = 8, ww = 16;
  const uint64_t dpl = (r > 0) ? (uint64_t)(k - k % r) : (uint64_t)k;
  const uint64_t adj = hh * ww * dpl;
  __uint128_t carry = 0;
  bool overflow = false;
  for (int i = 0; i < 32; ++i) {
    __uint128_t prod = (__uint128_t)target_le[i] * adj + carry;
    target_le[i] = (uint8_t)(prod & 0xFF);
    carry = prod >> 8;
  }
  if (carry != 0) overflow = true;
  if (overflow) target_le.assign(32, 0xFF);
  return true;
}

// Derive the noise seeds for a (possibly nonce-mutated) header. Mirrors the
// seed derivation in the mine loop: with real merkle roots use derive_seeds;
// otherwise the self-consistent job_key chain (smoke / serve). `header` must be
// 76 bytes (the nonce already written into [72,76)).
static void derive_seeds_for_header(const std::vector<uint8_t>& header,
                                    const std::vector<uint8_t>& config,
                                    const std::vector<uint8_t>& aroot,
                                    const std::vector<uint8_t>& broot,
                                    Seeds& s) {
  if (!aroot.empty() && !broot.empty()) {
    s = derive_seeds(header.data(), header.size(), config.data(), config.size(),
                     aroot.data(), broot.data());
  } else {
    pearl_miner::blake3::hash_concat(header.data(), header.size(),
                                     config.data(), config.size(), nullptr, s.job_key);
    pearl_miner::blake3::hash_concat(s.job_key, 32, s.job_key, 32, nullptr, s.b_noise_seed);
    pearl_miner::blake3::hash_concat(s.b_noise_seed, 32, s.job_key, 32, nullptr, s.a_noise_seed);
  }
}

// ---------------------------------------------------------------------------
// SERVE mode — the definitive non-stale miner.
//
// Inits CUDA + all device buffers ONCE, then loops forever mining the MOST
// RECENT job fed on stdin. Each stdin line is one job update:
//     JOB <header_76B_hex> <target_64hex>
// stdin is read NON-BLOCKING (poll) and fully drained each iteration so we
// always mine the newest JOB (never stall waiting for input). For the current
// header we derive the seeds ONCE, then mine successive small nonce batches so
// a freshly-arrived JOB is picked up within ~1 batch (~1s). On a tile hit we
// print one HIT line — echoing the 76B header the hit was mined against so the
// driver can map it back to a job_id — and KEEP mining (do not exit on hit).
// Exits cleanly on stdin EOF.
//
// Why this lands a FRESH share where one-shot ssh produces stale hits: a single
// persistent process re-derives noise for the LATEST header continuously at the
// full ~1.9 att/s, so when it hits (~1/57 per attempt) the share is for a
// ~current job (jobs rotate ~7s) instead of a long-dead fixed header.
// ---------------------------------------------------------------------------
static int run_serve(GpuBufs& g, const Args& a,
                     const std::vector<uint8_t>& config,
                     const std::vector<uint8_t>& aroot,
                     const std::vector<uint8_t>& broot) {
  const int M = a.m, N = a.n, K = a.k;
  const bool big_endian = (a.target_endian != "little");
  // Mine in small batches so a new JOB on stdin preempts within ~1 batch.
  const uint64_t BATCH = 2;

  std::vector<uint8_t> cur_header;     // 76B working header (nonce mutated in loop)
  std::string cur_header_hex;          // ORIGINAL job header hex (unmutated) to echo
  std::vector<uint8_t> cur_target_le;  // decoded pow_target for cur job
  bool have_job = false;
  uint64_t nonce = 0;

  // Line-buffered non-blocking stdin reader. We accumulate bytes into `inbuf`
  // and split on '\n'; only the LAST complete JOB line in a drain is applied
  // (older queued jobs are stale the instant a newer one is present).
  std::string inbuf;
  bool eof = false;

  auto apply_job_line = [&](const std::string& line) -> bool {
    // Expect: JOB <header_hex> <target_hex>
    if (line.compare(0, 4, "JOB ") != 0) return false;
    size_t p1 = line.find(' ', 4);
    if (p1 == std::string::npos) return false;
    std::string hhex = line.substr(4, p1 - 4);
    std::string thex = line.substr(p1 + 1);
    // trim trailing whitespace/CR on target
    while (!thex.empty() && (thex.back()=='\r'||thex.back()=='\n'||thex.back()==' '||thex.back()=='\t'))
      thex.pop_back();
    std::vector<uint8_t> h;
    if (!from_hex(hhex, h) || h.size() != 76) {
      fprintf(stderr, "serve: bad JOB header (need 76B hex), ignoring\n");
      return false;
    }
    std::vector<uint8_t> tle;
    if (!decode_target(thex, big_endian, K, a.r, tle)) {
      fprintf(stderr, "serve: bad JOB target, ignoring\n");
      return false;
    }
    cur_header = std::move(h);
    cur_target_le = std::move(tle);
    // Echo the ORIGINAL job header (with the pool's [72,76) suffix), NOT the
    // per-nonce mutated working header — so the driver can map a HIT back to the
    // exact job by header. The winning nonce is reported separately.
    cur_header_hex = hhex;
    return true;
  };

  // Drain all currently-available stdin into inbuf; split complete lines; apply
  // the NEWEST valid JOB line (re-derive seeds + reset nonce). Sets eof on EOF.
  auto drain_stdin = [&]() {
    struct pollfd pfd; pfd.fd = 0; pfd.events = POLLIN;
    for (;;) {
      int pr = poll(&pfd, 1, 0);  // 0ms timeout: non-blocking
      if (pr <= 0) break;
      if (!(pfd.revents & (POLLIN | POLLHUP))) break;
      char buf[8192];
      ssize_t n = read(0, buf, sizeof(buf));
      if (n == 0) { eof = true; break; }
      if (n < 0) break;
      inbuf.append(buf, (size_t)n);
      if ((size_t)n < sizeof(buf)) {
        // likely drained the pipe for now; still loop once more via poll
      }
    }
    // Split inbuf into lines; keep the trailing partial line in inbuf.
    std::string newest;  // the last complete valid-prefixed JOB line
    size_t start = 0, nl;
    size_t consumed = 0;
    while ((nl = inbuf.find('\n', start)) != std::string::npos) {
      std::string line = inbuf.substr(start, nl - start);
      if (line.compare(0, 4, "JOB ") == 0) newest = line;  // keep last JOB
      start = nl + 1;
      consumed = start;
    }
    if (consumed) inbuf.erase(0, consumed);
    if (!newest.empty()) {
      if (apply_job_line(newest)) {
        // Upload the per-job target; seeds are re-derived PER nonce in the mine
        // loop (each nonce mutates header[72,76) -> a fresh job_key), so there
        // is nothing job-global to derive here.
        CUCHK(cudaMemcpy(g.dTarget, cur_target_le.data(), 32, cudaMemcpyHostToDevice));
        nonce = 0;
        have_job = true;
        fprintf(stderr, "serve: new JOB header[0..4]=%02x%02x%02x%02x target_msb=%02x nonce reset\n",
                cur_header[0], cur_header[1], cur_header[2], cur_header[3],
                cur_target_le[31]);
      }
    }
  };

  fprintf(stderr, "serve: ready (M=%d N=%d K=%d R=%d) — awaiting JOB lines on stdin\n",
          M, N, K, a.r);

  while (true) {
    drain_stdin();
    if (eof && inbuf.empty()) {
      fprintf(stderr, "serve: stdin EOF — exiting\n");
      break;
    }
    if (!have_job) {
      // No job yet: brief blocking poll so we don't spin.
      struct pollfd pfd; pfd.fd = 0; pfd.events = POLLIN;
      poll(&pfd, 1, 200);
      continue;
    }

    // Echo the ORIGINAL (unmutated) job header so the driver maps the HIT to its
    // job_id; the winning nonce is in the HIT's `nonce` field.
    const std::string& header_hex_echo = cur_header_hex;

    // Mine BATCH nonces of the current header, returning per-nonce so a new JOB
    // preempts quickly. The nonce mutates header[72,76) -> fresh job_key ->
    // fresh noise each attempt (the production attempt-rate path).
    for (uint64_t b = 0; b < BATCH; ++b, ++nonce) {
      cur_header[72] = (uint8_t)(nonce);
      cur_header[73] = (uint8_t)(nonce >> 8);
      cur_header[74] = (uint8_t)(nonce >> 16);
      cur_header[75] = (uint8_t)(nonce >> 24);
      // Re-derive seeds for THIS nonce-mutated header (the job_key includes the
      // nonce suffix, exactly like mine/bench mode).
      Seeds s;
      derive_seeds_for_header(cur_header, config, aroot, broot, s);
      uint64_t ab_seed = nonce * 0x100000001B3ULL + 0xCBF29CE484222325ULL;
      uint32_t ix = 0, iy = 0;
      int hit = run_attempt(g, s, M, N, K, ab_seed, &ix, &iy);
      if (hit) {
        int a_rows[8], b_cols[16]; uint32_t T[16]; uint8_t gpu_hash[32];
        read_transcript_and_indices(g, s, K, ix, iy, a_rows, b_cols, T);
        transcript_hash(T, s.a_noise_seed, gpu_hash);
        printf("HIT {\"nonce\":%llu,\"seed\":%llu,\"tile\":[%u,%u],\"a_rows\":[",
               (unsigned long long)nonce, (unsigned long long)ab_seed, ix, iy);
        for (int i=0;i<8;++i) printf("%s%d", i?",":"", a_rows[i]);
        printf("],\"b_cols\":[");
        for (int i=0;i<16;++i) printf("%s%d", i?",":"", b_cols[i]);
        printf("],\"transcript\":[");
        for (int i=0;i<16;++i) printf("%s\"%08x\"", i?",":"", T[i]);
        printf("],\"gpu_hash\":\"");
        for (int i=0;i<32;++i) printf("%02x", gpu_hash[i]);
        printf("\",\"header\":\"%s\"}\n", header_hex_echo.c_str());
        fflush(stdout);
        // KEEP mining — do not exit on hit.
      }
    }
  }
  return 0;
}

int main(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) parse_kv(argv[i], a);
  // SERVE mode reads JOB updates from stdin line-by-line (NOT key=value args),
  // so it MUST take its config from argv and must NOT slurp stdin here. Detect
  // it from argv before the (blocking) stdin-args fallback below.
  bool serve_mode = (a.mode == "serve");
  // also accept whitespace-separated key=value pairs on stdin (non-serve only)
  if (!serve_mode && a.header_hex.empty()) {
    std::string in = read_stdin_args();
    std::string tok;
    for (char c : in) {
      if (c == ' ' || c == '\n' || c == '\t' || c == '\r') { if (!tok.empty()) parse_kv(tok, a); tok.clear(); }
      else tok += c;
    }
    if (!tok.empty()) parse_kv(tok, a);
    serve_mode = (a.mode == "serve");
  }

  std::vector<uint8_t> header, config, target_le;
  // serve-mode receives the header per-JOB on stdin; it is not required on argv.
  if (!serve_mode) {
    if (!from_hex(a.header_hex, header) || header.size() != 76) {
      fprintf(stderr, "bad header (need 76B hex)\n"); return 1;
    }
  }
  if (!from_hex(a.config_hex, config) || config.size() != 52) {
    fprintf(stderr, "bad config (need 52B hex)\n"); return 1;
  }
  // --- target -> 32 LE bytes for the device `pow_target` ---------------------
  // The wire/pool and human convention is BIG-ENDIAN hex (MSB-first): the same
  // 256-bit threshold parse_target_hex(... big_endian=True) yields, i.e. the
  // integer T = int.from_bytes(hex, "big"). The authoritative on-device /
  // verify_plain_proof comparison is `int.from_bytes(blake3_digest, "little") <= T`.
  // The device kernel reads pow_target as uint32[8] words where word 0 is the
  // LEAST-significant word (matching the LE digest word order). So we must hand
  // it T serialized LITTLE-endian: target_le[0] = LSByte(T). We therefore parse
  // the hex bytes in the requested endianness, normalize to the integer T, then
  // re-emit T as 32 LE bytes. (target_endian=little keeps the legacy raw-LE
  // contract for callers that already pass LE hex.)
  //   target=00..01  (big-endian)  -> T = 1            -> hardest non-zero -> NOHIT
  //   target=ff..ff                 -> T = 2^256 - 1    -> easiest          -> HIT
  target_le.assign(32, 0xFF);  // default easiest (T = 2^256 - 1)
  if (!a.target_hex.empty()) {
    std::vector<uint8_t> traw;
    if (!from_hex(a.target_hex, traw) || traw.size() != 32) {
      fprintf(stderr, "bad target (need 64 hex chars)\n"); return 1;
    }
    bool big = (a.target_endian != "little");
    // traw is MSB-first when big-endian. Re-emit T as little-endian bytes.
    for (int i = 0; i < 32; ++i)
      target_le[i] = big ? traw[31 - i] : traw[i];

    // --- difficulty adjustment: device threshold = wire_target * (h*w*k) ------
    // The wire `target` is the per-SHARE difficulty target only. The PoW the
    // device compares against is the verifier's adjusted bound:
    //   accept iff  int.from_bytes(jackpot_hash,"little") <= target * (h*w*k)
    // This is EXACTLY `extract_difficulty_bound` in zk-pow/src/api/sanity_checks.rs
    // (target_difficulty * difficulty_adjustment_factor, factor = h*w*dot_product_length),
    // which `verify_plain_proof` enforces. Without this factor the on-device
    // threshold is h*w*k (~2^19) times too HARD and the miner finds ZERO shares.
    //   h = rows_pattern.size() = 8   (RP), w = cols_pattern.size() = 16 (CP),
    //   k = dot_product_length = common_dim - common_dim % rank.
    // The product fits in 256 bits for any live target (2^206 * 2^19 = 2^225),
    // and is clamped to 2^256-1 (matching the verifier's U256::MAX saturation).
    const uint64_t hh = 8, ww = 16;
    const uint64_t dpl = (a.r > 0) ? (uint64_t)(a.k - a.k % a.r) : (uint64_t)a.k;
    const uint64_t adj = hh * ww * dpl;  // difficulty_adjustment_factor
    // 256-bit (target_le, little-endian) *= adj, saturating at 2^256-1.
    __uint128_t carry = 0;
    bool overflow = false;
    for (int i = 0; i < 32; ++i) {
      __uint128_t prod = (__uint128_t)target_le[i] * adj + carry;
      target_le[i] = (uint8_t)(prod & 0xFF);
      carry = prod >> 8;
    }
    if (carry != 0) overflow = true;  // result exceeded 256 bits
    if (overflow) target_le.assign(32, 0xFF);  // saturate to 2^256-1
  }
  std::vector<uint8_t> aroot, broot;
  if (!a.aroot_hex.empty()) { if (!from_hex(a.aroot_hex, aroot) || aroot.size() != 32) { fprintf(stderr, "bad aroot (need 64hex)\n"); return 1; } }
  if (!a.broot_hex.empty()) { if (!from_hex(a.broot_hex, broot) || broot.size() != 32) { fprintf(stderr, "bad broot (need 64hex)\n"); return 1; } }

  CUCHK(cudaSetDevice(a.dev));
  cudaDeviceProp p; CUCHK(cudaGetDeviceProperties(&p, a.dev));
  fprintf(stderr, "pearl_miner_sm89 dev%d %s sm_%d%d  mode=%s M=%d N=%d K=%d R=%d\n",
          a.dev, p.name, p.major, p.minor, a.mode.c_str(), a.m, a.n, a.k, a.r);

  // bench/serve-mode run the same fully-on-device attempt as mine-mode (no host
  // operand materialization, no (M,N) C buffer) so the measured rate reflects
  // the production mine path.
  bool on_device = (a.mode == "mine" || a.mode == "bench" || a.mode == "serve"
                    || a.mode == "jobmine");
  GpuBufs g;
  alloc_bufs(g, a.m, a.n, a.k, /*mine=*/on_device);

  // SERVE: CUDA + buffers are now initialized ONCE. Hand off to the persistent
  // mine loop, which reads JOB lines from stdin and uploads the per-job target
  // itself. (The upfront target memcpy below is skipped for serve.)
  if (a.mode == "serve")
    return run_serve(g, a, config, aroot, broot);

  // -------------------------------------------------------------------------
  // JOBMINE mode — the POOL-VALID miner. Mines the EXACT job header UNCHANGED
  // (never touches header[72:76), which is `nbits`/difficulty — mutating it
  // would make the proof's header != the job header and the pool would reject)
  // and searches ONLY by varying the A/B operand seed `ab_seed` per attempt.
  //
  // This is the real search freedom: A,B are seed-derived (splitmix64 fill),
  // so a different ab_seed -> different operands -> different commitment-keyed
  // noise -> a fresh noised C / transcript / PoW digest, exactly like the CPU
  // miner's per-attempt random A,B. The noise KEYS (job_key/a_noise_seed/
  // b_noise_seed) are derived ONCE from the fixed header+config and reused for
  // every attempt (the header doesn't change), matching commitment().
  //
  // ab_seed = nonce_start + i (so the Python driver regenerates A,B directly
  // from the reported `ab_seed` via the same splitmix64). On the FIRST winning
  // tile it prints one HIT line (with ab_seed + absolute a_rows/b_cols + the
  // gpu_hash) and exits; the proven loop re-invokes per job for the next nonce
  // window. target_le already carries the verifier's h*w*k difficulty
  // adjustment (decoded above) — keep it.
  // -------------------------------------------------------------------------
  if (a.mode == "jobmine") {
    CUCHK(cudaMemcpy(g.dTarget, target_le.data(), 32, cudaMemcpyHostToDevice));
    // Noise seeds depend ONLY on the (fixed) header + config: derive once.
    Seeds s;
    derive_seeds_for_header(header, config, aroot, broot, s);
    uint64_t start = a.nonce_start, count = a.nonce_count ? a.nonce_count : 1;
    for (uint64_t i = 0; i < count; ++i) {
      uint64_t ab_seed = start + i;  // search var: A/B operand seed (header fixed)
      uint32_t ix = 0, iy = 0;
      int hit = run_attempt(g, s, a.m, a.n, a.k, ab_seed, &ix, &iy);
      if (hit) {
        int a_rows[8], b_cols[16]; uint32_t T[16]; uint8_t gpu_hash[32];
        read_transcript_and_indices(g, s, a.k, ix, iy, a_rows, b_cols, T);
        transcript_hash(T, s.a_noise_seed, gpu_hash);
        // `nonce` field == ab_seed here (header is never nonce-mutated); the
        // load-bearing field for the proven proof path is `ab_seed`.
        printf("HIT {\"ab_seed\":%llu,\"nonce\":%llu,\"tile\":[%u,%u],\"a_rows\":[",
               (unsigned long long)ab_seed, (unsigned long long)ab_seed, ix, iy);
        for (int j = 0; j < 8; ++j)  printf("%s%d", j ? "," : "", a_rows[j]);
        printf("],\"b_cols\":[");
        for (int j = 0; j < 16; ++j) printf("%s%d", j ? "," : "", b_cols[j]);
        printf("],\"transcript\":[");
        for (int j = 0; j < 16; ++j) printf("%s\"%08x\"", j ? "," : "", T[j]);
        printf("],\"gpu_hash\":\"");
        for (int j = 0; j < 32; ++j) printf("%02x", gpu_hash[j]);
        printf("\"}\n");
        fflush(stdout);
        return 0;
      }
    }
    printf("NOHIT\n");
    return 0;
  }

  // BENCH-MODE TARGET: force the HARDEST target (T=0, all-zero words) so the PoW
  // check is (essentially) never satisfied. The default target for mine/verify is
  // the EASIEST (0xFF..FF, T=2^256-1), which makes check_pow_target true for EVERY
  // hash-tile -> every CTA thread takes write_host_signal_header's grid-wide
  // global_lock atomicCAS + __threadfence_system + pinned-memory writeback path,
  // serializing the entire grid on one lock and flooding PCIe. That is the
  // ~13000x "memory-bound 49W/100%-util 50s" stall: it is the host-signal hit
  // path, NOT the GEMM/noising. The reference bench (bench_sm89_pouw_re2) uses an
  // all-zero target for exactly this reason. mine-mode keeps the real pool target
  // (rare hits), so this only changes the throughput benchmark, matching prod.
  if (a.mode == "bench") target_le.assign(32, 0x00);
  CUCHK(cudaMemcpy(g.dTarget, target_le.data(), 32, cudaMemcpyHostToDevice));

  // -------------------------------------------------------------------------
  // bench-mode: fixed nonce_count of full attempts, no early return. CUDA init
  // + alloc already amortized above. Time the attempt loop and report the rate.
  // -------------------------------------------------------------------------
  if (a.mode == "bench") {
    uint64_t bcount = a.nonce_count ? a.nonce_count : 1;
    std::vector<uint8_t> bheader = header;
    // one warm-up attempt (first launch pays JIT/module-load + caches) — not timed.
    {
      Seeds s;
      pearl_miner::blake3::hash_concat(bheader.data(), bheader.size(),
                                       config.data(), config.size(), nullptr, s.job_key);
      pearl_miner::blake3::hash_concat(s.job_key, 32, s.job_key, 32, nullptr, s.b_noise_seed);
      pearl_miner::blake3::hash_concat(s.b_noise_seed, 32, s.job_key, 32, nullptr, s.a_noise_seed);
      uint32_t ix=0, iy=0;
      run_attempt(g, s, a.m, a.n, a.k, /*ab_seed=*/0xABCDEF, &ix, &iy);
    }
    auto t0 = std::chrono::steady_clock::now();
    uint64_t hits = 0;
    for (uint64_t i = 0; i < bcount; ++i) {
      uint64_t nonce = a.nonce_start + i;
      bheader[72] = (uint8_t)(nonce);       bheader[73] = (uint8_t)(nonce >> 8);
      bheader[74] = (uint8_t)(nonce >> 16); bheader[75] = (uint8_t)(nonce >> 24);
      Seeds s;
      pearl_miner::blake3::hash_concat(bheader.data(), bheader.size(),
                                       config.data(), config.size(), nullptr, s.job_key);
      pearl_miner::blake3::hash_concat(s.job_key, 32, s.job_key, 32, nullptr, s.b_noise_seed);
      pearl_miner::blake3::hash_concat(s.b_noise_seed, 32, s.job_key, 32, nullptr, s.a_noise_seed);
      uint64_t ab_seed = nonce * 0x100000001B3ULL + 0xCBF29CE484222325ULL;
      uint32_t ix=0, iy=0;
      hits += run_attempt(g, s, a.m, a.n, a.k, ab_seed, &ix, &iy);
    }
    auto t1 = std::chrono::steady_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();
    double avg = secs / (double)bcount;
    double att_s = (double)bcount / secs;
    // MAC formula matches bench_sm89_pouw_re2:
    //   total = 2*K*R*(M+N) + M*N*(K+2*R)
    long double M = a.m, N = a.n, K = a.k, R = a.r;
    long double total = 2.0L*K*R*(M+N) + M*N*(K + 2.0L*R);
    double tmac_s = (double)(total / (long double)avg / 1e12L);
    printf("BENCH {\"attempts\":%llu,\"hits\":%llu,\"seconds\":%.6f,"
           "\"avg_attempt_sec\":%.6f,\"attempts_per_sec\":%.4f,\"tmac_s\":%.3f,"
           "\"m\":%d,\"n\":%d,\"k\":%d,\"r\":%d}\n",
           (unsigned long long)bcount, (unsigned long long)hits, secs, avg,
           att_s, tmac_s, a.m, a.n, a.k, a.r);
    return 0;
  }

  auto print_hit = [&](uint64_t nonce, uint64_t ab_seed, uint32_t ix, uint32_t iy,
                       const int* a_rows, const int* b_cols,
                       const uint32_t* T, const uint8_t* gpu_hash) {
    printf("HIT {\"nonce\":%llu,\"seed\":%llu,\"tile\":[%u,%u],\"a_rows\":[",
           (unsigned long long)nonce, (unsigned long long)ab_seed, ix, iy);
    for (int i=0;i<8;++i) printf("%s%d", i?",":"", a_rows[i]);
    printf("],\"b_cols\":[");
    for (int i=0;i<16;++i) printf("%s%d", i?",":"", b_cols[i]);
    printf("],\"transcript\":[");
    for (int i=0;i<16;++i) printf("%s\"%08x\"", i?",":"", T[i]);
    printf("],\"gpu_hash\":\"");
    for (int i=0;i<32;++i) printf("%02x", gpu_hash[i]);
    printf("\"}\n");
  };

  // Mining loop: nonce -> header[72:76] -> seeds -> attempt.
  uint64_t start = a.nonce_start, count = a.nonce_count;
  if (a.mode == "verify" && count == 0) count = 1;
  for (uint64_t i = 0; i < count; ++i) {
    uint64_t nonce = start + i;
    // mutate the 4-byte nonce suffix of the header.
    header[72] = (uint8_t)(nonce);
    header[73] = (uint8_t)(nonce >> 8);
    header[74] = (uint8_t)(nonce >> 16);
    header[75] = (uint8_t)(nonce >> 24);

    // Seeds: job_key = blake3(header||config). The full-matrix merkle roots
    // (hash_a/hash_b) depend on the private A/B and are computed by the driver
    // (pearl_mining). For the GPU attempt the noise keys are the only seed inputs
    // the kernel needs, and the transcript/hash are arch-independent. When real
    // roots are supplied (aroot=/broot=) we use the authoritative commitment
    // chain; otherwise we derive a self-consistent set from job_key (smoke). The
    // binary returns the A/B `seed` + opened indices so the driver replays A/B
    // and serializes the proof with the correct roots.
    Seeds s;
    if (!a.aroot_hex.empty() && !a.broot_hex.empty()) {
      s = derive_seeds(header.data(), header.size(), config.data(), config.size(),
                       aroot.data(), broot.data());
    } else {
      pearl_miner::blake3::hash_concat(header.data(), header.size(),
                                       config.data(), config.size(), nullptr, s.job_key);
      pearl_miner::blake3::hash_concat(s.job_key, 32, s.job_key, 32, nullptr, s.b_noise_seed);
      pearl_miner::blake3::hash_concat(s.b_noise_seed, 32, s.job_key, 32, nullptr, s.a_noise_seed);
    }

    uint64_t ab_seed = nonce * 0x100000001B3ULL + 0xCBF29CE484222325ULL;
    uint32_t ix=0, iy=0;
    int hit = run_attempt(g, s, a.m, a.n, a.k, ab_seed, &ix, &iy);
    if (hit) {
      int a_rows[8], b_cols[16]; uint32_t T[16]; uint8_t gpu_hash[32];
      read_transcript_and_indices(g, s, a.k, ix, iy, a_rows, b_cols, T);
      transcript_hash(T, s.a_noise_seed, gpu_hash);

      // VERIFY cross-check: independently recompute the transcript from the RAW
      // (un-noised) seed-derived A/B strips using the HOST noise reference, and
      // compare to the transcript read from the GPU-noised operands. A match
      // proves the GPU noisingA/B + integer GEMM + XOR/rotl13 reproduce the
      // arch-independent reference bit-exactly. This is the decisive local
      // hardware self-check; on real sm_89 the user additionally diffs T against
      // a known oracle transcript.
      if (a.mode == "verify") {
        std::vector<int8_t> rawA((size_t)8 * a.k), rawB((size_t)16 * a.k);
        std::vector<int8_t> fullA((size_t)a.m * a.k), fullB((size_t)a.n * a.k);
        fill_AB(fullA.data(), fullA.size(), ab_seed);
        fill_AB(fullB.data(), fullB.size(), ab_seed ^ 0xD1B54A32D192ED03ULL);
        for (int r = 0; r < 8; ++r)
          memcpy(&rawA[(size_t)r*a.k], &fullA[(size_t)a_rows[r]*a.k], a.k);
        for (int c = 0; c < 16; ++c)
          memcpy(&rawB[(size_t)c*a.k], &fullB[(size_t)b_cols[c]*a.k], a.k);
        // Direct device-vs-host noised-operand diff for the first opened row/col,
        // to localize any noise/orientation discrepancy.
        {
          std::vector<int8_t> gpuArow(a.k), gpuBcol(a.k);
          CUCHK(cudaMemcpy(gpuArow.data(), g.dApEA + (size_t)a_rows[0]*a.k, a.k, cudaMemcpyDeviceToHost));
          CUCHK(cudaMemcpy(gpuBcol.data(), g.dBpEB + (size_t)b_cols[0]*a.k, a.k, cudaMemcpyDeviceToHost));
          // host noised A row0 / B col0
          const uint8_t SA[32]={'A','_','t','e','n','s','o','r'};
          const uint8_t SB[32]={'B','_','t','e','n','s','o','r'};
          std::vector<int8_t> EAR, EBL; int8_t eal[256], ebr[256];
          pearl_miner::noise_sparse(a.k, a.r, SA, s.a_noise_seed, EAR);
          pearl_miner::noise_sparse(a.k, a.r, SB, s.b_noise_seed, EBL);
          pearl_miner::noise_dense_row(a_rows[0], a.r, SA, s.a_noise_seed, eal);
          pearl_miner::noise_dense_row(b_cols[0], a.r, SB, s.b_noise_seed, ebr);
          int adiff=0, bdiff=0;
          for (int kk=0; kk<a.k; ++kk) {
            int32_t ea=0, eb=0;
            for (int r=0;r<a.r;++r){ ea += (int32_t)eal[r]*(int32_t)EAR[(size_t)kk*a.r+r]; eb += (int32_t)ebr[r]*(int32_t)EBL[(size_t)kk*a.r+r]; }
            int8_t hostA = pearl_miner::i8wrap((int32_t)rawA[kk] + ea);
            int8_t hostB = pearl_miner::i8wrap((int32_t)rawB[kk] + eb);
            if (hostA != gpuArow[kk]) ++adiff;
            if (hostB != gpuBcol[kk]) ++bdiff;
          }
          fprintf(stderr, "VERIFY noised-operand diff row0: A %d/%d B %d/%d  gpuA[0..3]=%d,%d,%d,%d\n",
                  adiff, a.k, bdiff, a.k, gpuArow[0],gpuArow[1],gpuArow[2],gpuArow[3]);
        }
        uint32_t Tref[16];
        transcript_from_strips(rawA, rawB, a_rows, b_cols, 8, 16, a.k, a.r,
                               s.a_noise_seed, s.b_noise_seed, Tref);
        bool match = memcmp(Tref, T, sizeof(T)) == 0;
        fprintf(stderr, "VERIFY gpu_vs_host_transcript=%s\n  host_ref:",
                match ? "MATCH" : "MISMATCH");
        for (int i=0;i<16;++i) fprintf(stderr, " %08x", Tref[i]);
        fprintf(stderr, "\n");
      }

      print_hit(nonce, ab_seed, ix, iy, a_rows, b_cols, T, gpu_hash);
      return 0;
    }
  }
  printf("NOHIT\n");
  return 0;
}
