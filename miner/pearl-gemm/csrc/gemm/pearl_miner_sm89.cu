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
#include <set>      // emit_plain_proof_b64 (was relying on a transitive include)
#include <string>
#include <vector>
#include <ctime>    // time(2) for the random ab_seed job start
#include <random>
#include <chrono>
#include <array>    // deferred HIT emission (2026-06-11)
#include <atomic>
#include <memory>
#include <thread>
#include <utility>  // std::swap (pipelined noising double-buffer, 2026-06-12)

// ── serve-mode stdin portability (Linux poll/read vs Windows pipe peek) ──
// The serve loop drains JOB lines from a parent-piped stdin without blocking
// the GPU. On Linux this is poll(fd0)+read(2); on Windows the daemon pipes
// into us, so we PeekNamedPipe for available bytes + ReadFile. The Linux
// branch is byte-identical to the original (the fleet binary builds from this).
#ifdef _WIN32
  #ifndef NOMINMAX
  #define NOMINMAX            // keep windows.h from clobbering std::min/max (CUTLASS uses them)
  #endif
  #ifndef WIN32_LEAN_AND_MEAN
  #define WIN32_LEAN_AND_MEAN // trim windows.h to avoid macro collisions with CuTe
  #endif
  #include <windows.h>
  #include <process.h>
  #define getpid _getpid
  using ssize_t_compat = long long;
  // The target-adjust multiply (byte*adj + carry, adj = h*w*k < 2^23) never
  // exceeds ~2^57, so 64 bits is exact — MSVC has no __uint128_t.
  using pm_wide_t = unsigned long long;
  // >0 data available, 0 nothing yet, <0 EOF/broken pipe.
  static inline int pm_stdin_ready() {
    HANDLE h = GetStdHandle(STD_INPUT_HANDLE);
    if (h == INVALID_HANDLE_VALUE || h == NULL) return -1;
    if (GetFileType(h) == FILE_TYPE_PIPE) {
      DWORD avail = 0;
      if (!PeekNamedPipe(h, NULL, 0, NULL, &avail, NULL)) return -1;  // broken => EOF
      return avail > 0 ? 1 : 0;
    }
    return (WaitForSingleObject(h, 0) == WAIT_OBJECT_0) ? 1 : 0;  // console fallback
  }
  static inline ssize_t_compat pm_stdin_read(char* buf, size_t n) {
    HANDLE h = GetStdHandle(STD_INPUT_HANDLE);
    DWORD got = 0;
    if (!ReadFile(h, buf, (DWORD)n, &got, NULL)) return 0;  // EOF/broken
    return (ssize_t_compat)got;
  }
  static inline void pm_idle_wait(int ms) { Sleep((DWORD)ms); }
#else
  #include <poll.h>     // serve-mode non-blocking stdin
  #include <unistd.h>   // read(2)
  using ssize_t_compat = ssize_t;
  using pm_wide_t = __uint128_t;
  static inline int pm_stdin_ready() {
    struct pollfd pfd; pfd.fd = 0; pfd.events = POLLIN;
    int pr = poll(&pfd, 1, 0);
    if (pr <= 0) return 0;
    if (!(pfd.revents & (POLLIN | POLLHUP))) return 0;
    return 1;
  }
  static inline ssize_t_compat pm_stdin_read(char* buf, size_t n) { return read(0, buf, n); }
  static inline void pm_idle_wait(int ms) {
    struct pollfd pfd; pfd.fd = 0; pfd.events = POLLIN; poll(&pfd, 1, ms);
  }
#endif

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
// DENOISE-OFF search variant: bit-identical transcript/PoW digest (the digest is
// folded from the int32 accumulator in the mainloop, before denoise; denoise only
// builds the useful-work C, which the share proof does not use). ~1.6x faster.
extern "C" void pearl_gemm_sm89_pow_128x256x128_R256_nodenoise_nostore(
    int8_t const* A, int64_t lda, int8_t const* B, int64_t ldb,
    cutlass::half_t* C, int64_t ldc, float const* A_scales, float const* B_scales,
    cutlass::half_t const* EAL, cutlass::half_t const* EBR,
    cutlass::half_t const* AxEBL, cutlass::half_t const* EARxBpEB,
    uint32_t const* pow_target, uint32_t const* pow_key,
    void* host_signal_sync, void* host_signal_header_pinned,
    uint64_t* inner_hash_counter, int M, int N, int K, cudaStream_t stream);
extern "C" void pearl_gemm_sm89_pow_128x256x64_R256_nodenoise_nostore_s2(
    int8_t const* A, int64_t lda, int8_t const* B, int64_t ldb,
    cutlass::half_t* C, int64_t ldc, float const* A_scales, float const* B_scales,
    cutlass::half_t const* EAL, cutlass::half_t const* EBR,
    cutlass::half_t const* AxEBL, cutlass::half_t const* EARxBpEB,
    uint32_t const* pow_target, uint32_t const* pow_key,
    void* host_signal_sync, void* host_signal_header_pinned,
    uint64_t* inner_hash_counter, int M, int N, int K, cudaStream_t stream);
extern "C" void pearl_gemm_sm89_pow_128x256x64_R256_nodenoise_nostore_s3(
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
// On-GPU keyed-BLAKE3 per-chunk CVs (matrix stays in VRAM); host reduces to root.
extern "C" void pearl_blake3_chunk_cvs_sm89(const uint8_t* d_data, size_t n_bytes,
                                            const uint8_t key[32], uint32_t* d_cvs,
                                            cudaStream_t stream);
// On-GPU tree reduce of those CVs -> 32-byte root (clobbers d_cvs; d_scratch >= n_chunks*4 u32).
extern "C" void pearl_blake3_root_sm89(uint32_t* d_cvs, uint32_t n_chunks, const uint8_t key[32],
                                       uint32_t* d_scratch, uint8_t out_root[32], cudaStream_t stream);
// Split an R-major (rows,256) int8 matrix into two contiguous (rows,128) halves.
extern "C" void pearl_miner_split_rmajor_256_sm89(
    const int8_t* in, int8_t* out_lo, int8_t* out_hi, size_t rows,
    cudaStream_t stream);
// Single-pass R=256 noising (2026-06-12): out = i8wrap(X + D@S^T) consuming the
// noisegen R-major outputs directly — replaces the chained 2x R128 passes, the
// 4 repack kernels, and the dead denoise-scratch outputs (~16.9ms -> ~7ms).
extern "C" void pearl_noising_fused_r256(
    const int8_t* X, const int8_t* D, const int8_t* S, int8_t* out,
    int rows, int K, cudaStream_t stream);
// PEARL_LEGACY_NOISING=1 restores the chained R128 noising path (A/B fallback).
static bool pm_use_fused_noising() {
  static const bool legacy = [](){
    const char* v = std::getenv("PEARL_LEGACY_NOISING");
    return v && v[0] && v[0] != '0';
  }();
  return !legacy;
}

#include "blake3_tree_host.hpp"  // canonical keyed-BLAKE3 matrix root (== pearl_mining)

using pearl_miner::Seeds;
using pearl_miner::derive_seeds;
using pearl_miner::transcript_hash;
using pearl_miner::transcript_from_strips;

#define CUCHK(x) do { auto _e = (x); if (_e != cudaSuccess) { \
    fprintf(stderr, "CUDA %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
    std::exit(2); } } while (0)

static constexpr int R_DIM = 256;
static constexpr int TILE_M = 128, TILE_N = 256;  // PoW kernel tile
// Production hash-tile strided pattern (pearlhash-150, 2026-06-09): h=8 x w=16,
// matching the (2,4) warp grid in kernel_traits_sm89.hpp — per-thread fragment
// rows {r,r+8}+32a (a<4), cols {c,c+1}+32b (b<8). BYTE-IDENTICAL to lpminer's
// pool-accepted mining_config (rows [0,8,32,40,64,72,96,104],
// cols [0,1,32,33,...,224,225]) so the pool's PeriodicPattern::from_list
// derivation reproduces the same 52-byte config the daemon passes as
// PEARL_CONFIG_HEX (rows bytes 070101030000, cols bytes 00010f070000).
// MUST stay consistent with the kernel warp grid AND the daemon config hex —
// an inconsistent flip = silent share drops (wave-8 failure mode).
// (Pre-2026-06-09 history: h=2 x w=64 with RP={0,8}, CP={0,1,8,9,...,249}
// matched the old (8,1) warp grid; config hex rows 070100000000 cols 0001031f0000.)
static constexpr int HASH_H = 8, HASH_W = 16;
static const int RP[HASH_H] = {0, 8, 32, 40, 64, 72, 96, 104};
static const int CP[HASH_W] = {
  0,1,32,33,64,65,96,97,128,129,160,161,192,193,224,225};

// ---- tiny arg parsing ----
struct Args {
  std::string header_hex, config_hex, target_hex, mode = "mine";
  std::string aroot_hex, broot_hex;  // optional: real A/B merkle roots (hash_a/hash_b)
  std::string target_endian = "big";  // "big" (pool/human, MSB-first) or "little"
  int m = 131072, n = 131072, k = 4096, r = 256, dev = 0;
  uint64_t nonce_start = 0, nonce_count = 1;
  int real_commit = 0;  // 1 = search with the real per-nonce A/B merkle-root commitment
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
  else if (k == "real_commit") a.real_commit = atoi(v.c_str());
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
  // ---- real-commitment mining (per-nonce A/B keyed-BLAKE3 merkle roots) ----
  bool real_commit=false;                    // search with the verifier's real key
  std::vector<uint8_t> header, config;       // job header(76)+config(52) for derive_seeds
  uint32_t *d_cvs=0;                          // device per-chunk CVs (on-GPU root)
  uint32_t *d_root_scratch=0;                 // device ping-pong scratch for the on-GPU tree reduce
  uint8_t  *h_cvs=0;                          // pinned host CV array (16MB; HIT-proof + debug oracle)
  Seeds cur_seeds;                           // seeds actually used this attempt (real or fallback)
  // ---- prep-ahead overlap (2026-06-10): derive attempt i+1's operands +
  //      commitment seeds on a non-blocking stream WHILE attempt i's PoW kernel
  //      runs on the legacy stream (hides the ~15-17ms fill_AB + merkle commit
  //      of the ~450ms attempt). State below carries the prep across calls. ----
  cudaStream_t pow_stream=0;                 // optional high-priority current-attempt stream
  cudaStream_t prep_stream=0;                // cudaStreamNonBlocking (mine mode only)
  bool prep_valid=false;                     // dA/dB + prep_seeds hold prep_seed's data
  uint64_t prep_seed=0;                      // ab_seed the prep-ahead ran for
  std::vector<uint8_t> prep_header;          // job header the prep seeds were derived for
  Seeds prep_seeds;                          // commitment-chain seeds for prep_seed
  // ---- fixed-B search experiment -------------------------------------------
  // PEARL_MINER_FIXED_B=1: hold raw B/BpEB fixed for the current job header
  // and vary only A. This is protocol-legal because
  //   b_noise_seed = blake3(job_key || B_root)
  // is independent of A_root; a_noise_seed still varies per A_root.
  bool fixed_b_valid=false;
  uint64_t fixed_b_seed=0;
  std::vector<uint8_t> fixed_b_header;
  uint8_t fixed_b_root[32]{};
  uint8_t fixed_b_noise_seed[32]{};
  // ---- pipelined noising (2026-06-12): the prep-ahead ALSO runs the full
  //      noisegen -> split -> chained noisingA/B for the NEXT attempt on the
  //      prep stream, writing into the ALT operand pair below while the PoW
  //      kernel reads the current pair (~17ms of formerly-serial GPU work
  //      hidden under the ~418ms kernel). On adoption the pairs swap. ----
  int8_t  *dApEA2=0,*dBpEB2=0;               // alt noised-operand pair (mine mode only)
  bool prep_noised=false;                    // dApEA2/dBpEB2 hold prep_seed's noised operands
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
  // Zero the four fp16 denoise factors ONCE at alloc. In mine/serve/bench mode
  // nothing ever writes them again (noisegen passes nullptr for fp16 outputs;
  // the PoW kernels take them const) — hoisted out of run_attempt_mine, where
  // re-zeroing cost 256 MiB of memset (~0.5 ms) per attempt. Verify mode still
  // re-zeros per attempt in its own path.
  CUCHK(cudaMemset(g.dEAL_fp16, 0, size_t(M)*R_DIM*2));
  CUCHK(cudaMemset(g.dEBR_fp16, 0, size_t(N)*R_DIM*2));
  CUCHK(cudaMemset(g.dAxEBL_fp16, 0, size_t(M)*R_DIM*2));
  CUCHK(cudaMemset(g.dEARxBpEB_fp16, 0, size_t(N)*R_DIM*2));

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
    // On-GPU merkle-root scratch: per-chunk CVs (32B each). 1 CV per 1024 bytes.
    size_t max_chunks = (size_t(M)*K > size_t(N)*K ? size_t(M)*K : size_t(N)*K) / 1024;
    CUCHK(cudaMalloc(&g.d_cvs, max_chunks * 32));
    CUCHK(cudaMalloc(&g.d_root_scratch, max_chunks * 16));  // ping-pong: holds n_chunks/2 nodes (n_chunks*4 u32)
    CUCHK(cudaMallocHost(&g.h_cvs, max_chunks * 32));
    if (const char* v = std::getenv("PEARL_MINER_HIGH_PRIORITY_POW");
        v && v[0] && v[0] != '0') {
      int least_priority = 0, greatest_priority = 0;
      CUCHK(cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority));
      CUCHK(cudaStreamCreateWithPriority(
          &g.pow_stream, cudaStreamNonBlocking, greatest_priority));
    }
    // Non-blocking stream for the prep-ahead overlap: kernels on it run
    // concurrently with legacy-default-stream work (a plain stream would
    // implicitly serialize against stream 0 and void the overlap).
    CUCHK(cudaStreamCreateWithFlags(&g.prep_stream, cudaStreamNonBlocking));
    // Alt noised-operand pair for the pipelined noising double-buffer (+1 GB).
    CUCHK(cudaMalloc(&g.dApEA2, size_t(M)*K));
    CUCHK(cudaMalloc(&g.dBpEB2, size_t(N)*K));
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
    const char* names[6] = {"operands+seeds", "noisegen", "split_rmajor",
                            "noisingAB", "sync_upload", "PoW_kernel"};
    float tot = 0;
    for (int i = 0; i < 6; ++i) {
      float ms = 0; CUCHK(cudaEventElapsedTime(&ms, e[i], e[i+1])); tot += ms;
      fprintf(stderr, "PROFILE phase%d %-13s %9.3f ms\n", i+1, names[i], ms);
    }
    fprintf(stderr, "PROFILE total %25.3f ms\n", tot);
  }
};

// Real-commitment seed derivation for `ab_seed` on `stream`: fill dA/dB
// (deterministic splitmix64), compute the keyed-BLAKE3 merkle roots ON-GPU
// (per-chunk CVs + tree reduce; only 32 bytes leave the device per root, ==
// pearl_mining.MerkleTree.root), then run the host commitment chain.
// Touches ONLY dA/dB + d_cvs/d_root_scratch — disjoint from every buffer the
// PoW kernel reads — so on the non-blocking prep stream this runs CONCURRENTLY
// with the legacy-stream PoW kernel (the prep-ahead overlap, 2026-06-10).
// Blocks the host until the roots are back (the root helper syncs `stream`
// only). PEARL_DEBUG_ROOT's host-oracle cross-check is only wired for the
// serial path (dbg=true implies stream 0; its sync memcpys would serialize the
// legacy stream and void the overlap).
static Seeds derive_real_seeds_on(GpuBufs& g, int M, int N, int K,
                                  uint64_t ab_seed, cudaStream_t stream, bool dbg) {
  pearl_miner_fill_AB_sm89(g.dA, size_t(M)*K, ab_seed, stream);
  pearl_miner_fill_AB_sm89(g.dB, size_t(N)*K, ab_seed ^ 0xD1B54A32D192ED03ULL, stream);
  uint8_t job_key[32], a_root[32], b_root[32];
  pearl_miner::blake3::hash_concat(g.header.data(), g.header.size(),
                                   g.config.data(), g.config.size(), nullptr, job_key);
  size_t a_chunks = size_t(M)*K / 1024, b_chunks = size_t(N)*K / 1024;
  uint8_t a_root_ref[32], b_root_ref[32]; int a_ok = 1, b_ok = 1;

  pearl_blake3_chunk_cvs_sm89((const uint8_t*)g.dA, size_t(M)*K, job_key, g.d_cvs, stream);
  if (dbg) {  // host oracle BEFORE the GPU reduce clobbers d_cvs
    CUCHK(cudaStreamSynchronize(stream));
    CUCHK(cudaMemcpy(g.h_cvs, g.d_cvs, a_chunks*32, cudaMemcpyDeviceToHost));
    pearl_miner::b3tree::blake3_root_from_chunk_cvs(job_key, (const uint32_t*)g.h_cvs, a_chunks, a_root_ref);
  }
  pearl_blake3_root_sm89(g.d_cvs, (uint32_t)a_chunks, job_key, g.d_root_scratch, a_root, stream);

  pearl_blake3_chunk_cvs_sm89((const uint8_t*)g.dB, size_t(N)*K, job_key, g.d_cvs, stream);
  if (dbg) {
    CUCHK(cudaStreamSynchronize(stream));
    CUCHK(cudaMemcpy(g.h_cvs, g.d_cvs, b_chunks*32, cudaMemcpyDeviceToHost));
    pearl_miner::b3tree::blake3_root_from_chunk_cvs(job_key, (const uint32_t*)g.h_cvs, b_chunks, b_root_ref);
  }
  pearl_blake3_root_sm89(g.d_cvs, (uint32_t)b_chunks, job_key, g.d_root_scratch, b_root, stream);

  if (dbg) {
    a_ok = memcmp(a_root, a_root_ref, 32) == 0; b_ok = memcmp(b_root, b_root_ref, 32) == 0;
    fprintf(stderr, "DEBUG_ROOT ab_seed=%llu a_chunks=%zu gpu_a_match=%d gpu_b_match=%d a_root=",
            (unsigned long long)ab_seed, a_chunks, a_ok, b_ok);
    for (int i = 0; i < 32; ++i) fprintf(stderr, "%02x", a_root[i]);
    fprintf(stderr, " b_root=");
    for (int i = 0; i < 32; ++i) fprintf(stderr, "%02x", b_root[i]);
    fprintf(stderr, "\n");
  }
  return derive_seeds(g.header.data(), g.header.size(),
                      g.config.data(), g.config.size(), a_root, b_root);
}

static void compute_matrix_root_on(GpuBufs& g, const int8_t* d_matrix,
                                   size_t bytes, const uint8_t job_key[32],
                                   uint8_t out_root[32], cudaStream_t stream) {
  size_t chunks = bytes / 1024;
  pearl_blake3_chunk_cvs_sm89((const uint8_t*)d_matrix, bytes, job_key,
                              g.d_cvs, stream);
  pearl_blake3_root_sm89(g.d_cvs, (uint32_t)chunks, job_key,
                         g.d_root_scratch, out_root, stream);
}

static bool fixed_b_enabled() {
  static const bool enabled = [](){
    const char* v = std::getenv("PEARL_MINER_FIXED_B");
    return v && v[0] && v[0] != '0';
  }();
  return enabled && pm_use_fused_noising();
}

static void prepare_fixed_b_for_job(GpuBufs& g, int M, int N, int K,
                                    uint64_t b_seed, cudaStream_t stream) {
  if (g.fixed_b_valid && g.fixed_b_header == g.header &&
      g.fixed_b_seed == b_seed) {
    return;
  }

  uint8_t job_key[32];
  pearl_miner::blake3::hash_concat(g.header.data(), g.header.size(),
                                   g.config.data(), g.config.size(), nullptr,
                                   job_key);
  pearl_miner_fill_AB_sm89(g.dB, size_t(N) * K, b_seed, stream);
  compute_matrix_root_on(g, g.dB, size_t(N) * K, job_key,
                         g.fixed_b_root, stream);
  pearl_miner::blake3::hash_concat(job_key, 32, g.fixed_b_root, 32, nullptr,
                                   g.fixed_b_noise_seed);

  CUCHK(cudaMemcpyAsync(g.dKeyB, g.fixed_b_noise_seed, 32,
                        cudaMemcpyHostToDevice, stream));
  pearl_miner_noisegen_sm89_R256(
      nullptr, g.dEBR_i8, nullptr, nullptr, g.dEBL_R, nullptr,
      g.dKeyA, g.dKeyB, M, N, K, stream);
  pearl_noising_fused_r256(g.dB, g.dEBR_i8, g.dEBL_R, g.dBpEB, N, K, stream);

  g.fixed_b_valid = true;
  g.fixed_b_seed = b_seed;
  g.fixed_b_header = g.header;
}

static Seeds derive_fixed_b_a_seed_on(GpuBufs& g, int M, int K,
                                      uint64_t ab_seed, cudaStream_t stream) {
  Seeds s{};
  uint8_t a_root[32];
  pearl_miner::blake3::hash_concat(g.header.data(), g.header.size(),
                                   g.config.data(), g.config.size(), nullptr,
                                   s.job_key);
  pearl_miner_fill_AB_sm89(g.dA, size_t(M) * K, ab_seed, stream);
  compute_matrix_root_on(g, g.dA, size_t(M) * K, s.job_key, a_root, stream);
  memcpy(s.b_noise_seed, g.fixed_b_noise_seed, 32);
  pearl_miner::blake3::hash_concat(s.b_noise_seed, 32, a_root, 32, nullptr,
                                   s.a_noise_seed);
  return s;
}

static void noising_fixed_b_a_on(GpuBufs& g, const Seeds& s, int M, int N, int K,
                                 int8_t* out_A, cudaStream_t stream) {
  (void)N;
  CUCHK(cudaMemcpyAsync(g.dKeyA, s.a_noise_seed, 32,
                        cudaMemcpyHostToDevice, stream));
  pearl_miner_noisegen_sm89_R256(
      g.dEAL_i8, nullptr, g.dEAR_R, nullptr, nullptr, nullptr,
      g.dKeyA, g.dKeyB, M, N, K, stream);
  pearl_noising_fused_r256(g.dA, g.dEAL_i8, g.dEAR_R, out_A, M, K, stream);
}

static int run_attempt_mine(GpuBufs& g, const Seeds& s, int M, int N, int K,
                            uint64_t ab_seed, uint32_t* tile_ix, uint32_t* tile_iy,
                            bool prep_next = false, uint64_t next_seed = 0) {
  PhaseProf prof;
  prof.mark(0);
  const bool dbg = std::getenv("PEARL_DEBUG_ROOT") != nullptr;
  static const bool stream_sync_only = [](){
    const char* v = std::getenv("PEARL_MINER_STREAM_SYNC");
    return v && v[0] && v[0] != '0';
  }();
  cudaStream_t const cur_stream = g.pow_stream ? g.pow_stream : cudaStream_t(0);

  // 1. operands + this attempt's seeds. Real-commitment mode computes the
  // verifier's key from the ACTUAL A/B (without it the search uses header-only
  // fallback seeds -> spurious hits the pool rejects). If the PREVIOUS call's
  // prep-ahead already derived this (seed, header)'s operands + seeds during
  // its PoW window, reuse them — dA/dB already hold this seed's data (the
  // prior call's cudaDeviceSynchronize ordered the prep-stream writes).
  bool adopted_noised = false;   // pipelined prep already noised this attempt's operands
  bool const use_fixed_b = g.real_commit && fixed_b_enabled() && !dbg;
  if (g.real_commit) {
    if (use_fixed_b) {
      uint64_t const b_seed =
          (g.fixed_b_valid && g.fixed_b_header == g.header)
              ? g.fixed_b_seed
              : (ab_seed ^ 0xD1B54A32D192ED03ULL);
      prepare_fixed_b_for_job(g, M, N, K, b_seed, cur_stream);
    }
    if (stream_sync_only && g.prep_valid && !dbg) {
      CUCHK(cudaStreamSynchronize(g.prep_stream));
    }
    if (g.prep_valid && g.prep_seed == ab_seed && g.prep_header == g.header && !dbg) {
      g.cur_seeds = g.prep_seeds;
      if (g.prep_noised) {
        // Pipelined noising (2026-06-12): the previous call's prep stream
        // already produced this attempt's noised operands in the ALT pair
        // (ordered by its cudaDeviceSynchronize). Swap pairs and skip the
        // serial noisegen/split/noising phases entirely.
        std::swap(g.dApEA, g.dApEA2);
        if (!use_fixed_b) std::swap(g.dBpEB, g.dBpEB2);
        adopted_noised = true;
      }
    } else {
      g.cur_seeds = use_fixed_b
          ? derive_fixed_b_a_seed_on(g, M, K, ab_seed, cur_stream)
          : derive_real_seeds_on(g, M, N, K, ab_seed, cur_stream, dbg);
    }
    g.prep_valid = false; g.prep_noised = false;
  } else {
    pearl_miner_fill_AB_sm89(g.dA, size_t(M)*K, ab_seed, cur_stream);
    pearl_miner_fill_AB_sm89(g.dB, size_t(N)*K, ab_seed ^ 0xD1B54A32D192ED03ULL, cur_stream);
    g.cur_seeds = s;
  }
  prof.mark(1);
  const Seeds& cs = g.cur_seeds;

  if (!adopted_noised) {
  if (use_fixed_b) {
    noising_fixed_b_a_on(g, cs, M, N, K, g.dApEA, cur_stream);
    prof.mark(2);
    prof.mark(3);
  } else {
  // 2. on-device noise factors (R=256) from a/b_noise_seed.
  CUCHK(cudaMemcpy(g.dKeyA, cs.a_noise_seed, 32, cudaMemcpyHostToDevice));
  CUCHK(cudaMemcpy(g.dKeyB, cs.b_noise_seed, 32, cudaMemcpyHostToDevice));
  pearl_miner_noisegen_sm89_R256(
      g.dEAL_i8, g.dEBR_i8, g.dEAR_R, g.dEAR_K, g.dEBL_R, g.dEBL_K,
      g.dKeyA, g.dKeyB, M, N, K, cur_stream);
  prof.mark(2);

  if (pm_use_fused_noising()) {
    // Fused single-pass noising: consumes EAL/EBR (rows,256) + EAR_R/EBL_R
    // (K,256) directly — no repack, no chaining, no denoise scratch.
    prof.mark(3);
    pearl_noising_fused_r256(g.dA, g.dEAL_i8, g.dEAR_R, g.dApEA, M, K, cur_stream);
    pearl_noising_fused_r256(g.dB, g.dEBR_i8, g.dEBL_R, g.dBpEB, N, K, cur_stream);
  } else {
  // 2b. Repack the R-major factors (EAL/EAR for A, EBR/EBL for B) into two
  //     contiguous (rows,128) R-halves. The R=128 noising kernel hard-codes
  //     ld=128, so it cannot consume a strided slice of the (rows,256) buffer.
  //     The K-major sparse factors (EBL_K for A, EAR_K for B) are already
  //     contiguous per half (row-blocks), so they are sliced directly.
  pearl_miner_split_rmajor_256_sm89(g.dEAL_i8, g.dEAL_h[0], g.dEAL_h[1], M, cur_stream);
  pearl_miner_split_rmajor_256_sm89(g.dEAR_R,  g.dEAR_h[0], g.dEAR_h[1], K, cur_stream);
  pearl_miner_split_rmajor_256_sm89(g.dEBR_i8, g.dEBR_h[0], g.dEBR_h[1], N, cur_stream);
  pearl_miner_split_rmajor_256_sm89(g.dEBL_R,  g.dEBL_h[0], g.dEBL_h[1], K, cur_stream);
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
        g.dApEA, g.dAxEBL_i32, M, K, cur_stream);
    pearl::sm89::pearl_noisingB_sm89_64x128x64_R128_int32(
        bIn,
        g.dEBR_h[h],                 // EBR half h, contiguous (N,128)
        g.dEBL_h[h],                 // EBL_R half h, contiguous (K,128)
        g.dEAR_K + size_t(roff)*K,   // EAR_K_major row-block [roff:, :] (ld=K)
        g.dBpEB, g.dEARxBpEB_i32, N, K, cur_stream);
  }
  }  // end legacy (non-fused) noising path
  }
  } else {
    prof.mark(2); prof.mark(3);    // phases hidden under the previous PoW kernel
  }
  prof.mark(4);

  // 4. denoise fp16 factors are irrelevant to the transcript/PoW and are never
  // written in mine mode — zeroed ONCE in alloc_bufs (hoisted 2026-06-10; was
  // 256 MiB of per-attempt memset).

  CUCHK(cudaMemcpy(g.dKey, cs.a_noise_seed, 32, cudaMemcpyHostToDevice));
  HostSignalSync zsync{};
  CUCHK(cudaMemcpy(g.dSync, &zsync, sizeof(zsync), cudaMemcpyHostToDevice));
  memset(g.hHeader, 0, host_signal_header_size);
  prof.mark(5);

  // Default to the denoise-OFF search kernel (~1.6x faster, bit-identical
  // transcript/digest). Set PEARL_MINER_DENOISE=1 to force the legacy denoise-ON
  // path (e.g. for an on-rig A/B or if a future proof needs the useful-work C).
  static const bool force_denoise = [](){
    const char* v = std::getenv("PEARL_MINER_DENOISE");
    return v && v[0] && v[0] != '0';
  }();
  static const int pow_bk64_stage = [](){
    const char* v = std::getenv("PEARL_SM89_POW_BK64_STAGE");
    if (!v || !v[0]) return 0;
    int stage = std::atoi(v);
    return (stage == 2 || stage == 3) ? stage : 0;
  }();
  if (force_denoise) {
    pearl::sm89::pearl_gemm_sm89_pow_128x256x128_R256_nostore(
        g.dApEA, K, g.dBpEB, K, nullptr, N, g.dAs, g.dBs,
        g.dEAL_fp16, g.dEBR_fp16, g.dAxEBL_fp16, g.dEARxBpEB_fp16,
        g.dTarget, g.dKey, g.dSync, g.hHeader, nullptr, M, N, K, cur_stream);
  } else if (pow_bk64_stage == 2) {
    pearl::sm89::pearl_gemm_sm89_pow_128x256x64_R256_nodenoise_nostore_s2(
        g.dApEA, K, g.dBpEB, K, nullptr, N, g.dAs, g.dBs,
        g.dEAL_fp16, g.dEBR_fp16, g.dAxEBL_fp16, g.dEARxBpEB_fp16,
        g.dTarget, g.dKey, g.dSync, g.hHeader, nullptr, M, N, K, cur_stream);
  } else if (pow_bk64_stage == 3) {
    pearl::sm89::pearl_gemm_sm89_pow_128x256x64_R256_nodenoise_nostore_s3(
        g.dApEA, K, g.dBpEB, K, nullptr, N, g.dAs, g.dBs,
        g.dEAL_fp16, g.dEBR_fp16, g.dAxEBL_fp16, g.dEARxBpEB_fp16,
        g.dTarget, g.dKey, g.dSync, g.hHeader, nullptr, M, N, K, cur_stream);
  } else {
    pearl::sm89::pearl_gemm_sm89_pow_128x256x128_R256_nodenoise_nostore(
        g.dApEA, K, g.dBpEB, K, nullptr, N, g.dAs, g.dBs,
        g.dEAL_fp16, g.dEBR_fp16, g.dAxEBL_fp16, g.dEARxBpEB_fp16,
        g.dTarget, g.dKey, g.dSync, g.hHeader, nullptr, M, N, K, cur_stream);
  }
  // PREP-AHEAD (overlap lever, 2026-06-10): while the PoW kernel runs on the
  // legacy stream (~424ms), derive the NEXT attempt's operands + commitment
  // seeds on the non-blocking prep stream (~15-17ms): fill_AB + on-GPU merkle
  // roots touch only dA/dB + d_cvs/d_root_scratch, which the PoW kernel never
  // reads. The host blocks briefly on the prep stream's two 32B root copies,
  // then waits out the remaining PoW time in the device sync below.
  static const bool no_prep = [](){          // PEARL_NO_PREP=1: serial A/B knob
    const char* v = std::getenv("PEARL_NO_PREP");
    return v && v[0] && v[0] != '0';
  }();
  bool prepped = false;
  if (g.real_commit && prep_next && !dbg && !no_prep) {
    g.prep_seeds = use_fixed_b
        ? derive_fixed_b_a_seed_on(g, M, K, next_seed, g.prep_stream)
        : derive_real_seeds_on(g, M, N, K, next_seed, g.prep_stream, false);
    // PIPELINED NOISING (2026-06-12): with next_seed's commitment seeds in
    // hand, run its ENTIRE noise chain (key upload -> noisegen -> split ->
    // chained noisingA/B) on the prep stream into the ALT operand pair, all
    // overlapped with the PoW kernel still running on stream 0. The PoW kernel
    // reads only dApEA/dBpEB (current pair) + the never-written fp16 zeros, so
    // every buffer touched here (factors, i32 scratch, dKeyA/B, alt pair,
    // dA/dB) is disjoint from its reads. The end-of-call device sync below
    // orders everything; the next call adopts via pointer swap and skips its
    // serial phases (~17ms/attempt formerly serial, now hidden).
    if (use_fixed_b) {
      noising_fixed_b_a_on(g, g.prep_seeds, M, N, K, g.dApEA2, g.prep_stream);
    } else {
      CUCHK(cudaMemcpyAsync(g.dKeyA, g.prep_seeds.a_noise_seed, 32,
                            cudaMemcpyHostToDevice, g.prep_stream));
      CUCHK(cudaMemcpyAsync(g.dKeyB, g.prep_seeds.b_noise_seed, 32,
                            cudaMemcpyHostToDevice, g.prep_stream));
      pearl_miner_noisegen_sm89_R256(
          g.dEAL_i8, g.dEBR_i8, g.dEAR_R, g.dEAR_K, g.dEBL_R, g.dEBL_K,
          g.dKeyA, g.dKeyB, M, N, K, g.prep_stream);
      if (pm_use_fused_noising()) {
      pearl_noising_fused_r256(g.dA, g.dEAL_i8, g.dEAR_R, g.dApEA2, M, K, g.prep_stream);
      pearl_noising_fused_r256(g.dB, g.dEBR_i8, g.dEBL_R, g.dBpEB2, N, K, g.prep_stream);
      } else {
    pearl_miner_split_rmajor_256_sm89(g.dEAL_i8, g.dEAL_h[0], g.dEAL_h[1], M, g.prep_stream);
    pearl_miner_split_rmajor_256_sm89(g.dEAR_R,  g.dEAR_h[0], g.dEAR_h[1], K, g.prep_stream);
    pearl_miner_split_rmajor_256_sm89(g.dEBR_i8, g.dEBR_h[0], g.dEBR_h[1], N, g.prep_stream);
    pearl_miner_split_rmajor_256_sm89(g.dEBL_R,  g.dEBL_h[0], g.dEBL_h[1], K, g.prep_stream);
    for (int h = 0; h < 2; ++h) {
      int roff = h * 128;
      const int8_t* aIn = (h == 0) ? g.dA : g.dApEA2;
      const int8_t* bIn = (h == 0) ? g.dB : g.dBpEB2;
      pearl_noisingA_sm89_64x128x64_R128_int32(
          aIn, g.dEAL_h[h], g.dEAR_h[h], g.dEBL_K + size_t(roff)*K,
          g.dApEA2, g.dAxEBL_i32, M, K, g.prep_stream);
      pearl::sm89::pearl_noisingB_sm89_64x128x64_R128_int32(
          bIn, g.dEBR_h[h], g.dEBL_h[h], g.dEAR_K + size_t(roff)*K,
          g.dBpEB2, g.dEARxBpEB_i32, N, K, g.prep_stream);
    }
    }  // end legacy (non-fused) prep noising
    }
    g.prep_seed = next_seed;
    g.prep_header = g.header;
    g.prep_valid = true;
    g.prep_noised = true;
    prepped = true;
  }
  prof.mark(6);
  if (stream_sync_only) {
    CUCHK(cudaStreamSynchronize(cur_stream));
  } else {
    CUCHK(cudaDeviceSynchronize());
  }
  cudaError_t le = cudaGetLastError();
  if (le != cudaSuccess) { fprintf(stderr, "PoW(nostore) launch: %s\n", cudaGetErrorString(le)); std::exit(2); }
  prof.report();

  if (g.hHeader->status == kSignalTriggered) {
    if (prepped) {
      if (stream_sync_only) {
        CUCHK(cudaStreamSynchronize(g.prep_stream));
      }
      // The prep-ahead overwrote dA/dB with the NEXT seed's operands, but the
      // HIT path (emit_plain_proof_b64) re-reads THIS attempt's raw A/B.
      // Restore them (deterministic fill, ~3ms; hits are ~1/60 attempts) and
      // invalidate the prep — the next call falls back to the serial path.
      // (The prep stream is already idle here — the device sync above ordered
      // its noising writes — so the refill cannot race the pipelined reads;
      // the alt pair's contents are simply discarded with the prep.)
      pearl_miner_fill_AB_sm89(g.dA, size_t(M)*K, ab_seed, cur_stream);
      if (!use_fixed_b) {
        pearl_miner_fill_AB_sm89(g.dB, size_t(N)*K,
                                 ab_seed ^ 0xD1B54A32D192ED03ULL, cur_stream);
      }
      CUCHK(cudaDeviceSynchronize());
      g.prep_valid = false; g.prep_noised = false;
    }
    *tile_ix = g.hHeader->tileCoord[0];
    *tile_iy = g.hHeader->tileCoord[1];
    return 1;
  }
  return 0;
}

static int run_attempt(GpuBufs& g, const Seeds& s, int M, int N, int K,
                       uint64_t ab_seed, uint32_t* tile_ix, uint32_t* tile_iy,
                       bool prep_next = false, uint64_t next_seed = 0) {
  if (g.mine) return run_attempt_mine(g, s, M, N, K, ab_seed, tile_ix, tile_iy,
                                      prep_next, next_seed);
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
  if (std::getenv("PEARL_DEBUG_CELLS")) {
    bool rseen[128] = {false}, cseen[256] = {false};
    int nr = 0, nc = 0;
    for (int j = 0; j < nreg; ++j) {
      int r = g.hHeader->thread_rows[j], c = g.hHeader->thread_cols[j];
      if (r >= 0 && r < 128 && !rseen[r]) { rseen[r] = true; ++nr; }
      if (c >= 0 && c < 256 && !cseen[c]) { cseen[c] = true; ++nc; }
    }
    fprintf(stderr, "DEBUG_CELLS nreg=%d uniq_rows=%d uniq_cols=%d ro=%d co=%d\n  rows(rel):", nreg, nr, nc, ro, co);
    for (int r = 0; r < 128; ++r) if (rseen[r]) fprintf(stderr, " %d", r - ro);
    fprintf(stderr, "\n  cols(rel):");
    for (int c = 0; c < 256; ++c) if (cseen[c]) fprintf(stderr, " %d", c - co);
    fprintf(stderr, "\n");
  }
  if (ro + RP[HASH_H-1] >= TILE_M) ro = TILE_M - 1 - RP[HASH_H-1];   // keep H rows in-tile
  if (co + CP[HASH_W-1] >= TILE_N) co = TILE_N - 1 - CP[HASH_W-1];   // keep W cols in-tile
  if (ro < 0) ro = 0; if (co < 0) co = 0;
  int row0 = (int)tile_ix * TILE_M + ro;
  int col0 = (int)tile_iy * TILE_N + co;
  for (int a = 0; a < HASH_H; ++a) a_rows[a] = row0 + RP[a];
  for (int b = 0; b < HASH_W; ++b) b_cols[b] = col0 + CP[b];

  // Copy the H noised-A rows (ApEA) + W noised-B cols (BpEB) back to host.
  std::vector<int8_t> An((size_t)HASH_H * K), Bn((size_t)HASH_W * K);
  for (int a = 0; a < HASH_H; ++a)
    CUCHK(cudaMemcpy(&An[(size_t)a*K], g.dApEA + (size_t)a_rows[a]*K, K, cudaMemcpyDeviceToHost));
  for (int b = 0; b < HASH_W; ++b)
    CUCHK(cudaMemcpy(&Bn[(size_t)b*K], g.dBpEB + (size_t)b_cols[b]*K, K, cudaMemcpyDeviceToHost));

  // Transcript over the NOISED operands — CUMULATIVE across R-chunks, exactly as
  // the verifier (zk-pow jackpot/helper.rs: the jackpot tile is declared OUTSIDE
  // the R-step loop and `+=` accumulates, never reset). Per R-chunk rc:
  //   acc_tile[i][j] += An[i][p:p+R] . Bn[j][p:p+R] ; x = XOR-reduce(cumulative
  //   acc_tile) ; T[rc%16] = rotl13(T[rc%16]) ^ x.   (H x W = 128 cells.)
  for (int i = 0; i < 16; ++i) transcript[i] = 0;
  std::vector<int32_t> acc_tile((size_t)HASH_H * HASH_W, 0);
  int nK = K / R_DIM;
  for (int rc = 0; rc < nK; ++rc) {
    int p = rc * R_DIM;
    for (int i = 0; i < HASH_H; ++i)
      for (int j = 0; j < HASH_W; ++j) {
        int32_t acc = 0;
        const int8_t* ar = &An[(size_t)i*K + p];
        const int8_t* br = &Bn[(size_t)j*K + p];
        for (int r = 0; r < R_DIM; ++r) acc += (int32_t)ar[r] * (int32_t)br[r];
        acc_tile[(size_t)i*HASH_W + j] += acc;   // cumulative prefix (helper.rs:25)
      }
    uint32_t x = 0;
    for (size_t t = 0; t < acc_tile.size(); ++t) x ^= (uint32_t)acc_tile[t];
    int idx = rc % 16;
    transcript[idx] = ((transcript[idx] << 13) | (transcript[idx] >> 19)) ^ x;
  }
}

// --- standard base64 ---
static std::string b64_encode(const uint8_t* d, size_t n) {
  static const char* T = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string out; out.reserve((n + 2) / 3 * 4);
  size_t i = 0;
  for (; i + 3 <= n; i += 3) {
    uint32_t v = ((uint32_t)d[i] << 16) | ((uint32_t)d[i+1] << 8) | d[i+2];
    out.push_back(T[(v>>18)&63]); out.push_back(T[(v>>12)&63]);
    out.push_back(T[(v>>6)&63]);  out.push_back(T[v&63]);
  }
  if (n - i == 1) { uint32_t v = (uint32_t)d[i] << 16;
    out.push_back(T[(v>>18)&63]); out.push_back(T[(v>>12)&63]); out.push_back('='); out.push_back('='); }
  else if (n - i == 2) { uint32_t v = ((uint32_t)d[i] << 16) | ((uint32_t)d[i+1] << 8);
    out.push_back(T[(v>>18)&63]); out.push_back(T[(v>>12)&63]); out.push_back(T[(v>>6)&63]); out.push_back('='); }
  return out;
}

// Build the full bincode-serialized PlainProof (== pearl_mining PlainProof.to_base64)
// directly in the binary, from the on-GPU A/B + per-chunk merkle CVs. Lets the daemon
// submit with ZERO host proof-build. Field order is the Rust struct order:
//   PlainProof{ m,n,k,noise_rank, a:MMProof, bt:MMProof }
//   MMProof{ proof:MerkleProof{ leaf_data, leaf_indices, total_leaves, root, siblings }, row_indices }
// bincode default: fixint LE; Vec = u64 len + elems; usize = u64; [u8;N] = N raw bytes.
//
// DEFERRED EMISSION SPLIT (2026-06-11): the build is two halves —
//   1. extract_hit_device_data: everything that touches the GPU (opened leaf
//      bytes + the full per-chunk CV arrays of A and B). MUST run before the
//      next attempt overwrites dA/dB. ~15-25ms.
//   2. plain_proof_b64_from_host: pure host work (two 524K-leaf multileaf
//      merkle proofs + bincode + base64), measured ~47ms per tree on an idle
//      core and far more under CPU-miner contention. serve-mode runs this on a
//      detached worker thread so the GPU starts the next attempt immediately;
//      jobmine keeps the old synchronous behavior via the same two calls.
struct HitHostData {
  std::vector<size_t>  a_leaf_idx, b_leaf_idx;
  std::vector<uint8_t> a_leaf_data, b_leaf_data;   // 1024B per leaf, opened rows/cols only
  std::vector<uint8_t> a_cvs, b_cvs;               // full chunk-CV arrays (total_leaves*32)
  size_t a_total_leaves = 0, b_total_leaves = 0;
  uint8_t job_key[32];
  int a_rows[HASH_H], b_cols[HASH_W];
  int M = 0, N = 0, K = 0, R = 0;
};

static void leaf_indices_for_rows(const int* rows, int n_rows, int K,
                                  std::vector<size_t>& out) {
  const int lpr = K / 1024;                 // leaves per row (K=4096 -> 4)
  std::set<size_t> liset;
  for (int r = 0; r < n_rows; ++r)
    for (int j = 0; j < lpr; ++j) liset.insert((size_t)rows[r]*lpr + j);
  out.assign(liset.begin(), liset.end());
}

// GPU half: opened leaf bytes + full chunk-CV arrays for A and B. Synchronous
// on stream 0; after it returns, the next attempt may freely overwrite dA/dB.
static void extract_hit_device_data(GpuBufs& g, const int* a_rows, const int* b_cols,
                                    int M, int N, int K, int R, HitHostData& h) {
  h.M = M; h.N = N; h.K = K; h.R = R;
  memcpy(h.a_rows, a_rows, sizeof(h.a_rows));
  memcpy(h.b_cols, b_cols, sizeof(h.b_cols));
  pearl_miner::blake3::hash_concat(g.header.data(), g.header.size(),
                                   g.config.data(), g.config.size(), nullptr, h.job_key);
  leaf_indices_for_rows(a_rows, HASH_H, K, h.a_leaf_idx);
  leaf_indices_for_rows(b_cols, HASH_W, K, h.b_leaf_idx);
  auto grab = [&](const int8_t* dMat, size_t mat_rows, const std::vector<size_t>& li,
                  std::vector<uint8_t>& leaf_data, std::vector<uint8_t>& cvs,
                  size_t& total_leaves) {
    total_leaves = mat_rows * (size_t)K / 1024;
    leaf_data.resize(li.size() * 1024);
    for (size_t i = 0; i < li.size(); ++i)
      CUCHK(cudaMemcpy(&leaf_data[i*1024], (const uint8_t*)dMat + li[i]*1024, 1024,
                       cudaMemcpyDeviceToHost));
    pearl_blake3_chunk_cvs_sm89((const uint8_t*)dMat, mat_rows*(size_t)K, h.job_key,
                                g.d_cvs, 0);
    CUCHK(cudaMemcpy(g.h_cvs, g.d_cvs, total_leaves*32, cudaMemcpyDeviceToHost));
    cvs.assign(g.h_cvs, g.h_cvs + total_leaves*32);   // own copy: g.h_cvs is reused
  };
  grab(g.dA, (size_t)M, h.a_leaf_idx, h.a_leaf_data, h.a_cvs, h.a_total_leaves);
  grab(g.dB, (size_t)N, h.b_leaf_idx, h.b_leaf_data, h.b_cvs, h.b_total_leaves);
}

// Host half: pure CPU, no GPU/GpuBufs access — safe on a worker thread.
static std::string plain_proof_b64_from_host(const HitHostData& h) {
  std::vector<uint8_t> buf;
  auto wu64 = [&](uint64_t v) { for (int i = 0; i < 8; ++i) buf.push_back((uint8_t)(v >> (8*i))); };
  auto wbytes = [&](const uint8_t* p, size_t n) { buf.insert(buf.end(), p, p + n); };
  wu64((uint64_t)h.M); wu64((uint64_t)h.N); wu64((uint64_t)h.K); wu64((uint64_t)h.R);
  auto emit_matrix = [&](const std::vector<size_t>& li, const std::vector<uint8_t>& leaf_data,
                         const std::vector<uint8_t>& cvs, size_t total_leaves,
                         const int* rows, int n_rows) {
    uint8_t root[32]; std::vector<uint8_t> siblings;
    pearl_miner::b3tree::multileaf_proof_from_chunk_cvs(h.job_key, (const uint32_t*)cvs.data(),
                                                        total_leaves, li, root, siblings);
    // MerkleProof: leaf_data, leaf_indices, total_leaves, root, siblings.
    // bincode serializes each [u8;1024] (N>32) as a length-prefixed seq: u64(1024)+bytes.
    wu64(li.size());
    for (size_t i = 0; i < li.size(); ++i) { wu64(1024); wbytes(&leaf_data[i*1024], 1024); }
    wu64(li.size()); for (size_t idx : li) wu64((uint64_t)idx);
    wu64(total_leaves);
    wbytes(root, 32);
    wu64(siblings.size() / 32); wbytes(siblings.data(), siblings.size());
    // row_indices
    wu64((uint64_t)n_rows); for (int r = 0; r < n_rows; ++r) wu64((uint64_t)rows[r]);
  };
  emit_matrix(h.a_leaf_idx, h.a_leaf_data, h.a_cvs, h.a_total_leaves, h.a_rows, HASH_H);
  emit_matrix(h.b_leaf_idx, h.b_leaf_data, h.b_cvs, h.b_total_leaves, h.b_cols, HASH_W);
  return b64_encode(buf.data(), buf.size());
}

static std::string emit_plain_proof_b64(GpuBufs& g, const int* a_rows, int n_arows,
                                        const int* b_cols, int n_bcols,
                                        int M, int N, int K, int R) {
  (void)n_arows; (void)n_bcols;   // always HASH_H/HASH_W (the mining_config ticket shape)
  HitHostData h;
  extract_hit_device_data(g, a_rows, b_cols, M, N, K, R, h);
  return plain_proof_b64_from_host(h);
}

// Compose the full serve-mode HIT line into ONE string so it is written with a
// single stdio call — line-atomic vs the main loop's STAT printf even when the
// write happens on the deferred-emission worker thread (stdio locks per call).
static std::string build_serve_hit_line(uint64_t ab_seed, uint32_t ix, uint32_t iy,
                                        const int* a_rows, const int* b_cols,
                                        const uint32_t* T, const uint8_t* gpu_hash,
                                        const std::string& proof_b64,
                                        const std::string& header_hex) {
  char tmp[128];
  std::string out;
  out.reserve(proof_b64.size() + header_hex.size() + 1024);
  snprintf(tmp, sizeof(tmp), "HIT {\"ab_seed\":%llu,\"tile\":[%u,%u],\"a_rows\":[",
           (unsigned long long)ab_seed, ix, iy);
  out += tmp;
  for (int i = 0; i < HASH_H; ++i) { snprintf(tmp, sizeof(tmp), "%s%d", i ? "," : "", a_rows[i]); out += tmp; }
  out += "],\"b_cols\":[";
  for (int i = 0; i < HASH_W; ++i) { snprintf(tmp, sizeof(tmp), "%s%d", i ? "," : "", b_cols[i]); out += tmp; }
  out += "],\"transcript\":[";
  for (int i = 0; i < 16; ++i) { snprintf(tmp, sizeof(tmp), "%s\"%08x\"", i ? "," : "", T[i]); out += tmp; }
  out += "],\"gpu_hash\":\"";
  for (int i = 0; i < 32; ++i) { snprintf(tmp, sizeof(tmp), "%02x", gpu_hash[i]); out += tmp; }
  out += "\",\"proof\":\"" + proof_b64 + "\",\"header\":\"" + header_hex + "\"}\n";
  return out;
}

// Deferred-emission backpressure: >2 proof builds in flight falls back to a
// synchronous build (never queue unbounded host work; hits are ~1/min).
static std::atomic<int> g_hit_emit_inflight{0};

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
  const uint64_t hh = HASH_H, ww = HASH_W;  // 2*64 == old 8*16 == 128
  const uint64_t dpl = (r > 0) ? (uint64_t)(k - k % r) : (uint64_t)k;
  const uint64_t adj = hh * ww * dpl;
  pm_wide_t carry = 0;
  bool overflow = false;
  for (int i = 0; i < 32; ++i) {
    pm_wide_t prod = (pm_wide_t)target_le[i] * adj + carry;
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
  // POOL-VALID persistent miner: FIXED job header, search ab_seed with the REAL
  // on-GPU commitment (real_commit) and emit the full plain_proof per HIT — the
  // exact valid path as jobmine, but the process stays RESIDENT (no ~335ms CUDA
  // re-init per window) and preempts on a new JOB so shares are never stale.
  g.real_commit = (a.real_commit != 0);
  g.config = config;                   // fixed for the session; keys the on-GPU root

  std::vector<uint8_t> cur_header;     // 76B job header (FIXED — never mutated)
  std::string cur_header_hex;          // job header hex to echo in the HIT
  std::string cur_target_hex;          // normalized target hex for duplicate JOB suppression
  std::vector<uint8_t> cur_target_le;  // decoded pow_target for cur job
  bool have_job = false;
  uint64_t ab_seed = 0;                // per-job A/B operand search seed
  uint64_t total_attempts = 0;         // cumulative; emitted as STAT for the daemon's hashrate
  const bool serve_log_jobs = (std::getenv("PEARL_SM89_SERVE_LOG_JOBS") != nullptr);
  int stat_interval = 32;
  if (const char* e = std::getenv("PEARL_SM89_STAT_INTERVAL")) {
    int v = std::atoi(e);
    if (v >= 1 && v <= 1024) stat_interval = v;
  }

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
    if (have_job && hhex == cur_header_hex && thex == cur_target_hex) {
      return false;
    }
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
    cur_target_hex = std::move(thex);
    // Echo the ORIGINAL job header (with the pool's [72,76) suffix), NOT the
    // per-nonce mutated working header — so the driver can map a HIT back to the
    // exact job by header. The winning nonce is reported separately.
    cur_header_hex = hhex;
    return true;
  };

  // Drain all currently-available stdin into inbuf; split complete lines; apply
  // the NEWEST valid JOB line (re-derive seeds + reset nonce). Sets eof on EOF.
  auto drain_stdin = [&]() {
    for (;;) {
      int ready = pm_stdin_ready();  // non-blocking: >0 data, 0 none, <0 EOF
      if (ready < 0) { eof = true; break; }
      if (ready == 0) break;
      char buf[8192];
      ssize_t_compat n = pm_stdin_read(buf, sizeof(buf));
      if (n == 0) { eof = true; break; }
      if (n < 0) break;
      inbuf.append(buf, (size_t)n);
      if ((size_t)n < sizeof(buf)) {
        // likely drained the pipe for now; still loop once more
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
      std::string prev_header_hex = cur_header_hex;  // before apply overwrites it
      if (apply_job_line(newest)) {
        // Upload the per-job target; the on-GPU commitment is keyed on g.header/
        // g.config (set here), and the A/B operands vary per ab_seed.
        CUCHK(cudaMemcpy(g.dTarget, cur_target_le.data(), 32, cudaMemcpyHostToDevice));
        g.header = cur_header;   // FIXED job header for run_attempt_mine's job_key
        if (have_job && cur_header_hex == prev_header_hex) {
          // SAME-HEADER re-notify (daemon reconnect, pool re-send, or a
          // target-only vardiff retarget): keep the ab_seed search position.
          // The seed->A/B schedule is deterministic, so resetting to 0 would
          // re-mine already-covered seed space and re-find the IDENTICAL hits
          // -> pool code-22 duplicate rejects (the ~0.9% dup leak).
          if (serve_log_jobs) {
            fprintf(stderr, "serve: same-header JOB target_msb=%02x ab_seed kept at %llu\n",
                    cur_target_le[31], (unsigned long long)ab_seed);
          }
        } else {
          // Fresh search space for the new job. Start at a RANDOM 64-bit seed
          // (not 0): the seed->share map is uniform so any start point has
          // equal share probability, and a random start makes the residual
          // duplicate windows (binary restart re-feeding the same job, A->B->A
          // job bounce, any cross-process same-header overlap) statistically
          // impossible — a restart from 0 deterministically re-found the same
          // early-seed hits (code-22 duplicate rejects).
          uint64_t t = (uint64_t)time(nullptr) ^ ((uint64_t)getpid() << 32) ^ ab_seed;
          t += 0x9E3779B97F4A7C15ULL;
          t = (t ^ (t >> 30)) * 0xBF58476D1CE4E5B9ULL;
          t = (t ^ (t >> 27)) * 0x94D049BB133111EBULL;
          ab_seed = t ^ (t >> 31);
          if (serve_log_jobs) {
            fprintf(stderr, "serve: new JOB header[0..4]=%02x%02x%02x%02x target_msb=%02x ab_seed=%llu (random start)\n",
                    cur_header[0], cur_header[1], cur_header[2], cur_header[3],
                    cur_target_le[31], (unsigned long long)ab_seed);
          }
        }
        have_job = true;
      }
    }
  };

  fprintf(stderr, "serve: ready (M=%d N=%d K=%d R=%d) — awaiting JOB lines on stdin\n",
          M, N, K, a.r);

  while (true) {
    drain_stdin();
    if (eof && inbuf.empty()) {
      // Drain any in-flight deferred HIT emission before exiting so the final
      // line is never torn by exit-time stream cleanup racing the worker.
      while (g_hit_emit_inflight.load() > 0) pm_idle_wait(50);
      fprintf(stderr, "serve: stdin EOF — exiting\n");
      break;
    }
    if (!have_job) {
      // No job yet: brief wait so we don't spin.
      pm_idle_wait(200);
      continue;
    }

    // One attempt on the FIXED job header at the current ab_seed. real_commit=1
    // makes run_attempt_mine compute the verifier's commitment ON-GPU from the
    // actual (seed-derived) A/B merkle roots — the pool-valid path (s unused when
    // real_commit; zero-init so a real_commit=0 misconfig can't mine on garbage
    // seeds). drain_stdin ran this iteration, so a new JOB preempts within
    // one attempt (~0.5s) -> shares are for the ~current job, never stale.
    // prep_next/ab_seed+1: while this attempt's PoW kernel runs, the NEXT
    // attempt's operands + commitment seeds are derived on the prep stream
    // (the overlap lever) — wasted only when a new JOB preempts (~1/300).
    Seeds s{};
    uint32_t ix = 0, iy = 0;
    int hit = run_attempt(g, s, M, N, K, ab_seed, &ix, &iy,
                          /*prep_next=*/true, /*next_seed=*/ab_seed + 1);
    if (hit) {
      const Seeds& hs = g.real_commit ? g.cur_seeds : s;
      int a_rows[HASH_H], b_cols[HASH_W]; uint32_t T[16]; uint8_t gpu_hash[32];
      read_transcript_and_indices(g, hs, K, ix, iy, a_rows, b_cols, T);
      transcript_hash(T, hs.a_noise_seed, gpu_hash);
      // The HIT line echoes the job header so the daemon maps it to a job_id
      // (an in-flight HIT for a just-superseded job still carries that job's
      // header).
      if (g.real_commit) {
        // DEFERRED EMISSION (2026-06-11): pull the proof's device data NOW
        // (the next attempt overwrites dA/dB and the noised operands), then
        // run the pure-host half (two 524K-leaf multileaf merkle proofs +
        // bincode + base64 + the stdout write, ~100ms+, worse under CPU-miner
        // contention) on a detached worker thread so the GPU starts the next
        // attempt immediately instead of idling behind host proof-build.
        auto h = std::make_shared<HitHostData>();
        extract_hit_device_data(g, a_rows, b_cols, M, N, K, a.r, *h);
        std::array<uint32_t,16> Tc; memcpy(Tc.data(), T, sizeof(T));
        std::array<uint8_t,32> gh; memcpy(gh.data(), gpu_hash, sizeof(gpu_hash));
        std::string hdr_hex = cur_header_hex;
        uint64_t hseed = ab_seed; uint32_t hix = ix, hiy = iy;
        auto emit_fn = [h, Tc, gh, hdr_hex, hseed, hix, hiy]() {
          std::string proof = plain_proof_b64_from_host(*h);
          std::string line = build_serve_hit_line(hseed, hix, hiy, h->a_rows, h->b_cols,
                                                  Tc.data(), gh.data(), proof, hdr_hex);
          fwrite(line.data(), 1, line.size(), stdout);
          fflush(stdout);
          g_hit_emit_inflight.fetch_sub(1);
        };
        bool spawned = false;
        if (g_hit_emit_inflight.fetch_add(1) < 2) {
          try { std::thread(emit_fn).detach(); spawned = true; }
          catch (...) { /* thread creation failed -> synchronous fallback */ }
        }
        if (!spawned) emit_fn();
      } else {
        // smoke path (no proof) — cheap, emit inline.
        std::string line = build_serve_hit_line(ab_seed, ix, iy, a_rows, b_cols,
                                                T, gpu_hash, std::string(), cur_header_hex);
        fwrite(line.data(), 1, line.size(), stdout);
        fflush(stdout);
      }
      // KEEP mining — persistent process, do not exit on hit.
    }
    ++ab_seed; ++total_attempts;
    // Periodic cumulative attempt count -> the daemon computes the attempt-rate hashrate.
    if ((total_attempts % (uint64_t)stat_interval) == 0) {
      printf("STAT %llu\n", (unsigned long long)total_attempts);
      fflush(stdout);
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
    const uint64_t hh = HASH_H, ww = HASH_W;  // 2*64 == old 8*16 == 128
    const uint64_t dpl = (a.r > 0) ? (uint64_t)(a.k - a.k % a.r) : (uint64_t)a.k;
    const uint64_t adj = hh * ww * dpl;  // difficulty_adjustment_factor
    // 256-bit (target_le, little-endian) *= adj, saturating at 2^256-1.
    pm_wide_t carry = 0;
    bool overflow = false;
    for (int i = 0; i < 32; ++i) {
      pm_wide_t prod = (pm_wide_t)target_le[i] * adj + carry;
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
  // PEARL_MINER_BLOCKING_SYNC=1: yield the CPU during device syncs instead of
  // the default spin-wait (~93% of a core per GPU process at the ~0.45s attempt
  // cadence). Keep the lower-latency spin default on big-CPU minis; enable on
  // the 2C/4T Pentium 6-GPU rigs (rig04/rig05) where six spinning serve
  // processes saturate the CPU. On CUDA 12 cudaSetDevice already initializes
  // the primary context; cudaSetDeviceFlags is documented to OVERWRITE the
  // flags of an initialized device, so this ordering (after cudaSetDevice, so
  // the flags land on a.dev and not device 0) is correct and cannot fail with
  // cudaErrorSetOnActiveProcess on 12.x.
  if (const char* bsync = std::getenv("PEARL_MINER_BLOCKING_SYNC");
      bsync && bsync[0] && bsync[0] != '0') {
    CUCHK(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));
    fprintf(stderr, "pearl_miner_sm89: blocking-sync host waits enabled\n");
  }
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
    // real_commit=1: run_attempt_mine computes the verifier's commitment per
    // ab_seed from the ACTUAL A/B merkle roots (ON-GPU, fast). Without this the
    // search uses the header-only fallback seed (or externally-passed aroot/broot)
    // -> wrong commitment -> the pool rejects every share.
    g.real_commit = (a.real_commit != 0);
    g.header = header; g.config = config;  // run_attempt_mine keys the on-GPU root
                                           // (job_key=blake3(header||config)) on these.
    // Fallback / external-root seeds (used only when real_commit==0).
    Seeds s;
    derive_seeds_for_header(header, config, aroot, broot, s);
    uint64_t start = a.nonce_start, count = a.nonce_count ? a.nonce_count : 1;
    for (uint64_t i = 0; i < count; ++i) {
      uint64_t ab_seed = start + i;  // search var: A/B operand seed (header fixed)
      uint32_t ix = 0, iy = 0;
      // prep_next mirrors serve-mode (sequential seeds), so jobmine exercises —
      // and the deterministic replay test validates — the pipelined prep path.
      int hit = run_attempt(g, s, a.m, a.n, a.k, ab_seed, &ix, &iy,
                            /*prep_next=*/(i + 1 < count), /*next_seed=*/ab_seed + 1);
      if (hit) {
        // Use the seeds the kernel ACTUALLY searched with (real per-ab_seed
        // commitment when real_commit, else the fallback s) so the read-back
        // transcript + gpu_hash match what was gated.
        const Seeds& hs = g.real_commit ? g.cur_seeds : s;
        int a_rows[HASH_H], b_cols[HASH_W]; uint32_t T[16]; uint8_t gpu_hash[32];
        read_transcript_and_indices(g, hs, a.k, ix, iy, a_rows, b_cols, T);
        transcript_hash(T, hs.a_noise_seed, gpu_hash);
        // Build the FULL submittable plain_proof in-binary (only when real_commit,
        // where the on-GPU A/B + commitment match the verifier). The daemon submits
        // this directly — no host proof build. Empty otherwise (diag/Python path).
        std::string proof_b64 = g.real_commit
            ? emit_plain_proof_b64(g, a_rows, HASH_H, b_cols, HASH_W, a.m, a.n, a.k, a.r)
            : std::string();
        printf("HIT {\"ab_seed\":%llu,\"nonce\":%llu,\"tile\":[%u,%u],\"a_rows\":[",
               (unsigned long long)ab_seed, (unsigned long long)ab_seed, ix, iy);
        for (int j = 0; j < HASH_H; ++j)  printf("%s%d", j ? "," : "", a_rows[j]);
        printf("],\"b_cols\":[");
        for (int j = 0; j < HASH_W; ++j) printf("%s%d", j ? "," : "", b_cols[j]);
        printf("],\"transcript\":[");
        for (int j = 0; j < 16; ++j) printf("%s\"%08x\"", j ? "," : "", T[j]);
        printf("],\"gpu_hash\":\"");
        for (int j = 0; j < 32; ++j) printf("%02x", gpu_hash[j]);
        printf("\",\"proof\":\"%s\"}\n", proof_b64.c_str());
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
    // bench real_commit=1: measure the PRODUCTION serve path (per-attempt
    // on-GPU merkle commit + the prep-ahead overlap). Default real_commit=0
    // keeps the historical bench semantics (no commit, no prep).
    g.real_commit = (a.real_commit != 0);
    g.header = header; g.config = config;
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
      uint64_t next_ab = (nonce + 1) * 0x100000001B3ULL + 0xCBF29CE484222325ULL;
      uint32_t ix=0, iy=0;
      hits += run_attempt(g, s, a.m, a.n, a.k, ab_seed, &ix, &iy,
                          /*prep_next=*/(i + 1 < bcount), next_ab);
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
    for (int i=0;i<HASH_H;++i) printf("%s%d", i?",":"", a_rows[i]);
    printf("],\"b_cols\":[");
    for (int i=0;i<HASH_W;++i) printf("%s%d", i?",":"", b_cols[i]);
    printf("],\"transcript\":[");
    for (int i=0;i<16;++i) printf("%s\"%08x\"", i?",":"", T[i]);
    printf("],\"gpu_hash\":\"");
    for (int i=0;i<32;++i) printf("%02x", gpu_hash[i]);
    printf("\"}\n");
  };

  // Mining loop: nonce -> seeds -> attempt.
  // Real-commitment mining keeps the job header FIXED (mutating header[72:76]
  // would make the proof header != the job header -> pool reject) and searches by
  // varying ab_seed; run_attempt_mine derives the verifier's real key from each A/B.
  g.real_commit = (a.real_commit != 0);
  g.header = header; g.config = config;
  uint64_t start = a.nonce_start, count = a.nonce_count;
  if (a.mode == "verify" && count == 0) count = 1;
  for (uint64_t i = 0; i < count; ++i) {
    uint64_t nonce = start + i;
    // Smoke/self-consistent mode varies the header nonce; real-commitment mode
    // keeps the header fixed (search diversity comes from ab_seed -> A/B).
    if (!g.real_commit) {
      header[72] = (uint8_t)(nonce);
      header[73] = (uint8_t)(nonce >> 8);
      header[74] = (uint8_t)(nonce >> 16);
      header[75] = (uint8_t)(nonce >> 24);
    }

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
      int a_rows[HASH_H], b_cols[HASH_W]; uint32_t T[16]; uint8_t gpu_hash[32];
      // Use the seeds the kernel ACTUALLY searched with (real per-nonce commitment
      // when real_commit, else the fallback s) so gpu_hash/opened indices match.
      const Seeds& hs = g.real_commit ? g.cur_seeds : s;
      read_transcript_and_indices(g, hs, a.k, ix, iy, a_rows, b_cols, T);
      transcript_hash(T, hs.a_noise_seed, gpu_hash);

      // VERIFY cross-check: independently recompute the transcript from the RAW
      // (un-noised) seed-derived A/B strips using the HOST noise reference, and
      // compare to the transcript read from the GPU-noised operands. A match
      // proves the GPU noisingA/B + integer GEMM + XOR/rotl13 reproduce the
      // arch-independent reference bit-exactly. This is the decisive local
      // hardware self-check; on real sm_89 the user additionally diffs T against
      // a known oracle transcript.
      if (a.mode == "verify") {
        std::vector<int8_t> rawA((size_t)HASH_H * a.k), rawB((size_t)HASH_W * a.k);
        std::vector<int8_t> fullA((size_t)a.m * a.k), fullB((size_t)a.n * a.k);
        fill_AB(fullA.data(), fullA.size(), ab_seed);
        fill_AB(fullB.data(), fullB.size(), ab_seed ^ 0xD1B54A32D192ED03ULL);
        for (int r = 0; r < HASH_H; ++r)
          memcpy(&rawA[(size_t)r*a.k], &fullA[(size_t)a_rows[r]*a.k], a.k);
        for (int c = 0; c < HASH_W; ++c)
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
          pearl_miner::noise_sparse(a.k, a.r, SA, hs.a_noise_seed, EAR);
          pearl_miner::noise_sparse(a.k, a.r, SB, hs.b_noise_seed, EBL);
          pearl_miner::noise_dense_row(a_rows[0], a.r, SA, hs.a_noise_seed, eal);
          pearl_miner::noise_dense_row(b_cols[0], a.r, SB, hs.b_noise_seed, ebr);
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
        transcript_from_strips(rawA, rawB, a_rows, b_cols, HASH_H, HASH_W, a.k, a.r,
                               hs.a_noise_seed, hs.b_noise_seed, Tref);
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
