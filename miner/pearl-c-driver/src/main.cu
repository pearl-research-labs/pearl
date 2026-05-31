// SPDX-License-Identifier: see LICENSE
//
// pearl-c-driver SKELETON — C++ hot-loop benchmark for the Pearl W19R miner.
//
// Goal: measure pure C++ iteration rate of the production hot-loop kernel
// sequence, to quantify the Python-orchestration ceiling (~25 effective TOPS
// at 725 att/s from _w19r_full_orchestrator_v2.py).
//
// What this binary does (bench-only, NO pool, NO stratum):
//   * Allocates device buffers matching the production shape:
//       M=N=2048, K=4096, R=128, BATCH=256.
//   * Drives the C-linkage trampoline `pearl_gemm_sm89_w19r_64x64x128_R128_prod`
//     directly (the same symbol that pearl_gemm_w19_cuda.so exports and that
//     `pg.gemm_sm89_w19r_64x64_prod` dispatches to from Python).
//   * Inner loop = BATCH×GEMM launches per "attempt"; this is exactly what the
//     v2 orchestrator does once the noise has been generated.
//   * pow_target = all-zeros so the kernel never signals a hit (matches the
//     bench convention used by csrc/gemm/_bench_w19r_multinonce.cu).
//
// SKELETON SCOPE INTENTIONALLY EXCLUDES:
//   * noise_gen_blake3_persistent (currently 0.27 ms / attempt — amortized
//     across 256 GEMMs — not a hot-path bottleneck).
//   * extract_sparse_indices + apply_sparse_noise (also one-shot per attempt
//     for n_sp; the per-nonce apply_sparse runs 256× but is ~10% of GEMM cost).
//   * Stratum + share submission (placeholder for next phase).
//
// The question this binary answers: "If the Python driver achieves 725 att/s
// (= 185,600 GEMM launches/sec) at production shape, how fast can a pure
// C++ loop do the same launches?"
//
// Compile via Makefile:  make -j
// Run:                   ./pearl_c_driver --bench --batch 256 --iters 200
//                        ./pearl_c_driver --bench --batch 256 --iters 200 --shape prod

#include <cuda_runtime.h>
#include <cutlass/numeric_types.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// ----------------------------------------------------------------------------
// extern "C" entry points from pearl_gemm_w19_cuda.so.
// Signatures copied verbatim from
// /home/pearl-deploy/pearl-gemm/csrc/gemm/pearl_gemm_w19_pybind.cu (~L267).
// ----------------------------------------------------------------------------

extern "C" void pearl_gemm_sm89_w19r_64x64x128_R128_prod(
    int8_t const* A, int64_t lda,
    int8_t const* B, int64_t ldb,
    cutlass::bfloat16_t* C, int64_t ldc,
    float const* A_scales,
    float const* B_scales,
    cutlass::half_t const* EAL,
    cutlass::half_t const* EBR,
    cutlass::half_t const* AxEBL,
    cutlass::half_t const* EARxBpEB,
    uint32_t const* pow_target,
    uint32_t const* pow_key,
    void* host_signal_sync,
    void* host_signal_header_pinned,
    uint64_t* inner_hash_counter,
    int M, int N, int K,
    cudaStream_t stream);

// ----------------------------------------------------------------------------
// CUDA error check.
// ----------------------------------------------------------------------------
#define CUCHK(x)                                                              \
  do {                                                                        \
    cudaError_t _e = (x);                                                     \
    if (_e != cudaSuccess) {                                                  \
      std::fprintf(stderr, "CUDA %s:%d %s\n", __FILE__, __LINE__,             \
                   cudaGetErrorString(_e));                                   \
      std::exit(1);                                                           \
    }                                                                         \
  } while (0)

// ----------------------------------------------------------------------------
// CLI parsing.
// ----------------------------------------------------------------------------
struct Cli {
  int  device     = 0;
  int  M          = 2048;
  int  N          = 2048;
  int  K          = 4096;
  int  batch      = 256;
  int  iters      = 200;
  int  warmup     = 5;
  bool graph_mode = false;   // capture all `batch` GEMM launches into one graph
  bool bench      = true;
};

static Cli parse_cli(int argc, char** argv) {
  Cli c;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "missing arg after %s\n", a.c_str());
        std::exit(2);
      }
      return argv[++i];
    };
    if      (a == "--device")    c.device = std::stoi(next());
    else if (a == "--M")         c.M = std::stoi(next());
    else if (a == "--N")         c.N = std::stoi(next());
    else if (a == "--K")         c.K = std::stoi(next());
    else if (a == "--batch")     c.batch = std::stoi(next());
    else if (a == "--iters")     c.iters = std::stoi(next());
    else if (a == "--warmup")    c.warmup = std::stoi(next());
    else if (a == "--graph")     c.graph_mode = true;
    else if (a == "--bench")     c.bench = true;
    else if (a == "--shape") {
      std::string s = next();
      if (s == "prod") { c.M = 2048; c.N = 2048; c.K = 4096; }
      else if (s == "small") { c.M = 256; c.N = 256; c.K = 512; }
      else { std::fprintf(stderr, "unknown shape %s\n", s.c_str()); std::exit(2); }
    }
    else if (a == "--help" || a == "-h") {
      std::printf(
        "Usage: %s [--device N] [--M M] [--N N] [--K K] [--batch B] [--iters I]\n"
        "          [--warmup W] [--graph] [--shape prod|small]\n", argv[0]);
      std::exit(0);
    }
    else {
      std::fprintf(stderr, "unknown arg %s\n", a.c_str());
      std::exit(2);
    }
  }
  return c;
}

// ----------------------------------------------------------------------------
// Device buffers.
// ----------------------------------------------------------------------------
struct DevBuf {
  int8_t*               A          = nullptr;   // (M, K) int8
  int8_t*               B          = nullptr;   // (N, K) int8
  cutlass::bfloat16_t*  C          = nullptr;   // (M, N) bf16
  float*                A_scales   = nullptr;   // (M,) float32
  float*                B_scales   = nullptr;   // (N,) float32
  cutlass::half_t*      EAL        = nullptr;   // (M, R) half
  cutlass::half_t*      EBR        = nullptr;   // (N, R) half
  cutlass::half_t*      AxEBL      = nullptr;   // (M, R) half
  cutlass::half_t*      EARxBpEB   = nullptr;   // (N, R) half
  uint32_t*             pow_target = nullptr;   // (8,) uint32
  uint32_t*             pow_key    = nullptr;   // (8,) uint32
  uint64_t*             hash_ctr   = nullptr;   // (1,) uint64
};

static constexpr int R_DIM = 128;

static DevBuf alloc_buffers(const Cli& c) {
  DevBuf b;
  const size_t bytes_A   = size_t(c.M) * c.K;
  const size_t bytes_B   = size_t(c.N) * c.K;
  const size_t bytes_C   = size_t(c.M) * c.N * 2;            // bf16
  const size_t bytes_EAL = size_t(c.M) * R_DIM * 2;          // half
  const size_t bytes_EBR = size_t(c.N) * R_DIM * 2;          // half

  CUCHK(cudaMalloc(&b.A,          bytes_A));
  CUCHK(cudaMalloc(&b.B,          bytes_B));
  CUCHK(cudaMalloc(&b.C,          bytes_C));
  CUCHK(cudaMalloc(&b.A_scales,   c.M * 4));
  CUCHK(cudaMalloc(&b.B_scales,   c.N * 4));
  CUCHK(cudaMalloc(&b.EAL,        bytes_EAL));
  CUCHK(cudaMalloc(&b.EBR,        bytes_EBR));
  CUCHK(cudaMalloc(&b.AxEBL,      bytes_EAL));
  CUCHK(cudaMalloc(&b.EARxBpEB,   bytes_EBR));
  CUCHK(cudaMalloc(&b.pow_target, 8 * 4));
  CUCHK(cudaMalloc(&b.pow_key,    8 * 4));
  CUCHK(cudaMalloc(&b.hash_ctr,   8));

  // Zero everything. pow_target=0 ensures the kernel never signals a hit,
  // so host_signal_sync / host_signal_header_pinned can stay nullptr
  // (matches the bench convention in _bench_w19r_multinonce.cu).
  CUCHK(cudaMemset(b.A,          0, bytes_A));
  CUCHK(cudaMemset(b.B,          0, bytes_B));
  CUCHK(cudaMemset(b.C,          0, bytes_C));
  CUCHK(cudaMemset(b.A_scales,   0, c.M * 4));
  CUCHK(cudaMemset(b.B_scales,   0, c.N * 4));
  CUCHK(cudaMemset(b.EAL,        0, bytes_EAL));
  CUCHK(cudaMemset(b.EBR,        0, bytes_EBR));
  CUCHK(cudaMemset(b.AxEBL,      0, bytes_EAL));
  CUCHK(cudaMemset(b.EARxBpEB,   0, bytes_EBR));
  CUCHK(cudaMemset(b.pow_target, 0, 8 * 4));
  CUCHK(cudaMemset(b.pow_key,    0, 8 * 4));
  CUCHK(cudaMemset(b.hash_ctr,   0, 8));

  return b;
}

static void free_buffers(DevBuf& b) {
  cudaFree(b.A);          cudaFree(b.B);          cudaFree(b.C);
  cudaFree(b.A_scales);   cudaFree(b.B_scales);
  cudaFree(b.EAL);        cudaFree(b.EBR);
  cudaFree(b.AxEBL);      cudaFree(b.EARxBpEB);
  cudaFree(b.pow_target); cudaFree(b.pow_key);
  cudaFree(b.hash_ctr);
  b = {};
}

// Drive ONE GEMM call (one "nonce") with all production-shaped pointers.
static inline void launch_gemm(const DevBuf& b, int M, int N, int K,
                               cudaStream_t s) {
  pearl_gemm_sm89_w19r_64x64x128_R128_prod(
      b.A,        /*lda=*/K,
      b.B,        /*ldb=*/K,
      b.C,        /*ldc=*/N,
      b.A_scales, b.B_scales,
      b.EAL,      b.EBR,
      b.AxEBL,    b.EARxBpEB,
      b.pow_target, b.pow_key,
      /*host_signal_sync=*/nullptr,
      /*host_signal_header_pinned=*/nullptr,
      b.hash_ctr,
      M, N, K, s);
}

// ----------------------------------------------------------------------------
// Bench mode.
// ----------------------------------------------------------------------------
static int bench_main(const Cli& c) {
  std::printf("pearl-c-driver SKELETON bench\n");
  std::printf("  device=%d M=%d N=%d K=%d R=%d batch=%d iters=%d warmup=%d graph=%s\n",
              c.device, c.M, c.N, c.K, R_DIM, c.batch, c.iters, c.warmup,
              c.graph_mode ? "yes" : "no");

  CUCHK(cudaSetDevice(c.device));
  cudaDeviceProp prop{};
  CUCHK(cudaGetDeviceProperties(&prop, c.device));
  std::printf("  GPU: %s (sm_%d%d) %d SMs\n", prop.name, prop.major, prop.minor,
              prop.multiProcessorCount);

  DevBuf B = alloc_buffers(c);

  cudaStream_t stream;
  CUCHK(cudaStreamCreate(&stream));

  // ---------------------------------------------------------------------------
  // Define the "one attempt" closure (batch GEMM launches on `stream`).
  // ---------------------------------------------------------------------------
  auto run_attempt = [&]() {
    for (int b = 0; b < c.batch; ++b) {
      launch_gemm(B, c.M, c.N, c.K, stream);
    }
  };

  // Optionally capture into a CUDA graph for minimum launch overhead.
  cudaGraph_t      graph     = nullptr;
  cudaGraphExec_t  graph_exec = nullptr;
  if (c.graph_mode) {
    std::printf("  capturing CUDA graph (%d GEMMs)...\n", c.batch);
    CUCHK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    run_attempt();
    CUCHK(cudaStreamEndCapture(stream, &graph));
    CUCHK(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
    std::printf("  graph captured + instantiated\n");
  }

  // ---------------------------------------------------------------------------
  // Warmup.
  // ---------------------------------------------------------------------------
  std::printf("  warmup %d attempts...\n", c.warmup);
  for (int w = 0; w < c.warmup; ++w) {
    if (c.graph_mode) CUCHK(cudaGraphLaunch(graph_exec, stream));
    else              run_attempt();
  }
  CUCHK(cudaStreamSynchronize(stream));

  // ---------------------------------------------------------------------------
  // Bench loop.
  // ---------------------------------------------------------------------------
  std::printf("  benching %d attempts...\n", c.iters);
  std::vector<double> per_attempt_ms;
  per_attempt_ms.reserve(c.iters);

  cudaEvent_t e_start, e_end;
  CUCHK(cudaEventCreate(&e_start));
  CUCHK(cudaEventCreate(&e_end));

  auto wall_start = std::chrono::steady_clock::now();
  CUCHK(cudaEventRecord(e_start, stream));

  for (int it = 0; it < c.iters; ++it) {
    cudaEvent_t t0, t1;
    CUCHK(cudaEventCreate(&t0));
    CUCHK(cudaEventCreate(&t1));
    CUCHK(cudaEventRecord(t0, stream));
    if (c.graph_mode) CUCHK(cudaGraphLaunch(graph_exec, stream));
    else              run_attempt();
    CUCHK(cudaEventRecord(t1, stream));
    CUCHK(cudaEventSynchronize(t1));
    float ms = 0;
    CUCHK(cudaEventElapsedTime(&ms, t0, t1));
    per_attempt_ms.push_back(double(ms));
    CUCHK(cudaEventDestroy(t0));
    CUCHK(cudaEventDestroy(t1));
  }

  CUCHK(cudaEventRecord(e_end, stream));
  CUCHK(cudaStreamSynchronize(stream));
  auto wall_end = std::chrono::steady_clock::now();

  float total_ms = 0;
  CUCHK(cudaEventElapsedTime(&total_ms, e_start, e_end));
  double wall_s =
      std::chrono::duration<double>(wall_end - wall_start).count();

  // ---------------------------------------------------------------------------
  // Report.
  // ---------------------------------------------------------------------------
  std::sort(per_attempt_ms.begin(), per_attempt_ms.end());
  double med = per_attempt_ms[per_attempt_ms.size() / 2];
  double p10 = per_attempt_ms[per_attempt_ms.size() /  10];
  double p90 = per_attempt_ms[(per_attempt_ms.size() * 9) / 10];
  double mn  = per_attempt_ms.front();
  double mx  = per_attempt_ms.back();
  double sum = 0;
  for (double v : per_attempt_ms) sum += v;
  double avg = sum / per_attempt_ms.size();

  // attempts/sec from GPU-event wall time = pure kernel throughput.
  double att_per_sec_gpu = double(c.iters) / (total_ms / 1000.0);
  // attempts/sec from steady_clock = end-to-end including host loop overhead.
  double att_per_sec_wall = double(c.iters) / wall_s;

  // Effective TOPS at this shape:
  //   2 * M * N * K ops per GEMM * batch GEMMs per attempt * attempts/sec
  double ops_per_attempt = 2.0 * c.M * c.N * c.K * double(c.batch);
  double tops_gpu  = att_per_sec_gpu  * ops_per_attempt / 1e12;
  double tops_wall = att_per_sec_wall * ops_per_attempt / 1e12;

  // "Nonce" rate = per-GEMM-launch rate; this is what the Python framework
  // reports as "att/s" (the v2 orchestrator's `nonces_per_s = attempts_per_s
  // * batch`). The Python driver achieves 725 nonces/sec at 24.91 TOPS.
  double nonces_per_sec_gpu  = att_per_sec_gpu  * c.batch;
  double nonces_per_sec_wall = att_per_sec_wall * c.batch;

  std::printf("\n");
  std::printf("  per-attempt ms : min=%.3f p10=%.3f med=%.3f avg=%.3f p90=%.3f max=%.3f\n",
              mn, p10, med, avg, p90, mx);
  std::printf("  total GPU ms     : %.2f\n", total_ms);
  std::printf("  wall sec         : %.3f\n", wall_s);
  std::printf("  attempts/s (gpu) : %.2f\n", att_per_sec_gpu);
  std::printf("  attempts/s (wall): %.2f\n", att_per_sec_wall);
  std::printf("  nonces/s   (gpu) : %.0f\n", nonces_per_sec_gpu);
  std::printf("  nonces/s   (wall): %.0f\n", nonces_per_sec_wall);
  std::printf("  TOPS       (gpu) : %.2f\n", tops_gpu);
  std::printf("  TOPS       (wall): %.2f\n", tops_wall);
  std::printf("\n");
  std::printf("  baseline Python driver: 725 nonces/s (24.91 TOPS effective)\n");
  std::printf("  C-skeleton speedup    : %.2fx nonces/s vs python\n",
              nonces_per_sec_wall / 725.0);

  if (c.graph_mode) {
    CUCHK(cudaGraphExecDestroy(graph_exec));
    CUCHK(cudaGraphDestroy(graph));
  }
  CUCHK(cudaEventDestroy(e_start));
  CUCHK(cudaEventDestroy(e_end));
  CUCHK(cudaStreamDestroy(stream));
  free_buffers(B);

  return 0;
}

int main(int argc, char** argv) {
  Cli c = parse_cli(argc, argv);
  if (c.bench) return bench_main(c);
  std::fprintf(stderr, "no mode selected — pass --bench\n");
  return 2;
}
