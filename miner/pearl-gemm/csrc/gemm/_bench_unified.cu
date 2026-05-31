// Unified Pearl-GEMM sm_89 throughput bench.
//
// Single C++ standalone for the cases where Python/pybind overhead might
// confuse perf measurement. Same shapes, same configs, same TOPS formula
// as tools/bench_pearl_gemm.py. Outputs CSV-compatible rows (`csv_row=...`
// lines so the Python driver can parse them).
//
// Configs supported:
//   --rank      64 | 128
//   --pow       hard | disabled       (NEVER 'easy' — see project_pearl_perf_postmortem)
//   --streams   N (CUDA streams running the chain concurrently)
//   --warmup    N
//   --repeats   N
//   --shapes    "1024,2048,4096,8192,16384,4096x4096x8192,16384x4096x4096"
//   --out       /path/to/file.csv     (optional)
//
// Build via tools/build_bench_unified.sh (calls nvcc with the standard flags).
// Run with `./_bench_unified --rank 64 --pow hard --streams 1 --repeats 30`.
//
// MAC accounting (per pipeline iter):
//   noisingA + noisingB + main gemm w/ denoise (+ optional PoW)
//   Total MACs = 2*K*R*(M+N) + M*N*(K + 2*R)
//   Main MACs  = 2*M*N*K  (what alpha-miner's tmac_s reports against)

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cutlass/numeric_types.h>

#include "host_signal_header.hpp"

// ------------------------------------------------------------------------
// Forward decls of every sm_89 C symbol we touch.
// Keep names in sync with csrc/gemm/pearl_gemm_sm89_*_inst.cu.
// ------------------------------------------------------------------------
extern "C" void pearl_noisingA_sm89_64x64x64_R64_int32(
    int8_t const*, int8_t const*, int8_t const*, int8_t const*,
    int8_t*, int32_t*, int, int, cudaStream_t);
namespace pearl { namespace sm89 {
extern "C" void pearl_noisingB_sm89_64x64x64_R64_int32(
    int8_t const*, int8_t const*, int8_t const*, int8_t const*,
    int8_t*, int32_t*, int, int, cudaStream_t);
extern "C" void pearl_gemm_sm89_denoise_128x128x64_R64(
    int8_t const*, int64_t, int8_t const*, int64_t,
    cutlass::bfloat16_t*, int64_t, float const*, float const*,
    cutlass::half_t const*, cutlass::half_t const*,
    cutlass::half_t const*, cutlass::half_t const*,
    int, int, int, cudaStream_t);
extern "C" void pearl_gemm_sm89_pow_128x128x64_R64(
    int8_t const*, int64_t, int8_t const*, int64_t,
    cutlass::bfloat16_t*, int64_t, float const*, float const*,
    cutlass::half_t const*, cutlass::half_t const*,
    cutlass::half_t const*, cutlass::half_t const*,
    uint32_t const*, uint32_t const*,
    void*, void*,
    uint64_t*,
    int, int, int, cudaStream_t);
}}  // namespace pearl::sm89

#define CUCHK(x) do { auto _e = (x); if (_e != cudaSuccess) { \
    fprintf(stderr, "CUDA %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
    std::exit(1); } } while (0)

// ------------------------------------------------------------------------
// One per-shape per-stream tensor set
// ------------------------------------------------------------------------
struct TensorSet {
  int M, N, K, R;
  int8_t  *dA = nullptr, *dB = nullptr;
  int8_t  *dEAL = nullptr, *dEBR = nullptr;
  int8_t  *dEAR_R = nullptr, *dEBL_R = nullptr;
  int8_t  *dEAR_K = nullptr, *dEBL_K = nullptr;
  int8_t  *dApEA = nullptr, *dBpEB = nullptr;
  int32_t *dAxEBL_i32 = nullptr, *dEARxBpEB_i32 = nullptr;
  cutlass::half_t *dEAL_fp16 = nullptr, *dEBR_fp16 = nullptr;
  cutlass::half_t *dAxEBL_fp16 = nullptr, *dEARxBpEB_fp16 = nullptr;
  float *dAs = nullptr, *dBs = nullptr;
  cutlass::bfloat16_t *dC = nullptr;
  // PoW scratch
  uint32_t *dPowTarget = nullptr, *dPowKey = nullptr;
  void *dSignalSync = nullptr, *dSignalHeader = nullptr;
  uint64_t *dHashCounter = nullptr;

  void alloc(int M_, int N_, int K_, int R_) {
    M = M_; N = N_; K = K_; R = R_;
    CUCHK(cudaMalloc(&dA,  size_t(M)*K));
    CUCHK(cudaMalloc(&dB,  size_t(N)*K));
    CUCHK(cudaMalloc(&dEAL, size_t(M)*R));
    CUCHK(cudaMalloc(&dEBR, size_t(N)*R));
    CUCHK(cudaMalloc(&dEAR_R, size_t(K)*R));
    CUCHK(cudaMalloc(&dEBL_R, size_t(K)*R));
    CUCHK(cudaMalloc(&dEAR_K, size_t(R)*K));
    CUCHK(cudaMalloc(&dEBL_K, size_t(R)*K));
    CUCHK(cudaMalloc(&dApEA, size_t(M)*K));
    CUCHK(cudaMalloc(&dBpEB, size_t(N)*K));
    CUCHK(cudaMalloc(&dAxEBL_i32,     size_t(M)*R*sizeof(int32_t)));
    CUCHK(cudaMalloc(&dEARxBpEB_i32,  size_t(N)*R*sizeof(int32_t)));
    CUCHK(cudaMalloc(&dEAL_fp16,      size_t(M)*R*sizeof(cutlass::half_t)));
    CUCHK(cudaMalloc(&dEBR_fp16,      size_t(N)*R*sizeof(cutlass::half_t)));
    CUCHK(cudaMalloc(&dAxEBL_fp16,    size_t(M)*R*sizeof(cutlass::half_t)));
    CUCHK(cudaMalloc(&dEARxBpEB_fp16, size_t(N)*R*sizeof(cutlass::half_t)));
    CUCHK(cudaMalloc(&dAs, M*sizeof(float)));
    CUCHK(cudaMalloc(&dBs, N*sizeof(float)));
    CUCHK(cudaMalloc(&dC,  size_t(M)*N*sizeof(cutlass::bfloat16_t)));
    CUCHK(cudaMalloc(&dPowTarget,    8*sizeof(uint32_t)));
    CUCHK(cudaMalloc(&dPowKey,       8*sizeof(uint32_t)));
    CUCHK(cudaMalloc(&dSignalSync,   sizeof(HostSignalSync)));
    CUCHK(cudaMalloc(&dSignalHeader, host_signal_header_size));
    CUCHK(cudaMalloc(&dHashCounter,  sizeof(uint64_t)));
    // 'hard' (impossible) target — every thread bails out of write_host_signal_header
    // before the atomicCAS spin. Don't change to 0xFFFFFFFF: see postmortem.
    CUCHK(cudaMemset(dPowTarget, 0, 8*sizeof(uint32_t)));
    CUCHK(cudaMemset(dPowKey, 0, 8*sizeof(uint32_t)));
    CUCHK(cudaMemset(dSignalSync, 0, sizeof(HostSignalSync)));
    CUCHK(cudaMemset(dSignalHeader, 0, host_signal_header_size));
    CUCHK(cudaMemset(dHashCounter, 0, sizeof(uint64_t)));
  }

  void free_() {
    for (void* p : {(void*)dA,(void*)dB,(void*)dEAL,(void*)dEBR,
                    (void*)dEAR_R,(void*)dEBL_R,(void*)dEAR_K,(void*)dEBL_K,
                    (void*)dApEA,(void*)dBpEB,
                    (void*)dAxEBL_i32,(void*)dEARxBpEB_i32,
                    (void*)dEAL_fp16,(void*)dEBR_fp16,
                    (void*)dAxEBL_fp16,(void*)dEARxBpEB_fp16,
                    (void*)dAs,(void*)dBs,(void*)dC,
                    (void*)dPowTarget,(void*)dPowKey,
                    dSignalSync,dSignalHeader,(void*)dHashCounter}) {
      if (p) cudaFree(p);
    }
  }
};

// ------------------------------------------------------------------------
// Chain runner — noisingA + noisingB + denoise GEMM (with optional PoW).
// Only R=64 has both denoise and pow variants compiled into the standalone.
// R=128 currently launches the denoise variant only; --pow hard with R=128
// will report 'unsupported' from this binary (Python harness handles R=128
// PoW via the pybind path).
// ------------------------------------------------------------------------
static void run_chain_r64(TensorSet& t, bool pow_on, cudaStream_t stream) {
  pearl_noisingA_sm89_64x64x64_R64_int32(
      t.dA, t.dEAL, t.dEAR_R, t.dEBL_K, t.dApEA, t.dAxEBL_i32, t.M, t.K, stream);
  pearl::sm89::pearl_noisingB_sm89_64x64x64_R64_int32(
      t.dB, t.dEBR, t.dEBL_R, t.dEAR_K, t.dBpEB, t.dEARxBpEB_i32, t.N, t.K, stream);
  if (pow_on) {
    pearl::sm89::pearl_gemm_sm89_pow_128x128x64_R64(
        t.dApEA, t.K, t.dBpEB, t.K, t.dC, t.N, t.dAs, t.dBs,
        t.dEAL_fp16, t.dEBR_fp16, t.dAxEBL_fp16, t.dEARxBpEB_fp16,
        t.dPowTarget, t.dPowKey,
        t.dSignalSync, t.dSignalHeader, t.dHashCounter,
        t.M, t.N, t.K, stream);
  } else {
    pearl::sm89::pearl_gemm_sm89_denoise_128x128x64_R64(
        t.dApEA, t.K, t.dBpEB, t.K, t.dC, t.N, t.dAs, t.dBs,
        t.dEAL_fp16, t.dEBR_fp16, t.dAxEBL_fp16, t.dEARxBpEB_fp16,
        t.M, t.N, t.K, stream);
  }
}

// ------------------------------------------------------------------------
// Bench one (shape, config) pair
// ------------------------------------------------------------------------
struct Result {
  std::string shape;
  std::string config;
  int rank;
  std::string pow_mode;
  int streams;
  int iters;
  double median_ms;
  double p99_ms;
  double min_ms;
  double mean_ms;
  double attempts_per_s;
  double main_tops;
  double full_tops;
  std::string status;
  std::string note;
};

static Result bench_one(int M, int N, int K, int rank, bool pow_on,
                        int n_streams, int warmup, int repeats) {
  Result r;
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%dx%dx%d", M, N, K);
  r.shape = buf;
  std::snprintf(buf, sizeof(buf), "r%d-%s-%ds",
                rank, pow_on ? "hard" : "disabled", n_streams);
  r.config = buf;
  r.rank = rank;
  r.pow_mode = pow_on ? "hard" : "disabled";
  r.streams = n_streams;
  r.iters = repeats;
  r.median_ms = r.p99_ms = r.min_ms = r.mean_ms = 0.0/0.0;
  r.attempts_per_s = r.main_tops = r.full_tops = 0.0/0.0;

  if (rank != 64) {
    r.status = "unsupported";
    r.note   = "C++ standalone has only R=64 wired; use Python harness for R=128";
    return r;
  }

  std::vector<TensorSet> sets(n_streams);
  std::vector<cudaStream_t> streams(n_streams);
  for (int i = 0; i < n_streams; ++i) {
    sets[i].alloc(M, N, K, rank);
    CUCHK(cudaStreamCreate(&streams[i]));
  }

  // Warmup
  for (int w = 0; w < warmup; ++w) {
    for (int i = 0; i < n_streams; ++i) run_chain_r64(sets[i], pow_on, streams[i]);
    CUCHK(cudaDeviceSynchronize());
  }

  // Per-iter events
  std::vector<cudaEvent_t> e0(repeats), e1(repeats);
  for (int i = 0; i < repeats; ++i) {
    CUCHK(cudaEventCreate(&e0[i]));
    CUCHK(cudaEventCreate(&e1[i]));
  }

  CUCHK(cudaDeviceSynchronize());
  for (int it = 0; it < repeats; ++it) {
    CUCHK(cudaEventRecord(e0[it]));
    for (int i = 0; i < n_streams; ++i) run_chain_r64(sets[i], pow_on, streams[i]);
    // Make e1 record after all per-stream work has completed.
    for (int i = 0; i < n_streams; ++i) {
      cudaEvent_t join; cudaEventCreate(&join);
      CUCHK(cudaEventRecord(join, streams[i]));
      CUCHK(cudaStreamWaitEvent(0, join));
      cudaEventDestroy(join);
    }
    CUCHK(cudaEventRecord(e1[it]));
  }
  CUCHK(cudaDeviceSynchronize());

  std::vector<double> ms(repeats);
  for (int i = 0; i < repeats; ++i) {
    float t = 0.f; CUCHK(cudaEventElapsedTime(&t, e0[i], e1[i]));
    ms[i] = double(t);
  }
  std::vector<double> sorted_ms = ms;
  std::sort(sorted_ms.begin(), sorted_ms.end());
  r.median_ms = sorted_ms[sorted_ms.size() / 2];
  r.p99_ms    = sorted_ms[std::min<size_t>(sorted_ms.size() - 1,
                                            size_t(0.99 * (sorted_ms.size() - 1)))];
  r.min_ms    = sorted_ms.front();
  double sum = 0.0;
  for (double v : ms) sum += v;
  r.mean_ms = sum / repeats;

  double sec = r.median_ms / 1000.0;
  r.attempts_per_s = double(n_streams) / sec;
  double main_macs = 2.0 * double(M) * double(N) * double(K) * double(n_streams);
  double full_macs = (2.0 * double(K) * double(rank) * (double(M) + double(N))
                    + double(M) * double(N) * (double(K) + 2.0 * double(rank))) * double(n_streams);
  r.main_tops = main_macs / sec * 1e-12;
  r.full_tops = full_macs / sec * 1e-12;
  r.status = "ok";

  for (int i = 0; i < repeats; ++i) {
    cudaEventDestroy(e0[i]); cudaEventDestroy(e1[i]);
  }
  for (int i = 0; i < n_streams; ++i) {
    cudaStreamDestroy(streams[i]);
    sets[i].free_();
  }
  return r;
}

// ------------------------------------------------------------------------
// Args + CSV writing
// ------------------------------------------------------------------------
static std::vector<std::tuple<int,int,int>> parse_shapes(std::string const& spec) {
  std::vector<std::tuple<int,int,int>> out;
  std::stringstream ss(spec);
  std::string tok;
  while (std::getline(ss, tok, ',')) {
    if (tok.empty()) continue;
    if (tok.find('x') == std::string::npos) {
      int n = std::stoi(tok);
      out.emplace_back(n, n, n);
    } else {
      std::stringstream ts(tok);
      std::string a, b, c;
      std::getline(ts, a, 'x');
      std::getline(ts, b, 'x');
      std::getline(ts, c, 'x');
      out.emplace_back(std::stoi(a), std::stoi(b), std::stoi(c));
    }
  }
  return out;
}

static const char* get_opt(int argc, char** argv, char const* key, char const* dflt) {
  for (int i = 1; i + 1 < argc; ++i)
    if (std::strcmp(argv[i], key) == 0) return argv[i + 1];
  return dflt;
}

int main(int argc, char** argv) {
  for (int i = 1; i < argc; ++i)
    if (!std::strcmp(argv[i], "-h") || !std::strcmp(argv[i], "--help")) {
      std::printf("Usage: %s [--rank 64|128] [--pow hard|disabled] [--streams N] "
                  "[--warmup N] [--repeats N] [--shapes csv] [--device N] [--out path]\n", argv[0]);
      return 0;
    }

  int rank     = std::atoi(get_opt(argc, argv, "--rank",    "64"));
  std::string pow_s = get_opt(argc, argv, "--pow",     "hard");
  int streams  = std::atoi(get_opt(argc, argv, "--streams", "1"));
  int warmup   = std::atoi(get_opt(argc, argv, "--warmup",  "3"));
  int repeats  = std::atoi(get_opt(argc, argv, "--repeats", "30"));
  int dev      = std::atoi(get_opt(argc, argv, "--device",  "0"));
  std::string shapes_s = get_opt(argc, argv, "--shapes",
      "1024,2048,4096,8192,16384,4096x4096x8192,16384x4096x4096");
  std::string out_path = get_opt(argc, argv, "--out", "");

  if (pow_s != "hard" && pow_s != "disabled") {
    fprintf(stderr, "ERROR: --pow must be 'hard' or 'disabled' "
                    "(NOT 'easy' — see project_pearl_perf_postmortem)\n");
    return 2;
  }
  bool pow_on = (pow_s == "hard");

  CUCHK(cudaSetDevice(dev));
  cudaDeviceProp p; CUCHK(cudaGetDeviceProperties(&p, dev));
  printf("device %d: %s sm_%d%d  L2=%dMB SMs=%d\n",
         dev, p.name, p.major, p.minor, int(p.l2CacheSize/1024/1024), p.multiProcessorCount);
  printf("config: rank=%d pow=%s streams=%d warmup=%d repeats=%d\n",
         rank, pow_s.c_str(), streams, warmup, repeats);

  auto shapes = parse_shapes(shapes_s);

  std::vector<Result> results;
  printf("\n  %16s  %22s  %9s  %9s  %10s  %10s  %8s  status\n",
         "shape", "config", "med ms", "p99 ms", "main TOPS", "full TOPS", "att/s");
  printf("  ----------------------------------------------------------"
         "---------------------------------------------\n");
  for (auto [M, N, K] : shapes) {
    Result r = bench_one(M, N, K, rank, pow_on, streams, warmup, repeats);
    results.push_back(r);
    if (r.status == "ok") {
      printf("  %16s  %22s  %9.3f  %9.3f  %10.2f  %10.2f  %8.1f  %s\n",
             r.shape.c_str(), r.config.c_str(),
             r.median_ms, r.p99_ms, r.main_tops, r.full_tops,
             r.attempts_per_s, r.status.c_str());
    } else {
      printf("  %16s  %22s  %9s  %9s  %10s  %10s  %8s  %s (%s)\n",
             r.shape.c_str(), r.config.c_str(),
             "n/a", "n/a", "n/a", "n/a", "n/a",
             r.status.c_str(), r.note.c_str());
    }
  }

  if (!out_path.empty()) {
    std::ofstream f(out_path);
    f << "shape,config,rank,pow_mode,streams,iters,"
         "median_ms,p99_ms,min_ms,mean_ms,attempts_per_s,main_tops,full_tops,"
         "device_name,device_cap,status,note\n";
    for (auto const& r : results) {
      f << r.shape << "," << r.config << "," << r.rank << "," << r.pow_mode << ","
        << r.streams << "," << r.iters << ","
        << r.median_ms << "," << r.p99_ms << "," << r.min_ms << "," << r.mean_ms << ","
        << r.attempts_per_s << "," << r.main_tops << "," << r.full_tops << ","
        << p.name << "," << p.major << "." << p.minor << ","
        << r.status << "," << r.note << "\n";
    }
    printf("\nwrote %s\n", out_path.c_str());
  }

  return 0;
}
