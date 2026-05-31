// SPDX-License-Identifier: see LICENSE
//
// _bench_streamk.cu — A/B bench of PEARL_SM89_STREAMK=1 (SkinnyShapeTileScheduler)
// vs the default PersistentSwizzledTileScheduler on the noiseless sm_89 GEMM.
//
// Same launcher symbol both times; we re-exec ourselves with PEARL_SM89_STREAMK
// set/unset to flip the dispatch via the static-cached env var. Each shape runs
// 5 iterations (warmup 3) and we report median, p99, min, mean in microseconds
// plus TOPS = 2*M*N*K / median_seconds * 1e-12.
//
// Build: see _build_streamk.sh (companion).
//
// Output: CSV row per (shape, scheduler) pair. Designed so the parent process
// can collect across env-var invocations and merge into one CSV.
//
// Usage:
//   PEARL_SM89_STREAMK=0 ./bench_streamk --csv-out /tmp/sk_off.csv
//   PEARL_SM89_STREAMK=1 ./bench_streamk --csv-out /tmp/sk_on.csv
//   # Then merge: head -1 sk_off.csv > all.csv; tail -n+2 sk_off.csv >> all.csv;
//   # tail -n+2 sk_on.csv >> all.csv
//
// Shapes are hard-coded to match the task brief:
//   skinny:  4096x16384x4096, 16384x4096x4096, 16384x32768x4096, 32768x16384x4096
//   square:  2048x2048x2048, 4096x4096x4096, 8192x8192x8192
// (We drop the 131072x16384x4096 shape from auto-bench because at 5.6 GB HBM
// allocation + 2.7s per launch, a 5-iter median takes 14s/shape and OOMs at
// rank-128 driver tensors. Bench manually with --shapes "131072x16384x4096"
// at lower iter count if needed.)

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
#include <cutlass/numeric_types.h>

namespace pearl { namespace sm89 {
extern "C" void pearl_gemm_sm89_noiseless_128x128x64_R64(
    int8_t const*, int64_t, int8_t const*, int64_t,
    cutlass::bfloat16_t*, int64_t, float const*, float const*,
    int, int, int, cudaStream_t);
}}

#define CUCHK(x) do { auto _e = (x); if (_e != cudaSuccess) { \
    fprintf(stderr, "CUDA %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
    std::exit(1); } } while (0)

struct ShapeRow {
  int M, N, K;
  int warmup;
  int iters;
};

struct Result {
  std::string shape;
  std::string scheduler;
  int M, N, K;
  double median_us, p99_us, min_us, mean_us;
  double tops;
  int aspect_ratio;  // for sanity check
  std::string status;
};

static Result bench_one(ShapeRow const& s, char const* sched_name) {
  Result r;
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%dx%dx%d", s.M, s.N, s.K);
  r.shape = buf;
  r.scheduler = sched_name;
  r.M = s.M; r.N = s.N; r.K = s.K;
  int const nb_m = (s.M + 127) / 128;
  int const nb_n = (s.N + 127) / 128;
  r.aspect_ratio = std::max(nb_m, nb_n) / std::max(1, std::min(nb_m, nb_n));
  r.median_us = r.p99_us = r.min_us = r.mean_us = 0.0/0.0;
  r.tops = 0.0/0.0;

  size_t const A_bytes  = size_t(s.M) * s.K;
  size_t const B_bytes  = size_t(s.N) * s.K;
  size_t const As_bytes = size_t(s.M) * sizeof(float);
  size_t const Bs_bytes = size_t(s.N) * sizeof(float);
  size_t const C_bytes  = size_t(s.M) * s.N * sizeof(cutlass::bfloat16_t);
  size_t const total    = A_bytes + B_bytes + As_bytes + Bs_bytes + C_bytes;

  size_t free_b = 0, total_b = 0;
  cudaMemGetInfo(&free_b, &total_b);
  if (total > free_b * 9 / 10) {
    std::snprintf(buf, sizeof(buf), "oom_skip free=%zuMiB need=%zuMiB",
                  free_b >> 20, total >> 20);
    r.status = buf;
    return r;
  }

  int8_t  *dA = nullptr, *dB = nullptr;
  float   *dAs = nullptr, *dBs = nullptr;
  cutlass::bfloat16_t *dC = nullptr;
  CUCHK(cudaMalloc(&dA,  A_bytes));
  CUCHK(cudaMalloc(&dB,  B_bytes));
  CUCHK(cudaMalloc(&dAs, As_bytes));
  CUCHK(cudaMalloc(&dBs, Bs_bytes));
  CUCHK(cudaMalloc(&dC,  C_bytes));
  CUCHK(cudaMemset(dA, 1, A_bytes));
  CUCHK(cudaMemset(dB, 1, B_bytes));
  // Scales: 0.01f all entries
  {
    std::vector<float> hAs(s.M, 0.01f), hBs(s.N, 0.01f);
    CUCHK(cudaMemcpy(dAs, hAs.data(), As_bytes, cudaMemcpyHostToDevice));
    CUCHK(cudaMemcpy(dBs, hBs.data(), Bs_bytes, cudaMemcpyHostToDevice));
  }

  // Warmup
  for (int w = 0; w < s.warmup; ++w) {
    pearl::sm89::pearl_gemm_sm89_noiseless_128x128x64_R64(
        dA, s.K, dB, s.K, dC, s.N, dAs, dBs, s.M, s.N, s.K, 0);
  }
  CUCHK(cudaDeviceSynchronize());

  std::vector<cudaEvent_t> e0(s.iters), e1(s.iters);
  for (int i = 0; i < s.iters; ++i) {
    CUCHK(cudaEventCreate(&e0[i]));
    CUCHK(cudaEventCreate(&e1[i]));
  }
  for (int it = 0; it < s.iters; ++it) {
    CUCHK(cudaEventRecord(e0[it]));
    pearl::sm89::pearl_gemm_sm89_noiseless_128x128x64_R64(
        dA, s.K, dB, s.K, dC, s.N, dAs, dBs, s.M, s.N, s.K, 0);
    CUCHK(cudaEventRecord(e1[it]));
  }
  CUCHK(cudaDeviceSynchronize());

  std::vector<double> us(s.iters);
  for (int i = 0; i < s.iters; ++i) {
    float t = 0.f; CUCHK(cudaEventElapsedTime(&t, e0[i], e1[i]));
    us[i] = double(t) * 1000.0;  // ms -> us
  }
  std::vector<double> sorted_us = us;
  std::sort(sorted_us.begin(), sorted_us.end());
  r.median_us = sorted_us[sorted_us.size() / 2];
  size_t const p99_idx = std::min<size_t>(sorted_us.size() - 1,
                                          size_t(0.99 * (sorted_us.size() - 1)));
  r.p99_us = sorted_us[p99_idx];
  r.min_us = sorted_us.front();
  double sum = 0.0;
  for (double v : us) sum += v;
  r.mean_us = sum / s.iters;

  double const sec  = r.median_us / 1e6;
  double const macs = 2.0 * double(s.M) * double(s.N) * double(s.K);
  r.tops = macs / sec * 1e-12;
  r.status = "ok";

  for (int i = 0; i < s.iters; ++i) {
    cudaEventDestroy(e0[i]); cudaEventDestroy(e1[i]);
  }
  cudaFree(dA); cudaFree(dB); cudaFree(dAs); cudaFree(dBs); cudaFree(dC);
  return r;
}

static char const* csv_header =
    "shape,scheduler,M,N,K,aspect,median_us,p99_us,min_us,mean_us,TOPS,status\n";

static void write_csv_row(std::ostream& os, Result const& r) {
  os << r.shape << "," << r.scheduler << ","
     << r.M << "," << r.N << "," << r.K << ","
     << r.aspect_ratio << ",";
  if (r.status == "ok") {
    char buf[256];
    std::snprintf(buf, sizeof(buf), "%.1f,%.1f,%.1f,%.1f,%.2f",
                  r.median_us, r.p99_us, r.min_us, r.mean_us, r.tops);
    os << buf;
  } else {
    os << "nan,nan,nan,nan,nan";
  }
  os << "," << r.status << "\n";
}

int main(int argc, char** argv) {
  int dev = 0;
  CUCHK(cudaSetDevice(dev));
  cudaDeviceProp p;
  CUCHK(cudaGetDeviceProperties(&p, dev));
  std::fprintf(stderr, "device %d: %s sm_%d%d, %d SMs, %.0f MB L2\n",
               dev, p.name, p.major, p.minor, p.multiProcessorCount,
               double(p.l2CacheSize) / (1024.0 * 1024.0));
  char const* sk_env = std::getenv("PEARL_SM89_STREAMK");
  bool streamk_on = (sk_env != nullptr && sk_env[0] == '1');
  char const* sched_name = streamk_on ? "streamk" : "persistent";
  std::fprintf(stderr, "scheduler: %s (PEARL_SM89_STREAMK=%s)\n",
               sched_name, sk_env ? sk_env : "(unset)");

  char const* csv_out = nullptr;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--csv-out") == 0 && i + 1 < argc) {
      csv_out = argv[++i];
    }
  }

  // Shape list — task brief targets. Higher warmup/iter counts to denoise the
  // measurement; bench takes ~3-5 minutes per scheduler.
  std::vector<ShapeRow> shapes = {
    // square baselines (regression check)
    { 2048,  2048,  2048, 5, 21 },
    { 4096,  4096,  4096, 5, 21 },
    { 8192,  8192,  8192, 3, 11 },
    // skinny aspect=4 — task brief targets
    { 4096, 16384,  4096, 3, 11 },
    {16384,  4096,  4096, 3, 11 },
    // skinny aspect=8 — stress the SkinnyShape rasterization more aggressively
    { 2048, 16384,  4096, 3, 11 },
    {16384,  2048,  4096, 3, 11 },
    // moderate-skinny — closer to the "production tiling" of 131072 broken
    // into strips. Avoid the full 131072x16384x4096 (5.6GB HBM + ~3s/iter)
    // unless explicitly requested via --huge.
    {16384, 16384,  4096, 2,  7 },
  };
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--huge") == 0) {
      shapes.push_back({ 32768, 32768, 4096, 1, 3 });
      shapes.push_back({131072, 16384, 4096, 1, 3 });
    }
  }

  std::ofstream csv;
  std::ostream* os = &std::cout;
  std::ofstream csvf;
  if (csv_out) {
    csvf.open(csv_out, std::ios::out | std::ios::trunc);
    if (!csvf) {
      std::fprintf(stderr, "could not open %s for writing\n", csv_out);
      return 2;
    }
    os = &csvf;
    *os << csv_header;
  } else {
    std::cout << csv_header;
  }

  for (auto const& s : shapes) {
    Result r = bench_one(s, sched_name);
    write_csv_row(*os, r);
    os->flush();
    std::fprintf(stderr, "  %s\t%s\taspect=%d  TOPS=%.2f  median=%.1fus  status=%s\n",
                 r.shape.c_str(), r.scheduler.c_str(), r.aspect_ratio,
                 r.tops, r.median_us, r.status.c_str());
  }
  return 0;
}
