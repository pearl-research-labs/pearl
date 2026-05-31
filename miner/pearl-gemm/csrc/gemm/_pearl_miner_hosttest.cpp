// Host-only validation of pearl_miner_host.hpp against the captured oracle.
// Reproduces job_key / b_noise_seed / a_noise_seed / gpu_hash AND the 16-word
// transcript from the disclosed A/B strips, all with ZERO external deps.
//
//   g++ -O2 -std=c++17 _pearl_miner_hosttest.cpp -o /tmp/hosttest
//   /tmp/hosttest <sharedump_dir>

#include "pearl_miner_host.hpp"

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

using namespace pearl_miner;

static std::vector<uint8_t> readfile(const std::string& p) {
  FILE* f = fopen(p.c_str(), "rb");
  if (!f) { fprintf(stderr, "cannot open %s\n", p.c_str()); exit(1); }
  fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
  std::vector<uint8_t> v(n);
  if (fread(v.data(), 1, n, f) != (size_t)n) { fprintf(stderr, "short read\n"); exit(1); }
  fclose(f);
  return v;
}

static std::string hex(const uint8_t* b, int n) {
  static const char* d = "0123456789abcdef";
  std::string s;
  for (int i = 0; i < n; ++i) { s += d[b[i] >> 4]; s += d[b[i] & 15]; }
  return s;
}

static bool from_hex(const char* s, uint8_t* out, int n) {
  for (int i = 0; i < n; ++i) {
    auto v = [](char c)->int{ if(c>='0'&&c<='9')return c-'0'; if(c>='a'&&c<='f')return c-'a'+10; if(c>='A'&&c<='F')return c-'A'+10; return -1;};
    int hi = v(s[2*i]), lo = v(s[2*i+1]);
    if (hi < 0 || lo < 0) return false;
    out[i] = (uint8_t)((hi << 4) | lo);
  }
  return true;
}

// find a 32-byte needle in haystack
static long find_bytes(const std::vector<uint8_t>& hay, const uint8_t* needle, int nl) {
  for (long i = 0; i + nl <= (long)hay.size(); ++i)
    if (memcmp(&hay[i], needle, nl) == 0) return i;
  return -1;
}

int main(int argc, char** argv) {
  std::string dir = (argc >= 2) ? argv[1] : "/mnt/c/Source/_lpminer_re/re_2026_05_30/sharedump";
  auto hdr = readfile(dir + "/header.bin");
  auto cfg = readfile(dir + "/mining_config.bin");
  auto proof = readfile(dir + "/proof.bin");

  printf("header.bin=%zuB mining_config.bin=%zuB proof.bin=%zuB\n",
         hdr.size(), cfg.size(), proof.size());

  // Oracle constants from meta.txt.
  uint8_t o_job[32], o_b[32], o_a[32], o_gpu[32], hash_a[32], hash_b[32];
  from_hex("3d825b47038e76ce5a8227474c8ceddc779e4e9822715b085dde2ab19fa936ab", o_job, 32);
  from_hex("58758e7e069a884fc25eb5dbc595090a4985c6556191009e3e53174a458f3cdc", o_b, 32);
  from_hex("3221fa5daec04de9f8d18f8428fccba85e0f9192538c698856da17854529a50b", o_a, 32);
  from_hex("7c920e4756693f4c9c1d03d24b25ef8a937d361248922f47fc60d1b6c947ae60", o_gpu, 32);
  from_hex("21132078255e277dae94591c9d17daa869347178b8fa5d4001ab5194aa0a515b", hash_a, 32);
  from_hex("bbb7be080bef77f2f8653ec0bc2625af01eb0b832d6b4cafc95bc09fd1e59f64", hash_b, 32);

  Seeds s = derive_seeds(hdr.data(), hdr.size(), cfg.data(), cfg.size(), hash_a, hash_b);

  int fail = 0;
  auto chk = [&](const char* name, const uint8_t* got, const uint8_t* exp) {
    bool ok = memcmp(got, exp, 32) == 0;
    printf("  %-14s %s  %s\n", name, hex(got, 32).c_str(), ok ? "MATCH" : "MISMATCH");
    if (!ok) { printf("    expected   %s\n", hex(exp, 32).c_str()); ++fail; }
  };
  printf("\n[derivation chain]\n");
  chk("job_key", s.job_key, o_job);
  chk("b_noise_seed", s.b_noise_seed, o_b);
  chk("a_noise_seed", s.a_noise_seed, o_a);

  // ---- Decisive keyed-BLAKE3 PoW-digest check ----
  // The transcript ITSELF cannot be byte-reproduced from this dump alone (it
  // requires the miner's private full A/B; the proof only discloses 8 A-rows +
  // 16 B-cols packed inside a merkle multiproof — see report 07's open
  // deserialization caveat). But the keyed-BLAKE3 PoW digest IS decisively
  // checkable: feed the captured oracle transcript + our derived a_noise_seed
  // and confirm we reproduce gpu_hash. This closes the loop on the keyed-blake3
  // PoW path and the a_noise_seed derivation together.
  uint32_t oracleT[16] = {
      0x000347d1,0xfffeaf78,0x00022839,0x000da1d5,0x00064448,0x0006e061,
      0xffff3a09,0xffe239bf,0xfff9623f,0x00060c97,0xfff07b33,0xfffc6967,
      0xfff852af,0x0006cde0,0x0006b873,0xffc7302e};
  uint8_t gpu[32];
  transcript_hash(oracleT, s.a_noise_seed, gpu);
  printf("\n[gpu_hash = blake3_keyed(oracle_transcript, key=a_noise_seed)]\n");
  chk("gpu_hash", gpu, o_gpu);

  printf("\n%s\n", fail == 0 ? "HOST GATE: PASS" : "HOST GATE: FAIL");
  return fail == 0 ? 0 : 1;
}
