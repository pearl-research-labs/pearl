// CPU B preparation kernels for vllm_miner.b_prepare_cpu: the keyed BLAKE3
// noise lines and the fused B' row chain, bit-identical to its torch
// reference except the peel's mid half (a tolerance surface).
//
// Build with -ffp-contract=off: every fp32 multiply must round on its own.

#include <ATen/Parallel.h>
#include <torch/extension.h>

#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#ifdef __AVX512F__
#include <immintrin.h>
#endif

namespace {

constexpr int kRank = 32;
constexpr int kPrequantBlock = 8;
constexpr int64_t kTileRows = 16;
constexpr int64_t kTileCols = 256;
constexpr int64_t kLineGrain = 256;

// ---- BLAKE3, one root block of a keyed hash ----

constexpr std::array<uint32_t, 8> kBlake3Iv = {
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
    0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19};
constexpr uint8_t kBlake3Schedule[7][16] = {
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    {2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8},
    {3, 4, 10, 12, 13, 2, 7, 14, 6, 5, 9, 0, 11, 15, 8, 1},
    {10, 7, 12, 9, 14, 3, 13, 15, 4, 0, 11, 2, 5, 8, 1, 6},
    {12, 13, 9, 11, 15, 10, 14, 8, 7, 2, 5, 3, 0, 1, 6, 4},
    {9, 14, 11, 5, 8, 12, 15, 1, 13, 3, 0, 10, 2, 6, 4, 7},
    {11, 15, 5, 0, 1, 9, 8, 6, 14, 10, 2, 12, 3, 4, 7, 13},
};
constexpr uint32_t kBlake3BlockLen = 64;
// CHUNK_START | CHUNK_END | ROOT | KEYED_HASH: a keyed hash of one block.
constexpr uint32_t kKeyedRootBlockFlags = 1 | 2 | 8 | 16;
constexpr size_t kKeyBytes = 32;
constexpr size_t kLabelBytes = 2;

inline uint32_t rotr32(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

inline void blake3_g(uint32_t* s, int a, int b, int c, int d, uint32_t x, uint32_t y) {
  s[a] = s[a] + s[b] + x;
  s[d] = rotr32(s[d] ^ s[a], 16);
  s[c] = s[c] + s[d];
  s[b] = rotr32(s[b] ^ s[c], 12);
  s[a] = s[a] + s[b] + y;
  s[d] = rotr32(s[d] ^ s[a], 8);
  s[c] = s[c] + s[d];
  s[b] = rotr32(s[b] ^ s[c], 7);
}

// The first kRank output bytes of the keyed hash of the 64-byte block `m`
// (little-endian words; this host is little-endian).
void keyed_block_hash(const uint32_t* key, const uint32_t* m, uint8_t* out) {
  uint32_t s[16];
  for (int i = 0; i < 8; ++i) s[i] = key[i];
  for (int i = 0; i < 4; ++i) s[8 + i] = kBlake3Iv[i];
  s[12] = 0;
  s[13] = 0;
  s[14] = kBlake3BlockLen;
  s[15] = kKeyedRootBlockFlags;
  for (const auto& p : kBlake3Schedule) {
    blake3_g(s, 0, 4, 8, 12, m[p[0]], m[p[1]]);
    blake3_g(s, 1, 5, 9, 13, m[p[2]], m[p[3]]);
    blake3_g(s, 2, 6, 10, 14, m[p[4]], m[p[5]]);
    blake3_g(s, 3, 7, 11, 15, m[p[6]], m[p[7]]);
    blake3_g(s, 0, 5, 10, 15, m[p[8]], m[p[9]]);
    blake3_g(s, 1, 6, 11, 12, m[p[10]], m[p[11]]);
    blake3_g(s, 2, 7, 8, 13, m[p[12]], m[p[13]]);
    blake3_g(s, 3, 4, 9, 14, m[p[14]], m[p[15]]);
  }
  for (int i = 0; i < kRank / 4; ++i) {
    const uint32_t word = s[i] ^ s[i + 8];
    std::memcpy(out + 4 * i, &word, 4);
  }
}

// ---- Number formats ----

constexpr uint32_t kBf16NaN = 0x7FC0;
constexpr float kQuantMax = 448.0f;
// e4m3 from fp32 in [-448, 448] (c10's conversion, round to nearest even):
// below 2^-6 through the denormal mask, else re-bias the exponent and round.
constexpr uint32_t kE4m3DenormMask = UINT32_C(141) << 23;
constexpr uint32_t kE4m3MinNormal = UINT32_C(121) << 23;
constexpr uint32_t kE4m3NormalBias = (static_cast<uint32_t>(7 - 127) << 23) + 0x7FFFF;
constexpr uint32_t kE4m3Nan = 0x7F;

// The strip rewrite (protocol_constants.STRIP_*); g is the committed device's.
constexpr float kStripSmallRms = 8.0f;
constexpr int kStripExpOffset = 9;
constexpr int kStripCoinBlock = 4;
constexpr int kStripCoinShift = 8;
constexpr uint32_t kStripCoinThreshold = 112;

inline uint32_t f32_bits(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  return u;
}

inline float f32_from_bits(uint32_t u) {
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

// c10::BFloat16: round to nearest even, NaN to the canonical quiet NaN.
inline uint16_t bf16_bits(float f) {
  const uint32_t u = f32_bits(f);
  const uint16_t rounded = static_cast<uint16_t>((u + 0x7FFF + ((u >> 16) & 1)) >> 16);
  return std::isnan(f) ? static_cast<uint16_t>(kBf16NaN) : rounded;
}

inline float bf16_value(uint16_t b) { return f32_from_bits(static_cast<uint32_t>(b) << 16); }

inline float round_bf16(float f) { return bf16_value(bf16_bits(f)); }

// The verifier's e4m3 cast: round to nearest even, saturating to +-448.
inline uint8_t e4m3_bits(float f) {
  f = std::min(std::max(f, -kQuantMax), kQuantMax);
  uint32_t bits = f32_bits(f);
  const uint32_t sign = bits & UINT32_C(0x80000000);
  bits ^= sign;
  const uint32_t denorm =
      f32_bits(f32_from_bits(bits) + f32_from_bits(kE4m3DenormMask)) - kE4m3DenormMask;
  const uint32_t normal = (bits + kE4m3NormalBias + ((bits >> 20) & 1)) >> 20;
  const uint32_t magnitude = bits < kE4m3MinNormal ? denorm : normal;
  return static_cast<uint8_t>(magnitude | (sign >> 24));
}

float e4m3_value(uint8_t b) {
  const int exponent = (b >> 3) & 0xF;
  const int mantissa = b & 0x7;
  float magnitude = exponent == 0 ? std::ldexp(static_cast<float>(mantissa), -9)
                                  : std::ldexp(static_cast<float>(8 + mantissa), exponent - 10);
  if (exponent == 0xF && mantissa == 0x7) magnitude = NAN;
  return (b & 0x80) ? -magnitude : magnitude;
}

const std::array<float, 256> kE4m3Values = [] {
  std::array<float, 256> values{};
  for (int b = 0; b < 256; ++b) values[b] = e4m3_value(static_cast<uint8_t>(b));
  return values;
}();

// float64 -> fp32 rounded towards zero.
inline float toward_zero_f32(double v) {
  const float nearest = static_cast<float>(v);
  const bool away = std::fabs(static_cast<double>(nearest)) > std::fabs(v);
  return away ? f32_from_bits(f32_bits(nearest) - 1) : nearest;
}

uint64_t isqrt(uint64_t x) {
  uint64_t r = static_cast<uint64_t>(std::sqrt(static_cast<double>(x)));
  while (r * r > x) --r;
  while ((r + 1) * (r + 1) <= x) ++r;
  return r;
}

// A row's strip state from alpha and its floored l2: the magnitude bound
// |e4m3(g * s)| and the safe exponent already in code position, s = alpha * l2.
struct Strip {
  uint32_t bound;
  uint32_t code;
};

inline Strip strip_of(float alpha, float l2, float strip_g, float strip_g_small_rms) {
  const float s = alpha * l2;
  const float g = s < kStripSmallRms ? strip_g_small_rms : strip_g;
  const uint32_t field = ((f32_bits(s) >> 23) & 0xFF) - (127 + kStripExpOffset - 7);
  return {static_cast<uint32_t>(e4m3_bits(g * s) & 0x7F), ((field & 0xF) << 3) | 1};
}

inline bool coin_passes(float mixed) {
  return ((f32_bits(mixed) >> kStripCoinShift) & 0x7F) < kStripCoinThreshold;
}

inline uint8_t rewrite(uint8_t q, Strip strip, bool coin) {
  return coin && (q & 0x7Fu) < strip.bound ? static_cast<uint8_t>((q & 0x87u) | strip.code) : q;
}

// ---- Integer noise dot ----
// An e4m3 value is sig * 2^(q - kE4m3UnitExp) with |sig| <= 15 and
// q = max(exponent, 1). A line whose q span is at most kMaxLineSpan holds
// exactly as int16 sig << (q - q_min) times 2^(q_min - kE4m3UnitExp); a row
// and column whose spans sum to at most kMaxDotSpan dot exactly in int32
// (32 * 15^2 * 2^18 < 2^31). kMaxDotSpan is within the atom's untruncated
// span (b_prepare_cpu._UNTRUNCATED_EXP_SPAN), so that dot is the atom's.
constexpr int kE4m3UnitExp = 10;
constexpr int kBlackwellWindowBits = 26;
constexpr int kE4m3MaxQ = 15;
constexpr int kMaxLineSpan = 11;
constexpr int kMaxDotSpan = 18;
constexpr int kPairs = kRank / 2;
// Columns per group of F_B's paired form: one AVX-512 register of int32. The
// layout is built either way, so this is not under __AVX512F__.
constexpr int kLanes = 16;

inline int e4m3_q(uint8_t b) { return std::max((b >> 3) & 0xF, 1); }

// The q span of the kRank values line[j * stride]; a NaN spans everything.
int q_span(const uint8_t* line, int64_t stride) {
  int lo = kE4m3MaxQ, hi = 1;
  for (int j = 0; j < kRank; ++j) {
    const uint8_t b = line[j * stride];
    if ((b & 0x7F) == kE4m3Nan) return INT_MAX / 2;
    lo = std::min(lo, e4m3_q(b));
    hi = std::max(hi, e4m3_q(b));
  }
  return hi - lo;
}

// line[j * stride] == ints[j] * (the returned power of two), for a line
// within kMaxLineSpan.
float integer_line(const uint8_t* line, int64_t stride, int16_t* ints) {
  int lo = kE4m3MaxQ;
  for (int j = 0; j < kRank; ++j) lo = std::min(lo, e4m3_q(line[j * stride]));
  for (int j = 0; j < kRank; ++j) {
    const uint8_t b = line[j * stride];
    const int exponent = (b >> 3) & 0xF, mantissa = b & 0x7;
    const int magnitude = (exponent ? (mantissa | 8) : mantissa) << (e4m3_q(b) - lo);
    ints[j] = static_cast<int16_t>((b & 0x80) ? -magnitude : magnitude);
  }
  return std::ldexp(1.0f, lo - kE4m3UnitExp);
}

inline int32_t int_dot(const int16_t* e, const int16_t* f) {
  int32_t sum = 0;
  for (int j = 0; j < kRank; ++j) sum += static_cast<int32_t>(e[j]) * f[j];
  return sum;
}

// ---- Windowed noise dot (the verifier's windowed_group_sum) ----
// One K = kRank FP8 atom from +0: each product sig * 2^(exp - 23), sig =
// (sa * sb) << 17, is truncated to the window (`width` significand bits,
// anchored at the largest exponent), the terms are summed exactly, and the
// sum is truncated to `width` bits. A zero result is +0.
constexpr int kGFloatBits = 24;
constexpr int kF32MinExponent = -126;

struct E4m3Term {
  int8_t sig;  // signed significand
  int8_t q;    // max(exponent field, 1)
};

inline E4m3Term e4m3_term(uint8_t b) {
  const int exponent = (b >> 3) & 0xF, mantissa = b & 0x7;
  const int sig = exponent ? (mantissa | 8) : mantissa;
  return {static_cast<int8_t>((b & 0x80) ? -sig : sig), static_cast<int8_t>(e4m3_q(b))};
}

float windowed_dot(const E4m3Term* e, const E4m3Term* f, int width) {
  int max_exp = INT_MIN;
  for (int j = 0; j < kRank; ++j) {
    if (e[j].sig != 0 && f[j].sig != 0) max_exp = std::max(max_exp, e[j].q + f[j].q - 14);
  }
  if (max_exp == INT_MIN) return 0.0f;
  const int diff = std::abs(width - kGFloatBits);
  int64_t acc = 0;
  for (int j = 0; j < kRank; ++j) {
    const int product = e[j].sig * f[j].sig;
    if (product == 0) continue;
    const int shift = max_exp - (e[j].q + f[j].q - 14);
    if (shift >= 32) continue;
    const uint32_t sig = static_cast<uint32_t>(std::abs(product)) << 17;
    const uint32_t shifted = width < kGFloatBits ? sig >> diff : sig << diff;
    const int64_t aligned = shifted >> shift;
    acc += product < 0 ? -aligned : aligned;
  }
  if (acc == 0) return 0.0f;
  uint64_t magnitude = static_cast<uint64_t>(acc < 0 ? -acc : acc);
  const int bits = 64 - __builtin_clzll(magnitude);
  int exponent = max_exp + bits - width;
  magnitude = bits > width ? magnitude >> (bits - width) : magnitude << (width - bits);
  if (exponent < kF32MinExponent) {
    const int shift = kF32MinExponent - exponent;
    magnitude = shift >= 32 ? 0 : magnitude >> shift;
    exponent = kF32MinExponent;
  }
  magnitude = width < kGFloatBits ? magnitude << diff : magnitude >> diff;
  if (magnitude == 0) return 0.0f;
  const float value = std::ldexp(static_cast<float>(magnitude), exponent - 23);
  return acc < 0 ? -value : value;
}

// out[r][c] = windowed_dot(E_B row r0 + r, F_B column c0 + c) for a tile.
void windowed_noise_tile(
    const E4m3Term* e_terms, int64_t rows, const E4m3Term* f2_terms, int64_t c0, int64_t cols,
    int width, float* out) {
  for (int64_t r = 0; r < rows; ++r) {
    for (int64_t c = 0; c < cols; ++c) {
      out[r * kTileCols + c] =
          windowed_dot(e_terms + r * kRank, f2_terms + (c0 + c) * kRank, width);
    }
  }
}

// The integer sums of noise_tile as the atom's fp32 noise (rounded towards
// zero; scaling by the powers of two row_scale * col_scale is exact).
void integer_noise_values(
    const int32_t* sums, int64_t rows, const float* row_scale, const float* col_scale,
    int64_t cols, float* out) {
  for (int64_t r = 0; r < rows; ++r) {
    for (int64_t c = 0; c < cols; ++c) {
      const double exact = static_cast<double>(sums[r * kTileCols + c]) * row_scale[r] * col_scale[c];
      out[r * kTileCols + c] = toward_zero_f32(exact);
    }
  }
}

// One B cell's fp32 mix: fma(beta, noise, (alpha * scale) * code), one
// rounding per operation.
inline float mix_cell(float coefficient, float b, int8_t code, float noise) {
  return std::fma(b, noise, coefficient * static_cast<float>(code));
}

#ifdef __AVX512F__

inline __m512i splat(uint32_t v) { return _mm512_set1_epi32(static_cast<int>(v)); }

inline __m512 join(__m256 lo, __m256 hi) {
  return _mm512_insertf32x8(_mm512_castps256_ps512(lo), hi, 1);
}

inline __m512 round_bf16_v(__m512 x) {
  const __m512i u = _mm512_castps_si512(x);
  const __m512i lsb = _mm512_and_si512(_mm512_srli_epi32(u, 16), splat(1));
  const __m512i rounded = _mm512_and_si512(
      _mm512_add_epi32(u, _mm512_add_epi32(lsb, splat(0x7FFF))), splat(0xFFFF0000));
  const __mmask16 nan = _mm512_cmp_ps_mask(x, x, _CMP_UNORD_Q);
  return _mm512_castsi512_ps(_mm512_mask_blend_epi32(nan, rounded, splat(kBf16NaN << 16)));
}

// e4m3_bits on 16 lanes, as int32.
inline __m512i e4m3_bits_v(__m512 f) {
  f = _mm512_max_ps(_mm512_set1_ps(-kQuantMax), _mm512_min_ps(_mm512_set1_ps(kQuantMax), f));
  __m512i bits = _mm512_castps_si512(f);
  const __m512i sign = _mm512_and_si512(bits, splat(0x80000000));
  bits = _mm512_xor_si512(bits, sign);
  const __m512 denorm_mask = _mm512_castsi512_ps(splat(kE4m3DenormMask));
  const __m512i denorm = _mm512_sub_epi32(
      _mm512_castps_si512(_mm512_add_ps(_mm512_castsi512_ps(bits), denorm_mask)),
      splat(kE4m3DenormMask));
  const __m512i mant_odd = _mm512_and_si512(_mm512_srli_epi32(bits, 20), splat(1));
  const __m512i normal = _mm512_srli_epi32(
      _mm512_add_epi32(_mm512_add_epi32(bits, splat(kE4m3NormalBias)), mant_odd), 20);
  const __mmask16 small = _mm512_cmplt_epu32_mask(bits, splat(kE4m3MinNormal));
  const __m512i magnitude = _mm512_mask_blend_epi32(small, normal, denorm);
  return _mm512_and_si512(
      _mm512_or_si512(magnitude, _mm512_srli_epi32(sign, 24)), splat(0xFF));
}

// kE4m3Values on 16 int32 lanes of e4m3 bits.
inline __m512 e4m3_value_v(__m512i q) {
  constexpr uint32_t kToF32Bias = static_cast<uint32_t>(127 - 7) << 23;
  constexpr uint32_t kF32QuietNan = 0x7FC00000;
  constexpr int kMantissaShift = 20;
  constexpr uint32_t kMinNormalBits = 8;
  const float kSubnormalUnit = std::ldexp(1.0f, -9);
  const __m512i magnitude = _mm512_and_si512(q, splat(0x7F));
  const __m512i normal =
      _mm512_add_epi32(_mm512_slli_epi32(magnitude, kMantissaShift), splat(kToF32Bias));
  const __m512 subnormal =
      _mm512_mul_ps(_mm512_cvtepi32_ps(magnitude), _mm512_set1_ps(kSubnormalUnit));
  __m512i value = _mm512_mask_blend_epi32(
      _mm512_cmplt_epu32_mask(magnitude, splat(kMinNormalBits)), normal,
      _mm512_castps_si512(subnormal));
  value = _mm512_mask_blend_epi32(
      _mm512_cmpeq_epi32_mask(magnitude, splat(kE4m3Nan)), value, splat(kF32QuietNan));
  const __m512i sign = _mm512_slli_epi32(_mm512_and_si512(q, splat(0x80)), 24);
  return _mm512_castsi512_ps(_mm512_or_si512(value, sign));
}

// rewrite on 16 lanes of whole coin groups, as int32.
inline __m512i rewrite_v(__m512i q, __m512 mixed, Strip strip) {
  // Each lane reads its group's first column (lane & ~3).
  const __m512i group_start = _mm512_set_epi32(12, 12, 12, 12, 8, 8, 8, 8, 4, 4, 4, 4, 0, 0, 0, 0);
  const __m512i field = _mm512_and_si512(
      _mm512_srli_epi32(
          _mm512_permutexvar_epi32(group_start, _mm512_castps_si512(mixed)), kStripCoinShift),
      splat(0x7F));
  const __mmask16 coin = _mm512_cmplt_epu32_mask(field, splat(kStripCoinThreshold));
  const __mmask16 inside = _mm512_mask_cmplt_epu32_mask(
      coin, _mm512_and_si512(q, splat(0x7F)), splat(strip.bound));
  const __m512i rewritten = _mm512_or_si512(_mm512_and_si512(q, splat(0x87)), splat(strip.code));
  return _mm512_mask_blend_epi32(inside, q, rewritten);
}

#endif

#ifdef __AVX512BF16__
inline __m512bh as_bf16_pairs(__m512i x) { return reinterpret_cast<__m512bh&>(x); }
#endif

inline uint32_t load_pair(const void* p) {
  uint32_t pair;
  std::memcpy(&pair, p, sizeof(pair));
  return pair;
}

// One row's cells [0, cols) of a column tile (cols a multiple of the coin
// block, starting on one) under the atom's fp32 noise. Writes B' and the
// peel's mid input diff = bf16(beta * bf16(noise) - B').
void prepare_row(
    float a, float b, Strip strip, const int8_t* code, const uint16_t* scale,
    const float* noises, int64_t cols, uint8_t* bp, uint16_t* diff) {
  int64_t c = 0;
#ifdef __AVX512F__
  const __m512 a_v = _mm512_set1_ps(a);
  const __m512 b_v = _mm512_set1_ps(b);
  constexpr int kBf16Shift = 16;
  for (; c + kLanes <= cols; c += kLanes) {
    const __m512 codes = _mm512_cvtepi32_ps(
        _mm512_cvtepi8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(code + c))));
    const int64_t block = c / kPrequantBlock;
    const __m512 coefficient = _mm512_mul_ps(
        a_v, join(
                 _mm256_set1_ps(bf16_value(scale[block])),
                 _mm256_set1_ps(bf16_value(scale[block + 1]))));
    const __m512 noise = _mm512_loadu_ps(noises + c);
    const __m512 mixed = _mm512_fmadd_ps(b_v, noise, _mm512_mul_ps(coefficient, codes));
    __m512i q = e4m3_bits_v(mixed);
    q = rewrite_v(q, mixed, strip);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(bp + c), _mm512_cvtepi32_epi8(q));
    const __m512 d =
        round_bf16_v(_mm512_fmsub_ps(b_v, round_bf16_v(noise), e4m3_value_v(q)));
    _mm256_storeu_si256(
        reinterpret_cast<__m256i*>(diff + c),
        _mm512_cvtepi32_epi16(_mm512_srli_epi32(_mm512_castps_si512(d), kBf16Shift)));
  }
#endif
  bool coin = false;
  for (; c < cols; ++c) {
    const float coefficient = a * bf16_value(scale[c / kPrequantBlock]);
    const float mixed = mix_cell(coefficient, b, code[c], noises[c]);
    if (c % kStripCoinBlock == 0) coin = coin_passes(mixed);
    uint8_t q = e4m3_bits(mixed);
    q = rewrite(q, strip, coin);
    bp[c] = q;
    diff[c] = bf16_bits(std::fma(b, round_bf16(noises[c]), -kE4m3Values[q]));
  }
}

// out[r][c] = sum_j e_ints[r][j] * f2_ints[c0 + c][j], exact in int32.
// f2_pairs holds F_B's columns in groups of kLanes as [pair][lane] int16 pairs.
void noise_tile(
    const int16_t* e_ints, int64_t rows, const int16_t* f2_ints, const int32_t* f2_pairs,
    int64_t c0, int64_t cols, int32_t* out) {
  int64_t c = 0;
#ifdef __AVX512VNNI__
  constexpr int64_t kRowStep = 8;
  for (; c + kLanes <= cols; c += kLanes) {
    const int32_t* f = f2_pairs + (c0 + c) / kLanes * (kPairs * kLanes);
    int64_t r = 0;
    for (; r + kRowStep <= rows; r += kRowStep) {
      __m512i acc[kRowStep];
      for (auto& row : acc) row = _mm512_setzero_si512();
      for (int p = 0; p < kPairs; ++p) {
        const __m512i fv = _mm512_loadu_si512(f + p * kLanes);
        for (int64_t i = 0; i < kRowStep; ++i) {
          const uint32_t e_pair = load_pair(e_ints + (r + i) * kRank + 2 * p);
          acc[i] = _mm512_dpwssd_epi32(acc[i], splat(e_pair), fv);
        }
      }
      for (int64_t i = 0; i < kRowStep; ++i) {
        _mm512_storeu_si512(out + (r + i) * kTileCols + c, acc[i]);
      }
    }
    for (; r < rows; ++r) {
      for (int64_t cc = c; cc < c + kLanes; ++cc) {
        out[r * kTileCols + cc] = int_dot(e_ints + r * kRank, f2_ints + (c0 + cc) * kRank);
      }
    }
  }
#endif
  for (int64_t r = 0; r < rows; ++r) {
    for (int64_t cc = c; cc < cols; ++cc) {
      out[r * kTileCols + cc] = int_dot(e_ints + r * kRank, f2_ints + (c0 + cc) * kRank);
    }
  }
}

// acc[r][j] += sum_c diff[r][c] * f1[j][c0 + c] in fp32 (the peel's mid
// half). f1_pairs holds F_A as BF16 [column pair][j][2].
void mid_tile(
    const uint16_t* diff, int64_t rows, const float* f1_t, const uint16_t* f1_pairs, int64_t c0,
    int64_t cols, float* acc) {
  int64_t r = 0;
#ifdef __AVX512BF16__
  constexpr int64_t kRowStep = 8;
  for (; r + kRowStep <= rows; r += kRowStep) {
    __m512 sums[kRowStep][2];
    for (int64_t i = 0; i < kRowStep; ++i) {
      sums[i][0] = _mm512_loadu_ps(acc + (r + i) * kRank);
      sums[i][1] = _mm512_loadu_ps(acc + (r + i) * kRank + kLanes);
    }
    for (int64_t c = 0; c < cols; c += 2) {
      const uint16_t* f = f1_pairs + (c0 + c) * kRank;
      const __m512bh f_lo = as_bf16_pairs(_mm512_loadu_si512(f));
      const __m512bh f_hi = as_bf16_pairs(_mm512_loadu_si512(f + 2 * kLanes));
      for (int64_t i = 0; i < kRowStep; ++i) {
        const __m512bh d = as_bf16_pairs(splat(load_pair(diff + (r + i) * kTileCols + c)));
        sums[i][0] = _mm512_dpbf16_ps(sums[i][0], d, f_lo);
        sums[i][1] = _mm512_dpbf16_ps(sums[i][1], d, f_hi);
      }
    }
    for (int64_t i = 0; i < kRowStep; ++i) {
      _mm512_storeu_ps(acc + (r + i) * kRank, sums[i][0]);
      _mm512_storeu_ps(acc + (r + i) * kRank + kLanes, sums[i][1]);
    }
  }
#endif
  for (; r < rows; ++r) {
    for (int64_t c = 0; c < cols; ++c) {
      const float d = bf16_value(diff[r * kTileCols + c]);
      for (int j = 0; j < kRank; ++j) acc[r * kRank + j] += d * f1_t[(c0 + c) * kRank + j];
    }
  }
}

// ---- Entry points ----

// Whether the integer noise dot is exact (and so the Blackwell atom's) for
// these lines: every E_B row and F_B column within kMaxLineSpan, the widest
// pair within kMaxDotSpan.
bool integer_noise_dot_exact(const torch::Tensor& e2, const torch::Tensor& f2) {
  TORCH_CHECK(e2.scalar_type() == at::kFloat8_e4m3fn && e2.is_contiguous() && e2.size(1) == kRank);
  TORCH_CHECK(f2.scalar_type() == at::kFloat8_e4m3fn && f2.is_contiguous() && f2.size(0) == kRank);
  const auto* e = reinterpret_cast<const uint8_t*>(e2.data_ptr());
  const auto* f = reinterpret_cast<const uint8_t*>(f2.data_ptr());
  const int64_t n = e2.size(0), k = f2.size(1);
  const auto widest = [](int64_t count, auto span_of) {
    return at::parallel_reduce(
        0, count, kLineGrain, 0,
        [&](int64_t begin, int64_t end, int acc) {
          for (int64_t i = begin; i < end; ++i) acc = std::max(acc, span_of(i));
          return acc;
        },
        [](int a, int b) { return std::max(a, b); });
  };
  const int e_span = widest(n, [&](int64_t r) { return q_span(e + r * kRank, 1); });
  const int f_span = widest(k, [&](int64_t c) { return q_span(f + c, k); });
  return e_span <= kMaxLineSpan && f_span <= kMaxLineSpan && e_span + f_span <= kMaxDotSpan;
}

torch::Tensor noise_lines(
    const std::string& key, const std::string& label, int64_t count, int64_t target_norm,
    int64_t sqrt_prec) {
  TORCH_CHECK(key.size() == kKeyBytes, "noise line key must be 32 bytes");
  TORCH_CHECK(label.size() == kLabelBytes, "noise line label must be 2 bytes");
  uint32_t key_words[8];
  std::memcpy(key_words, key.data(), kKeyBytes);
  const uint32_t label_word =
      static_cast<uint8_t>(label[0]) | (static_cast<uint32_t>(static_cast<uint8_t>(label[1])) << 8);
  const float scaled_norm = round_bf16(static_cast<float>(target_norm * sqrt_prec));
  const uint64_t prec_sq = static_cast<uint64_t>(sqrt_prec * sqrt_prec);

  auto out = torch::empty({count, kRank}, torch::dtype(at::kFloat8_e4m3fn));
  auto* lines = reinterpret_cast<uint8_t*>(out.data_ptr());
  at::parallel_for(0, count, kLineGrain, [&](int64_t begin, int64_t end) {
    uint32_t m[16] = {};
    uint8_t draws[kRank];
    int32_t values[kRank];
    for (int64_t i = begin; i < end; ++i) {
      // label || u32 LE(i), zero-padded to one block.
      const uint32_t index = static_cast<uint32_t>(i);
      m[0] = label_word | (index << 16);
      m[1] = index >> 16;
      keyed_block_hash(key_words, m, draws);
      uint64_t sumsq = 0;
      for (int t = 0; t < kRank; ++t) {
        const int32_t magnitude = (draws[t] & 0x7F) + 1;
        values[t] = (draws[t] >> 7) ? -magnitude : magnitude;
        sumsq += static_cast<uint64_t>(magnitude * magnitude);
      }
      const float norm = round_bf16(static_cast<float>(isqrt(sumsq * prec_sq)));
      const float scale = round_bf16(scaled_norm / norm);
      for (int t = 0; t < kRank; ++t) {
        lines[i * kRank + t] = e4m3_bits(round_bf16(static_cast<float>(values[t]) * scale));
      }
    }
  });
  return out;
}

// (B' (n, k) e4m3, the peel's mid half (n, R) bf16) of every B row, under the
// committed device's noise-dot window (`window_bits`) and strip rewrite.
std::vector<torch::Tensor> prepare_rows(
    const torch::Tensor& codes, const torch::Tensor& scales, const torch::Tensor& alpha,
    const torch::Tensor& beta, const torch::Tensor& l2, const torch::Tensor& e2,
    const torch::Tensor& f1, const torch::Tensor& f2, int64_t window_bits,
    double strip_g, double strip_g_small_rms) {
  TORCH_CHECK(codes.dim() == 2 && codes.scalar_type() == at::kChar && codes.is_contiguous());
  const int64_t n = codes.size(0), k = codes.size(1);
  TORCH_CHECK(k % kTileCols == 0, "k must be a multiple of the column tile");
  TORCH_CHECK(scales.scalar_type() == at::kBFloat16 && scales.is_contiguous());
  TORCH_CHECK(scales.size(0) == n && scales.size(1) == k / kPrequantBlock);
  for (const auto* row : {&alpha, &beta, &l2}) {
    TORCH_CHECK(row->scalar_type() == at::kBFloat16 && row->numel() == n && row->is_contiguous());
  }
  TORCH_CHECK(e2.scalar_type() == at::kFloat8_e4m3fn && e2.is_contiguous());
  TORCH_CHECK(e2.size(0) == n && e2.size(1) == kRank);
  for (const auto* f : {&f1, &f2}) {
    TORCH_CHECK(f->scalar_type() == at::kFloat8_e4m3fn && f->is_contiguous());
    TORCH_CHECK(f->size(0) == kRank && f->size(1) == k);
  }

  const auto* code_p = codes.data_ptr<int8_t>();
  const auto* scale_p = reinterpret_cast<const uint16_t*>(scales.data_ptr());
  const auto* alpha_p = reinterpret_cast<const uint16_t*>(alpha.data_ptr());
  const auto* beta_p = reinterpret_cast<const uint16_t*>(beta.data_ptr());
  const auto* l2_p = reinterpret_cast<const uint16_t*>(l2.data_ptr());
  const auto* e2_p = reinterpret_cast<const uint8_t*>(e2.data_ptr());
  const auto* f1_p = reinterpret_cast<const uint8_t*>(f1.data_ptr());
  const auto* f2_p = reinterpret_cast<const uint8_t*>(f2.data_ptr());
  // The untruncated Blackwell atom is the exact integer dot; any other cell
  // runs the windowed group sum.
  const int width = static_cast<int>(window_bits);
  const bool integer_dot = width == kBlackwellWindowBits && integer_noise_dot_exact(e2, f2);

  // F_B's columns in integer form (also as kLanes-column groups of pairs) or
  // as atom terms; F_A as (k, R) fp32 and as BF16 [column pair][j][2].
  std::vector<int16_t> f2_ints(static_cast<size_t>(integer_dot ? k * kRank : 0));
  std::vector<float> f2_scale(static_cast<size_t>(k));
  std::vector<int32_t> f2_pairs(static_cast<size_t>(integer_dot ? k * kPairs : 0));
  std::vector<E4m3Term> f2_terms(static_cast<size_t>(integer_dot ? 0 : k * kRank));
  std::vector<float> f1_t(static_cast<size_t>(k * kRank));
  std::vector<uint16_t> f1_pairs(static_cast<size_t>(k * kRank));
  at::parallel_for(0, k, kTileCols, [&](int64_t begin, int64_t end) {
    for (int64_t c = begin; c < end; ++c) {
      if (integer_dot) {
        int16_t* ints = f2_ints.data() + c * kRank;
        f2_scale[c] = integer_line(f2_p + c, k, ints);
        int32_t* group = f2_pairs.data() + c / kLanes * (kPairs * kLanes) + c % kLanes;
        for (int p = 0; p < kPairs; ++p) group[p * kLanes] = load_pair(ints + 2 * p);
      } else {
        for (int j = 0; j < kRank; ++j) f2_terms[c * kRank + j] = e4m3_term(f2_p[j * k + c]);
      }
      for (int j = 0; j < kRank; ++j) {
        const float value = kE4m3Values[f1_p[j * k + c]];
        f1_t[c * kRank + j] = value;
        f1_pairs[(c / 2 * kRank + j) * 2 + c % 2] = bf16_bits(value);
      }
    }
  });

  auto b_prime = torch::empty({n, k}, torch::dtype(at::kFloat8_e4m3fn));
  auto mid = torch::empty({n, kRank}, torch::dtype(at::kBFloat16));
  auto* bp_p = reinterpret_cast<uint8_t*>(b_prime.data_ptr());
  auto* mid_p = reinterpret_cast<uint16_t*>(mid.data_ptr());
  const int64_t row_tiles = (n + kTileRows - 1) / kTileRows;

  at::parallel_for(0, row_tiles, 1, [&](int64_t begin, int64_t end) {
    std::vector<int32_t> sums(static_cast<size_t>(kTileRows * kTileCols));
    std::vector<float> noise(static_cast<size_t>(kTileRows * kTileCols));
    std::vector<uint16_t> diff(static_cast<size_t>(kTileRows * kTileCols));
    int16_t e_ints[kTileRows * kRank];
    E4m3Term e_terms[kTileRows * kRank];
    float e_scale[kTileRows];
    float mid_acc[kTileRows * kRank];
    for (int64_t t = begin; t < end; ++t) {
      const int64_t r0 = t * kTileRows;
      const int64_t rows = std::min(kTileRows, n - r0);
      for (int64_t i = 0; i < rows; ++i) {
        const uint8_t* line = e2_p + (r0 + i) * kRank;
        if (integer_dot) {
          e_scale[i] = integer_line(line, 1, e_ints + i * kRank);
        } else {
          for (int j = 0; j < kRank; ++j) e_terms[i * kRank + j] = e4m3_term(line[j]);
        }
      }
      std::fill(mid_acc, mid_acc + rows * kRank, 0.0f);
      for (int64_t c0 = 0; c0 < k; c0 += kTileCols) {
        if (integer_dot) {
          noise_tile(e_ints, rows, f2_ints.data(), f2_pairs.data(), c0, kTileCols, sums.data());
          integer_noise_values(
              sums.data(), rows, e_scale, f2_scale.data() + c0, kTileCols, noise.data());
        } else {
          windowed_noise_tile(e_terms, rows, f2_terms.data(), c0, kTileCols, width, noise.data());
        }
        for (int64_t i = 0; i < rows; ++i) {
          const int64_t r = r0 + i;
          const float a = bf16_value(alpha_p[r]);
          const Strip strip = strip_of(a, bf16_value(l2_p[r]), strip_g, strip_g_small_rms);
          prepare_row(
              a, bf16_value(beta_p[r]), strip,
              code_p + r * k + c0, scale_p + r * (k / kPrequantBlock) + c0 / kPrequantBlock,
              noise.data() + i * kTileCols, kTileCols, bp_p + r * k + c0,
              diff.data() + i * kTileCols);
        }
        mid_tile(diff.data(), rows, f1_t.data(), f1_pairs.data(), c0, kTileCols, mid_acc);
      }
      for (int64_t i = 0; i < rows * kRank; ++i) mid_p[r0 * kRank + i] = bf16_bits(mid_acc[i]);
    }
  });
  return {b_prime, mid};
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("noise_lines", &noise_lines, "(count, R) e4m3 keyed-BLAKE3 noise lines");
  m.def("prepare_rows", &prepare_rows, "(B', the peel's mid half) of every B row");
}
