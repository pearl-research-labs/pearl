//! CPU B preparation: one job's B side of a layer computed on the host, for a
//! miner that prepares a layer's next B operand off the GPU.
//!
//! [`noise_lines`] draws one factor's keyed noise lines ([`sample_line`]).
//! [`prepare_rows`] builds, for every B row, `B'` and the mid half of the B
//! peel, `(beta (.) E_B@F_B - B') @ F_A^T`, from the row's prequant codes and
//! scales, its `alpha`, `beta` and `l2`, and the job's lines.
//!
//! `B'` is bit-identical to `Fp8E4M3Quant::noisy_quantize` on the committed
//! device: the noise dot is that device's FP8 atom (an exact integer dot when
//! the Blackwell atom truncates nothing, else its windowed group sum), the mix
//! is one fp32 FMA, and the strip rewrite is the verifier's. The mid half
//! accumulates in fp32, so it is a tolerance surface.
//!
//! On x86-64 the row chain (AVX-512 F/DQ), the integer noise dot (VNNI) and the
//! mid half (BF16) take AVX-512 paths the CPU reports at run time; the scalar
//! paths compute the same `B'`.

use anyhow::{Result, ensure};
use plonky2_maybe_rayon::*;

use crate::v4::api::dtype::fp8_e4m3_to_f32;
use crate::v4::api::noise::sample_line;
use crate::v4::api::prequant::BLOCK_SIZE;
use crate::v4::api::primitives::Hash256;

/// Entries per noise line.
pub const RANK: usize = 32;
/// [`prepare_rows`] works on column tiles of this many columns; `k` must be a
/// multiple.
pub const TILE_COLS: usize = 256;
const TILE_ROWS: usize = 16;
const LINE_GRAIN: usize = 256;
/// Columns per AVX-512 register of fp32 or int32.
const LANES: usize = 16;
const PAIRS: usize = RANK / 2;
// The vector row chain loads one register of codes as two whole scale blocks.
const _: () = assert!(LANES == 2 * BLOCK_SIZE);

/// The committed device's arithmetic for [`prepare_rows`].
#[derive(Clone, Copy, Debug)]
pub struct DeviceArithmetic {
    /// Significand bits of the FP8 noise atom's window (26 on Blackwell, 14 on
    /// Hopper).
    pub window_bits: u16,
    /// The strip rewrite's `g` for a row with `alpha * l2 >= 8`, and below it.
    pub strip_g: (f32, f32),
}

/// One layer's B rows and one job's noise lines (BF16 as `u16`, E4M3 as `u8`,
/// all row-major).
pub struct BRows<'a> {
    /// `(n, k)` prequant codes.
    pub codes: &'a [i8],
    /// `(n, k / BLOCK_SIZE)` BF16 block scales.
    pub scales: &'a [u16],
    /// Per-row BF16 `alpha`, `beta` and floored `l2`, `n` each.
    pub alpha: &'a [u16],
    pub beta: &'a [u16],
    pub l2: &'a [u16],
    /// `(n, RANK)` E_B.
    pub e_b: &'a [u8],
    /// `(RANK, k)` F_A and F_B.
    pub f_a: &'a [u8],
    pub f_b: &'a [u8],
}

/// Draws `lines.len() / RANK` noise lines of one `(side, factor)` address
/// under the noise-line key `key`: line `i` is [`sample_line`] at index `i`.
pub fn noise_lines(key: &Hash256, side: u8, factor: u8, lines: &mut [u8]) -> Result<()> {
    ensure!(lines.len().is_multiple_of(RANK), "noise lines must be whole lines of {RANK}");
    ensure!(lines.len() / RANK <= u32::MAX as usize + 1, "noise line index overflows u32");
    lines
        .par_chunks_mut(RANK * LINE_GRAIN)
        .enumerate()
        .for_each(|(chunk, lines)| {
            for (i, line) in lines.chunks_mut(RANK).enumerate() {
                sample_line(key, side, factor, (chunk * LINE_GRAIN + i) as u32, line);
            }
        });
    Ok(())
}

/// Writes every B row's `B'` (`(n, k)` E4M3) and the mid half of its peel
/// (`(n, RANK)` BF16).
pub fn prepare_rows(rows: &BRows, arithmetic: DeviceArithmetic, b_prime: &mut [u8], mid: &mut [u16]) -> Result<()> {
    let isa = Isa::detect();
    let integer_dot = arithmetic.window_bits == BLACKWELL_WINDOW_BITS && integer_noise_dot_exact(rows.e_b, rows.f_b);
    prepare_rows_with(isa, integer_dot, rows, arithmetic, b_prime, mid)
}

// ---- Number formats ----

const BF16_NAN: u16 = 0x7FC0;
const QUANT_MAX: f32 = 448.0;
// E4M3 from fp32 in [-448, 448], round to nearest even: below 2^-6 through the
// denormal mask, else re-bias the exponent and round.
const E4M3_DENORM_MASK: u32 = 141 << 23;
const E4M3_MIN_NORMAL: u32 = 121 << 23;
const E4M3_NORMAL_BIAS: u32 = ((7i32 - 127) as u32).wrapping_shl(23).wrapping_add(0x7FFFF);
const E4M3_NAN: u8 = 0x7F;

fn bf16_bits(f: f32) -> u16 {
    if f.is_nan() {
        return BF16_NAN;
    }
    let u = f.to_bits();
    (u.wrapping_add(0x7FFF + ((u >> 16) & 1)) >> 16) as u16
}

fn bf16_value(b: u16) -> f32 {
    f32::from_bits(u32::from(b) << 16)
}

fn round_bf16(f: f32) -> f32 {
    bf16_value(bf16_bits(f))
}

/// The verifier's E4M3 cast: round to nearest even, saturating to ±448 (a NaN
/// stays a NaN).
fn e4m3_bits(f: f32) -> u8 {
    let f = if f < -QUANT_MAX { -QUANT_MAX } else { f };
    let f = if QUANT_MAX < f { QUANT_MAX } else { f };
    let signed = f.to_bits();
    let sign = signed & 0x8000_0000;
    let bits = signed ^ sign;
    let denorm = (f32::from_bits(bits) + f32::from_bits(E4M3_DENORM_MASK))
        .to_bits()
        .wrapping_sub(E4M3_DENORM_MASK);
    let normal = bits.wrapping_add(E4M3_NORMAL_BIAS).wrapping_add((bits >> 20) & 1) >> 20;
    let magnitude = if bits < E4M3_MIN_NORMAL { denorm } else { normal };
    (magnitude | (sign >> 24)) as u8
}

fn e4m3_value(b: u8) -> f32 {
    if b & 0x7F == E4M3_NAN { f32::NAN } else { fp8_e4m3_to_f32(b) }
}

/// float64 -> fp32 rounded towards zero.
fn toward_zero_f32(v: f64) -> f32 {
    let nearest = v as f32;
    if f64::from(nearest).abs() > v.abs() {
        f32::from_bits(nearest.to_bits() - 1)
    } else {
        nearest
    }
}

// ---- Strip rewrite ----

const STRIP_SMALL_RMS: f32 = 8.0;
const STRIP_EXP_OFFSET: u32 = 9;
const STRIP_COIN_BLOCK: usize = 4;
const STRIP_COIN_SHIFT: u32 = 8;
const STRIP_COIN_THRESHOLD: u32 = 112;

/// A row's strip: the magnitude bound `|e4m3(g * s)|` and the safe exponent
/// already in code position, for `s = alpha * l2`.
#[derive(Clone, Copy)]
struct Strip {
    bound: u32,
    code: u32,
}

fn strip_of(alpha: f32, l2: f32, (strip_g, strip_g_small_rms): (f32, f32)) -> Strip {
    let s = alpha * l2;
    let g = if s < STRIP_SMALL_RMS { strip_g_small_rms } else { strip_g };
    let field = ((s.to_bits() >> 23) & 0xFF).wrapping_sub(127 + STRIP_EXP_OFFSET - 7);
    Strip {
        bound: u32::from(e4m3_bits(g * s) & 0x7F),
        code: ((field & 0xF) << 3) | 1,
    }
}

fn coin_passes(mixed: f32) -> bool {
    ((mixed.to_bits() >> STRIP_COIN_SHIFT) & 0x7F) < STRIP_COIN_THRESHOLD
}

fn rewrite(q: u8, strip: Strip, coin: bool) -> u8 {
    if coin && u32::from(q & 0x7F) < strip.bound {
        (q & 0x87) | strip.code as u8
    } else {
        q
    }
}

// ---- Integer noise dot ----
// An E4M3 value is sig * 2^(q - E4M3_UNIT_EXP) with |sig| <= 15 and
// q = max(exponent, 1). A line whose q span is at most MAX_LINE_SPAN holds
// exactly as int16 sig << (q - q_min) times 2^(q_min - E4M3_UNIT_EXP); a row and
// column whose spans sum to at most MAX_DOT_SPAN dot exactly in int32
// (32 * 15^2 * 2^18 < 2^31). MAX_DOT_SPAN is within the Blackwell atom's
// untruncated span (25 fractional bits less the 6 of an E4M3 product), so that
// dot is the atom's.

const E4M3_UNIT_EXP: i32 = 10;
const E4M3_MAX_Q: i32 = 15;
const BLACKWELL_WINDOW_BITS: u16 = 26;
const MAX_LINE_SPAN: i32 = 11;
const MAX_DOT_SPAN: i32 = 18;

fn e4m3_q(b: u8) -> i32 {
    i32::from((b >> 3) & 0xF).max(1)
}

/// The q span of the RANK values `line[j * stride]`; a NaN spans everything.
fn q_span(line: &[u8], stride: usize) -> i32 {
    let (mut lo, mut hi) = (E4M3_MAX_Q, 1);
    for j in 0..RANK {
        let b = line[j * stride];
        if b & 0x7F == E4M3_NAN {
            return i32::MAX / 2;
        }
        lo = lo.min(e4m3_q(b));
        hi = hi.max(e4m3_q(b));
    }
    hi - lo
}

/// `line[j * stride] == ints[j] * (the returned power of two)`, for a line
/// within MAX_LINE_SPAN.
fn integer_line(line: &[u8], stride: usize, ints: &mut [i16]) -> f32 {
    let lo = (0..RANK).map(|j| e4m3_q(line[j * stride])).min().unwrap_or(E4M3_MAX_Q);
    for j in 0..RANK {
        let b = line[j * stride];
        let (exponent, mantissa) = ((b >> 3) & 0xF, i16::from(b & 0x7));
        let magnitude = (if exponent != 0 { mantissa | 8 } else { mantissa }) << (e4m3_q(b) - lo);
        ints[j] = if b & 0x80 != 0 { -magnitude } else { magnitude };
    }
    f32::from_bits(((lo - E4M3_UNIT_EXP + 127) as u32) << 23)
}

fn int_dot(e: &[i16], f: &[i16]) -> i32 {
    e[..RANK]
        .iter()
        .zip(&f[..RANK])
        .map(|(&e, &f)| i32::from(e) * i32::from(f))
        .sum()
}

/// Whether the integer noise dot is exact (and so the Blackwell atom's) for
/// these lines: every E_B row and F_B column within MAX_LINE_SPAN, the widest
/// pair within MAX_DOT_SPAN.
fn integer_noise_dot_exact(e_b: &[u8], f_b: &[u8]) -> bool {
    let k = f_b.len() / RANK;
    let e_span = e_b
        .par_chunks(RANK * LINE_GRAIN)
        .map(|lines| lines.chunks(RANK).map(|line| q_span(line, 1)).max().unwrap_or(0))
        .max();
    let f_span = (0..k).into_par_iter().map(|c| q_span(&f_b[c..], k)).max();
    let (e_span, f_span) = (e_span.unwrap_or(0), f_span.unwrap_or(0));
    e_span <= MAX_LINE_SPAN && f_span <= MAX_LINE_SPAN && e_span + f_span <= MAX_DOT_SPAN
}

// ---- Windowed noise dot (the verifier's windowed group sum) ----
// One K = RANK FP8 atom from +0: each product sig * 2^(exp - 23), sig =
// (sa * sb) << 17, is truncated to the window (`width` significand bits,
// anchored at the largest exponent), the terms are summed exactly, and the sum
// is truncated to `width` bits. A zero result is +0.

const GFLOAT_BITS: i32 = 24;
const F32_MIN_EXPONENT: i32 = -126;

/// One int16 lane of the vector windowed dot: `sig` the low byte, `q` the high.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct E4m3Term {
    /// Signed significand.
    sig: i8,
    q: i8,
}

fn e4m3_term(b: u8) -> E4m3Term {
    let (exponent, mantissa) = ((b >> 3) & 0xF, (b & 0x7) as i8);
    let sig = if exponent != 0 { mantissa | 8 } else { mantissa };
    E4m3Term {
        sig: if b & 0x80 != 0 { -sig } else { sig },
        q: e4m3_q(b) as i8,
    }
}

fn windowed_dot(e: &[E4m3Term], f: &[E4m3Term], width: i32) -> f32 {
    let (e, f) = (&e[..RANK], &f[..RANK]);
    let exp = |j: usize| i32::from(e[j].q) + i32::from(f[j].q) - 14;
    let Some(max_exp) = (0..RANK).filter(|&j| e[j].sig != 0 && f[j].sig != 0).map(exp).max() else {
        return 0.0;
    };
    let diff = (width - GFLOAT_BITS).unsigned_abs();
    let mut acc = 0i64;
    for j in 0..RANK {
        let product = i32::from(e[j].sig) * i32::from(f[j].sig);
        let shift = max_exp - exp(j);
        if product == 0 || shift >= 32 {
            continue;
        }
        let sig = product.unsigned_abs() << 17;
        let shifted = if width < GFLOAT_BITS { sig >> diff } else { sig << diff };
        let aligned = i64::from(shifted >> shift);
        acc += if product < 0 { -aligned } else { aligned };
    }
    window_round(acc, max_exp, width)
}

/// The windowed sum `acc` of terms anchored at `max_exp`, truncated to
/// `width` bits, as fp32.
fn window_round(acc: i64, max_exp: i32, width: i32) -> f32 {
    if acc == 0 {
        return 0.0;
    }
    let diff = (width - GFLOAT_BITS).unsigned_abs();
    let mut magnitude = acc.unsigned_abs();
    let bits = (64 - magnitude.leading_zeros()) as i32;
    let mut exponent = max_exp + bits - width;
    magnitude = if bits > width {
        magnitude >> (bits - width)
    } else {
        magnitude << (width - bits)
    };
    if exponent < F32_MIN_EXPONENT {
        let shift = F32_MIN_EXPONENT - exponent;
        magnitude = if shift >= 32 { 0 } else { magnitude >> shift };
        exponent = F32_MIN_EXPONENT;
    }
    magnitude = if width < GFLOAT_BITS {
        magnitude << diff
    } else {
        magnitude >> diff
    };
    if magnitude == 0 {
        return 0.0;
    }
    // At most 24 significand bits at or above 2^-149: exact in fp32.
    let value = (magnitude as f64 * f64::from_bits(((exponent - 23 + 1023) as u64) << 52)) as f32;
    if acc < 0 { -value } else { value }
}

// ---- Instruction sets ----

// Only the x86_64 `vector` dispatch reads the fields.
#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
#[derive(Clone, Copy, Debug)]
struct Isa {
    /// AVX-512 F and DQ: the row chain.
    avx512: bool,
    /// AVX-512 VNNI: the integer noise dot.
    vnni: bool,
    /// AVX-512 BF16: the mid half.
    bf16: bool,
}

impl Isa {
    #[cfg(any(test, not(target_arch = "x86_64")))]
    const SCALAR: Self = Self {
        avx512: false,
        vnni: false,
        bf16: false,
    };

    fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            let avx512 = std::arch::is_x86_feature_detected!("avx512f") && std::arch::is_x86_feature_detected!("avx512dq");
            Self {
                avx512,
                vnni: avx512 && std::arch::is_x86_feature_detected!("avx512vnni"),
                bf16: avx512 && std::arch::is_x86_feature_detected!("avx512bf16"),
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        Self::SCALAR
    }
}

// ---- Row chain ----

/// One row's scales and strip.
#[derive(Clone, Copy)]
struct RowScales {
    alpha: f32,
    beta: f32,
    strip: Strip,
}

/// One row's cells of a column tile (whole coin groups) under the atom's fp32
/// noise: `B'` and the peel's mid input `diff = bf16(beta * bf16(noise) - B')`.
fn prepare_row(isa: Isa, row: RowScales, code: &[i8], scale: &[u16], noises: &[f32], bp: &mut [u8], diff: &mut [u16]) {
    let cols = bp.len();
    assert!(code.len() == cols && noises.len() == cols && diff.len() == cols && scale.len() * BLOCK_SIZE == cols);
    let c = vector::prepare_row(isa, row, code, scale, noises, bp, diff);
    let mut coin = false;
    for c in c..cols {
        let coefficient = row.alpha * bf16_value(scale[c / BLOCK_SIZE]);
        let mixed = row.beta.mul_add(noises[c], coefficient * f32::from(code[c]));
        if c % STRIP_COIN_BLOCK == 0 {
            coin = coin_passes(mixed);
        }
        let q = rewrite(e4m3_bits(mixed), row.strip, coin);
        bp[c] = q;
        diff[c] = bf16_bits(row.beta.mul_add(round_bf16(noises[c]), -e4m3_value(q)));
    }
}

/// `out[r][c] = sum_j e_ints[r][j] * f_ints[c0 + c][j]` for a column tile,
/// exact in int32. `f_pairs` holds F_B's columns in groups of LANES as
/// `[pair][lane]` int16 pairs.
fn noise_tile(isa: Isa, e_ints: &[i16], rows: usize, f_ints: &[i16], f_pairs: &[u32], c0: usize, out: &mut [i32]) {
    let r = vector::noise_tile(isa, e_ints, rows, f_pairs, c0, out);
    for r in r..rows {
        for c in 0..TILE_COLS {
            out[r * TILE_COLS + c] = int_dot(&e_ints[r * RANK..], &f_ints[(c0 + c) * RANK..]);
        }
    }
}

/// `out[r][c] = windowed_dot(E_B row r, F_B column c0 + c)` for a column tile.
/// `f_terms` holds F_B as `(RANK, k)` terms.
fn windowed_tile(isa: Isa, e_terms: &[E4m3Term], rows: usize, f_terms: &[E4m3Term], c0: usize, width: i32, out: &mut [f32]) {
    let k = f_terms.len() / RANK;
    let r = vector::windowed_tile(isa, e_terms, rows, f_terms, c0, width, out);
    let mut column = [E4m3Term::default(); RANK];
    for c in 0..TILE_COLS {
        for (j, term) in column.iter_mut().enumerate() {
            *term = f_terms[j * k + c0 + c];
        }
        for i in r..rows {
            out[i * TILE_COLS + c] = windowed_dot(&e_terms[i * RANK..], &column, width);
        }
    }
}

/// The integer sums of [`noise_tile`] as the atom's fp32 noise, rounded towards
/// zero (scaling by the powers of two row and column scale is exact).
/// `f_scale` holds the tile's column scales.
fn integer_noise_values(isa: Isa, sums: &[i32], rows: usize, e_scale: &[f32], f_scale: &[f32], out: &mut [f32]) {
    let r = vector::integer_noise_values(isa, sums, rows, e_scale, f_scale, out);
    for i in r..rows {
        for c in 0..TILE_COLS {
            let exact = f64::from(sums[i * TILE_COLS + c]) * f64::from(e_scale[i]) * f64::from(f_scale[c]);
            out[i * TILE_COLS + c] = toward_zero_f32(exact);
        }
    }
}

/// `acc[r][j] += sum_c diff[r][c] * F_A[j][c0 + c]` in fp32 over a column tile.
/// `f_a_t` holds F_A as `(k, RANK)` fp32, `f_a_pairs` as BF16
/// `[column pair][j][2]`.
fn mid_tile(isa: Isa, diff: &[u16], rows: usize, f_a_t: &[f32], f_a_pairs: &[u16], c0: usize, acc: &mut [f32]) {
    let r = vector::mid_tile(isa, diff, rows, f_a_pairs, c0, acc);
    for r in r..rows {
        for c in 0..TILE_COLS {
            let d = bf16_value(diff[r * TILE_COLS + c]);
            for j in 0..RANK {
                acc[r * RANK + j] += d * f_a_t[(c0 + c) * RANK + j];
            }
        }
    }
}

fn load_pair(pair: &[i16]) -> u32 {
    u32::from(pair[0] as u16) | (u32::from(pair[1] as u16) << 16)
}

fn prepare_rows_with(
    isa: Isa,
    integer_dot: bool,
    rows: &BRows,
    arithmetic: DeviceArithmetic,
    b_prime: &mut [u8],
    mid: &mut [u16],
) -> Result<()> {
    let n = rows.alpha.len();
    let k = rows.f_b.len() / RANK;
    let width = i32::from(arithmetic.window_bits);
    ensure!(
        k > 0 && k.is_multiple_of(TILE_COLS),
        "k={k} must be a positive multiple of {TILE_COLS}"
    );
    ensure!((2..=30).contains(&width), "noise window of {width} bits is not supported");
    ensure!(
        rows.codes.len() == n * k && b_prime.len() == n * k,
        "codes and B' must be (n, k)"
    );
    ensure!(
        rows.scales.len() == n * k / BLOCK_SIZE,
        "scales must be (n, k / {BLOCK_SIZE})"
    );
    ensure!(
        rows.beta.len() == n && rows.l2.len() == n,
        "alpha, beta and l2 must hold one value per row"
    );
    ensure!(
        rows.e_b.len() == n * RANK && mid.len() == n * RANK,
        "E_B and the mid half must be (n, {RANK})"
    );
    ensure!(
        rows.f_a.len() == RANK * k && rows.f_b.len() == RANK * k,
        "F_A and F_B must be ({RANK}, k)"
    );

    // F_B's columns in integer form (also as LANES-column groups of pairs) or
    // as (RANK, k) atom terms; F_A as (k, RANK) fp32 and as BF16
    // [column pair][j][2].
    let mut f_b_ints = vec![0i16; if integer_dot { k * RANK } else { 0 }];
    let mut f_b_scale = vec![0f32; k];
    let mut f_b_pairs = vec![0u32; if integer_dot { k * PAIRS } else { 0 }];
    let mut f_b_terms = vec![E4m3Term::default(); if integer_dot { 0 } else { k * RANK }];
    let mut f_a_t = vec![0f32; k * RANK];
    let mut f_a_pairs = vec![0u16; k * RANK];
    // Column tiles in parallel; LANES and column pairs divide TILE_COLS, so
    // every layout's tile is one contiguous chunk.
    if integer_dot {
        f_b_ints
            .par_chunks_mut(TILE_COLS * RANK)
            .zip(f_b_scale.par_chunks_mut(TILE_COLS))
            .zip(f_b_pairs.par_chunks_mut(TILE_COLS * PAIRS))
            .enumerate()
            .for_each(|(tile, ((ints, scale), pairs))| {
                for c in 0..TILE_COLS {
                    let ints = &mut ints[c * RANK..(c + 1) * RANK];
                    scale[c] = integer_line(&rows.f_b[tile * TILE_COLS + c..], k, ints);
                    for p in 0..PAIRS {
                        pairs[c / LANES * (PAIRS * LANES) + p * LANES + c % LANES] = load_pair(&ints[2 * p..]);
                    }
                }
            });
    } else {
        f_b_terms
            .par_chunks_mut(k)
            .zip(rows.f_b.par_chunks(k))
            .for_each(|(terms, line)| {
                for (term, &b) in terms.iter_mut().zip(line) {
                    *term = e4m3_term(b);
                }
            });
    }
    f_a_t
        .par_chunks_mut(TILE_COLS * RANK)
        .zip(f_a_pairs.par_chunks_mut(TILE_COLS * RANK))
        .enumerate()
        .for_each(|(tile, (t, pairs))| {
            for c in 0..TILE_COLS {
                for j in 0..RANK {
                    let value = e4m3_value(rows.f_a[j * k + tile * TILE_COLS + c]);
                    t[c * RANK + j] = value;
                    pairs[(c / 2 * RANK + j) * 2 + c % 2] = bf16_bits(value);
                }
            }
        });

    b_prime
        .par_chunks_mut(TILE_ROWS * k)
        .zip(mid.par_chunks_mut(TILE_ROWS * RANK))
        .enumerate()
        .for_each(|(tile, (bp, mid))| {
            let r0 = tile * TILE_ROWS;
            let tile_rows = bp.len() / k;
            let mut sums = vec![0i32; TILE_ROWS * TILE_COLS];
            let mut noise = vec![0f32; TILE_ROWS * TILE_COLS];
            let mut diff = vec![0u16; TILE_ROWS * TILE_COLS];
            let mut e_ints = [0i16; TILE_ROWS * RANK];
            let mut e_terms = [E4m3Term::default(); TILE_ROWS * RANK];
            let mut e_scale = [0f32; TILE_ROWS];
            let mut mid_acc = [0f32; TILE_ROWS * RANK];
            for i in 0..tile_rows {
                let line = &rows.e_b[(r0 + i) * RANK..(r0 + i + 1) * RANK];
                if integer_dot {
                    e_scale[i] = integer_line(line, 1, &mut e_ints[i * RANK..(i + 1) * RANK]);
                } else {
                    for j in 0..RANK {
                        e_terms[i * RANK + j] = e4m3_term(line[j]);
                    }
                }
            }
            let row_scales: Vec<RowScales> = (r0..r0 + tile_rows)
                .map(|r| {
                    let alpha = bf16_value(rows.alpha[r]);
                    RowScales {
                        alpha,
                        beta: bf16_value(rows.beta[r]),
                        strip: strip_of(alpha, bf16_value(rows.l2[r]), arithmetic.strip_g),
                    }
                })
                .collect();
            for c0 in (0..k).step_by(TILE_COLS) {
                if integer_dot {
                    noise_tile(isa, &e_ints, tile_rows, &f_b_ints, &f_b_pairs, c0, &mut sums);
                    let f_scale = &f_b_scale[c0..c0 + TILE_COLS];
                    integer_noise_values(isa, &sums, tile_rows, &e_scale, f_scale, &mut noise);
                } else {
                    windowed_tile(isa, &e_terms, tile_rows, &f_b_terms, c0, width, &mut noise);
                }
                for (i, row) in row_scales.iter().enumerate() {
                    let r = r0 + i;
                    let cells = i * TILE_COLS..(i + 1) * TILE_COLS;
                    prepare_row(
                        isa,
                        *row,
                        &rows.codes[r * k + c0..r * k + c0 + TILE_COLS],
                        &rows.scales[(r * k + c0) / BLOCK_SIZE..(r * k + c0 + TILE_COLS) / BLOCK_SIZE],
                        &noise[cells.clone()],
                        &mut bp[i * k + c0..i * k + c0 + TILE_COLS],
                        &mut diff[cells],
                    );
                }
                mid_tile(isa, &diff, tile_rows, &f_a_t, &f_a_pairs, c0, &mut mid_acc);
            }
            for (out, &acc) in mid.iter_mut().zip(&mid_acc) {
                *out = bf16_bits(acc);
            }
        });
    Ok(())
}

#[cfg(target_arch = "x86_64")]
mod vector {
    //! Runs each AVX-512 path when the CPU has the instruction sets it needs
    //! and returns how many leading rows (cells, for `prepare_row`) it
    //! completed; the caller's scalar loop computes the rest.

    use super::{E4m3Term, Isa, RowScales, simd};

    pub(super) fn prepare_row(
        isa: Isa,
        row: RowScales,
        code: &[i8],
        scale: &[u16],
        noises: &[f32],
        bp: &mut [u8],
        diff: &mut [u16],
    ) -> usize {
        if isa.avx512 {
            // SAFETY: the CPU reports AVX-512 F and DQ.
            unsafe { simd::prepare_row(row, code, scale, noises, bp, diff) }
        } else {
            0
        }
    }

    pub(super) fn noise_tile(isa: Isa, e_ints: &[i16], rows: usize, f_pairs: &[u32], c0: usize, out: &mut [i32]) -> usize {
        if isa.vnni {
            // SAFETY: the CPU reports AVX-512 F and VNNI.
            unsafe { simd::noise_tile(e_ints, rows, f_pairs, c0, out) }
        } else {
            0
        }
    }

    pub(super) fn windowed_tile(
        isa: Isa,
        e_terms: &[E4m3Term],
        rows: usize,
        f_terms: &[E4m3Term],
        c0: usize,
        width: i32,
        out: &mut [f32],
    ) -> usize {
        if isa.avx512 {
            // SAFETY: the CPU reports AVX-512 F and DQ.
            unsafe { simd::windowed_tile(e_terms, rows, f_terms, c0, width, out) }
        } else {
            0
        }
    }

    pub(super) fn integer_noise_values(
        isa: Isa,
        sums: &[i32],
        rows: usize,
        e_scale: &[f32],
        f_scale: &[f32],
        out: &mut [f32],
    ) -> usize {
        if isa.avx512 {
            // SAFETY: the CPU reports AVX-512 F and DQ.
            unsafe { simd::integer_noise_values(sums, rows, e_scale, f_scale, out) }
        } else {
            0
        }
    }

    pub(super) fn mid_tile(isa: Isa, diff: &[u16], rows: usize, f_a_pairs: &[u16], c0: usize, acc: &mut [f32]) -> usize {
        if isa.bf16 {
            // SAFETY: the CPU reports AVX-512 F and BF16.
            unsafe { simd::mid_tile(diff, rows, f_a_pairs, c0, acc) }
        } else {
            0
        }
    }
}

#[cfg(not(target_arch = "x86_64"))]
mod vector {
    //! This architecture has no vector paths: every function reports that it
    //! completed nothing, so the caller's scalar loop computes everything.

    use super::{E4m3Term, Isa, RowScales};

    pub(super) fn prepare_row(_: Isa, _: RowScales, _: &[i8], _: &[u16], _: &[f32], _: &mut [u8], _: &mut [u16]) -> usize {
        0
    }

    pub(super) fn noise_tile(_: Isa, _: &[i16], _: usize, _: &[u32], _: usize, _: &mut [i32]) -> usize {
        0
    }

    pub(super) fn windowed_tile(_: Isa, _: &[E4m3Term], _: usize, _: &[E4m3Term], _: usize, _: i32, _: &mut [f32]) -> usize {
        0
    }

    pub(super) fn integer_noise_values(_: Isa, _: &[i32], _: usize, _: &[f32], _: &[f32], _: &mut [f32]) -> usize {
        0
    }

    pub(super) fn mid_tile(_: Isa, _: &[u16], _: usize, _: &[u16], _: usize, _: &mut [f32]) -> usize {
        0
    }
}

#[cfg(target_arch = "x86_64")]
mod simd {
    //! The AVX-512 paths of the row chain, the integer noise dot and the mid
    //! half; each computes the scalar code's cells on LANES at once.

    use std::arch::x86_64::*;

    use super::*;

    #[inline]
    #[target_feature(enable = "avx512f")]
    fn splat(v: u32) -> __m512i {
        _mm512_set1_epi32(v as i32)
    }

    #[inline]
    #[target_feature(enable = "avx512f,avx512dq")]
    fn join(lo: __m256, hi: __m256) -> __m512 {
        _mm512_insertf32x8::<1>(_mm512_castps256_ps512(lo), hi)
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    fn round_bf16_v(x: __m512) -> __m512 {
        let u = _mm512_castps_si512(x);
        let lsb = _mm512_and_si512(_mm512_srli_epi32::<16>(u), splat(1));
        let rounded = _mm512_and_si512(_mm512_add_epi32(u, _mm512_add_epi32(lsb, splat(0x7FFF))), splat(0xFFFF_0000));
        let nan = _mm512_cmp_ps_mask::<_CMP_UNORD_Q>(x, x);
        _mm512_castsi512_ps(_mm512_mask_blend_epi32(nan, rounded, splat(u32::from(BF16_NAN) << 16)))
    }

    /// e4m3_bits on LANES, as int32.
    #[inline]
    #[target_feature(enable = "avx512f")]
    fn e4m3_bits_v(f: __m512) -> __m512i {
        // MIN/MAX return their second operand on NaN, so a NaN stays a NaN.
        let f = _mm512_max_ps(_mm512_set1_ps(-QUANT_MAX), _mm512_min_ps(_mm512_set1_ps(QUANT_MAX), f));
        let signed = _mm512_castps_si512(f);
        let sign = _mm512_and_si512(signed, splat(0x8000_0000));
        let bits = _mm512_xor_si512(signed, sign);
        let denorm_mask = _mm512_castsi512_ps(splat(E4M3_DENORM_MASK));
        let denorm = _mm512_sub_epi32(
            _mm512_castps_si512(_mm512_add_ps(_mm512_castsi512_ps(bits), denorm_mask)),
            splat(E4M3_DENORM_MASK),
        );
        let mant_odd = _mm512_and_si512(_mm512_srli_epi32::<20>(bits), splat(1));
        let normal = _mm512_srli_epi32::<20>(_mm512_add_epi32(_mm512_add_epi32(bits, splat(E4M3_NORMAL_BIAS)), mant_odd));
        let small = _mm512_cmplt_epu32_mask(bits, splat(E4M3_MIN_NORMAL));
        let magnitude = _mm512_mask_blend_epi32(small, normal, denorm);
        _mm512_and_si512(_mm512_or_si512(magnitude, _mm512_srli_epi32::<24>(sign)), splat(0xFF))
    }

    /// e4m3_value on LANES of int32 E4M3 bits.
    #[inline]
    #[target_feature(enable = "avx512f")]
    fn e4m3_value_v(q: __m512i) -> __m512 {
        const TO_F32_BIAS: u32 = (127 - 7) << 23;
        const F32_QUIET_NAN: u32 = 0x7FC0_0000;
        const MIN_NORMAL_BITS: u32 = 8;
        const SUBNORMAL_UNIT: f32 = 1.0 / 512.0;
        let magnitude = _mm512_and_si512(q, splat(0x7F));
        let normal = _mm512_add_epi32(_mm512_slli_epi32::<20>(magnitude), splat(TO_F32_BIAS));
        let subnormal = _mm512_mul_ps(_mm512_cvtepi32_ps(magnitude), _mm512_set1_ps(SUBNORMAL_UNIT));
        let value = _mm512_mask_blend_epi32(
            _mm512_cmplt_epu32_mask(magnitude, splat(MIN_NORMAL_BITS)),
            normal,
            _mm512_castps_si512(subnormal),
        );
        let value = _mm512_mask_blend_epi32(
            _mm512_cmpeq_epi32_mask(magnitude, splat(u32::from(E4M3_NAN))),
            value,
            splat(F32_QUIET_NAN),
        );
        let sign = _mm512_slli_epi32::<24>(_mm512_and_si512(q, splat(0x80)));
        _mm512_castsi512_ps(_mm512_or_si512(value, sign))
    }

    /// rewrite on LANES of whole coin groups, as int32.
    #[inline]
    #[target_feature(enable = "avx512f")]
    fn rewrite_v(q: __m512i, mixed: __m512, strip: Strip) -> __m512i {
        // Each lane reads its group's first column (lane & !3).
        let group_start = _mm512_set_epi32(12, 12, 12, 12, 8, 8, 8, 8, 4, 4, 4, 4, 0, 0, 0, 0);
        let field = _mm512_and_si512(
            _mm512_srli_epi32::<STRIP_COIN_SHIFT>(_mm512_permutexvar_epi32(group_start, _mm512_castps_si512(mixed))),
            splat(0x7F),
        );
        let coin = _mm512_cmplt_epu32_mask(field, splat(STRIP_COIN_THRESHOLD));
        let inside = _mm512_mask_cmplt_epu32_mask(coin, _mm512_and_si512(q, splat(0x7F)), splat(strip.bound));
        let rewritten = _mm512_or_si512(_mm512_and_si512(q, splat(0x87)), splat(strip.code));
        _mm512_mask_blend_epi32(inside, q, rewritten)
    }

    /// The row chain on every whole LANES group; returns the columns done.
    #[target_feature(enable = "avx512f,avx512dq")]
    pub(super) fn prepare_row(
        row: RowScales,
        code: &[i8],
        scale: &[u16],
        noises: &[f32],
        bp: &mut [u8],
        diff: &mut [u16],
    ) -> usize {
        let cols = bp.len();
        let alpha = _mm512_set1_ps(row.alpha);
        let beta = _mm512_set1_ps(row.beta);
        let mut c = 0;
        while c + LANES <= cols {
            // SAFETY: every load and store covers [c, c + LANES), within the
            // caller's equal-length slices.
            let (codes, noise) = unsafe {
                (
                    _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128(code.as_ptr().add(c).cast()))),
                    _mm512_loadu_ps(noises.as_ptr().add(c)),
                )
            };
            let block = c / BLOCK_SIZE;
            let coefficient = _mm512_mul_ps(
                alpha,
                join(
                    _mm256_set1_ps(bf16_value(scale[block])),
                    _mm256_set1_ps(bf16_value(scale[block + 1])),
                ),
            );
            let mixed = _mm512_fmadd_ps(beta, noise, _mm512_mul_ps(coefficient, codes));
            let q = rewrite_v(e4m3_bits_v(mixed), mixed, row.strip);
            let d = round_bf16_v(_mm512_fmsub_ps(beta, round_bf16_v(noise), e4m3_value_v(q)));
            let d = _mm512_cvtepi32_epi16(_mm512_srli_epi32::<16>(_mm512_castps_si512(d)));
            // SAFETY: as above.
            unsafe {
                _mm_storeu_si128(bp.as_mut_ptr().add(c).cast(), _mm512_cvtepi32_epi8(q));
                _mm256_storeu_si256(diff.as_mut_ptr().add(c).cast(), d);
            }
            c += LANES;
        }
        c
    }

    /// The integer noise dot of a column tile on every whole ROW_STEP group of
    /// rows; returns the rows done.
    #[target_feature(enable = "avx512f,avx512vnni")]
    pub(super) fn noise_tile(e_ints: &[i16], rows: usize, f_pairs: &[u32], c0: usize, out: &mut [i32]) -> usize {
        const ROW_STEP: usize = 8;
        let mut r = 0;
        while r + ROW_STEP <= rows {
            for c in (0..TILE_COLS).step_by(LANES) {
                let f = &f_pairs[(c0 + c) / LANES * (PAIRS * LANES)..(c0 + c) / LANES * (PAIRS * LANES) + PAIRS * LANES];
                let mut acc = [_mm512_setzero_si512(); ROW_STEP];
                for p in 0..PAIRS {
                    // SAFETY: f holds PAIRS groups of LANES pairs.
                    let fv = unsafe { _mm512_loadu_si512(f.as_ptr().add(p * LANES).cast()) };
                    for (i, acc) in acc.iter_mut().enumerate() {
                        let e_pair = load_pair(&e_ints[(r + i) * RANK + 2 * p..]);
                        *acc = _mm512_dpwssd_epi32(*acc, splat(e_pair), fv);
                    }
                }
                for (i, acc) in acc.iter().enumerate() {
                    let cells = &mut out[(r + i) * TILE_COLS + c..(r + i) * TILE_COLS + c + LANES];
                    // SAFETY: cells holds LANES int32.
                    unsafe { _mm512_storeu_si512(cells.as_mut_ptr().cast(), *acc) };
                }
            }
            r += ROW_STEP;
        }
        r
    }

    /// The mid half of a column tile on every whole ROW_STEP group of rows;
    /// returns the rows done.
    #[target_feature(enable = "avx512f,avx512bf16")]
    pub(super) fn mid_tile(diff: &[u16], rows: usize, f_a_pairs: &[u16], c0: usize, acc: &mut [f32]) -> usize {
        const ROW_STEP: usize = 8;
        let mut r = 0;
        while r + ROW_STEP <= rows {
            let mut sums = [[_mm512_setzero_ps(); 2]; ROW_STEP];
            for (i, sums) in sums.iter_mut().enumerate() {
                let row = &acc[(r + i) * RANK..(r + i + 1) * RANK];
                // SAFETY: row holds RANK = 2 * LANES fp32.
                *sums = unsafe { [_mm512_loadu_ps(row.as_ptr()), _mm512_loadu_ps(row.as_ptr().add(LANES))] };
            }
            for c in (0..TILE_COLS).step_by(2) {
                let f = &f_a_pairs[(c0 + c) * RANK..(c0 + c + 2) * RANK];
                // SAFETY: f holds 2 * RANK BF16, two registers of pairs.
                let (f_lo, f_hi): (__m512bh, __m512bh) = unsafe {
                    (
                        std::mem::transmute::<__m512i, __m512bh>(_mm512_loadu_si512(f.as_ptr().cast())),
                        std::mem::transmute::<__m512i, __m512bh>(_mm512_loadu_si512(f.as_ptr().add(2 * LANES).cast())),
                    )
                };
                for (i, sums) in sums.iter_mut().enumerate() {
                    let cell = (r + i) * TILE_COLS + c;
                    let pair = u32::from(diff[cell]) | (u32::from(diff[cell + 1]) << 16);
                    // SAFETY: an int32 register reinterpreted as BF16 pairs.
                    let d: __m512bh = unsafe { std::mem::transmute(splat(pair)) };
                    sums[0] = _mm512_dpbf16_ps(sums[0], d, f_lo);
                    sums[1] = _mm512_dpbf16_ps(sums[1], d, f_hi);
                }
            }
            for (i, sums) in sums.iter().enumerate() {
                let row = &mut acc[(r + i) * RANK..(r + i + 1) * RANK];
                // SAFETY: as above.
                unsafe {
                    _mm512_storeu_ps(row.as_mut_ptr(), sums[0]);
                    _mm512_storeu_ps(row.as_mut_ptr().add(LANES), sums[1]);
                }
            }
            r += ROW_STEP;
        }
        r
    }

    /// integer_noise_values on every row; returns the rows done.
    #[target_feature(enable = "avx512f")]
    pub(super) fn integer_noise_values(sums: &[i32], rows: usize, e_scale: &[f32], f_scale: &[f32], out: &mut [f32]) -> usize {
        for i in 0..rows {
            let e = _mm512_set1_ps(e_scale[i]);
            for c in (0..TILE_COLS).step_by(LANES) {
                let at = i * TILE_COLS + c;
                let (s, f, o) = (&sums[at..at + LANES], &f_scale[c..c + LANES], &mut out[at..at + LANES]);
                // SAFETY: s, f and o hold LANES values each.
                unsafe {
                    let s = _mm512_cvt_roundepi32_ps::<{ _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC }>(_mm512_loadu_si512(
                        s.as_ptr().cast(),
                    ));
                    // Powers of two at least 2^-18: both products are exact.
                    let scale = _mm512_mul_ps(e, _mm512_loadu_ps(f.as_ptr()));
                    _mm512_storeu_ps(o.as_mut_ptr(), _mm512_mul_ps(s, scale));
                }
            }
        }
        rows
    }

    /// LANES terms from `terms[at..]`: their significands and q, as int32.
    #[inline]
    #[target_feature(enable = "avx512f")]
    fn terms_v(terms: &[E4m3Term], at: usize) -> (__m512i, __m512i) {
        let terms = &terms[at..at + LANES];
        // SAFETY: terms holds LANES two-byte terms.
        let x = _mm512_cvtepi16_epi32(unsafe { _mm256_loadu_si256(terms.as_ptr().cast()) });
        (_mm512_srai_epi32::<24>(_mm512_slli_epi32::<24>(x)), _mm512_srai_epi32::<8>(x))
    }

    /// window_round on 8 lanes of int64 sums anchored at int32 `max_exp`:
    /// the fp32 significand (scale it by 2^(exponent - 23)), the exponent, and
    /// which lanes are negative.
    #[inline]
    #[target_feature(enable = "avx512f,avx512dq")]
    fn window_round_v(acc: __m512i, max_exp: __m256i, width: i32) -> (__m256, __m256i, __mmask8) {
        let zero = _mm512_setzero_si512();
        let magnitude = _mm512_abs_epi64(acc);
        // floor(log2 |acc|), exact: |acc| < 2^53.
        let top = _mm512_cvttpd_epi64(_mm512_getexp_pd(_mm512_cvtepi64_pd(magnitude)));
        let top_bit = _mm512_set1_epi64(i64::from(width) - 1);
        let mut exponent = _mm512_sub_epi64(_mm512_add_epi64(_mm512_cvtepi32_epi64(max_exp), top), top_bit);
        // To exactly `width` bits.
        let up = _mm512_sub_epi64(top_bit, top);
        let mut m = _mm512_mask_blend_epi64(
            _mm512_cmplt_epi64_mask(up, zero),
            _mm512_sllv_epi64(magnitude, up),
            _mm512_srlv_epi64(magnitude, _mm512_sub_epi64(zero, up)),
        );
        let floor = _mm512_set1_epi64(F32_MIN_EXPONENT.into());
        let under = _mm512_sub_epi64(floor, exponent);
        // A count of 64 or more zeroes, as the scalar's 32 does on `width` bits.
        m = _mm512_mask_srlv_epi64(m, _mm512_cmpgt_epi64_mask(under, zero), m, under);
        exponent = _mm512_max_epi64(exponent, floor);
        let diff = _mm512_set1_epi64(i64::from((width - GFLOAT_BITS).unsigned_abs()));
        m = if width < GFLOAT_BITS {
            _mm512_sllv_epi64(m, diff)
        } else {
            _mm512_srlv_epi64(m, diff)
        };
        let negative = _mm512_mask_cmpneq_epi64_mask(_mm512_cmplt_epi64_mask(acc, zero), m, zero);
        (_mm512_cvtepi64_ps(m), _mm512_cvtepi64_epi32(exponent), negative)
    }

    /// The windowed noise dot of a column tile on every row, LANES columns at
    /// a time; returns the rows done.
    #[target_feature(enable = "avx512f,avx512dq")]
    pub(super) fn windowed_tile(
        e_terms: &[E4m3Term],
        rows: usize,
        f_terms: &[E4m3Term],
        c0: usize,
        width: i32,
        out: &mut [f32],
    ) -> usize {
        let k = f_terms.len() / RANK;
        let diff = (width - GFLOAT_BITS).unsigned_abs();
        // A product's sig << 17, moved to the window.
        let (left, right) = if width < GFLOAT_BITS { (17, diff) } else { (17 + diff, 0) };
        let (left, right) = (splat(left), splat(right));
        for i in 0..rows {
            let e = &e_terms[i * RANK..(i + 1) * RANK];
            for c in (0..TILE_COLS).step_by(LANES) {
                let mut max_exp = splat(i32::MIN as u32);
                for (j, term) in e.iter().enumerate() {
                    if term.sig != 0 {
                        let (sig, q) = terms_v(f_terms, j * k + c0 + c);
                        let exp = _mm512_add_epi32(q, splat((i32::from(term.q) - 14) as u32));
                        max_exp = _mm512_mask_max_epi32(max_exp, _mm512_test_epi32_mask(sig, sig), max_exp, exp);
                    }
                }
                let (mut lo, mut hi) = (_mm512_setzero_si512(), _mm512_setzero_si512());
                for (j, term) in e.iter().enumerate() {
                    if term.sig != 0 {
                        let (sig, q) = terms_v(f_terms, j * k + c0 + c);
                        let product = _mm512_mullo_epi32(sig, splat(i32::from(term.sig) as u32));
                        let exp = _mm512_add_epi32(q, splat((i32::from(term.q) - 14) as u32));
                        // A count of 32 or more (a term below the window, or a
                        // zero product's) zeroes.
                        let shift = _mm512_sub_epi32(max_exp, exp);
                        let windowed = _mm512_srlv_epi32(_mm512_sllv_epi32(_mm512_abs_epi32(product), left), right);
                        let aligned = _mm512_srlv_epi32(windowed, shift);
                        let signed =
                            _mm512_mask_sub_epi32(aligned, _mm512_movepi32_mask(product), _mm512_setzero_si512(), aligned);
                        lo = _mm512_add_epi64(lo, _mm512_cvtepi32_epi64(_mm512_castsi512_si256(signed)));
                        hi = _mm512_add_epi64(hi, _mm512_cvtepi32_epi64(_mm512_extracti64x4_epi64::<1>(signed)));
                    }
                }
                let (lo, lo_exp, lo_neg) = window_round_v(lo, _mm512_castsi512_si256(max_exp), width);
                let (hi, hi_exp, hi_neg) = window_round_v(hi, _mm512_extracti64x4_epi64::<1>(max_exp), width);
                let exponent = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(lo_exp), hi_exp);
                // Exact: at most 24 significand bits at or above 2^-149.
                let value = _mm512_scalef_ps(join(lo, hi), _mm512_cvtepi32_ps(_mm512_sub_epi32(exponent, splat(23))));
                let negative = u16::from(lo_neg) | (u16::from(hi_neg) << 8);
                let value = _mm512_mask_sub_ps(value, negative, _mm512_setzero_ps(), value);
                let cells = &mut out[i * TILE_COLS + c..i * TILE_COLS + c + LANES];
                // SAFETY: cells holds LANES fp32.
                unsafe { _mm512_storeu_ps(cells.as_mut_ptr(), value) };
            }
        }
        rows
    }
}

#[cfg(test)]
mod tests {
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    use super::*;
    use crate::v4::api::dtype::f32_to_bf16;
    use crate::v4::api::noise::{Noise, OperandNoise, sample_noise};
    use crate::v4::api::prequant::exact_norms;
    use crate::v4::api::primitives::Sides;
    use crate::v4::api::public_params::Device;
    use crate::v4::api::quantization::Fp8E4M3Quant;
    use crate::v4::api::transcript::{LABEL_NOISE_LINE, subkey};

    const SEEDS: Sides<Hash256> = Sides {
        a: [0x22; 32],
        b: [0x11; 32],
    };

    /// The arithmetic each device commits to (the miner's per-device constants).
    fn arithmetic(device: Device) -> DeviceArithmetic {
        match device {
            Device::B200 => DeviceArithmetic {
                window_bits: 26,
                strip_g: (0.08, 0.1),
            },
            Device::H100 => DeviceArithmetic {
                window_bits: 14,
                strip_g: (0.105, 0.105),
            },
        }
    }

    fn job_noise(n: usize, k: usize) -> Noise {
        let b_cols: Vec<u32> = (0..n as u32).collect();
        sample_noise(k, RANK as u16, SEEDS, &[], &b_cols)
    }

    /// `(k, RANK)` -> `(RANK, k)`.
    fn transposed(f: &[u8]) -> Vec<u8> {
        let k = f.len() / RANK;
        (0..RANK * k).map(|i| f[(i % k) * RANK + i / k]).collect()
    }

    fn planes(n: usize, k: usize, seed: u64) -> (Vec<i8>, Vec<u16>) {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let codes = (0..n * k).map(|_| rng.random_range(-127..=127)).collect();
        let scales = (0..n * k / BLOCK_SIZE)
            .map(|_| f32_to_bf16(rng.random_range(1e-4f32..1e-2)).unwrap())
            .collect();
        (codes, scales)
    }

    struct Prepared {
        b_prime: Vec<u8>,
        mid: Vec<u16>,
    }

    fn prepare(isa: Isa, integer_dot: bool, rows: &BRows, arithmetic: DeviceArithmetic) -> Prepared {
        let (n, k) = (rows.alpha.len(), rows.f_b.len() / RANK);
        let mut prepared = Prepared {
            b_prime: vec![0; n * k],
            mid: vec![0; n * RANK],
        };
        prepare_rows_with(isa, integer_dot, rows, arithmetic, &mut prepared.b_prime, &mut prepared.mid).unwrap();
        prepared
    }

    #[test]
    fn noise_lines_are_the_sampled_lines() {
        let (n, k) = (700, 512);
        let noise = job_noise(n, k);
        let key = subkey(LABEL_NOISE_LINE, Some(&SEEDS.b));
        for (side, factor, count, expected) in [(1, 0, n, &noise.b.e), (0, 1, k, &noise.a.f), (1, 1, k, &noise.b.f)] {
            let mut lines = vec![0u8; count * RANK];
            noise_lines(&key, side, factor, &mut lines).unwrap();
            assert_eq!(&lines, expected, "side {side} factor {factor}");
        }
        assert!(noise_lines(&key, 1, 0, &mut [0u8; RANK + 1]).is_err());
    }

    /// The windowed dot is each device's atom, and on lines within the spans
    /// the integer dot is Blackwell's.
    #[test]
    fn noise_dots_are_the_device_atoms() {
        let (m, n) = (24, 40);
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let mut lines = |count: usize| -> Vec<u8> {
            (0..count * RANK)
                .map(|_| {
                    let b: u8 = rng.random();
                    if b & 0x7F == E4M3_NAN { b ^ 1 } else { b }
                })
                .collect()
        };
        let noise = job_noise(m, n);
        let cases = [(lines(m), lines(n)), (noise.b.e.clone(), noise.b.f[..n * RANK].to_vec())];
        for (e, f) in &cases {
            let e_terms: Vec<E4m3Term> = e.iter().map(|&b| e4m3_term(b)).collect();
            let f_terms: Vec<E4m3Term> = f.iter().map(|&b| e4m3_term(b)).collect();
            for device in Device::ALL {
                let expected = device.matmul_fp8(e, f, None, m, n, RANK).unwrap();
                let width = i32::from(arithmetic(device).window_bits);
                for (cell, &expected) in expected.iter().enumerate() {
                    let (r, c) = (cell / n, cell % n);
                    let got = windowed_dot(&e_terms[r * RANK..][..RANK], &f_terms[c * RANK..][..RANK], width);
                    assert_eq!(got.to_bits(), expected.to_bits(), "{device:?} cell ({r}, {c})");
                }
            }
        }
        let (e, f) = &cases[1];
        assert!(integer_noise_dot_exact(e, &transposed(f)));
        let expected = Device::B200.matmul_fp8(e, f, None, m, n, RANK).unwrap();
        let (mut e_ints, mut f_ints) = (vec![0i16; RANK], vec![0i16; RANK]);
        for (cell, &expected) in expected.iter().enumerate() {
            let (r, c) = (cell / n, cell % n);
            let e_scale = integer_line(&e[r * RANK..], 1, &mut e_ints);
            let f_scale = integer_line(&f[c * RANK..], 1, &mut f_ints);
            let exact = f64::from(int_dot(&e_ints, &f_ints)) * f64::from(e_scale) * f64::from(f_scale);
            assert_eq!(toward_zero_f32(exact).to_bits(), expected.to_bits(), "cell ({r}, {c})");
        }
    }

    /// B' is the verifier's noisy quantization on every path, and the scalar
    /// and AVX-512 mid halves agree with each other and with fp64.
    #[test]
    fn rows_match_the_verifiers_noisy_quantization() {
        for (n, k) in [(37, 512), (16, 256)] {
            let (codes, scales) = planes(n, k, (n * k) as u64);
            let noise = job_noise(n, k);
            let (f_a, f_b) = (transposed(&noise.a.f), transposed(&noise.b.f));
            let norms = exact_norms(&codes, &scales, n, k, BLOCK_SIZE).unwrap();
            for device in Device::ALL {
                let operand = OperandNoise {
                    e: noise.b.e.clone(),
                    f: noise.b.f.clone(),
                };
                let built = Fp8E4M3Quant::new(device)
                    .noisy_quantize(&codes, &scales, &operand, &norms)
                    .unwrap();
                let rows = BRows {
                    codes: &codes,
                    scales: &scales,
                    alpha: &built.alpha,
                    beta: &built.beta,
                    l2: &built.l2,
                    e_b: &noise.b.e,
                    f_a: &f_a,
                    f_b: &f_b,
                };
                let integer = device == Device::B200 && integer_noise_dot_exact(&noise.b.e, &f_b);
                assert_eq!(
                    integer,
                    device == Device::B200,
                    "sampled lines take the integer dot on Blackwell"
                );
                let scalar = prepare(Isa::SCALAR, false, &rows, arithmetic(device));
                let mut paths = vec![("scalar", &scalar)];
                let detected = prepare(Isa::detect(), integer, &rows, arithmetic(device));
                paths.push(("detected", &detected));
                let integer_scalar = prepare(Isa::SCALAR, integer, &rows, arithmetic(device));
                paths.push(("integer scalar", &integer_scalar));
                for (name, prepared) in &paths {
                    assert!(prepared.b_prime == built.noised_part, "{device:?} {n}x{k} {name} B'");
                }

                let peel_mid = mid_f64(&built.noised_part, &built.beta, &operand, &f_a, device, k);
                for (name, prepared) in &paths {
                    let mid: Vec<f64> = prepared.mid.iter().map(|&b| f64::from(bf16_value(b))).collect();
                    let error = relative_error(&mid, &peel_mid);
                    assert!(error < 1.0 / 128.0, "{device:?} {n}x{k} {name} mid half off by {error}");
                }
            }
        }
    }

    /// The vector noise tiles compute the scalar tiles' cells, at every window.
    #[test]
    fn vector_noise_tiles_match_scalar() {
        let k = 2 * TILE_COLS;
        let mut rng = ChaCha8Rng::seed_from_u64(9);
        let mut lines = |count: usize| -> Vec<E4m3Term> {
            (0..count)
                .map(|i| {
                    let b: u8 = rng.random();
                    // Signed zeros, and no NaN.
                    let b = if i % 11 == 0 { b & 0x80 } else { b };
                    e4m3_term(if b & 0x7F == E4M3_NAN { b ^ 1 } else { b })
                })
                .collect()
        };
        let (e_terms, f_terms) = (lines(TILE_ROWS * RANK), lines(RANK * k));
        let cells = TILE_ROWS * TILE_COLS;
        for width in 2..=30 {
            for c0 in [0, TILE_COLS] {
                let (mut vector, mut scalar) = (vec![0f32; cells], vec![0f32; cells]);
                windowed_tile(Isa::detect(), &e_terms, TILE_ROWS, &f_terms, c0, width, &mut vector);
                windowed_tile(Isa::SCALAR, &e_terms, TILE_ROWS, &f_terms, c0, width, &mut scalar);
                for (cell, (v, s)) in vector.iter().zip(&scalar).enumerate() {
                    assert_eq!(v.to_bits(), s.to_bits(), "width {width} tile {c0} cell {cell}");
                }
            }
        }

        let sums: Vec<i32> = (0..cells).map(|_| rng.random()).collect();
        let mut power = || f32::from_bits(((rng.random_range(-9..=5) + 127) as u32) << 23);
        let e_scale: Vec<f32> = (0..TILE_ROWS).map(|_| power()).collect();
        let f_scale: Vec<f32> = (0..TILE_COLS).map(|_| power()).collect();
        let (mut vector, mut scalar) = (vec![0f32; cells], vec![0f32; cells]);
        integer_noise_values(Isa::detect(), &sums, TILE_ROWS, &e_scale, &f_scale, &mut vector);
        integer_noise_values(Isa::SCALAR, &sums, TILE_ROWS, &e_scale, &f_scale, &mut scalar);
        assert!(vector.iter().zip(&scalar).all(|(v, s)| v.to_bits() == s.to_bits()));
    }

    #[test]
    fn rows_reject_malformed_operands() {
        let (n, k) = (4, 256);
        let (codes, scales) = planes(n, k, 1);
        let noise = job_noise(n, k);
        let (f_a, f_b) = (transposed(&noise.a.f), transposed(&noise.b.f));
        let ones = vec![0x3F80u16; n];
        let rows = BRows {
            codes: &codes,
            scales: &scales,
            alpha: &ones,
            beta: &ones,
            l2: &ones,
            e_b: &noise.b.e,
            f_a: &f_a,
            f_b: &f_b,
        };
        let (mut b_prime, mut mid) = (vec![0u8; n * k], vec![0u16; n * RANK]);
        let b200 = arithmetic(Device::B200);
        assert!(prepare_rows(&rows, b200, &mut b_prime, &mut mid).is_ok());
        assert!(prepare_rows(&rows, b200, &mut b_prime[1..], &mut mid).is_err());
        let short = BRows {
            beta: &ones[1..],
            ..rows
        };
        assert!(prepare_rows(&short, b200, &mut b_prime, &mut mid).is_err());
        let wide = DeviceArithmetic { window_bits: 40, ..b200 };
        assert!(prepare_rows(&rows, wide, &mut b_prime, &mut mid).is_err());
    }

    /// `(beta (.) E_B@F_B - B') @ F_A^T` in fp64 over the atom's noise.
    fn mid_f64(b_prime: &[u8], beta: &[u16], noise: &OperandNoise, f_a: &[u8], device: Device, k: usize) -> Vec<f64> {
        let n = beta.len();
        let atom = device.matmul_fp8(&noise.e, &noise.f, None, n, k, RANK).unwrap();
        let mut mid = vec![0f64; n * RANK];
        for r in 0..n {
            let b = f64::from(bf16_value(beta[r]));
            for c in 0..k {
                let diff = b * f64::from(round_bf16(atom[r * k + c])) - f64::from(e4m3_value(b_prime[r * k + c]));
                for j in 0..RANK {
                    mid[r * RANK + j] += diff * f64::from(e4m3_value(f_a[j * k + c]));
                }
            }
        }
        mid
    }

    fn relative_error(got: &[f64], expected: &[f64]) -> f64 {
        let error: f64 = got.iter().zip(expected).map(|(g, e)| (g - e).powi(2)).sum();
        let scale: f64 = expected.iter().map(|e| e.powi(2)).sum();
        (error / scale).sqrt()
    }
}
