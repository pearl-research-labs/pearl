//! Deterministic low-rank FP16 noise factors for one A100 tile.
//!
//! The direct analogue of the FP8 noise module ([`crate::v4::api::noise`]): the
//! per-matmul noise `N = E @ F^T` is a stack of keyed-BLAKE3 "lines", one rule
//! ([`sample_line`]) for every factor. A line is `r` XOF bytes -> sign x UNIFORM
//! magnitude in `[1, 128]` (never zero, no modulo bias), L2-NORMALIZED to the
//! shared constant norm [`NOISE_TARGET_NORM`] with the exact integer-`isqrt` +
//! one-BF16-division recipe (byte-identical on any host), then rounded. The ONLY
//! difference from the FP8 draw is the final element type: FP16 (`u16`) rather
//! than FP8 E4M3 (`u8`), because the A100 datapath consumes FP16 lines and
//! [`super::quantization::noisy_quantize`] expects `E`/`F` as `u16`.
//!
//! A constant line norm makes an `E@F` entry (`<e_row, f_col>`) have a known peak
//! (`~c^2`) and rms (`~c^2/sqrt(r)`), so the quant scheme derives its per-row
//! scales from `X`'s norms alone, never measuring `E@F` — see the FP8 rationale.
//!
//! # Seed chain (B then A), FP16 domain
//!
//! The FP16 scheme is a parallel scheme to FP8, so it gets its own transcript
//! domain (`pearl/v4/FP16/...`): a shared label would let the two schemes collide
//! on a reused seed. The chain mirrors the FP8 dense derivation:
//!
//! ```text
//! keyA         := H_"key-A"(proposed_header)                      (A-side tree key)
//! keyB         := H_"key-B"(ancestor_header)                      (B-side tree key)
//! noise seedB  := H_"seed-B"(root_B || keyB || pB)
//! noise seedA  := H_"seed-A"(root_A || noise_seedB || keyA || pA)
//! ```
//!
//! `root_X` is the operand's Merkle commitment root; `pX` is the per-side public
//! parameter encoding ([`crate::v5::api::plain_proof`]). `E_A` keys off
//! `seedA`; both `F` bases and `E_B` key off `seedB` (with distinct `Side`
//! addresses), exactly as FP8.
//!
//! ## Window-authenticated ancestor keying
//!
//! Mirroring the full FP8 scheme ([`crate::v4::api::transcript`]), the A-side
//! tree is keyed by `H_"key-A"(proposed_header)` (`σ̂`) and the B-side tree by
//! `H_"key-B"(ancestor_header)` (`σ_Δ`), a header drawn from the depth-`D` state
//! window preceding the proposed header. The ancestor is authenticated as a
//! member of that window by the consensus verifier's `check_fp16_certificate_ancestors`
//! (a SHA256d hash-walk from the proposed header through the supplied full
//! ancestor headers; see `zk-pow/bindings/go/src/fp16.rs`) before verification.
//! Because B's commitment root only rebuilds under `keyB`, an
//! ancestor that is not the one the operand was committed against — whether
//! out-of-window or merely unauthenticated — fails the B-side Merkle rebuild.
//! The ancestor is also folded into `pB` and hence the seed chain.

use pearl_blake3::blake3_digest;

use super::dtype::f32_to_fp16;
use crate::v4::api::compute::{bf16_div, bf16_mul};
use crate::v4::api::dtype::{bf16_to_f32, f32_to_bf16};
use crate::v4::api::quantization::NOISE_TARGET_NORM;
use crate::v4::api::primitives::{Hash256, IncompleteBlockHeader, Sides};

/// Fixed-point factor carrying `log2(32) = 5` fractional norm bits through the
/// exact integer `isqrt`. It cancels in the scale division; it only preserves
/// precision in the floored square root. Identical to the FP8 constant.
const INT_SQRT_PREC: u64 = 32;

/// FP16 transcript labels. A distinct domain from the FP8 `pearl/v4/FP8/`
/// labels so the two schemes never derive the same seed from the same inputs.
/// `key-A`/`key-B` mirror the FP8 per-side opening keys: the A-side tree keys on
/// the proposed header, the B-side tree on the window-authenticated ancestor.
const LABEL_KEY_A: &[u8] = b"pearl/v4/FP16/key-A";
const LABEL_KEY_B: &[u8] = b"pearl/v4/FP16/key-B";
const LABEL_SEED_A: &[u8] = b"pearl/v4/FP16/seed-A";
const LABEL_SEED_B: &[u8] = b"pearl/v4/FP16/seed-B";
const LABEL_NOISE_LINE: &[u8] = b"pearl/v4/FP16/noise-line";

/// Derive a 32-byte role key from an FP16 label and an optional parent.
pub(crate) fn subkey(label: &[u8], parent: Option<&Hash256>) -> Hash256 {
    blake3_digest(label, parent.copied())
}

/// Hash `message` under [`subkey`] of `label` (unkeyed parent).
fn hash_labelled(message: &[u8], label: &[u8]) -> Hash256 {
    blake3_digest(message, Some(subkey(label, None)))
}

/// `keyA = H_"key-A"(proposed_header)` — the A-side tree/opening key.
pub(crate) fn key_a(proposed_header: &IncompleteBlockHeader) -> Hash256 {
    hash_labelled(&proposed_header.to_bytes(), LABEL_KEY_A)
}

/// `keyB = H_"key-B"(ancestor_header)` — the B-side tree/opening key, keyed on
/// the proof-carried, window-authenticated ancestor header.
pub(crate) fn key_b(ancestor_header: &IncompleteBlockHeader) -> Hash256 {
    hash_labelled(&ancestor_header.to_bytes(), LABEL_KEY_B)
}

/// `Sides { a: keyA, b: keyB }` — the per-side opening keys, mirroring
/// [`crate::v4::api::transcript`]. `proposed_header` (`σ̂`) keys the A side;
/// `ancestor_header` (`σ_Δ`) keys the B side.
pub(crate) fn commitment_keys(
    proposed_header: &IncompleteBlockHeader,
    ancestor_header: &IncompleteBlockHeader,
) -> Sides<Hash256> {
    Sides {
        a: key_a(proposed_header),
        b: key_b(ancestor_header),
    }
}

/// The per-side noise seeds, derived B-then-A from the committed roots, the
/// per-side opening keys and the per-side public-parameter encodings. `seedB`
/// keys off `keyB` (the ancestor), `seedA` off `keyA` (the proposed header). See
/// the module docs.
pub fn noise_seeds(keys: &Sides<Hash256>, roots: &Sides<Hash256>, p: &Sides<Vec<u8>>) -> Sides<Hash256> {
    let message_b = [roots.b.as_slice(), keys.b.as_slice(), p.b.as_slice()].concat();
    let seed_b = hash_labelled(&message_b, LABEL_SEED_B);
    let message_a = [roots.a.as_slice(), seed_b.as_slice(), keys.a.as_slice(), p.a.as_slice()].concat();
    let seed_a = hash_labelled(&message_a, LABEL_SEED_A);
    Sides { a: seed_a, b: seed_b }
}

/// Which operand a noise line belongs to. The discriminants are committed bytes.
#[repr(u8)]
pub(crate) enum Side {
    A = 0,
    B = 1,
}

/// Which factor a line contributes to: the row/col-keyed `E` or the shared `F`.
#[repr(u8)]
pub(crate) enum NoiseFactor {
    E = 0,
    F = 1,
}

/// One operand's paired noise factors, FP16 bit patterns, whose product
/// `N = E @ F^T` is the injected noise.
pub struct NoiseFactors16 {
    /// `(num_rows x r)` FP16 values, row-major (the row/col-keyed E-lines).
    pub e: Vec<u16>,
    /// `(k x r)` FP16 values, row-major (the shared F basis).
    pub f: Vec<u16>,
}

/// Both operands' deterministic FP16 noise factors.
pub type Noise16 = Sides<NoiseFactors16>;

/// Draws one keyed, L2-normalized line of `rank` FP16 entries.
///
/// Derives the line key as [`subkey`] of [`LABEL_NOISE_LINE`] under `seed`, then
/// keyed-BLAKE3-XOF's the address `side | factor | line(u32 LE)` — zero-padded to
/// a constant 64 bytes (one BLAKE3 block) — to `rank` output bytes and normalizes
/// them (see [`normalize_line`]).
pub(crate) fn sample_line(seed: &Hash256, side: Side, factor: NoiseFactor, line: u32, rank: u16) -> Vec<u16> {
    normalize_line(&sample_line_xof_bytes(seed, side, factor, line, rank))
}

/// The raw keyed-BLAKE3-XOF bytes a line is drawn from, *before* [`normalize_line`]. These are the
/// free-witness bytes the circuit's NoiseStark normalizes (its seed-keyed-XOF binding is a later
/// increment); exposing them lets the batch driver feed NoiseStark the REAL per-line bytes so its
/// normalized output is bit-exact with [`sample_line`]/[`sample_noise`].
pub(crate) fn sample_line_xof_bytes(seed: &Hash256, side: Side, factor: NoiseFactor, line: u32, rank: u16) -> Vec<u8> {
    let key = subkey(LABEL_NOISE_LINE, Some(seed));
    let mut material = Vec::with_capacity(64);
    material.push(side as u8);
    material.push(factor as u8);
    material.extend_from_slice(&line.to_le_bytes());
    assert!(material.len() <= 64, "noise line material must fit one BLAKE3 block");
    material.resize(64, 0);

    let mut bytes = vec![0u8; usize::from(rank)];
    let mut hasher = blake3::Hasher::new_keyed(&key);
    hasher.update(&material);
    hasher.finalize_xof().fill(&mut bytes);
    bytes
}

/// The `noise-line` subkey label ([`LABEL_NOISE_LINE`]) — the 24-byte message whose keyed-BLAKE3
/// digest under a `seed` is that seed's `line_key` ([`subkey`]). Exposed so the FP16 ZK noise
/// binding ([`crate::v5::circuit::noise_blake3`]) can pin the subkey compression's message to this
/// exact public constant.
pub(crate) fn noise_line_label() -> &'static [u8] {
    LABEL_NOISE_LINE
}

/// The 64-byte keyed-BLAKE3 material block of one noise line — `side | factor | line(u32 LE)` then
/// zero-padded to one BLAKE3 block, byte-identical to the block [`sample_line_xof_bytes`] hashes
/// under the seed's `line_key`. Exposed so the ZK noise binding can pin each line compression's
/// message to this exact public constant (the material is a pure function of the public line
/// address, never free witness).
pub(crate) fn noise_line_material(side: Side, factor: NoiseFactor, line: u32) -> [u8; 64] {
    let mut material = [0u8; 64];
    material[0] = side as u8;
    material[1] = factor as u8;
    material[2..6].copy_from_slice(&line.to_le_bytes());
    material
}

/// Decodes `bytes` into a signed integer line and renormalizes it to L2 norm
/// [`NOISE_TARGET_NORM`], cast to FP16.
///
/// `norm_scaled = floor(||x||_2 * INT_SQRT_PREC)` (exact integer `isqrt`), then
/// `scale = bf16(NOISE_TARGET_NORM * INT_SQRT_PREC) / bf16(norm_scaled)` (one
/// BF16 rounding), then `entry_i = fp16(bf16(x_i) * scale)`. Identical to the FP8
/// recipe except the final cast target.
fn normalize_line(bytes: &[u8]) -> Vec<u16> {
    // Each `x_i` in ±[1, 128]: bit 7 the sign, `(b & 0x7F) + 1` the magnitude.
    let x: Vec<i64> = bytes
        .iter()
        .map(|&b| {
            let sign = 1 - 2 * ((b >> 7) as i64); // +1 (bit 7 = 0) or -1
            let magnitude = ((b & 0x7F) as i64) + 1; // uniform in [1, 128], never 0
            sign * magnitude
        })
        .collect();

    let sumsq: u64 = x.iter().map(|&xi| (xi * xi) as u64).sum();
    let norm_scaled = (sumsq * (INT_SQRT_PREC * INT_SQRT_PREC)).isqrt();

    let numer = f32_to_bf16((NOISE_TARGET_NORM * INT_SQRT_PREC as f64) as f32).expect("8192 is representable");
    let denom = f32_to_bf16(norm_scaled as f32).expect("norm_scaled < 2^24 is representable");
    let scale = bf16_div(numer, denom).expect("noise-line scale is finite");

    x.iter()
        .map(|&xi| {
            let xb = f32_to_bf16(xi as f32).expect("|x_i| <= 128 is representable in bf16");
            let entry = bf16_mul(xb, scale).expect("noise entry is finite");
            f32_to_fp16(bf16_to_f32(entry)).expect("noise entry is representable in FP16")
        })
        .collect()
}

/// Draws the four FP16 noise factors for one tile.
///
/// `a_rows`/`b_cols` are the selected global row/column indices (the `E` lines key
/// off them); both `F` bases are `0..k` lines. `rank` is the peel rank `r`. `E_A`
/// keys off `seeds.a`; `E_B`, `F_A`, `F_B` all key off `seeds.b` with distinct
/// `Side` addresses.
pub(crate) fn sample_noise(k: usize, rank: u16, seeds: Sides<Hash256>, a_rows: &[u32], b_cols: &[u32]) -> Noise16 {
    let line = |seed: &Hash256, side: Side, factor: NoiseFactor, idx: u32| sample_line(seed, side, factor, idx, rank);

    let e_a: Vec<u16> = a_rows.iter().flat_map(|&row| line(&seeds.a, Side::A, NoiseFactor::E, row)).collect();
    let e_b: Vec<u16> = b_cols.iter().flat_map(|&col| line(&seeds.b, Side::B, NoiseFactor::E, col)).collect();
    let f_a: Vec<u16> = (0..k as u32).flat_map(|i| line(&seeds.b, Side::A, NoiseFactor::F, i)).collect();
    let f_b: Vec<u16> = (0..k as u32).flat_map(|i| line(&seeds.b, Side::B, NoiseFactor::F, i)).collect();

    Noise16 {
        a: NoiseFactors16 { e: e_a, f: f_a },
        b: NoiseFactors16 { e: e_b, f: f_b },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::v5::api::dtype::fp16_to_f32;

    fn seeds() -> Sides<Hash256> {
        Sides { a: [0x22u8; 32], b: [0x11u8; 32] }
    }

    /// Additive, test-only oracle dump for the sm_80 miner `fp16_noise_lines`
    /// kernel. Writes XOF bytes, per-seed line keys, and full `sample_noise`
    /// E/F `u16` outputs for several seeds/shapes so the GA100 kernel can be
    /// asserted bit-exact. Ignored by default; run with
    /// `PEARL_FP16_NOISE_VEC=<path> cargo test -p zk-pow api::fp16::noise::dump_noise_vectors -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn dump_noise_vectors() {
        use std::fmt::Write as _;

        let path = std::env::var("PEARL_FP16_NOISE_VEC")
            .unwrap_or_else(|_| "/tmp/fp16_noise_vectors.txt".to_string());

        fn hx(b: &[u8]) -> String {
            b.iter().map(|x| format!("{x:02x}")).collect()
        }
        fn us(v: &[u16]) -> String {
            v.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(" ")
        }

        // (seed_a, seed_b, k, r, a_rows, b_cols)
        let cases: Vec<([u8; 32], [u8; 32], usize, u16, Vec<u32>, Vec<u32>)> = vec![
            ([0x22u8; 32], [0x11u8; 32], 128, 32, vec![0, 8, 64], vec![1, 2]),
            ([0x01u8; 32], [0xfeu8; 32], 64, 32, vec![0, 1, 2, 255], vec![7, 300, 1000]),
            ([0xa5u8; 32], [0x5au8; 32], 256, 32, vec![0], vec![0]),
            // Non-protocol ranks to exercise XOF beyond 32/64 bytes.
            ([0x33u8; 32], [0x44u8; 32], 40, 16, vec![3, 9], vec![4]),
            ([0x7fu8; 32], [0x80u8; 32], 48, 48, vec![0, 17], vec![1, 2, 65535]),
            ([0x00u8; 32], [0xffu8; 32], 24, 72, vec![5], vec![6]),
        ];

        let mut out = String::new();
        for (ci, (seed_a, seed_b, k, r, a_rows, b_cols)) in cases.iter().enumerate() {
            let key_a = subkey(LABEL_NOISE_LINE, Some(seed_a));
            let key_b = subkey(LABEL_NOISE_LINE, Some(seed_b));
            writeln!(out, "CASE {ci}").unwrap();
            writeln!(out, "seed_a {}", hx(seed_a)).unwrap();
            writeln!(out, "seed_b {}", hx(seed_b)).unwrap();
            writeln!(out, "line_key_a {}", hx(&key_a)).unwrap();
            writeln!(out, "line_key_b {}", hx(&key_b)).unwrap();
            writeln!(out, "k {k}").unwrap();
            writeln!(out, "r {r}").unwrap();
            writeln!(out, "a_rows {}", a_rows.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(",")).unwrap();
            writeln!(out, "b_cols {}", b_cols.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(",")).unwrap();

            // Isolate the hash: raw XOF bytes for a few representative addresses.
            let addrs: Vec<(u8, char, u8, char, u32)> = vec![
                (0, 'a', 0, 'E', a_rows[0]),      // Side::A, NoiseFactor::E
                (1, 'b', 0, 'E', b_cols[0]),      // Side::B, NoiseFactor::E
                (0, 'b', 1, 'F', 0u32),           // Side::A, NoiseFactor::F
                (1, 'b', 1, 'F', (k - 1) as u32), // Side::B, NoiseFactor::F
            ];
            for (side_b, sc, _factor_b, fc, line) in addrs {
                let seed = if sc == 'a' { seed_a } else { seed_b };
                let side = if side_b == 0 { Side::A } else { Side::B };
                let factor = if fc == 'E' { NoiseFactor::E } else { NoiseFactor::F };
                let xof = sample_line_xof_bytes(&seed, side, factor, line, *r);
                writeln!(out, "XOF {} {} {} {}", fc, side_b, line, hx(&xof)).unwrap();
            }

            let noise = sample_noise(*k, *r, Sides { a: *seed_a, b: *seed_b }, a_rows, b_cols);
            writeln!(out, "e_a {}", us(&noise.a.e)).unwrap();
            writeln!(out, "f_a {}", us(&noise.a.f)).unwrap();
            writeln!(out, "e_b {}", us(&noise.b.e)).unwrap();
            writeln!(out, "f_b {}", us(&noise.b.f)).unwrap();
            writeln!(out, "END").unwrap();
        }
        std::fs::write(&path, out).unwrap();
        eprintln!("wrote fp16 noise vectors to {path}");
    }

    #[test]
    fn discriminants_are_wire_stable() {
        assert_eq!(Side::A as u8, 0);
        assert_eq!(Side::B as u8, 1);
        assert_eq!(NoiseFactor::E as u8, 0);
        assert_eq!(NoiseFactor::F as u8, 1);
    }

    #[test]
    fn noise_shapes_and_determinism() {
        let (k, r) = (128usize, 32u16);
        let noise = sample_noise(k, r, seeds(), &[0, 8, 64], &[1, 2]);
        assert_eq!(noise.a.e.len(), 3 * r as usize);
        assert_eq!(noise.b.e.len(), 2 * r as usize);
        assert_eq!(noise.a.f.len(), k * r as usize);
        assert_eq!(noise.b.f.len(), k * r as usize);

        let again = sample_noise(k, r, seeds(), &[0, 8, 64], &[1, 2]);
        assert_eq!(noise.a.e, again.a.e);
        assert_eq!(noise.b.f, again.b.f);
    }

    #[test]
    fn line_is_normalized_to_target_norm() {
        let r = 64u16;
        let noise = sample_noise(64, r, Sides { a: [4u8; 32], b: [3u8; 32] }, &[7], &[]);
        let sq_norm: f32 = noise.a.e.iter().map(|&c| fp16_to_f32(c).powi(2)).sum();
        let norm = sq_norm.sqrt();
        assert!(
            (norm - NOISE_TARGET_NORM as f32).abs() < 0.1 * NOISE_TARGET_NORM as f32,
            "line L2 norm {norm} should be near {NOISE_TARGET_NORM}"
        );
        assert!(noise.a.e.iter().all(|&c| fp16_to_f32(c) != 0.0), "noise never draws zero");
    }

    #[test]
    fn lines_key_on_address_side_and_seed() {
        let base = Sides { a: [9u8; 32], b: [7u8; 32] };
        let a_changed = Sides { a: [8u8; 32], b: [7u8; 32] };
        let b_changed = Sides { a: [9u8; 32], b: [6u8; 32] };
        let n0 = sample_noise(64, 32, base, &[0], &[0]);
        let n_a = sample_noise(64, 32, a_changed, &[0], &[0]);
        let n_b = sample_noise(64, 32, b_changed, &[0], &[0]);

        // Global addressing: only E keys on the selected indices.
        let n_cols = sample_noise(64, 32, base, &[0], &[256]);
        assert_ne!(n0.b.e, n_cols.b.e, "distinct global B-columns must draw distinct E-lines");
        assert_eq!(n0.a.e, n_cols.a.e, "the A E-line depends only on its row");
        assert_eq!(n0.a.f, n_cols.a.f, "the F basis never varies across instances");

        // Side addressing: FA and FB share seedB but never a Side address.
        assert_ne!(n0.a.f, n0.b.f, "FA and FB are distinct Side addresses under seedB");

        // Seed addressing: EA from seedA; both F bases and EB from seedB.
        assert_eq!(n0.a.f, n_a.a.f, "FA is independent of seedA");
        assert_ne!(n0.a.e, n_a.a.e, "EA draws from seedA");
        assert_eq!(n0.b.e, n_a.b.e, "EB is independent of seedA");
        assert_ne!(n0.a.f, n_b.a.f, "FA draws from seedB");
        assert_ne!(n0.b.f, n_b.b.f, "FB draws from seedB");
    }

    #[test]
    fn seed_chain_binds_roots_key_and_params() {
        let keys = Sides { a: [0x5au8; 32], b: [0x5bu8; 32] };
        let roots = Sides { a: [1u8; 32], b: [2u8; 32] };
        let p = Sides { a: vec![10, 11], b: vec![20, 21] };
        let base = noise_seeds(&keys, &roots, &p);

        // seedB folds root_B, keyB and pB; seedA folds root_A, seedB, keyA and pA.
        let other_root_b = noise_seeds(&keys, &Sides { a: roots.a, b: [9u8; 32] }, &p);
        assert_ne!(base.b, other_root_b.b, "seedB binds root_B");
        assert_ne!(base.a, other_root_b.a, "seedA binds seedB (and hence root_B)");

        let other_root_a = noise_seeds(&keys, &Sides { a: [9u8; 32], b: roots.b }, &p);
        assert_eq!(base.b, other_root_a.b, "seedB is independent of root_A");
        assert_ne!(base.a, other_root_a.a, "seedA binds root_A");

        // keyB feeds seedB (and hence seedA); keyA feeds only seedA.
        let other_key_b = noise_seeds(&Sides { a: keys.a, b: [0u8; 32] }, &roots, &p);
        assert_ne!(base.b, other_key_b.b, "seedB binds keyB");
        assert_ne!(base.a, other_key_b.a, "seedA binds keyB through seedB");

        let other_key_a = noise_seeds(&Sides { a: [0u8; 32], b: keys.b }, &roots, &p);
        assert_eq!(base.b, other_key_a.b, "seedB is independent of keyA");
        assert_ne!(base.a, other_key_a.a, "seedA binds keyA");

        let other_pa = noise_seeds(&keys, &roots, &Sides { a: vec![99], b: p.b.clone() });
        assert_ne!(base.a, other_pa.a, "seedA binds pA");
        assert_eq!(base.b, other_pa.b, "seedB is independent of pA");
    }
}
