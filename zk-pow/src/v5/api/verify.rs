//! Plaintext tile verification for the FP16 (A100) scheme.
//!
//! Mirrors the FP8 plaintext verifier ([`crate::v4::api::verify::verify_plain_proof`])
//! but over the FP16 datapath: given the opened FP16 operand strips and the
//! deterministic noise, it rebuilds the noised operands, replays the A100 tile
//! bit-for-bit, runs the unpredictable-accumulation-steps policy, folds the tile
//! into the jackpot ticket, and checks the difficulty target.
//!
//! The lottery extractor ([`xor_fold_extract`]), lane assignment, ticket hash,
//! and difficulty check are scheme-independent and reused from the FP8 modules;
//! only the operand open + noise + quantization + matmul + policy are FP16.
//!
//! Commitment wiring (Merkle openings, seed derivation, wire codec) is a
//! separate layer: this function takes the opened operands and noise factors
//! directly, which is exactly the boundary the FP8 `open_and_noisy_quantize`
//! sits above.

use anyhow::{bail, Result};

use super::params::Fp16Params;
use super::plain_proof::Fp16PlainProof;
use super::policy::{check_shared_gates, replay_and_evaluate, PolicyReport};
use super::quantization::{noisy_quantize, row_norms};
use crate::v4::api::transcript::{compute_jackpot_ticket, Ticket};
use crate::v4::api::utils::xor_fold_extract;
use crate::v4::api::layout::{lane_assignment, AxisPattern};
use crate::v4::api::primitives::{Hash256, IncompleteBlockHeader};
use crate::v4::api::proof_utils::check_jackpot_difficulty;

/// The rank-`r` noise for one operand side (the low-rank factors `E` and `F`,
/// FP16 bit patterns; `N = E @ F^T`).
pub struct OperandNoise16<'a> {
    /// `num_rows x r`.
    pub e: &'a [u16],
    /// `k x r`.
    pub f: &'a [u16],
}

/// The result of a successful tile replay: the recomputed tile, the policy
/// report, and the jackpot ticket.
pub struct TileVerify {
    pub tile: Vec<f32>,
    pub report: PolicyReport,
    pub ticket: Ticket,
}

/// Rebuilds, replays, and scores an FP16 tile. Errors if the jackpot policy
/// rejects the tile; otherwise returns the ticket for the difficulty check.
///
/// `a_rows` is `h x k` and `b_rows` is `w x k` FP16 bit patterns (the operands
/// as committed). `rows_pattern`/`cols_pattern` are the periodic partitions that
/// define the 16-lane extractor layout. `seed_a` keys the jackpot hash.
pub fn verify_tile(
    params: &Fp16Params,
    a_rows: &[u16],
    b_rows: &[u16],
    noise_a: &OperandNoise16,
    noise_b: &OperandNoise16,
    rows_pattern: &AxisPattern,
    cols_pattern: &AxisPattern,
    seed_a: &Hash256,
) -> Result<TileVerify> {
    params.validate()?;
    let (h, w, k, r) = (params.h, params.w, params.k, params.r);
    anyhow::ensure!(a_rows.len() == h * k && b_rows.len() == w * k, "operand shape mismatch");

    // Rebuild the noised FP16 operands: A' = Q(alpha_a*A + beta_a*E_a@F_a), likewise B'.
    let norms_a: Vec<_> = (0..h).map(|i| row_norms(&a_rows[i * k..i * k + k])).collect::<Result<_>>()?;
    let norms_b: Vec<_> = (0..w).map(|j| row_norms(&b_rows[j * k..j * k + k])).collect::<Result<_>>()?;
    let built_a = noisy_quantize(a_rows, noise_a.e, noise_a.f, &norms_a, r)?;
    let built_b = noisy_quantize(b_rows, noise_b.e, noise_b.f, &norms_b, r)?;

    // Shared entry-liveness + noise-floor gates (bound degenerate operands)
    // before the unpredictable-accumulation-steps density gate below.
    check_shared_gates(a_rows, &built_a, b_rows, &built_b, k)
        .map_err(|e| anyhow::anyhow!("the jackpot is not admissible: {e}"))?;

    // Replay the A100 tile and score it.
    let (tile, report) = replay_and_evaluate(&built_a.noised_part, &built_b.noised_part, h, w, k);
    if !report.accept {
        bail!("the jackpot is not admissible (f_bp={:.4}, rho={:.4})", report.f_bp, report.rho);
    }

    // Fold the tile into the ticket over the committed 16-subtile lane layout.
    let lanes = lane_assignment(rows_pattern, cols_pattern);
    let message = xor_fold_extract(&tile, &lanes);
    let ticket = compute_jackpot_ticket(seed_a, &message);
    Ok(TileVerify { tile, report, ticket })
}

/// Full plaintext acceptance: [`verify_tile`] plus the difficulty target check.
pub fn verify_tile_proof(
    params: &Fp16Params,
    a_rows: &[u16],
    b_rows: &[u16],
    noise_a: &OperandNoise16,
    noise_b: &OperandNoise16,
    rows_pattern: &AxisPattern,
    cols_pattern: &AxisPattern,
    seed_a: &Hash256,
    nbits: u32,
) -> Result<TileVerify> {
    let v = verify_tile(params, a_rows, b_rows, noise_a, noise_b, rows_pattern, cols_pattern, seed_a)?;
    check_jackpot_difficulty(&v.ticket.jackpot, nbits, params.h as u32, params.w as u32, params.k as u32)?;
    Ok(v)
}

/// Full certificate acceptance: the FP16 analogue of
/// [`crate::v4::api::verify::verify_plain_proof`]. Parses and authenticates the wire
/// certificate ([`Fp16PlainProof::parse_proof`]: Merkle open+rebuild of both
/// operand trees + the B-then-A noise-seed chain + deterministic FP16 noise),
/// replays and scores the tile, folds the jackpot ticket, and checks difficulty.
///
/// `nbits_override` replaces the proposed header's difficulty (e.g. a pool share).
/// `Ok(())` accepts; any failure (bad opening, inadmissible tile, or unmet
/// difficulty) is an `Err`.
pub fn verify_fp16_plain_proof(
    proposed_header: &IncompleteBlockHeader,
    proof: &Fp16PlainProof,
    nbits_override: Option<u32>,
) -> Result<()> {
    let opened = proof.parse_proof(proposed_header)?;
    let noise_a = OperandNoise16 { e: &opened.noise.a.e, f: &opened.noise.a.f };
    let noise_b = OperandNoise16 { e: &opened.noise.b.e, f: &opened.noise.b.f };
    let nbits = nbits_override.unwrap_or(proposed_header.nbits);
    verify_tile_proof(
        &opened.params,
        &opened.a_rows,
        &opened.b_rows,
        &noise_a,
        &noise_b,
        &opened.rows_pattern,
        &opened.cols_pattern,
        &opened.seed_a,
        nbits,
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::v5::api::dtype::f32_to_fp16;
    use crate::v4::api::layout::DimType::{Blake, Fold};

    const H: usize = 4;
    const W: usize = 64;
    const K: usize = 256;
    const R: usize = 32;

    fn params() -> Fp16Params {
        Fp16Params { device: super::super::params::Fp16Device::A100, h: H, w: W, k: K, r: R }
    }

    fn patterns() -> (AxisPattern, AxisPattern) {
        // 4 row subtiles x 4 col subtiles = 16 lanes; cols fold 16 each.
        (AxisPattern::new(&[(4, Blake)]).unwrap(), AxisPattern::new(&[(4, Blake), (16, Fold)]).unwrap())
    }

    /// Deterministic xorshift stream of FP16 values with spread magnitudes,
    /// so the tile has dense breakpoints (the honest/realistic regime).
    struct Gen(u64);
    impl Gen {
        fn next_u64(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn operand(&mut self, n: usize) -> Vec<u16> {
            (0..n)
                .map(|_| {
                    let r = self.next_u64();
                    let sign = if r & 1 == 0 { 1.0 } else { -1.0 };
                    let exp = ((r >> 1) % 9) as i32 - 3; // 2^-3 .. 2^5
                    let mant = 1.0 + ((r >> 8) % 1024) as f32 / 1024.0;
                    f32_to_fp16(sign * mant * 2f32.powi(exp)).unwrap()
                })
                .collect()
        }
        fn noise(&mut self, n: usize) -> Vec<u16> {
            let scale = 256.0 / (R as f32).sqrt();
            (0..n).map(|_| f32_to_fp16(if self.next_u64() & 1 == 0 { scale } else { -scale }).unwrap()).collect()
        }
    }

    #[test]
    fn accepts_honest_tile_and_ticket_is_deterministic() {
        let mut g = Gen(0x1234_5678_9abc_def1);
        let a = g.operand(H * K);
        let b = g.operand(W * K);
        let (ea, fa) = (g.noise(H * R), g.noise(K * R));
        let (eb, fb) = (g.noise(W * R), g.noise(K * R));
        let (rp, cp) = patterns();
        let seed = [7u8; 32];
        let na = OperandNoise16 { e: &ea, f: &fa };
        let nb = OperandNoise16 { e: &eb, f: &fb };

        let v = verify_tile(&params(), &a, &b, &na, &nb, &rp, &cp, &seed).expect("honest tile accepts");
        assert!(v.report.accept && v.report.f_bp >= 0.30 && v.report.rho >= 1.2);
        // Deterministic: re-running yields the identical ticket.
        let v2 = verify_tile(&params(), &a, &b, &na, &nb, &rp, &cp, &seed).unwrap();
        assert_eq!(v.ticket.jackpot, v2.ticket.jackpot);
        // Easy target accepts; an impossible (all-zero) target rejects.
        check_jackpot_difficulty(&v.ticket.jackpot, 0x207fffff, H as u32, W as u32, K as u32).unwrap();
        assert!(check_jackpot_difficulty(&v.ticket.jackpot, 0, H as u32, W as u32, K as u32).is_err());
    }

    #[test]
    fn rejects_flat_tile() {
        // Flat +-1 operands with no noise: products barely truncate, so the
        // policy's breakpoint density falls below the gate.
        let one = f32_to_fp16(1.0).unwrap();
        let a = vec![one; H * K];
        let b = vec![one; W * K];
        let (ea, fa) = (vec![0u16; H * R], vec![0u16; K * R]);
        let (eb, fb) = (vec![0u16; W * R], vec![0u16; K * R]);
        let (rp, cp) = patterns();
        let na = OperandNoise16 { e: &ea, f: &fa };
        let nb = OperandNoise16 { e: &eb, f: &fb };
        let err = match verify_tile(&params(), &a, &b, &na, &nb, &rp, &cp, &[0u8; 32]) {
            Ok(_) => panic!("flat tile must be rejected"),
            Err(e) => e,
        };
        assert!(format!("{err:#}").contains("not admissible"), "got: {err:#}");
    }

    /// Dumps the full `verify_tile` chain (per-stage + final) for several
    /// params/shapes/seeds, as the bit-exactness oracle for the sm_80
    /// `pearl_gemm.fp16_pipeline` driver. Additive and `#[ignore]`d; run with:
    /// ```text
    /// PEARL_FP16_ORACLE_OUT=/path/oracle.json cargo test -p zk-pow --lib -- \
    ///   api::fp16::verify::tests::dump_pipeline_oracle --ignored --exact --nocapture
    /// ```
    #[test]
    #[ignore = "dumps the FP16 verify_tile oracle vectors for the sm_80 pipeline; run explicitly"]
    fn dump_pipeline_oracle() {
        use crate::v4::api::layout::AxisPattern;
        use serde_json::{json, Value};

        fn axis_json(p: &AxisPattern) -> Value {
            Value::Array(p.dims().iter().map(|&(l, t)| json!([l, t as u8])).collect())
        }

        // Fully dumps one case: always computes the whole chain (even when the
        // policy rejects), so every stage boundary can be checked downstream.
        #[allow(clippy::too_many_arguments)]
        fn case(
            name: &str,
            h: usize,
            w: usize,
            k: usize,
            r: usize,
            a: &[u16],
            b: &[u16],
            ea: &[u16],
            fa: &[u16],
            eb: &[u16],
            fb: &[u16],
            rp: &AxisPattern,
            cp: &AxisPattern,
            seed: &[u8; 32],
            nbits: u32,
        ) -> Value {
            let norms_a: Vec<(u16, u16)> =
                (0..h).map(|i| row_norms(&a[i * k..i * k + k]).unwrap()).collect();
            let norms_b: Vec<(u16, u16)> =
                (0..w).map(|j| row_norms(&b[j * k..j * k + k]).unwrap()).collect();
            let built_a = noisy_quantize(a, ea, fa, &norms_a, r).unwrap();
            let built_b = noisy_quantize(b, eb, fb, &norms_b, r).unwrap();
            let (tile, report) =
                replay_and_evaluate(&built_a.noised_part, &built_b.noised_part, h, w, k);
            let lanes = lane_assignment(rp, cp);
            let message = xor_fold_extract(&tile, &lanes);
            let ticket = compute_jackpot_ticket(seed, &message);
            let diff_ok =
                check_jackpot_difficulty(&ticket.jackpot, nbits, h as u32, w as u32, k as u32).is_ok();
            json!({
                "name": name, "h": h, "w": w, "k": k, "r": r,
                "a_rows": a, "b_rows": b, "ea": ea, "fa": fa, "eb": eb, "fb": fb,
                "rows_pattern": axis_json(rp), "cols_pattern": axis_json(cp),
                "seed_a": seed.to_vec(), "nbits": nbits,
                "norms_a": norms_a.iter().map(|&(l, i)| json!([l, i])).collect::<Vec<_>>(),
                "norms_b": norms_b.iter().map(|&(l, i)| json!([l, i])).collect::<Vec<_>>(),
                "built_a": built_a.noised_part, "built_b": built_b.noised_part,
                "alpha_a": built_a.alpha, "beta_a": built_a.beta, "l2_a": built_a.l2,
                "tile_bits": tile.iter().map(|x| x.to_bits()).collect::<Vec<u32>>(),
                "f_bp_bits": report.f_bp.to_bits(), "rho_bits": report.rho.to_bits(),
                "f_bp": report.f_bp, "rho": report.rho, "accept": report.accept,
                "message": message.to_vec(), "ticket": ticket.jackpot.to_vec(),
                "difficulty_ok": diff_ok,
            })
        }

        let mut cases: Vec<Value> = Vec::new();

        // Honest spread-magnitude operands (accept), two seeds, h=4/w=64.
        for (name, gseed, hseed) in [
            ("honest_a", 0x1234_5678_9abc_def1u64, [7u8; 32]),
            ("honest_b", 0xfeed_face_cafe_b0bau64, [0x5au8; 32]),
        ] {
            let mut g = Gen(gseed);
            let a = g.operand(H * K);
            let b = g.operand(W * K);
            let (ea, fa) = (g.noise(H * R), g.noise(K * R));
            let (eb, fb) = (g.noise(W * R), g.noise(K * R));
            let (rp, cp) = patterns();
            cases.push(case(name, H, W, K, R, &a, &b, &ea, &fa, &eb, &fb, &rp, &cp, &hseed, EASY));
        }

        // Flat operands, zero noise (reject); still dumps the full chain.
        {
            let one = f32_to_fp16(1.0).unwrap();
            let a = vec![one; H * K];
            let b = vec![one; W * K];
            let (ea, fa) = (vec![0u16; H * R], vec![0u16; K * R]);
            let (eb, fb) = (vec![0u16; W * R], vec![0u16; K * R]);
            let (rp, cp) = patterns();
            cases.push(case(
                "flat_reject", H, W, K, R, &a, &b, &ea, &fa, &eb, &fb, &rp, &cp, &[0u8; 32], EASY,
            ));
        }

        // h = 16 (no A-side pad), 4x16 col subtiles: a different fold geometry.
        {
            let (h4, w4) = (16usize, 64usize);
            let rp = AxisPattern::new(&[(4, Blake), (4, Fold)]).unwrap();
            let cp = AxisPattern::new(&[(4, Blake), (16, Fold)]).unwrap();
            assert_eq!(rp.tile_size() as usize, h4);
            assert_eq!(cp.tile_size() as usize, w4);
            let mut g = Gen(0x0bad_f00d_1337_d00d);
            let a = g.operand(h4 * K);
            let b = g.operand(w4 * K);
            let (ea, fa) = (g.noise(h4 * R), g.noise(K * R));
            let (eb, fb) = (g.noise(w4 * R), g.noise(K * R));
            cases.push(case(
                "h16", h4, w4, K, R, &a, &b, &ea, &fa, &eb, &fb, &rp, &cp, &[0x11u8; 32], EASY,
            ));
        }

        let out = std::env::var("PEARL_FP16_ORACLE_OUT")
            .unwrap_or_else(|_| "/tmp/fp16_pipeline_oracle.json".to_string());
        std::fs::write(&out, serde_json::to_vec_pretty(&cases).unwrap()).unwrap();
        println!("wrote {} FP16 pipeline oracle cases to {}", cases.len(), out);
    }

    const EASY: u32 = 0x207f_ffff;

    // ---- End-to-end certificate path (commit -> open -> wire -> verify) ----

    mod cert {
        use super::{Blake, Fold, Gen};
        use crate::v5::api::commitment::{commit_operand, open_rows};
        use crate::v5::api::noise::{key_a, key_b};
        use crate::v5::api::params::Fp16Device;
        use crate::v5::api::plain_proof::{Fp16JobParams, Fp16MatrixProof, Fp16OperandParams, Fp16PlainProof};
        use crate::v5::api::verify::verify_fp16_plain_proof;
        use crate::v4::api::public_params::HashId;
        use crate::v4::api::layout::AxisPattern;
        use crate::v4::api::primitives::{IncompleteBlockHeader, Sides};

        // Total committed rows (> tile sizes, so some rows stay unopened and the
        // openings carry real siblings).
        const M: usize = 8;
        const N: usize = 128;
        const K: usize = 256;
        const R: usize = 32;
        const EASY_NBITS: u32 = 0x207f_ffff;
        const HASH: HashId = HashId::Blake3Chunk1024;

        fn patterns() -> (AxisPattern, AxisPattern) {
            (
                AxisPattern::new(&[(4, Blake)]).unwrap(),          // h = 4
                AxisPattern::new(&[(4, Blake), (16, Fold)]).unwrap(), // w = 64
            )
        }

        /// Builds a wire certificate from full operands: commits the A tree under
        /// keyA (proposed header) and the B tree under keyB (the ancestor header),
        /// then opens the pattern tile rows. `ancestor` is the proof-carried `σ_Δ`.
        fn build_proof_with_ancestor(
            header: &IncompleteBlockHeader,
            ancestor: &IncompleteBlockHeader,
            a_full: &[u16],
            b_full: &[u16],
        ) -> Fp16PlainProof {
            let (rp, cp) = patterns();
            let tree_a = commit_operand(a_full, M, K, HASH, key_a(header)).unwrap();
            let tree_b = commit_operand(b_full, N, K, HASH, key_b(ancestor)).unwrap();
            let a_idx: Vec<usize> = rp.tile_offsets().iter().map(|&o| o as usize).collect();
            let b_idx: Vec<usize> = cp.tile_offsets().iter().map(|&o| o as usize).collect();
            let pa = open_rows(&tree_a, &a_idx, M, K, HASH).unwrap();
            let pb = open_rows(&tree_b, &b_idx, N, K, HASH).unwrap();
            Fp16PlainProof {
                job: Fp16JobParams {
                    ancestor_header: *ancestor,
                    device: Fp16Device::A100,
                    k: K as u32,
                    r: R as u32,
                    operands: Sides {
                        a: Fp16OperandParams { num_rows: M as u32, hash_id: HASH, pattern: rp },
                        b: Fp16OperandParams { num_rows: N as u32, hash_id: HASH, pattern: cp },
                    },
                },
                values: Sides {
                    a: Fp16MatrixProof { proof: pa, row_indices: a_idx },
                    b: Fp16MatrixProof { proof: pb, row_indices: b_idx },
                },
            }
        }

        /// Dense-fixture helper: the proposed header is also the ancestor (`σ_Δ =
        /// σ̂`), the honest single-block case. Keeps the existing call sites intact.
        fn build_proof(header: &IncompleteBlockHeader, a_full: &[u16], b_full: &[u16]) -> Fp16PlainProof {
            build_proof_with_ancestor(header, header, a_full, b_full)
        }

        /// An honest, policy-accepting fixture: spread-magnitude operands whose
        /// noised tile clears the gate (same regime as `accepts_honest_tile`).
        fn honest_fixture() -> (IncompleteBlockHeader, Fp16PlainProof) {
            // Asymmetric prev_block/merkle_root (NOT palindromic under byte
            // reversal) so the committed Go fixture actually exercises header
            // byte-orientation across the FFI seam -- a reversal-invariant header
            // would let a wrong-orientation conversion pass undetected.
            let header = IncompleteBlockHeader {
                prev_block: std::array::from_fn(|i| i as u8),
                merkle_root: std::array::from_fn(|i| 0x40u8 + i as u8),
                ..IncompleteBlockHeader::new_for_test(EASY_NBITS)
            };
            let mut g = Gen(0xdead_beef_0bad_f00d);
            let a_full = g.operand(M * K);
            let b_full = g.operand(N * K);
            (header, build_proof(&header, &a_full, &b_full))
        }

        #[test]
        fn verifier_accepts_the_honest_fixture() {
            let (header, proof) = honest_fixture();
            verify_fp16_plain_proof(&header, &proof, None)
                .unwrap_or_else(|e| panic!("the honest FP16 certificate must verify: {e:#}"));
        }

        /// Writes the committed Go/FFI test vector for the honest A100 certificate.
        /// Format: `header(76) | u32le proof_len | proof_bytes` — the proposed header
        /// (which keys the commitment + seed chain), then the serialized
        /// [`Fp16PlainProof`]. The header's own `nbits` is the difficulty target, so
        /// the verifier is called with `nbits_override = None` (FFI `0`).
        ///
        /// Run explicitly to regenerate:
        /// ```text
        /// cargo test -p zk-pow --lib -- \
        ///   api::fp16::verify::tests::cert::regenerate_go_fixture --ignored --exact
        /// ```
        #[test]
        #[ignore = "regenerates the committed Go/FFI fixture; run explicitly"]
        fn regenerate_go_fixture() {
            let (header, proof) = honest_fixture();
            // Self-check: the vector we are about to commit must verify.
            verify_fp16_plain_proof(&header, &proof, None).expect("fixture must verify before writing");

            let proof_bytes = proof.to_bytes().expect("serialize Fp16PlainProof");
            let proof_len = u32::try_from(proof_bytes.len()).expect("proof length fits u32");
            let mut fixture = Vec::with_capacity(76 + 4 + proof_bytes.len());
            fixture.extend_from_slice(&header.to_bytes());
            fixture.extend_from_slice(&proof_len.to_le_bytes());
            fixture.extend_from_slice(&proof_bytes);

            let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../node/zkpow/testdata/fp16_plain_proof_a100.bin");
            std::fs::write(&path, &fixture).expect("write FP16 Go fixture");
            println!("FP16 A100 Go fixture ({} bytes) written to {}", fixture.len(), path.display());
        }

        /// Additive dump for the Python FP16 opener/assembler round-trip
        /// (`miner_base.fp16_commitment` / `fp16_block_submission`). Writes the
        /// exact operands + keys + roots behind the committed
        /// `fp16_plain_proof_a100.bin` fixture, so the Python side can rebuild the
        /// trees and assert its assembled `Fp16PlainProof.to_bytes()` equals the
        /// fixture bytes (and that its tree roots equal `commit_operand`'s).
        /// Binary layout: `header(76) | m,n,k (u32 le) | a_full (m*k u16 le) |
        /// b_full (n*k u16 le) | keyA(32) | keyB(32) | root_A(32) | root_B(32)`.
        /// ```text
        /// PEARL_FP16_CERT_OPERANDS_OUT=/path/operands.bin cargo test -p zk-pow --lib -- \
        ///   api::fp16::verify::tests::cert::dump_cert_operands --ignored --exact --nocapture
        /// ```
        #[test]
        #[ignore = "dumps the FP16 cert operands for the Python opener/assembler round-trip; run explicitly"]
        fn dump_cert_operands() {
            let header = IncompleteBlockHeader::new_for_test(EASY_NBITS);
            let mut g = Gen(0xdead_beef_0bad_f00d);
            let a_full = g.operand(M * K);
            let b_full = g.operand(N * K);
            let ka = key_a(&header);
            let kb = key_b(&header);
            let root_a = commit_operand(&a_full, M, K, HASH, ka).unwrap().root();
            let root_b = commit_operand(&b_full, N, K, HASH, kb).unwrap().root();

            let mut out = Vec::new();
            out.extend_from_slice(&header.to_bytes());
            for v in [M as u32, N as u32, K as u32] {
                out.extend_from_slice(&v.to_le_bytes());
            }
            for &x in &a_full {
                out.extend_from_slice(&x.to_le_bytes());
            }
            for &x in &b_full {
                out.extend_from_slice(&x.to_le_bytes());
            }
            out.extend_from_slice(&ka);
            out.extend_from_slice(&kb);
            out.extend_from_slice(&root_a);
            out.extend_from_slice(&root_b);

            let path = std::env::var("PEARL_FP16_CERT_OPERANDS_OUT")
                .unwrap_or_else(|_| "/tmp/fp16_cert_operands.bin".to_string());
            std::fs::write(&path, &out).unwrap();
            println!("wrote {} bytes ({} + {} operand u16s) to {}", out.len(), M * K, N * K, path);
        }

        #[test]
        fn parses_a_non_origin_tile_with_global_indices() {
            // A winner off the origin: row tile t_r=1 (global rows [4,8)) and col
            // tile t_c=1 (global cols [64,128)). The opener discloses the GLOBAL
            // rows (`base + tile_offsets()`); `parse_proof` must accept them and
            // open exactly those rows (the E noise then keys on the right global
            // indices -- the reconciliation that makes the full-matrix search and
            // the cert agree for a non-origin winner).
            let header = IncompleteBlockHeader::new_for_test(EASY_NBITS);
            let mut g = Gen(0x0bad_c0de_0bad_cafe);
            let a_full = g.operand(M * K);
            let b_full = g.operand(N * K);
            let (rp, cp) = patterns();
            let (h, w) = (rp.tile_size() as usize, cp.tile_size() as usize);
            let (tr, tc) = (1usize, 1usize);

            let tree_a = commit_operand(&a_full, M, K, HASH, key_a(&header)).unwrap();
            let tree_b = commit_operand(&b_full, N, K, HASH, key_b(&header)).unwrap();
            let a_idx: Vec<usize> = rp.tile_offsets().iter().map(|&o| tr * h + o as usize).collect();
            let b_idx: Vec<usize> = cp.tile_offsets().iter().map(|&o| tc * w + o as usize).collect();
            let pa = open_rows(&tree_a, &a_idx, M, K, HASH).unwrap();
            let pb = open_rows(&tree_b, &b_idx, N, K, HASH).unwrap();

            let proof = Fp16PlainProof {
                job: Fp16JobParams {
                    ancestor_header: header,
                    device: Fp16Device::A100,
                    k: K as u32,
                    r: R as u32,
                    operands: Sides {
                        a: Fp16OperandParams { num_rows: M as u32, hash_id: HASH, pattern: rp.clone() },
                        b: Fp16OperandParams { num_rows: N as u32, hash_id: HASH, pattern: cp.clone() },
                    },
                },
                values: Sides {
                    a: Fp16MatrixProof { proof: pa, row_indices: a_idx.clone() },
                    b: Fp16MatrixProof { proof: pb, row_indices: b_idx.clone() },
                },
            };

            let opened = proof.parse_proof(&header).expect("a non-origin tile must parse");
            assert_eq!(opened.a_rows, a_full[tr * h * K..(tr * h + h) * K].to_vec());
            assert_eq!(opened.b_rows, b_full[tc * w * K..(tc * w + w) * K].to_vec());

            // A base that is not a valid periodic tile offset (row base must be a
            // multiple of h) is rejected, even though the indices are still a
            // contiguous run of valid rows.
            let bad_a: Vec<usize> = rp.tile_offsets().iter().map(|&o| 1 + o as usize).collect();
            let bad_pa = open_rows(&tree_a, &bad_a, M, K, HASH).unwrap();
            let mut bad = proof.clone();
            bad.values.a = Fp16MatrixProof { proof: bad_pa, row_indices: bad_a };
            assert!(bad.parse_proof(&header).is_err(), "a non-periodic tile base must be rejected");
        }

        #[test]
        fn wire_roundtrips_and_rejects_trailing_bytes() {
            let (header, proof) = honest_fixture();
            let bytes = proof.to_bytes().expect("serialize");
            let back = Fp16PlainProof::from_bytes(&bytes).expect("deserialize");
            assert_eq!(back.job, proof.job);
            assert_eq!(back.values.a.row_indices, proof.values.a.row_indices);
            assert_eq!(back.values.a.proof.root, proof.values.a.proof.root);
            assert_eq!(back.values.b.proof.root, proof.values.b.proof.root);
            // The round-tripped certificate still verifies (against the fixture's
            // own header -- honest_fixture uses asymmetric prev_block/merkle_root).
            verify_fp16_plain_proof(&header, &back, None).expect("round-tripped cert must verify");
            // No compat ladder: a trailing byte is rejected.
            let mut trailing = bytes.clone();
            trailing.push(0);
            assert!(Fp16PlainProof::from_bytes(&trailing).is_err());
        }

        #[test]
        fn rejects_a_flipped_opened_row() {
            let (header, mut proof) = honest_fixture();
            proof.values.a.proof.leaf_data[0][0] ^= 0xFF;
            let err = verify_fp16_plain_proof(&header, &proof, None).expect_err("flipped leaf must be rejected");
            assert!(format!("{err:#}").contains("root"), "got: {err:#}");
        }

        #[test]
        fn rejects_a_wrong_root() {
            let (header, mut proof) = honest_fixture();
            proof.values.b.proof.root = [0u8; 32];
            assert!(verify_fp16_plain_proof(&header, &proof, None).is_err(), "wrong root must be rejected");
        }

        #[test]
        fn rejects_a_wrong_header() {
            // A different proposed header derives a different opening key, so the
            // committed trees no longer reconstruct their roots.
            let (_, proof) = honest_fixture();
            let wrong = IncompleteBlockHeader { timestamp: 0x1234_5678, ..IncompleteBlockHeader::new_for_test(EASY_NBITS) };
            let err = verify_fp16_plain_proof(&wrong, &proof, None).expect_err("wrong header must be rejected");
            assert!(format!("{err:#}").contains("root"), "got: {err:#}");
        }

        #[test]
        fn rejects_a_tampered_ancestor_header_via_the_b_side_key() {
            // The B-side tree is keyed by keyB = H_"key-B"(ancestor_header). The B
            // operand was committed under the honest ancestor's keyB, so swapping
            // the proof-carried ancestor (as an out-of-window / unauthenticated
            // ancestor would) re-derives keyB and the committed B root no longer
            // rebuilds: parse_proof REJECTS before any tile replay.
            let (header, proof) = honest_fixture();
            proof.parse_proof(&header).expect("the honest ancestor must open the B tree");
            let mut tampered = proof.clone();
            tampered.job.ancestor_header.prev_block[0] ^= 1;
            let err = match tampered.parse_proof(&header) {
                Ok(_) => panic!("a B-key from an unauthenticated ancestor must be rejected"),
                Err(e) => e,
            };
            assert!(format!("{err:#}").contains("root"), "expected a B-side Merkle rebuild failure, got: {err:#}");
        }

        #[test]
        fn rejects_an_out_of_window_ancestor_key() {
            // A certificate whose B tree is committed under one ancestor's keyB but
            // whose job advertises a different ancestor (e.g. one outside the state
            // window) must fail the B-side Merkle rebuild. We build the B tree under
            // `committed_ancestor` yet carry `claimed_ancestor` in the job.
            let header = IncompleteBlockHeader::new_for_test(EASY_NBITS);
            let committed_ancestor = IncompleteBlockHeader { prev_block: [0x33; 32], ..header };
            let claimed_ancestor = IncompleteBlockHeader { prev_block: [0x44; 32], ..header };
            let mut g = Gen(0x0bad_f00d_dead_beef);
            let a_full = g.operand(M * K);
            let b_full = g.operand(N * K);
            let mut proof = build_proof_with_ancestor(&header, &committed_ancestor, &a_full, &b_full);
            // Baseline: the matching ancestor verifies end-to-end.
            {
                let mut honest = proof.clone();
                honest.job.ancestor_header = committed_ancestor;
                verify_fp16_plain_proof(&header, &honest, None).expect("matching ancestor must verify");
            }
            // Swap in the mismatched (out-of-window) ancestor: keyB changes and the
            // committed B root no longer rebuilds.
            proof.job.ancestor_header = claimed_ancestor;
            let err = verify_fp16_plain_proof(&header, &proof, None)
                .expect_err("an out-of-window ancestor key must be rejected");
            assert!(format!("{err:#}").contains("root"), "expected a B-side Merkle rebuild failure, got: {err:#}");
        }

        #[test]
        fn policy_gate_rejects_a_flat_opened_tile() {
            // The policy gate that `verify_fp16_plain_proof` funnels into rejects
            // a degenerate tile. We exercise it honestly over the commitment
            // layer: commit + open flat +-1 operands, then replay with zero noise
            // (the full cert path cannot present a flat tile to the policy, since
            // the deterministic rank-r noise always injects admissible breakpoint
            // structure; the flat regime only arises absent that noise).
            use super::super::super::params::Fp16Params;
            use super::super::{verify_tile, OperandNoise16};
            use crate::v5::api::commitment::verify_and_open_rows;
            use crate::v5::api::dtype::f32_to_fp16;

            let header = IncompleteBlockHeader::new_for_test(EASY_NBITS);
            let one = f32_to_fp16(1.0).unwrap();
            let proof = build_proof(&header, &vec![one; M * K], &vec![one; N * K]);
            let (rp, cp) = patterns();
            let (h, w) = (rp.tile_size() as usize, cp.tile_size() as usize);
            // A keys on the proposed header, B on the ancestor (= header here).
            let a_rows = verify_and_open_rows(
                &proof.values.a.proof,
                &proof.values.a.row_indices,
                M,
                K,
                HASH,
                key_a(&header),
                &proof.values.a.proof.root,
            )
            .unwrap();
            let b_rows = verify_and_open_rows(
                &proof.values.b.proof,
                &proof.values.b.row_indices,
                N,
                K,
                HASH,
                key_b(&header),
                &proof.values.b.proof.root,
            )
            .unwrap();
            let (ea, fa) = (vec![0u16; h * R], vec![0u16; K * R]);
            let (eb, fb) = (vec![0u16; w * R], vec![0u16; K * R]);
            let na = OperandNoise16 { e: &ea, f: &fa };
            let nb = OperandNoise16 { e: &eb, f: &fb };
            let params = Fp16Params { device: Fp16Device::A100, h, w, k: K, r: R };
            let err = match verify_tile(&params, &a_rows, &b_rows, &na, &nb, &rp, &cp, &[0u8; 32]) {
                Ok(_) => panic!("a flat opened tile must be rejected by the policy"),
                Err(e) => e,
            };
            assert!(format!("{err:#}").contains("not admissible"), "got: {err:#}");
        }
    }
}
