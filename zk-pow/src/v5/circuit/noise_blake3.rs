//! In-circuit derivation of the FP16 noise lines' keyed-BLAKE3-XOF bytes (ZK binding, increment
//! 6e-3c).
//!
//! This closes the last soundness gap of the FP16 noise: in increment 6e-2 the raw keyed-XOF bytes
//! [`crate::v5::circuit::noise_stark::NoiseStark`] normalizes were **free witness**, so the noise
//! was still grindable (a prover could pick any bytes). Here the forked FP16 BLAKE3 engine
//! ([`super::blake3_fp16_stark`]) recomputes those bytes *in-circuit* from the real noise seeds, and
//! egresses each line's output `cv_out` so the egress CTL can bind them to NoiseStark's byte inputs.
//!
//! # The schedule, bit-exact with [`crate::v5::api::noise`]
//!
//! The plaintext draws line `(side, factor, line)` under `seed` as
//! `keyed_blake3(line_key, material64)` where `line_key = subkey(LABEL_NOISE_LINE, seed) =
//! keyed_blake3(seed, LABEL_NOISE_LINE)` and `material64 = side | factor | line(u32 LE) | zero-pad`
//! (see [`crate::v5::api::noise::sample_line_xof_bytes`]). The rank is [`FP16_QUANT_R`] `= 32`, so
//! the first `r = 32` XOF bytes are exactly the output `cv_out` (8 words LE). The schedule is:
//!
//! * **2 subkey compressions** — `subkey_A = keyed_blake3(seedA, label)` and
//!   `subkey_B = keyed_blake3(seedB, label)`, keyed by `seedA`/`seedB` (fed as the engine's
//!   `KEY_A`/`KEY_B` public inputs — the verifier pins them to the ROOT-DERIVED noise seeds
//!   [`crate::v5::circuit::driver::fp16_root_derived_seeds`], so they are a pure function of the
//!   committed operand roots; see the scope note below).
//! * **`h + w + 2k` line compressions** in NoiseStark's line order `[E_A (h), F_A (k), E_B (w),
//!   F_B (k)]`, each keyed by the appropriate subkey's `cv_out` (routed as a `Chain` CV), hashing
//!   that line's `material64`, egressing its `cv_out` under the per-line key base `line * 16`.
//! * the one mandatory lottery compression (the engine requires exactly one).
//!
//! Every message block (the subkey `label` and each line's `material64`) is PUBLIC per-line data; it
//! is pinned verifier-fixed via the engine's general message-word pins
//! ([`Fp16RawBlake3Program::msg_pins`]) so a prover cannot grind the noise through the message. The
//! only remaining witness freedom is the seeds.
//!
//! **Scope (increment 6e-3d — the seed<-root gap is now CLOSED).** The seeds are derived from the
//! committed operand roots: the driver sets this engine's `KEY_A`/`KEY_B` public inputs to
//! `seedA`/`seedB = noise_seeds(keys, {HASH_A, HASH_B}, p)` — exactly
//! [`crate::v5::api::noise::noise_seeds`] over the committed roots, the opening keys and the public
//! parameters ([`crate::v5::circuit::driver::fp16_root_derived_seeds`]). `verify` recomputes those
//! seeds from the proof's pinned `HASH_A`/`HASH_B` + keys + `p` and pins the noise-BLAKE3
//! `KEY_A`/`KEY_B` to them, so a prover can neither choose the noise *bytes* (they are the pinned
//! keyed-BLAKE3 XOF of the seeds) nor the *seeds* (they are forced to be `noise_seeds` of the
//! committed roots). The noise is now fully operand-dependent and bit-exact with the plaintext's
//! root-derived noise — the root-anchored anti-grind property holds.
//!
//! The only remaining test-vs-production boundary is shared with `HASH_A`/`HASH_B`: the opening keys
//! and the `p` encodings are supplied at the circuit boundary (by the driver / tests) rather than
//! derived from the block header in-circuit; the consensus layer supplies these header-derived values
//! the same way it supplies the expected `HASH_A`/`HASH_B`.
//!
//! # Egress -> NoiseStark binding
//!
//! Each line compression's `cv_out` (= the 32 XOF bytes) egresses as 16 little-endian 16-bit limbs
//! on keys `line*16 .. line*16 + 16` ([`super::blake3_fp16_stark::ctl::ctl_cv_egress_looking_blake3`]).
//! NoiseStark's egress-pair looked side ([`super::noise_stark::ctl::ctl_noise_egress_pair_looked`])
//! recomposes its own two consecutive XOF bytes into the same 16-bit limbs on the same keys, so the
//! CTL multiset balance forces NoiseStark's bytes to equal the in-circuit-derived XOF bytes of the
//! root-derived seeds — closing the "free-witness noise bytes" grind, and (6e-3d) the seeds are
//! themselves bound to the committed operand roots, so the noise is fully operand-bound anti-grind.

use pearl_blake3::{B3F_CHUNK_END, B3F_CHUNK_START, B3F_KEYED_HASH, B3F_ROOT, BLAKE3_MSG_LEN, blake3_digest};
use plonky2::hash::hash_types::RichField;

use crate::v5::api::noise::{NoiseFactor, Side, noise_line_label, noise_line_material};
use crate::v4::api::primitives::Hash256;
use crate::v5::circuit::blake3_commit::bytes32_to_words;
use crate::v5::circuit::blake3_fp16_stark::columns::{PI_HASH_JACKPOT, PI_KEY_A, PI_KEY_B};
use crate::v5::circuit::blake3_fp16_stark::{
    Fp16RawBlake3Instruction, Fp16RawBlake3Program, Fp16RawCvSource, Fp16RawMessageSource,
};

/// The keyed-BLAKE3 flags of a one-block root hash (chunk start + end + root + keyed).
const KEYED_ONE_BLOCK: u32 = (B3F_CHUNK_START | B3F_CHUNK_END | B3F_ROOT | B3F_KEYED_HASH) as u32;

/// The 16 little-endian `u32` words of one 64-byte message block — the per-row word pins the engine
/// fixes a compression's message to (`word 2j` / `word 2j+1` on row `j`).
fn block_pin_words(block: &[u8; 64]) -> [u32; 16] {
    core::array::from_fn(|i| u32::from_le_bytes(block[4 * i..4 * i + 4].try_into().unwrap()))
}

/// The `(side, factor, line_index)` address of NoiseStark line `L` for an `h x w` tile with inner
/// `k`, in the committed line order `[E_A (h), F_A (k), E_B (w), F_B (k)]`. `None` past the last
/// line.
fn line_address(l: usize, h: usize, w: usize, k: usize) -> Option<(Side, NoiseFactor, u32)> {
    if l < h {
        Some((Side::A, NoiseFactor::E, l as u32))
    } else if l < h + k {
        Some((Side::A, NoiseFactor::F, (l - h) as u32))
    } else if l < h + k + w {
        Some((Side::B, NoiseFactor::E, (l - h - k) as u32))
    } else if l < h + k + w + k {
        Some((Side::B, NoiseFactor::F, (l - h - k - w) as u32))
    } else {
        None
    }
}

/// `E_A` lines key off `seedA` (subkey_A); `F_A`, `E_B`, `F_B` off `seedB` (subkey_B) — the exact
/// seed assignment of [`crate::v5::api::noise::sample_noise`].
fn line_subkey_instr(l: usize, h: usize) -> usize {
    if l < h { SUBKEY_A_INSTR } else { SUBKEY_B_INSTR }
}

/// Instruction index of the `subkey_A = keyed_blake3(seedA, label)` compression.
pub const SUBKEY_A_INSTR: usize = 0;
/// Instruction index of the `subkey_B = keyed_blake3(seedB, label)` compression.
pub const SUBKEY_B_INSTR: usize = 1;
/// Instruction index of line `L` (the two subkeys come first).
pub const fn line_instr(l: usize) -> usize {
    2 + l
}
/// The number of noise lines of an `h x w` tile with inner `k`.
pub const fn num_lines(h: usize, w: usize, k: usize) -> usize {
    h + w + 2 * k
}

/// The egress key base of line `L` (16 limbs per line, disjoint 16-wide ranges). NoiseStark's
/// egress-pair looked side uses the same `L*16 + j` keys.
pub const fn line_egress_base(l: usize) -> u64 {
    (l as u64) * 16
}

/// Builds the FP16 noise-derivation BLAKE3 program for an `h x w` tile with inner `k`: the two
/// subkey compressions, the `h + w + 2k` line compressions (each egressing its `cv_out`), and the
/// mandatory lottery compression. Every message block is pinned to its public constant via
/// `msg_pins`, so only the seeds (the engine's `KEY_A`/`KEY_B`, bound to the committed roots at the
/// batch) are witness. `height_bits`, when `Some`, forces an on-ladder trace height.
pub fn noise_blake3_program(h: usize, w: usize, k: usize, height_bits: Option<usize>) -> Fp16RawBlake3Program {
    let lines = num_lines(h, w, k);
    let label = noise_line_label();
    assert!(label.len() <= BLAKE3_MSG_LEN, "noise-line label must fit one BLAKE3 block");

    // Aux message 0 is the shared `label || zero-pad` block; aux message 1 + L is line L's material64.
    let mut label_block = [0u8; 64];
    label_block[..label.len()].copy_from_slice(label);
    let label_pins = block_pin_words(&label_block);

    let mut instructions = Vec::with_capacity(lines + 3);
    let mut msg_pins: Vec<(usize, [u32; 16])> = Vec::with_capacity(lines + 2);

    // subkey_A (keyed by seedA = KEY_A) and subkey_B (keyed by seedB = KEY_B), both hashing `label`.
    for (instr_idx, key) in [(SUBKEY_A_INSTR, Fp16RawCvSource::KeyA), (SUBKEY_B_INSTR, Fp16RawCvSource::KeyB)] {
        instructions.push(Fp16RawBlake3Instruction {
            cv: key,
            msg: Fp16RawMessageSource::AuxBytes { idx: 0 },
            counter: 0,
            block_len: label.len() as u32,
            flags: KEYED_ONE_BLOCK,
            bind: None,
            egress: None,
        });
        msg_pins.push((instr_idx, label_pins));
    }

    // Line compressions, each keyed by its subkey's cv_out (Chain) and egressing cv_out.
    for l in 0..lines {
        let (side, factor, line) = line_address(l, h, w, k).expect("line index in range");
        let material = noise_line_material(side, factor, line);
        let instr_idx = line_instr(l);
        instructions.push(Fp16RawBlake3Instruction {
            cv: Fp16RawCvSource::Chain(line_subkey_instr(l, h)),
            msg: Fp16RawMessageSource::AuxBytes { idx: 1 + l },
            counter: 0,
            block_len: BLAKE3_MSG_LEN as u32,
            flags: KEYED_ONE_BLOCK,
            bind: None,
            egress: Some(line_egress_base(l)),
        });
        msg_pins.push((instr_idx, block_pin_words(&material)));
    }

    instructions.push(Fp16RawBlake3Instruction::lottery());

    Fp16RawBlake3Program {
        instructions,
        num_aux_msgs: 1 + lines,
        num_aux_cvs: 0,
        routing_pins: Vec::new(),
        moe: None,
        msg_pins,
        height_bits,
    }
}

/// The witness aux-message blocks for [`noise_blake3_program`]: aux 0 = `label || zero-pad`, aux
/// `1 + L` = line `L`'s `material64`. Seed-independent (the seeds enter as `KEY_A`/`KEY_B`), so the
/// prover has no freedom here — every block equals the pinned public constant.
pub fn noise_blake3_aux_msgs(h: usize, w: usize, k: usize) -> Vec<[u8; 64]> {
    let lines = num_lines(h, w, k);
    let label = noise_line_label();
    let mut aux = Vec::with_capacity(1 + lines);
    let mut label_block = [0u8; 64];
    label_block[..label.len()].copy_from_slice(label);
    aux.push(label_block);
    for l in 0..lines {
        let (side, factor, line) = line_address(l, h, w, k).expect("line index in range");
        aux.push(noise_line_material(side, factor, line));
    }
    aux
}

/// The engine's `(KEY_A, KEY_B)` word form for the two noise seeds: `KEY_A = seedA`, `KEY_B = seedB`.
/// The subkey compressions key off these, so binding them (as the engine's public inputs) to the
/// natively-recomputed [`crate::v5::api::noise::noise_seeds`] is what makes the whole derivation
/// seed-bound.
pub fn noise_seed_key_words(seed_a: &Hash256, seed_b: &Hash256) -> ([u32; 8], [u32; 8]) {
    (bytes32_to_words(seed_a), bytes32_to_words(seed_b))
}

/// The noise-BLAKE3 program's fixed jackpot key / lottery words (the engine requires one lottery
/// compression; the noise derivation does not use it, so it is pinned to a constant and its
/// `HASH_JACKPOT` is an inert public input).
pub const NOISE_BLAKE3_JACKPOT_KEY: Hash256 = [0u8; 32];
/// The noise-BLAKE3 program's fixed lottery message words (all zero; see [`NOISE_BLAKE3_JACKPOT_KEY`]).
pub const NOISE_BLAKE3_LOTTERY_WORDS: [u32; 16] = [0u32; 16];

/// The noise-BLAKE3 table's 64 public inputs for seeds `(seed_a, seed_b)`: `KEY_A = seedA`,
/// `KEY_B = seedB` (the pinned seeds the verifier checks — the seed binding), `JACKPOT_KEY =
/// NOISE_BLAKE3_JACKPOT_KEY`, `HASH_JACKPOT` the inert lottery digest, every other slot zero (the
/// program binds no operand/routing/offsets hash). Deterministic, so [`crate::v5::circuit::driver`]
/// can pin them in `verify` without reproving.
pub fn noise_blake3_public_inputs<F: RichField>(seed_a: &Hash256, seed_b: &Hash256) -> Vec<F> {
    let mut pis = vec![F::ZERO; crate::v5::circuit::blake3_fp16_stark::columns::NUM_FP16_RAW_BLAKE3_PUBLIC_INPUTS];
    let (ka, kb) = noise_seed_key_words(seed_a, seed_b);
    for i in 0..8 {
        pis[PI_KEY_A + i] = F::from_canonical_u32(ka[i]);
        pis[PI_KEY_B + i] = F::from_canonical_u32(kb[i]);
    }
    // KEY space PI_JACKPOT_KEY (16..24): NOISE_BLAKE3_JACKPOT_KEY is all-zero, so those stay zero.
    let lottery_bytes: Vec<u8> = NOISE_BLAKE3_LOTTERY_WORDS.iter().flat_map(|w| w.to_le_bytes()).collect();
    let jackpot = blake3_digest(&lottery_bytes, Some(NOISE_BLAKE3_JACKPOT_KEY));
    let jackpot_words = bytes32_to_words(&jackpot);
    for i in 0..8 {
        pis[PI_HASH_JACKPOT + i] = F::from_canonical_u32(jackpot_words[i]);
    }
    pis
}

/// The on-ladder trace height (in degree bits) for the noise-BLAKE3 table of an `h x w` tile with
/// inner `k`: the smallest [`crate::v5::circuit::driver::FP16_REACHABLE_DEGREE_BITS`] member whose
/// height covers the live rows (`8 * (num_lines + 3)` — two subkeys, the lines, the lottery). Snapping
/// to the ladder keeps the table on the consensus fold ladder (the universal-wrapper prerequisite).
pub fn noise_blake3_height_bits(h: usize, w: usize, k: usize) -> usize {
    let live_rows = 8 * (num_lines(h, w, k) + 3);
    crate::v5::circuit::driver::FP16_REACHABLE_DEGREE_BITS
        .iter()
        .copied()
        .filter(|&b| (1usize << b) >= live_rows)
        .min()
        .expect("noise-BLAKE3 live rows exceed the top ladder height")
}

#[cfg(test)]
mod tests {
    use core::borrow::Borrow;

    use plonky2::field::goldilocks_field::GoldilocksField;
    use plonky2::field::types::{Field, PrimeField64};
    use starky::constraint_consumer::ConstraintConsumer;
    use starky::evaluation_frame::{StarkEvaluationFrame, StarkFrame};
    use starky::stark::Stark;

    use super::*;
    use crate::v5::api::noise::{sample_line_xof_bytes, subkey};
    use crate::v4::api::public_params::HashId;
    use crate::v5::circuit::blake3_fp16_stark::columns::{
        FP16_RAW_BLAKE3_COL_MAP, NUM_FP16_RAW_BLAKE3_COLUMNS, NUM_FP16_RAW_BLAKE3_PUBLIC_INPUTS,
    };
    use crate::v5::circuit::blake3_fp16_stark::{Fp16RawBlake3ColumnsView, Fp16RawBlake3Stark, Fp16RawBlake3TraceInputs};

    const D: usize = 2;
    type F = GoldilocksField;
    type S = Fp16RawBlake3Stark<F, D>;

    fn trace_inputs<'a>(aux: &'a [[u8; 64]], key_a: [u32; 8], key_b: [u32; 8]) -> Fp16RawBlake3TraceInputs<'a> {
        Fp16RawBlake3TraceInputs {
            a_values: &[],
            a_scales: &[],
            b_values: &[],
            b_scales: &[],
            routing_words: &[],
            offsets_words: &[],
            aux_msgs: aux,
            aux_cvs: &[],
            lottery_words: [0; 16],
            key_a,
            key_b,
            jackpot_key: [0x5a5a_5a5au32; 8],
            a_hash_id: HashId::Blake3Chunk1024,
            b_hash_id: HashId::Blake3Chunk1024,
            routing_hash_id: HashId::Blake3Chunk1024,
            offsets_hash_id: HashId::Blake3Chunk1024,
        }
    }

    fn constraints_violated(
        stark: &S,
        rows: &[[F; NUM_FP16_RAW_BLAKE3_COLUMNS]],
        pis: &[F; NUM_FP16_RAW_BLAKE3_PUBLIC_INPUTS],
    ) -> bool {
        let n = rows.len();
        (0..n).any(|i| {
            let frame = StarkFrame::from_values(&rows[i], &rows[(i + 1) % n], pis);
            let mut consumer = ConstraintConsumer::new(
                vec![F::from_canonical_u64(2), F::from_canonical_u64(0x876543210)],
                F::from_bool(i != n - 1),
                F::from_bool(i == 0),
                F::from_bool(i == n - 1),
            );
            stark.eval_packed_generic(&frame, &mut consumer);
            consumer.accumulators().into_iter().any(|acc| acc != F::ZERO)
        })
    }

    /// The row index of a compression's finalization row (row 7 of instruction `c`).
    fn fin_row(c: usize) -> usize {
        8 * c + 7
    }

    /// Read a finalization row's `cv_out` as 32 LE bytes.
    fn cv_out_bytes(row: &Fp16RawBlake3ColumnsView<F>) -> [u8; 32] {
        let mut out = [0u8; 32];
        for w in 0..8 {
            let word = row.cv_out[w].to_canonical_u64() as u32;
            out[4 * w..4 * w + 4].copy_from_slice(&word.to_le_bytes());
        }
        out
    }

    /// The in-circuit program reproduces, bit-for-bit, the plaintext noise derivation: each subkey's
    /// `cv_out == subkey(LABEL_NOISE_LINE, seed)` and each line's egressed `cv_out ==
    /// sample_line_xof_bytes(seed, side, factor, line, 32)`; the whole forked AIR is satisfied.
    #[test]
    fn noise_blake3_is_bit_exact_vs_sample_line_xof() {
        let seed_a: Hash256 = [0x3c; 32];
        let seed_b: Hash256 = [0xa7; 32];
        // A spread of tile shapes (small, on the wrapper envelope k >= 16 and asymmetric).
        for (h, w, k) in [(1usize, 1usize, 16usize), (4, 4, 16), (2, 8, 32)] {
            let program = noise_blake3_program(h, w, k, None);
            let aux = noise_blake3_aux_msgs(h, w, k);
            let (ka, kb) = noise_seed_key_words(&seed_a, &seed_b);
            let data = trace_inputs(&aux, ka, kb);
            let (rows, pis) = program.generate_trace::<F>(&data);

            // (a) The two subkeys equal subkey(LABEL, seed).
            let sub_a: &Fp16RawBlake3ColumnsView<F> = rows[fin_row(SUBKEY_A_INSTR)].borrow();
            let sub_b: &Fp16RawBlake3ColumnsView<F> = rows[fin_row(SUBKEY_B_INSTR)].borrow();
            assert_eq!(cv_out_bytes(sub_a), subkey(noise_line_label(), Some(&seed_a)), "subkey_A mismatch {h}x{w}x{k}");
            assert_eq!(cv_out_bytes(sub_b), subkey(noise_line_label(), Some(&seed_b)), "subkey_B mismatch {h}x{w}x{k}");

            // (b) Every line's egressed cv_out equals sample_line_xof_bytes over the right seed.
            for l in 0..num_lines(h, w, k) {
                let (side, factor, line) = line_address(l, h, w, k).unwrap();
                let seed = if l < h { seed_a } else { seed_b };
                let row: &Fp16RawBlake3ColumnsView<F> = rows[fin_row(line_instr(l))].borrow();
                // The egress flag and key base ride this row.
                assert_eq!(row.is_egress_cv, F::ONE, "line {l} must egress");
                assert_eq!(row.ctl_key_base, F::from_canonical_u64(line_egress_base(l)), "line {l} egress key");
                // cv_out == the first r = 32 XOF bytes of the plaintext draw.
                let want = sample_line_xof_bytes(&seed, side_copy(side), factor_copy(factor), line, 32);
                assert_eq!(cv_out_bytes(row).as_slice(), want.as_slice(), "line {l} ({h}x{w}x{k}) XOF mismatch");
                // The egress limbs recompose to cv_out[w] (what the egress CTL exports).
                for w2 in 0..8 {
                    let lo = row.cv_egress_limbs[2 * w2].to_canonical_u64();
                    let hi = row.cv_egress_limbs[2 * w2 + 1].to_canonical_u64();
                    assert_eq!((lo + (hi << 16)) as u32, row.cv_out[w2].to_canonical_u64() as u32, "limb recompose");
                }
            }

            // (c) The whole forked AIR (keyed chain, message-word pins, egress) is satisfied.
            assert!(!constraints_violated(&S::new(program), &rows, &pis), "forked AIR violated for {h}x{w}x{k}");
        }
    }

    /// Grinding the noise is fail-closed: changing a line's material (the public address block) away
    /// from its pinned constant breaks the engine's word-pin constraint, and changing the seed
    /// changes every derived byte (so a prover cannot pick a favorable line that isn't the real
    /// seed-derived one).
    #[test]
    fn material_pin_and_seed_are_load_bearing() {
        let (h, w, k) = (2usize, 2usize, 16usize);
        let seed_a: Hash256 = [0x11; 32];
        let seed_b: Hash256 = [0x22; 32];
        let program = noise_blake3_program(h, w, k, None);
        let mut aux = noise_blake3_aux_msgs(h, w, k);
        let (ka, kb) = noise_seed_key_words(&seed_a, &seed_b);

        // Honest trace passes.
        let (rows, pis) = program.generate_trace::<F>(&trace_inputs(&aux, ka, kb));
        assert!(!constraints_violated(&S::new(program.clone()), &rows, &pis), "honest must pass");

        // Tamper line 0's material (aux idx 1) away from its pinned public constant: the pins still
        // hold the honest words, so the ingested word != WORD_PIN and the pin constraint rejects.
        aux[1][0] ^= 0xFF;
        let (bad_rows, bad_pis) = program.generate_trace::<F>(&trace_inputs(&aux, ka, kb));
        assert!(
            constraints_violated(&S::new(program), &bad_rows, &bad_pis),
            "a tampered (unpinned) material block must break the word-pin constraint"
        );

        // A different seed yields entirely different XOF bytes (grinding the seed is only possible by
        // changing the committed roots, which the batch binds).
        let other = sample_line_xof_bytes(&[0x33; 32], Side::A, NoiseFactor::E, 0, 32);
        let honest = sample_line_xof_bytes(&seed_a, Side::A, NoiseFactor::E, 0, 32);
        assert_ne!(other, honest, "distinct seeds must give distinct XOF bytes");
    }

    // `Side`/`NoiseFactor` are not `Copy`; rebuild them from the address for the second use.
    fn side_copy(s: Side) -> Side {
        match s as u8 {
            0 => Side::A,
            _ => Side::B,
        }
    }
    fn factor_copy(f: NoiseFactor) -> NoiseFactor {
        match f as u8 {
            0 => NoiseFactor::E,
            _ => NoiseFactor::F,
        }
    }
}
