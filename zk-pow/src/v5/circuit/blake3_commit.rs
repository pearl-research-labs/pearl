//! FP16 driver for the shared, scheme-neutral BLAKE3 STARK engine
//! ([`crate::v4::circuit::blake3_stark`]).
//!
//! The FP8 bridge [`Blake3Program::from_blake_program`] hard-asserts the FP8 prequant four-plane
//! shape (A values / A scales / B values / B scales, each root produced exactly once) and panics
//! on anything else. The FP16 scheme instead commits **one keyed-BLAKE3 Merkle tree per operand
//! directly over the operand's `u16` LE row bytes** (no scales, no prequant fold — see
//! [`crate::v5::api::commitment::commit_operand`]), and folds the 16 lottery words into
//! `HASH_JACKPOT` under `JACKPOT_KEY`. So `from_blake_program` cannot be used.
//!
//! This module assembles the [`Blake3Instruction`] schedule **directly** for the FP16 shape,
//! reusing the shared [`Blake3Stark`] AIR, [`Blake3Program::generate_trace`], and public-input
//! layout unchanged. Two things are proven, bit-exact against the plaintext:
//!
//! * **jackpot hash** ([`jackpot_program`]): one keyed compression of the 16 lottery words under
//!   `JACKPOT_KEY`, output bound to `HASH_JACKPOT` — bit-exact with
//!   [`crate::v4::api::transcript::compute_jackpot_ticket`] (the FP16 scheme reuses the
//!   scheme-neutral jackpot transcript). This is [`Blake3Instruction::lottery`], already supported
//!   by the engine.
//! * **operand Merkle roots** ([`operand_tree_program`] / [`fp16_blake3_program`]): one keyed tree
//!   per operand over the chunk-padded `u16` row bytes, root bound to `HASH_A` / `HASH_B` —
//!   bit-exact with `commit_operand(..).root()` (which is `pearl_blake3::MerkleTree::with_chunk_len`
//!   under the side key). The per-leaf chunk, chunk-chaining, odd-tail promotion and ROOT
//!   finalization mirror [`pearl_blake3::MerkleTree`] for any `HashId` leaf size (128/256/512/1024).
//!
//! Scope: this delivers the Blake3 *constructor + generate_trace + public-binding* step (design
//! doc `docs/fp16_scheme/zk_binding_design.md` §5a item 2). It does **not** wire the batch driver
//! or the noisy_quant / XorFold counterparties; the CTL looking-sides are exposed here as
//! parameterized hooks ([`ctl_lottery_words_looking_blake3`], [`ctl_operand_bytes_looking_blake3`])
//! for the batch-integration step, exactly as the sibling FP16 sub-STARKs expose theirs.

use pearl_blake3::{B3F_CHUNK_END, B3F_CHUNK_START, B3F_KEYED_HASH, B3F_PARENT, B3F_ROOT, BLAKE3_MSG_LEN};

use crate::v4::api::public_params::HashId;
use crate::v4::circuit::blake3_stark::{
    Blake3Instruction, Blake3Program, CvRef, CvSource, MessageSource, PlaneId, PublicBinding,
};

/// 64-byte blocks per `chunk_len`-byte Merkle leaf (BLAKE3 blocks are 64 bytes).
fn blocks_per_leaf(chunk_len: usize) -> usize {
    chunk_len / BLAKE3_MSG_LEN
}

/// Appends one operand's keyed Merkle tree over `raw_len` chunk-padded `u16` row bytes, returning
/// the index of the root instruction (whose `bind` is `bind`). Mirrors
/// [`pearl_blake3::MerkleTree::with_chunk_len`] bit-for-bit:
///
/// * each leaf is a BLAKE3 chunk of `chunk_len` bytes (`chunk_len / 64` blocks), chunk counter =
///   leaf index, `CHUNK_START` on block 0 and `CHUNK_END` on the last block, chained in-table
///   ([`CvSource::Chain`]) across the leaf's blocks;
/// * leaf CVs are paired bottom-up with keyed `PARENT` compressions, odd tails promoted unchanged;
/// * the single top merge carries `ROOT`; a one-leaf tree instead ROOT-finalizes that leaf's last
///   block (matching `MerkleTree`'s `data.len() <= chunk_len` branch, `hasher.hash(data)`).
///
/// `plane` selects the committed byte stream and `key` its opening key (`KeyA`/`KeyB`); the trace
/// inputs must supply that plane's bytes padded under the same `HashId`. `ctl_base` is the byte
/// offset of the plane's first element in the operand stream (0 for a single-operand program).
fn append_operand_tree(
    instrs: &mut Vec<Blake3Instruction>,
    plane: PlaneId,
    key: CvSource,
    raw_len: usize,
    chunk_len: usize,
    ctl_base: u64,
    bind: Option<PublicBinding>,
) -> usize {
    assert!(
        HashId::from_chunk_len(chunk_len).is_some(),
        "chunk_len {chunk_len} is not an allowed BLAKE3 Merkle leaf size"
    );
    assert!(
        matches!(key, CvSource::KeyA | CvSource::KeyB),
        "operand tree key must be KeyA/KeyB"
    );
    let bpl = blocks_per_leaf(chunk_len);
    // Chunk-padded leaf count (matches HashId::padded_len / MerkleTree leaf count).
    let padded = raw_len.div_ceil(chunk_len).max(1) * chunk_len;
    let leaves = padded / chunk_len;

    // ---- Leaf chunks. ----
    let mut cvs: Vec<usize> = Vec::with_capacity(leaves);
    for leaf in 0..leaves {
        for b in 0..bpl {
            let mut flags = B3F_KEYED_HASH as u32;
            if b == 0 {
                flags |= B3F_CHUNK_START as u32;
            }
            let last_block = b + 1 == bpl;
            // A single-leaf tree ROOT-finalizes the leaf itself (MerkleTree's `hash(data)` branch).
            let is_tree_root = leaves == 1 && last_block;
            if last_block {
                flags |= B3F_CHUNK_END as u32;
                if is_tree_root {
                    flags |= B3F_ROOT as u32;
                }
            }
            let offset = leaf * chunk_len + b * BLAKE3_MSG_LEN;
            instrs.push(Blake3Instruction {
                cv: if b == 0 {
                    key
                } else {
                    CvSource::Chain(instrs.len() - 1)
                },
                msg: MessageSource::PlaneBytes {
                    plane,
                    offset,
                    ctl_base: ctl_base + offset as u64,
                },
                counter: leaf as u64,
                block_len: BLAKE3_MSG_LEN as u32,
                flags,
                bind: if is_tree_root { bind } else { None },
            });
        }
        cvs.push(instrs.len() - 1);
    }

    // ---- Parent layers (keyed), bottom-up, odd tail promoted; top merge is ROOT. ----
    while cvs.len() > 1 {
        let is_root_layer = cvs.len() == 2;
        let mut next = Vec::with_capacity(cvs.len().div_ceil(2));
        for pair in cvs.chunks(2) {
            if let [left, right] = *pair {
                let mut flags = (B3F_KEYED_HASH | B3F_PARENT) as u32;
                if is_root_layer {
                    flags |= B3F_ROOT as u32;
                }
                instrs.push(Blake3Instruction {
                    cv: key,
                    msg: MessageSource::Parent {
                        left: CvRef::Instruction(left),
                        right: CvRef::Instruction(right),
                    },
                    counter: 0,
                    block_len: BLAKE3_MSG_LEN as u32,
                    flags,
                    bind: if is_root_layer { bind } else { None },
                });
                next.push(instrs.len() - 1);
            } else {
                next.push(pair[0]); // Odd tail: promoted unchanged.
            }
        }
        cvs = next;
    }
    cvs[0]
}

/// The jackpot-only program: one keyed compression of the 16 lottery words under `JACKPOT_KEY`,
/// output bound to `HASH_JACKPOT`. This is the smallest FP16 Blake3 statement and its XorFold
/// lottery-words counterparty ([`crate::v5::circuit::xor_fold_stark`]) already exists.
pub fn jackpot_program() -> Blake3Program {
    Blake3Program {
        instructions: vec![Blake3Instruction::lottery()],
        num_aux_msgs: 0,
        num_aux_cvs: 0,
        routing_pins: Vec::new(),
        moe: None,
    }
}

/// A single operand's tree (root bound to `bind`), plus the mandatory lottery compression (the
/// engine's [`Blake3Program::validate`] requires exactly one). `side` picks the opening key and
/// committed plane.
pub fn operand_tree_program(raw_len: usize, chunk_len: usize, bind: PublicBinding) -> Blake3Program {
    let (plane, key) = match bind {
        PublicBinding::HashA => (PlaneId::AValues, CvSource::KeyA),
        PublicBinding::HashB => (PlaneId::BValues, CvSource::KeyB),
        other => panic!("operand_tree_program binds HashA/HashB, not {other:?}"),
    };
    let mut instructions = Vec::new();
    append_operand_tree(&mut instructions, plane, key, raw_len, chunk_len, 0, Some(bind));
    instructions.push(Blake3Instruction::lottery());
    Blake3Program {
        instructions,
        num_aux_msgs: 0,
        num_aux_cvs: 0,
        routing_pins: Vec::new(),
        moe: None,
    }
}

/// The full FP16 Blake3 program: operand-A tree (→ `HASH_A`, keyed `KEY_A`), operand-B tree
/// (→ `HASH_B`, keyed `KEY_B`), and the lottery compression (→ `HASH_JACKPOT`). `a_len`/`b_len`
/// are the raw (pre-chunk-padding) `u16` row-byte lengths (`num_rows * k * 2`); `a_chunk`/`b_chunk`
/// are the per-side [`HashId::chunk_len`]. The B plane's CTL keys are offset past A's byte range so
/// a future operand-bytes channel keeps the two sides disjoint.
pub fn fp16_blake3_program(a_len: usize, a_chunk: usize, b_len: usize, b_chunk: usize) -> Blake3Program {
    let mut instructions = Vec::new();
    append_operand_tree(
        &mut instructions,
        PlaneId::AValues,
        CvSource::KeyA,
        a_len,
        a_chunk,
        0,
        Some(PublicBinding::HashA),
    );
    append_operand_tree(
        &mut instructions,
        PlaneId::BValues,
        CvSource::KeyB,
        b_len,
        b_chunk,
        a_len as u64,
        Some(PublicBinding::HashB),
    );
    instructions.push(Blake3Instruction::lottery());
    Blake3Program {
        instructions,
        num_aux_msgs: 0,
        num_aux_cvs: 0,
        routing_pins: Vec::new(),
        moe: None,
    }
}

/// 8 LE `u32` limbs of a 32-byte key / hash (the [`crate::v4::circuit::blake3_stark::Blake3TraceInputs`]
/// key-word form, and the public-input word form).
pub fn bytes32_to_words(b: &[u8; 32]) -> [u32; 8] {
    core::array::from_fn(|i| u32::from_le_bytes(b[4 * i..4 * i + 4].try_into().unwrap()))
}

/// The inverse of [`bytes32_to_words`]: the 32 LE bytes of 8 `u32` public-input / key words.
pub fn words_to_bytes32(w: &[u32; 8]) -> [u8; 32] {
    core::array::from_fn(|i| w[i / 4].to_le_bytes()[i % 4])
}

// ==================================================================================================
// Parameterized CTL looking-side hooks (for the batch-integration step; not wired here).
// ==================================================================================================

use plonky2::field::types::Field;
use starky::cross_table_lookup::{TableIdx, TableWithColumns};
use starky::lookup::{Column, Filter};

use crate::v4::circuit::blake3_stark::columns::BLAKE3_COL_MAP;

/// Blake3's looking side of the **lottery words** channel: 16 `(word_pos, BLAKE3_MSG[pos])` tuples
/// on the lottery message-load row (row 0 holds the message), filter
/// `IS_BIND_JACKPOT_HASH * IS_NEW_BLAKE` (fires exactly once). The counterparty is
/// [`crate::v5::circuit::xor_fold_stark::ctl::ctl_lottery_words_looked_xor_fold`] (filter
/// `IS_LANE_FINAL`): pairing both sides proves the 16 jackpot message words are exactly XorFold's
/// 16 lane outputs, and `HASH_JACKPOT` then binds their keyed BLAKE3 digest. Mirrors the FP8
/// `ctl_lottery_words_looking_blake3`, but takes the Blake3 table's batch index explicitly (the
/// FP16 batch does not yet register the table — the index is supplied at integration time).
pub fn ctl_lottery_words_looking_blake3<F: Field>(blake3_table: usize) -> Vec<TableWithColumns<F>> {
    let m = &BLAKE3_COL_MAP;
    (0..16)
        .map(|j| {
            TableWithColumns::new(
                TableIdx::from(blake3_table),
                vec![Column::constant(F::from_canonical_usize(j)), Column::single(m.blake3_msg[j])],
                Filter::new(
                    vec![(Column::single(m.is_bind_jackpot_hash), Column::single(m.is_new_blake))],
                    vec![],
                ),
            )
        })
        .collect()
}

/// Blake3's looking side of the **operand bytes** channel: 4 `(CTL_KEY_BASE + 2j, UINT8_DATA[2j] +
/// 2^8*UINT8_DATA[2j+1])` tuples per committed operand-values row — one `u16` operand element per
/// tuple (LE), keyed by its low byte's offset in the operand stream. Filter `IS_INT8_MESSAGE`,
/// which [`Blake3Program::generate_trace`] sets on exactly the live (non-chunk-padding) rows of an
/// `AValues`/`BValues` plane, so every committed `u16` crosses exactly once and padding/parent rows
/// do not. The B plane's `CTL_KEY_BASE` carries the `a_len` byte offset
/// ([`fp16_blake3_program`]), keeping the two operands' key spaces disjoint on one shared channel.
///
/// The pair packing is sound because every message byte is BYTES2-checked by the engine's
/// `blake3_lut_lookups`. This is the Blake3↔noisy_quant counterparty of the design doc §4; the
/// `noisy_quant` looked side (keyed by the same byte offset, value = the raw `u16`) is the
/// integration-step follow-on. Takes the Blake3 table's batch index explicitly.
pub fn ctl_operand_bytes_looking_blake3<F: Field>(blake3_table: usize) -> Vec<TableWithColumns<F>> {
    let m = &BLAKE3_COL_MAP;
    let byte_shift = F::from_canonical_u64(1 << 8);
    (0..4)
        .map(|j| {
            TableWithColumns::new(
                TableIdx::from(blake3_table),
                vec![
                    Column::linear_combination_with_constant([(m.ctl_key_base, F::ONE)], F::from_canonical_usize(2 * j)),
                    Column::linear_combination([(m.uint8_data[2 * j], F::ONE), (m.uint8_data[2 * j + 1], byte_shift)]),
                ],
                Filter::from_column(Column::single(m.is_int8_message)),
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use plonky2::field::goldilocks_field::GoldilocksField;
    use plonky2::field::types::PrimeField64;
    use starky::constraint_consumer::ConstraintConsumer;
    use starky::evaluation_frame::{StarkEvaluationFrame, StarkFrame};
    use starky::stark::Stark;

    use super::*;
    use crate::v5::api::commitment::commit_operand;
    use crate::v5::api::dtype::f32_to_fp16;
    use crate::v4::api::transcript::compute_jackpot_ticket;
    use crate::v4::circuit::blake3_stark::columns::{
        NUM_BLAKE3_COLUMNS, NUM_BLAKE3_PUBLIC_INPUTS, PI_HASH_A, PI_HASH_B, PI_HASH_JACKPOT,
    };
    use crate::v4::circuit::blake3_stark::{Blake3Stark, Blake3TraceInputs};
    use pearl_blake3::blake3_digest;

    const D: usize = 2;
    type F = GoldilocksField;
    type S = Blake3Stark<F, D>;

    /// Read an 8-word public-input hash as 32 LE bytes.
    fn pi_bytes(pis: &[F; NUM_BLAKE3_PUBLIC_INPUTS], base: usize) -> [u8; 32] {
        let w: [u32; 8] = core::array::from_fn(|i| pis[base + i].to_canonical_u64() as u32);
        words_to_bytes32(&w)
    }

    /// Full AIR check over every row pair, including the last→first wrap (the real vanishing
    /// polynomial). Returns true if any constraint is violated. Mirrors the fp8 blake3 test harness.
    fn constraints_violated(stark: &S, rows: &[[F; NUM_BLAKE3_COLUMNS]], pis: &[F; NUM_BLAKE3_PUBLIC_INPUTS]) -> bool {
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

    /// Minimal Blake3TraceInputs for a program that uses only the AValues/BValues planes + lottery
    /// (no scales, routing, offsets, aux).
    fn inputs<'a>(
        a_values: &'a [u8],
        b_values: &'a [u8],
        lottery_words: [u32; 16],
        key_a: [u32; 8],
        key_b: [u32; 8],
        jackpot_key: [u32; 8],
        a_hash_id: HashId,
        b_hash_id: HashId,
    ) -> Blake3TraceInputs<'a> {
        Blake3TraceInputs {
            a_values,
            a_scales: &[],
            b_values,
            b_scales: &[],
            routing_words: &[],
            offsets_words: &[],
            aux_msgs: &[],
            aux_cvs: &[],
            lottery_words,
            key_a,
            key_b,
            jackpot_key,
            a_hash_id,
            b_hash_id,
            routing_hash_id: HashId::Blake3Chunk1024,
            offsets_hash_id: HashId::Blake3Chunk1024,
        }
    }

    /// A finite FP16 row-major operand of spread magnitudes.
    fn operand(num_rows: usize, k: usize, salt: u32) -> Vec<u16> {
        (0..num_rows * k)
            .map(|i| f32_to_fp16(((i as u32).wrapping_mul(salt) % 193) as f32 * 0.25 - 24.0).unwrap())
            .collect()
    }

    // ---------------------------------------------------------------- jackpot

    #[test]
    fn jackpot_hash_is_bit_exact_vs_plaintext() {
        let seed_a = [0x5au8; 32];
        let jk = crate::v4::api::transcript::jackpot_key(&seed_a);
        let msg: [u8; 64] = core::array::from_fn(|i| (i as u8).wrapping_mul(37).wrapping_add(11));
        let lottery_words: [u32; 16] = core::array::from_fn(|i| u32::from_le_bytes(msg[4 * i..4 * i + 4].try_into().unwrap()));

        let program = jackpot_program();
        let data = inputs(
            &[],
            &[],
            lottery_words,
            [0; 8],
            [0; 8],
            bytes32_to_words(&jk),
            HashId::Blake3Chunk1024,
            HashId::Blake3Chunk1024,
        );
        let (rows, pis) = program.generate_trace::<F>(&data);

        // Bit-exact against the scheme-neutral jackpot transcript the plaintext verifier uses.
        let expected = compute_jackpot_ticket(&seed_a, &msg).jackpot;
        assert_eq!(pi_bytes(&pis, PI_HASH_JACKPOT), expected, "HASH_JACKPOT != compute_jackpot_ticket");
        // And against the raw keyed-BLAKE3 primitive, to catch a transcript-helper regression.
        assert_eq!(pi_bytes(&pis, PI_HASH_JACKPOT), blake3_digest(&msg, Some(jk)));

        // The honest trace satisfies the full reused AIR.
        assert!(!constraints_violated(&S::new(program), &rows, &pis));
    }

    #[test]
    fn jackpot_hash_rejects_tampered_lottery_word() {
        let seed_a = [0x11u8; 32];
        let jk = crate::v4::api::transcript::jackpot_key(&seed_a);
        let msg: [u8; 64] = core::array::from_fn(|i| i as u8);
        let mut lottery_words: [u32; 16] =
            core::array::from_fn(|i| u32::from_le_bytes(msg[4 * i..4 * i + 4].try_into().unwrap()));
        // Flip one lottery word: the proven HASH_JACKPOT must diverge from the honest digest.
        lottery_words[7] ^= 0xDEAD_BEEF;

        let program = jackpot_program();
        let data = inputs(
            &[],
            &[],
            lottery_words,
            [0; 8],
            [0; 8],
            bytes32_to_words(&jk),
            HashId::Blake3Chunk1024,
            HashId::Blake3Chunk1024,
        );
        let (_, pis) = program.generate_trace::<F>(&data);
        assert_ne!(
            pi_bytes(&pis, PI_HASH_JACKPOT),
            compute_jackpot_ticket(&seed_a, &msg).jackpot,
            "a tampered lottery word must change HASH_JACKPOT"
        );
    }

    // ------------------------------------------------------------ operand trees

    /// For every allowed leaf size and a spread of operand shapes (single leaf, odd-promote,
    /// multi-leaf cascade), HASH_A / HASH_B equal `commit_operand(..).root()` bit-for-bit.
    #[test]
    fn operand_roots_are_bit_exact_vs_commit_operand() {
        let key_a_bytes = [3u8; 32];
        let key_b_bytes = [200u8; 32];
        // (num_rows, k): row bytes = 2*k; shapes chosen to exercise 1 leaf, odd promote, >2 leaves
        // across the different chunk sizes.
        let shapes = [(1usize, 16usize), (2, 64), (3, 96), (8, 256), (5, 300)];
        for hash_id in HashId::ALL {
            let chunk = hash_id.chunk_len();
            for &(nr, k) in &shapes {
                let a = operand(nr, k, 0x9E37);
                let b = operand(nr, k, 0x85EB);
                let a_bytes = crate::v5::api::commitment::rows_to_bytes(&a);
                let b_bytes = crate::v5::api::commitment::rows_to_bytes(&b);

                let program = fp16_blake3_program(a_bytes.len(), chunk, b_bytes.len(), chunk);
                let data = inputs(
                    &a_bytes,
                    &b_bytes,
                    [0; 16],
                    bytes32_to_words(&key_a_bytes),
                    bytes32_to_words(&key_b_bytes),
                    [0; 8],
                    hash_id,
                    hash_id,
                );
                let (rows, pis) = program.generate_trace::<F>(&data);

                let root_a = commit_operand(&a, nr, k, hash_id, key_a_bytes).unwrap().root();
                let root_b = commit_operand(&b, nr, k, hash_id, key_b_bytes).unwrap().root();
                assert_eq!(
                    pi_bytes(&pis, PI_HASH_A),
                    root_a,
                    "HASH_A mismatch: chunk={chunk} shape={nr}x{k}"
                );
                assert_eq!(
                    pi_bytes(&pis, PI_HASH_B),
                    root_b,
                    "HASH_B mismatch: chunk={chunk} shape={nr}x{k}"
                );

                // The honest trace satisfies the full reused AIR (soundness of the schedule).
                assert!(
                    !constraints_violated(&S::new(program), &rows, &pis),
                    "AIR violated: chunk={chunk} shape={nr}x{k}"
                );
            }
        }
    }

    #[test]
    fn operand_root_rejects_tampered_byte() {
        let key_a_bytes = [7u8; 32];
        let (nr, k) = (4usize, 128usize);
        let hash_id = HashId::Blake3Chunk256;
        let chunk = hash_id.chunk_len();
        let a = operand(nr, k, 0x1234);
        let mut a_bytes = crate::v5::api::commitment::rows_to_bytes(&a);

        let honest_root = commit_operand(&a, nr, k, hash_id, key_a_bytes).unwrap().root();
        let program = operand_tree_program(a_bytes.len(), chunk, PublicBinding::HashA);

        // Flip one committed byte: the proven HASH_A must diverge from the honest root.
        a_bytes[13] ^= 0xFF;
        let data = inputs(
            &a_bytes,
            &[],
            [0; 16],
            bytes32_to_words(&key_a_bytes),
            [0; 8],
            [0; 8],
            hash_id,
            HashId::Blake3Chunk1024,
        );
        let (_, pis) = program.generate_trace::<F>(&data);
        assert_ne!(pi_bytes(&pis, PI_HASH_A), honest_root, "a tampered operand byte must change HASH_A");
    }

    #[test]
    fn full_program_binds_all_three_hashes() {
        let key_a_bytes = [9u8; 32];
        let key_b_bytes = [19u8; 32];
        let seed_a = [0x77u8; 32];
        let jk = crate::v4::api::transcript::jackpot_key(&seed_a);
        let (nr, k) = (3usize, 80usize);
        let hash_id = HashId::Blake3Chunk512;
        let chunk = hash_id.chunk_len();
        let a = operand(nr, k, 0xABCD);
        let b = operand(nr + 1, k, 0xBEEF);
        let a_bytes = crate::v5::api::commitment::rows_to_bytes(&a);
        let b_bytes = crate::v5::api::commitment::rows_to_bytes(&b);
        let msg: [u8; 64] = core::array::from_fn(|i| (i as u8) ^ 0x3C);
        let lottery_words: [u32; 16] = core::array::from_fn(|i| u32::from_le_bytes(msg[4 * i..4 * i + 4].try_into().unwrap()));

        let program = fp16_blake3_program(a_bytes.len(), chunk, b_bytes.len(), chunk);
        let data = inputs(
            &a_bytes,
            &b_bytes,
            lottery_words,
            bytes32_to_words(&key_a_bytes),
            bytes32_to_words(&key_b_bytes),
            bytes32_to_words(&jk),
            hash_id,
            hash_id,
        );
        let (rows, pis) = program.generate_trace::<F>(&data);

        assert_eq!(pi_bytes(&pis, PI_HASH_A), commit_operand(&a, nr, k, hash_id, key_a_bytes).unwrap().root());
        assert_eq!(pi_bytes(&pis, PI_HASH_B), commit_operand(&b, nr + 1, k, hash_id, key_b_bytes).unwrap().root());
        assert_eq!(pi_bytes(&pis, PI_HASH_JACKPOT), compute_jackpot_ticket(&seed_a, &msg).jackpot);
        assert!(!constraints_violated(&S::new(program), &rows, &pis));
    }

    // ------------------------------------------------------------ CTL hooks

    #[test]
    fn ctl_hooks_are_well_formed() {
        // Any placeholder table index is fine; the real index is supplied at batch integration.
        assert_eq!(ctl_lottery_words_looking_blake3::<F>(3).len(), 16);
        assert_eq!(ctl_operand_bytes_looking_blake3::<F>(3).len(), 4);
    }
}
