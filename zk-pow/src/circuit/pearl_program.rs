use log::debug;
use pearl_blake3::{B3F_CHUNK_END, B3F_CHUNK_START, B3F_KEYED_HASH, B3F_ROOT};
use plonky2_maybe_rayon::*;

use crate::api::proof::Hash256;
use crate::api::proof_utils::CompiledPublicParams;
use crate::circuit::chip::blake3::blake3_compress::Blake3Tweak;
use crate::circuit::chip::blake3::logic::{BlakeRoundLogic, MessageDataType};
use crate::circuit::chip::blake3::program::{BLOCK_LEN, DWORD_SIZE, MatDwordId, MessageType, ROUNDS_PER_BLAKE_INSTRUCTION};
use crate::circuit::chip::{BitRegDst, BitRegSrc, JackpotLogic, MatmulLogic};
use anyhow::{Result, ensure};

pub use blake3::{BLOCK_LEN as BLAKE3_MSG_LEN, CHUNK_LEN as BLAKE3_CHUNK_LEN, OUT_LEN as BLAKE3_DIGEST_SIZE};

// We tile the matrix A using
//   TILE_H × TILE_D  tiles,
// and tile B matrix using
//   TILE_D × TILE_H  tiles,
// and our minimal hardware multiplier is
//   (TILE_H × TILE_D) × (TILE_D × TILE_H)
pub const TILE_D: usize = 16; // Tile depth
pub const TILE_H: usize = 2; // tile height (viewing B as B^t)
pub const JACKPOT_SIZE: usize = 16; // number of uint32 in jackpot
pub const MIN_STARK_LEN: usize = 1 << 13;
pub const LROT_PER_TILE: u32 = 13;

#[derive(Clone, Debug, Copy, Default)]
pub struct RowLogic {
    pub blake: BlakeRoundLogic,
    pub matmul: MatmulLogic,
    pub jackpot: JackpotLogic,
}

impl CompiledPublicParams {
    /// MoE routing Blake **witness** layout: `num_routing_strips` × 64 bytes (= 8 Pearl dwords each strip).
    /// Strips are **deduplicated** by 64-byte chunk index; multiple A tile rows can share one strip.
    pub fn num_routing_dwords(&self) -> usize {
        if self.moe.is_some() {
            self.blake_proof.num_routing_strips * (BLOCK_LEN / DWORD_SIZE)
        } else {
            debug_assert!(self.blake_proof.num_routing_strips == 0);
            0
        }
    }

    pub fn expected_num_rows(&self) -> usize {
        // Blake rows: one row per round per instruction (commitment hash only, jackpot is appended separately).
        let blake_proof_len = self.blake_proof.instructions.len() * ROUNDS_PER_BLAKE_INSTRUCTION;

        // The trailing jackpot blake compression adds one more instruction's worth of rows.
        (blake_proof_len + ROUNDS_PER_BLAKE_INSTRUCTION).max(self.matmul_and_jackpot_num_rows())
    }

    /// Number of rows emitted by `structure_matmul_in_stark` for the matmul/jackpot streams.
    fn matmul_and_jackpot_num_rows(&self) -> usize {
        let tiles_per_r = self.r / TILE_D;
        let xor_instructions = TILE_H * TILE_H + 1;
        let base = tiles_per_r.max(xor_instructions);
        let total_subtiles = (self.h / TILE_H) * (self.w / TILE_H);
        let num_chunks = self.k / self.r;

        let extra_shifts_per_subtile: usize = (0..JACKPOT_SIZE.min(num_chunks))
            .map(|tid| {
                let occurrences = num_chunks / JACKPOT_SIZE + usize::from(tid < num_chunks % JACKPOT_SIZE);
                let num_shift3 = compute_back_shift_ops(occurrences).0;
                (xor_instructions + num_shift3).saturating_sub(base)
            })
            .sum();

        total_subtiles * num_chunks * base + (total_subtiles - 1) * extra_shifts_per_subtile
    }

    /// log2 num rows
    pub fn degree_bits(&self) -> usize {
        // Round up to the next power of two and return its log2
        self.expected_num_rows().next_power_of_two().max(MIN_STARK_LEN).ilog2() as usize
    }

    pub fn num_dwords_per_strip(&self) -> usize {
        self.blake_proof.strip_length / DWORD_SIZE
    }

    /// Total number of dwords in matrix A (is_b=false) or B (is_b=true).
    pub fn num_dwords(&self, is_b: bool) -> usize {
        let num_strips = if is_b { self.w } else { self.h };
        num_strips * self.num_dwords_per_strip()
    }

    /// Convert a dword position to a global index.
    pub fn dword_to_index(&self, dword: MatDwordId) -> usize {
        let n = self.num_dwords_per_strip();
        let base = if dword.is_b_strip { self.num_dwords(false) } else { 0 };
        base + (dword.strip_idx / TILE_H) * n * TILE_H + (dword.idx_in_strip / DWORD_SIZE) * TILE_H + dword.strip_idx % TILE_H
    }

    /// The inverse map: convert a global index back to MatDwordId.
    pub fn index_to_dword(&self, index: usize) -> MatDwordId {
        let n = self.num_dwords_per_strip();
        let num_a_dwords = self.num_dwords(false);
        let is_b = index >= num_a_dwords;
        let i = index - if is_b { num_a_dwords } else { 0 };
        let (chunk, rem) = (i / (n * TILE_H), i % (n * TILE_H));
        MatDwordId {
            is_b_strip: is_b,
            strip_idx: chunk * TILE_H + rem % TILE_H,
            idx_in_strip: (rem / TILE_H) * DWORD_SIZE,
        }
    }

    pub fn structure_proof(&self) -> Result<Vec<RowLogic>> {
        let commitment_hash_proof = self.structure_commitment_hash_in_stark()?;
        let blake_proof_len = commitment_hash_proof.len();

        let (matmul_proof, jackpot_proof) = self.structure_matmul_in_stark()?;
        let matmul_and_jackpot_proof_len = matmul_proof.len(); // same as jackpot_proof.len()

        let num_rows = (blake_proof_len + ROUNDS_PER_BLAKE_INSTRUCTION).max(matmul_and_jackpot_proof_len);

        debug!(
            "blake3 circuit len: {:?} || matmul/jackpot len: {:?} || h: {:?} w: {:?}",
            blake_proof_len, matmul_and_jackpot_proof_len, self.h, self.w
        );

        let jackpot_blake_start = num_rows - ROUNDS_PER_BLAKE_INSTRUCTION;
        let jackpot_blake = structure_jackpot_blake();

        let padded_n = num_rows.next_power_of_two().max(MIN_STARK_LEN);

        ensure!(
            num_rows == self.expected_num_rows(),
            "structure_proof length {} does not match expected num rows {}",
            num_rows,
            self.expected_num_rows()
        );

        let res: Vec<RowLogic> = (0..padded_n)
            .into_par_iter()
            .map(|i| {
                if i >= num_rows {
                    return RowLogic::default();
                }
                RowLogic {
                    blake: if i < blake_proof_len {
                        commitment_hash_proof[i]
                    } else if i >= jackpot_blake_start {
                        jackpot_blake[i - jackpot_blake_start]
                    } else {
                        BlakeRoundLogic::default()
                    },
                    matmul: matmul_proof.get(i).copied().unwrap_or_default(),
                    jackpot: jackpot_proof.get(i).copied().unwrap_or_default(),
                }
            })
            .collect();

        Ok(res)
    }

    pub fn a_noise_seed(&self) -> Hash256 {
        self.commitment_hash.1
    }

    pub fn b_noise_seed(&self) -> Hash256 {
        self.commitment_hash.0
    }
}

/// Compute the number of shift3 operations and store type needed to achieve a given back-shift.
/// Returns (num_shift3, store_dst) such that: 3 * num_shift3 + store_shift ≡ back_shift (mod 32)
fn compute_back_shift_ops(back_shifts: usize) -> (usize, BitRegDst) {
    let target = back_shifts % 32;
    let store_dst = match target % 3 {
        0 => BitRegDst::Store0,
        1 => BitRegDst::Store1,
        2 => BitRegDst::Store2,
        _ => unreachable!(),
    };
    (target / 3, store_dst)
}

impl CompiledPublicParams {
    fn structure_commitment_hash_in_stark(&self) -> Result<Vec<BlakeRoundLogic>> {
        let instructions = &self.blake_proof.instructions;
        let expects_routing = self.moe.is_some();
        let n = instructions.len();

        let (mut has_a, mut has_b, mut has_routing) = (false, false, false);
        for inst in instructions {
            has_a |= inst.is_hash_a;
            has_b |= inst.is_hash_b;
            has_routing |= inst.is_hash_routing;

            if matches!(inst.msg, MessageType::Parent { .. }) {
                ensure!(inst.is_cv_key, "Parent instruction should use cv_key");
            }
        }
        ensure!(has_a && has_b, "Blake program must output hash of A and B");
        ensure!(
            (!expects_routing || has_routing),
            "Blake program must output hash of routing in MoE proof"
        );

        // Each `emit_instruction_rounds` call returns `ROUNDS_PER_BLAKE_INSTRUCTION` rows,
        // so the row index of instruction `k` in the flattened output is the last row of
        // the k-th `ROUNDS_PER_BLAKE_INSTRUCTION`-sized chunk.
        let inst_to_row: Vec<usize> = (0..n)
            .map(|i| i * ROUNDS_PER_BLAKE_INSTRUCTION + (ROUNDS_PER_BLAKE_INSTRUCTION - 1))
            .collect();

        let mut res: Vec<BlakeRoundLogic> = vec![BlakeRoundLogic::default(); n * ROUNDS_PER_BLAKE_INSTRUCTION];

        res.par_chunks_exact_mut(ROUNDS_PER_BLAKE_INSTRUCTION)
            .zip(instructions.par_iter().enumerate())
            .for_each(|(chunk, (i, inst))| {
                debug_assert!(i > 0 || inst.is_cv_key, "first instruction must be cv_key");
                let read_cv_from = (!inst.is_cv_key).then(|| inst_to_row[i - 1]);
                chunk.copy_from_slice(&inst.emit_instruction_rounds(&inst_to_row, read_cv_from));
            });

        Ok(res)
    }
}

/// Generate one `BlakeRoundLogic` per round for hashing the 64-byte jackpot.
fn structure_jackpot_blake() -> [BlakeRoundLogic; ROUNDS_PER_BLAKE_INSTRUCTION] {
    let tweak = Blake3Tweak {
        counter_low: 0,
        counter_high: 0,
        block_len: BLAKE3_MSG_LEN as u32,
        flags: (B3F_KEYED_HASH | B3F_CHUNK_START | B3F_CHUNK_END | B3F_ROOT).into(),
    };
    std::array::from_fn(|i| BlakeRoundLogic {
        data_source: if i == 7 {
            MessageDataType::Jackpot
        } else {
            MessageDataType::None
        },
        blake3_tweak: if i == 0 { Some(tweak) } else { None },
        round_idx: i + 1,
        idx_of_row_whence_to_read_cv: None,
        is_hash_a: false,
        is_hash_b: false,
        is_hash_routing: false,
        is_hash_jackpot: i == 7,
        cv_is_commitment: i == 0,
    })
}

impl CompiledPublicParams {
    fn structure_matmul_in_stark(&self) -> Result<(Vec<MatmulLogic>, Vec<JackpotLogic>)> {
        let mut matmul_res = vec![];
        let mut jackpot_res = vec![];
        let h = self.h;
        let w = self.w;
        let k = self.k;
        let r = self.r;
        let tiles_per_r = r / TILE_D;
        // Base rows per ll: 1 LOAD + 4 XORs (last XOR also stores and dumps cumsum)
        let xor_instructions = TILE_H * TILE_H + 1;
        // If k=dot_length is not multiple of TILE_D, then MAT_FREQ will not be uniform.
        ensure!(k.is_multiple_of(TILE_D), "k must be a multiple of TILE_D");
        ensure!(h.is_multiple_of(TILE_H), "h must be a multiple of TILE_H");
        ensure!(w.is_multiple_of(TILE_H), "w must be a multiple of TILE_H");
        ensure!(h * w <= 256, "h * w must be at most 256");

        let num_subtiles_h = h / TILE_H;
        let num_subtiles_w = w / TILE_H;
        let total_subtiles = num_subtiles_h * num_subtiles_w;

        // Precompute: for each tid, which ll is the last one that writes to it, and how many times
        let mut last_ll_for_tid = [0usize; JACKPOT_SIZE];
        let mut num_occurrences = [0usize; JACKPOT_SIZE];
        for ll in (r..=k).step_by(r) {
            let tid = (ll / r - 1) % JACKPOT_SIZE;
            last_ll_for_tid[tid] = ll;
            num_occurrences[tid] += 1;
        }

        // Iterate over subtiles in row-major order, processing all r-chunks along k.
        //
        // Each subsequent subtile's loads rotate this subtile's accumulated contributions further.
        // To compensate, at the last write to each tid within a subtile (for non-last subtiles),
        // apply a back shift = (remaining_subtiles) * (num_occurrences of tid) * LROT_PER_TILE.
        for i in 0..num_subtiles_h {
            for j in 0..num_subtiles_w {
                let subtile_idx = i * num_subtiles_w + j;
                let is_last_subtile = subtile_idx == total_subtiles - 1;

                // Compute dot product along common dimension in r-sized chunks.
                for ll in (r..=k).step_by(r) {
                    let tid = (ll / r - 1) % JACKPOT_SIZE;
                    let is_last_occurrence_of_tid = ll == last_ll_for_tid[tid];

                    // Generate matmul logic for each TILE_D chunk in this r-chunk
                    for l in 0..tiles_per_r {
                        let idx_in_strip = ll - r + l * TILE_D;
                        matmul_res.push(MatmulLogic {
                            is_reset_cumsum: idx_in_strip == 0,
                            a_dword: Some(MatDwordId {
                                is_b_strip: false,
                                strip_idx: i * TILE_H,
                                idx_in_strip,
                            }),
                            b_dword: Some(MatDwordId {
                                is_b_strip: true,
                                strip_idx: j * TILE_H,
                                idx_in_strip,
                            }),
                        });
                    }

                    // Row (xor_instructions - 1): XOR + is_dump_cumsum_buffer + store
                    // Determine the store type based on position
                    let (num_shift3, final_store) = if is_last_subtile {
                        // Last subtile: Store0, no back-shift needed
                        (0, BitRegDst::Store0)
                    } else if is_last_occurrence_of_tid {
                        // Last occurrence of this tid on non-last subtile:
                        // Apply back-shift to compensate for subsequent subtiles' loads
                        compute_back_shift_ops(num_occurrences[tid])
                    } else {
                        // Non-last occurrence: Store0 to allow rotation accumulation
                        (0, BitRegDst::Store0)
                    };

                    let bitgreg_instructions = xor_instructions + num_shift3;

                    // Pad matmul/jackpot entries to align the two instruction streams
                    matmul_res.extend(std::iter::repeat_n(
                        MatmulLogic::default(),
                        bitgreg_instructions.saturating_sub(tiles_per_r),
                    ));
                    jackpot_res.extend(std::iter::repeat_n(
                        JackpotLogic::default(),
                        tiles_per_r.saturating_sub(bitgreg_instructions),
                    ));

                    // === Jackpot state machine for this subtile's r-chunk ===
                    //
                    // LOAD: bit_reg = jackpot[tid].rotate_left(LROT) (implicit rotation on load)
                    // XOR:  bit_reg ^= tile_buffer[0]
                    // Store0: jackpot[tid] = bit_reg (no shift)
                    // Store1: jackpot[tid] = bit_reg >>> LROT (back-shift by LROT)
                    // Store2: jackpot[tid] = bit_reg >>> 2*LROT (back-shift by 2*LROT)
                    // Shift3: bit_reg >>>= 3*LROT (39 ≡ 7 mod 32)
                    //
                    // Row structure:
                    //   Row 0: LOAD - loads jackpot[tid].rotate_left(LROT) into bit_reg
                    //   Rows 1-3: XOR + Store0 - accumulate XOR, store intermediate
                    //   Row 4: XOR + is_dump_cumsum_buffer + final store
                    //
                    // For non-last occurrence: Store0 (allow rotation to accumulate via next load)
                    // For last occurrence on last subtile: Store0 (final result)
                    // For last occurrence on non-last subtile: shift3 ops + Store to back-shift

                    // Row 0: LOAD (src=Jackpot, dst=Store1) - NOP that loads rotated jackpot into bit_reg
                    jackpot_res.push(JackpotLogic {
                        src: BitRegSrc::Jackpot,
                        dst: BitRegDst::Store1,
                        jackpot_idx: tid,
                        is_dump_cumsum_buffer: false,
                    });

                    // Rows 1 to (xor_instructions - 2): XOR + Store0 (intermediate stores)
                    for _ in 1..(xor_instructions - 1) {
                        jackpot_res.push(JackpotLogic {
                            src: BitRegSrc::Xor,
                            dst: BitRegDst::Store0,
                            jackpot_idx: tid,
                            is_dump_cumsum_buffer: false,
                        });
                    }

                    for ss in 0..(num_shift3 + 1) {
                        jackpot_res.push(JackpotLogic {
                            src: if ss == 0 { BitRegSrc::Xor } else { BitRegSrc::Shift3 },
                            dst: if ss == num_shift3 { final_store } else { BitRegDst::Store0 },
                            jackpot_idx: tid,
                            is_dump_cumsum_buffer: ss == num_shift3,
                        });
                    }
                }
            }
        }

        Ok((matmul_res, jackpot_res))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit::chip::blake3::program::BlakeProgram;

    /// Minimal params for exercising `structure_matmul_in_stark` row counting.
    /// Only `h`, `w`, `k`, `r` affect the matmul/jackpot streams.
    fn matmul_params(h: usize, w: usize, k: usize, r: usize) -> CompiledPublicParams {
        CompiledPublicParams {
            job_key: [0u8; 32],
            k,
            h,
            w,
            r,
            blake_proof: BlakeProgram {
                num_a_rows: h,
                num_b_cols: w,
                strip_length: k,
                num_routing_strips: 0,
                num_auxiliary_msgs: 0,
                num_auxiliary_cvs: 0,
                instructions: vec![],
            },
            a_rows_indices: vec![],
            b_cols_indices: vec![],
            commitment_hash: ([0u8; 32], [0u8; 32]),
            moe: None,
        }
    }

    #[test]
    fn expected_num_rows_accounts_for_shift3_back_rotation() {
        let (h, w, r) = (2, 4, TILE_D);
        let k = 33 * r;
        let params = matmul_params(h, w, k, r);

        let (matmul, jackpot) = params.structure_matmul_in_stark().unwrap();
        assert_eq!(matmul.len(), jackpot.len(), "the two streams must stay aligned");

        // The row-counting helper must match the actual emitted length exactly.
        assert_eq!(
            params.matmul_and_jackpot_num_rows(),
            matmul.len(),
            "matmul_and_jackpot_num_rows must equal the emitted row count"
        );

        // This config exercises the back-rotation path: the old flat formula
        // (which ignored Shift3 rows) under-counts by exactly one row here.
        let xor_instructions = TILE_H * TILE_H + 1;
        let tiles_per_r = r / TILE_D;
        let total_subtiles = (h / TILE_H) * (w / TILE_H);
        let num_r_per_tile = k / r;
        let old_flat = total_subtiles * num_r_per_tile * tiles_per_r.max(xor_instructions);
        assert_eq!(matmul.len(), old_flat + 1, "expected exactly one extra Shift3 row");

        // And a back-rotation Shift3 row was indeed emitted.
        assert!(
            jackpot.iter().any(|j| matches!(j.src, BitRegSrc::Shift3)),
            "a back-rotation Shift3 row should be present"
        );
    }

    /// Exhaustively check `matmul_and_jackpot_num_rows` against the actual
    /// emitted stream lengths over the full parameter grid permitted by
    /// `public_params_sanity_check`.
    #[test]
    fn matmul_and_jackpot_num_rows_matches_structure_exhaustively() {
        let hw_cases = [(2usize, 16usize), (16, 16)];

        let mut checked = 0usize;
        for r in [32usize, 64, 128, 256, 512, 1024] {
            let k_min = (16 * r).max(1024);
            let k_max = (4 * r * r).min(1 << 16);
            for k in (k_min..=k_max).step_by(64) {
                for &(h, w) in &hw_cases {
                    let params = matmul_params(h, w, k, r);
                    let (matmul, jackpot) = params.structure_matmul_in_stark().unwrap();
                    assert_eq!(matmul.len(), jackpot.len(), "streams misaligned at r={r} k={k} h={h} w={w}");
                    assert_eq!(
                        params.matmul_and_jackpot_num_rows(),
                        matmul.len(),
                        "row counter mismatch at r={r} k={k} h={h} w={w}"
                    );
                    checked += 1;
                }
            }
        }
        assert!(checked > 7000, "grid unexpectedly small: {checked}");
    }
}
