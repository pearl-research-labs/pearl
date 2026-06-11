use crate::v1::circuit::chip::blake3::logic::MessageDataType;
use crate::v1::circuit::chip::blake3::program::{Blake3Tweak, BlakeMsg, DWORD_SIZE, MatDwordId};
use crate::v1::circuit::chip::{BitRegDst, BitRegSrc};

use crate::v1::{
    api::proof_utils::CompiledPublicParams,
    circuit::{
        pearl_layout::{BITS_PER_LIMB, NOISE_PACKING_BASE, pearl_columns},
        pearl_noise::{MMSlice, compute_noise},
        pearl_program::RowLogic,
        utils::trace_utils::{deg2_muxer_bits, i64_pack_base, i64_to_u64},
    },
};
use anyhow::Result;
use plonky2::field::goldilocks_field::GoldilocksField;
use plonky2_maybe_rayon::*;

type F = GoldilocksField;

/// Pack blake3 tweak parameters into a u64.
///
/// Little-endian byte layout:
///   [counter_low (4) | counter_high (2) | flags (1) | block_len (1)]
fn pack_blake3_tweak(tweak: &Blake3Tweak) -> u64 {
    debug_assert!(tweak.flags < 32, "Blake3 flags must be < 32");
    debug_assert_eq!(tweak.block_len, 64, "Blake3 block_len must be 64");
    (tweak.counter_low as u64)
        | ((tweak.counter_high as u64 & 0xFFFF) << 32)
        | (((tweak.flags as u8) as u64) << 48)
        | (((tweak.block_len as u8) as u64) << 56)
}

pub fn read_dword_from_matrix(strips: &MMSlice, dword_id: MatDwordId) -> [i8; DWORD_SIZE] {
    let mat = if dword_id.is_b_strip { &strips.b } else { &strips.a };
    let strip = &mat[dword_id.strip_idx];
    strip[dword_id.idx_in_strip..dword_id.idx_in_strip + DWORD_SIZE]
        .try_into()
        .unwrap()
}

pub fn read_blake_msg_from_matrix(strips: &MMSlice, blake_msg: BlakeMsg) -> [i8; 64] {
    let mut result = [0i8; 64];
    for (dword_idx, &dword_id) in blake_msg.dwords.iter().enumerate() {
        let dword = read_dword_from_matrix(strips, dword_id);
        let dst_start = dword_idx * DWORD_SIZE;
        result[dst_start..dst_start + DWORD_SIZE].copy_from_slice(&dword);
    }
    result
}

// Returns:
//   1. List of preprocessed columns. Each preprocessed column has num rows identical to STARK num rows.
//   2. Noise matrices: (noise_A, noise_B). A (B) is array of TILE_H noise rows (columns).
// Used both by prover and verifier. All other logic in this file is prover-only.
#[allow(clippy::type_complexity)]
pub(crate) fn generate_preprocessed(
    public_params: &CompiledPublicParams,
    circuit: Option<&[RowLogic]>,
) -> Result<(Vec<(usize, Vec<u64>)>, MMSlice)> {
    let mut owned_circuit: Option<Vec<RowLogic>> = None;
    let circuit: &[RowLogic] = match circuit {
        Some(c) => c,
        None => {
            owned_circuit = Some(public_params.structure_proof()?);
            owned_circuit.as_ref().unwrap().as_slice()
        }
    };

    let num_rows = circuit.len();

    // Fake MAT_ID base for non-matrix rows (beyond all real dword indices)
    let fake_mat_id_base = (public_params.num_dwords(false) + public_params.num_dwords(true)) as u64;

    // Noise matrices are needed by the NOISE_PACKED_PREP column; compute once up front.
    let noise_prep_slice = compute_noise(public_params);

    // Number of bit slots in the CONTROL_PREP packed layout before MAT_ID:
    // 21 static flags + JACKPOT_IDX_LEN muxer bits.
    const STATIC_BITS_LEN: usize = 21;
    const CONTROL_BITS_LEN: usize = STATIC_BITS_LEN + pearl_columns::JACKPOT_IDX_LEN;

    // Pre-allocate the four output columns and fill them in a single parallel
    // pass over `circuit`. The previous implementation re-scanned the ~19 MB
    // `Vec<RowLogic>` four times and pattern-matched `blake.data_source` up to
    // three times per row; this version visits each row once.
    let mut control_prep = vec![0u64; num_rows];
    let mut noise_packed_prep = vec![0u64; num_rows];
    let mut cv_or_tweak_prep = vec![0u64; num_rows];
    let mut ab_id_prep = vec![0u64; num_rows];

    control_prep
        .par_iter_mut()
        .zip(noise_packed_prep.par_iter_mut())
        .zip(cv_or_tweak_prep.par_iter_mut())
        .zip(ab_id_prep.par_iter_mut())
        .enumerate()
        .for_each(|(row_idx, (((control_out, noise_out), cv_or_tweak_out), ab_id_out))| {
            let RowLogic {
                blake: blake_data,
                matmul: matmul_data,
                jackpot: jackpot_data,
            } = circuit[row_idx];

            // Single match on `blake.data_source` covers the IS_MSG_* flags, the MAT_ID
            // lookup, and the noise-dword lookup for NOISE_PACKED_PREP.
            let (is_msg_mat, is_msg_jackpot, is_msg_aux_data, is_msg_cv, mat_id, noise_dword_id) = match blake_data.data_source {
                MessageDataType::Matrix { dword_id } => (
                    true,
                    false,
                    false,
                    false,
                    public_params.dword_to_index(dword_id) as u64,
                    Some(dword_id),
                ),
                MessageDataType::Jackpot => (false, true, false, false, fake_mat_id_base + row_idx as u64, None),
                MessageDataType::AuxiliaryData { .. } => (false, false, true, false, fake_mat_id_base + row_idx as u64, None),
                MessageDataType::PreviousCv { .. } => (false, false, false, true, fake_mat_id_base + row_idx as u64, None),
                MessageDataType::None => (false, false, false, false, fake_mat_id_base + row_idx as u64, None),
            };

            let is_bitreg_load = matches!(jackpot_data.src, BitRegSrc::Jackpot);

            // ---- CONTROL_PREP: pack 21 static flags + JACKPOT_IDX muxer bits + MAT_ID.
            let static_bits: [bool; STATIC_BITS_LEN] = [
                matmul_data.is_reset_cumsum,
                matmul_data.is_update_cumsum(),
                blake_data.is_use_job_key(),
                blake_data.is_use_commitment_hash(),
                blake_data.is_hash_a,
                blake_data.is_hash_b,
                blake_data.is_hash_jackpot,
                blake_data.idx_of_row_whence_to_read_cv.is_some(), // IS_CV_IN
                blake_data.round_idx == 1,                         // IS_NEW_BLAKE
                blake_data.round_idx == 8,                         // IS_LAST_ROUND
                is_msg_mat,
                is_msg_jackpot,
                is_msg_aux_data,
                is_msg_cv,
                is_bitreg_load,                                // IS_LOAD
                matches!(jackpot_data.src, BitRegSrc::Xor),    // IS_XOR
                matches!(jackpot_data.src, BitRegSrc::Shift3), // IS_SHIFT3
                matches!(jackpot_data.dst, BitRegDst::Store0),
                matches!(jackpot_data.dst, BitRegDst::Store1),
                matches!(jackpot_data.dst, BitRegDst::Store2),
                jackpot_data.is_dump_cumsum_buffer,
            ];
            // JACKPOT_IDX encoding: 0..15 for load, 16..31 for store.
            let muxer_bits = deg2_muxer_bits::<{ pearl_columns::JACKPOT_IDX_LEN }>(Some(if is_bitreg_load {
                jackpot_data.jackpot_idx
            } else {
                jackpot_data.jackpot_idx + 16
            }));
            let mut bits_packed: u64 = 0;
            for (i, &b) in static_bits.iter().enumerate() {
                bits_packed |= (b as u64) << i;
            }
            for (i, &b) in muxer_bits.iter().enumerate() {
                bits_packed |= (b as u64) << (STATIC_BITS_LEN + i);
            }
            *control_out = bits_packed + (mat_id << CONTROL_BITS_LEN);

            // ---- NOISE_PACKED_PREP: only matrix rows have non-zero noise.
            *noise_out = match noise_dword_id {
                Some(dword_id) => {
                    let noise_dword = read_dword_from_matrix(&noise_prep_slice, dword_id);
                    i64_to_u64::<F>(i64_pack_base(&noise_dword, NOISE_PACKING_BASE))
                }
                None => 0,
            };

            // ---- CV_OR_TWEAK_PREP: CV_IDX when reading CV, else blake3_tweak of prev row.
            let cv_idx = blake_data.idx_of_row_whence_to_read_cv;
            let prev_row_idx = if row_idx == 0 { num_rows - 1 } else { row_idx - 1 };
            let prev_tweak = circuit[prev_row_idx].blake.blake3_tweak;
            debug_assert!(
                cv_idx.is_none() || prev_tweak.is_none(),
                "CV_IDX and blake3_tweak cases must be disjoint"
            );
            *cv_or_tweak_out = match cv_idx {
                Some(idx) => idx as u64,
                None => pack_blake3_tweak(&prev_tweak.unwrap_or_default()),
            };

            // ---- AB_ID_PREP: A_ID || (B_ID << 2*BITS_PER_LIMB).
            let a_id = matmul_data
                .a_dword
                .map(|d| public_params.dword_to_index(d) as u64)
                .unwrap_or(0);
            let b_id = matmul_data
                .b_dword
                .map(|d| public_params.dword_to_index(d) as u64)
                .unwrap_or(0);
            *ab_id_out = a_id + (b_id << (2 * BITS_PER_LIMB));
        });

    // The returned (index, column) pairs are not ordered; consumers that care
    // about ordering (e.g. `PearlStark::preprocessed_columns`) must sort / look
    // up by the column index.
    let res = vec![
        (pearl_columns::CONTROL_PREP, control_prep),
        (pearl_columns::NOISE_PACKED_PREP, noise_packed_prep),
        (pearl_columns::CV_OR_TWEAK_PREP, cv_or_tweak_prep),
        (pearl_columns::AB_ID_PREP, ab_id_prep),
    ];

    if let Some(owned) = owned_circuit.take() {
        rayon::spawn(move || drop(owned));
    }

    Ok((res, noise_prep_slice))
}
