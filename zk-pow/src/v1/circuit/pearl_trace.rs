use hashbrown::HashMap;

use plonky2::field::extension::Extendable;
use plonky2::field::polynomial::PolynomialValues;
use plonky2::hash::hash_types::RichField;
use plonky2_maybe_rayon::*;
use starky::lookup::Lookup;

use crate::v1::api::proof::PrivateProofParams;
use crate::v1::circuit::chip::AuxData;
use crate::v1::circuit::chip::blake3::program::{DWORD_SIZE, MatDwordId};
#[cfg(debug_assertions)]
use crate::v1::circuit::chip::compute_jackpot;
use crate::v1::circuit::pearl_layout::{BYTES_PER_GOLDILOCKS, pearl_columns, pearl_public};
use crate::v1::circuit::pearl_noise::MMSlice;
use crate::v1::circuit::pearl_preprocess::{generate_preprocessed, read_dword_from_matrix};
use crate::v1::circuit::pearl_program::{TILE_D, TILE_H};
use crate::v1::circuit::pearl_stark::{PearlStarkChips, PearlStarkConfig};
use crate::v1::circuit::utils::trace_utils::{RowBuilder, read_from_trace, u64_pack_le};

// returns TILE_H × TILE_D flattened row major
pub fn read_tile(strips: &MMSlice, dword_id: MatDwordId) -> [i8; TILE_H * TILE_D] {
    let mut res = [0i8; TILE_H * TILE_D];
    let mut dst_idx = 0;
    for strip_id in 0..TILE_H {
        for dword_idx in 0..TILE_D / DWORD_SIZE {
            let dword = read_dword_from_matrix(
                strips,
                MatDwordId {
                    is_b_strip: dword_id.is_b_strip,
                    strip_idx: dword_id.strip_idx + strip_id,
                    idx_in_strip: dword_id.idx_in_strip + dword_idx * DWORD_SIZE,
                },
            );
            res[dst_idx..dst_idx + DWORD_SIZE].copy_from_slice(&dword);
            dst_idx += DWORD_SIZE;
        }
    }
    res
}

pub fn generate_trace<F: RichField + Extendable<D>, const D: usize>(
    chips: &PearlStarkChips,
    config: &PearlStarkConfig,

    lookup_data: &[Lookup<F>],
    public_params: &crate::v1::api::proof::PublicProofParams,
    private_params: PrivateProofParams,
) -> (Vec<[F; pearl_columns::TOTAL]>, [F; pearl_public::TOTAL]) {
    debug_assert!(F::BITS == 64, "Goldilocks field is assumed");
    let compiled_params = &config.compiled_public_params;
    let circuit = &config.structured_circuit;
    let (preprocessed, noise) = generate_preprocessed(compiled_params, Some(circuit)).unwrap();

    // Debug only: compute expected jackpot before private_params is consumed
    #[cfg(debug_assertions)]
    let expected_jackpot = compute_jackpot(compiled_params, &private_params.s_a, &private_params.s_b, &noise);

    let num_rows = circuit.len();

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Fill trace
    //////////////////////////////////////////////////////////////////////////////////////////////////////////
    let secret_strips = MMSlice {
        a: private_params.s_a,
        b: private_params.s_b,
    };
    let aux_data = AuxData {
        external_msgs: private_params.external_msgs,
        external_cvs: private_params.external_cvs,
    };

    // Initialize empty trace (parallel for NUMA locality and memory bandwidth)
    let mut trace: Vec<[F; pearl_columns::TOTAL]> = (0..num_rows)
        .into_par_iter()
        .map(|_| [F::ZERO; pearl_columns::TOTAL])
        .collect();

    let mut matmul_chip = chips.matmul_chip.clone();

    for row_idx in 0..num_rows {
        ///////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Fill preprocessed columns
        ///////////////////////////////////////////////////////////////////////////////////////////////////////////
        for (col_id, col) in &preprocessed {
            trace[row_idx][*col_id] = F::from_canonical_u64(col[row_idx]);
        }

        let mut row_builder = RowBuilder {
            row: &mut trace[row_idx],
            offset: 0,
        };

        ///////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Fill constant lookup tables (extend last value beyond table size)
        // Each table is followed by a FREQ column which we skip
        ///////////////////////////////////////////////////////////////////////////////////////////////////////////
        chips.urange8_chip.fill_row_trace(row_idx, &mut row_builder);
        chips.urange13_chip.fill_row_trace(row_idx, &mut row_builder);
        chips.irange7p1_chip.fill_row_trace(row_idx, &mut row_builder);
        chips.irange8_chip.fill_row_trace(row_idx, &mut row_builder);
        chips.i8u8_chip.fill_row_trace(row_idx, &mut row_builder);

        // Control flags and MAT_ID packed
        chips.control_and_matid_packer.fill_row_trace(row_idx, &mut row_builder);

        // Stark row chip
        chips.stark_row_chip.fill_row_trace(row_idx, &mut row_builder);

        // Input chip
        chips.input_chip.fill_row_trace(
            row_idx,
            &mut row_builder,
            &secret_strips,
            &aux_data,
            &config.blake3_chip_config,
        );

        // Blake3 chip
        chips.blake3_chip.fill_row_trace(row_idx, &mut row_builder);

        // Matmul chip
        matmul_chip.fill_row_trace(row_idx, &mut row_builder, &secret_strips, &noise, &config.matmul_chip_config);

        // Jackpot chip
        chips.jackpot_chip.fill_row_trace(row_idx, &mut row_builder);

        row_builder.assert_end();
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Fill CUMSUM_BUFFER, BIT_REG, JACKPOT_MSG
    //////////////////////////////////////////////////////////////////////////////////////////////////////////
    chips
        .jackpot_chip
        .fill_full_trace(&config.jackpot_chip_config, config, &mut trace);

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Fill Blake3-related columns (CV_IN, BLAKE3_CV, CV_OUT, BLAKE3_MSG, BLAKE3_MSG_BUFFER)
    //////////////////////////////////////////////////////////////////////////////////////////////////////////
    chips
        .blake3_chip
        .fill_full_trace(&config.blake3_chip_config, config, &mut trace);

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Blind 192 bits of the trace so that zeta reveals no additional information about witness
    //////////////////////////////////////////////////////////////////////////////////////////////////////////
    blind_trace(&mut trace, compiled_params.expected_num_rows());

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Fill lookup table frequencies after processing all rows
    //////////////////////////////////////////////////////////////////////////////////////////////////////////
    fill_lookup_table_frequencies(lookup_data, &mut trace, num_rows);

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Fill PUBLIC_INPUTS
    let mut public_inputs = [F::ZERO; pearl_public::TOTAL];
    let mut public_inputs_builder = RowBuilder {
        row: &mut public_inputs,
        offset: pearl_public::JOB_KEY,
    };

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Fill JOB_KEY (PUBLIC)
    for chunk in compiled_params.job_key.chunks_exact(BYTES_PER_GOLDILOCKS) {
        public_inputs_builder.dump_u64(u64_pack_le(chunk, 8));
    }
    debug_assert_eq!(public_inputs_builder.offset, pearl_public::JOB_KEY_END);
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Fill COMMITMENT_HASH (PUBLIC)
    debug_assert_eq!(public_inputs_builder.offset, pearl_public::COMMITMENT_HASH);
    let (_, commitment_hash) = compiled_params.commitment_hash;
    for chunk in commitment_hash.chunks_exact(BYTES_PER_GOLDILOCKS) {
        public_inputs_builder.dump_u64(u64_pack_le(chunk, 8));
    }
    debug_assert_eq!(public_inputs_builder.offset, pearl_public::COMMITMENT_HASH_END);

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Fill HASH_A (PUBLIC)
    debug_assert_eq!(public_inputs_builder.offset, pearl_public::HASH_A);
    for chunk in public_params.hash_a.chunks_exact(BYTES_PER_GOLDILOCKS) {
        public_inputs_builder.dump_u64(u64_pack_le(chunk, 8));
    }
    debug_assert_eq!(public_inputs_builder.offset, pearl_public::HASH_A_END);

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Fill HASH_B (PUBLIC)
    debug_assert_eq!(public_inputs_builder.offset, pearl_public::HASH_B);
    for chunk in public_params.hash_b.chunks_exact(BYTES_PER_GOLDILOCKS) {
        public_inputs_builder.dump_u64(u64_pack_le(chunk, 8));
    }
    debug_assert_eq!(public_inputs_builder.offset, pearl_public::HASH_B_END);

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Fill HASH_JACKPOT (PUBLIC)
    // The jackpot blake is at the end of the pre-padding rows, not at num_rows-1 (which is after padding)
    let jackpot_row = compiled_params.expected_num_rows() - 1;
    debug_assert_eq!(public_inputs_builder.offset, pearl_public::HASH_JACKPOT);
    // Note: we could have just computed it as below: blake3(jackpot, commitment_hash)
    let jackpot_hash: [u32; 8] = read_from_trace(&trace[jackpot_row], pearl_columns::CV_OUT);
    for &word in &jackpot_hash {
        public_inputs_builder.dump_u64(word as u64);
    }
    debug_assert_eq!(public_inputs_builder.offset, pearl_public::HASH_JACKPOT_END);

    // Debug verification: compare JACKPOT_MSG and hash with expected computation
    #[cfg(debug_assertions)]
    {
        use crate::v1::circuit::utils::trace_utils::bytes_to_words;

        let jackpot_msg_from_trace: [u32; 16] = read_from_trace(&trace[jackpot_row], pearl_columns::JACKPOT_MSG);
        assert_eq!(
            jackpot_msg_from_trace, expected_jackpot,
            "JACKPOT_MSG from trace doesn't match compute_jackpot result"
        );

        // Verify the hash matches expected blake3 computation
        let jackpot_as_bytes: Vec<u8> = jackpot_msg_from_trace.iter().flat_map(|elem| elem.to_le_bytes()).collect();
        let expected_hash = pearl_blake3::blake3_digest(&jackpot_as_bytes, Some(commitment_hash));
        let expected_hash_words: [u32; 8] = bytes_to_words(&expected_hash);
        assert_eq!(
            jackpot_hash, expected_hash_words,
            "HASH_JACKPOT from trace doesn't match expected blake3 computation"
        );
    }

    public_inputs_builder.assert_end();

    (trace, public_inputs)
}

/// Inject 192 bits of cryptographic randomness into the trace to blind the witness.
/// Overwrite `UINT8_DATA` on rounds 0-2 of the jackpot-blake compression.
fn blind_trace<F: RichField + Extendable<D>, const D: usize>(trace: &mut [[F; pearl_columns::TOTAL]], num_rows: usize) {
    let blinding_rows = [num_rows - 8, num_rows - 7, num_rows - 6];
    let random = F::rand_vec(blinding_rows.len() * pearl_columns::UINT8_DATA_LEN);
    for (&row_idx, chunk) in blinding_rows.iter().zip(random.chunks(pearl_columns::UINT8_DATA_LEN)) {
        for (c, f) in chunk.iter().enumerate() {
            trace[row_idx][pearl_columns::UINT8_DATA + c] = F::from_canonical_u64(f.to_canonical_u64() as u8 as u64);
        }
    }
}

fn fill_lookup_table_frequencies<F: RichField + Extendable<D>, const D: usize>(
    lookup_data: &[Lookup<F>],
    trace: &mut Vec<[F; pearl_columns::TOTAL]>,
    num_rows: usize,
) {
    let mut participating_columns = vec![false; pearl_columns::TOTAL];
    for col_idx in lookup_data.iter().flat_map(|lookup| {
        lookup
            .columns
            .iter()
            .chain([&lookup.table_column, &lookup.frequencies_column])
            .flat_map(|col| col.relevant_columns())
            .chain(lookup.filter_columns.iter().flat_map(|filter| filter.relevant_columns()))
    }) {
        participating_columns[col_idx] = true;
    }

    let column_values: Vec<PolynomialValues<F>> = participating_columns
        .iter()
        .enumerate()
        .collect::<Vec<_>>()
        .into_par_iter()
        .map(|(col_idx, &is_participating)| {
            if is_participating {
                PolynomialValues::new(trace.iter().map(|row| row[col_idx]).collect())
            } else {
                PolynomialValues::zero(1)
            }
        })
        .collect();

    // Batch evaluate all columns, filters, and table columns in parallel
    let all_cols: Vec<_> = lookup_data.iter().flat_map(|l| l.columns.iter()).collect();
    let all_filters: Vec<_> = lookup_data.iter().flat_map(|l| l.filter_columns.iter()).collect();
    let all_table_cols: Vec<_> = lookup_data.iter().map(|l| &l.table_column).collect();

    let all_col_evals: Vec<Vec<F>> = all_cols.par_iter().map(|col| col.eval_all_rows(&column_values)).collect();
    let all_filter_evals: Vec<Vec<F>> = all_filters
        .par_iter()
        .map(|filter| filter.eval_all_rows(&column_values))
        .collect();
    let all_table_evals: Vec<Vec<F>> = all_table_cols
        .par_iter()
        .map(|col| col.eval_all_rows(&column_values))
        .collect();

    // Build column offset indices for each lookup
    let offsets: Vec<_> = lookup_data
        .iter()
        .scan(0, |offset, lookup| {
            let start = *offset;
            *offset += lookup.columns.len();
            Some((start, lookup.columns.len()))
        })
        .collect();

    // Compute frequency columns in parallel
    let freq_columns: Vec<(usize, Vec<F>)> = lookup_data
        .par_iter()
        .enumerate()
        .map(|(lookup_idx, lookup)| {
            let (offset, num_cols) = offsets[lookup_idx];
            let col_evals = &all_col_evals[offset..offset + num_cols];
            let filter_evals = &all_filter_evals[offset..offset + num_cols];
            let table_values = &all_table_evals[lookup_idx];

            // Build reverse index: table value -> array index
            let value_to_idx: HashMap<F, usize> = table_values.iter().enumerate().map(|(i, v)| (*v, i)).collect();

            // Count frequencies in parallel over columns, then reduce
            let freq_counts: Vec<u64> = col_evals
                .par_iter()
                .zip(filter_evals.par_iter())
                .map(|(col_vals, filter_vals)| {
                    let mut counts = vec![0u64; table_values.len()];
                    for i in 0..num_rows {
                        if filter_vals[i] == F::ONE
                            && let Some(&idx) = value_to_idx.get(&col_vals[i])
                        {
                            counts[idx] += 1;
                        }
                    }
                    counts
                })
                .reduce(
                    || vec![0u64; table_values.len()],
                    |mut a, b| {
                        a.iter_mut().zip(b.iter()).for_each(|(x, y)| *x += *y);
                        a
                    },
                );

            let freq_col_idx = lookup
                .frequencies_column
                .get_single_column_idx()
                .expect("Frequency column should be a single column");

            let freq_column: Vec<F> = freq_counts.into_iter().map(F::from_canonical_u64).collect();
            (freq_col_idx, freq_column)
        })
        .collect();

    // Write back frequency columns
    for i in 0..num_rows {
        for (freq_col_idx, freq_column) in &freq_columns {
            trace[i][*freq_col_idx] = freq_column[i];
        }
    }
}
