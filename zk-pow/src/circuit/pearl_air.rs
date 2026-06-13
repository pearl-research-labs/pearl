//! Pearl AIR: Constraint Definitions for the Pearl STARK
//! =====================================================
//!
//! The Pearl STARK trace has four interleaved chips sharing every row:
//!   Input, Blake3, Matmul, and Jackpot.
//!
//! Each row's behavior is determined by preprocessed control columns (see pearl_preprocess.rs),
//! which encode the program defined in pearl_program.rs. The Blake3 instruction stream and the
//! Matmul+Jackpot instruction stream run at independent lengths and are interleaved into
//! the same trace. Padding rows are no-ops for all chips.
//!
//! Preprocessed Columns
//! --------------------
//! 1. CONTROL_PREP: Control logic specifying what each row does across all chips. Also
//!    contains MAT_ID, an identifier for the original location of this row's matrix input,
//!    as well as indices and flags related to outer_indices (in the MoE case).
//! 2. NOISE_PACKED_PREP: The precomputed noise to add to the matrix data fed to this row.
//! 3. CV_OR_TWEAK_PREP: Additional Blake3 control. Either a row index to read a previous
//!    blake3 output from (parsed as CV, or as message for a parent compression), or the
//!    blake3 compression tweak flags (counter, block_len, flags).
//! 4. AB_ID_PREP: The ID numbers of the two TILE_H × TILE_D tiles of A and B to load
//!    from the input (matching MAT_ID of other rows via RAM lookup).
//!
//! Range Tables and Lookups
//! ------------------------
//! Five range tables ensure values stay in their declared ranges (uint8, uint13, int7+1,
//! int8, and an int8-to-uint8 conversion table). The I8U8 table enables verified conversion
//! between signed and unsigned byte representations, conditional on IS_MSG_MAT rows.
//!
//! RAM lookups (via logup) enforce that A_NOISED/B_NOISED tile reads are consistent with
//! the NOISED_PACKED values at matching MAT_ID rows. CV routing lookups enforce that CV_IN
//! reads are consistent with CV_OUT values at the referenced row.
//!
//! Public Input Binding
//! --------------------
//! CV_OUT on rows flagged IS_HASH_A, IS_HASH_B, IS_HASH_JACKPOT is constrained to match
//! the public inputs HASH_A, HASH_B, HASH_JACKPOT. JOB_KEY and COMMITMENT_HASH enter the
//! circuit as CV inputs to Blake3, verified in the CV routing constraints.

use crate::circuit::pearl_stark::PearlStarkChips;
use crate::circuit::utils::evaluator::Evaluator;
use starky::evaluation_frame::{StarkEvaluationFrame, StarkFrame};

use crate::circuit::pearl_layout::{pearl_columns, pearl_public};
use crate::circuit::utils::air_utils::RowView;

pub(crate) fn eval_constraints<V, S, E>(
    chips: &PearlStarkChips,
    vars: &StarkFrame<V, S, { pearl_columns::TOTAL }, { pearl_public::TOTAL }>,
    eval: &mut E,
) where
    V: Copy + Default,
    S: Copy + Default,
    E: Evaluator<V, S>,
{
    // Read the prover's evaluation at the challenge point.
    let local_trace = vars.get_local_values();
    let next_trace = vars.get_next_values();
    let public_inputs = vars.get_public_inputs();
    let mut row_view = RowView::new(local_trace);

    // Check consecutive range tables
    chips.urange8_chip.eval_constraints(&mut row_view, next_trace, eval);
    chips.urange13_chip.eval_constraints(&mut row_view, next_trace, eval);
    chips.irange7p1_chip.eval_constraints(&mut row_view, next_trace, eval);
    chips.irange8_chip.eval_constraints(&mut row_view, next_trace, eval);
    chips.i8u8_chip.eval_constraints(&mut row_view, next_trace, eval);

    // Control flags and MatId Packer
    let (_matmul_cf, blake3_cf, jackpot_cf) = chips.control_and_matid_packer.eval_constraints(&mut row_view, eval);

    // StarkRow chip
    chips.stark_row_chip.eval_constraints(&mut row_view, next_trace, eval);

    // Input chip
    let uint8_data = chips.input_chip.eval_constraints(eval, &mut row_view);

    // Blake3 chip
    let blake3_output = chips
        .blake3_chip
        .eval_constraints(vars, eval, &mut row_view, &blake3_cf, &uint8_data);

    // MatMul Chip
    let cumsum_tile = chips.matmul_chip.eval_constraints(vars, eval, &mut row_view);

    // Jackpot Chip
    chips
        .jackpot_chip
        .eval_constraints(vars, eval, &mut row_view, &jackpot_cf, &cumsum_tile);

    // Check HASH_A agrees with blake3_output
    for i in 0..pearl_public::HASH_A_LEN {
        let pub_hash_a = eval.scalar(public_inputs[pearl_public::HASH_A + i]);
        let out_cv = blake3_output[i];
        eval.constraint_eq_if(blake3_cf.is_hash_a, pub_hash_a, out_cv);
    }

    // Check HASH_B agrees with blake3_output
    for i in 0..pearl_public::HASH_B_LEN {
        let pub_hash_b = eval.scalar(public_inputs[pearl_public::HASH_B + i]);
        let out_cv = blake3_output[i];
        eval.constraint_eq_if(blake3_cf.is_hash_b, pub_hash_b, out_cv);
    }

    // Check HASH_JACKPOT agrees with blake3_output
    for i in 0..pearl_public::HASH_JACKPOT_LEN {
        let pub_hash_jackpot = eval.scalar(public_inputs[pearl_public::HASH_JACKPOT + i]);
        let out_cv = blake3_output[i];
        eval.constraint_eq_if(blake3_cf.is_hash_jackpot, pub_hash_jackpot, out_cv);
    }

    // Check HASH_ROUTING agrees with blake3_output
    for i in 0..pearl_public::HASH_ROUTING_LEN {
        let pub_hash_routing = eval.scalar(public_inputs[pearl_public::HASH_ROUTING + i]);
        let out_cv = blake3_output[i];
        eval.constraint_eq_if(blake3_cf.is_hash_routing, pub_hash_routing, out_cv);
    }

    debug_assert_eq!(row_view.offset, pearl_columns::TOTAL);
    row_view.assert_end();
}
