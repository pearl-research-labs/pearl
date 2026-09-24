//! Class (a) ("known") column assembly for the fp8 batch system.
//!
//! Every fp8 STARK leads with a block of class (a) columns — values the verifier can
//! recompute from the compiled programs and public data alone (schedules, indices, noise
//! decode fields; see each `columns.rs`). Under the batch multi-STARK commitment these are
//! **committed with the trace** like any online column (they are per-job data, so they cannot
//! live in the setup-time preprocessed oracle the way the LUTs of `super::luts` do), and the
//! verifier *re-derives* them: `starky::batch_prover::batch_prove` absorbs the
//! [`BatchKnownColumns`] digest into the Fiat-Shamir transcript, and
//! `starky::batch_verifier::batch_verify` recomputes each known column's openings at `zeta`
//! and `g*zeta` from the values assembled here and checks them against the proof's claimed
//! trace openings. A prover therefore cannot lie about any class (a) column without breaking
//! the FRI binding of the trace commitment itself.
//!
//! The digest slot is not a hash of those column values. It is the job's `statement_digest`
//! (`PublicParams::digest`), bound at prove/verify — not at assembly. The columns are
//! uniquely determined by that statement, so a collision-resistant hash of the statement is
//! a valid Fiat-Shamir salt.
//!
//! The per-table generators are `Blake3Program::known_values`,
//! `InputQuantProgram::known_values`, `ScaleProgram::known_values`,
//! `MatmulProgram::known_values`, `XorFoldProgram::known_values` and
//! `TamedProgram::known_values` — each bit-exact with its `generate_trace` fill (asserted by
//! `super::consistency`). [`fp8_known_columns`] packs their outputs into the
//! [`BatchKnownColumns`] handed to both the batch prover and verifier.

use plonky2::field::polynomial::PolynomialValues;
use plonky2::hash::hash_types::{HashOut, RichField};
use primitive_types::U256;
use starky::batch_verifier::BatchKnownColumns;

use super::blake3_stark::columns::NUM_BLAKE3_KNOWN_COLUMNS;
use super::ctl::{NUM_ALL_TABLES, NUM_TABLES};
use super::input_quant_stark::columns::NUM_INPUT_QUANT_KNOWN_COLUMNS;
use super::matmul_b200_stark::columns::NUM_MATMUL_B200_KNOWN_COLUMNS;
use super::scale_stark::columns::NUM_SCALE_KNOWN_COLUMNS;
use super::tamed_stark::columns::NUM_TAMED_KNOWN_COLUMNS;
use super::xor_fold_stark::columns::NUM_XOR_FOLD_KNOWN_COLUMNS;
use crate::api::primitives::Hash256;

/// Known-column count of every main table, in `super::ctl::Table` order. Each table's known
/// block is its *leading* columns (indices `0..count`), by the `columns.rs` layouts. The LUT
/// tables of the batch carry no known columns — their static halves are *preprocessed* (bound
/// by the setup cap, `super::luts`), and their multiplicity columns are ordinary online data.
pub const KNOWN_COLUMNS_PER_TABLE: [usize; NUM_TABLES] = [
    NUM_BLAKE3_KNOWN_COLUMNS,
    NUM_INPUT_QUANT_KNOWN_COLUMNS,
    NUM_SCALE_KNOWN_COLUMNS,
    NUM_MATMUL_B200_KNOWN_COLUMNS,
    NUM_XOR_FOLD_KNOWN_COLUMNS,
    NUM_TAMED_KNOWN_COLUMNS,
];

/// Reduce a 32-byte hash into four Goldilocks elements (little-endian integer mod `p^4`).
pub(crate) fn hash256_to_hash_out<F: RichField>(hash: Hash256) -> HashOut<F> {
    let p = U256::from(F::ORDER);
    let mut v = U256::from_little_endian(&hash);
    let mut elements = [F::ZERO; 4];
    for e in elements.iter_mut() {
        *e = F::from_canonical_u64((v % p).as_u64());
        v /= p;
    }
    HashOut { elements }
}

/// Packs the six main tables' known-column values (in `Table` order, each from its
/// program's `known_values`) into the [`BatchKnownColumns`] fed to
/// `batch_prove`/`batch_verify`, covering all [`NUM_ALL_TABLES`] batch tables (the LUT
/// entries empty): the leading-block column indices and the values the verifier reopens at
/// `zeta`. The Fiat-Shamir digest slot is left empty; prove/verify bind `statement_digest`
/// in place (`HashOut` is `Copy`) before absorbing the struct.
///
/// Panics if a table's column count differs from [`KNOWN_COLUMNS_PER_TABLE`] or a table's
/// columns have mismatched or non-power-of-two heights — signs of a mis-ordered argument.
pub fn fp8_known_columns<F: RichField>(values_per_table: [Vec<PolynomialValues<F>>; NUM_TABLES]) -> BatchKnownColumns<F> {
    for (t, values) in values_per_table.iter().enumerate() {
        assert_eq!(
            values.len(),
            KNOWN_COLUMNS_PER_TABLE[t],
            "table {t}: wrong known-column count (tables mis-ordered?)"
        );
        for col in values {
            assert_eq!(col.len(), values[0].len(), "table {t}: ragged known columns");
        }
        assert!(values[0].len().is_power_of_two(), "table {t}: height not a power of two");
    }
    let mut columns_per_table: Vec<Vec<usize>> = KNOWN_COLUMNS_PER_TABLE.iter().map(|&n| (0..n).collect()).collect();
    columns_per_table.resize(NUM_ALL_TABLES, Vec::new());
    let mut values_per_table: Vec<Vec<PolynomialValues<F>>> = values_per_table.into();
    values_per_table.resize(NUM_ALL_TABLES, Vec::new());
    BatchKnownColumns {
        digest: None,
        columns_per_table,
        values_per_table,
    }
}
