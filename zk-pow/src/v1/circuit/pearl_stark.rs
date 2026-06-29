use std::marker::PhantomData;

use sorted_vec::SortedSet;

use anyhow::{Result, ensure};
use plonky2::field::extension::{Extendable, FieldExtension};
use plonky2::field::packed::PackedField;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::ext_target::ExtensionTarget;
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2_field::types::Field;
use starky::constraint_consumer::{ConstraintConsumer, RecursiveConstraintConsumer};
use starky::evaluation_frame::StarkFrame;
use starky::lookup::{Column, Filter, Lookup};
use starky::stark::Stark;

use crate::v1::api::proof::{PrivateProofParams, PublicProofParams};
use crate::v1::api::proof_utils::CompiledPublicParams;
use crate::v1::circuit::chip::blake3::program::{AuxiliaryCvLocation, AuxiliaryMsgLocation, DWORD_SIZE};
use crate::v1::circuit::chip::blake3::{Blake3Chip, Blake3ChipConfig};
use crate::v1::circuit::chip::{
    ControlAndMatIDPackedChip, I8U8Chip, IRange7P1Chip, IRange8Chip, InputChip, MatmulChip, MatmulChipConfig, StarkRowChip,
    URange8Chip, URange13Chip,
};
use crate::v1::circuit::chip::{JackpotChip, JackpotChipConfig};
use crate::v1::circuit::pearl_air::eval_constraints;
use crate::v1::circuit::pearl_layout::{pearl_columns, pearl_public};
use crate::v1::circuit::pearl_preprocess::generate_preprocessed;
use crate::v1::circuit::pearl_program::{RowLogic, TILE_D, TILE_H};
use crate::v1::circuit::pearl_trace::generate_trace;
use crate::v1::circuit::utils::native_evaluator::NativeEvaluator;
use crate::v1::circuit::utils::symbolic_evaluator::SymbolicEvaluator;

#[derive(Clone, Debug, Default)]
pub struct PearlStarkChips {
    pub urange8_chip: URange8Chip,
    pub urange13_chip: URange13Chip,
    pub irange7p1_chip: IRange7P1Chip,
    pub irange8_chip: IRange8Chip,
    pub i8u8_chip: I8U8Chip,
    pub stark_row_chip: StarkRowChip,
    pub blake3_chip: Blake3Chip,
    pub jackpot_chip: JackpotChip,
    pub matmul_chip: MatmulChip,
    pub input_chip: InputChip,
    pub control_and_matid_packer: ControlAndMatIDPackedChip,
}

#[derive(Clone, Debug)]
pub struct PearlStarkConfig {
    pub compiled_public_params: CompiledPublicParams,
    pub blake3_chip_config: Blake3ChipConfig,
    pub jackpot_chip_config: JackpotChipConfig,
    pub matmul_chip_config: MatmulChipConfig,

    pub blake3_msg_locations: Vec<AuxiliaryMsgLocation>,
    pub blake3_cv_locations: Vec<AuxiliaryCvLocation>,

    /// Structured row logic produced by `CompiledPublicParams::structure_proof`.
    /// Cached here so that trace generation and preprocessed-column generation do not
    /// have to recompile the STARK program a second time.
    pub structured_circuit: Vec<RowLogic>,
}

impl PearlStarkConfig {
    pub fn new(public_params: &PublicProofParams) -> Self {
        let (compiled_public_params, blake3_msg_locations, blake3_cv_locations) = public_params.compile();
        let structured_circuit = compiled_public_params.structure_proof().unwrap();
        let blake_logic: Vec<_> = structured_circuit.iter().map(|r| r.blake).collect();
        let matmul_logic: Vec<_> = structured_circuit.iter().map(|r| r.matmul).collect();
        let jackpot_logic: Vec<_> = structured_circuit.iter().map(|r| r.jackpot).collect();
        let blake3_chip_config = Blake3ChipConfig::new(blake_logic);
        let matmul_chip_config = MatmulChipConfig::new(matmul_logic);
        let jackpot_chip_config = JackpotChipConfig::new(jackpot_logic);

        Self {
            compiled_public_params,
            blake3_chip_config,
            blake3_msg_locations,
            blake3_cv_locations,
            jackpot_chip_config,
            matmul_chip_config,
            structured_circuit,
        }
    }
}

/// STARK for the Pearl proving system.
#[derive(Default, Clone, Debug)]
pub struct PearlStark<F: RichField + Extendable<D>, const D: usize> {
    _phantom: std::marker::PhantomData<F>,
    pub chips: PearlStarkChips,
    pub config: Option<PearlStarkConfig>,
}

impl<F: RichField + Extendable<D>, const D: usize> Stark<F, D> for PearlStark<F, D> {
    type EvaluationFrame<FE, P, const D2: usize>
        = StarkFrame<P, P::Scalar, { pearl_columns::TOTAL }, { pearl_public::TOTAL }>
    where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>;

    type EvaluationFrameTarget =
        StarkFrame<ExtensionTarget<D>, ExtensionTarget<D>, { pearl_columns::TOTAL }, { pearl_public::TOTAL }>;

    fn eval_packed_generic<FE, P, const D2: usize>(
        &self,
        vars: &Self::EvaluationFrame<FE, P, D2>,
        yield_constr: &mut ConstraintConsumer<P>,
    ) where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>,
    {
        let mut evaluator = NativeEvaluator::new(yield_constr);
        eval_constraints(&self.chips, vars, &mut evaluator);
    }

    fn eval_ext_circuit(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        vars: &Self::EvaluationFrameTarget,
        yield_constr: &mut RecursiveConstraintConsumer<F, D>,
    ) {
        let mut evaluator = SymbolicEvaluator::new(builder, yield_constr);
        eval_constraints(&self.chips, vars, &mut evaluator);
    }

    fn lookups(&self) -> Vec<Lookup<F>> {
        // Constant tables inclusion checks
        let mut lookup_data = vec![
            unfiltered_lookup(
                Column::<F>::single(pearl_columns::URANGE8_TABLE),
                Column::<F>::single(pearl_columns::URANGE8_FREQ),
                Column::<F>::singles(pearl_columns::UINT8_DATA_RANGE).collect(),
            ),
            unfiltered_lookup(
                Column::<F>::single(pearl_columns::URANGE13_TABLE),
                Column::<F>::single(pearl_columns::URANGE13_FREQ),
                Column::<F>::singles(pearl_columns::MAT_ID_LIMBS_RANGE.chain(pearl_columns::AB_ID_LIMBS_RANGE)).collect(),
            ),
            // Signal is in [-64, 64].
            unfiltered_lookup(
                Column::<F>::single(pearl_columns::IRANGE7P1_TABLE),
                Column::<F>::single(pearl_columns::IRANGE7P1_FREQ),
                Column::<F>::singles(pearl_columns::MAT_UNPACK_RANGE.chain(pearl_columns::NOISE_UNPACK_RANGE)).collect(),
            ),
            unfiltered_lookup(
                Column::<F>::single(pearl_columns::IRANGE8_TABLE),
                Column::<F>::single(pearl_columns::IRANGE8_FREQ),
                Column::<F>::singles(pearl_columns::A_NOISED_UNPACK_RANGE.chain(pearl_columns::B_NOISED_UNPACK_RANGE)).collect(),
            ),
            // Conversion: MAT_UNPACK: int7 -> UINT8_DATA: uint8, conditional on IS_MSG_MAT.
            Lookup {
                columns: pearl_columns::UINT8_DATA_RANGE
                    .zip(pearl_columns::MAT_UNPACK_RANGE)
                    .map(|(mu8, mi8)| indexed_column::<F>(mu8, mi8, 1 << 8, 0))
                    .collect(),
                table_column: Column::<F>::single(pearl_columns::I8U8_TABLE),
                frequencies_column: Column::<F>::single(pearl_columns::I8U8_FREQ),
                filter_columns: vec![
                    Filter::<F>::from_column(Column::<F>::single(pearl_columns::IS_MSG_MAT));
                    pearl_columns::UINT8_DATA_LEN
                ],
            },
        ];

        // RAM by indexing: load noised matrix tiles (A, B) indexed by (stark)row and (matrix)strip.
        let (num_words, num_rows, num_strips) = (pearl_columns::NOISED_PACKED_LEN, TILE_D / DWORD_SIZE, TILE_H);
        let noised_a_b = |word_idx, row_idx, strip_idx| {
            let out_col = word_idx + num_words * (row_idx + num_rows * strip_idx);
            let idx_shift = strip_idx + row_idx * num_strips;
            [
                indexed_column::<F>(pearl_columns::A_NOISED + out_col, pearl_columns::A_ID, 1u64 << 32, idx_shift),
                indexed_column::<F>(pearl_columns::B_NOISED + out_col, pearl_columns::B_ID, 1u64 << 32, idx_shift),
            ]
        };
        for word_idx in 0..num_words {
            lookup_data.push(Lookup {
                columns: (0..num_rows)
                    .flat_map(|row| (0..num_strips).flat_map(move |strip| noised_a_b(word_idx, row, strip)))
                    .collect(),
                table_column: indexed_column::<F>(pearl_columns::NOISED_PACKED + word_idx, pearl_columns::MAT_ID, 1u64 << 32, 0),
                frequencies_column: Column::<F>::single(pearl_columns::MAT_FREQ),
                filter_columns: vec![
                    Filter::<F>::from_column(Column::<F>::single(pearl_columns::IS_UPDATE_CUMSUM));
                    2 * num_rows * num_strips
                ],
            });
        }

        for i in 0..pearl_columns::CV_OUT_LEN {
            lookup_data.push(Lookup {
                columns: vec![indexed_column::<F>(
                    pearl_columns::CV_IN + i,
                    pearl_columns::CV_OR_TWEAK_PREP, // CV_OR_TWEAK_PREP serves as cv_idx
                    1u64 << 32,
                    0,
                )],
                frequencies_column: Column::<F>::single(pearl_columns::CV_OUT_FREQ),
                table_column: indexed_column::<F>(pearl_columns::CV_OUT + i, pearl_columns::STARK_ROW_IDX, 1u64 << 32, 0),
                filter_columns: vec![Filter::<F>::from_column(Column::<F>::single(pearl_columns::IS_CV_IN)); 1],
            });
        }

        lookup_data
    }

    fn constraint_degree(&self) -> usize {
        3
    }
}

impl<F: RichField + Extendable<D>, const D: usize> PearlStark<F, D> {
    pub fn new_with_params(public_params: &PublicProofParams) -> Self {
        Self {
            _phantom: PhantomData,
            chips: PearlStarkChips::default(),
            config: Some(PearlStarkConfig::new(public_params)),
        }
    }

    pub fn preprocessed_indices() -> SortedSet<usize> {
        SortedSet::from_unsorted(vec![
            pearl_columns::CONTROL_PREP,
            pearl_columns::NOISE_PACKED_PREP,
            pearl_columns::CV_OR_TWEAK_PREP,
            pearl_columns::AB_ID_PREP,
        ])
    }

    pub fn generate_trace(
        &self,
        public_params: &PublicProofParams,
        private_params: PrivateProofParams,
    ) -> (Vec<[F; pearl_columns::TOTAL]>, [F; pearl_public::TOTAL]) {
        let dotprod_length = public_params.dot_product_length();
        assert_eq!(private_params.s_a.len(), public_params.h(), "s_a must have h strips");
        assert_eq!(private_params.s_b.len(), public_params.w(), "s_b must have w strips");
        for strip in private_params.s_a.iter().chain(private_params.s_b.iter()) {
            assert_eq!(
                strip.len(),
                dotprod_length,
                "strips has length={} but should be {}",
                strip.len(),
                dotprod_length
            );
        }

        generate_trace(
            &self.chips,
            self.config.as_ref().unwrap(),
            &self.lookups(),
            public_params,
            private_params,
        )
    }

    /// Returns preprocessed column values in `preprocessed_indices()` order.
    ///
    /// The output order matches `preprocessed_indices()`.
    pub fn preprocessed_columns(public_params: &CompiledPublicParams) -> Result<Vec<Vec<F>>> {
        let (mut preprocessed, _) = generate_preprocessed(public_params, None)?;
        preprocessed.sort_by_key(|(idx, _)| *idx);

        let got_indices: Vec<usize> = preprocessed.iter().map(|(idx, _)| *idx).collect();
        let expected_indices: Vec<usize> = Self::preprocessed_indices().iter().copied().collect();
        ensure!(
            got_indices == expected_indices,
            "generate_preprocessed returned columns {:?} but preprocessed_indices() expects {:?}",
            got_indices,
            expected_indices,
        );

        Ok(preprocessed
            .into_iter()
            .map(|(_, col)| col.into_iter().map(F::from_canonical_u64).collect())
            .collect())
    }
}

/// Lookup with no filter (all rows participate).
pub fn unfiltered_lookup<F: Field>(table_column: Column<F>, frequencies_column: Column<F>, columns: Vec<Column<F>>) -> Lookup<F> {
    let filter_columns = vec![Filter::<F>::default(); columns.len()];
    Lookup {
        columns,
        table_column,
        frequencies_column,
        filter_columns,
    }
}

/// Virtual column: col_elem + factor * (col_idx + shift), for RAM-style indexing.
fn indexed_column<F: Field>(elem: usize, idx: usize, factor: u64, shift: usize) -> Column<F> {
    Column::linear_combination_with_constant(
        vec![(elem, F::ONE), (idx, F::from_canonical_u64(factor))],
        F::from_canonical_u64(factor * shift as u64),
    )
}
