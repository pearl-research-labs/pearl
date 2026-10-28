//! The fp8 batch driver: one proof for the whole 21-table system.
//!
//! [`Fp8System`] is the *statement* of one job: the five compiled programs and the public
//! recompute inputs of the class (a) columns (plane
//! byte lengths, routing words, the noise codes). Both the prover and the verifier build it
//! from public data alone, and it derives everything batching needs:
//!
//! - the **batch order**: canonical table order (`Table` 0..4, then the
//!   device's LUT inventory) — profile-independent within each device, so Fiat-Shamir observes every
//!   trace/auxiliary/quotient cap in the same order for every job (a
//!   universal-verifier prerequisite), and CTL table indices need no renumbering.
//!   Each role (trace / auxiliary / quotient) commits all 21 tables in one *merged*
//!   multi-height tree (`merged = true`): every table enters as a per-row digest
//!   at its runtime height, in table-index order at ties — so a proof
//!   carries one cap per role instead of `21`, and the universal verifier walks one
//!   conditional path per role per query. Batched FRI still folds by height
//!   internally (`starky::batch_stark` sorts the *distinct* heights for its
//!   instances; the setup tree's flat column order stays job-independent). The
//!   verifier derives the heights from its own class (a) recompute — never from the
//!   proof — so a prover cannot influence the layout;
//! - the consensus [`StarkConfig`], whose fixed FRI reduction schedule is the consensus
//!   ladder over every *reachable* table height — a job folds the ladder suffix from its
//!   tallest table, so schedules agree across jobs ([`fp8_stark_config`]);
//! - the [`BatchKnownColumns`]: the class (a) values at their batch positions. The
//!   Fiat-Shamir digest slot is `statement_digest` (labelled hash of the proposed header
//!   and [`crate::v4::api::public_params::PublicParams`]), bound via
//!   [`Fp8System::bind_statement_digest`] — not at construction — then absorbed into
//!   Fiat-Shamir; the verifier re-opens the values;
//! - the LUT setup-commitment positions ([`Fp8System::preprocessed_data`] wraps
//!   `lut_preprocessed_data`; [`Fp8System::verify`] takes only the consensus cap and pins
//!   the per-table preprocessed column indices itself).
//!
//! [`Fp8System::prove`] regenerates the five main traces from the private strips, chaining
//! each table's inputs from its producer exactly like the consistency driver (InputQuant from
//! the strips, Scale from InputQuant's group tuples, device-specific Matmul from the noised
//! codes, XorFold from the cell results and census counts, Blake3 from the strip bytes and the folded lottery words),
//! accumulates the committed-LUT multiplicities ([`LutChecker`]; serving validation is
//! a debug/test-side concern owned by [`LutChecker::check_trace`]), assembles the sixteen LUT traces,
//! observes the generated lottery digest through its `derive_statement_digest` callback
//! (the Fiat-Shamir salt may depend on `J`), binds the returned `statement_digest`, and
//! hands the batch to `starky::batch_prover::batch_prove`. The main tables' public
//! inputs are whatever the witness generates (keys, roots, the lottery digest) — they ship
//! inside the proof.
//!
//! [`Fp8System::verify`] pins the proof's degree profile and public inputs to the
//! statement (the caller supplies the expected per-table public inputs — deriving the
//! bindable slots and reading the claim slots out of the proof is the API layer's job,
//! `crate::v4::api::zk`), requires the caller to have bound `statement_digest`
//! beforehand, and hands the proof to `starky::batch_verifier::batch_verify`,
//! which binds the trace openings to the recomputed class (a) columns, the consensus LUT cap
//! and the CTL multiset equalities under one batched FRI argument.
//!
//! **Scheme boundary.** This driver proves exactly one scheme: prequant fp8
//! ([`Quant::Fp8E4M3Prequant`](crate::v4::api::public_params::Quant::Fp8E4M3Prequant)).
//! Callers building a [`Fp8System`] from a parsed job must reject any other
//! quant discriminant at their entry; the `Blake3Program::from_blake_program` bridge
//! independently rejects programs that are not the four-plane prequant shape.

use core::borrow::Borrow;

use anyhow::{Result, anyhow, ensure};
use plonky2::field::extension::Extendable;
use plonky2::field::polynomial::PolynomialValues;
use plonky2::fri::FriConfig;
use plonky2::fri::reduction_strategies::FriReductionStrategy;
use plonky2::hash::hash_types::{HashOut, RichField};
use plonky2::hash::merkle_tree::MerkleCap;
use plonky2::plonk::config::GenericConfig;
use plonky2::timed;
use plonky2::util::log2_strict;
use plonky2::util::timing::TimingTree;
use primitive_types::U256;
use starky::batch_proof::BatchStarkProofWithPublicInputs;
use starky::batch_prover::{BatchStarkPreprocessedData, BatchStarkPreprocessedVerifierData, batch_prove};
use starky::batch_stark::BatchStark;
use starky::batch_universal::UniversalVerifierEnvelope;
use starky::batch_verifier::{BatchKnownColumns, batch_verify};
use starky::config::StarkConfig;
use starky::cross_table_lookup::CrossTableLookup;

use super::blake3_stark::columns::{NUM_BLAKE3_COLUMNS, NUM_BLAKE3_PUBLIC_INPUTS, PI_HASH_JACKPOT};
use super::blake3_stark::stark::{Blake3KnownInputs, Blake3Program, Blake3Stark, Blake3TraceInputs, calloc_rows};
use super::ctl::{NUM_ALL_TABLES, NUM_LUT_TABLES, NUM_TABLES, Table, all_cross_table_lookups, lut_inventories, lut_tables};
use super::input_quant_stark::columns::NUM_INPUT_QUANT_PUBLIC_INPUTS;
use super::input_quant_stark::stark::{InputQuantProgram, InputQuantStark};
use super::luts::stark::boxed_lut_stark;
use super::luts::{LutChecker, lut_height, lut_preprocessed_data, lut_trace, num_precommitted_columns};
use super::matmul_b200_stark::columns::NUM_MATMUL_PUBLIC_INPUTS;
use super::matmul_b200_stark::{MatmulB200ColumnsView, MatmulStarkB200};
use super::matmul_h100::MatmulStarkH100;
use super::scale_stark::columns::NUM_SCALE_PUBLIC_INPUTS;
use super::scale_stark::stark::{ScaleProgram, ScaleStark};
use super::xor_fold_stark::columns::NUM_XOR_FOLD_PUBLIC_INPUTS;
use super::xor_fold_stark::stark::{XorFoldProgram, XorFoldStark};
use crate::v2::api::proof_utils::u32_field_array_to_hash;
use crate::v4::api::primitives::Hash256;
use crate::v4::api::public_params::{Device, HashId};

// ==================================================================================================
// The consensus proof shape
// ==================================================================================================

/// Targeted (conjectured) security level in bits: `queries * rate_bits + proof_of_work_bits`.
/// The 120-bit target of the recursive wrapper stages and of V1/V2
/// (`v2::circuit::pearl_circuit::SECURITY_BITS`); `consensus_config_meets_the_security_target`
/// asserts the parameters reach it.
pub const STARK_SECURITY_BITS: usize = 120;
/// Logup/CTL challenge repetitions (soundness `~(rows + instances)/|F|` per repetition; three
/// repetitions push the batched-lookup error far below the FRI error).
pub const STARK_NUM_CHALLENGES: usize = 3;
/// FRI rate `2^-1` — the fastest prover rate admitting the degree-3 constraint envelope.
pub const STARK_RATE_BITS: usize = 1;
/// Merkle cap height of every oracle and FRI commit layer.
pub const STARK_CAP_HEIGHT: usize = 4;
/// FRI grinding bits.
pub const STARK_POW_BITS: u32 = 18;
/// FRI query rounds: the minimal count meeting the security target at this rate (102).
pub const STARK_QUERY_ROUNDS: usize = (STARK_SECURITY_BITS - STARK_POW_BITS as usize).div_ceil(STARK_RATE_BITS);

/// Every trace height (degree bits) reachable by some table of some envelope-legal job,
/// descending — the consensus FRI ladder's fold boundaries. The wire envelope
/// (`api::layout` lottery-tile bounds) is
/// `k ∈ [2^10, 2^16]` with `32 | k`, `h ≥ 4`, `w ≥ 16`, `h·w ∈ [256, 2048]`,
/// `(h + w)·k ≤ 2^22`; heights are `next_pow2` of each table's row formula:
///
/// | table      | rows             | bits    |
/// |------------|------------------|---------|
/// | InputQuant | `max(h,w)·k`     | 14..=22 |
/// | Matmul     | `h·w·(k/32)`     | 13..=22 |
/// | Blake3     | `8·compressions` | 13..=19 |
/// | XorFold    | `h·w`            |  8..=11 |
/// | Scale      | `h + w`          |  5..=10 |
/// | LUTs       | [`lut_height`]   | {17, 16, 14, 11, 10, 8, 6, 5} |
///
/// Blake3 work is bounded in closed form from the statement's scalars before
/// compilation (`PublicParams::check_work_bounds`): four plane trees, MoE
/// routing/offsets trees, two root folds, and the jackpot together consume at
/// most `2^22` compression-message bytes. Thus there are at most `2^16`
/// compressions and `2^19` rows. The minimum legal tile still has
/// more than `2^12` Blake3 rows. The union is `[5, 22] \ {12}`: no table lands on `2^12`
/// (Scale and XorFold top out at `2^10`/`2^11`, and Matmul, InputQuant, and Blake3 all start
/// at `2^13`).
/// Coverage and minimality are asserted by
/// `ladder_covers_the_envelope`; real jobs are gated onto the ladder at
/// `Fp8Job::derive`.
pub const FP8_REACHABLE_DEGREE_BITS: [usize; 17] = [22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 11, 10, 9, 8, 7, 6, 5];

/// Per-main-table inclusive degree-bits ranges `[lo, hi]` over the wire envelope, in
/// canonical `Table` order — the rows of [`FP8_REACHABLE_DEGREE_BITS`]'s table. Exactness
/// (each bound is attained by some envelope-legal job) is asserted by
/// `ladder_covers_the_envelope`; Blake3's bounds are proven, not swept (see above).
pub const FP8_MAIN_TABLE_DEGREE_RANGES: [(usize, usize); NUM_TABLES] = [(13, 19), (14, 22), (5, 10), (13, 22), (8, 11)];

/// Batch indices of the *grouped* tables — the LUTs
/// (`NUM_TABLES..NUM_ALL_TABLES`). Their heights are consensus constants. Each
/// online table enters the merged tree as a row digest at its height; the proof
/// carries one cap per role, and the
/// recursive verifier walks one tree per role per query.
pub const FP8_GROUPED_TABLES: [usize; NUM_LUT_TABLES] = {
    let mut tables = [0; NUM_LUT_TABLES];
    let mut i = 0;
    while i < NUM_LUT_TABLES {
        tables[i] = NUM_TABLES + i;
        i += 1;
    }
    tables
};

/// The consensus universal-verifier envelope: the main tables range
/// over [`FP8_MAIN_TABLE_DEGREE_RANGES`], the LUT tables are fixed at their
/// consensus heights and grouped ([`FP8_GROUPED_TABLES`]), and the fold ladder is
/// [`FP8_REACHABLE_DEGREE_BITS`]. One stage-1 wrapper circuit built on this envelope
/// verifies every envelope-legal job (the job's degree profile becomes a
/// public input).
pub fn fp8_universal_envelope(device: Device) -> UniversalVerifierEnvelope {
    let mut degree_ranges: Vec<(usize, usize)> = FP8_MAIN_TABLE_DEGREE_RANGES.to_vec();
    degree_ranges.extend(lut_tables(device).iter().map(|&table| {
        let bits = log2_strict(lut_height(table));
        (bits, bits)
    }));
    UniversalVerifierEnvelope {
        degree_ranges,
        ladder: FP8_REACHABLE_DEGREE_BITS.to_vec(),
        grouped_tables: FP8_GROUPED_TABLES.to_vec(),
    }
}

/// The consensus [`StarkConfig`] for a job whose batch tables have the given heights
/// (`degree_bits` per table, any order, duplicates allowed). Everything is a consensus
/// constant, including the FRI reduction strategy: a [`FriReductionStrategy::Ladder`] whose
/// boundaries are every *reachable* height ([`FP8_REACHABLE_DEGREE_BITS`]). A job's schedule
/// is the ladder's suffix from its tallest table down to the `2^5` bottom, so any two jobs'
/// schedules agree on their shared span, and — because the ladder serializes as the boundary
/// list, not the suffix — the Fiat-Shamir transcript absorbs the *same* config for every
/// envelope job: the universal-verifier prerequisite. Batch FRI injects each instance when
/// the folded codeword reaches its LDE size and folds plainly through instance-free
/// boundaries.
///
/// Sub-envelope profiles (unit-test geometries) add their own heights as extra boundaries;
/// the config stays a pure function of the degree profile, and equals the consensus
/// constant for every on-ladder profile (real jobs are ladder-gated at `Fp8Job::derive`).
pub fn fp8_stark_config(degree_bits: &[usize]) -> StarkConfig {
    let min = degree_bits.iter().copied().min().expect("at least one table");
    assert!(
        min + STARK_RATE_BITS >= STARK_CAP_HEIGHT,
        "the smallest table's LDE must cover the Merkle cap"
    );
    let mut boundaries: Vec<usize> = FP8_REACHABLE_DEGREE_BITS.iter().chain(degree_bits).copied().collect();
    boundaries.sort_unstable_by_key(|&bits| core::cmp::Reverse(bits));
    boundaries.dedup();
    StarkConfig::new(
        STARK_SECURITY_BITS,
        STARK_NUM_CHALLENGES,
        FriConfig {
            rate_bits: STARK_RATE_BITS,
            cap_height: STARK_CAP_HEIGHT,
            proof_of_work_bits: STARK_POW_BITS,
            reduction_strategy: FriReductionStrategy::Ladder(boundaries),
            num_query_rounds: STARK_QUERY_ROUNDS,
        },
    )
}

// ==================================================================================================
// The statement
// ==================================================================================================

/// The public data defining one fp8 statement: the five compiled programs and the
/// class (a) recompute inputs. The caller derives all of it from the parsed job; **it must
/// reject any job with `mma_type != Bf16ToFp8Fp32` before building a statement**. The
/// main tables' public inputs are not part of the statement object: the prover generates
/// them with the traces and the verifier pins them via [`Fp8System::verify`]'s
/// `expected_public_inputs` argument.
pub struct Fp8PublicData {
    pub blake3: Blake3Program,
    pub input_quant: InputQuantProgram,
    pub scale: ScaleProgram,
    pub xor_fold: XorFoldProgram,
    /// Raw byte lengths of the four opened strip planes (A/B values, A/B scales) — the
    /// Blake3 class (a) schedule inputs. The MoE routing statement needs no length here:
    /// its public part is the pin schedule riding [`Blake3Program::routing_pins`], and the
    /// opened hotspot blocks themselves are witness data ([`Fp8Witness::routing_words`]).
    pub a_values_len: usize,
    pub a_scales_len: usize,
    pub b_values_len: usize,
    pub b_scales_len: usize,
    /// The job's bf16 noise codes, natively recomputed from the public seeds
    /// (`(E @ F)[elem]`), A then B element order.
    pub a_noise: Vec<u16>,
    pub b_noise: Vec<u16>,
}

/// Verifier-recomputable leading columns for the five main tables. This is only an
/// intermediate representation: [`Fp8System::assemble_known_columns`] is the sole place
/// that converts it to the positional layout the batch verifier requires.
struct MainTableKnownValues<F: RichField> {
    blake3: Vec<PolynomialValues<F>>,
    input_quant: Vec<PolynomialValues<F>>,
    scale: Vec<PolynomialValues<F>>,
    matmul: Vec<PolynomialValues<F>>,
    xor_fold: Vec<PolynomialValues<F>>,
}

impl<F: RichField> MainTableKnownValues<F> {
    fn into_batch_known_columns(self, device: Device) -> BatchKnownColumns<F> {
        let values_per_table = vec![self.blake3, self.input_quant, self.scale, self.matmul, self.xor_fold];
        for table in Table::ALL {
            let columns = &values_per_table[table as usize];
            assert_eq!(
                columns.len(),
                table.known_column_count(device),
                "{table:?}: wrong known-column count"
            );
            // Matmul has no known columns to shape-check.
            if let Some(first) = columns.first() {
                for column in columns {
                    assert_eq!(column.len(), first.len(), "{table:?}: ragged known columns");
                }
                assert!(
                    first.len().is_power_of_two(),
                    "{table:?}: known columns do not have power-of-two height"
                );
            }
        }

        let mut columns_per_table = Table::ALL
            .into_iter()
            .map(|table| (0..table.known_column_count(device)).collect())
            .collect::<Vec<Vec<usize>>>();
        columns_per_table.resize(NUM_ALL_TABLES, Vec::new());
        let mut values_per_table = values_per_table;
        values_per_table.resize(NUM_ALL_TABLES, Vec::new());
        BatchKnownColumns {
            digest: None,
            columns_per_table,
            values_per_table,
        }
    }
}

/// Reduces a statement digest into four Goldilocks elements (little-endian integer mod `p^4`).
pub(super) fn statement_digest_to_hash_out<F: RichField>(digest: Hash256) -> HashOut<F> {
    let p = U256::from(F::ORDER);
    let mut value = U256::from_little_endian(&digest);
    let mut elements = [F::ZERO; 4];
    for element in &mut elements {
        *element = F::from_canonical_u64((value % p).as_u64());
        value /= p;
    }
    HashOut { elements }
}

/// The device-specific Matmul AIR of one FP8 statement. It owns the geometry the AIR
/// constrains; callers do not maintain a second dimension-only program alongside it.
enum MatmulBackend<F: RichField + Extendable<D>, const D: usize> {
    H100(MatmulStarkH100<F, D>),
    B200(MatmulStarkB200<F, D>),
}

impl<F: RichField + Extendable<D>, const D: usize> MatmulBackend<F, D> {
    fn new(device: Device, h: usize, w: usize, k: usize) -> Self {
        match device {
            Device::H100 => Self::H100(MatmulStarkH100::new(h, w, k)),
            Device::B200 => Self::B200(MatmulStarkB200::new(h, w, k)),
        }
    }

    /// Geometry-derived trace height, including power-of-two padding.
    fn num_rows(&self) -> usize {
        match self {
            Self::H100(stark) => stark.num_rows(),
            Self::B200(stark) => stark.num_rows(),
        }
    }

    /// The selected AIR's trace width (the joined fill's column destination count).
    fn num_columns(&self) -> usize {
        match self {
            Self::H100(_) => super::matmul_h100::columns::NUM_MATMUL_COLUMNS,
            Self::B200(_) => super::matmul_b200_stark::columns::NUM_MATMUL_B200_COLUMNS,
        }
    }

    /// The geometry public inputs of the selected AIR.
    fn public_inputs(&self) -> [F; NUM_MATMUL_PUBLIC_INPUTS] {
        match self {
            Self::H100(stark) => stark.public_inputs(),
            Self::B200(stark) => stark.public_inputs(),
        }
    }

    /// Borrows the selected AIR at the heterogeneous batch boundary.
    ///
    /// The returned trait object is read by native batch proving and verification, and
    /// while compiling the recursive wrapper. It neither constructs a trace nor selects
    /// an AIR from proof-controlled data.
    fn as_stark(&self) -> &dyn BatchStark<F, D> {
        match self {
            Self::H100(stark) => stark,
            Self::B200(stark) => stark,
        }
    }

    /// The joined-schedule trace fill: [`Self::generate_trace`]'s trace written
    /// column-major straight into the caller's zeroed column buffers, with the cell
    /// results (`cell_words`, `cell_skips`) returned by the cell workers instead of read
    /// back out of the trace. Only the H100 backend has the direct fill; B200 falls
    /// back to the row fill + transpose.
    fn fill_columns(
        &self,
        a_codes: &[u8],
        b_codes: &[u8],
        a_lambdas: &[u64],
        b_lambdas: &[u64],
        cols: &mut [&mut [F]],
    ) -> (Vec<u32>, Vec<u64>) {
        match self {
            Self::H100(stark) => stark.generate_columns(a_codes, b_codes, a_lambdas, b_lambdas, cols),
            Self::B200(stark) => {
                let (rows, _) = stark.generate_trace(a_codes, b_codes, a_lambdas, b_lambdas);
                let trace = column_major(&rows);
                // Scatter the transposed columns into the destination buffers (identical
                // geometry on both backends).
                for (dst, src) in cols.iter_mut().zip(trace) {
                    dst.copy_from_slice(&src.values);
                }
                let mut cell_words = vec![0u32; stark.h * stark.w];
                let mut cell_skips = vec![0u64; stark.h * stark.w];
                for row in &rows {
                    let value: &MatmulB200ColumnsView<F> = row.borrow();
                    if value.is_cell_final == F::ONE && value.is_padding == F::ZERO {
                        let cell =
                            value.out_row.to_canonical_u64() as usize * stark.w + value.out_col.to_canonical_u64() as usize;
                        cell_words[cell] = (value.cell_result_f32_lo.to_canonical_u64()
                            | (value.cell_result_f32_hi.to_canonical_u64() << 16))
                            as u32;
                        cell_skips[cell] = value.cell_skips.to_canonical_u64();
                    }
                }
                (cell_words, cell_skips)
            }
        }
    }
}

/// One job's fully derived proving/verifying context. See the module docs.
pub struct Fp8System<F: RichField + Extendable<D>, const D: usize> {
    blake3: Blake3Stark<F, D>,
    input_quant: InputQuantStark<F, D>,
    scale: ScaleStark<F, D>,
    matmul: MatmulBackend<F, D>,
    xor_fold: XorFoldStark<F, D>,
    lut_tables: [super::luts::LutTable; NUM_LUT_TABLES],
    /// The device's LUT AIRs at their exact widths, in committed order (boxed: each table
    /// instantiates its own width).
    luts: [Box<dyn BatchStark<F, D>>; NUM_LUT_TABLES],
    /// Class (a) recompute inputs kept for trace generation and witness validation.
    plane_lens: [usize; 4],
    a_noise: Vec<u16>,
    b_noise: Vec<u16>,
    /// Per-table trace degree bits in the canonical batch order (`Table` 0..4, then
    /// the device's committed LUT inventory) — not height-sorted; batched FRI sorts the distinct
    /// heights internally.
    degree_bits: [usize; NUM_ALL_TABLES],
    /// The consensus config for this job's degree profile.
    config: StarkConfig,
    /// Class (a) columns in canonical table order. The digest slot is empty until
    /// `bind_statement_digest` fills it.
    known: BatchKnownColumns<F>,
    /// All 23 CTL channels, table indices in canonical order.
    ctls: Vec<CrossTableLookup<F>>,
}

impl<F: RichField + Extendable<D>, const D: usize> Fp8System<F, D> {
    /// Derives the five main tables' verifier-known columns from a statement. This is
    /// the only cross-table owner of their canonical batch order and metadata.
    pub(super) fn derive_known_columns(data: &Fp8PublicData) -> BatchKnownColumns<F> {
        let ((blake3, input_quant), (scale, xor_fold)) = plonky2_maybe_rayon::join(
            || {
                plonky2_maybe_rayon::join(
                    || {
                        data.blake3.known_values::<F>(&Blake3KnownInputs {
                            a_values_len: data.a_values_len,
                            a_scales_len: data.a_scales_len,
                            b_values_len: data.b_values_len,
                            b_scales_len: data.b_scales_len,
                        })
                    },
                    || data.input_quant.known_values::<F>(&data.a_noise, &data.b_noise),
                )
            },
            || plonky2_maybe_rayon::join(|| data.scale.known_values::<F>(), || data.xor_fold.known_values::<F>()),
        );
        Self::assemble_known_columns(data, blake3, input_quant, scale, xor_fold)
    }

    fn assemble_known_columns(
        data: &Fp8PublicData,
        blake3: Vec<PolynomialValues<F>>,
        input_quant: Vec<PolynomialValues<F>>,
        scale: Vec<PolynomialValues<F>>,
        xor_fold: Vec<PolynomialValues<F>>,
    ) -> BatchKnownColumns<F> {
        debug_assert_eq!(
            data.input_quant.device, data.scale.device,
            "InputQuant and Scale must use the same committed device"
        );
        let device = data.scale.device;
        MainTableKnownValues {
            blake3,
            input_quant,
            scale,
            matmul: Vec::new(),
            xor_fold,
        }
        .into_batch_known_columns(device)
    }

    /// Derives the full batching context from the statement: class (a) values, per-table
    /// heights (each table's program pins its own), the consensus config, the CTL set and
    /// the known-column data — all in the canonical table order.
    pub fn new(data: Fp8PublicData) -> Self {
        let known = Self::derive_known_columns(&data);
        Self::with_known_columns(data, known)
    }

    /// Reuses columns generated alongside the programs; canonical layout and shape
    /// checks remain owned by this driver.
    pub(crate) fn new_with_known_values(
        data: Fp8PublicData,
        blake3: Vec<PolynomialValues<F>>,
        input_quant: Vec<PolynomialValues<F>>,
        scale: Vec<PolynomialValues<F>>,
        xor_fold: Vec<PolynomialValues<F>>,
    ) -> Self {
        let known = Self::assemble_known_columns(&data, blake3, input_quant, scale, xor_fold);
        Self::with_known_columns(data, known)
    }

    fn with_known_columns(data: Fp8PublicData, known: BatchKnownColumns<F>) -> Self {
        let Fp8PublicData {
            blake3,
            input_quant,
            scale,
            xor_fold,
            a_values_len,
            a_scales_len,
            b_values_len,
            b_scales_len,
            a_noise,
            b_noise,
        } = data;
        debug_assert_eq!(
            input_quant.device, scale.device,
            "InputQuant and Scale must use the same committed device"
        );
        let device = scale.device;
        let (h, w, k) = (input_quant.h, input_quant.w, input_quant.k);
        let lut_tables = lut_tables(device);
        let matmul = MatmulBackend::new(device, h, w, k);
        // Heights come uniformly from each table's statement-derived program
        // (power-of-two padded), never from the proof. Known columns, when
        // present, are generated at exactly these heights.
        let mut heights = [0usize; NUM_ALL_TABLES];
        heights[Table::Blake3 as usize] = blake3.num_rows();
        heights[Table::InputQuant as usize] = input_quant.num_rows();
        heights[Table::Scale as usize] = scale.num_rows();
        heights[Table::Matmul as usize] = matmul.num_rows();
        heights[Table::XorFold as usize] = xor_fold.num_rows();
        for table in Table::ALL {
            debug_assert!(
                known.values_per_table[table as usize]
                    .iter()
                    .all(|column| column.len() == heights[table as usize]),
                "{table:?} known columns must be generated at the table height"
            );
        }
        for (i, &table) in lut_tables.iter().enumerate() {
            heights[NUM_TABLES + i] = lut_height(table);
        }

        // The batch order is the canonical table order itself — profile-independent, so
        // the Fiat-Shamir cap/opening order, the statement digest and the CTL indices
        // are the same for every job; batched FRI folds by height internally.
        let degree_bits: [usize; NUM_ALL_TABLES] = core::array::from_fn(|t| log2_strict(heights[t]));
        let config = fp8_stark_config(&degree_bits);

        // Every channel in the same canonical index space.
        let ctls = all_cross_table_lookups::<F>(device, &scale);

        Self {
            blake3: Blake3Stark::new(blake3),
            input_quant: InputQuantStark::new(input_quant),
            scale: ScaleStark::new(scale),
            matmul,
            xor_fold: XorFoldStark::new(xor_fold),
            luts: lut_tables.map(boxed_lut_stark::<F, D>),
            lut_tables,
            plane_lens: [a_values_len, a_scales_len, b_values_len, b_scales_len],
            a_noise,
            b_noise,
            degree_bits,
            config,
            known,
            ctls,
        }
    }

    /// The consensus config of this job's degree profile.
    pub fn config(&self) -> &StarkConfig {
        &self.config
    }

    /// Per-table trace degree bits in the canonical batch order (`Table` 0..4, then
    /// the device's committed LUT inventory) — the expected `proof.degree_bits`, and the
    /// shape identity of any compiled-circuit cache key: per-table assignments, not the
    /// height multiset, so two statements sharing a multiset cannot collide.
    pub fn degree_bits(&self) -> &[usize; NUM_ALL_TABLES] {
        &self.degree_bits
    }

    pub(crate) fn device(&self) -> Device {
        self.scale.program.device
    }

    /// The Blake3 witness bounds of this statement's schedule — how many auxiliary
    /// 64-byte messages and auxiliary sibling CVs an honest witness must supply
    /// (the unopened subtrees of the operand forests). Prover-side sizing input.
    pub fn blake3_aux_bounds(&self) -> (usize, usize) {
        (self.blake3.program.num_aux_msgs, self.blake3.program.num_aux_cvs)
    }

    #[cfg(test)]
    pub fn matmul_num_columns(&self) -> usize {
        self.matmul.as_stark().num_columns()
    }

    /// The batch positions of the sixteen device-specific LUT tables
    /// order. The batch order is canonical, so LUT `i` sits at `NUM_TABLES + i`.
    pub fn lut_positions(&self) -> [usize; NUM_LUT_TABLES] {
        core::array::from_fn(|i| NUM_TABLES + i)
    }

    pub fn committed_lut_tables(&self) -> &[super::luts::LutTable; NUM_LUT_TABLES] {
        &self.lut_tables
    }

    /// The batch positions of the five main tables, in canonical `Table` order — where a
    /// caller finds each table's public inputs inside `proof.public_inputs`. The batch
    /// order is canonical, so table `t` sits at `t`.
    pub fn main_table_positions(&self) -> [usize; NUM_TABLES] {
        core::array::from_fn(|t| t)
    }

    /// The setup-time commitment to the precommitted LUT columns,
    /// placed at this job's batch positions. The underlying flat column list (and
    /// hence the Merkle cap) is job-independent — the LUTs keep their
    /// relative order under any geometry — so the cap is a consensus constant;
    /// only the position bookkeeping varies per job.
    pub fn preprocessed_data<C: GenericConfig<D, F = F>>(&self, timing: &mut TimingTree) -> BatchStarkPreprocessedData<F, C, D> {
        lut_preprocessed_data::<F, C, D>(NUM_ALL_TABLES, self.lut_positions(), &self.lut_tables, &self.config, timing)
    }

    /// The verifier's view of the LUT precommitment: the consensus `cap` plus the
    /// per-table preprocessed column indices, which the statement pins itself (a caller
    /// cannot smuggle in a different column split).
    pub fn preprocessed_verifier_data<C: GenericConfig<D, F = F>>(
        &self,
        cap: &MerkleCap<F, C::Hasher>,
    ) -> BatchStarkPreprocessedVerifierData<F, C, D> {
        let mut columns_per_table = vec![Vec::new(); NUM_ALL_TABLES];
        for (i, &table) in self.lut_tables.iter().enumerate() {
            columns_per_table[NUM_TABLES + i] = (0..num_precommitted_columns(table)).collect();
        }
        BatchStarkPreprocessedVerifierData {
            cap: cap.clone(),
            columns_per_table,
        }
    }

    /// The all-channel CTL set, table indices in canonical order (for the recursive
    /// wrapper, which re-runs the batch verifier in-circuit).
    pub(super) fn ctls(&self) -> &[CrossTableLookup<F>] {
        &self.ctls
    }

    /// The class (a) known columns in canonical table order (for the recursive wrapper:
    /// column indices shape the circuit; the values feed the native evaluation-at-zeta
    /// recompute). The digest slot is empty until `bind_statement_digest` fills it.
    pub(super) fn known(&self) -> &BatchKnownColumns<F> {
        &self.known
    }

    pub(super) fn has_moe(&self) -> bool {
        self.blake3.program.moe.is_some()
    }

    /// Binds `statement_digest` into the known-column Fiat-Shamir slot. The caller of
    /// [`Self::verify`] must invoke this (with the digest of the statement it will verify
    /// against) before the verification: `verify` fails closed on an unbound slot.
    /// [`Self::prove`] performs the bind itself.
    pub fn bind_statement_digest(&mut self, statement_digest: Hash256) {
        self.known.digest = Some(statement_digest_to_hash_out(statement_digest));
    }

    /// Cross-checks a caller-supplied `statement_digest` (the wrapper paths take it
    /// as a parameter) against the one bound into this system, if any: they must
    /// agree — a mismatch means system and digest came from different statements.
    pub fn ensure_statement_digest_binding(&self, statement_digest: Hash256) -> Result<()> {
        ensure!(
            self.known
                .digest
                .is_none_or(|bound| bound == statement_digest_to_hash_out(statement_digest)),
            "the statement_digest disagrees with the one bound into the system's known columns"
        );
        Ok(())
    }

    /// The batch tables, canonical order.
    pub(super) fn batch_starks(&self) -> [&dyn BatchStark<F, D>; NUM_ALL_TABLES] {
        let main_tables: [&dyn BatchStark<F, D>; NUM_TABLES] = [
            &self.blake3,
            &self.input_quant,
            &self.scale,
            self.matmul.as_stark(),
            &self.xor_fold,
        ];
        core::array::from_fn(|t| {
            if t < NUM_TABLES {
                main_tables[t]
            } else {
                self.luts[t - NUM_TABLES].as_ref()
            }
        })
    }

    /// The main tables' public inputs, padded with the LUT tables' empty vectors to the
    /// canonical batch shape.
    pub(super) fn batch_public_inputs(&self, main_pis: &[Vec<F>; NUM_TABLES]) -> [Vec<F>; NUM_ALL_TABLES] {
        core::array::from_fn(|t| if t < NUM_TABLES { main_pis[t].clone() } else { Vec::new() })
    }

    /// Proves one job: regenerates the five main traces from the witness (chaining each
    /// table's inputs from its producer), accumulates the committed-LUT multiplicities,
    /// assembles the LUT traces and runs the batch prover. `preprocessed` is the
    /// setup-time LUT commitment for this job's layout ([`Self::preprocessed_data`]).
    ///
    /// `derive_statement_digest` receives the lottery digest `J` (the 32-byte
    /// jackpot hash generated with the traces -- the block's shipped claim) and returns the
    /// Fiat-Shamir salt to absorb into the known-column slot.
    pub fn prove<C: GenericConfig<D, F = F>>(
        &mut self,
        witness: &Fp8Witness<'_>,
        derive_statement_digest: impl FnOnce(Hash256) -> Hash256,
        preprocessed: &BatchStarkPreprocessedData<F, C, D>,
        timing: &mut TimingTree,
    ) -> Result<BatchStarkProofWithPublicInputs<F, C, D>> {
        let [a_values_len, a_scales_len, b_values_len, b_scales_len] = self.plane_lens;
        ensure!(
            witness.a_values.len() == a_values_len
                && witness.a_scales.len() == a_scales_len
                && witness.b_values.len() == b_values_len
                && witness.b_scales.len() == b_scales_len,
            "witness plane lengths differ from the statement's"
        );

        // ---- Native InputQuant advice: per-group side witnesses plus the compact
        // per-side outputs (noised fp8 codes, summand scores, per-matrix-row scale
        // tuples) the downstream tables consume — one fused parallel build, shared
        // by `Arc` across the dead padding groups. The witness byte/code
        // conversions stream hundreds of MiB and are pure maps: run all four
        // concurrently on the pool instead of four serial passes. ----
        let (a_int8, b_int8, a_scale_codes, b_scale_codes) = {
            timing.push("trace: witness convert", log::Level::Debug);
            let as_int8 = |bytes: &[u8]| -> Vec<i8> {
                use plonky2_maybe_rayon::{IndexedParallelIterator, MaybeParChunks, MaybeParChunksMut, ParallelIterator};
                let mut out = vec![0i8; bytes.len()];
                out.par_chunks_mut(1 << 20)
                    .zip(bytes.par_chunks(1 << 20))
                    .for_each(|(o, i)| {
                        for (o, &b) in o.iter_mut().zip(i) {
                            *o = b as i8;
                        }
                    });
                out
            };
            let le_codes = |bytes: &[u8]| -> Vec<u16> {
                use plonky2_maybe_rayon::{IndexedParallelIterator, MaybeParChunks, MaybeParChunksMut, ParallelIterator};
                let n = bytes.len() / 2;
                let mut out = vec![0u16; n];
                out.par_chunks_mut(1 << 20)
                    .zip(bytes.par_chunks(1 << 21))
                    .for_each(|(o, i)| {
                        for (o, p) in o.iter_mut().zip(i.chunks(2)) {
                            *o = u16::from_le_bytes([p[0], p[1]]);
                        }
                    });
                out
            };
            let (a_int8, b_int8) = plonky2_maybe_rayon::join(
                || as_int8(witness.a_values),
                || as_int8(witness.b_values),
            );
            let (a_scale_codes, b_scale_codes) = plonky2_maybe_rayon::join(
                || le_codes(witness.a_scales),
                || le_codes(witness.b_scales),
            );
            timing.pop();
            (a_int8, b_int8, a_scale_codes, b_scale_codes)
        };
        let iq_native = timed!(
            timing,
            "trace: input_quant",
            self.input_quant
                .program
                .build_native(&a_int8, &a_scale_codes, &self.a_noise, &b_int8, &b_scale_codes, &self.b_noise,)
        );

        // ---- Joined trace generation. Given the native advice, the fills are
        // independent (Scale's tuples and Matmul's codes/lambdas come from `iq_native`,
        // not from the IQ trace) except the Matmul -> XorFold -> Blake3 chain: XorFold
        // folds Matmul's cell results and Blake3's lottery block is XorFold's lane
        // outputs. Two branches run overlapped across the rayon pool — (a) the IQ
        // column fill and (b) the matmul column fill feeding XorFold and Blake3 in
        // sequence — with Scale's fill (sub-millisecond) absorbing pool slack.
        // Every column destination is a kernel-zeroed allocation: `alloc_zeroed`
        // maps lazy zero pages, so no serial memset stands ahead of the fills and
        // the padding rows need no explicit zeroing pass. ----
        let iq_num_rows = self.input_quant.program.num_rows();
        let mm_num_rows = self.matmul.num_rows();
        let mut iq_cols = calloc_columns::<F>(super::input_quant_stark::columns::NUM_INPUT_QUANT_COLUMNS, iq_num_rows);
        let mut mm_cols = calloc_columns::<F>(self.matmul.num_columns(), mm_num_rows);

        timing.push("trace: joined-fill", log::Level::Debug);
        let ((iq_pis,), (_, (scale_rows, scale_pis, xf_rows, xf_pis, b3_rows, b3_pis))) = {
            // Column-slice handles: disjoint mutable borrows over the calloc'd columns,
            // handed once to each fill (the fills write only disjoint rows).
            let mut iq_slices: Vec<&mut [F]> = iq_cols.iter_mut().map(|c| c.as_mut_slice()).collect();
            let mut mm_slices: Vec<&mut [F]> = mm_cols.iter_mut().map(|c| c.as_mut_slice()).collect();
            plonky2_maybe_rayon::join(
                || {
                    self.input_quant.program.fill_columns::<F>(&iq_native, &mut iq_slices);
                    (self.input_quant.program.public_inputs(),)
                },
                || {
                    let ((cell_words, cell_skips), (scale_rows, scale_pis)) = plonky2_maybe_rayon::join(
                        || {
                            self.matmul.fill_columns(
                                &iq_native.a_codes,
                                &iq_native.b_codes,
                                &iq_native.a_lambdas,
                                &iq_native.b_lambdas,
                                &mut mm_slices,
                            )
                        },
                        || {
                            self.scale
                                .program
                                .generate_trace::<F>(&iq_native.a_tuples, &iq_native.b_tuples)
                        },
                    );
                    // ---- XorFold: fold Matmul's finished cell words into the lottery
                    // lanes, returning the lane-final lottery words for Blake3's
                    // jackpot block (same fold, no trace read-back). ----
                    let (xf_rows, lottery_words) = self.xor_fold.program.generate_rows_and_lottery::<F>(&cell_words, &cell_skips);
                    let xf_pis = [F::from_canonical_u64(self.xor_fold.program.skip_limit)];
                    // ---- Blake3: the strip bytes and the folded lottery words under the
                    // job schedule (native Phase A, then the parallel row fill). ----
                    let inputs = Blake3TraceInputs {
                        a_values: witness.a_values,
                        a_scales: witness.a_scales,
                        b_values: witness.b_values,
                        b_scales: witness.b_scales,
                        routing_words: witness.routing_words,
                        offsets_words: witness.offsets_words,
                        aux_msgs: witness.aux_msgs,
                        aux_cvs: witness.aux_cvs,
                        lottery_words,
                        key_a: witness.key_a,
                        key_b: witness.key_b,
                        jackpot_key: witness.jackpot_key,
                        a_hash_id: witness.a_hash_id,
                        b_hash_id: witness.b_hash_id,
                        routing_hash_id: witness.routing_hash_id,
                        offsets_hash_id: witness.offsets_hash_id,
                    };
                    let schedule = self.blake3.program.native_schedule(&inputs);
                    let mut b3_rows = calloc_rows::<F, NUM_BLAKE3_COLUMNS>(self.blake3.program.num_rows());
                    let b3_pis = self.blake3.program.parallel_fill::<F>(&inputs, &schedule, &mut b3_rows);
                    (
                        (cell_words, cell_skips),
                        (scale_rows, scale_pis, xf_rows, xf_pis, b3_rows, b3_pis),
                    )
                },
            )
        };
        timing.pop();
        let mat_pis = self.matmul.public_inputs().to_vec();

        // The proof ships the public inputs the witness generates (keys, roots, the geometry
        // and dead-limit slots, the lottery digest); the verifier pins every slot to its own
        // expectations in [`Self::verify`].
        let generated: [Vec<F>; NUM_TABLES] = [b3_pis.to_vec(), iq_pis.to_vec(), scale_pis.to_vec(), mat_pis, xf_pis.to_vec()];

        // ---- Column-major traces, canonical order. The IQ and Matmul traces' columns
        // were filled in place; wrap the calloc'd vectors without a transpose pass. ----
        timing.push("trace: to-column-major", log::Level::Debug);
        let iq_trace: Vec<PolynomialValues<F>> = iq_cols.into_iter().map(PolynomialValues::new).collect();
        let mat_trace: Vec<PolynomialValues<F>> = mm_cols.into_iter().map(PolynomialValues::new).collect();
        let mut traces: Vec<Vec<PolynomialValues<F>>> = vec![
            column_major(&b3_rows),
            iq_trace,
            column_major(&scale_rows),
            mat_trace,
            column_major(&xf_rows),
        ];
        timing.pop();

        // ---- Committed-LUT multiplicities: walk every AIR's inventory over its trace,
        // accumulating counts only — the serving validation is a debug/test-side
        // concern ([`LutChecker::check_trace`], exercised by the consistency
        // driver) and never changes the committed counts. The five walks run
        // overlapped (one job per table, each internally row-parallel); each
        // job owns its counters, and the results merge exactly (u64 addition
        // commutes). ----
        let inventories = lut_inventories::<F>(self.device(), &self.scale.program);
        {
            let label = "trace: lut-check".to_string();
            timing.push(&label, log::Level::Debug);
            let mut counters: Vec<LutChecker<F>> = inventories
                .iter()
                .map(|(_, lookups)| {
                    let mut tables: Vec<_> = lookups.iter().map(|l| l.table).collect();
                    tables.sort_unstable();
                    tables.dedup();
                    LutChecker::<F>::new(&tables)
                })
                .collect::<Vec<_>>();
            {
                use plonky2_maybe_rayon::{IndexedParallelIterator as _, MaybeParIterMut as _, ParallelIterator as _};
                counters
                    .par_iter_mut()
                    .zip(&inventories)
                    .try_for_each(|(counter, (table, lookups))| {
                        let t = usize::from(*table);
                        counter.count_trace(lookups, &traces[t], &generated[t], &format!("{table:?}"))
                    })
                    .map_err(|e: String| anyhow!(e))?;
            }
            let mut checker = LutChecker::<F>::new(&self.lut_tables);
            for counter in &counters {
                checker.multiplicities.merge(&counter.multiplicities);
            }
            for table in self.lut_tables {
                let label = format!("trace: lut-trace {table:?}");
                timing.push(&label, log::Level::Debug);
                let trace = lut_trace::<F>(table, checker.multiplicities.table_columns(table));
                timing.pop();
                traces.push(trace);
            }
            timing.pop();
        }

        // ---- The traces were assembled in canonical order, which is the batch order. ----
        let batch_traces: [Vec<PolynomialValues<F>>; NUM_ALL_TABLES] = traces.try_into().map_err(|_| anyhow!("table count"))?;
        for (t, trace) in batch_traces.iter().enumerate() {
            ensure!(
                log2_strict(trace[0].len()) == self.degree_bits[t],
                "batch table {t}: trace height differs from the statement's"
            );
        }

        // Bind the salt after the traces exist (`J` is in `generated`), before absorbing.
        let statement_digest = derive_statement_digest(hash_jackpot(&generated));
        self.bind_statement_digest(statement_digest);
        batch_prove::<F, C, D, NUM_ALL_TABLES>(
            &self.batch_starks(),
            &self.config,
            batch_traces,
            &self.batch_public_inputs(&generated),
            &self.ctls,
            &FP8_GROUPED_TABLES,
            true,
            Some(preprocessed),
            Some(&self.known),
            timing,
        )
    }

    /// Verifies one job's proof against this statement, the caller's expected public inputs
    /// (canonical `Table` order), the consensus LUT cap, and `statement_digest`
    /// (bound into the known-column Fiat-Shamir slot): the degree profile and every public
    /// input slot must equal the expectation, and the batch verifier then checks every
    /// constraint, the class (a) openings against the statement's recomputed values, the
    /// CTL balances and the batched FRI argument.
    pub fn verify<C: GenericConfig<D, F = F>>(
        &self,
        proof: &BatchStarkProofWithPublicInputs<F, C, D>,
        expected_public_inputs: &[Vec<F>; NUM_TABLES],
        lut_cap: &MerkleCap<F, C::Hasher>,
    ) -> Result<()> {
        for (t, expected) in [
            NUM_BLAKE3_PUBLIC_INPUTS,
            NUM_INPUT_QUANT_PUBLIC_INPUTS,
            NUM_SCALE_PUBLIC_INPUTS,
            NUM_MATMUL_PUBLIC_INPUTS,
            NUM_XOR_FOLD_PUBLIC_INPUTS,
        ]
        .into_iter()
        .enumerate()
        {
            ensure!(
                expected_public_inputs[t].len() == expected,
                "table {t}: wrong expected public input count"
            );
        }
        ensure!(
            proof.proof.degree_bits == self.degree_bits,
            "the proof's degree profile differs from the statement's"
        );
        ensure!(
            proof.public_inputs == self.batch_public_inputs(expected_public_inputs),
            "the proof's public inputs differ from the expected ones"
        );
        ensure!(
            self.known.digest.is_some(),
            "the statement_digest is unbound: call bind_statement_digest before verify"
        );
        batch_verify::<F, C, D, NUM_ALL_TABLES>(
            &self.batch_starks(),
            &self.config,
            proof,
            &self.ctls,
            &FP8_GROUPED_TABLES,
            true,
            Some(&self.preprocessed_verifier_data::<C>(lut_cap)),
            Some(&self.known),
            &Default::default(),
        )
    }
}

/// The lottery digest `J` from Blake3's witness-generated public inputs (`PI_HASH_JACKPOT`).
fn hash_jackpot<F: RichField>(public_inputs: &[Vec<F>; NUM_TABLES]) -> Hash256 {
    let blake3 = &public_inputs[Table::Blake3 as usize];
    let limbs: &[F; 8] = blake3[PI_HASH_JACKPOT..PI_HASH_JACKPOT + 8].try_into().unwrap();
    u32_field_array_to_hash(limbs)
}

/// The private witness of one job: the opened strip planes (raw bytes, exactly the
/// statement's plane lengths) and Blake3's auxiliary compression inputs.
pub struct Fp8Witness<'a> {
    pub a_values: &'a [u8],
    pub a_scales: &'a [u8],
    pub b_values: &'a [u8],
    pub b_scales: &'a [u8],
    /// The opened MoE routing hotspot blocks as u32 words (16 per block, hotspot order;
    /// empty for a dense job). Witness data: the statement pins only the sampled entries
    /// ([`Blake3Program::routing_pins`]); the neighbor words are bound by the hash chain to
    /// `HASH_ROUTING`.
    pub routing_words: &'a [u32],
    /// The full MoE offsets list `O` as u32 words (chunk-padded; empty for a dense job),
    /// bound to `HASH_OFFSETS` with the public scalars pinned in-circuit.
    pub offsets_words: &'a [u32],
    /// Auxiliary 64-byte messages (unopened blocks) of the Blake3 forest.
    pub aux_msgs: &'a [[u8; 64]],
    /// Auxiliary chaining values of the Blake3 forest (32 raw bytes each).
    pub aux_cvs: &'a [[u8; 32]],
    pub key_a: [u32; 8],
    pub key_b: [u32; 8],
    pub jackpot_key: [u32; 8],
    /// Merkle leaf sizes from [`PublicParams`] (not stored on [`Blake3Program`]).
    pub a_hash_id: HashId,
    pub b_hash_id: HashId,
    pub routing_hash_id: HashId,
    pub offsets_hash_id: HashId,
}

/// Row-major trace rows -> the column-major `PolynomialValues` layout of the batch prover.
///
/// Parallel over blocks of 8 adjacent columns: every 64-byte source line serves exactly one
/// block's job, so the whole matrix streams from DRAM once (the naive per-column pass reads
/// it once per column). Column buffers are written through their `MaybeUninit` spare
/// capacity — each `(column, row)` slot is written exactly once before the length is set.
///
/// Parallel jobs are ROW tiles: each job streams its tile of source rows
/// once, fully using every cache line (the column-blocked variant rereads
/// the whole matrix once per 8-column block — for Blake3's 452 columns
/// that is ~57 full re-streams). Column writes scatter to `N` buffers,
/// one 8-byte element per (row, column), written exactly once via raw
/// pointers into each column's spare capacity; disjoint across tiles.
fn column_major<F: RichField, const N: usize>(rows: &[[F; N]]) -> Vec<PolynomialValues<F>> {
    /// Rows per job, tuned so the tile streams ~1 MiB of source rows
    /// (L2-resident through the job's lifetime) and a deg-22 table
    /// produces hundreds of jobs.
    const ROW_TILE: usize = 4096;

    let num_rows = rows.len();
    let mut cols: Vec<Vec<F>> = (0..N).map(|_| Vec::with_capacity(num_rows)).collect();
    {
        // SAFETY wrapper: tiles write disjoint (row) regions of each
        // column's spare capacity; `cols` outlives the parallel scope and
        // is not read until after the join + set_len.
        use plonky2_maybe_rayon::{MaybeIntoParIter as _, ParallelIterator as _};
        struct ColPtr<F>(*mut F);
        unsafe impl<F> Send for ColPtr<F> {}
        unsafe impl<F> Sync for ColPtr<F> {}
        let ptrs: Vec<ColPtr<F>> = cols.iter_mut().map(|c| ColPtr(c.as_mut_ptr())).collect();

        (0..num_rows.div_ceil(ROW_TILE))
            .into_par_iter()
            .for_each(|tile| {
                let r_lo = tile * ROW_TILE;
                let r_hi = ((tile + 1) * ROW_TILE).min(num_rows);
                for (i, row) in rows[r_lo..r_hi].iter().enumerate() {
                    let i = r_lo + i;
                    // SAFETY: the write targets column c's spare capacity
                    // at offset i < num_rows; tiles own disjoint row ranges.
                    unsafe {
                        for (c, p) in ptrs.iter().enumerate() {
                            *p.0.add(i) = row[c];
                        }
                    }
                }
            });
    }
    for col in cols.iter_mut() {
        unsafe { col.set_len(num_rows) };
    }
    cols.into_iter().map(PolynomialValues::new).collect()
}

/// One kernel-zeroed `Vec<F>` per trace column, `num_rows` long: `alloc_zeroed` maps lazy
/// zero pages, so the allocation costs no serial memset and the fills' padding rows come
/// pre-zeroed. The zeroed memory is a valid all-zero `[F]` for every supported field.
fn calloc_columns<F: RichField>(num_columns: usize, num_rows: usize) -> Vec<Vec<F>> {
    (0..num_columns)
        .map(|_| {
            let layout = std::alloc::Layout::array::<F>(num_rows).expect("column layout overflows");
            // SAFETY: `alloc_zeroed` returns a valid, aligned, zeroed block of `layout` bytes.
            let ptr = unsafe { std::alloc::alloc_zeroed(layout) } as *mut F;
            // SAFETY: `ptr` covers exactly `num_rows` initialized (zero) elements and the
            // allocation is uniquely owned here.
            unsafe { Vec::from_raw_parts(ptr, num_rows, num_rows) }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use plonky2::field::goldilocks_field::GoldilocksField;
    use plonky2::field::types::Field;
    use plonky2::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};

    use super::super::consistency::{Fp8Fixture, build_fixture, build_fixture_from_job, fixture_job};
    use super::*;
    use crate::v4::api::public_params::PublicParams;

    const D: usize = 2;
    type C = PoseidonGoldilocksConfig;
    type F = GoldilocksField;

    /// The blocked `column_major` must be value-identical to the naive per-column gather.
    #[test]
    fn column_major_matches_naive_gather() {
        const N: usize = 13;
        let rows: Vec<[F; N]> = (0..1024u32)
            .map(|i| std::array::from_fn(|c| F::from_canonical_u32(i * 31 + c as u32 * 7 + (i * c as u32) % 5)))
            .collect();
        let blocked = column_major::<F, N>(&rows);
        for c in 0..N {
            let naive: Vec<F> = rows.iter().map(|r| r[c]).collect();
            assert_eq!(blocked[c].values, naive, "column {c} diverged");
        }
    }

    /// Arbitrary 32-byte salt for driver tests that do not intercept `J`.
    const STATEMENT_DIGEST: Hash256 = [0x11; 32];

    /// `verify` must fail for the given (mutated) argument set, after binding the digest.
    fn rejects(
        system: &mut Fp8System<F, D>,
        proof: &BatchStarkProofWithPublicInputs<F, C, D>,
        expected: &[Vec<F>; NUM_TABLES],
        cap: &MerkleCap<F, <C as GenericConfig<D>>::Hasher>,
        digest: Hash256,
    ) -> bool {
        system.bind_statement_digest(digest);
        system.verify::<C>(proof, expected, cap).is_err()
    }

    /// The boundary bits the config's schedule folds through, from the profile's max down.
    fn schedule_boundaries(degree_bits: &[usize]) -> Vec<usize> {
        let config = fp8_stark_config(degree_bits);
        let max = *degree_bits.iter().max().unwrap();
        let mut reached = vec![max];
        for a in config.fri_params(max).reduction_arity_bits {
            assert!((1..=3).contains(&a), "arity steps stay <= 3 bits");
            reached.push(reached.last().unwrap() - a);
        }
        reached
    }

    #[test]
    fn fri_arities_fold_through_every_instance() {
        // Envelope-shaped profiles (max ∈ [14, 22]) and sub-envelope test geometries: the
        // schedule's partial sums must hit every table height (each instance's LDE) and
        // every ladder boundary below the profile's max, and stop at the lowest boundary —
        // `min(2^5, smallest table)`: the ladder always folds to its own bottom, so the
        // final-polynomial length is a consensus constant (every real job carries the
        // `2^5` LUT anyway; only sub-envelope test profiles can dip below it).
        for degrees in [
            vec![19, 17, 16, 13, 10, 8, 6, 5],
            vec![22, 16, 5],
            vec![17, 16, 9, 3],
            vec![12, 7, 3],
            vec![12],
        ] {
            let reached = schedule_boundaries(&degrees);
            for &d in &degrees {
                assert!(
                    reached.contains(&d),
                    "schedule misses degree {d} ({degrees:?} -> {reached:?})"
                );
            }
            let (min, max) = (*degrees.last().unwrap(), degrees[0]);
            for &bits in FP8_REACHABLE_DEGREE_BITS.iter().filter(|b| **b <= max) {
                assert!(reached.contains(&bits), "schedule skips ladder stop {bits} ({degrees:?})");
            }
            let bottom = min.min(*FP8_REACHABLE_DEGREE_BITS.last().unwrap());
            assert_eq!(*reached.last().unwrap(), bottom, "folding must stop at the lowest boundary");
        }

        // The universal-verifier property: one consensus ladder, so a shallower job's
        // schedule is exactly the suffix of a deeper one's from its own tallest table
        // (on-ladder maxima only: an off-ladder max adds its own boundary and diverges).
        let deep = schedule_boundaries(&[22, 5]);
        for max in [21, 20, 19, 18, 17, 16, 15, 14, 13] {
            let shallow = schedule_boundaries(&[max, 5]);
            let offset = deep.iter().position(|&b| b == max).unwrap();
            assert_eq!(
                shallow.as_slice(),
                &deep[offset..],
                "max {max} schedule is not a ladder suffix"
            );
        }

        // The Fiat-Shamir prerequisite: every on-ladder profile yields the *same* config
        // (same absorbed serialization), whatever its height span.
        let consensus = fp8_stark_config(&FP8_REACHABLE_DEGREE_BITS);
        for degrees in [vec![22, 16, 5], vec![14, 5], vec![17, 16, 10, 8, 6, 5]] {
            let config = fp8_stark_config(&degrees);
            assert_eq!(
                config.fri_config.reduction_strategy, consensus.fri_config.reduction_strategy,
                "on-ladder profile {degrees:?} must produce the consensus strategy"
            );
        }
    }

    /// The consensus parameters meet [`FP8_V2_SECURITY_BITS`]: the batch prove/verify path
    /// never calls [`StarkConfig::check_config`] at runtime, so enforce it here.
    #[test]
    fn consensus_config_meets_the_security_target() {
        fp8_stark_config(&FP8_REACHABLE_DEGREE_BITS)
            .check_config::<F, D>()
            .expect("consensus FRI parameters must meet the security target");
    }

    /// Construct the full consensus envelope, including independent main tables
    /// whose maximum degrees coincide. No geometry-specific circuit is selected.
    #[test]
    fn universal_envelopes_build_for_both_devices() {
        use hashbrown::HashMap;
        use plonky2::plonk::circuit_builder::CircuitBuilder;
        use starky::batch_recursive_verifier::BatchKnownColumnsTarget;
        use starky::batch_universal::verify_universal_batch_stark_proof_circuit;

        for device in [Device::H100, Device::B200] {
            let (header, mut plain) = fixture_job();
            plain.job.common.device = device;
            let system = build_system(&build_fixture_from_job(header, plain));
            let config = fp8_stark_config(&FP8_REACHABLE_DEGREE_BITS);
            let envelope = fp8_universal_envelope(device);
            let mut builder = CircuitBuilder::<F, D>::new(super::super::circuit_utils::build_recursion_config(3, 18, 1, false));
            // Only the cap shape matters when constructing the verifier gates.
            let cap = MerkleCap(vec![HashOut::ZERO; 1 << config.fri_config.cap_height]);
            let prep = system.preprocessed_verifier_data::<C>(&cap).constant_target(&mut builder);
            let columns = &system.known().columns_per_table;
            let known = BatchKnownColumnsTarget {
                digest: Some(builder.add_virtual_hash()),
                columns_per_table: columns.clone(),
                evals_at_zeta: columns
                    .iter()
                    .map(|c| builder.add_virtual_extension_targets(c.len()))
                    .collect(),
                evals_at_g_zeta: columns
                    .iter()
                    .map(|c| builder.add_virtual_extension_targets(c.len()))
                    .collect(),
            };
            let target = verify_universal_batch_stark_proof_circuit::<F, C, D, NUM_ALL_TABLES>(
                &mut builder,
                &system.batch_starks(),
                &config,
                &envelope,
                system.ctls(),
                Some(&prep),
                Some(&known),
                &HashMap::new(),
            )
            .expect("one consensus circuit must support the entire device envelope");
            let proof = &target.proof_with_pis.proof;
            assert_eq!(proof.trace_caps.len(), 1);
            assert_eq!(proof.auxiliary_polys_caps.as_ref().unwrap().len(), 1);
            assert_eq!(proof.quotient_polys_caps.as_ref().unwrap().len(), 1);
            for (t, flags) in target.degree_flags.iter().enumerate() {
                let (lo, hi) = envelope.degree_ranges[t];
                assert_eq!(
                    flags.len(),
                    envelope.ladder.iter().filter(|&&d| (lo..=hi).contains(&d)).count()
                );
            }
            println!("{device:?} consensus verifier: {} gates before padding", builder.num_gates());
        }
    }

    /// [`FP8_REACHABLE_DEGREE_BITS`] is exactly the envelope's reach: sweep every lottery
    /// tile `(h, w)` and the job `k` range, apply each table's height formula, and compare
    /// against the ladder (with Blake3 contributing its bounded `13..=19` — see the
    /// constant's docs). Guards both directions: a formula landing off the ladder breaks
    /// batch FRI injection; a ladder stop nothing reaches taxes every proof with a dead
    /// fold layer.
    #[test]
    fn ladder_covers_the_envelope() {
        use std::collections::BTreeSet;

        // Per-main-table reach, canonical `Table` order (Blake3's bounded 13..=19 is
        // documented on the constant, not swept).
        let mut per_table: [BTreeSet<usize>; NUM_TABLES] = Default::default();
        per_table[Table::Blake3 as usize].extend(13..=19);
        // Envelope (`api::layout` tile bounds): h >= 4
        // (MIN_TILE_ROWS), w >= 16 (MIN_TILE_COLS), h*w <= 2048 (MAX_TILE_ELEMS),
        // h*w >= 16*16 (JACKPOT_ENTRIES lanes of >= MIN_SUBTILE_ELEMS), and the
        // opened-strips bound (h + w)*k <= 2^22 with k in
        // [PublicParams::MIN_K, PublicParams::MAX_K], 32 | k.
        let bits = |rows: usize| rows.next_power_of_two().trailing_zeros() as usize;
        for h in 4..=128 {
            for w in 16..=(2048 / h) {
                if h * w < 256 {
                    continue;
                }
                let k_max = ((1 << 22) / (h + w) / 32 * 32).min(PublicParams::MAX_K);
                if k_max < PublicParams::MIN_K {
                    continue;
                }
                per_table[Table::Scale as usize].insert(bits(h + w));
                per_table[Table::XorFold as usize].insert(bits(h * w));
                // InputQuant `max(h,w)*k` and Matmul `h*w*(k/32)` are monotone in k, and
                // consecutive k steps (+32 on k >= PublicParams::MIN_K) never double a value, so
                // sweeping k hits exactly the bit range between the two endpoints.
                per_table[Table::InputQuant as usize].extend(bits(h.max(w) * PublicParams::MIN_K)..=bits(h.max(w) * k_max));
                per_table[Table::Matmul as usize].extend(bits(h * w * (PublicParams::MIN_K / 32))..=bits(h * w * (k_max / 32)));
            }
        }

        // The universal envelope's per-table ranges are exactly each table's reach.
        for (t, reach) in per_table.iter().enumerate() {
            let (lo, hi) = FP8_MAIN_TABLE_DEGREE_RANGES[t];
            assert_eq!(
                (*reach.first().unwrap(), *reach.last().unwrap()),
                (lo, hi),
                "table {t}: envelope range must equal the swept reach"
            );
        }

        let mut reachable: BTreeSet<usize> = per_table.into_iter().flatten().collect();
        for device in [Device::H100, Device::B200] {
            reachable.extend(lut_tables(device).into_iter().map(|t| bits(lut_height(t))));
        }

        assert!(!reachable.contains(&12), "2^12 must stay unreachable (the ladder's gap)");
        let ladder: BTreeSet<usize> = FP8_REACHABLE_DEGREE_BITS.into_iter().collect();
        assert_eq!(reachable, ladder, "the consensus ladder must equal the envelope's reach");

        // The universal envelope is well-formed: bounds on the ladder,
        // fixed LUT slots, and the ladder headed by the tallest reachable table.
        for device in [Device::H100, Device::B200] {
            let envelope = fp8_universal_envelope(device);
            assert_eq!(envelope.ladder, FP8_REACHABLE_DEGREE_BITS.to_vec());
            assert_eq!(envelope.degree_ranges.len(), NUM_ALL_TABLES);
            for (t, &(lo, hi)) in envelope.degree_ranges.iter().enumerate() {
                assert!(lo <= hi, "table {t}");
                assert!(
                    ladder.contains(&lo) && ladder.contains(&hi),
                    "table {t}: bounds on the ladder"
                );
                assert!(t < NUM_TABLES || lo == hi, "LUT slot {t} must be fixed");
            }
        }
    }

    fn public_data(fx: &Fp8Fixture) -> Fp8PublicData {
        Fp8PublicData {
            blake3: fx.blake3.clone(),
            input_quant: fx.input_quant.clone(),
            scale: fx.scale.clone(),
            xor_fold: fx.xor_fold.clone(),
            a_values_len: fx.a_values.len(),
            a_scales_len: fx.a_scales.len(),
            b_values_len: fx.b_values.len(),
            b_scales_len: fx.b_scales.len(),
            a_noise: fx.a_noise.clone(),
            b_noise: fx.b_noise.clone(),
        }
    }

    fn build_system(fx: &Fp8Fixture) -> Fp8System<F, D> {
        Fp8System::new(public_data(fx))
    }

    #[test]
    fn admission_work_covers_derived_known_layout() {
        use crate::v4::api::prequant::BLOCK_SIZE;
        use crate::v4::api::primitives::IncompleteBlockHeader;
        use crate::v4::api::public_params::test_fixtures::{dense_params, empty_jackpot};
        use crate::v4::api::public_params::{HashId, known_column_evaluation_products};
        use crate::v4::api::zk::Fp8Job;
        use crate::v4::circuit::blake3_stark::stark::ROWS_PER_COMPRESSION;

        for device in [Device::H100, Device::B200] {
            for hash_id in HashId::ALL {
                let mut job = dense_params().job().clone();
                job.common.device = device;
                job.operands.a.hash_id = hash_id;
                job.operands.b.hash_id = hash_id;
                let params = PublicParams::try_new(job, empty_jackpot(), None).unwrap();
                let system = Fp8Job::derive(&params, &IncompleteBlockHeader::zero()).unwrap().system;
                let (h, w, k, r) = (
                    params.h() as u64,
                    params.w() as u64,
                    params.common_dim() as u64,
                    params.rank() as u64,
                );
                let blocks = params.blake3_work_block_bound().unwrap();
                let heights = [
                    (blocks * ROWS_PER_COMPRESSION as u64).next_power_of_two(),
                    (h.max(w) * k).next_power_of_two(),
                    (h + w).next_power_of_two(),
                    (h * w * (k / 32)).next_power_of_two(),
                    (h * w).next_power_of_two(),
                ];
                let compiled_bytes = (h + w) * k + 2 * (h + w) * (k / BLOCK_SIZE as u64) + blocks * blake3::BLOCK_LEN as u64;
                let mut work = (h + w) * k * r + 2 * r * (h + w + 2 * k) + compiled_bytes;
                for table in Table::ALL {
                    let index = table as usize;
                    let actual_height = 1u64 << system.degree_bits()[index];
                    if table == Table::Blake3 {
                        assert!(heights[index] >= actual_height, "{device:?} {hash_id:?}: Blake3 bound");
                    } else {
                        assert_eq!(heights[index], actual_height, "{device:?} {hash_id:?}: {table:?} height");
                    }
                    let values = &system.known().values_per_table[index];
                    assert_eq!(values.len(), table.known_column_count(device), "{table:?} width");
                    assert_eq!(values.len(), system.known().columns_per_table[index].len());
                    assert!(values.iter().all(|column| column.len() as u64 == actual_height));
                    if !values.is_empty() {
                        let columns = values.len() as u64;
                        work += heights[index] * columns + known_column_evaluation_products(heights[index], columns).unwrap();
                    }
                }
                assert_eq!(params.verifier_work_units(blocks).unwrap(), work);
                assert!(work <= PublicParams::MAX_VERIFIER_WORK);
            }
        }
    }

    fn witness<'a>(fx: &'a Fp8Fixture) -> Fp8Witness<'a> {
        Fp8Witness {
            a_values: &fx.a_values,
            a_scales: &fx.a_scales,
            b_values: &fx.b_values,
            b_scales: &fx.b_scales,
            routing_words: &fx.routing_words,
            offsets_words: &fx.offsets_words,
            aux_msgs: &fx.aux_msgs,
            aux_cvs: &fx.aux_cvs,
            key_a: fx.key_a,
            key_b: fx.key_b,
            jackpot_key: fx.jackpot_key,
            a_hash_id: fx.a_hash_id,
            b_hash_id: fx.b_hash_id,
            routing_hash_id: fx.routing_hash_id,
            offsets_hash_id: fx.offsets_hash_id,
        }
    }

    /// The batch order is the canonical table order: identity positions, per-table
    /// `degree_bits` (LUT slots pinned to their constants), and — the property the
    /// cache key and the Fiat-Shamir transcript rely on — an order independent of the
    /// height profile, hence *not* height-sorted for this fixture (its small main tables
    /// precede the 2^16 LUTs).
    #[test]
    fn batch_order_is_canonical() {
        let fx = build_fixture(false);
        let system = build_system(&fx);

        assert_eq!(system.main_table_positions(), core::array::from_fn(|t| t));
        assert_eq!(system.lut_positions(), core::array::from_fn(|i| NUM_TABLES + i));
        for (i, &table) in system.committed_lut_tables().iter().enumerate() {
            assert_eq!(
                system.degree_bits()[NUM_TABLES + i],
                log2_strict(lut_height(table)),
                "LUT {i} degree bits"
            );
        }
        assert!(
            !system.degree_bits().windows(2).all(|w| w[0] >= w[1]),
            "the canonical profile must not be height-sorted for this fixture"
        );
    }

    /// The known-column Fiat-Shamir slot is empty at construction and becomes the quartet
    /// encoding of the caller's `statement_digest` after bind — not a Poseidon hash of the
    /// known-column values.
    #[test]
    fn known_digest_is_statement_digest_quartet() {
        let fx = build_fixture(false);
        let mut system = build_system(&fx);
        assert!(system.known().digest.is_none(), "digest is unbound at construction");
        system.bind_statement_digest(STATEMENT_DIGEST);
        assert_eq!(
            system.known().digest,
            Some(statement_digest_to_hash_out(STATEMENT_DIGEST)),
            "BatchKnownColumns.digest must be the statement_digest quartet"
        );
    }

    #[test]
    fn compact_hopper_batch_area_matches_the_lookup_inventory() {
        let fx = build_fixture(false);
        let mut data = public_data(&fx);
        data.input_quant.device = Device::H100;
        data.scale.device = Device::H100;
        let system = Fp8System::<F, D>::new(data);
        let starks = system.batch_starks();
        let matmul = Table::Matmul as usize;
        let config = system.config();
        assert_eq!(config.num_challenges, 3);
        assert_eq!(config.fri_config.rate_bits, 1);
        assert_eq!(starks[matmul].constraint_degree(), 3);
        let (helpers, zs, _) = CrossTableLookup::num_ctl_helpers_zs_all(system.ctls(), matmul, config.num_challenges, 3);
        assert_eq!((starks[matmul].num_columns(), helpers, zs), (339, 276, 18));
        assert_eq!(starks[matmul].num_columns() + helpers + zs + 6, 639);
        let pow2g = NUM_TABLES
            + system
                .committed_lut_tables()
                .iter()
                .position(|&t| t == super::super::luts::LutTable::Pow2G)
                .unwrap();
        let (helpers, zs, _) = CrossTableLookup::num_ctl_helpers_zs_all(system.ctls(), pow2g, config.num_challenges, 3);
        assert_eq!((starks[pow2g].num_columns(), helpers, zs), (12, 6, 3));
        assert_eq!(starks[pow2g].num_columns() + helpers + zs + 6, 27);
        let blake3 = Table::Blake3 as usize;
        let (helpers, zs, _) = CrossTableLookup::num_ctl_helpers_zs_all(system.ctls(), blake3, config.num_challenges, 3);
        assert_eq!((starks[blake3].num_columns(), helpers, zs), (452, 303, 24));
        assert_eq!(system.ctls().len(), 23);
        assert_eq!(starks.len(), 21);
    }

    /// The full driver roundtrip on the consistency fixture: setup-time LUT precommitment,
    /// batch proof, verification — then the rejection surface: tampered public inputs, a
    /// wrong consensus cap, a verifier whose class (a) recompute differs (different noise
    /// codes), a diverging `statement_digest`, and a tampered opening.
    #[test]
    fn batch_proof_roundtrips_and_rejects_tampering() {
        for device in [Device::H100, Device::B200] {
            let (header, mut plain) = fixture_job();
            plain.job.common.device = device;
            check_batch_roundtrip(build_fixture_from_job(header, plain));
        }
    }

    fn check_batch_roundtrip(fx: Fp8Fixture) {
        let mut system = build_system(&fx);

        let mut timing = TimingTree::default();
        let preprocessed = system.preprocessed_data::<C>(&mut timing);
        let cap = preprocessed.cap();

        let proof = system
            .prove::<C>(&witness(&fx), |_| STATEMENT_DIGEST, &preprocessed, &mut timing)
            .expect("proving must succeed");

        // Verify fails closed on an unbound digest: `fresh` never bound one, while
        // `system` carries the bind `prove` performed (its verify below is the round trip).
        let fresh = build_system(&fx);
        assert!(
            fresh.verify::<C>(&proof, &fx.public_inputs, &cap).is_err(),
            "an unbound statement_digest must fail closed"
        );
        system.bind_statement_digest(STATEMENT_DIGEST);
        system
            .verify::<C>(&proof, &fx.public_inputs, &cap)
            .expect("the honest proof must verify");

        // The proof ships the witness-generated public inputs at the tables' batch positions.
        for (t, &p) in system.main_table_positions().iter().enumerate() {
            assert_eq!(
                proof.public_inputs[p], fx.public_inputs[t],
                "table {t}: shipped public inputs"
            );
        }

        // A tampered public input (the first nonempty table's leading element) is rejected
        // by the expectation equality check.
        let mut tampered = proof.clone();
        tampered
            .public_inputs
            .iter_mut()
            .find(|pis| !pis.is_empty())
            .expect("some table has public inputs")[0] += F::ONE;
        assert!(
            rejects(&mut system, &tampered, &fx.public_inputs, &cap, STATEMENT_DIGEST),
            "tampered public input accepted"
        );

        // Symmetrically, a diverging expectation rejects the honest proof.
        let mut wrong_expected = fx.public_inputs.clone();
        wrong_expected[0][0] += F::ONE;
        assert!(
            rejects(&mut system, &proof, &wrong_expected, &cap, STATEMENT_DIGEST),
            "diverging expected public input accepted"
        );

        // A wrong consensus LUT cap is rejected (Fiat-Shamir and FRI both diverge).
        let mut bad_cap = cap.clone();
        bad_cap.0[0].elements[0] += F::ONE;
        assert!(
            rejects(&mut system, &proof, &fx.public_inputs, &bad_cap, STATEMENT_DIGEST),
            "wrong LUT cap accepted"
        );

        // A verifier whose statement carries different noise codes recomputes different
        // class (a) columns and must reject the proof (the known-column openings no longer
        // match the trace commitment). Same digest as the proof: failure is the openings,
        // not Fiat-Shamir.
        let mut data = public_data(&fx);
        data.a_noise[0] ^= 0x0080;
        let mut wrong_statement = Fp8System::<F, D>::new(data);
        assert!(
            rejects(&mut wrong_statement, &proof, &fx.public_inputs, &cap, STATEMENT_DIGEST),
            "a diverging class (a) recompute must reject the proof"
        );

        // A diverging statement_digest (header or PublicParams changed) must reject: the
        // Fiat-Shamir transcript absorbs that digest.
        let mut flipped = STATEMENT_DIGEST;
        flipped[0] ^= 1;
        assert!(
            rejects(&mut system, &proof, &fx.public_inputs, &cap, flipped),
            "a diverging statement_digest must reject the proof"
        );

        // A tampered trace opening breaks the FRI binding.
        let mut tampered = proof.clone();
        tampered.proof.openings[0].local_values[0] += F::ONE.into();
        assert!(
            rejects(&mut system, &tampered, &fx.public_inputs, &cap, STATEMENT_DIGEST),
            "tampered opening accepted"
        );
    }
}
