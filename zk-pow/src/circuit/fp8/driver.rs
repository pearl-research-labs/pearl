//! Batch proving and verification for the six FP8 AIRs and their committed lookup tables.
//!
//! # Fixed layout and public geometry
//!
//! [`Fp8System`] derives the programs, verifier-known columns and per-table degree
//! profile from public job data. Tables always appear in [`Table`] order, followed
//! by [`LUT_TABLES`]. This fixes both the cross-table indices and the order in which
//! Fiat–Shamir absorbs commitments, even when job geometry changes.
//!
//! The six main tables commit separately. Lookup tables share a tree in fixed
//! table order. Their heights and commitment layout are derived by the verifier.
//! The FRI low-degree test uses a fixed schedule covering all supported heights,
//! so one recursive verifier can handle every supported degree profile.
//!
//! # Proving
//!
//! [`Fp8System::prove`] generates each trace from its producer's outputs, then
//! checks lookup membership and fills the lookup multiplicity columns. Once the
//! lottery digest is known, the `derive_statement_digest` callback derives the
//! statement digest, including the jackpot claim. That digest is absorbed into
//! Fiat–Shamir before the batch proof is generated.
//!
//! # Verification
//!
//! Before calling [`Fp8System::verify`], the caller binds the statement digest with
//! [`Fp8System::bind_statement_digest`] and supplies the expected public inputs.
//! Verification checks:
//!
//! - every table's public inputs and trace height against the statement;
//! - verifier-known column openings against independently recomputed values;
//! - the lookup setup commitment against the trusted cap;
//! - all arithmetic constraints and cross-table multiset equalities;
//! - the batched FRI low-degree argument.
//!
//! Only `Quant::Fp8E4M3Prequant` is supported. Callers must reject other quantization
//! modes before constructing this system.

use core::borrow::Borrow;

use anyhow::{Result, anyhow, ensure};
use plonky2::field::extension::Extendable;
use plonky2::field::polynomial::PolynomialValues;
use plonky2::fri::FriConfig;
use plonky2::fri::reduction_strategies::FriReductionStrategy;
use plonky2::hash::hash_types::RichField;
use plonky2::hash::merkle_tree::MerkleCap;
use plonky2::plonk::config::GenericConfig;
use plonky2::util::log2_strict;
use plonky2::util::timing::TimingTree;
use starky::batch_proof::BatchStarkProofWithPublicInputs;
use starky::batch_prover::{BatchStarkPreprocessedData, BatchStarkPreprocessedVerifierData, batch_prove};
use starky::batch_stark::BatchStark;
use starky::batch_universal::UniversalVerifierEnvelope;
use starky::batch_verifier::{BatchKnownColumns, batch_verify};
use starky::config::StarkConfig;
use starky::cross_table_lookup::CrossTableLookup;

use super::blake3_stark::columns::{NUM_BLAKE3_PUBLIC_INPUTS, PI_HASH_JACKPOT};
use super::blake3_stark::ctl::blake3_lut_lookups;
use super::blake3_stark::stark::{Blake3KnownInputs, Blake3Program, Blake3Stark, Blake3TraceInputs};
use super::ctl::{LUT_TABLES, LutTable, NUM_ALL_TABLES, NUM_LUT_TABLES, NUM_TABLES, Table, all_cross_table_lookups};
use super::input_quant_stark::columns::{InputQuantColumnsView, NUM_INPUT_QUANT_PUBLIC_INPUTS};
use super::input_quant_stark::ctl::input_quant_lut_lookups;
use super::input_quant_stark::stark::{InputQuantProgram, InputQuantStark};
use super::known_values::{fp8_known_columns, hash256_to_hash_out};
use super::luts::{
    B200AlignStark, Bytes2Stark, Clamp22Stark, Div448Stark, ExpInfoStark, Int8DecStark, Log16Stark, LutChecker, Pair128Stark,
    Pow2DStark, Pow2GbStark, QcastStark, Range16Stark, RneRndStark, Width16Stark, Width32Stark, XfPow2Stark, lut_height,
    lut_preprocessed_data, lut_trace, num_precommitted_columns,
};
use super::matmul_b200_stark::MatmulB200ColumnsView;
use super::matmul_b200_stark::columns::NUM_MATMUL_PUBLIC_INPUTS;
use super::matmul_b200_stark::ctl::matmul_b200_lut_lookups;
use super::matmul_b200_stark::stark::{MatmulB200Stark, MatmulProgram, generate_b200_trace};
use super::scale_stark::columns::{NUM_SCALE_PUBLIC_INPUTS, ScaleColumnsView};
use super::scale_stark::ctl::{SIGMA_EXP_OFFSET, scale_lut_lookups};
use super::scale_stark::stark::{ScaleProgram, ScaleRowTuple, ScaleStark};
use super::tamed_stark::columns::NUM_TAMED_PUBLIC_INPUTS;
use super::tamed_stark::ctl::tamed_lut_lookups;
use super::tamed_stark::stark::{TamedProgram, TamedStark};
use super::xor_fold_stark::columns::{NUM_XOR_FOLD_PUBLIC_INPUTS, XorFoldColumnsView};
use super::xor_fold_stark::ctl::xor_fold_lut_lookups;
use super::xor_fold_stark::stark::{XorFoldProgram, XorFoldStark};
use crate::api::fp8::public_params::HashId;
use crate::api::primitives::Hash256;
use crate::v2::api::proof_utils::u32_field_array_to_hash;

// The consensus proof shape

/// Targeted conjectured security in bits: `queries * rate_bits + proof_of_work_bits`.
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
/// `k ∈ [2^11, 2^16]` with `32 | k`, `h ≥ 4`, `w ≥ 16`, `h·w ∈ [256, 2048]`,
/// `(h + w)·k ≤ 2^22`; heights are `next_pow2` of each table's row formula:
///
/// | table      | rows             | bits    |
/// |------------|------------------|---------|
/// | InputQuant | `max(h,w)·k`     | 15..=22 |
/// | Matmul     | `h·w·(k/32)`     | 14..=22 |
/// | Blake3     | `8·compressions` | 14..=21 |
/// | XorFold    | `h·w`            |  8..=11 |
/// | Tamed      | `h·w`            |  8..=11 |
/// | Scale      | `h + w`          |  5..=10 |
/// | LUTs       | [`lut_height`]   | {17, 16, 14, 11, 10, 8, 6, 5} |
///
/// Blake3 bounds: opened bytes are `1.25·(h+w)·k ∈ [5·2^14, 5·2^20]` (values `k` plus
/// scales `k/4` bytes per strip), so leaf compressions alone are `≥ 1280 > 2^10`
/// (rows `> 2^13`), while leaves (`16·chunks ≤ 16·((5/4096)·2^22 + 4(h+w)) < 2^17`) +
/// strip membership paths (`≤ 2·chunks + 30·2(h+w)`) + MoE hotspot rows stay `< 2^18`
/// (rows `≤ 2^21`). The union is `[5, 22] \ {12, 13}`: no table lands on `2^12` or
/// `2^13` (Scale and XorFold/Tamed top out at `2^10`/`2^11`, the tallest sub-2^14 LUT
/// is the `2^11` XFPOW2, Matmul rows are `≥ 256·2^11/32 = 2^14`, InputQuant `≥ 2^15`,
/// Blake3 `> 2^13`). Coverage and minimality are asserted by
/// `ladder_covers_the_envelope`; real jobs are gated onto the ladder at
/// `Fp8Job::derive`.
pub const FP8_REACHABLE_DEGREE_BITS: [usize; 16] = [22, 21, 20, 19, 18, 17, 16, 15, 14, 11, 10, 9, 8, 7, 6, 5];

/// Per-main-table inclusive degree-bits ranges `[lo, hi]` over the wire envelope, in
/// canonical `Table` order — the rows of [`FP8_REACHABLE_DEGREE_BITS`]'s table. Exactness
/// (each bound is attained by some envelope-legal job) is asserted by
/// `ladder_covers_the_envelope`; Blake3's bounds are proven, not swept (see above).
pub const FP8_MAIN_TABLE_DEGREE_RANGES: [(usize, usize); NUM_TABLES] = [(14, 21), (15, 22), (5, 10), (14, 22), (8, 11), (8, 11)];

/// LUT batch indices. Their fixed heights allow one shared multi-height Merkle tree
/// per oracle role, reducing both caps and recursive authentication paths.
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
pub fn fp8_universal_envelope() -> UniversalVerifierEnvelope {
    let mut degree_ranges: Vec<(usize, usize)> = FP8_MAIN_TABLE_DEGREE_RANGES.to_vec();
    degree_ranges.extend(LUT_TABLES.iter().map(|&table| {
        let bits = log2_strict(lut_height(table));
        (bits, bits)
    }));
    UniversalVerifierEnvelope {
        degree_ranges,
        ladder: FP8_REACHABLE_DEGREE_BITS.to_vec(),
        grouped_tables: FP8_GROUPED_TABLES.to_vec(),
    }
}

/// FRI configuration for the supplied table heights (any order, duplicates allowed).
/// Production profiles use the fixed [`FP8_REACHABLE_DEGREE_BITS`] ladder. Jobs fold
/// its suffix from their tallest table, but absorb the complete ladder into Fiat–Shamir,
/// so one recursive verifier can cover them all.
/// Test profiles outside the envelope add their heights as extra boundaries.
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

// The statement

/// Public programs and recomputation inputs for one FP8 statement.
/// Callers must validate the supported prequant mode before construction.
/// Public inputs are checked separately through `Fp8System::verify`.
pub struct Fp8PublicData {
    pub blake3: Blake3Program,
    pub input_quant: InputQuantProgram,
    pub scale: ScaleProgram,
    pub matmul: MatmulProgram,
    pub xor_fold: XorFoldProgram,
    /// Raw byte lengths of the four opened strip planes (A/B values, A/B scales) — the
    /// Blake3 verifier-known schedule inputs. The MoE routing statement needs no length here:
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

/// The sixteen LUT AIRs at their exact widths, in [`LUT_TABLES`] order (boxed: each table
/// instantiates its own width).
struct LutStarks<F: RichField + Extendable<D>, const D: usize> {
    starks: Vec<Box<dyn BatchStark<F, D>>>,
}

/// One LUT AIR at its table's exact width (the width is a compile-time constant per table,
/// so the dispatch is a static match).
fn boxed_lut_stark<F: RichField + Extendable<D>, const D: usize>(table: LutTable) -> Box<dyn BatchStark<F, D>> {
    match table {
        LutTable::RneRnd => Box::new(RneRndStark::<F, D>::new(table)),
        LutTable::Range16 => Box::new(Range16Stark::<F, D>::new(table)),
        LutTable::Bytes2 => Box::new(Bytes2Stark::<F, D>::new(table)),
        LutTable::Qcast => Box::new(QcastStark::<F, D>::new(table)),
        LutTable::Div448 => Box::new(Div448Stark::<F, D>::new(table)),
        LutTable::Pair128 => Box::new(Pair128Stark::<F, D>::new(table)),
        LutTable::Clamp22 => Box::new(Clamp22Stark::<F, D>::new(table)),
        LutTable::Int8Dec => Box::new(Int8DecStark::<F, D>::new(table)),
        LutTable::ExpInfo => Box::new(ExpInfoStark::<F, D>::new(table)),
        LutTable::Pow2D => Box::new(Pow2DStark::<F, D>::new(table)),
        LutTable::B200Align => Box::new(B200AlignStark::<F, D>::new(table)),
        LutTable::Pow2Gb => Box::new(Pow2GbStark::<F, D>::new(table)),
        LutTable::Width32 => Box::new(Width32Stark::<F, D>::new(table)),
        LutTable::XfPow2 => Box::new(XfPow2Stark::<F, D>::new(table)),
        LutTable::Width16 => Box::new(Width16Stark::<F, D>::new(table)),
        LutTable::Log16 => Box::new(Log16Stark::<F, D>::new(table)),
    }
}

impl<F: RichField + Extendable<D>, const D: usize> LutStarks<F, D> {
    fn new() -> Self {
        Self {
            starks: LUT_TABLES.iter().map(|&t| boxed_lut_stark::<F, D>(t)).collect(),
        }
    }

    /// The AIRs as batch tables, in [`LUT_TABLES`] order.
    fn as_dyn(&self) -> [&dyn BatchStark<F, D>; NUM_LUT_TABLES] {
        core::array::from_fn(|i| self.starks[i].as_ref())
    }
}

/// Generates the Matmul trace from the noised fp8 codes and their summand scores: the
/// column-major polynomials, the public inputs, the finished cells' f32 words (XorFold's
/// inputs), and the per-cell `(CELL_MAGNITUDE_EXPONENT, CELL_SKIPS)` tuples (Tamed's inputs — exactly the
/// E-cell channel tuples).
#[allow(clippy::type_complexity)]
fn matmul_trace<F: RichField>(
    program: &MatmulProgram,
    a_codes: &[u8],
    b_codes: &[u8],
    a_lambdas: &[u64],
    b_lambdas: &[u64],
) -> (Vec<PolynomialValues<F>>, Vec<F>, Vec<u32>, Vec<(u64, u64)>) {
    let cells = program.h * program.w;
    let (mut cell_words, mut cell_tuples) = (vec![0u32; cells], vec![(0u64, 0u64); cells]);
    let (rows, pis) = generate_b200_trace::<F>(program, a_codes, b_codes, a_lambdas, b_lambdas);
    for r in &rows {
        let v: &MatmulB200ColumnsView<F> = r.borrow();
        // Live finals only: the trailing phantom cells' finals carry no cell.
        if v.is_cell_final == F::ONE && v.is_padding == F::ZERO {
            let cell = v.cell_id.to_canonical_u64() as usize;
            cell_words[cell] = (v.cell_result_f32_lo.to_canonical_u64() | (v.cell_result_f32_hi.to_canonical_u64() << 16)) as u32;
            cell_tuples[cell] = (v.cell_magnitude_exponent.to_canonical_u64(), v.cell_skips.to_canonical_u64());
        }
    }
    (column_major(&rows), pis.to_vec(), cell_words, cell_tuples)
}

/// One job's fully derived proving/verifying context. See the module docs.
pub struct Fp8System<F: RichField + Extendable<D>, const D: usize> {
    blake3: Blake3Stark<F, D>,
    input_quant: InputQuantStark<F, D>,
    scale: ScaleStark<F, D>,
    matmul: MatmulB200Stark<F, D>,
    xor_fold: XorFoldStark<F, D>,
    tamed: TamedStark<F, D>,
    luts: LutStarks<F, D>,
    /// Verifier-known recompute inputs kept for trace generation and witness validation.
    plane_lens: [usize; 4],
    a_noise: Vec<u16>,
    b_noise: Vec<u16>,
    /// Per-table trace degree bits in canonical batch order, matching the expected proof profile.
    degree_bits: [usize; NUM_ALL_TABLES],
    /// The consensus config for this job's degree profile.
    config: StarkConfig,
    /// Verifier-known columns in canonical table order. The digest slot is empty until
    /// `bind_statement_digest` fills it.
    known: BatchKnownColumns<F>,
    /// All 24 CTL channels, table indices in canonical order.
    ctls: Vec<CrossTableLookup<F>>,
}

impl<F: RichField + Extendable<D>, const D: usize> Fp8System<F, D> {
    /// Derives the full batching context from the statement: verifier-known values (which fix
    /// every main table's height), the consensus config, the CTL set and the known-column
    /// data — all in the canonical table order.
    pub fn new(data: Fp8PublicData) -> Self {
        let Fp8PublicData {
            blake3,
            input_quant,
            scale,
            matmul,
            xor_fold,
            a_values_len,
            a_scales_len,
            b_values_len,
            b_scales_len,
            a_noise,
            b_noise,
        } = data;
        // Tamed's program is the shared geometry itself.
        let tamed = TamedProgram {
            h: matmul.h,
            w: matmul.w,
            k: matmul.k,
        };

        // Verifier-known recompute — also the (only) source of the main tables' heights: the
        // verifier must never take the batch layout from the proof.
        let known_values = [
            blake3.known_values::<F>(&Blake3KnownInputs {
                a_values_len,
                a_scales_len,
                b_values_len,
                b_scales_len,
            }),
            input_quant.known_values::<F>(&a_noise, &b_noise),
            scale.known_values::<F>(),
            matmul.known_values::<F>(),
            xor_fold.known_values::<F>(),
            tamed.known_values::<F>(),
        ];
        let mut heights = [0usize; NUM_ALL_TABLES];
        for (t, columns) in known_values.iter().enumerate() {
            heights[t] = columns[0].len();
        }
        for (i, &table) in LUT_TABLES.iter().enumerate() {
            heights[NUM_TABLES + i] = lut_height(table);
        }

        // The batch order is the canonical table order itself — profile-independent, so
        // the Fiat-Shamir cap/opening order, the statement digest and the CTL indices
        // are the same for every job; batched FRI folds by height internally.
        let degree_bits: [usize; NUM_ALL_TABLES] = core::array::from_fn(|t| log2_strict(heights[t]));
        let config = fp8_stark_config(&degree_bits);

        let known = fp8_known_columns::<F>(known_values);
        // All 24 channels, in the same canonical index space.
        let ctls = all_cross_table_lookups::<F>(&scale);

        Self {
            blake3: Blake3Stark::new(blake3),
            input_quant: InputQuantStark::new(input_quant),
            scale: ScaleStark::new(scale),
            matmul: MatmulB200Stark::new(matmul),
            xor_fold: XorFoldStark::new(xor_fold),
            tamed: TamedStark::new(tamed),
            luts: LutStarks::new(),
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

    /// Per-table trace degree bits in canonical batch order, matching the expected proof profile.
    pub fn degree_bits(&self) -> &[usize; NUM_ALL_TABLES] {
        &self.degree_bits
    }

    /// Job geometry used by tests; the universal circuit takes geometry and degrees as public inputs.
    #[cfg(test)]
    pub fn geometry(&self) -> [usize; 3] {
        let p = &self.input_quant.program;
        [p.h, p.w, p.k]
    }

    /// The batch positions of the sixteen LUT tables, in [`LUT_TABLES`]
    /// order. The batch order is canonical, so LUT `i` sits at `NUM_TABLES + i`.
    pub fn lut_positions(&self) -> [usize; NUM_LUT_TABLES] {
        core::array::from_fn(|i| NUM_TABLES + i)
    }

    /// The batch positions of the six main tables, in canonical `Table` order — where a
    /// caller finds each table's public inputs inside `proof.public_inputs`. The batch
    /// order is canonical, so table `t` sits at `t`.
    pub fn main_table_positions(&self) -> [usize; NUM_TABLES] {
        core::array::from_fn(|t| t)
    }

    /// Builds the fixed LUT setup commitment in canonical table order.
    pub fn preprocessed_data<C: GenericConfig<D, F = F>>(&self, timing: &mut TimingTree) -> BatchStarkPreprocessedData<F, C, D> {
        lut_preprocessed_data::<F, C, D>(NUM_ALL_TABLES, self.lut_positions(), &self.config, timing)
    }

    /// The verifier's view of the LUT precommitment: the consensus `cap` plus the
    /// per-table preprocessed column indices, which the statement pins itself (a caller
    /// cannot smuggle in a different column split).
    pub fn preprocessed_verifier_data<C: GenericConfig<D, F = F>>(
        &self,
        cap: &MerkleCap<F, C::Hasher>,
    ) -> BatchStarkPreprocessedVerifierData<F, C, D> {
        let mut columns_per_table = vec![Vec::new(); NUM_ALL_TABLES];
        for (i, &table) in LUT_TABLES.iter().enumerate() {
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

    /// The verifier-known columns in canonical table order (for the recursive wrapper:
    /// column indices shape the circuit; the values feed the native evaluation-at-zeta
    /// recompute). The digest slot is empty until `bind_statement_digest` fills it.
    pub(super) fn known(&self) -> &BatchKnownColumns<F> {
        &self.known
    }

    /// Binds `statement_digest` into the known-column Fiat-Shamir slot. The caller of
    /// [`Self::verify`] must invoke this (with the digest of the statement it will verify
    /// against) before the verification: `verify` fails closed on an unbound slot.
    /// [`Self::prove`] performs the bind itself.
    pub fn bind_statement_digest(&mut self, statement_digest: Hash256) {
        self.known.digest = Some(hash256_to_hash_out(statement_digest));
    }

    /// Checks a supplied digest against the system's bound digest, if present.
    pub fn ensure_statement_digest_binding(&self, statement_digest: Hash256) -> Result<()> {
        ensure!(
            self.known
                .digest
                .is_none_or(|bound| bound == hash256_to_hash_out(statement_digest)),
            "the statement_digest disagrees with the one bound into the system's known columns"
        );
        Ok(())
    }

    /// The batch tables, canonical order.
    pub(super) fn batch_starks(&self) -> [&dyn BatchStark<F, D>; NUM_ALL_TABLES] {
        let mut tables: Vec<&dyn BatchStark<F, D>> = vec![
            &self.blake3,
            &self.input_quant,
            &self.scale,
            &self.matmul,
            &self.xor_fold,
            &self.tamed,
        ];
        tables.extend(self.luts.as_dyn());
        core::array::from_fn(|t| tables[t])
    }

    /// The main tables' public inputs, padded with the LUT tables' empty vectors to the
    /// canonical batch shape.
    pub(super) fn batch_public_inputs(&self, main_pis: &[Vec<F>; NUM_TABLES]) -> [Vec<F>; NUM_ALL_TABLES] {
        core::array::from_fn(|t| if t < NUM_TABLES { main_pis[t].clone() } else { Vec::new() })
    }

    /// Generates the main traces, LUT multiplicities and batch proof using the fixed setup data.
    /// `derive_statement_digest` receives the generated lottery digest and returns the
    /// statement digest to bind into Fiat–Shamir before proving.
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

        // InputQuant: the opened strips + the statement's noise codes.
        let as_int8 = |bytes: &[u8]| -> Vec<i8> { bytes.iter().map(|&b| b as i8).collect() };
        let le_codes = |bytes: &[u8]| -> Vec<u16> { bytes.chunks(2).map(|p| u16::from_le_bytes([p[0], p[1]])).collect() };
        let (a_int8, b_int8) = (as_int8(witness.a_values), as_int8(witness.b_values));
        let (a_scale_codes, b_scale_codes) = (le_codes(witness.a_scales), le_codes(witness.b_scales));
        let (iq_rows, iq_pis) = self.input_quant.program.generate_trace::<F>(
            &a_int8,
            &a_scale_codes,
            &self.a_noise,
            &b_int8,
            &b_scale_codes,
            &self.b_noise,
        );

        // Scale: one aggregate tuple per matrix row, exactly what InputQuant committed.
        let (h, w, k) = (
            self.input_quant.program.h,
            self.input_quant.program.w,
            self.input_quant.program.k,
        );
        let tuples = |b_side: bool| -> Vec<ScaleRowTuple> {
            (0..if b_side { w } else { h })
                .map(|g| {
                    let v: &InputQuantColumnsView<F> = iq_rows[g * k + k - 1].borrow();
                    if b_side {
                        ScaleRowTuple {
                            l2_frame_sum: v.block_l2_b.running_l2_frame_sum.to_canonical_u64(),
                            frame_doubled_scale_exponent: v.block_l2_b.frame_doubled_scale_exponent.to_canonical_u64() as u32,
                            max_abs: v.max_abs_b.to_canonical_u64() as u32,
                            dead_count: v.dead_count_b.to_canonical_u64() as u32,
                        }
                    } else {
                        ScaleRowTuple {
                            l2_frame_sum: v.block_l2_a.running_l2_frame_sum.to_canonical_u64(),
                            frame_doubled_scale_exponent: v.block_l2_a.frame_doubled_scale_exponent.to_canonical_u64() as u32,
                            max_abs: v.max_abs_a.to_canonical_u64() as u32,
                            dead_count: v.dead_count_a.to_canonical_u64() as u32,
                        }
                    }
                })
                .collect()
        };
        let (scale_rows, scale_pis) = self.scale.program.generate_trace::<F>(&tuples(false), &tuples(true));

        // Matmul: the noised fp8 codes and summand scores InputQuant committed (same
        // element order). Live rows only: past a side's
        // `h*k`/`w*k` elements the InputQuant trace carries the dead phantom fill.
        let codes = |b_side: bool| -> Vec<u8> {
            let live = if b_side { w * k } else { h * k };
            iq_rows[..live]
                .iter()
                .map(|r| {
                    let v: &InputQuantColumnsView<F> = r.borrow();
                    (if b_side { v.code_noised_b } else { v.code_noised_a }).to_canonical_u64() as u8
                })
                .collect()
        };
        let lambdas = |b_side: bool| -> Vec<u64> {
            let live = if b_side { w * k } else { h * k };
            iq_rows[..live]
                .iter()
                .map(|r| {
                    let v: &InputQuantColumnsView<F> = r.borrow();
                    (if b_side { v.summand_score_b } else { v.summand_score_a }).to_canonical_u64()
                })
                .collect()
        };
        let (mat_trace, mat_pis, cell_words, cell_tuples) = matmul_trace(
            &self.matmul.program,
            &codes(false),
            &codes(true),
            &lambdas(false),
            &lambdas(true),
        );

        // XorFold: fold Matmul's finished cell words into the lottery lanes.
        let (xf_rows, xf_pis) = self.xor_fold.program.generate_trace::<F>(&cell_words);

        // Tamed: Matmul's per-cell binade-and-census tuples against Scale's per-row
        // sigma frames, reassembled exactly as the channels export them.
        let sigma_frames = |rows: core::ops::Range<usize>| -> Vec<(u64, u64)> {
            rows.map(|g| {
                let v: &ScaleColumnsView<F> = scale_rows[g].borrow();
                (
                    v.sigma_significand.to_canonical_u64(),
                    v.alpha_exp.to_canonical_u64() + v.l2_floored_exponent.to_canonical_u64() + SIGMA_EXP_OFFSET,
                )
            })
            .collect()
        };
        let (tamed_rows, tamed_pis) =
            self.tamed
                .program
                .generate_trace::<F>(&cell_tuples, &sigma_frames(0..h), &sigma_frames(h..h + w));

        // Blake3: the strip bytes and the folded lottery words under the job schedule.
        let mut lottery_words = [0u32; 16];
        for r in &xf_rows {
            let v: &XorFoldColumnsView<F> = r.borrow();
            if v.is_lane_final == F::ONE {
                lottery_words[v.lane_id.to_canonical_u64() as usize] = (v.rotation_input_top13.to_canonical_u64()
                    + (v.rotation_input_bottom19_limb_0.to_canonical_u64() << 13)
                    + (v.rotation_input_bottom19_limb_1.to_canonical_u64() << 29))
                    as u32;
            }
        }
        let (b3_rows, b3_pis) = self.blake3.program.generate_trace::<F>(&Blake3TraceInputs {
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
        });

        // The proof ships the public inputs the witness generates (keys, roots, the geometry
        // and dead-limit slots, the lottery digest); the verifier pins every slot to its own
        // expectations in [`Self::verify`].
        let generated: [Vec<F>; NUM_TABLES] = [
            b3_pis.to_vec(),
            iq_pis.to_vec(),
            scale_pis.to_vec(),
            mat_pis,
            xf_pis.to_vec(),
            tamed_pis.to_vec(),
        ];

        // Column-major traces, canonical order.
        let mut traces: Vec<Vec<PolynomialValues<F>>> = vec![
            column_major(&b3_rows),
            column_major(&iq_rows),
            column_major(&scale_rows),
            mat_trace,
            column_major(&xf_rows),
            column_major(&tamed_rows),
        ];

        // Check LUT membership and accumulate multiplicities, reporting invalid witness tuples early.
        let mut checker = LutChecker::<F>::new();
        let inventories = [
            (blake3_lut_lookups::<F>(), "Blake3"),
            (input_quant_lut_lookups::<F>(), "InputQuant"),
            (scale_lut_lookups::<F>(&self.scale.program), "Scale"),
            (matmul_b200_lut_lookups::<F>(), "Matmul"),
            (xor_fold_lut_lookups::<F>(), "XorFold"),
            (tamed_lut_lookups::<F>(), "Tamed"),
        ];
        for (t, (lookups, name)) in inventories.iter().enumerate() {
            checker
                .check_trace(lookups, &traces[t], &generated[t], name)
                .map_err(|e| anyhow!(e))?;
        }
        for table in LUT_TABLES {
            traces.push(lut_trace::<F>(table, checker.multiplicities.table_columns(table)));
        }

        // The traces were assembled in canonical order, which is the batch order.
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
            Some(preprocessed),
            Some(&self.known),
            timing,
        )
    }

    /// Verifies against the expected public inputs, degree profile, fixed LUT cap and
    /// previously bound statement digest. The batch verifier checks all AIRs, known-column
    /// openings, CTL balances and FRI.
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
            NUM_TAMED_PUBLIC_INPUTS,
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
    /// Merkle leaf sizes from [`PublicParams`](crate::api::fp8::public_params::PublicParams)
    /// (not stored on [`Blake3Program`]).
    pub a_hash_id: HashId,
    pub b_hash_id: HashId,
    pub routing_hash_id: HashId,
    pub offsets_hash_id: HashId,
}

/// Row-major trace rows -> the column-major `PolynomialValues` layout of the batch prover.
fn column_major<F: RichField, const N: usize>(rows: &[[F; N]]) -> Vec<PolynomialValues<F>> {
    (0..N)
        .map(|c| PolynomialValues::new(rows.iter().map(|r| r[c]).collect()))
        .collect()
}

#[cfg(test)]
mod tests {
    use plonky2::field::goldilocks_field::GoldilocksField;
    use plonky2::field::types::Field;
    use plonky2::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};

    use super::super::consistency::{Fp8Fixture, build_fixture};
    use super::super::known_values::hash256_to_hash_out;
    use super::*;

    const D: usize = 2;
    type C = PoseidonGoldilocksConfig;
    type F = GoldilocksField;

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
        for max in [21, 20, 19, 18, 17, 16, 15, 14] {
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

    /// Check the configured FRI parameters against [`STARK_SECURITY_BITS`].
    #[test]
    fn consensus_config_meets_the_security_target() {
        fp8_stark_config(&FP8_REACHABLE_DEGREE_BITS)
            .check_config::<F, D>()
            .expect("consensus FRI parameters must meet the security target");
    }

    /// [`FP8_REACHABLE_DEGREE_BITS`] is exactly the envelope's reach: sweep every lottery
    /// tile `(h, w)` and the job `k` range, apply each table's height formula, and compare
    /// against the ladder (with Blake3 contributing its bounded `14..=21` — see the
    /// constant's docs). Guards both directions: a formula landing off the ladder breaks
    /// batch FRI injection; a ladder stop nothing reaches taxes every proof with a dead
    /// fold layer.
    #[test]
    fn ladder_covers_the_envelope() {
        use std::collections::BTreeSet;

        // Per-main-table reach, canonical `Table` order (Blake3's bounded 14..=21 is
        // documented on the constant, not swept).
        let mut per_table: [BTreeSet<usize>; NUM_TABLES] = Default::default();
        per_table[Table::Blake3 as usize].extend(14..=21);
        // Envelope (`api::layout` tile bounds): h >= 4
        // (MIN_TILE_ROWS), w >= 16 (MIN_TILE_COLS), h*w <= 2048 (MAX_TILE_ELEMS),
        // h*w >= 16*16 (JACKPOT_ENTRIES lanes of >= MIN_SUBTILE_ELEMS), and the
        // opened-strips bound (h + w)*k <= 2^22 with k in [2048, 2^16], 32 | k.
        let bits = |rows: usize| rows.next_power_of_two().trailing_zeros() as usize;
        for h in 4..=128 {
            for w in 16..=(2048 / h) {
                if h * w < 256 {
                    continue;
                }
                let k_max = ((1 << 22) / (h + w) / 32 * 32).min(1 << 16);
                if k_max < 2048 {
                    continue;
                }
                per_table[Table::Scale as usize].insert(bits(h + w));
                per_table[Table::XorFold as usize].insert(bits(h * w));
                per_table[Table::Tamed as usize].insert(bits(h * w));
                // InputQuant `max(h,w)*k` and Matmul `h*w*(k/32)` are monotone in k, and
                // consecutive k steps (+32 on k >= 2048) never double a value, so
                // sweeping k hits exactly the bit range between the two endpoints.
                per_table[Table::InputQuant as usize].extend(bits(h.max(w) * 2048)..=bits(h.max(w) * k_max));
                per_table[Table::Matmul as usize].extend(bits(h * w * (2048 / 32))..=bits(h * w * (k_max / 32)));
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
        reachable.extend(LUT_TABLES.into_iter().map(|t| bits(lut_height(t))));

        assert!(
            !reachable.contains(&12) && !reachable.contains(&13),
            "2^12 and 2^13 must stay unreachable (the ladder's gap)"
        );
        let ladder: BTreeSet<usize> = FP8_REACHABLE_DEGREE_BITS.into_iter().collect();
        assert_eq!(reachable, ladder, "the consensus ladder must equal the envelope's reach");

        // The universal envelope is well-formed: bounds on the ladder,
        // fixed LUT slots, and the ladder headed by the tallest reachable table.
        let envelope = fp8_universal_envelope();
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

    fn public_data(fx: &Fp8Fixture) -> Fp8PublicData {
        Fp8PublicData {
            blake3: fx.blake3.clone(),
            input_quant: fx.input_quant.clone(),
            scale: fx.scale.clone(),
            matmul: fx.matmul.clone(),
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
        for (i, &table) in LUT_TABLES.iter().enumerate() {
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
            Some(hash256_to_hash_out(STATEMENT_DIGEST)),
            "BatchKnownColumns.digest must be the statement_digest quartet"
        );
    }

    /// The full driver roundtrip on the consistency fixture: setup-time LUT precommitment,
    /// batch proof, verification — then the rejection surface: tampered public inputs, a
    /// wrong consensus cap, a verifier whose verifier-known recompute differs (different noise
    /// codes), a diverging `statement_digest`, and a tampered opening.
    #[test]
    fn batch_proof_roundtrips_and_rejects_tampering() {
        let fx = build_fixture(false);
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
        // verifier-known columns and must reject the proof (the known-column openings no longer
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
