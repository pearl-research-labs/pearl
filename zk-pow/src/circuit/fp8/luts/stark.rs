//! Committed LUT contents, trace generation, setup commitment, and AIR.
//! See the [parent module](super) for the lookup and padding invariants.

use core::marker::PhantomData;

use plonky2::field::extension::{Extendable, FieldExtension};
use plonky2::field::packed::PackedField;
use plonky2::field::polynomial::PolynomialValues;
use plonky2::field::types::Field;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::ext_target::ExtensionTarget;
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2::plonk::config::GenericConfig;
use plonky2::util::timing::TimingTree;
use starky::batch_prover::BatchStarkPreprocessedData;
use starky::batch_stark::BatchStark;
use starky::config::StarkConfig;
use starky::constraint_consumer::{ConstraintConsumer, RecursiveConstraintConsumer};
use starky::evaluation_frame::StarkFrame;
use starky::stark::Stark;

use super::super::ctl::{LUT_TABLES, NUM_LUT_TABLES};
use super::super::input_quant_stark::stark::qcast;
use super::super::matmul_b200_stark::stark::B200Product;
use super::super::scale_stark::stark::{CODE_448, rnernd_reference};
use super::super::unpredictability::{log2_fixed, sig_nonzero, sig_width};
use super::LutTable;
use super::columns::{
    XFPOW2_CAP, XFPOW2_LIMBS, XFPOW2_ZERO_POINT, lut_height, lut_num_columns, num_precommitted_columns, num_slots, slot_height,
};
use crate::api::fp8::compute::bf16_div;

/// Generates one table (or one slot of a folded table): the stored value columns, in the order
/// the consumers' `LutLookup::values` bind them.
pub fn generate<F: Field>(table: LutTable, slot: usize) -> Vec<Vec<F>> {
    assert!(slot < num_slots(table), "slot {slot} out of range for {table:?}");
    let height = slot_height(table);
    let f = F::from_canonical_u64;
    let mut columns: Vec<Vec<F>> = match table {
        LutTable::Range16 => vec![],
        _ => {
            let arity = match table {
                LutTable::Bytes2 | LutTable::Pair128 | LutTable::Width32 | LutTable::Width16 => 2,
                LutTable::Int8Dec | LutTable::RneRnd => 4,
                LutTable::B200Align => 5,
                LutTable::XfPow2 => XFPOW2_LIMBS,
                _ => 1,
            };
            vec![Vec::with_capacity(height); arity]
        }
    };

    for key in 0..height as u64 {
        match table {
            LutTable::Range16 => {}
            // All byte pairs (paired 8-bit range check); a lone byte keys (d, 0).
            LutTable::Bytes2 => {
                columns[0].push(f(key & 0xFF));
                columns[1].push(f(key >> 8));
            }
            // All 7-bit pairs (mantissa fields, ScaleStark's sqrt/l2-linf pairs).
            LutTable::Pair128 => {
                columns[0].push(f(key & 0x7F));
                columns[1].push(f(key >> 7));
            }
            // fp8 quantization of a bf16 code: `f32_to_fp8_e4m3(bf16_to_f32(clamp_±448(x)))`,
            // via InputQuant's mirror. Non-finite keys (exponent field 255) cannot decode; they
            // are unreachable because each consumer range-checks `254 - out_exp`; their rows
            // hold the saturation code.
            LutTable::Qcast => {
                let code = key as u16;
                let out = if code & 0x7F80 == 0x7F80 {
                    (code >> 8) as u8 & 0x80 | 0x7E
                } else {
                    qcast(code)
                };
                columns[0].push(f(u64::from(out)));
            }
            // Exact bf16 decode fields of the int8 value: byte 0 -> +0, else the normalized
            // `(SIGN, EXP, MANTISSA)` with `EXP_IS_ZERO = 0` (every int8 fits 8 significand
            // bits, so the decode is exact; byte 0x80 = -128 -> (1, 134, 0, 0)).
            LutTable::Int8Dec => {
                let v = key as u8 as i8;
                let (sign, exp, man, eiz) = if v == 0 {
                    (0, 0, 0, 1)
                } else {
                    let a = (v as i64).unsigned_abs(); // in [1, 128]
                    let w = 64 - a.leading_zeros() as u64;
                    (u64::from(v < 0), 126 + w, (a << (8 - w)) - 128, 0)
                };
                columns[0].push(f(sign));
                columns[1].push(f(exp));
                columns[2].push(f(man));
                columns[3].push(f(eiz));
            }
            // The alpha definition `RNE_bf16(448 / noised_bound)` via the native `bf16_div`.
            // Error paths (zero/subnormal-tiny denominators, non-finite keys) hold the
            // exponent-255 sentinel the alpha validity RCs reject; unreachable on the live
            // domain (the 2^-32 norm floors keep the noised bound at or above 2^-32).
            LutTable::Div448 => {
                let code = key as u16;
                let sentinel = code & 0x8000 | 0x7F80;
                let out = if code & 0x7F80 == 0x7F80 {
                    sentinel
                } else {
                    bf16_div(CODE_448, code).unwrap_or(sentinel)
                };
                columns[0].push(f(u64::from(out)));
            }
            // Subnormal flag of an exponent field; the [0, 254] key domain is the finiteness
            // proof.
            LutTable::ExpInfo => columns[0].push(f(u64::from(key == 0))),
            // The RNERND slot for fade argument `x = key - 400`: `clamp(x, -7, 18) + 7`.
            LutTable::Clamp22 => columns[0].push(f((key as i64 - 400 + 7).clamp(0, 25) as u64)),
            LutTable::Pow2D => columns[0].push(f(1 << key)),
            // Matmul's carry alignment: the 26-cap floors a far accumulator to zero
            // (`4*GROUP_OUTPUT_SIGNIFICAND < 2^26`), exactly as the window drops it.
            LutTable::Pow2Gb => columns[0].push(f(1 << key.min(26))),
            // Matmul's truncate-toward-zero back-end: W = key + 1 (the stored key column
            // is the shifted ramp [1, 32] — no key 0), value (2^max(W-24,0), 2^max(24-W,0)).
            LutTable::Width32 => {
                let w = key + 1;
                columns[0].push(f(1 << w.saturating_sub(24)));
                columns[1].push(f(1 << 24u64.saturating_sub(w)));
            }
            // Shared bf16 RNE back-end (ScaleStark's mirror; one slot per signed cut). See the
            // module header for the InputQuant cancellation case that also needs the LSB scale.
            LutTable::RneRnd => {
                let (mant, wa, oiz, oez) = rnernd_reference(key, slot as u64);
                columns[0].push(f(mant));
                columns[1].push(f(wa));
                columns[2].push(f(u64::from(oiz)));
                columns[3].push(f(u64::from(oez)));
            }
            // The whole per-lane fp8 product, pre-truncated to the B200 window (Matmul's
            // mirror; slot = REL). `ALIGNED_LANE_TERMS = ±floor(P*2^19 / 2^REL)` toward zero with
            // the sign folded; slots REL >= 27 hold 0 (P < 2^8). `PRODUCT_BIASED_EXPONENT`
            // is the biased stored-exponent sum (sentinel 0 for zero products,
            // slot-independent). `OPERAND_CODES_A/B` echo the key's operand bytes, pinning the
            // looking side's code columns individually. `BINADE` is the product's check-3
            // binade `floor(log2 |prod|) + 139` (0 for zero products, same in every slot).
            LutTable::B200Align => {
                let p = B200Product::new((key & 0xFF) as u8, (key >> 8) as u8);
                let aligned = if p.is_zero {
                    F::ZERO
                } else {
                    let mag = f((p.sig << 19) >> slot.min(63));
                    if p.sign { -mag } else { mag }
                };
                columns[0].push(aligned);
                columns[1].push(f(p.biased_exponent));
                columns[2].push(f(key & 0xFF));
                columns[3].push(f(key >> 8));
                columns[4].push(f(p.biased_binade()));
            }
            // TamedStark's squared-comparison power: key `t` decodes `D = 2*(t - 1024)`, the
            // value is `2^min(max(D,0), 80)` as base-2^16 limbs (exactly one limb nonzero —
            // a power of two).
            LutTable::XfPow2 => {
                let d = 2 * (key as i64 - XFPOW2_ZERO_POINT as i64);
                let a = (d.max(0) as u64).min(XFPOW2_CAP);
                for u in 0..XFPOW2_LIMBS as u64 {
                    columns[u as usize].push(if a / 16 == u { f(1 << (a % 16)) } else { F::ZERO });
                }
            }
            // Jackpot check 4's significand-product width: the bit length of the 16-bit key
            // and its nonzero flag (key 0 holds the (0, 0) sentinel).
            LutTable::Width16 => {
                columns[0].push(f(sig_width(key)));
                columns[1].push(f(sig_nonzero(key)));
            }
            // Jackpot check 4's fixed-point log: `floor(64 * log2 key)` (0 at key 0).
            LutTable::Log16 => columns[0].push(f(log2_fixed(key))),
        }
    }
    columns
}

/// Generates the table's full precommitted column block (`columns[c][row]`, the leading
/// [`num_precommitted_columns`] of its trace): the consensus-frozen data behind the setup
/// commitment. Deterministic; padding repeats valid entries (module docs).
pub fn lut_precommitted_values<F: Field>(table: LutTable) -> Vec<Vec<F>> {
    let height = lut_height(table);
    let live = slot_height(table);
    let mut cols: Vec<Vec<F>> = Vec::with_capacity(num_precommitted_columns(table));
    match table {
        // The stored key tuple is the whole precommitted block (live == height: both are
        // powers of two, so there is no padding).
        LutTable::Bytes2 | LutTable::Pair128 => cols.extend(generate::<F>(table, 0)),
        // Sub-height (or exactly-full) small tables: a saturated key column `min(i, live - 1)`
        // and the value columns padded alike — padding repeats the last live row's fact.
        LutTable::Int8Dec | LutTable::ExpInfo | LutTable::Clamp22 | LutTable::Pow2D | LutTable::Pow2Gb => {
            cols.push((0..height).map(|i| F::from_canonical_usize(i.min(live - 1))).collect());
            for mut col in generate::<F>(table, 0) {
                let last = *col.last().unwrap();
                col.resize(height, last);
                cols.push(col);
            }
        }
        // WIDTH32's key domain is [1, 32] — a *shifted* ramp (no key 0 row exists, so no
        // multiplicity can serve a zero width claim); live == height, no padding.
        LutTable::Width32 => {
            cols.push((0..height).map(|i| F::from_canonical_usize(i + 1)).collect());
            cols.extend(generate::<F>(table, 0));
        }
        // Ramp-keyed tables fill their height exactly (live == height, a power of two).
        _ => {
            cols.push((0..height).map(F::from_canonical_usize).collect());
            match table {
                LutTable::Range16 => {} // the ramp is the whole table
                LutTable::Qcast | LutTable::Div448 | LutTable::XfPow2 | LutTable::Width16 | LutTable::Log16 => {
                    cols.extend(generate::<F>(table, 0))
                }
                // Folded: the per-slot aligned column in slot-major order, then the
                // slot-independent columns (product decode facts — computed from the key
                // alone; asserted in tests).
                LutTable::B200Align => {
                    let mut slot0 = generate::<F>(table, 0);
                    let shared = slot0.split_off(1);
                    cols.extend(slot0);
                    for slot in 1..num_slots(table) {
                        let mut columns = generate::<F>(table, slot);
                        columns.truncate(1);
                        cols.extend(columns);
                    }
                    cols.extend(shared);
                }
                // Folded: all four value columns per slot.
                LutTable::RneRnd => {
                    for slot in 0..num_slots(table) {
                        cols.extend(generate::<F>(table, slot));
                    }
                }
                _ => unreachable!("handled above"),
            }
        }
    }
    debug_assert_eq!(cols.len(), num_precommitted_columns(table));
    debug_assert!(cols.iter().all(|c| c.len() == height));
    cols
}

/// Assembles one LUT AIR's full constraint-view trace: the precommitted block followed by the
/// per-slot multiplicity columns (e.g. from [`super::witness::LutMultiplicities::table_columns`]). Under the
/// batch commitment only the multiplicity tail is committed per proof — the leading block is
/// served by [`lut_preprocessed_data`]'s setup oracle; both halves address identically inside
/// constraints and CTLs.
pub fn lut_trace<F: Field>(table: LutTable, multiplicities: Vec<Vec<F>>) -> Vec<PolynomialValues<F>> {
    assert_eq!(multiplicities.len(), num_slots(table));
    for col in &multiplicities {
        assert_eq!(col.len(), lut_height(table));
    }
    let mut cols = lut_precommitted_values::<F>(table);
    cols.extend(multiplicities);
    cols.into_iter().map(PolynomialValues::new).collect()
}

/// One committed LUT's AIR: a pure lookup table of `WIDTH` columns — the precommitted block,
/// then the per-slot multiplicity columns. It has **no constraints of its own**: the
/// precommitted columns are bound by the consensus setup cap, and the multiplicity columns
/// only by the table's CTL channel balance ([`super::super::ctl::lut_cross_table_lookups`]).
///
/// `WIDTH` must equal [`lut_num_columns`]`(table)` (asserted by [`Self::new`]; the per-table
/// aliases below pin it) — the `Stark` trait wants the column count as a compile-time constant.
#[derive(Clone, Copy, Debug)]
pub struct LutStark<F, const D: usize, const WIDTH: usize> {
    pub table: LutTable,
    _phantom: PhantomData<F>,
}

impl<F, const D: usize, const WIDTH: usize> LutStark<F, D, WIDTH> {
    pub fn new(table: LutTable) -> Self {
        assert_eq!(WIDTH, lut_num_columns(table), "wrong AIR width for {table:?}");
        Self {
            table,
            _phantom: PhantomData,
        }
    }
}

impl<F: RichField + Extendable<D>, const D: usize, const WIDTH: usize> Stark<F, D> for LutStark<F, D, WIDTH> {
    type EvaluationFrame<FE, P, const D2: usize>
        = StarkFrame<P, P::Scalar, WIDTH, 0>
    where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>;
    type EvaluationFrameTarget = StarkFrame<ExtensionTarget<D>, ExtensionTarget<D>, WIDTH, 0>;

    fn eval_packed_generic<FE, P, const D2: usize>(
        &self,
        _vars: &Self::EvaluationFrame<FE, P, D2>,
        _yield_constr: &mut ConstraintConsumer<P>,
    ) where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>,
    {
    }

    fn eval_ext_circuit(
        &self,
        _builder: &mut CircuitBuilder<F, D>,
        _vars: &Self::EvaluationFrameTarget,
        _yield_constr: &mut RecursiveConstraintConsumer<F, D>,
    ) {
    }

    fn constraint_degree(&self) -> usize {
        3
    }

    fn requires_ctls(&self) -> bool {
        true
    }
}

/// The AIRs at their exact widths (the batch instantiates sixteen of these, in [`LUT_TABLES`]
/// order).
pub type RneRndStark<F, const D: usize> = LutStark<F, D, { lut_num_columns(LutTable::RneRnd) }>;
pub type Range16Stark<F, const D: usize> = LutStark<F, D, { lut_num_columns(LutTable::Range16) }>;
pub type Bytes2Stark<F, const D: usize> = LutStark<F, D, { lut_num_columns(LutTable::Bytes2) }>;
pub type QcastStark<F, const D: usize> = LutStark<F, D, { lut_num_columns(LutTable::Qcast) }>;
pub type Div448Stark<F, const D: usize> = LutStark<F, D, { lut_num_columns(LutTable::Div448) }>;
pub type Pair128Stark<F, const D: usize> = LutStark<F, D, { lut_num_columns(LutTable::Pair128) }>;
pub type Clamp22Stark<F, const D: usize> = LutStark<F, D, { lut_num_columns(LutTable::Clamp22) }>;
pub type Int8DecStark<F, const D: usize> = LutStark<F, D, { lut_num_columns(LutTable::Int8Dec) }>;
pub type ExpInfoStark<F, const D: usize> = LutStark<F, D, { lut_num_columns(LutTable::ExpInfo) }>;
pub type Pow2DStark<F, const D: usize> = LutStark<F, D, { lut_num_columns(LutTable::Pow2D) }>;
pub type B200AlignStark<F, const D: usize> = LutStark<F, D, { lut_num_columns(LutTable::B200Align) }>;
pub type Pow2GbStark<F, const D: usize> = LutStark<F, D, { lut_num_columns(LutTable::Pow2Gb) }>;
pub type Width32Stark<F, const D: usize> = LutStark<F, D, { lut_num_columns(LutTable::Width32) }>;
pub type XfPow2Stark<F, const D: usize> = LutStark<F, D, { lut_num_columns(LutTable::XfPow2) }>;
pub type Width16Stark<F, const D: usize> = LutStark<F, D, { lut_num_columns(LutTable::Width16) }>;
pub type Log16Stark<F, const D: usize> = LutStark<F, D, { lut_num_columns(LutTable::Log16) }>;

/// One LUT AIR at its table's exact width (the width is a compile-time constant per table,
/// so the dispatch is a static match).
pub(crate) fn boxed_lut_stark<F: RichField + Extendable<D>, const D: usize>(table: LutTable) -> Box<dyn BatchStark<F, D>> {
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

/// The `BatchStarkPreprocessedData::new` inputs for a batch of `num_tables` tables in which
/// LUT `i` (of [`LUT_TABLES`]) is table `table_positions[i]`:
/// `(values_per_table, columns_per_table)` — each LUT's precommitted columns at its position
/// (empty elsewhere). Positions must be strictly increasing (LUT heights descend in
/// [`LUT_TABLES`] order and the batch orders tables by descending height).
pub fn lut_preprocessed_inputs<F: Field>(
    num_tables: usize,
    table_positions: [usize; NUM_LUT_TABLES],
) -> (Vec<Vec<PolynomialValues<F>>>, Vec<Vec<usize>>) {
    assert!(
        table_positions.windows(2).all(|w| w[0] < w[1]) && table_positions[NUM_LUT_TABLES - 1] < num_tables,
        "table positions must be strictly increasing and within the batch"
    );
    let mut values = vec![Vec::new(); num_tables];
    let mut columns = vec![Vec::new(); num_tables];
    for (i, &table) in LUT_TABLES.iter().enumerate() {
        values[table_positions[i]] = lut_precommitted_values::<F>(table)
            .into_iter()
            .map(PolynomialValues::new)
            .collect();
        columns[table_positions[i]] = (0..num_precommitted_columns(table)).collect();
    }
    (values, columns)
}

/// Precommits every LUT column except the per-proof multiplicities:
/// builds the setup-time `BatchStarkPreprocessedData` — LDEs, the batched
/// Merkle tree, and the cap that becomes a consensus constant. The prover copies this data
/// into every proof's preprocessed oracle; the verifier checks proofs against the cap;
/// neither regenerates or recommits the tables per job.
pub fn lut_preprocessed_data<F, C, const D: usize>(
    num_tables: usize,
    table_positions: [usize; NUM_LUT_TABLES],
    config: &StarkConfig,
    timing: &mut TimingTree,
) -> BatchStarkPreprocessedData<F, C, D>
where
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
{
    let (values, columns) = lut_preprocessed_inputs::<F>(num_tables, table_positions);
    BatchStarkPreprocessedData::new(values, columns, config, timing)
}
