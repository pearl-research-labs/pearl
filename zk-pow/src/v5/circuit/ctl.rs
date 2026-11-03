//! Cross-table lookups of the FP16 / A100 multi-STARK batch.
//!
//! The FP16 batch is wire-separate from the FP8 system (it does not extend the FP8 `Table`/`Device`
//! enums). It has eight main tables — the A100 matmul AIR (0), the rho/breakpoint policy AIR (1), the
//! XorFold lottery mixer (2), the Blake3 jackpot commitment (3), the three quantization-chain
//! AIRs: the per-row scale derivation (4), the fused noisy-quant G1/G3 roundings (5), and the
//! single-rounding G2 FMA (6), and the noise matmul AIR (`N = E@F^T`, 7) — plus the five committed
//! LUTs their auxiliary columns are served by.
//! Each LUT is its own AIR of the batch; its channel's looking side collects every instance of all
//! main tables' inventories and its looked side is the LUT's slots filtered by their per-proof
//! multiplicity columns. This is what makes each main STARK's own verification enforce its
//! decode/shift/width/byte-range auxiliary columns.
//!
//! On top of the LUT channels there are seven main-table channels: the matmul -> policy census
//! import, the matmul -> XorFold **cell results** (binding the proven output tile to the folded
//! lottery lanes), the XorFold -> Blake3 **lottery words** (binding the 16 lane outputs to the
//! jackpot message, hence to the `HASH_JACKPOT` public input), the two internal quant-chain
//! channels (the row-scale -> noisy-quant **scales** import and the G1/G3 <-> G2 **FMA pairing**),
//! and the two operand-provenance channels (increments 6c/6d) that close audit Finding 3: the
//! Blake3 -> noisy-quant **operand bytes** (committed byte pair `== raw`, keyed by byte offset) and
//! the matmul -> noisy-quant **operand codes** (matmul code `== noised out`, keyed by element index
//! with the `w`/`h` reuse multiplicity). The first three bind the matmul output tile to
//! `HASH_JACKPOT`; the next two prove the noised-operand arithmetic `noised = Q(alpha*raw + beta*N)`
//! end to end in-batch; the last two bind that one operand set end to end — committed bytes (Blake3)
//! `== raw` (quant) `--Q-->` `out` (quant) `== matmul operand codes` — so the matmul provably
//! multiplies the noised quantization of the committed operands. The noise `N` is now bound too
//! (increment 6e-1): a second matmul instance (table 7) proves `N = E@F^T` in-batch and the
//! noise-word channel binds its cell results to the noise noisy-quant consumes (keyed by element
//! index). The noise derivation itself is now bound too (increment 6e-3, CLOSED): the forked BLAKE3
//! AIR (`blake3_fp16_stark`) egresses each noise line's keyed-XOF bytes and the noise-BLAKE3 egress
//! CTL binds them to the seeds derived from the committed operand roots, so `E`/`F` are no longer
//! free witness and the noise is not grindable.
//!
//! The committed LUTs reuse the FP8 [`LutStark`](crate::v4::circuit::luts::LutStark) machinery:
//! `RANGE16`, `WIDTH32` and `BYTES2` are the very FP8 tables (`BYTES2` serves the Blake3 AIR's
//! message-byte range checks); `FP16DECODE` and `FP16POW2` are new [`LutTable`] variants (serving
//! FP16's `2^16`-key operand decode and its wider alignment power-of-two) that no FP8 device commits,
//! so the FP8 consensus layout is untouched.

use plonky2::field::types::Field;
use starky::cross_table_lookup::{CrossTableLookup, TableIdx, TableWithColumns};
use starky::lookup::{Column, Filter};

use super::blake3_commit::{ctl_lottery_words_looking_blake3, ctl_operand_bytes_looking_blake3};
use super::blake3_fp16_stark::ctl::{blake3_lut_lookups as noise_blake3_lut_lookups, ctl_cv_egress_looking_blake3};
use super::matmul_a100_stark::columns::MATMUL_A100_COL_MAP;
use super::matmul_a100_stark::ctl::{
    ctl_cell_results_looked_matmul, ctl_cell_results_looked_matmul_offset, ctl_operand_codes_looking_matmul,
    ctl_operand_codes_looking_matmul_offset, matmul_a100_lut_lookups,
};
use super::noise_stark::ctl::{ctl_noise_egress_pair_looked, ctl_noise_operands_looked, noise_lut_lookups};
use super::noisy_quant_fma_stark::ctl::{ctl_fma_pairing_looking, fma_lut_lookups};
use super::noisy_quant_stark::columns::NOISY_QUANT_COL_MAP;
use super::noisy_quant_stark::ctl::{
    ctl_fma_hook_looking, ctl_noise_word_looked_noisy_quant, ctl_operand_bytes_looked_noisy_quant,
    ctl_operand_codes_looked_noisy_quant, noisy_quant_lut_lookups,
};
use super::policy_stark::columns::POLICY_A100_COL_MAP;
use super::policy_stark::ctl::policy_a100_lut_lookups;
use super::row_scale_stark::columns::ROW_SCALE_COL_MAP;
use super::row_scale_stark::ctl::row_scale_lut_lookups;
use super::xor_fold_stark::ctl::{
    ctl_cell_results_looking_xor_fold, ctl_lottery_words_looked_xor_fold, xor_fold_lut_lookups,
};
use crate::v4::circuit::blake3_stark::ctl::blake3_lut_lookups;
use crate::v4::circuit::luts::LutTable;
use crate::v4::circuit::luts::columns::num_slots;
use crate::v4::circuit::luts::ctl::LutLookup;
use crate::v4::circuit::luts::ctl_looked_lut_slot;

/// The A100 matmul AIR is table 0 of the FP16 batch.
pub const MATMUL_A100_TABLE: usize = 0;

/// The FP16 rho/breakpoint policy AIR is table 1 of the FP16 batch.
pub const POLICY_A100_TABLE: usize = 1;

/// The FP16 XorFold lottery-mixer AIR is table 2 of the FP16 batch.
pub const XOR_FOLD_A100_TABLE: usize = 2;

/// The FP16 Blake3 jackpot-commitment AIR is table 3 of the FP16 batch.
pub const BLAKE3_A100_TABLE: usize = 3;

/// The FP16 per-row scale-derivation AIR is table 4 of the FP16 batch (quant chain, increment 6b).
pub const ROW_SCALE_A100_TABLE: usize = 4;

/// The FP16 fused noisy-quantization AIR (G1 + G3) is table 5 of the FP16 batch (increment 6b).
pub const NOISY_QUANT_A100_TABLE: usize = 5;

/// The FP16 single-rounding FMA AIR (G2) is table 6 of the FP16 batch (increment 6b).
pub const NOISY_QUANT_FMA_A100_TABLE: usize = 6;

/// The A-side FP16 noise matmul AIR (`N_A = E_A @ F_A^T`, output `h x k`, inner `r`) is table 7 of
/// the FP16 batch (increment 6e-2). A [`super::matmul_a100_stark::MatmulStarkA100`] instance whose
/// proven cell results are the A-side noise matrix noisy-quant's G1 rounding consumes; the noise-word
/// CTL binds them to noisy-quant's `(NOISE_LO, NOISE_HI)`. Its `E_A`/`F_A` operands are bound to
/// NoiseStark's proven normalized lines by the E/F-operand channel (6e-2), so they are the real,
/// distinct seed-derived lines (seed-keyed-XOF binding is 6e-3).
pub const NOISE_MATMUL_A_A100_TABLE: usize = 7;

/// The B-side FP16 noise matmul AIR (`N_B = E_B @ F_B^T`, output `w x k`, inner `r`) is table 8 of the
/// FP16 batch (increment 6e-2). Restoring the plaintext's DISTINCT `F_A`/`F_B` (6e-1 used one shared
/// `F`): B keys `F_B` off `Side::B`, so `N_B` matches `sample_noise` bit-for-bit. Its cell results
/// bind to noisy-quant's B-side noise words (cell-id offset `h*k`), its `E_B`/`F_B` operands to
/// NoiseStark (index offset `(h+k)*r`).
pub const NOISE_MATMUL_B_A100_TABLE: usize = 8;

/// The FP16 noise-line normalization AIR (NoiseStark) is table 9 of the FP16 batch (increment 6e-2).
/// It proves `normalize_line` for every noise line of the tile (`E_A`, `F_A`, `E_B`, `F_B`) and binds
/// its normalized FP16 output to the two noise matmuls' `E`/`F` operands. In increment 6e-3c its raw
/// XOF bytes are no longer free witness: the egress-pair CTL binds them to the noise-BLAKE3 table.
pub const NOISE_STARK_A100_TABLE: usize = 9;

/// The FP16 noise-BLAKE3 derivation AIR (forked engine) is table 10 of the FP16 batch (increment
/// 6e-3c). It recomputes every noise line's keyed-BLAKE3-XOF bytes in-circuit from the noise seeds
/// (its `KEY_A`/`KEY_B` public inputs): two subkey compressions then `h+w+2k` per-line compressions,
/// each keyed by the line's subkey and egressing its `cv_out`. The egress-pair CTL binds those
/// `cv_out` limbs to NoiseStark's XOF-byte inputs, so the noise is no longer grindable through those
/// bytes. Every message block is pinned to its public constant via the engine's message-word pins.
pub const NOISE_BLAKE3_A100_TABLE: usize = 10;

/// Number of FP16 main (non-LUT) tables: the A100 matmul AIR, the policy AIR, the XorFold lottery
/// mixer, the Blake3 jackpot commitment, the three quantization-chain AIRs (row-scale, G1/G3
/// noisy-quant, G2 FMA), the two noise matmul AIRs (`N_A`/`N_B`), the NoiseStark line AIR, and the
/// noise-BLAKE3 derivation AIR (6e-3c).
pub const NUM_FP16_MAIN_TABLES: usize = 11;

/// The committed LUTs of the FP16 batch, in descending committed height (the batch order; FRI
/// folds by height internally). `FP16DECODE`/`RANGE16`/`BYTES2` are `2^16` tall, `FP16POW2` is
/// `256`, `WIDTH32` is `32`. `BYTES2` serves the Blake3 AIR's message-byte range checks (the
/// `blake3_lut_lookups` BYTES2 instances); it is the FP8 byte-pair table, committed by no FP8 device
/// change here.
pub const FP16_LUT_TABLES: [LutTable; NUM_FP16_LUT_TABLES] = [
    LutTable::Fp16Decode,
    LutTable::Range16,
    LutTable::Bytes2,
    LutTable::Fp16Pow2,
    LutTable::Width32,
];

/// Number of committed LUT tables in the FP16 batch.
pub const NUM_FP16_LUT_TABLES: usize = 5;

/// Total tables of the FP16 batch: the eight main tables, then the five LUTs.
pub const NUM_FP16_TABLES: usize = NUM_FP16_MAIN_TABLES + NUM_FP16_LUT_TABLES;

/// Number of CTL channels: one per committed LUT, plus the nine main-table channels (matmul ->
/// policy census import, matmul -> XorFold cell results, XorFold -> Blake3 lottery words, the two
/// internal quant-chain channels (row-scale -> noisy-quant scales, G1/G3 <-> G2 FMA pairing), the
/// two operand-provenance channels (Blake3 -> noisy-quant operand bytes [6c], matmul -> noisy-quant
/// operand codes [6d]), the noise-word channel (both noise matmuls -> noisy-quant `N` [6e-1/6e-2]),
/// and the E/F-operand channel (both noise matmuls' `E`/`F` -> NoiseStark normalized lines [6e-2])).
pub const NUM_FP16_CTL_CHANNELS: usize = NUM_FP16_LUT_TABLES + 10;

/// Batch table index of the `i`-th committed LUT (LUTs follow the main tables).
pub const fn fp16_lut_table_idx(i: usize) -> usize {
    NUM_FP16_MAIN_TABLES + i
}

/// The matmul -> policy census-import cross-table lookup (gap 3 linkage). The matmul AIR exports,
/// per live group step, the tuple `(operand_index_base_a, operand_index_base_b, group_breakpoint,
/// sum products_truncated_flag)`; the policy AIR imports the same tuple on its matching row. The
/// `(base_a, base_b)` pair is unique per live row (base_a fixes `(r, j)`, base_b fixes `(c, j)`),
/// so the multiset equality forces the policy's per-step census to equal the matmul's
/// bit-exactly — the policy can no longer score a census different from the one the matmul proved.
/// Both sides are filtered to live rows (`1 - is_padding`).
pub fn census_import_ctl<F: Field>() -> CrossTableLookup<F> {
    let m = &MATMUL_A100_COL_MAP;
    let p = &POLICY_A100_COL_MAP;
    let one = F::ONE;
    let neg = -F::ONE;

    // Matmul (looking): keys (base_a, base_b); values (group_breakpoint, sum of the 8 lane
    // products-truncated flags). The sum is a plain linear combination of the flag columns.
    let matmul_pt_sum = Column::linear_combination(m.products_truncated_flag.iter().map(|&c| (c, one)));
    let looking = TableWithColumns::new(
        TableIdx::from(MATMUL_A100_TABLE),
        vec![
            Column::single(m.operand_index_base_a),
            Column::single(m.operand_index_base_b),
            Column::single(m.group_breakpoint),
            matmul_pt_sum,
        ],
        Filter::from_column(Column::linear_combination_with_constant([(m.is_padding, neg)], one)),
    );

    // Policy (looked): the same tuple on the same grid.
    let looked = TableWithColumns::new(
        TableIdx::from(POLICY_A100_TABLE),
        vec![
            Column::single(p.operand_index_base_a),
            Column::single(p.operand_index_base_b),
            Column::single(p.group_breakpoint),
            Column::single(p.products_truncated),
        ],
        Filter::from_column(Column::linear_combination_with_constant([(p.is_padding, neg)], one)),
    );

    CrossTableLookup::new(vec![looking], vec![looked])
}

/// The internal quant-chain scales channel, keyed by operand-row index (the FP8 `InputQuant`
/// `element_index`/offset pattern): the row-scale AIR exports, per live row, the tuple
/// `(OPERAND_ROW_INDEX, alpha, beta)`; the noisy-quant AIR imports the same tuple on every live
/// element. Both sides are filtered to live rows (`1 - IS_PAD`). For each operand row `i`, the
/// row-scale table replicates `(i, alpha_i, beta_i)` on each of its `k` live block rows and the
/// noisy-quant table consumes `(i, alpha_i, beta_i)` on each of its `k` live elements, so the
/// multiset balance forces every element to consume *its own* operand row's proven `(alpha, beta)`.
/// The shared index space (`A` rows `0..h`, `B` rows `h..h+w`) covers BOTH operands in one channel.
pub fn scales_import_ctl<F: Field>() -> CrossTableLookup<F> {
    let rs = &ROW_SCALE_COL_MAP;
    let nq = &NOISY_QUANT_COL_MAP;
    let one = F::ONE;
    let neg = -F::ONE;
    // noisy-quant (looking): `(operand_row_index, alpha, beta)` per live element.
    let looking = TableWithColumns::new(
        TableIdx::from(NOISY_QUANT_A100_TABLE),
        Column::singles([nq.operand_row_index, nq.alpha, nq.beta]).collect(),
        Filter::from_column(Column::linear_combination_with_constant([(nq.is_pad, neg)], one)),
    );
    // row-scale (looked): `(operand_row_index, alpha_code, beta_code)` per live row.
    let looked = TableWithColumns::new(
        TableIdx::from(ROW_SCALE_A100_TABLE),
        Column::singles([rs.operand_row_index, rs.alpha_code, rs.beta_code]).collect(),
        Filter::from_column(Column::linear_combination_with_constant([(rs.is_pad, neg)], one)),
    );
    CrossTableLookup::new(vec![looking], vec![looked])
}

/// Every committed-LUT instance the FP16 batch consumes, tagged with the batch table index of its
/// consumer: the matmul AIR's full inventory at [`MATMUL_A100_TABLE`], the policy AIR's RANGE16
/// inventory at [`POLICY_A100_TABLE`], the XorFold AIR's RC16 inventory at [`XOR_FOLD_A100_TABLE`],
/// the Blake3 AIR's BYTES2/RC16 inventory at [`BLAKE3_A100_TABLE`], and the three quantization-chain
/// AIRs' FP16DECODE/RANGE16/FP16POW2/WIDTH32 inventories (all shared FP16-batch LUTs).
fn lut_inventories<F: Field>() -> [(usize, Vec<LutLookup<F>>); NUM_FP16_MAIN_TABLES] {
    [
        (MATMUL_A100_TABLE, matmul_a100_lut_lookups::<F>()),
        (POLICY_A100_TABLE, policy_a100_lut_lookups::<F>()),
        (XOR_FOLD_A100_TABLE, xor_fold_lut_lookups::<F>()),
        (BLAKE3_A100_TABLE, blake3_lut_lookups::<F>()),
        (ROW_SCALE_A100_TABLE, row_scale_lut_lookups::<F>()),
        (NOISY_QUANT_A100_TABLE, noisy_quant_lut_lookups::<F>()),
        (NOISY_QUANT_FMA_A100_TABLE, fma_lut_lookups::<F>()),
        // The two noise matmuls reuse the exact A100 matmul inventory (same decode/shift/width/range
        // auxiliary columns), so their decode/alignment columns are served by the same committed LUTs.
        (NOISE_MATMUL_A_A100_TABLE, matmul_a100_lut_lookups::<F>()),
        (NOISE_MATMUL_B_A100_TABLE, matmul_a100_lut_lookups::<F>()),
        // NoiseStark's per-line normalization inventory (RC16/PAIR128/POW2D) — shared FP16-batch LUTs.
        (NOISE_STARK_A100_TABLE, noise_lut_lookups::<F>()),
        // The noise-BLAKE3 derivation table reuses the forked engine's BYTES2/RC16 inventory (the same
        // message-byte and egress-limb range checks), served by the shared FP16-batch LUTs (6e-3c).
        (NOISE_BLAKE3_A100_TABLE, noise_blake3_lut_lookups::<F>()),
    ]
}

/// All FP16 cross-table lookups: one channel per committed LUT, then the eight main-table channels
/// (census import, cell results, lottery words, scales import, FMA pairing, operand bytes [6c],
/// operand codes [6d], noise word [6e-1]). Each LUT channel's looking side collects every
/// instance of that table across *all* main tables' inventories (so the shared RANGE16 table
/// serves the matmul's result limbs, the policy's gate-slack / per-step-count limbs, and the
/// XorFold mixer limbs in one channel, and BYTES2 serves the Blake3 message bytes), and its looked
/// side is the LUT's slots at [`fp16_lut_table_idx`]. Every LUT must have at least one consumer (a
/// consumerless table is a wiring bug).
pub fn all_cross_table_lookups<F: Field>(h: usize, w: usize, k: usize) -> Vec<CrossTableLookup<F>> {
    // `w` enters only through the noise-word B offset `h*k` and the E/F B offset `(h+k)*r`, both of
    // which are expressed via `h`/`k`; it is kept in the signature for geometric symmetry.
    let _ = w;
    let r = super::driver::FP16_QUANT_R;
    let inventories = lut_inventories::<F>();
    let mut ctls: Vec<CrossTableLookup<F>> = FP16_LUT_TABLES
        .iter()
        .enumerate()
        .map(|(i, &table)| {
            let looking: Vec<TableWithColumns<F>> = inventories
                .iter()
                .flat_map(|(table_idx, lookups)| {
                    let table_idx = *table_idx;
                    lookups.iter().filter(|l| l.table == table).map(move |l| {
                        TableWithColumns::new(
                            TableIdx::from(table_idx),
                            l.keys.iter().chain(&l.values).cloned().collect(),
                            l.filter.clone(),
                        )
                    })
                })
                .collect();
            assert!(!looking.is_empty(), "{table:?} has no looking instances");
            let looked = (0..num_slots(table))
                .map(|slot| ctl_looked_lut_slot(TableIdx::from(fp16_lut_table_idx(i)), table, slot))
                .collect();
            CrossTableLookup::new(looking, looked)
        })
        .collect();
    // The three main-table channels: the matmul -> policy census import, the matmul -> XorFold cell
    // results (binding the proven output tile to the folded lottery lanes), and the XorFold ->
    // Blake3 lottery words (binding the 16 lane outputs to the jackpot message, hence HASH_JACKPOT).
    ctls.push(census_import_ctl::<F>());
    ctls.push(CrossTableLookup::new(
        vec![ctl_cell_results_looking_xor_fold::<F>(XOR_FOLD_A100_TABLE)],
        vec![ctl_cell_results_looked_matmul::<F>(MATMUL_A100_TABLE)],
    ));
    ctls.push(CrossTableLookup::new(
        ctl_lottery_words_looking_blake3::<F>(BLAKE3_A100_TABLE),
        vec![ctl_lottery_words_looked_xor_fold::<F>(XOR_FOLD_A100_TABLE)],
    ));
    // The two internal quant-chain channels (increment 6b): the row-scale -> noisy-quant scales
    // import, and the G1/G3 noisy-quant <-> G2 FMA pairing. The pairing's shared tuple
    // `(alpha, raw, t_sign, t_mant, t_exp, noised_lo, noised_hi)` binds G1's proven `t` (fed to G2)
    // and G3's cast input `noised` (proven by G2) to the single-rounding FMA. Both sides are filtered
    // to live rows.
    ctls.push(scales_import_ctl::<F>());
    ctls.push(CrossTableLookup::new(
        vec![ctl_fma_hook_looking::<F>(NOISY_QUANT_A100_TABLE)],
        vec![ctl_fma_pairing_looking::<F>(NOISY_QUANT_FMA_A100_TABLE)],
    ));
    // The two operand-provenance channels (increments 6c/6d) that close audit Finding 3: the matmul
    // provably multiplies the NOISED quantization of the COMMITTED operands.
    // 6c — operand-bytes: Blake3's committed operand byte pairs (-> HASH_A/HASH_B) equal the quant
    // chain's `raw` inputs, keyed by byte offset `2*e`. So `raw` is no longer free witness: it is the
    // committed operand the noised codes derive from.
    ctls.push(CrossTableLookup::new(
        ctl_operand_bytes_looking_blake3::<F>(BLAKE3_A100_TABLE),
        vec![ctl_operand_bytes_looked_noisy_quant::<F>(NOISY_QUANT_A100_TABLE)],
    ));
    // 6d — noised->matmul: the matmul's operand codes equal the noisy-quant `out` column, keyed by the
    // shared element index, with the w/h reuse multiplicity on the looked side. So the matmul's
    // operands are no longer free: they are the proven noised quantization of the committed `raw`, and
    // cells of a row/column provably reuse the same operand element.
    ctls.push(CrossTableLookup::new(
        ctl_operand_codes_looking_matmul::<F>(MATMUL_A100_TABLE),
        vec![ctl_operand_codes_looked_noisy_quant::<F>(NOISY_QUANT_A100_TABLE)],
    ));
    // 6e-1/6e-2 — noise-word: the two noise matmuls' proven cell results (`N_A = E_A@F_A^T`, `h x k`;
    // `N_B = E_B@F_B^T`, `w x k`; both `CELL_ID = i*k + j`) equal the noisy-quant `(NOISE_LO,
    // NOISE_HI)` the G1 rounding consumes, keyed by the shared global element index. A-side cells map
    // directly (`[0, h*k)`); B-side cells are shifted by `h*k` into `[h*k, (h+w)*k)`. So the `beta*N`
    // noise is provably `E@F` with distinct `F_A`/`F_B`.
    ctls.push(CrossTableLookup::new(
        vec![
            ctl_cell_results_looked_matmul::<F>(NOISE_MATMUL_A_A100_TABLE),
            ctl_cell_results_looked_matmul_offset::<F>(NOISE_MATMUL_B_A100_TABLE, h * k),
        ],
        vec![ctl_noise_word_looked_noisy_quant::<F>(NOISY_QUANT_A100_TABLE)],
    ));
    // 6e-2 — E/F-operand: the two noise matmuls' `E`/`F` operand codes equal NoiseStark's proven
    // normalized line entries, keyed by the global entry index `line*r + entry`. NoiseStark lays its
    // lines in the order `[E_A, F_A, E_B, F_B]`: matmul A's own `[0, (h+k)*r)` operand index space
    // matches the leading `E_A`,`F_A` blocks directly (offset 0); matmul B's `[0, (w+k)*r)` space is
    // shifted by `(h+k)*r` onto the trailing `E_B`,`F_B` blocks. NoiseStark's looked side carries the
    // per-element reuse multiplicity (`k` for `E`, `h`/`w` for `F`), so each proven entry balances the
    // matmul operands' per-cell reuse. This binds `E`/`F` to the seed-derived normalized lines.
    let mut ef_looking = ctl_operand_codes_looking_matmul_offset::<F>(NOISE_MATMUL_A_A100_TABLE, 0);
    ef_looking.extend(ctl_operand_codes_looking_matmul_offset::<F>(NOISE_MATMUL_B_A100_TABLE, (h + k) * r));
    ctls.push(CrossTableLookup::new(
        ef_looking,
        vec![ctl_noise_operands_looked::<F>(NOISE_STARK_A100_TABLE)],
    ));
    // 6e-3c — noise-XOF egress: the noise-BLAKE3 table recomputes every line's keyed-BLAKE3-XOF bytes
    // in-circuit from the seeds (its pinned `KEY_A`/`KEY_B` public inputs) and egresses each line
    // compression's `cv_out` as 16 little-endian 16-bit limbs on keys `line*16 .. line*16 + 16`.
    // NoiseStark's egress-pair looked side recomposes its own two consecutive XOF bytes into the same
    // limbs on the same keys, so the multiset balance forces NoiseStark's normalized line's raw XOF
    // bytes to equal the seed-derived, material-pinned keyed-XOF — the noise's last free witness is
    // closed, so it is no longer grindable through those bytes.
    ctls.push(CrossTableLookup::new(
        ctl_cv_egress_looking_blake3::<F>(NOISE_BLAKE3_A100_TABLE),
        vec![ctl_noise_egress_pair_looked::<F>(NOISE_STARK_A100_TABLE)],
    ));
    ctls
}

/// The per-main-table committed-LUT inventories, for the prover's multiplicity accounting
/// (mirrors [`lut_inventories`] but keyed for the driver's [`crate::v4::circuit::luts::LutChecker`]).
pub fn fp16_lut_inventories<F: Field>() -> [(usize, Vec<LutLookup<F>>); NUM_FP16_MAIN_TABLES] {
    lut_inventories::<F>()
}

#[cfg(test)]
mod tests {
    use plonky2::field::goldilocks_field::GoldilocksField;
    use plonky2::field::polynomial::PolynomialValues;
    use plonky2::field::types::Field;
    use starky::cross_table_lookup::debug_utils::check_ctls;

    use super::*;
    use crate::v5::api::accumulate::GROUP as ACC_GROUP;
    use crate::v5::api::policy::{evaluate, replay_and_evaluate};
    use crate::v4::api::transcript::jackpot_key;
    use crate::v4::circuit::luts::{LutChecker, lut_trace};
    use crate::v5::circuit::driver::Fp16System;
    use crate::v5::circuit::matmul_a100_stark::columns::{MATMUL_A100_COL_MAP, NUM_MATMUL_A100_COLUMNS};
    use crate::v5::circuit::matmul_a100_stark::stark::MatmulStarkA100;
    use crate::v5::circuit::policy_stark::columns::{NUM_POLICY_A100_COLUMNS, POLICY_A100_COL_MAP};
    use crate::v5::circuit::policy_stark::stark::PolicyStarkA100;

    /// The full honest nine-table batch for a 4x4 tile (16 output cells, one per lottery lane),
    /// replicated from one accepting cell: the traces (canonical order) plus the per-table public
    /// inputs (only Blake3's are nonempty). Built via the driver so the output -> jackpot chain is
    /// wired exactly as a real proof.
    fn full_batch() -> (Vec<Vec<PolynomialValues<F>>>, Vec<Vec<F>>, usize) {
        use crate::v4::api::public_params::HashId;
        let (k, a0, b0) = accepting_tile();
        let (a, b) = (a0.repeat(4), b0.repeat(4));
        // Operand-tree keys / leaf size are passed in for now (seed-chain derivation is a later
        // increment); the CTL balance is independent of these choices.
        let system = Fp16System::<F, D>::new(4, 4, k, HashId::Blake3Chunk512, HashId::Blake3Chunk512);
        let (traces, blake3_pis) = system
            .generate_batch_traces(&a, &b, [0x11; 32], [0x22; 32], jackpot_key(&[0x5a; 32]))
            .expect("every honest instance must be served");
        let mut all_pis: Vec<Vec<F>> = vec![Vec::new(); NUM_FP16_TABLES];
        all_pis[BLAKE3_A100_TABLE] = blake3_pis;
        (traces.to_vec(), all_pis, k)
    }

    type F = GoldilocksField;
    const D: usize = 2;
    type S = MatmulStarkA100<F, D>;
    type P = PolicyStarkA100<F, D>;

    const VECTORS: &str = include_str!("../api/testdata/a100_dot_vectors.txt");

    /// An accepting `1 x 1 x k` tile from the GPU reference vectors: operands that clear the
    /// policy gate (so the policy slacks are nonnegative and its RANGE16 limbs are in-domain) and
    /// whose matmul output is normal (no subnormal branch).
    fn accepting_tile() -> (usize, Vec<u16>, Vec<u16>) {
        for line in VECTORS.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let t: Vec<&str> = line.split_whitespace().collect();
            let k: usize = t[0].parse().unwrap();
            if k % ACC_GROUP != 0 {
                continue;
            }
            let a: Vec<u16> = t[1..1 + k].iter().map(|x| x.parse().unwrap()).collect();
            let b: Vec<u16> = t[1 + k..1 + 2 * k].iter().map(|x| x.parse().unwrap()).collect();
            if replay_and_evaluate(&a, &b, 1, 1, k).1.accept {
                return (k, a, b);
            }
        }
        panic!("reference vectors must contain an accepting 1x1 tile");
    }

    fn mat_columns(rows: &[[F; NUM_MATMUL_A100_COLUMNS]]) -> Vec<PolynomialValues<F>> {
        (0..NUM_MATMUL_A100_COLUMNS)
            .map(|c| PolynomialValues::new(rows.iter().map(|r| r[c]).collect()))
            .collect()
    }

    fn policy_columns(rows: &[[F; NUM_POLICY_A100_COLUMNS]]) -> Vec<PolynomialValues<F>> {
        (0..NUM_POLICY_A100_COLUMNS)
            .map(|c| PolynomialValues::new(rows.iter().map(|r| r[c]).collect()))
            .collect()
    }

    /// Assembles the full FP16 batch trace set from (possibly tampered) matmul rows and a policy
    /// trace generated from the same operands: `[matmul, policy, Fp16Decode, Range16, Fp16Pow2,
    /// Width32]`. Runs `LutChecker` over both main tables' inventories; `Err` if any instance is
    /// not served by its committed table.
    fn build_batch(
        mat_rows: &[[F; NUM_MATMUL_A100_COLUMNS]],
        pol_rows: &[[F; NUM_POLICY_A100_COLUMNS]],
    ) -> Result<Vec<Vec<PolynomialValues<F>>>, String> {
        let mat_cols = mat_columns(mat_rows);
        let pol_cols = policy_columns(pol_rows);
        let mut checker = LutChecker::<F>::new(&FP16_LUT_TABLES);
        checker.check_trace(&matmul_a100_lut_lookups::<F>(), &mat_cols, &[], "MatmulA100")?;
        checker.check_trace(&policy_a100_lut_lookups::<F>(), &pol_cols, &[], "PolicyA100")?;
        let mut traces = vec![mat_cols, pol_cols];
        for table in FP16_LUT_TABLES {
            traces.push(lut_trace::<F>(table, checker.multiplicities.table_columns(table)));
        }
        Ok(traces)
    }

    #[test]
    fn fp16_ctls_assemble() {
        let ctls = all_cross_table_lookups::<F>(4, 4, 16);
        assert_eq!(
            ctls.len(),
            NUM_FP16_CTL_CHANNELS,
            "one channel per LUT, plus census import, cell results, lottery words, scales, FMA pairing, \
             operand bytes (6c), operand codes (6d), noise word (6e-1/6e-2), and E/F operands (6e-2)"
        );
        assert_eq!(
            NUM_FP16_TABLES, 16,
            "matmul, policy, xor_fold, blake3, row_scale, noisy_quant, noisy_quant_fma, noise_matmul_a, \
             noise_matmul_b, noise_stark, noise_blake3, five LUTs"
        );
    }

    #[test]
    fn lut_tables_descend_in_height() {
        use crate::v4::circuit::luts::lut_height;
        let heights: Vec<usize> = FP16_LUT_TABLES.iter().map(|&t| lut_height(t)).collect();
        assert!(heights.windows(2).all(|w| w[0] >= w[1]), "committed LUTs must descend: {heights:?}");
    }

    /// The headline of gaps 1 and 3: every decode/shift/width auxiliary column of the matmul AIR
    /// and every RANGE16 fact of the policy AIR is served by a committed table (`LutChecker`), and
    /// every CTL channel balances — the four LUT channels *and* the matmul -> policy census-import
    /// channel (`check_ctls`). So the batch's own verification enforces those columns and binds the
    /// policy census to the matmul's.
    #[test]
    fn honest_batch_serves_and_balances_every_channel() {
        let (traces, all_pis, k) = full_batch();
        check_ctls(&traces, &all_pis, &all_cross_table_lookups::<F>(4, 4, k), &Default::default());
    }

    /// Negative: forging an auxiliary column the LUTs are responsible for must be caught by the
    /// committed tables (a value mismatch or an out-of-domain key).
    #[test]
    fn forged_auxiliary_columns_are_rejected_by_the_luts() {
        let (k, a, b) = accepting_tile();
        let mat = S::new(1, 1, k).generate_trace(&a, &b, None);
        let pol = P::new(1, 1, k).generate_trace(&a, &b);
        let m = &MATMUL_A100_COL_MAP;
        build_batch(&mat, &pol).expect("baseline honest trace must serve");

        let live = (0..mat.len())
            .find(|&r| {
                mat[r][m.is_padding] == F::ZERO
                    && mat[r][m.group_nonempty] == F::ONE
                    && mat[r][m.group_sum_is_zero] == F::ZERO
            })
            .expect("fixture has a live non-cancelling row");

        let forge = |col: usize, delta: F| -> Result<(), String> {
            let mut t = mat.clone();
            t[live][col] += delta;
            build_batch(&t, &pol).map(|_| ())
        };

        assert!(forge(m.sig_a[1], F::ONE).is_err(), "forged sig_a must be rejected by FP16DECODE");
        assert!(forge(m.eps_b[1], F::ONE).is_err(), "forged eps_b must be rejected by FP16DECODE");
        assert!(forge(m.lane_shift_power[1], F::ONE).is_err(), "forged lane_shift_power must be rejected by FP16POW2");
        assert!(
            forge(m.product_biased_exp[1], F::from_canonical_u64(1000)).is_err(),
            "a product exponent above the window max must leave the FP16POW2 key domain"
        );
        assert!(forge(m.truncation_power, F::ONE).is_err(), "forged truncation_power must be rejected by WIDTH32");
        // MA12: a floor-witness hi limb pushed to >= 2^10 (a field-fraction alias of the floor
        // quotient) leaves the `2^6`-scaled RANGE16 domain — the limb-range soundness fix.
        assert!(
            forge(m.aligned_mag_hi[1], F::from_canonical_u64(1024)).is_err(),
            "an out-of-range floor-witness hi limb must be rejected by RANGE16"
        );

        // The multiset balance itself breaks: honest LUT multiplicities against a tampered matmul
        // value is a CTL imbalance that `check_ctls` panics on. (Full nine-table batch so every
        // channel — LUTs, census, cell-results, lottery-words — is present.)
        let (honest, all_pis, k) = full_batch();
        let mut tampered = honest.clone();
        let live = (0..tampered[MATMUL_A100_TABLE][m.is_padding].values.len())
            .find(|&r| {
                tampered[MATMUL_A100_TABLE][m.is_padding].values[r] == F::ZERO
                    && tampered[MATMUL_A100_TABLE][m.group_sum_is_zero].values[r] == F::ZERO
            })
            .expect("a live non-cancelling row");
        tampered[MATMUL_A100_TABLE][m.sig_a[1]].values[live] += F::ONE;
        let ctls = all_cross_table_lookups::<F>(4, 4, k);
        let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            check_ctls(&tampered, &all_pis, &ctls, &Default::default());
        }))
        .is_err();
        assert!(panicked, "a tampered matmul value against honest LUT multiplicities must unbalance a CTL channel");
    }

    /// Regression guard for the NORM_SIG normalized-range pin (the dropped B200 MB7 check).
    ///
    /// The MA7 width identity `GROUP_SUM_ABS * LIFTING_POWER = NORM_SIG * TRUNCATION_POWER +
    /// TRUNC_REM` only pins `GROUP_SUM_WIDTH` to the true bit-width when `NORM_SIG` is forced into
    /// `[2^23, 2^24)`. The looser MA12 split alone (`NORM_SIG_HI < 2^10`) lets a prover pick a false
    /// width, choose `TRUNC_REM != 0`, and so forge `RZ_DROPPED` (MA11) → `GROUP_BREAKPOINT` → the
    /// jackpot census (`f_bp` / `rho`). A `NORM_SIG_HI` outside `[128, 256)` but still inside the
    /// looser `2^6*hi < 2^16` domain is exactly the shape of such a forge; the committed RANGE16
    /// normalized-range instance must reject it.
    #[test]
    fn forged_norm_sig_width_is_rejected_by_the_normalized_range() {
        let (k, a, b) = accepting_tile();
        let mat = S::new(1, 1, k).generate_trace(&a, &b, None);
        let pol = P::new(1, 1, k).generate_trace(&a, &b);
        let m = &MATMUL_A100_COL_MAP;
        // Baseline: the honest (normalized) trace is served, so the new pin does not reject it.
        build_batch(&mat, &pol).expect("baseline honest trace must serve");

        let live = (0..mat.len())
            .find(|&r| mat[r][m.is_padding] == F::ZERO && mat[r][m.group_sum_is_zero] == F::ZERO)
            .expect("fixture has a live nonzero-sum row");

        // Each forged hi is below 128 (hi = 0) or at/above 256, i.e. outside the normalized band
        // `[128, 256)` — yet all stay inside the looser MA12 `2^6*hi < 2^16` (hi < 2^10) split that
        // the pre-fix inventory relied on, so only the new normalized-range check can catch them.
        for forged_hi in [0u64, 256, 512] {
            assert!((1u64 << 6) * forged_hi < (1 << 16), "forged hi stays inside the MA12 2^6-split domain");
            let mut t = mat.clone();
            t[live][m.norm_sig_hi] = F::from_canonical_u64(forged_hi);
            assert!(
                build_batch(&t, &pol).is_err(),
                "norm_sig_hi={forged_hi} is outside [128,256) and must be rejected by the normalized-range RANGE16"
            );
        }
    }

    /// The census-import channel binds the policy census to the matmul's: forging a policy
    /// per-step census value away from the matmul's (keeping the matmul honest) unbalances the
    /// `matmul -> policy` channel.
    #[test]
    fn forged_policy_census_breaks_the_import_channel() {
        let p = &POLICY_A100_COL_MAP;
        let (honest, all_pis, k) = full_batch();
        let live = (0..honest[POLICY_A100_TABLE][p.is_padding].values.len())
            .find(|&r| honest[POLICY_A100_TABLE][p.is_padding].values[r] == F::ZERO)
            .unwrap();

        let ctls = all_cross_table_lookups::<F>(4, 4, k);
        for col in [p.group_breakpoint, p.products_truncated] {
            let mut tampered = honest.clone();
            tampered[POLICY_A100_TABLE][col].values[live] += F::ONE;
            let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                check_ctls(&tampered, &all_pis, &ctls, &Default::default());
            }))
            .is_err();
            assert!(panicked, "a forged policy census value must unbalance the census-import channel");
        }
    }

    /// 6d — the noised->matmul operand-codes channel binds the matmul's operand codes to the quant
    /// chain's noised `out`. Forging the noisy-quant `OUT` column at a live element (keeping the
    /// matmul honest) unbalances the operand-codes channel — and ONLY that channel, since `OUT` is
    /// wired nowhere else (no LUT, no other CTL), so the panic is attributable to it.
    #[test]
    fn forged_noised_out_breaks_the_operand_codes_channel() {
        let nq = &NOISY_QUANT_COL_MAP;
        let (honest, all_pis, k) = full_batch();
        let live = (0..honest[NOISY_QUANT_A100_TABLE][nq.is_pad].values.len())
            .find(|&r| honest[NOISY_QUANT_A100_TABLE][nq.is_pad].values[r] == F::ZERO)
            .expect("a live noisy-quant element");
        let mut tampered = honest.clone();
        tampered[NOISY_QUANT_A100_TABLE][nq.out].values[live] += F::ONE;
        let ctls = all_cross_table_lookups::<F>(4, 4, k);
        let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            check_ctls(&tampered, &all_pis, &ctls, &Default::default());
        }))
        .is_err();
        assert!(panicked, "a forged noised OUT must unbalance the noised->matmul operand-codes channel");
    }

    /// 6e-1 — the noise-word channel binds the `N = E@F^T` noise matmul's proven cell results to the
    /// noise noisy-quant consumes. Forging the noisy-quant `NOISE_LO` at a live element, and (keeping
    /// the matmul honest) forging the noise matmul's `CELL_RESULT_F32_LO` at a finished cell, each
    /// unbalance the noise-word channel — `NOISE_LO`/`NOISE_HI` and the noise matmul's
    /// `CELL_RESULT_F32_*` are wired into no other CTL channel, so the panic is attributable to it.
    #[test]
    fn forged_noise_word_breaks_the_noise_channel() {
        use crate::v5::circuit::matmul_a100_stark::columns::MATMUL_A100_COL_MAP;
        let nq = &NOISY_QUANT_COL_MAP;
        let m = &MATMUL_A100_COL_MAP;
        let (honest, all_pis, k) = full_batch();
        let ctls = all_cross_table_lookups::<F>(4, 4, k);

        // (a) the consuming side: a forged noise word noisy-quant claims to consume.
        let nq_live = (0..honest[NOISY_QUANT_A100_TABLE][nq.is_pad].values.len())
            .find(|&r| honest[NOISY_QUANT_A100_TABLE][nq.is_pad].values[r] == F::ZERO)
            .expect("a live noisy-quant element");
        // (b) the producing side: a forged cell result of the noise matmul at a finished live cell.
        let noise_final = (0..honest[NOISE_MATMUL_A_A100_TABLE][m.is_cell_final].values.len())
            .find(|&r| {
                honest[NOISE_MATMUL_A_A100_TABLE][m.is_cell_final].values[r] == F::ONE
                    && honest[NOISE_MATMUL_A_A100_TABLE][m.is_padding].values[r] == F::ZERO
            })
            .expect("a finished live noise-matmul cell");

        for (table, col, row) in [
            (NOISY_QUANT_A100_TABLE, nq.noise_lo, nq_live),
            (NOISE_MATMUL_A_A100_TABLE, m.cell_result_f32_lo, noise_final),
        ] {
            let mut tampered = honest.clone();
            tampered[table][col].values[row] += F::ONE;
            let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                check_ctls(&tampered, &all_pis, &ctls, &Default::default());
            }))
            .is_err();
            assert!(panicked, "a forged noise word must unbalance the noise-word (6e-1) channel");
        }
    }

    /// 6c — the operand-bytes channel binds Blake3's committed operand bytes to the quant chain's
    /// `raw`. Forging the committed `UINT8_DATA` at a live operand-values row unbalances the
    /// operand-bytes channel (its committed byte pair no longer equals the quant `raw`), so a prover
    /// cannot commit one operand and quantize a different one.
    #[test]
    fn forged_committed_byte_breaks_the_operand_bytes_channel() {
        use crate::v4::circuit::blake3_stark::columns::BLAKE3_COL_MAP;
        let bm = &BLAKE3_COL_MAP;
        let (honest, all_pis, k) = full_batch();
        let live = (0..honest[BLAKE3_A100_TABLE][bm.is_int8_message].values.len())
            .find(|&r| honest[BLAKE3_A100_TABLE][bm.is_int8_message].values[r] == F::ONE)
            .expect("a live operand-values row");
        let mut tampered = honest.clone();
        tampered[BLAKE3_A100_TABLE][bm.uint8_data[0]].values[live] += F::ONE;
        let ctls = all_cross_table_lookups::<F>(4, 4, k);
        let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            check_ctls(&tampered, &all_pis, &ctls, &Default::default());
        }))
        .is_err();
        assert!(panicked, "a forged committed operand byte must unbalance the operand-bytes channel");
    }

    /// 6e-2 — the E/F-operand channel binds the two noise matmuls' `E`/`F` operand codes to
    /// NoiseStark's proven normalized line entries. Forging a NoiseStark line entry (`ENTRY_FP16`) at a
    /// live row unbalances the E/F channel: the proven normalized line no longer matches the matmul
    /// operand it is bound to. `ENTRY_FP16` is wired into no other CTL channel, so the panic is
    /// attributable to the E/F binding.
    #[test]
    fn forged_noise_line_breaks_the_ef_channel() {
        use crate::v5::circuit::noise_stark::columns::NOISE_COL_MAP;
        let (honest, all_pis, k) = full_batch();
        let ns = &NOISE_COL_MAP;
        let live = (0..honest[NOISE_STARK_A100_TABLE][ns.is_pad].values.len())
            .find(|&r| honest[NOISE_STARK_A100_TABLE][ns.is_pad].values[r] == F::ZERO)
            .expect("a live noise line entry");
        let mut tampered = honest.clone();
        tampered[NOISE_STARK_A100_TABLE][ns.entry_fp16].values[live] += F::ONE;
        let ctls = all_cross_table_lookups::<F>(4, 4, k);
        let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            check_ctls(&tampered, &all_pis, &ctls, &Default::default());
        }))
        .is_err();
        assert!(panicked, "a forged NoiseStark line entry must unbalance the E/F-operand channel");
    }

    /// 6e-3c — the noise-XOF egress channel binds NoiseStark's raw XOF bytes to the noise-BLAKE3
    /// table's in-circuit, seed-derived keyed-XOF `cv_out` limbs. Grinding the noise is fail-closed:
    /// * forging NoiseStark's recomposed limb (`BYTE_PAIR`) at a live egress-pair row unbalances the
    ///   egress channel (its XOF bytes no longer match the seed-derived keyed-XOF);
    /// * forging the noise-BLAKE3 table's egressed limb (`CV_EGRESS_LIMBS`) at a line finalization row
    ///   unbalances the SAME channel from the producing side.
    /// `BYTE_PAIR` / `CV_EGRESS_LIMBS` are wired into no other CTL channel, so each panic is
    /// attributable to the egress binding — the noise is no longer grindable through those bytes.
    #[test]
    fn forged_noise_xof_breaks_the_egress_channel() {
        use crate::v5::circuit::blake3_fp16_stark::columns::FP16_RAW_BLAKE3_COL_MAP as BM;
        use crate::v5::circuit::noise_stark::columns::NOISE_COL_MAP;
        let (honest, all_pis, k) = full_batch();
        let ctls = all_cross_table_lookups::<F>(4, 4, k);
        let ns = &NOISE_COL_MAP;

        // (a) the consuming side: forge NoiseStark's recomposed limb at a live egress-pair row.
        let pair_row = (0..honest[NOISE_STARK_A100_TABLE][ns.is_egress_pair].values.len())
            .find(|&r| honest[NOISE_STARK_A100_TABLE][ns.is_egress_pair].values[r] == F::ONE)
            .expect("a live egress-pair row");
        // (b) the producing side: forge a noise-BLAKE3 egress limb at a line finalization (egress) row.
        let egress_row = (0..honest[NOISE_BLAKE3_A100_TABLE][BM.is_egress_cv].values.len())
            .find(|&r| honest[NOISE_BLAKE3_A100_TABLE][BM.is_egress_cv].values[r] == F::ONE)
            .expect("a noise-BLAKE3 egress row");

        for (table, col, row) in [
            (NOISE_STARK_A100_TABLE, ns.byte_pair, pair_row),
            (NOISE_BLAKE3_A100_TABLE, BM.cv_egress_limbs[0], egress_row),
        ] {
            let mut tampered = honest.clone();
            tampered[table][col].values[row] += F::ONE;
            let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                check_ctls(&tampered, &all_pis, &ctls, &Default::default());
            }))
            .is_err();
            assert!(panicked, "a forged noise XOF limb must unbalance the egress (6e-3c) channel");
        }
    }

    /// `evaluate` is the oracle the policy trace is built to match; this keeps the import test's
    /// tile genuinely accepting even if the reference set changes.
    #[test]
    fn accepting_tile_is_accepted_by_the_oracle() {
        let (k, a, b) = accepting_tile();
        assert!(evaluate(&[replay_census(&a, &b)], k).accept);
    }

    fn replay_census(a: &[u16], b: &[u16]) -> Vec<crate::v5::api::accumulate::PolicyStep> {
        use crate::v5::api::accumulate::a100_dot;
        let mut steps = Vec::new();
        a100_dot(a, b, 0.0, Some(&mut steps));
        steps
    }
}

