//! One FP16 tile, six traces, every CTL channel balanced — and an end-to-end batched-FRI proof.
//!
//! This is the FP16 analogue of `crate::v4::circuit::consistency`. It builds an honest,
//! policy-accepting tile from the GPU reference vectors, generates the matmul and policy traces
//! and the four committed-LUT traces, and exercises two drivers:
//!
//! - [`one_fp16_job_balances_every_ctl_channel`]: every AIR's constraints on its own trace, every
//!   committed-LUT instance served ([`LutChecker`]), and all five CTL channels balanced over the
//!   full six-table batch (`check_ctls`) — the four LUT channels and the `matmul -> policy`
//!   census import;
//! - [`batched_fp16_proof_roundtrips_and_rejects_tampering`]: the real
//!   [`Fp16System`] batched proof — it generates, verifies, and rejects a tampered trace cell, a
//!   forged per-step census value, and a forged decode column.

use plonky2::field::goldilocks_field::GoldilocksField;
use plonky2::field::types::Field;
use plonky2::plonk::config::PoseidonGoldilocksConfig;
use plonky2::util::timing::TimingTree;
use starky::constraint_consumer::ConstraintConsumer;
use starky::cross_table_lookup::debug_utils::check_ctls;
use starky::evaluation_frame::{StarkEvaluationFrame, StarkFrame};
use starky::stark::Stark;

use super::ctl::{
    BLAKE3_A100_TABLE, MATMUL_A100_TABLE, NUM_FP16_TABLES, POLICY_A100_TABLE, XOR_FOLD_A100_TABLE,
    all_cross_table_lookups,
};
use super::driver::{Fp16System, default_lane_layout, fp16_noised_operands};
use super::matmul_a100_stark::columns::MATMUL_A100_COL_MAP;
use super::matmul_a100_stark::stark::MatmulStarkA100;
use super::policy_stark::columns::POLICY_A100_COL_MAP;
use super::policy_stark::stark::PolicyStarkA100;
use super::xor_fold_stark::columns::XOR_FOLD_COL_MAP;
use crate::v5::api::accumulate::{GROUP as ACC_GROUP, a100_dot};
use crate::v5::api::policy::replay_and_evaluate;
use crate::v4::api::transcript::{compute_jackpot_ticket, jackpot_key};
use crate::v4::api::utils::xor_fold_extract;
use crate::v4::api::layout::lane_assignment;
use crate::v4::api::primitives::Hash256;
use crate::v4::circuit::blake3_stark::columns::PI_HASH_JACKPOT;

type F = GoldilocksField;
type C = PoseidonGoldilocksConfig;
const D: usize = 2;
type S = MatmulStarkA100<F, D>;
type P = PolicyStarkA100<F, D>;

const VECTORS: &str = include_str!("../api/testdata/a100_dot_vectors.txt");

/// A small accepting `1 x 1 x k` tile from the GPU reference vectors: operands that clear the
/// policy gate (nonnegative slacks, in-domain RANGE16 limbs) and whose matmul output is normal.
/// `min_k` lets callers pick a tile tall enough to cover the FRI Merkle cap.
fn accepting_tile(min_k: usize) -> (usize, Vec<u16>, Vec<u16>) {
    let mut best: Option<(usize, Vec<u16>, Vec<u16>)> = None;
    for line in VECTORS.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let t: Vec<&str> = line.split_whitespace().collect();
        let k: usize = t[0].parse().unwrap();
        if k % ACC_GROUP != 0 || k < min_k {
            continue;
        }
        // Prefer the smallest qualifying k (fastest FRI).
        if best.as_ref().is_some_and(|(bk, _, _)| k >= *bk) {
            continue;
        }
        let a0: Vec<u16> = t[1..1 + k].iter().map(|x| x.parse().unwrap()).collect();
        let b0: Vec<u16> = t[1 + k..1 + 2 * k].iter().map(|x| x.parse().unwrap()).collect();
        // The batch's matmul/policy are over the NOISED operands, so the gate must clear on the
        // noised 4x4 tile (the tile shape the capstone test proves).
        let (a, b) = (a0.repeat(4), b0.repeat(4));
        let Ok((noised_a, noised_b)) = fp16_noised_operands(4, 4, k, &a, &b, KEY_A, KEY_B, OP_HASH_ID, OP_HASH_ID) else { continue };
        if replay_and_evaluate(&noised_a, &noised_b, 4, 4, k).1.accept {
            best = Some((k, a0, b0));
        }
    }
    best.unwrap_or_else(|| panic!("reference vectors must contain a cell whose noised 4x4 tile accepts with k >= {min_k}"))
}

/// Replicates one accepting cell `(a, b)` into an `h x w` tile: `a` on every A row, `b` on every
/// B column. Every cell then replays the same accepting dot product, so the whole tile clears the
/// policy gate (both ratios are averages of identical cells). Returns `(a_full, b_full)`.
fn replicate(a: &[u16], b: &[u16], h: usize, w: usize) -> (Vec<u16>, Vec<u16>) {
    (a.repeat(h), b.repeat(w))
}

/// Runs every constraint of `$stark` over every row pair of its own trace with starky's
/// `z_last`/`L_first`/`L_last` semantics (transitions excluded on the last -> first wrap).
macro_rules! assert_constraints {
    ($stark:expr, $rows:expr, $name:literal) => {{
        let stark = $stark;
        let n = $rows.len();
        for i in 0..n {
            let frame = StarkFrame::from_values(&$rows[i], &$rows[(i + 1) % n], &[]);
            let mut consumer = ConstraintConsumer::new(
                vec![F::from_canonical_u64(2), F::from_canonical_u64(0x876543210)],
                F::from_bool(i != n - 1),
                F::from_bool(i == 0),
                F::from_bool(i == n - 1),
            );
            stark.eval_packed_generic(&frame, &mut consumer);
            assert!(
                consumer.accumulators().into_iter().all(|acc| acc == F::ZERO),
                "{} constraints violated at row {i}",
                $name
            );
        }
    }};
}

/// A fixed noise seed for the jackpot key derivation in these tests.
const SEED_A: Hash256 = [0x5a; 32];
/// Operand-tree opening keys and leaf size for these tests (passed in; the seed-chain derivation is
/// a later increment). `Blake3Chunk512` keeps the Blake3 table's height a power of two the config
/// handles natively.
const KEY_A: Hash256 = [0x11; 32];
const KEY_B: Hash256 = [0x22; 32];
const OP_HASH_ID: crate::v4::api::public_params::HashId = crate::v4::api::public_params::HashId::Blake3Chunk512;

/// The 32-byte jackpot digest the plaintext extractor produces for the `h x w` tile whose cell
/// `(r, c)` is the A100 dot of the NOISED operand rows `(noised_a[r], noised_b[c])` — the same noised
/// operands the batch's matmul multiplies: fold the tile under the committed lane layout, then
/// `compute_jackpot_ticket`. The committed ZK `HASH_JACKPOT` must equal this bit-for-bit.
fn plaintext_jackpot(noised_a: &[u16], noised_b: &[u16], h: usize, w: usize, k: usize) -> Hash256 {
    let mut tile = vec![0f32; h * w];
    for r in 0..h {
        for c in 0..w {
            tile[r * w + c] = a100_dot(&noised_a[r * k..r * k + k], &noised_b[c * k..c * k + k], 0.0, None);
        }
    }
    let (row_axis, col_axis) = default_lane_layout(h, w);
    let lanes = lane_assignment(&row_axis, &col_axis);
    let msg = xor_fold_extract(&tile, &lanes);
    compute_jackpot_ticket(&SEED_A, &msg).jackpot
}

/// 32-byte digest -> 8 LE `u32` field words, the Blake3 public-input form of `HASH_JACKPOT`.
fn hash_to_words(h: &Hash256) -> Vec<F> {
    (0..8)
        .map(|i| F::from_canonical_u32(u32::from_le_bytes(h[4 * i..4 * i + 4].try_into().unwrap())))
        .collect()
}

/// The channel-balance driver: every AIR satisfied on its own trace, every committed-LUT instance
/// served, and all CTL channels balanced over the full nine-table batch.
#[test]
fn one_fp16_job_balances_every_ctl_channel() {
    // A 4x4 tile replicated from one accepting cell: 16 output cells, one per lottery lane.
    let (k, a0, b0) = accepting_tile(16);
    let (a, b) = replicate(&a0, &b0, 4, 4);
    let matmul = S::new(4, 4, k);
    let policy = P::new(4, 4, k);

    // Every AIR satisfied on its own trace — over the NOISED operands the batch's matmul/policy
    // multiply (the quant chain's `out`), so the policy gate clears exactly as in the batch.
    let (noised_a, noised_b) =
        fp16_noised_operands(4, 4, k, &a, &b, KEY_A, KEY_B, OP_HASH_ID, OP_HASH_ID).expect("in-envelope noised operands");
    assert_constraints!(&matmul, &matmul.generate_trace(&noised_a, &noised_b, None), "MatmulA100");
    assert_constraints!(&policy, &policy.generate_trace(&noised_a, &noised_b), "PolicyA100");

    // The driver assembles all nine traces (committed-LUT serving re-checked inside).
    let system = Fp16System::<F, D>::new(4, 4, k, OP_HASH_ID, OP_HASH_ID);
    let jk = jackpot_key(&SEED_A);
    let (traces, blake3_pis) = system
        .generate_batch_traces(&a, &b, KEY_A, KEY_B, jk)
        .expect("every honest instance must be served");
    assert_eq!(traces.len(), NUM_FP16_TABLES);

    // Every CTL channel balances: the five LUT channels, the census-import channel, the matmul ->
    // XorFold cell-results and XorFold -> Blake3 lottery-words channels, the two internal quant-chain
    // channels, and the two operand-provenance channels (operand bytes [6c], operand codes [6d]).
    let mut all_pis: Vec<Vec<F>> = vec![Vec::new(); NUM_FP16_TABLES];
    all_pis[BLAKE3_A100_TABLE] = blake3_pis;
    check_ctls(&traces.to_vec(), &all_pis, &all_cross_table_lookups::<F>(4, 4, k), &Default::default());
}

/// The capstone: a real batched-FRI proof of an honest tile generates and verifies, `HASH_JACKPOT`
/// equals the plaintext jackpot, and tampering (a trace cell, a policy census, a decode column) is
/// rejected by the batch verifier.
#[test]
fn batched_fp16_proof_roundtrips_and_rejects_tampering() {
    // A 4x4 tile (replicated accepting cell): 16 output cells feeding the 16 lottery lanes; the
    // 2^16 committed LUTs dominate the proof cost regardless.
    let (k, a0, b0) = accepting_tile(16);
    let (a, b) = replicate(&a0, &b0, 4, 4);
    let (h, w) = (4usize, 4usize);
    let mut system = Fp16System::<F, D>::new(h, w, k, OP_HASH_ID, OP_HASH_ID);
    let jk = jackpot_key(&SEED_A);
    let mut timing = TimingTree::default();
    let preprocessed = system.preprocessed_data::<C>(&mut timing);
    let lut_cap = preprocessed.cap();

    // ---- Honest proof: generates and verifies. ----
    let proof = system
        .prove::<C>(&a, &b, KEY_A, KEY_B, jk, &preprocessed, &mut timing)
        .expect("honest FP16 tile must prove");
    let blake3_pis = proof.public_inputs[BLAKE3_A100_TABLE].clone();
    system
        .verify::<C>(&proof, &blake3_pis, &lut_cap)
        .expect("honest FP16 proof must verify");

    // HASH_JACKPOT is bit-exact with the plaintext extractor -> jackpot transcript over the NOISED
    // operands the matmul multiplies (the quant chain's `out`).
    let (noised_a, noised_b) = system.noised_operands(&a, &b, KEY_A, KEY_B);
    let expected = plaintext_jackpot(&noised_a, &noised_b, h, w, k);
    assert_eq!(
        &blake3_pis[PI_HASH_JACKPOT..PI_HASH_JACKPOT + 8],
        hash_to_words(&expected).as_slice(),
        "committed HASH_JACKPOT must equal the plaintext jackpot"
    );

    // ---- Tampering: mutate one honest trace cell (keeping the honest LUT multiplicities), prove,
    // and assert the batch verifier rejects. ----
    let m = &MATMUL_A100_COL_MAP;
    let p = &POLICY_A100_COL_MAP;
    let xf = &XOR_FOLD_COL_MAP;

    let (honest, _) = system.generate_batch_traces(&a, &b, KEY_A, KEY_B, jk).expect("honest traces");
    let mat_live = (0..system.num_rows() - 1)
        .find(|&r| {
            honest[MATMUL_A100_TABLE][m.is_padding].values[r] == F::ZERO
                && honest[MATMUL_A100_TABLE][m.group_sum_is_zero].values[r] == F::ZERO
                && honest[MATMUL_A100_TABLE][m.is_cell_final].values[r] == F::ZERO
        })
        .expect("a live mid-cell matmul row");
    let pol_live = (0..system.num_rows())
        .find(|&r| honest[POLICY_A100_TABLE][p.is_padding].values[r] == F::ZERO)
        .expect("a live policy row");
    // A live (non-padding) XorFold row carries a folded cell word.
    let xf_rows = honest[XOR_FOLD_A100_TABLE][xf.is_pad].values.len();
    let xf_live = (0..xf_rows)
        .find(|&r| honest[XOR_FOLD_A100_TABLE][xf.is_pad].values[r] == F::ZERO)
        .expect("a live XorFold row");

    let prove_tampered = |col_table: usize, col: usize, row: usize| {
        let (mut traces, blake3_pis) = system.generate_batch_traces(&a, &b, KEY_A, KEY_B, jk).expect("honest traces");
        traces[col_table][col].values[row] += F::ONE;
        let mut timing = TimingTree::default();
        // The prover may succeed (it does not re-check constraints), but verification must fail.
        match system.prove_batch_traces::<C>(traces, &blake3_pis, &preprocessed, &mut timing) {
            Ok(bad) => system.verify::<C>(&bad, &blake3_pis, &lut_cap).is_err(),
            Err(_) => true, // the prover itself refused the malformed trace
        }
    };

    // (a) a trace cell: the accumulated group-sum magnitude (MA5 arithmetic).
    assert!(
        prove_tampered(MATMUL_A100_TABLE, m.group_sum_abs, mat_live),
        "a tampered matmul trace cell must be rejected"
    );
    // (b) a census value: the policy's per-step breakpoint (census-import channel + PP gate).
    assert!(
        prove_tampered(POLICY_A100_TABLE, p.group_breakpoint, pol_live),
        "a forged policy census value must be rejected"
    );
    // (c) a decode column the LUTs are responsible for: a lane significand (FP16DECODE channel).
    assert!(
        prove_tampered(MATMUL_A100_TABLE, m.sig_a[1], mat_live),
        "a forged decode column must be rejected by the FP16DECODE CTL"
    );
    // (d) the matmul output tile a cell feeds into the lottery: tampering a XorFold cell-result limb
    // unbalances the matmul -> XorFold cell-results channel. (The full output -> jackpot chain,
    // including a tampered lottery word, is exercised by the driver end-to-end test.)
    assert!(
        prove_tampered(XOR_FOLD_A100_TABLE, xf.cell_result_f32_lo, xf_live),
        "a tampered folded cell word must be rejected by the cell-results CTL"
    );

    // (e) a mismatched statement: verifying the honest proof against a different geometry must fail
    // on the degree-profile / known-column binding.
    let other = Fp16System::<F, D>::new(h, w, k * 2, OP_HASH_ID, OP_HASH_ID);
    let other_prep = other.preprocessed_data::<C>(&mut TimingTree::default());
    assert!(
        other.verify::<C>(&proof, &blake3_pis, &other_prep.cap()).is_err(),
        "a proof must not verify against a different statement"
    );
}
