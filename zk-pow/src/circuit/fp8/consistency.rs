//! One job, twenty-two traces, every CTL channel balanced.
//!
//! [`build_fixture`] assembles the same fixture the real verifier parses and generates every
//! main table's trace from that one witness — InputQuant consumes the opened strips, Scale
//! consumes InputQuant's group tuples, Matmul
//! consumes InputQuant's noised fp8 codes, XorFold consumes Matmul's cell results, Tamed
//! consumes Matmul's E-cell binades and Scale's sigma frames, Blake3 consumes the strips'
//! bytes and XorFold's folded lottery words under the deployed `BlakeProgram` schedule. It is
//! shared by two drivers:
//!
//! - this module's channel-balance test, which checks every AIR's constraints on its own
//!   trace, every program's `known_values` (class (a) recompute) against its trace's leading
//!   columns, every committed-LUT instance against the committed tables
//!   ([`LutChecker`]), and the balance of all 24 CTL channels over the full 22-table batch
//!   via starky's `check_ctls`;
//! - the batch driver's end-to-end proof test (`super::driver`), which proves the same job
//!   with `starky::batch_prover::batch_prove` and verifies it back.

use core::borrow::Borrow;

use pearl_blake3::{BLAKE3_CHUNK_LEN, MerkleTree, pad_to_chunk_boundary};
use plonky2::field::goldilocks_field::GoldilocksField;
use plonky2::field::polynomial::PolynomialValues;
use plonky2::field::types::{Field, PrimeField64};
use starky::constraint_consumer::ConstraintConsumer;
use starky::cross_table_lookup::debug_utils::check_ctls;
use starky::evaluation_frame::{StarkEvaluationFrame, StarkFrame};
use starky::stark::Stark;

use super::blake3_stark::columns::{Blake3ColumnsView, NUM_BLAKE3_COLUMNS};
use super::blake3_stark::ctl::blake3_lut_lookups;
use super::blake3_stark::stark::{Blake3KnownInputs, Blake3Program, Blake3Stark, Blake3TraceInputs, MoeSchedule};
use super::ctl::{LUT_TABLES, LutTable, NUM_TABLES, all_cross_table_lookups};
use super::input_quant_stark::columns::{InputQuantColumnsView, NUM_INPUT_QUANT_COLUMNS};
use super::input_quant_stark::ctl::input_quant_lut_lookups;
use super::input_quant_stark::stark::{InputQuantProgram, InputQuantStark};
use super::known_values::{KNOWN_COLUMNS_PER_TABLE, fp8_known_columns};
use super::luts::{LutChecker, lut_trace};
use super::matmul_b200_stark::MatmulB200ColumnsView;
use super::matmul_b200_stark::columns::NUM_MATMUL_B200_COLUMNS;
use super::matmul_b200_stark::ctl::matmul_b200_lut_lookups;
use super::matmul_b200_stark::stark::{MatmulB200Stark, MatmulProgram, generate_b200_trace};
use super::scale_stark::columns::{NUM_SCALE_COLUMNS, ScaleColumnsView};
use super::scale_stark::ctl::{SIGMA_EXP_OFFSET, scale_lut_lookups};
use super::scale_stark::stark::{ScaleProgram, ScaleRowTuple, ScaleStark};
use super::tamed_stark::columns::NUM_TAMED_COLUMNS;
use super::tamed_stark::ctl::tamed_lut_lookups;
use super::tamed_stark::stark::{TamedProgram, TamedStark};
use super::xor_fold_stark::columns::{NUM_XOR_FOLD_COLUMNS, XorFoldColumnsView};
use super::xor_fold_stark::ctl::xor_fold_lut_lookups;
use super::xor_fold_stark::stark::{XorFoldProgram, XorFoldStark};
use crate::api::fp8::openings::stream_words;
use crate::api::fp8::plain_proof::{MoeWitness, PlainProofV4};
use crate::api::fp8::prequant::BLOCK_SIZE;
use crate::api::fp8::public_params::{CommonParams, Device, HashId, JobParams, MoeParams, OperandParams, Quant};
use crate::api::fp8::transcript::{key_a, key_b};
use crate::api::layout::{AxisPattern, DimType};
use crate::api::primitives::IncompleteBlockHeader;
#[cfg(test)]
use crate::api::verify::verify_plain_proof;
use crate::ffi::plain_proof::MatrixMerkleProof;

type F = GoldilocksField;
const D: usize = 2;

fn to_u64(x: F) -> u64 {
    x.to_canonical_u64()
}

/// Row-major trace rows -> the column-major `PolynomialValues` layout `check_ctls` reads.
fn columns<const N: usize>(rows: &[[F; N]]) -> Vec<PolynomialValues<F>> {
    (0..N)
        .map(|c| PolynomialValues::new(rows.iter().map(|r| r[c]).collect()))
        .collect()
}

/// Runs every constraint of `stark` over every row pair of its own trace, with starky's
/// `z_last`/`L_first`/`L_last` semantics (transitions excluded on the last -> first wrap).
macro_rules! assert_constraints {
    ($stark:expr, $rows:expr, $pis:expr, $name:literal) => {{
        let stark = $stark;
        let n = $rows.len();
        for i in 0..n {
            let frame = StarkFrame::from_values(&$rows[i], &$rows[(i + 1) % n], $pis);
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

/// One fully assembled fp8 job: the compiled programs and public recompute inputs
/// (the statement; Tamed's program is the shared geometry itself), the private strip planes
/// and Blake3 auxiliary data (the witness), and the six generated main traces with their
/// public inputs.
pub(crate) struct Fp8Fixture {
    // The statement: programs plus the class (a) recompute inputs and expected public inputs.
    pub blake3: Blake3Program,
    pub input_quant: InputQuantProgram,
    pub scale: ScaleProgram,
    pub matmul: MatmulProgram,
    pub xor_fold: XorFoldProgram,
    pub a_noise: Vec<u16>,
    pub b_noise: Vec<u16>,
    pub public_inputs: [Vec<F>; NUM_TABLES],
    // The witness: the opened strip planes (raw bytes) and Blake3's auxiliary inputs.
    pub a_values: Vec<u8>,
    pub a_scales: Vec<u8>,
    pub b_values: Vec<u8>,
    pub b_scales: Vec<u8>,
    /// The opened routing hotspot blocks as u32 words (MoE; empty for dense jobs).
    pub routing_words: Vec<u32>,
    /// The full padded offsets list as u32 words (MoE; empty for dense jobs).
    pub offsets_words: Vec<u32>,
    pub aux_msgs: Vec<[u8; 64]>,
    pub aux_cvs: Vec<[u8; 32]>,
    pub key_a: [u32; 8],
    pub key_b: [u32; 8],
    pub jackpot_key: [u32; 8],
    pub a_hash_id: HashId,
    pub b_hash_id: HashId,
    pub routing_hash_id: HashId,
    pub offsets_hash_id: HashId,
    // The generated traces (row-major) and their geometry.
    pub b3_rows: Vec<[F; NUM_BLAKE3_COLUMNS]>,
    pub iq_rows: Vec<[F; NUM_INPUT_QUANT_COLUMNS]>,
    pub scale_rows: Vec<[F; NUM_SCALE_COLUMNS]>,
    pub mat_rows: Vec<[F; NUM_MATMUL_B200_COLUMNS]>,
    pub xf_rows: Vec<[F; NUM_XOR_FOLD_COLUMNS]>,
    pub tamed_rows: Vec<[F; NUM_TAMED_COLUMNS]>,
}

/// The fixture job as the wire would carry it: the block header and the fp8 `PlainProof`
/// (keyed Merkle trees over pseudo-random prequant planes).
///
/// Geometry: `m = n = 32` committed rows, `k = 2048`, `h = w = 16` opened strips (per axis
/// `[(4, Fold), (4, Blake)]`: the tile `{0..15}`) — the smallest pad-free shape that
/// satisfies every table and the consensus envelope at once
/// (`api::layout` needs rank 32, 4 x 4 = 16 lottery lanes, a 16-element
/// subtile and 16 opened cols; `h*k`, `h + w` and `h*w` are all powers of two here, so no
/// table pads). Shared with the API round-trip test (`crate::api::fp8::zk`), which drives the
/// same job through the wire-level entry points; [`fixture_job_asym`] covers the padded
/// geometries and [`fixture_job_k32`] the non-power-of-two `k` envelope.
pub(crate) fn fixture_job() -> (IncompleteBlockHeader, PlainProofV4) {
    let dims = &[(4, DimType::Fold), (4, DimType::Blake)];
    fixture_job_with(32, 2048, dims, dims)
}

/// The v4 opening keys for a fixture header: `keyA = H_"key-A"(σ̂)`,
/// `keyB = H_"key-B"(σ_Δ)`, where the fixture's σ_Δ is σ̂ itself.
pub(crate) fn fixture_tree_keys(header: &IncompleteBlockHeader) -> ([u8; 32], [u8; 32]) {
    (key_a(header), key_b(header))
}

fn keyed_plane_proof(rows_bytes: &[Vec<u8>], row_indices: &[usize], key: [u8; 32], hash_id: HashId) -> MatrixMerkleProof {
    let flat: Vec<u8> = rows_bytes.iter().flatten().copied().collect();
    let chunk_len = hash_id.chunk_len();
    let tree = MerkleTree::with_chunk_len(&hash_id.pad(&flat), key, chunk_len).unwrap();
    let leaves =
        MerkleTree::compute_leaf_indices_from_rows(row_indices, (rows_bytes.len(), rows_bytes[0].len()), chunk_len).unwrap();
    MatrixMerkleProof {
        proof: tree.get_multileaf_proof(&leaves),
        row_indices: row_indices.to_vec(),
    }
}

/// The asymmetric fixture job: `m = n = 32`, `k = 2048`, rows `[(4, Fold), (4, Blake)]` ->
/// `h = 16`, cols `[(5, Fold), (4, Blake)]` -> `w = 20` (a 20-element subtile, still inside
/// `MIN_SUBTILE_ELEMS..=MAX_SUBTILE_ELEMS`). Exercises every padding path [`fixture_job`]
/// misses: `h != w` with a dead A-region (InputQuant, `16*2048` live A rows against
/// `20*2048` live B rows), Scale pad rows (`h + w = 36` -> 64), XorFold pad rows
/// (`h*w = 320` -> 512), and Matmul trailing phantom cells (`320*32` -> 2^14).
pub(crate) fn fixture_job_asym() -> (IncompleteBlockHeader, PlainProofV4) {
    fixture_job_with(
        32,
        2048,
        &[(4, DimType::Fold), (4, DimType::Blake)],
        &[(5, DimType::Fold), (4, DimType::Blake)],
    )
}

/// The `k % 32` fixture job: `m = n = 32`, `k = 2080` (`= 32 * 65`, so `k % 64 = 32` — the
/// envelope the wire layer admits since dword tiling replaced whole-block tiling). Committed
/// rows are 2080 bytes (values) and 264 bytes (scales), so row boundaries inside a plane are
/// never 64-byte aligned and the Blake3 schedule is dense with straddling blocks: cross-strip
/// `MatrixLeaf`s across the contiguous opened band `{0..15}` on both the values and scales
/// planes, and `SplitLeaf`s where a block runs into an unopened row or the plane's chunk
/// padding. Downstream, `k/32 = 65` rows per Matmul cell and `k/8 = 260` whole quantizer
/// blocks per InputQuant group.
pub(crate) fn fixture_job_k32() -> (IncompleteBlockHeader, PlainProofV4) {
    let dims = &[(4, DimType::Fold), (4, DimType::Blake)];
    fixture_job_with(32, 2080, dims, dims)
}

/// The mid-size fixture job: `m = n = 32` committed rows, `k = 4096`, `h = w = 16` opened
/// strips with interleaved dims `[(2, Fold), (2, Blake), (2, Fold), (2, Blake)]` per axis
/// (fold `{0, 1, 4, 5}`, blake `{0, 2, 8, 10}` — a non-contiguous lane geometry, unlike
/// [`fixture_job`]'s) — InputQuant at 2^16 rows, Blake3 and Matmul at 2^15. Geometry
/// comparable to the existing recursion test jobs, big enough that per-proof overheads
/// stop dominating; used by the medium API round trip (`crate::api::fp8::zk`).
pub(crate) fn fixture_job_medium() -> (IncompleteBlockHeader, PlainProofV4) {
    let dims = &[
        (2, DimType::Fold),
        (2, DimType::Blake),
        (2, DimType::Fold),
        (2, DimType::Blake),
    ];
    fixture_job_with(32, 4096, dims, dims)
}

/// A policy-rejected wire job ([`fixture_job`]'s geometry): both sides commit the *same*
/// value plane, so each diagonal tile cell replays a coherent `<v, v>` accumulation — all
/// squares, no cancellation — whose magnitude blows the tamed-products allowance (jackpot
/// check 3, 16 untamed of 256 cells > the `eps_tame = 1/64` allowance of 4). Checks 1, 2
/// and 4 all pass: lifting `eps_tame` alone accepts the tile.
///
/// `k = 8192` is forced by `tau_tame = 256`: a diagonal cell's ratio `M / (sigma_i*sigma_j)`
/// is Cauchy-Schwarz-capped at `~5k` (`4k` clean + `~k` noise), so untamed needs
/// `4k > tau_tame * sqrt(k)`, i.e. `sqrt(k) > 64`. This `k` keeps the same `~1.4x`
/// clean-cap headroom over the bound that `k = 2048` gave at `tau_tame = 128`.
pub(crate) fn fixture_job_untamed() -> (IncompleteBlockHeader, PlainProofV4) {
    let dims = &[(4, DimType::Fold), (4, DimType::Blake)];
    let (m, k) = (32, 8192);
    let plane: Vec<Vec<u8>> = (0..m).map(|i| (0..k).map(|j| plane_byte(0, i, j)).collect()).collect();
    fixture_job_with_planes(m, k, dims, dims, plane.clone(), plane)
}

/// The shared fixture-job builder: `m = n` committed rows and, per axis, the opened strips
/// = the tile of that axis' committed [`AxisPattern`] dims (based at 0). The dims must carry
/// Blake sizes multiplying to the 16 lottery lanes (4 per axis here) and Fold sizes whose
/// product is a legal subtile to stay inside the consensus envelope
/// (`api::layout`); the two axes need not agree, so padded geometries
/// (`h != w`, non-power-of-two `h + w` / `h*w`) are expressible.
fn fixture_job_with(
    m: usize,
    k: usize,
    row_dims: &[(u32, DimType)],
    col_dims: &[(u32, DimType)],
) -> (IncompleteBlockHeader, PlainProofV4) {
    fixture_job_with_zero_a_rows(m, k, row_dims, col_dims, &[])
}

/// One committed int8-plane byte: a splitmix64-style mix of `(seed, i, j)`, so committed rows
/// are mutually incoherent like honest workloads. (Overlapping ramps would make `A @ B^T`
/// cells coherent and fail the jackpot policy's tamed-products allowance, which
/// `TamedProgram::generate_trace` enforces.)
fn plane_byte(seed: usize, i: usize, j: usize) -> u8 {
    let mut x = ((seed as u64) << 48) ^ ((i as u64) << 24) ^ j as u64;
    x = x.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    x ^= x >> 31;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27;
    (x % 251) as u8
}

/// [`fixture_job_with`] with the given committed A rows' int8 values zeroed (scales stay
/// normal, so those rows decode to `X = 0` everywhere): the all-zero-row envelope. The scheme
/// floors both norms at `2^-32` (ledger N5), so alpha stays finite, beta strictly positive,
/// and the row's noised elements are pure noise — the ticket must remain provable end to end.
fn fixture_job_with_zero_a_rows(
    m: usize,
    k: usize,
    row_dims: &[(u32, DimType)],
    col_dims: &[(u32, DimType)],
    zero_a_rows: &[usize],
) -> (IncompleteBlockHeader, PlainProofV4) {
    let plane = |seed: usize| -> Vec<Vec<u8>> { (0..m).map(|i| (0..k).map(|j| plane_byte(seed, i, j)).collect()).collect() };
    let mut a_values = plane(0);
    for &i in zero_a_rows {
        a_values[i].fill(0);
    }
    fixture_job_with_planes(m, k, row_dims, col_dims, a_values, plane(7))
}

/// [`fixture_job_with`] on explicit committed int8 value planes (`m` rows of `k` bytes
/// each): adversarial planes ride the same wire layout as the honest fixtures.
fn fixture_job_with_planes(
    m: usize,
    k: usize,
    row_dims: &[(u32, DimType)],
    col_dims: &[(u32, DimType)],
    a_values: Vec<Vec<u8>>,
    b_values: Vec<Vec<u8>>,
) -> (IncompleteBlockHeader, PlainProofV4) {
    let n = m;
    let n_blocks = k / BLOCK_SIZE;

    // ---- The committed matrices and the sampled strips: each axis' tile (fold (+) blake). ----
    let rows_pattern = AxisPattern::new(row_dims).unwrap();
    let cols_pattern = AxisPattern::new(col_dims).unwrap();
    let tile = |p: &AxisPattern| -> Vec<usize> { p.tile_offsets().iter().map(|&o| o as usize).collect() };
    let (a_rows, b_rows) = (tile(&rows_pattern), tile(&cols_pattern));
    let header = IncompleteBlockHeader::new_for_test(0x207FFFFF);
    let (key_a, key_b) = fixture_tree_keys(&header);

    let scale_tree = |seed: usize| -> Vec<Vec<u8>> {
        (0..m)
            .map(|i| {
                (0..n_blocks)
                    .flat_map(|b| (0x3f80u16 + ((seed + i + b) % 16) as u16).to_le_bytes())
                    .collect()
            })
            .collect()
    };
    let a_hash_id = HashId::Blake3Chunk256;
    let b_hash_id = HashId::Blake3Chunk128;
    let proof = PlainProofV4 {
        job: JobParams {
            // σ̂ and σ_Δ coincide in this fixture (the ancestor is the proposed header).
            ancestor_header: header,
            common: CommonParams {
                k: k as u32,
                r: 32,
                quant: Quant::Fp8E4M3Prequant,
                device: Device::B200,
            },
            operands: crate::api::primitives::Sides {
                a: OperandParams {
                    num_rows: m as u32,
                    hash_id: a_hash_id,
                    pattern: rows_pattern,
                },
                b: OperandParams {
                    num_rows: n as u32,
                    hash_id: b_hash_id,
                    pattern: cols_pattern,
                },
            },
            moe: None,
        },
        values: crate::api::primitives::Sides {
            a: keyed_plane_proof(&a_values, &a_rows, key_a, a_hash_id),
            b: keyed_plane_proof(&b_values, &b_rows, key_b, b_hash_id),
        },
        scales: crate::api::primitives::Sides {
            a: keyed_plane_proof(&scale_tree(1), &a_rows, key_a, a_hash_id),
            b: keyed_plane_proof(&scale_tree(5), &b_rows, key_b, b_hash_id),
        },
        moe_witness: None,
    };
    (header, proof)
}

/// The MoE fixture job: `m = 256` global tokens, `e = 4` experts
/// with `top_k = 2` (512 flat routing entries — a two-chunk routing tree; the deployed
/// membership verifier requires more than one chunk, see `MerkleProof::compute_root`), the
/// proof for `expert_idx = 1` whose region spans slots `128..256` — the sampled entries
/// (inner = the rows tile `{0..3, 8..11, 16..19, 24..27}` -> slots `{128..131, 136..139,
/// 144..147, 152..155}`) straddle routing blocks 8 and 9, so the Blake3 routing tree opens
/// two hotspot strips with interleaved pinned/neighbor words (the rest of the tree rides as
/// auxiliary messages and CVs). The A tree commits the global 256-token activation plane and
/// opens the 16 outer rows; the B^T tree commits all four experts' weight columns
/// (stacked `n = n_e * e = 64`) and opens expert 1's `[16..32)`.
///
/// Geometry: rows `[(4, Fold), (2, Null), (4, Blake)]` (the Null dim spreads the subtiles
/// out inside the expert's region), cols `[(4, Fold), (4, Blake)]` — `h = w = 16` opened
/// strips, the same consensus envelope as [`fixture_job`] (`k = 2048`, 16 lanes,
/// 16-element subtiles).
pub(crate) fn fixture_job_moe() -> (IncompleteBlockHeader, PlainProofV4) {
    // Each expert gets 0, 2, ..., 254 (strictly increasing). The same tokens repeat
    // across experts; never twice in one expert.
    let routing_flat: Vec<u32> = (0..4).flat_map(|_| (0..128u32).map(|i| 2 * i)).collect();
    fixture_job_moe_with(routing_flat)
}

/// Exclusive ends [128, 256, 384, 512]; expert 1 is `routing[128..256)`.
/// `I_A` is that slice at the rows tile.
fn fixture_job_moe_with(routing_flat: Vec<u32>) -> (IncompleteBlockHeader, PlainProofV4) {
    let (m, n_e, k) = (256usize, 16usize, 2048usize);
    let (e, _top_k, expert_idx) = (4usize, 2usize, 1u16);
    let n_blocks = k / BLOCK_SIZE;
    assert_eq!(routing_flat.len(), 512);

    let rows_pattern = AxisPattern::new(&[(4, DimType::Fold), (2, DimType::Null), (4, DimType::Blake)]).unwrap();
    let cols_pattern = AxisPattern::new(&[(4, DimType::Fold), (4, DimType::Blake)]).unwrap();

    let routing_end_offsets: Vec<u32> = vec![128, 256, 384, 512];
    let inner_a_rows: Vec<usize> = rows_pattern.tile_offsets().iter().map(|&o| o as usize).collect();
    let outer_rows: Vec<usize> = inner_a_rows.iter().map(|&i| routing_flat[128 + i] as usize).collect();

    let header = IncompleteBlockHeader::new_for_test(0x207FFFFF);
    let (key_a, key_b) = fixture_tree_keys(&header);

    let value_tree = |rows: usize, seed: usize| -> Vec<Vec<u8>> {
        (0..rows).map(|i| (0..k).map(|j| plane_byte(seed, i, j)).collect()).collect()
    };
    let scale_tree = |rows: usize, seed: usize| -> Vec<Vec<u8>> {
        (0..rows)
            .map(|i| {
                (0..n_blocks)
                    .flat_map(|b| (0x3f80u16 + ((seed + i + b) % 16) as u16).to_le_bytes())
                    .collect()
            })
            .collect()
    };
    // The routing Merkle proof: the flat u32 entries as 4-byte "rows", opened at the
    // sampled global slots (mirrors the miner's `build_routing_proof`). The offset list
    // is disclosed raw; only its claimed root rides the witness.
    let routing_bytes: Vec<u8> = routing_flat.iter().flat_map(|&x| x.to_le_bytes()).collect();
    let routing_tree = MerkleTree::new(&pad_to_chunk_boundary(&routing_bytes), key_a);
    let offsets_bytes: Vec<u8> = routing_end_offsets.iter().flat_map(|&x| x.to_le_bytes()).collect();
    let offsets_tree = MerkleTree::new(&pad_to_chunk_boundary(&offsets_bytes), key_a);
    let routing_leaves = crate::api::fp8::public_params::MoEStatement {
        w: expert_idx,
        o_w_prev: 128,
        o_w: 256,
        o_last: 512,
        hash_routing: [0u8; 32],
        hash_offsets: [0u8; 32],
        i_a: vec![],
    }
    .opened_routing_blocks()
    .into_iter()
    .map(|r| r as usize)
    .collect::<Vec<_>>();
    // Unique-minimal merkle leaves covering those 64-byte virtual rows.
    let routing_leaf_indices =
        MerkleTree::compute_leaf_indices_from_rows(&routing_leaves, (routing_bytes.len().div_ceil(64), 64), BLAKE3_CHUNK_LEN)
            .unwrap();
    let routing_proof = routing_tree.get_multileaf_proof(&routing_leaf_indices);

    // B^T commits every expert's columns; expert 1's global columns are [16, 32).
    let bt_rows: Vec<usize> = (n_e..2 * n_e).collect();
    let hash_id = HashId::Blake3Chunk1024;
    let proof = PlainProofV4 {
        job: JobParams {
            // σ̂ and σ_Δ coincide in this fixture (the ancestor is the proposed header).
            ancestor_header: header,
            common: CommonParams {
                k: k as u32,
                r: 32,
                quant: Quant::Fp8E4M3Prequant,
                device: Device::B200,
            },
            operands: crate::api::primitives::Sides {
                a: OperandParams {
                    num_rows: m as u32,
                    hash_id,
                    pattern: rows_pattern,
                },
                b: OperandParams {
                    num_rows: (n_e * e) as u32,
                    hash_id,
                    pattern: cols_pattern,
                },
            },
            moe: Some(MoeParams {
                experts: e as u16,
                hash_id_r: hash_id,
                hash_id_o: hash_id,
            }),
        },
        values: crate::api::primitives::Sides {
            a: keyed_plane_proof(&value_tree(m, 0), &outer_rows, key_a, hash_id),
            b: keyed_plane_proof(&value_tree(n_e * e, 7), &bt_rows, key_b, hash_id),
        },
        scales: crate::api::primitives::Sides {
            a: keyed_plane_proof(&scale_tree(m, 1), &outer_rows, key_a, hash_id),
            b: keyed_plane_proof(&scale_tree(n_e * e, 5), &bt_rows, key_b, hash_id),
        },
        moe_witness: Some(MoeWitness {
            w: expert_idx,
            offsets: routing_end_offsets.clone(),
            offsets_root: offsets_tree.root(),
            routing: routing_proof,
        }),
    };
    (header, proof)
}

/// Builds the shared end-to-end fixture: verifier-parsed
/// job -> five mutually consistent traces (the Matmul trace,
/// its bit-exact cell results feeding XorFold and, through the lottery words, Blake3).
/// With `moe`, the job is [`fixture_job_moe`]: the Blake3 forest gains the routing tree,
/// the pin schedule rides the program, and the opened hotspot words ride the witness.
pub(crate) fn build_fixture(moe: bool) -> Fp8Fixture {
    let (header, proof) = if moe { fixture_job_moe() } else { fixture_job() };
    build_fixture_from_job(header, proof)
}

/// [`build_fixture`] on an explicit wire-level job: the geometry (`h`, `w`, `k`, `r`, the
/// lane assignment) is read from the parse, so any reference-legal job — in particular the
/// padded [`fixture_job_asym`] shapes — drives the same five traces.
pub(crate) fn build_fixture_from_job(header: IncompleteBlockHeader, proof: PlainProofV4) -> Fp8Fixture {
    let (private, public) = proof.parse_proof(&header).expect("fixture must parse");
    let (compiled, _, _) = public.compile(&header).expect("fixture must compile");
    let (h, w, k, r) = (compiled.h, compiled.w, compiled.k, compiled.r);

    // ---- InputQuantStark: the opened strips + the job's noise codes. ----
    let iq_program = InputQuantProgram {
        h,
        w,
        k,
        block_size: BLOCK_SIZE,
        r,
    };
    let a_int8: Vec<i8> = private.operands.a.values.as_bytes().iter().map(|&b| b as i8).collect();
    let b_int8: Vec<i8> = private.operands.b.values.as_bytes().iter().map(|&b| b as i8).collect();
    let le_codes = |bytes: &[u8]| -> Vec<u16> { bytes.as_chunks::<2>().0.iter().map(|p| u16::from_le_bytes(*p)).collect() };
    let a_scale_codes = le_codes(private.operands.a.scales.as_bytes());
    let b_scale_codes = le_codes(private.operands.b.scales.as_bytes());
    // Normal-or-zero bf16 noise codes (the real ones are `(E @ F)[elem]`, recomputed natively).
    let noise = |len: usize, salt: u64| -> Vec<u16> {
        const POOL: [u16; 8] = [0x0000, 0x3F80, 0xBF00, 0x3E80, 0x4000, 0xBF80, 0x3D80, 0xC000];
        (0..len)
            .map(|i| POOL[((i as u64).wrapping_mul(salt) >> 7) as usize % POOL.len()])
            .collect()
    };
    let a_noise = noise(h * k, 0xD1B54A32D192ED03);
    let b_noise = noise(w * k, 0x2545F4914F6CDD1D);
    let (iq_rows, iq_pis) = iq_program.generate_trace::<F>(&a_int8, &a_scale_codes, &a_noise, &b_int8, &b_scale_codes, &b_noise);

    // ---- ScaleStark: one tuple per matrix row, exactly what InputQuant committed. ----
    let scale_program = ScaleProgram::new(h, w, k, r);
    let tuples = |b_side: bool| -> Vec<ScaleRowTuple> {
        (0..if b_side { w } else { h })
            .map(|g| {
                let v: &InputQuantColumnsView<F> = iq_rows[g * k + k - 1].borrow();
                if b_side {
                    ScaleRowTuple {
                        l2_frame_sum: to_u64(v.block_l2_b.running_l2_frame_sum),
                        frame_doubled_scale_exponent: to_u64(v.block_l2_b.frame_doubled_scale_exponent) as u32,
                        max_abs: to_u64(v.max_abs_b) as u32,
                        dead_count: to_u64(v.dead_count_b) as u32,
                    }
                } else {
                    ScaleRowTuple {
                        l2_frame_sum: to_u64(v.block_l2_a.running_l2_frame_sum),
                        frame_doubled_scale_exponent: to_u64(v.block_l2_a.frame_doubled_scale_exponent) as u32,
                        max_abs: to_u64(v.max_abs_a) as u32,
                        dead_count: to_u64(v.dead_count_a) as u32,
                    }
                }
            })
            .collect()
    };
    let (scale_rows, scale_pis) = scale_program.generate_trace::<F>(&tuples(false), &tuples(true));

    // ---- Matmul: the noised fp8 codes and summand scores InputQuant committed (same element
    // order; live rows only — past `h*k`/`w*k` the InputQuant trace is the dead phantom
    // fill). ----
    let codes = |b_side: bool| -> Vec<u8> {
        let live = if b_side { w * k } else { h * k };
        iq_rows[..live]
            .iter()
            .map(|r| {
                let v: &InputQuantColumnsView<F> = r.borrow();
                to_u64(if b_side { v.code_noised_b } else { v.code_noised_a }) as u8
            })
            .collect()
    };
    let lambdas = |b_side: bool| -> Vec<u64> {
        let live = if b_side { w * k } else { h * k };
        iq_rows[..live]
            .iter()
            .map(|r| {
                let v: &InputQuantColumnsView<F> = r.borrow();
                to_u64(if b_side { v.lambda_b } else { v.lambda_a })
            })
            .collect()
    };
    let matmul_program = MatmulProgram { h, w, k };
    let mut cell_words = vec![0u32; h * w];
    let mut cell_tuples = vec![(0u64, 0u64); h * w];
    let (mat_rows, mat_pis) =
        generate_b200_trace::<F>(&matmul_program, &codes(false), &codes(true), &lambdas(false), &lambdas(true));
    for r in &mat_rows {
        let v: &MatmulB200ColumnsView<F> = r.borrow();
        // Live finals only: trailing phantom cells' finals carry no cell.
        if v.is_cell_final == F::ONE && v.is_padding == F::ZERO {
            cell_words[to_u64(v.cell_id) as usize] = (to_u64(v.cell_result_f32_lo) | (to_u64(v.cell_result_f32_hi) << 16)) as u32;
            cell_tuples[to_u64(v.cell_id) as usize] = (to_u64(v.e_cell), to_u64(v.cell_skips));
        }
    }

    // ---- Tamed: Matmul's per-cell binade-and-census tuples against Scale's per-row sigma
    // frames, reassembled exactly as the channels export them. ----
    let sigma_frames = |rows: core::ops::Range<usize>| -> Vec<(u64, u64)> {
        rows.map(|g| {
            let v: &ScaleColumnsView<F> = scale_rows[g].borrow();
            (
                to_u64(v.sigma_significand),
                to_u64(v.alpha_exp) + to_u64(v.l2_floored_exponent) + SIGMA_EXP_OFFSET,
            )
        })
        .collect()
    };
    let tamed_program = TamedProgram { h, w, k };
    let (tamed_rows, tamed_pis) = tamed_program.generate_trace::<F>(&cell_tuples, &sigma_frames(0..h), &sigma_frames(h..h + w));

    // ---- XorFoldStark: fold Matmul's cell results into the 16 lottery lanes — the
    // committed patterns' lane assignment, exactly what the API layer derives. ----
    let xf_program = XorFoldProgram {
        lanes: public.lane_assignment(),
    };
    let (xf_rows, xf_pis) = xf_program.generate_trace::<F>(&cell_words);

    // ---- Blake3Stark: the deployed program over the strips, keyed by the real job key, with
    // the lottery block = XorFold's folded words. ----
    let mut lottery_words = [0u32; 16];
    for r in &xf_rows {
        let v: &XorFoldColumnsView<F> = r.borrow();
        if v.is_lane_final == F::ONE {
            lottery_words[to_u64(v.lane_id) as usize] = (to_u64(v.rotation_input_top13)
                + (to_u64(v.rotation_input_bottom19_limb_0) << 13)
                + (to_u64(v.rotation_input_bottom19_limb_1) << 29))
                as u32;
        }
    }
    let (routing_pins, moe_schedule) = match (public.moe_statement(), &compiled.moe) {
        (Some(moe), Some(cm)) => (
            moe.routing_pins(&cm.inner_indices),
            Some(
                MoeSchedule::new(moe, public.moe().expect("MoE").experts, public.m())
                    .expect("fixture MoE schedule must be well-formed"),
            ),
        ),
        _ => (vec![], None),
    };
    let blake_program = Blake3Program::from_blake_program(&compiled.blake_proof, k, routing_pins, moe_schedule);
    let routing_words = stream_words(&private.s_routing);
    let offsets_words = stream_words(&private.s_offsets);
    let a_values = private.operands.a.values.as_bytes().to_vec();
    let b_values = private.operands.b.values.as_bytes().to_vec();
    let a_scales = private.operands.a.scales.as_bytes().to_vec();
    let b_scales = private.operands.b.scales.as_bytes().to_vec();
    let key_a: [u32; 8] = core::array::from_fn(|i| 0x1000_0001u32.wrapping_mul(i as u32 + 1));
    let key_b: [u32; 8] = core::array::from_fn(|i| 0x4000_0005u32.wrapping_mul(i as u32 + 1));
    let jackpot_key: [u32; 8] = core::array::from_fn(|i| 0x3000_0007u32.wrapping_mul(i as u32 + 1));
    let a_hash_id = public.a().hash_id;
    let b_hash_id = public.b().hash_id;
    let routing_hash_id = public.moe().map(|m| m.hash_id_r).unwrap_or(HashId::Blake3Chunk1024);
    let offsets_hash_id = public.moe().map(|m| m.hash_id_o).unwrap_or(HashId::Blake3Chunk1024);
    let (b3_rows, b3_pis) = blake_program.generate_trace::<F>(&Blake3TraceInputs {
        a_values: &a_values,
        a_scales: &a_scales,
        b_values: &b_values,
        b_scales: &b_scales,
        routing_words: &routing_words,
        offsets_words: &offsets_words,
        aux_msgs: &private.external_msgs,
        aux_cvs: &private.external_cvs,
        lottery_words,
        key_a,
        key_b,
        jackpot_key,
        a_hash_id,
        b_hash_id,
        routing_hash_id,
        offsets_hash_id,
    });

    Fp8Fixture {
        blake3: blake_program,
        input_quant: iq_program,
        scale: scale_program,
        matmul: matmul_program,
        xor_fold: xf_program,
        a_noise,
        b_noise,
        public_inputs: [
            b3_pis.to_vec(),
            iq_pis.to_vec(),
            scale_pis.to_vec(),
            mat_pis.to_vec(),
            xf_pis.to_vec(),
            tamed_pis.to_vec(),
        ],
        a_values,
        a_scales,
        b_values,
        b_scales,
        routing_words,
        offsets_words,
        aux_msgs: private.external_msgs,
        aux_cvs: private.external_cvs,
        key_a,
        key_b,
        jackpot_key,
        a_hash_id,
        b_hash_id,
        routing_hash_id,
        offsets_hash_id,
        b3_rows,
        iq_rows,
        scale_rows,
        mat_rows,
        xf_rows,
        tamed_rows,
    }
}

/// The end-to-end consistency driver: every AIR satisfied on its own trace, class (a)
/// recompute bit-exact with the traces, every LUT instance served by the
/// committed oracle, and all 24 CTL channels balanced over the full 22-table batch.
fn check_job_balances_every_ctl_channel(fx: Fp8Fixture) {
    let (h, w, k) = (fx.input_quant.h, fx.input_quant.w, fx.input_quant.k);

    // ---- Early diagnostics: the CTL surfaces have the expected row counts. ----
    let count = |get: fn(&Blake3ColumnsView<F>) -> F| {
        fx.b3_rows
            .iter()
            .filter(|r| {
                let v: &Blake3ColumnsView<F> = (*r).borrow();
                get(v) == F::ONE
            })
            .count()
    };
    assert_eq!(count(|v| v.is_int8_message), (h + w) * k / 8);
    assert_eq!(count(|v| v.is_scale_message), (h + w) * k / 32);

    // ---- Column-major views (Table order: Blake3, InputQuant, Scale, Matmul, XorFold,
    // Tamed; the sixteen LUT traces are appended below once their multiplicities are
    // known). ----
    let mut traces = vec![
        columns(&fx.b3_rows),
        columns(&fx.iq_rows),
        columns(&fx.scale_rows),
        columns(&fx.mat_rows),
        columns(&fx.xf_rows),
        columns(&fx.tamed_rows),
    ];

    // ---- Class (a) ("known") columns: every program's `known_values` — recomputed from the
    // program and public data alone — must equal its trace's leading columns bit for bit.
    // This is exactly what the batch verifier recomputes and checks the trace-commitment
    // openings against (`BatchKnownColumns`), so any divergence here is a soundness hole. ----
    let tamed_program = TamedProgram { h, w, k };
    let known = [
        fx.blake3.known_values::<F>(&Blake3KnownInputs {
            a_values_len: fx.a_values.len(),
            a_scales_len: fx.a_scales.len(),
            b_values_len: fx.b_values.len(),
            b_scales_len: fx.b_scales.len(),
        }),
        fx.input_quant.known_values::<F>(&fx.a_noise, &fx.b_noise),
        fx.scale.known_values::<F>(),
        fx.matmul.known_values::<F>(),
        fx.xor_fold.known_values::<F>(),
        tamed_program.known_values::<F>(),
    ];
    for (t, known_cols) in known.iter().enumerate() {
        assert_eq!(known_cols.len(), KNOWN_COLUMNS_PER_TABLE[t]);
        for (ci, col) in known_cols.iter().enumerate() {
            assert_eq!(&traces[t][ci], col, "table {t}: known column {ci} diverges from the trace");
        }
    }
    let batch_known = fp8_known_columns::<F>(known);
    assert!(
        batch_known.digest.is_none(),
        "the Fiat-Shamir digest is unbound until prove/verify"
    );
    assert_eq!(
        batch_known.columns_per_table[0],
        (0..KNOWN_COLUMNS_PER_TABLE[0]).collect::<Vec<_>>()
    );

    // ---- The committed LUT oracle serves every instance of every AIR's inventory: each key
    // resolves to an in-domain (slot, row) of the generated tables and the bound values equal
    // the precommitted columns there ([`LutChecker`], precise per-instance errors — the CTL
    // balance check below would only report an unbalanced multiset). ----
    let mut checker = LutChecker::<F>::new();
    checker
        .check_trace(&blake3_lut_lookups::<F>(), &traces[0], &fx.public_inputs[0], "Blake3")
        .unwrap();
    checker
        .check_trace(
            &input_quant_lut_lookups::<F>(),
            &traces[1],
            &fx.public_inputs[1],
            "InputQuant",
        )
        .unwrap();
    checker
        .check_trace(&scale_lut_lookups::<F>(&fx.scale), &traces[2], &fx.public_inputs[2], "Scale")
        .unwrap();
    checker
        .check_trace(&matmul_b200_lut_lookups::<F>(), &traces[3], &fx.public_inputs[3], "Matmul")
        .unwrap();
    checker
        .check_trace(&xor_fold_lut_lookups::<F>(), &traces[4], &fx.public_inputs[4], "XorFold")
        .unwrap();
    checker
        .check_trace(&tamed_lut_lookups::<F>(), &traces[5], &fx.public_inputs[5], "Tamed")
        .unwrap();
    // Unfiltered inventories give exact totals: RNERND x6 per InputQuant row and x3 per Scale
    // row; the Matmul backend table (B200ALIGN) x32 per Matmul row (one per
    // lane of the accumulation group).
    let mults = &checker.multiplicities;
    assert_eq!(
        mults.table_total(LutTable::RneRnd),
        (6 * fx.iq_rows.len() + 3 * fx.scale_rows.len()) as u64
    );
    assert_eq!(mults.table_total(LutTable::B200Align), (32 * fx.mat_rows.len()) as u64);

    // ---- The sixteen LUT AIRs' traces (batch tables 6..=21, [`LUT_TABLES`] order):
    // each is its precommitted block plus the accumulated multiplicity columns — the
    // per-proof online half of the committed LUT oracle. ----
    for table in LUT_TABLES {
        traces.push(lut_trace::<F>(table, mults.table_columns(table)));
    }

    // All 22 channels, assembled before the programs move into their starks below.
    let all_ctls = all_cross_table_lookups::<F>(&fx.scale);

    // ---- Every AIR satisfied on its own trace (the LUT AIRs have no constraints). ----
    let pis: [&[F]; NUM_TABLES] = core::array::from_fn(|t| fx.public_inputs[t].as_slice());
    assert_constraints!(Blake3Stark::<F, D>::new(fx.blake3), &fx.b3_rows, pis[0], "Blake3");
    assert_constraints!(
        InputQuantStark::<F, D>::new(fx.input_quant),
        &fx.iq_rows,
        pis[1],
        "InputQuant"
    );
    assert_constraints!(ScaleStark::<F, D>::new(fx.scale), &fx.scale_rows, pis[2], "Scale");
    assert_constraints!(MatmulB200Stark::<F, D>::new(fx.matmul), &fx.mat_rows, pis[3], "Matmul");
    assert_constraints!(XorFoldStark::<F, D>::new(fx.xor_fold), &fx.xf_rows, pis[4], "XorFold");
    assert_constraints!(TamedStark::<F, D>::new(tamed_program), &fx.tamed_rows, pis[5], "Tamed");

    // ---- Every CTL channel balances over the full 22-table batch: the eight main channels
    // and one per committed LUT. `check_ctls` reads non-binary filter values as
    // multiplicities — the operand channels' `w/h * IS_EVEN_ROW` looked sides, the sigma
    // channel's `W_MULT`/`H_MULT` looked side, and the LUT channels' multiplicity columns.
    // The main tables' public inputs feed InputQuant's CTL geometry terms and Scale's sigma
    // multiplicities; the LUT tables have none. ----
    let mut all_pis: Vec<Vec<F>> = fx.public_inputs.to_vec();
    all_pis.resize(traces.len(), vec![]);
    check_ctls(&traces, &all_pis, &all_ctls, &Default::default());
}

#[test]
fn one_job_balances_every_ctl_channel() {
    check_job_balances_every_ctl_channel(build_fixture(false));
}

/// The MoE fixture through the same driver: the routing tree joins the Blake3 forest (pins
/// from the public routing statement, hotspot words from the witness), and every AIR,
/// class (a) recompute, LUT instance and CTL channel must still close.
#[test]
fn one_moe_job_balances_every_ctl_channel() {
    check_job_balances_every_ctl_channel(build_fixture(true));
}

/// The compiled routing schedule is the unique-minimal opening of `R[w]`'s hotspot
/// blocks: same leaf set as the miner-side `compute_leaf_indices_from_rows`, same
/// sibling ranges as the wire proof.
#[test]
fn moe_routing_schedule_is_the_unique_minimal_opening() {
    use crate::api::fp8::openings::TreeSchedule;
    use crate::circuit::chip::blake3::program::{BLOCK_LEN, BlakeProgram, ProofSource};

    let (header, proof) = fixture_job_moe();
    let (_, public) = proof.parse_proof(&header).expect("MoE fixture must parse");
    let (_, _, cv_locs) = BlakeProgram::compile(&public);

    let moe = public.moe().expect("MoE");
    let bytes = public.num_padded_routing_entries().expect("MoE") * std::mem::size_of::<u32>();
    let schedule = TreeSchedule::new(&cv_locs, ProofSource::Routing, bytes, moe.hash_id_r);

    let hotspots: Vec<usize> = public
        .moe_statement()
        .expect("MoE")
        .opened_routing_blocks()
        .iter()
        .map(|&r| r as usize)
        .collect();
    let minimal =
        MerkleTree::compute_leaf_indices_from_rows(&hotspots, (bytes / BLOCK_LEN, BLOCK_LEN), moe.hash_id_r.chunk_len()).unwrap();
    assert_eq!(
        schedule.leaf_indices, minimal,
        "routing leaf schedule != unique-minimal leaf set"
    );

    let routing = &proof.moe_witness.as_ref().unwrap().routing;
    let mut proof_ranges: Vec<(usize, usize)> = routing
        .compute_sibling_ranges(schedule.padded_len)
        .into_iter()
        .map(|(s, e, _)| (s, e))
        .collect();
    proof_ranges.sort_unstable();
    assert_eq!(
        schedule.sibling_ranges, proof_ranges,
        "routing aux-CV ranges != proof sibling ranges"
    );
}

/// Two identical tokens in `R[w]` (unsampled neighbor slots of expert 1). The Merkle
/// opening is still valid; parse and plaintext verify must reject the duplicate.
#[test]
fn moe_duplicate_winner_token_fails_parse_and_verify() {
    let mut routing: Vec<u32> = (0..4).flat_map(|_| (0..128u32).map(|i| 2 * i)).collect();
    // Copy the token at unsampled inner 5 onto inner 4 (both in expert 1, not on the tile).
    routing[128 + 4] = routing[128 + 5];
    let (header, proof) = fixture_job_moe_with(routing);
    let err = proof
        .parse_proof(&header)
        .expect_err("duplicate R[w] entries must fail parse");
    assert!(
        format!("{err:#}").contains("strictly increasing"),
        "rejection must come from the R[w] uniqueness check, got: {err:#}"
    );
    assert!(
        verify_plain_proof(&header, &proof, None).is_err(),
        "plaintext verify must reject a duplicate in R[w]"
    );
}

/// A wrong claimed `HO` on an otherwise honest proof: the statement (and thus
/// seed-A) would bind the forged root, so the openings check must catch the
/// mismatch against the root recomputed from the disclosed `O`.
#[test]
fn moe_tampered_offsets_root_fails_parse_and_verify() {
    let (header, mut proof) = fixture_job_moe();
    proof.moe_witness.as_mut().unwrap().offsets_root[0] ^= 1;
    let err = proof.parse_proof(&header).expect_err("a forged offsets root must fail parse");
    assert!(
        format!("{err:#}").contains("offsets"),
        "rejection must come from the offsets tree reconstruction, got: {err:#}"
    );
    assert!(
        verify_plain_proof(&header, &proof, None).is_err(),
        "plaintext verify must reject a forged offsets root"
    );
}

/// [`fixture_job`]'s geometry with the first opened A row's int8 values all zero — the
/// protocol-legal all-zero-row envelope (ledger N5). The scheme floors both norms at `2^-32`,
/// alpha stays finite, beta strictly positive, and that row's noised elements are pure noise;
/// every AIR (InputQuant's floored witness, ScaleStark's in-circuit H0 floors, Matmul on the
/// noise-only codes), the class (a) recompute, the LUT domains and all 24 CTL channels must
/// still close. Regression: the InputQuant witness used to derive alpha from the unfloored
/// zero norms and panic.
#[test]
fn one_job_with_an_all_zero_opened_row_balances_every_ctl_channel() {
    let dims = &[(4, DimType::Fold), (4, DimType::Blake)];
    let (header, proof) = fixture_job_with_zero_a_rows(32, 2048, dims, dims, &[0]);
    check_job_balances_every_ctl_channel(build_fixture_from_job(header, proof));
}

/// The padded-geometry fixture ([`fixture_job_asym`]: `h = 16 != w = 20`) through the same
/// driver: every liveness/padding flag path — InputQuant's dead
/// A-region, Scale and XorFold pad rows, Matmul trailing phantom cells — must satisfy the
/// constraints, match the class (a) recompute, stay in the LUT domains, and keep all 22
/// channels balanced.
#[test]
fn one_asymmetric_job_balances_every_ctl_channel() {
    let (header, proof) = fixture_job_asym();
    check_job_balances_every_ctl_channel(build_fixture_from_job(header, proof));
}

/// The `k % 32` fixture ([`fixture_job_k32`]: `k = 2080`, committed rows that never align to
/// 64-byte Blake3 blocks) through the same driver: the schedule's
/// straddling blocks — cross-strip `PlaneBytes` and `SplitLeaf`s of both orders — must
/// satisfy the Blake3 constraints, match the class (a) recompute, and keep the values/scales
/// CTL surface exact so all 22 channels still balance against InputQuant's demand at a
/// non-power-of-two `k` (65 live rows per Matmul cell).
#[test]
fn one_k_mod_32_job_balances_every_ctl_channel() {
    let (header, proof) = fixture_job_k32();
    check_job_balances_every_ctl_channel(build_fixture_from_job(header, proof));
}

/// Below the envelope: `k = 2064` sits in `[2048, 2^16]` but `k % 32 = 16`, so
/// `PublicParams::try_new` must reject at parse time.
#[test]
fn wire_layer_rejects_k_not_multiple_of_32() {
    let dims = &[(4, DimType::Fold), (4, DimType::Blake)];
    let (header, proof) = fixture_job_with(32, 2064, dims, dims);
    let err = proof.parse_proof(&header).expect_err("k % 32 != 0 must be rejected");
    assert!(
        format!("{err:#}").contains("k must be divisible by 32"),
        "expected k-divisible-by-32 rejection, got: {err:#}"
    );
}

/// The ZK pipeline refuses to trace a policy-rejected job: TamedStark's trace generation
/// panics on the untamed fixture (its J6 gate has no satisfying row), so no proof of the
/// job can exist. The plain verifier rejects the same job with "not admissible"
/// (`crate::api::verify` tests).
#[test]
#[should_panic(expected = "untamed cells exceed the eps_tame allowance")]
fn an_untamed_job_has_no_zk_trace() {
    let (header, proof) = fixture_job_untamed();
    let _ = build_fixture_from_job(header, proof);
}
