//! Public statements and their fixed-width wire encoding.
//!
//! [`PublicParams`] is the statement shared by the plaintext and ZK verifiers:
//! the job tuple ([`JobParams`] — the per-operand parameter tuples plus the
//! proof-carried ancestor header) plus the per-proof selections (tile bases,
//! the jackpot claim, and the MoE projection). Anything *derivable* from these
//! — the per-side opening keys (`keyA`/`keyB`), aggregate operand roots
//! (`HA`/`HB`), noise seeds, and jackpot key — is computed from the statement
//! plus the caller's proposed header in [`crate::v4::api::transcript`], never
//! stored here.
//!
//! [`PublicParams::to_bytes`] / [`PublicParams::from_bytes`] use this layout:
//!
//! ```text
//! public_data := σ_d(108) ‖ pB ‖ HB_values ‖ HB_scales ‖ pA ‖ HA_values ‖ HA_scales ‖ tA ‖ tB ‖ J ‖ [ MoE tail ]
//! MoE tail    := w ‖ O_{w-1} ‖ O_w ‖ O_{e-1} ‖ HR ‖ HO ‖ I_A
//! ```
//!
//! `σ_d` (the complete 108-byte [`BlockHeader`] of the proof-carried
//! ancestor, proof commitment included) leads the blob; the depth itself is not
//! transmitted. `tA`/`tB` (4 bytes each) and `J` (32 bytes) are consumed
//! verbatim.
//! `pB`/`pA` are the per-operand tuples below, each followed by its two plane
//! commitment roots (32 bytes each: int8 values then BF16 scales). `pB`
//! precedes `pA` so its trailing `e` field (the MoE discriminator) is available
//! before `pA`'s MoE-conditional suffix is read. The MoE tail is present iff
//! `pB` encodes `e != 0`: `w` (2 bytes) and `O_{w-1}`/`O_w`/`O_{e-1}` (4 bytes
//! each) are scalars, `HR`/`HO` are 32-byte roots, and `I_A` is the A-side index
//! set (`outer_indices`, u32 LE each, no length prefix — its cardinality is
//! `rows_pattern.tile_size()`, which the params already carry). The raw
//! routing/offset lists (`O`, `R[w]`, `πR`) are witness data and never carried
//! here.
//!
//! `pB` is 21 bytes and carries the common parameters and expert count `e`.
//! It precedes `pA` so the decoder knows whether the MoE suffix is present.
//! Dense jobs encode `e = 0` and an 11-byte `pA`; MoE adds `hash_idR`/`hash_idO`
//! to `pA` (13 bytes) and includes the tail.
//!
//! `I_A` contains `rows_pattern.tile_size()` little-endian u32 indices without
//! a length prefix. Raw offsets and routing openings are private witness data.

#![allow(dead_code)]

use anyhow::{Result, bail, ensure};
use serde::{Deserialize, Serialize};

use crate::v4::api::layout::{AxisPattern, MAX_TILE_ROWS, check_lottery_layout, lane_assignment};
use crate::v4::api::prequant::BLOCK_SIZE;
use crate::v4::api::primitives::{BlockHeader, Hash256, IncompleteBlockHeader, Sides};

/// State-window depth `D`: `σ_d` must be one of the `D` most recent ancestors
/// `σ_1..σ_D` of the proposed header, never the proposed header itself.
pub const STATE_WINDOW_DEPTH: usize = 4;

/// Whitelisted keyed-BLAKE3 Merkle chunk size. Encoded as one byte (`0..=3`)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
#[cfg_attr(feature = "pyo3", pyo3::pyclass(eq, eq_int))]
pub enum HashId {
    Blake3Chunk128 = 0,
    Blake3Chunk256 = 1,
    Blake3Chunk512 = 2,
    Blake3Chunk1024 = 3,
}

impl HashId {
    pub const ALL: [Self; 4] = [
        Self::Blake3Chunk128,
        Self::Blake3Chunk256,
        Self::Blake3Chunk512,
        Self::Blake3Chunk1024,
    ];

    pub const fn chunk_len(self) -> usize {
        match self {
            Self::Blake3Chunk128 => 128,
            Self::Blake3Chunk256 => 256,
            Self::Blake3Chunk512 => 512,
            Self::Blake3Chunk1024 => 1024,
        }
    }

    pub const fn from_chunk_len(n: usize) -> Option<Self> {
        match n {
            128 => Some(Self::Blake3Chunk128),
            256 => Some(Self::Blake3Chunk256),
            512 => Some(Self::Blake3Chunk512),
            1024 => Some(Self::Blake3Chunk1024),
            _ => None,
        }
    }

    pub fn pad(self, data: &[u8]) -> Vec<u8> {
        let mut padded = data.to_vec();
        padded.resize(self.padded_len(data.len()), 0);
        padded
    }

    pub const fn padded_len(self, raw_len: usize) -> usize {
        let chunk = self.chunk_len();
        raw_len.div_ceil(chunk) * chunk
    }
}

impl TryFrom<u8> for HashId {
    type Error = anyhow::Error;

    fn try_from(value: u8) -> Result<Self> {
        match value {
            0 => Ok(Self::Blake3Chunk128),
            1 => Ok(Self::Blake3Chunk256),
            2 => Ok(Self::Blake3Chunk512),
            3 => Ok(Self::Blake3Chunk1024),
            other => bail!("unknown HashId discriminant: {other}"),
        }
    }
}

const _: () = {
    assert!(HashId::ALL.len() == pearl_blake3::ALLOWED_CHUNK_LENS.len());
    let mut i = 0;
    while i < HashId::ALL.len() {
        assert!(HashId::ALL[i].chunk_len() == pearl_blake3::ALLOWED_CHUNK_LENS[i]);
        i += 1;
    }
};

/// Committed quantization scheme. Encoded as one byte; `0` is [`Self::Fp8E4M3Prequant`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
#[cfg_attr(feature = "pyo3", pyo3::pyclass(eq, eq_int))]
pub enum Quant {
    Fp8E4M3Prequant = 0,
}

impl TryFrom<u8> for Quant {
    type Error = anyhow::Error;

    fn try_from(value: u8) -> Result<Self> {
        match value {
            0 => Ok(Self::Fp8E4M3Prequant),
            other => bail!("unknown Quant discriminant: {other}"),
        }
    }
}

/// Mining device whose matmul datapath an FP8 proof reproduces; committed in
/// `pB`'s device byte.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
#[cfg_attr(feature = "pyo3", pyo3::pyclass(eq, eq_int))]
pub enum Device {
    H100 = 0,
    B200 = 1,
}

impl Device {
    pub const ALL: [Self; 2] = [Self::H100, Self::B200];

    /// Base-two logarithm of the protocol noise-to-signal ratio used by
    /// quantization, jackpot-policy thresholds, and their matching AIR
    /// encodings.
    pub const fn lg2_delta(self) -> i32 {
        match self {
            Self::H100 => 0,
            Self::B200 => -1,
        }
    }

    /// Protocol noise-to-signal ratio `2^lg2_delta`, shared by plaintext
    /// quantization, policy replay, and the ZK witness.
    pub const fn delta(self) -> f64 {
        f64::from_bits(((self.lg2_delta() + 1023) as u64) << 52)
    }

    /// Number of fractional bits retained below this device's FP8
    /// accumulation-window anchor. The unpredictability policy and AIR use it
    /// to decide which summands hardware can drop.
    pub const fn fp8_window_bits(self) -> u32 {
        match self {
            Self::H100 => 13,
            Self::B200 => 25,
        }
    }

    /// BF16-code increment for the liveness bound: adding it to the floored-L2 code
    /// multiplies the bound by `tau_idle * delta` (an exact power of two, so the
    /// bound stays value-ordered in code space) — `+256` (`x4`) on B200,
    /// `+384` (`x8`) on H100.
    pub const fn liveness_code_shift(self) -> u64 {
        ((3 + self.lg2_delta()) * 128) as u64
    }

    /// Additive term in `enc(delta * alpha * l2) = E(alpha) + E(l2f) + wide + offset`:
    /// folds the `2^-268` BF16-unit product bias and `-log2(delta)` — `13` on B200
    /// (`delta = 2^-1`), `14` on H100 (`delta = 1`).
    pub const fn sigma_encoding_offset(self) -> u64 {
        (14 + self.lg2_delta()) as u64
    }
}

impl TryFrom<u8> for Device {
    type Error = anyhow::Error;

    fn try_from(value: u8) -> Result<Self> {
        match value {
            0 => Ok(Device::H100),
            1 => Ok(Device::B200),
            other => bail!("unsupported Device discriminant: {other}"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "pyo3", pyo3::pyclass(name = "CommonParams", get_all))]
pub struct CommonParams {
    pub k: u32,
    pub r: u16,
    pub quant: Quant,
    pub device: Device,
}

impl CommonParams {
    /// `r == 32`, `k ∈ [1024, 2^16]`, `32 | k`.
    fn check(&self) -> Result<()> {
        let k = self.k as usize;
        let r = self.r as usize;
        ensure!(r == 32, "Rank must be exactly 32 || r={r}");
        ensure!(k.is_multiple_of(32), "k must be divisible by 32 || k={k}");
        ensure!(k >= PublicParams::MIN_K, "k must be >= {} || k={k}", PublicParams::MIN_K);
        ensure!(k <= PublicParams::MAX_K, "k must be <= {} || k={k}", PublicParams::MAX_K);
        Ok(())
    }
}

/// Parameters of one matmul operand. The statement carries two of these:
/// `a` is A (`m` rows), `b` is B (`n` rows), for the product `A * Bᵀ`.
/// Both are stored as collections of row vectors, so `pattern` selects rows
/// of that operand (columns of C when this is the B side).
///
/// For MoE, `b.num_rows` is the **stacked** B height (`η · e`); `try_new`
/// requires `e | n` and tile-fits against `η = n / e`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "pyo3", pyo3::pyclass(name = "OperandParams", get_all))]
pub struct OperandParams {
    pub num_rows: u32,
    pub hash_id: HashId,
    pub pattern: AxisPattern,
}

impl OperandParams {
    fn check(&self) -> Result<()> {
        ensure!(self.num_rows >= 1, "num_rows must be >= 1 || num_rows={}", self.num_rows);
        ensure!(
            self.num_rows <= (1 << 24),
            "num_rows must be <= 2^24 || num_rows={}",
            self.num_rows
        );
        Ok(())
    }
}

/// MoE-only public fields: expert count plus the routing/offset hash IDs
/// appended to `pA` as `hash_idR` / `hash_idO`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "pyo3", pyo3::pyclass(name = "MoeParams", get_all))]
pub struct MoeParams {
    /// Expert count; `1 ≤ e ≤ 1024`.
    pub experts: u16,
    pub hash_id_r: HashId,
    pub hash_id_o: HashId,
}

impl MoeParams {
    /// Maximum expert count.
    pub const MAX_NUM_EXPERTS: u16 = 1024;
}

/// The job tuple shared by the witness ([`crate::v4::api::plain_proof::PlainProofV4`])
/// and the statement ([`PublicParams`]): the common tuple, the per-operand
/// parameter tuples, and the proof-carried ancestor header. `b` precedes `a` —
/// the protocol's commitment order (the B-side keys and seeds derive first).
///
/// # Whitepaper symbol map
/// - `ancestor_header` = `σ_d` — the selected ancestor block header, complete
///   (108 bytes, proof commitment included), carried by the proof. Verifiers
///   authenticate it as one of the window headers `σ_1..σ_D` preceding the
///   proposed header with [`Self::check_ancestry`].
/// - The proposed header `σ̂` is *not* part of the job: it is the caller's
///   header argument (`proposed_header`), which omits the proof commitment.
///
/// Both wire codecs share the part encoders ([`Self::encode_p_b`] /
/// [`Self::encode_p_a`]): the consensus `public_data` interleaves the
/// statement's plane roots between `pB` and `pA`, while the `PlainProofV4`
/// bincode carries the job contiguously via [`Self::to_wire_bytes`]
/// (`ancestor_header(108) || pB || pA`).
#[derive(Debug, Clone, PartialEq)]
pub struct JobParams {
    /// The proof-carried ancestor header `σ_d` (complete 108-byte header).
    pub ancestor_header: BlockHeader,
    /// The common tuple (`k`, `r`, `Quant`, `Device`) — rides `pB` on the wire.
    pub common: CommonParams,
    /// Operand tuples: A (`m`, `hash_idA`, `Prow`) and B (`n`, `hash_idB`, `Pcol`).
    /// Encoded B-first because `pB = common ‖ b ‖ e` determines `pA`'s MoE suffix.
    pub operands: Sides<OperandParams>,
    /// `Some` iff the job is MoE (then `b.num_rows` is the stacked height `η · e`).
    pub moe: Option<MoeParams>,
}

impl JobParams {
    /// Wire length of `pB = (n, k, r, Quant, Device, hash_idB, Pcol, e)`.
    const P_B_LEN: usize = 4 + 4 + 2 + 1 + 1 + 1 + AxisPattern::NUM_DIMS + 2;
    /// Wire length of dense `pA = (m, hash_idA, Prow)` (MoE appends two hash-ids).
    const P_A_LEN: usize = 4 + 1 + AxisPattern::NUM_DIMS;

    /// The single wire encoding of the job tuple: `ancestor_header(108) || pB || pA`.
    /// [`MoeParams`] rides inside the parts: `pB` ends with `e` (`0` when dense) and an
    /// MoE `pA` appends `hash_idR || hash_idO`.
    ///
    /// The consensus `public_data` codec interleaves the statement's plane
    /// roots between the same parts ([`PublicParams::to_bytes`]); the
    /// `PlainProofV4` bincode carries the job contiguously via
    /// [`Self::from_wire_bytes`].
    pub(crate) fn to_wire_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(BlockHeader::SERIALIZED_SIZE + Self::P_B_LEN + Self::P_A_LEN);
        out.extend_from_slice(&self.ancestor_header.to_bytes());
        out.extend_from_slice(&self.encode_p_b());
        out.extend_from_slice(&self.encode_p_a());
        out
    }

    /// Inverse of [`Self::to_wire_bytes`]: parses `ancestor_header(108) || pB || pA`
    /// (no statement material — the plane roots and tile bases ride separately).
    pub(crate) fn from_wire_bytes(wire: &[u8]) -> Result<Self> {
        let mut s = wire;
        let ancestor_header = BlockHeader::from_bytes(take(&mut s, BlockHeader::SERIALIZED_SIZE)?)?;
        let (common, b, experts) = parse_p_b(&mut s)?;
        let (a, moe_hash_ids) = parse_p_a(&mut s, experts)?;
        ensure!(s.is_empty(), "trailing bytes in fp8 job tuple");
        let moe = moe_hash_ids.map(|(hash_id_r, hash_id_o)| MoeParams {
            experts,
            hash_id_r,
            hash_id_o,
        });
        Ok(Self {
            ancestor_header,
            common,
            operands: Sides { a, b },
            moe,
        })
    }

    /// Check individual fields. [`PublicParams::try_new`] checks relationships
    /// between them, including tile fit and `e | n`.
    fn check(&self) -> Result<()> {
        self.common.check()?;
        self.operands.b.check()?;
        self.operands.a.check()?;
        if let Some(moe) = self.moe {
            ensure!(
                moe.experts >= 1 && moe.experts <= MoeParams::MAX_NUM_EXPERTS,
                "experts must be in 1..={} || e={}",
                MoeParams::MAX_NUM_EXPERTS,
                moe.experts
            );
        }
        Ok(())
    }

    /// Authenticates `σ_d` as the depth-`d ≤ D` ancestor of `proposed_header` (`σ̂`):
    /// `chain` is exactly `σ_1..σ_{d-1}`, parent first, and each link is the SHA256d
    /// of the complete parent header.
    pub fn check_ancestry(&self, proposed_header: &IncompleteBlockHeader, chain: &[BlockHeader]) -> Result<()> {
        ensure!(
            chain.len() < STATE_WINDOW_DEPTH,
            "v4 ancestor chain holds {} headers, at most {} fit the state window",
            chain.len(),
            STATE_WINDOW_DEPTH - 1
        );
        let mut child = proposed_header;
        for (depth, parent) in (1..).zip(chain.iter().chain([&self.ancestor_header])) {
            ensure!(
                child.prev_block == parent.block_hash(),
                "v4 ancestor header at depth {depth} does not connect"
            );
            child = &parent.incomplete;
        }
        Ok(())
    }

    /// The B-side tuple `pB = (n, k, r, Quant, Device, hash_idB, Pcol, e)` —
    /// the common tuple rides `pB`. MoE `n` is stacked (`η · e`).
    pub(crate) fn encode_p_b(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(Self::P_B_LEN);
        out.extend_from_slice(&self.operands.b.num_rows.to_le_bytes());
        out.extend_from_slice(&self.common.k.to_le_bytes());
        out.extend_from_slice(&self.common.r.to_le_bytes());
        out.push(self.common.quant as u8);
        out.push(self.common.device as u8);
        out.push(self.operands.b.hash_id as u8);
        out.extend_from_slice(&self.operands.b.pattern.to_bytes());
        out.extend_from_slice(&self.experts().to_le_bytes());
        out
    }

    /// The A-side tuple `pA = (m, hash_idA, Prow[, hash_idR, hash_idO])`.
    pub(crate) fn encode_p_a(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(Self::P_A_LEN);
        out.extend_from_slice(&self.operands.a.num_rows.to_le_bytes());
        out.push(self.operands.a.hash_id as u8);
        out.extend_from_slice(&self.operands.a.pattern.to_bytes());
        if let Some(moe) = self.moe {
            out.push(moe.hash_id_r as u8);
            out.push(moe.hash_id_o as u8);
        }
        out
    }

    /// `0` when dense.
    pub(crate) fn experts(&self) -> u16 {
        self.moe.map_or(0, |m| m.experts)
    }
}

/// MoE public fields: winner, routing counts and roots, and selected A rows.
/// The routing and offsets lists are private witness data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct MoEStatement {
    /// Winner expert `w`.
    pub w: u16,
    /// Cumulative routing counts bracketing the winner's slice: `O_{w-1}`, `O_w`.
    pub o_w_prev: u32,
    pub o_w: u32,
    /// Total routing count `O_{e-1}`.
    pub o_last: u32,
    /// Routing root `HR`.
    pub hash_routing: Hash256,
    /// Offset root `HO`.
    pub hash_offsets: Hash256,
    /// The A-side index set `I_A` (`outer_indices`); length = `rows_pattern.tile_size()`.
    pub i_a: Vec<u32>,
}

impl MoEStatement {
    fn check(&self, moe: MoeParams, a: &OperandParams, t_a: u32) -> Result<()> {
        ensure!(
            self.w < moe.experts,
            "winner expert must satisfy 0 <= w < e || w={} e={}",
            self.w,
            moe.experts
        );
        let h = a.pattern.tile_size() as usize;
        ensure!(
            self.i_a.len() == h,
            "I_A length must equal h || |I_A|={} h={}",
            self.i_a.len(),
            h
        );
        ensure!(self.i_a.windows(2).all(|w| w[0] < w[1]), "I_A must be strictly increasing");
        ensure!(
            self.i_a.iter().all(|&i| i < a.num_rows),
            "I_A must be a subset of [0, m) || m={}",
            a.num_rows
        );
        ensure!(
            self.o_w_prev <= self.o_w,
            "O_{{w-1}} must be <= O_w || O_{{w-1}}={} O_w={}",
            self.o_w_prev,
            self.o_w
        );
        ensure!(
            self.o_w <= self.o_last,
            "O_w must be <= O_{{e-1}} || O_w={} O_{{e-1}}={}",
            self.o_w,
            self.o_last
        );
        if self.w == 0 {
            ensure!(
                self.o_w_prev == 0,
                "O_{{w-1}} must be 0 when w=0 || O_{{w-1}}={}",
                self.o_w_prev
            );
        }
        if self.w == moe.experts - 1 {
            ensure!(
                self.o_w == self.o_last,
                "O_w must equal O_{{e-1}} when w=e-1 || O_w={} O_{{e-1}}={}",
                self.o_w,
                self.o_last
            );
        }
        let winner_row_count = self.o_w - self.o_w_prev;
        ensure!(
            winner_row_count <= a.num_rows,
            "O_w - O_{{w-1}} must be <= m || s_w={} m={}",
            winner_row_count,
            a.num_rows
        );
        // Routing byte spans stay <= 2^31 after pad-to-16 (2^29 is 16-divisible):
        // every downstream offset is plain `usize`/`u32` arithmetic on any platform.
        ensure!(
            self.o_last <= 1 << 29,
            "O_{{e-1}} must be <= 2^29 || O_{{e-1}}={}",
            self.o_last
        );
        let last_selected_row = t_a
            .checked_add(a.pattern.tile_max())
            .ok_or_else(|| anyhow::anyhow!("tA + tile max overflows u32"))?;
        ensure!(
            last_selected_row < winner_row_count,
            "tA={} + tile max={} must be < |R[w]|={} (U_A subset of [0, s_w))",
            t_a,
            a.pattern.tile_max(),
            winner_row_count
        );
        Ok(())
    }

    /// Indices of the 64-byte BLAKE3 blocks covering the winner's routing entries.
    /// Returns no blocks for an empty slice. These are message blocks, smaller than
    /// the Merkle chunks selected by [`HashId`].
    pub(crate) fn opened_routing_blocks(&self) -> Vec<u32> {
        const BLOCK: usize = pearl_blake3::BLAKE3_MSG_LEN;
        let byte_start = self.o_w_prev as usize * std::mem::size_of::<u32>();
        let byte_end = self.o_w as usize * std::mem::size_of::<u32>();
        if byte_end <= byte_start {
            return vec![];
        }
        let start = byte_start / BLOCK;
        let end = byte_end.div_ceil(BLOCK);
        (start..end).map(|i| i as u32).collect()
    }
}

/// The proof selection carried by the statement: the tile bases, the jackpot
/// claim, and the per-side aggregate commitment digests. Chosen after the
/// commitment and before the openings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct JackpotStatement {
    /// Tile bases `(tA, tB)`: the row/column offsets of the lottery tile `T`.
    pub tile_bases: Sides<u32>,
    /// The claimed jackpot digest `J` (the keyed lottery ticket to compare
    /// against the difficulty bound).
    pub hash_jackpot: Hash256,
    /// `H_A`: the A-side commitment digest.
    pub hash_a: Hash256,
    /// `H_B`: the B-side commitment digest.
    pub hash_b: Hash256,
}

impl JackpotStatement {
    fn check(&self, a: &OperandParams, b: &OperandParams, b_row_bound: u32) -> Result<()> {
        ensure!(
            a.pattern.offset_is_valid(self.tile_bases.a),
            "tA={} must be a valid tile base for the A-side pattern",
            self.tile_bases.a
        );
        ensure!(
            b.pattern.offset_is_valid(self.tile_bases.b),
            "tB={} must be a valid tile base for the B-side pattern",
            self.tile_bases.b
        );
        ensure!(
            self.tile_bases.a + a.pattern.tile_max() < a.num_rows,
            "tA={} + tile max={} must be < m={}",
            self.tile_bases.a,
            a.pattern.tile_max(),
            a.num_rows
        );
        ensure!(
            self.tile_bases.b + b.pattern.tile_max() < b_row_bound,
            "tB={} + tile max={} must be < {} (per-expert B rows)",
            self.tile_bases.b,
            b.pattern.tile_max(),
            b_row_bound
        );
        Ok(())
    }
}

/// The public statement: the job tuple plus per-proof selections (tile bases,
/// jackpot claim, MoE projection). Wire encoding is [`Self::to_bytes`].
#[derive(Debug, Clone)]
pub struct PublicParams {
    job: JobParams,
    /// The jackpot selection (tile bases, jackpot claim, committed plane roots).
    jackpot_statement: JackpotStatement,
    /// Public MoE fields; `None` for dense jobs.
    moe_statement: Option<MoEStatement>,
}

impl PublicParams {
    /// Inclusive consensus bounds for the common matmul dimension (`k ∈ [1024, 2^16]`).
    pub const MIN_K: usize = 1 << 10;
    pub const MAX_K: usize = 1 << 16;
    /// Compression-message bytes in the four operand trees, optional MoE trees,
    /// the two root folds, and final jackpot. Touched Merkle parents count;
    /// unopened sibling CVs are supplied without compression work.
    pub const MAX_BLAKE3_WORK_BYTES: usize = 1 << 22;

    /// Byte length of a dense (non-MoE) statement:
    /// σ_d (108) + pB (21) + HB (32) + pA (11) + HA (32) + tile bases (8) + J (32).
    pub const WIRE_SIZE: usize = 244;

    /// Largest statement byte length: the dense core plus the largest supported MoE tail.
    pub const MAX_WIRE_SIZE: usize = Self::WIRE_SIZE + 2 + 2 + 3 * 4 + 2 * 32 + 4 * MAX_TILE_ROWS;

    /// Whether `len` is a plausible statement length. Exact validation happens in
    /// [`Self::from_bytes`].
    pub fn is_valid_wire_size(len: usize) -> bool {
        (Self::WIRE_SIZE..=Self::MAX_WIRE_SIZE).contains(&len)
    }

    pub(crate) fn try_new(
        job: JobParams,
        jackpot_statement: JackpotStatement,
        moe_statement: Option<MoEStatement>,
    ) -> Result<Self> {
        job.check()?;
        check_lottery_layout(&job.operands.a.pattern, &job.operands.b.pattern)?;
        check_opened_strips_bound(
            job.operands.a.pattern.tile_size() as usize,
            job.operands.b.pattern.tile_size() as usize,
            job.common.k as usize,
        )?;
        ensure!(
            job.moe.is_some() == moe_statement.is_some(),
            "moe and moe_statement must both be present or both absent"
        );
        let b_row_bound = if let (Some(moe), Some(moe_statement)) = (job.moe, &moe_statement) {
            moe_statement.check(moe, &job.operands.a, jackpot_statement.tile_bases.a)?;
            ensure!(
                job.operands.b.num_rows.is_multiple_of(u32::from(moe.experts)),
                "n must be divisible by e || n={} e={}",
                job.operands.b.num_rows,
                moe.experts
            );
            job.operands.b.num_rows / u32::from(moe.experts)
        } else {
            job.operands.b.num_rows
        };
        jackpot_statement.check(&job.operands.a, &job.operands.b, b_row_bound)?;
        let params = Self {
            job,
            jackpot_statement,
            moe_statement,
        };
        params.check_blake3_work_bound()?;
        Ok(params)
    }

    /// Repeat [`Self::try_new`]'s MoE checks before verification to catch
    /// statements modified after construction.
    pub(crate) fn recheck_moe(&self) -> Result<()> {
        if let (Some(moe), Some(stmt)) = (self.moe(), self.moe_statement.as_ref()) {
            stmt.check(*moe, self.a(), self.jackpot_statement.tile_bases.a)?;
        }
        self.check_blake3_work_bound()?;
        Ok(())
    }

    /// Bound the compressions the public Blake3 opening forest will compile, from the
    /// statement's scalars alone, and reject the statement when the bound exceeds
    /// [`Self::MAX_BLAKE3_WORK_BYTES`]. Runs in [`Self::try_new`], before job
    /// derivation lets the compiler allocate its per-instruction state.
    fn check_blake3_work_bound(&self) -> Result<()> {
        ensure!(
            self.blake3_work_block_bound()? <= (Self::MAX_BLAKE3_WORK_BYTES / blake3::BLOCK_LEN) as u64,
            "FP8 Blake3 work exceeds 2^22 bytes"
        );
        Ok(())
    }

    fn blake3_work_block_bound(&self) -> Result<u64> {
        let mut count: u64 = 3; // Two operand-root folds and the final jackpot compression.
        let k = u64::from(self.common_dim());
        let scale_row_bytes = 2 * (k / BLOCK_SIZE as u64);
        let a_rows = self.a_rows_indices();
        let b_rows = self.b_rows_indices();

        for (total_rows, row_bytes, hash_id, rows) in [
            (u64::from(self.m()), k, self.a().hash_id, &a_rows),
            (u64::from(self.n()), k, self.b().hash_id, &b_rows),
            (u64::from(self.m()), scale_row_bytes, self.a().hash_id, &a_rows),
            (u64::from(self.n()), scale_row_bytes, self.b().hash_id, &b_rows),
        ] {
            count += blake_tree_block_bound(
                total_rows * row_bytes,
                hash_id,
                rows.iter()
                    .map(|&row| (u64::from(row) * row_bytes, (u64::from(row) + 1) * row_bytes)),
            )?;
        }
        if let (Some(moe), Some(stmt)) = (self.moe(), self.moe_statement()) {
            let width = std::mem::size_of::<u32>() as u64;
            count += blake_tree_block_bound(
                u64::from(stmt.o_last) * width,
                moe.hash_id_r,
                std::iter::once((u64::from(stmt.o_w_prev) * width, u64::from(stmt.o_w) * width)),
            )?;
            let offsets_bytes = u64::from(moe.experts) * width;
            count += blake_tree_block_bound(offsets_bytes, moe.hash_id_o, std::iter::once((0, offsets_bytes)))?;
        }
        Ok(count)
    }

    /// Builds the statement from its wire [`Self::to_bytes`] encoding.
    pub fn from_bytes(public_data: &[u8]) -> Result<Self> {
        let mut remaining = public_data;

        // ---- the proof-carried ancestor header σ_d (108 bytes) ----
        let ancestor_header = BlockHeader::from_bytes(take(&mut remaining, BlockHeader::SERIALIZED_SIZE)?)?;

        let (common, b, experts) = parse_p_b(&mut remaining)?;
        let hash_b: Hash256 = take(&mut remaining, 32)?.try_into().unwrap();

        let (a, moe_hash_ids) = parse_p_a(&mut remaining, experts)?;
        let hash_a: Hash256 = take(&mut remaining, 32)?.try_into().unwrap();

        let t_a = u32::from_le_bytes(take(&mut remaining, 4)?.try_into().unwrap());
        let t_b = u32::from_le_bytes(take(&mut remaining, 4)?.try_into().unwrap());
        let hash_jackpot: Hash256 = take(&mut remaining, 32)?.try_into().unwrap();

        let (moe, moe_statement) = if experts != 0 {
            let w = u16::from_le_bytes(take(&mut remaining, 2)?.try_into().unwrap());
            let winner_start = u32::from_le_bytes(take(&mut remaining, 4)?.try_into().unwrap());
            let o_w = u32::from_le_bytes(take(&mut remaining, 4)?.try_into().unwrap());
            let routing_count = u32::from_le_bytes(take(&mut remaining, 4)?.try_into().unwrap());
            let hash_routing: Hash256 = take(&mut remaining, 32)?.try_into().unwrap();
            let hash_offsets: Hash256 = take(&mut remaining, 32)?.try_into().unwrap();
            // The pattern controls `I_A`'s length. Check the tile bounds and remaining
            // bytes before allocating, so a small malformed blob cannot request a
            // large buffer.
            check_lottery_layout(&a.pattern, &b.pattern)?;
            let i_a_len = a.pattern.tile_size() as usize;
            let i_a_bytes = i_a_len
                .checked_mul(std::mem::size_of::<u32>())
                .ok_or_else(|| anyhow::anyhow!("|I_A| * 4 overflows usize || |I_A|={i_a_len}"))?;
            ensure!(remaining.len() >= i_a_bytes, "truncated fp8 public_data");
            let mut i_a = Vec::with_capacity(i_a_len);
            for _ in 0..i_a_len {
                i_a.push(u32::from_le_bytes(take(&mut remaining, 4)?.try_into().unwrap()));
            }
            let (hash_id_r, hash_id_o) = moe_hash_ids.expect("MoE pA carries the routing/offset hash-ids");
            (
                Some(MoeParams {
                    experts,
                    hash_id_r,
                    hash_id_o,
                }),
                Some(MoEStatement {
                    w,
                    o_w_prev: winner_start,
                    o_w,
                    o_last: routing_count,
                    hash_routing,
                    hash_offsets,
                    i_a,
                }),
            )
        } else {
            (None, None)
        };
        ensure!(remaining.is_empty(), "trailing bytes in fp8 public_data");

        let params = Self::try_new(
            JobParams {
                ancestor_header,
                common,
                operands: Sides { a, b },
                moe,
            },
            JackpotStatement {
                tile_bases: Sides { a: t_a, b: t_b },
                hash_jackpot,
                hash_a,
                hash_b,
            },
            moe_statement,
        )?;
        ensure!(
            params.to_bytes().as_slice() == public_data,
            "non-canonical V4 public_data encoding"
        );
        Ok(params)
    }

    /// Encodes the full statement to the wire [`Self::from_bytes`] layout.
    pub(crate) fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(Self::MAX_WIRE_SIZE);
        out.extend_from_slice(&self.job.ancestor_header.to_bytes());
        out.extend_from_slice(&self.job.encode_p_b());
        out.extend_from_slice(&self.jackpot_statement.hash_b); // H_B
        out.extend_from_slice(&self.job.encode_p_a());
        out.extend_from_slice(&self.jackpot_statement.hash_a); // H_A
        out.extend_from_slice(&self.jackpot_statement.tile_bases.a.to_le_bytes());
        out.extend_from_slice(&self.jackpot_statement.tile_bases.b.to_le_bytes());
        out.extend_from_slice(&self.jackpot_statement.hash_jackpot);
        if let Some(stmt) = &self.moe_statement {
            out.extend_from_slice(&stmt.w.to_le_bytes());
            out.extend_from_slice(&stmt.o_w_prev.to_le_bytes());
            out.extend_from_slice(&stmt.o_w.to_le_bytes());
            out.extend_from_slice(&stmt.o_last.to_le_bytes());
            out.extend_from_slice(&stmt.hash_routing);
            out.extend_from_slice(&stmt.hash_offsets);
            for &idx in &stmt.i_a {
                out.extend_from_slice(&idx.to_le_bytes());
            }
        }
        out
    }

    /// The job tuple the statement was constructed from.
    pub(crate) fn job(&self) -> &JobParams {
        &self.job
    }

    /// The proof-carried ancestor header `σ_d`, authenticated by [`JobParams::check_ancestry`].
    pub fn ancestor_header(&self) -> &BlockHeader {
        &self.job.ancestor_header
    }

    pub(crate) fn common(&self) -> &CommonParams {
        &self.job.common
    }

    pub(crate) fn a(&self) -> &OperandParams {
        &self.job.operands.a
    }

    pub(crate) fn b(&self) -> &OperandParams {
        &self.job.operands.b
    }

    pub(crate) fn moe(&self) -> Option<&MoeParams> {
        self.job.moe.as_ref()
    }

    pub(crate) fn tile_bases(&self) -> Sides<u32> {
        self.jackpot_statement.tile_bases
    }

    pub(crate) fn hash_jackpot(&self) -> Hash256 {
        self.jackpot_statement.hash_jackpot
    }

    /// Set the jackpot after proving. `parse_proof` initializes it to zero;
    /// verification binds the statement's claim to the proven digest.
    pub(crate) fn set_hash_jackpot(&mut self, hash: Hash256) {
        self.jackpot_statement.hash_jackpot = hash;
    }

    /// The aggregate A-side commitment digest `H_A` (the folded `hash_a`).
    pub(crate) fn hash_a(&self) -> Hash256 {
        self.jackpot_statement.hash_a
    }

    /// The aggregate B-side commitment digest `H_B` (the folded `hash_b`).
    pub(crate) fn hash_b(&self) -> Hash256 {
        self.jackpot_statement.hash_b
    }

    /// The per-side aggregate commitment digests `Sides { a: H_A, b: H_B }` for the
    /// noise-seed chain.
    pub(crate) fn commitment_digests(&self) -> Sides<Hash256> {
        Sides {
            a: self.hash_a(),
            b: self.hash_b(),
        }
    }

    pub(crate) fn moe_statement(&self) -> Option<&MoEStatement> {
        self.moe_statement.as_ref()
    }

    #[cfg(test)]
    pub(crate) fn moe_statement_mut(&mut self) -> Option<&mut MoEStatement> {
        self.moe_statement.as_mut()
    }

    /// Replaces the committed device for proof-transfer rejection tests.
    #[cfg(test)]
    pub(crate) fn set_device_for_test(&mut self, device: Device) {
        self.job.common.device = device;
    }

    pub(crate) fn value_row_bytes(&self) -> usize {
        usize::try_from(self.job.common.k).expect("k fits usize")
    }

    pub(crate) fn scale_row_bytes(&self) -> usize {
        2 * (self.value_row_bytes() / BLOCK_SIZE)
    }
    pub(crate) fn lane_assignment(&self) -> Vec<Vec<usize>> {
        lane_assignment(&self.job.operands.a.pattern, &self.job.operands.b.pattern)
    }

    /// Common matmul dimension `k`.
    pub(crate) fn common_dim(&self) -> u32 {
        self.job.common.k
    }

    /// Additive-noise rank `r`.
    pub(crate) fn rank(&self) -> u32 {
        self.job.common.r as u32
    }

    /// Rows of the lottery tile: `rows_pattern.tile_size()`.
    pub(crate) fn h(&self) -> u32 {
        self.job.operands.a.pattern.tile_size()
    }

    /// Cols of the lottery tile: `cols_pattern.tile_size()`.
    pub(crate) fn w(&self) -> u32 {
        self.job.operands.b.pattern.tile_size()
    }

    /// `m`: number of rows of `A`.
    pub(crate) fn m(&self) -> u32 {
        self.job.operands.a.num_rows
    }

    /// `n`: number of rows of `B` (the committed B-tree height).
    ///
    /// Dense: rows of one B matrix. MoE: the **total** across all experts
    /// (`η · e`), not the per-expert count `η` — see [`Self::expert_rows`].
    pub(crate) fn n(&self) -> u32 {
        self.job.operands.b.num_rows
    }

    /// Per-expert B rows `η`. Dense: same as [`Self::n`]. MoE: `n / e`
    /// (`try_new` requires `e | n`).
    pub(crate) fn expert_rows(&self) -> u32 {
        match self.job.moe {
            None => self.job.operands.b.num_rows,
            Some(moe) => self.job.operands.b.num_rows / u32::from(moe.experts),
        }
    }

    /// The lottery tile's A-row indices (dense): `rows_pattern` tile offsets based at `tA`.
    pub(crate) fn a_inner_indices(&self) -> Vec<u32> {
        let base = self.jackpot_statement.tile_bases.a;
        self.job
            .operands
            .a
            .pattern
            .tile_offsets()
            .into_iter()
            .map(|i| i + base)
            .collect()
    }

    /// The A-row indices the committed values tree opens: MoE uses the winner's outer
    /// indices (`I_A`), dense uses [`Self::a_inner_indices`].
    pub(crate) fn a_rows_indices(&self) -> Vec<u32> {
        match &self.moe_statement {
            Some(stmt) => stmt.i_a.clone(),
            None => self.a_inner_indices(),
        }
    }

    /// The lottery tile's B-row indices: `cols_pattern` tile offsets based at `tB`,
    /// offset by the winner's expert column block (`w · η`) in MoE.
    pub(crate) fn b_rows_indices(&self) -> Vec<u32> {
        let base = self.jackpot_statement.tile_bases.b;
        let expert_row_offset = self
            .moe_statement
            .as_ref()
            .map_or(0, |stmt| self.expert_rows() * u32::from(stmt.w));
        self.job
            .operands
            .b
            .pattern
            .tile_offsets()
            .into_iter()
            .map(|i| expert_row_offset + base + i)
            .collect()
    }

    /// Number of real (unpadded) entries in the flattened routing: the total cumulative
    /// count `O_{e-1}` (`o_last`). `None` outside the MoE case.
    pub(crate) fn num_routing_entries(&self) -> Option<usize> {
        self.moe_statement.as_ref().map(|stmt| stmt.o_last as usize)
    }

    /// [`Self::num_routing_entries`] rounded up to a multiple of 16 so the routing byte
    /// length (`u32` entries → bytes) tiles evenly into 64-byte virtual rows.
    pub(crate) fn num_padded_routing_entries(&self) -> Option<usize> {
        self.num_routing_entries().map(|n| n.next_multiple_of(16))
    }

    /// Unpadded byte length of the offsets list `O` (`e` u32 entries). `None` outside MoE.
    pub(crate) fn num_offsets_bytes(&self) -> Option<usize> {
        self.moe().map(|moe| moe.experts as usize * std::mem::size_of::<u32>())
    }

    /// The winner expert `w`; `None` for dense.
    pub(crate) fn expert_idx(&self) -> Option<u16> {
        self.moe_statement.as_ref().map(|stmt| stmt.w)
    }
}

/// Bound one tree's Blake3 compressions from its opened byte intervals, in closed
/// form from the statement's scalars. Exact on the dominant term and within
/// root-path slack overall:
///
/// - Leaves: chunks are the hashing unit, so every touched chunk costs exactly
///   `chunk_len / 64` compressions, whether fully opened or not. Touched chunks
///   per interval are exactly `ceil(end / chunk) - floor(start / chunk)`, with
///   overlapping intervals clamped so shared boundary chunks count once.
/// - Parents: `BlakeProgram::recursive_compilation` compiles an internal node iff
///   its span touches an opened chunk (untouched subtrees arrive as auxiliary CVs
///   and cost nothing). Nodes whose span lies wholly inside one interval number
///   fewer than that interval's touched chunks; every other compiled node
///   straddles an interval edge, so it sits on the root path of the interval's
///   first or last chunk — at most `2 * (depth + 1)` more per interval.
///
/// The compiler therefore never exceeds this bound, and
/// `blake3_bound_dominates_the_compiled_stark_for_all_chunk_sizes` pins both the
/// domination and its tightness.
fn blake_tree_block_bound(raw_bytes: u64, hash_id: HashId, opened_bytes: impl IntoIterator<Item = (u64, u64)>) -> Result<u64> {
    let chunk = hash_id.chunk_len() as u64;
    // Identical to `padded_len(raw_bytes) / chunk`: every raw byte lands in a
    // chunk, and padding adds at most one partial chunk.
    let total_chunks = raw_bytes.div_ceil(chunk);
    ensure!(total_chunks > 0, "FP8 Blake3 tree must contain at least one chunk");
    let mut ranges: Vec<(u64, u64)> = opened_bytes
        .into_iter()
        .filter(|&(start, end)| start < end)
        .map(|(start, end)| (start / chunk, end.div_ceil(chunk)))
        .collect();
    ranges.sort_unstable();
    let mut merged: Vec<(u64, u64)> = Vec::with_capacity(ranges.len());
    for (start, end) in ranges {
        ensure!(end <= total_chunks, "FP8 Blake3 opening exceeds its tree");
        if let Some(last) = merged.last_mut()
            && start <= last.1
        {
            last.1 = last.1.max(end);
        } else {
            merged.push((start, end));
        }
    }

    let chunk_blocks = chunk / blake3::BLOCK_LEN as u64;
    let depth = total_chunks.next_power_of_two().trailing_zeros() as u64;
    let blocks = merged
        .iter()
        .map(|(start, end)| (end - start) * (chunk_blocks + 1) + 2 * (depth + 1))
        .sum();
    Ok(blocks)
}

/// `k · (|I_A| + |I_B|) ≤ 2^22`: the opened-strips envelope (worker input, trace heights).
fn check_opened_strips_bound(h: usize, w: usize, k: usize) -> Result<()> {
    let opened = h
        .checked_add(w)
        .and_then(|hw| hw.checked_mul(k))
        .ok_or_else(|| anyhow::anyhow!("(h+w)·k overflows usize || h={h} w={w} k={k}"))?;
    ensure!(
        opened <= (1 << 22),
        "k·(|IA|+|IB|) must be <= 2^22 || (h+w)·k={opened} h={h} w={w} k={k}"
    );
    Ok(())
}

/// Consume the next `n` bytes of `s` and return them (or `Err` if short).
fn take<'a>(remaining: &mut &'a [u8], n: usize) -> Result<&'a [u8]> {
    ensure!(remaining.len() >= n, "truncated fp8 public_data");
    let (head, tail) = remaining.split_at(n);
    *remaining = tail;
    Ok(head)
}

/// Decode `pB = (n, k, r, Quant, Device, hash_idB, Pcol, e)`;
/// inverse of [`JobParams::encode_p_b`].
fn parse_p_b(remaining: &mut &[u8]) -> Result<(CommonParams, OperandParams, u16)> {
    let num_rows = u32::from_le_bytes(take(remaining, 4)?.try_into().unwrap());
    let k = u32::from_le_bytes(take(remaining, 4)?.try_into().unwrap());
    let r = u16::from_le_bytes(take(remaining, 2)?.try_into().unwrap());
    let quant = Quant::try_from(take(remaining, 1)?[0])?;
    let device = Device::try_from(take(remaining, 1)?[0])?;
    let hash_id = HashId::try_from(take(remaining, 1)?[0])?;
    let pattern = AxisPattern::from_bytes(take(remaining, AxisPattern::NUM_DIMS)?)?;
    let experts = u16::from_le_bytes(take(remaining, 2)?.try_into().unwrap());
    Ok((
        CommonParams { k, r, quant, device },
        OperandParams {
            num_rows,
            hash_id,
            pattern,
        },
        experts,
    ))
}

/// Decode `pA = (m, hash_idA, Prow[, hash_idR, hash_idO])`;
/// inverse of [`JobParams::encode_p_a`].
fn parse_p_a(remaining: &mut &[u8], experts: u16) -> Result<(OperandParams, Option<(HashId, HashId)>)> {
    let num_rows = u32::from_le_bytes(take(remaining, 4)?.try_into().unwrap());
    let hash_id = HashId::try_from(take(remaining, 1)?[0])?;
    let pattern = AxisPattern::from_bytes(take(remaining, AxisPattern::NUM_DIMS)?)?;
    let moe_hash_ids = if experts != 0 {
        Some((
            HashId::try_from(take(remaining, 1)?[0])?,
            HashId::try_from(take(remaining, 1)?[0])?,
        ))
    } else {
        None
    };
    Ok((
        OperandParams {
            num_rows,
            hash_id,
            pattern,
        },
        moe_hash_ids,
    ))
}

#[cfg(feature = "pyo3")]
#[pyo3::pymethods]
impl CommonParams {
    #[new]
    fn py_new(k: u32, r: u16, quant: Quant, device: Device) -> Self {
        Self { k, r, quant, device }
    }
}

#[cfg(feature = "pyo3")]
#[pyo3::pymethods]
impl OperandParams {
    #[new]
    fn py_new(num_rows: u32, hash_id: HashId, pattern: AxisPattern) -> Self {
        Self {
            num_rows,
            hash_id,
            pattern,
        }
    }
}

#[cfg(feature = "pyo3")]
#[pyo3::pymethods]
impl MoeParams {
    #[new]
    fn py_new(experts: u16, hash_id_r: HashId, hash_id_o: HashId) -> Self {
        Self {
            experts,
            hash_id_r,
            hash_id_o,
        }
    }
}

/// Statement fixtures shared by this file's tests and the transcript-keying
/// tests in [`crate::v4::api::transcript`]. Test-only.
#[cfg(test)]
pub(crate) mod test_fixtures {
    use super::*;
    use crate::v4::api::layout::DimType;

    pub(crate) fn rows() -> AxisPattern {
        AxisPattern::new(&[(4, DimType::Blake)]).unwrap()
    }

    pub(crate) fn cols() -> AxisPattern {
        AxisPattern::new(&[(2, DimType::Fold), (4, DimType::Blake), (16, DimType::Fold)]).unwrap()
    }

    pub(crate) fn empty_jackpot() -> JackpotStatement {
        JackpotStatement {
            tile_bases: Sides { a: 0, b: 0 },
            hash_jackpot: [0; 32],
            hash_a: [0; 32],
            hash_b: [0; 32],
        }
    }

    /// A linked header chain of the given depth: `(σ̂, σ_1..σ_{depth-1}, σ_depth)`,
    /// each `prev_block` the hash of its parent, `σ̂` being
    /// [`IncompleteBlockHeader::new_for_test`] with only its `prev_block` relinked.
    pub(crate) fn ancestry(depth: usize) -> (IncompleteBlockHeader, Vec<BlockHeader>, BlockHeader) {
        assert!(depth >= 1, "σ_d is at least the parent");
        let ancestor = BlockHeader::new_for_test(0x207FFFFF);
        // Built oldest first: each header is its predecessor's child.
        let mut lineage = vec![ancestor];
        for _ in 1..depth {
            let parent = lineage.last().unwrap();
            lineage.push(BlockHeader {
                incomplete: IncompleteBlockHeader {
                    prev_block: parent.block_hash(),
                    timestamp: parent.incomplete.timestamp + 1,
                    ..parent.incomplete
                },
                ..*parent
            });
        }
        let proposed = IncompleteBlockHeader {
            prev_block: lineage.last().unwrap().block_hash(),
            ..IncompleteBlockHeader::new_for_test(0x207FFFFF)
        };
        let chain = lineage[1..].iter().rev().copied().collect();
        (proposed, chain, ancestor)
    }

    pub(crate) fn empty_moe_statement(i_a: Vec<u32>) -> MoEStatement {
        MoEStatement {
            w: 1,
            o_w_prev: 128,
            o_w: 256,
            o_last: 512,
            hash_routing: [0; 32],
            hash_offsets: [0; 32],
            i_a,
        }
    }

    pub(crate) fn dense_params() -> PublicParams {
        PublicParams::try_new(
            JobParams {
                ancestor_header: BlockHeader::zero(),
                common: CommonParams {
                    k: 2048,
                    r: 32,
                    quant: Quant::Fp8E4M3Prequant,
                    device: Device::B200,
                },
                operands: Sides {
                    a: OperandParams {
                        num_rows: 256,
                        hash_id: HashId::Blake3Chunk1024,
                        pattern: rows(),
                    },
                    b: OperandParams {
                        num_rows: 128,
                        hash_id: HashId::Blake3Chunk1024,
                        pattern: cols(),
                    },
                },
                moe: None,
            },
            empty_jackpot(),
            None,
        )
        .unwrap()
    }

    pub(crate) fn stacked_b(dense: &PublicParams, experts: u16) -> OperandParams {
        OperandParams {
            num_rows: dense.job.operands.b.num_rows * u32::from(experts),
            ..dense.job.operands.b.clone()
        }
    }

    pub(crate) fn moe_params() -> PublicParams {
        let p = dense_params();
        let b = stacked_b(&p, 4);
        PublicParams::try_new(
            JobParams {
                ancestor_header: p.job.ancestor_header,
                common: p.job.common,
                operands: Sides {
                    a: p.job.operands.a.clone(),
                    b,
                },
                moe: Some(MoeParams {
                    experts: 4,
                    hash_id_r: HashId::Blake3Chunk1024,
                    hash_id_o: HashId::Blake3Chunk1024,
                }),
            },
            p.jackpot_statement,
            Some(empty_moe_statement(vec![0, 2, 4, 6])),
        )
        .unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::v4::api::layout::DimType::{Blake, Fold};
    use crate::v4::api::public_params::test_fixtures::{
        ancestry, cols, dense_params, empty_jackpot, empty_moe_statement, moe_params, rows, stacked_b,
    };

    const DENSE_PA: &str = "00010000030e0303030303";
    const DENSE_PB: &str = "80000000000800002000000103050e3d0303030000";
    const MOE_PA: &str = "00010000030e03030303030303";
    const MOE_PB: &str = "00020000000800002000000103050e3d0303030400";

    fn hex_bytes(s: &str) -> Vec<u8> {
        (0..s.len() / 2)
            .map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap())
            .collect()
    }

    #[test]
    fn encode_pins() {
        let dense = dense_params();
        assert_eq!(dense.job.experts(), 0);
        assert_eq!(dense.job.encode_p_a(), hex_bytes(DENSE_PA));
        assert_eq!(dense.job.encode_p_b(), hex_bytes(DENSE_PB));
        assert_eq!(dense.value_row_bytes(), 2048);
        assert_eq!(dense.scale_row_bytes(), 2 * (2048 / BLOCK_SIZE));
        assert_eq!(dense.n(), 128);
        assert_eq!(dense.expert_rows(), 128);

        let moe = moe_params();
        assert_eq!(moe.job.encode_p_a(), hex_bytes(MOE_PA));
        assert_eq!(moe.job.encode_p_b(), hex_bytes(MOE_PB));
        assert_eq!(dense.job.encode_p_a().len() + 2, moe.job.encode_p_a().len());
        assert_eq!(moe.n(), 512);
        assert_eq!(moe.expert_rows(), 128);
    }

    /// A truncated MoE tail must be rejected with an error, not panic — the
    /// `public_data` blob is proof-controlled input.
    #[test]
    fn from_bytes_rejects_truncated_moe_tail() {
        let full = moe_params().to_bytes();
        // Cut mid-`I_A` (past `w`, the O scalars, and the two roots).
        let cut = full.len() - 2 * 4;
        let truncated = &full[..cut];
        let err = PublicParams::from_bytes(truncated).unwrap_err();
        assert!(err.to_string().contains("truncated fp8 public_data"));
    }

    /// A short malformed blob must not cause a large allocation from its
    /// claimed tile size. Reject the pattern before reserving `I_A`.
    #[test]
    fn from_bytes_rejects_oversized_pattern_before_i_a_alloc() {
        let huge = AxisPattern::new(&[(1 << 24, Fold)]).unwrap();
        assert_eq!(huge.tile_size(), 1 << 24);

        let mut job = moe_params().job().clone();
        job.operands.a.pattern = huge;

        // Assemble `public_data` exactly as `to_bytes` lays it out, with the
        // fixed MoE tail fields present and no `I_A` entries.
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&job.ancestor_header.to_bytes());
        bytes.extend_from_slice(&job.encode_p_b());
        bytes.extend_from_slice(&[0u8; 32]); // H_B
        bytes.extend_from_slice(&job.encode_p_a());
        bytes.extend_from_slice(&[0u8; 32]); // H_A
        bytes.extend_from_slice(&0u32.to_le_bytes()); // tA
        bytes.extend_from_slice(&0u32.to_le_bytes()); // tB
        bytes.extend_from_slice(&[0u8; 32]); // J
        bytes.extend_from_slice(&1u16.to_le_bytes()); // w
        bytes.extend_from_slice(&128u32.to_le_bytes()); // O_{w-1}
        bytes.extend_from_slice(&256u32.to_le_bytes()); // O_w
        bytes.extend_from_slice(&512u32.to_le_bytes()); // O_{e-1}
        bytes.extend_from_slice(&[0u8; 32]); // HR
        bytes.extend_from_slice(&[0u8; 32]); // HO

        let err = PublicParams::from_bytes(&bytes).unwrap_err();
        assert!(
            err.to_string().contains("h*w must be <="),
            "expected lottery-envelope rejection before the I_A allocation, got: {err}"
        );
    }

    #[test]
    fn construction_rejects() {
        let prow = rows().to_bytes();
        let pcol = cols().to_bytes();

        // Builds params from unchecked parts so the rejection paths stay exercised.
        let build = |m: usize, k: usize, row: &[u8], experts: Option<usize>| -> Result<PublicParams> {
            let (moe, moe_statement) = match experts {
                None => (None, None),
                Some(e) => (
                    Some(MoeParams {
                        experts: u16::try_from(e)?,
                        hash_id_r: HashId::Blake3Chunk1024,
                        hash_id_o: HashId::Blake3Chunk1024,
                    }),
                    Some(empty_moe_statement(vec![0, 2, 4, 6])),
                ),
            };
            PublicParams::try_new(
                JobParams {
                    ancestor_header: BlockHeader::zero(),
                    common: CommonParams {
                        k: u32::try_from(k)?,
                        r: 32,
                        quant: Quant::Fp8E4M3Prequant,
                        device: Device::B200,
                    },
                    operands: Sides {
                        a: OperandParams {
                            num_rows: u32::try_from(m)?,
                            hash_id: HashId::Blake3Chunk1024,
                            pattern: AxisPattern::from_bytes(row)?,
                        },
                        b: OperandParams {
                            num_rows: 128,
                            hash_id: HashId::Blake3Chunk1024,
                            pattern: AxisPattern::from_bytes(&pcol)?,
                        },
                    },
                    moe,
                },
                empty_jackpot(),
                moe_statement,
            )
        };

        assert!(build(256, 2048, &prow, None).is_ok());
        assert!(build(256, 7, &prow, None).is_err());
        assert!(build(256, 2048, &prow, Some(0)).is_err());
        let p = dense_params();
        let moe = Some(MoeParams {
            experts: 4,
            hash_id_r: HashId::Blake3Chunk1024,
            hash_id_o: HashId::Blake3Chunk1024,
        });
        assert!(
            PublicParams::try_new(
                JobParams {
                    ancestor_header: p.job.ancestor_header,
                    common: p.job.common,
                    operands: Sides {
                        a: p.job.operands.a.clone(),
                        b: p.job.operands.b.clone(),
                    },
                    moe,
                },
                p.jackpot_statement.clone(),
                None,
            )
            .is_err()
        );
        assert!(
            PublicParams::try_new(
                JobParams {
                    ancestor_header: p.job.ancestor_header,
                    common: p.job.common,
                    operands: Sides {
                        a: p.job.operands.a.clone(),
                        b: p.job.operands.b.clone(),
                    },
                    moe: None,
                },
                p.jackpot_statement.clone(),
                Some(empty_moe_statement(vec![0, 2, 4, 6])),
            )
            .is_err()
        );
        assert!(build((u32::MAX as usize).saturating_add(1), 1024, &prow, None).is_err());
        assert!(build(256, 1024, &[0x00, 0x03, 0x03, 0x03, 0x03, 0x03], None).is_err());
        assert!(Quant::try_from(1u8).is_err());
        assert_eq!(Device::try_from(0u8).unwrap(), Device::H100);
        assert!(Device::try_from(2u8).is_err());
        assert_eq!(Device::try_from(1u8).unwrap(), Device::B200);
        assert!(HashId::try_from(4u8).is_err());
        assert_eq!(HashId::from_chunk_len(128), Some(HashId::Blake3Chunk128));
        assert_eq!(HashId::from_chunk_len(256), Some(HashId::Blake3Chunk256));
        assert_eq!(HashId::from_chunk_len(512), Some(HashId::Blake3Chunk512));
        assert_eq!(HashId::from_chunk_len(1024), Some(HashId::Blake3Chunk1024));
        assert!(HashId::from_chunk_len(64).is_none());
        assert_eq!(HashId::ALL.map(HashId::chunk_len), [128, 256, 512, 1024]);
        assert_eq!(HashId::Blake3Chunk128.padded_len(1), 128);
        assert_eq!(HashId::Blake3Chunk128.pad(&[1]).len(), 128);
        assert_eq!(HashId::Blake3Chunk1024.padded_len(1), 1024);
    }

    #[test]
    fn try_new_rejects_non_monotonic_i_a() {
        let p = dense_params();
        let moe = Some(MoeParams {
            experts: 4,
            hash_id_r: HashId::Blake3Chunk1024,
            hash_id_o: HashId::Blake3Chunk1024,
        });
        let b = stacked_b(&p, 4);
        let build = |i_a: Vec<u32>| {
            PublicParams::try_new(
                JobParams {
                    ancestor_header: p.job.ancestor_header,
                    common: p.job.common,
                    operands: Sides {
                        a: p.job.operands.a.clone(),
                        b: b.clone(),
                    },
                    moe,
                },
                p.jackpot_statement.clone(),
                Some(empty_moe_statement(i_a)),
            )
        };
        assert!(build(vec![0, 2, 4, 6]).is_ok());
        assert!(build(vec![0, 2, 2, 6]).is_err());
        assert!(build(vec![6, 4, 2, 0]).is_err());
    }

    #[test]
    fn wire_roundtrip_and_truncation_reject() {
        for device in [Device::H100, Device::B200] {
            let mut dense = dense_params();
            dense.job.common.device = device;
            let mut moe = moe_params();
            moe.job.common.device = device;
            for params in [dense, moe] {
                let bytes = params.to_bytes();
                let parsed = PublicParams::from_bytes(&bytes).unwrap();
                assert_eq!(parsed.common().device, device);
                assert_eq!(parsed.to_bytes(), bytes, "decode must be canonical");

                // Regression: truncating inside the MoE i_a tail used to panic in
                // `from_bytes` (unwrap on a short slice); every strict prefix must
                // now return an error instead.
                for len in 0..bytes.len() {
                    assert!(
                        PublicParams::from_bytes(&bytes[..len]).is_err(),
                        "strict prefix of {len} bytes must be rejected"
                    );
                }

                let mut trailing = bytes.clone();
                trailing.push(0);
                assert!(
                    PublicParams::from_bytes(&trailing).is_err(),
                    "trailing bytes must be rejected"
                );
            }
        }
    }

    #[test]
    fn try_new_rejects_invalid_tile_bases() {
        let p = dense_params();
        let build = |t_a, t_b| {
            let mut jackpot = p.jackpot_statement.clone();
            jackpot.tile_bases = Sides { a: t_a, b: t_b };
            PublicParams::try_new(p.job.clone(), jackpot, None)
        };
        assert!(build(0, 0).is_ok());
        assert!(build(4, 0).is_ok(), "next A-period tA=4 still fits in m");
        assert!(
            build(4, 128).is_err(),
            "tB=128 is on the B lattice but tB + tile_max must be < n"
        );
        assert!(build(1, 0).is_err(), "tA must sit on the A-side lattice");
        assert!(build(0, 1).is_err(), "tB must sit on the B-side lattice");
        assert!(build(u32::MAX, 0).is_err(), "tA must not wrap tile_offsets");
        assert!(build(0, u32::MAX).is_err(), "tB must not wrap tile_offsets");
    }

    fn with_kr(rank: u16, k: u32) -> Result<PublicParams> {
        let p = dense_params();
        PublicParams::try_new(
            JobParams {
                ancestor_header: p.job.ancestor_header,
                common: CommonParams {
                    k,
                    r: rank,
                    quant: p.job.common.quant,
                    device: p.job.common.device,
                },
                operands: Sides {
                    a: p.job.operands.a.clone(),
                    b: p.job.operands.b.clone(),
                },
                moe: None,
            },
            p.jackpot_statement,
            None,
        )
    }

    #[test]
    fn envelope_launch_geometry_passes() {
        with_kr(32, PublicParams::MIN_K as u32).unwrap();
        with_kr(32, 2048).unwrap();
    }

    #[test]
    fn rank_zero_is_rejected() {
        let err = with_kr(0, 2048).unwrap_err();
        assert!(err.to_string().contains("Rank must be exactly 32"));
    }

    #[test]
    fn legacy_rank_16_is_rejected() {
        let err = with_kr(16, 2048).unwrap_err();
        assert!(err.to_string().contains("Rank must be exactly 32"));
    }

    #[test]
    fn k_below_minimum_is_rejected() {
        let err = with_kr(32, (PublicParams::MIN_K - 32) as u32).unwrap_err();
        assert!(err.to_string().contains(&format!("k must be >= {}", PublicParams::MIN_K)));
    }

    #[test]
    fn minimal_4x64_tile_passes() {
        let p = dense_params();
        let rows = AxisPattern::new(&[(4, Blake)]).unwrap();
        let cols = AxisPattern::new(&[(4, Blake), (16, Fold)]).unwrap();
        PublicParams::try_new(
            JobParams {
                ancestor_header: p.job.ancestor_header,
                common: p.job.common,
                operands: Sides {
                    a: OperandParams {
                        num_rows: 4,
                        hash_id: p.job.operands.a.hash_id,
                        pattern: rows,
                    },
                    b: OperandParams {
                        num_rows: 64,
                        hash_id: p.job.operands.b.hash_id,
                        pattern: cols,
                    },
                },
                moe: None,
            },
            empty_jackpot(),
            None,
        )
        .unwrap();
    }

    #[test]
    fn zero_m_is_rejected() {
        let p = dense_params();
        let err = PublicParams::try_new(
            JobParams {
                ancestor_header: p.job.ancestor_header,
                common: p.job.common,
                operands: Sides {
                    a: OperandParams {
                        num_rows: 0,
                        hash_id: p.job.operands.a.hash_id,
                        pattern: p.job.operands.a.pattern.clone(),
                    },
                    b: p.job.operands.b.clone(),
                },
                moe: None,
            },
            p.jackpot_statement,
            None,
        )
        .unwrap_err();
        assert!(err.to_string().contains("num_rows must be >= 1"));
    }

    #[test]
    fn zero_n_is_rejected() {
        let p = dense_params();
        let err = PublicParams::try_new(
            JobParams {
                ancestor_header: p.job.ancestor_header,
                common: p.job.common,
                operands: Sides {
                    a: p.job.operands.a.clone(),
                    b: OperandParams {
                        num_rows: 0,
                        hash_id: p.job.operands.b.hash_id,
                        pattern: p.job.operands.b.pattern.clone(),
                    },
                },
                moe: None,
            },
            p.jackpot_statement,
            None,
        )
        .unwrap_err();
        assert!(err.to_string().contains("num_rows must be >= 1"));
    }

    #[test]
    fn opened_strips_bound_rejects_4x64_at_k_2_16() {
        let p = dense_params();
        let rows = AxisPattern::new(&[(4, Blake)]).unwrap();
        let cols = AxisPattern::new(&[(4, Blake), (16, Fold)]).unwrap();
        let err = PublicParams::try_new(
            JobParams {
                ancestor_header: p.job.ancestor_header,
                common: CommonParams {
                    k: PublicParams::MAX_K as u32,
                    r: 32,
                    quant: p.job.common.quant,
                    device: p.job.common.device,
                },
                operands: Sides {
                    a: OperandParams {
                        num_rows: 4,
                        hash_id: p.job.operands.a.hash_id,
                        pattern: rows,
                    },
                    b: OperandParams {
                        num_rows: 64,
                        hash_id: p.job.operands.b.hash_id,
                        pattern: cols,
                    },
                },
                moe: None,
            },
            empty_jackpot(),
            None,
        )
        .unwrap_err();
        assert!(err.to_string().contains("k·(|IA|+|IB|) must be <= 2^22"));
    }

    #[test]
    fn try_new_rejects_too_many_experts() {
        let p = dense_params();
        let err = PublicParams::try_new(
            JobParams {
                ancestor_header: p.job.ancestor_header,
                common: p.job.common,
                operands: Sides {
                    a: p.job.operands.a.clone(),
                    b: p.job.operands.b.clone(),
                },
                moe: Some(MoeParams {
                    experts: MoeParams::MAX_NUM_EXPERTS + 1,
                    hash_id_r: HashId::Blake3Chunk1024,
                    hash_id_o: HashId::Blake3Chunk1024,
                }),
            },
            p.jackpot_statement.clone(),
            Some(empty_moe_statement(vec![0, 2, 4, 6])),
        )
        .unwrap_err();
        assert!(err.to_string().contains("experts must be in 1..="));
    }

    #[test]
    fn try_new_rejects_winner_out_of_range() {
        let p = dense_params();
        let mut stmt = empty_moe_statement(vec![0, 2, 4, 6]);
        stmt.w = 4;
        let err = PublicParams::try_new(
            JobParams {
                ancestor_header: p.job.ancestor_header,
                common: p.job.common,
                operands: Sides {
                    a: p.job.operands.a.clone(),
                    b: p.job.operands.b.clone(),
                },
                moe: Some(MoeParams {
                    experts: 4,
                    hash_id_r: HashId::Blake3Chunk1024,
                    hash_id_o: HashId::Blake3Chunk1024,
                }),
            },
            p.jackpot_statement.clone(),
            Some(stmt),
        )
        .unwrap_err();
        assert!(err.to_string().contains("0 <= w < e"));
    }

    #[test]
    fn try_new_rejects_i_a_length_and_range() {
        let p = dense_params();
        let moe = Some(MoeParams {
            experts: 4,
            hash_id_r: HashId::Blake3Chunk1024,
            hash_id_o: HashId::Blake3Chunk1024,
        });
        let build = |i_a: Vec<u32>| {
            PublicParams::try_new(
                JobParams {
                    ancestor_header: p.job.ancestor_header,
                    common: p.job.common,
                    operands: Sides {
                        a: p.job.operands.a.clone(),
                        b: p.job.operands.b.clone(),
                    },
                    moe,
                },
                p.jackpot_statement.clone(),
                Some(empty_moe_statement(i_a)),
            )
        };
        assert!(
            build(vec![0, 2, 4])
                .unwrap_err()
                .to_string()
                .contains("I_A length must equal h")
        );
        assert!(
            build(vec![0, 2, 4, 256])
                .unwrap_err()
                .to_string()
                .contains("I_A must be a subset of [0, m)")
        );
    }

    #[test]
    fn try_new_rejects_o_scalars_and_u_a() {
        let p = dense_params();
        let moe = Some(MoeParams {
            experts: 4,
            hash_id_r: HashId::Blake3Chunk1024,
            hash_id_o: HashId::Blake3Chunk1024,
        });
        let build = |stmt: MoEStatement| {
            PublicParams::try_new(
                JobParams {
                    ancestor_header: p.job.ancestor_header,
                    common: p.job.common,
                    operands: Sides {
                        a: p.job.operands.a.clone(),
                        b: p.job.operands.b.clone(),
                    },
                    moe,
                },
                p.jackpot_statement.clone(),
                Some(stmt),
            )
        };
        let mut stmt = empty_moe_statement(vec![0, 2, 4, 6]);
        stmt.o_w = 100;
        stmt.o_w_prev = 128;
        assert!(build(stmt).unwrap_err().to_string().contains("O_{w-1} must be <= O_w"));

        let mut stmt = empty_moe_statement(vec![0, 2, 4, 6]);
        stmt.o_w = 600;
        stmt.o_last = 512;
        assert!(build(stmt).unwrap_err().to_string().contains("O_w must be <= O_{e-1}"));

        let mut stmt = empty_moe_statement(vec![0, 2, 4, 6]);
        stmt.o_w_prev = 0;
        stmt.o_w = 2;
        assert!(build(stmt).unwrap_err().to_string().contains("U_A subset of [0, s_w)"));
    }

    #[test]
    fn try_new_rejects_n_not_divisible_by_e() {
        let p = dense_params();
        let err = PublicParams::try_new(
            JobParams {
                ancestor_header: p.job.ancestor_header,
                common: p.job.common,
                operands: Sides {
                    a: p.job.operands.a.clone(),
                    b: p.job.operands.b.clone(),
                },
                moe: Some(MoeParams {
                    experts: 3,
                    hash_id_r: HashId::Blake3Chunk1024,
                    hash_id_o: HashId::Blake3Chunk1024,
                }),
            },
            p.jackpot_statement,
            Some(empty_moe_statement(vec![0, 2, 4, 6])),
        )
        .unwrap_err();
        assert!(err.to_string().contains("n must be divisible by e"));
    }

    #[test]
    fn try_new_rejects_o_last_above_the_2p29_cap() {
        let p = dense_params();
        let build = |o_last: u32| {
            let mut stmt = empty_moe_statement(vec![0, 2, 4, 6]);
            stmt.o_last = o_last;
            PublicParams::try_new(
                JobParams {
                    ancestor_header: p.job.ancestor_header,
                    common: p.job.common,
                    operands: Sides {
                        a: p.job.operands.a.clone(),
                        b: p.job.operands.b.clone(),
                    },
                    moe: Some(MoeParams {
                        experts: 3,
                        hash_id_r: HashId::Blake3Chunk1024,
                        hash_id_o: HashId::Blake3Chunk1024,
                    }),
                },
                p.jackpot_statement.clone(),
                Some(stmt),
            )
        };
        // At and above the cap: rejected by MoEStatement::check.
        let err = build((1 << 29) + 1).unwrap_err();
        assert!(err.to_string().contains("O_{e-1} must be <= 2^29"), "{err}");
        let err = build(u32::MAX).unwrap_err();
        assert!(err.to_string().contains("O_{e-1} must be <= 2^29"), "{err}");
    }

    // ---- check_ancestry: σ_d authenticated by the SHA256d walk from σ̂ ----

    fn job_with_ancestor(ancestor_header: BlockHeader) -> JobParams {
        JobParams {
            ancestor_header,
            ..dense_params().job.clone()
        }
    }

    fn ancestry_err(proposed: &IncompleteBlockHeader, chain: &[BlockHeader], ancestor: BlockHeader) -> String {
        job_with_ancestor(ancestor)
            .check_ancestry(proposed, chain)
            .unwrap_err()
            .to_string()
    }

    #[test]
    fn check_ancestry_accepts_every_window_depth() {
        for depth in 1..=STATE_WINDOW_DEPTH {
            let (proposed, chain, ancestor) = ancestry(depth);
            assert_eq!(chain.len(), depth - 1);
            job_with_ancestor(ancestor).check_ancestry(&proposed, &chain).unwrap();
        }
    }

    #[test]
    fn check_ancestry_rejects_depth_beyond_the_window() {
        let (proposed, chain, ancestor) = ancestry(STATE_WINDOW_DEPTH + 1);
        let err = ancestry_err(&proposed, &chain, ancestor);
        assert!(err.contains("at most 3 fit the state window"), "{err}");
    }

    #[test]
    fn check_ancestry_rejects_a_broken_link() {
        let (proposed, chain, ancestor) = ancestry(3);
        for (i, depth) in [(0, 1), (1, 2)] {
            let mut broken = chain.clone();
            broken[i].incomplete.merkle_root[0] ^= 1;
            let err = ancestry_err(&proposed, &broken, ancestor);
            assert!(err.contains(&format!("depth {depth} does not connect")), "{err}");
        }
    }

    #[test]
    fn check_ancestry_rejects_extra_headers() {
        let (proposed, mut chain, ancestor) = ancestry(2);
        // σ_d repeated at the end of the chain: the walk overshoots it.
        chain.push(ancestor);
        let err = ancestry_err(&proposed, &chain, ancestor);
        assert!(err.contains("depth 3 does not connect"), "{err}");
    }

    #[test]
    fn check_ancestry_rejects_a_wrong_sigma_d() {
        let (proposed, chain, ancestor) = ancestry(2);
        // The proof commitment is part of the authenticated header.
        let recommitted = BlockHeader {
            proof_commitment: [0xee; 32],
            ..ancestor
        };
        let err = ancestry_err(&proposed, &chain, recommitted);
        assert!(err.contains("depth 2 does not connect"), "{err}");
        // σ̂ itself is never in the window.
        let err = ancestry_err(
            &proposed,
            &[],
            BlockHeader {
                incomplete: proposed,
                proof_commitment: [0; 32],
            },
        );
        assert!(err.contains("depth 1 does not connect"), "{err}");
    }

    #[test]
    fn blake3_bound_dominates_the_compiled_stark_for_all_chunk_sizes() {
        use crate::v4::circuit::blake3_stark::stark::{Blake3Program, MoeSchedule};
        use crate::v4::circuit::chip::blake3::program::BlakeProgram;

        for is_moe in [false, true] {
            for hash_id in HashId::ALL {
                for k in [1024, 2080, 16384] {
                    let mut seed = if is_moe { moe_params() } else { dense_params() };
                    seed.job.common.k = k;
                    seed.job.operands.a.hash_id = hash_id;
                    seed.job.operands.b.hash_id = hash_id;
                    if let Some(moe) = seed.job.moe.as_mut() {
                        moe.hash_id_r = hash_id;
                        moe.hash_id_o = hash_id;
                    }
                    let params = PublicParams::try_new(seed.job, seed.jackpot_statement, seed.moe_statement).unwrap();
                    let (program, _, _) = BlakeProgram::compile(&params);
                    let (pins, schedule) = match (params.moe(), params.moe_statement()) {
                        (Some(moe), Some(stmt)) => (
                            stmt.routing_pins(&params.a_inner_indices()),
                            Some(MoeSchedule::new(stmt, moe.experts, params.m()).unwrap()),
                        ),
                        (None, None) => (vec![], None),
                        _ => unreachable!(),
                    };
                    let stark = Blake3Program::from_blake_program(&program, k as usize, pins, schedule);
                    let bound = params.blake3_work_block_bound().unwrap();
                    let compiled = stark.instructions.len() as u64;
                    assert!(
                        bound >= compiled,
                        "the bound must dominate the compiler: MoE={is_moe} hash_id={hash_id:?} k={k}: {bound} < {compiled}"
                    );
                    assert!(
                        bound <= compiled + 4096,
                        "the bound must stay within root-path slack: MoE={is_moe} hash_id={hash_id:?} k={k}: {bound} >> {compiled}"
                    );
                    assert!(compiled * blake3::BLOCK_LEN as u64 <= PublicParams::MAX_BLAKE3_WORK_BYTES as u64);
                }
            }
        }
    }

    #[test]
    fn aggregate_blake3_cap_rejects_before_compilation() {
        let mut seed = dense_params();
        seed.job.common.k = 31744;
        assert!((seed.h() as usize + seed.w() as usize) * seed.common_dim() as usize <= (1 << 22));
        let err = PublicParams::try_new(seed.job, seed.jackpot_statement, None).unwrap_err();
        assert!(err.to_string().contains("Blake3 work exceeds 2^22 bytes"), "{err}");
    }

    #[test]
    fn maximal_moe_scalars_bound_and_compile_without_overflow() {
        use crate::v4::circuit::blake3_stark::stark::{Blake3Program, MoeSchedule};
        use crate::v4::circuit::chip::blake3::program::BlakeProgram;

        // The maximal routing scalars: `o_last` at the 2^29 cap (a ~2^31-byte
        // virtual routing tree), winner slice at the very front. The routing
        // bound is O(tree depth) thanks to aux-CV pruning, so `try_new` must
        // pass the work gate and the compiler must produce a program the bound
        // dominates.
        let p = dense_params();
        let build = |stmt: MoEStatement| {
            PublicParams::try_new(
                JobParams {
                    ancestor_header: p.job.ancestor_header,
                    common: p.job.common,
                    operands: Sides {
                        a: p.job.operands.a.clone(),
                        b: stacked_b(&p, 4),
                    },
                    moe: Some(MoeParams {
                        experts: 4,
                        hash_id_r: HashId::Blake3Chunk1024,
                        hash_id_o: HashId::Blake3Chunk1024,
                    }),
                },
                p.jackpot_statement.clone(),
                Some(stmt),
            )
            .unwrap()
        };

        // Baseline: identical params except the routing tree is tiny
        // (`o_last = 512` -> a 2048-byte tree).
        let small = build(empty_moe_statement(vec![0, 1, 2, 3]));
        let mut stmt = empty_moe_statement(vec![0, 1, 2, 3]);
        stmt.w = 0;
        stmt.o_w_prev = 0;
        stmt.o_w = 16;
        stmt.o_last = 1 << 29;
        let params = build(stmt);

        // The maximal tree adds only root-path parents: ~2 * depth-of-2^24-chunks
        // extra blocks over the baseline, nothing proportional to 2^34 bytes.
        let (bound, small_bound) = (
            params.blake3_work_block_bound().unwrap(),
            small.blake3_work_block_bound().unwrap(),
        );
        assert!(
            bound <= small_bound + 128,
            "maximal routing scalars must bound at O(depth): {bound} vs baseline {small_bound}"
        );
        assert!(
            bound <= (PublicParams::MAX_BLAKE3_WORK_BYTES / blake3::BLOCK_LEN) as u64,
            "maximal routing scalars must pass the work gate"
        );

        let (program, _, _) = BlakeProgram::compile(&params);
        let (pins, schedule) = (
            params.moe_statement().unwrap().routing_pins(&params.a_inner_indices()),
            Some(MoeSchedule::new(params.moe_statement().unwrap(), params.moe().unwrap().experts, params.m()).unwrap()),
        );
        let stark = Blake3Program::from_blake_program(&program, 2048, pins, schedule);
        let compiled = stark.instructions.len() as u64;
        assert!(
            bound >= compiled,
            "the bound must dominate the compiled stark under maximal scalars: {bound} < {compiled}"
        );
        assert!(compiled * blake3::BLOCK_LEN as u64 <= PublicParams::MAX_BLAKE3_WORK_BYTES as u64);
    }
}
