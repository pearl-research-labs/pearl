//! Blake3 program types: instruction-level abstractions for Blake3 hashing.

use std::mem;

use anyhow::{Result, bail, ensure};
pub use blake3::BLOCK_LEN;
use pearl_blake3::{B3F_CHUNK_END, B3F_CHUNK_START, B3F_KEYED_HASH, B3F_PARENT, B3F_ROOT, BLAKE3_MSG_LEN};

use crate::{
    api::{
        fp8::{
            openings::PrivateProofParams,
            prequant::{BLOCK_SIZE, PrequantOperand},
            public_params::{HashId, PublicParams},
        },
        primitives::{Hash256, Sides},
    },
    circuit::chip::blake3::{
        blake3_compress::blake3_compress,
        logic::{AuxDataType, BlakeRoundLogic, MessageDataType},
    },
    ensure_eq,
};

pub use super::blake3_compress::Blake3Tweak;

pub const DWORD_SIZE: usize = 8; // bytes per dword

/// Reads one dword of an opened committed operand. Value and scale strips share
/// one index space: `0..row_count` are values, then scales of the same rows.
fn read_dword(operand: &PrequantOperand, dword_id: MatDwordId) -> [u8; DWORD_SIZE] {
    let n_values = operand.values.row_count();
    let row = if dword_id.strip_idx < n_values {
        operand.values.row(dword_id.strip_idx)
    } else {
        operand.scales.row(dword_id.strip_idx - n_values)
    };
    row[dword_id.idx_in_strip..dword_id.idx_in_strip + DWORD_SIZE]
        .try_into()
        .unwrap()
}

/// Assembles a 64-byte Blake3 message from the opened committed operand.
fn read_blake_msg(operand: &PrequantOperand, blake_msg: BlakeMsg) -> [u8; 64] {
    let mut result = [0u8; 64];
    for (dword_idx, &dword_id) in blake_msg.dwords.iter().enumerate() {
        let dword = read_dword(operand, dword_id);
        let dst_start = dword_idx * DWORD_SIZE;
        result[dst_start..dst_start + DWORD_SIZE].copy_from_slice(&dword);
    }
    result
}

/// Number of STARK rows emitted per [`BlakeInstruction`] (one blake3 compression:
/// 7 mixing rounds + 1 output/finalization row).
pub const ROUNDS_PER_BLAKE_INSTRUCTION: usize = 8;

#[derive(Clone, Debug, Copy, Hash, Eq, PartialEq, Default)]
pub struct MatDwordId {
    pub is_b_strip: bool, // If true, then B matrix. If false, then A matrix.
    /// Index into the side's opened strips. Values occupy `0..h` (A) / `0..w` (B);
    /// when scales are present they follow at `h..2h` / `w..2w`. The program is
    /// byte-oriented, so the two trees' strips occupy disjoint index ranges of
    /// the same list.
    pub strip_idx: usize,
    pub idx_in_strip: usize, // divisible by 8, idx of first byte element.
}

#[derive(Clone, Debug, Copy, Hash, Eq, PartialEq)]
pub struct BlakeMsg {
    pub dwords: [MatDwordId; 8], // Ordered same way message enters blake. All dwords share same is_b_strip.
}

impl BlakeMsg {
    /// Construct a BlakeMsg with 8 consecutive dwords from the same strip (64 contiguous bytes).
    pub fn contiguous(is_b_strip: bool, strip_idx: usize, start_idx: usize) -> Self {
        BlakeMsg {
            dwords: std::array::from_fn(|i| MatDwordId {
                is_b_strip,
                strip_idx,
                idx_in_strip: start_idx + i * DWORD_SIZE,
            }),
        }
    }
}

#[derive(Clone, Debug, Copy)]
pub enum CvType {
    Instruction { idx: usize }, // cv is output of instruction at index=idx
    Auxiliary { idx: usize },   // cv is given as auxiliary CV at index=idx
}

#[derive(Clone, Debug, Copy)]
pub enum MessageType {
    /// 64 bytes of opened strip data. The 8 dwords are independent
    /// [`MatDwordId`]s: normally 64 contiguous bytes of one strip, but when a
    /// committed row is not a multiple of 64 bytes (prequant rows at
    /// `k % 64 != 0` / scales rows at `k % 256 != 0`) a block can straddle a
    /// row boundary and its dwords come from two adjacent opened strips.
    MatrixLeaf {
        mat_data: BlakeMsg,
    },
    RoutingLeaf {
        hotspot_idx: usize,
    }, // 64 contiguous bytes from a routing hotspot block (each block is exactly one blake3 block).
    OffsetsLeaf {
        block_idx: usize,
    }, // 64 contiguous bytes of the offsets list (fully opened: every block is scheduled).
    AuxiliaryLeaf {
        idx: usize,
    }, // index among auxiliary messages.
    /// A block straddling an opening boundary: dword `i` comes from an opened
    /// strip where `mat_dwords[i]` is `Some`, and from auxiliary message
    /// `aux_idx` (the full 64-byte block, unopened-neighbor bytes included)
    /// where it is `None`. Only the `None` dwords of the auxiliary message are
    /// ever read.
    SplitLeaf {
        mat_dwords: [Option<MatDwordId>; 8],
        aux_idx: usize,
    },
    Parent {
        cv_low: CvType,
        cv_high: CvType,
    }, // bytes 0..32 and 32..64 of message
}

/// The CV source of a keyed compression: `Prev` chains the previous
/// instruction's output; `KeyA`/`KeyB` ingest the per-side opening key
/// (`keyA`/`keyB`); `Jackpot` ingests the lottery `jackpot_key`. Matches the
/// circuit's CV-source mux (one compiled slot per variant).
#[derive(Clone, Debug, Copy, PartialEq, Eq)]
pub enum KeySource {
    Prev,
    KeyA,
    KeyB,
    Jackpot,
}

/// Which role an instruction's output hash plays, if any: a reconstructed
/// tree root (`A`/`B`, their prequant `Scales` trees, the MoE `Routing`
/// root), the keyed `Offsets` root,
/// or nothing (`None` for interior compressions and the jackpot).
#[derive(Clone, Debug, Copy, PartialEq, Eq)]
pub enum HashOut {
    None,
    A,
    B,
    AScales,
    BScales,
    Routing,
    Offsets,
    Jackpot,
}

#[derive(Clone, Debug, Copy)]
pub struct BlakeInstruction {
    pub key_source: KeySource,
    pub tweak: Blake3Tweak, // last 16 bytes of the initial state
    pub msg: MessageType,
    pub out: HashOut,
}

impl BlakeInstruction {
    /// Generate one [`BlakeRoundLogic`] entry per round (see [`ROUNDS_PER_BLAKE_INSTRUCTION`]).
    pub(crate) fn emit_instruction_rounds(
        &self,
        inst_to_row: &[usize],
        read_cv_from: Option<usize>,
    ) -> [BlakeRoundLogic; ROUNDS_PER_BLAKE_INSTRUCTION] {
        std::array::from_fn(|i| {
            let data_source = match self.msg {
                MessageType::MatrixLeaf { mat_data } => MessageDataType::Matrix {
                    dword_id: mat_data.dwords[i],
                },
                MessageType::RoutingLeaf { hotspot_idx } => MessageDataType::RoutingData {
                    hotspot_idx,
                    idx_in_block: i * DWORD_SIZE,
                },
                MessageType::OffsetsLeaf { .. } => {
                    unreachable!("offsets sections exist only in fp8 programs, which do not use round logic")
                }
                MessageType::AuxiliaryLeaf { idx } => MessageDataType::AuxiliaryData {
                    aux_type: AuxDataType::Msg { aux_msg_idx: idx },
                    dword_idx: i,
                },
                MessageType::SplitLeaf { mat_dwords, aux_idx } => match mat_dwords[i] {
                    Some(dword_id) => MessageDataType::Matrix { dword_id },
                    None => MessageDataType::AuxiliaryData {
                        aux_type: AuxDataType::Msg { aux_msg_idx: aux_idx },
                        dword_idx: i,
                    },
                },
                MessageType::Parent { cv_low, cv_high } => {
                    let (cv, local_i) = if i < 4 { (cv_low, i) } else { (cv_high, i - 4) };
                    match cv {
                        CvType::Instruction { idx } if local_i == 3 => MessageDataType::PreviousCv {
                            source_row_idx: inst_to_row[idx],
                        },
                        CvType::Auxiliary { idx } => MessageDataType::AuxiliaryData {
                            aux_type: AuxDataType::Cv { aux_cv_idx: idx },
                            dword_idx: local_i,
                        },
                        _ => MessageDataType::None,
                    }
                }
            };
            let idx_of_row_whence_to_read_cv = match data_source {
                MessageDataType::PreviousCv { source_row_idx } => Some(source_row_idx),
                _ if i == 0 => read_cv_from,
                _ => None,
            };
            debug_assert!(
                i + 1 != ROUNDS_PER_BLAKE_INSTRUCTION || !matches!(data_source, MessageDataType::None),
                "Last round (IS_LAST_ROUND) must have a data source to constrain BLAKE3_MSG_BUFFER"
            );
            BlakeRoundLogic {
                data_source,
                blake3_tweak: if i == 0 { Some(self.tweak) } else { None },
                round_idx: i + 1,
                idx_of_row_whence_to_read_cv,
                is_hash_a: i == 7 && self.out == HashOut::A,
                is_hash_b: i == 7 && self.out == HashOut::B,
                is_hash_routing: i == 7 && self.out == HashOut::Routing,
                is_hash_jackpot: i == 7 && self.out == HashOut::Jackpot,
                cv_is_commitment: false,
            }
        })
    }
}

#[derive(Clone, Debug)]
pub struct BlakeProgram {
    pub num_a_rows: usize,   // Number of A rows being proved (h)
    pub num_b_cols: usize,   // Number of B columns being proved (w)
    pub strip_length: usize, // length of the strips relevant to the proof (k-k%r)
    /// MoE: Number of distinct opened strips for outer indices.
    pub num_routing_strips: usize,
    /// MoE: Number of 64-byte offsets blocks.
    pub num_offsets_strips: usize,
    pub num_auxiliary_msgs: usize, // each msg being 64 bytes
    pub num_auxiliary_cvs: usize,  // each cv being 32 bytes
    pub instructions: Vec<BlakeInstruction>,
}

/// Which Merkle commitment an auxiliary message/CV belongs to. `A`/`B` are the
/// (values) trees of each side; `AScales`/`BScales` are the prequant scales
/// trees; `Routing`/`Offsets` are the MoE routing and offsets commitments.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProofSource {
    A,
    B,
    AScales,
    BScales,
    Routing,
    Offsets,
}

/// The Merkle roots reconstructed by [`BlakeProgram::evaluate_blake`]. The
/// scales roots are `Some` exactly for prequantized (FP8) programs.
#[derive(Debug, Clone, Copy)]
pub struct BlakeRoots {
    pub hash_a: Hash256,
    pub hash_b: Hash256,
    pub hash_a_scales: Option<Hash256>,
    pub hash_b_scales: Option<Hash256>,
}

#[derive(Debug, Clone, Copy)]
pub struct AuxiliaryMsgLocation {
    pub global_start: usize, // index in the global matrix pointing to first byte in message
    pub source: ProofSource,
}

#[derive(Debug, Clone, Copy)]
pub struct AuxiliaryCvLocation {
    pub global_start: usize, // start index in the global matrix
    pub global_end: usize,   // end index in the global matrix (exclusive)
    pub source: ProofSource,
}

/// 64-byte row indices into the unpadded `m × top_k × 4` routing byte layout for each inner A-row index.
pub fn routing_blake_hotspot_rows(routing_start_offset: u32, inner_indices: &[u32]) -> Vec<u32> {
    let mut hotspots: Vec<u32> = inner_indices
        .iter()
        .map(|&i| ((routing_start_offset as u64 + i as u64) * 4 / BLOCK_LEN as u64) as u32)
        .collect();

    hotspots.sort_unstable();
    hotspots.dedup();
    hotspots
}

impl BlakeProgram {
    /// Compile the ONE Blake membership program of a proof, derived entirely
    /// from PUBLIC data (shapes, widths, opened indices — never strip
    /// contents), so the ZK verifier can compile the identical program to
    /// generate its preprocessed columns.
    ///
    /// The compilation is byte-oriented and content-agnostic; the scheme only
    /// selects which committed trees exist and their row widths:
    /// - Int7: the A and B trees at the mining-config widths.
    /// - Prequant (`int8 blk8 bf16s` FP8): four trees — int8 values (`k`
    ///   bytes/row, strip indices `0..h` / `0..w`) then BF16 scales
    ///   (`2 * k/BLOCK_SIZE` bytes/row, strip indices `h..2h` / `w..2w`), so
    ///   the strips fed to Blake3 are the committed bytes, not opened BF16.
    /// - MoE (either scheme): plus the routing commitment.
    pub fn compile(params: &PublicParams) -> (Self, Vec<AuxiliaryMsgLocation>, Vec<AuxiliaryCvLocation>) {
        let mut instructions = Vec::new();
        let mut msgs: Vec<AuxiliaryMsgLocation> = vec![];
        let mut cvs: Vec<AuxiliaryCvLocation> = vec![];
        let m = params.m() as usize;
        let total_b_cols = params.n() as usize;

        // Committed row stride and exposed strip prefix of the values trees:
        // prequant commits 1 byte per int8 element and exposes full rows.
        let (row_bytes, strip_bytes) = (params.common_dim() as usize, params.common_dim() as usize);

        /// Compile one committed tree over its whole padded byte range,
        /// returning the root instruction's index.
        #[allow(clippy::too_many_arguments)]
        fn section(
            row_bytes: usize,
            strip_bytes: usize,
            num_rows: usize,
            hotspots: &[u32],
            source: ProofSource,
            strip_idx_base: usize,
            hash_id: HashId,
            instructions: &mut Vec<BlakeInstruction>,
            msgs: &mut Vec<AuxiliaryMsgLocation>,
            cvs: &mut Vec<AuxiliaryCvLocation>,
        ) -> usize {
            let cv = recursive_compilation(
                0,
                hash_id.padded_len(num_rows * row_bytes),
                num_rows,
                row_bytes,
                hotspots,
                strip_bytes,
                source,
                strip_idx_base,
                hash_id,
                instructions,
                msgs,
                cvs,
            );
            let CvType::Instruction { idx } = cv else { unreachable!() };
            idx
        }

        let idx = section(
            row_bytes,
            strip_bytes,
            m,
            &params.a_rows_indices(),
            ProofSource::A,
            0,
            params.a().hash_id,
            &mut instructions,
            &mut msgs,
            &mut cvs,
        );
        instructions[idx].out = HashOut::A;

        let idx = section(
            row_bytes,
            strip_bytes,
            total_b_cols,
            &params.b_rows_indices(),
            ProofSource::B,
            0,
            params.b().hash_id,
            &mut instructions,
            &mut msgs,
            &mut cvs,
        );
        instructions[idx].out = HashOut::B;

        // The scales trees: one BF16 scale (2 bytes) per BLOCK_SIZE int8
        // values, full rows exposed. Their strips follow the values strips
        // in each side's strip list (indices h.. / w..).
        let scale_row_bytes = 2 * (params.common_dim() as usize / BLOCK_SIZE);
        let idx = section(
            scale_row_bytes,
            scale_row_bytes,
            m,
            &params.a_rows_indices(),
            ProofSource::AScales,
            params.h() as usize,
            params.a().hash_id,
            &mut instructions,
            &mut msgs,
            &mut cvs,
        );
        instructions[idx].out = HashOut::AScales;

        let idx = section(
            scale_row_bytes,
            scale_row_bytes,
            total_b_cols,
            &params.b_rows_indices(),
            ProofSource::BScales,
            params.w() as usize,
            params.b().hash_id,
            &mut instructions,
            &mut msgs,
            &mut cvs,
        );
        instructions[idx].out = HashOut::BScales;

        let mut num_routing_strips = 0usize;
        let mut num_offsets_strips = 0usize;
        if let Some(moe) = params.moe_statement() {
            // Routing is jagged: `Rflat` holds exactly `O_{e-1}` u32 entries.
            // We pad the entry count up to a multiple of 16 so the byte length tiles
            // evenly into the 64-byte virtual rows the commitment uses below.
            let total_routing_entries = params.num_padded_routing_entries().expect("MoE params present");
            let full_routing_size = total_routing_entries * mem::size_of::<u32>();
            // Routing and offsets are flat byte strings with no natural row structure, so
            // both sections below use virtual rows of one Blake3 block (64 bytes) each.
            let num_rows = full_routing_size / BLOCK_LEN;
            let hotspots = moe.opened_routing_blocks();
            num_routing_strips = hotspots.len();
            let routing_hash_id = params.moe().expect("MoE").hash_id_r;

            let idx = section(
                BLOCK_LEN,
                BLOCK_LEN,
                num_rows,
                &hotspots,
                ProofSource::Routing,
                0,
                routing_hash_id,
                &mut instructions,
                &mut msgs,
                &mut cvs,
            );
            instructions[idx].out = HashOut::Routing;

            // Offsets: the cumulative-count list `O` (zero-padded to the `hash_idO` chunk
            // length) as 64-byte virtual rows, every one opened — the circuit checks the
            // whole list, so the whole tree is word-level data.
            let offsets_hash_id = params.moe().expect("MoE").hash_id_o;
            let offsets_bytes = offsets_hash_id.padded_len(params.num_offsets_bytes().expect("MoE"));
            let offsets_rows = offsets_bytes / BLOCK_LEN;
            let all_rows: Vec<u32> = (0..offsets_rows as u32).collect();
            num_offsets_strips = offsets_rows;

            let idx = section(
                BLOCK_LEN,
                BLOCK_LEN,
                offsets_rows,
                &all_rows,
                ProofSource::Offsets,
                0,
                offsets_hash_id,
                &mut instructions,
                &mut msgs,
                &mut cvs,
            );
            instructions[idx].out = HashOut::Offsets;
        }

        // A prequant program must expose exactly one scales root per side.
        debug_assert_eq!(
            instructions.iter().filter(|i| i.out == HashOut::AScales).count(),
            1,
            "prequant <=> exactly one A scales root"
        );
        debug_assert_eq!(
            instructions.iter().filter(|i| i.out == HashOut::BScales).count(),
            1,
            "prequant <=> exactly one B scales root"
        );

        (
            BlakeProgram {
                num_a_rows: params.h() as usize,
                num_b_cols: params.w() as usize,
                strip_length: strip_bytes,
                num_routing_strips,
                num_offsets_strips,
                num_auxiliary_msgs: msgs.len(),
                num_auxiliary_cvs: cvs.len(),
                instructions,
            },
            msgs,
            cvs,
        )
    }

    /// Replay the compiled opening forest under `keys`. For MoE,
    /// `opt_moe_roots` is `(HR, HO)`: the routing and offsets tree roots the
    /// replayed sections must reproduce.
    pub fn evaluate_blake(
        &self,
        keys: Sides<Hash256>,
        private_params: &PrivateProofParams,
        opt_moe_roots: Option<(Hash256, Hash256)>,
    ) -> Result<BlakeRoots> {
        ensure_eq!(private_params.operands.a.num_rows()?, self.num_a_rows);
        ensure_eq!(private_params.operands.b.num_rows()?, self.num_b_cols);
        let mut cvs = vec![];

        for instruction in &self.instructions {
            let cv_in = match instruction.key_source {
                KeySource::Prev => *cvs.last().unwrap(),
                KeySource::KeyA => keys.a,
                KeySource::KeyB => keys.b,
                KeySource::Jackpot => {
                    bail!("opening evaluation does not run jackpot-keyed instructions")
                }
            };
            let msg = match instruction.msg {
                MessageType::MatrixLeaf { mat_data } => {
                    let operand = private_params.operands.side(mat_data.dwords[0].is_b_strip);
                    read_blake_msg(operand, mat_data)
                }
                MessageType::RoutingLeaf { hotspot_idx } => {
                    let strip = &private_params.s_routing[hotspot_idx];
                    std::array::from_fn(|i| strip[i])
                }
                MessageType::OffsetsLeaf { block_idx } => {
                    let strip = &private_params.s_offsets[block_idx];
                    std::array::from_fn(|i| strip[i])
                }
                MessageType::AuxiliaryLeaf { idx } => private_params.external_msgs[idx],
                MessageType::SplitLeaf { mat_dwords, aux_idx } => {
                    let mut msg = private_params.external_msgs[aux_idx];
                    for (d, dword_id) in mat_dwords.iter().enumerate() {
                        if let Some(dword_id) = dword_id {
                            let operand = private_params.operands.side(dword_id.is_b_strip);
                            let dword = read_dword(operand, *dword_id);
                            msg[d * DWORD_SIZE..(d + 1) * DWORD_SIZE].copy_from_slice(&dword);
                        }
                    }
                    msg
                }
                MessageType::Parent { cv_low, cv_high } => {
                    let get_cv = |cv: CvType| match cv {
                        CvType::Auxiliary { idx } => private_params.external_cvs[idx],
                        CvType::Instruction { idx } => cvs[idx],
                    };
                    [get_cv(cv_low), get_cv(cv_high)].concat().try_into().unwrap()
                }
            };
            cvs.push(blake3_compress(&msg, cv_in, instruction.tweak));
        }

        let (mut hash_a, mut hash_b, mut hash_a_scales, mut hash_b_scales) = (None, None, None, None);
        let (mut hash_routing, mut hash_offsets) = (None, None);
        for (idx, inst) in self.instructions.iter().enumerate() {
            match inst.out {
                HashOut::A => hash_a = Some(cvs[idx]),
                HashOut::B => hash_b = Some(cvs[idx]),
                HashOut::AScales => hash_a_scales = Some(cvs[idx]),
                HashOut::BScales => hash_b_scales = Some(cvs[idx]),
                HashOut::Routing => hash_routing = Some(cvs[idx]),
                HashOut::Offsets => hash_offsets = Some(cvs[idx]),
                HashOut::None | HashOut::Jackpot => {}
            }
        }
        ensure!(hash_a.is_some() && hash_b.is_some());
        if let Some((expected_hash_routing, expected_hash_offsets)) = opt_moe_roots {
            let got = hash_routing
                .ok_or_else(|| anyhow::anyhow!("Blake program has no is_hash_routing output but hash_routing was expected"))?;
            ensure_eq!(
                got,
                expected_hash_routing,
                "hash_routing mismatch between Blake evaluation and public commitment"
            );
            let got = hash_offsets
                .ok_or_else(|| anyhow::anyhow!("Blake program has no offsets output but hash_offsets was expected"))?;
            ensure_eq!(
                got,
                expected_hash_offsets,
                "hash_offsets mismatch between Blake evaluation and the claimed HO"
            );
        }
        Ok(BlakeRoots {
            hash_a: hash_a.unwrap(),
            hash_b: hash_b.unwrap(),
            hash_a_scales,
            hash_b_scales,
        })
    }
}

#[allow(clippy::too_many_arguments)]
fn recursive_compilation(
    start: usize,
    end: usize,
    num_rows: usize,
    row_len: usize,         // row length in bytes
    hotspot_strips: &[u32], // sorted global row indices into the committed matrix
    strip_len: usize,       // prefix length of each row exposed to the circuit
    source: ProofSource,    // which committed tree this section covers
    strip_idx_base: usize,  // offset of this tree's strips within its side's strip list
    hash_id: HashId,
    instructions: &mut Vec<BlakeInstruction>,
    out_msgs: &mut Vec<AuxiliaryMsgLocation>,
    out_cvs: &mut Vec<AuxiliaryCvLocation>,
) -> CvType {
    let chunk_len = hash_id.chunk_len();
    debug_assert!(strip_len <= row_len);
    // Rows need only tile into whole dwords: a 64-byte block may straddle a row boundary
    // (handled per dword below), but a dword may not.
    debug_assert!(row_len.is_multiple_of(DWORD_SIZE), "row_len must be divisible by 8");
    debug_assert!(strip_len.is_multiple_of(DWORD_SIZE), "strip_len must be divisible by 8");
    debug_assert!(
        hotspot_strips.windows(2).all(|w| w[0] < w[1]),
        "hotspot_strips must be sorted with no duplicates"
    );

    let is_routing = matches!(source, ProofSource::Routing);
    let is_offsets = matches!(source, ProofSource::Offsets);
    let is_b_matrix = matches!(source, ProofSource::B | ProofSource::BScales);
    let key_source = if is_b_matrix { KeySource::KeyB } else { KeySource::KeyA };

    let is_root = start == 0 && end == hash_id.padded_len(num_rows * row_len);
    let root_flag = is_root as u8 * B3F_ROOT;

    // Binary search: first strip whose byte range ends after `start`
    // i.e. first s where s * row_len + strip_len > start
    let lo = hotspot_strips.partition_point(|&s| s as usize * row_len + strip_len <= start);
    let intersects_strips = hotspot_strips[lo..].first().is_some_and(|&s| (s as usize * row_len) < end);

    if !intersects_strips && end - start >= chunk_len {
        out_cvs.push(AuxiliaryCvLocation {
            global_start: start,
            global_end: end,
            source,
        });
        CvType::Auxiliary { idx: out_cvs.len() - 1 }
    } else if end - start <= chunk_len {
        // end - start < chunk_len or end - start == chunk_len and intersects_strips
        debug_assert!(start < end);
        let chunk_idx = start / chunk_len;
        let mut is_first_in_chunk = true;
        let mut msg_start = start;
        while msg_start < end {
            let block_len = (end - msg_start).min(BLAKE3_MSG_LEN);
            let is_last_in_chunk = msg_start + block_len == end;
            let flags = B3F_KEYED_HASH
                | (is_first_in_chunk as u8 * B3F_CHUNK_START)
                | (is_last_in_chunk as u8 * (B3F_CHUNK_END | root_flag));
            let tweak = Blake3Tweak {
                counter_low: chunk_idx as u32,
                counter_high: (chunk_idx >> 32) as u16,
                block_len: block_len as u32,
                flags: flags.into(),
            };

            // Resolve the block per dword: rows are dword-aligned, so each 8-byte dword lies
            // in exactly one committed row, but the 64-byte block itself may straddle one row
            // boundary when `row_len % 64 != 0` (prequant int8 rows at `k % 64 != 0`, scales
            // rows at `k % 256 != 0`). A dword is strip-sourced iff it lies wholly in the
            // exposed prefix of an opened (hotspot) row.
            debug_assert_eq!(block_len % DWORD_SIZE, 0);
            let mat_dwords: [Option<MatDwordId>; 8] = std::array::from_fn(|d| {
                let g = msg_start + d * DWORD_SIZE;
                if g + DWORD_SIZE > msg_start + block_len {
                    return None;
                }
                let offset_in_row = g % row_len;
                (offset_in_row + DWORD_SIZE <= strip_len)
                    .then(|| hotspot_strips.binary_search(&((g / row_len) as u32)).ok())
                    .flatten()
                    .map(|hotspot_idx| MatDwordId {
                        is_b_strip: is_b_matrix,
                        strip_idx: strip_idx_base + hotspot_idx,
                        idx_in_strip: offset_in_row,
                    })
            });
            let num_strip_dwords = mat_dwords.iter().filter(|d| d.is_some()).count();
            let msg = if num_strip_dwords > 0 {
                // Slices are chunk-padded, so a block that intersects strips is always full.
                assert_eq!(block_len, BLAKE3_MSG_LEN, "strip-intersecting blocks are whole");
                if is_routing || is_offsets {
                    // Each routing/offsets strip is exactly one whole blake3 block: never straddling.
                    let d0 = mat_dwords[0].expect("routing/offsets blocks are wholly opened");
                    assert_eq!(d0.idx_in_strip, 0);
                    assert_eq!(num_strip_dwords, BLAKE3_MSG_LEN / DWORD_SIZE);
                    let idx = d0.strip_idx - strip_idx_base;
                    if is_routing {
                        MessageType::RoutingLeaf { hotspot_idx: idx }
                    } else {
                        MessageType::OffsetsLeaf { block_idx: idx }
                    }
                } else if num_strip_dwords == BLAKE3_MSG_LEN / DWORD_SIZE {
                    MessageType::MatrixLeaf {
                        mat_data: BlakeMsg {
                            dwords: mat_dwords.map(|d| d.expect("all dwords strip-sourced")),
                        },
                    }
                } else {
                    // Mixed block: expose the opened dwords, and carry the unopened remainder
                    // as an auxiliary message (the full block — the extractor serves it from
                    // the same chunk bytes the proof already transports).
                    out_msgs.push(AuxiliaryMsgLocation {
                        global_start: msg_start,
                        source,
                    });
                    MessageType::SplitLeaf {
                        mat_dwords,
                        aux_idx: out_msgs.len() - 1,
                    }
                }
            } else {
                out_msgs.push(AuxiliaryMsgLocation {
                    global_start: msg_start,
                    source,
                });
                MessageType::AuxiliaryLeaf { idx: out_msgs.len() - 1 }
            };
            instructions.push(BlakeInstruction {
                key_source: if is_first_in_chunk { key_source } else { KeySource::Prev },
                tweak,
                msg,
                out: HashOut::None,
            });
            is_first_in_chunk = false;
            msg_start += block_len;
        }
        CvType::Instruction {
            idx: instructions.len() - 1,
        }
    } else {
        // intersects_strips && end - start > chunk_len: recurse
        let ss = (end - start).next_power_of_two() / 2;
        let mid = start + ss;
        debug_assert!(mid < end);

        let left_cv = recursive_compilation(
            start,
            mid,
            num_rows,
            row_len,
            hotspot_strips,
            strip_len,
            source,
            strip_idx_base,
            hash_id,
            instructions,
            out_msgs,
            out_cvs,
        );
        let right_cv = recursive_compilation(
            mid,
            end,
            num_rows,
            row_len,
            hotspot_strips,
            strip_len,
            source,
            strip_idx_base,
            hash_id,
            instructions,
            out_msgs,
            out_cvs,
        );
        instructions.push(BlakeInstruction {
            key_source,
            tweak: Blake3Tweak {
                counter_low: 0,
                counter_high: 0,
                block_len: BLAKE3_MSG_LEN as u32,
                flags: (B3F_PARENT | B3F_KEYED_HASH | root_flag).into(),
            },
            msg: MessageType::Parent {
                cv_low: left_cv,
                cv_high: right_cv,
            },
            out: HashOut::None,
        });
        CvType::Instruction {
            idx: instructions.len() - 1,
        }
    }
}
