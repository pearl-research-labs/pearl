//! Blake3 program types: instruction-level abstractions for Blake3 hashing.

use std::mem;

use anyhow::{Result, ensure};
pub use blake3::BLOCK_LEN;
use pearl_blake3::{B3F_CHUNK_END, B3F_CHUNK_START, B3F_KEYED_HASH, B3F_PARENT, B3F_ROOT, BLAKE3_CHUNK_LEN, BLAKE3_MSG_LEN};
use serde::{Deserialize, Serialize};

use crate::{
    api::proof::{Hash256, PrivateProofParams, PublicProofParams},
    circuit::{
        chip::blake3::{
            blake3_compress::blake3_compress,
            logic::{AuxDataType, BlakeRoundLogic, MessageDataType},
        },
        pearl_noise::MMSlice,
        pearl_preprocess::read_blake_msg_from_matrix,
    },
    ensure_eq,
};

pub use super::blake3_compress::Blake3Tweak;

pub const DWORD_SIZE: usize = 8; // bytes per dword

/// Number of STARK rows emitted per [`BlakeInstruction`] (one blake3 compression:
/// 7 mixing rounds + 1 output/finalization row).
pub const ROUNDS_PER_BLAKE_INSTRUCTION: usize = 8;

#[derive(Serialize, Deserialize, Clone, Debug, Copy, Hash, Eq, PartialEq, Default)]
pub struct MatDwordId {
    pub is_b_strip: bool,    // If true, then B matrix. If false, then A matrix.
    pub strip_idx: usize,    // If a, 0..h; If b, 0..w
    pub idx_in_strip: usize, // divisible by 8, idx of first byte element.
}

#[derive(Serialize, Deserialize, Clone, Debug, Copy, Hash, Eq, PartialEq)]
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
    MatrixLeaf { mat_data: BlakeMsg },          // location of the message in the A/B matrix.
    RoutingLeaf { hotspot_idx: usize }, // 64 contiguous bytes from a routing hotspot block (each block is exactly one blake3 block).
    AuxiliaryLeaf { idx: usize },       // index among auxiliary messages.
    Parent { cv_low: CvType, cv_high: CvType }, // bytes 0..32 and 32..64 of message
}

#[derive(Clone, Debug, Copy)]
pub struct BlakeInstruction {
    pub is_cv_key: bool,    // Is CV the blockheader-derived key? If false, then CV is previous hash.
    pub tweak: Blake3Tweak, // last 16 bytes of the initial state
    pub msg: MessageType,
    pub is_hash_a: bool,       // Is this instruction output hash A?
    pub is_hash_b: bool,       // Is this instruction output hash B?
    pub is_hash_routing: bool, // Is this instruction output hash of routing?
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
                MessageType::AuxiliaryLeaf { idx } => MessageDataType::AuxiliaryData {
                    aux_type: AuxDataType::Msg { aux_msg_idx: idx },
                    dword_idx: i,
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
                is_hash_a: i == 7 && self.is_hash_a,
                is_hash_b: i == 7 && self.is_hash_b,
                is_hash_routing: i == 7 && self.is_hash_routing,
                is_hash_jackpot: false,
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
    pub num_auxiliary_msgs: usize, // each msg being 64 bytes
    pub num_auxiliary_cvs: usize,  // each cv being 32 bytes
    pub instructions: Vec<BlakeInstruction>,
}

#[derive(Debug, Clone, Copy)]
pub enum ProofSource {
    A,
    B,
    Routing,
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
    pub fn new(params: &PublicProofParams) -> (Self, Vec<AuxiliaryMsgLocation>, Vec<AuxiliaryCvLocation>) {
        let mut instructions = Vec::new();
        let mut msgs: Vec<AuxiliaryMsgLocation> = vec![];
        let mut cvs: Vec<AuxiliaryCvLocation> = vec![];
        let m = params.m() as usize;
        let common_dim = params.common_dim();
        let total_b_cols = params.total_b_cols();

        let cv = recursive_compilation(
            0,
            pearl_blake3::padded_chunk_len(m * common_dim),
            m,
            common_dim,
            &params.a_rows_indices(),
            params.dot_product_length(),
            false,
            false,
            &mut instructions,
            &mut msgs,
            &mut cvs,
        );
        let CvType::Instruction { idx } = cv else { unreachable!() };
        instructions[idx].is_hash_a = true;

        let cv = recursive_compilation(
            0,
            pearl_blake3::padded_chunk_len(total_b_cols * common_dim),
            total_b_cols,
            common_dim,
            &params.b_cols_indices(),
            params.dot_product_length(),
            true,
            false,
            &mut instructions,
            &mut msgs,
            &mut cvs,
        );
        let CvType::Instruction { idx } = cv else { unreachable!() };
        instructions[idx].is_hash_b = true;

        let mut num_routing_strips = 0usize;
        if let Some(moe) = params.moe.as_ref() {
            // Routing is jagged, but the per-expert lists always sum to `m * top_k`
            // (every token is routed to exactly `top_k` experts). We pad the entry
            // count up to a multiple of 16 so the byte length tiles evenly into the
            // 64-byte virtual rows the commitment uses below.
            let total_routing_entries = params.padded_routing_entries().expect("MoE params present");
            let full_routing_size = total_routing_entries * mem::size_of::<u32>();
            let routing_row_len = 64; // Just to satisfy the conditions.
            let num_rows = full_routing_size / routing_row_len;
            let inner = params.a_inner_indices();
            let hotspots = routing_blake_hotspot_rows(moe.expert_start_offset(), &inner);
            num_routing_strips = hotspots.len();

            let cv = recursive_compilation(
                0,
                pearl_blake3::padded_chunk_len(num_rows * routing_row_len),
                num_rows,
                routing_row_len,
                &hotspots,
                routing_row_len,
                false,
                true,
                &mut instructions,
                &mut msgs,
                &mut cvs,
            );
            let CvType::Instruction { idx } = cv else { unreachable!() };
            instructions[idx].is_hash_routing = true;
        }

        (
            BlakeProgram {
                num_a_rows: params.h(),
                num_b_cols: params.w(),
                strip_length: params.dot_product_length(),
                num_routing_strips,
                num_auxiliary_msgs: msgs.len(),
                num_auxiliary_cvs: cvs.len(),
                instructions,
            },
            msgs,
            cvs,
        )
    }

    pub fn evaluate_blake(
        &self,
        job_key: Hash256,
        private_params: &PrivateProofParams,
        opt_hash_routing: Option<Hash256>,
    ) -> Result<(Hash256, Hash256)> {
        ensure_eq!(private_params.s_a.len(), self.num_a_rows);
        ensure_eq!(private_params.s_b.len(), self.num_b_cols);
        let strips = MMSlice {
            a: private_params.s_a.clone(),
            b: private_params.s_b.clone(),
            routing: private_params.s_routing.clone(),
        };
        let mut cvs = vec![];

        for instruction in &self.instructions {
            let cv_in = if instruction.is_cv_key {
                job_key
            } else {
                *cvs.last().unwrap()
            };
            let msg = match instruction.msg {
                MessageType::MatrixLeaf { mat_data } => read_blake_msg_from_matrix(&strips, mat_data).map(|b| b as u8),
                MessageType::RoutingLeaf { hotspot_idx } => {
                    let strip = &strips.routing[hotspot_idx];
                    std::array::from_fn(|i| strip[i])
                }
                MessageType::AuxiliaryLeaf { idx } => private_params.external_msgs[idx],
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

        let (mut hash_a, mut hash_b, mut hash_routing) = (None, None, None);
        for (idx, inst) in self.instructions.iter().enumerate() {
            if inst.is_hash_a {
                hash_a = Some(cvs[idx]);
            }
            if inst.is_hash_b {
                hash_b = Some(cvs[idx]);
            }
            if inst.is_hash_routing {
                hash_routing = Some(cvs[idx]);
            }
        }
        ensure!(hash_a.is_some() && hash_b.is_some());
        if let Some(expected_hash_routing) = opt_hash_routing {
            let got = hash_routing
                .ok_or_else(|| anyhow::anyhow!("Blake program has no is_hash_routing output but hash_routing was expected"))?;
            ensure_eq!(
                got,
                expected_hash_routing,
                "hash_routing mismatch between Blake evaluation and public commitment"
            );
        }
        Ok((hash_a.unwrap(), hash_b.unwrap()))
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
    is_b_matrix: bool,
    is_routing: bool,
    instructions: &mut Vec<BlakeInstruction>,
    out_msgs: &mut Vec<AuxiliaryMsgLocation>,
    out_cvs: &mut Vec<AuxiliaryCvLocation>,
) -> CvType {
    debug_assert!(strip_len <= row_len);
    debug_assert!(row_len.is_multiple_of(BLAKE3_MSG_LEN), "row_len must be divisible by 64");
    debug_assert!(
        hotspot_strips.windows(2).all(|w| w[0] < w[1]),
        "hotspot_strips must be sorted with no duplicates"
    );

    let source = if is_routing {
        ProofSource::Routing
    } else if is_b_matrix {
        ProofSource::B
    } else {
        ProofSource::A
    };

    let is_root = start == 0 && end == pearl_blake3::padded_chunk_len(num_rows * row_len);
    let root_flag = is_root as u8 * B3F_ROOT;

    // Binary search: first strip whose byte range ends after `start`
    // i.e. first s where s * row_len + strip_len > start
    let lo = hotspot_strips.partition_point(|&s| s as usize * row_len + strip_len <= start);
    let intersects_strips = hotspot_strips[lo..].first().is_some_and(|&s| (s as usize * row_len) < end);

    if !intersects_strips && end - start >= BLAKE3_CHUNK_LEN {
        out_cvs.push(AuxiliaryCvLocation {
            global_start: start,
            global_end: end,
            source,
        });
        CvType::Auxiliary { idx: out_cvs.len() - 1 }
    } else if end - start <= BLAKE3_CHUNK_LEN {
        // end - start < BLAKE3_CHUNK_LEN or end - start == BLAKE3_CHUNK_LEN and intersects_strips
        debug_assert!(start < end);
        let chunk_idx = start / BLAKE3_CHUNK_LEN;
        let mut is_first_in_chunk = true;
        let mut msg_start = start;
        while msg_start < end {
            let mut is_auxiliary_msg = true;
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

            // Assumes row_len divisible by 64. Since row_len % 64 == 0, a 64-byte block contains at most one row.
            // Here, we check whether one of the hotpot strings is part of the current block.
            // The row this block falls in, and its byte offset within that row:
            let candidate_row = (msg_start / row_len) as u32;
            let offset_in_row = msg_start % row_len;
            if offset_in_row < strip_len {
                // This block is within the strip prefix of its row — check if that
                // row is one of the hotspot strips.
                if let Ok(hotspot_idx) = hotspot_strips.binary_search(&candidate_row) {
                    let msg = if is_routing {
                        // Each strip has length exactly one blake3 block.
                        assert_eq!(offset_in_row, 0);
                        MessageType::RoutingLeaf { hotspot_idx }
                    } else {
                        MessageType::MatrixLeaf {
                            mat_data: BlakeMsg::contiguous(is_b_matrix, hotspot_idx, offset_in_row),
                        }
                    };
                    instructions.push(BlakeInstruction {
                        is_cv_key: is_first_in_chunk,
                        tweak,
                        msg,
                        is_hash_a: false,
                        is_hash_b: false,
                        is_hash_routing: false,
                    });
                    is_auxiliary_msg = false;
                }
            }

            if is_auxiliary_msg {
                out_msgs.push(AuxiliaryMsgLocation {
                    global_start: msg_start,
                    source,
                });
                instructions.push(BlakeInstruction {
                    is_cv_key: is_first_in_chunk,
                    tweak,
                    msg: MessageType::AuxiliaryLeaf { idx: out_msgs.len() - 1 },
                    is_hash_a: false,
                    is_hash_b: false,
                    is_hash_routing: false,
                });
            }
            is_first_in_chunk = false;
            msg_start += block_len;
        }
        CvType::Instruction {
            idx: instructions.len() - 1,
        }
    } else {
        // intersects_strips && end - start > BLAKE3_CHUNK_LEN: recurse
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
            is_b_matrix,
            is_routing,
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
            is_b_matrix,
            is_routing,
            instructions,
            out_msgs,
            out_cvs,
        );
        instructions.push(BlakeInstruction {
            is_cv_key: true,
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
            is_hash_a: false,
            is_hash_b: false,
            is_hash_routing: false,
        });
        CvType::Instruction {
            idx: instructions.len() - 1,
        }
    }
}
