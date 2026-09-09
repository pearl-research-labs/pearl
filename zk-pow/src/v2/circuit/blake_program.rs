//! Blake3 program types: instruction-level abstractions for Blake3 hashing.

use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};

use crate::v2::{
    api::proof::{Hash256, PrivateProofParams},
    circuit::{blake3::blake3_compress::blake3_compress, pearl_noise::MMSlice, pearl_preprocess::read_blake_msg_from_matrix},
    ensure_eq,
};

pub use crate::v2::circuit::blake3::blake3_compress::Blake3Tweak;

pub const DWORD_SIZE: usize = 8; // bytes per dword

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
    MatrixLeaf { mat_data: BlakeMsg },          // location of the message in the matrix.
    AuxiliaryLeaf { idx: usize },               // index among auxiliary messages.
    Parent { cv_low: CvType, cv_high: CvType }, // bytes 0..32 and 32..64 of message
}

#[derive(Clone, Debug, Copy)]
pub struct BlakeInstruction {
    pub is_cv_key: bool,    // Is CV the blockheader-derived key? If false, then CV is previous hash.
    pub tweak: Blake3Tweak, // last 16 bytes of the initial state
    pub msg: MessageType,
    pub is_hash_a: bool, // Is this instruction output hash A?
    pub is_hash_b: bool, // Is this instruction output hash B?
}

#[derive(Clone, Debug)]
pub struct BlakeProgram {
    pub num_a_rows: usize,         // Number of A rows being proved (h)
    pub num_b_cols: usize,         // Number of B columns being proved (w)
    pub strip_length: usize,       // length of the strips relevant to the proof (k-k%r)
    pub num_auxiliary_msgs: usize, // each msg being 64 bytes
    pub num_auxiliary_cvs: usize,  // each cv being 32 bytes
    pub instructions: Vec<BlakeInstruction>,
}

impl BlakeProgram {
    pub fn evaluate_blake(&self, job_key: Hash256, private_params: &PrivateProofParams) -> Result<(Hash256, Hash256)> {
        ensure_eq!(private_params.s_a.len(), self.num_a_rows);
        ensure_eq!(private_params.s_b.len(), self.num_b_cols);
        let strips = MMSlice {
            a: private_params.s_a.clone(),
            b: private_params.s_b.clone(),
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

        let (mut hash_a, mut hash_b) = (None, None);
        for (idx, inst) in self.instructions.iter().enumerate() {
            if inst.is_hash_a {
                hash_a = Some(cvs[idx]);
            }
            if inst.is_hash_b {
                hash_b = Some(cvs[idx]);
            }
        }
        ensure!(hash_a.is_some() && hash_b.is_some());
        Ok((hash_a.unwrap(), hash_b.unwrap()))
    }
}
