use crate::v1::circuit::chip::blake3::{blake3_compress::Blake3Tweak, program::MatDwordId};

#[derive(Clone, Debug, Copy)]
pub enum AuxDataType {
    /// Auxiliary message (64 bytes = 8 dwords). dword_idx ranges 0..8.
    Msg { aux_msg_idx: usize },
    /// Auxiliary CV (32 bytes = 4 dwords). dword_idx ranges 0..4.
    Cv { aux_cv_idx: usize },
}

/// Describe how to populate BLAKE3_MSG_BUFFER in each blake round.
#[derive(Clone, Debug, Copy, Default)]
pub enum MessageDataType {
    /// Load a single dword (8 bytes / 2 packed elements) from the matrix into BLAKE3_MSG_BUFFER.
    Matrix { dword_id: MatDwordId },
    /// Load a single dword from auxiliary data (message or CV) into BLAKE3_MSG_BUFFER.
    AuxiliaryData { aux_type: AuxDataType, dword_idx: usize },
    /// Load 4 dwords (32 bytes) from a previous CV_OUT into BLAKE3_MSG_BUFFER.
    PreviousCv { source_row_idx: usize },
    /// Load the entire BLAKE3_MSG_BUFFER from jackpot.
    Jackpot,

    #[default]
    None, // No data loading in this round. Used for rounds that don't need to override BLAKE3_MSG_BUFFER.
}

#[derive(Clone, Debug, Copy)]
pub struct BlakeRoundLogic {
    /// Specifies what data to load into BLAKE3_MSG_BUFFER this round.
    /// In round 8 the loaded data (in BLAKE3_MSG_BUFFER) should match the correct message that blake3 were processing since first round.
    pub data_source: MessageDataType,
    pub(crate) blake3_tweak: Option<Blake3Tweak>, // Some only at round 1
    /// Round index within a blake3 compression, 1-indexed: 1,2,3,4,5,6,7,8.
    pub(crate) round_idx: usize,

    /// Which STARK row to read its CV_OUT into this row's CV_IN.
    pub idx_of_row_whence_to_read_cv: Option<usize>,
    pub is_hash_a: bool,        // true if this row outputs hash A
    pub is_hash_b: bool,        // true if this row outputs hash B
    pub is_hash_jackpot: bool,  // true if this row outputs hash of jackpot
    pub cv_is_commitment: bool, // true if BLAKE3_CV should be commitment_hash
}

impl Default for BlakeRoundLogic {
    fn default() -> Self {
        Self {
            data_source: MessageDataType::None,
            blake3_tweak: None,
            round_idx: 1, // Most permissive option, no constraints imposed.
            idx_of_row_whence_to_read_cv: None,
            is_hash_a: false,
            is_hash_b: false,
            is_hash_jackpot: false,
            cv_is_commitment: false,
        }
    }
}

impl BlakeRoundLogic {
    pub fn is_use_job_key(&self) -> bool {
        (self.idx_of_row_whence_to_read_cv.is_none() || matches!(self.data_source, MessageDataType::PreviousCv { .. }))
            && !self.is_use_commitment_hash()
    }
    pub fn is_use_commitment_hash(&self) -> bool {
        self.cv_is_commitment
    }
}
