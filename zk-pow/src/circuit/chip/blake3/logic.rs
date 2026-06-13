use crate::circuit::{
    chip::blake3::{blake3_compress::Blake3Tweak, program::MatDwordId},
    utils::evaluator::Evaluator,
};

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
    /// Load a single dword (8 bytes / 2 packed elements) from the A/B matrix into BLAKE3_MSG_BUFFER.
    Matrix { dword_id: MatDwordId },
    /// Load a single dword from routing data into BLAKE3_MSG_BUFFER.
    /// Routing goes through the auxiliary path (IS_MSG_AUX_DATA) so that MAT_UNPACK stays zero.
    /// `hotspot_idx` is the index of the 64-byte long hotspot block at hand.
    /// `idx_in_block` is the index, within the hotspot, of the current DWORD's first byte.
    RoutingData { hotspot_idx: usize, idx_in_block: usize },
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
    pub is_hash_routing: bool,  // true if this row outputs hash of routing
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
            is_hash_routing: false,
            is_hash_jackpot: false,
            cv_is_commitment: false,
        }
    }
}

/// 3-bit encoding of [`MessageDataType`] into `[bit0, bit1, bit2]` for the IS_MSG_BITS
/// preprocessed columns. The encoding is consumed on the AIR side by [`decode_is_msg_bits`].
///
/// | Variant            | bit0 | bit1 | bit2 |
/// |--------------------|------|------|------|
/// | Matrix             |  1   |  0   |  0   |
/// | Jackpot            |  0   |  1   |  0   |
/// | AuxiliaryData /    |  0   |  1   |  1   |
/// |   RoutingData      |      |      |      |
/// | PreviousCv         |  0   |  0   |  1   |
/// | None               |  0   |  0   |  0   |
///
/// The AIR derives three signals from these bits (see [`decode_is_msg_bits`]):
/// - `is_msg_uint8_data = bit0 + bit1*bit2` -- fires for `100` and `011`: load UINT8_DATA
///   into blake3_msg. For Matrix rows, UINT8_DATA additionally goes through the int7→uint8
///   lookup (filtered by bit0 directly in `pearl_stark.rs`). For Aux/Routing rows, UINT8_DATA
///   is loaded without conversion; IS_FIRST_OUTER / IS_SECOND_OUTER further constrain whether
///   UINT8_DATA must match preprocessed outer indices (see `constraints.rs`).
/// - `is_msg_jackpot = bit1*(1-bit2)` -- fires for `010`: loads the jackpot slice into the
///   full msg_buffer.
/// - `is_msg_cv = bit2*(1-bit1)` -- fires for `001`: loads CV_IN into msg_buffer[8..16].
pub fn encode_is_msg_bits(data_source: &MessageDataType) -> [bool; 3] {
    match data_source {
        MessageDataType::Matrix { .. } => [true, false, false],
        MessageDataType::Jackpot => [false, true, false],
        MessageDataType::AuxiliaryData { .. } | MessageDataType::RoutingData { .. } => [false, true, true],
        MessageDataType::PreviousCv { .. } => [false, false, true],
        MessageDataType::None => [false, false, false],
    }
}

/// Decodes the 3 IS_MSG_BITS into the AIR signal expressions used by the blake3 constraints.
/// This is the constraint-side inverse of [`encode_is_msg_bits`]. Returns three signals:
///
/// - `is_msg_jackpot    = bit1 * (1-bit2)`  (only `010` activates)
/// - `is_msg_uint8_data = bit0 + bit1*bit2` (`100` or `011` -- load UINT8_DATA into blake3_msg;
///   for Matrix rows UINT8_DATA also goes through int7→uint8; for Aux/Routing rows it does not,
///   and IS_FIRST_OUTER / IS_SECOND_OUTER further constrain whether UINT8_DATA must match
///   preprocessed outer indices)
/// - `is_msg_cv         = bit2 * (1-bit1)`  (only `001` activates)
///
/// Note: `is_msg_mat = bit0` is not returned here; it is only used as a lookup filter column
/// in `pearl_stark.rs` (int7→uint8 table), where the raw bit column is referenced directly.
pub(crate) fn decode_is_msg_bits<V: Copy, E: Evaluator<V, S>, S: Copy>(eval: &mut E, one: V, bits: [V; 3]) -> (V, V, V) {
    let [bit0, bit1, bit2] = bits;
    let not_bit_1 = eval.sub(one, bit1);
    let not_bit_2 = eval.sub(one, bit2);
    let is_msg_jackpot = eval.mul(not_bit_2, bit1);
    let is_msg_uint8_data = {
        let is_msg_aux_data = eval.mul(bit1, bit2);
        eval.add(bit0, is_msg_aux_data)
    };
    let is_msg_cv = eval.mul(bit2, not_bit_1);
    (is_msg_jackpot, is_msg_uint8_data, is_msg_cv)
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
