use layout_macro::define_layout;

pub(crate) const BYTES_PER_GOLDILOCKS: usize = 4; // Packing factor of i7/i8/u8 into Goldilocks.
pub(crate) const BITS_PER_LIMB: usize = 13;
pub(crate) const BITS_PER_OUTER_INDEX: usize = 2 * BITS_PER_LIMB; // Number of bits in the outer index
pub(crate) const NOISE_PACKING_BASE: i64 = 129; // Range [-64, 64] has 129 values

define_layout! {
    mod pearl_columns {
        URANGE8_TABLE: 1, // 0..=255
        URANGE8_FREQ: 1,
        URANGE13_TABLE: 1, // 0..=8191 (BITS_PER_LIMB = 13)
        URANGE13_FREQ: 1,
        IRANGE7P1_TABLE: 1, // -64..=64
        IRANGE7P1_FREQ: 1,
        IRANGE8_TABLE: 1, // -128..=127
        IRANGE8_FREQ: 1,
        I8U8_TABLE: 1, // 2^8 table (0..256).map((x as u8) + 256 * (x as i8)) : i8 -> u8 conversion
        I8U8_AUX: 1, // 0..1 column, helping ensure I8U8_TABLE is correct.
        I8U8_FREQ: 1,

        CONTROL_PREP: 1, // Unpacked le: control bits || MAT_ID
        IS_RESET_CUMSUM: 1, // bit, should reset jackpot tile with a matmul?
        IS_UPDATE_CUMSUM: 1, // bit, should update jackpot tile with a matmul? If not, we do not load matrices.
        IS_USE_JOB_KEY: 1, // if true we use JOB_KEY, otherwise prev row's CV_OUT.
        IS_USE_COMMITMENT_HASH: 1, // if true we use COMMITMENT_HASH, otherwise prev row's CV_OUT.
        IS_HASH_A: 1, // whether this row outputs hash A.
        IS_HASH_B: 1, // whether this row outputs hash B.
        IS_HASH_ROUTING: 1, // whether this row outputs hash of routing.
        IS_HASH_JACKPOT: 1, // whether this row outputs hash of jackpot.
        IS_CV_IN: 1, // Do we even want to load CV_IN?
        IS_NEW_BLAKE: 1, // Is blake3 in current not continuing previous row's blake3? (either IS_LAST_ROUND in previous row, or it did not compute blake3)
        IS_LAST_ROUND: 1, // Is this the 8th (last) round of a blake3 compression?
        IS_MSG_BITS: 3, // 3-bit encoding of the data source loaded into BLAKE3_MSG_BUFFER. See encode_is_msg_bits / decode_is_msg_bits in blake3/logic.rs for the full table.
        IS_FIRST_OUTER: 1, // set to 1 when, during routing, the first UINT8_DATA u32 limb is an outer index
        IS_SECOND_OUTER: 1, // set to 1 when, during routing, the second UINT8_DATA u32 limb is an outer index
        IS_LOAD: 1, // load jackpot to BIT_REG and CUMSUM_TILE to CUMSUM_BUFFER?
        IS_XOR: 1, // XOR CUMSUM_BUFFER intermediate to BIT_REG?
        IS_SHIFT3: 1, // shift BIT_REG >>>= 3?
        STORE_BITS: 2, // 2-bit encoding of store rotation: Store0=1, Store1=2, Store2=3
        IS_DUMP_CUMSUM_BUFFER: 1, // dump CUMSUM_TILE to CUMSUM_BUFFER?
        JACKPOT_IDX: 8, // indicators: is_store[i] for i in 0..16 || is_load[i] for i in 0..16
        MAT_ID_LIMBS: 2, // range check for MAT_ID
        MAT_ID: 1, // Compact matrix index, derived from CONTROL_PREP.

        OUTER_INDICES_PACKED_PREP: 1, // preprocessed column. Holds a tight encoding of `UINT8_DATA`, in case this row is hashing outer indices, as part of the routing Merkle proof.
        OUTER_INDEX_FIRST: 2, // Unpacking half of OUTER_INDICES_PACKED. Equal to first u32 in UINT8_DATA in case IS_FIRST_OUTER=true.
        OUTER_INDEX_SECOND: 2, // Unpacking half of OUTER_INDICES_PACKED. Equal to second u32 in UINT8_DATA in case IS_SECOND_OUTER=true.

        STARK_ROW_IDX: 1,

        MAT_UNPACK: 8, // 8 int7 elements
        UINT8_DATA: 8, // If IS_MSG_MAT: MAT_UNPACK converted to u8. Otherwise: auxiliary data.
        NOISE_PACKED_PREP: 1, // Noise associated with the mat in MAT_PACKED_IDXED.
        NOISE_UNPACK: 8, // 8 int7 elements

        NOISED_PACKED: 2, // MAT + NOISE packed as 4 i8 elements per Goldilocks.
        MAT_FREQ: 1, // Number of times NOISED_PACKED is read, for matmul purposes.

        BLAKE3_MSG_BUFFER: 16, // Blake3 msg buffering. In round 8, it contains the data that enters blake at round 1.

        CV_OR_TWEAK_PREP: 1, // either cv_idx or blake3_tweak
        CV_IN: 8, // CV for BLAKE3, read from CV_OUT_PACKED using logup with CV_OR_TWEAK_PREP as index.
        BLAKE3_MSG: 16, // message entering blake3; packed le, 4 bytes per goldilocks (uint8).
        BLAKE3_CV: 8, // CVs ready for blake3. packed le, 4 bytes per goldilocks.
        BLAKE3_ROUND: 1056, // AIR that a blake3 round done correctly; CV_OUT of last round contains blake3 output.

        CV_OUT: 8, // u32 le encoding of hash; Output CV of BLAKE3.
        CV_OUT_FREQ: 1, // Frequency of logup of CV_OUT.

        AB_ID_PREP: 1, // A_ID || B_ID (both MAT_ID's)
        AB_ID_LIMBS: 4, // range check for AB_ID_PREP.
        A_ID: 1,
        B_ID: 1,
        A_NOISED: 8, // TILE_H × TILE_D / 4
        A_NOISED_UNPACK: 32, // TILE_H × TILE_D
        B_NOISED: 8,  // TILE_W × TILE_D / 4
        B_NOISED_UNPACK: 32, // TILE_W × TILE_D
        CUMSUM_TILE: 4, // TILE_H × TILE_W. int32
        CUMSUM_BUFFER: 4, // Buffering of CUMSUM_TILE. int32.
        JACKPOT_MSG: 16, // jackpot blake3 message. uint32.
        BIT_REG: 32, // Bitwise representation. Helps xoring 32-bit integers between rows.

    }
}

define_layout! {
    mod pearl_public {
        JOB_KEY: 8, // Blake3(BlockHeader || MiningConfiguration)
        COMMITMENT_HASH: 8, // Commitment hash a.k.a. a_noise_seed
        HASH_A: 8, // Blake3(A, key=JOB_KEY).
        HASH_B: 8, // Blake3(B^t, key=JOB_KEY).
        HASH_ROUTING: 8, // Blake3(routing, key=JOB_KEY).
        HASH_JACKPOT: 8, // Blake3(JACKPOT_MSG, key=COMMITMENT_HASH).
    }
}
