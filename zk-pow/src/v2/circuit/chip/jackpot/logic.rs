#[derive(Clone, Debug, Copy, Default)]
pub enum BitRegSrc {
    #[default]
    Jackpot, // bit_reg = jackpot[idx]
    Xor,    // bit_reg ^= prev_tile_buffer[0]
    Shift3, // bit_reg >>>= 3*LROT_PER_TILE
}

// How to store bit_reg
#[derive(Clone, Debug, Copy, Default)]
pub enum BitRegDst {
    Store0, // jackpot[idx] = bit_reg >>> 0
    #[default]
    Store1, // jackpot[idx] = bit_reg >>> LROT_PER_TILE
    Store2, // jackpot[idx] = bit_reg >>> 2*LROT_PER_TILE
}

#[derive(Clone, Debug, Copy, Default)]
pub struct JackpotLogic {
    // Where to load bit_reg from
    pub src: BitRegSrc,
    // Where to store bit_reg to. If Store1 and src=Jackpot, then both unchanged.
    pub dst: BitRegDst,
    // Which jackpot idx to write BitReg to.
    pub jackpot_idx: usize,
    // Dump CUMSUM_TILE to BUFFER_CUMSUM?
    pub is_dump_cumsum_buffer: bool,
}
