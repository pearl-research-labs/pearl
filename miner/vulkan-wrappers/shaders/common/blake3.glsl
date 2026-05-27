// ============================================================
// BLAKE3 constants (from blake3_constants.hpp)
// ============================================================
const uint B3_IV0 = 0x6A09E667u;
const uint B3_IV1 = 0xBB67AE85u;
const uint B3_IV2 = 0x3C6EF372u;
const uint B3_IV3 = 0xA54FF53Au;
const uint B3_IV4 = 0x510E527Fu;
const uint B3_IV5 = 0x9B05688Cu;
const uint B3_IV6 = 0x1F83D9ABu;
const uint B3_IV7 = 0x5BE0CD19u;

const uint B3_KEYED_HASH = 1u << 4;
const uint B3_CHUNK_START = 1u << 0;
const uint B3_CHUNK_END = 1u << 1;
const uint B3_ROOT = 1u << 3;
const uint B3_PARENT = 1u << 2;

const uint B3_FLAGS_SINGLE = B3_KEYED_HASH | B3_CHUNK_START | B3_CHUNK_END | B3_ROOT;

// ============================================================
// BLAKE3 G function
// ============================================================
uint b3_rotr(uint x, uint n) {
    return (x >> n) | (x << (32u - n));
}

void b3_G(inout uint a, inout uint b, inout uint c, inout uint d, uint x, uint y) {
    a = a + b + x;
    d = b3_rotr(d ^ a, 16);
    c = c + d;
    b = b3_rotr(b ^ c, 12);
    a = a + b + y;
    d = b3_rotr(d ^ a, 8);
    c = c + d;
    b = b3_rotr(b ^ c, 7);
}

// ============================================================
// One BLAKE3 round (8 G operations)
// ============================================================
void b3_round(inout uint[16] state, uint[16] block) {
    b3_G(state[0], state[4], state[8],  state[12], block[0],  block[1]);
    b3_G(state[1], state[5], state[9],  state[13], block[2],  block[3]);
    b3_G(state[2], state[6], state[10], state[14], block[4],  block[5]);
    b3_G(state[3], state[7], state[11], state[15], block[6],  block[7]);
    b3_G(state[0], state[5], state[10], state[15], block[8],  block[9]);
    b3_G(state[1], state[6], state[11], state[12], block[10], block[11]);
    b3_G(state[2], state[7], state[8],  state[13], block[12], block[13]);
    b3_G(state[3], state[4], state[9],  state[14], block[14], block[15]);
}

// ============================================================
// BLAKE3 message word permutation
// ============================================================
void b3_permute(inout uint[16] block) {
    uint[16] orig = block;
    block[0]  = orig[2];
    block[1]  = orig[6];
    block[2]  = orig[3];
    block[3]  = orig[10];
    block[4]  = orig[7];
    block[5]  = orig[0];
    block[6]  = orig[4];
    block[7]  = orig[13];
    block[8]  = orig[1];
    block[9]  = orig[11];
    block[10] = orig[12];
    block[11] = orig[5];
    block[12] = orig[9];
    block[13] = orig[14];
    block[14] = orig[15];
    block[15] = orig[8];
}

// ============================================================
// BLAKE3 keyed compression: single 64-byte block
// Matches compress_msg_block_u32 exactly.
// Input message is 16 uint32, key is 8 uint32.
// Output hash[8] is overwritten.
// ============================================================
void b3_compress_keyed(
    uint[16] message,
    uint[8] key,
    uint flags,
    inout uint[8] out_hash
) {
    uint[16] state;
    uint[16] block = message;

    for (int i = 0; i < 8; i++) {
        state[i] = key[i];
    }
    state[8]  = B3_IV0;
    state[9]  = B3_IV1;
    state[10] = B3_IV2;
    state[11] = B3_IV3;
    state[12] = 0u;
    state[13] = 0u;
    state[14] = 64u;
    state[15] = flags;

    for (int i = 0; i < 6; i++) {
        b3_round(state, block);
        b3_permute(block);
    }
    b3_round(state, block);

    for (int i = 0; i < 8; i++) {
        out_hash[i] = state[i] ^ state[i + 8];
    }
}
