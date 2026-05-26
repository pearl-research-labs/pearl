# Specification: Vulkan Standalone Pearl Miner

**Date:** 2026-05-23
**Status:** DRAFT
**Type:** Specification
**Requirements:** [REQ-VULKAN-MINER.md](../requirements/REQ-VULKAN-MINER.md)

## Overview

This document specifies the implementation of a Vulkan 1.3 compute pipeline
for the pearl-gemm Proof-of-Work mining algorithm. The miner runs as a
standalone daemon — no LLM, no vLLM — connecting to the pearl gateway RPC
for job acquisition and block submission.

All arithmetic is integer (int8 × int8 → int32). BLAKE3 hashing follows
CUDA reference exactly. GEMM uses manual tiling with shared memory,
subgroup reductions, and no vendor-specific cooperative matrix extensions.

## Language Choices

| Component | Language | Rationale |
|-----------|----------|-----------|
| Host binary | Rust | BLAKE3 crate, gateway RPC (Unix socket), `ash` Vulkan bindings |
| Shaders | GLSL 460 | Widest Vulkan compatibility, compile via `glslc` offline |
| SPIR-V compilation | `glslc` (build.rs) | Offline during `cargo build`, no runtime compiler |

A Rust host is preferred because the existing codebase already has Rust
crates for BLAKE3, Merkle trees (`pearl-blake3`), and gateway types
(`zk-pow`, `pearl-gateway`). A pure-C alternative is possible if `C` linkage
is preferred for the mining loop.

## Crate Layout

```
miner/vulkan-miner/
├── Cargo.toml              # deps: ash 0.38+, blake3, tokio, serde, bincode, tracing
├── build.rs                # compiles GLSL → SPIR-V via glslc
├── src/
│   ├── main.rs             # Entry: args, device selection, mining loop
│   ├── context.rs          # Vulkan context (instance, device, queues)
│   ├── buffers.rs          # Buffer + allocation management
│   ├── pipelines.rs        # VkComputePipeline per kernel
│   ├── mining.rs           # MiningLoop: dispatch orchestration
│   ├── proof.rs            # Merkle proof construction after block found
│   ├── rng.rs              # CPU-side RNG for job_key (blake3-based)
│   └── gateway/
│       ├── mod.rs
│       └── client.rs       # Unix domain socket RPC to pearl-gateway
├── shaders/
│   ├── k1_random_fill.comp       # K1: fill A/B with uniform random int8
│   ├── k2_noise_gen.comp         # K2: BLAKE3 noise matrices
│   └── k3_noised_gemm.comp       # K3: noised GEMM + XOR jackpot accumulation
│       # K4 (jackpot hash) is folded into K3 — see §K3
└── tests/
    ├── k1_random_fill_test.rs    # CPU comparison of RNG output
    ├── k2_noise_gen_test.rs      # Comparison vs pearl_noise.rs reference
    └── k3_noised_gemm_test.rs    # Full pipeline test vs mine.rs CPU reference
```

### `Cargo.toml` (key dependencies)

```toml
[package]
name = "vulkan-miner"
version = "0.1.0"

[dependencies]
ash = { version = "0.38", features = ["linked"] }
blake3 = "1.5"                     # host-side hashing, CPU Merkle trees
pearl-blake3 = { path = "../../../pearl-blake3" }  # MerkleTree
zk-pow-types = { path = "../../../zk-pow/src/ffi" }  # PlainProof
tokio = { version = "1", features = ["full"] }
serde = { version = "1", features = ["derive"] }
bincode = "1"
tracing = "0.1"
tracing-subscriber = "0.3"
anyhow = "1"
clap = { version = "4", features = ["derive"] }

[build-dependencies]
# build.rs invokes `glslc` directly; no crate needed
```

## Vulkan Initialization

### Required Features and Extensions

```rust
// Vulkan 1.3 core features (all required):
//   VK_KHR_shader_float16_int8          → shaderInt8
//   VK_KHR_shader_subgroup_extended_types → subgroup int8 ops
//   VK_KHR_8bit_storage                 → SSBO access to int8
//   VK_KHR_buffer_device_address        → optional, for push-constant-style buf refs

// Device creation pNext chain:
VkPhysicalDeviceShaderFloat16Int8FeaturesKHR int8_feat{
    .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES_KHR,
    .shaderInt8 = VK_TRUE,
};
VkPhysicalDevice8BitStorageFeaturesKHR int8_storage{
    .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES_KHR,
    .storageBuffer8BitAccess = VK_TRUE,
    .uniformAndStorageBuffer8BitAccess = VK_TRUE,
};
VkPhysicalDeviceVulkan13Features vulkan_13{
    .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
    .pNext = &int8_feat,
    .subgroupSizeControl = VK_TRUE,
    .computeFullSubgroups = VK_TRUE,
    .synchronization2 = VK_TRUE,
    .dynamicRendering = VK_TRUE,
};
```

### Device Selection

Select the first physical device supporting all required features and
belonging to `VK_QUEUE_COMPUTE_BIT` family. Prefer discrete GPU over
integrated.

### Queue

One compute queue. No concurrent queue sharing needed for single-card mining.

## Buffer Layout

### SSBO Binding Convention

All kernels use `set = 0`. Binding slots per kernel:

| Kernel | Binding 0 | Binding 1 | Binding 2 | Binding 3 | Binding 4 | Binding 5 | Binding 6 |
|--------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
| K1 random fill | A (int8) | B (int8) | — | — | — | — | — |
| K2 noise gen | EAL (int8) | EAR (int8) | EBL (int8) | EBR (int8) | hash_A (uint32[8]) | hash_B (uint32[8]) | — |
| K3 noised GEMM | A (int8) | B (int8) | EAL (int8) | EAR (int8) | EBL (int8) | EBR (int8) | jackpot (uint32[16]) |

### Push Constants

K3 uses push constants for tile configuration:

```glsl
layout(push_constant) uniform PushConstants {
    uint matrix_m;         // rows of A
    uint matrix_n;         // cols of B
    uint matrix_k;         // common dim
    uint noise_rank;       // r
    uint tile_offset_m;    // tile row start
    uint tile_offset_n;    // tile col start
    // Total: 6 × 4 = 24 bytes (well within 128-byte limit)
};
```

### Memory Allocation Strategy

All buffers are allocated from a single `VkDeviceMemory` pool using
sub-allocation (or use `gpu-allocator` crate). All are
`VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT` for performance.

| Buffer | Size (1024×1024, r=128) | Format |
|--------|------------------------|--------|
| A | m×k × 1 = 1,048,576 B | int8 |
| B | k×n × 1 = 1,048,576 B | int8 |
| EAL | m×r × 1 = 131,072 B | int8 |
| EAR | k×r × 1 = 131,072 B | int8 |
| EBL | k×r × 1 = 131,072 B | int8 |
| EBR | n×r × 1 = 131,072 B | int8 |
| jackpot | 16 × 4 = 64 B | uint32 |
| block_found | 2 × 4 = 8 B | uint32 (tile_coord) |
| **Total** | **~2.6 MB** | |

Buffers are created once and reused across iterations. If matrix dimensions
change between jobs, buffers are recreated.

### Buffer Striding

- A is row-major: element `A[row][col]` = `buf[row * k + col]` (originally m×k)
- B is also row-major: element `B[row][col]` = `buf[row * n + col]` (originally k×n)

Wait — B is k×n. The CUDA code has A as m×k, B as k×n, and the noise
matrices handle transpose via the sparse structure. Let me re-verify from
the Rust reference.

From `noise_generation_kernel.h:18-25`:
- EAL (M, R), EAR (K, R), EBL (K, R), EBR (N, R)
- The noised GEMM does: (A + EAL·EAR^T) × (B + EBL·EBR^T)
  = (A[m×k] + EAL[m×r]·EAR^T[r×k]) × (B[k×n] + EBL[k×r]·EBR^T[r×n])

So A is m×k, B is k×n, both row-major.

## GLSL Shaders

### K1: Random Fill (`k1_random_fill.comp`)

```glsl
#version 460
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, r8) writeonly uniform image2D img_A;
layout(set = 0, binding = 1, r8) writeonly uniform image2D img_B;

layout(push_constant) uniform Params {
    uint m;
    uint n;
    uint k;
    uint seed;   // per-iteration seed from job
} params;

// xorshift RNG returning int8 in [-64, 63]
int8_t xorshift_rng(inout uint state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return int8_t(state & 0x7F) - int8_t(64);  // [-64, 63]
}

void main() {
    uint idx = gl_GlobalInvocationID.x;

    // Each invocation fills one element
    uint state = params.seed ^ (idx * 0x9E3779B9);

    if (idx < params.m * params.k) {
        uint row = idx / params.k;
        uint col = idx % params.k;
        imageStore(img_A, ivec2(col, row), uvec4(uint8_t(xorshift_rng(state))));
    }

    uint b_offset = params.m * params.k;
    if (idx < params.k * params.n) {
        uint row = idx / params.n;
        uint col = idx % params.n;
        imageStore(img_B, ivec2(col, row), uvec4(uint8_t(xorshift_rng(state))));
    }
}
```

**Workgroup dispatch:** `ceil(m*k / 256)` × 1 × 1 for A, combined with
B into one dispatch over `max(m*k, k*n)` invocations.

**Note:** `image2D` with `r8` format is used for writing individual int8
pixels. An alternative is `int8_t[]` SSBO with explicit array indexing —
which is cleaner for GEMM reads. The SSBO approach is preferred.

### K1 Alternative — SSBO-based

```glsl
#version 460
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_shader_8bit_storage : require

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) buffer A_buf { int8_t data[]; } A;
layout(set = 0, binding = 1, std430) buffer B_buf { int8_t data[]; } B;

layout(push_constant) uniform Params {
    uint m;
    uint n;
    uint k;
    uint seed;
} params;

// xorshift as above

void main() {
    uint idx = gl_GlobalInvocationID.x;
    uint state = params.seed ^ (idx * 0x9E3779B9);

    if (idx < params.m * params.k)
        A.data[idx] = xorshift_rng(state);

    uint b_off = params.m * params.k;
    if (idx < params.k * params.n)
        B.data[idx] = xorshift_rng(state);
}
```

### K2: BLAKE3 Noise Generation (`k2_noise_gen.comp`)

This is the most critical port. The CUDA noise generation kernel
(`noise_generation_kernel.h`) calls one BLAKE3 compression per 32 bytes
of output. Each thread processes one 32-byte chunk.

**BLAKE3 compression function in GLSL:**

```glsl
#version 460
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_shader_16bit_storage : require    // for uint16_t if needed
#extension GL_EXT_shader_8bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) buffer EAL { int8_t data[]; } eal;
layout(set = 0, binding = 1, std430) buffer EAR { int8_t data[]; } ear;
layout(set = 0, binding = 2, std430) buffer EBL { int8_t data[]; } ebl;
layout(set = 0, binding = 3, std430) buffer EBR { int8_t data[]; } ebr;
layout(set = 0, binding = 4, std430) buffer HashA { uint data[]; } hash_a;
layout(set = 0, binding = 5, std430) buffer HashB { uint data[]; } hash_b;

layout(push_constant) uniform Params {
    uint m;
    uint n;
    uint k;
    uint rank;      // noise rank r (64 or 128)
} params;

// BLAKE3 constants
const uint IV0 = 0x6A09E667;
const uint IV1 = 0xBB67AE85;
const uint IV2 = 0x3C6EF372;
const uint IV3 = 0xA54FF53A;
const uint IV4 = 0x510E527F;
const uint IV5 = 0x9B05688C;
const uint IV6 = 0x1F83D9AB;
const uint IV7 = 0x5BE0CD19;

const uint KEYED_HASH = 1u << 4;
const uint CHUNK_START = 1u << 0;
const uint CHUNK_END = 1u << 1;
const uint ROOT = 1u << 3;
```

**BLAKE3 round function** — direct port from blake3.cuh:

```glsl
// One column round: G(mixed, a, b, c, d, x, y)
void G(inout uint a, inout uint b, inout uint c, inout uint d, uint x, uint y) {
    a = a + b + x;
    d = (d ^ a); d = (d >> 16) | (d << 16);  // rotr 16 -> rotl 16
    c = c + d;
    b = (b ^ c); b = (b >> 12) | (b << 20);  // rotr 12 -> rotl 20
    a = a + b + y;
    d = (d ^ a); d = (d >> 8) | (d << 24);   // rotr 8
    c = c + d;
    b = (b ^ c); b = (b >> 7) | (b << 25);   // rotr 7
}

// A single BLAKE3 round of 8 G operations
void blake3_round(inout uint[16] state, in uint[16] block) {
    // Column step
    G(state[0], state[4], state[8],  state[12], block[0],  block[1]);
    G(state[1], state[5], state[9],  state[13], block[2],  block[3]);
    G(state[2], state[6], state[10], state[14], block[4],  block[5]);
    G(state[3], state[7], state[11], state[15], block[6],  block[7]);
    // Diagonal step
    G(state[0], state[5], state[10], state[15], block[8],  block[9]);
    G(state[1], state[6], state[11], state[12], block[10], block[11]);
    G(state[2], state[7], state[8],  state[13], block[12], block[13]);
    G(state[3], state[4], state[9],  state[14], block[14], block[15]);
}
```

**BLAKE3 keyed compression** — computes `blake3_keyed(message, key).digest()`:

```glsl
void blake3_compress_keyed(
    uint[16] message,    // 64 bytes = 16 uint32
    uint[8] key,         // 32 bytes = 8 uint32
    uint flags,
    out uint[8] out_hash
) {
    uint[16] state;
    uint[16] block = message;

    // Initialize
    state[0] = key[0]; state[1] = key[1]; state[2] = key[2]; state[3] = key[3];
    state[4] = key[4]; state[5] = key[5]; state[6] = key[6]; state[7] = key[7];
    state[8]  = IV0; state[9]  = IV1; state[10] = IV2; state[11] = IV3;
    state[12] = 0;   state[13] = 0;   state[14] = 64;  state[15] = flags; // counter=0, block_len=64

    // 7 rounds (6 + 1 final)
    for (int i = 0; i < 7; i++) {
        blake3_round(state, block);
        // Permute block
        uint[16] orig = block;
        block[0] = orig[2]; block[1] = orig[6]; block[2] = orig[3];  block[3] = orig[10];
        block[4] = orig[7]; block[5] = orig[0]; block[6] = orig[4];  block[7] = orig[13];
        block[8] = orig[1]; block[9] = orig[11]; block[10] = orig[12]; block[11] = orig[5];
        block[12] = orig[9]; block[13] = orig[14]; block[14] = orig[15]; block[15] = orig[8];
    }

    // Finalize: output[0..7] = state[0..7] ^ state[8..15]
    for (int i = 0; i < 8; i++)
        out_hash[i] = state[i] ^ state[i + 8];
}
```

**Main noise generation** — each thread produces 32 bytes of one noise matrix:

```glsl
void main() {
    uint global_idx = gl_GlobalInvocationID.x;
    uint rank = params.rank;

    // Each thread processes one 32-byte output element
    // Determine which matrix and which position
    uint total_elems = params.m * rank      // EAL
                     + params.k * rank      // EAR
                     + params.k * rank      // EBL
                     + params.n * rank;     // EBR

    if (global_idx >= total_elems) return;

    uint[16] msg = uint[16](0);
    uint[8] key;
    uint[8] output;

    // Determine (matrix_idx, offset_within_matrix)
    uint cursor = 0;
    uint mat;  // 0=EAL, 1=EAR, 2=EBL, 3=EBR
    uint stride, row, offset;
    // ... decode global_idx into matrix + position ...

    // Seed selection: "A_tensor" for EAL/EAR, "B_tensor" for EBL/EBR
    // key = hash_A or hash_B

    // Rule (*) from noise_generation_kernel.h:56:
    // msg[0] = r + 1 for EAL/EBR (dense), msg[1] = r + 1 for EAR/EBL (sparse)
    // where r = global linear index of this 32-byte chunk

    msg[0] = global_idx + 1;  // simplified; actual logic per matrix

    // Copy key from SSBO
    for (int i = 0; i < 8; i++)
        key[i] = hash_a.data[i];  // or hash_b for B-side

    blake3_compress_keyed(msg, key, KEYED_HASH | CHUNK_START | CHUNK_END | ROOT, output);

    // Map output bytes to noise values
    // Dense (EAL/EBR): each byte → int8 in [-32, 32)
    // Sparse (EAR/EBL): each 32-bit word → find +1 and -1 positions
    int8_t noise_vals[32];
    for (int i = 0; i < 32; i++) {
        uint byte_val = (output[i / 4] >> ((i % 4) * 8)) & 0xFF;
        noise_vals[i] = int8_t(byte_val & 0x3F) - int8_t(32);  // [-32, 32)
    }

    // Write to appropriate buffer
    // ... per-matrix logic ...
}
```

The BLAKE3 port in GLSL requires:
1. **BLAKE3 constants** from `blake3_constants.hpp` (IV0-IV7, flag values).
2. **G function** as shown above (direct translation).
3. **Round + permutation** exactly matching blake3.cuh (6 full rounds + permutation + 1 final round).
4. **GLSL limitations**: no `#pragma unroll`, but `[unroll]` attribute works.
   No `uint[16]` as function parameter — use explicit array arguments.

**Workgroup dispatch:** Each workgroup produces `local_size_x` × 32 bytes
of noise output. For m=1024, k=1024, n=1024, r=128:
- EAL: 1024×128 = 131072 outputs → ceil(131072/64) = 2048 workgroups
- Total across all 4 matrices: (1024+1024+1024+1024)×128/64 = 8192 workgroups
  Actually per thread generating 32 bytes = 32 int8 values:
  - EAL threads: 1024 × 128 / 32 = 4096 threads
  - Total: (1024+1024)×128/32 × 2 (for A-side) + (1024+1024)×128/32 × 2 (for B-side)
  → simpler: dispatch `ceil(total_elems / 64)` × 1 × 1

**GLSL Pitfalls:**
- No `genType` templates; use concrete `uint` arrays.
- No `asType` for bitcasting; pack/unpack manually.
- `int8_t` may need `int16_t` staging in practice because GLSL int8
  arithmetic is limited on many implementations. The `int8_t` suffix `h`
  (as in `0x3Fh`) is available with `GL_EXT_shader_explicit_arithmetic_types_int8`.

### K3: Noised GEMM + Jackpot Accumulation (`k3_noised_gemm.comp`)

This is the computationally dominant kernel. Its structure mirrors the CUDA
`pearl_gemm_kernel.h` but without WGMMA/TMA — using manual tiling with
shared memory.

```glsl
#version 460
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_shader_8bit_storage : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_EXT_shader_subgroup_extended_types_int8 : require

layout(local_size_x = 128, local_size_y = 2, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) readonly buffer A_buf { int8_t data[]; } A;
layout(set = 0, binding = 1, std430) readonly buffer B_buf { int8_t data[]; } B;
layout(set = 0, binding = 2, std430) readonly buffer EAL_buf { int8_t data[]; } EAL;
layout(set = 0, binding = 3, std430) readonly buffer EAR_buf { int8_t data[]; } EAR;
layout(set = 0, binding = 4, std430) readonly buffer EBL_buf { int8_t data[]; } EBL;
layout(set = 0, binding = 5, std430) readonly buffer EBR_buf { int8_t data[]; } EBR;
layout(set = 0, binding = 6, std430) buffer Jackpot_buf { uint data[]; } jackpot;

layout(push_constant) uniform Params {
    uint M;         // rows of A
    uint N;         // cols of B
    uint K;         // common dim
    uint R;         // noise rank
    uint tile_M;    // tile rows (128)
    uint tile_N;    // tile cols (256)
    uint tile_K;    // tile inner dim (128)
} params;

// Shared memory for one A tile + one B tile
shared int8_t sh_A[128 * 128];  // tile_M × tile_K
shared int8_t sh_B[128 * 256];  // tile_K × tile_N

// Each workgroup processes one output tile: tile_M × tile_N
void main() {
    uint tid = gl_LocalInvocationIndex;  // = x + y * local_size_x
    uint warp_id  = tid / gl_SubgroupSize;  // which subgroup within workgroup
    uint lane_id  = gl_SubgroupInvocationID;

    uint tile_row = gl_WorkGroupID.y;  // tile index in M dimension
    uint tile_col = gl_WorkGroupID.x;  // tile index in N dimension

    uint row_base = tile_row * params.tile_M;
    uint col_base = tile_col * params.tile_N;

    // Per-invocation accumulator
    int32_t acc = 0;
    uint jackpot_idx = 0;  // which of 16 jackpot slots (set per K-tile)

    // Iterate over K in tile_K-sized steps
    for (uint k_base = 0; k_base < params.K; k_base += params.tile_K) {
        // Cooperatively load A tile into shared memory
        // Each thread loads one element
        // A[row][k] = A.data[row * params.K + k]
        uint a_idx_in_tile = tid;
        if (a_idx_in_tile < params.tile_M * params.tile_K) {
            uint a_row = row_base + (a_idx_in_tile / params.tile_K);
            uint a_k   = k_base + (a_idx_in_tile % params.tile_K);
            // Noised load: A_noised = A[row][k] + noise_A[row][k]
            // noise_A[row][k] = sum_{r=0..R-1} EAL[row][r] * EAR[k][r]
            //   For each r: EAL[row][r] * EAR[k][r] where EAR[k] has one +1, one -1
            //   → noise_A[row][k] = EAL[row][pos] - EAL[row][neg]
            //   where pos/neg are the sparse indices in EAR[k]
            // This is expensive to compute per element; precompute in a separate kernel
            // or compute on-the-fly with shared memory for EAR/EAL tiles.

            // Simplified: load raw A_noised from SSBO
            // (In practice, noise application is fused into the inner loop)
        }

        // Cooperatively load B tile into shared memory
        // B[k][col] = B.data[k * params.N + col]

        memoryBarrierShared();
        barrier();

        // Inner GEMM loop over tile_K
        for (uint kk = 0; kk < params.tile_K; kk++) {
            // Compute A[k][kk] * B[kk][col] and accumulate
            // Use subgroup broadcast or manual load from shared mem
        }

        memoryBarrierShared();
        barrier();

        // End of K-tile: XOR-reduce accumulator → jackpot[tid % 16]
        // XOR-reduce across the tile's accumulator using subgroup XOR
        uint xored = subgroupXor(uint(acc));
        if (lane_id == 0) {
            // rotl(jackpot[tid_rot], 13) ^ xored
            uint j_idx = (warp_id * params.tile_N / 16 /* approx */) % 16;

            // Atomic XOR with rotation into global jackpot
            // GPU atomic XOR on uint32
            // jackpot.data[j_idx] = rotl(jackpot.data[j_idx], 13) ^ xored
        }
    }

    // Note: K4 (jackpot hash + PoW check) runs as a separate kernel
    // after all workgroups complete
}
```

The actual K3 implementation needs careful design. Here is the proposed
thread mapping:

- **Workgroup:** 128 × 2 = 256 threads (4 subgroups of 64)
- **Output tile:** 128 × 256
- **Inner K-tile:** 128
- **Noise application** is fused: each thread loads A[tid] + noise_A[tid]
  where noise_A = EAL[row] dot EAR[k] for the sparse pair at k
- **Shared memory:** `tile_M × tile_K + tile_K × tile_N` = 128×128 + 128×256 = 49152 bytes
  This exceeds the typical 32KB shared memory limit — will need to tile
  further within the inner loop or use a different tile size.

Reduced tile sizes to fit shared memory:

| Tile M | Tile N | Tile K | Shared (A+B) | Notes |
|--------|--------|--------|-------------|-------|
| 64 | 128 | 128 | 64×128 + 128×128 = 24576 B | Fits 32KB |
| 64 | 64 | 128 | 64×128 + 128×64 = 16384 B | Conservative |
| 128 | 64 | 128 | 128×128 + 128×64 = 24576 B | Fits 32KB |
| 64 | 128 | 64 | 64×64 + 64×128 = 12288 B | Small |

**Recommended default:** 64×128×128 (M_tile × N_tile × K_tile) requiring
24KB shared memory.

### K4: Jackpot Hash + PoW Check

This runs as a separate dispatch with a single workgroup after K3
completes. It reads the jackpot[16] uint32 array, computes
`blake3_keyed(jackpot, commitment_hash_A)`, and compares the result
as a uint256 LE integer against the difficulty target.

```glsl
#version 460

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) buffer Jackpot { uint data[16]; } jackpot;
layout(set = 0, binding = 1, std430) buffer HashA { uint data[8]; } hash_a;
layout(set = 0, binding = 2, std430) buffer Target { uint data[8]; } target;
layout(set = 0, binding = 3, std430) buffer Result {
    uint found;    // 1 if block found
    uint tile_row; // winning tile row
    uint tile_col; // winning tile col
} result;

void main() {
    // Build message from jackpot[0..3] = first 16 bytes
    uint[16] msg;
    for (int i = 0; i < 16; i++)
        // jackpot[i] is already uint32; use as-is for message block
        // The message is 64 bytes: first 64 = jackpot[0..15] as uint32[16]
        // Actually: jackpot is uint32[16] = 64 bytes exactly
        msg[i] = jackpot.data[i];  // but only first 4 words (16 bytes) matter
                                   // according to CUDA code...
    // Need to verify the exact blake3 input format.
    // From pow_utils.hpp line ~200: 
    //   blake3_keyed(jackpot_32_bytes, commitment_hash_A)
    // Where jackpot_32_bytes is first 8 uint32 values of the 16-element array.
    // Actually re-read: Jackpot size is 16 elements, but only first 8 are used.

    uint[8] key;
    for (int i = 0; i < 8; i++)
        key[i] = hash_a.data[i];

    uint[16] message;
    for (int i = 0; i < 16; i++)
        message[i] = (i < 8) ? jackpot.data[i] : 0u;  // zero-pad to 64 bytes

    uint[8] output;
    blake3_compress_keyed(message, key, KEYED_HASH | CHUNK_START | CHUNK_END | ROOT, output);

    // Compare output as LE uint256 against target
    bit found = true;
    for (int i = 7; i >= 0; i--) {  // MSB comparison (uint256 LE)
        if (output[i] > target.data[i]) { found = false; break; }
        if (output[i] < target.data[i]) { break; }  // strictly less → found
        // if equal, continue to next word
    }

    result.found = found ? 1u : 0u;
    if (found) {
        result.tile_row = gl_WorkGroupID.x; // not available; use push constants
        result.tile_col = gl_WorkGroupID.y;
    }
}
```

**Workgroup dispatch:** 1 × 1 × 1 (single subgroup, trivial).

### Jackpot Construction Detail

From CUDA `pow_utils.hpp` lines ~200-250, the jackpot hash computation:

```
1. Start with `jackpot[16] = {0}`
2. For each K-tile iteration (every `rank` elements along K):
   a. Compute C_tile = (A + noise_A) × (B + noise_B)
   b. XOR-reduce C_tile to a single uint32 (`xored`)
   c. tid = (k_iteration / rank - 1) % 16  (which jackpot slot this tile goes to)
   d. jackpot[tid] = rotl(jackpot[tid], 13) ^ xored
3. After all K-tiles:
   jackpot_hash = blake3_keyed(jackpot[0..8] as bytes, commitment_hash_A)
4. If jackpot_hash < target:
   signal block found
```

The XOR reduction uses a 3-input XOR tree (`xor3_lop3` in CUDA) which
reduces N uint32 values to 1 in O(log_3 N) layers. In GLSL without PTX
intrinsics, use sequential XOR (`^`) which compiles to efficient ALU ops.

## Host-Side Mining Loop

### Rust Pseudocode

```rust
// src/mining.rs
pub struct MiningLoop {
    context: VulkanContext,
    pipelines: KernelPipelines,
    buffers: MiningBuffers,
    gateway: GatewayClient,
}

impl MiningLoop {
    pub async fn run(&mut self) -> Result<()> {
        loop {
            let job = self.gateway.get_job().await?;
            let job_key = blake3::hash(&[job.block_header, job.mining_config]);
            let seed = compute_seed(&job_key);

            loop {  // inner mining loop — same job, different seeds
                // 1. Record command buffer
                self.record_iteration(&job, seed, &job_key)?;

                // 2. Submit to compute queue, wait for fence
                self.context.submit(&self.cmd_buffer)?;
                self.context.wait_for_fence()?;

                // 3. Check result
                if self.buffers.block_found() {
                    let (tile_row, tile_col) = self.buffers.read_tile_coord();
                    let proof = self.build_merkle_proof(
                        &job, tile_row, tile_col, &job_key
                    );
                    self.gateway.submit_block(proof).await?;
                }

                seed = seed.wrapping_add(1);
            }
        }
    }
}
```

### Command Buffer Recording

```rust
// Pseudo-code for one iteration:
fn record_iteration(&mut self, job: &Job, seed: u64, job_key: &[u8; 32]) {
    let cmd = &self.context.command_buffer;

    cmd.begin();

    // K1: Random fill
    cmd.bind_pipeline(VK_PIPELINE_BIND_POINT_COMPUTE, &self.pipelines.k1);
    cmd.push_constants(&[job.m, job.n, job.k, seed]);
    cmd.dispatch(ceil_div(job.m * job.k, 256), 1, 1);
    cmd.pipeline_barrier(...);  // K1 writes → K2 reads

    // K2: Noise generation
    cmd.bind_pipeline(VK_PIPELINE_BIND_POINT_COMPUTE, &self.pipelines.k2);
    cmd.dispatch(noise_thread_count, 1, 1);
    cmd.pipeline_barrier(...);  // K2 writes → K3 reads

    // K3: Noised GEMM + jackpot accumulation
    cmd.bind_pipeline(VK_PIPELINE_BIND_POINT_COMPUTE, &self.pipelines.k3);
    cmd.push_constants(&[job.m, job.n, job.k, job.r,
                         tile_m, tile_n, tile_k]);
    let wg_x = ceil_div(job.n, tile_n);  // tile columns
    let wg_y = ceil_div(job.m, tile_m);  // tile rows
    cmd.dispatch(wg_x, wg_y, 1);
    cmd.pipeline_barrier(...);  // K3 writes (jackpot) → K4 reads

    // K4: Jackpot hash + PoW check
    cmd.bind_pipeline(VK_PIPELINE_BIND_POINT_COMPUTE, &self.pipelines.k4);
    cmd.dispatch(1, 1, 1);
    // No barrier needed after K4 — next iteration overwrites all buffers

    cmd.end();
}
```

## Merkle Proof Construction

After a block is found, the CPU builds a Merkle proof for the winning tile.

**Reference:** `zk-pow/src/circuit/pearl_program.rs` Merkle tree logic.

```rust
// src/proof.rs — uses pearl_blake3::MerkleTree
use pearl_blake3::MerkleTree;

pub fn build_merkle_proof(
    a_matrix: &[i8],         // CPU-side copy of A
    b_matrix: &[i8],         // CPU-side copy of B
    m: usize, k: usize, n: usize,
    tile_row: u32, tile_col: u32,
    tile_m: u32, tile_n: u32,
    rows_pattern: &[u32],    // from MiningConfiguration
    cols_pattern: &[u32],
    job_key: &[u8; 32],
) -> PlainProof {
    // Extract winning row indices in A
    let a_rows: Vec<usize> = rows_pattern.iter()
        .map(|&r| (tile_row * tile_m + r) as usize)
        .collect();
    let b_cols: Vec<usize> = cols_pattern.iter()
        .map(|&c| (tile_col * tile_n + c) as usize)
        .collect();

    // Build Merkle tree from padded A and B matrices
    let tree_a = MerkleTree::new(&a_matrix, &job_key);
    let tree_bt = MerkleTree::new(&transpose(b_matrix), &job_key);

    PlainProof {
        m,
        n,
        k,
        noise_rank: 128,  // from mining config
        a: MatrixMerkleProof {
            proof: tree_a.get_multileaf_proof(&a_rows),
            row_indices: a_rows,
        },
        bt: MatrixMerkleProof {
            proof: tree_bt.get_multileaf_proof(&b_cols),
            row_indices: b_cols,
        },
    }
}
```

Note: `MerkleTree` exists in `pearl-blake3` crate. The CPU needs a copy
of the A and B matrices to build the tree. For a 1024×1024 int8 matrix,
this is 1 MB — trivial to read back from GPU after each iteration or
recompute from seed.

**Optimization:** Instead of reading back A/B on every iteration, read
back only on block found and recompute on CPU from the same seed. Saves
PCIe bandwidth at the cost of one extra CPU BLAKE3 computation.

## Gateway RPC

```rust
// src/gateway/client.rs

pub struct GatewayClient {
    socket: tokio::net::UnixStream,
}

impl GatewayClient {
    pub async fn connect(path: &str) -> Result<Self> {
        let socket = tokio::net::UnixStream::connect(path).await?;
        Ok(Self { socket })
    }

    pub async fn get_job(&mut self) -> Result<MiningJob> {
        // Send GET_JOB request
        // Receive IncompleteBlockHeader + MiningConfiguration
        // (Fixed 52-byte mining config struct)
    }

    pub async fn submit_block(&mut self, proof: PlainProof) -> Result<()> {
        let bytes = bincode::serialize(&proof)?;
        let b64 = base64::encode(&bytes);
        // POST bytes to gateway
    }
}
```

## SPIR-V Compilation (`build.rs`)

```rust
// build.rs
fn main() {
    let shader_dir = std::path::Path::new("shaders");
    let out_dir = std::env::var("OUT_DIR").unwrap();

    for entry in std::fs::read_dir(shader_dir).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        if path.extension().map_or(false, |e| e == "comp") {
            let out_path = std::path::Path::new(&out_dir)
                .join(path.file_stem().unwrap())
                .with_extension("spv");
            let status = std::process::Command::new("glslc")
                .arg("-fshader-stage=compute")
                .arg(&path)
                .arg("-o")
                .arg(&out_path)
                .status()
                .expect("glslc not found; install Vulkan SDK or shaderc");
            assert!(status.success(), "Failed to compile {:?}", path);

            // Rerun if shader changes
            println!("cargo:rerun-if-changed={}", path.display());
        }
    }
}
```

## Testing Strategy

### Test 1: BLAKE3 GLSL → CPU comparison

Compile a minimal GLSL shader that runs BLAKE3 on known test vectors.
Dispatch with a single workgroup and read back the output. Compare
against the Rust `blake3` crate output.

**Test vectors** (from BLAKE3 spec):
- Input: empty, expected: `AF1349B9...`
- Input: 64 zero bytes, expected: `8EE9E4E3...`

### Test 2: Noise generation comparison

Run K2 with the same inputs as the Rust `pearl_noise.rs` reference.
Read back output matrices and compare byte-for-byte.

### Test 3: Full pipeline comparison

Run K1+K2+K3+K4 with fixed seed. Read back jackpot array and hash.
Compare against Rust CPU reference `mine.rs`.

### Test 4: End-to-end PoW match

Run full mining loop with a very easy target (so a block is found
quickly), read back the tile coordinates and proof data, compare
against reference CPU miner's output.

## Performance Considerations

### Shared Memory
- Default tile size 64×128×128 = 24KB shared memory.
- Need to check `maxComputeSharedMemorySize` (usually 32KB or 48KB).
- Could split the N-tile further (e.g., 64×64×128 = 16KB).

### Subgroup Utilization
- Subgroup size varies (32 on NVIDIA, 64 on AMD, 32/64 on Intel).
- Workgroup size must be a multiple of subgroup size.
- Use `computeFullSubgroups = VK_TRUE` to ensure full subgroups.

### Memory Bandwidth
- Each iteration: read ~2.6 MB (A,B,noise matrices), write ~64B (jackpot).
- On a PCIe 4.0 ×16 link (32 GB/s), transfer time is negligible.
- All buffers are device-local — no host reads until block found.

### Instruction Throughput
- The GEMM inner loop is int8 multiply + int32 accumulate + add + XOR.
- Expect ~60-80% of peak int8 TOPS on AMD RDNA3, ~40-50% on Intel ARC.
- Much lower than CUDA (which hits ~90% of SM90 peak), but acceptable.

## Open Implementation Questions

1. **BLAKE3 in GLSL: array size limits.** GLSL does not support
   variable-length arrays as function arguments. Need to use fixed-size
   arrays (`uint[16]`, `uint[8]`) or `uvec4`×4 for block. The macro-based
   approach from `blake3.cuh` (BLAKE3_ROUND, BLAKE3_PERMUTE) avoids
   function call overhead and sidesteps GLSL restrictions.

2. **int8 SSBO support.** `VK_KHR_8bit_storage` with `storageBuffer8BitAccess`
   allows `int8_t[]` in SSBOs, but some drivers (early Intel, AMD pre-RDNA3)
   may not support it. Fallback: pack 4 int8 into 1 uint32 per SSBO element.

3. **Jackpot slot indexing.** The CUDA code uses thread index modulo 16
   to determine which of 16 jackpot slots a tile feeds into. Need to map
   workgroup+subgroup+thread IDs to the same 16-way partitioning.

4. **Noise fusing.** The CUDA GEMM kernel preloads noise matrices via
   shared memory and fuses noise application into the inner loop. The
   same pattern works in GLSL, but the extra shared memory pressure may
   require smaller tile sizes.

5. **Sparse noise on-the-fly.** EAR/EBL have exactly one +1 and one -1
   per row. Instead of loading the full EAR/EBL matrices, compute the
   noise contribution from just two int8 values per K-row: one positive
   and one negative index from the collision hash.

6. **glslc availability.** `build.rs` assumes `glslc` is in PATH. Should
   add a `--glslc-path` environment variable fallback.

## File Map

| File | Purpose |
|------|---------|
| `src/main.rs` | CLI args, init context, start mining loop |
| `src/context.rs` | `VulkanContext` — instance, device, queues, command pool |
| `src/buffers.rs` | `MiningBuffers` — all SSBO allocation and binding |
| `src/pipelines.rs` | `KernelPipelines` — 4 compute pipelines + pipeline layout |
| `src/mining.rs` | `MiningLoop` — job polling, command buffer recording, dispatch |
| `src/proof.rs` | `build_merkle_proof()` — block submission |
| `src/gateway/client.rs` | `GatewayClient` — Unix socket RPC |
| `shaders/k1_random_fill.comp` | K1: random A/B fill |
| `shaders/k2_noise_gen.comp` | K2: BLAKE3-based noise generation |
| `shaders/k3_noised_gemm.comp` | K3: tiled noised GEMM + XOR jackpot |
| `shaders/common/blake3.glsl` | Shared BLAKE3 compression function header |
| `build.rs` | glslc SPIR-V compilation |
| `Cargo.toml` | Dependencies |

## Reference CUDA-to-Vulkan Mapping

| CUDA Concept | Vulkan Equivalent |
|---|---|
| `__global__` kernel | GLSL `void main()` with `layout(local_size_...)` |
| `__device__` function | GLSL function |
| `__shared__` memory | `shared` storage class in GLSL |
| `__constant__` memory | `const` or specialization constants |
| CUDA thread | GLSL invocation |
| CUDA warp | Vulkan subgroup (`gl_SubgroupSize`) |
| `__syncthreads()` | `barrier()` + `memoryBarrierShared()` |
| `atomicAdd` | `atomicAdd` (Vulkan 1.3 core) |
| CUDA stream | `VkQueue` + separate command buffer |
| `cudaMemcpy` | `vkCmdCopyBuffer` |
| `cudaMalloc` | `vkCreateBuffer` + `vkAllocateMemory` |
| CUTLASS/WGMMA | Manual tiled matmul with shared memory |
| TMA (Tensor Memory Accelerator) | Not available; use explicit `shared` loads |
| `lop3.b32` XOR3 | Regular `^` XOR operator |

## Dependencies

| Crate | Version | Purpose |
|-------|---------|---------|
| `ash` | 0.38+ | Raw Vulkan bindings |
| `blake3` | 1.5 | CPU-side BLAKE3 for job_key, commitment hash |
| `pearl-blake3` | workspace | Merkle tree construction for proofs |
| `zk-pow-types` | workspace | PlainProof, PublicProofParams, MiningConfiguration |
| `tokio` | 1 | Async I/O for gateway RPC |
| `serde` + `bincode` | stable | Proof serialization |
| `tracing` | 0.1 | Logging |
| `clap` | 4 | CLI argument parsing |
| `gpu-allocator` | 0.25 | Optional: Vulkan memory allocation |

## Directory Layout (under `miner/`)

```
miner/
├── docs/
│   ├── requirements/
│   │   └── REQ-VULKAN-MINER.md
│   └── specifications/
│       └── SPEC-VULKAN-MINER.md      ← this file
├── vulkan-miner/                     ← new Rust crate
│   ├── Cargo.toml
│   ├── build.rs
│   ├── src/
│   └── shaders/
├── RESEARCH-VULKAN.md
├── RESEARCH-OPENCL.md
└── RESEARCH-PROCESS.md
```
