# Requirements: Vulkan Standalone Pearl Miner

**Date:** 2026-05-23
**Status:** DRAFT
**Type:** Requirements
**Specification:** [SPEC-VULKAN-MINER.md](../specifications/SPEC-VULKAN-MINER.md)

> **Implementation status:** REQ only. No code.

## Purpose

Port the pearl-gemm PoW miner from CUDA (Hopper SM90, CUTLASS) to Vulkan
compute, enabling GPU-agnostic mining on AMD, Intel, older NVIDIA, and Apple
Silicon (via MoltenVK). The resulting miner must produce identical PoW output
to the reference Rust CPU miner (`zk-pow/src/ffi/mine.rs`).

## Design Principles

1. **No LLM dependency.** Stage 1 is a standalone GPU miner — no vLLM, no
   denoising, no LLM inference. The mining algorithm is a self-contained PoW
   loop: generate random matrices → noise → GEMM → hash → check target.

2. **Bit-exact correctness.** Every Vulkan kernel output must match the
   reference Rust CPU miner byte-for-byte. Floating-point divergence is not
   permitted (all arithmetic is integer).

3. **One submission model.** The miner connects to the pearl gateway RPC
   (`pearl-gateway`), receives block headers with difficulty targets, and
   submits found blocks as `PlainProof` (see `zk-pow/src/ffi/plain_proof.rs`).

4. **Vulkan 1.3 minimum.** Required extensions may extend the surface but the
   baseline is `VK_API_VERSION_1_3`.

## Scope

### In Scope

- Vulkan compute pipeline for the mining algorithm: matrix generation, noise
  generation, noisy GEMM, hash accumulation, jackpot hash, PoW target check.
- Host-side mining loop connecting to the pearl gateway RPC.
- Block submission with Merkle proof construction from GPU-generated data.
- Support for configurable matrix dimensions (m, n, k) and noise rank (r).
- Single-GPU mining only.

### Out of Scope

- ZK proof generation (node-side, not miner-side).
- Denoising / LLM tensor recovery (Stage 2).
- Multi-GPU mining.
- PyTorch vLLM integration (Stage 2).
- WebGPU or OpenCL backends (Vulkan only for Stage 1).

## Functional Requirements

### FR-1: Vulkan Compute Pipeline

The system SHALL implement a GPU compute pipeline consisting of at least the
following stages, executable without CPU intermediate data transfer between
kernel dispatches:

1. **FR-1.1** — Generate random int8 matrices A (m×k) and B (k×n) on the GPU,
   with values uniformly distributed in [-64, +63].

2. **FR-1.2** — Generate noise matrices EAL (m×r), EAR (k×r), EBL (k×r),
   EBR (n×r) on the GPU using the BLAKE3-based noise generation algorithm
   from `zk-pow/src/circuit/pearl_noise.rs`. The noise MUST include both:
   - Dense uniform noise (EAL, EBR): each byte from BLAKE3 output mapped to
     [-32, +32).
   - Sparse noise (EAR, EBL): each row has exactly one +1 and one -1, rest 0.

3. **FR-1.3** — Compute the noised GEMM: C = (A+N_A) · (B+N_B), where
   N_A = EAL·EAR<sup>T</sup> and N_B = EBL·EBR<sup>T</sup>, using int8
   multiplication with int32 accumulation.

4. **FR-1.4** — For each GEMM tile, XOR-reduce the tile's accumulator to a
   single uint32 value and accumulate into a 16-element jackpot array using
   rotate-xor mixing: `jackpot[tid] = rotl(jackpot[tid], 13) ^ xored`.

5. **FR-1.5** — Compute the jackpot hash:
   `blake3_keyed(jackpot_16_bytes, commitment_hash_A)`. Compare the result as
   a uint256 little-endian integer against the difficulty target. If the hash
   is less than the target, signal a block-found event.

### FR-2: Host Mining Loop

The host SHALL:

- **FR-2.1** — Poll the pearl gateway for mining jobs (block headers, difficulty
  targets, mining configurations).
- **FR-2.2** — For each job, orchestrate GPU execution of FR-1.1 through FR-1.5
  using Vulkan command buffers.
- **FR-2.3** — On block-found event: read back the winning tile coordinates
  from GPU memory, reconstruct the winning row/column data on CPU, build a
  Merkle proof using `pearl_blake3::MerkleTree`, assemble a `PlainProof`, and
  submit it to the gateway.
- **FR-2.4** — Repeat the loop indefinitely; the miner has no termination
  condition (daemon).

### FR-3: Block Submission

The submitted `PlainProof` SHALL contain (see `zk-pow/src/ffi/plain_proof.rs`):

- Matrix dimensions: m, n, k, noise_rank.
- Merkle proof for the winning A rows: row indices and Merkle siblings.
- Merkle proof for the winning B columns: column indices and Merkle siblings.
- Commitment hashes and jackpot hash (derived from `PublicProofParams`).

### FR-4: Buffer Management

The system SHALL manage GPU memory for the following buffers:

- A and B matrices (m×k and k×n, int8).
- Noise matrices EAL, EAR, EBL, EBR (int8).
- Jackpot accumulator array (16 uint32).
- Tile coordinate output buffer (2 uint32 for block-found signal).

Buffer sizes SHALL be computed from the active mining job's dimensions and
allocated/reallocated when dimensions change.

### FR-5: Gateway RPC

- **FR-5.1** — Connect to the pearl gateway via Unix domain socket at
  `/tmp/pearlgw.sock`.
- **FR-5.2** — Receive jobs as `IncompleteBlockHeader` + `MiningConfiguration`
  (see `zk-pow/src/api/proof.rs`).
- **FR-5.3** — Submit `PlainProof` serialized via bincode → base64 → POST.

### FR-6: Configuration

The miner SHALL support the following configurable parameters:

- Matrix dimensions: m (A rows, default 1024), n (B cols, default 1024),
  k (common dim, default 1024).
- Noise rank r (default 64 or 128).
- GPU device selection (index or UUID).
- Tile dimensions for the GEMM kernel (defaults matching CUDA: 128×256 with
  K-tile 128).

## Non-Functional Requirements

### NFR-1: Correctness

The jackpot hash output SHALL match the Rust CPU reference miner output for
all equivalent inputs. The system SHALL include a test harness that compares
GPU output against CPU reference output for small matrix sizes.

### NFR-2: Performance

The miner SHALL complete one full mining iteration (fill A/B → noise → GEMM →
jackpot hash → PoW check) for 1024×1024×1024 matrices at rank 128 in under
10 seconds on a desktop-class GPU (e.g., AMD Radeon RX 7900 XTX or
NVIDIA RTX 4090).

### NFR-3: Memory

Total GPU memory usage SHALL NOT exceed 32 MB for a 1024×1024×1024 mining job
with rank 128.

### NFR-4: Portability

The miner SHALL run on any Vulkan 1.3-capable GPU without vendor-specific
extensions. If optional extensions (e.g., `VK_KHR_cooperative_matrix`) are
unavailable, the implementation MUST fall back to a software-path equivalent
that produces identical results.

## Cross-References

- [SPEC-VULKAN-MINER.md](../specifications/SPEC-VULKAN-MINER.md) — Implementation specification.
- [zk-pow/src/ffi/mine.rs](../../../zk-pow/src/ffi/mine.rs) — Reference Rust CPU miner.
- [zk-pow/src/circuit/pearl_noise.rs](../../../zk-pow/src/circuit/pearl_noise.rs) — Noise generation algorithm.
- [zk-pow/src/circuit/pearl_program.rs](../../../zk-pow/src/circuit/pearl_program.rs) — Jackpot hash constants.
- [zk-pow/src/ffi/plain_proof.rs](../../../zk-pow/src/ffi/plain_proof.rs) — Block submission format.
- [zk-pow/src/api/proof.rs](../../../zk-pow/src/api/proof.rs) — MiningConfiguration, block header types.
- [miner/pearl-gateway](../../pearl-gateway/) — Gateway RPC implementation.
- [RESEARCH-VULKAN.md](../../RESEARCH-VULKAN.md) — Earlier Vulkan research.
