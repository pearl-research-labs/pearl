# RESEARCH-VULKAN: Porting Pearl Miner to Vulkan Compute

## Purpose
Document findings, decisions, and progress for porting the pearl-gemm CUDA miner
to Vulkan compute shaders. Updated iteratively as research progresses.

---

## 2026-05-22: Initial Assessment

### What We're Porting

The pearl-gemm CUDA implementation (`miner/pearl-gemm/csrc/`) consists of:

| Component | CUDA Files | Vulkan Equivalent |
|-----------|-----------|-------------------|
| GEMM kernel (noisy/vanilla) | `gemm/pearl_gemm_kernel.h`, `gemm/collective_mainloop.hpp`, `gemm/collective_epilogue.hpp` | `VK_PIPELINE_BIND_POINT_COMPUTE` with GLSL compute shaders |
| Noising A kernel | `gemm/pearl_noisingA_kernel.h` | Custom compute shader + SSBO |
| Noise generation (BLAKE3-based) | `gemm/noise_generation_kernel.h` | BLAKE3 in GLSL compute shader |
| Denoise converter | `gemm/denoise_converter_kernel.h` | Simple element-wise compute shader |
| Tensor hash (BLAKE3 + Merkle tree) | `tensor_hash/tensor_hash_host.hpp` | Multi-stage compute pipeline |
| Quantization kernel | `gemm/quantize_kernel.hpp` | Element-wise compute shader |
| Inner hash kernel | `gemm/inner_hash_kernel.cu` | Compute shader |
| BLAKE3 implementation | `blake3/blake3.cuh` | GLSL port or SPIR-V |

### CUDA Features Used (and Vulkan Counterparts)

| CUDA Feature | Usage | Vulkan Equivalent |
|-------------|-------|-------------------|
| TMA (Tensor Memory Accelerator) | Async global->shared loads | `VK_EXT_device_generated_commands` or manual SSBO loads |
| WGMMA (Warp Group MMA) | int8 matmul on Hopper | Manual `gl_WorkGroupID` tiled matmul with `shared` memory |
| Shared memory (SMEM) | Pipeline buffers, tile storage | `shared` keyword in GLSL compute (max ~48KB, device-dependent) |
| CUDA streams | Async kernel launches | `VkCommandBuffer` + `VkQueue` |
| CUTLASS library | GEMM template infrastructure | No direct equivalent - must implement or use Vulkan GLSL math |
| CUDA dynamic parallelism | Not used currently | N/A |
| cuBLAS | Not used directly | N/A |

### Key Challenges for Vulkan Port

1. **No equivalent to CUTLASS/CuTe** - The entire tiled GEMM infrastructure must be
   reimplemented in GLSL compute shaders, including tile scheduling, shared memory
   layouts, and warp-level matrix operations.

2. **No TMA (Tensor Memory Accelerator)** - Vulkan has no direct equivalent to
   NVIDIA's TMA for async global->shared copies. Must use plain `shared` memory
   with manual `barrier()` for thread synchronization.

3. **No WGMMA** - Vulkan has no warp-group-level matrix multiply-accumulate.
   Matrix multiplication must be done at the invocation level with explicit
   loop nests over tiles.

4. **Subgroup operations** - Vulkan supports subgroups (`GL_KHR_shader_subgroup`),
   which can partially replace CUDA warp intrinsics, but WG-level (warp-group)
   operations are not available.

5. **SPIR-V vs GLSL** - Compute shaders must compile to SPIR-V. GLSL is the
   primary authoring language, but inline SPIR-V assembly is possible for
   critical paths.

6. **PyTorch integration** - Current integration uses PyTorch CUDA custom ops.
   Vulkan compute kernels would need a different integration path (e.g.,
   `torch.backends.vulkan`, or custom buffer management via `VkBuffer`).

### Vulkan Extensions to Investigate

- `VK_KHR_shader_float16_int8` - int8/fp16 support in compute shaders
- `VK_KHR_shader_subgroup_extended_types` - subgroup ops for int8
- `VK_KHR_cooperative_matrix` - cooperative matrix operations (if available)
- `VK_KHR_8bit_storage` - 8-bit buffer access
- `VK_KHR_buffer_device_address` - pointer-like buffer access

### Initial Porting Strategy (Tentative)

```
Phase 1: No-op Vulkan compute pipeline skeleton
  - Create VkDevice, VkCommandPool, VkQueue
  - Implement trivial "copy" compute shader
  - Benchmark kernel launch overhead vs CUDA

Phase 2: Port quantization kernel
  - Simplest kernel (element-wise)
  - Proves compute shader pipeline works end-to-end

Phase 3: Port BLAKE3 hash kernel
  - Straightforward GLSL implementation (no CUDA-specific features)
  - Used by noise_gen and tensor_hash

Phase 4: Port noise generation
  - Uses BLAKE3; verify correctness

Phase 5: Port GEMM (noisy and vanilla)
  - Most complex component
  - Implement tiled matmul with shared memory
  - Tile scheduler in GLSL
  - Denoising epilogue

Phase 6: Tensor hash / Merkle tree
  - Multi-kernel pipeline
  - Complex shared memory patterns
```

### Open Questions

1. Can we use `VK_KHR_cooperative_matrix` to replace WGMMA?
   - Only available on select hardware (Android, some ARM GPUs)
   - Desktop support is limited

2. Is Vulkan's `shared` memory large enough for our tile sizes?
   - Typical max: 16KB-48KB per workgroup
   - CUDA usage: ~64KB+ per block (Hopper)
   - May need smaller tiles

3. Can we achieve competitive performance without TMA?
   - Manual global->shared copies add instruction overhead
   - TMA is hardware-accelerated on Hopper

---

## Iteration Log

| Date | Entry |
|------|-------|
| 2026-05-22 | Initial assessment created. GEMM kernel identified as most complex port. |
| 2026-05-23 | **Major simplification.** Code audit revealed the vLLM integration is optional. The reference Rust CPU miner (`zk-pow/src/ffi/mine.rs`) is a pure PoW loop — no LLM, no denoising, no inference. Stage 1 is now a standalone GPU miner with K1-K4 kernels only. LLM integration deferred to Stage 2. See `VULKAN-REQUIREMENTS.md` for full spec. |

### Key Docs
- **REQ:** `docs/requirements/REQ-VULKAN-MINER.md` — pure requirements (what)
- **SPEC:** `docs/specifications/SPEC-VULKAN-MINER.md` — implementation (how)

### Next Actions
- [x] Code audit complete — confirmed algorithm is self-contained PoW
- [x] REQ/SPEC split complete: requirements → pure REQ, implementation → SPEC
- [x] Vulkan/GLSL/BLAKE3 research complete
- [ ] **Phase 1:** Implement BLAKE3 GLSL header and verify against test vectors
- [ ] **Phase 2:** Implement K1 random fill shader
- [ ] **Phase 3:** Implement K2 noise gen shader (depends on Phase 1)
- [ ] **Phase 4:** Implement K3 noised GEMM shader (most complex)
- [ ] **Phase 5:** Implement K4 jackpot hash shader (depends on Phase 1)
- [ ] **Phase 6:** Wire up Vulkan API (instance, device, pipelines, dispatch)
- [ ] **Phase 7:** Wire up host mining loop + gateway RPC
- [ ] **Phase 8:** Integration testing vs Rust CPU reference
- [ ] **Phase 9:** Benchmark and optimize tile sizes
