# RESEARCH-OPENCL: Porting Pearl Miner to OpenCL

## Purpose
Document findings, decisions, and progress for porting the pearl-gemm CUDA miner
to OpenCL. Updated iteratively as research progresses.

---

## 2026-05-22: Initial Assessment

### What We're Porting

Same components as RESEARCH-VULKAN.md. The pearl-gemm CUDA implementation
(`miner/pearl-gemm/csrc/`) must be ported kernel-by-kernel.

### CUDA Features Used (and OpenCL Counterparts)

| CUDA Feature | Usage | OpenCL Equivalent |
|-------------|-------|-------------------|
| Thread blocks + warps | Parallel hierarchy | `work-group` + `work-item` (`get_local_id()`, `get_local_size()`) |
| Shared memory | Per-block fast memory | `__local` memory qualifier |
| Global memory | Device-wide buffers | `__global` memory qualifier |
| CUDA streams | Async execution | `cl_command_queue` (in-order or out-of-order) |
| CUTLASS/CuTe | Tiled GEMM templates | Must implement manually in OpenCL C/C++ |
| TMA (Tensor Memory Accelerator) | Async global->shared | `async_work_group_copy()` or manual loops |
| WGMMA | Warp-group matmul | Subgroups (`cl_khr_subgroups`) + manual matmul |
| BLAKE3 intrinsics | Rotations, additions | Standard C operators (already infrastructure) |
| cuBLAS | Not used directly | N/A |

### OpenCL Strengths for This Project

1. **Simpler API than Vulkan** - Much less verbose setup code. Kernels are
   written in OpenCL C (C99-based) which is easier to port from CUDA C++.

2. **Vendor portability** - OpenCL 3.0 runs on NVIDIA, AMD, Intel, ARM, and
   Qualcomm GPUs. The current code is NVIDIA-only (requires Hopper SM90).

3. **Existing BLAKE3 in OpenCL** - BLAKE3 hash can be written in pure OpenCL C
   without vendor extensions.

4. **Shared memory semantics** - OpenCL `__local` maps closely to CUDA `__shared__`.
   Pattern: declare `__local` buffer, manually copy from `__global`, `barrier()`.

5. **Subgroups (optional)** - `cl_khr_subgroups` extension provides warp-level
   operations similar to CUDA.

6. **SPIR-V ingestion** (OpenCL 3.1) - Kernels can be compiled offline to SPIR-V
   for faster loading. Shared same IR with Vulkan.

### OpenCL Weaknesses for This Project

1. **No CUTLASS equivalent** - Must implement all tile scheduling and matrix
   multiplication from scratch in OpenCL C.

2. **No hardware matmul acceleration via standard API** - NVIDIA GPUs have
   tensor cores, but OpenCL doesn't expose them in the standard API. Some
   extensions exist but aren't portable.

3. **Limited cooperative matrix support** - OpenCL 3.0 has
   `cl_khr_cooperative_matrix` but support is sparse across vendors.

4. **Smaller ecosystem** - Fewer tools, fewer examples, less community support
   than CUDA or Vulkan compute.

5. **PyTorch integration is weaker** - PyTorch's OpenCL backend is less mature
   than CUDA or Vulkan backends. May need custom `torch.Tensor` buffer sharing.

6. **Performance portability is hard** - A kernel optimized for NVIDIA might
   not perform well on AMD or Intel GPUs without significant tuning.

### OpenCL Kernel Port Mapping

```c
// CUDA kernel pattern
__global__ void quantize_kernel(...) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // ... kernel body using __shared__ memory
}

// OpenCL equivalent
__kernel void quantize_kernel(__global ...) {
    int idx = get_global_id(0);
    __local int shared_buf[256];
    // ... same kernel body
    barrier(CLK_LOCAL_MEM_FENCE);
}
```

### Initial Porting Strategy (Tentative)

```
Phase 1: OpenCL runtime setup
  - Platform, device, context, command queue
  - Buffer allocation from clCreateBuffer
  - Kernel compilation from source/SPIR-V
  - Benchmark launch overhead vs CUDA

Phase 2: Port quantization kernel
  - Simplest kernel, proves pipeline works
  - Compare performance to CUDA

Phase 3: Port BLAKE3 hash kernel
  - Pure OpenCL C implementation
  - Only uses standard operators (add, xor, rotate)
  - Test against CUDA BLAKE3 output

Phase 4: Port noise generation kernel
  - Uses BLAKE3 + indexed writes
  - Verify deterministic output matches CUDA

Phase 5: Port GEMM (noisy and vanilla)
  - Tiled implementation using __local memory
  - Tile scheduler logic
  - Denoising epilogue
  - This is the hardest part

Phase 6: Tensor hash / Merkle tree
  - Multi-stage kernel pipeline
  - Synchronization between stages
```

### Comparing Vulkan vs OpenCL for This Project

| Factor | Vulkan | OpenCL |
|--------|--------|--------|
| API complexity | Very high (verbose) | Moderate |
| Kernel language | GLSL/SPIR-V | OpenCL C / C++ for OpenCL |
| CUDA migration ease | Harder (different paradigm) | Easier (similar concepts) |
| Hardware support | Very wide (all GPUs) | Wide (all GPUs + CPUs + FPGAs) |
| PyTorch integration | Experimental backend | Community effort |
| Mature compute ecosystem | Smaller (gaming focus) | Larger (HPC focus) |
| Shared memory | `shared` keyword | `__local` qualifier |
| Subgroup/warp ops | `GL_KHR_shader_subgroup` | `cl_khr_subgroups` |
| Cooperative matmul | `VK_KHR_cooperative_matrix` | `cl_khr_cooperative_matrix` |

### Preliminary Recommendation

**Start with OpenCL** for the following reasons:
1. Much simpler API reduces porting time
2. OpenCL C maps more directly from CUDA C++ than GLSL compute
3. Same BLAKE3 code can compile for both (pure standard C)
4. Shared memory semantics are nearly identical
5. Can later layer on Vulkan via CLVK if desired

**Keep Vulkan as a future option** for:
1. Wider hardware support (especially mobile/integrated GPUs)
2. Tighter integration with graphics pipeline if needed
3. Potentially lower driver overhead on some platforms

---

## Iteration Log

| Date | Entry |
|------|-------|
| 2026-05-22 | Initial assessment created. OpenCL recommended as initial target. |

### Next Actions
- [ ] Check system for OpenCL ICD (install if needed): `sudo apt install opencl-headers ocl-icd-opencl-dev`
- [ ] Install `clinfo` to verify GPU device support
- [ ] Test BLAKE3 in OpenCL C with reference test vectors
- [ ] Prototype quantization kernel in OpenCL and compare output
- [ ] Benchmark shared memory bandwidth vs CUDA equivalent
