# pearl-gemm sm_89 Port Spec — Integrated 20-Agent Output

Working directory: `C:/Source/pearl/miner/pearl-gemm/`
Target: RTX 4070 Ti SUPER (sm_89, Ada Lovelace, AD103)
Source state: CUTLASS 3.x kernel hardcoded to `arch=compute_90a,code=sm_90a` in [setup.py:88](miner/pearl-gemm/setup.py:88), `get_min_capability() == 9` gate at [vllm_kernels.py:70](miner/vllm-miner/src/vllm_miner/vllm_kernels.py:70).

---

## 0. THE LOAD-BEARING CONTRADICTION — read first

The 20 agents converged on the codebase structure, the substitution mapping, and the port plan, but **two of them disagree by ~6800× on whether 75 TH/s is reachable.** This is not a minor calibration disagreement; it's a fundamentally different reading of what "TH/s" means, and the answer dictates whether the port is worth doing.

### Agent D14 (perf model, **NOT REACHABLE**)
- `inner_hash_count` formula from [tests/test_pearl_gemm.py:1116-1135](miner/pearl-gemm/tests/test_pearl_gemm.py): `ceil(M/tile_m) × ceil(N/tile_n) × (K/R) × num_threads_per_cta`
- For default tile `128×256×128`, R=128, num_threads_per_cta = 128
- At 4070 Ti SUPER peak int8 (353 TOPS dense), every tile = 4.19M MACs → max 84M tiles/sec → max **~10.7 × 10⁹ inner-hashes/sec = 0.011 TH/s**
- 75 TH/s would require **~6800× peak hardware** → physically impossible

### Agent E20 (metric definition, **REACHABLE**)
- Same `inner_hash_count` formula
- But asserts "175 TOPS ÷ 50 cycles/hash = 3.5 trillion hashes/sec" → 75 TH/s = 21% utilization → easy
- The 50 cycles/hash figure has **no source in the code**. It treats inner-hash as a standalone unit decoupled from MMA work.

### Resolution
**D14 is correct.** The inner-hash is not a free-standing kernel — it's a XOR-reduction *fused into the GEMM mainloop* ([pow_utils.hpp:171-180](miner/pearl-gemm/csrc/gemm/pow_utils.hpp)), consuming MMA accumulator fragments. There is a hard ratio of ~32,000 MACs per inner hash. You cannot do more inner hashes per second than (peak MACs/sec) ÷ (MACs/hash). Sm_89 peak ÷ 32K = ~10¹⁰ hashes/sec. 75 × 10¹² is unreachable by ~3 orders of magnitude.

### So what *can* a 4070 Ti SUPER hit?

| Metric | Sm_89 ceiling (well-tuned port) |
|---|---|
| Int8 TOPS (raw GEMM ops/sec, what some marketing calls "TH/s") | ~250 TOPS = **0.25 "PTOps/s"**, or 250 GTOPS, at ~71% of dense peak |
| Inner hashes/sec (D14's reading) | ~10¹⁰/sec = **~10 GH/s**, or **0.01 TH/s** |
| Tile commits/sec | ~6 × 10⁷/sec = **60 MH/s**, or 6 × 10⁻⁵ TH/s |

The 75 figure looks like it came from a session that conflated **TH/s with TOPS** (treating each MAC as a "hash" — a marketing trick, not the codebase's definition). If 75 *GH/s* (10⁹) was meant, that's ~7× peak still — also impossible. If 75 *TOPS* was meant, that's reachable.

**Recommendation:** before any code lands, confirm which metric the original prediction referred to. If it was "inner hashes per second", the target is unreachable on this hardware and the port should be sized by other goals (parity with sm_90, energy/$, throughput-per-rig, etc.). If it was "75 TOPS effective on the noisy-gemm path", that's a reasonable port target.

---

## 1. Repo anatomy (agents B5, B6, B7, B8 consolidated)

```
pearl/miner/pearl-gemm/
├── setup.py                         # ★ arch hardcoded to sm_90a (line 88)
├── third_party/cutlass/             # submodule, ~3.5+ — has sm_80 builders
└── csrc/
    ├── gemm/
    │   ├── pearl_gemm_api.cpp       # PyBind: gemm(), noisy_gemm(), noise_A(), noise_B(), …
    │   ├── pearl_gemm_kernel.h      # ★ __global__ hopper_gemm_ws (sm_90a)
    │   ├── pearl_gemm_launch_template.h # ★ launch_kernel_on_cluster (sm_90a)
    │   ├── pearl_gemm_host.h        # ★ ClusterShape, num_clusters_m/n (sm_90a)
    │   ├── kernel_traits.hpp        # ★★ ALL sm_90 substrate
    │   ├── collective_mainloop.hpp  # ★★ producer-warp TMA / WGMMA pipeline (sm_90a)
    │   ├── collective_epilogue.hpp  # ★★ TMA store, denoise WGMMA (sm_90a)
    │   ├── named_barrier.hpp        # OK on sm_89 (bar.sync N,M; ≤16 IDs)
    │   ├── pow_utils.hpp            # OK on sm_89 (lop3, shf — both sm_50+)
    │   ├── inner_hash_kernel.cu     # OK on sm_89 (single-thread microbench)
    │   ├── noise_generation.cu      # OK on sm_89 (BLAKE3, no MMA)
    │   ├── pearl_noisingA_kernel.h  # ★ ArchTag=Sm90, TMA+WGMMA (rewrite needed)
    │   ├── pearl_noisingB_kernel.h  # ★ ArchTag=Sm90, TMA+WGMMA (rewrite needed)
    │   ├── denoise_converter.cu     # mostly OK; int32→fp16 cast
    │   └── quantize_kernel.cu       # OK on sm_89 (cvt.rni.sat.s8.f32 PTX)
    ├── blake3/                      # OK on sm_89 (no sm_90 intrinsics)
    └── tensor_hash/
        └── merkle_tree_roots_kernel.hpp  # ★ ArchTag=Sm90, TMA load (rewrite needed)
```

★ = Hopper-specific, needs port. ★★ = central rewrite.

Inventory totals (agents A1, A2): ~90 sm_90-specific construct occurrences. Breakdown:
- TMA atoms (`SM90_TMA_LOAD/STORE/LOAD_MULTICAST/REDUCE_ADD`): **30+**
- GMMA (`GMMA::ss_op_selector`): **8**
- Cluster + named barriers: **20+**
- `warpgroup_reg_alloc/dealloc` (`setmaxnreg`): **6**
- `SM90_U32x4_STSM_N` STSM: 3 (works on sm_89 — the PTX `stmatrix.m8n8.x4` is sm_75+, the CUTE name is a misnomer)
- `PipelineTmaAsync`: 5

---

## 2. Sm_89 substitution map (agents A1, A2, C9, C10, C11, C12, C13 consolidated)

| Hopper construct | Sm_89 substitute | Confidence | Source |
|---|---|---|---|
| `SM90_TMA_LOAD[_MULTICAST]` | `SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>` + manual `cp.async` loop | HIGH | A2, C9 |
| `SM90_TMA_STORE` | `AutoVectorizingCopyWithAssumedAlignment<128>` (plain `st.global.v4`) | HIGH | A2, C10 |
| `GMMA::ss_op_selector` int8 | `MMA_Atom<SM80_16x8x32_S32S8S8S32_TN>` | HIGH | C9, C12 |
| `GMMA::ss_op_selector` fp16 (denoise) | `MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>` | HIGH | C10, C12 |
| `cutlass::PipelineTmaAsync<N>` | `cutlass::PipelineAsync<N>` | HIGH | A2 |
| `ss_smem_selector<GMMA::Major::K, int8>` | `composition(Swizzle<3,4,3>, Layout<Shape<_8,_128>,Stride<_128,_1>>)` | HIGH | C12 |
| `ss_smem_selector<GMMA::Major::K, fp16/bf16>` | `composition(Swizzle<3,3,3>, ...)` | HIGH | C12 |
| `cluster_arrive_relaxed()` / `cluster_wait()` | `__syncthreads()` | HIGH | A3 |
| `NamedBarrier::sync(N, id)` | unchanged — `cutlass::arch::NamedBarrier` works on sm_80+ | HIGH | C11 |
| `warpgroup_reg_alloc/dealloc<N>` | delete; use `__launch_bounds__(threads, blocks)` | HIGH | C9, E18 |
| `warpgroup_fence_operand / arrive / commit_batch / wait<0>` | delete; sm_80 `mma.sync` is synchronous | HIGH | C9 |
| `mainloop_pipeline.producer_tail()` | `cp.async.wait_group<0>` + `__syncthreads()` | HIGH | C11 |
| `ClusterShape_MNK` template + `cM, cN` | force `Shape<_1,_1,_1>`, cM=cN=1 | HIGH | C9 |
| `cudaLaunchKernelEx` + cluster attrs | plain `<<<grid, block, smem, stream>>>` | HIGH | C9, D16 |
| `cuTensorMapEncodeTiled` / `make_tma_copy` | delete entirely | HIGH | C9 |
| Producer-warp warp specialization | **unified-warp model** — every warp loads + computes | HIGH | C9 |

---

## 3. Tile sizing & smem budget (agent A4)

Sm_89 dynamic smem cap = **100 KB/CTA** (`cudaFuncAttributeMaxDynamicSharedMemorySize`). H100 = 228 KB. **All current `default_compiled_kernels.py` configs exceed 100 KB on sm_89 and won't fit.**

Recommended sm_89 configs (replace `MATMUL_KERNELS` for arch=89 only):

| bM | bN | bK | R | stages | Smem est. | Predicted TOPS at 71% util | Notes |
|---|---|---|---|---|---|---|---|
| 128 | 128 | 128 | 64 | 3 | ~98 KB | **~250** | int8 tile sweet spot, 1 CTA/SM |
| 128 | 128 | 64 | 64 | 4 | ~96 KB | ~230 | smaller K, more pipeline |
| 128 | 256 | 64 | 64 | 3 | ~80 KB | ~245 | wider N tile |
| 64 | 128 | 128 | 64 | 4 | ~84 KB | ~210 | 2 CTAs/SM achievable |

`heuristics.hpp:26-50` `get_pipeline_stages()` must be patched to read `dprops.sharedMemPerBlockOptin` for sm_89 (≈101376 bytes) instead of H100's 232448.

---

## 4. Build plan (agent D16)

`setup.py` diff at L87-88:
```python
TARGET_ARCH = os.getenv("PEARL_GEMM_TARGET_ARCH", "90a").lower()
_ARCH_GENCODES = {
    "89":  ["arch=compute_89,code=sm_89"],
    "90a": ["arch=compute_90a,code=sm_90a"],
    "all": ["arch=compute_89,code=sm_89", "arch=compute_90a,code=sm_90a"],
}
COMPUTE_CAPABILITIES = _ARCH_GENCODES[TARGET_ARCH]
ENABLED_ARCHES = {"89": [89], "90a": [90], "all": [89, 90]}[TARGET_ARCH]
```

L351:
```python
arch_flags = []
for gencode in COMPUTE_CAPABILITIES:
    arch_flags += ["-gencode", gencode]
```

`pearl_gemm_build_utils/kernel_configs/default_compiled_kernels.py` — add `arch: int` field to `MatmulKernelConfig`/`NoisingAKernelConfig`/`NoisingBKernelConfig`. Codegen filename suffix `_sm{arch}.cu` in `generate_instantiations.py`.

`vllm_kernels.py:70`:
```python
@classmethod
def get_min_capability(cls) -> int:
    import pearl_gemm_cuda
    return getattr(pearl_gemm_cuda, "_min_compute_capability", 9)
```
Set `_min_compute_capability = 8` from `pearl_gemm_api.cpp` when built with `-DPEARL_GEMM_BUILD_SM89`.

CUTLASS submodule pin `291300ff…` (~v3.5/3.6) already has the sm_80 `CollectiveBuilder` we need — no upgrade required.

---

## 5. Implementation sequencing (agents D17, E18 consolidated)

1. **Scaffolding (no kernels rewritten yet)** — land setup.py + codegen + dispatch changes. `PEARL_GEMM_TARGET_ARCH=90a` build stays bit-identical. `PEARL_GEMM_TARGET_ARCH=89` fails at nvcc (intentional — no sm_89 sources yet).
2. **Sm_89 `gemm()` (no noise, no PoW)** — write `kernel_traits_sm89.hpp` + `collective_mainloop_sm89.hpp` + `collective_epilogue_sm89.hpp` + `pearl_gemm_kernel_sm89.h`. Validate against `TestGEMM::test_noiseless_int7_gemm`. Bit-exact int32 path is the validation oracle.
3. **Sm_89 noisingA / noisingB** — easier than the main GEMM (already non-cluster, smaller smem). Validate with `TestNoiseA::test_int7_test_noise_a`. **bit-exact** because power-of-two scale.
4. **Sm_89 `noisy_gemm()` with PoW** — full port; validates against `TestNoisyGEMM` + `TestInnerHashCounting` (the formula match is the load-bearing check).
5. **Tune** — tile sweep, kStages, smem swizzle, vectorized stores. See §6.

Estimated effort: **3-5 weeks of engineering for one experienced CUTLASS dev**. Most of that is in steps 2-4. Step 1 alone is ~2 days.

---

## 6. Validation (agent D17)

**Premise correction from D17:** `pearl-gemm/tests/reference_outputs.json` **does not exist**. The only `reference_outputs.json` is at `vllm-miner/tests/` and contains LLM text strings, not tensors. Pearl-gemm tests compute their references in-process via `torch._int_mm` and Python helpers — no architecture-baked oracle.

Bit-exact gates (must hold on sm_89):
- `test_noise_gen.py` — BLAKE3 byte stream, `npt.assert_equal`
- `test_inner_hash.py` — uint32 mixing, `np.testing.assert_equal`
- `test_tensor_hash.py` — BLAKE3 Merkle, `torch.equal`
- `test_pearl_gemm.py::TestNoiseA/B::int7_*` — both int32 and fp16 outputs (`torch.equal`; power-of-two scale is exact)
- `test_pearl_gemm.py::TestInnerHashCounting` — count match against the formula

Tolerance gates (loosened bf16 epilogue):
- `test_pearl_gemm.py::TestGEMM/TestNoisyGEMM` — existing `atol=1e-1 rtol=1e-2`. Do not loosen.

---

## 7. Top risks (agent E18, top 3)

1. **HIGH/5 — Register pressure without `setmaxnreg`.** Hopper grants the producer warp's regs to consumers via `warpgroup_reg_alloc<256>`. Sm_89 has no equivalent; persistent producer/consumer kernels with the same accumulator footprint will spill. Mitigation: **unified-warp model** (every warp does cp.async + MMA), which is what every CUTLASS sm_80 example uses.
2. **HIGH/5 — Smem capacity collapse.** Sm_89 = 100 KB, Hopper = 228 KB. All current tile configs exceed budget. Mitigation: retune per §3.
3. **MED/5 — 75 TH/s target may be infeasible.** See §0. Confirm the unit before scoping.

---

## 8. Files the port touches (master list)

Must change:
- `miner/pearl-gemm/setup.py` (arch flag, codegen filter)
- `miner/pearl-gemm-build-utils/src/pearl_gemm_build_utils/kernel_configs/default_compiled_kernels.py` (sm_89 tile configs)
- `miner/pearl-gemm-build-utils/src/pearl_gemm_build_utils/generate_instantiations.py` (filename suffix)
- `miner/pearl-gemm/csrc/gemm/pearl_gemm_api.cpp` (export `_min_compute_capability`)
- `miner/pearl-gemm/csrc/gemm/pearl_gemm_launch_template.h` (arch dispatch)
- `miner/pearl-gemm/csrc/gemm/heuristics.hpp` (sm_89 smem budget)
- `miner/vllm-miner/src/vllm_miner/vllm_kernels.py:70` (read exported min capability)

Must add (new files, sm_89-only):
- `miner/pearl-gemm/csrc/gemm/kernel_traits_sm89.hpp`
- `miner/pearl-gemm/csrc/gemm/collective_mainloop_sm89.hpp`
- `miner/pearl-gemm/csrc/gemm/collective_epilogue_sm89.hpp`
- `miner/pearl-gemm/csrc/gemm/pearl_gemm_kernel_sm89.h`
- `miner/pearl-gemm/csrc/gemm/pearl_noisingA_kernel_sm89.h`
- `miner/pearl-gemm/csrc/gemm/pearl_noisingB_kernel_sm89.h`

Unchanged (pure sm_80+ code, ports for free):
- `csrc/blake3/*`, `csrc/tensor_hash/{tensor_hash.cu, compute_blake_mt_kernel.hpp, reduce_roots_kernel.h, commitment_hash_from_merkle_roots_kernel.hpp}`, `csrc/gemm/{quantize_kernel.cu, noise_generation.cu, named_barrier.hpp, pow_utils.hpp, inner_hash_kernel.cu}`

---

## 9. sm_120 (consumer Blackwell, RTX 50-series) addendum

Consumer Blackwell is an sm_80-PTX strict superset, so the sm_89 source tree
above also targets sm_120 — no new kernel files. The port is a multi-gencode
build switch plus extending the `#if defined(PEARL_GEMM_BUILD_SM89)` gates to
also fire on `PEARL_GEMM_BUILD_SM120`.

### Build invocations

```bash
# RTX 50-series only (RTX 5090 / 5080 / 5070):
PEARL_GEMM_TARGET_ARCH=120 pip install -e miner/pearl-gemm

# RTX 40 + 50-series fat binary (what MeowMiner ships):
PEARL_GEMM_TARGET_ARCH=all-consumer pip install -e miner/pearl-gemm
```

CUDA ≥ 12.8 is required (already enforced by `setup.py:get_wheel_url`).

### What carries over from sm_89

- Same MMA atoms (`SM80_16x8x32_S32S8S8S32_TN` for the int8 main GEMM, `SM80_16x8x16_F32F16F16F32_TN` for fp16 denoise).
- Same cp.async loads (`SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>`).
- Same ~100 KB dynamic smem opt-in cap on consumer Blackwell → same `default_compiled_kernels.py` narrow tile grid (R=64 bM=bN=128; R=128 bM=64 bN∈{64,128}; R=128 bM=128 bN=256 wave-10 path).
- Same KernelTraitsSm89 / CollectiveMainloopSm89 / CollectiveEpilogueSm89 templates — controlled by `PEARL_GEMM_USE_SM89_PATH` (defined in `pearl_gemm_launch_template.h` when either SM89 or SM120 is enabled).
- `heuristics.hpp::get_swizzle_size` and `get_pipeline_stages` already read `cudaDeviceProp` dynamically (`l2CacheSize`, `multiProcessorCount`, `sharedMemPerBlockOptin`) — no per-arch constants to patch.

### What's *not* done (deliberate; the user asked for kernel-only)

- **Tile retune for 5090 (170 SMs, ~96 MB L2).** The sm_89 grid is reused as-is. Wave-efficiency improves for free because the 5090 has ~2.8× the SMs (more parallel CTAs in a wave), but bM/bN/bK may not be at sm_120's optimum. Sweep `bench_sm89_r128_*.cu` on a 5090 to retune.
- **MeowMiner packaging.** This change makes the wheel build sm_120-capable; it does not bundle Pearl into the MeowMiner archive (no `--algo pearl` launcher wiring, no `start-pearl.{sh,bat}`, no Pearl runtime libs in the tarball).
- **No sm_100 (datacenter Blackwell) path.** sm_100's tcgen05 cluster MMA would need a separate kernel; this addendum is consumer-only (sm_120).

### Smoke test

After building with `PEARL_GEMM_TARGET_ARCH=120` on a 5090 host:

```bash
python miner/pearl-gemm/csrc/gemm/sm89_smoke.py        # noiseless GEMM
python miner/pearl-gemm/csrc/gemm/sm89_smoke_r128.py   # R=128 bit-exact
python miner/pearl-gemm/csrc/gemm/sm89_noisy_smoke.py  # full noisy_gemm pipeline
```

`_min_compute_capability` should report `8` (sm_120 is treated as Ampere+
because all atoms are sm_80-PTX). `pg._min_compute_capability` is exported
from `pearl_gemm_api.cpp:1350`.
