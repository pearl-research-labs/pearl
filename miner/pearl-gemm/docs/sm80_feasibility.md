# SM80 (A100 / GA100) FP16 GEMM — feasibility probe

**Verdict: GO. Backend: raw CUDA (`mma.sync.m16n8k16.f32.f16`) via `torch.utils.cpp_extension.load_inline` with `-arch=sm_80`.**

## What was tested

The FP16 scheme's verifier replays the A100 `HMMA.16816.F32` accumulation,
modeled in `zk-pow/src/api/fp16/accumulate.rs` (`a100_dot` / `a100_matmul`): the
k axis in groups of `G = 8`, per-group alignment exponent `eta`, truncate-toward-
zero onto `2^(eta-24)`, exact integer sum, round-toward-zero to FP32 after every
group. That model was **measured from this exact silicon**, so a straight sm_80
FP16 tensor-core GEMM with FP32 accumulation and a pinned ascending-k reduction
(no split-k, no atomics) should reproduce it bit-for-bit natively.

The probe (`feas.py`, archived) compiles a single-warp kernel that computes one
`16x8` FP32 output tile = `A(16xK) · B(8xK)ᵀ`, looping over k in steps of 16 with
one `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` per chunk, chaining the
FP32 accumulator forward (no split-k). It compares the output **bit pattern**
(`float32` → `uint32`) against a Python integer-exact port of `a100_dot`
(`tests/helpers/a100_fp16_reference.py`, itself validated bit-exact against all
238 reference vectors in `testdata/a100_dot_vectors.txt`).

## Result

Hardware: NVIDIA CMP 170HX (GA100, compute cap 8.0), CUDA 12.8, nvcc `-arch=sm_80`.

```
K=16:  0/128 mismatches
K=32:  0/128
K=64:  0/128
K=128: 0/128
K=256: 0/128
TOTAL: 0 / 640  => BIT-EXACT
```

The native sm_80 `mma.sync.m16n8k16` chain with pinned k-order **is** the
verifier model, bit-for-bit, on real GA100 tensor cores.

## Backend choice: raw CUDA, not CuTe

`pyproject.toml` pins `nvidia-cutlass-dsl[cu13]>=4.6.0`, but **4.5.0** is what is
installed. Rather than fight that toolchain pin on sm_80 (and because the
existing `_kernel_sm120.py` warp-mma path is deeply specialized for the *FP8*
lottery datapath — f8f6f4 atoms, peel/unscale, hit-signal), the bit-exact FP16
tile is shipped as a small hand-written CUDA kernel compiled with
`-arch=sm_80` and loaded via `cpp_extension`. This is the "proven to work"
fallback named in the task brief and keeps the FP16 path independent of the FP8
CuTe stack.

## Scope shipped vs deferred

- **Shipped:** the bit-exact sm_80 FP16→FP32 GEMM tile (the datapath the verifier
  replays and the ticket hashes), `Arch.SM80` + protocol constants, the Python
  oracle, and an on-GPU bit-exact + determinism test.
- **Deferred (documented, not implemented):** on-GPU lottery hit-signal, the C''
  peel/denoise, and the FP16 noisy-quant / noise-line / tensor-hash kernels.
  These are the FP8-style fusion follow-ons; the GEMM tile is the core
  correctness target and is complete.
