# FP16 / A100 scheme — reproducible hardware validation

This directory is the **reproducible hardware oracle and calibration harness** for
the FP16 accumulation-hardness scheme (`docs/fp16_scheme/fp16_scheme.tex`). It
exists to close three gaps a review of the scheme flagged:

1. the committed reference corpus
   (`zk-pow/src/api/fp16/testdata/a100_dot_vectors.txt`) topped out at **k=256**
   and carried no explicit subnormal / cancellation / overflow edges;
2. the accumulation model was validated only **circularly** in-tree
   (`accumulate.rs`, the Python reference, and the vector file are three
   expressions of the *same* model) — nothing checked it against the **device**;
3. the **RZ-vs-RNE** rounding question (does the A100 round toward zero, as the
   model assumes, or to nearest-even?) had no in-tree silicon resolution, and the
   Appendix "Empirical validation" numbers (f_bp / rho distributions, shortcut
   rates) were prose with **no runnable harness**.

Everything here runs on a real **sm_80** GPU (A100 / GA100 / CMP 170HX) with a
standard CUDA toolkit (tested: nvcc 12.8, CMP 170HX 64 GB, driver 610.43.02). It
deliberately links only the CUDA runtime — **no torch, no pearl-gemm** — so it
builds and runs on an sm_80 box that cannot install the py3.12 miner stack.

## Contents

| file | what it does |
|---|---|
| `a100_hmma_capture.cu` | Standalone capture tool. Issues the real `mma.sync.m16n8k16.f32.f16` (asm + fragment layout copied verbatim from the validated production kernel `fp16_gemm/_kernel_sm80.cu`) and dumps device FP32 result bits for each input dot product. |
| `generate_and_capture.py` | Generates an edge-case corpus (k up to 1024; subnormal operands / accumulators / outputs; cancellation; grid-boundary; max-magnitude), captures it on silicon, and **cross-checks the software model against the device** bit-for-bit. Writes `a100_dot_vectors_k1024.txt`. |
| `rounding_mode_probe.py` | Builds four model variants (per-product ∈ {RZ,RNE} × per-group ∈ {RZ,RNE}), finds vectors where they disagree, captures them on silicon, and reports which rounding the device matches. **Resolves RZ-vs-RNE empirically.** |
| `calibration.py` | Reproduces the Appendix policy-metric study: per-strategy `f_bp`, `rho`, and shortcut-predictor reproduction rates, checked against the gate (`f_bp>=0.30`, `rho>=1.2`) and the paper's claimed regime. CPU only. `--rust-noise` drives the **real consensus noise pipeline** (next row) instead of a Python reimplementation. |
| `attack_cost_benchmark.cu` | Times the honest tensor-core GEMM against the per-product truncation-correction kernel (which must run on CUDA cores) on silicon, measuring the attacker's cost multiplier — the paper's "6-13x / memory-bound far more" claim. |
| `../../../zk-pow/examples/fp16_noise_tool.rs` | Rust example that emits seed-exact noised operands through the real `fp16_noised_operands` pipeline (root-derived seeds -> BLAKE3 noise lines -> `noisy_quantize`); `calibration.py --rust-noise` shells out to it. Build: `cd zk-pow && cargo build --release --example fp16_noise_tool`. |

## How to run

```sh
cd docs/fp16_scheme/validation
nvcc -arch=sm_80 -O2 -o a100_hmma_capture a100_hmma_capture.cu   # build once
python3 generate_and_capture.py        # edge corpus + model-vs-silicon cross-check
python3 rounding_mode_probe.py          # RZ-vs-RNE resolution on silicon
python3 calibration.py                  # policy-metric calibration (Python noise; no GPU)

# seed-exact consensus noise + the attack-cost benchmark:
(cd ../../../zk-pow && cargo build --release --example fp16_noise_tool)
python3 calibration.py --rust-noise     # same calibration, real consensus noise bytes
nvcc -arch=sm_80 -O3 -o attack_cost_benchmark attack_cost_benchmark.cu
./attack_cost_benchmark 512 512 1024 50 # honest vs truncation-correction cost
```

The regenerated `a100_dot_vectors_k1024.txt` is copied to
`zk-pow/src/api/fp16/testdata/` and consumed by the Rust cross-check test
`api::fp16::accumulate::tests::matches_hardware_capture_k1024`, so CI enforces
"model == captured silicon bits" (including a k=1024 and a subnormal-output
assertion). The capture binary itself is a build artifact (git-ignored).

## Recorded results (on CMP 170HX, sm_80, nvcc 12.8)

**Model vs silicon — bit-exact.** The capture tool reproduces all **238**
pre-existing committed vectors bit-for-bit, and the model reproduces all **272**
newly captured edge vectors (k up to 1024; 16 of them subnormal-FP32 outputs;
64 subnormal-FP32 carry-ins). Total: **910** independent silicon dot products
agree with the model, **0 mismatches** (238 committed + 272 edge + 400
rounding-discriminating).

**RZ-vs-RNE — RZ confirmed at both stages.** On **400** vectors constructed so the
four rounding variants disagree, the device matched the committed `(RZ, RZ)`
model on **400/400**; the alternatives matched only where they happen to coincide
with RZ and were excluded elsewhere (`RZ/RNE` 113/400, `RNE/RZ` 109/400,
`RNE/RNE` 60/400). **Conclusion: the A100/GA100 `HMMA.16816.F32` rounds toward
zero** both when aligning each product/accumulator onto the `2^(eta-24)` grid and
when encoding the per-group sum to FP32 — exactly what `accumulate.rs` models.

**Overflow is unreachable from a single FP16 dot product.** Max-magnitude
(|operand| up to 65504) k=1024 tiles with ±max-finite FP32 carry-in all stayed
finite on silicon; no generated vector overflowed. The accumulation guard against
non-finite results is therefore defensive, consistent with the feasibility note
that FP16 products align at `eta >= 99` and the carry model forbids a cell-start
carry-in.

**Subnormal-FP32 output is reachable only via a surviving subnormal carry-in.**
With FP16 operands the smallest nonzero product is `2^-48`, so any nonzero product
dominates a subnormal (`< 2^-126`) accumulator and pushes the output normal; a
nonzero subnormal output arises only when a subnormal FP32 carry-in survives
products that are zero or exactly cancel. The corpus includes 16 such cases
(captured and model-matched), which exercise the device's subnormal encode — the
`matmul_a100_stark` MA13 branch the ZK audit noted was otherwise silicon-untested.
This confirms MA13 is a sound guard on the reachable sub-case.

**Policy calibration (`calibration.py`, 16×16 tile, k=1024, rank-32 δ=1/2 noise).**
Across 8 adversarial/realistic strategies, with the **shape-faithful Python**
noise: `f_bp ∈ [0.453, 0.512]`, `rho ∈ [1.67, 1.94]`, shortcut reproduction
≤ 1.56%. With the **seed-exact Rust consensus** noise (`--rust-noise`, via
`fp16_noise_tool`): `f_bp ∈ [0.448, 0.518]`, `rho ∈ [1.674, 1.939]`, shortcut
≤ 2.34%. Both match the paper's claimed regime (`f_bp ∈ [0.36,0.51]`,
`rho ≥ 1.38`, cheapest shortcut ≤ ~1.6%), all well above the `0.30 / 1.2` gate —
and the two noise sources agree closely, so the Python proxy was faithful.

**Truncation-correction attack cost (`attack_cost_benchmark.cu`, on CMP 170HX).**
Granting the attacker the exact sums and the breakpoint oracle for free, the
unavoidable per-product correction (decode + multiply + per-group truncation on
CUDA cores) measured **37.6×** the honest bit-exact tensor-core GEMM at
`512×512×1024` and **43.0×** at `1024³`. The honest baseline is the miner's own
kernel style (one warp per 16×8 subtile, no shared-memory pipelining — exactly
`fp16_gemm/_kernel_sm80.cu`), so the ratio is representative, not pessimistic. It
sits well above the whitepaper's "6–13× compute-bound-optimistic" floor and below
its "220–470× memory-bound" ceiling, consistent with the hardness claim that the
corrections cannot be batched onto tensor cores.

## Scope / remaining

- The accumulation model and both roundings are silicon-validated; the
  policy-metric distributions are reproducible with **both** a Python proxy and
  the **real consensus noise pipeline** (`--rust-noise`); and the attack-cost
  multiplier is measured on silicon. The three gaps the review flagged are closed.
- The attack benchmark measures the *lower-bound* elementwise correction cost; a
  production attacker's kernel could be tuned, but so could the honest one (which
  benefits from tensor cores — the asymmetry the scheme relies on). The strategy
  set in `calibration.py` is extensible for broader adversarial sweeps.
