# pearl-gemm

CuTe DSL kernels for Pearl's FP8 scheme, installed as a member of the repo's
root uv workspace. Everything is pure Python: kernels
JIT-compile at runtime via `nvidia-cutlass-dsl`, so there is no C++ build,
and autotuning is a plain parameter sweep (`pearl_gemm.autotune`).

## The 4-op API (`pearl_gemm.api`)

| Op | Kernel | Status |
|---|---|---|
| `pre_quant` | `pre_quant/` | block-scaled INT8 quantization (BF16 scale per 8 codes) into the two committed blobs |
| `tensor_hash_plus_stats` | `tensor_hash_plus_stats/` | keyed BLAKE3 Merkle chain over each block-scaled activation blob (codes, scales) |
| `noisy_quant` | `noisy_quant/` | fused stats combine, E₁, quantization, and peel; opens the committed code/scale blobs in registers |
| `mixed_gemm` | `mixed_gemm/` | persistent FP8 GEMM (SM100 tcgen05+TMEM), first-winner publish into the persistent hit signal (`pow/`), BF16 peel, and unscale |

All operations are functional: callers allocate and pass inputs, outputs, and
workspaces. Host modules validate those buffers and cache compiled launchers
by shape and config, never by tensor pointer. The integration tests contain
the complete allocation and launch wiring.

`pre_quant` is the only kernel that reads BF16 A: the downstream ops all
consume the two committed blobs (`(m, k)` int8 codes and `(m, k/8)` BF16
scales, 1.25 B/element in total), so the committed bytes and the bytes prep
reads are the same bytes by construction.

Each operation package keeps its host launch in `_host.py` beside its device
modules. The full Merkle and finalize pipeline lives together under
`tensor_hash_plus_stats/`, which also fuses the per-block stats partials into
the codes blob's roots kernel (they factor over the block scale, so nothing
is dequantized); cross-operation helpers live under `_utils/`.
The same hash engine is also exposed as `tensor_hash` for arbitrary contiguous
CUDA tensor bytes without commitment finalization.

## Layout

- `src/pearl_gemm/` — operation packages, API, constants, and autotuning
- `tests/` — kernel tests (`pytest`; GPU required for most): one file per op,
  shape grids, consistency, and CUDA-graph coverage
- `tests/helpers/` — B-side preprocessing and full miner wiring

## Autotune

```bash
python -m pearl_gemm.autotune --shapes 4096x4096x4096,8192x8192x8192
python -m pearl_gemm.autotune --kernels mixed_gemm --shapes 4096x6144x16384
```

merges its winners into `src/pearl_gemm/autotune_configs/<device>.json`,
keyed by `(kernel, shape)`: a subset retune keeps every shape it did not
sweep, and `--kernels` keeps every op it did not name. `mixed_gemm` injects
the committed lottery for `(n, k)` so GLM widths that are not 128-aligned
are still in-space. At runtime `autotune.get_tuned("noisy_quant", m=..., k=...)`
returns the best-known config fields (exact shape match, else nearest in
log2 space). `tensor_hash_plus_stats` does that lookup itself when `config`
is omitted (`get_tensor_hash_plus_stats_config(m, k)`, resolved for the input
tensor's device), LRU-caching the record. Pass `--out` a copy to preview a
retune before adopting it.

## Verification

```bash
scripts/run_tests.sh pr
scripts/run_tests.sh full
```
