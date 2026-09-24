# vLLM - PearlMiner

A vLLM plugin that mines the Pearl FP8/FP10 scheme inside serving forwards on
SM100 (B200) GPUs, over BF16 or quantized HuggingFace checkpoints.

## How it works

Select the plugin with `--quantization pearl` (BF16 checkpoints carry no
quantization config; the plugin default-constructs). At load time every
supported linear layer is encoded to the schema's committed FP10 planes (one
int8 code per value + one BF16 scale per 8 values) with the reference
`PrequantMatrix.encode`, and an FP8 fallback operand is derived from those
planes. Eligible prefills run the four-op mined pipeline
(`pre_quant -> tensor_hash_plus_stats -> noisy_quant -> mixed_gemm`). Decode,
small batches, capture windows, no-job periods, and mining failures serve
through the FP8 fallback GEMM; startup also remains available on that fallback
when initial B preparation is not yet ready. Excluded and unsupported layers
keep vLLM's unquantized BF16 path. All mined launches share one persistent
`pearl_gemm.HitSignal` per CUDA device; the event-gated consumer atomically
copies and re-arms it before validating a candidate, so the first-wins latch is
reusable across launches.

Lottery candidates are validated against the reference `is_winning`, opened
from the retained FP10 planes into certificate-v3 `PlainProof` objects, and
canonically verified. Jackpot-policy-inadmissible candidates are filtered;
admissible proofs are submitted to `pearl-gateway`. The gateway and Go node still do
not support the certificate-v3 block wire format, so production node acceptance
remains unavailable; integration uses the real miner RPC plus the consensus
PlainPeel verifier. B-side per-job preparation runs the SM100 GPU chain
(`tensor_hash_plus_stats_b -> noise_lines -> noisy_quant_b`) off the serving
path. The CPU `build_b_rows` path is retained only as an independent oracle for
rare winner validation; it never prepares production launch operands.

Capability gates:

- SM100 (B200) only: the `pearl` quantization method refuses to load on any
  other GPU. TP, DP, and EP workers independently commit and mine
  each eligible process-local dense shard; there is no cross-rank weight
  reconstruction or proof aggregation.
- Dense linear layers only; a mined local shard needs `k % 512 == 0`,
  `n % 128 == 0`, and `n, k % 16 == 0` (FP8 fallback alignment). Layers that
  cannot mine on this device/shape keep their original BF16 path.
- MoE expert weights are not mined yet. MLA `.kv_b_proj` parameters also stay
  on their source scheme because the model reads those weights directly instead
  of exclusively through the linear quantization method.
- The worker subtracts a bounded post-profile allowance for the persistent hit
  signal and one reusable A-side workspace for each runtime CUDA stream that can
  retain an allocator population. The allowance counts one workspace per
  runtime CUDA stream; completion callbacks retain only host continuations.
  Same-stream allocator blocks are ordered and reused. The allowance uses the
  larger of these runtime workspaces and the mutually exclusive
  checkpoint-reload transient.
- Pearl-owned dense operands must remain CUDA-resident after registration.

### Quantized source checkpoints

`--quantization pearl` also accepts a checkpoint that carries its own
`quant_method` — plain `fp8` (block or per-tensor) and `compressed-tensors`
FP8 / NVFP4 / MXFP4. The plugin claims such a checkpoint through
`override_quantization_method` and delegates loading to the matching vLLM quant
method, so the model keeps its compact footprint. Only mineable dense linears
are then decoded back to BF16 through the delegate's own kernels and encoded to
FP10; every other layer — MoE experts, KV cache, embeddings, MLA `.kv_b_proj`,
`PEARL_IGNORED_LAYERS` matches, and unsupported local shards — keeps the
checkpoint's source scheme.

This upcast is **lossy in the sense that it cannot recover what the checkpoint
already discarded**: mining and serving then run over already-quantized
weights. Prefer a dynamic-activation checkpoint; a checkpoint-provided static
activation scale adds a further uniform rounding factor and is warned about
once at load.

## Installation

The package is a member of the top-level `uv` workspace (see `pyproject.toml`
at the repository root). Install it with `uv sync` from the repository root:

```bash
uv sync --package vllm-miner
```

## Serving

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct --quantization pearl ...
```

### vLLM 0.26 reload and sleep mode

The worker hooks support checkpoint-format `reload_weights` and both CuMem
sleep levels. They first close all mining-producer admission, drain exact CUDA
completions, and forget every layer's prepared job. Reload
re-encodes FP10 and FP8 operands **in place**, preserving every model/graph
pointer; wake verifies CuMem restored the same virtual addresses before mining
restarts. A standalone `collective_rpc("reload_weights")` first pauses the
engine scheduler, aborts active requests, clears prefix/MM/encoder caches, and
waits asynchronously for the final executor step before touching weights; it
resumes scheduling only after every worker succeeds. Level-2 sleep follows
vLLM's required order: wake `weights`, call `collective_rpc("reload_weights")`,
then wake `kv_cache`. Kernel-format reload
is rejected because it does not contain Pearl's derived operands. A failed
transition leaves mining suspended and requires worker restart rather than
exposing partially updated state.

The authoritative names, defaults, bounds, and example for all `PEARL_*`
execution controls are under [Runtime configuration](#runtime-configuration).
Initial mining readiness and job loss fail open to FP8 serving; a job change
re-prepares each layer's B operands in place on the serving stream the first
time that layer launches under the new job. CUDA ownership failures remain
fail-closed and require worker restart.

Shared miner knobs (`MINER_`-prefixed, `miner_base.settings`): `MINER_NO_MINING`,
`MINER_NO_GATEWAY`, `MINER_SKIP_BLOCK_SUBMISSION`, and
`MINER_SUBMISSION_INFLIGHT_LIMIT` (bounded CPU proof/RPC handoffs, default `8`).

## Running the Docker image

```bash
docker run --rm -it --gpus all -p 8000:8000 -p 8337:8337 -p 8339:8339 \
  -e MINER_NO_GATEWAY=0 -e PEARLD_RPC_URL=http://172.17.0.1:44107/ -e HF_TOKEN=<TOKEN> \
  -v /.cache/huggingface:/root/.cache/huggingface --shm-size 8g \
  vllm_miner:latest meta-llama/Llama-3.1-8B-Instruct \
  --quantization pearl --host 0.0.0.0 --port 8000 \
  --max-model-len 8192 --gpu-memory-utilization 0.9
```

`--quantization pearl` encodes the mineable layers and every eligible served
forward mines; the container passes `"$@"` through to `vllm serve` unchanged.
Set `MINER_NO_MINING=true` to run serve-only.

### Container lifecycle

The entrypoint starts `pearl-gateway` in the background, waits for its miner
RPC socket, then `exec`s `vllm serve`. vLLM's worker shutdown closes mining
admission, waits for process-local CUDA and continuation memory safety, and
then runs native cleanup. Final hash accounting or an in-flight proof may be
dropped.

### Mining kill switch

`MINER_NO_MINING=true` is the operator kill switch and stops mining
altogether: no request-time mining, no B preparation and no kernel warmup.
Selected layers are still FP10-encoded at load and serve on the FP8 fallback
GEMM.

Notes / constraints:

- CUDA graphs work with vLLM's default `cudagraph_mode`; captured graphs
  always replay the FP8 fallback, so mining runs on eager prefills.
- TP/DP/EP workers own independent runtime, accounting, winner, and drain
  state. Fused expert GEMMs remain native; only dense `LinearBase` shards
  route through Pearl.

## Runtime configuration

`RuntimeSettings` reads case-insensitive `PEARL_*` environment variables once
per worker process. These controls configure the launch runtime; the current
schema-mining kernels require SM100/B200 and supported layer shapes.
Unsupported devices or layer shapes stay on the serving fallback.

| Variable | Default | Valid values | Applies to | Effect |
|---|---:|---|---|---|
| `PEARL_IGNORED_LAYERS` | empty | comma-separated exact prefixes or `re:` regular expressions | registration | Excludes matching framework layer prefixes from mining. |
| `PEARL_MIN_MINING_TOKENS` | `1024` | integer `>= 4` | request-time/busy | Sends smaller serving forwards through the FP8 fallback. |
| `PEARL_M_BUCKETS` | `2048,8192` | non-empty comma-separated positive multiples of 64 | request-time | Bounds compiled kernel variants. Request-time mining pads to the smallest fitting bucket and falls back above the largest. |
| `PEARL_WARMUP_COMPILE` | `true` | boolean | startup/job preparation | Compiles every configured pipeline variant off the serving path before mining engages. |
| `PEARL_WINNER_CHECK_INFLIGHT_LIMIT` | `8` | integer `>= 1` | all mining launches | Caps retained launches awaiting event-gated inspection of the persistent hit signal. At the cap, mining is declined and serving falls back. |
| `PEARL_COMPLETION_INFLIGHT_LIMIT` | `8` | integer `1..256` | all mining launches | Caps all event-gated credited launches, including no-gateway launches that retain no winner. At the cap, mining is declined before GPU allocation. Also sizes the mining memory kept off the KV cache (peak reservation scales ~linearly with it); raise only when profiling shows spare memory. |
| `PEARL_OOM_COOLDOWN_S` | `30` | finite float `> 0` | all mining GPU work | Suspends mining device-wide after OOM, then admits exactly one recovery probe; a no-op/stale attempt does not reopen ordinary admission. Serving continues on fallback without cache flush or device synchronization. |
| `PEARL_RECON_CHUNK_ROWS` | `4096` | integer `>= 1` | quantized-checkpoint load | Identity rows fed per delegate `apply` when decoding a quantized weight back to BF16. Bounds the upcast's transient workspace; the free-memory budget check is the real OOM gate, so lower this only for a pathological shape whose per-chunk transient is too large. |

Every credited launch reports **difficulty-normalized effective work** (`m*n*k`)
to the gateway, so `total_hash`/hashrate is comparable across shapes/`k` and to
network difficulty.

Proof/RPC occupancy intentionally does not gate mining launches:
`MINER_SUBMISSION_INFLIGHT_LIMIT` bounds actual proof work, while the independent
winner/completion limits bound retained launch state. If a rare winner arrives
with every proof slot occupied, its handoff waits up to 30 seconds.

`PEARL_COMPLETION_INFLIGHT_LIMIT` also sizes the mining memory reserved off the
KV cache. The default of `8` keeps enough concurrency to overlap mining with
serving while leaving KV headroom; raise it only after confirming spare device
memory in the startup log's `Reserved ... for bounded Pearl mining work` line.

Example (SM100 request-time mining):

```bash
PEARL_IGNORED_LAYERS='' \
PEARL_MIN_MINING_TOKENS=1024 \
PEARL_M_BUCKETS=2048,8192 \
PEARL_WARMUP_COMPILE=true \
PEARL_WINNER_CHECK_INFLIGHT_LIMIT=8 \
PEARL_COMPLETION_INFLIGHT_LIMIT=8 \
PEARL_OOM_COOLDOWN_S=30 \
PEARL_RECON_CHUNK_ROWS=4096 \
vllm serve meta-llama/Llama-3.1-8B-Instruct --quantization pearl
```

## Running Tests

The suite runs on the B200 GPU CI job. Device-heavy files such as
`tests/test_vllm_adapter.py` and `tests/test_runtime_gpu.py` need an
SM100 GPU and carry no hardware skip guards.

```bash
pytest miner/vllm-miner/tests
```

## Building the Docker Image

```bash
docker buildx build -t vllm_miner . -f miner/vllm-miner/Dockerfile
```
