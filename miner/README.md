# Pearl Miners

This directory contains the FP8 plain-peel miner stack. Distribution and
directory names use hyphens; Python import packages use underscores.

## Package roles

- **`miner-base`** owns framework-independent protocol, gateway, accounting,
  and proof-submission primitives. Concrete miner execution loops
  belong to the framework package.
- **`vllm-miner`** is the vLLM plugin (`vllm_miner`): a single image that
  serves and, when mining is enabled, mines on the served dense layers'
  forwards. See [`vllm-miner/README.md`](vllm-miner/README.md).
- **`pearl-gemm`** is the SM100 (B200) CuTe DSL kernel package the miner
  launches (`pre_quant`, `tensor_hash_plus_stats`, `noisy_quant`,
  `mixed_gemm` and the B-side chain).
- **`pearl-gateway`** is the node-to-miner bridge the miner submits proofs
  through; **`miner-utils`** holds the shared logging helpers.

The FP8 protocol math lives in `miner-base` (commitments, noise, quantization, scheme, hardware).

## Distributed dense mining

Each vLLM CUDA worker independently commits and mines the actual 2-D weight
shard loaded in that process. TP, DP, and EP retain this process-local proof
contract. Local shape/device/verifier bounds decide eligibility, and an
ineligible shard stays on the framework's native BF16 path. Replicated workers
report additive process-local credits through the existing gateway metric.

Only dense framework linear layers participate. Fused expert GEMMs remain
native. Pipeline-parallel stages mine their resident dense layers during
native forwards. Only CUDA worker processes own mining managers. Framework teardown
closes process-local mining admission, waits only for CUDA and host-continuation
memory safety, and then reaps the process tree. Final hash reports or an
extremely unlikely opened proof may be dropped during shutdown.

## Docker images

Build from the repository root:

```bash
docker buildx build -t vllm_miner . -f miner/vllm-miner/Dockerfile
```
