# miner-base

Framework- and kernel-independent infrastructure every Pearl miner shares,
plus the stable FP8 plain-peel protocol surface.

`miner_base` owns:

- commitments, extractor layout, prequantized planes, deterministic noise,
  FP8 quantization, lottery policy, scheme types, and the hardware contract
  (`miner_base.commitment`, `layout`, `noise`, `quantization`, `scheme`,
  `hardware`, …). :func:`miner_base.hardware.hardware_for` resolves Blackwell
  (SM100) only.
- `async_loop_manager`: gateway job polling, credited-hash accounting, bounded
  proof handoff. CUDA event/stream ownership belongs to
  the `vllm_miner` package in `vllm-miner`.
- `block_submission`: validated PlainPeel openings, proof construction,
  verification, policy filtering, and gateway handoff.
- `mining_config`, commitment helpers, settings, and gateway client.

The shared Torch/CUDA execution loop belongs in `vllm-miner`.

## Installation

```bash
uv sync --package miner-base
```

## Runtime configuration

`MinerSettings` reads case-insensitive `MINER_*` environment variables. The
retained miner stack consumes these settings:

| Variable | Default | Valid values | Effect |
|---|---:|---|---|
| `MINER_NO_GATEWAY` | `false` | boolean | Uses the deterministic offline job and refuses winner inspection, proof construction, and submission. |
| `MINER_SKIP_BLOCK_SUBMISSION` | `false` | boolean | Disables winner retention, proof construction, and submission; successfully completed lottery work remains credited. |
| `MINER_SUBMISSION_INFLIGHT_LIMIT` | `8` | integer `>= 1` | Bounds proof construction plus gateway RPC work admitted by each process. |

`MINER_NO_MINING` and `MINER_REGISTER_QUANTIZATION` are declared here so every miner parses one settings contract, but execution
ownership stays in the vLLM adapter, which documents each control's activation
and hardware constraints.

## Tests

```bash
uv run --package miner-base pytest miner/miner-base/tests
```
