# miner-base

Framework- and kernel-independent infrastructure shared by Pearl GPU miners,
plus the host-side FP8 plain-peel protocol surface. `miner_base` owns:

- `protocol`: cert-v4 parameters, transcript keys/seeds, device mapping, and lottery layouts.
- `commitment`: keyed BLAKE3 Merkle commitments and openings.
- `prequant`: block-scaled int8 + BF16 prequantized planes.
- `block_submission`: PlainPeel opening validation, proof construction, local verification, and gateway handoff.
- `gateway_client` & `async_loop_manager`: gateway JSON-RPC client, job polling, credited-hash accounting, and bounded proof submission.
- `settings`: `MINER_*` runtime settings.

## Installation & Tests

```bash
uv sync --package miner-base
uv run --package miner-base pytest miner/miner-base/tests
```
