# Receipt-Sidechannel H200 Result, 2026-05-18

The H200 run validated the core receipt-sidechannel / Hardy split theory on
real GPU infrastructure.

## What Passed

- D1 produced a real private soft-pool accepted AKO-012 share on an H200:
  `f428aa13-f3f3-47a7-9737-69408106effe`.
- D2 measured the accepted-path proof-GEMM no-hit speed at `464.19` hot-loop /
  `531.42` NoisyGEMM TMAD/s.
- D3 rebuilt and staged the proof-node-side type-3 share from the D1 proof
  artifact, submitted it to the private soft pool, and received
  `accepted=true`, `outcome_code=0`.
- The D3 split-side share id was
  `00000000-0000-4000-8000-000000000005`.

## Evidence Package

All durable run artifacts are committed under:

```text
tools/akoya_bridge/artifacts/20260518_h200_receipt_sidechannel/
```

The package includes the D1 direct summary, D1 server result, D2 speed/canary
artifacts, the failed full `noisy_gemm` benchmark artifact, P1K165 diagnostic
artifact, D3 staged frame metadata, D3 client/server results, and SHA256 sums.

## Current Interpretation

The split/proof-artifact path is validated for private soft-pool acceptance.
This is not public-pool or mainnet payout validation yet.

The next engineering blocker is the full fused `noisy_gemm` normal-path failure:

```text
CUDA error: an illegal instruction was encountered at
miner/pearl-gemm/csrc/gemm/pearl_noisingB_host.h:54
```

Fix that path, rerun D1/D2/D3 on H200, then attempt public-pool or mainnet
validation.

