# H200 Receipt-Sidechannel D1/D2/D3 Result

UTC run window: 2026-05-18T17:51:07Z to 2026-05-18T18:14:20Z.
Hardware: rented Vast NVIDIA H200, instance 37021208, now destroyed.
Build: 7199b8db59a0ec54caf72c23245576d3f59f9fb6+dirty.

## Result to Tell the Team

The receipt-sidechannel / Hardy split path is no longer a no-GPU replay theory.
On a real H200, AKO-012 produced a private soft-pool accepted share, then the
saved proof artifact was split, staged as a proof-node-side type-3 share, and
accepted by the private soft pool with outcome code 0.

Paste-ready speed line:

```text
Accepted path, AKO-012 geometry, build 7199b8db59a0ec54caf72c23245576d3f59f9fb6+dirty: 464.19 hot-loop / 531.42 NoisyGEMM TMAD/s, soft-pool accepted=true, share_id f428aa13-f3f3-47a7-9737-69408106effe, 2026-05-18T18:12:35Z.
```

## D1 Direct Accepted Share

- Status: PASS.
- Direct share id: `f428aa13-f3f3-47a7-9737-69408106effe`.
- Soft-pool result: `accepted=true`, `outcome_code=0`.
- Server message: `Mining solution verified successfully`.
- Proof artifact source: `d1_direct_run_summary.json`.
- Direct server evidence: `d1_server_result.json`.

## D2 Measured H200 Speed

- Status: PASS with caveat.
- Measured no-hit proof-GEMM path: `464.19` hot-loop / `531.42` NoisyGEMM TMAD/s.
- Wall-inclusive speed: `432.74` TMAD/s.
- Artifact: `d2_nohit_speed.json`.
- Forced-hit canary: `d2_forced_hit_canary.json`, status `verified`.
- Caveat: full fused `noisy_gemm` normal-path benchmark failed with CUDA illegal instruction at `pearl_noisingB_host.h:54`; see `d2_full_noisygemm_failed.json`.
- P1K165 two-phase run was also executed but is diagnostic-only: `405.08` hot-loop / `438.38` NoisyGEMM TMAD/s; see `d2_p1k165_diagnostic.json`.

## D3 Hardy Split

- Status: PASS.
- Split-side share id: `00000000-0000-4000-8000-000000000005`.
- Split client result: `accepted=true`, `outcome_code=0`.
- Split server result: `accepted=true`.
- Staged type-3 frame SHA256: `068586f2edc38b1a13b3ed9010c9eaa647c730a2ca297c50e7f4ef8da0db4337`.
- Evidence: `d3_verify_split.json`, `d3_stage_metadata.json`, `d3_client_result.json`, `d3_server_result.json`, and `d3_type3_plain_proof_share.msgpack`.

## Next Step

Fix the full fused `noisy_gemm` noisingB illegal-instruction path, then rerun
the same D1/D2/D3 gate and advance to public-pool or mainnet validation. Do
not claim public payout or 1K TMAD from this run.

