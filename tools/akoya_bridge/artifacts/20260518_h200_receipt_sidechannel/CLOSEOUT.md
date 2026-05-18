# CLOSEOUT

D1: PASS. Private soft-pool accepted=true, share_id f428aa13-f3f3-47a7-9737-69408106effe, outcome_code 0, proof artifact saved in /srv/pearl/overnight/artifacts/h200-37021208/ako012-20260518T175107Z/direct_run/summary.json.

D2: PASS with caveat. Paste-ready line is in TMAD_RESULT.md: 464.19 hot-loop / 531.42 NoisyGEMM TMAD/s on H200, normal benchmark path. The full noisy_gemm path failed with CUDA illegal instruction in pearl_noisingB_host.h; the proof-GEMM no-hit run is paired with a verified forced-hit canary and D1 soft-pool acceptance. P1K165 two-phase was also run separately and is diagnostic-only.

D3: PASS. Split-side staged type-3 share accepted=true / outcome_code 0, share_id 00000000-0000-4000-8000-000000000005; frame sha256 068586f2edc38b1a13b3ed9010c9eaa647c730a2ca297c50e7f4ef8da0db4337.

Instance: NOT destroyed. Active instance 37021208 is still running on Vast H200 at $3.908602150537634/hr because operator explicitly objected to deletion. Estimated current run spend at last check: about $2.15; Vast reported uptime about 33 minutes. Cleanup requires explicit operator approval.

Single next action: decide whether to keep the H200 for more testing or explicitly authorize cleanup of instance 37021208.

UTC written: 2026-05-18T18:16:27Z
