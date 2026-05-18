Accepted path, AKO-012 geometry, build 7199b8db59a0ec54caf72c23245576d3f59f9fb6+dirty: 464.19 hot-loop / 531.42 NoisyGEMM TMAD/s, soft-pool accepted=true, share_id f428aa13-f3f3-47a7-9737-69408106effe, 2026-05-18T18:12:35Z.

Evidence:
- D1 soft-pool accepted: True, share_id f428aa13-f3f3-47a7-9737-69408106effe, outcome_code 0, message: Mining solution verified successfully.
- D2 measured artifact: /srv/pearl/overnight/artifacts/h200-37021208/d2-proofgemm-nohit-20260518T181235Z/direct_gpu_hotloop_benchmark_20260518T181235Z.json
- D2 status: measured_unverified; mining_validity: normal_benchmark_path; proof_validity_status: unknown_missing_canary.
- D2 wall-inclusive throughput: 432.74 TMAD/s; GPU: NVIDIA H200.
- D2 canary artifact: /srv/pearl/overnight/artifacts/h200-37021208/d2-proofgemm-canary-20260518T181141Z/direct_gpu_hotloop_benchmark_20260518T181142Z.json; status verified; proof_validity verified_self.
- D2 full noisy_gemm normal-path attempt failed with: CUDA error: an illegal instruction was encountered at /workspace/pearl-src/miner/pearl-gemm/csrc/gemm/pearl_noisingB_host.h:54.
- D2 P1K165 diagnostic only: 405.08 hot-loop / 438.38 NoisyGEMM TMAD/s; mining_validity invalid_diagnostic_not_mining.
- Build/worktree: /srv/pearl/direct_runs/worktrees/p1k167-ops-integrated-1 at 7199b8db59a0ec54caf72c23245576d3f59f9fb6+dirty.
