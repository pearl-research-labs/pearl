# AKO-012 Closeout: Private Akoya Soft-Pool Acceptance

Verdict: PASS.

The current direct GPU Akoya submit path produced one locally valid proof and submitted it to the private Akoya-compatible soft pool. The server returned a type-4 accepted result.

Key evidence:
- Run id: `ako012_soft_pool_20260517T154755Z_36947111`
- Local artifact root: `/home/bereket/pearl-ops/artifacts/ako012-soft-pool/ako012_soft_pool_20260517T154755Z_36947111`
- Summary: `accepted=true`, `direct_status=accepted`, `server_accepted=true`, `server_message="Mining solution verified successfully"`
- Direct runner: `status=accepted`, `submission.outcome_code=0`, `submission.accepted=true`
- Server result: `accepted=true`, `share_id=9014c1b5-cf51-4e3f-9b41-b4d9b692ae15`
- Difficulty: `share_difficulty=0x1e3fffff` (`507510783`), `network_nbits=0x207fffff`
- H200 attempt: one attempt, proof verify valid, share verify valid
- GPU box auto-destroyed after the run; Vast state showed no active instances and no stopped storage.

Important interpretation:
This proves our direct runner can produce and submit a type-3 PlainProofShare that a private Akoya-compatible endpoint validates and accepts as a type-4 share result. It does not prove public Akoya acceptance at real pool difficulty. Public-pool lottery remains uneconomic at current loop speed.

Follow-up:
The next useful step is not another blind public run. It is either:
1. serializer/protocol comparison against real Akoya if public acceptance is still needed, or
2. performance work on the accepted direct path, with this private soft-pool as the fast correctness gate.
