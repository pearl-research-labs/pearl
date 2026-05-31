"""pearl-stratum-srv — solo Pearl mining pool.

Exposes a Stratum-v1 TCP listener that the production `alpha-miner` and our own
`vllm-miner` already speak. Reuses the existing `pearl-gateway` services for
template fetching (PearlNodeClient + WorkCache) and block submission
(SubmissionService). The only new code is the protocol-translation layer.

Designed for SOLO operation on a trusted LAN:
  - no per-user accounting / payouts (coinbase → single mining address)
  - no `pearl.challenge` DDoS handshake (LAN-only, no untrusted clients)
  - no TLS (LAN-only; add stunnel/Caddy in front if you ever expose it)
  - no vardiff (we accept every share for liveness; only blocks matter)

Public API kept minimal so the protocol/job modules can be imported and
tested without the full pearl-gateway dependency tree.
"""
