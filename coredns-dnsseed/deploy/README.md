# Pearl DNS Seeder (CoreDNS Plugin)

A CoreDNS plugin that crawls the Pearl P2P network and serves peer IP addresses
via DNS A/AAAA records.

## Local Development

### Prerequisites

- A running `pearld` node (regtest mode is simplest)
- Docker

### Quick Start

1. Start a regtest node:

```bash
pearld --regtest --listen 127.0.0.1:18444
```

2. Build the CoreDNS image from the repo root:

```bash
docker build -f coredns-dnsseed/deploy/Dockerfile -t pearl-seeder .
```

The binary is built from [`coredns-dnsseed/cmd/coredns`](../cmd/coredns/main.go), a
custom CoreDNS main that compiles in the `dnsseed` plugin plus a curated set of
standard plugins (`bind`, `debug`, `errors`, `health`, `log`, `prometheus`,
`ready`, `reload`). The CoreDNS version comes from the repo's
`go.mod`. To use another standard plugin in a Corefile, add its import and
directive entry to `cmd/coredns/main.go`.

To build the binary natively instead of via Docker:

```bash
go build -o coredns ./coredns-dnsseed/cmd/coredns
```

3. Run with the local dev Corefile:

```bash
docker run --rm --network host \
  -v $(pwd)/coredns-dnsseed/deploy/Corefile.local:/etc/coredns/Corefile \
  pearl-seeder
```

4. Query (records exist only at the zone apex, which is `.` in the dev
   Corefile):

```bash
dig @127.0.0.1 -p 1053 . A +short
```

### Tests

Unit tests (use controlled peers on ephemeral loopback ports):

```bash
go test -race ./coredns-dnsseed/dnsseed
```

End-to-end test — builds `pearld` and the seeder binary, runs both, and
verifies DNS answers over UDP/TCP, the synthesized SOA, negative responses,
readiness, health, and metrics. The same test gates the image build in
[`dnsseed_image.yml`](../../.github/workflows/dnsseed_image.yml):

```bash
go test -tags e2e -v -timeout 10m ./coredns-dnsseed/integration
```

The end-to-end fixture binds the regtest P2P port (18444), so stop any local
regtest node first. Run unit and end-to-end tests as separate invocations.
Before merge, require the existing repository CI gates: `task fmt:go fmt:rust
tidy`, `task build:blockchain`, and `task test:go` (which includes the race detector).

### Configuration

The seeder is configured via a CoreDNS Corefile. See the
[plugin documentation](../dnsseed/README.md) for the `dnsseed` directive
syntax, the fixed peer-serving policy, exported metrics, and readiness
behavior.

### Production Deployment

For production, create a separate Corefile with real domains and bootstrap peers,
and mount it into the container at `/etc/coredns/Corefile`. Example:

```
seed.example.org:1053 {
    dnsseed {
        network mainnet
        bootstrap_peers node1.example.org:44108 node2.example.org:44108
        crawl_interval 15m
    }
    prometheus
    health :8180
    ready :8181
}
```

### Kubernetes

- The image runs as a non-root user without capabilities, so it can only bind
  unprivileged ports. Bind `:1053` (as in the examples) and map it in the
  Service (`port: 53`, `targetPort: 1053`, both UDP and TCP).
- Probes: readinessProbe on `:8181 /ready` (true once the seeder has at least
  one verified address) and livenessProbe on `:8180 /health`.
- Run at least 2 replicas. The address book is in-memory, so a restarted pod
  serves nothing until its first compliant default-port handshake; readiness
  keeps it out of the Service until then, without waiting for gossip.
- The crawler needs outbound access to the P2P port (44108 on mainnet) for
  bootstrap peers and discovered nodes; account for this in egress policies.
- The runtime image is `scratch` (binary plus CA bundle, no shell); use
  `kubectl debug` with an ephemeral container for in-pod troubleshooting.

### Rollout checks

Ship through the existing seeder image pipeline. During rollout, check `/ready`,
`coredns_dnsseed_addresses`, crawl completion logs, and process open-file-descriptor
usage. Logs include probes started, completed verifications, new endpoints
admitted against the 4096-per-crawl limit, dropped over-budget advertisements,
and the outcome of each bootstrap probe.
The 64-worker bound covers pending and open crawler connections, not DNS serving
sockets or every descriptor in the process. A completed or canceled crawl waits
for workers to close their sockets; the in-memory address book remains available
for DNS responses. No new Corefile directives or persistent storage are needed.
