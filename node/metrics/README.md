# metrics

Opt-in Prometheus metrics for `pearld`, covering chain state, P2P traffic,
JSON-RPC performance, and the mempool.

No HTTP endpoint is served unless `--metricslisten` is given. Collection itself
is always on, because incrementing a counter costs about as much as checking a
flag; the exception is the per-frame P2P wire hooks, which are only installed
when the endpoint is configured.

## Enabling

Pass `--metricslisten` on the command line or set `metricslisten` in
`pearld.conf`:

```bash
pearld --metricslisten=127.0.0.1:9105
```

Repeat the flag to bind more than one interface:

```bash
pearld --metricslisten=127.0.0.1:9105 --metricslisten=[::1]:9105
```

Metrics are served at `/metrics`, with a health check at `/healthz`.

### Choosing a bind address

The address has to be reachable by whatever scrapes it, so loopback only works
when Prometheus runs on the same host:

| Scrape setup | Bind address |
|--------------|--------------|
| Prometheus or an agent on the same host | `127.0.0.1:9105` |
| Central Prometheus over the network | the host's private/VPC address, or `0.0.0.0:9105` |
| Kubernetes `PodMonitor` / `ServiceMonitor` | `0.0.0.0:9105`, so the pod IP is reachable |

`/metrics` is unauthenticated, so any address beyond loopback is reachable by
anything that can route to the node. Restrict it with a firewall, security
group, or Kubernetes `NetworkPolicy`. pearld logs a warning at startup when the
endpoint is bound beyond loopback.

## Metrics

Everything uses the `pearld_` namespace. The standard Go runtime (`go_*`) and
process (`process_*`) collectors are exported on the same endpoint.

Identity labels such as `node` and `region` are attached by Prometheus at
scrape time rather than by pearld, so every node in a fleet produces identical
metric names.

### Node

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `pearld_info` | Gauge | `version`, `network`, `protocol_version` | Always 1; carries build and network identity as labels |

### Chain

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `pearld_chain_tip_height` | Gauge | | Current chain tip height |
| `pearld_chain_tip_timestamp_seconds` | Gauge | | Unix timestamp of the tip block |
| `pearld_chain_total_transactions` | Gauge | | Total transactions in the main chain |
| `pearld_chain_is_current` | Gauge | | 1 when synced, 0 while syncing |
| `pearld_chain_blocks_connected_total` | Counter | | Blocks connected to the main chain |
| `pearld_chain_blocks_disconnected_total` | Counter | | Blocks disconnected from the main chain |
| `pearld_chain_blocks_accepted_total` | Counter | | Blocks accepted into the blockchain |

Tip freshness is derived from the timestamp rather than exported as an age, so
`time() - pearld_chain_tip_timestamp_seconds` stays correct even if the node
stalls between scrapes.

### P2P

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `pearld_p2p_peers` | Gauge | `direction` | Connected peers |
| `pearld_p2p_peer_connects_total` | Counter | `direction` | Peer connections established |
| `pearld_p2p_peer_disconnects_total` | Counter | `direction` | Peer disconnections |
| `pearld_p2p_peers_banned_total` | Counter | | Peers banned for misbehavior |
| `pearld_p2p_peers_rejected_total` | Counter | `reason` | Rejected connection attempts |
| `pearld_p2p_wire_bytes_total` | Counter | `direction`, `command` | Bytes per wire message type |
| `pearld_p2p_wire_messages_total` | Counter | `direction`, `command` | Wire messages by type |
| `pearld_net_totalbytes_recv_total` | Counter | | Cumulative bytes received |
| `pearld_net_totalbytes_sent_total` | Counter | | Cumulative bytes sent |

`direction` is `inbound` or `outbound`. `reason` is one of `max_peers`,
`banned`, `undesired_agent`, `shutdown`, or `other`.

### JSON-RPC

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `pearld_rpc_requests_total` | Counter | `method`, `result` | Requests processed |
| `pearld_rpc_request_duration_seconds` | Histogram | `method` | Handler execution latency |
| `pearld_rpc_auth_failures_total` | Counter | | Failed authentication attempts |
| `pearld_rpc_websocket_clients` | Gauge | | Connected WebSocket clients |

`result` is `success` or `error`. `method` is restricted to registered RPC
methods; anything else collapses to `unknown` so an unauthenticated caller
cannot inflate series count with arbitrary method names.

### Mempool

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `pearld_mempool_transactions` | Gauge | | Transactions in the mempool |
| `pearld_mempool_bytes` | Gauge | | Mempool size in bytes |
| `pearld_mempool_max_bytes` | Gauge | | Configured mempool size limit |
| `pearld_mempool_last_updated_timestamp_seconds` | Gauge | | When the mempool last changed |

## Scraping

### Standalone hosts

Bind an address the Prometheus server can reach, then point a job at it:

```yaml
scrape_configs:
  - job_name: 'pearld'
    static_configs:
      - targets: ['10.0.1.20:9105']
        labels:
          node: 'validator-1'
          region: 'us-east'
```

For a larger fleet prefer `file_sd_configs`, so targets can be added without
editing and reloading the Prometheus configuration by hand.

### Kubernetes

Bind the wildcard address so the pod IP is reachable, and name the container
port so a monitor can reference it:

```yaml
spec:
  containers:
    - name: pearld
      image: ghcr.io/pearl-research-labs/pearl/pearld:latest
      args:
        - --metricslisten=0.0.0.0:9105
      ports:
        - name: metrics
          containerPort: 9105
```

With the Prometheus Operator installed, a `PodMonitor` completes the wiring:

```yaml
apiVersion: monitoring.coreos.com/v1
kind: PodMonitor
metadata:
  name: pearld
spec:
  selector:
    matchLabels:
      app: pearld
  podMetricsEndpoints:
    - port: metrics
      path: /metrics
      interval: 30s
```

A `ServiceMonitor` referencing the same named port works if the pods sit behind
a `Service`. Pair either with a `NetworkPolicy` admitting only the Prometheus
namespace.

## Design

Metrics live in a private `prometheus.Registry` rather than the default one, so
tests stay isolated and nothing is registered implicitly by importing a
package.

The HTTP server builds its own `http.ServeMux`. It must not reuse
`http.DefaultServeMux`, because `node/btcd.go` imports `net/http/pprof`, which
registers profiling handlers there — serving the default mux would expose
`/debug/pprof` on the metrics port. `server_test.go` guards this by asserting
`/debug/pprof/` returns 404.

Values that can be read cheaply from live state — chain tip, peer counts,
mempool size — are collected at scrape time through `Source`, a struct of
accessor functions supplied by `package main`. There is no background polling
goroutine. Event counts that cannot be sampled, such as block connections and
RPC latency, are pushed from their call sites.

The package does no logging. `Server` reports bound addresses through
`ListenAddrs` and unexpected serve failures through the `Errors` channel, both
consumed by `pearldMain`, so there is no subsystem logger to wire up.

The record helpers deliberately carry no enabled/disabled flag. All metrics are
registered in `init()` regardless, so a flag would save no memory and no
registration work — only an atomic increment, which costs about the same as the
atomic load needed to check it. The one place the distinction pays for itself is
`newPeerConfig` in `node/server.go`, which skips installing the `OnRead`/
`OnWrite` closures entirely when no metrics endpoint is configured, keeping the
per-frame path identical to an uninstrumented build.

Instrumentation lives entirely in `package main` and hooks that already exist
(`peer.Config.Listeners.OnRead`/`OnWrite`, `blockchain.Subscribe`), so shared
packages like `node/blockchain` and `node/peer` gain no Prometheus dependency.
That matters because they are imported by `spv/`, `wallet/`, `dnsseeder/`, and
`coredns-dnsseed/`.
