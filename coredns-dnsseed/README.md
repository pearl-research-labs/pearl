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

3. Run with the local dev Corefile:

```bash
docker run --rm --network host \
  -v $(pwd)/coredns-dnsseed/deploy/Corefile.local:/etc/coredns/Corefile \
  pearl-seeder
```

4. Query:

```bash
dig @127.0.0.1 -p 1053 localhost A +short
```

### Configuration

The seeder is configured via a CoreDNS Corefile. The `dnsseed` block supports:

| Directive         | Default | Description                              |
| ----------------- | ------- | ---------------------------------------- |
| `network`         | —       | Pearl network: mainnet, testnet, testnet2, regtest, signet, simnet |
| `bootstrap_peers` | —       | Space-separated `host:port` list         |
| `crawl_interval`  | `15m`   | How often to re-crawl the network        |
| `record_ttl`      | `3600`  | DNS record TTL in seconds                |
| `max_answers`     | `25`    | Max IPs per DNS response                 |

### Production Deployment

For production, create a separate Corefile with real domains and bootstrap peers,
and mount it into the container at `/etc/coredns/Corefile`. Example:

```
mainnet.seeder.example.com {
    dnsseed {
        network mainnet
        bootstrap_peers node1.example.com:44108 node2.example.com:44108
        crawl_interval 15m
        record_ttl 600
    }
    prometheus
    health
}
```
