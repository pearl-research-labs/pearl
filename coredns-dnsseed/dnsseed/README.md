# dnsseed

## Name

*dnsseed* - serves verified Pearl full-node addresses for P2P network bootstrapping.

## Description

The *dnsseed* plugin crawls the Pearl P2P network in the background and answers
A and AAAA queries with the IP addresses of verified full nodes, making CoreDNS
act as a DNS seeder for new nodes joining the network.

A peer is served only when it completes the P2P version handshake and satisfies
the serving policy: it speaks at least the minimum wire protocol version,
carries the required full-node service bits, reports a chain height at or above
the network's latest checkpoint (which weeds out nodes stranded on a pre-fork
chain), and listens on the network's default P2P port. The policy is fixed in
code, not configurable.

Each crawl re-verifies the served addresses, sends `getaddr` to the live peers,
and verifies addresses received through `addrv2`. Newly verified peers are
queried during the same crawl, so discovery continues until no candidate
address arrives for 30 seconds or the crawl interval expires. Peer connections
are then closed; the verified address book remains in memory for DNS serving
and the next crawl.

A served peer that fails re-verification on two consecutive crawls stops being
served and enters a three-hour cooldown; a gossiped address that fails its
first verification enters the cooldown immediately. Cooled-down addresses are
not re-dialed until the cooldown expires, after which gossip naturally
rediscovers and re-verifies them. Should the address book empty out entirely
(every known peer died, or the instance lost egress), the crawler starts over
from the bootstrap peers, which are always dialed regardless of cooldown.

Records exist only at the zone apex: A and AAAA queries answer with a random
sample of up to 25 peer addresses (TTL 3600), SOA queries answer with a
synthesized SOA record, other query types receive an empty NOERROR, and names
below the apex are NXDOMAIN. Negative responses carry the SOA in the AUTHORITY section so
resolvers can cache them (RFC 2308) instead of re-querying continuously. The
SOA is synthesized - the primary name server is the zone apex and the mailbox
is `hostmaster.<zone>` - since the zone has no real primary or mailbox.

Startup performs no network I/O; bootstrapping retries in the background and
readiness gates traffic until at least one verified address is available.

## Compilation

This plugin is compiled into the Pearl seeder binary by
[`coredns-dnsseed/cmd/coredns`](../cmd/coredns/main.go), which pins the plugin
set and directive order. To include it in a stock CoreDNS build instead, add to
[plugin.cfg](https://github.com/coredns/coredns/blob/master/plugin.cfg):

~~~ txt
dnsseed:github.com/pearl-research-labs/pearl/coredns-dnsseed/dnsseed
~~~

## Syntax

~~~ txt
dnsseed {
    network NETWORK
    bootstrap_peers PEER...
    crawl_interval DURATION
    min_protocol_version VERSION
}
~~~

* `network` **NETWORK** - required. The Pearl network to crawl: `mainnet`,
  `testnet`, `testnet2`, `regtest`, `signet` or `simnet`.
* `bootstrap_peers` **PEER...** - required. One or more `host:port` peers used
  to join the P2P network. Unreachable peers are retried every 30 seconds.
* `crawl_interval` **DURATION** - how often to re-crawl the network and the
  maximum duration of one crawl. Defaults to `15m`.
* `min_protocol_version` **VERSION** - the lowest wire protocol version
  served. Defaults to the node's own peering floor
  (`peer.MinAcceptableProtocolVersion`); values below it are rejected since
  such peers cannot complete the handshake anyway. Raise it during a protocol
  upgrade window to stop handing pre-upgrade peers to bootstrapping nodes
  before the node floor rises. CoreDNS substitutes environment variables in
  the Corefile, so the floor can be set per deployment with
  `min_protocol_version {$SEEDER_MIN_PROTOCOL_VERSION}`.

## Metrics

If monitoring is enabled (via the *prometheus* plugin) the following metrics
are exported:

* `coredns_dnsseed_request_count_total{server}` - count of queries handled by
  the plugin.
* `coredns_dnsseed_addresses` - number of verified peer addresses available to
  serve, updated after each crawl.

The `server` label indicates which server handled the request, see the
*metrics* plugin for details.

## Ready

This plugin reports readiness to the *ready* plugin once the crawler has at
least one verified address to serve. A freshly started instance is not ready
until it has bootstrapped and booked a peer.

## Examples

Serve seeder answers for `seed.example.org` on an unprivileged port, with
probes and metrics:

~~~ corefile
seed.example.org:1053 {
    dnsseed {
        network mainnet
        bootstrap_peers node1.example.org:44108 node2.example.org:44108
    }
    prometheus
    health :8180
    ready :8181
}
~~~

Local regtest development against a single node:

~~~ corefile
.:1053 {
    dnsseed {
        network regtest
        bootstrap_peers 127.0.0.1:18444
        crawl_interval 2m
    }
    log
}
~~~

## Differences from bitcoin-seeder

Compared to the reference [sipa/bitcoin-seeder](https://github.com/sipa/bitcoin-seeder),
this plugin deliberately omits:

* **Persistence** - the address book is in-memory. Readiness gating plus
  multiple replicas cover restarts; a fresh instance rebuilds its book in one
  bootstrap and crawl.
* **Reliability windows** - the reference scores peers over exponential
  2h/8h/1d/1w statistics windows with ban and ignore formulas. Two-strike
  tolerance plus a fixed cooldown protects a 2000-peer book without the
  bookkeeping.
* **Service-bit subdomain filtering** (`x1.`, `x9.`, ...) - `pearld` never
  queries filtered seed names; no Pearl network sets `DNSSeed.HasFiltering`.
* **Crawling deprecated nodes for gossip** - the reference keeps requesting
  addresses from nodes too old to serve. That conflicts with the hardfork
  policy here: deprecated peers are rejected at the version handshake.
* **NS records and operator-named SOA** - both need an operator-supplied
  hostname and mailbox (the reference's `-n`/`-m` flags). Delegation already
  lives in the parent zone; operators who want NS or a custom SOA can compile
  in the *template* plugin.
* **Tor/proxy support and a hand-rolled DNS layer** - transport concerns
  (EDNS0, TCP, truncation), probes, metrics, and reload belong to CoreDNS;
  delegating them is the point of this design.

An alternative design - the crawler writing a zone file for the *file* or
*auto* plugin to serve - was rejected: it loses per-query answer rotation (no
stock plugin subsets records; *loadbalance* only reorders), needs a writable
volume in the otherwise read-only `scratch` image, and adds reload latency
plus zone-serial bookkeeping.

## Also See

The [deployment guide](../deploy/README.md) for Docker and Kubernetes usage,
and the [CoreDNS manual](https://coredns.io/manual).
