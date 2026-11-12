# coredns-dnsseed

A CoreDNS plugin that serves Pearl network peer addresses via DNS.

## Components

- **crawler** — crawls the Pearl P2P network and maintains an address book of reachable peers
- **dnsseed** — CoreDNS plugin that answers DNS queries with addresses from the crawler

## Building



go build ./...
go test ./...

