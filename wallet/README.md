# Oyster

[![Build Status](https://github.com/pearl-research-labs/pearl/workflows/Build%20and%20Test/badge.svg)](https://github.com/pearl-research-labs/pearl/actions)
[![ISC License](https://img.shields.io/badge/license-ISC-blue.svg)](http://copyfree.org)

Oyster is the Pearl wallet daemon. It handles wallet functionality for a single
user, acting as both an RPC client to pearld and an RPC server for wallet
clients and legacy RPC applications.

Public and private keys are derived using the hierarchical deterministic format
described by [BIP0032](https://github.com/bitcoin/bips/blob/master/bip-0032.mediawiki).
Unencrypted private keys are not supported and are never written to disk. Oyster
uses the `m/44'/<coin type>'/<account>'/<branch>/<address index>` HD path for
all derived addresses, as described by
[BIP0044](https://github.com/bitcoin/bips/blob/master/bip-0044.mediawiki).

Due to the sensitive nature of public data in a BIP0032 wallet, Oyster provides
the option of encrypting not just private keys, but public data as well. This is
intended to thwart privacy risks where a wallet file is compromised without
exposing all current and future addresses (public keys) managed by the wallet.

Oyster connects to a pearld instance for blockchain queries and notifications
over websockets. An SPV mode is also available via the
[Pearl light client](https://github.com/pearl-research-labs/pearl/tree/master/spv).

Wallet clients can use one of two RPC servers:

  1. A legacy JSON-RPC server

     This server is enabled by default.

  2. An experimental gRPC server

     The gRPC server uses a new API built for Oyster, but the API is not
     stabilized and the server is feature gated behind a config option
     (`--experimentalrpclisten`). The gRPC server is documented
     [here](./rpc/documentation/README.md).

For day-to-day use from a terminal, [oystercli](cmd/oystercli/) provides an
interactive menu-driven client over the JSON-RPC server, including guided
wallet creation and troubleshooting tools.

## Requirements

- [Go](https://golang.org) 1.26 or newer
- [Task](https://taskfile.dev) runner

## Building

From the repository root:

```bash
task build:blockchain
```

This builds Oyster (along with pearld, prlctl, and oystercli) into `bin/`.

## Getting Started

1. Start pearld:

```bash
pearld -u rpcuser -P rpcpass
```

2. Create a wallet:

```bash
oyster -u rpcuser -P rpcpass --create
```

3. Start Oyster:

```bash
oyster -u rpcuser -P rpcpass
```

See [sample-oyster.conf](sample-oyster.conf) for the full list of options.

## Issue Tracker

The [integrated GitHub issue tracker](https://github.com/pearl-research-labs/pearl/issues)
is used for this project.

## License

Oyster is licensed under the [copyfree](http://copyfree.org) ISC License.
