# Pearl Light Client

[![Build Status](https://github.com/pearl-research-labs/pearl/workflows/Build%20and%20Test/badge.svg)](https://github.com/pearl-research-labs/pearl/actions)
[![ISC License](https://img.shields.io/badge/license-ISC-blue.svg)](http://copyfree.org)
[![GoDoc](https://img.shields.io/badge/godoc-reference-blue.svg)](https://pkg.go.dev/github.com/pearl-research-labs/pearl/spv)

A privacy-preserving Pearl light client. It uses compact block filters
(BIP-157/BIP-158) to minimize bandwidth and storage use on the client side,
while preserving privacy and minimizing processor load on full nodes serving
light clients.

## How It Works

The light client synchronizes only block headers and a chain of compact block
filter headers specifying the correct filters for each block. Filters are loaded
lazily and stored in the database upon request; blocks are loaded lazily and not
saved.

## Usage

The client is instantiated using `NewChainService` and then started. Upon start,
the client sets up its database, connects to the P2P network, and becomes
available for queries.

### Queries

There are various types of queries supported by the client. Block headers can be
retrieved by height and hash; full blocks can be fetched from the network using
`GetBlockFromNetwork` by hash. The most useful methods are specifically tailored
to scan the blockchain for data relevant to a wallet:

#### Rescan

`Rescan` allows a wallet to scan the chain for specific TXIDs, outputs, and
addresses. A start and end block may be specified along with other options. If
no end block is specified, the rescan continues until stopped. While a rescan
runs, it notifies the client of each connected and disconnected block. It's
important to note that "recvtx" and "redeemingtx" notifications are only sent
when a transaction is confirmed, not when it enters the mempool.

#### GetUtxo

`GetUtxo` allows a wallet to check that a UTXO exists on the blockchain and has
not been spent. It is **highly recommended** to specify a start block; otherwise,
in the event that the UTXO doesn't exist, the client will download all the
filters back to block 1 searching for it. It returns a `SpendReport` containing
either a `TxOut` including the `PkScript` required to spend the output, or
information about the spending transaction.

### Stopping the Client

Calling `Stop` on the `ChainService` client shuts it down cleanly; the method
blocks until shutdown is complete.

## Installation

```bash
go get github.com/pearl-research-labs/pearl/spv
```

## License

The Pearl light client is licensed under the [copyfree](http://copyfree.org) ISC
License.
