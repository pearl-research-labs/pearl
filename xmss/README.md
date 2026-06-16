# xmss

[![ISC License](https://img.shields.io/badge/license-ISC-blue.svg)](http://copyfree.org)

Go bindings for the XMSS (eXtended Merkle Signature Scheme) post-quantum signature library used by the Pearl network.

XMSS is a hash-based signature scheme that is secure against quantum adversaries. Pearl uses it to implement the `OP_CHECKXMSSSIG` opcode, providing a post-quantum spend path for Taproot addresses.

## Key Properties

- **Post-quantum secure** — security relies solely on hash function assumptions (SHAKE256)
- **Stateful** — each secret key supports at most `MaxSigns` (32) signatures; reusing a `msg_uid` is catastrophic
- **Deterministic** — key generation and signing are fully deterministic from seeds
- **Signature size** — 2340 bytes per signature

## Constants

| Constant | Size | Description |
|---|---|---|
| `PrivateSeedLen` | 64 bytes | Private seed (must remain secret) |
| `PublicSeedLen` | 32 bytes | Public seed |
| `PKLen` | 64 bytes | Public key |
| `SKLen` | 128 bytes | Secret key |
| `MsgLen` | 32 bytes | Message (hash) to sign |
| `SignatureLen` | 2340 bytes | Signature output |
| `MaxSigns` | 32 | Maximum signatures per key pair |

## Usage

```go
// Key generation
pk, sk, err := xmss.Keygen(privateSeed, publicSeed)

// Sign — msg_uid must be unique per key pair (0 <= msg_uid < MaxSigns)
sig, err := xmss.Sign(msgUID, sk, msg)

// Verify
valid := xmss.Verify(pk, msg, sig)
```

> **Warning:** Never sign two messages with the same `msg_uid` under the same key pair. Doing so allows an attacker to forge signatures.

## Building

The C library must be compiled before the Go bindings can be used. Build with the `xmss` tag:

```bash
# Build the static library
make -C xmss

# Build Go code with xmss support
go build -tags xmss ./...
```

Without the `xmss` build tag, a stub implementation is used that returns errors for all operations.

## License

Licensed under the [copyfree](http://copyfree.org) ISC License.
