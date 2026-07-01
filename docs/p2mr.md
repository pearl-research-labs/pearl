# P2MR — Pay-to-Merkle-Root (SegWit v2) Addresses

P2MR is Pearl's **witness version 2** output type: a bech32m address / script that
commits directly to the Merkle root of a tapscript tree, with **no internal key**.
It is Pearl's implementation of BIP 360 and serves as the network's
**quantum-resistant** spend path (script-path only, typically an XMSS leaf).

This document is the reference for anyone who needs to **understand, decode,
display, index, integrate with, or spend** P2MR outputs.

> **Naming note.** Pearl calls this type "P2MR / Pay-to-Merkle-Root" and tracks it
> under "BIP 360". Upstream Bitcoin's BIP 360 draft is titled "Pay to Quantum
> Resistant Hash (P2QRH)". Pearl implemented its own concrete variant under that
> number; do not assume byte-for-byte parity with the upstream draft. **This
> document (and the Pearl source) is the authoritative spec — there is no separate
> written standard.**

---

## TL;DR

| Property                | Value                                                        |
| ----------------------- | ------------------------------------------------------------ |
| Witness version         | **2**                                                        |
| Address encoding        | **bech32m** (BIP 350) — v0 is rejected                       |
| HRP (mainnet)           | `prl` → addresses look like `prl1z…`                         |
| Witness program         | **exactly 32 bytes** — the Merkle root of a tapscript tree   |
| scriptPubKey            | `OP_2 <32-byte root>` = `0x52 0x20 ‖ root` (**34 bytes**)    |
| Key-path spend          | **None.** Script-path only (no internal key)                 |
| Script class name       | `witness_v2_merkleroot`                                      |
| Consensus status        | Standard, fully validated (alongside P2TR and OP_RETURN)     |

**To just decode + display a P2MR address you only need two rules:**

1. `scriptPubKey` → address: match `0x52 0x20 <32 bytes>`, then bech32m-encode
   `(hrp="prl", witver=2, program=<32 bytes>)`.
2. address → `scriptPubKey`: bech32m-decode, first 5-bit group is the version
   (`2`), regroup the rest 5→8 bits into a 32-byte program, emit `OP_2 <program>`.

You do **not** need any quantum / XMSS logic to decode or index these.

---

## 1. Why P2MR exists

Taproot (v1) outputs are protected by a single 32-byte Schnorr **output key**. That
key-path is a secp256k1 public key and is therefore vulnerable to a
cryptographically-relevant quantum computer (Shor's algorithm recovers the private
key from the public key).

P2MR removes that exposure. There is **no internal key and no key-path spend** — the
32-byte witness program is *directly* the Merkle root of a script tree, and the only
way to spend is to reveal a leaf script and satisfy it. When the leaf is a
hash-based post-quantum scheme (Pearl uses **XMSS** via `OP_CHECKXMSSSIG`), the
whole output's security reduces to hash-function assumptions, which are considered
quantum-safe.

So P2MR is best understood as **"script-path-only Taproot for post-quantum
spending."** It reuses Taproot's BIP 341/342 tapscript tree and tagged-hash
machinery verbatim; the only differences are (a) no internal key / key-path, and
(b) the witness program is compared to the reconstructed Merkle root directly
instead of via a tweak.

---

## 2. Address format

### 2.1 Encoding

P2MR addresses are **bech32m** (BIP 350). Pearl's segwit decoder rejects witness v0
entirely and requires bech32m for every supported version (v1 Taproot, v2 P2MR).
A v2 string that carries a plain-bech32 checksum is invalid.

### 2.2 Human-readable part (HRP)

| Network              | HRP    | Example prefix |
| -------------------- | ------ | -------------- |
| Mainnet              | `prl`  | `prl1z…`       |
| Testnet / Signet     | `tprl` | `tprl1z…`      |
| Regtest / Simnet     | `rprl` | `rprl1z…`      |

The BIP-44 coin type for mainnet is `808276` (ASCII "PRL", `0xC5554`).

### 2.3 Witness version character

In bech32(m) the first data symbol after the `1` separator encodes the witness
version directly (it is not part of the 5→8 bit regrouping). Using the bech32
charset `qpzry9x8gf2tvdw0s3jn54khce6mua7l`:

| Char after `<hrp>1` | Witness version | Type            |
| ------------------- | --------------- | --------------- |
| `q`                 | 0               | (rejected)      |
| `p`                 | 1               | Taproot (P2TR)  |
| `z`                 | 2               | **P2MR**        |

So a Pearl mainnet P2MR address always starts with **`prl1z`**, and Taproot always
starts with `prl1p`.

### 2.4 Program length

The witness program is **exactly 32 bytes** (the Merkle root). This is enforced on
address construction, on script recognition, and by the script engine. Programs of
any other length are not valid P2MR.

### 2.5 Worked example (test vector)

The mainnet address:

```
prl1zqu04ax80tw03rs0v90rel24a77z722yyrdum43fcdvdtfgug4svquahpxa
```

decodes as:

| Field         | Value                                                              |
| ------------- | ------------------------------------------------------------------ |
| HRP           | `prl` (mainnet)                                                    |
| Checksum      | bech32m                                                            |
| Witness ver   | `2` (P2MR)                                                         |
| Program (32B) | `071f5e98ef5b9f11c1ec2bc79faabdf785e528841b79bac5386b1ab4a388ac18` |
| scriptPubKey  | `5220071f5e98ef5b9f11c1ec2bc79faabdf785e528841b79bac5386b1ab4a388ac18` (34 bytes) |

---

## 3. scriptPubKey format

A P2MR output script is a standard SegWit witness program with version bumped to 2:

```
OP_2 OP_DATA_32 <32-byte merkle root>
0x52 0x20       <32 bytes>
```

Total length **34 bytes**. There is nothing Pearl-specific about the *encoding*:
`OP_2` is the standard numeric opcode `0x52`, and `OP_DATA_32` (`0x20`) is the
canonical minimal push of 32 bytes — the exact same shape as Taproot's
`OP_1 OP_DATA_32 <key>`, only the version opcode differs.

The Pearl-specific part is the *meaning* (the pushed 32 bytes are a Merkle root, not
a tweaked key) and the *spend rules* (§5).

---

## 4. Script classification & consensus rules

Pearl recognizes exactly three standard output types at the consensus level; every
other script shape is `NonStandardTy` and is **rejected** during block validation:

| Script class            | scriptPubKey                     |
| ----------------------- | -------------------------------- |
| `WitnessV1TaprootTy`    | `OP_1 <32-byte key>`             |
| `WitnessV2MerkleRootTy` | `OP_2 <32-byte merkle root>`     |
| `NullDataTy`            | `OP_RETURN <data>`               |

Recognition is a pure structural check: length `== 34`, `script[0] == OP_2`,
`script[1] == OP_DATA_32`, and bytes `[2:34]` are the program. Because only v1 and
v2 are classified as standard, witness versions 3–16 never appear in valid mainnet
outputs (they would be non-standard and rejected).

---

## 5. Spend semantics (only needed if you sign, not to index)

> If you only decode/display P2MR outputs, skip this section. It is included so the
> document fully describes the type.

P2MR is **script-path only**. A valid spend reveals a leaf of the committed tree
plus a Merkle inclusion proof, and the engine checks the reconstructed root equals
the witness program.

### 5.1 Witness stack layout

```
[ <leaf input args…>, <revealed leaf script>, <control block> ]
```

- The **last** element is the control block.
- The **second-to-last** element is the revealed tapscript leaf.
- Everything before that is the input stack the leaf consumes.

A single-element witness is explicitly rejected (`ErrMerkleRootNoKeyPathSpend`):
unlike Taproot, there is no key-path spend.

For the canonical XMSS leaf `<xmss_pubkey> OP_CHECKXMSSSIG`, the stack is:

```
[ sig1, sig2, sig3, sig4, sig5, <leaf script>, <control block> ]
```

(The XMSS signature is 2340 bytes, split into five 468-byte chunks to fit the
520-byte stack-element limit; the pubkey is embedded in the leaf script itself.)

### 5.2 Control block format

The P2MR control block differs from Taproot's — there is **no 32-byte internal
key**:

```
control block = c[0] ‖ node_0 ‖ node_1 ‖ … ‖ node_{m-1}
```

- `c[0]`: leaf-version-and-parity byte. Leaf version is `c[0] & 0xFE`; the low bit
  is the parity bit and **must be 1** (`ErrMerkleRootControlBlockInvalidParity`).
  For the base tapscript leaf version `0xC0`, `c[0] == 0xC1`.
- `node_i`: the Merkle inclusion proof, zero or more 32-byte sibling hashes.
- Total size: `1 + 32·m` bytes. Min 1 byte; max `1 + 32·128`.

Only the base tapscript leaf version (`0xC0`, BIP 342 semantics) is executed.

### 5.3 Verification algorithm

1. Parse and validate the control block (size multiple of 32 after the first byte,
   parity bit set).
2. Recompute the Merkle root from the revealed leaf and the inclusion proof, using
   the **same tagged hashes as Taproot**: `TapLeaf` for the leaf, `TapBranch`
   (lexicographically ordered) for each proof step.
3. Compare the recomputed root **directly** to the 32-byte witness program (no
   internal-key tweak, no parity adjustment). Mismatch →
   `ErrMerkleRootMerkleProofInvalid`.
4. Execute the revealed leaf under the BIP 342 tapscript rules.

### 5.4 XMSS leaf / `OP_CHECKXMSSSIG`

The post-quantum leaf script is `<xmss_pubkey> OP_CHECKXMSSSIG`. Key facts:

- XMSS parameters: SHAKE256, `n = 64`; **public key 64 bytes**, **signature 2340
  bytes**.
- `OP_CHECKXMSSSIG` is valid **only in tapscript context**; anywhere else it errors
  like a reserved opcode.
- Stack: `[… sig1 sig2 sig3 sig4 sig5 pubkey] → [… bool]`.
- XMSS is a stateful hash-based scheme: **each key can sign at most 32 times**
  (`MaxSigns = 32`). This is a signer-side operational constraint; it does not
  affect decoding or indexing.

---

## 6. Comparison: Taproot (v1) vs P2MR (v2)

| Aspect                | Taproot (P2TR, v1)                   | P2MR (v2)                          |
| --------------------- | ------------------------------------ | ---------------------------------- |
| Witness version       | 1 (`prl1p…`)                         | 2 (`prl1z…`)                       |
| scriptPubKey          | `OP_1 <32B>`                         | `OP_2 <32B>`                       |
| 32-byte program is…   | tweaked Schnorr output key           | Merkle root of script tree         |
| Key-path spend        | Yes (Schnorr sig)                    | **No**                             |
| Script-path spend     | Yes (control block has internal key) | Yes (control block has **no** key) |
| Control block byte[0] | leaf version + key-y parity          | leaf version + parity (**must 1**) |
| Quantum exposure      | key-path is secp256k1 (exposed)      | none (hash-based leaf)             |
| Encoding              | bech32m                              | bech32m                            |

Both reuse the identical BIP 341/342 tapscript tree and tagged-hash construction.

---

## 7. Integration guide (decode / index / display)

### 7.1 The only two rules you need

**scriptPubKey → address** (e.g. to display the recipient of an output):

```text
if len(spk) == 34 and spk[0] == 0x52 (OP_2) and spk[1] == 0x20 (OP_DATA_32):
    program = spk[2:34]              # 32 bytes
    address = bech32m_encode(hrp="prl", data=[2] + convertbits(program, 8, 5, pad=True))
```

**address → scriptPubKey** (e.g. to build an output that pays a P2MR address):

```text
hrp, data, spec = bech32_decode(address)     # spec must be bech32m
assert hrp in {"prl","tprl","rprl"}
version = data[0]                              # == 2
program = convertbits(data[1:], 5, 8, pad=False)
assert len(program) == 32
spk = bytes([0x52, 0x20]) + program            # OP_2 OP_DATA_32 <program>
```

Note the version byte (`data[0]`) is **not** part of the 5→8 regrouping; strip it
first, then convert the remaining symbols.

### 7.2 Implementation notes

P2MR reuses the **standard SegWit v1+ bech32m** machinery — the same encoding path
as Taproot. Any wallet or library that already handles Taproot (v1) can support
P2MR by **allowing witness version 2** with a 32-byte program; no new cryptography
is required to decode, validate, or display these addresses.

- Use a **bech32m** codec (BIP 350), not plain bech32. The witness version is the
  first 5-bit symbol and is **excluded** from the 5↔8 bit regrouping.
- **Reject witness v0** and enforce the **32-byte** program length.
- The `scriptPubKey` is a canonical minimal push: `OP_2 OP_DATA_32 <program>` —
  build and match it exactly as you already do for Taproot's `OP_1 OP_DATA_32 <key>`.

If your Taproot support hardcodes version 1 (some libraries do), the only change
needed is to accept version 2 through the same generic bech32m path. A canonical
implementation lives in the Pearl node source — see §8.

---

## 8. Source reference index

| Concern                              | File(s) in the Pearl repo                                  |
| ------------------------------------ | ---------------------------------------------------------- |
| Address encode/decode (bech32m)      | `node/btcutil/address.go`, `node/btcutil/bech32/bech32.go` |
| Network HRPs, coin type              | `node/chaincfg/params.go`                                  |
| scriptPubKey build/recognize, classes| `node/txscript/standard.go`, `node/txscript/pkscript.go`   |
| Witness dispatch, spend verification | `node/txscript/engine.go`                                  |
| Control block, Merkle commitment     | `node/txscript/taproot.go`                                 |
| `OP_CHECKXMSSSIG` opcode             | `node/txscript/opcode.go`                                  |
| P2MR error codes                     | `node/txscript/error.go`                                   |
| Consensus standardness               | `node/blockchain/validate.go`                              |
| Wallet address type / key scopes     | `wallet/waddrmgr/address.go`, `wallet/waddrmgr/scoped_manager.go` |
| Spending P2MR (author)               | `wallet/wallet/txauthor/author.go`                         |
| Size/weight constants                | `wallet/wallet/txsizes/size.go`                            |
| XMSS signature scheme                | `xmss/`                                                    |
| Reference indexer parser             | Blockbook `bchain/coins/pearl/pearlparser.go`              |
