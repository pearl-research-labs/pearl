# Multisig Architecture

This document explains the multisig branch of the Pearl desktop wallet in a review-friendly way.

It is written for two use cases:

1. Reviewers who want to inspect every file changed by `feature/multisig` relative to `master`.
2. Developers who want the multisig flow from A to Z without jumping across the codebase.

The branch keeps multisig entirely local to the desktop wallet. There is no separate remote multisig backend. The wallet stores vault state, pending PSBTs, and sent activity locally, then syncs them against the Pearl chain and Blockbook where needed.

## Architecture Map

The multisig feature is split into clean layers:

- `src/main/*`
  - Electron main process integration.
  - Wallet process lifecycle.
  - RPC access to Pearl chain data.
  - IPC bridge to the renderer.
- `src/preload/*` and `src/types/*`
  - Renderer bridge exposure.
  - Type-safe API surface.
- `src/renderer/src/chains/*`
  - Pearl address, network, and Taproot multisig primitives.
- `src/renderer/src/crypto/*`
  - Base64, BIP32, and descriptor helpers.
- `src/renderer/src/lib/*`
  - Shared derivation path helpers.
- `src/renderer/src/services/*`
  - Core multisig state machine.
  - Renderer RPC wrappers.
  - Wallet session seed tracking.
- `src/renderer/src/pages/multisig/*`
  - End-user multisig UI.
- `src/renderer/src/pages/*`
  - Existing wallet pages updated to support multisig session and refresh behavior.

## File-by-File Change Map

### Main Process

| File | Change |
| --- | --- |
| `apps/apps/pearl-desktop-wallet/package.json` | Added multisig crypto/runtime dependencies: `@scure/bip32`, `@scure/bip39`, `@scure/btc-signer`, `@noble/curves`, `@noble/hashes`, `vite-plugin-top-level-await`, and `vite-plugin-wasm`. |
| `src/main/clients/blockbook-client.ts` | Added transaction-info lookup used for confirmations and broadcast monitoring. |
| `src/main/ipc/register-wallet-ipc.ts` | Added IPC handlers for multisig key derivation, vault balance, vault UTXOs, tx info, broadcast, and wallet lock/unlock. |
| `src/main/services/manager-service.ts` | Added wallet lifecycle handling needed by multisig: stop/start wallet process, select wallet, lock wallet, force-lock wallet, and seed persistence. |
| `src/main/services/pearl-rpc.ts` | Added the Pearl RPC client with endpoint pooling, raw transaction broadcast, balance walking, and UTXO scanning. |
| `src/main/services/wallet-seed-store.ts` | Added storage for the active wallet seed/session context. |
| `src/main/services/wallet-service/wallet-rpc-methods.ts` | Added wallet RPC wrappers used by multisig, including `deriveMultisigKey()` and wallet lock/unlock helpers. |
| `src/main/services/wallet-service/wallet-service.ts` | Added Blockbook-backed transaction info lookup and kept transaction sorting/filtering behavior. |

### Bridge And Types

| File | Change |
| --- | --- |
| `src/preload/index.ts` | Exposed the new wallet and multisig APIs to the renderer bridge. |
| `src/types/app-bridge.ts` | Updated the TypeScript definition of the bridge to include the new multisig methods. |

### Shared Chain And Crypto Primitives

| File | Change |
| --- | --- |
| `src/renderer/src/chains/pearl/address.ts` | Added Pearl taproot address encode/decode helpers for the `prl` network format. |
| `src/renderer/src/chains/pearl/multisig.ts` | Added Taproot multisig descriptor construction, vault address derivation, and script generation. |
| `src/renderer/src/chains/pearl/network.ts` | Added Pearl network parameters and explorer URL helpers. |
| `src/renderer/src/crypto/base64.ts` | Added base64 helpers for PSBT import/export. |
| `src/renderer/src/crypto/bip32.ts` | Added BIP32 derivation helpers for signer key derivation. |
| `src/renderer/src/crypto/descriptor.ts` | Added cosigner descriptor encode/decode helpers. |
| `src/renderer/src/lib/pearl-multisig-path.ts` | Added the canonical derivation path helper for Pearl multisig signer slots. |

### Renderer Services

| File | Change |
| --- | --- |
| `src/renderer/src/services/multisig.ts` | Added the core multisig engine: vault records, pending/sent state, PSBT import, PSBT validation, signing, broadcasting, txid prediction, on-chain sync, and active removal of broadcasted txs from pending. |
| `src/renderer/src/services/pearl-rpc.ts` | Added renderer-side Pearl RPC wrappers that call the main-process bridge. |
| `src/renderer/src/services/wallet-seed.ts` | Added session wallet seed tracking so the active wallet context survives page transitions. |

### Multisig Pages

| File | Change |
| --- | --- |
| `src/renderer/src/pages/multisig/CreateMultisig.tsx` | Added the vault creation UI and cosigner import/validation flow. |
| `src/renderer/src/pages/multisig/MultisigDashboard.tsx` | Added the vault dashboard with pending txs, sent activity, proposal import, descriptor export, and refresh behavior. |
| `src/renderer/src/pages/multisig/MultisigRelaySign.tsx` | Added the proposal import/sign relay page. |
| `src/renderer/src/pages/multisig/MultisigSend.tsx` | Added the compose-spend PSBT builder page. |
| `src/renderer/src/pages/multisig/MultisigTx.tsx` | Added the detailed pending transaction page with signing, broadcasting, copy actions, txid display, and post-broadcast fallback. |
| `src/renderer/src/pages/multisig/Page.tsx` | Added the multisig landing page that lists vaults and sent/pending summaries. |
| `src/renderer/src/pages/multisig/components/header.tsx` | Added the shared multisig header component. |
| `src/renderer/src/pages/multisig/helpers.ts` | Added shared formatting, clipboard, and descriptor parsing helpers. |

### Existing Wallet Pages Updated For Multisig

| File | Change |
| --- | --- |
| `src/renderer/src/pages/ImportAccount.tsx` | Updated account import to persist session seed state and unlock the wallet. |
| `src/renderer/src/pages/WalletDashboard.tsx` | Updated refresh logic to avoid flicker and stale updates during silent refresh. |
| `src/renderer/src/pages/WalletUnlock.tsx` | Added wallet selection, unlock retry handling, and session cleanup. |
| `src/renderer/src/pages/create-wallet/CreateWallet.tsx` | Cleared stale wallet data before creation and wrote the new session seed state. |

## End-to-End Multisig Flow

### Step 1: Wallet unlock and identity selection

The multisig flow starts with a live wallet process.

- `WalletUnlock.tsx` lets the user choose a wallet and unlock it.
- `ImportAccount.tsx` also unlocks after import.
- `CreateWallet.tsx` clears stale state and records the generated seed for the new wallet.
- `manager-service.ts` is the main process owner of wallet lifecycle.
- `wallet-rpc-methods.ts` provides the underlying RPC methods for unlocking, locking, and deriving the multisig key.

If the wallet has been idle long enough, the address manager can become locked again. In that case multisig derivation fails until the wallet is unlocked again.

### Step 2: Create a multisig vault

`CreateMultisig.tsx` builds the local vault record.

- The user picks:
  - label,
  - threshold,
  - total signer count,
  - vault account,
  - key index.
- The app derives the local x-only pubkey from the wallet.
- The user pastes the other cosigner descriptors or raw x-only pubkeys.
- The UI validates:
  - threshold <= total signers,
  - every cosigner slot is filled,
  - every signer is unique,
  - the local signer belongs to the vault.
- `chains/pearl/multisig.ts` turns the signer set into a Taproot multisig descriptor.
- The vault record stores:
  - sorted pubkeys,
  - the local signer slot,
  - the Pearl address,
  - the derivation metadata,
  - creation timestamp.

### Step 3: Export the cosigner descriptor

The vault can export a descriptor JSON for the other participants.

- It contains the x-only pubkey.
- It contains the origin derivation path.
- It contains the label.

This is the artifact the rest of the signer set imports to stay aligned on the same vault.

### Step 4: Compose a spend

`MultisigSend.tsx` creates a local PSBT draft.

- The user enters:
  - destination address,
  - amount,
  - fee rate.
- `parsePearlAmountToGrains()` converts the display amount into grains.
- `composeVaultSend()`:
  - fetches vault UTXOs from the Pearl chain,
  - sorts UTXOs by value,
  - selects inputs until amount + fee is covered,
  - builds a PSBT with `@scure/btc-signer`,
  - adds destination output,
  - adds change output when needed,
  - computes the fee/change preview.
- `persistComposedAsPending()` stores the draft locally.
- The predicted txid is computed immediately from the PSBT and stored with the pending record.

### Step 5: Import a proposal from another signer

`MultisigRelaySign.tsx` and `services/multisig.ts` accept two input formats:

- proposal artifact JSON,
- raw base64 PSBT.

The import path checks:

- vault address,
- threshold,
- signer set,
- output preview,
- input ownership.

The imported PSBT is only accepted if it belongs to the vault and matches the expected spend shape.

### Step 6: Sign locally

Signing happens in the multisig service and is surfaced through the relay-sign and tx detail pages.

- The wallet signs only vault-owned inputs.
- Any foreign input is rejected.
- Any foreign signer is rejected.
- The PSBT is re-inspected after signing.
- The signer list is updated.
- The exported artifact is regenerated so the next cosigner can continue from the same PSBT.

This branch explicitly fixed the bug where the code previously only inspected the first input.
Now all inputs are checked.

### Step 7: Broadcast the final tx

When the threshold is met, `broadcastPendingTx()` finalizes and broadcasts the transaction.

- The PSBT is validated again before broadcast.
- The tx is finalized only when the threshold is satisfied.
- The raw transaction is broadcast through Pearl RPC.
- The pending record becomes `broadcast`.
- A sent-activity record is written so the tx shows up in the activity list.

### Step 8: Track txs on chain

This branch adds active tx monitoring.

- Every pending tx stores an `expectedTxid`.
- `listPendingTxs()` calls `syncBroadcastTxs()`.
- `syncBroadcastTxs()` checks the blockchain for tracked txids.
- If a tx is found:
  - the pending record is removed,
  - a sent record is created or updated,
  - confirmations and blockhash are refreshed.

This is what allows another participant to broadcast the tx and still have it appear in local activity.

### Step 9: Review the tx detail page

`MultisigTx.tsx` gives the full tx view.

- Copy proposal artifact.
- Copy PSBT.
- Copy txid.
- Add my signature.
- Send transaction.
- Open explorer.
- Display the predicted txid.
- Display signer state and output summary.
- Poll the state in the background so pending txs disappear when they are seen on chain.

### Step 10: See sent activity

`MultisigDashboard.tsx` and `Page.tsx` show the resulting activity.

- Pending transactions are listed separately.
- Sent transactions are listed in activity.
- Confirmations are displayed.
- The UI refreshes silently to avoid flicker.

## Security And Validation Rules

The branch explicitly enforces the following rules:

- A PSBT is not trusted because the first input looks valid.
- Every input must belong to the vault.
- Foreign signers are rejected.
- Output preview must match the composed draft.
- The predicted txid is derived from the PSBT.
- Pending txs are removed when the chain shows they have already been broadcast.
- Sent activity is preserved even if the pending entry disappears.

The Taproot multisig descriptor uses an unspendable internal key, so there is no hidden master key built into the output.

## Review Checklist

If you want to review the branch in a clean order, this is the best path:

1. Read `src/renderer/src/chains/pearl/multisig.ts` to understand the vault descriptor shape.
2. Read `src/renderer/src/services/multisig.ts` to understand state, validation, signing, and sync.
3. Read `src/renderer/src/pages/multisig/MultisigSend.tsx` and `MultisigRelaySign.tsx` to see the two input paths.
4. Read `src/renderer/src/pages/multisig/MultisigTx.tsx` to see the detailed pending tx UX.
5. Read `src/main/ipc/register-wallet-ipc.ts` and `src/main/services/pearl-rpc.ts` to see the bridge to the wallet and chain.
6. Read `src/renderer/src/pages/multisig/MultisigDashboard.tsx` and `src/renderer/src/pages/multisig/Page.tsx` for the resulting activity model.

## Summary

The branch converts the desktop wallet into a full local multisig coordinator:

- create vault,
- share cosigner descriptors,
- compose PSBTs,
- sign proposals,
- broadcast finalized txs,
- monitor chain state,
- remove broadcasted txs from pending,
- keep sent activity visible.

Every file listed above is part of that flow.
