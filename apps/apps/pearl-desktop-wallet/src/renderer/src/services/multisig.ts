import * as btc from "@scure/btc-signer";
import { decodeTaprootAddress, encodeTaprootAddress } from "../chains/pearl/address";
import { pearlParams, type PearlNetwork } from "../chains/pearl/network";
import { pearlMultisigPath } from "../lib/pearl-multisig-path";
import { base64ToBytes, bytesToBase64 } from "../crypto/base64";
import {
  bytesToHex,
  encodePubkeyDescriptor,
  hexToBytes,
  parsePubkeyDescriptor,
  type PearlMultisigPubkeyDescriptor,
} from "../crypto/descriptor";
import {
  vaultDescriptorFromPubkeys,
  type VaultDescriptor,
} from "../chains/pearl/multisig";
import {
  broadcastPearlTx,
  type PrlUtxo,
} from "./pearl-rpc";

const STORAGE_KEY = "pearl-desktop-wallet.multisig-v1";
const PEARL_DEFAULT_FEERATE_SATS_PER_VBYTE = 2n;
const PER_INPUT_VBYTES_MULTISIG = 100n;
const PER_P2TR_OUTPUT_VBYTES = 43n;
const FIXED_OVERHEAD_VBYTES = 11n;
const DUST_LIMIT_GRAINS = 546n;
const EMPTY_STATE: PersistedState = { vaults: [], pendingTxs: [], sentTxs: [] };

export interface VaultRecord {
  id: string;
  version: 1;
  label: string;
  threshold: number;
  total: number;
  sortedPubkeysHex: string[];
  myPubkeyHex: string;
  myOriginPath: string;
  myVaultAccount: number;
  myKeyIndex: number;
  pearlAddress: string;
  network: "mainnet";
  createdAt: number;
}

export interface VaultPendingTxRecord {
  id: string;
  vaultId: string;
  psbtBase64: string;
  expectedTxid?: string;
  signersHex: string[];
  createdAt: number;
  updatedAt: number;
  status: "drafting" | "ready" | "broadcast" | "failed";
  txid?: string;
  preview: {
    destination: string;
    amountGrains: string;
    feeGrains: string;
    changeGrains: string;
    inputCount: number;
  };
}

export interface VaultSentTxRecord {
  id: string;
  vaultId: string;
  txid: string;
  destination: string;
  amountGrains: string;
  feeGrains: string;
  changeGrains: string;
  time: number;
  updatedAt: number;
  confirmations: number;
  blockhash: string;
  status: "broadcast" | "confirmed";
}

interface PersistedState {
  vaults: VaultRecord[];
  pendingTxs: VaultPendingTxRecord[];
  sentTxs: VaultSentTxRecord[];
}

export interface ProposalArtifactV1 {
  version: 1;
  kind: "pearl-multisig-proposal";
  id: string;
  vaultAddress: string;
  threshold: number;
  sortedPubkeysHex: string[];
  psbtBase64: string;
  preview: VaultPendingTxRecord["preview"];
  createdAt: number;
  updatedAt: number;
}

interface BroadcastTxInfo {
  txid: string;
  confirmations: number;
  blockhash: string;
  time: number;
}

function isLikelyPsbtBase64(text: string): boolean {
  try {
    const bytes = base64ToBytes(text.trim());
    return (
      bytes.length >= 5 &&
      bytes[0] === 0x70 &&
      bytes[1] === 0x73 &&
      bytes[2] === 0x62 &&
      bytes[3] === 0x74 &&
      bytes[4] === 0xff
    );
  } catch {
    return false;
  }
}

function pendingFromPsbt(opts: {
  vault: VaultRecord;
  psbtBase64: string;
}): VaultPendingTxRecord {
  const tx = btc.Transaction.fromPSBT(base64ToBytes(opts.psbtBase64));
  assertPsbtInputsBelongToVault(tx, opts.vault);
  const info = inspectPsbt(opts.psbtBase64, opts.vault.threshold, opts.vault.sortedPubkeysHex);
  const dest = info.outputs[0];
  if (!dest || !dest.address) {
    throw new Error("E_PROPOSAL_PSBT_UNSUPPORTED: raw PSBT must have a Pearl destination output");
  }

  const change = info.outputs[1];
  const changeGrains =
    change && change.address === opts.vault.pearlAddress ? change.amountGrains.toString() : "0";
  const expectedTxid = deriveExpectedTxid(opts.psbtBase64);

  return {
    id: newUuid(),
    vaultId: opts.vault.id,
    psbtBase64: opts.psbtBase64,
    expectedTxid,
    signersHex: info.signersHex,
    createdAt: Date.now(),
    updatedAt: Date.now(),
    status: info.thresholdMet ? "ready" : "drafting",
    preview: {
      destination: dest.address,
      amountGrains: dest.amountGrains.toString(),
      feeGrains: info.feeUnknown ? "0" : info.feeGrains.toString(),
      changeGrains,
      inputCount: info.inputCount,
    },
  };
}

function deriveExpectedTxid(psbtBase64: string): string {
  const tx = btc.Transaction.fromPSBT(base64ToBytes(psbtBase64));
  return tx.id;
}

function assertPsbtInputsBelongToVault(tx: btc.Transaction, vault: VaultRecord): void {
  const vaultOutputScriptHex = bytesToHex(descriptorFromRecord(vault).outputScript);
  for (let i = 0; i < tx.inputsLength; i++) {
    const input = tx.getInput(i) as {
      witnessUtxo?: { script: Uint8Array; amount: bigint };
    };
    if (!input.witnessUtxo || bytesToHex(input.witnessUtxo.script) !== vaultOutputScriptHex) {
      throw new Error("E_MULTISIG_PSBT_FOREIGN_INPUT");
    }
  }
}

function newUuid(): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}-${Math.random().toString(36).slice(2)}`;
}

function normalizePersistedState(value: unknown): PersistedState {
  if (!value || typeof value !== "object") {
    return EMPTY_STATE;
  }
  const parsed = value as Partial<PersistedState>;
  return {
    vaults: Array.isArray(parsed.vaults) ? (parsed.vaults as VaultRecord[]) : [],
    pendingTxs: Array.isArray(parsed.pendingTxs) ? (parsed.pendingTxs as VaultPendingTxRecord[]) : [],
    sentTxs: Array.isArray(parsed.sentTxs) ? (parsed.sentTxs as VaultSentTxRecord[]) : [],
  };
}

function readLegacyLocalState(): PersistedState | null {
  if (typeof localStorage === "undefined") {
    return null;
  }
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    return normalizePersistedState(JSON.parse(raw));
  } catch {
    return null;
  }
}

async function loadState(): Promise<PersistedState> {
  if (typeof window === "undefined" || !window.appBridge?.wallet?.getMultisigState) {
    return EMPTY_STATE;
  }
  const state = normalizePersistedState(await window.appBridge.wallet.getMultisigState());
  if (state.vaults.length === 0 && state.pendingTxs.length === 0 && state.sentTxs.length === 0) {
    const legacy = readLegacyLocalState();
    if (legacy && (legacy.vaults.length > 0 || legacy.pendingTxs.length > 0 || legacy.sentTxs.length > 0)) {
      await saveState(legacy);
      return legacy;
    }
  }
  return state;
}

async function saveState(state: PersistedState): Promise<void> {
  if (typeof window === "undefined" || !window.appBridge?.wallet?.saveMultisigState) {
    return;
  }
  await window.appBridge.wallet.saveMultisigState(state);
}

async function updateState(mutator: (state: PersistedState) => PersistedState): Promise<PersistedState> {
  const next = mutator(await loadState());
  await saveState(next);
  return next;
}

function upsertSentTx(state: PersistedState, rec: VaultSentTxRecord): PersistedState {
  return {
    ...state,
    sentTxs: [rec, ...state.sentTxs.filter((p) => p.id !== rec.id && p.txid !== rec.txid)],
  };
}

function sentFromPending(
  pending: VaultPendingTxRecord,
  txid: string,
  txInfo?: BroadcastTxInfo | null,
): VaultSentTxRecord {
  const now = Date.now();
  return {
    id: pending.id,
    vaultId: pending.vaultId,
    txid,
    destination: pending.preview.destination,
    amountGrains: pending.preview.amountGrains,
    feeGrains: pending.preview.feeGrains,
    changeGrains: pending.preview.changeGrains,
    time: txInfo?.time ?? now,
    updatedAt: now,
    confirmations: txInfo?.confirmations ?? 0,
    blockhash: txInfo?.blockhash ?? "",
    status: txInfo && txInfo.confirmations > 0 ? "confirmed" : "broadcast",
  };
}

function sameHexLists(a: string[], b: string[]): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (a[i]!.toLowerCase() !== b[i]!.toLowerCase()) return false;
  }
  return true;
}

function parseProposalArtifactJson(text: string): ProposalArtifactV1 {
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch (err) {
    throw new Error(`E_PROPOSAL_ARTIFACT_PARSE: ${err instanceof Error ? err.message : String(err)}`);
  }
  if (!parsed || typeof parsed !== "object") {
    throw new Error("E_PROPOSAL_ARTIFACT_BAD_SHAPE");
  }
  const o = parsed as Record<string, unknown>;
  const preview = o.preview as Record<string, unknown> | undefined;
  if (
    o.kind !== "pearl-multisig-proposal" ||
    o.version !== 1 ||
    typeof o.id !== "string" ||
    typeof o.vaultAddress !== "string" ||
    !Array.isArray(o.sortedPubkeysHex) ||
    typeof o.psbtBase64 !== "string" ||
    typeof o.threshold !== "number" ||
    typeof o.createdAt !== "number" ||
    typeof o.updatedAt !== "number" ||
    !preview ||
    typeof preview.destination !== "string" ||
    typeof preview.amountGrains !== "string" ||
    typeof preview.feeGrains !== "string" ||
    typeof preview.changeGrains !== "string" ||
    typeof preview.inputCount !== "number"
  ) {
    throw new Error("E_PROPOSAL_ARTIFACT_BAD_SHAPE");
  }
  return {
    version: 1,
    kind: "pearl-multisig-proposal",
    id: o.id,
    vaultAddress: o.vaultAddress,
    threshold: o.threshold,
    sortedPubkeysHex: o.sortedPubkeysHex.map((v) => String(v)),
    psbtBase64: o.psbtBase64,
    preview: {
      destination: preview.destination,
      amountGrains: preview.amountGrains,
      feeGrains: preview.feeGrains,
      changeGrains: preview.changeGrains,
      inputCount: preview.inputCount,
    },
    createdAt: o.createdAt,
    updatedAt: o.updatedAt,
  };
}

export function decodeProposalArtifact(text: string): ProposalArtifactV1 {
  const trimmed = text.trim();
  if (trimmed.length === 0) throw new Error("E_PROPOSAL_ARTIFACT_EMPTY");
  return parseProposalArtifactJson(trimmed);
}

export function encodeProposalArtifact(opts: {
  vault: VaultRecord;
  pending: VaultPendingTxRecord;
}): string {
  const artifact: ProposalArtifactV1 = {
    version: 1,
    kind: "pearl-multisig-proposal",
    id: opts.pending.id,
    vaultAddress: opts.vault.pearlAddress,
    threshold: opts.vault.threshold,
    sortedPubkeysHex: [...opts.vault.sortedPubkeysHex],
    psbtBase64: opts.pending.psbtBase64,
    preview: { ...opts.pending.preview },
    createdAt: opts.pending.createdAt,
    updatedAt: opts.pending.updatedAt,
  };
  return JSON.stringify(artifact, null, 2);
}

export async function importProposalArtifact(opts: {
  vault: VaultRecord;
  artifactText: string;
}): Promise<VaultPendingTxRecord> {
  const artifact = decodeProposalArtifact(opts.artifactText);
  if (artifact.vaultAddress !== opts.vault.pearlAddress) {
    throw new Error("E_PROPOSAL_VAULT_MISMATCH");
  }
  if (artifact.threshold !== opts.vault.threshold) {
    throw new Error("E_PROPOSAL_THRESHOLD_MISMATCH");
  }
  if (!sameHexLists(artifact.sortedPubkeysHex, opts.vault.sortedPubkeysHex)) {
    throw new Error("E_PROPOSAL_SIGNER_SET_MISMATCH");
  }

  const tx = btc.Transaction.fromPSBT(base64ToBytes(artifact.psbtBase64));
  assertPsbtInputsBelongToVault(tx, opts.vault);
  const info = inspectPsbt(artifact.psbtBase64, opts.vault.threshold, opts.vault.sortedPubkeysHex);
  assertPsbtMatchesPreview(info, artifact.preview, opts.vault.pearlAddress);

  const rec: VaultPendingTxRecord = {
    id: artifact.id,
    vaultId: opts.vault.id,
    psbtBase64: artifact.psbtBase64,
    expectedTxid: deriveExpectedTxid(artifact.psbtBase64),
    signersHex: info.signersHex,
    createdAt: artifact.createdAt,
    updatedAt: Date.now(),
    status: info.thresholdMet ? "ready" : "drafting",
    preview: { ...artifact.preview },
  };
  await savePendingTx(rec);
  return rec;
}

export async function importProposalArtifactOrPsbt(opts: {
  vault: VaultRecord;
  artifactText: string;
}): Promise<VaultPendingTxRecord> {
  try {
    return await importProposalArtifact(opts);
  } catch (err) {
    if (!isLikelyPsbtBase64(opts.artifactText)) {
      throw err;
    }
    const rec = pendingFromPsbt({
      vault: opts.vault,
      psbtBase64: opts.artifactText.trim(),
    });
    await savePendingTx(rec);
    return rec;
  }
}

export function exportProposalArtifact(opts: {
  vault: VaultRecord;
  pending: VaultPendingTxRecord;
}): string {
  return encodeProposalArtifact(opts);
}

function estimateMultisigFee(numInputs: number, numOutputs: number, feerate: bigint): bigint {
  const vbytes =
    FIXED_OVERHEAD_VBYTES +
    BigInt(numInputs) * PER_INPUT_VBYTES_MULTISIG +
    BigInt(numOutputs) * PER_P2TR_OUTPUT_VBYTES;
  return vbytes * feerate;
}

function p2trScriptToPearlAddress(scriptBytes: Uint8Array, network: PearlNetwork): string | null {
  if (scriptBytes.length !== 34) return null;
  if (scriptBytes[0] !== 0x51) return null;
  if (scriptBytes[1] !== 0x20) return null;
  try {
    return encodeTaprootAddress(scriptBytes.slice(2), pearlParams(network));
  } catch {
    return null;
  }
}

function p2trScriptFromPearlAddress(address: string, network: PearlNetwork): Uint8Array {
  const program = decodeTaprootAddress(address, pearlParams(network));
  const script = new Uint8Array(34);
  script[0] = 0x51;
  script[1] = 0x20;
  script.set(program, 2);
  return script;
}

async function deriveMultisigKey(vaultAccount: number, keyIndex: number): Promise<{
  pubkeyHex: string;
  privKeyHex: string;
  originPath: string;
}> {
  if (!Number.isInteger(vaultAccount) || vaultAccount < 0) {
    throw new Error("Invalid vault account");
  }
  if (!Number.isInteger(keyIndex) || keyIndex < 0) {
    throw new Error("Invalid key index");
  }

  if (typeof window === "undefined" || !window.appBridge?.wallet?.deriveMultisigKey) {
    throw new Error("Wallet engine is unavailable");
  }

  const derived = await window.appBridge.wallet.deriveMultisigKey(vaultAccount, keyIndex);
  return derived;
}

export interface ExportedPubkeyDescriptor {
  json: string;
  pubkeyHex: string;
  originPath: string;
}

export async function exportMyCosignerDescriptor(opts: {
  vaultAccount: number;
  keyIndex: number;
  label: string;
}): Promise<ExportedPubkeyDescriptor> {
  const { pubkeyHex, originPath } = await deriveMultisigKey(opts.vaultAccount, opts.keyIndex);
  const json = encodePubkeyDescriptor({
    xOnlyPubkey: hexToBytes(pubkeyHex),
    originPath,
    label: opts.label,
  });
  return { json, pubkeyHex, originPath };
}

export function importCosignerDescriptor(json: string): {
  descriptor: PearlMultisigPubkeyDescriptor;
  pubkeyHex: string;
} {
  const { descriptor, xOnlyPubkey } = parsePubkeyDescriptor(json);
  return { descriptor, pubkeyHex: bytesToHex(xOnlyPubkey) };
}

export async function listVaults(): Promise<VaultRecord[]> {
  const state = await loadState();
  return [...state.vaults].sort((a, b) => b.createdAt - a.createdAt);
}

export async function getVault(id: string): Promise<VaultRecord | undefined> {
  const state = await loadState();
  return state.vaults.find((v) => v.id === id);
}

async function getBroadcastTxInfo(txid: string): Promise<BroadcastTxInfo | null> {
  if (typeof window === "undefined" || !window.appBridge?.wallet?.getTransactionInfo) {
    return null;
  }
  try {
    const tx = await window.appBridge.wallet.getTransactionInfo(txid);
    if (!tx) return null;
    return {
      txid: tx.txid,
      confirmations: tx.confirmations ?? 0,
      blockhash: tx.blockhash ?? "",
      time: tx.time ?? Date.now(),
    };
  } catch {
    return null;
  }
}

async function syncBroadcastTxs(vaultId: string): Promise<void> {
  const state = await loadState();
  const trackedTxids = new Set<string>();
  const expectedTxidUpdates = new Map<string, string>();
  const pendingIdsToRemove = new Set<string>();
  const sentRecords: VaultSentTxRecord[] = [];
  let changed = false;

  for (const pending of state.pendingTxs) {
    if (pending.vaultId !== vaultId) continue;
    let txid = pending.txid ?? pending.expectedTxid ?? null;
    if (!txid) {
      try {
        txid = deriveExpectedTxid(pending.psbtBase64);
      } catch {
        continue;
      }
    }
    trackedTxids.add(txid);
    if (pending.expectedTxid !== txid) {
      expectedTxidUpdates.set(pending.id, txid);
      changed = true;
    }
  }

  for (const sent of state.sentTxs) {
    if (sent.vaultId === vaultId && sent.txid) {
      trackedTxids.add(sent.txid);
    }
  }

  if (trackedTxids.size === 0) {
    if (changed) {
      await updateState((current) => {
        let next = current;
        if (expectedTxidUpdates.size > 0) {
          next = {
            ...next,
            pendingTxs: next.pendingTxs.map((p) =>
              expectedTxidUpdates.has(p.id) ? { ...p, expectedTxid: expectedTxidUpdates.get(p.id)! } : p,
            ),
          };
        }
        return next;
      });
    }
    return;
  }

  const txInfoCache = new Map<string, BroadcastTxInfo | null>();
  const lookup = async (txid: string): Promise<BroadcastTxInfo | null> => {
    if (txInfoCache.has(txid)) {
      return txInfoCache.get(txid) ?? null;
    }
    const txInfo = await getBroadcastTxInfo(txid);
    txInfoCache.set(txid, txInfo);
    return txInfo;
  };

  for (const txid of trackedTxids) {
    const txInfo = await lookup(txid);
    if (!txInfo) {
      continue;
    }

    const pending = state.pendingTxs.find(
      (p) => p.vaultId === vaultId && (p.txid === txid || p.expectedTxid === txid),
    );
    const sent = state.sentTxs.find((p) => p.vaultId === vaultId && p.txid === txid);
    if (pending) {
      pendingIdsToRemove.add(pending.id);
      changed = true;
    }

    const sentRecord: VaultSentTxRecord | null = sent
      ? {
          ...sent,
          confirmations: txInfo.confirmations,
          blockhash: txInfo.blockhash || sent.blockhash,
          time: sent.time || txInfo.time,
          updatedAt: Date.now(),
          status: txInfo.confirmations > 0 ? "confirmed" : "broadcast",
        } as VaultSentTxRecord
      : pending
        ? sentFromPending(pending, txid, txInfo)
        : null;

    if (!sentRecord) {
      continue;
    }

    sentRecords.push(sentRecord);
    changed = true;
  }

  if (changed) {
    await updateState((current) => {
      let next = current;
      if (expectedTxidUpdates.size > 0) {
        next = {
          ...next,
          pendingTxs: next.pendingTxs.map((p) =>
            expectedTxidUpdates.has(p.id) ? { ...p, expectedTxid: expectedTxidUpdates.get(p.id)! } : p,
          ),
        };
      }
      if (pendingIdsToRemove.size > 0) {
        next = {
          ...next,
          pendingTxs: next.pendingTxs.filter((p) => !pendingIdsToRemove.has(p.id)),
        };
      }
      if (sentRecords.length > 0) {
        for (const rec of sentRecords) {
          next = {
            ...next,
            sentTxs: upsertSentTx(next, rec).sentTxs,
          };
        }
      }
      return next;
    });
  }
}

export async function deleteVault(id: string): Promise<void> {
  await updateState((state) => ({
    vaults: state.vaults.filter((v) => v.id !== id),
    pendingTxs: state.pendingTxs.filter((p) => p.vaultId !== id),
    sentTxs: state.sentTxs.filter((p) => p.vaultId !== id),
  }));
}

export async function createVault(input: {
  label: string;
  threshold: number;
  cosignerPubkeysHex: string[];
  myPubkeyHex: string;
  myVaultAccount: number;
  myKeyIndex: number;
  network: "mainnet";
}): Promise<VaultRecord> {
  const label = input.label.trim();
  if (label.length === 0 || label.length > 64) throw new Error("E_VAULT_BAD_LABEL");
  if (!Number.isInteger(input.myVaultAccount) || input.myVaultAccount < 0) throw new Error("E_VAULT_BAD_ORIGIN");
  if (!Number.isInteger(input.myKeyIndex) || input.myKeyIndex < 0) throw new Error("E_VAULT_BAD_ORIGIN");
  const pubkeys = input.cosignerPubkeysHex.map((h) => hexToBytes(h));
  const params = pearlParams(input.network);
  const descriptor = vaultDescriptorFromPubkeys(input.threshold, pubkeys, params);

  const myHex = input.myPubkeyHex.toLowerCase();
  const isMember = descriptor.sortedPubkeys.some((p) => bytesToHex(p) === myHex);
  if (!isMember) throw new Error("E_VAULT_NOT_A_COSIGNER");

  const verify = await deriveMultisigKey(input.myVaultAccount, input.myKeyIndex);
  if (verify.pubkeyHex.toLowerCase() !== myHex) throw new Error("E_VAULT_PUBKEY_PATH_MISMATCH");

  const record: VaultRecord = {
    id: newUuid(),
    version: 1,
    label,
    threshold: descriptor.threshold,
    total: descriptor.total,
    sortedPubkeysHex: descriptor.sortedPubkeys.map((p) => bytesToHex(p)),
    myPubkeyHex: myHex,
    myOriginPath: pearlMultisigPath(input.myVaultAccount, input.myKeyIndex),
    myVaultAccount: input.myVaultAccount,
    myKeyIndex: input.myKeyIndex,
    pearlAddress: descriptor.address,
    network: input.network,
    createdAt: Date.now(),
  };

  await updateState((state) => ({ ...state, vaults: [record, ...state.vaults.filter((v) => v.id !== record.id)] }));
  return record;
}

export function descriptorFromRecord(rec: VaultRecord): VaultDescriptor {
  return vaultDescriptorFromPubkeys(
    rec.threshold,
    rec.sortedPubkeysHex.map((h) => hexToBytes(h)),
    pearlParams(rec.network),
  );
}

export function wireDescriptorFromRecord(rec: VaultRecord): {
  threshold: number;
  sortedPubkeysHex: string[];
  network: "mainnet";
} {
  return {
    threshold: rec.threshold,
    sortedPubkeysHex: rec.sortedPubkeysHex,
    network: rec.network,
  };
}

export interface VaultBalance {
  grains: bigint;
  degraded: boolean;
}

function decodeVaultBalance(result: { grains: string; degraded: boolean }): VaultBalance {
  return {
    grains: BigInt(result.grains),
    degraded: result.degraded,
  };
}

function decodeVaultUtxos(result: {
  utxos: Array<{
    txid: string;
    vout: number;
    valueGrains: string;
    scriptHex: string;
  }>;
  degraded: boolean;
}): { utxos: PrlUtxo[]; degraded: boolean } {
  return {
    utxos: result.utxos.map((u) => ({
      txid: u.txid,
      vout: u.vout,
      valueGrains: BigInt(u.valueGrains),
      scriptHex: u.scriptHex,
    })),
    degraded: result.degraded,
  };
}

export async function fetchVaultBalance(rec: VaultRecord): Promise<VaultBalance> {
  const result = await window.appBridge.wallet.getVaultBalance(rec.pearlAddress);
  return decodeVaultBalance(result);
}

export async function fetchVaultUtxos(rec: VaultRecord): Promise<{ utxos: PrlUtxo[]; degraded: boolean }> {
  const result = await window.appBridge.wallet.getVaultUtxos(rec.pearlAddress);
  return decodeVaultUtxos(result);
}

export interface ComposeVaultSendOpts {
  vault: VaultRecord;
  destination: string;
  amountGrains: bigint;
  feerateSatPerVbyte?: bigint;
}

export interface ComposedVaultSend {
  psbtBase64: string;
  utxos: PrlUtxo[];
  outputs: { address: string; amountGrains: bigint }[];
  feeGrains: bigint;
  changeGrains: bigint;
  degraded: boolean;
  amountGrains: bigint;
  destination: string;
}

export async function composeVaultSend(opts: ComposeVaultSendOpts): Promise<ComposedVaultSend> {
  const feerate = opts.feerateSatPerVbyte ?? PEARL_DEFAULT_FEERATE_SATS_PER_VBYTE;
  const { utxos: avail, degraded } = await fetchVaultUtxos(opts.vault);
  if (avail.length === 0) throw new Error("E_NO_UTXOS");
  const vaultDescriptor = descriptorFromRecord(opts.vault);

  const sorted = [...avail].sort((a, b) =>
    a.valueGrains > b.valueGrains ? -1 : a.valueGrains < b.valueGrains ? 1 : 0,
  );

  let numOutputs = 2;
  const picked: PrlUtxo[] = [];
  let sum = 0n;
  for (const u of sorted) {
    picked.push(u);
    sum += u.valueGrains;
    const fee = estimateMultisigFee(picked.length, numOutputs, feerate);
    if (sum >= opts.amountGrains + fee) break;
  }
  let fee = estimateMultisigFee(picked.length, numOutputs, feerate);
  let need = opts.amountGrains + fee;
  if (sum < need) throw new Error("E_INSUFFICIENT_FUNDS");

  let change = sum - need;
  if (change < DUST_LIMIT_GRAINS) {
    numOutputs -= 1;
    const recomputed = estimateMultisigFee(picked.length, numOutputs, feerate);
    if (sum < opts.amountGrains + recomputed) throw new Error("E_INSUFFICIENT_FUNDS");
    fee = sum - opts.amountGrains;
    change = 0n;
  }

  const outputs: { address: string; amountGrains: bigint }[] = [
    { address: opts.destination, amountGrains: opts.amountGrains },
  ];
  if (change > 0n) {
    outputs.push({ address: opts.vault.pearlAddress, amountGrains: change });
  }

  const tx = new btc.Transaction({ allowUnknownOutputs: false });
  for (const u of picked) {
    if (
      typeof u.txid !== "string" ||
      !/^[0-9a-fA-F]{64}$/.test(u.txid) ||
      typeof u.vout !== "number" ||
      u.vout < 0 ||
      typeof u.scriptHex !== "string" ||
      !/^[0-9a-fA-F]+$/.test(u.scriptHex)
    ) {
      throw new Error("E_PEARL_UTXO_SHAPE");
    }
    const valueGrains = u.valueGrains;
    if (valueGrains <= 0n) throw new Error("E_PEARL_UTXO_VALUE");
    const scriptBytes = hexToBytes(u.scriptHex);
    if (bytesToHex(scriptBytes) !== bytesToHex(vaultDescriptor.outputScript)) {
      throw new Error("E_MULTISIG_UTXO_NOT_VAULT");
    }
    tx.addInput({
      txid: hexToBytes(u.txid),
      index: u.vout,
      witnessUtxo: { amount: valueGrains, script: scriptBytes },
      tapInternalKey: vaultDescriptor.internalKey,
      tapLeafScript: vaultDescriptor.tapLeafScript,
    });
  }
  for (const o of outputs) {
    tx.addOutput({
      script: p2trScriptFromPearlAddress(o.address, opts.vault.network),
      amount: o.amountGrains,
    });
  }

  return {
    psbtBase64: bytesToBase64(tx.toPSBT()),
    utxos: picked,
    outputs,
    feeGrains: fee,
    changeGrains: change,
    degraded,
    amountGrains: opts.amountGrains,
    destination: opts.destination,
  };
}

export async function signVaultPsbt(opts: {
  vault: VaultRecord;
  psbtBase64: string;
}): Promise<{ psbtBase64: string }> {
  const key = await deriveMultisigKey(opts.vault.myVaultAccount, opts.vault.myKeyIndex);
  const tx = btc.Transaction.fromPSBT(base64ToBytes(opts.psbtBase64));
  assertPsbtInputsBelongToVault(tx, opts.vault);
  for (let i = 0; i < tx.inputsLength; i++) {
    tx.signIdx(hexToBytes(key.privKeyHex), i);
  }
  return { psbtBase64: bytesToBase64(tx.toPSBT()) };
}

export interface PsbtOutputInfo {
  address: string | null;
  amountGrains: bigint;
  scriptHex: string;
}

export interface PsbtSignerInfo {
  signerCount: number;
  signersHex: string[];
  foreignSignersHex: string[];
  thresholdMet: boolean;
  inputCount: number;
  witnessScriptHex: string;
  outputs: PsbtOutputInfo[];
  inputAmountsGrains: bigint[];
  totalInputGrains: bigint;
  totalOutputGrains: bigint;
  feeGrains: bigint;
  feeUnknown: boolean;
  network: "mainnet";
}

export function inspectPsbt(
  psbtBase64: string,
  threshold: number,
  validPubkeysHex?: readonly string[],
): PsbtSignerInfo {
  if (typeof psbtBase64 !== "string" || psbtBase64.length === 0) {
    throw new Error("E_MULTISIG_BAD_PSBT");
  }
  let tx: btc.Transaction;
  try {
    tx = btc.Transaction.fromPSBT(base64ToBytes(psbtBase64));
  } catch (err) {
    throw new Error(`E_MULTISIG_PSBT_PARSE: ${err instanceof Error ? err.message : String(err)}`);
  }
  if (tx.inputsLength === 0) throw new Error("E_PEARL_NO_INPUTS");

  const input0 = tx.getInput(0) as {
    tapScriptSig?: Array<[{ pubKey: Uint8Array; leafHash: Uint8Array }, Uint8Array]>;
    witnessUtxo?: { script: Uint8Array; amount: bigint };
  };
  const inputAmountsGrains: bigint[] = [];
  let totalInputGrains = 0n;
  let feeUnknown = false;
  const seen = new Set<string>();
  const foreignSeen = new Set<string>();
  let thresholdMet = true;
  const validSet = validPubkeysHex ? new Set(validPubkeysHex.map((h) => h.toLowerCase())) : null;
  for (let i = 0; i < tx.inputsLength; i++) {
    const inp = tx.getInput(i) as {
      witnessUtxo?: { amount: bigint };
      tapScriptSig?: Array<[{ pubKey: Uint8Array; leafHash: Uint8Array }, Uint8Array]>;
    };
    const amt = inp.witnessUtxo?.amount;
    if (typeof amt !== "bigint") {
      feeUnknown = true;
      inputAmountsGrains.push(0n);
    } else {
      inputAmountsGrains.push(amt);
      totalInputGrains += amt;
    }
    const inputSeen = new Set<string>();
    const sigEntries = inp.tapScriptSig ?? [];
    for (const [{ pubKey }] of sigEntries) {
      const hex = bytesToHex(pubKey);
      if (validSet) {
        if (validSet.has(hex.toLowerCase())) {
          seen.add(hex);
          inputSeen.add(hex.toLowerCase());
        } else {
          foreignSeen.add(hex);
        }
      } else {
        seen.add(hex);
        inputSeen.add(hex.toLowerCase());
      }
    }
    if (validSet && inputSeen.size < threshold) {
      thresholdMet = false;
    }
  }

  const signersHex = Array.from(seen);
  const foreignSignersHex = validSet ? Array.from(foreignSeen) : [];
  if (!validSet) {
    thresholdMet = signersHex.length >= threshold;
  }

  const outputs: PsbtOutputInfo[] = [];
  let totalOutputGrains = 0n;
  for (let i = 0; i < tx.outputsLength; i++) {
    const o = tx.getOutput(i) as { script?: Uint8Array; amount?: bigint };
    const script = o.script ?? new Uint8Array(0);
    const amt = o.amount ?? 0n;
    outputs.push({
      address: p2trScriptToPearlAddress(script, "mainnet"),
      amountGrains: amt,
      scriptHex: bytesToHex(script),
    });
    totalOutputGrains += amt;
  }

  const feeGrains = feeUnknown ? 0n : totalInputGrains > totalOutputGrains ? totalInputGrains - totalOutputGrains : 0n;

  return {
    signerCount: signersHex.length,
    signersHex,
    foreignSignersHex,
    thresholdMet,
    inputCount: tx.inputsLength,
    witnessScriptHex: input0.witnessUtxo?.script ? bytesToHex(input0.witnessUtxo.script) : "",
    outputs,
    inputAmountsGrains,
    totalInputGrains,
    totalOutputGrains,
    feeGrains,
    feeUnknown,
    network: "mainnet",
  };
}

export function feeSuspiciousReason(info: PsbtSignerInfo): string | null {
  if (info.feeUnknown) {
    return "Input amounts are missing from the PSBT — fee can't be verified, so the wallet refuses to sign.";
  }
  if (info.totalInputGrains <= 0n) return null;
  const feePct = (info.feeGrains * 100n) / info.totalInputGrains;
  if (feePct > 20n) {
    return `Fee is ${feePct}% of the spend (${info.feeGrains} grains of ${info.totalInputGrains}). That's far above normal — refusing.`;
  }
  return null;
}

export function assertPsbtMatchesPreview(
  info: PsbtSignerInfo,
  preview: VaultPendingTxRecord["preview"],
  vaultAddress: string,
): void {
  if (info.feeUnknown) {
    throw new Error("E_MULTISIG_OUTPUT_MISMATCH: PSBT inputs are missing amount data");
  }
  const expectedAmount = BigInt(preview.amountGrains);
  const expectedChange = BigInt(preview.changeGrains);
  const expectedFee = BigInt(preview.feeGrains);
  if (info.outputs.length === 0) throw new Error("E_MULTISIG_OUTPUT_MISMATCH: PSBT has no outputs");
  const dest = info.outputs[0]!;
  if (dest.address !== preview.destination) {
    throw new Error(`E_MULTISIG_OUTPUT_MISMATCH: destination is ${dest.address ?? "non-Pearl script"} (expected ${preview.destination})`);
  }
  if (dest.amountGrains !== expectedAmount) {
    throw new Error(`E_MULTISIG_OUTPUT_MISMATCH: destination amount is ${dest.amountGrains} (expected ${expectedAmount})`);
  }
  if (expectedChange > 0n) {
    if (info.outputs.length < 2) throw new Error("E_MULTISIG_OUTPUT_MISMATCH: change output is missing");
    const chg = info.outputs[1]!;
    if (chg.address !== vaultAddress) {
      throw new Error(`E_MULTISIG_OUTPUT_MISMATCH: change goes to ${chg.address ?? "non-Pearl script"} (expected ${vaultAddress})`);
    }
    if (chg.amountGrains !== expectedChange) {
      throw new Error(`E_MULTISIG_OUTPUT_MISMATCH: change amount is ${chg.amountGrains} (expected ${expectedChange})`);
    }
  }
  const expectedCount = expectedChange > 0n ? 2 : 1;
  if (info.outputs.length > expectedCount) {
    throw new Error(`E_MULTISIG_OUTPUT_MISMATCH: PSBT has ${info.outputs.length} outputs (expected ${expectedCount})`);
  }
  if (info.feeGrains !== expectedFee) {
    throw new Error(`E_MULTISIG_OUTPUT_MISMATCH: fee is ${info.feeGrains} grains (expected ${expectedFee})`);
  }
}

export function psbtOutputsEqual(a: PsbtOutputInfo[], b: PsbtOutputInfo[]): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (a[i]!.scriptHex !== b[i]!.scriptHex) return false;
    if (a[i]!.amountGrains !== b[i]!.amountGrains) return false;
  }
  return true;
}

export function finalizeVaultPsbt(psbtBase64: string): { rawHex: string } {
  let tx: btc.Transaction;
  try {
    tx = btc.Transaction.fromPSBT(base64ToBytes(psbtBase64));
  } catch (err) {
    throw new Error(`E_MULTISIG_PSBT_PARSE: ${err instanceof Error ? err.message : String(err)}`);
  }
  try {
    tx.finalize();
  } catch (err) {
    throw new Error(`E_MULTISIG_PSBT_NOT_FINALIZABLE: ${err instanceof Error ? err.message : String(err)}`);
  }
  return { rawHex: tx.hex };
}

export async function broadcastVaultTx(rawHex: string): Promise<string> {
  return await broadcastPearlTx(rawHex);
}

export async function listPendingTxs(vaultId: string): Promise<VaultPendingTxRecord[]> {
  await syncBroadcastTxs(vaultId);
  const state = await loadState();
  return [...state.pendingTxs.filter((p) => p.vaultId === vaultId)].sort((a, b) => b.createdAt - a.createdAt);
}

export async function listSentTxs(vaultId: string): Promise<VaultSentTxRecord[]> {
  const state = await loadState();
  return [...state.sentTxs.filter((p) => p.vaultId === vaultId)].sort((a, b) => b.time - a.time);
}

export async function getPendingTx(id: string): Promise<VaultPendingTxRecord | undefined> {
  const state = await loadState();
  const pending = state.pendingTxs.find((p) => p.id === id);
  if (!pending || (!pending.expectedTxid && !pending.txid)) {
    return pending;
  }
  await syncBroadcastTxs(pending.vaultId);
  return (await loadState()).pendingTxs.find((p) => p.id === id);
}

export async function getSentTx(id: string): Promise<VaultSentTxRecord | undefined> {
  const state = await loadState();
  return state.sentTxs.find((p) => p.id === id);
}

export async function savePendingTx(rec: VaultPendingTxRecord): Promise<void> {
  await updateState((state) => ({
    ...state,
    pendingTxs: [rec, ...state.pendingTxs.filter((p) => p.id !== rec.id)],
  }));
}

export async function saveSentTx(rec: VaultSentTxRecord): Promise<void> {
  await updateState((state) => upsertSentTx(state, rec));
}

export async function deletePendingTx(id: string): Promise<void> {
  await updateState((state) => ({ ...state, pendingTxs: state.pendingTxs.filter((p) => p.id !== id) }));
}

export async function persistComposedAsPending(opts: {
  vault: VaultRecord;
  psbtBase64: string;
  preview: VaultPendingTxRecord["preview"];
}): Promise<VaultPendingTxRecord> {
  const tx = btc.Transaction.fromPSBT(base64ToBytes(opts.psbtBase64));
  assertPsbtInputsBelongToVault(tx, opts.vault);
  const info = inspectPsbt(opts.psbtBase64, opts.vault.threshold, opts.vault.sortedPubkeysHex);
  const rec: VaultPendingTxRecord = {
    id: newUuid(),
    vaultId: opts.vault.id,
    psbtBase64: opts.psbtBase64,
    expectedTxid: deriveExpectedTxid(opts.psbtBase64),
    signersHex: info.signersHex,
    createdAt: Date.now(),
    updatedAt: Date.now(),
    status: info.thresholdMet ? "ready" : "drafting",
    preview: opts.preview,
  };
  await savePendingTx(rec);
  return rec;
}

export async function signPendingTx(opts: {
  vault: VaultRecord;
  pending: VaultPendingTxRecord;
}): Promise<VaultPendingTxRecord> {
  const myHex = opts.vault.myPubkeyHex.toLowerCase();
  const tx = btc.Transaction.fromPSBT(base64ToBytes(opts.pending.psbtBase64));
  assertPsbtInputsBelongToVault(tx, opts.vault);
  const pre = inspectPsbt(opts.pending.psbtBase64, opts.vault.threshold, opts.vault.sortedPubkeysHex);
  assertPsbtMatchesPreview(pre, opts.pending.preview, opts.vault.pearlAddress);
  if (pre.foreignSignersHex.length > 0) throw new Error("E_MULTISIG_FOREIGN_SIGNER_PRESENT");
  if (pre.signersHex.includes(myHex)) return opts.pending;
  const { psbtBase64 } = await signVaultPsbt({ vault: opts.vault, psbtBase64: opts.pending.psbtBase64 });
  const info = inspectPsbt(psbtBase64, opts.vault.threshold, opts.vault.sortedPubkeysHex);
  const updated: VaultPendingTxRecord = {
    ...opts.pending,
    psbtBase64,
    expectedTxid: deriveExpectedTxid(psbtBase64),
    signersHex: info.signersHex,
    status: info.thresholdMet ? "ready" : "drafting",
    updatedAt: Date.now(),
  };
  await savePendingTx(updated);
  return updated;
}

export async function broadcastPendingTx(opts: {
  vault: VaultRecord;
  pending: VaultPendingTxRecord;
}): Promise<VaultPendingTxRecord> {
  const tx = btc.Transaction.fromPSBT(base64ToBytes(opts.pending.psbtBase64));
  assertPsbtInputsBelongToVault(tx, opts.vault);
  const info = inspectPsbt(opts.pending.psbtBase64, opts.vault.threshold, opts.vault.sortedPubkeysHex);
  assertPsbtMatchesPreview(info, opts.pending.preview, opts.vault.pearlAddress);
  if (info.foreignSignersHex.length > 0) throw new Error("E_MULTISIG_FOREIGN_SIGNER_PRESENT");
  if (!info.thresholdMet) throw new Error("E_MULTISIG_THRESHOLD_NOT_MET");

  const { rawHex } = finalizeVaultPsbt(opts.pending.psbtBase64);
  let txid: string;
  try {
    txid = await broadcastPearlTx(rawHex);
  } catch (err) {
    const failed: VaultPendingTxRecord = { ...opts.pending, status: "failed", updatedAt: Date.now() };
    await savePendingTx(failed);
    throw err;
  }
  const broadcast: VaultPendingTxRecord = {
    ...opts.pending,
    status: "broadcast",
    txid,
    expectedTxid: opts.pending.expectedTxid ?? txid,
    updatedAt: Date.now(),
  };
  await savePendingTx(broadcast);
  await saveSentTx(sentFromPending(broadcast, txid));
  return broadcast;
}

export { PEARL_DEFAULT_FEERATE_SATS_PER_VBYTE, PER_INPUT_VBYTES_MULTISIG, PER_P2TR_OUTPUT_VBYTES, FIXED_OVERHEAD_VBYTES, DUST_LIMIT_GRAINS };
