import { p2tr, p2tr_ms, TAPROOT_UNSPENDABLE_KEY } from "@scure/btc-signer";
import type { BTC_NETWORK } from "@scure/btc-signer/utils.js";
import { encodeTaprootAddress } from "./address";
import type { PearlNetworkParams } from "./network";

export const MULTISIG_MIN_THRESHOLD = 1;
export const MULTISIG_MAX_COSIGNERS = 15;

export interface VaultDescriptor {
  threshold: number;
  total: number;
  sortedPubkeys: Uint8Array[];
  address: string;
  outputScript: Uint8Array;
  outputKey: Uint8Array;
  leafScript: Uint8Array;
  leafVersion: number;
  internalKey: Uint8Array;
  tapLeafScript: Array<
    [
      { version: number; internalKey: Uint8Array; merklePath: Uint8Array[] },
      Uint8Array,
    ]
  >;
  network: "mainnet";
}

function isXOnlyPubkey(b: Uint8Array): boolean {
  return b.length === 32;
}

function pearlNetworkToBtcNetwork(params: PearlNetworkParams): BTC_NETWORK {
  return {
    bech32: params.hrp,
    pubKeyHash: 0,
    scriptHash: 0,
    wif: 0,
  };
}

export function sortPubkeysBip67(pubkeys: Uint8Array[]): Uint8Array[] {
  const copy = pubkeys.map((p) => Uint8Array.from(p));
  copy.sort((a, b) => {
    for (let i = 0; i < a.length; i++) {
      const da = a[i]!;
      const db = b[i]!;
      if (da !== db) return da - db;
    }
    return 0;
  });
  return copy;
}

export function vaultDescriptorFromPubkeys(
  threshold: number,
  pubkeys: Uint8Array[],
  params: PearlNetworkParams,
): VaultDescriptor {
  if (!Number.isInteger(threshold) || threshold < MULTISIG_MIN_THRESHOLD) throw new Error("E_MULTISIG_BAD_THRESHOLD");
  if (pubkeys.length === 0 || pubkeys.length > MULTISIG_MAX_COSIGNERS) throw new Error("E_MULTISIG_BAD_COSIGNER_COUNT");
  if (threshold > pubkeys.length) throw new Error("E_MULTISIG_THRESHOLD_EXCEEDS_COSIGNERS");
  for (const p of pubkeys) {
    if (!isXOnlyPubkey(p)) throw new Error("E_MULTISIG_BAD_PUBKEY_LEN");
  }
  const seen = new Set<string>();
  for (const p of pubkeys) {
    const k = Array.from(p).map((b) => b.toString(16).padStart(2, "0")).join("");
    if (seen.has(k)) throw new Error("E_MULTISIG_DUPLICATE_PUBKEY");
    seen.add(k);
  }

  const sorted = sortPubkeysBip67(pubkeys);
  const leaf = p2tr_ms(threshold, sorted);
  const tr = p2tr(
    TAPROOT_UNSPENDABLE_KEY,
    leaf as unknown as Parameters<typeof p2tr>[1],
    pearlNetworkToBtcNetwork(params),
    false,
  );
  const address = encodeTaprootAddress(tr.tweakedPubkey, params);
  const tls = tr.tapLeafScript;
  if (!tls || tls.length !== 1) throw new Error("E_MULTISIG_UNEXPECTED_LEAF_COUNT");
  const [, scriptPlusVer] = tls[0]!;
  const leafVersion = scriptPlusVer[scriptPlusVer.length - 1]!;
  const leafScript = scriptPlusVer.slice(0, -1);

  return {
    threshold,
    total: pubkeys.length,
    sortedPubkeys: sorted,
    address,
    outputScript: tr.script,
    outputKey: tr.tweakedPubkey,
    leafScript,
    leafVersion,
    internalKey: TAPROOT_UNSPENDABLE_KEY,
    tapLeafScript: tls as VaultDescriptor["tapLeafScript"],
    network: params.name,
  };
}

export function vaultAddressFromPubkeys(
  threshold: number,
  pubkeys: Uint8Array[],
  params: PearlNetworkParams,
): string {
  return vaultDescriptorFromPubkeys(threshold, pubkeys, params).address;
}

export const PEARL_MULTISIG_NUMS_INTERNAL_KEY = TAPROOT_UNSPENDABLE_KEY;
