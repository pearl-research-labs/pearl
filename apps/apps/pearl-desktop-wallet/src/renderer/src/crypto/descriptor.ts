export interface PearlMultisigPubkeyDescriptor {
  version: 1;
  type: "pearl-multisig-pubkey";
  network: "mainnet";
  xOnlyPubkey: string;
  originPath: string;
  label: string;
}

export function bytesToHex(bytes: Uint8Array): string {
  return Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");
}

export function hexToBytes(hex: string): Uint8Array {
  if (!/^(?:[0-9a-fA-F]{2})*$/.test(hex)) throw new Error("E_HEX");
  const out = new Uint8Array(hex.length / 2);
  for (let i = 0; i < out.length; i++) {
    out[i] = Number.parseInt(hex.slice(i * 2, i * 2 + 2), 16);
  }
  return out;
}

export function encodePubkeyDescriptor(input: {
  xOnlyPubkey: Uint8Array;
  originPath: string;
  label: string;
}): string {
  const label = input.label.trim();
  if (label.length === 0 || label.length > 64) throw new Error("E_DESCRIPTOR_BAD_LABEL");
  if (input.xOnlyPubkey.length !== 32) throw new Error("E_DESCRIPTOR_BAD_PUBKEY_LEN");
  if (input.originPath.trim().length === 0) throw new Error("E_DESCRIPTOR_BAD_PATH");
  const obj: PearlMultisigPubkeyDescriptor = {
    version: 1,
    type: "pearl-multisig-pubkey",
    network: "mainnet",
    xOnlyPubkey: bytesToHex(input.xOnlyPubkey),
    originPath: input.originPath,
    label,
  };
  return JSON.stringify(obj, null, 2);
}

export function parsePubkeyDescriptor(json: string): {
  descriptor: PearlMultisigPubkeyDescriptor;
  xOnlyPubkey: Uint8Array;
} {
  let raw: unknown;
  try {
    raw = JSON.parse(json);
  } catch {
    throw new Error("E_DESCRIPTOR_BAD_JSON");
  }
  if (typeof raw !== "object" || raw === null || Array.isArray(raw)) {
    throw new Error("E_DESCRIPTOR_BAD_SHAPE");
  }
  const o = raw as Partial<PearlMultisigPubkeyDescriptor>;
  if (o.version !== 1) throw new Error("E_DESCRIPTOR_BAD_VERSION");
  if (o.type !== "pearl-multisig-pubkey") throw new Error("E_DESCRIPTOR_BAD_TYPE");
  if (o.network !== "mainnet") throw new Error("E_DESCRIPTOR_BAD_NETWORK");
  if (typeof o.xOnlyPubkey !== "string") throw new Error("E_DESCRIPTOR_BAD_PUBKEY");
  if (typeof o.originPath !== "string") throw new Error("E_DESCRIPTOR_BAD_PATH");
  if (typeof o.label !== "string") throw new Error("E_DESCRIPTOR_BAD_LABEL");
  const label = o.label.trim();
  if (label.length === 0 || label.length > 64) throw new Error("E_DESCRIPTOR_BAD_LABEL");
  const xOnlyPubkey = hexToBytes(o.xOnlyPubkey);
  if (xOnlyPubkey.length !== 32) throw new Error("E_DESCRIPTOR_BAD_PUBKEY_LEN");
  return {
    descriptor: {
      version: 1,
      type: "pearl-multisig-pubkey",
      network: "mainnet",
      xOnlyPubkey: o.xOnlyPubkey,
      originPath: o.originPath,
      label,
    },
    xOnlyPubkey,
  };
}
