import { bytesToHex, hexToBytes } from "@/crypto/descriptor";
import { importCosignerDescriptor } from "@/services/multisig";

export function formatGrains(grains: bigint): string {
  const whole = grains / 100_000_000n;
  const frac = (grains % 100_000_000n).toString().padStart(8, "0").replace(/0+$/, "");
  return frac.length > 0 ? `${whole}.${frac}` : `${whole}`;
}

export function shortHex(hex: string, keep = 8): string {
  if (hex.length <= keep * 2) return hex;
  return `${hex.slice(0, keep)}...${hex.slice(-keep)}`;
}

export function copyToClipboard(text: string): Promise<void> {
  return navigator.clipboard.writeText(text);
}

export function descriptorTextToPubkeyHex(input: string): { pubkeyHex: string; originPath?: string } {
  const trimmed = input.trim();
  if (trimmed.length === 0) throw new Error("descriptor is empty");
  if (trimmed.startsWith("{")) {
    const { descriptor, pubkeyHex } = importCosignerDescriptor(trimmed);
    return { pubkeyHex, originPath: descriptor.originPath };
  }
  const normalized = trimmed.replace(/^0x/i, "");
  if (!/^[0-9a-fA-F]{64}$/.test(normalized)) {
    throw new Error("expected a 32-byte x-only pubkey or descriptor JSON");
  }
  return { pubkeyHex: normalized.toLowerCase() };
}

export function isLikelyVaultToken(value: string): boolean {
  return /^[A-Za-z0-9_-]{43}$/.test(value.trim());
}

export function hexToLower(hex: string): string {
  return bytesToHex(hexToBytes(hex)).toLowerCase();
}
