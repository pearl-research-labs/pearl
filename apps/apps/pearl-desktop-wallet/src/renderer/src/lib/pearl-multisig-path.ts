export const PEARL_COIN_TYPE = 808276;
export const PEARL_MULTISIG_ACCOUNT_PREFIX = 100;

export function pearlMultisigPath(vaultAccount: number, keyIndex: number): string {
  if (!Number.isInteger(vaultAccount) || vaultAccount < 0 || vaultAccount > 0x7fffffff) {
    throw new Error(`pearlMultisigPath: bad vaultAccount ${vaultAccount}`);
  }
  if (!Number.isInteger(keyIndex) || keyIndex < 0 || keyIndex > 0x7fffffff) {
    throw new Error(`pearlMultisigPath: bad keyIndex ${keyIndex}`);
  }
  return `m/86'/${PEARL_COIN_TYPE}'/${PEARL_MULTISIG_ACCOUNT_PREFIX}'/${vaultAccount}'/${keyIndex}`;
}
