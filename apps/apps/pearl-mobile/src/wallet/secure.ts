/**
 * 敏感数据存储：助记词只进系统安全区
 * （iOS Keychain / Android Keystore），并在应用在前台期间
 * 以内存引用供派生与签名使用；锁屏即清除内存副本。
 */

import * as SecureStore from 'expo-secure-store';
import {SECURE_STORE_SERVICE} from '@pearl/pearl-mobile-core';

const KEY_MNEMONIC = 'mnemonic';

const OPTIONS: SecureStore.SecureStoreOptions = {
  keychainService: SECURE_STORE_SERVICE,
  // 设备解锁后可读；不可迁移到其他设备
  keychainAccessible: SecureStore.WHEN_UNLOCKED_THIS_DEVICE_ONLY,
};

export async function saveMnemonic(mnemonic: string): Promise<void> {
  await SecureStore.setItemAsync(KEY_MNEMONIC, mnemonic, OPTIONS);
}

export async function loadMnemonic(): Promise<string | null> {
  return SecureStore.getItemAsync(KEY_MNEMONIC, OPTIONS);
}

export async function hasMnemonic(): Promise<boolean> {
  const v = await loadMnemonic();
  return v !== null && v.length > 0;
}

export async function deleteMnemonic(): Promise<void> {
  await SecureStore.deleteItemAsync(KEY_MNEMONIC, OPTIONS);
}

/**
 * 内存中的助记词：应用会话内使用，App 退到后台或锁屏时清空。
 * 需要签名时若内存为空，则从安全存储重新加载（可触发生物识别）。
 */
let inMemoryMnemonic: string | null = null;

export function setSessionMnemonic(m: string | null): void {
  inMemoryMnemonic = m;
}

export function getSessionMnemonic(): string | null {
  return inMemoryMnemonic;
}

/** 获取签名所需助记词：优先内存，缺失时读安全存储 */
export async function requireMnemonic(): Promise<string> {
  if (inMemoryMnemonic) return inMemoryMnemonic;
  const m = await loadMnemonic();
  if (!m) throw new Error('钱包不存在');
  inMemoryMnemonic = m;
  return m;
}
