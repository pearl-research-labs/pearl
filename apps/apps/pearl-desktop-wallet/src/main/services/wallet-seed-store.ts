import { createCipheriv, createDecipheriv, randomBytes, scryptSync } from 'crypto';
import fs from 'fs';

interface SeedBlobV1 {
  version: 1;
  kdf: 'scrypt';
  salt: string;
  iv: string;
  tag: string;
  ciphertext: string;
}

function toBase64(bytes: Buffer): string {
  return bytes.toString('base64');
}

function fromBase64(value: string): Buffer {
  return Buffer.from(value, 'base64');
}

function encryptSeed(seed: string, password: string): SeedBlobV1 {
  const salt = randomBytes(16);
  const key = scryptSync(password, salt, 32);
  const iv = randomBytes(12);
  const cipher = createCipheriv('aes-256-gcm', key, iv);
  const ciphertext = Buffer.concat([cipher.update(seed, 'utf8'), cipher.final()]);
  const tag = cipher.getAuthTag();
  return {
    version: 1,
    kdf: 'scrypt',
    salt: toBase64(salt),
    iv: toBase64(iv),
    tag: toBase64(tag),
    ciphertext: toBase64(ciphertext),
  };
}

function decryptSeed(blob: SeedBlobV1, password: string): string {
  if (blob.version !== 1 || blob.kdf !== 'scrypt') {
    throw new Error('Unsupported seed blob');
  }
  const salt = fromBase64(blob.salt);
  const iv = fromBase64(blob.iv);
  const tag = fromBase64(blob.tag);
  const ciphertext = fromBase64(blob.ciphertext);
  const key = scryptSync(password, salt, 32);
  const decipher = createDecipheriv('aes-256-gcm', key, iv);
  decipher.setAuthTag(tag);
  const plaintext = Buffer.concat([decipher.update(ciphertext), decipher.final()]);
  return plaintext.toString('utf8');
}

export function writeEncryptedSeed(filePath: string, seed: string, password: string): void {
  const blob = encryptSeed(seed, password);
  fs.writeFileSync(filePath, JSON.stringify(blob, null, 2), { mode: 0o600 });
}

export function readEncryptedSeed(filePath: string, password: string): string | null {
  if (!fs.existsSync(filePath)) return null;
  const raw = fs.readFileSync(filePath, 'utf8');
  const parsed = JSON.parse(raw) as SeedBlobV1;
  return decryptSeed(parsed, password);
}
