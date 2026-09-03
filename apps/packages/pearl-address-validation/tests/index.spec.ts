import validate, { getAddressInfo, Network } from '../src/index';
import { expect, describe, it } from 'vitest';

const MAINNET_P2TR = 'prl1px8h7f4q0sra6ed4d9kasz5re6zj3u945u0pk2t7yxyaf5a9geqtsvum09k';
const MAINNET_P2MR = 'prl1zx8h7f4q0sra6ed4d9kasz5re6zj3u945u0pk2t7yxyaf5a9geqtsypzqta';
const TESTNET_P2TR = 'tprl1px8h7f4q0sra6ed4d9kasz5re6zj3u945u0pk2t7yxyaf5a9geqts8nl36r';
const TESTNET_P2MR = 'tprl1zx8h7f4q0sra6ed4d9kasz5re6zj3u945u0pk2t7yxyaf5a9geqts0wx75g';
const SIMNET_P2TR = 'rprl1px8h7f4q0sra6ed4d9kasz5re6zj3u945u0pk2t7yxyaf5a9geqtsa2rflt';
const SIMNET_P2MR = 'rprl1zx8h7f4q0sra6ed4d9kasz5re6zj3u945u0pk2t7yxyaf5a9geqts4h6x3q';

describe('Pearl Address Validation', () => {
  describe('Valid P2TR (witness v1) addresses', () => {
    it('validates Mainnet P2TR', () => {
      expect(validate(MAINNET_P2TR)).toBe(true);
      expect(getAddressInfo(MAINNET_P2TR)).toEqual({
        type: 'p2tr',
        network: 'mainnet',
        address: MAINNET_P2TR,
      });
    });

    it('validates Testnet P2TR', () => {
      expect(validate(TESTNET_P2TR)).toBe(true);
      expect(getAddressInfo(TESTNET_P2TR)).toEqual({
        type: 'p2tr',
        network: 'testnet',
        address: TESTNET_P2TR,
      });
    });

    it('validates Simnet P2TR', () => {
      expect(validate(SIMNET_P2TR)).toBe(true);
      expect(getAddressInfo(SIMNET_P2TR)).toEqual({
        type: 'p2tr',
        network: 'simnet',
        address: SIMNET_P2TR,
      });
    });
  });

  describe('Valid P2MR (witness v2) addresses', () => {
    it('validates Mainnet P2MR', () => {
      expect(validate(MAINNET_P2MR)).toBe(true);
      expect(getAddressInfo(MAINNET_P2MR)).toEqual({
        type: 'p2mr',
        network: 'mainnet',
        address: MAINNET_P2MR,
      });
    });

    it('validates Testnet P2MR', () => {
      expect(validate(TESTNET_P2MR)).toBe(true);
      expect(getAddressInfo(TESTNET_P2MR)).toEqual({
        type: 'p2mr',
        network: 'testnet',
        address: TESTNET_P2MR,
      });
    });

    it('validates Simnet P2MR', () => {
      expect(validate(SIMNET_P2MR)).toBe(true);
      expect(getAddressInfo(SIMNET_P2MR)).toEqual({
        type: 'p2mr',
        network: 'simnet',
        address: SIMNET_P2MR,
      });
    });
  });

  describe('Validation with network parameter', () => {
    it('accepts an address whose network matches', () => {
      expect(validate(MAINNET_P2TR, Network.mainnet)).toBe(true);
      expect(validate(MAINNET_P2MR, Network.mainnet)).toBe(true);
      expect(validate(TESTNET_P2MR, Network.testnet)).toBe(true);
      expect(validate(SIMNET_P2MR, Network.simnet)).toBe(true);
    });

    it('rejects an address whose network does not match', () => {
      expect(validate(MAINNET_P2TR, Network.testnet)).toBe(false);
      expect(validate(MAINNET_P2MR, Network.simnet)).toBe(false);
      expect(validate(TESTNET_P2TR, Network.mainnet)).toBe(false);
    });
  });

  describe('Invalid addresses', () => {
    it('rejects witness v0 (legacy SegWit) addresses', () => {
      // Bitcoin-style v0 P2WPKH/P2WSH addresses fail bech32m checksum verification.
      expect(validate('prl1qw508d6qejxtdg4y5r3zarvary0c5xw7k6lzukf')).toBe(false);
      expect(validate('tprl1qw508d6qejxtdg4y5r3zarvary0c5xw7knq72ye')).toBe(false);
    });

    it('rejects witness versions other than v1 and v2', () => {
      // Mainnet bech32m, witness v3, 32-byte program — well-formed but unsupported.
      const v3 = 'prl1rx8h7f4q0sra6ed4d9kasz5re6zj3u945u0pk2t7yxyaf5a9geqtsm2j9kr';
      expect(validate(v3)).toBe(false);
    });

    it('rejects v2 addresses with a non-32-byte witness program', () => {
      // Mainnet bech32m, witness v2, 20-byte program.
      const shortProgram = 'prl1zx8h7f4q0sra6ed4d9kasz5re6zj3u945rvmc5t';
      expect(validate(shortProgram)).toBe(false);
    });

    it('rejects addresses with invalid checksum', () => {
      // Last char tampered.
      const bad = MAINNET_P2TR.slice(0, -1) + (MAINNET_P2TR.slice(-1) === 'k' ? 'l' : 'k');
      expect(validate(bad)).toBe(false);
    });

    it('rejects addresses with unknown HRP', () => {
      // Bitcoin mainnet P2TR address — valid bech32m but wrong network prefix.
      expect(validate('bc1paardr2nczq0rx5rqpfwnvpzm497zvux64y0f7wjgcs7xuuuh2nnq3xju0r')).toBe(false);
    });

    it('rejects bogus inputs', () => {
      expect(validate('x')).toBe(false);
      expect(validate('invalid')).toBe(false);
      expect(validate('')).toBe(false);
    });
  });

  describe('Case sensitivity', () => {
    it('accepts uppercase addresses', () => {
      expect(validate(MAINNET_P2TR.toUpperCase())).toBe(true);
      expect(validate(MAINNET_P2MR.toUpperCase())).toBe(true);
    });
  });

  describe('Error behavior', () => {
    it('throws "Invalid address" for unsupported witness versions', () => {
      const v3 = 'prl1rx8h7f4q0sra6ed4d9kasz5re6zj3u945u0pk2t7yxyaf5a9geqtsm2j9kr';
      expect(() => getAddressInfo(v3)).toThrow('Invalid address');
    });

    it('throws "Invalid address" for bogus input', () => {
      expect(() => getAddressInfo('invalid')).toThrow('Invalid address');
    });
  });
});
