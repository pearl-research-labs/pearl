import validate, { getAddressInfo, Network } from '../src/index';
import { expect, describe, it } from 'vitest';

// Vectors: all share one 32-byte witness program. The mainnet vector is a
// real, known-good Pearl address; testnet/simnet vectors were re-encoded
// from the same program with a self-checked bech32m codec (BIP-350).
const MAINNET = 'prl1p62v09vuzyd8kdz9l23jaf3kph4wwx6jqcmhkkhg8lhr2qlxky8psu3zw9d';
const TESTNET = 'tprl1p62v09vuzyd8kdz9l23jaf3kph4wwx6jqcmhkkhg8lhr2qlxky8psh7xs6c';
const SIMNET = 'rprl1p62v09vuzyd8kdz9l23jaf3kph4wwx6jqcmhkkhg8lhr2qlxky8psd86gls';

describe('Taproot Address Validation', () => {
  describe('Valid Taproot Addresses', () => {
    it('validates Mainnet P2TR', () => {
      expect(validate(MAINNET)).toBe(true);
      expect(getAddressInfo(MAINNET)).toEqual({
        bech32: true,
        type: 'p2tr',
        network: 'mainnet',
        address: MAINNET,
      });
    });

    it('validates Testnet P2TR', () => {
      expect(validate(TESTNET)).toBe(true);
      expect(getAddressInfo(TESTNET)).toEqual({
        bech32: true,
        type: 'p2tr',
        network: 'testnet',
        address: TESTNET,
      });
    });

    it('validates Simnet (rprl) P2TR', () => {
      expect(validate(SIMNET)).toBe(true);
      expect(getAddressInfo(SIMNET)).toEqual({
        bech32: true,
        type: 'p2tr',
        network: 'simnet',
        address: SIMNET,
      });
    });
  });

  describe('Validation with Network Parameter', () => {
    it('validates Mainnet P2TR with network parameter', () => {
      expect(validate(MAINNET, Network.mainnet)).toBe(true);
    });

    it('validates Testnet P2TR with network parameter', () => {
      expect(validate(TESTNET, Network.testnet)).toBe(true);
    });

    it('validates Simnet P2TR with network parameter', () => {
      expect(validate(SIMNET, Network.simnet)).toBe(true);
    });

    it('rejects mainnet address when validating against testnet', () => {
      expect(validate(MAINNET, Network.testnet)).toBe(false);
    });

    it('casts testnet to regtest via options', () => {
      const info = getAddressInfo(TESTNET, { castTestnetTo: Network.regtest });
      expect(info.network).toBe('regtest');
      expect(validate(TESTNET, Network.regtest, { castTestnetTo: Network.regtest })).toBe(true);
    });
  });

  describe('Invalid/Rejected Addresses', () => {
    it('rejects Legacy SegWit v0 P2WPKH addresses (Pearl is v1+ only)', () => {
      // Same program as MAINNET, witness version 0
      expect(
        validate('prl1q62v09vuzyd8kdz9l23jaf3kph4wwx6jqcmhkkhg8lhr2qlxky8psr6jtcn'),
      ).toBe(false);
    });

    it('rejects future witness versions (v2)', () => {
      expect(
        validate('prl1z62v09vuzyd8kdz9l23jaf3kph4wwx6jqcmhkkhg8lhr2qlxky8ps5vmptx'),
      ).toBe(false);
    });

    it('rejects non-32-byte witness programs', () => {
      // 20-byte program, witness v1
      expect(validate('prl1p62v09vuzyd8kdz9l23jaf3kph4wwx6jqmu48ck')).toBe(false);
    });

    it('rejects base58 addresses (Pearl has no base58 addresses)', () => {
      // A valid Bitcoin P2PKH address — must NOT validate as a Pearl address
      expect(validate('17VZNX1SN5NtKa8UQFxwQbFeFc3iqRYhem')).toBe(false);
      expect(() => getAddressInfo('17VZNX1SN5NtKa8UQFxwQbFeFc3iqRYhem')).toThrow('Invalid address');
    });

    it('rejects addresses with the wrong human-readable part', () => {
      // Same program, Bitcoin mainnet HRP
      expect(validate('bc1p62v09vuzyd8kdz9l23jaf3kph4wwx6jqcmhkkhg8lhr2qlxky8psyzhlne')).toBe(false);
    });

    it('rejects invalid bech32m checksum', () => {
      // MAINNET with the last character corrupted
      expect(validate('prl1p62v09vuzyd8kdz9l23jaf3kph4wwx6jqcmhkkhg8lhr2qlxky8psu3zw99')).toBe(false);
    });

    it('rejects bogus addresses', () => {
      expect(validate('x')).toBe(false);
      expect(validate('invalid')).toBe(false);
      expect(validate('')).toBe(false);
    });
  });

  describe('Case Sensitivity', () => {
    it('validates uppercase Taproot addresses', () => {
      const address = 'PRL1P62V09VUZYD8KDZ9L23JAF3KPH4WWX6JQCMHKKHG8LHR2QLXKY8PSU3ZW9D';
      expect(validate(address)).toBe(true);
      expect(getAddressInfo(address).network).toBe('mainnet');
    });

    it('rejects mixed-case addresses', () => {
      expect(validate('prl1P62v09vuzyd8kdz9l23jaf3kph4wwx6jqcmhkkhg8lhr2qlxky8psu3zw9d')).toBe(false);
    });
  });

  describe('Error Messages', () => {
    it('throws error for v0 addresses', () => {
      expect(() =>
        getAddressInfo('prl1q62v09vuzyd8kdz9l23jaf3kph4wwx6jqcmhkkhg8lhr2qlxky8psr6jtcn'),
      ).toThrow('Invalid address');
    });

    it('throws descriptive error for invalid address', () => {
      expect(() => getAddressInfo('invalid')).toThrow('Invalid address');
    });
  });
});
