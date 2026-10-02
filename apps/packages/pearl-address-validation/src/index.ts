import { bech32, bech32m } from 'bech32';

enum Network {
  mainnet = 'mainnet',
  testnet = 'testnet',
  regtest = 'regtest',
  simnet = 'simnet',
}

// Pearl is Taproot-only (bech32m, witness v1+, 32-byte programs — see
// node/btcutil/address.go `decodeSegWitAddress` in pearl-research-labs/pearl).
// The legacy members below are kept only for API compatibility; the parser
// never returns them.
enum AddressType {
  p2pkh = 'p2pkh',
  p2sh = 'p2sh',
  p2wpkh = 'p2wpkh',
  p2wsh = 'p2wsh',
  p2tr = 'p2tr',
}

type AddressInfo = {
  bech32: boolean;
  network: Network;
  address: string;
  type: AddressType;
};

type Options = {
  castTestnetTo?: Network.regtest | Network.simnet;
};

function castTestnetTo(fromNetwork: Network, toNetwork?: Network.regtest | Network.simnet): Network {
  if (!toNetwork) {
    return fromNetwork;
  }

  if (fromNetwork === Network.mainnet) {
    throw new Error('Cannot cast mainnet to non-mainnet');
  }

  return toNetwork;
}

const normalizeAddressInfo = (addressInfo: AddressInfo, options?: Options): AddressInfo => {
  return {
    ...addressInfo,
    network: castTestnetTo(addressInfo.network, options?.castTestnetTo),
  };
};

const parseBech32 = (address: string, options?: Options): AddressInfo => {
  let decoded;

  const lowerAddress = address.toLowerCase();
  // Only accept Taproot addresses (witness v1) - reject legacy SegWit v0
  // Check if address starts with 'p' (witness v1)
  if (!lowerAddress.startsWith('prl1p') && !lowerAddress.startsWith('tprl1p') && !lowerAddress.startsWith('rprl1p')) {
    throw new Error('Invalid address');
  }

  try {
    // Taproot uses bech32m encoding
    decoded = bech32m.decode(address);
  } catch (error) {
    throw new Error('Invalid address');
  }

  const mapPrefixToNetwork: { [key: string]: Network } = {
    prl: Network.mainnet,
    tprl: Network.testnet,
    rprl: Network.simnet,
  };

  const network: Network | undefined = mapPrefixToNetwork[decoded.prefix];

  if (network === undefined) {
    throw new Error('Invalid address');
  }

  const witnessVersion = decoded.words[0];

  // Only accept witness version 1 (Taproot)
  if (witnessVersion !== 1) {
    throw new Error('Only Taproot (witness v1) addresses are supported. Found witness version: ' + witnessVersion);
  }

  const data = bech32.fromWords(decoded.words.slice(1));

  // Taproot addresses must have 32-byte programs
  if (data.length !== 32) {
    throw new Error('Invalid Taproot address: witness program must be 32 bytes');
  }

  const type = AddressType.p2tr;

  return normalizeAddressInfo(
    {
      bech32: true,
      network,
      address,
      type,
    },
    options,
  );
};

const getAddressInfo = (address: string, options?: Options): AddressInfo => {
  const lowerAddress = address.toLowerCase();
  // Pearl addresses are bech32m segwit (Taproot, witness v1+). There are no
  // base58 Pearl addresses — base58 strings (e.g. Bitcoin '1…'/'3…' or any
  // other chain's) are never valid here and must be rejected.
  if (lowerAddress.startsWith('prl1') || lowerAddress.startsWith('tprl1') || lowerAddress.startsWith('rprl1')) {
    try {
      return parseBech32(address, options);
    } catch (error) {
      throw new Error('Invalid address');
    }
  }

  throw new Error('Invalid address');
};

const validate = (address: string, network?: Network, options?: Options) => {
  try {
    const addressInfo = getAddressInfo(address, options);

    if (network) {
      return network === addressInfo.network;
    }

    return true;
  } catch (error) {
    return false;
  }
};

export { getAddressInfo, Network, AddressType, validate };
export type { AddressInfo };
export default validate;
