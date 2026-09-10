import { bech32m } from 'bech32';

enum Network {
  mainnet = 'mainnet',
  testnet = 'testnet',
  simnet = 'simnet',
}

enum AddressType {
  p2tr = 'p2tr',
  p2mr = 'p2mr',
}

type AddressInfo = {
  network: Network;
  address: string;
  type: AddressType;
};

const prefixToNetwork: { [key: string]: Network } = {
  prl: Network.mainnet,
  tprl: Network.testnet,
  rprl: Network.simnet,
};

const witnessVersionToType: { [key: number]: AddressType } = {
  1: AddressType.p2tr,
  2: AddressType.p2mr,
};

const getAddressInfo = (address: string): AddressInfo => {
  let decoded;

  try {
    decoded = bech32m.decode(address);
  } catch {
    throw new Error('Invalid address');
  }

  const network = prefixToNetwork[decoded.prefix];

  if (network === undefined) {
    throw new Error('Invalid address');
  }

  const witnessVersion = decoded.words[0];

  if (witnessVersion === undefined) {
    throw new Error('Invalid address');
  }

  const type = witnessVersionToType[witnessVersion];

  if (type === undefined) {
    throw new Error('Invalid address');
  }

  const program = bech32m.fromWords(decoded.words.slice(1));

  if (program.length !== 32) {
    throw new Error('Invalid address');
  }

  return { network, address, type };
};

const validate = (address: string, network?: Network): boolean => {
  try {
    const addressInfo = getAddressInfo(address);

    if (network) {
      return network === addressInfo.network;
    }

    return true;
  } catch {
    return false;
  }
};

export { getAddressInfo, Network, AddressType, validate };
export type { AddressInfo };
export default validate;
