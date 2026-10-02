# pearl-address-validation

Validate Pearl Taproot (P2TR) addresses using bech32m encoding for mainnet, testnet, regtest, and simnet networks.

Pearl is Taproot-only: valid addresses are bech32m (BIP-350), witness version
1, with a 32-byte witness program. Legacy SegWit v0 addresses, future witness
versions, non-32-byte programs, and base58 addresses (Pearl has never had any)
are all rejected — matching `pearld`'s `decodeSegWitAddress`.

```js
validate('prl1p62v09vuzyd8kdz9l23jaf3kph4wwx6jqcmhkkhg8lhr2qlxky8psu3zw9d');
==> true

getAddressInfo('prl1p62v09vuzyd8kdz9l23jaf3kph4wwx6jqcmhkkhg8lhr2qlxky8psu3zw9d');
==> {
  bech32: true,
  network: 'mainnet',
  address: 'prl1p62v09vuzyd8kdz9l23jaf3kph4wwx6jqcmhkkhg8lhr2qlxky8psu3zw9d',
  type: 'p2tr'
}
```

## Installation

Add `pearl-address-validation` to your Javascript project dependencies using Yarn:

```bash
yarn add pearl-address-validation
```

Or NPM:

```bash
npm install pearl-address-validation --save
```

## Usage

### Importing

```js
import { validate, getAddressInfo } from 'pearl-address-validation';
```

### Validating addresses

`validate(address)` returns `true` for valid Pearl addresses or `false` for invalid Pearl addresses.

```js
validate('prl1p62v09vuzyd8kdz9l23jaf3kph4wwx6jqcmhkkhg8lhr2qlxky8psu3zw9d')
==> true

validate('17VZNX1SN5NtKa8UQFxwQbFeFc3iqRYhem') // a Bitcoin base58 address
==> false

validate('invalid')
==> false
```

#### Network validation

`validate(address, network)` allows you to validate whether an address is valid and belongs to `network`.

```js
validate('prl1p62v09vuzyd8kdz9l23jaf3kph4wwx6jqcmhkkhg8lhr2qlxky8psu3zw9d', 'mainnet')
==> true

validate('prl1p62v09vuzyd8kdz9l23jaf3kph4wwx6jqcmhkkhg8lhr2qlxky8psu3zw9d', 'testnet')
==> false

validate('tprl1p62v09vuzyd8kdz9l23jaf3kph4wwx6jqcmhkkhg8lhr2qlxky8psh7xs6c', 'testnet')
==> true
```

### Address information

`getAddressInfo(address)` parses the input address and returns information about its type and network.

If the input address is invalid, an exception will be thrown.

```js
getAddressInfo('prl1p62v09vuzyd8kdz9l23jaf3kph4wwx6jqcmhkkhg8lhr2qlxky8psu3zw9d')
==> {
  address: 'prl1p62v09vuzyd8kdz9l23jaf3kph4wwx6jqcmhkkhg8lhr2qlxky8psu3zw9d',
  type: 'p2tr',
  network: 'mainnet',
  bech32: true
}
```

### Networks

This library supports the following Pearl networks: `mainnet`, `testnet`, `regtest` and `simnet`.

Address prefixes: `prl` (mainnet), `tprl` (testnet), `rprl` (regtest/simnet).

#### Casting testnet addresses to regtest or simnet

You can use the `options` parameter to cast `testnet` addresses to `regtest` or `simnet`.

```js
// Default - No casting
getAddressInfo('tprl1p62v09vuzyd8kdz9l23jaf3kph4wwx6jqcmhkkhg8lhr2qlxky8psh7xs6c');
==> {
  address: 'tprl1p62v09vuzyd8kdz9l23jaf3kph4wwx6jqcmhkkhg8lhr2qlxky8psh7xs6c',
  type: 'p2tr',
  network: 'testnet',
  bech32: true
}

// Cast testnet to regtest
getAddressInfo('tprl1p62v09vuzyd8kdz9l23jaf3kph4wwx6jqcmhkkhg8lhr2qlxky8psh7xs6c', {
  castTestnetTo: 'regtest'
})
==> {
  address: 'tprl1p62v09vuzyd8kdz9l23jaf3kph4wwx6jqcmhkkhg8lhr2qlxky8psh7xs6c',
  type: 'p2tr',
  network: 'regtest',
  bech32: true
}

// Validating and casting
validate('tprl1p62v09vuzyd8kdz9l23jaf3kph4wwx6jqcmhkkhg8lhr2qlxky8psh7xs6c', 'regtest', {
  castTestnetTo: 'regtest'
})
==> true
```

### TypeScript support

If you're using TypeScript, the following types are provided with this library:

```ts
enum Network {
  mainnet = 'mainnet',
  testnet = 'testnet',
  regtest = 'regtest',
  simnet = 'simnet',
}

enum AddressType {
  p2pkh = 'p2pkh', // legacy, never returned for Pearl addresses
  p2sh = 'p2sh',   // legacy, never returned for Pearl addresses
  p2wpkh = 'p2wpkh', // legacy, never returned for Pearl addresses
  p2wsh = 'p2wsh',   // legacy, never returned for Pearl addresses
  p2tr = 'p2tr',
}

type AddressInfo = {
  bech32: boolean;
  network: Network;
  address: string;
  type: AddressType;
};
```

#### TypeScript usage

```ts
import { validate, getAddressInfo, Network, AddressInfo } from 'pearl-address-validation';

validate('prl1p62v09vuzyd8kdz9l23jaf3kph4wwx6jqcmhkkhg8lhr2qlxky8psu3zw9d', Network.mainnet);
==> true

const addressInfo: AddressInfo = getAddressInfo('tprl1p62v09vuzyd8kdz9l23jaf3kph4wwx6jqcmhkkhg8lhr2qlxky8psh7xs6c');
addressInfo.network;

==> 'testnet'
```

## License

The MIT License (MIT).
