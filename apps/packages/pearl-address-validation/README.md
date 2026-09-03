# pearl-address-validation

Validate Pearl Taproot (P2TR) addresses using bech32m encoding for mainnet, testnet, and simnet networks.

```js
validate('prl1p5cyxnuxmeuwuvkwfem96lqzszd02n6xdcjrs20cac6yqjjwudpxqkedrcr');
==> true

getAddressInfo('prl1p5cyxnuxmeuwuvkwfem96lqzszd02n6xdcjrs20cac6yqjjwudpxqkedrcr');
==> {
  bech32: true,
  network: 'mainnet',
  address: 'prl1p5cyxnuxmeuwuvkwfem96lqzszd02n6xdcjrs20cac6yqjjwudpxqkedrcr',
  type: 'p2tr'
}
```

## Installation

Add `@pearl/pearl-address-validation` to your JavaScript project dependencies using pnpm:

```bash
pnpm add @pearl/pearl-address-validation
```

Or NPM:

```bash
npm install @pearl/pearl-address-validation --save
```

Or Yarn:

```bash
yarn add @pearl/pearl-address-validation
```

## Usage

### Importing

```js
import { validate, getAddressInfo } from '@pearl/pearl-address-validation';
```

### Validating addresses

`validate(address)` returns `true` for valid Pearl addresses or `false` for invalid Pearl addresses.

```js
validate('prl1p5cyxnuxmeuwuvkwfem96lqzszd02n6xdcjrs20cac6yqjjwudpxqkedrcr')
==> true

validate('invalid')
==> false
```

#### Network validation

`validate(address, network)` allows you to validate whether an address is valid and belongs to `network`.

```js
validate('prl1p5cyxnuxmeuwuvkwfem96lqzszd02n6xdcjrs20cac6yqjjwudpxqkedrcr', 'mainnet')
==> true

validate('prl1p5cyxnuxmeuwuvkwfem96lqzszd02n6xdcjrs20cac6yqjjwudpxqkedrcr', 'testnet')
==> false

validate('tprl1p5cyxnuxmeuwuvkwfem96lqzszd02n6xdcjrs20cac6yqjjwudpxqkedrcr', 'testnet')
==> true
```

### Address information

`getAddressInfo(address)` parses the input address and returns information about its type and network.

If the input address is invalid, an exception will be thrown.

```js
getAddressInfo('prl1p5cyxnuxmeuwuvkwfem96lqzszd02n6xdcjrs20cac6yqjjwudpxqkedrcr')
==> {
  address: 'prl1p5cyxnuxmeuwuvkwfem96lqzszd02n6xdcjrs20cac6yqjjwudpxqkedrcr',
  type: 'p2tr',
  network: 'mainnet',
  bech32: true
}
```

### Networks

Pearl uses the following address prefixes:

| Network | Prefix | Example |
|---|---|---|
| mainnet | prl1p... | prl1p5cyxnux... |
| testnet | tprl1p... | tprl1p5cyxnux... |
| simnet | rprl1p... | rprl1p5cyxnux... |

This library supports the following Pearl networks: `mainnet`, `testnet`, `regtest` and `simnet`.

#### Casting testnet addresses to regtest or simnet

```js
getAddressInfo('tprl1p5cyxnuxmeuwuvkwfem96lqzszd02n6xdcjrs20cac6yqjjwudpxqkedrcr', {
  castTestnetTo: 'simnet'
})
==> {
  address: 'tprl1p5cyxnuxmeuwuvkwfem96lqzszd02n6xdcjrs20cac6yqjjwudpxqkedrcr',
  type: 'p2tr',
  network: 'simnet',
  bech32: true
}
```

### TypeScript support

```ts
import { validate, getAddressInfo, Network, AddressInfo } from '@pearl/pearl-address-validation';

validate('prl1p5cyxnuxmeuwuvkwfem96lqzszd02n6xdcjrs20cac6yqjjwudpxqkedrcr', Network.mainnet);
==> true

const addressInfo: AddressInfo = getAddressInfo('tprl1p5cyxnuxmeuwuvkwfem96lqzszd02n6xdcjrs20cac6yqjjwudpxqkedrcr');
addressInfo.network;
==> 'testnet'
```

## License

The MIT License (MIT).
