# pearl-address-validation

Validate Pearl P2TR (witness v1) and P2MR (witness v2) addresses encoded with bech32m, across mainnet, testnet, and simnet.

```js
validate('prl1px8h7f4q0sra6ed4d9kasz5re6zj3u945u0pk2t7yxyaf5a9geqtsvum09k');
==> true

getAddressInfo('prl1zx8h7f4q0sra6ed4d9kasz5re6zj3u945u0pk2t7yxyaf5a9geqtsypzqta');
==> {
  network: 'mainnet',
  address: 'prl1zx8h7f4q0sra6ed4d9kasz5re6zj3u945u0pk2t7yxyaf5a9geqtsypzqta',
  type: 'p2mr'
}
```

## Installation

```bash
pnpm add pearl-address-validation
```

## Usage

### Importing

```js
import { validate, getAddressInfo } from 'pearl-address-validation';
```

### Validating addresses

`validate(address)` returns `true` for valid Pearl addresses or `false` otherwise.

```js
validate('prl1px8h7f4q0sra6ed4d9kasz5re6zj3u945u0pk2t7yxyaf5a9geqtsvum09k')
==> true

validate('invalid')
==> false
```

#### Network validation

`validate(address, network)` checks that an address is valid _and_ belongs to `network`.

```js
validate('prl1px8h7f4q0sra6ed4d9kasz5re6zj3u945u0pk2t7yxyaf5a9geqtsvum09k', 'mainnet')
==> true

validate('prl1px8h7f4q0sra6ed4d9kasz5re6zj3u945u0pk2t7yxyaf5a9geqtsvum09k', 'testnet')
==> false

validate('tprl1px8h7f4q0sra6ed4d9kasz5re6zj3u945u0pk2t7yxyaf5a9geqts8nl36r', 'testnet')
==> true
```

### Address information

`getAddressInfo(address)` parses the input and returns its type and network. Throws if the address is invalid.

```js
getAddressInfo('prl1px8h7f4q0sra6ed4d9kasz5re6zj3u945u0pk2t7yxyaf5a9geqtsvum09k')
==> {
  address: 'prl1px8h7f4q0sra6ed4d9kasz5re6zj3u945u0pk2t7yxyaf5a9geqtsvum09k',
  type: 'p2tr',
  network: 'mainnet'
}
```

### Address types

- `p2tr` — Pay-to-Taproot (witness v1, BIP 341), 32-byte tweaked output key.
- `p2mr` — Pay-to-Merkle-Root (witness v2, BIP 360), 32-byte script-tree merkle root.

Both encode as bech32m with a 32-byte witness program. Witness versions other than 1 and 2 are rejected.

### Networks

Supported networks: `mainnet` (`prl`), `testnet` (`tprl`), `simnet` (`rprl`).

### TypeScript

```ts
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
```

```ts
import { validate, getAddressInfo, Network, AddressInfo } from 'pearl-address-validation';

validate('prl1zx8h7f4q0sra6ed4d9kasz5re6zj3u945u0pk2t7yxyaf5a9geqtsypzqta', Network.mainnet);
==> true

const info: AddressInfo = getAddressInfo('tprl1zx8h7f4q0sra6ed4d9kasz5re6zj3u945u0pk2t7yxyaf5a9geqts0wx75g');
info.network;
==> 'testnet'
info.type;
==> 'p2mr'
```

## License

The MIT License (MIT).
