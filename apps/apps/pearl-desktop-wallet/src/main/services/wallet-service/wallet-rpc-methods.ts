import { RpcClient } from '../rpc-client';
import type { SyncProgress } from '../../../types/app-bridge';

class WalletRpcMethods {
  constructor(private readonly rpc: RpcClient) { }

  getInfo() {
    return this.rpc.call<unknown>('getinfo', []);
  }

  getSyncProgress(): Promise<SyncProgress> {
    return this.rpc
      .call<{
        header_height: number;
        filter_header_height: number;
        block_height: number;
        best_peer_height: number;
        synced: boolean;
      }>('getsyncprogress', [])
      .then(r => ({
        headerHeight: r.header_height,
        filterHeaderHeight: r.filter_header_height,
        blockHeight: r.block_height,
        bestPeerHeight: r.best_peer_height,
        synced: r.synced,
      }));
  }

  chainSynced() {
    return this.rpc.call<boolean>('chainsynced', []);
  }

  getAddressesByAccount(account: string = 'default') {
    return this.rpc.call<string[]>('getaddressesbyaccount', [account]);
  }

  importPrivateKey(privateKey: string, label: string = '', rescan: boolean = false) {
    return this.rpc.call<void>('importprivkey', [privateKey, label, rescan]);
  }

  getNewAddress(account: string = 'default') {
    return this.rpc.call<string>('getnewaddress', [account]);
  }

  getBalance(account: string = 'default', minconf: number = 1) {
    return this.rpc.call<number>('getbalance', [account, minconf]);
  }

  getReceivedByAddress(address: string) {
    return this.rpc.call<number>('getreceivedbyaddress', [address]);
  }

  listAllTransactions() {
    return this.rpc.call<unknown>('listalltransactions', []);
  }

  listTransactions(count: number = 10, from: number = 0) {
    // Passing account gives an error: Transactions are not yet grouped by account
    return this.rpc.call<unknown>('listtransactions', [undefined, count, from]);
  }

  unlockWallet(passphrase: string, timeout: number = 3600) {
    return this.rpc.call<void>('walletpassphrase', [passphrase, timeout]);
  }

  lockWallet() {
    return this.rpc.call<void>('walletlock', []);
  }

  changeWalletPassphrase(oldPassphrase: string, newPassphrase: string) {
    return this.rpc.call<void>('walletpassphrasechange', [oldPassphrase, newPassphrase]);
  }

  sendFromDefaultAccount(toAddress: string, amount: number, feeRate: number, minconf: number = 0) {
    const outputs = { [toAddress]: amount };
    return this.rpc.call<string>('sendmany', ['default', outputs, feeRate, minconf]);
  }

  async validateAddress(address: string) {
    const validationResult = await this.rpc.call<{ isvalid: boolean }>('validateaddress', [address]);
    return { isValid: validationResult.isvalid };
  }
}

export { WalletRpcMethods };
