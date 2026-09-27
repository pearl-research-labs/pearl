import { RpcClient, RpcConfig } from '../rpc-client.ts';
import { formatAndSortTransactions } from './transaction-formatter.ts';
import { WalletRpcMethods } from './wallet-rpc-methods.ts';
import { WalletApi } from '../../../types/app-bridge.ts';
import { BlockbookClient } from '../../clients/blockbook-client.ts';

class WalletService extends WalletRpcMethods implements WalletApi {
  constructor(config: RpcConfig) {
    super(
      new RpcClient({
        rpcHost: config.rpcHost,
        rpcPort: config.rpcPort,
        rpcUser: config.rpcUser,
        rpcPassword: config.rpcPassword,
      })
    );
  }

  override async listAllTransactions() {
    const transactions = await super.listAllTransactions();
    return formatAndSortTransactions(transactions);
  }

  override async listTransactions(count: number = 10, from: number = 0) {
    // Using listtransactions rpc call doesn't work properly. In addition,
    // passing account gives an error: "Transactions are not yet grouped by
    // account"
    // Therefore using listAllTransactions and paginate ourselves
    const allTransactions = await this.listAllTransactions();

    // THIS IS A HACK TO REMOVE THE SENT TRANSACTIONS WITH NO FEE AND TO HIDE ACTIVITIES THAT WAS CREATED
    // BY CURRENT WALLET (TO HIDE USED UTXOS THAT WASNT SPENT TOTALLY)
    const filteredTransactions = allTransactions.filter(
      ({ type, fee }) => !(type === 'sent' && fee === 0)
    );
    const transactions = filteredTransactions.slice(from, from + count);
    return transactions;
  }

  async getTransactionInfo(txid: string) {
    return await BlockbookClient.getTransactionInfo(txid);
  }
}

export { WalletService };
