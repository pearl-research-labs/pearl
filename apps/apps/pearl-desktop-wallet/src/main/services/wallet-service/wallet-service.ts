import { RpcClient, RpcConfig } from '../rpc-client.ts';
import { formatAndSortTransactions, formatTransaction, sortNewestFirst } from './transaction-formatter.ts';
import { WalletRpcMethods } from './wallet-rpc-methods.ts';
import { WalletApi } from '../../../types/app-bridge.ts';
import { Transaction } from '../../../types/transaction.ts';

const RECENT_PAGE_SIZE = 20;

// THIS IS A HACK TO REMOVE THE SENT TRANSACTIONS WITH NO FEE AND TO HIDE ACTIVITIES THAT WAS CREATED
// BY CURRENT WALLET (TO HIDE USED UTXOS THAT WASNT SPENT TOTALLY)
function isShown({ type, fee }: Transaction) {
  return !(type === 'sent' && fee === 0);
}

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
    // Activity pages by slicing one time-sorted list of shown rows; only the whole history keeps that order stable
    // from page to page.
    const allTransactions = await this.listAllTransactions();
    return allTransactions.filter(isShown).slice(from, from + count);
  }

  // The dashboard polls this, so it reads history newest first a page at a time rather than listing all of it.
  // Unmined transactions come first in no time order and the filter can hide whole pages (a miner's immature
  // coinbases), so read on until count shown rows are mined; every row after those is older.
  async listRecentTransactions(count: number) {
    const shown: Transaction[] = [];
    let shownMined = 0;
    for (let from = 0; shownMined < count; from += RECENT_PAGE_SIZE) {
      const page = await super.listTransactions(RECENT_PAGE_SIZE, from);
      if (page.length === 0) break;
      const rows: Transaction[] = page.map(formatTransaction).filter(isShown);
      shown.push(...rows);
      shownMined += rows.filter(tx => tx.blockhash).length;
    }
    return sortNewestFirst(shown).slice(0, count);
  }
}

export { WalletService };
