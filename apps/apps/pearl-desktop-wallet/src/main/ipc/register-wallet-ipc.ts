import { ipcMain } from 'electron';
import { ManagerService } from '../services/manager-service';
import { BlockbookClient } from '../clients/blockbook-client';
import { broadcastPearlTx, fetchPrlBalanceGrains, fetchPrlUtxos } from '../services/pearl-rpc';

function registerWalletIpc(ms: ManagerService) {
  ipcMain.handle('wallet-unlock', async (_event, passphrase: string, timeout: number = 60) => {
    await ms.ensureWalletService().unlockWallet(passphrase, timeout);
    return ms.unlockCurrentWalletSeed(passphrase);
  });
  ipcMain.handle('wallet-derive-multisig-key', (_event, vaultAccount: number, keyIndex: number) =>
    ms.ensureWalletService().deriveMultisigKey(vaultAccount, keyIndex)
  );
  ipcMain.handle('wallet-lock', async _event => ms.lockWallet());
  ipcMain.handle('wallet-force-lock', async _event => ms.forceLockWallet());
  ipcMain.handle('wallet-change-password', async (_event, currentPassword: string, newPassword: string) => {
    await ms.ensureWalletService().changeWalletPassphrase(currentPassword, newPassword);
    await ms.rotateCurrentWalletSeedPassword(currentPassword, newPassword);
  });
  ipcMain.handle('wallet-send-from-default-account', (_event, toAddress: string, amount: number, feeRate: number) =>
    ms.ensureWalletService().sendFromDefaultAccount(toAddress, amount, feeRate)
  );
  ipcMain.handle('wallet-list-all-transactions', _event =>
    ms.ensureWalletService().listAllTransactions()
  );
  ipcMain.handle('wallet-list-transactions', (_event, count: number = 10, from: number = 0) =>
    ms.ensureWalletService().listTransactions(count, from)
  );
  ipcMain.handle('wallet-get-balance', (_event, account: string, minconf: number = 1) =>
    ms.ensureWalletService().getBalance(account, minconf)
  );
  ipcMain.handle('wallet-get-vault-balance', async (_event, address: string) => {
    const result = await fetchPrlBalanceGrains(address);
    return {
      grains: result.grains.toString(),
      degraded: result.degraded,
    };
  });
  ipcMain.handle('wallet-get-vault-utxos', async (_event, address: string) => {
    const result = await fetchPrlUtxos(address);
    return {
      utxos: result.utxos.map((u) => ({
        txid: u.txid,
        vout: u.vout,
        valueGrains: u.valueGrains.toString(),
        scriptHex: u.scriptHex,
      })),
      degraded: result.degraded,
      droppedNoScript: result.droppedNoScript,
    };
  });
  ipcMain.handle('wallet-broadcast-pearl-tx', async (_event, rawHex: string) => {
    return await broadcastPearlTx(rawHex);
  });
  ipcMain.handle('wallet-get-transaction-info', async (_event, txid: string) => {
    return await ms.ensureWalletService().getTransactionInfo(txid);
  });
  ipcMain.handle('wallet-validate-address', (_event, address: string) =>
    ms.ensureWalletService().validateAddress(address)
  );
  ipcMain.handle('wallet-get-new-address', _event => ms.ensureWalletService().getNewAddress());
  ipcMain.handle('wallet-estimate-fee', (_event, numBlocks: number) =>
    BlockbookClient.estimateFee(numBlocks)
  );
}

export { registerWalletIpc };
