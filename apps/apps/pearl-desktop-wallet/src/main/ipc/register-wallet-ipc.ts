import { ipcMain } from 'electron';
import { ManagerService } from '../services/manager-service';
import { BlockbookClient } from '../clients/blockbook-client';

function registerWalletIpc(ms: ManagerService) {
  ipcMain.handle('wallet-unlock', (_event, passphrase: string, timeout: number = 60) =>
    ms.ensureWalletService().unlockWallet(passphrase, timeout)
  );
  ipcMain.handle('wallet-lock', _event => ms.lockWallet());
  ipcMain.handle('wallet-force-lock', _event => ms.forceLockWallet());
  ipcMain.handle('wallet-change-password', (_event, currentPassword: string, newPassword: string) =>
    ms.ensureWalletService().changeWalletPassphrase(currentPassword, newPassword)
  );
  ipcMain.handle('wallet-send-from-default-account', (_event, toAddress: string, amount: number, feeRate: number) =>
    ms.ensureWalletService().sendFromDefaultAccount(toAddress, amount, feeRate)
  );
  ipcMain.handle('wallet-remove-transaction', (_event, txid: string) =>
    ms.ensureWalletService().removeTransaction(txid)
  );
  ipcMain.handle('wallet-rebroadcast-transaction', (_event, txid: string) =>
    ms.ensureWalletService().rebroadcastTransaction(txid)
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
  ipcMain.handle('wallet-validate-address', (_event, address: string) =>
    ms.ensureWalletService().validateAddress(address)
  );
  ipcMain.handle('wallet-get-new-address', _event => ms.ensureWalletService().getNewAddress());
  ipcMain.handle('wallet-estimate-fee', (_event, numBlocks: number) =>
    BlockbookClient.estimateFee(numBlocks)
  );
}

export { registerWalletIpc };
