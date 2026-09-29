package legacyrpc

import (
	"github.com/pearl-research-labs/pearl/node/blockchain"
	"github.com/pearl-research-labs/pearl/node/btcjson"
	"github.com/pearl-research-labs/pearl/wallet/wallet"
)

// listEntries renders txs as listtransactions entries: for every output a transaction shows, a send when the wallet
// paid it out and a receive when it pays the wallet.
func listEntries(txs []wallet.Tx) []btcjson.ListTransactionsResult {
	entries := []btcjson.ListTransactionsResult{}
	for i := range txs {
		tx := &txs[i]
		base := btcjson.ListTransactionsResult{
			TxID:            tx.Hash.String(),
			Generated:       blockchain.IsCoinBaseTx(&tx.MsgTx),
			Time:            tx.Received.Unix(),
			TimeReceived:    tx.Received.Unix(),
			WalletConflicts: []string{},
		}
		if tx.Block.Height != -1 {
			base.BlockHash = tx.Block.Hash.String()
			base.BlockTime = tx.Block.Time.Unix()
			base.Confirmations = int64(tx.Confirmations)
		}
		fee := (-tx.Fee).ToPRL()

		for _, out := range tx.Outputs {
			entry := base
			entry.Vout = out.Index
			entry.Address = out.Address
			if out.Sent {
				send := entry
				send.Category = "send"
				send.Amount = -out.Amount.ToPRL()
				send.Fee = &fee
				entries = append(entries, send)
			}
			if out.Received {
				receive := entry
				receive.Category = tx.ReceiveCategory.String()
				receive.Amount = out.Amount.ToPRL()
				receive.Account = out.Account
				entries = append(entries, receive)
			}
		}
	}
	return entries
}
